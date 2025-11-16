#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script runs a data augmentation experiment by
CLASSIFYING PURELY IN THE NORMALIZED LATENT SPACE.

1.  Real Data: Raw -> Scale -> Encode -> Normalize -> Flatten
2.  Synthetic Data: DDPM -> Rescale -> Flatten
3.  Combines these latent vectors and trains the classifier.
"""

import math, json, random
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings

# ==================================
# ======== MAIN CONFIGURATION ========
# ==================================
AUGMENT_RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TRAIN_CSV = Path(r"../MyDataset/ftir_train_wn.csv")
TEST_CSV = Path(r"../MyDataset/ftir_test_wn.csv")
LDM_DIR = Path("ldm_out")
AE_WEIGHTS_FILE = LDM_DIR / "ae_conv1d.pt"
DDPM_CHECKPOINT_FILE = LDM_DIR / "ddpm_latent_unet.pt"
AE_META_FILE = LDM_DIR / "ae_meta.json"
OUT_DIR = Path(r"Augmentation_Results_Pure_Latent")  # New output folder
SAMPLE_STEPS = 1000
SEED = 42
ENCODER_BATCH_SIZE = 128
# ==================================

# Setup
warnings.filterwarnings('ignore', category=UserWarning)
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
random.seed(SEED);
np.random.seed(SEED);
torch.manual_seed(SEED)


# =================================================================
# PART 1: PLS-DA CLASSIFIER CODE
# (Helper functions: venetian_blinds_folds, tune_pls_da_auc, cm_as_matrix)
# =================================================================

def venetian_blinds_folds(y: np.ndarray, k: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(y));
    folds = []
    classes = np.unique(y);
    class_pos = {c: np.where(y == c)[0] for c in classes}
    for fold in range(k):
        val = []
        for c, pos in class_pos.items():
            take = pos[np.arange(len(pos)) % k == fold]
            val.append(take)
        val = np.concatenate(val) if val else np.array([], dtype=int)
        tr = np.setdiff1d(idx, val, assume_unique=False);
        folds.append((tr, val))
    return folds


def tune_pls_da_auc(X: np.ndarray, y: np.ndarray, k: int, max_lv: int = 30):
    best_auc, best_c = -np.inf, None
    folds = venetian_blinds_folds(y, k)
    upper = max(1, min(max_lv, min(X.shape[0] - 1, X.shape[1])))
    min_class_samples_cv = min(np.sum(y == 0), np.sum(y == 1)) * (1 - 1 / k)
    if upper > min_class_samples_cv:
        upper = max(1, int(min_class_samples_cv) - 1)

    for ncomp in range(1, upper + 1):
        oof = np.zeros_like(y, dtype=float);
        seen = np.zeros_like(y, dtype=bool)
        for tr_idx, va_idx in folds:
            if len(va_idx) == 0 or len(tr_idx) == 0: continue
            n_eff = max(1, min(ncomp, min(X[tr_idx].shape[0] - 1, X.shape[1])))
            min_class_fold = min(np.sum(y[tr_idx] == 0), np.sum(y[tr_idx] == 1))
            if n_eff < 1 or n_eff > min_class_fold: continue

            try:
                pls = PLSRegression(n_components=n_eff, scale=False)
                pls.fit(X[tr_idx], y[tr_idx])
                oof[va_idx] = pls.predict(X[va_idx]).ravel()
                seen[va_idx] = True
            except ValueError:
                continue
        if not seen.any(): continue
        auc = roc_auc_score(y[seen], oof[seen])
        if auc > best_auc:
            best_auc, best_c = auc, ncomp
    if best_c is None:
        best_c, best_auc = 2, float("nan")
    return best_c, best_auc


def cm_as_matrix(y_true, y_pred):
    labels = np.unique(np.concatenate((y_true, y_pred)))
    if len(labels) == 1:
        if labels[0] == 0:
            return np.array([[len(y_true), 0], [0, 0]])
        else:
            return np.array([[0, 0], [0, len(y_true)]])
    return confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()


# =================================================================
# PART 2: GENERATIVE MODEL CODE
# (Full AE and UNet definitions)
# =================================================================

def gnorm(c): return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class ConvAE(nn.Module):
    # Unchanged - we still need the encoder
    def __init__(self, F: int, downs: int = 4, base: int = 64, latent_c: int = 64):
        super().__init__()
        self.F = F
        c = base;
        in_c = 1;
        enc = []
        for i in range(downs):
            out_c = latent_c if i == downs - 1 else c
            enc += [nn.Conv1d(in_c, c, 5, padding=2), nn.SiLU(), nn.Conv1d(c, out_c, 5, stride=2, padding=2), nn.SiLU()]
            in_c = out_c;
            c = min(c * 2, 256)
        self.encoder = nn.Sequential(*enc)
        dec = [];
        c_cur = latent_c
        for i in range(downs):
            c_mid = max(c_cur // 2, base) if i < downs - 1 else base
            c_out = base if i < downs - 1 else 32
            dec += [nn.ConvTranspose1d(c_cur, c_mid, 4, stride=2, padding=1), nn.SiLU(),
                    nn.Conv1d(c_mid, c_out, 5, padding=2), nn.SiLU()]
            c_cur = c_out
        self.decoder = nn.Sequential(*dec)
        self.to_raw = nn.Conv1d(c_cur, 1, kernel_size=3, padding=1)
        self.align_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(latent_c, 256), nn.SiLU(),
                                        nn.Linear(256, F))

    def encode(self, x):  # (B, 1, F)
        return self.encoder(x)  # (B, C, L)

    def decode(self, h):  # (B, C, L)
        y = self.decoder(h)
        if y.shape[-1] != self.F:
            if y.shape[-1] > self.F:
                diff = y.shape[-1] - self.F;
                start = diff // 2
                y = y[..., start: start + self.F]
            else:
                pad = self.F - y.shape[-1]
                y = Fnn.pad(y, (pad // 2, pad - pad // 2))
        return self.to_raw(y)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=128, max_period=10000.0):
        super().__init__();
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False);
        self.dim = dim

    def forward(self, t):
        t = t.float().unsqueeze(1);
        ang = t * self.freqs.unsqueeze(0)
        return torch.cat([ang.sin(), ang.cos()], dim=1)


class ClassEmbedding(nn.Module):
    def __init__(self, num_classes=2, dim=32):
        super().__init__();
        self.emb = nn.Embedding(num_classes, dim)

    def forward(self, y): return self.emb(y)


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1);
        self.gn1 = gnorm(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1);
        self.gn2 = gnorm(cout)
        self.act = nn.SiLU()
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, cout))
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cvec):
        h = self.act(self.gn1(self.conv1(x)));
        b = self.cond(cvec).unsqueeze(-1)
        h = self.conv2(h);
        h = self.gn2(h + b);
        h = self.act(h)
        return h + self.skip(x)


class UNet1D_Cond(nn.Module):
    # Unchanged
    def __init__(self, in_ch=64, base=128, out_ch=64, time_dim=128, class_dim=32, num_classes=2):
        super().__init__()
        self.temb = SinusoidalTimeEmbedding(time_dim);
        self.yemb = ClassEmbedding(num_classes=num_classes, dim=class_dim)
        self.proj = nn.Sequential(nn.Linear(time_dim + class_dim, base), nn.SiLU());
        cond_dim = base
        self.rb1 = ResBlock1D(in_ch, base, cond_dim);
        self.down1 = nn.Conv1d(base, base, 4, stride=2, padding=1)
        self.rb2 = ResBlock1D(base, base * 2, cond_dim);
        self.down2 = nn.Conv1d(base * 2, base * 2, 4, stride=2, padding=1)
        self.mid1 = ResBlock1D(base * 2, base * 4, cond_dim);
        self.mid2 = ResBlock1D(base * 4, base * 4, cond_dim)
        self.up2_conv = nn.Conv1d(base * 4, base * 2, 1);
        self.rb_up2a = ResBlock1D(base * 2 + base * 2, base * 2, cond_dim);
        self.rb_up2b = ResBlock1D(base * 2, base * 2, cond_dim)
        self.up1_conv = nn.Conv1d(base * 2, base, 1);
        self.rb_up1a = ResBlock1D(base + base, base, cond_dim);
        self.rb_up1b = ResBlock1D(base, base, cond_dim)
        self.head = nn.Conv1d(base, out_ch, 3, padding=1)

    def forward(self, zt, t, y):
        c = torch.cat([self.temb(t), self.yemb(y)], dim=1);
        c = self.proj(c)
        h1 = self.rb1(zt, c);
        x = self.down1(h1);
        h2 = self.rb2(x, c);
        x = self.down2(h2)
        x = self.mid1(x, c);
        x = self.mid2(x, c)
        x = Fnn.interpolate(x, size=h2.shape[-1], mode="linear", align_corners=False);
        x = self.up2_conv(x);
        x = torch.cat([x, h2], dim=1)
        x = self.rb_up2a(x, c);
        x = self.rb_up2b(x, c)
        x = Fnn.interpolate(x, size=h1.shape[-1], mode="linear", align_corners=False);
        x = self.up1_conv(x);
        x = torch.cat([x, h1], dim=1)
        x = self.rb_up1a(x, c);
        x = self.rb_up1b(x, c)
        return self.head(x)


def cosine_beta_schedule(T, s=0.008):
    steps = T + 1;
    x = torch.linspace(0, T, steps, dtype=torch.float32)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0];
    betas = 1 - (ac[1:] / ac[:-1])
    return betas.clamp(1e-8, 0.999)


@torch.no_grad()
def p_sample_loop_latent(unet, T, steps, y_class, n,
                         latent_c, latent_L):
    """
    NEW sampling loop.
    Generates and returns the final *normalized* latent vector.
    """
    betas_s = cosine_beta_schedule(T).to(DEVICE)
    alphas_s = 1.0 - betas_s
    ac_s = torch.cumprod(alphas_s, dim=0)
    sqrt_recip_alphas = (1.0 / torch.sqrt(alphas_s)).to(DEVICE)

    z_t = torch.randn(n, latent_c, latent_L, device=DEVICE)
    y = torch.full((n,), int(y_class), device=DEVICE, dtype=torch.long)
    ts = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=DEVICE)

    for t_val in ts:
        t = t_val.repeat(n)
        eps_hat = unet(z_t, t, y)
        beta_t = betas_s[t].view(-1, 1, 1)
        sqrt_one_minus_ac_t = torch.sqrt(1.0 - ac_s[t]).view(-1, 1, 1)
        sqrt_recip_alpha_t = sqrt_recip_alphas[t].view(-1, 1, 1)
        mean = sqrt_recip_alpha_t * (z_t - beta_t / sqrt_one_minus_ac_t * eps_hat)
        if (t_val > 0):
            noise = torch.randn_like(z_t)
            z_t = mean + torch.sqrt(beta_t) * noise
        else:
            z_t = mean

    # z_t is now the final predicted *normalized* z0

    # === Rescaling Fix ===
    z_gen_mean = z_t.mean(dim=0, keepdim=True)
    z_gen_std = z_t.std(dim=0, keepdim=True)
    z_t_renormed = (z_t - z_gen_mean) / z_gen_std.clamp(1e-6)

    # === RETURN THE LATENT ===
    return z_t_renormed.cpu().numpy()  # (N, C, L)


# =================================================================
# PART 3: NEW PREPROCESSING & MAIN EXPERIMENT SCRIPT
# =================================================================

def preprocess_with_encoder_and_normalize(X_raw_np: np.ndarray, ae_model: ConvAE,
                                          scaler_mean: np.ndarray, scaler_std: np.ndarray,
                                          z_mu_np: np.ndarray, z_std_np: np.ndarray,
                                          batch_size: int) -> np.ndarray:
    """
    NEW preprocessing function.
    Raw -> Scale -> Encode -> Normalize -> Flatten
    """
    ae_model.eval()

    # 1. Scale the raw data
    X_scaled_np = (X_raw_np - scaler_mean) / (scaler_std + 1e-12)

    # 2. Convert to batched torch tensor
    X_tensor = torch.from_numpy(X_scaled_np).float().unsqueeze(1)  # (N, 1, F)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_latents = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            z = ae_model.encode(xb)  # (B, C, L)
            all_latents.append(z.cpu())

    z_full = torch.cat(all_latents, dim=0)  # (N, C, L)

    # 3. Normalize the latents
    z_mu_t = torch.from_numpy(z_mu_np).cpu()
    z_std_t = torch.from_numpy(z_std_np).cpu()
    z_norm_full = (z_full - z_mu_t) / z_std_t.clamp(1e-6)

    # 4. Concatenate and flatten
    z_norm_flat = z_norm_full.numpy().reshape(z_norm_full.shape[0], -1)  # (N, C*L)

    return z_norm_flat


def run_pls_analysis(X_tr: np.ndarray, y_tr: np.ndarray,
                     X_te: np.ndarray, y_te: np.ndarray,
                     n_total_train: int, ratio: float):
    """
    Tunes, trains, and evaluates the PLS-DA model.
    """
    print(f"  Tuning PLS-DA (N={n_total_train}, F={X_tr.shape[1]})...")
    best_lv, cv_auc = tune_pls_da_auc(X_tr, y_tr, k=10, max_lv=min(40, X_tr.shape[1] - 1))

    best_lv = max(1, min(best_lv, min(X_tr.shape[0] - 1, X_tr.shape[1])))

    pls = PLSRegression(n_components=best_lv, scale=False)
    pls.fit(X_tr, y_tr)

    prob_te = pls.predict(X_te).ravel()
    auc = roc_auc_score(y_te, prob_te)
    yhat_te = (prob_te >= 0.5).astype(int)

    acc = accuracy_score(y_te, yhat_te)
    cm = confusion_matrix(y_te, yhat_te)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
    else:
        (tn, fp), (fn, tp) = cm_as_matrix(y_te, yhat_te)
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0

    print(
        f"  Test Results (Ratio {ratio * 100}%): LV={best_lv}, AUC={auc:.4f}, Acc={acc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}")

    return {
        "ratio": f"{ratio * 100:.0f}%",
        "n_total_train": n_total_train,
        "n_synthetic": n_total_train - len(y_te),  # (Hacky)
        "features": X_tr.shape[1],
        "best_lv": best_lv,
        "cv_auc": cv_auc,
        "test_auc": auc,
        "test_acc": acc,
        "test_sens": sens,
        "test_spec": spec,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }


def main():
    print("Starting PURE LATENT SPACE augmentation experiment...")

    # --- 1. Load Original Data ---
    if not (TRAIN_CSV.exists() and TEST_CSV.exists() and AE_META_FILE.exists()):
        print("Error: Missing one or more input files.")
        return

    df_tr_orig = pd.read_csv(TRAIN_CSV)
    df_te = pd.read_csv(TEST_CSV)

    with open(AE_META_FILE, "r") as f:
        meta = json.load(f)
    spec_cols = meta["cols"]

    X_tr_orig_raw = df_tr_orig[spec_cols].to_numpy(dtype=np.float32)
    y_tr_orig = (df_tr_orig["classes"].values != 0).astype(int)

    X_te_raw = df_te[spec_cols].to_numpy(dtype=np.float32)
    y_te = (df_te["classes"].values != 0).astype(int)

    n_orig_healthy = np.sum(y_tr_orig == 0)
    n_orig_cancer = np.sum(y_tr_orig == 1)
    print(f"Loaded original train data: {n_orig_healthy} Healthy, {n_orig_cancer} Cancer. Total: {len(y_tr_orig)}")

    # --- 2. Load Generative Models & Scalers ---
    print("Loading generative models...")
    ckpt = torch.load(DDPM_CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)
    gen_meta = ckpt["meta"]

    F_LEN = int(gen_meta["F"]);
    downs = int(gen_meta["downs"])
    latent_c = int(gen_meta["latent_channels"]);
    latent_L = int(gen_meta["latent_length"])
    T_trained = int(ckpt["T"])

    # Scalers for AE
    scaler_mean_np = np.array(gen_meta["scaler_mean"], dtype=np.float32)
    scaler_std_np = np.array(gen_meta["scaler_std"], dtype=np.float32)

    # Scalers for Latent Space
    z_mu_np = ckpt["z_mu"].cpu().numpy()
    z_std_np = ckpt["z_std"].cpu().numpy()

    ae = ConvAE(F_LEN, downs=downs, latent_c=latent_c).to(DEVICE)
    ae_sd = torch.load(AE_WEIGHTS_FILE, map_location=DEVICE, weights_only=False)
    ae.load_state_dict(ae_sd, strict=False)
    ae.eval()

    unet = UNet1D_Cond(in_ch=latent_c, base=128, out_ch=latent_c).to(DEVICE)
    unet.load_state_dict(ckpt["model"])
    unet.eval()
    print("All models loaded.")

    # --- 3. Preprocess Real Data (Done once using NEW method) ---
    print("Preprocessing test data (Raw -> Scale -> Encode -> Normalize)...")
    X_te_preproc = preprocess_with_encoder_and_normalize(
        X_te_raw, ae, scaler_mean_np, scaler_std_np,
        z_mu_np, z_std_np, ENCODER_BATCH_SIZE
    )
    print(f"Test data preprocessed to shape: {X_te_preproc.shape}")

    print("Preprocessing original train data (Raw -> Scale -> Encode -> Normalize)...")
    X_tr_orig_preproc = preprocess_with_encoder_and_normalize(
        X_tr_orig_raw, ae, scaler_mean_np, scaler_std_np,
        z_mu_np, z_std_np, ENCODER_BATCH_SIZE
    )
    print(f"Original train data preprocessed to shape: {X_tr_orig_preproc.shape}")

    # --- 4. Run Experiment Loop ---
    all_results = []

    for ratio in AUGMENT_RATIOS:
        print("\n" + "=" * 50)
        print(f"RUNNING EXPERIMENT: {ratio * 100:.0f}% Augmentation")
        print("=" * 50)

        if ratio == 0.0:
            X_tr_aug_preproc = X_tr_orig_preproc
            y_tr_aug = y_tr_orig
            print(f"  Using 0% synthetic data (Baseline). N={len(y_tr_aug)}")

        else:
            n_healthy_gen = int(n_orig_healthy * ratio)
            n_cancer_gen = int(n_orig_cancer * ratio)
            print(f"  Generating {n_healthy_gen} healthy and {n_cancer_gen} cancer (normalized latents)...")

            z_gen_healthy = p_sample_loop_latent(
                unet=unet, T=T_trained, steps=min(SAMPLE_STEPS, T_trained),
                y_class=0, n=n_healthy_gen,
                latent_c=latent_c, latent_L=latent_L
            )
            y_gen_healthy = np.zeros(n_healthy_gen, dtype=int)
            # Flatten the generated latents
            z_gen_healthy_flat = z_gen_healthy.reshape(n_healthy_gen, -1)

            z_gen_cancer = p_sample_loop_latent(
                unet=unet, T=T_trained, steps=min(SAMPLE_STEPS, T_trained),
                y_class=1, n=n_cancer_gen,
                latent_c=latent_c, latent_L=latent_L
            )
            y_gen_cancer = np.ones(n_cancer_gen, dtype=int)
            # Flatten the generated latents
            z_gen_cancer_flat = z_gen_cancer.reshape(n_cancer_gen, -1)

            print(f"  Generated {len(z_gen_healthy_flat)} + {len(z_gen_cancer_flat)} new latent vectors.")

            # Combine with original *preprocessed* data
            X_tr_aug_preproc = np.vstack([X_tr_orig_preproc, z_gen_healthy_flat, z_gen_cancer_flat])
            y_tr_aug = np.hstack([y_tr_orig, y_gen_healthy, y_gen_cancer])
            print(f"  New training set size: {len(y_tr_aug)}")

        # --- Train (Data is already preprocessed) ---
        n_syn = len(y_tr_aug) - len(y_tr_orig)
        results = run_pls_analysis(X_tr_aug_preproc, y_tr_aug, X_te_preproc, y_te, len(y_tr_aug), ratio)
        results["n_synthetic"] = n_syn
        all_results.append(results)

    # --- 5. Final Report ---
    print("\n" + "=" * 60)
    print("     PURE LATENT SPACE AUGMENTATION EXPERIMENT SUMMARY")
    print("=" * 60)

    df_results = pd.DataFrame(all_results)
    df_results.set_index("ratio", inplace=True)

    results_csv_path = OUT_DIR / "augmentation_results_pure_latent.csv"
    df_results.to_csv(results_csv_path)
    print(f"Saved results table to: {results_csv_path}")

    print("\nTest Set Performance vs. Augmentation Ratio (Pure Latent Space):")
    print(df_results[['n_total_train', 'n_synthetic', 'features', 'best_lv', 'test_auc', 'test_acc', 'test_sens',
                      'test_spec']].to_string(float_format="%.4f"))

    # --- Plot key metrics ---
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    fig.suptitle("PLS-DA Test Performance vs. Augmentation Ratio (Pure Latent Space)", fontsize=16)

    ratios_str = df_results.index.values

    axes[0, 0].plot(ratios_str, df_results['test_auc'], 'o-', label="Test AUC")
    axes[0, 0].set_title("Test AUC")
    axes[0, 0].grid(True, linestyle='--')

    axes[0, 1].plot(ratios_str, df_results['test_acc'], 'o-', label="Test Accuracy", color='tab:green')
    axes[0, 1].set_title("Test Accuracy")
    axes[0, 1].grid(True, linestyle='--')

    axes[1, 0].plot(ratios_str, df_results['test_sens'], 'o-', label="Test Sensitivity (Cancer)", color='tab:red')
    axes[1, 0].set_title("Test Sensitivity")
    axes[1, 0].set_xlabel("Augmentation Ratio")
    axes[1, 0].grid(True, linestyle='--')

    axes[1, 1].plot(ratios_str, df_results['test_spec'], 'o-', label="Test Specificity (Healthy)", color='tab:blue')
    axes[1, 1].set_title("Test Specificity")
    axes[1, 1].set_xlabel("Augmentation Ratio")
    axes[1, 1].grid(True, linestyle='--')

    for ax in axes.flat:
        ax.set_ylim(bottom=max(0.0, df_results[['test_auc', 'test_acc', 'test_sens', 'test_spec']].min().min() - 0.1),
                    top=1.0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path = OUT_DIR / "augmentation_metrics_plot_pure_latent.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\nSaved metrics plot to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()