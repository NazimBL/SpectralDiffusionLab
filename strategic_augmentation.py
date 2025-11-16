#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script tests a "Clean Spectrum" + "Balance-then-Augment" strategy
using an XGBoost classifier.

1.  Real Data is preprocessed: Raw -> 2nd-Derivative -> Clean Spectra
2.  Synthetic Data is generated: DDPM -> Decode -> Clean Spectra
3.  The classifier is trained purely on these (N, 235) clean spectra.
"""

import math, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from scipy.signal import savgol_filter
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings

# ==================================
# ======== MAIN CONFIG ========
# ==================================
AUGMENT_RATIOS = [0.0, 0.2,0.4, 0.6,0.8, 1,1.5, 2.0]
TRAIN_CSV = Path(r"MyDataset/ftir_train_wn.csv")
TEST_CSV = Path(r"MyDataset/ftir_test_wn.csv")
LDM_DIR = Path("ldm_out")
AE_WEIGHTS_FILE = LDM_DIR / "ae_conv1d.pt"
DDPM_CHECKPOINT_FILE = LDM_DIR / "ddpm_latent_unet.pt"
AE_META_FILE = LDM_DIR / "ae_meta.json"
OUT_DIR = Path(r"Augmentation_Results_XGBoost")
SAMPLE_STEPS = 1000
SEED = 42
GENERATOR_BATCH_SIZE = 64  # Batch size for generating new samples
GUIDANCE_SCALE = 0.5
LATENT_C_MODEL = 12  # Must match train_ae.py
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
# PART 1: preprocessing functions
# =================================================================

def preprocess_row(x_row: np.ndarray) -> np.ndarray:
    """ Must match train_ae.py """
    win = 5 if x_row.size >= 5 else (x_row.size // 2 * 2 + 1)
    if win % 2 == 0: win += 1
    z = savgol_filter(x_row, window_length=win, polyorder=2, deriv=2)
    n = np.linalg.norm(z) + 1e-12
    return (z / n).astype(np.float32)


def spectral_cols(df):
    cols = []
    for c in df.columns:
        if "class" in c or "group" in c or "obs" in c: continue
        try:
            float(c);
            cols.append(c)
        except:
            pass
    if not cols: raise ValueError("No wavenumber columns detected.")
    return [c for c in df.columns if c in set(cols)]


# =================================================================
# PART 2: GENERATIVE MODEL DEFINITIONS
# matches other script ... to be organized later sorry
# =================================================================

def gnorm(c):
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class ConvAE(nn.Module):
    def __init__(self, F: int, downs: int = 4, base: int = 64, latent_c: int = 12):
        super().__init__()
        self.F = F
        self.latent_c = latent_c
        c = base
        enc = []
        in_c = 1
        for i in range(downs):
            out_c = latent_c if i == downs - 1 else c
            enc += [
                nn.Conv1d(in_c, c, kernel_size=5, stride=1, padding=2),
                nn.SiLU(),
                nn.Conv1d(c, out_c, kernel_size=5, stride=2, padding=2),
                nn.SiLU(),
            ]
            in_c = out_c
            c = min(c * 2, 256)
        self.encoder = nn.Sequential(*enc)
        with torch.no_grad():
            probe = torch.zeros(1, 1, F)
            feat = self.encoder(probe)
            self.latent_L = feat.shape[2]
        dec = []
        c_cur = self.latent_c
        for i in range(downs):
            c_mid = max(c_cur // 2, base) if i < downs - 1 else base
            c_out = base if i < downs - 1 else 32
            dec += [
                nn.ConvTranspose1d(c_cur, c_mid, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv1d(c_mid, c_out, kernel_size=5, padding=2),
                nn.SiLU(),
            ]
            c_cur = c_out
        self.decoder = nn.Sequential(*dec)
        self.to_raw = nn.Conv1d(c_cur, 1, kernel_size=3, padding=1)

    def decode(self, z):
        y = self.decoder(z)
        if y.shape[-1] != self.F:
            if y.shape[-1] > self.F:
                start = (y.shape[-1] - self.F) // 2
                y = y[..., start:start + self.F]
            else:
                pad = self.F - y.shape[-1]
                y = Fnn.pad(y, (pad // 2, pad - pad // 2))
        return self.to_raw(y)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=128, max_period=10000.0):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)
        self.dim = dim

    def forward(self, t):
        t = t.float().unsqueeze(1)
        ang = t * self.freqs.unsqueeze(0)
        return torch.cat([ang.sin(), ang.cos()], dim=1)


class ClassEmbedding(nn.Module):
    def __init__(self, num_classes=2, dim=32):
        super().__init__()
        self.emb = nn.Embedding(num_classes + 1, dim)
        self.num_classes = num_classes

    def forward(self, y):
        return self.emb(y)


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.gn1 = gnorm(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.gn2 = gnorm(cout)
        self.act = nn.SiLU()
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, cout))
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cvec):
        h = self.act(self.gn1(self.conv1(x)))
        b = self.cond(cvec).unsqueeze(-1)
        h = self.conv2(h)
        h = self.gn2(h + b)
        h = self.act(h)
        return h + self.skip(x)


class UNet1D_Cond(nn.Module):
    def __init__(self, in_ch=12, base=128, out_ch=12, time_dim=128, class_dim=32, num_classes=2):
        super().__init__()
        self.temb = SinusoidalTimeEmbedding(time_dim)
        self.yemb = ClassEmbedding(num_classes=num_classes, dim=class_dim)
        self.null_class_idx = num_classes
        self.proj = nn.Sequential(nn.Linear(time_dim + class_dim, base), nn.SiLU())
        cond_dim = base
        self.rb1 = ResBlock1D(in_ch, base, cond_dim)
        self.down1 = nn.Conv1d(base, base, 4, stride=2, padding=1)
        self.rb2 = ResBlock1D(base, base * 2, cond_dim)
        self.down2 = nn.Conv1d(base * 2, base * 2, 4, stride=2, padding=1)
        self.mid1 = ResBlock1D(base * 2, base * 4, cond_dim)
        self.mid2 = ResBlock1D(base * 4, base * 4, cond_dim)
        self.up2_conv = nn.Conv1d(base * 4, base * 2, 1)
        self.rb_up2a = ResBlock1D(base * 2 + base * 2, base * 2, cond_dim)
        self.rb_up2b = ResBlock1D(base * 2, base * 2, cond_dim)
        self.up1_conv = nn.Conv1d(base * 2, base, 1)
        self.rb_up1a = ResBlock1D(base + base, base, cond_dim)
        self.rb_up1b = ResBlock1D(base, base, cond_dim)
        self.head = nn.Conv1d(base, out_ch, 3, padding=1)

    def forward(self, zt, t, y):
        c = torch.cat([self.temb(t), self.yemb(y)], dim=1)
        c = self.proj(c)
        h1 = self.rb1(zt, c)
        x = self.down1(h1)
        h2 = self.rb2(x, c)
        x = self.down2(h2)
        x = self.mid1(x, c)
        x = self.mid2(x, c)
        x = Fnn.interpolate(x, size=h2.shape[-1], mode="linear", align_corners=False)
        x = self.up2_conv(x)
        x = torch.cat([x, h2], dim=1)
        x = self.rb_up2a(x, c)
        x = self.rb_up2b(x, c)
        x = Fnn.interpolate(x, size=h1.shape[-1], mode="linear", align_corners=False)
        x = self.up1_conv(x)
        x = torch.cat([x, h1], dim=1)
        x = self.rb_up1a(x, c)
        x = self.rb_up1b(x, c)
        return self.head(x)


def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps, dtype=torch.float32)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return betas.clamp(1e-8, 0.999)


# =================================================================
# PART 3: GENERATOR FUNCTION (SAMPLER)
# =================================================================

@torch.no_grad()
def generate_clean_spectra(unet, ae, z_mu, z_std, z_tr_std, T_trained, steps, y_class, n, w):
    """
    This is the full generative pipeline.
    It runs p_sample_loop and decodes the result.
    """
    betas_s = cosine_beta_schedule(T_trained).to(DEVICE)
    alphas_s = 1.0 - betas_s
    ac_s = torch.cumprod(alphas_s, dim=0)
    sqrt_recip_alphas = (1.0 / torch.sqrt(alphas_s)).to(DEVICE)

    latent_c = unet.rb1.conv1.in_channels
    latent_L = ae.latent_L  # Get L from the AE

    # Scale the initial noise
    z_t = torch.randn(n, latent_c, latent_L, device=DEVICE) * z_tr_std

    y_cond = torch.full((n,), int(y_class), device=DEVICE, dtype=torch.long)
    y_uncond = torch.full((n,), unet.null_class_idx, device=DEVICE, dtype=torch.long)

    ts = torch.linspace(T_trained - 1, 0, steps, dtype=torch.long, device=DEVICE)
    for t_val in ts:
        t = t_val.repeat(n)
        eps_cond = unet(z_t, t, y_cond)
        eps_uncond = unet(z_t, t, y_uncond)
        eps_hat = eps_uncond + w * (eps_cond - eps_uncond)
        beta_t = betas_s[t].view(-1, 1, 1)
        sqrt_one_minus_ac_t = torch.sqrt(1.0 - ac_s[t]).view(-1, 1, 1)
        sqrt_recip_alpha_t = sqrt_recip_alphas[t].view(-1, 1, 1)
        mean = sqrt_recip_alpha_t * (z_t - beta_t / sqrt_one_minus_ac_t * eps_hat)
        if (t_val > 0):
            noise = torch.randn_like(z_t)
            z_t = mean + torch.sqrt(beta_t) * noise
        else:
            z_t = mean

    z_final_norm = z_t

    # Un-normalize the latent
    z_t_unnorm = z_final_norm * z_std + z_mu

    # Decode the un-normalized latent to a clean spectrum
    x_clean = ae.decode(z_t_unnorm)

    return x_clean.squeeze(1).detach().cpu().numpy()  # (n, F)


# =================================================================
# PART 4: CLASSIFIER FUNCTION
# =================================================================

def run_classifier_analysis(X_tr: np.ndarray, y_tr: np.ndarray,
                            X_te: np.ndarray, y_te: np.ndarray,
                            strategy_name: str, n_syn_h: int, n_syn_c: int):
    n_total_train = X_tr.shape[0]
    n_syn_total = n_syn_h + n_syn_c

    print(f"  Training XGBClassifier (N={n_total_train}, F={X_tr.shape[1]})...")

    n_healthy = np.sum(y_tr == 0)
    n_cancer = np.sum(y_tr == 1)
    weight = n_healthy / (n_cancer + 1e-6)

    model = XGBClassifier(
        random_state=SEED,
        scale_pos_weight=weight,
        n_estimators=200,
        early_stopping_rounds=20,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    prob_te = model.predict_proba(X_te)[:, 1]
    yhat_te = model.predict(X_te)

    auc = roc_auc_score(y_te, prob_te)
    acc = accuracy_score(y_te, yhat_te)
    cm = confusion_matrix(y_te, yhat_te, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(
        f"  Test Results (Strategy '{strategy_name}'): AUC={auc:.4f}, Acc={acc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}")

    return {
        "strategy": strategy_name,
        "n_total_train": n_total_train,
        "n_syn_healthy": n_syn_h,
        "n_syn_cancer": n_syn_c,
        "features": X_tr.shape[1],
        "test_auc": auc,
        "test_acc": acc,
        "test_sens": sens,
        "test_spec": spec,
    }


# =================================================================
# PART 5: MAIN EXPERIMENT SCRIPT
# =================================================================

def main():
    print("Starting 'Clean Spectrum' + 'Balance-then-Augment' experiment (with XGBoost)...")

    # --- 1. Original Data ---
    if not (TRAIN_CSV.exists() and TEST_CSV.exists() and AE_META_FILE.exists()):
        print("Error: Missing one or more input files.")
        return

    df_tr_orig = pd.read_csv(TRAIN_CSV)
    df_te = pd.read_csv(TEST_CSV)

    with open(AE_META_FILE, "r") as f:
        meta = json.load(f)
    spec_cols = meta["cols"]
    F_LEN = int(meta["F"])

    X_tr_orig_raw = df_tr_orig[spec_cols].to_numpy(dtype=np.float32)
    y_tr_orig = (df_tr_orig["classes"].values != 0).astype(int)

    X_te_raw = df_te[spec_cols].to_numpy(dtype=np.float32)
    y_te = (df_te["classes"].values != 0).astype(int)

    n_orig_healthy = np.sum(y_tr_orig == 0)
    n_orig_cancer = np.sum(y_tr_orig == 1)
    n_to_balance = n_orig_cancer - n_orig_healthy
    print(
        f"Loaded original train data: {n_orig_healthy} Healthy, {n_orig_cancer} Cancer. (Imbalance: {n_to_balance} samples)")
    print(f"Loaded original test data: {np.sum(y_te == 0)} Healthy, {np.sum(y_te == 1)} Cancer. Total: {len(y_te)}")

    # --- 2. Load Generative Models & configs ---
    print("Loading generative models...")
    ckpt = torch.load(DDPM_CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)

    downs = int(meta["downs"])
    latent_c = int(meta["latent_channels"])

    if latent_c != LATENT_C_MODEL:
        print(f"ERROR: Model mismatch. Script config LATENT_C={LATENT_C_MODEL}, but ae_meta.json has {latent_c}")
        return

    T_trained = int(ckpt["T"])

    # Latent Stats
    z_mu = ckpt["z_mu"].to(DEVICE)
    z_std = ckpt["z_std"].to(DEVICE)

    # We need the std of the *normalized* training latents for the sampler
    # We load latent_train.pt just for this one value
    try:
        tr_latents = torch.load(LDM_DIR / "latent_train.pt", map_location=DEVICE, weights_only=False)
        z_tr_norm_std = ((tr_latents['z'] - z_mu) / z_std.clamp(1e-6)).std()
        print(f"Loaded normalized latent std: {z_tr_norm_std.item():.4f}")
    except Exception as e:
        print(f"Error loading latent_train.pt to get std. Did you run cache_latents.py?")
        print(f"Error: {e}")
        return

    ae = ConvAE(F_LEN, downs=downs, latent_c=latent_c).to(DEVICE)
    ae_sd = torch.load(AE_WEIGHTS_FILE, map_location=DEVICE, weights_only=False)
    ae.load_state_dict(ae_sd, strict=False)
    ae.eval()

    unet = UNet1D_Cond(in_ch=latent_c, base=128, out_ch=latent_c).to(DEVICE)
    unet.load_state_dict(ckpt["model"])
    unet.eval()
    print("All models loaded.")

    # --- 3. Preprocess ALL Data ---
    print("Preprocessing all data to 2nd-Derivative domain...")
    X_tr_orig_clean = np.vstack([preprocess_row(r) for r in X_tr_orig_raw]).astype(np.float32)
    X_te_clean = np.vstack([preprocess_row(r) for r in X_te_raw]).astype(np.float32)
    print(f"Train data preprocessed to shape: {X_tr_orig_clean.shape}")
    print(f"Test data preprocessed to shape: {X_te_clean.shape}")

    all_results = []

    # --- 4. Run ORIGINAL Imbalanced Baseline ---
    print("\n" + "=" * 50)
    print("RUNNING EXPERIMENT: Strategy 'Original_Imbalanced'")
    print("=" * 50)
    results = run_classifier_analysis(
        X_tr_orig_clean, y_tr_orig,
        X_te_clean, y_te,
        "Orig_Imbalanced", 0, 0
    )
    all_results.append(results)

    # --- 5. Create the new "Base Balanced Set" ---
    print("\n" + "=" * 50)
    print("Creating Base Balanced Dataset")
    print("=" * 50)
    print(f"  Generating {n_to_balance} 'Healthy' clean spectra...")

    X_gen_balance_h = generate_clean_spectra(
        unet=unet, ae=ae, z_mu=z_mu, z_std=z_std, z_tr_std=z_tr_norm_std,
        T_trained=T_trained, steps=min(SAMPLE_STEPS, T_trained),
        y_class=0, n=n_to_balance, w=GUIDANCE_SCALE
    )
    y_gen_balance_h = np.zeros(n_to_balance, dtype=int)

    # This is our new "base" training set of preprocessed spectra
    X_tr_base_clean = np.vstack([X_tr_orig_clean, X_gen_balance_h])
    y_tr_base = np.hstack([y_tr_orig, y_gen_balance_h])

    n_base_size = len(y_tr_base)
    n_base_healthy = np.sum(y_tr_base == 0)
    n_base_cancer = np.sum(y_tr_base == 1)

    print(f"  New base training set: {n_base_healthy} H, {n_base_cancer} C (Total: {n_base_size})")

    # --- 6. Run Experiment Loop (starting from balanced set) ---
    for ratio in AUGMENT_RATIOS:
        strategy_name = f"Balanced + {ratio * 100:.0f}%"
        print("\n" + "=" * 50)
        print(f"RUNNING EXPERIMENT: Strategy '{strategy_name}'")
        print("=" * 50)

        if ratio == 0.0:
            X_tr_aug_clean = X_tr_base_clean
            y_tr_aug = y_tr_base
            n_syn_h, n_syn_c = n_to_balance, 0
            print(f"  Using 0% additional data (Balanced Baseline).")
        else:
            n_base_class_size = n_base_cancer
            n_syn_per_class = int(n_base_class_size * ratio)

            # Batch generation to avoid OOM
            X_gen_healthy, X_gen_cancer = [], []
            n_left_h = n_syn_per_class
            n_left_c = n_syn_per_class

            print(f"  Generating {n_syn_per_class} new healthy and {n_syn_per_class} new cancer spectra...")
            while n_left_h > 0 or n_left_c > 0:
                if n_left_h > 0:
                    n_batch_h = min(n_left_h, GENERATOR_BATCH_SIZE)
                    X_batch_h = generate_clean_spectra(
                        unet=unet, ae=ae, z_mu=z_mu, z_std=z_std, z_tr_std=z_tr_norm_std,
                        T_trained=T_trained, steps=min(SAMPLE_STEPS, T_trained),
                        y_class=0, n=n_batch_h, w=GUIDANCE_SCALE
                    )
                    X_gen_healthy.append(X_batch_h)
                    n_left_h -= n_batch_h

                if n_left_c > 0:
                    n_batch_c = min(n_left_c, GENERATOR_BATCH_SIZE)
                    X_batch_c = generate_clean_spectra(
                        unet=unet, ae=ae, z_mu=z_mu, z_std=z_std, z_tr_std=z_tr_norm_std,
                        T_trained=T_trained, steps=min(SAMPLE_STEPS, T_trained),
                        y_class=1, n=n_batch_c, w=GUIDANCE_SCALE
                    )
                    X_gen_cancer.append(X_batch_c)
                    n_left_c -= n_batch_c

            X_gen_healthy_all = np.vstack(X_gen_healthy)
            X_gen_cancer_all = np.vstack(X_gen_cancer)
            y_gen_healthy = np.zeros(n_syn_per_class, dtype=int)
            y_gen_cancer = np.ones(n_syn_per_class, dtype=int)

            X_tr_aug_clean = np.vstack([X_tr_base_clean, X_gen_healthy_all, X_gen_cancer_all])
            y_tr_aug = np.hstack([y_tr_base, y_gen_healthy, y_gen_cancer])

            n_syn_h = n_to_balance + n_syn_per_class
            n_syn_c = n_syn_per_class

        n_h_total = np.sum(y_tr_aug == 0)
        n_c_total = np.sum(y_tr_aug == 1)
        print(f"  New training set: {n_h_total} H, {n_c_total} C (Total: {len(y_tr_aug)})")
        print(f"  Training data shape: {X_tr_aug_clean.shape}")

        results = run_classifier_analysis(
            X_tr_aug_clean, y_tr_aug,
            X_te_clean, y_te,
            strategy_name, n_syn_h, n_syn_c
        )
        all_results.append(results)

    # --- 7. Final Report ---
    print("\n" + "=" * 60)
    print("     CLEAN SPECTRUM 'BALANCE-THEN-AUGMENT' EXPERIMENT (XGBOOST)")
    print("=" * 60)

    df_results = pd.DataFrame(all_results)
    df_results.set_index("strategy", inplace=True)

    results_csv_path = OUT_DIR / "augmentation_results_clean_spectrum_xgboost.csv"
    df_results.to_csv(results_csv_path)
    print(f"Saved results table to: {results_csv_path}")

    print("\nTest Set Performance vs. Augmentation Strategy (Clean Spectrum, XGBoost):")
    print(df_results[['n_total_train', 'n_syn_healthy', 'n_syn_cancer', 'features', 'test_auc', 'test_acc', 'test_sens',
                      'test_spec']].to_string(float_format="%.4f"))

    # --- Plots ---
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    fig.suptitle("XGBClassifier Test Performance: Balance-then-Augment Strategy (Clean Spectrum)", fontsize=16)

    strategies_str = df_results.index.values

    axes[0, 0].plot(strategies_str, df_results['test_auc'], 'o-', label="Test AUC")
    axes[0, 0].set_title("Test AUC")
    axes[0, 0].grid(True, linestyle='--')

    axes[0, 1].plot(strategies_str, df_results['test_acc'], 'o-', label="Test Accuracy", color='tab:green')
    axes[0, 1].set_title("Test Accuracy")
    axes[0, 1].grid(True, linestyle='--')

    axes[1, 0].plot(strategies_str, df_results['test_sens'], 'o-', label="Test Sensitivity (Cancer)", color='tab:red')
    axes[1, 0].set_title("Test Sensitivity")
    axes[1, 0].set_xlabel("Augmentation Strategy")
    axes[1, 0].grid(True, linestyle='--')

    axes[1, 1].plot(strategies_str, df_results['test_spec'], 'o-', label="Test Specificity (Healthy)", color='tab:blue')
    axes[1, 1].set_title("Test Specificity")
    axes[1, 1].set_xlabel("Augmentation Strategy")
    axes[1, 1].grid(True, linestyle='--')

    min_y = max(0.0, df_results[['test_auc', 'test_acc', 'test_sens', 'test_spec']].min().min() - 0.1)
    max_y = min(1.0, df_results[['test_auc', 'test_acc', 'test_sens', 'test_spec']].max().max() + 0.05)

    for ax in axes.flat:
        ax.set_ylim(bottom=min_y, top=max_y)
        ax.tick_params(axis='x', rotation=25)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path = OUT_DIR / "augmentation_metrics_plot_clean_spectrum_xgboost.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\nSaved metrics plot to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()