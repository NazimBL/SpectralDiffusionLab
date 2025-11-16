#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script runs a focused experiment to compare three balancing strategies:
1.  Orig_Imbalanced: Real, imbalanced data (168H, 241C)
2.  Real_Balanced_Undersample: Real, undersampled data (168H, 168C)
3.  Synthetic_Balanced: Real data + synthetic balancing data (241H, 241C)

This will isolate the effect of balancing from augmentation.
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
TRAIN_CSV = Path(r"MyDataset/ftir_train_wn.csv")
TEST_CSV = Path(r"MyDataset/ftir_test_wn.csv")
LDM_DIR = Path("ldm_out")
AE_WEIGHTS_FILE = LDM_DIR / "ae_conv1d.pt"
DDPM_CHECKPOINT_FILE = LDM_DIR / "ddpm_latent_unet.pt"
AE_META_FILE = LDM_DIR / "ae_meta.json"
OUT_DIR = Path(r"Balancing_Experiment_Results")
SAMPLE_STEPS = 1000
SEED = 42
GUIDANCE_SCALE = 5
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
RNG = np.random.default_rng(SEED)


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
# PART 2: Generative model definitions
# =================================================================

def gnorm(c):
    return nn.GroupNorm(num_groups=min(4, c), num_channels=c)  # 4 groups for 12 channels


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
    latent_L = ae.latent_L

    # *** FINAL FIX: Scale the initial noise ***
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
    z_t_unnorm = z_final_norm * z_std + z_mu
    x_clean = ae.decode(z_t_unnorm)

    return x_clean.squeeze(1).detach().cpu().numpy()  # (n, F)


# =================================================================
# PART 4: CLASSIFIER FUNCTION
# =================================================================

def run_classifier_analysis(X_tr: np.ndarray, y_tr: np.ndarray,
                            X_te: np.ndarray, y_te: np.ndarray,
                            strategy_name: str, n_train_h: int, n_train_c: int):
    """
    Trains and evaluates a robust XGBoost classifier.
    """
    n_total_train = X_tr.shape[0]

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
        "n_train_h": n_train_h,
        "n_train_c": n_train_c,
        "features": X_tr.shape[1],
        "test_auc": auc,
        "test_acc": acc,
        "test_sens": sens,
        "test_spec": spec,
    }


# =================================================================
# PART 5: MAIN EXPERIMENT
# =================================================================

def main():
    print("Starting Balancing Strategies Comparison (XGBoost)...")

    # --- 1. Original Data ---
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

    n_orig_healthy = np.sum(y_tr_orig == 0)  # 168
    n_orig_cancer = np.sum(y_tr_orig == 1)  # 241
    n_to_balance = n_orig_cancer - n_orig_healthy  # 73
    print(
        f"Loaded original train data: {n_orig_healthy} Healthy, {n_orig_cancer} Cancer. (Imbalance: {n_to_balance} samples)")

    # --- 2. Load Generative Models & configs ---
    print("Loading generative models...")
    ckpt = torch.load(DDPM_CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)

    downs = int(meta["downs"])
    latent_c = int(meta["latent_channels"])

    if latent_c != LATENT_C_MODEL:
        print(f"ERROR: Model mismatch. Script config LATENT_C={LATENT_C_MODEL}, but ae_meta.json has {latent_c}")
        return

    T_trained = int(ckpt["T"])
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

    all_results = []

    # --- STRATEGY 2: Original Imbalanced (168 H, 241 C) ---
    print("\n" + "=" * 50)
    print("RUNNING EXPERIMENT: Strategy 'Orig_Imbalanced'")
    print("=" * 50)
    results = run_classifier_analysis(
        X_tr_orig_clean, y_tr_orig,
        X_te_clean, y_te,
        "Orig_Imbalanced", n_orig_healthy, n_orig_cancer
    )
    all_results.append(results)

    # --- STRATEGY 1: Real_Balanced_Undersample (168 H, 168 C) ---
    print("\n" + "=" * 50)
    print("RUNNING EXPERIMENT: Strategy 'Real_Balanced_Undersample'")
    print("=" * 50)

    # Get all healthy samples
    X_h_real = X_tr_orig_clean[y_tr_orig == 0]
    y_h_real = y_tr_orig[y_tr_orig == 0]

    # Get all cancer samples
    X_c_real = X_tr_orig_clean[y_tr_orig == 1]
    y_c_real = y_tr_orig[y_tr_orig == 1]

    # Randomly select 168 cancer samples
    n_healthy = len(y_h_real)
    cancer_indices = RNG.choice(len(y_c_real), size=n_healthy, replace=False)

    X_c_real_under = X_c_real[cancer_indices]
    y_c_real_under = y_c_real[cancer_indices]

    # Combine them
    X_tr_under = np.vstack([X_h_real, X_c_real_under])
    y_tr_under = np.hstack([y_h_real, y_c_real_under])

    print(f"  Created undersampled dataset: {len(y_h_real)} H, {len(y_c_real_under)} C")

    results = run_classifier_analysis(
        X_tr_under, y_tr_under,
        X_te_clean, y_te,
        "Real_Balanced_Undersample", len(y_h_real), len(y_c_real_under)
    )
    all_results.append(results)

    # --- STRATEGY 3: Synthetic_Balanced (241 H, 241 C) ---
    print("\n" + "=" * 50)
    print("RUNNING EXPERIMENT: Strategy 'Synthetic_Balanced (Balanced + 0%)'")
    print("=" * 50)
    print(f"  Generating {n_to_balance} 'Healthy' clean spectra to balance data...")

    X_gen_balance_h = generate_clean_spectra(
        unet=unet, ae=ae, z_mu=z_mu, z_std=z_std, z_tr_std=z_tr_norm_std,
        T_trained=T_trained, steps=min(SAMPLE_STEPS, T_trained),
        y_class=0, n=n_to_balance, w=GUIDANCE_SCALE
    )
    y_gen_balance_h = np.zeros(n_to_balance, dtype=int)

    # This is the "base" training set of preprocessed spectra
    X_tr_syn_balanced = np.vstack([X_tr_orig_clean, X_gen_balance_h])
    y_tr_syn_balanced = np.hstack([y_tr_orig, y_gen_balance_h])

    n_h_total = np.sum(y_tr_syn_balanced == 0)  # Should be 241
    n_c_total = np.sum(y_tr_syn_balanced == 1)  # Should be 241
    print(f"  New synthetically balanced set: {n_h_total} H, {n_c_total} C")

    results = run_classifier_analysis(
        X_tr_syn_balanced, y_tr_syn_balanced,
        X_te_clean, y_te,
        "Synthetic_Balanced", n_h_total, n_c_total
    )
    all_results.append(results)

    # --- 7. Final Report ---
    print("\n" + "=" * 60)
    print("     BALANCING STRATEGY COMPARISON (XGBOOST)")
    print("=" * 60)

    df_results = pd.DataFrame(all_results)
    df_results.set_index("strategy", inplace=True)

    results_csv_path = OUT_DIR / "balancing_strategy_comparison.csv"
    df_results.to_csv(results_csv_path)
    print(f"Saved results table to: {results_csv_path}")

    print("\nTest Set Performance vs. Balancing Strategy:")
    print(df_results[['n_train_h', 'n_train_c', 'features', 'test_auc', 'test_acc', 'test_sens',
                      'test_spec']].to_string(float_format="%.4f"))

    # --- Plots ---
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    fig.suptitle("XGBClassifier Test Performance: Balancing Strategy Comparison", fontsize=16)

    strategies_str = df_results.index.values

    axes[0, 0].bar(strategies_str, df_results['test_auc'], color='tab:blue')
    axes[0, 0].set_title("Test AUC")
    axes[0, 0].grid(True, linestyle='--', axis='y')

    axes[0, 1].bar(strategies_str, df_results['test_acc'], color='tab:green')
    axes[0, 1].set_title("Test Accuracy")
    axes[0, 1].grid(True, linestyle='--', axis='y')

    axes[1, 0].bar(strategies_str, df_results['test_sens'], color='tab:red')
    axes[1, 0].set_title("Test Sensitivity (Cancer)")
    axes[1, 0].set_xlabel("Strategy")
    axes[1, 0].grid(True, linestyle='--', axis='y')

    axes[1, 1].bar(strategies_str, df_results['test_spec'], color='tab:purple')
    axes[1, 1].set_title("Test Specificity (Healthy)")
    axes[1, 1].set_xlabel("Strategy")
    axes[1, 1].grid(True, linestyle='--', axis='y')

    min_y = max(0.0, df_results[['test_auc', 'test_acc', 'test_sens', 'test_spec']].min().min() - 0.1)
    max_y = min(1.0, df_results[['test_auc', 'test_acc', 'test_sens', 'test_spec']].max().max() + 0.05)

    for ax in axes.flat:
        if min_y < max_y:
            ax.set_ylim(bottom=min_y, top=max_y)
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path = OUT_DIR / "balancing_strategy_comparison.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\nSaved metrics plot to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()