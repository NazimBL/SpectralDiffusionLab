#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script tests the "Balance-then-Augment" strategy using
a SIMPLE Logistic Regression classifier to isolate the
effect of the augmentation.

It uses the Savgol-filter preprocessing.
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
from sklearn.linear_model import LogisticRegression  # <-- NEW MODEL
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings

# ==================================
# ======== MAIN CONFIGURATION ========
# ==================================

# --- Ratios to add *after* balancing ---
AUGMENT_RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]

# --- Paths ---
TRAIN_CSV = Path(r"../MyDataset/ftir_train_wn.csv")
TEST_CSV = Path(r"../MyDataset/ftir_test_wn.csv")
LDM_DIR = Path("ldm_out")
AE_WEIGHTS_FILE = LDM_DIR / "ae_conv1d.pt"
DDPM_CHECKPOINT_FILE = LDM_DIR / "ddpm_latent_unet.pt"  # Assumes (lambda=0.05) model
AE_META_FILE = LDM_DIR / "ae_meta.json"
OUT_DIR = Path(r"Augmentation_Results_Logit_Balanced")  # New output folder

# --- Model Config ---
SAMPLE_STEPS = 1000
SEED = 42
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
# PART 1: PREPROCESSING CODE
# =================================================================

META_COLS = {"groupnumbers", "classes", "class_name", "binary_label", "groupcodes", "obsnames"}


def detect_spectral_cols(df: pd.DataFrame) -> List[str]:
    spec = []
    for c in df.columns:
        if c in META_COLS: continue
        try:
            float(c); spec.append(c)
        except Exception:
            pass
    if not spec:
        spec = [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
    if not spec: raise ValueError("No spectral columns detected.")
    return [c for c in df.columns if c in set(spec)]


def second_derivative_savgol(X: np.ndarray, window: int = 5, poly: int = 2) -> np.ndarray:
    if window % 2 == 0: window += 1
    return savgol_filter(X, window_length=window, polyorder=poly, deriv=2, axis=1)


def vector_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def preprocess_numpy_array(X_raw: np.ndarray) -> np.ndarray:
    Xd2 = second_derivative_savgol(X_raw, window=5, poly=2)
    Xn = vector_normalize_rows(Xd2)
    return Xn


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
# (All models are identical to the previous script)
# =================================================================

def gnorm(c): return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class ConvAE(nn.Module):
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

    def decode(self, h):
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
def p_sample_loop(unet, ae, T, steps, y_class, n,
                  latent_c, latent_L,
                  x_mean_t, x_std_t, z_mu_t, z_std_t):
    if n == 0:
        return np.array([], dtype=np.float32).reshape(0, F_LEN)

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

    z_gen_mean = z_t.mean(dim=0, keepdim=True)
    z_gen_std = z_t.std(dim=0, keepdim=True)
    z_t_renormed = (z_t - z_gen_mean) / z_gen_std.clamp(1e-6)

    z_t_unnorm = z_t_renormed * z_std_t + z_mu_t
    x_scaled = ae.decode(z_t_unnorm)
    x_raw = (x_scaled * x_std_t + x_mean_t).squeeze(1).detach().cpu().numpy()
    return x_raw


# =================================================================
# PART 3: MAIN EXPERIMENT SCRIPT (with new classifier)
# =================================================================

def run_lr_analysis(X_tr: np.ndarray, y_tr: np.ndarray,
                    X_te: np.ndarray, y_te: np.ndarray,
                    strategy_name: str, n_syn_h: int, n_syn_c: int):
    """
    Trains and evaluates a simple Logistic Regression model.
    """
    n_total_train = X_tr.shape[0]
    n_syn_total = n_syn_h + n_syn_c

    print(f"  Training Logistic Regression (N={n_total_train}, F={X_tr.shape[1]})...")

    # Use a standard, robust Logistic Regression (Ridge)
    # class_weight='balanced' helps the model internally handle any class imbalance
    model = LogisticRegression(
        penalty='l2',
        solver='liblinear',
        class_weight='balanced',
        C=1.0,  # Default regularization
        random_state=SEED,
        max_iter=1000
    )

    model.fit(X_tr, y_tr)

    # Evaluate on TEST
    prob_te = model.predict_proba(X_te)[:, 1]  # Probabilities for class 1
    yhat_te = model.predict(X_te)

    auc = roc_auc_score(y_te, prob_te)
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
        f"  Test Results (Strategy '{strategy_name}'): AUC={auc:.4f}, Acc={acc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}")

    return {
        "strategy": strategy_name,
        "n_total_train": n_total_train,
        "n_syn_healthy": n_syn_h,
        "n_syn_cancer": n_syn_c,
        "n_syn_total": n_syn_total,
        "features": X_tr.shape[1],
        "test_auc": auc,
        "test_acc": acc,
        "test_sens": sens,
        "test_spec": spec,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }


# Global F_LEN to be set in main
F_LEN = -1


def main():
    global F_LEN
    print("Starting Savgol Filter 'Balance-then-Augment' experiment (with Logistic Regression)...")

    # --- 1. Load Original Data ---
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

    # --- 2. Load Generative Models & Scalers ---
    print("Loading generative models...")
    ckpt = torch.load(DDPM_CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)
    gen_meta = ckpt["meta"]

    downs = int(gen_meta["downs"])
    latent_c = int(gen_meta["latent_channels"]);
    latent_L = int(gen_meta["latent_length"])
    T_trained = int(ckpt["T"])

    x_mean_t = torch.from_numpy(np.array(gen_meta["scaler_mean"], dtype=np.float32)).to(DEVICE).view(1, 1, -1)
    x_std_t = torch.from_numpy(np.array(gen_meta["scaler_std"], dtype=np.float32)).to(DEVICE).view(1, 1, -1)
    z_mu_t = ckpt["z_mu"].to(DEVICE)
    z_std_t = ckpt["z_std"].to(DEVICE)

    ae = ConvAE(F_LEN, downs=downs, latent_c=latent_c).to(DEVICE)
    ae_sd = torch.load(AE_WEIGHTS_FILE, map_location=DEVICE, weights_only=False)
    ae.load_state_dict(ae_sd, strict=False)
    ae.eval()

    unet = UNet1D_Cond(in_ch=latent_c, base=128, out_ch=latent_c).to(DEVICE)
    unet.load_state_dict(ckpt["model"])
    unet.eval()
    print("All models loaded.")

    # --- 3. Preprocess Test Data (Done once using Savgol) ---
    print("Preprocessing test data (Savgol filter)...")
    X_te_preproc = preprocess_numpy_array(X_te_raw)
    print(f"Test data preprocessed to shape: {X_te_preproc.shape}")

    all_results = []

    # --- 4. Run ORIGINAL IMbalanced Baseline (for comparison) ---
    print("\n" + "=" * 50)
    print("RUNNING EXPERIMENT: Strategy 'Original_Imbalanced'")
    print("=" * 50)
    X_tr_preproc = preprocess_numpy_array(X_tr_orig_raw)
    print(f"  Training set: {n_orig_healthy} H, {n_orig_cancer} C (Total: {len(y_tr_orig)})")
    results = run_lr_analysis(  # <-- NEW FUNCTION
        X_tr_preproc, y_tr_orig,
        X_te_preproc, y_te,
        "Orig_Imbalanced", 0, 0
    )
    all_results.append(results)

    # --- 5. Create the new "Base Balanced Set" ---
    print("\n" + "=" * 50)
    print("Creating Base Balanced Dataset")
    print("=" * 50)
    print(f"  Generating {n_to_balance} 'Healthy' samples to balance dataset...")
    x_gen_balance_h = p_sample_loop(
        unet=unet, ae=ae, T=T_trained, steps=min(SAMPLE_STEPS, T_trained),
        y_class=0, n=n_to_balance, latent_c=latent_c, latent_L=latent_L,
        x_mean_t=x_mean_t, x_std_t=x_std_t, z_mu_t=z_mu_t, z_std_t=z_std_t
    )
    y_gen_balance_h = np.zeros(n_to_balance, dtype=int)

    X_tr_base_raw = np.vstack([X_tr_orig_raw, x_gen_balance_h])
    y_tr_base = np.hstack([y_tr_orig, y_gen_balance_h])
    n_base_size = len(y_tr_base)  # This is our new baseline size (e.g., 482)
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
            # This is the "Balanced_Baseline" run
            X_tr_aug_raw = X_tr_base_raw
            y_tr_aug = y_tr_base
            n_syn_h, n_syn_c = n_to_balance, 0
            print(f"  Using 0% additional data (Balanced Baseline).")

        else:
            # Calculate new samples to add *in a balanced way*
            n_base_class_size = n_base_cancer
            n_syn_per_class = int(n_base_class_size * ratio)

            print(f"  Generating {n_syn_per_class} new healthy and {n_syn_per_class} new cancer samples...")

            x_gen_healthy = p_sample_loop(
                unet=unet, ae=ae, T=T_trained, steps=min(SAMPLE_STEPS, T_trained),
                y_class=0, n=n_syn_per_class, latent_c=latent_c, latent_L=latent_L,
                x_mean_t=x_mean_t, x_std_t=x_std_t, z_mu_t=z_mu_t, z_std_t=z_std_t
            )
            y_gen_healthy = np.zeros(n_syn_per_class, dtype=int)

            x_gen_cancer = p_sample_loop(
                unet=unet, ae=ae, T=T_trained, steps=min(SAMPLE_STEPS, T_trained),
                y_class=1, n=n_syn_per_class, latent_c=latent_c, latent_L=latent_L,
                x_mean_t=x_mean_t, x_std_t=x_std_t, z_mu_t=z_mu_t, z_std_t=z_std_t
            )
            y_gen_cancer = np.ones(n_syn_per_class, dtype=int)

            X_tr_aug_raw = np.vstack([X_tr_base_raw, x_gen_healthy, x_gen_cancer])
            y_tr_aug = np.hstack([y_tr_base, y_gen_healthy, y_gen_cancer])

            n_syn_h = n_to_balance + n_syn_per_class
            n_syn_c = n_syn_per_class

        n_h_total = np.sum(y_tr_aug == 0)
        n_c_total = np.sum(y_tr_aug == 1)
        print(f"  New training set: {n_h_total} H, {n_c_total} C (Total: {len(y_tr_aug)})")

        # --- Preprocess and Train (using Savgol) ---
        print("  Preprocessing augmented training data (Savgol filter)...")
        X_tr_aug_preproc = preprocess_numpy_array(X_tr_aug_raw)
        print(f"  Training data preprocessed to shape: {X_tr_aug_preproc.shape}")

        results = run_lr_analysis(  # <-- NEW FUNCTION
            X_tr_aug_preproc, y_tr_aug,
            X_te_preproc, y_te,
            strategy_name, n_syn_h, n_syn_c
        )
        all_results.append(results)

    # --- 7. Final Report ---
    print("\n" + "=" * 60)
    print("     SAVGOL 'BALANCE-THEN-AUGMENT' EXPERIMENT (LOGISTIC REGRESSION)")
    print("=" * 60)

    df_results = pd.DataFrame(all_results)
    df_results.set_index("strategy", inplace=True)

    results_csv_path = OUT_DIR / "augmentation_results_logit_balanced.csv"
    df_results.to_csv(results_csv_path)
    print(f"Saved results table to: {results_csv_path}")

    print("\nTest Set Performance vs. Augmentation Strategy (Savgol Filter, Logistic Regression):")
    print(df_results[['n_total_train', 'n_syn_healthy', 'n_syn_cancer', 'test_auc', 'test_acc', 'test_sens',
                      'test_spec']].to_string(float_format="%.4f"))

    # --- Plot key metrics ---
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    fig.suptitle("Logistic Regression Test Performance: Balance-then-Augment Strategy (Savgol)", fontsize=16)

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

    for ax in axes.flat:
        ax.set_ylim(bottom=max(0.0, df_results[['test_auc', 'test_acc', 'test_sens', 'test_spec']].min().min() - 0.1),
                    top=max(1.0, df_results[['test_auc', 'test_acc', 'test_sens', 'test_spec']].max().max() + 0.05))
        ax.tick_params(axis='x', rotation=25)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path = OUT_DIR / "augmentation_metrics_plot_logit_balanced.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\nSaved metrics plot to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()