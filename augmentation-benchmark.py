#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script runs a "Strategic Augmentation" experiment comparing LDM, cGAN, and SMOTE.

It uses the EXACT LDM implementation from the original strategic_augmentation.py,
and adds parallel comparisons for cGAN and SMOTE.

1. Starts with real, balanced data (undersampled majority).
2. Iteratively adds synthetic data at specific ratios using LDM, cGAN, and SMOTE.
3. Trains XGBoost and tracks metrics on held-out test set.
4. Plots trends for all three methods.
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
# --- NEW IMPORT FOR SMOTE ---
from imblearn.over_sampling import SMOTE
# ----------------------------
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

# --- NEW CONFIG FOR CGAN ---
GAN_GENERATOR_WEIGHTS = Path("gan_out/cgan_generator_final.pt")
# ---------------------------

OUT_DIR = Path(r"Strategic_Augmentation_Comparison")

# Ratios of synthetic data to add relative to the REAL BALANCED dataset size.
AUGMENT_RATIOS = [0.0, 0.4, 0.8,1, 1.5,2.0]

SAMPLE_STEPS = 500
SEED = 42
GUIDANCE_SCALE = 0.5
LATENT_C_MODEL = 12
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
# PART 1: Preprocessing (MATCHING ORIGINAL EXACTLY)
# =================================================================
def preprocess_row(x_row: np.ndarray) -> np.ndarray:
    """ Must match train_ae.py """
    win = 5 if x_row.size >= 5 else (x_row.size // 2 * 2 + 1)
    if win % 2 == 0: win += 1
    z = savgol_filter(x_row, window_length=win, polyorder=2, deriv=2)
    n = np.linalg.norm(z) + 1e-12
    return (z / n).astype(np.float32)


# =================================================================
# PART 2: LDM & AE Model Definitions (MATCHING ORIGINAL EXACTLY)
# =================================================================
def gnorm(c):
    return nn.GroupNorm(num_groups=min(4, c), num_channels=c)

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
        # --- CRITICAL FIX: THIS BLOCK WAS MISSING IN PREVIOUS ATTEMPT ---
        with torch.no_grad():
            probe = torch.zeros(1, 1, F)
            feat = self.encoder(probe)
            self.latent_L = feat.shape[2]
        # ----------------------------------------------------------------
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
# PART 3: LDM Sampler (MATCHING ORIGINAL EXACTLY)
# =================================================================
@torch.no_grad()
def generate_clean_spectra_ldm(unet, ae, z_mu, z_std, z_tr_std, T_trained, steps, y_class, n, w):
    betas_s = cosine_beta_schedule(T_trained).to(DEVICE)
    alphas_s = 1.0 - betas_s
    ac_s = torch.cumprod(alphas_s, dim=0)
    sqrt_recip_alphas = (1.0 / torch.sqrt(alphas_s)).to(DEVICE)

    latent_c = unet.rb1.conv1.in_channels
    # This line depends on the fix in ConvAE.__init__
    latent_L = ae.latent_L

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

    return x_clean.squeeze(1).detach().cpu().numpy()

# =================================================================
# PART 4: cGAN Model & Sampler (NEW)
# =================================================================
class cGAN_Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, feature_dim, model_dim=64):
        super(cGAN_Generator, self).__init__()
        self.feature_dim = feature_dim
        self.initial_feature_size = feature_dim // 16
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.latent_layer = nn.Sequential(
            nn.Linear(latent_dim + num_classes, model_dim * 4 * self.initial_feature_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.model = nn.Sequential(
            nn.ConvTranspose1d(model_dim * 4, model_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(model_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(model_dim * 2, model_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(model_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(model_dim, model_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(model_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(model_dim // 2, 1, kernel_size=4, stride=2, padding=1),
        )
    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        x = self.latent_layer(x)
        x = x.view(x.shape[0], -1, self.initial_feature_size)
        x = self.model(x)
        if x.shape[-1] != self.feature_dim:
             x = Fnn.interpolate(x, size=self.feature_dim, mode='linear', align_corners=False)
        return x

@torch.no_grad()
def generate_clean_spectra_gan(generator, latent_dim, feature_dim, y_class, n):
    z = torch.randn(n, latent_dim).to(DEVICE)
    labels = torch.full((n,), int(y_class), device=DEVICE, dtype=torch.long)
    gen_spectra = generator(z, labels)
    return gen_spectra.squeeze(1).cpu().numpy() # (n, F)

# =================================================================
# PART 5: Classifier Function
# =================================================================
def run_classifier_analysis(X_tr: np.ndarray, y_tr: np.ndarray,
                            X_te: np.ndarray, y_te: np.ndarray,
                            description: str):
    """ Trains XGBoost and returns metrics. """
    n_train = X_tr.shape[0]
    n_h = np.sum(y_tr == 0)
    n_c = np.sum(y_tr == 1)
    print(f"  -> Training Classifier on {n_train} samples ({n_h} H, {n_c} C)...")

    # Even though data is balanced, slight weighting can help stability
    weight = n_h / (n_c + 1e-6)

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

    print(f"     Results: AUC={auc:.4f}, Acc={acc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}")

    return {
        "description": description,
        "ratio": description.split(" ")[0], # extract ratio for plotting
        "n_train": n_train,
        "auc": auc,
        "acc": acc,
        "sens": sens,
        "spec": spec,
    }


# =================================================================
# PART 6: MAIN EXPERIMENT LOOP
# =================================================================
def main():
    print("Starting Strategic Augmentation Comparison (LDM vs cGAN vs SMOTE)...")

    # --- 1. Load & Preprocess Real Data ---
    df_tr_orig = pd.read_csv(TRAIN_CSV)
    df_te = pd.read_csv(TEST_CSV)

    with open(AE_META_FILE, "r") as f:
        meta = json.load(f)
    spec_cols = meta["cols"]
    F_LEN = int(meta["F"])

    X_tr_raw = df_tr_orig[spec_cols].to_numpy(dtype=np.float32)
    y_tr_orig = (df_tr_orig["classes"].values != 0).astype(int)
    X_te_raw = df_te[spec_cols].to_numpy(dtype=np.float32)
    y_te = (df_te["classes"].values != 0).astype(int)

    print("Preprocessing real data...")
    X_tr_clean = np.vstack([preprocess_row(r) for r in X_tr_raw]).astype(np.float32)
    X_te_clean = np.vstack([preprocess_row(r) for r in X_te_raw]).astype(np.float32)

    # --- 2. Create Real Balanced Baseline (Undersampling) ---
    n_h_orig = np.sum(y_tr_orig == 0)
    n_c_orig = np.sum(y_tr_orig == 1)
    n_min = min(n_h_orig, n_c_orig) # Should be 168 based on previous runs

    X_h = X_tr_clean[y_tr_orig == 0]
    y_h = y_tr_orig[y_tr_orig == 0]
    X_c = X_tr_clean[y_tr_orig == 1]
    y_c = y_tr_orig[y_tr_orig == 1]

    # Undersample majority to match minority
    idx_h = RNG.choice(len(X_h), size=n_min, replace=False)
    idx_c = RNG.choice(len(X_c), size=n_min, replace=False)

    X_tr_balanced = np.vstack([X_h[idx_h], X_c[idx_c]])
    y_tr_balanced = np.hstack([y_h[idx_h], y_c[idx_c]])
    n_bal_total = len(y_tr_balanced) # e.g., 168+168 = 336
    print(f"\nBaseline: Real Balanced Data created ({n_min} H, {n_min} C). Total: {n_bal_total}")

    # --- 3. Load Generative Models ---
    # LDM
    print("Loading LDM & AE models...")
    ckpt = torch.load(DDPM_CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)
    downs = int(meta["downs"])
    latent_c = int(meta["latent_channels"])
    T_trained = int(ckpt["T"])
    z_mu = ckpt["z_mu"].to(DEVICE)
    z_std = ckpt["z_std"].to(DEVICE)
    try:
        tr_latents = torch.load(LDM_DIR / "latent_train.pt", map_location=DEVICE, weights_only=False)
        z_tr_norm_std = ((tr_latents['z'] - z_mu) / z_std.clamp(1e-6)).std()
    except: return

    ae = ConvAE(F_LEN, downs=downs, latent_c=latent_c).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS_FILE, map_location=DEVICE, weights_only=False), strict=False)
    ae.eval()
    unet = UNet1D_Cond(in_ch=latent_c, base=128, out_ch=latent_c).to(DEVICE)
    unet.load_state_dict(ckpt["model"])
    unet.eval()

    # cGAN
    print("Loading cGAN model...")
    GAN_LATENT_DIM = 100; GAN_NUM_CLASSES = 2; GAN_MODEL_DIM = 64
    if not GAN_GENERATOR_WEIGHTS.exists(): return
    cgan_generator = cGAN_Generator(GAN_LATENT_DIM, GAN_NUM_CLASSES, F_LEN, GAN_MODEL_DIM).to(DEVICE)
    cgan_generator.load_state_dict(torch.load(GAN_GENERATOR_WEIGHTS, map_location=DEVICE, weights_only=True))
    cgan_generator.eval()

    # --- 4. Strategic Augmentation Loops ---
    results_ldm = []
    results_cgan = []
    results_smote = []

    # ---- LDM Loop (EXACTLY AS ORIGINAL) ----
    print("\n" + "="*30 + "\nStarting LDM Augmentation Loop\n" + "="*30)


    for ratio in AUGMENT_RATIOS:
        n_syn_to_add = int(n_bal_total * ratio)
        desc = f"{ratio} (+{n_syn_to_add} LDM)"
        print(f"\nRunning Ratio: {desc}")

        if ratio == 0.0:
            X_curr, y_curr = X_tr_balanced, y_tr_balanced
        else:
            n_per_class = n_syn_to_add // 2
            print(f"  Generating {n_per_class} H and {n_per_class} C with LDM...")
            # Generate Healthy (0)
            X_gen_h = generate_clean_spectra_ldm(unet, ae, z_mu, z_std, z_tr_norm_std, T_trained, min(SAMPLE_STEPS, T_trained), 0, n_per_class, GUIDANCE_SCALE)
            # Generate Cancer (1)
            X_gen_c = generate_clean_spectra_ldm(unet, ae, z_mu, z_std, z_tr_norm_std, T_trained, min(SAMPLE_STEPS, T_trained), 1, n_per_class, GUIDANCE_SCALE)

            X_curr = np.vstack([X_tr_balanced, X_gen_h, X_gen_c])
            y_curr = np.hstack([y_tr_balanced, np.zeros(n_per_class), np.ones(n_per_class)])

        res = run_classifier_analysis(X_curr, y_curr, X_te_clean, y_te, desc)
        results_ldm.append(res)

    # ---- cGAN Loop (NEW) ----
    print("\n" + "="*30 + "\nStarting cGAN Augmentation Loop\n" + "="*30)
    for ratio in AUGMENT_RATIOS:
        n_syn_to_add = int(n_bal_total * ratio)
        desc = f"{ratio} (+{n_syn_to_add} cGAN)"
        print(f"\nRunning Ratio: {desc}")

        if ratio == 0.0:
            X_curr, y_curr = X_tr_balanced, y_tr_balanced
        else:
            n_per_class = n_syn_to_add // 2
            print(f"  Generating {n_per_class} H and {n_per_class} C with cGAN...")
            # Generate Healthy (0)
            X_gen_h = generate_clean_spectra_gan(cgan_generator, GAN_LATENT_DIM, F_LEN, 0, n_per_class)
            # Generate Cancer (1)
            X_gen_c = generate_clean_spectra_gan(cgan_generator, GAN_LATENT_DIM, F_LEN, 1, n_per_class)

            X_curr = np.vstack([X_tr_balanced, X_gen_h, X_gen_c])
            y_curr = np.hstack([y_tr_balanced, np.zeros(n_per_class), np.ones(n_per_class)])

        res = run_classifier_analysis(X_curr, y_curr, X_te_clean, y_te, desc)
        results_cgan.append(res)

    # ---- SMOTE Loop (NEW) ----
    print("\n" + "="*30 + "\nStarting SMOTE Augmentation Loop\n" + "="*30)
    for ratio in AUGMENT_RATIOS:
        n_syn_to_add = int(n_bal_total * ratio)
        desc = f"{ratio} (+{n_syn_to_add} SMOTE)"
        print(f"\nRunning Ratio: {desc}")

        if ratio == 0.0:
            X_curr, y_curr = X_tr_balanced, y_tr_balanced
        else:
            n_per_class = n_syn_to_add // 2
            target_h = n_min + n_per_class
            target_c = n_min + n_per_class
            print(f"  Applying SMOTE to reach {target_h} H and {target_c} C...")

            # Configure SMOTE to achieve exact target counts
            smote = SMOTE(sampling_strategy={0: target_h, 1: target_c}, random_state=SEED)
            # Apply to balanced data
            X_curr, y_curr = smote.fit_resample(X_tr_balanced, y_tr_balanced)

        res = run_classifier_analysis(X_curr, y_curr, X_te_clean, y_te, desc)
        results_smote.append(res)


    # --- 5. Save and Plot Results ---
    print("\nAnalyzing results...")
    df_ldm = pd.DataFrame(results_ldm)
    df_cgan = pd.DataFrame(results_cgan)
    df_smote = pd.DataFrame(results_smote)

    # Combine for CSV saving
    df_ldm['Method'] = 'LDM'
    df_cgan['Method'] = 'cGAN'
    df_smote['Method'] = 'SMOTE'
    df_all = pd.concat([df_ldm, df_cgan, df_smote], ignore_index=True)
    csv_path = OUT_DIR / "strategic_augmentation_comparison_fixed.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"Saved combined results to {csv_path}")

    # Plotting
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    fig.suptitle("Strategic Augmentation: LDM vs cGAN vs SMOTE", fontsize=16)

    metrics = {'auc': 'AUC', 'acc': 'Accuracy', 'sens': 'Sensitivity (Cancer)', 'spec': 'Specificity (Healthy)'}
    ratios = [float(r) for r in df_ldm['ratio']]

    for i, (metric_key, metric_name) in enumerate(metrics.items()):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        # Plot LDM (Green)
        ax.plot(ratios, df_ldm[metric_key], marker='o', linewidth=2, color='tab:green', label='LDM')
        # Plot cGAN (Red)
        ax.plot(ratios, df_cgan[metric_key], marker='s', linewidth=2, color='tab:red', label='cGAN')
        # Plot SMOTE (Purple)
        ax.plot(ratios, df_smote[metric_key], marker='^', linewidth=2, color='tab:purple', label='SMOTE')

        ax.set_title(f"Test {metric_name}")
        ax.set_xlabel("Augmentation Ratio (Added / Original Balanced)")
        ax.set_ylabel("Score")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticks(ratios)

        # Set consistent y-limits for easier comparison
        if metric_key == 'auc': ax.set_ylim(bottom=0.70, top=0.85)
        elif metric_key == 'sens': ax.set_ylim(bottom=0.65, top=0.90)
        else: ax.set_ylim(bottom=0.50, top=0.85)

        if i == 0: ax.legend() # Legend only on first plot

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = OUT_DIR / "strategic_augmentation_comparison_plot_fixed.png"
    plt.savefig(plot_path, dpi=200)
    print(f"Saved comparison plot to {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()