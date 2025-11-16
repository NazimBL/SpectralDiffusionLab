#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is the DDPM training script modified to include Triplet Loss
to force the model to learn discriminative features.
"""

import math, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

print("Fnn type:", type(Fnn))

# ===================== CONFIG =====================
OUT_DIR = Path("ldm_out_triplet")  # New output dir
EPOCHS = 300
BATCH = 64
LR = 2e-4
T = 1000  # DDPM steps (training)
SAVE_EVERY = 50
SEED = 42

# --- NEW: Loss Lambdas for Ablation Study ---
LAMBDA_MSE = 1.0  # Weight for original MSE(eps, eps_hat)
LAMBDA_PEAK = 0.1  # Weight for decoded peak-weighted MSE
LAMBDA_TRIPLET = 0.1  # Weight for new triplet loss
TRIPLET_MARGIN = 0.2  # Margin for triplet loss (e.g., 0.2 - 1.0)
# --- End New Config ---

# Peak-weighted loss
SIGMA_BINS = 2.0
PEAKS_CM1 = [1716.0, 1446.0, 1377.0, 1234.0, 1045.0, 900.0]

# Optional sampling after training
DO_SAMPLE_AFTER_TRAIN = True
SAMPLES_PER_CLASS = 8
SAMPLE_STEPS = 1000
# ==================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED);
np.random.seed(SEED);
random.seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Using device: {DEVICE}")
print(f"Loss weights: MSE={LAMBDA_MSE}, Peak={LAMBDA_PEAK}, Triplet={LAMBDA_TRIPLET}")

# ----------------- Load AE meta + decoder -----------------
# Load from original 'ldm_out' directory
with open(Path("ldm_out") / "ae_meta.json", "r") as f:
    meta = json.load(f)
F_LEN = int(meta["F"])
print("F_LEN:", F_LEN)
downs = int(meta["downs"])
latent_c = int(meta["latent_c" if "latent_c" in meta else "latent_channels"])
latent_L = int(meta["latent_length"])
mean_np = np.array(meta["scaler_mean"], dtype=np.float32)
std_np = np.array(meta["scaler_std"], dtype=np.float32)
cols = meta["cols"]
wns_np = np.array([float(c) for c in cols], dtype=np.float32)


def gnorm(c):  # safe group norm
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class ConvAE(nn.Module):
    """Only need the decoder for peak loss and final rendering."""

    def __init__(self, F: int, downs: int = 4, base: int = 64, latent_c: int = 64):
        super().__init__()
        self.F = F
        c = base;
        in_c = 1;
        enc = []
        for i in range(downs):
            out_c = latent_c if i == downs - 1 else c
            enc += [
                nn.Conv1d(in_c, c, 5, padding=2), nn.SiLU(),
                nn.Conv1d(c, out_c, 5, stride=2, padding=2), nn.SiLU()
            ]
            in_c = out_c;
            c = min(c * 2, 256)
        self.encoder = nn.Sequential(*enc)
        dec = [];
        c_cur = latent_c
        for i in range(downs):
            c_mid = max(c_cur // 2, base) if i < downs - 1 else base
            c_out = base if i < downs - 1 else 32
            dec += [
                nn.ConvTranspose1d(c_cur, c_mid, 4, stride=2, padding=1), nn.SiLU(),
                nn.Conv1d(c_mid, c_out, 5, padding=2), nn.SiLU()
            ]
            c_cur = c_out
        self.decoder = nn.Sequential(*dec)
        self.to_raw = nn.Conv1d(c_cur, 1, kernel_size=3, padding=1)

    def decode(self, h):
        y = self.decoder(h)
        if y.shape[-1] != self.F:
            if y.shape[-1] > self.F:
                diff = y.shape[-1] - self.F
                y = y[..., diff // 2: diff // 2 + self.F]
            else:
                pad = self.F - y.shape[-1]
                y = Fnn.pad(y, (pad // 2, pad - pad // 2))
        return self.to_raw(y)


# Load AE from original directory
ae = ConvAE(F_LEN, downs=downs, latent_c=latent_c).to(DEVICE)
sd = torch.load(Path("ldm_out") / "ae_conv1d.pt", map_location=DEVICE, weights_only=False)
ae.load_state_dict(sd, strict=False)
for p in ae.parameters(): p.requires_grad = False
ae.eval()

x_mean = torch.from_numpy(mean_np).to(DEVICE).view(1, 1, -1)
x_std = torch.from_numpy(std_np).to(DEVICE).view(1, 1, -1)


# ----------------- U-Net Model (Unchanged) -----------------
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
        self.emb = nn.Embedding(num_classes, dim)

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
    def __init__(self, in_ch=64, base=128, out_ch=64, time_dim=128, class_dim=32, num_classes=2):
        super().__init__()
        self.temb = SinusoidalTimeEmbedding(time_dim)
        self.yemb = ClassEmbedding(num_classes=num_classes, dim=class_dim)
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
        h1 = self.rb1(zt, c);
        x = self.down1(h1);
        h2 = self.rb2(x, c);
        x = self.down2(h2)
        x = self.mid1(x, c);
        x = self.mid2(x, c)
        x = Fnn.interpolate(x, size=h2.shape[-1], mode="linear", align_corners=False)
        x = self.up2_conv(x);
        x = torch.cat([x, h2], dim=1)
        x = self.rb_up2a(x, c);
        x = self.rb_up2b(x, c)
        x = Fnn.interpolate(x, size=h1.shape[-1], mode="linear", align_corners=False)
        x = self.up1_conv(x);
        x = torch.cat([x, h1], dim=1)
        x = self.rb_up1a(x, c);
        x = self.rb_up1b(x, c)
        return self.head(x)


# ----------------- DDPM schedule + helpers -----------------
def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps, device=DEVICE)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return betas.clamp(1e-8, 0.999)


betas = cosine_beta_schedule(T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_ac = torch.sqrt(alphas_cumprod)
sqrt_omc = torch.sqrt(1.0 - alphas_cumprod)
one_over_sqrt_ac = 1.0 / sqrt_ac


def q_sample(z0, t, noise=None):
    if noise is None: noise = torch.randn_like(z0)
    s1 = sqrt_ac[t].view(-1, 1, 1)
    s2 = sqrt_omc[t].view(-1, 1, 1)
    return s1 * z0 + s2 * noise, noise


def predict_x0_from_eps(z_t, t, eps_hat):
    s1 = one_over_sqrt_ac[t].view(-1, 1, 1)
    s2 = sqrt_omc[t].view(-1, 1, 1)
    return s1 * (z_t - s2 * eps_hat)


# ----------------- Peak Gaussian weights -----------------
def gaussian_peak_weights(wns, peaks_cm1, sigma_bins=2.0):
    Flen = wns.shape[0]
    idxs = [int(np.argmin(np.abs(wns - v))) for v in peaks_cm1]
    w = np.zeros(Flen, dtype=np.float32)
    xs = np.arange(Flen, dtype=np.float32)
    for i0 in idxs:
        w += np.exp(-0.5 * ((xs - i0) / sigma_bins) ** 2)
    w = w / (w.mean() + 1e-8)
    return torch.from_numpy(w).view(1, 1, -1).to(DEVICE), idxs


w_peaks, peak_indices = gaussian_peak_weights(wns_np, PEAKS_CM1, SIGMA_BINS)

# ----------------- Data -----------------
# Load from original directory
tr = torch.load(Path("ldm_out") / "latent_train.pt", map_location=DEVICE, weights_only=False)
va = torch.load(Path("ldm_out") / "latent_val.pt", map_location=DEVICE, weights_only=False)
z_tr, y_tr = tr["z"].to(DEVICE), tr["y"].to(DEVICE)
z_va, y_va = va["z"].to(DEVICE), va["y"].to(DEVICE)

z_mu = torch.from_numpy(tr["meta"]["z_mu"]).to(DEVICE)
z_std = torch.from_numpy(tr["meta"]["z_std"]).to(DEVICE)
z_tr = (z_tr - z_mu) / z_std
z_va = (z_va - z_mu) / z_std

# === NEW: WEIGHTED SAMPLER TO FIX IMBALANCE ===
# This is crucial for Triplet Loss, which needs both classes in a batch.
y_tr_cpu = y_tr.cpu().numpy()
class_counts = np.bincount(y_tr_cpu)
class_weights = 1. / class_counts
print(f"Balancing training batches...")
print(f"  Class counts (H, C): {class_counts[0]}, {class_counts[1]}")
print(f"  Class weights (H, C): {class_weights[0]:.4f}, {class_weights[1]:.4f}")
sample_weights = class_weights[y_tr_cpu]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
# ===============================================

# Use sampler, turn off shuffle
train_loader = DataLoader(TensorDataset(z_tr, y_tr), batch_size=BATCH, sampler=sampler, drop_last=True)
val_loader = DataLoader(TensorDataset(z_va, y_va), batch_size=BATCH, shuffle=False)

# ----------------- Model / Optim / Losses -----------------
unet = UNet1D_Cond(in_ch=latent_c, base=128, out_ch=latent_c, time_dim=128, class_dim=32, num_classes=2).to(DEVICE)
opt = torch.optim.AdamW(unet.parameters(), lr=LR)
mse = nn.MSELoss()


# --- NEW: TRIPLET LOSS FUNCTION ---
def compute_triplet_loss(embeddings, labels, margin=TRIPLET_MARGIN, p=2):
    """
    Computes batch-all triplet loss.
    Embeddings are (B, C, L), labels are (B,).
    """
    # Flatten embeddings for distance calculation
    B = embeddings.shape[0]
    embeddings_flat = embeddings.view(B, -1)  # (B, C*L)

    # Get pairwise distance matrix
    pairwise_dist = torch.cdist(embeddings_flat, embeddings_flat, p=p)  # (B, B)

    # Create masks for positive and negative pairs
    # (B,) -> (B, 1) -> (B, B)
    mask_anchor_positive = (labels.unsqueeze(0) == labels.unsqueeze(1))
    mask_anchor_negative = (labels.unsqueeze(0) != labels.unsqueeze(1))

    # Mask out the diagonal (self-comparison)
    mask_anchor_positive[torch.eye(B, dtype=torch.bool, device=DEVICE)] = False

    triplet_loss = 0.0
    num_valid_triplets = 0

    for i in range(B):  # For each anchor
        # Get distances to all positives and all negatives
        dist_pos = pairwise_dist[i][mask_anchor_positive[i]]
        dist_neg = pairwise_dist[i][mask_anchor_negative[i]]

        if dist_pos.shape[0] == 0 or dist_neg.shape[0] == 0:
            continue  # No valid triplets for this anchor (e.g., batch has one class)

        # Batch-All: Compare every positive to every negative
        # (N_pos, 1) - (1, N_neg) -> (N_pos, N_neg)
        loss_matrix = dist_pos.unsqueeze(1) - dist_neg.unsqueeze(0) + margin

        # Clamp at 0 and sum up all violations
        loss_matrix_clamped = Fnn.relu(loss_matrix)

        triplet_loss += loss_matrix_clamped.sum()
        num_valid_triplets += loss_matrix_clamped.numel()  # Count all (A,P,N) pairs

    if num_valid_triplets == 0:
        return torch.tensor(0.0, device=DEVICE)

    return triplet_loss / num_valid_triplets


# ----------------- Modified Training Loop -----------------

@torch.no_grad()
def mse_peak_term(z0_hat_norm, z0_true_norm):
    z0_hat = z0_hat_norm * z_std + z_mu
    z0_true = z0_true_norm * z_std + z_mu
    x_pred_s = ae.decode(z0_hat)
    x_true_s = ae.decode(z0_true)
    x_pred_r = x_pred_s * x_std + x_mean
    x_true_r = x_true_s * x_std + x_mean
    diff = (x_pred_r - x_true_r) * w_peaks
    return (diff ** 2).mean()


def train_one_epoch():
    unet.train()
    tot, tot_mse, tot_peak, tot_triplet = 0.0, 0.0, 0.0, 0.0
    # Recalculate N based on sampler, not dataset
    N_train = len(train_loader) * BATCH
    for z0, y in train_loader:
        opt.zero_grad()
        B = z0.size(0)
        t = torch.randint(0, T, (B,), device=DEVICE, dtype=torch.int64)
        z_t, eps_true = q_sample(z0, t)

        eps_hat = unet(z_t, t, y)
        z0_hat = predict_x0_from_eps(z_t, t, eps_hat)

        # --- Calculate all 3 losses ---
        loss_mse = mse(eps_hat, eps_true)
        loss_peak = mse_peak_term(z0_hat, z0)
        loss_triplet = compute_triplet_loss(z0_hat, y)

        loss = (LAMBDA_MSE * loss_mse) + \
               (LAMBDA_PEAK * loss_peak) + \
               (LAMBDA_TRIPLET * loss_triplet)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()

        tot += loss.item() * B
        tot_mse += loss_mse.item() * B
        tot_peak += loss_peak.item() * B
        tot_triplet += loss_triplet.item() * B

    return tot / N_train, tot_mse / N_train, tot_peak / N_train, tot_triplet / N_train


@torch.no_grad()
def eval_one_epoch():
    unet.eval()
    tot, tot_mse, tot_peak, tot_triplet = 0.0, 0.0, 0.0, 0.0
    N_val = len(val_loader.dataset)
    for z0, y in val_loader:
        B = z0.size(0)
        # Use a fixed timestep for consistent validation
        t = torch.full((B,), T // 2, device=DEVICE, dtype=torch.int64)
        z_t, eps_true = q_sample(z0, t)

        eps_hat = unet(z_t, t, y)
        z0_hat = predict_x0_from_eps(z_t, t, eps_hat)

        loss_mse = mse(eps_hat, eps_true)
        loss_peak = mse_peak_term(z0_hat, z0)
        loss_triplet = compute_triplet_loss(z0_hat, y)

        loss = (LAMBDA_MSE * loss_mse) + \
               (LAMBDA_PEAK * loss_peak) + \
               (LAMBDA_TRIPLET * loss_triplet)

        tot += loss.item() * B
        tot_mse += loss_mse.item() * B
        tot_peak += loss_peak.item() * B
        tot_triplet += loss_triplet.item() * B

    return tot / N_val, tot_mse / N_val, tot_peak / N_val, tot_triplet / N_val


best_val = float("inf")
for ep in range(1, EPOCHS + 1):
    tr_all, tr_mse, tr_peak, tr_triplet = train_one_epoch()
    va_all, va_mse, va_peak, va_triplet = eval_one_epoch()

    print(f"Epoch {ep:03d} | "
          f"train total {tr_all:.6f} (mse {tr_mse:.6f}, peak {tr_peak:.6f}, triplet {tr_triplet:.6f}) | "
          f"val total {va_all:.6f} (mse {va_mse:.6f}, peak {va_peak:.6f}, triplet {va_triplet:.6f})")

    if va_all < best_val - 1e-6:
        best_val = va_all
        print(f"  ^ New best validation loss: {va_all:.6f} (saving model)")
        torch.save({
            "model": unet.state_dict(),
            "T": T,
            "betas": cosine_beta_schedule(T).cpu(),
            "lambda_mse": LAMBDA_MSE,
            "lambda_peak": LAMBDA_PEAK,
            "lambda_triplet": LAMBDA_TRIPLET,
            "triplet_margin": TRIPLET_MARGIN,
            "meta": meta,
            "z_mu": z_mu.detach().cpu(), "z_std": z_std.detach().cpu(),
        }, OUT_DIR / "ddpm_latent_unet.pt")

    if ep % SAVE_EVERY == 0:
        torch.save(unet.state_dict(), OUT_DIR / f"ddpm_unet_ep{ep}.pt")

print("Training done. Best val:", best_val)


# ----------------- DDPM ancestral sampler -----------------
# (Sampler is unchanged, but we'll load the new checkpoint)
@torch.no_grad()
def p_sample_loop(unet, steps, y_class, n, ckpt_data):
    """
    steps: number of reverse steps (<= T)
    y_class: int (0 healthy / 1 cancer)
    returns: raw spectra (n, F) in original units
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
    # === END FIX ===

    z_mu_chk = ckpt_data["z_mu"].to(DEVICE)
    z_std_chk = ckpt_data["z_std"].to(DEVICE)

    z_t_unnorm = z_t_renormed * z_std_chk + z_mu_chk

    x_scaled = ae.decode(z_t_unnorm)
    x_raw = (x_scaled * x_std + x_mean).squeeze(1).detach().cpu().numpy()
    return x_raw


if DO_SAMPLE_AFTER_TRAIN:
    print("Sampling from best model...")
    ckpt = torch.load(OUT_DIR / "ddpm_latent_unet.pt", map_location=DEVICE, weights_only=False)
    unet.load_state_dict(ckpt["model"])
    unet.eval()

    # Load checkpoint data for the sampler
    ckpt_data_for_sampler = {
        "z_mu": ckpt["z_mu"],
        "z_std": ckpt["z_std"]
    }

    for cls, name in [(0, "healthy"), (1, "cancer")]:
        xraw = p_sample_loop(unet, steps=min(SAMPLE_STEPS, T), y_class=cls, n=SAMPLES_PER_CLASS,
                             ckpt_data=ckpt_data_for_sampler)
        np.save(OUT_DIR / f"samples_{name}.npy", xraw)
        print(f"Saved {name} samples ->", OUT_DIR / f"samples_{name}.npy")