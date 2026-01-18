#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.utils.data import TensorDataset, DataLoader


# ===================== CONFIG =====================
OUT_DIR = Path("ldm_out")  # contains ae_meta.json, ae_conv1d.pt, latent_*.pt
EPOCHS = 300
BATCH = 64
LR = 2e-4
T = 1000  # DDPM steps (training)
SAVE_EVERY = 50
SEED = 42
LAMBDA_PEAKS = 0
SIGMA_BINS = 2.0
PEAKS_CM1 = [1716.0, 1446.0, 1377.0, 1234.0, 1045.0, 900.0]
DO_SAMPLE_AFTER_TRAIN = True
SAMPLES_PER_CLASS = 20
SAMPLE_STEPS = 1000
# ==================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED);
np.random.seed(SEED);
random.seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ... AE meta loading  ...
with open(OUT_DIR / "ae_meta.json", "r") as f:
    meta = json.load(f)
F_LEN = int(meta["F"])
print("F_LEN:", F_LEN)
downs = int(meta["downs"])
latent_c = int(meta["latent_channels"])
latent_L = int(meta["latent_length"])
cols = meta["cols"]
wns_np = np.array([float(c) for c in cols], dtype=np.float32)


def gnorm(c):
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


# ... (ConvAE class) ...
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


# Initialize AE and load weights
ae = ConvAE(F_LEN, downs=downs, base=64, latent_c=latent_c).to(DEVICE)
sd = torch.load(OUT_DIR / "ae_conv1d.pt", map_location=DEVICE, weights_only=False)
ae.load_state_dict(sd, strict=False)
for p in ae.parameters(): p.requires_grad = False
ae.eval()


# ... (U-Net, DDPM schedule, data loading, etc.) ...
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
    def __init__(self, in_ch=64, base=128, out_ch=64, time_dim=128, class_dim=32, num_classes=2):
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



# --- Data Loading (This is correct) ---
tr = torch.load(OUT_DIR / "latent_train.pt", map_location=DEVICE, weights_only=False)
va = torch.load(OUT_DIR / "latent_val.pt", map_location=DEVICE, weights_only=False)
z_tr, y_tr = tr["z"].to(DEVICE), tr["y"].to(DEVICE)
z_va, y_va = va["z"].to(DEVICE), va["y"].to(DEVICE)
z_mu = torch.from_numpy(tr["meta"]["z_mu"]).to(DEVICE)
z_std = torch.from_numpy(tr["meta"]["z_std"]).to(DEVICE)
z_tr = (z_tr - z_mu) / z_std
z_va = (z_va - z_mu) / z_std

print(f"z_tr_norm stats:  mean = {z_tr.mean().item():.6f},  std = {z_tr.std().item():.6f}")

train_loader = DataLoader(TensorDataset(z_tr, y_tr), batch_size=BATCH, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(z_va, y_va), batch_size=BATCH, shuffle=False)

# --- Model / Optim ---
unet = UNet1D_Cond(in_ch=latent_c, base=128, out_ch=latent_c, time_dim=128, class_dim=32, num_classes=2).to(DEVICE)
opt = torch.optim.AdamW(unet.parameters(), lr=LR)
mse = nn.MSELoss()


# --- Training loops  ---
def train_one_epoch():
    unet.train()
    tot = 0.0
    for z0, y in train_loader:
        opt.zero_grad()
        B = z0.size(0)
        t = torch.randint(0, T, (B,), device=DEVICE, dtype=torch.int64)
        prob = 0.1
        y_masked = y.clone()
        mask = torch.rand(B, device=DEVICE) < prob
        y_masked[mask] = unet.null_class_idx
        z_t, eps = q_sample(z0, t)
        eps_hat = unet(z_t, t, y_masked)
        loss = mse(eps_hat, eps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        tot += loss.item() * B
    N = len(train_loader.dataset)
    return tot / N


@torch.no_grad()
def eval_one_epoch():
    unet.eval()
    tot = 0.0
    for z0, y in val_loader:
        B = z0.size(0)
        t = torch.randint(0, T, (B,), device=DEVICE, dtype=torch.int64)
        z_t, eps = q_sample(z0, t)
        eps_hat = unet(z_t, t, y)
        loss_eps = mse(eps_hat, eps)
        tot += loss_eps.item() * B
    N = len(val_loader.dataset)
    return tot / N


# --- Main training loop ---
best_val = float("inf")
print("Starting DDPM training on AE latents...")  # <--- Renamed for clarity
for ep in range(1, EPOCHS + 1):
    tr_loss = train_one_epoch()
    va_loss = eval_one_epoch()
    print(f"Epoch {ep:03d} | "
          f"train loss {tr_loss:.4f} | "
          f"val loss {va_loss:.4f}")

    if va_loss < best_val - 1e-6:
        best_val = va_loss
        torch.save({
            "model": unet.state_dict(),
            "T": T,
            "meta": meta,
            "z_mu": z_mu.detach().cpu(),
            "z_std": z_std.detach().cpu(),
        }, OUT_DIR / "ddpm_latent_unet.pt")

    if ep % SAVE_EVERY == 0:
        torch.save(unet.state_dict(), OUT_DIR / f"ddpm_unet_ep{ep}.pt")
print("Training done. Best val:", best_val)


# -----------------  SAMPLING LOOP -----------------
@torch.no_grad()
def p_sample_loop(unet, z_mu_arg, z_std_arg, steps, y_class, n, w=7.5):  # <--- FIX: Added z_mu/z_std as args
    """
    Generates final spectra.
    Uses z_mu_arg / z_std_arg passed from the main script.
    """
    betas_s = cosine_beta_schedule(T).to(DEVICE)
    alphas_s = 1.0 - betas
    ac_s = torch.cumprod(alphas_s, dim=0)
    sqrt_recip_alphas = (1.0 / torch.sqrt(alphas_s)).to(DEVICE)

    z_t = torch.randn(n, latent_c, latent_L, device=DEVICE)

    y_cond = torch.full((n,), int(y_class), device=DEVICE, dtype=torch.long)
    y_uncond = torch.full((n,), unet.null_class_idx, device=DEVICE, dtype=torch.long)

    ts = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=DEVICE)
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

    # --- FIX 2 : Use the arguments, don't load the file ---

        # this z_final in the main loop ---
    z_final_norm = z_t
    z_final_norm = z_final_norm * (z_tr.std() / (z_final_norm.std() + 1e-8))

        # Un-normalize
    z_t_unnorm = z_final_norm * z_std_arg + z_mu_arg

        # Decode
    x_scaled = ae.decode(z_t_unnorm)

        # ---  Return z_final_norm and the decoded spectrum ---
    return z_final_norm, x_scaled.squeeze(1).detach().cpu().numpy()


# ---  SAMPLING BLOCK---
if DO_SAMPLE_AFTER_TRAIN:
    print("Generating samples...")
    # Load the *best* model state we just saved
    ckpt = torch.load(OUT_DIR / "ddpm_latent_unet.pt", map_location=DEVICE, weights_only=False)
    unet.load_state_dict(ckpt["model"])
    unet.eval()

    # ---  Load the z_mu/z_std from the checkpoint  ---
    z_mu_sample = ckpt["z_mu"].to(DEVICE)
    z_std_sample = ckpt["z_std"].to(DEVICE)

    for cls, name in [(0, "healthy"), (1, "cancer")]:

        z_final, xraw = p_sample_loop(unet, z_mu_sample, z_std_sample,
                                      steps=min(SAMPLE_STEPS, T), y_class=cls, n=SAMPLES_PER_CLASS, w=7.5)

        if cls == 0:  # Only print this once
            print(f"z_final stats (w={1.5}):  mean = {z_final.mean().item():.6f},  std = {z_final.std().item():.6f}")


        np.save(OUT_DIR / f"samples_{name}.npy", xraw)
        print(f"Saved {name} samples -> {OUT_DIR / f'samples_{name}.npy'}")