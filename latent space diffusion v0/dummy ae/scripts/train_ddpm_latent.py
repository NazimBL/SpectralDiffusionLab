#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.utils.data import TensorDataset, DataLoader

print("Fnn type:", type(Fnn))

# ===================== CONFIG =====================
OUT_DIR        = Path("ldm_out")   # contains ae_meta.json, ae_conv1d.pt, latent_*.pt
EPOCHS         = 300
BATCH          = 64
LR             = 2e-4
T              = 1000                 # DDPM steps (training)
SAVE_EVERY     = 50
SEED           = 42

# Peak-weighted loss
LAMBDA_PEAKS   = 0.5                  # try 0.25–1.0
SIGMA_BINS     = 2.0                  # Gaussian σ around each peak (in bins)
PEAKS_CM1      = [1716.0, 1446.0, 1377.0, 1234.0, 1045.0, 900.0]

# Optional sampling after training
DO_SAMPLE_AFTER_TRAIN = True
SAMPLES_PER_CLASS     = 10
SAMPLE_STEPS          = 1000          # ancestral DDPM steps (<= T)
# ==================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Load AE meta + decoder -----------------
with open(OUT_DIR/"ae_meta.json", "r") as f:
    meta = json.load(f)
F_LEN         = int(meta["F"])
print("F_LEN:", F_LEN)
downs       = int(meta["downs"])
latent_c    = int(meta["latent_c" if "latent_c" in meta else "latent_channels"])
latent_L    = int(meta["latent_length"])
mean_np     = np.array(meta["scaler_mean"], dtype=np.float32)
std_np      = np.array(meta["scaler_std"],  dtype=np.float32)
cols        = meta["cols"]
wns_np      = np.array([float(c) for c in cols], dtype=np.float32)

def gnorm(c):  # safe group norm
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)

class ConvAE(nn.Module):
    """Only need the decoder for peak loss and final rendering."""
    def __init__(self, F:int, downs:int=4, base:int=64, latent_c:int=64):
        super().__init__()
        self.F = F
        # encoder (unused) to match state dict
        c = base; in_c = 1; enc=[]
        for i in range(downs):
            out_c = latent_c if i == downs-1 else c
            enc += [
                nn.Conv1d(in_c, c, 5, padding=2), nn.SiLU(),
                nn.Conv1d(c, out_c, 5, stride=2, padding=2), nn.SiLU()
            ]
            in_c = out_c; c = min(c*2, 256)
        self.encoder = nn.Sequential(*enc)

        # decoder mirrors encoder
        dec=[]; c_cur = latent_c
        for i in range(downs):
            c_mid = max(c_cur//2, base) if i < downs-1 else base
            c_out = base if i < downs-1 else 32
            dec += [
                nn.ConvTranspose1d(c_cur, c_mid, 4, stride=2, padding=1), nn.SiLU(),
                nn.Conv1d(c_mid, c_out, 5, padding=2), nn.SiLU()
            ]
            c_cur = c_out
        self.decoder = nn.Sequential(*dec)
        self.to_raw = nn.Conv1d(c_cur, 1, kernel_size=3, padding=1)

    def decode(self, h):  # (B,latent_c,latent_L) -> (B,1,F) scaled units
        y = self.decoder(h)
        if y.shape[-1] != self.F:
            if y.shape[-1] > self.F:
                diff = y.shape[-1] - self.F
                y = y[..., diff//2 : diff//2 + self.F]
            else:
                pad = self.F - y.shape[-1]
                y = Fnn.pad(y, (pad // 2, pad - pad // 2))
        return self.to_raw(y)

ae = ConvAE(F_LEN, downs=downs, latent_c=latent_c).to(DEVICE)
sd = torch.load(OUT_DIR/"ae_conv1d.pt", map_location=DEVICE, weights_only=False)
ae.load_state_dict(sd, strict=False)

# After building/loading the ae
with torch.no_grad():
    probe = torch.randn(2, latent_c, latent_L, device=DEVICE)
    out = ae.decode(probe)
    assert out.shape[-1] == F_LEN, f"Decoder returned {out.shape[-1]} != F_LEN={F_LEN}"

for p in ae.parameters(): p.requires_grad = False
ae.eval()

x_mean = torch.from_numpy(mean_np).to(DEVICE).view(1,1,-1)
x_std  = torch.from_numpy(std_np ).to(DEVICE).view(1,1,-1)

# ----------------- Old-style conditioned U-Net -----------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=128, max_period=10000.0):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)
        self.dim = dim
    def forward(self, t):  # t: (B,)
        t = t.float().unsqueeze(1)
        ang = t * self.freqs.unsqueeze(0)
        return torch.cat([ang.sin(), ang.cos()], dim=1)

class ClassEmbedding(nn.Module):
    def __init__(self, num_classes=2, dim=32):
        super().__init__()
        self.emb = nn.Embedding(num_classes, dim)
    def forward(self, y):  # y: (B,)
        return self.emb(y)

class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.gn1   = gnorm(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.gn2   = gnorm(cout)
        self.act   = nn.SiLU()
        self.cond  = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, cout))
        self.skip  = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
    def forward(self, x, cvec):
        h = self.act(self.gn1(self.conv1(x)))
        b = self.cond(cvec).unsqueeze(-1)  # (B,cout,1)
        h = self.conv2(h)
        h = self.gn2(h + b)
        h = self.act(h)
        return h + self.skip(x)

class UNet1D_Cond(nn.Module):
    """
    in:  z_t (B,64,15), t(B,), y(B,)
    out: eps_hat (B,64,15)
    """
    def __init__(self, in_ch=64, base=128, out_ch=64, time_dim=128, class_dim=32, num_classes=2):
        super().__init__()
        self.temb = SinusoidalTimeEmbedding(time_dim)
        self.yemb = ClassEmbedding(num_classes=num_classes, dim=class_dim)
        self.proj = nn.Sequential(nn.Linear(time_dim+class_dim, base), nn.SiLU())
        cond_dim = base

        # Encoder
        self.rb1   = ResBlock1D(in_ch,    base,   cond_dim)      # L=15
        self.down1 = nn.Conv1d(base,      base,   4, stride=2, padding=1)  # 15->7
        self.rb2   = ResBlock1D(base,     base*2, cond_dim)      # L=7
        self.down2 = nn.Conv1d(base*2,    base*2, 4, stride=2, padding=1)  # 7->3

        # Mid
        self.mid1 = ResBlock1D(base*2, base*4, cond_dim)
        self.mid2 = ResBlock1D(base*4, base*4, cond_dim)

        # Decoder
        self.up2_conv = nn.Conv1d(base*4, base*2, 1)
        self.rb_up2a  = ResBlock1D(base*2 + base*2, base*2, cond_dim)
        self.rb_up2b  = ResBlock1D(base*2,          base*2, cond_dim)

        self.up1_conv = nn.Conv1d(base*2, base, 1)
        self.rb_up1a  = ResBlock1D(base + base,     base,   cond_dim)
        self.rb_up1b  = ResBlock1D(base,            base,   cond_dim)

        self.head = nn.Conv1d(base, out_ch, 3, padding=1)

    def forward(self, zt, t, y):
        c = torch.cat([self.temb(t), self.yemb(y)], dim=1)
        c = self.proj(c)

        h1 = self.rb1(zt, c)      # (B,base,15)
        x  = self.down1(h1)       # (B,base,7)
        h2 = self.rb2(x, c)       # (B,base*2,7)
        x  = self.down2(h2)       # (B,base*2,3)

        x  = self.mid1(x, c)      # (B,base*4,3)
        x  = self.mid2(x, c)

        x  = Fnn.interpolate(x, size=h2.shape[-1], mode="linear", align_corners=False)
        x  = self.up2_conv(x)     # (B,base*2,7)
        x  = torch.cat([x, h2], dim=1)
        x  = self.rb_up2a(x, c)
        x  = self.rb_up2b(x, c)

        x  = Fnn.interpolate(x, size=h1.shape[-1], mode="linear", align_corners=False)
        x  = self.up1_conv(x)     # (B,base,15)
        x  = torch.cat([x, h1], dim=1)
        x  = self.rb_up1a(x, c)
        x  = self.rb_up1b(x, c)

        return self.head(x)       # (B,64,15)

# ----------------- DDPM schedule + helpers -----------------
def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps, device=DEVICE)
    ac = torch.cos(((x/T) + s) / (1+s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return betas.clamp(1e-8, 0.999)

betas = cosine_beta_schedule(T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_ac   = torch.sqrt(alphas_cumprod)                  # (T,)
sqrt_omc  = torch.sqrt(1.0 - alphas_cumprod)            # (T,)
one_over_sqrt_ac = 1.0 / sqrt_ac
posterior_var = betas * (1.0 - alphas_cumprod[:-1].clone().detach().mean()).new_ones(T)  # not used; will compute per-step

def q_sample(z0, t, noise=None):
    if noise is None: noise = torch.randn_like(z0)
    s1 = sqrt_ac[t].view(-1,1,1)
    s2 = sqrt_omc[t].view(-1,1,1)
    return s1 * z0 + s2 * noise, noise

def predict_x0_from_eps(z_t, t, eps_hat):
    s1 = one_over_sqrt_ac[t].view(-1,1,1)
    s2 = sqrt_omc[t].view(-1,1,1)
    return s1 * (z_t - s2 * eps_hat)

# ----------------- Peak Gaussian weights on raw axis -----------------
def gaussian_peak_weights(wns, peaks_cm1, sigma_bins=2.0):
    Flen = wns.shape[0]
    idxs = [int(np.argmin(np.abs(wns - v))) for v in peaks_cm1]
    w = np.zeros(Flen, dtype=np.float32)
    xs = np.arange(Flen, dtype=np.float32)
    for i0 in idxs:
        w += np.exp(-0.5 * ((xs - i0)/sigma_bins)**2)
    w = w / (w.mean() + 1e-8)
    return torch.from_numpy(w).view(1,1,-1).to(DEVICE), idxs

w_peaks, peak_indices = gaussian_peak_weights(wns_np, PEAKS_CM1, SIGMA_BINS)

# ----------------- Data -----------------
tr = torch.load(OUT_DIR/"latent_train.pt", map_location=DEVICE, weights_only=False)
va = torch.load(OUT_DIR/"latent_val.pt",   map_location=DEVICE, weights_only=False)
z_tr, y_tr = tr["z"].to(DEVICE), tr["y"].to(DEVICE)
z_va, y_va = va["z"].to(DEVICE), va["y"].to(DEVICE)

# NEW: normalize latents
z_mu  = torch.from_numpy(tr["meta"]["z_mu"]).to(DEVICE)
z_std = torch.from_numpy(tr["meta"]["z_std"]).to(DEVICE)
z_tr = (z_tr - z_mu) / z_std
z_va = (z_va - z_mu) / z_std

train_loader = DataLoader(TensorDataset(z_tr, y_tr), batch_size=BATCH, shuffle=True, drop_last=True)
val_loader   = DataLoader(TensorDataset(z_va, y_va), batch_size=BATCH, shuffle=False)

# ----------------- Model / Optim -----------------
unet = UNet1D_Cond(in_ch=latent_c, base=128, out_ch=latent_c, time_dim=128, class_dim=32, num_classes=2).to(DEVICE)
opt  = torch.optim.AdamW(unet.parameters(), lr=LR)
mse  = nn.MSELoss()


@torch.no_grad()
def mse_peak_term(z0_hat_norm, z0_true_norm):
    # Un-normalize latents first!
    z0_hat = z0_hat_norm * z_std + z_mu
    z0_true = z0_true_norm * z_std + z_mu

    # Now decode the un-normalized latents
    x_pred_s = ae.decode(z0_hat)
    x_true_s = ae.decode(z0_true)

    # Un-scale spectra (this part was already correct)
    x_pred_r = x_pred_s * x_std + x_mean
    x_true_r = x_true_s * x_std + x_mean

    diff = (x_pred_r - x_true_r) * w_peaks
    return (diff ** 2).mean()

def train_one_epoch():
    unet.train()
    tot, tot_eps, tot_peak = 0.0, 0.0, 0.0
    for z0, y in train_loader:
        opt.zero_grad()
        B = z0.size(0)
        t = torch.randint(0, T, (B,), device=DEVICE, dtype=torch.int64)
        z_t, eps = q_sample(z0, t)
        eps_hat = unet(z_t, t, y)

        loss_eps  = mse(eps_hat, eps)
        z0_hat    = predict_x0_from_eps(z_t, t, eps_hat)
        loss_peak = mse_peak_term(z0_hat, z0)

        loss = loss_eps + LAMBDA_PEAKS * loss_peak
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()

        tot += loss.item()*B; tot_eps += loss_eps.item()*B; tot_peak += loss_peak.item()*B
    N = len(train_loader.dataset)
    return tot/N, tot_eps/N, tot_peak/N

@torch.no_grad()
def eval_one_epoch():
    unet.eval()
    tot, tot_eps, tot_peak = 0.0, 0.0, 0.0
    for z0, y in val_loader:
        B = z0.size(0)
        t = torch.randint(0, T, (B,), device=DEVICE, dtype=torch.int64)
        z_t, eps = q_sample(z0, t)
        eps_hat  = unet(z_t, t, y)

        loss_eps  = mse(eps_hat, eps)
        z0_hat    = predict_x0_from_eps(z_t, t, eps_hat)
        loss_peak = mse_peak_term(z0_hat, z0)

        tot += (loss_eps + LAMBDA_PEAKS*loss_peak).item()*B
        tot_eps += loss_eps.item()*B
        tot_peak += loss_peak.item()*B
    N = len(val_loader.dataset)
    return tot/N, tot_eps/N, tot_peak/N

best_val = float("inf")
for ep in range(1, EPOCHS+1):
    tr_all, tr_eps, tr_peak = train_one_epoch()
    va_all, va_eps, va_peak = eval_one_epoch()
    print(f"Epoch {ep:03d} | "
          f"train total {tr_all:.6f} (eps {tr_eps:.6f}, peak {tr_peak:.6f}) | "
          f"val total {va_all:.6f} (eps {va_eps:.6f}, peak {va_peak:.6f})")

    if va_all < best_val - 1e-6:
        best_val = va_all
        torch.save({
            "model": unet.state_dict(),
            "T": T,
            "betas": cosine_beta_schedule(T).cpu(),
            "lambda_peaks": LAMBDA_PEAKS,
            "sigma_bins": SIGMA_BINS,
            "peaks_cm1": PEAKS_CM1,
            "meta": meta,
            "z_mu": z_mu.detach().cpu(), "z_std": z_std.detach().cpu(),  # NEW
        }, OUT_DIR / "ddpm_latent_unet.pt")

    if ep % SAVE_EVERY == 0:
        torch.save(unet.state_dict(), OUT_DIR/f"ddpm_unet_ep{ep}.pt")

print("Training done. Best val:", best_val)

# ----------------- DDPM ancestral sampler -----------------
@torch.no_grad()
def p_sample_loop(unet, steps, y_class, n):
    """
    steps: number of reverse steps (<= T)
    y_class: int (0 healthy / 1 cancer)
    returns: raw spectra (n, F) in original units
    """
    betas_s = cosine_beta_schedule(T).to(DEVICE)            # (T,)
    alphas_s = 1.0 - betas_s
    ac_s = torch.cumprod(alphas_s, dim=0)
    sqrt_recip_alphas = (1.0 / torch.sqrt(alphas_s)).to(DEVICE)
    posterior_variance = betas_s * (1.0 - ac_s[:-1].clone().detach().mean()).new_ones(T)  # simple
    # Start from noise in latent space
    z_t = torch.randn(n, latent_c, latent_L, device=DEVICE)
    y   = torch.full((n,), int(y_class), device=DEVICE, dtype=torch.long)

    # Use evenly spaced timesteps for speed if steps < T
    ts = torch.linspace(T-1, 0, steps, dtype=torch.long, device=DEVICE)
    for t_val in ts:
        t = t_val.repeat(n)
        eps_hat = unet(z_t, t, y)
        # DDPM reverse step
        beta_t  = betas_s[t].view(-1,1,1)
        sqrt_one_minus_ac_t = torch.sqrt(1.0 - ac_s[t]).view(-1,1,1)
        sqrt_recip_alpha_t  = sqrt_recip_alphas[t].view(-1,1,1)

        x0_hat = (z_t - sqrt_one_minus_ac_t * eps_hat) / torch.sqrt(ac_s[t]).view(-1,1,1)
        mean   = sqrt_recip_alpha_t * (z_t - beta_t / sqrt_one_minus_ac_t * eps_hat)
        if (t_val > 0):
            noise = torch.randn_like(z_t)
            z_t = mean + torch.sqrt(beta_t) * noise
        else:
            z_t = mean

        # === NEW FIX: Rescale the generated latents ===

        # === END FIX ===

        z_mu_chk = ckpt["z_mu"].to(DEVICE)
        z_std_chk = ckpt["z_std"].to(DEVICE)

        # Un-normalize the *re-normalized* latent vector
        z_t_unnorm = z_t * z_std_chk + z_mu_chk

        # Decode to raw
        x_scaled = ae.decode(z_t_unnorm)  # (n,1,F) scaled
        x_raw = (x_scaled * x_std + x_mean).squeeze(1).detach().cpu().numpy()  # (n,F)
        return x_raw

if DO_SAMPLE_AFTER_TRAIN:
    ckpt = torch.load(OUT_DIR/"ddpm_latent_unet.pt", map_location=DEVICE, weights_only=False)
    unet.load_state_dict(ckpt["model"])
    unet.eval()

    # Load the latent stats needed for un-normalizing
    z_mu_chk = ckpt["z_mu"].to(DEVICE)
    z_std_chk = ckpt["z_std"].to(DEVICE)

    # The easiest way: just load them globally for sampling
    z_mu = ckpt["z_mu"].to(DEVICE)
    z_std = ckpt["z_std"].to(DEVICE)
    for cls, name in [(0,"healthy"),(1,"cancer")]:
        xraw = p_sample_loop(unet, steps=min(SAMPLE_STEPS, T), y_class=cls, n=SAMPLES_PER_CLASS)
        np.save(OUT_DIR / f"samples_{name}.npy", xraw)
        print(f"Saved {name} samples ->", OUT_DIR / f"samples_{name}.npy")

