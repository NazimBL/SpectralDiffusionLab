#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone script to load the trained AE and DDPM models
and generate a specified number of new samples for each class.
"""

import math, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

# ======== CONFIGURATION ========
# Samples to generate
N_HEALTHY_SAMPLES = 50
N_CANCER_SAMPLES = 50

# trained model files
OUT_DIR = Path("ldm_out")
AE_WEIGHTS_FILE = OUT_DIR / "ae_conv1d.pt"
DDPM_CHECKPOINT_FILE = OUT_DIR / "ddpm_latent_unet.pt"

# Output file
HEALTHY_OUT_FILE = OUT_DIR / "generated_healthy_data.npy"
CANCER_OUT_FILE = OUT_DIR / "generated_cancer_data.npy"

# Sampling steps
SAMPLE_STEPS = 1000
# ===============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# -----------------------------------------------------------------
# MODEL DEFINITIONS
# -----------------------------------------------------------------

def gnorm(c):
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class ConvAE(nn.Module):

    def __init__(self, F: int, downs: int = 4, base: int = 64, latent_c: int = 64):
        super().__init__()
        self.F = F
        # encoder
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

        # Decoder
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

        # latent space
        self.align_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(latent_c, 256), nn.SiLU(),
            nn.Linear(256, F)
        )

    def decode(self, h):  # (B,latent_c,latent_L) -> (B,1,F) scaled units
        y = self.decoder(h)
        if y.shape[-1] != self.F:
            if y.shape[-1] > self.F:
                diff = y.shape[-1] - self.F
                start = diff // 2
                y = y[..., start: start + self.F]
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
        self.gn1 = gnorm(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.gn2 = gnorm(cout)
        self.act = nn.SiLU()
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, cout))
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cvec):
        h = self.act(self.gn1(self.conv1(x)))
        b = self.cond(cvec).unsqueeze(-1)  # (B,cout,1)
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

        # Encoder
        self.rb1 = ResBlock1D(in_ch, base, cond_dim)
        self.down1 = nn.Conv1d(base, base, 4, stride=2, padding=1)
        self.rb2 = ResBlock1D(base, base * 2, cond_dim)
        self.down2 = nn.Conv1d(base * 2, base * 2, 4, stride=2, padding=1)
        # Mid
        self.mid1 = ResBlock1D(base * 2, base * 4, cond_dim)
        self.mid2 = ResBlock1D(base * 4, base * 4, cond_dim)
        # Decoder
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


# -----------------------------------------------------------------
# SAMPLING LOOP
# -----------------------------------------------------------------

def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps, dtype=torch.float32)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return betas.clamp(1e-8, 0.999)


@torch.no_grad()
def p_sample_loop(unet, ae, T, steps, y_class, n,
                  latent_c, latent_L,
                  x_mean, x_std, z_mu, z_std):

    # Load diffusion schedule parameters
    betas_s = cosine_beta_schedule(T).to(DEVICE)
    alphas_s = 1.0 - betas_s
    ac_s = torch.cumprod(alphas_s, dim=0)
    sqrt_recip_alphas = (1.0 / torch.sqrt(alphas_s)).to(DEVICE)

    # Start from noise in latent space
    z_t = torch.randn(n, latent_c, latent_L, device=DEVICE)
    y = torch.full((n,), int(y_class), device=DEVICE, dtype=torch.long)

    # Use evenly spaced timesteps
    ts = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=DEVICE)

    print(f"Sampling class {y_class} ({n} samples)...")
    for t_val in ts:
        t = t_val.repeat(n)
        eps_hat = unet(z_t, t, y)

        # DDPM reverse step
        beta_t = betas_s[t].view(-1, 1, 1)
        sqrt_one_minus_ac_t = torch.sqrt(1.0 - ac_s[t]).view(-1, 1, 1)
        sqrt_recip_alpha_t = sqrt_recip_alphas[t].view(-1, 1, 1)

        mean = sqrt_recip_alpha_t * (z_t - beta_t / sqrt_one_minus_ac_t * eps_hat)

        if (t_val > 0):
            noise = torch.randn_like(z_t)
            z_t = mean + torch.sqrt(beta_t) * noise
        else:
            z_t = mean  # Final step, no noise

    # z_t is now the final predicted *normalized* z0

    # === 1 FIX I made : Rescale the generated latents ===
    # Calculate the stats of the generated batch (per-position)
    z_gen_mean = z_t.mean(dim=0, keepdim=True)
    z_gen_std = z_t.std(dim=0, keepdim=True)

    # Re-normalize the latents to have std=1, mean=0
    z_t_renormed = (z_t - z_gen_mean) / z_gen_std.clamp(1e-6)

    # Un-normalize the *re-normalized* latent vector
    z_t_unnorm = z_t_renormed * z_std + z_mu

    # Decode to raw
    x_scaled = ae.decode(z_t_unnorm)

    # Un-scale spectra back to original absorbance units
    x_raw = (x_scaled * x_std + x_mean).squeeze(1).detach().cpu().numpy()
    return x_raw


# -----------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------

def main():
    if not DDPM_CHECKPOINT_FILE.exists():
        print(f"Error: DDPM checkpoint not found at: {DDPM_CHECKPOINT_FILE}")
        return
    if not AE_WEIGHTS_FILE.exists():
        print(f"Error: AE weights not found at: {AE_WEIGHTS_FILE}")
        return

    # --- 1. Load DDPM Checkpoint & Meta ---
    ckpt = torch.load(DDPM_CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)
    meta = ckpt["meta"]

    # Get AE parameters
    F_LEN = int(meta["F"])
    downs = int(meta["downs"])
    latent_c = int(meta["latent_channels"])
    latent_L = int(meta["latent_length"])
    T_trained = int(ckpt["T"])

    # Get scaling stats
    x_mean = torch.from_numpy(np.array(meta["scaler_mean"], dtype=np.float32)).to(DEVICE).view(1, 1, -1)
    x_std = torch.from_numpy(np.array(meta["scaler_std"], dtype=np.float32)).to(DEVICE).view(1, 1, -1)
    z_mu = ckpt["z_mu"].to(DEVICE)
    z_std = ckpt["z_std"].to(DEVICE)

    print(f"Loaded meta: F={F_LEN}, C={latent_c}, L={latent_L}, T={T_trained}")

    # --- 2. Load Autoencoder ---
    ae = ConvAE(F_LEN, downs=downs, latent_c=latent_c).to(DEVICE)
    ae_sd = torch.load(AE_WEIGHTS_FILE, map_location=DEVICE, weights_only=False)
    ae.load_state_dict(ae_sd, strict=False)  # strict=False to ignore align_head if missing
    for p in ae.parameters(): p.requires_grad = False
    ae.eval()
    print("Autoencoder loaded.")

    # --- 3. Load U-Net ---
    unet = UNet1D_Cond(in_ch=latent_c, base=128, out_ch=latent_c).to(DEVICE)
    unet.load_state_dict(ckpt["model"])
    for p in unet.parameters(): p.requires_grad = False
    unet.eval()
    print("DDPM U-Net loaded.")

    # --- 4. Generate Healthy Samples ---
    if N_HEALTHY_SAMPLES > 0:
        x_raw_healthy = p_sample_loop(
            unet=unet, ae=ae, T=T_trained, steps=min(SAMPLE_STEPS, T_trained),
            y_class=0, n=N_HEALTHY_SAMPLES,
            latent_c=latent_c, latent_L=latent_L,
            x_mean=x_mean, x_std=x_std, z_mu=z_mu, z_std=z_std
        )
        np.save(HEALTHY_OUT_FILE, x_raw_healthy)
        print(f"Saved {N_HEALTHY_SAMPLES} healthy samples -> {HEALTHY_OUT_FILE}")

    # --- 5. Generate Cancer Samples ---
    if N_CANCER_SAMPLES > 0:
        x_raw_cancer = p_sample_loop(
            unet=unet, ae=ae, T=T_trained, steps=min(SAMPLE_STEPS, T_trained),
            y_class=1, n=N_CANCER_SAMPLES,
            latent_c=latent_c, latent_L=latent_L,
            x_mean=x_mean, x_std=x_std, z_mu=z_mu, z_std=z_std
        )
        np.save(CANCER_OUT_FILE, x_raw_cancer)
        print(f"Saved {N_CANCER_SAMPLES} cancer samples -> {CANCER_OUT_FILE}")

    print("Generation complete.")


if __name__ == "__main__":
    main()