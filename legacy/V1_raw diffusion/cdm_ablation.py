#!/usr/bin/env python3
"""
Ablation runner for 1D conditional diffusion model on FTIR spectra.

- architecture (1D U-Net with self-attention + class-conditioning)
-  preprocessing (ALS baseline → SG 2nd derivative → vector norm, 900–1800 cm⁻¹ window)
- evaluation (FID on reconstructed x0, simple peak-region checks, exemplar plots)
- (noise, signal/x0, peak-weighted MSE, triplet)

Usage examples:
  python ablation_diffusion_runner.py --modes 1,2,3,4 --train_csv train_set.csv --val_csv val_set.csv --test_csv test_set.csv \
      --epochs 50 --batch_size 64 --lr 5e-5 --save_dir runs/ablation_v1



Notes:
- Binary setup: drop class==4 (Hyperplasia) then binary_class = (class != 0)
- Cond-embedding uses num_classes=2. Guidance scale supported for sampling.
- Torch compile is optional (enable via --compile) in case of older CUDA on cluster.
"""
import os
import math
import json
import time
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import savgol_filter
from scipy.linalg import sqrtm

FID_IMPL_TAG = "[eval] FID = eigensqrt + jittered cov; eps=1e-6; L2-normalized x0_p"

# -------------------------------
# 0. Args & small utils
# -------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, default="train_set.csv")
    p.add_argument("--val_csv", type=str, default="val_set.csv")
    p.add_argument("--test_csv", type=str, default="test_set.csv")
    p.add_argument("--wn_min", type=float, default=900.0)
    p.add_argument("--wn_max", type=float, default=1800.0)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--compile", action="store_true")

    p.add_argument("--timesteps", type=int, default=250)
    p.add_argument("--temb", type=int, default=128)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--guidance_scale", type=float, default=1.0)

    p.add_argument("--modes", type=str, default="1,2,3,4",
                   help="Comma list among {1,2,3,4}:\n"
                        "1: noise+signal, 2: +peak, 3: +triplet, 4: weighted (0.5,6,5,1)")
    p.add_argument("--save_dir", type=str, default="runs/ablation")
    return p.parse_args()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------
# 1. Preprocessing (ALS → SG(d2) → vector norm) and data loading
# -------------------------------

def baseline_als(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = np.diff(np.eye(L), 2)
    D = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(niter):
        W = np.diag(w)
        Z = np.linalg.inv(W + D)
        z = Z.dot(w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def apply_baseline_correction(X, lam=1e5, p=0.01, niter=10):
    X_corr = np.zeros_like(X)
    for i in range(X.shape[0]):
        bl = baseline_als(X[i, :], lam=lam, p=p, niter=niter)
        X_corr[i, :] = X[i, :] - bl
    return X_corr


def vector_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


def _select_feature_cols(df, wn_min, wn_max):
    non_feat = {"sample_id", "class", "patient_id", "binary_class"}
    feat_cols = [c for c in df.columns if c not in non_feat]
    wns = np.array([float(c) for c in feat_cols])
    mask = (wns >= wn_min) & (wns <= wn_max)
    return list(np.array(feat_cols)[mask])


def _prepare_df(df):
    df = df[df['class'] != 4].copy()
    df['binary_class'] = (df['class'] != 0).astype(int)
    return df


def load_split(csv_path, wn_min, wn_max, scaler_state=None):
    df = pd.read_csv(csv_path)
    df = _prepare_df(df)
    cols = _select_feature_cols(df, wn_min, wn_max)
    X = df[cols].values
    # Apply preprocessing
    X_bc = apply_baseline_correction(X)
    X_sg = savgol_filter(X_bc, window_length=5, polyorder=2, deriv=2, axis=1)
    X_norm = vector_normalize(X_sg)
    y = df['binary_class'].values.astype(np.int64)
    return X_norm, y, cols


# -------------------------------
# 2. Diffusion utilities
# -------------------------------

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x/timesteps)+s)/(1+s) * math.pi/2)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:]/alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def compute_alpha(betas):
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bars


def forward_diffusion_ddpm(x0, t, alpha_bars):
    t_idx = t - 1
    sqrt_ab = torch.sqrt(alpha_bars[t_idx]).unsqueeze(1).unsqueeze(1)
    sqrt_omb = torch.sqrt(1 - alpha_bars[t_idx]).unsqueeze(1).unsqueeze(1)
    noise = torch.randn_like(x0)
    x_t = sqrt_ab * x0 + sqrt_omb * noise
    return x_t, noise


# -------------------------------
# 3. Model (same as your working UNet1D with attention + cond)
# -------------------------------

def timestep_embedding(timesteps, embed_dim):
    half = embed_dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps.unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if embed_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class SelfAttention1D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.q = nn.Conv1d(in_ch, in_ch, 1)
        self.k = nn.Conv1d(in_ch, in_ch, 1)
        self.v = nn.Conv1d(in_ch, in_ch, 1)
        self.scale = in_ch**-0.5
    def forward(self, x):
        Q, K, V = self.q(x), self.k(x), self.v(x)
        w = torch.softmax(torch.bmm(Q.transpose(1,2), K) * self.scale, dim=-1)
        return torch.bmm(w, V.transpose(1,2)).transpose(1,2) + x


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(temb, out_ch)
        self.attn = SelfAttention1D(out_ch)
        self.act = nn.ReLU()
        self.skip = (in_ch == out_ch)
    def forward(self, x, t_emb):
        h = self.conv1(x) + self.time_mlp(t_emb).unsqueeze(-1)
        h = self.act(h)
        h = self.conv2(h)
        h = self.attn(h)
        if self.skip:
            h = h + x
        return self.act(h)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, temb)
        self.pool = nn.Conv1d(out_ch, out_ch, 2, stride=2)
    def forward(self, x, t_emb):
        skip = self.res(x, t_emb)
        return skip, self.pool(skip)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb):
        super().__init__()
        self.trans = nn.ConvTranspose1d(in_ch, out_ch, 2, stride=2)
        self.res = ResidualBlock(in_ch + out_ch, out_ch, temb)
    def forward(self, x, skip, t_emb):
        x = self.trans(x)
        diff = skip.size(-1) - x.size(-1)
        if diff > 0:
            x = F.pad(x, (0, diff))
        return self.res(torch.cat([x, skip], dim=1), t_emb)


class UNet1D(nn.Module):
    def __init__(self, num_classes=2, temb=128, base_ch=64):
        super().__init__()
        self.cond_emb = nn.Embedding(num_classes, temb)
        self.time_mlp = nn.Sequential(nn.Linear(temb, temb), nn.ReLU(), nn.Linear(temb, temb))
        self.down1 = DownBlock(1, base_ch, temb)
        self.down2 = DownBlock(base_ch, base_ch*2, temb)
        self.down3 = DownBlock(base_ch*2, base_ch*2, temb)
        self.mid   = ResidualBlock(base_ch*2, base_ch*2, temb)
        self.up1   = UpBlock(base_ch*2, base_ch*2, temb)
        self.up2   = UpBlock(base_ch*2, base_ch, temb)
        self.up3   = UpBlock(base_ch, base_ch, temb)
        self.out   = nn.Conv1d(base_ch, 1, kernel_size=1)
    def forward(self, x, t, cond):
        te = timestep_embedding(t, self.time_mlp[0].in_features)
        te = self.time_mlp(te)
        c_emb = self.cond_emb(cond) if cond is not None else torch.zeros_like(te)
        t_emb = te + c_emb
        s1, d1 = self.down1(x, t_emb)
        s2, d2 = self.down2(d1, t_emb)
        s3, d3 = self.down3(d2, t_emb)
        m     = self.mid(d3, t_emb)
        u1    = self.up1(m, s3, t_emb)
        u2    = self.up2(u1, s2, t_emb)
        u3    = self.up3(u2, s1, t_emb)
        return self.out(u3)


# -------------------------------
# 4. Sampling & evaluation helpers
# -------------------------------

def p_sample_ddpm(model, x_t, t, betas, alpha_bars, device, cond, guidance_scale=1.0):
    beta_t = betas[t-1]
    alpha_t = 1 - beta_t
    sqrt_rec = 1.0 / torch.sqrt(alpha_t)
    sqrt_omb = torch.sqrt(1 - alpha_bars[t-1])
    t_tensor = torch.full((x_t.size(0),), t, dtype=torch.long, device=device)
    un = model(x_t, t_tensor.float(), None)
    cn = model(x_t, t_tensor.float(), cond)
    pred = un + guidance_scale * (cn - un)
    denom = torch.sqrt(alpha_bars[t-1]).view(-1,1,1)
    x0 = (x_t - sqrt_omb * pred) / denom
    mean = sqrt_rec * (x_t - (beta_t / sqrt_omb) * pred)
    if t > 1:
        return mean + torch.sqrt(beta_t) * torch.randn_like(x_t)
    return mean


def sample_ddpm(model, cond, timesteps, betas, alpha_bars, device, guidance_scale, length):
    xt = torch.randn(1,1,length,device=device)
    for tt in range(timesteps, 0, -1):
        xt = p_sample_ddpm(model, xt, tt, betas, alpha_bars, device, cond, guidance_scale)
    return xt


def create_peak_mask(length=234,
                     start_wn=1797.53,
                     end_wn=898.764,
                     peak_positions=(1446.51, 1377.08, 1234.35, 1045.34, 902.622),
                     peak_weight=6.0,
                     window_size=2):
    wns = np.linspace(start_wn, end_wn, length)
    mask = np.ones(length, dtype=np.float32)
    for p in peak_positions:
        idx = np.abs(wns - p).argmin()
        for i in range(idx-window_size, idx+window_size+1):
            if 0 <= i < length:
                mask[i] = peak_weight
    return mask


def peak_weighted_mse(xp, xt, mask):
    w = torch.from_numpy(mask).to(xp.device).view(1,1,-1)
    return ((xp - xt)**2 * w).mean()


class TripletLossWithHardNegatives(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    def forward(self, anc, pos, neg):
        pd = F.pairwise_distance(anc, pos)
        nd = F.pairwise_distance(anc, neg)
        return F.relu(pd - nd + self.margin).mean()


def cosine_similarity(a, b):
    a_n = a / a.norm(dim=1, keepdim=True)
    b_n = b / b.norm(dim=1, keepdim=True)
    return (a_n @ b_n.T).squeeze()


def _cov(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
	c = np.cov(x, rowvar=False)
 # jitter on diag to avoid singularity
	return c + np.eye(c.shape[0]) * eps

def _sqrt_psd(m: np.ndarray) -> np.ndarray:
    # symmetric PSD matrix square root via eigh (more stable than sqrtm for our use)
    w, v = np.linalg.eigh(m)
    w = np.clip(w, 0.0, None)
    return (v * np.sqrt(w)) @ v.T


def calculate_fid(real: np.ndarray, fake: np.ndarray, eps: float = 1e-6) -> float:
	real = real.astype(np.float64)
	fake = fake.astype(np.float64)
	mu1, mu2 = real.mean(0), fake.mean(0)
	sig1, sig2 = _cov(real, eps), _cov(fake, eps)
	ssd = float(np.sum((mu1 - mu2) ** 2))
	covmean = _sqrt_psd(sig1) @ _sqrt_psd(sig2)
	fid = ssd + float(np.trace(sig1 + sig2 - 2.0 * covmean))
# numerical guard
	if np.isnan(fid) or np.isinf(fid):
		return float("inf")
	return fid


def plot_single_class(real, synth, class_label, length=234, save_dir="hpc_results"):
    os.makedirs(save_dir, exist_ok=True)
    real = np.asarray(real).squeeze()
    synth = np.asarray(synth).squeeze()
    wns = np.linspace(1797.53, 898.764, length)

    assert real.shape == (length,), f"real has shape {real.shape}, expected ({length},)"
    assert synth.shape == (length,), f"synth has shape {synth.shape}, expected ({length},)"

    plt.figure(figsize=(10,5))
    plt.plot(wns, real, label="Real")
    plt.plot(wns, synth, '--', label="Synth")
    for p in [1446,1377,1234,1045,900]:
        plt.axvline(p, linestyle=':', color='gray')
    plt.gca().invert_xaxis()
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{save_dir}/class_{class_label}.png", dpi=200)
    plt.close()


def verify_peak_regions(real, synth, wns, peaks, window_size=2):
    rows = []
    for p in peaks:
        idx = np.abs(wns - p).argmin()
        r = real[max(0,idx-window_size):idx+window_size+1]
        s = synth[max(0,idx-window_size):idx+window_size+1]
        wn_range = wns[max(0,idx-window_size):idx+window_size+1]
        area_r = np.trapz(r, wn_range)
        area_s = np.trapz(s, wn_range)
        pct = abs(area_r - area_s) / (abs(area_r) + 1e-8) * 100
        rows.append({"peak": p, "area_delta_percent": float(pct)})
    return rows


# -------------------------------
# 5. Training step with ablation weights
# -------------------------------

def train_one_mode(mode, args, device, X_train, y_train, X_val, y_val, length):
    # Prepare run dir
    run_dir = os.path.join(args.save_dir, f"mode{mode}")
    os.makedirs(run_dir, exist_ok=True)

    # Diffusion schedule
    betas = cosine_beta_schedule(args.timesteps).to(device)
    _, alpha_bars = compute_alpha(betas)

    # Model
    model = UNet1D(num_classes=2, temb=args.temb, base_ch=args.base_ch).to(device)
    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    mse_loss = nn.MSELoss()
    triplet_loss = TripletLossWithHardNegatives(margin=1.0)
    peak_mask = create_peak_mask(length=length)

    # Ablation weights
    if mode == 1:
        w_noise, w_sig, w_peak, w_trip = 1.0, 1.0, 0.0, 0.0
    elif mode == 2:
        w_noise, w_sig, w_peak, w_trip = 1.0, 1.0, 1.0, 0.0
    elif mode == 3:
        w_noise, w_sig, w_peak, w_trip = 1.0, 1.0, 1.0, 1.0
    else:
        w_noise, w_sig, w_peak, w_trip = 0.5, 6.0, 5.0, 1.0
    print(f"Mode {mode} weights: noise={w_noise}, signal={w_sig}, peak={w_peak}, triplet={w_trip}")

    # Data
    train_ds = TensorDataset(torch.tensor(X_train[:,None,:], dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    val_ds   = TensorDataset(torch.tensor(X_val[:,None,:], dtype=torch.float32),
                             torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # Train
    log_rows = []
    best_val = 1e9
    start_time = time.time()
    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            t = torch.randint(1, args.timesteps+1, (xb.size(0),), device=device)
            xt, noise = forward_diffusion_ddpm(xb, t, alpha_bars)
            cond = None if random.random() < 0.1 else yb

            pred_noise = model(xt, t.float(), cond)
            loss_n = mse_loss(pred_noise, noise)

            # Predict x0 and compute signal + peak losses
            t_idx = t - 1
            sqrt_ab = torch.sqrt(alpha_bars[t_idx]).view(-1,1,1)
            sqrt_omb = torch.sqrt(1 - alpha_bars[t_idx]).view(-1,1,1)
            x0_pred = (xt - sqrt_omb * pred_noise) / sqrt_ab
            x0_pred = F.normalize(x0_pred, p=2, dim=-1)

            loss_s = mse_loss(x0_pred, xb)
            loss_p = peak_weighted_mse(x0_pred, xb, peak_mask)

            # Triplet with simple hard neg construction within batch
            loss_t = torch.tensor(0.0, device=device)
            if w_trip > 0 and xb.size(0) >= 3:
                # pick anchors as xb, pos as same-class exemplar, neg as different-class exemplar
                pos_mask = (yb == yb[0])
                neg_mask = ~pos_mask
                if pos_mask.sum() > 1 and neg_mask.sum() > 0:
                    pos = xb[pos_mask][0].squeeze(1)
                    neg = xb[neg_mask][0].squeeze(1)
                    loss_t = triplet_loss(xb.squeeze(1), pos.repeat(xb.size(0),1), neg.repeat(xb.size(0),1))

            loss = w_noise*loss_n + w_sig*loss_s + w_peak*loss_p + w_trip*loss_t

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            losses.append(float(loss.item()))

        # "Validation" proxy: average training loss this epoch (you can swap in a real proxy if desired)
        val_metric = float(np.mean(losses))
        scheduler.step(val_metric)

        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": optimizer.param_groups[0]['lr']}
        log_rows.append(row)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Mode{mode} Epoch {epoch:02d} | Loss {row['train_loss']:.4f} | LR {row['lr']:.2e}")

        # Save best-by-proxy checkpoint
        if val_metric < best_val:
            best_val = val_metric
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "args": vars(args)},
                       os.path.join(run_dir, "best.pth"))

    torch.save(model.state_dict(), os.path.join(run_dir, f"model_final.pth"))
    pd.DataFrame(log_rows).to_csv(os.path.join(run_dir, "train_log.csv"), index=False)

    train_secs = time.time() - start_time

    # ---------------- Eval: exemplar per class + FID on recon ----------------
    model.eval()
    print(FID_IMPL_TAG)
    with torch.no_grad():
        classes = [0,1]
        class_results = {}
        for cl in classes:
            cond = torch.tensor([cl], dtype=torch.long, device=device)
            candidates = []
            for _ in range(10):
                synth = sample_ddpm(model, cond, args.timesteps, betas, alpha_bars, device, args.guidance_scale, length)
                candidates.append(synth)
            candidates = torch.cat(candidates, dim=0)

            # pick a real exemplar from training set
            real_mask = (torch.tensor(y_train) == cl)
            if real_mask.sum() == 0:
                continue
            real_ex = torch.tensor(X_train[real_mask][0][None,None,:], dtype=torch.float32, device=device)
            real_flat = F.normalize(real_ex.view(1, -1), p=2, dim=1)
            cand_flat = F.normalize(candidates.view(candidates.size(0), -1), p=2, dim=1)
            sims = cosine_similarity(cand_flat, real_flat)
            best = int(torch.argmax(sims).item())

            real_arr  = real_ex.squeeze().cpu().numpy()
            synth_arr = candidates[best].squeeze().cpu().numpy()
            class_results[cl] = {'real': real_arr, 'synth': synth_arr}

        # plots + peak checks
        wn_axis = np.linspace(1797.53, 898.764, length)
        peak_rows = []
        for cl, data in class_results.items():
            plot_single_class(data['real'], data['synth'], cl, length, save_dir=os.path.join(run_dir, "plots"))
            rows = verify_peak_regions(data['real'], data['synth'], wn_axis, [1446,1377,1234,1045,900])
            for r in rows:
                r.update({"class": int(cl)})
            peak_rows.extend(rows)
        if peak_rows:
            pd.DataFrame(peak_rows).to_csv(os.path.join(run_dir, "peak_checks.csv"), index=False)

        # FID on reconstructed x0 vs real
        real_embeds, fake_embeds = [], []
        # Use validation split to avoid eval on train
        val_tensor = torch.tensor(X_val[:,None,:], dtype=torch.float32, device=device)
        val_labels = torch.tensor(y_val, dtype=torch.long, device=device)

        for i in range(0, val_tensor.size(0), args.batch_size):
            xb = val_tensor[i:i+args.batch_size]
            yb = val_labels[i:i+args.batch_size]
            if xb.size(0) == 0:
                continue
            t_rand = torch.randint(1, args.timesteps+1, (xb.size(0),), device=device)
            noise = torch.randn_like(xb)
            x_t = xb * torch.sqrt(alpha_bars[t_rand-1]).view(-1,1,1) + noise * torch.sqrt(1 - alpha_bars[t_rand-1]).view(-1,1,1)
            pred_n = model(x_t, t_rand.float(), yb)
            x0_p = (x_t - torch.sqrt(1 - alpha_bars[t_rand-1]).view(-1,1,1) * pred_n) / torch.sqrt(alpha_bars[t_rand-1]).view(-1,1,1)
            x0_p = F.normalize(x0_p, p=2, dim=-1)
            real_embeds.append(xb.squeeze(1).cpu().numpy())
            fake_embeds.append(x0_p.squeeze(1).cpu().numpy())

        real_array = np.concatenate(real_embeds, axis=0) if real_embeds else np.zeros((1,length))
        fake_array = np.concatenate(fake_embeds, axis=0) if fake_embeds else np.zeros((1,length))
        fid_score = calculate_fid(real_array, fake_array)

    # Save summary
    summary = {
        "mode": int(mode),
        "weights": {"noise": w_noise, "signal": w_sig, "peak": w_peak, "triplet": w_trip},
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "length": int(length),
        "train_seconds": float(train_secs),
        "fid": float(fid_score),
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Mode {mode} finished | FID={fid_score:.2f} | secs={int(train_secs)} | saved to {run_dir}")
    return summary


# -------------------------------
# 6. Main: load data once, run modes
# -------------------------------

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load splits once with same preprocessing
    Xtr, ytr, cols_tr = load_split(args.train_csv, args.wn_min, args.wn_max)
    Xva, yva, cols_va = load_split(args.val_csv,   args.wn_min, args.wn_max)
    Xte, yte, cols_te = load_split(args.test_csv,  args.wn_min, args.wn_max)

    # Sanity on feature alignment
    if cols_tr != cols_va or cols_tr != cols_te:
        raise ValueError("Feature columns (wavenumbers) differ across splits. Rebuild CSVs with identical columns.")

    length = Xtr.shape[1]
    print(f"Loaded data | length={length} | train={len(Xtr)} val={len(Xva)} test={len(Xte)} | device={device}")

    modes = [int(m.strip()) for m in args.modes.split(',') if m.strip()]
    results = []
    for m in modes:
        res = train_one_mode(m, args, device, Xtr, ytr, Xva, yva, length)
        results.append(res)

    # Save global results table
    pd.DataFrame(results).to_csv(os.path.join(args.save_dir, "ablation_results.csv"), index=False)
    print("All modes done. Results written to ablation_results.csv")


if __name__ == "__main__":
    main()
