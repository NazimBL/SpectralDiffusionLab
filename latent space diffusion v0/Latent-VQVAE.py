#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, math, random
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as Fnn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# ======== EDIT ========
RAW_TRAIN = Path(r"../MyDataset/ftir_train_wn.csv")
OUT_DIR = Path(r"ldm_out")
BATCH = 64
LR = 3e-4
EPOCHS = 300
PATIENCE = 40
LATENT_C = 64
DOWNS = 4
NUM_CODES = 512
COMMIT_BETA = 0.25
REC_WEIGHT = 10.0  # <--- NEW: Force model to prioritize reconstruction
VAL_RATIO = 0.10
SEED = 42
# ======================

torch.manual_seed(SEED);
np.random.seed(SEED);
random.seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

META = {"groupnumbers", "classes", "class_name", "binary_label", "groupcodes", "obsnames"}


def spectral_cols(df):
    cols = [];
    for c in df.columns:
        if c in META: continue
        try:
            float(c);
            cols.append(c)
        except:
            pass
    if not cols: raise ValueError("No wavenumber columns detected.")
    return [c for c in df.columns if c in set(cols)]


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X): return (X - self.mean) / (self.std + 1e-12)

    def inverse_transform(self, Xn): return Xn * self.std + self.mean


class RawPairs(Dataset):
    def __init__(self, csv_path, scaler):
        df = pd.read_csv(csv_path)
        self.cols = spectral_cols(df)
        X = df[self.cols].to_numpy(dtype=np.float32)
        y = (df["classes"].values != 0).astype(np.int64)

        if scaler is None:
            mean = X.mean(axis=0)
            std = X.std(axis=0) + 1e-12
            self.scaler = Scaler(mean=mean.astype(np.float32), std=std.astype(np.float32))
        else:
            self.scaler = scaler

        Xn = self.scaler.transform(X).astype(np.float32)
        self.X = Xn
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i])[None, :]
        y = torch.tensor(self.y[i])
        return x, y


# -------- Vector Quantizer Layer (Unchanged from your file) --------
class VectorQuantizer(nn.Module):
    def __init__(self, num_codes, embed_dim, beta):
        super().__init__()
        self.K = num_codes
        self.C = embed_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.K, self.C)
        self.embedding.weight.data.uniform_(-1.0 / self.K, 1.0 / self.K)

    def forward(self, z_e):
        B, C, L = z_e.shape
        z_e_flat = z_e.permute(0, 2, 1).contiguous().view(-1, self.C)
        dist = torch.sum(z_e_flat ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(z_e_flat, self.embedding.weight.t())
        indices = torch.argmin(dist, dim=1)
        z_q_flat = self.embedding(indices)
        z_q = z_q_flat.view(B, L, C).permute(0, 2, 1).contiguous()

        loss_vq = Fnn.mse_loss(z_e.detach(), z_q)
        loss_commit = Fnn.mse_loss(z_e, z_q.detach())

        z_q_ste = z_e + (z_q - z_e).detach()
        return z_q_ste, loss_vq, loss_commit, indices.view(B, L)


# -------- Conv VQ-VAE (Unchanged from your file) --------
class ConvVQVAE(nn.Module):
    def __init__(self, F: int, downs: int, base: int, latent_c: int, num_codes: int, beta: float):
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
        self.pre_quant_conv = nn.Conv1d(latent_c, latent_c, 1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=latent_c)
        self.quantizer = VectorQuantizer(num_codes, latent_c, beta)
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

    def encode(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quant_conv(z_e)
        z_e = self.norm(z_e)
        _z_q_ste, _loss_vq, _loss_commit, indices = self.quantizer(z_e)
        return indices

    def decode(self, indices):
        z_q_flat = self.quantizer.embedding(indices.view(-1))
        B, L = indices.shape
        z_q = z_q_flat.view(B, L, self.latent_c).permute(0, 2, 1).contiguous()
        xr = self.decode_from_quantized(z_q)
        return xr

    # --- FORWARD PASS MODIFIED ---
    def forward(self, x):
        # 1. Encode
        z_e = self.encoder(x)
        z_e_norm = self.pre_quant_conv(z_e)
        z_e_norm = self.norm(z_e_norm)

        # 2. Quantize
        z_q, loss_vq, loss_commit, _indices = self.quantizer(z_e_norm)

        # 3. Decode
        xr = self.decode_from_quantized(z_q)

        return xr, loss_vq, loss_commit  # <--- Reverted to standard VQ-VAE outputs

    def decode_from_quantized(self, z_q):
        y = self.decoder(z_q)
        if y.shape[-1] != self.F:
            if y.shape[-1] > self.F:
                start = (y.shape[-1] - self.F) // 2
                y = y[..., start:start + self.F]
            else:
                pad = self.F - y.shape[-1]
                y = Fnn.pad(y, (pad // 2, pad - pad // 2))
        raw = self.to_raw(y)
        return raw


# --------- EarlyStopping (Unchanged) ---------
class EarlyStop:
    def __init__(self, patience=PATIENCE):
        self.best = None;
        self.wait = 0;
        self.stop = False

    def step(self, metric):
        if self.best is None or metric < self.best - 1e-6:
            self.best = float(metric);
            self.wait = 0
            return True
        else:
            self.wait += 1
            if self.wait >= PATIENCE: self.stop = True
            return False


# --------- Train / Val loop (MODIFIED) ---------
def train_val():
    df = pd.read_csv(RAW_TRAIN)
    cols = spectral_cols(df)
    X = df[cols].to_numpy(np.float32)
    y = (df["classes"].values != 0).astype(np.int64)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
    tr_idx, va_idx = next(sss.split(X, y))

    mean = X[tr_idx].mean(axis=0)
    std = X[tr_idx].std(axis=0) + 1e-12
    scaler = Scaler(mean=mean.astype(np.float32), std=std.astype(np.float32))

    ds_full = RawPairs(RAW_TRAIN, scaler=scaler)
    ds_tr = Subset(ds_full, tr_idx.tolist())
    ds_va = Subset(ds_full, va_idx.tolist())

    F = X.shape[1]
    model = ConvVQVAE(F, downs=DOWNS, base=64, latent_c=LATENT_C,
                      num_codes=NUM_CODES, beta=COMMIT_BETA).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    # --- STEP FUNCTION MODIFIED ---
    def step(loader, train=True):
        if train:
            model.train()
        else:
            model.eval()
        losses, recs, vqs, commits = [], [], [], []

        for xb, _yb in loader:
            xb = xb.to(DEVICE)
            if train: opt.zero_grad()

            # VQ-VAE Forward
            xr, loss_vq, loss_commit = model(xb)

            # 1. Reconstruction Loss (MSE)
            loss_rec = Fnn.mse_loss(xr.squeeze(1), xb.squeeze(1))

            # 2. Total VQ-VAE Loss (Restored standard loss + new weight)
            loss = (REC_WEIGHT * loss_rec) + loss_vq + (COMMIT_BETA * loss_commit)

            if train:
                loss.backward()
                opt.step()

            losses.append(loss.item())
            recs.append(loss_rec.item())
            vqs.append(loss_vq.item())
            commits.append(loss_commit.item())

        return np.mean(losses), np.mean(recs), np.mean(vqs), np.mean(commits)

    tr_loader = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, drop_last=True)
    va_loader = DataLoader(ds_va, batch_size=BATCH, shuffle=False)

    stopper = EarlyStop(patience=PATIENCE)
    best_state = None

    hist = []
    print(f"Starting VQ-VAE Training (K={NUM_CODES}, C={LATENT_C}, beta={COMMIT_BETA}, REC_WEIGHT={REC_WEIGHT})...")

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_rec, tr_vq, tr_com = step(tr_loader, train=True)
        va_loss, va_rec, va_vq, va_com = step(va_loader, train=False)

        hist.append({
            "epoch": ep,
            "tr_loss": tr_loss, "va_loss": va_loss,
            "tr_rec": tr_rec, "va_rec": va_rec,
            "tr_vq": tr_vq, "va_vq": va_vq,
            "tr_commit": tr_com, "va_commit": va_com,
        })

        # Updated print statement
        print(f"Ep {ep:03d} | L: {tr_loss:.4f}/{va_loss:.4f} | "
              f"Rec: {tr_rec:.4f}/{va_rec:.4f} | "
              f"VQ: {tr_vq:.4f}/{va_vq:.4f}")

        if stopper.step(va_loss):
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        if stopper.stop:
            print(f"Early stopping at epoch {ep}.")
            break

    assert best_state is not None
    torch.save(best_state, OUT_DIR / "ae_conv1d.pt")

    meta = {
        "model_type": "VQVAE",
        "F": int(F),
        "downs": int(DOWNS),
        "latent_channels": int(model.latent_c),
        "latent_length": int(model.latent_L),
        "num_codes": int(NUM_CODES),
        "commit_beta": float(COMMIT_BETA),
        "rec_weight": float(REC_WEIGHT),  # Added new param
        "cols": cols,
        "scaler_mean": ds_full.scaler.mean.tolist(),
        "scaler_std": ds_full.scaler.std.tolist(),
        "seed": SEED
    }
    with open(OUT_DIR / "ae_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Plot
    dfh = pd.DataFrame(hist)
    # *** FIX: Corrected typo OUT_DIT -> OUT_DIR ***
    dfh.to_csv(OUT_DIR / "ae_train_history.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(dfh["epoch"], dfh["tr_loss"], label="Train Loss")
    plt.plot(dfh["epoch"], dfh["va_loss"], label="Val Loss")
    plt.title("Total Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(dfh["epoch"], dfh["tr_rec"], label="Train Rec")
    plt.plot(dfh["epoch"], dfh["va_rec"], label="Val Rec")
    plt.title("Reconstruction MSE")
    plt.legend()

    plt.tight_layout();
    plt.savefig(OUT_DIR / "ae_loss.png", dpi=200);
    plt.close()
    print("Saved VQ-VAE artifacts.")


if __name__ == "__main__":
    train_val()