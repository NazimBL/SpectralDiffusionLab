#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, math, random
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# ======== EDIT ========
RAW_TRAIN = Path(r"../MyDataset/ftir_train_wn.csv")
OUT_DIR   = Path(r"ldm_out")
BATCH     = 64
LR        = 2e-4
EPOCHS    = 200
PATIENCE  = 20
LATENT_C  = 64           # channels in latent
DOWNS     = 4            # # of stride-2 downsamples (~F/16)
ALIGN_LMB = 1.0          # weight for align loss (preproc)
USE_CLIP  = False        # raw clipping for stability; keep False unless needed
CLIP_VAL  = 5.0
VAL_RATIO = 0.10
SEED      = 42
# ======================

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

META = {"groupnumbers","classes","class_name","binary_label","groupcodes","obsnames"}

def spectral_cols(df):
    cols=[];
    for c in df.columns:
        if c in META: continue
        try: float(c); cols.append(c)
        except: pass
    if not cols: raise ValueError("No wavenumber columns detected.")
    return [c for c in df.columns if c in set(cols)]  # preserve CSV order (descending wn)

def preprocess_row(x_row: np.ndarray) -> np.ndarray:
    win = 5 if x_row.size >= 5 else (x_row.size//2*2+1)
    z = savgol_filter(x_row, window_length=win, polyorder=2, deriv=2)
    n = np.linalg.norm(z) + 1e-12
    return z / n

@dataclass
class Scaler:
    mean: np.ndarray
    std:  np.ndarray
    def transform(self, X): return (X - self.mean) / self.std
    def inverse_transform(self, Xn): return Xn * self.std + self.mean

class RawPairs(Dataset):
    def __init__(self, csv_path, scaler, use_clip=False, clip_val=5.0):
        df = pd.read_csv(csv_path)
        self.cols = spectral_cols(df)
        X = df[self.cols].to_numpy(dtype=np.float32)
        print(X)
        y = (df["classes"].values != 0).astype(np.int64)  # Cancer=1, Healthy=0
        # build paired preprocessed targets
        Z = np.vstack([preprocess_row(r) for r in X]).astype(np.float32)

        # fit scaler if not given
        if scaler is None:
            mean = X.mean(axis=0)
            std  = X.std(axis=0) + 1e-12
            self.scaler = Scaler(mean=mean.astype(np.float32), std=std.astype(np.float32))
        else:
            self.scaler = scaler

        Xn = self.scaler.transform(X).astype(np.float32)
        if use_clip:
            Xn = np.clip(Xn, -clip_val, clip_val)

        self.X = Xn
        self.Z = Z
        self.y = y

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        # (C=1, L=F) for Conv1D
        x = torch.from_numpy(self.X[i])[None, :]              # (1,F)
        z = torch.from_numpy(self.Z[i])                       # (F,)
        y = torch.tensor(self.y[i])
        return x, z, y

# -------- Conv1D Autoencoder with Align Head --------
class ConvAE(nn.Module):
    def __init__(self, F: int, downs: int = 4, base: int = 64, latent_c: int = 64):
        super().__init__()
        self.F = F
        # Encoder: (B,1,F) -> (B,C,F/2^downs)
        c = base
        enc = []
        in_c = 1
        for i in range(downs):
            out_c = latent_c if i == downs-1 else c
            enc += [
                nn.Conv1d(in_c, c, kernel_size=5, stride=1, padding=2),
                nn.SiLU(),
                nn.Conv1d(c, out_c, kernel_size=5, stride=2, padding=2),  # down x2
                nn.SiLU(),
            ]
            in_c = out_c
            c = min(c*2, 256)
        self.encoder = nn.Sequential(*enc)

        # compute latent length dynamically
        with torch.no_grad():
            probe = torch.zeros(1,1,F)
            lat = self.encoder(probe)
            self.latent_c = lat.shape[1]
            self.latent_L = lat.shape[2]

        # Decoder mirrors encoder
        dec = []
        c_in = self.latent_c
        c_cur = c_in
        for i in range(downs):
            c_mid = max(c_cur//2, base) if i < downs-1 else base
            c_out = base if i < downs-1 else 32
            dec += [
                nn.ConvTranspose1d(c_cur, c_mid, kernel_size=4, stride=2, padding=1),  # up x2
                nn.SiLU(),
                nn.Conv1d(c_mid, c_out, kernel_size=5, padding=2),
                nn.SiLU(),
            ]
            c_cur = c_out
        self.decoder = nn.Sequential(*dec)
        self.to_raw = nn.Conv1d(c_cur, 1, kernel_size=3, padding=1)  # (B,1,F)

        # Align head: latent -> preprocessed vector (F)
        self.align_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),      # (B,latent_c,1)
            nn.Flatten(),                 # (B,latent_c)
            nn.Linear(self.latent_c, 256), nn.SiLU(),
            nn.Linear(256, F)
        )

    def encode(self, x):           # x: (B,1,F)
        return self.encoder(x)     # (B,latent_c,latent_L)
    def decode(self, h):           # h: (B,latent_c,latent_L)
        y = self.decoder(h)        # upsampled to ~F
        # crop/center-pad to exact F
        if y.shape[-1] != self.F:
            if y.shape[-1] > self.F:
                start = (y.shape[-1] - self.F)//2
                y = y[..., start:start+self.F]
            else:
                pad = self.F - y.shape[-1]
                y = nn.functional.pad(y, (pad//2, pad - pad//2))
        raw = self.to_raw(y)       # (B,1,F)
        return raw
    def align(self, h):            # (B,latent_c,latent_L) -> (B,F)
        return self.align_head(h)
    def forward(self, x):
        h = self.encode(x)
        xr = self.decode(h)
        zhat = self.align(h)
        return h, xr, zhat

# --------- EarlyStopping ---------
class EarlyStop:
    def __init__(self, patience=PATIENCE):
        self.best = None
        self.wait = 0
        self.stop = False
    def step(self, metric):
        if self.best is None or metric < self.best - 1e-8:
            self.best = float(metric)
            self.wait = 0
            return True  # new best
        else:
            self.wait += 1
            if self.wait >= PATIENCE:
                self.stop = True
            return False

# --------- Train / Val loop ---------
def train_val():
    # full dataset (fit scaler on full train split only, not val)
    df = pd.read_csv(RAW_TRAIN)
    cols = spectral_cols(df)
    X = df[cols].to_numpy(np.float32)
    y = (df["classes"].values != 0).astype(np.int64)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
    tr_idx, va_idx = next(sss.split(X, y))
    # scaler on train only
    mean = X[tr_idx].mean(axis=0)
    std  = X[tr_idx].std(axis=0) + 1e-12
    scaler = Scaler(mean.astype(np.float32), std.astype(np.float32))

    ds_full = RawPairs(RAW_TRAIN, scaler=scaler, use_clip=USE_CLIP, clip_val=CLIP_VAL)
    ds_tr = Subset(ds_full, tr_idx.tolist())
    ds_va = Subset(ds_full, va_idx.tolist())

    F = X.shape[1]
    model = ConvAE(F, downs=DOWNS, base=64, latent_c=LATENT_C).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    def step(loader, train=True):
        if train: model.train()
        else:     model.eval()
        losses, recs, aligns = [], [], []
        for xb, zb, _yb in loader:
            xb = xb.to(DEVICE)              # (B,1,F)
            zb = zb.to(DEVICE)              # (B,F)
            if train: opt.zero_grad()
            h, xr, zhat = model(xb)
            # losses
            loss_rec = nn.functional.mse_loss(xr.squeeze(1), xb.squeeze(1))
            loss_align = nn.functional.mse_loss(zhat, zb)
            loss = loss_rec + ALIGN_LMB * loss_align
            if train:
                loss.backward()
                opt.step()
            losses.append(loss.item()); recs.append(loss_rec.item()); aligns.append(loss_align.item())
        return np.mean(losses), np.mean(recs), np.mean(aligns)

    tr_loader = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, drop_last=True)
    va_loader = DataLoader(ds_va, batch_size=BATCH, shuffle=False)

    stopper = EarlyStop(patience=PATIENCE)
    best_state = None

    hist = []
    for ep in range(1, EPOCHS+1):
        tr_loss, tr_rec, tr_aln = step(tr_loader, train=True)
        va_loss, va_rec, va_aln = step(va_loader, train=False)
        hist.append({"epoch":ep,"tr_loss":tr_loss,"va_loss":va_loss,"tr_rec":tr_rec,"va_rec":va_rec,"tr_aln":tr_aln,"va_aln":va_aln})
        print(f"Epoch {ep:03d} | tr: {tr_loss:.5f} (rec {tr_rec:.5f}, aln {tr_aln:.5f}) | va: {va_loss:.5f} (rec {va_rec:.5f}, aln {va_aln:.5f})")

        if stopper.step(va_loss):
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
        if stopper.stop:
            print(f"Early stopping at epoch {ep}.")
            break

    # Save best
    assert best_state is not None, "Training did not run?"
    torch.save(best_state, OUT_DIR / "ae_conv1d.pt")

    # Save meta (for diffusion + sampling later)
    meta = {
        "F": int(F),
        "downs": int(DOWNS),
        "latent_channels": int(model.latent_c),
        "latent_length": int(model.latent_L),
        "cols": cols,
        "scaler_mean": ds_full.scaler.mean.tolist(),
        "scaler_std": ds_full.scaler.std.tolist(),
        "use_clip": USE_CLIP,
        "clip_val": CLIP_VAL,
        "align_lambda": ALIGN_LMB,
        "seed": SEED
    }
    with open(OUT_DIR/"ae_meta.json","w") as f:
        json.dump(meta, f, indent=2)

    # Quick plots
    dfh = pd.DataFrame(hist)
    dfh.to_csv(OUT_DIR/"ae_train_history.csv", index=False)
    plt.figure(figsize=(6,4)); plt.plot(dfh["epoch"], dfh["tr_loss"], label="train"); plt.plot(dfh["epoch"], dfh["va_loss"], label="val"); plt.legend(); plt.title("AE loss"); plt.tight_layout(); plt.savefig(OUT_DIR/"ae_loss.png", dpi=200); plt.close()

    print("Saved: ae_conv1d.pt, ae_meta.json, ae_train_history.csv, ae_loss.png")

if __name__ == "__main__":
    train_val()
