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


RAW_TRAIN = Path(r"MyDataset/ftir_train_wn.csv")
OUT_DIR = Path(r"ldm_out")
BATCH = 64
LR = 5e-4
EPOCHS = 300
PATIENCE = 40
LATENT_C = 12 # channels in latent
DOWNS = 4  # # of stride-2 downsamples (~F/16)
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
            float(c); cols.append(c)
        except:
            pass
    if not cols: raise ValueError("No wavenumber columns detected.")
    return [c for c in df.columns if c in set(cols)]


def preprocess_row(x_row: np.ndarray) -> np.ndarray:
    # SG 2nd derivative + L2 normalization
    win = 5 if x_row.size >= 5 else (x_row.size // 2 * 2 + 1)
    # Ensure window is odd
    if win % 2 == 0: win += 1

    z = savgol_filter(x_row, window_length=win, polyorder=2, deriv=2)
    n = np.linalg.norm(z) + 1e-12
    return (z / n).astype(np.float32)


@dataclass
class Scaler:
    # This scaler is for the ORIGINAL RAW data,
    # used only to create the preprocessed data.
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X): return (X - self.mean) / (self.std + 1e-12)

    def inverse_transform(self, Xn): return Xn * self.std + self.mean


class CleanSpectraDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.cols = spectral_cols(df)
        X_raw = df[self.cols].to_numpy(dtype=np.float32)
        self.y = (df["classes"].values != 0).astype(np.int64)

        # We only use the raw scaler to save it for later.
        # The AE itself will not use it.
        self.scaler_mean = X_raw.mean(axis=0).astype(np.float32)
        self.scaler_std = (X_raw.std(axis=0) + 1e-12).astype(np.float32)

        # Preprocess the  dataset into the clean 2nd-deriv form.
        # This is what the AE will be trained on.
        print("Preprocessing raw data to 2nd-derivative...")
        X_clean = np.vstack([preprocess_row(r) for r in X_raw]).astype(np.float32)

        # We store the clean data
        self.X = X_clean
        print(f"Dataset created. Clean data shape: {self.X.shape}")

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, i):
        # Return the clean spectrum (input) and its class label
        # Add channel dim for Conv1D
        x_clean = torch.from_numpy(self.X[i])[None, :]
        y = torch.tensor(self.y[i])
        return x_clean, y


# -------- Standard Conv1D Autoencoder --------
class ConvAE(nn.Module):
    def __init__(self, F: int, downs: int = 4, base: int = 64, latent_c: int = 64):
        super().__init__()
        self.F = F
        self.latent_c = latent_c

        # --- Encoder ---
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

        # --- Decoder ---
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
        # x is a clean 2nd-deriv spectrum
        return self.encoder(x)  # (B, C, L)

    def decode(self, z):
        # z is a latent vector
        y = self.decoder(z)
        if y.shape[-1] != self.F:
            if y.shape[-1] > self.F:
                start = (y.shape[-1] - self.F) // 2
                y = y[..., start:start + self.F]
            else:
                pad = self.F - y.shape[-1]
                y = Fnn.pad(y, (pad // 2, pad - pad // 2))
        return self.to_raw(y)  # (B, 1, F)

    def forward(self, x):
        z = self.encode(x)
        xr = self.decode(z)
        return xr


# --------- EarlyStopping ---------
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


# --------- Train / Val loop ---------
def train_val():
    # Load and preprocess the data
    ds_full = CleanSpectraDataset(RAW_TRAIN)
    X = ds_full.X
    y = ds_full.y
    F = X.shape[1]

    # Create train/val splits
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
    tr_idx, va_idx = next(sss.split(X, y))

    ds_tr = Subset(ds_full, tr_idx.tolist())
    ds_va = Subset(ds_full, va_idx.tolist())

    model = ConvAE(F, downs=DOWNS, base=64, latent_c=LATENT_C).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    def step(loader, train=True):
        if train:
            model.train()
        else:
            model.eval()
        total_loss = 0.0

        for xb, _yb in loader:
            xb = xb.to(DEVICE)  # This is X_clean
            if train: opt.zero_grad()

            # --- Simple AE Forward Pass ---
            xr = model(xb)  # This is X_recon

            # --- Simple Reconstruction Loss ---
            loss = Fnn.mse_loss(xr, xb)

            if train:
                loss.backward()
                opt.step()

            total_loss += loss.item() * xb.shape[0]

        return total_loss / len(loader.dataset)

    tr_loader = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, drop_last=True)
    va_loader = DataLoader(ds_va, batch_size=BATCH, shuffle=False)

    stopper = EarlyStop(patience=PATIENCE)
    best_state = None

    hist = []
    print(f"Starting AE Training on 2nd-Derivative Data...")

    for ep in range(1, EPOCHS + 1):
        tr_loss = step(tr_loader, train=True)
        va_loss = step(va_loader, train=False)

        hist.append({"epoch": ep, "tr_loss": tr_loss, "va_loss": va_loss})

        print(f"Ep {ep:03d} | Train Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f}")

        if stopper.step(va_loss):
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        if stopper.stop:
            print(f"Early stopping at epoch {ep}.")
            break

    assert best_state is not None
    torch.save(best_state, OUT_DIR / "ae_conv1d.pt")

    meta = {
        "model_type": "AE_2ndDeriv",
        "F": int(F),
        "downs": int(DOWNS),
        "latent_channels": int(model.latent_c),
        "latent_length": int(model.latent_L),
        "cols": ds_full.cols,
        # We save the RAW scalers just in case
        "scaler_mean": ds_full.scaler_mean.tolist(),
        "scaler_std": ds_full.scaler_std.tolist(),
        "seed": SEED
    }
    with open(OUT_DIR / "ae_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Plot
    dfh = pd.DataFrame(hist)
    dfh.to_csv(OUT_DIR / "ae_train_history.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(dfh["epoch"], dfh["tr_loss"], label="Train Loss")
    plt.plot(dfh["epoch"], dfh["va_loss"], label="Val Loss")
    plt.title("AE Reconstruction Loss (2nd-Derivative)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, linestyle='--')

    plt.tight_layout();
    plt.savefig(OUT_DIR / "ae_loss.png", dpi=200);
    plt.close()
    print("Saved AE artifacts.")


if __name__ == "__main__":
    train_val()