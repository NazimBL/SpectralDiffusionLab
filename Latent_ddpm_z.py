#!/usr/bin/env python3
# Latent_DDPM_Z.py

import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.signal import savgol_filter  # <--- ADDED

# ====== EDIT THESE ======
TRAIN_CSV = Path(r"../MyDataset/ftir_train_wn.csv")
OUT_DIR = Path(r"ldm_out")
BATCH = 256
VAL_RATIO = 0.15
SEED = 42
# ========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
META_COLS = {"groupnumbers", "classes", "class_name", "binary_label", "groupcodes", "obsnames"}


def spectral_cols(df):
    cols = []
    for c in df.columns:
        if c in META_COLS: continue
        try:
            float(c);
            cols.append(c)
        except:
            pass
    if not cols: raise ValueError("No wavenumber columns detected.")
    return [c for c in df.columns if c in set(cols)]


# --- ADDED: Preprocessing function (must match train_ae.py) ---
def preprocess_row(x_row: np.ndarray) -> np.ndarray:
    # SG 2nd derivative + L2 normalization
    win = 5 if x_row.size >= 5 else (x_row.size // 2 * 2 + 1)
    # Ensure window is odd
    if win % 2 == 0: win += 1

    z = savgol_filter(x_row, window_length=win, polyorder=2, deriv=2)
    n = np.linalg.norm(z) + 1e-12
    return (z / n).astype(np.float32)


# --- UPDATED: ConvAE Class (Must match train_ae.py) ---
class ConvAE(nn.Module):
    def __init__(self, F: int, downs: int = 4, base: int = 64, latent_c: int = 12):
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

    def encode(self, x):
        return self.encoder(x)  # (B, C, L)


def load_train_df(path: Path):
    df = pd.read_csv(path)
    cols = spectral_cols(df)
    X = df[cols].to_numpy(np.float32)
    y = (df["classes"].values != 0).astype(np.int64)
    wns = np.array([float(c) for c in cols], dtype=np.float32)
    if not np.all(np.diff(wns) < 0):
        wns = wns[::-1]
        X = X[:, ::-1].copy()
    return X, y, wns, cols


def main():
    torch.manual_seed(SEED);
    np.random.seed(SEED);
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load meta
    meta = json.loads((OUT_DIR / "ae_meta.json").read_text())
    F = int(meta["F"]);
    downs = int(meta["downs"]);
    latent_c = int(meta["latent_channels"])
    # We still load these raw scalers to save them in the new meta
    mean = np.array(meta["scaler_mean"], dtype=np.float32)
    std = np.array(meta["scaler_std"], dtype=np.float32)

    print(f"[META] F={F} downs={downs} latent_c={latent_c}")

    # Initialize AE
    ae = ConvAE(F, downs=downs, base=64, latent_c=latent_c).to(DEVICE)

    # Load Weights (strict=False is OK because decoder is missing)
    sd = torch.load(OUT_DIR / "ae_conv1d.pt", map_location=DEVICE)
    ae.load_state_dict(sd, strict=False)
    ae.eval()

    # --- START OF FIX ---
    # 1. load RAW train data
    X_raw, y, wns, cols = load_train_df(TRAIN_CSV)

    # 2. Apply the SAME preprocessing as the AE was trained on
    print("Preprocessing raw data to 2nd-derivative...")
    X_clean = np.vstack([preprocess_row(r) for r in X_raw]).astype(np.float32)
    print(f"Clean data shape: {X_clean.shape}")

    # 3. Split the CLEAN data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
    tr_idx, va_idx = next(sss.split(X_clean, y))  # Split X_clean

    # --- END OF FIX ---

    class TrainSet(Dataset):
        def __init__(self, X, y): self.X = X; self.y = y  # X is now X_clean

        def __len__(self): return len(self.y)

        def __getitem__(self, i):
            # Add channel dim for Conv1D
            return torch.from_numpy(self.X[i])[None, :], torch.tensor(self.y[i])

    def encode_split(Xs, ys, name):
        ds = TrainSet(Xs, ys)  # Xs is now X_clean[idx]
        dl = DataLoader(ds, batch_size=BATCH, shuffle=False)
        latents = [];
        labels = []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(DEVICE)  # xb is a clean 2nd-deriv spectrum
                # Feed the encoder the data it was trained on
                h = ae.encode(xb)
                latents.append(h.cpu())
                labels.append(yb)
        z = torch.cat(latents, dim=0)
        y_t = torch.cat(labels, dim=0)
        print(f"[{name}] z shape: {tuple(z.shape)}")
        return z, y_t

    # We encode the CLEAN splits
    z_tr, y_tr = encode_split(X_clean[tr_idx], y[tr_idx], "train")
    z_va, y_va = encode_split(X_clean[va_idx], y[va_idx], "val")

    # Stats
    z_mu = z_tr.mean(dim=(0,))  # shape (C, Lz)
    z_std = z_tr.std(dim=(0,)).clamp_min(1e-6)

    # Stats
    print("Latent mean (first 5):", z_mu.flatten()[:5])
    print("Latent std (first 5):", z_std.flatten()[:5])

    # save
    torch.save({
        "z": z_tr, "y": y_tr, "wns": torch.from_numpy(wns),
        "meta": {"F": F, "downs": downs, "latent_c": latent_c, "latent_L": z_tr.shape[-1],
                 "scaler_mean": mean, "scaler_std": std, "cols": cols,
                 "z_mu": z_mu.cpu().numpy(), "z_std": z_std.cpu().numpy()} # <--- ADD THIS
    }, OUT_DIR / "latent_train.pt")

    torch.save({
        "z": z_va, "y": y_va, "wns": torch.from_numpy(wns),
        "meta": {"F": F, "downs": downs, "latent_c": latent_c, "latent_L": z_tr.shape[-1],
                 "scaler_mean": mean, "scaler_std": std, "cols": cols,
                 "z_mu": z_mu.cpu().numpy(), "z_std": z_std.cpu().numpy()}
    }, OUT_DIR / "latent_val.pt")

    print("Saved latent_train.pt and latent_val.pt")


if __name__ == "__main__":
    main()