#!/usr/bin/env python3
# cache_latents.py

import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

# ====== EDIT THESE ======
#TRAIN_CSV = Path(r"../Baseline Results/raw_train_ks.csv")
TRAIN_CSV = Path(r"../MyDataset/ftir_train_wn.csv")
OUT_DIR   = Path(r"ldm_out")              # where ae_conv1d.pt + ae_meta.json are
BATCH     = 256
VAL_RATIO = 0.15
SEED      = 42
# ========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
META_COLS = {"groupnumbers","classes","class_name","binary_label","groupcodes","obsnames"}

def spectral_cols(df):
    cols=[]
    for c in df.columns:
        if c in META_COLS: continue
        try: float(c); cols.append(c)
        except: pass
    if not cols: raise ValueError("No wavenumber columns detected.")
    return [c for c in df.columns if c in set(cols)]

class ConvAE(nn.Module):
    def __init__(self, F: int, downs: int = 4, base: int = 64, latent_c: int = 64):
        super().__init__()
        self.F = F
        c = base; in_c = 1; enc=[]
        for i in range(downs):
            out_c = latent_c if i == downs-1 else c
            enc += [
                nn.Conv1d(in_c, c, kernel_size=5, stride=1, padding=2),
                nn.SiLU(),
                nn.Conv1d(c, out_c, kernel_size=5, stride=2, padding=2),
                nn.SiLU(),
            ]
            in_c = out_c; c = min(c*2, 256)
        self.encoder = nn.Sequential(*enc)

    def encode(self, x):  # (B,1,F) -> (B,C,Lz)
        return self.encoder(x)

def load_train_df(path: Path):
    df = pd.read_csv(path)
    cols = spectral_cols(df)
    X = df[cols].to_numpy(np.float32)
    y = (df["classes"].values != 0).astype(np.int64)
    wns = np.array([float(c) for c in cols], dtype=np.float32)
    # ensure descending
    if not np.all(np.diff(wns) < 0):
        wns = wns[::-1]
        X = X[:, ::-1].copy()
    return X, y, wns, cols

def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load meta + AE encoder
    meta = json.loads((OUT_DIR/"ae_meta.json").read_text())
    F = int(meta["F"]); downs = int(meta["downs"]); latent_c = int(meta["latent_channels"])
    mean = np.array(meta["scaler_mean"], dtype=np.float32)
    std  = np.array(meta["scaler_std"],  dtype=np.float32)

    print(f"[META] F={F} downs={downs} latent_c={latent_c} mean/std shapes={mean.shape}/{std.shape}")

    ae = ConvAE(F, downs=downs, base=64, latent_c=latent_c).to(DEVICE)
    sd = torch.load(OUT_DIR/"ae_conv1d.pt", map_location=DEVICE)
    ae.load_state_dict(sd, strict=False)
    ae.eval()

    # load train data (raw), z-score using TRAIN stats saved in meta
    X_raw, y, wns, cols = load_train_df(TRAIN_CSV)
    assert X_raw.shape[1] == F, f"Input length {X_raw.shape[1]} != meta F {F}"

    X = ((X_raw - mean) / (std + 1e-12)).astype(np.float32)

    # stratified split (by classes)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
    tr_idx, va_idx = next(sss.split(X, y))

    # simple Dataset-on-the-fly
    class TrainSet(Dataset):
        def __init__(self, X, y):
            self.X = X; self.y = y
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            return torch.from_numpy(self.X[i])[None,:], torch.tensor(self.y[i])

    def encode_split(Xs, ys, name):
        ds = TrainSet(Xs, ys)
        dl = DataLoader(ds, batch_size=BATCH, shuffle=False, pin_memory=(DEVICE=="cuda"))
        latents=[]; labels=[]
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(DEVICE)  # (B,1,F)
                h = ae.encode(xb)   # (B,C,Lz)
                latents.append(h.cpu())
                labels.append(yb)
        z = torch.cat(latents, dim=0)   # (N,C,Lz)
        y_t = torch.cat(labels, dim=0)  # (N,)
        print(f"[{name}] z shape: {tuple(z.shape)}, y counts: {(y_t==0).sum().item()} healthy / {(y_t==1).sum().item()} cancer")
        return z, y_t

    z_tr, y_tr = encode_split(X[tr_idx], y[tr_idx], "train")
    z_va, y_va = encode_split(X[va_idx], y[va_idx], "val")

    # NEW: latent normalization stats (per-channel or per-channel-per-pos)
    z_mu = z_tr.mean(dim=(0,))  # shape (C, Lz)
    z_std = z_tr.std(dim=(0,)).clamp_min(1e-6)

    print("mu and std: ")
    print(z_mu)
    print("\n")
    print(z_std)

    # save artifacts for diffusion training
    # save artifacts for diffusion training
    torch.save({
        "z": z_tr, "y": y_tr, "wns": torch.from_numpy(wns),
        "meta": {"F": F, "downs": downs, "latent_c": latent_c, "latent_L": z_tr.shape[-1],
                 "scaler_mean": mean, "scaler_std": std, "cols": cols,
                 "z_mu": z_mu.cpu().numpy(), "z_std": z_std.cpu().numpy()}  # NEW
    }, OUT_DIR / "latent_train.pt")

    torch.save({
        "z": z_va, "y": y_va, "wns": torch.from_numpy(wns),
        "meta": {"F": F, "downs": downs, "latent_c": latent_c, "latent_L": z_tr.shape[-1],
                 "scaler_mean": mean, "scaler_std": std, "cols": cols,
                 "z_mu": z_mu.cpu().numpy(), "z_std": z_std.cpu().numpy()}  # NEW
    }, OUT_DIR / "latent_val.pt")


print("Saved:", OUT_DIR/"latent_train.pt", "and", OUT_DIR/"latent_val.pt")

if __name__ == "__main__":
    main()
