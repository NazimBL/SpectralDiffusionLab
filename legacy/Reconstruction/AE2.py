# FTIR Autoencoder: predict baseline‐corrected (no derivative) spectra
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# === 1) Preprocessing Functions ===
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
    X_bc = np.zeros_like(X)
    for i in range(X.shape[0]):
        bl = baseline_als(X[i,:], lam=lam, p=p, niter=niter)
        X_bc[i,:] = X[i,:] - bl
    return X_bc

def vector_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms==0] = 1
    return X / norms

# === 2) Load & Preprocess ===
def preprocess_train_data(csv_path):
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in ("sample_id","class","patient_id")]
    X_raw = df[feat_cols].values
    # (a) baseline‐correct
    X_bc  = apply_baseline_correction(X_raw)
    # (b) derivative + norm
    X_sg  = savgol_filter(X_bc, window_length=5, polyorder=2, deriv=2, axis=1)
    X_norm= vector_normalize(X_sg)
    return X_norm, X_bc

# === 3) Autoencoder ===
class FTIRAutoencoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim,128), nn.ReLU(),
            nn.Linear(128,64),  nn.ReLU(),
            nn.Linear(64,32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32,64),  nn.ReLU(),
            nn.Linear(64,128), nn.ReLU(),
            nn.Linear(128,dim)
        )
    def forward(self,x):
        return self.decoder(self.encoder(x))

# === 4) Training ===
def train_autoencoder(X_in, X_out, epochs=100, lr=1e-3, bs=32):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TensorDataset(torch.tensor(X_in,dtype=torch.float32),
                       torch.tensor(X_out,dtype=torch.float32))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    model = FTIRAutoencoder(X_in.shape[1]).to(dev)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for e in range(epochs):
        tot=0
        for xb,yb in loader:
            xb,yb=xb.to(dev),yb.to(dev)
            pred = model(xb)
            l = lossf(pred,yb)
            opt.zero_grad(); l.backward(); opt.step()
            tot += l.item()
        if e%10==0:
            print(f"Epoch {e:03d}   loss={tot/len(loader):.6f}")
    return model

# === 5) Decode synthetic ===
def decode_synthetic(model, synth_dir):
    model.eval()
    dev = next(model.parameters()).device
    out=[]
    with torch.no_grad():
        for fn in os.listdir(synth_dir):
            if fn.endswith(".csv"):
                Xn = pd.read_csv(os.path.join(synth_dir,fn)).values
                xn = torch.tensor(Xn,dtype=torch.float32).to(dev)
                xb = model(xn).cpu().numpy()
                out.append(xb)
    return np.vstack(out)

# === 6) Run pipeline ===
def main():
    # load
    X_norm, X_bc = preprocess_train_data("train_set.csv")

    # train AE to map X_norm → X_bc
    ae = train_autoencoder(X_norm, X_bc, epochs=100, lr=1e-3, bs=32)

    # decode your synthetic preprocessed spectra
    dec_bc = decode_synthetic(ae, "Comparative Analysis/generated_spectra")

    # compare mean of baseline‐corrected domain
    mean_dec = dec_bc.mean(axis=0)
    mean_true= X_bc.mean(axis=0)

    plt.figure(figsize=(10,6))
    plt.plot(mean_dec, label="Mean Decoded Synthetic (baseline-corrected)", lw=2)
    plt.plot(mean_true,label="Mean Real baseline-corrected", ls="--", alpha=0.8)
    plt.xlabel("Wavenumber index"); plt.ylabel("Absorbance")
    plt.title("AE: baseline‐corrected decoded vs real")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("ae_bc_vs_true_bc.png", dpi=300)
    plt.show()

if __name__=="__main__":
    main()
