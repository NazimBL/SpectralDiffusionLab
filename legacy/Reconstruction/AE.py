# FTIR Autoencoder Pipeline with Peak-Weighted Loss (Option 2)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# === Preprocessing Functions ===
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
    X_corrected = np.zeros_like(X)
    for i in range(X.shape[0]):
        baseline = baseline_als(X[i, :], lam=lam, p=p, niter=niter)
        X_corrected[i, :] = X[i, :] - baseline
    return X_corrected

def vector_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms

def preprocess_train_data(csv_path):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ["sample_id", "class", "patient_id"]]
    X_raw = df[feature_cols].values             # raw absorbance
    X_bc  = apply_baseline_correction(X_raw)    # baseline correction
    X_sg  = savgol_filter(X_bc, window_length=5, polyorder=2, deriv=2, axis=1)
    X_norm= vector_normalize(X_sg)              # 2nd‐derivative + norm
    return X_norm, X_raw

# === Peak-Weighted Loss Helpers ===
def create_peak_mask(length=234,
                     start_wn=1797.53,
                     end_wn=898.764,
                     peak_positions=[1446.51,1377.08,1234.35,1045.34,902.622],
                     peak_weight=10.0,
                     window_size=2):
    wns = np.linspace(start_wn, end_wn, length)
    mask = np.ones(length, dtype=np.float32)
    for p in peak_positions:
        idx = np.abs(wns - p).argmin()
        for i in range(idx - window_size, idx + window_size + 1):
            if 0 <= i < length:
                mask[i] = peak_weight
    return mask

def peak_weighted_mse(xp, xt, mask):
    # xp, xt: (B, features)
    w = torch.from_numpy(mask).to(xp.device).view(1, -1)
    diff2 = (xp - xt)**2
    return (diff2 * w).mean()

# === Autoencoder Definition ===
class FTIRAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64,  32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32,  64),  nn.ReLU(),
            nn.Linear(64, 128),  nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# === Train Autoencoder (with optional peak weight) ===
def train_autoencoder(X_train, X_target,
                      epochs=100, lr=1e-3, batch_size=32,
                      lambda_peak=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor  = torch.tensor(X_train, dtype=torch.float32)
    X_target_tensor = torch.tensor(X_target, dtype=torch.float32)
    dataset = TensorDataset(X_train_tensor, X_target_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FTIRAutoencoder(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss  = nn.MSELoss()

    # build peak mask once
    peak_mask = create_peak_mask(length=X_train.shape[1], peak_weight=10.0)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)

            loss_mse = mse_loss(pred, yb)
            loss_peak = peak_weighted_mse(pred, yb, peak_mask) if lambda_peak > 0 else 0.0
            loss_l1 = pred.abs().mean() * 1e-4

            loss = loss_mse + lambda_peak * loss_peak + loss_l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch==epochs-1:
            avg = total_loss/len(loader)
            print(f"Epoch {epoch:03d} | Loss: {avg:.6f}")

    return model

# === Decode Synthetic Preprocessed CSVs ===
def decode_synthetic(model, synthetic_dir):
    decoded = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for fname in os.listdir(synthetic_dir):
            if fname.endswith(".csv"):
                arr = pd.read_csv(os.path.join(synthetic_dir, fname)).values
                x   = torch.tensor(arr, dtype=torch.float32).to(device)
                x_hat = model(x).cpu().numpy()
                decoded.append(x_hat)
    return np.vstack(decoded)

# === Run Full Pipeline ===
def main():
    # 1) Preprocess
    X_norm, X_raw = preprocess_train_data("../train_set.csv")

    # 2) Fit a y-scaler on the raw spectra
    y_scaler = StandardScaler()
    Y_scaled = y_scaler.fit_transform(X_raw)

    # 3) Train AE to predict scaled-raw with peak-weighted loss

    model = train_autoencoder(
        X_train=X_norm,
        X_target=Y_scaled,
        epochs=100,
        lr=1e-3,
        batch_size=32,
        lambda_peak=3.0
    )

    # 4) Decode your synthetic preprocessed CSVs
    decoded_synthetic = decode_synthetic(model, "../Comparative Analysis/generated_spectra")

    # 5) Invert scaling back to raw absorbance
    decoded_raw = y_scaler.inverse_transform(decoded_synthetic)

    # 6) Plot mean vs real
    mean_synth = decoded_raw.mean(axis=0)
    mean_real  = X_raw.mean(axis=0)

    plt.figure(figsize=(10,6))
    plt.plot(mean_synth, label="Mean Decoded Synthetic (Raw)", linewidth=2)
    plt.plot(mean_real,  label="Mean Real Raw", linestyle="--", alpha=0.8)
    plt.xlabel("Wavenumber Index")
    plt.ylabel("Absorbance")
    plt.title("Autoencoder Reconstructed vs Real Raw Spectrum (with peak loss)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ae_synthetic_vs_real_peakweighted.png", dpi=300)
    plt.show()

if __name__=="__main__":
    main()
