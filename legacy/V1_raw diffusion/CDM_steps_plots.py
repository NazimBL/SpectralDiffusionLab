import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --- Baseline Correction ---
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

# --- Vector Normalization ---
def vector_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms

# --- Load & Preprocess ---
def load_and_preprocess_spectrum(csv_path="train_set.csv", sample_index=0):
    df = pd.read_csv(csv_path)
    df = df[df['class'] != 4].copy()
    df['binary_class'] = (df['class'] != 0).astype(int)
    non_feat = {"sample_id","class","patient_id","binary_class"}
    feat_cols = [c for c in df.columns if c not in non_feat]
    wns = np.array([float(c) for c in feat_cols])
    mask = (wns >= 900) & (wns <= 1800)
    selected = np.array(feat_cols)[mask]
    X = df[selected].values
    y = df['binary_class'].values
    # Preprocess one spectrum
    x = X[sample_index]
    baseline = baseline_als(x)
    x_corr = x - baseline
    x_sg = savgol_filter(x_corr, window_length=5, polyorder=2, deriv=2)
    x_norm = x_sg / np.linalg.norm(x_sg)
    return wns[mask], x_norm

# --- Forward Diffusion (simulate z_t) ---
def simulate_forward_diffusion(x0, t, total_timesteps=300):
    s = 0.008
    steps = total_timesteps + 1
    x = np.linspace(0, total_timesteps, steps)
    alphas_cumprod = np.cos(((x/total_timesteps)+s)/(1+s) * np.pi/2)**2
    alphas_cumprod /= alphas_cumprod[0]
    sqrt_ab = np.sqrt(alphas_cumprod[t])
    sqrt_omb = np.sqrt(1 - alphas_cumprod[t])
    noise = np.random.randn(*x0.shape)
    xt = sqrt_ab * x0 + sqrt_omb * noise
    return xt

# --- Plotting Utility ---
def plot_spectrum(wns, spectrum, label, filename, color='blue'):
    plt.figure(figsize=(8,4))
    plt.plot(wns, spectrum, label=label, color=color)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(label)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# --- Main Execution ---
wns, x0 = load_and_preprocess_spectrum("train_set.csv", sample_index=0)
z1 = simulate_forward_diffusion(x0, t=5)
z2 = simulate_forward_diffusion(x0, t=1)
zT = simulate_forward_diffusion(x0, t=300)
plot_spectrum(wns, zT, "Fully Noised ($z_T$, t=300)", "zT.png", color='red')

# Save each plot separately
plot_spectrum(wns, x0, "Preprocessed Spectrum ($x_0$)", "x0.png", color='black')
plot_spectrum(wns, z1, "Forward Diffused ($z_1$, t=5)", "z1.png", color='orange')
plot_spectrum(wns, z2, "Forward Diffused ($z_2$, t=1)", "z2.png", color='green')
