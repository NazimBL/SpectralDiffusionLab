import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader, TensorDataset

#no scaling
# -------------------------------
# 1. Preprocessing (Baseline Correction Only)
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
    X_corrected = np.zeros_like(X)
    for i in range(X.shape[0]):
        baseline = baseline_als(X[i, :], lam=lam, p=p, niter=niter)
        X_corrected[i, :] = X[i, :] - baseline
    return X_corrected

def vector_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms
def load_and_preprocess_data(csv_path="train_set.csv"):
    df = pd.read_csv(csv_path)
    non_feature_cols = ["sample_id", "class", "patient_id"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    df['binary_class'] = df['class'].apply(lambda x: 0 if x == 0 else 1)
    X = df[feature_cols].values
    y = df["binary_class"].values
    # Only apply baseline correction (no additional scaling)
    X_bc = apply_baseline_correction(X, lam=1e5, p=0.01, niter=10)
    X_sg = savgol_filter(X_bc, window_length=5, polyorder=2, deriv=2, axis=1)
    X_norm = vector_normalize(X_sg)

    return X_norm, y

# -------------------------------
# 2. DDPM Schedules and Diffusion Utilities
# -------------------------------
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def compute_alpha(betas):
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bars

def forward_diffusion_ddpm(x0, t, alpha_bars):
    t_idx = t - 1
    sqrt_alpha_bar = torch.sqrt(alpha_bars[t_idx]).unsqueeze(1).unsqueeze(1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bars[t_idx]).unsqueeze(1).unsqueeze(1)
    noise = torch.randn_like(x0)
    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    return x_t, noise

# -------------------------------
# 3. 1D U-Net Architecture
# -------------------------------
def timestep_embedding(timesteps, embed_dim):
    half_dim = embed_dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device) / half_dim)
    args = timesteps.unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if embed_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.act = nn.ReLU()
        self.skip = (in_ch == out_ch)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        time_out = self.time_mlp(t_emb).unsqueeze(-1)
        h = h + time_out
        h = self.act(h)
        h = self.conv2(h)
        if self.skip:
            h = h + x
        return self.act(h)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, time_emb_dim)
        self.pool = nn.Conv1d(out_ch, out_ch, kernel_size=2, stride=2)
    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        down = self.pool(x)
        return x, down

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.trans = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.res = ResidualBlock(in_ch + out_ch, out_ch, time_emb_dim)
    def forward(self, x, skip, t_emb):
        x = self.trans(x)
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            x = nn.functional.pad(x, (0, diff))
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb)

class UNet1D(nn.Module):
    def __init__(self, num_classes, time_emb_dim=128, base_ch=128):
        super().__init__()
        self.cond_emb = nn.Embedding(num_classes, time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.down1 = DownBlock(1, base_ch, time_emb_dim)
        self.down2 = DownBlock(base_ch, base_ch * 2, time_emb_dim)
        self.down3 = DownBlock(base_ch * 2, base_ch * 2, time_emb_dim)
        self.mid = ResidualBlock(base_ch * 2, base_ch * 2, time_emb_dim)
        self.up1 = UpBlock(base_ch * 2, base_ch * 2, time_emb_dim)
        self.up2 = UpBlock(base_ch * 2, base_ch, time_emb_dim)
        self.up3 = UpBlock(base_ch, base_ch, time_emb_dim)
        self.out_conv = nn.Conv1d(base_ch, 1, kernel_size=1)
    def forward(self, x, t, cond):
        te = timestep_embedding(t, self.time_mlp[0].in_features)
        te = self.time_mlp(te)
        if cond is not None:
            cond_emb = self.cond_emb(cond)
        else:
            cond_emb = torch.zeros_like(te)
        t_emb = te + cond_emb

        skip1, x_d1 = self.down1(x, t_emb)
        skip2, x_d2 = self.down2(x_d1, t_emb)
        skip3, x_d3 = self.down3(x_d2, t_emb)
        x_mid = self.mid(x_d3, t_emb)
        x_up1 = self.up1(x_mid, skip3, t_emb)
        x_up2 = self.up2(x_up1, skip2, t_emb)
        x_up3 = self.up3(x_up2, skip1, t_emb)
        return self.out_conv(x_up3)

# -------------------------------
# 4. Reverse Diffusion (Sampling)
# -------------------------------
def p_sample_ddpm(model, x_t, t, betas, alpha_bars, device, cond, guidance_scale=5.0):
    beta_t = betas[t - 1]
    alpha_t = 1 - beta_t
    sqrt_recip_alpha_t = 1.0 / torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bars[t - 1])
    t_tensor = torch.tensor([t], dtype=torch.float32, device=device).expand(x_t.size(0))
    pred_uncond = model(x_t, t_tensor, None)
    pred_cond = model(x_t, t_tensor, cond)
    pred_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    # When reconstructing x0, unsqueeze the denominator so it broadcasts correctly.
    denom = torch.sqrt(alpha_bars[t - 1]).unsqueeze(0).unsqueeze(0)
    x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * pred_noise) / denom
    # Do not clamp x0_pred to preserve amplitude.
    coef = beta_t / sqrt_one_minus_alpha_bar_t
    mean = sqrt_recip_alpha_t * (x_t - coef * pred_noise)
    if t > 1:
        noise = torch.randn_like(x_t)
        sigma_t = torch.sqrt(beta_t)
        x_prev = mean + sigma_t * noise
    else:
        x_prev = mean
    return x_prev

def sample_ddpm(model, cond, num_steps, betas, alpha_bars, device, guidance_scale=5.0, length=200):
    x_t = torch.randn(1, 1, length, device=device)
    for t in reversed(range(1, num_steps + 1)):
        x_t = p_sample_ddpm(model, x_t, t, betas, alpha_bars, device, cond, guidance_scale)
    return x_t.squeeze(0)

# -------------------------------
# 5. Peak-Weighted Loss
# -------------------------------
def create_peak_mask(length=234, start_wn=1797.53, end_wn=898.764,
                     peak_positions=[1446.51, 1377.08, 1234.35, 1045.34, 902.622],
                     peak_weight=6.0, window_size=2):

    wave_numbers = np.linspace(start_wn, end_wn, length)
    mask = np.ones(length, dtype=np.float32)

    for peak in peak_positions:
        idx = np.argmin(np.abs(wave_numbers - peak))  # closest index to the actual peak
        for i in range(idx - window_size, idx + window_size + 1):
            if 0 <= i < length:
                mask[i] = peak_weight

    return mask

def peak_weighted_mse(x_pred, x_true, peak_mask):
    w = torch.from_numpy(peak_mask).to(x_pred.device).view(1, 1, -1)
    diff2 = (x_pred - x_true) ** 2
    return (diff2 * w).mean()

# -------------------------------
# 6. Contrastive Net & Plotting Functions
# -------------------------------
class ContrastiveNet(nn.Module):
    def __init__(self, length, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(16 * 16, embed_dim)
        )
    def forward(self, x):
        return self.net(x)

def cosine_similarity(a, b):
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=0, keepdim=True)
    return (a_norm * b_norm).sum(dim=1)

def select_best_candidate(candidates, contrastive_net, target_embedding):
    embeddings = contrastive_net(candidates)
    sims = cosine_similarity(embeddings, target_embedding)
    best_idx = torch.argmax(sims)
    return candidates[best_idx]

def plot_single_class(real_data, synth_data, class_label, length=234, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    start_wn = 1797.53
    end_wn = 898.764
    wave_numbers = np.linspace(start_wn, end_wn, length)
    plt.figure(figsize=(12, 6))
    plt.plot(wave_numbers, real_data, label="Real Spectrum", linewidth=1.5)
    plt.plot(wave_numbers, synth_data, label="Synthetic Spectrum", linestyle="--", linewidth=1.5, alpha=0.8)
    for wn in [1446, 1377, 1234, 1045, 900]:
        plt.axvline(x=wn, color='gray', linestyle=':', alpha=0.5, label='Peak Region' if wn == 1446 else None)
    plt.gca().invert_xaxis()
    plt.xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    plt.ylabel("Intensity", fontsize=12)
    plt.title(f"Class {class_label}: Real vs Synthetic Spectrum", fontsize=14)
    plt.xticks(np.arange(900, 1801, 100), rotation=45)
    plt.gca().set_xticks(np.arange(900, 1801, 50), minor=True)
    plt.grid(True, which='major', linestyle='--', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/class_{class_label}_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot for class {class_label} to {save_dir}/")

def verify_peak_regions(real, synth, wavenumbers, peak_positions, window_size=2):

    print("\nPeak Region Analysis:")
    for wn in peak_positions:
        idx = np.abs(wavenumbers - wn).argmin()
        start = max(0, idx - window_size)
        end = min(len(wavenumbers) - 1, idx + window_size)
        real_peak = real[start:end]
        synth_peak = synth[start:end]
        wn_range = wavenumbers[start:end]
        real_max = real_peak.max()
        synth_max = synth_peak.max()
        max_diff = abs(real_max - synth_max)
        real_area = np.trapz(real_peak, wn_range)
        synth_area = np.trapz(synth_peak, wn_range)
        area_diff = abs(real_area - synth_area) / (abs(real_area) + 1e-8) * 100
        print(f"Peak @ {wn} cm⁻¹:")
        print(f"  Max Intensity: Real={real_max:.2f}, Synth={synth_max:.2f} (Δ={max_diff:.2f})")
        print(f"  Peak Area: Real={real_area:.2f}, Synth={synth_area:.2f} ({area_diff:.1f}% difference)\n")
        if max_diff > 0.1:
            print(f"WARNING: Large intensity difference at {wn} cm⁻¹")
        if area_diff > 15:
            print(f"WARNING: Significant area variation at {wn} cm⁻¹ (>15%)")

# -------------------------------
# 7. Main Training and Evaluation
# -------------------------------
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    X_data, y = load_and_preprocess_data("train_set.csv")
    # Reshape to (N,1,length)
    X_data = X_data[:, None, :]
    length = X_data.shape[2]
    num_classes = int(np.max(y)) + 1

    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Diffusion hyperparameters
    timesteps = 300  # increased timesteps for finer denoising
    # Uncomment the next line to use cosine schedule:
    # betas = cosine_beta_schedule(timesteps, s=0.008).to(device)
    betas = linear_beta_schedule(timesteps, 1e-4, 0.02).to(device)
    _, alpha_bars = compute_alpha(betas)

    # U-Net model settings
    base_ch = 96
    model = UNet1D(num_classes=num_classes, time_emb_dim=128, base_ch=base_ch).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()

    # Loss weights and peak mask
    lambda_signal = 6
    lambda_peaks = 2
    peak_mask = create_peak_mask(length=length, start_wn=1797.53, end_wn=898.764,
                                 peak_positions=[1446.51, 1377.08, 1234.35, 1045.34, 902.622],
                                 peak_weight=2, window_size=2)

    epochs = 80
    drop_prob = 0.1
    guidance_scale = 1

    # Print hardware and training parameters
    print("\n=== Hardware Configuration ===")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Num CPUs: {os.cpu_count()}")

    print("\n=== Training Parameters ===")
    print(f"Timesteps: {timesteps}")
    print(f"Base channels: {base_ch}")
    print(f"Loss weights - Signal: {lambda_signal}, Peaks: {lambda_peaks}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Peak mask unique weights: {np.unique(peak_mask)}")

    print("Training 1D U-Net Diffusion Model with peak-weighted loss...")
    model.train()
    for epoch in range(epochs):
        losses = []
        noise_losses = []
        signal_losses = []
        peak_losses = []
        grad_norms = []

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)  # (B,1,length)
            batch_y = batch_y.to(device)

            t = torch.randint(1, timesteps + 1, (batch_x.size(0),), device=device)
            x_t, noise = forward_diffusion_ddpm(batch_x, t, alpha_bars)
            cond_input = None if random.random() < drop_prob else batch_y

            pred_noise = model(x_t, t.float(), cond_input)
            loss_noise = mse_loss(pred_noise, noise)
            noise_losses.append(loss_noise.item())

            # Reconstruct x0 with proper unsqueezing on the denominator for broadcasting
            t_idx = t - 1
            sqrt_alpha_bar = torch.sqrt(alpha_bars[t_idx]).unsqueeze(1).unsqueeze(1)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bars[t_idx]).unsqueeze(1).unsqueeze(1)
            x0_pred = (x_t - sqrt_one_minus_alpha_bar * pred_noise) / torch.sqrt(alpha_bars[t_idx]).unsqueeze(1).unsqueeze(1)
            loss_signal = mse_loss(x0_pred, batch_x)
            signal_losses.append(loss_signal.item())

            loss_peaks = peak_weighted_mse(x0_pred, batch_x, peak_mask)
            peak_losses.append(loss_peaks.item())

            loss_total = loss_noise + lambda_signal * loss_signal + lambda_peaks * loss_peaks

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(loss_total.item())

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.detach().data.norm(2).item() ** 2
            grad_norms.append(total_norm ** 0.5)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Total loss: {np.mean(losses):.4f}")
        print(f"Component losses - Noise: {np.mean(noise_losses):.4f}, Signal: {np.mean(signal_losses):.4f}, Peaks: {np.mean(peak_losses):.4f}")
        print(f"Gradient norm - Mean: {np.mean(grad_norms):.4f}, Std: {np.std(grad_norms):.4f}")
        if torch.cuda.is_available():
            print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # Train Contrastive Network (dummy)
    contrastive_net = ContrastiveNet(length=length, embed_dim=64).to(device)
    c_opt = optim.Adam(contrastive_net.parameters(), lr=1e-4)
    print("Training Contrastive Network (dummy) ...")
    contrastive_net.train()
    for epoch in range(10):
        cl_losses = []
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            emb = contrastive_net(batch_x)
            target = emb.mean(dim=0, keepdim=True).expand_as(emb)
            loss = mse_loss(emb, target)
            c_opt.zero_grad()
            loss.backward()
            c_opt.step()
            cl_losses.append(loss.item())
        print(f"Contrastive Epoch {epoch+1}/10 - Loss: {np.mean(cl_losses):.6f}")

    # Evaluation: Generate synthetic candidates and compare with real samples
    model.eval()
    contrastive_net.eval()
    class_results = {}
    classes = torch.unique(y_tensor).cpu().numpy()
    print(f"Detected classes: {classes}")


    for cl in classes:
        print(f"\nProcessing class {cl}...")
        cond = torch.tensor([cl], dtype=torch.long, device=device)
        candidates = []
        for _ in range(10):  # Reduced candidate count for memory
            synth = sample_ddpm(model, cond, timesteps, betas, alpha_bars, device, guidance_scale, length)
            candidates.append(synth.unsqueeze(0))
        candidates = torch.cat(candidates, dim=0)

        mask = (y_tensor == cl)
        if mask.sum() == 0:
            print(f"No real samples found for class {cl}")
            continue
        real_samples = X_tensor[mask].to(device)
        with torch.no_grad():
            target_emb = contrastive_net(real_samples).mean(dim=0)
        best_candidate = select_best_candidate(candidates, contrastive_net, target_emb)
        class_results[cl] = {
            'real': real_samples[0].cpu().numpy().squeeze(),
            'synth': best_candidate[0].cpu().detach().numpy().squeeze()
        }

    for cl in class_results:
        print(f"\nVisualizing class {cl}...")
        data = class_results[cl]
        plot_single_class(data['real'], data['synth'], cl, length=length, save_dir="hpc_results")
        wn_axis = np.linspace(1797.53, 898.764, length)
        verify_peak_regions(data['real'], data['synth'], wn_axis, [1446, 1377, 1234, 1045, 900])

if __name__ == "__main__":
    main()
