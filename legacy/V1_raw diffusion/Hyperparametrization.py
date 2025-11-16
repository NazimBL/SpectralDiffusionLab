import math
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader, TensorDataset
from scipy.linalg import sqrtm
from itertools import product


# -------------------------------
# 1. Preprocessing
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


def apply_baseline_correction(X):
    X_corrected = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_corrected[i, :] = X[i, :] - baseline_als(X[i, :])
    return X_corrected


def vector_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


def load_and_preprocess_data():
    df = pd.read_csv("train_set.csv")
    non_feature_cols = ["sample_id", "class", "patient_id"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    df['binary_class'] = df['class'].apply(lambda x: 0 if x == 0 else 1)
    X = df[feature_cols].values
    y = df["binary_class"].values
    X = apply_baseline_correction(X)
    X = savgol_filter(X, window_length=5, polyorder=2, deriv=2, axis=1)
    X = vector_normalize(X)
    return X[:, None, :], y  # Add channel dimension


# -------------------------------
# 2. Diffusion Components
# -------------------------------
def cosine_beta_schedule(timesteps):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos((x / timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
    alphas_cumprod /= alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def forward_diffusion(x0, t, alpha_bars):
    t_idx = t - 1
    sqrt_alpha_bar = torch.sqrt(alpha_bars[t_idx]).view(-1, 1, 1)
    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bars[t_idx]).view(-1, 1, 1)
    noise = torch.randn_like(x0)
    return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha * noise, noise


# -------------------------------
# 3. U-Net Architecture
# -------------------------------
class UNet1D(nn.Module):
    def __init__(self, num_classes, base_ch=64):
        super().__init__()
        time_emb_dim = 128
        self.cond_emb = nn.Embedding(num_classes, time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.down1 = self._make_down(1, base_ch, time_emb_dim)
        self.down2 = self._make_down(base_ch, base_ch * 2, time_emb_dim)
        self.down3 = self._make_down(base_ch * 2, base_ch * 2, time_emb_dim)
        self.mid = self._make_residual(base_ch * 2, time_emb_dim)
        self.up1 = self._make_up(base_ch * 2, base_ch * 2, time_emb_dim)
        self.up2 = self._make_up(base_ch * 2, base_ch, time_emb_dim)
        self.up3 = self._make_up(base_ch, 1, time_emb_dim)
        self.out = nn.Conv1d(1, 1, kernel_size=1)

    def _make_down(self, in_ch, out_ch, time_emb_dim):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.MaxPool1d(2)
        )

    def _make_residual(self, channels, time_emb_dim):
        return nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1)
        )

    def _make_up(self, in_ch, out_ch, time_emb_dim):
        return nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, 2, stride=2),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1)
        )

    def forward(self, x, t, cond):
        t_emb = self.time_mlp(self._timestep_embedding(t))
        if cond is not None:
            t_emb += self.cond_emb(cond)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x_mid = self.mid(x3) + x3
        x_up1 = self.up1(x_mid) + x2
        x_up2 = self.up2(x_up1) + x1
        x_up3 = self.up3(x_up2)
        return self.out(x_up3)

    def _timestep_embedding(self, t):
        embed_dim = self.time_mlp[0].in_features
        half_dim = embed_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


def calculate_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
# -------------------------------
# 4. Training with Batch-Hard Triplet Mining
# -------------------------------
def train_and_evaluate(params, dataloader, device):
    model = UNet1D(num_classes=2, base_ch=params['base_ch']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    betas = cosine_beta_schedule(params['timesteps']).to(device)
    alpha_bars = torch.cumprod(1 - betas, dim=0)
    triplet = nn.TripletMarginLoss(margin=1.0)
    mse = nn.MSELoss()

    for epoch in range(params.get('epochs', 20)):
        model.train()
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            t = torch.randint(1, params['timesteps'] + 1, (batch_x.size(0),), device=device)
            x_t, noise = forward_diffusion(batch_x, t, alpha_bars)

            # Classifier-free guidance dropout
            cond = batch_y if random.random() > 0.1 else None
            pred_noise = model(x_t, t, cond)

            # Noise prediction loss
            loss_noise = mse(pred_noise, noise)

            # Signal reconstruction loss
            x0_pred = (x_t - torch.sqrt(1 - alpha_bars[t - 1]).view(-1, 1, 1) * pred_noise) / torch.sqrt(
                alpha_bars[t - 1]).view(-1, 1, 1)
            loss_signal = mse(x0_pred, batch_x)

            # Batch-hard triplet loss
            anchors = x0_pred.squeeze(1)
            positives = batch_x.squeeze(1)
            labels = batch_y

            # Compute distance matrix and masks
            dist_mat = torch.cdist(anchors, positives)
            pos_mask = labels.view(-1, 1) == labels.view(1, -1)
            neg_mask = ~pos_mask

            # Find hardest negatives
            neg_dists = dist_mat.masked_fill(neg_mask, float('inf'))
            hardest_neg_idx = torch.argmin(neg_dists, dim=1)
            hardest_neg = torch.where(neg_mask.any(dim=1),
                                      positives[hardest_neg_idx],
                                      torch.zeros_like(positives[0]))

            # Compute triplet loss
            loss_triplet = triplet(anchors, positives, hardest_neg)

            # Total loss
            loss = loss_noise + params['lambda_signal'] * loss_signal + loss_triplet

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Calculate FID
    real, fake = [], []
    model.eval()
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            t = torch.full((batch_x.size(0),), params['timesteps'], device=device)
            x_t, _ = forward_diffusion(batch_x, t, alpha_bars)
            pred = model(x_t, t, None)
            real.append(batch_x.cpu().numpy())
            fake.append(pred.cpu().numpy())

    real = np.concatenate(real)
    fake = np.concatenate(fake)
    fid = calculate_fid(real, fake)
    return fid, model


# -------------------------------
# 5. Hyperparameter Search
# -------------------------------
def grid_search():
    param_grid = {
        'timesteps': [100, 200],
        'base_ch': [64, 128],
        'lr': [1e-4, 5e-5],
        'lambda_signal': [2, 4],
        'guidance_scale': [1.0, 2.0]
    }

    X, y = load_and_preprocess_data()
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_fid = float('inf')
    best_params = None
    best_model = None

    for params in product(*param_grid.values()):
        current_params = dict(zip(param_grid.keys(), params))
        fid, model = train_and_evaluate(current_params, dataloader, device)

        if fid < best_fid:
            best_fid = fid
            best_params = current_params
            best_model = model

    # Save best model
    os.makedirs("saved_models", exist_ok=True)
    torch.save(best_model.state_dict(),
               f"saved_models/best_model_{best_params['timesteps']}_{best_params['base_ch']}.pt")
    print(f"Best FID: {best_fid:.2f} with params: {best_params}")


if __name__ == "__main__":
    grid_search()