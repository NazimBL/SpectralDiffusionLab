#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trains a Conditional Wasserstein GAN with Gradient Penalty (cWGAN-GP)
on 1D spectral data.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn # <--- ADDED IMPORT
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from scipy.signal import savgol_filter
from pathlib import Path
import random
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
# Paths
RAW_TRAIN = Path(r"MyDataset/ftir_train_wn.csv")
OUT_DIR = Path(r"gan_out")

# Data
NUM_CLASSES = 2

# Model Hyperparameters
LATENT_DIM = 100   # Dimension of random noise vector z
MODEL_DIM = 64     # Base dimension for convolutional filters

# Training Hyperparameters
EPOCHS = 1000
BATCH_SIZE = 64
LR = 1e-4          # Learning rate for Adam optimization
B1 = 0.5           # Adam: decay of first order momentum of gradient
B2 = 0.9           # Adam: decay of second order momentum of gradient
N_CRITIC = 5       # Number of critic training steps per generator step
LAMBDA_GP = 10     # Gradient penalty lambda hyperparameter

# Generation
SAMPLES_TO_GENERATE = 500 # Number of samples per class to generate after training

# Reproducibility
SEED = 42
# Check for CUDA, then MPS (for Mac), then CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
# ==================================================

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using device: {DEVICE}")

# ==============================================================================
# 1. DATA LOADING & PREPROCESSING
# ==============================================================================
META_COLS = {"groupnumbers", "classes", "class_name", "binary_label", "groupcodes", "obsnames"}

def spectral_cols(df):
    cols = []
    for c in df.columns:
        if c in META_COLS: continue
        try:
            float(c)
            cols.append(c)
        except:
            pass
    if not cols: raise ValueError("No wavenumber columns detected.")
    return [c for c in df.columns if c in set(cols)]

def preprocess_row(x_row: np.ndarray) -> np.ndarray:
    # SG 2nd derivative + L2 normalization
    win = 5 if x_row.size >= 5 else (x_row.size // 2 * 2 + 1)
    if win % 2 == 0: win += 1
    z = savgol_filter(x_row, window_length=win, polyorder=2, deriv=2)
    n = np.linalg.norm(z) + 1e-12
    return (z / n).astype(np.float32)

class CleanSpectraDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.cols = spectral_cols(df)
        X_raw = df[self.cols].to_numpy(dtype=np.float32)
        self.y = (df["classes"].values != 0).astype(np.int64)
        self.F = X_raw.shape[1]

        print(f"Preprocessing {len(X_raw)} spectra to 2nd-derivative domain...")
        self.X_clean = np.vstack([preprocess_row(r) for r in X_raw]).astype(np.float32)
        print(f"Data shape: {self.X_clean.shape}")

        self.wns = np.array([float(c) for c in self.cols])

    def __len__(self): return self.X_clean.shape[0]

    def __getitem__(self, i):
        # Add channel dimension: (1, F)
        return torch.from_numpy(self.X_clean[i])[None, :], torch.tensor(self.y[i])


# ==============================================================================
# 2. MODEL DEFINITIONS (Generator & Critic)
# ==============================================================================

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, feature_dim, model_dim=64):
        super(Generator, self).__init__()
        self.feature_dim = feature_dim
        self.initial_feature_size = feature_dim // 16 # Starting size for upsampling

        # Class embedding and input projection
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.latent_layer = nn.Sequential(
            nn.Linear(latent_dim + num_classes, model_dim * 4 * self.initial_feature_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder-style upsampling layers
        self.model = nn.Sequential(
            # Input shape: (B, model_dim*4, F//16)
            nn.ConvTranspose1d(model_dim * 4, model_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(model_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: (B, model_dim*2, ~F//8)
            nn.ConvTranspose1d(model_dim * 2, model_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(model_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: (B, model_dim, ~F//4)
            nn.ConvTranspose1d(model_dim, model_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(model_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: (B, model_dim//2, ~F//2)
            nn.ConvTranspose1d(model_dim // 2, 1, kernel_size=4, stride=2, padding=1),
            # Final shape: (B, 1, ~F) -> might not be exactly F due to integer math
        )

    def forward(self, z, labels):
        # Concatenate noise z and label embeddings
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        # Project and reshape for convolution
        x = self.latent_layer(x)
        x = x.view(x.shape[0], -1, self.initial_feature_size)
        # Pass through upsampling layers
        x = self.model(x)

        # --- FIX: Force output size to match exact feature dimension F ---
        if x.shape[-1] != self.feature_dim:
             x = Fnn.interpolate(x, size=self.feature_dim, mode='linear', align_corners=False)
        # -----------------------------------------------------------------
        return x

class Critic(nn.Module):
    def __init__(self, num_classes, feature_dim, model_dim=64):
        super(Critic, self).__init__()

        # Class embedding and mixing layer
        self.label_emb = nn.Embedding(num_classes, feature_dim)
        # Mixes the spectral input with the label embedding
        self.concat_layer = nn.Sequential(
            nn.Conv1d(1 + 1, model_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Encoder-style downsampling layers
        # Note: No BatchNorm in WGAN-GP Critic
        self.model = nn.Sequential(
            # Input shape: (B, model_dim//2, F)
            nn.Conv1d(model_dim // 2, model_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: (B, model_dim, ~F//2)
            nn.Conv1d(model_dim, model_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: (B, model_dim*2, ~F//4)
            nn.Conv1d(model_dim * 2, model_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: (B, model_dim*4, ~F//8)
        )

        # Determine output size of convolutions dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, model_dim // 2, feature_dim)
            dummy_output = self.model(dummy_input)
            self.conv_out_size = dummy_output.view(1, -1).shape[1]

        # Final scoring layer
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conv_out_size, 1)
        )

    def forward(self, x, labels):
        # Create label channel matching the data's spatial dimension
        c = self.label_emb(labels).unsqueeze(1) # (B, 1, F)
        # Concatenate data and label channels
        # This is where the error occurred before the fix in Generator
        x = torch.cat([x, c], 1) # (B, 2, F)
        # Mix them
        x = self.concat_layer(x)
        # Downsample
        x = self.model(x)
        # Score
        validity = self.final_layer(x)
        return validity


# ==============================================================================
# 3. WGAN-GP TRAINING LOGIC
# ==============================================================================

def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1).to(DEVICE)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = D(interpolates, labels)

    fake = torch.ones(real_samples.size(0), 1).to(DEVICE)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ==============================================================================
# 4. MAIN TRAINING LOOP
# ==============================================================================

def main():
    # --- Setup Data ---
    dataset = CleanSpectraDataset(RAW_TRAIN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    F_LEN = dataset.F

    # --- Initialize Models ---
    generator = Generator(LATENT_DIM, NUM_CLASSES, F_LEN, MODEL_DIM).to(DEVICE)
    critic = Critic(NUM_CLASSES, F_LEN, MODEL_DIM).to(DEVICE)

    # --- Optimizers (Betas are important for WGAN) ---
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(B1, B2))
    optimizer_C = torch.optim.Adam(critic.parameters(), lr=LR, betas=(B1, B2))

    print("Starting WGAN-GP training...")
    # Lists to keep track of progress
    g_losses = []
    d_losses = []
    batches_done = 0

    for epoch in range(EPOCHS):
        for i, (imgs, labels) in enumerate(dataloader):

            # Configure input
            real_imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            batch_size = real_imgs.size(0)

            # ---------------------
            #  Train Critic
            # ---------------------
            optimizer_C.zero_grad()

            # Sample noise and labels as generator input
            z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            # Generate a batch of fake images
            fake_imgs = generator(z, labels)

            # Real images score
            real_validity = critic(real_imgs, labels)
            # Fake images score
            fake_validity = critic(fake_imgs.detach(), labels)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(critic, real_imgs, fake_imgs.detach(), labels)

            # Adversarial loss (Wasserstein objective with GP)
            # Minimize: -E[C(real)] + E[C(fake)] + lambda * GP
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gradient_penalty

            d_loss.backward()
            optimizer_C.step()
            d_losses.append(d_loss.item())

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator every N_CRITIC steps
            if i % N_CRITIC == 0:
                optimizer_G.zero_grad()

                # Generate a batch of fake images
                fake_imgs = generator(z, labels)
                # Score the fake images
                fake_validity = critic(fake_imgs, labels)

                # Adversarial loss (Generator wants to maximize Critic's score for fakes)
                # Minimize: -E[C(fake)]
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()
                g_losses.append(g_loss.item())

                # Print log
                if i % 10 == 0:
                    print(
                        f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                        f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                    )

                batches_done += 1

    # Save final generator weights
    torch.save(generator.state_dict(), OUT_DIR / "cgan_generator_final.pt")
    print(f"Training finished. Final generator weights saved to {OUT_DIR / 'cgan_generator_final.pt'}")

    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Critic Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(OUT_DIR / "gan_loss_plot.png")
    plt.close()

    # ==========================================================================
    # 5. POST-TRAINING GENERATION
    # ==========================================================================
    print(f"\nGenerating {SAMPLES_TO_GENERATE} samples per class for evaluation...")

    generator.eval()

    with torch.no_grad():
        # --- Generate Healthy (Class 0) ---
        z_h = torch.randn(SAMPLES_TO_GENERATE, LATENT_DIM).to(DEVICE)
        labels_h = torch.zeros(SAMPLES_TO_GENERATE, dtype=torch.long).to(DEVICE)
        gen_h = generator(z_h, labels_h).squeeze(1).cpu().numpy()

        # --- Generate Cancer (Class 1) ---
        z_c = torch.randn(SAMPLES_TO_GENERATE, LATENT_DIM).to(DEVICE)
        labels_c = torch.ones(SAMPLES_TO_GENERATE, dtype=torch.long).to(DEVICE)
        gen_c = generator(z_c, labels_c).squeeze(1).cpu().numpy()

    # Save as .npy files (compatible with your existing evaluation scripts)
    np.save(OUT_DIR / "gan_samples_healthy.npy", gen_h)
    np.save(OUT_DIR / "gan_samples_cancer.npy", gen_c)

    print(f"Saved healthy samples to {OUT_DIR / 'gan_samples_healthy.npy'}")
    print(f"Saved cancer samples to {OUT_DIR / 'gan_samples_cancer.npy'}")

    # Quick visualization of generated samples
    wns = dataset.wns
    plt.figure(figsize=(10, 5))
    plt.plot(wns, gen_h[:5].T, color='blue', alpha=0.3, label='Generated Healthy')
    plt.plot(wns, gen_c[:5].T, color='red', alpha=0.3, label='Generated Cancer')
    plt.title("Preview of Generated GAN Spectra (First 5)")
    plt.xlabel("Wavenumber")
    plt.ylabel("Normalized Absorbance (2nd Deriv)")
    plt.grid(True)
    plt.savefig(OUT_DIR / "gan_generated_preview.png")
    plt.close()

    print("Preview plot saved. You can now point your QC.py and evaluation scripts to the .npy files.")


if __name__ == "__main__":
    main()