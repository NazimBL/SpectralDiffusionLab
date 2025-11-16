import torch
import numpy as np
import pandas as pd
from savemodel import UNet1D, cosine_beta_schedule, compute_alpha, sample_ddpm  # Adjust import as needed

# ---------------------
# Load saved model
# ---------------------
checkpoint = torch.load("trained_unet_model.pth", map_location='cuda' if torch.cuda.is_available() else 'cpu')

# Extract saved components
model_state_dict = checkpoint['model_state_dict']
timesteps = checkpoint['timesteps']
betas = checkpoint['betas']
alpha_bars = checkpoint['alpha_bars']
length = checkpoint['length']
num_classes = checkpoint['num_classes']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and load weights
model = UNet1D(num_classes=num_classes, time_emb_dim=128, base_ch=64).to(device)
model.load_state_dict(model_state_dict)
model.eval()

# ---------------------
# Generate synthetic samples
# ---------------------

def generate_and_save_synthetic(class_label, num_samples=5, out_dir="generated_spectra"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    cond = torch.tensor([class_label], dtype=torch.long, device=device)

    for i in range(num_samples):
        synth = sample_ddpm(model, cond, timesteps, betas, alpha_bars, device, guidance_scale=1.0, length=length)
        spectrum = synth.cpu().detach().numpy().squeeze()

        # Save as CSV row
        csv_path = f"{out_dir}/class{class_label}_sample{i+1}.csv"
        pd.DataFrame(spectrum.reshape(1, -1)).to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

# Example usage
generate_and_save_synthetic(class_label=0, num_samples=10)
generate_and_save_synthetic(class_label=1, num_samples=10)
