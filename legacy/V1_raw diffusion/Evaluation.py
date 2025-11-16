import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from savemodel import load_and_preprocess_data, sample_ddpm, cosine_beta_schedule, compute_alpha, UNet1D

# -----------------------
# Parameters
# -----------------------
ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
trained_model_path = "trained_unet_model.pth"
real_data_csv = "train_set.csv"
val_csv = "val_set.csv"
test_csv = "test_set.csv"
results = []

# -----------------------
# Load Model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(trained_model_path, map_location=device)
model = UNet1D(num_classes=checkpoint['num_classes'], time_emb_dim=128, base_ch=64).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
timesteps = checkpoint['timesteps']
betas = checkpoint['betas']
alpha_bars = checkpoint['alpha_bars']
length = checkpoint['length']

# -----------------------
# Load and Preprocess Real Data
# -----------------------
X_train_real, y_train_real = load_and_preprocess_data(real_data_csv)
X_val, y_val = load_and_preprocess_data(val_csv)
X_test, y_test = load_and_preprocess_data(test_csv)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_real)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# Generate and Evaluate with Synthetic Data
# -----------------------
for ratio in ratios:
    num_synthetic = int(X_train_real.shape[0] * ratio)
    synth_X, synth_y = [], []

    for class_id in np.unique(y_train_real):
        n_class = (y_train_real == class_id).sum()
        n_samples = int(ratio * n_class)
        cond = torch.tensor([class_id] * n_samples, dtype=torch.long, device=device)
        for c in cond:
            x_gen = sample_ddpm(model, c.unsqueeze(0), timesteps, betas, alpha_bars, device, guidance_scale=5.0, length=length)
            synth_X.append(x_gen.detach().cpu().numpy())
            synth_y.append(c.item())

    synth_X = np.array(synth_X).squeeze()
    synth_y = np.array(synth_y)

    # Preprocess synthetic
    synth_X = normalize(torch.tensor(synth_X).unsqueeze(1), p=2, dim=-1).squeeze(1).numpy()
    synth_X_scaled = scaler.transform(synth_X)

    # Combine
    X_combined = np.vstack([X_train_scaled, synth_X_scaled])
    y_combined = np.concatenate([y_train_real, synth_y])

    # Train
    model_xgb = XGBClassifier(learning_rate=0.01, max_depth=4, n_estimators=500, use_label_encoder=False, eval_metric='logloss')
    model_xgb.fit(X_combined, y_combined)

    # Evaluate
    y_pred = model_xgb.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    results.append((ratio, acc))
    print(f"Ratio {int(ratio*100)}% synthetic → Accuracy: {acc:.4f}")

# -----------------------
# Plot Results
# -----------------------
import matplotlib.pyplot as plt

ratios_percent = [int(r*100) for r, _ in results]
accuracies = [a for _, a in results]

plt.plot(ratios_percent, accuracies, marker='o')
plt.xlabel('Synthetic Data % (of training set size)')
plt.ylabel('Accuracy on Test Set')
plt.title('XGBoost Accuracy vs Synthetic Data Ratio')
plt.grid(True)
plt.savefig("accuracy_vs_synthetic_ratio.png")
plt.show()
