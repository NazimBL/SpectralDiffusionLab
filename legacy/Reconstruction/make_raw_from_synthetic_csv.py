import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from inverse_preprocess import invert_preprocessing

# Step 1: Load all synthetic CSVs
synthetic_dir = "../Comparative Analysis/generated_spectra"
synthetic_list = []

for fname in os.listdir(synthetic_dir):
    if fname.endswith(".csv"):
        df = pd.read_csv(os.path.join(synthetic_dir, fname))
        synthetic_list.append(df.values)

synthetic_all = np.vstack(synthetic_list)

# Step 2: Load real raw spectra from train_set.csv
train_df = pd.read_csv("../train_set.csv")
non_feature_cols = ["sample_id", "class", "patient_id"]
feature_cols = [c for c in train_df.columns if c not in non_feature_cols]
X_real_raw = train_df[feature_cols].values

# Step 3: Run inverse preprocessing
reconstructed_synthetic = invert_preprocessing(synthetic_all, X_real_raw)

# Step 4: Average and compare
avg_synth_raw = reconstructed_synthetic.mean(axis=0)
avg_real_raw = X_real_raw.mean(axis=0)

# Step 5: Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(avg_synth_raw, label="Mean Synthetic (Reconstructed Raw)", linewidth=2)
plt.plot(avg_real_raw, label="Mean Real Raw", linestyle="--", alpha=0.8)
plt.xlabel("Wavenumber Index")
plt.ylabel("Absorbance")
plt.title("Average Raw Spectrum: Synthetic vs Real")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("synthetic_vs_real_average.png", dpi=300)
plt.show()
