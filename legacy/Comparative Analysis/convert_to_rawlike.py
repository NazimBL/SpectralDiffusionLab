import os
import pandas as pd
import numpy as np
from scipy.integrate import cumtrapz

# -------------------------
# Configuration
# -------------------------
#Comparative Analysis/generated_spectra
input_dir = "generated_spectra"
output_dir = "generated_spectra_rawlike"
os.makedirs(output_dir, exist_ok=True)

train_set = pd.read_csv("train_set.csv")
feature_cols = [col for col in train_set.columns if col not in ["sample_id", "class", "patient_id"]]
X_real = train_set[feature_cols].values

# -------------------------
# Estimate Reference Stats
# -------------------------
mean_norm = np.mean(np.linalg.norm(X_real, axis=1))
mean_baseline = np.mean(X_real, axis=0)

# -------------------------
# Inverse Preprocessing Function
# -------------------------
def approximate_raw_from_preprocessed(preprocessed_spectrum):
    # Step 1: Undo vector normalization
    approx_unscaled = preprocessed_spectrum * mean_norm

    # Step 2: Approximate inverse 2nd derivative
    integrated_once = cumtrapz(approx_unscaled, dx=1, initial=0)
    raw_like = cumtrapz(integrated_once, dx=1, initial=0)

    # Step 3: Add average baseline
    raw_approx = raw_like + mean_baseline

    return raw_approx

# -------------------------
# Process All Synthetic CSVs
# -------------------------
for filename in os.listdir(input_dir):
    if filename.endswith(".csv") and "class" in filename:
        file_path = os.path.join(input_dir, filename)
        df = pd.read_csv(file_path)
        spectrum = df.values.squeeze()

        raw_like_spectrum = approximate_raw_from_preprocessed(spectrum)
        df_rawlike = pd.DataFrame([raw_like_spectrum], columns=feature_cols)

        out_path = os.path.join(output_dir, filename.replace(".csv", "_RAWLIKE.csv"))
        df_rawlike.to_csv(out_path, index=False)
        print(f"[✓] Converted and saved: {out_path}")
