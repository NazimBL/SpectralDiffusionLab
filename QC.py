import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.signal import savgol_filter

# ======== CONFIG ========
OUT_DIR = Path("ldm_out")

# real test data
TEST_CSV = Path("../MyDataset/ftir_test_wn.csv")

# The .npy files from generator script (train_ddpm_latent.py also generates .npy s
GEN_HEALTHY_FILE = OUT_DIR / "samples_healthy.npy"
GEN_CANCER_FILE = OUT_DIR / "samples_cancer.npy"

# ===============================

def plot_mean_std(ax, data, wavenumbers, label_prefix, class_name, color, linestyle='-'):
    """Helper function to plot a single mean/std spectrum on a given axis."""
    if len(data) == 0:
        print(f"Warning: No data for {label_prefix} {class_name}")
        return

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    label = f"{label_prefix} {class_name} mean (n={len(data)})"

    ax.plot(wavenumbers, mean, label=label, color=color, linestyle=linestyle)
    ax.fill_between(wavenumbers, mean - std, mean + std, alpha=0.2, color=color)

    print(f"Stats for {label_prefix} {class_name}:")
    print(f"  Mean range: {mean.min():.3f} to {mean.max():.3f}")


#  added preprocessing function from train_ae.py ---
def preprocess_row(x_row: np.ndarray) -> np.ndarray:
    # SG 2nd derivative + L2 normalization
    win = 5 if x_row.size >= 5 else (x_row.size // 2 * 2 + 1)
    # Ensure window is odd
    if win % 2 == 0: win += 1

    z = savgol_filter(x_row, window_length=win, polyorder=2, deriv=2)
    n = np.linalg.norm(z) + 1e-12
    return (z / n).astype(np.float32)



def main():
    # --- 1. Load Wavenumbers (X-axis) ---
    meta_path = OUT_DIR / "ae_meta.json"
    if not meta_path.exists():
        print(f"Error: Could not find {meta_path}")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)

    wavenumbers = np.array([float(c) for c in meta["cols"]])
    print(f"Loaded {len(wavenumbers)} wavenumbers.")

    # --- 2. Load Real Test Data ---
    if not TEST_CSV.exists():
        print(f"Error: Could not find {TEST_CSV}")
        return

    df_test = pd.read_csv(TEST_CSV)
    # Ensure columns match the order from meta
    real_spectra_raw = df_test[meta["cols"]].to_numpy(dtype=np.float32)  # <--- MODIFIED: Renamed
    # Cancer=1, Healthy=0
    real_labels = (df_test["classes"].values != 0).astype(np.int64)

    real_healthy_raw = real_spectra_raw[real_labels == 0]
    real_cancer_raw = real_spectra_raw[real_labels == 1]

    print(f"Loaded real test data: {len(real_healthy_raw)} healthy, {len(real_cancer_raw)} cancer.")

    # --- 3. Preprocess Real Test Data ---
    print("Applying 2nd-derivative preprocessing to real test data for comparison...")
    real_healthy_clean = np.vstack([preprocess_row(r) for r in real_healthy_raw]).astype(np.float32)
    real_cancer_clean = np.vstack([preprocess_row(r) for r in real_cancer_raw]).astype(np.float32)

    # --- 4. Load Generated Data ---
    if not GEN_HEALTHY_FILE.exists() or not GEN_CANCER_FILE.exists():
        print(f"Error: Could not find generated .npy files in {OUT_DIR}")
        print(f"Checked for: {GEN_HEALTHY_FILE.name} and {GEN_CANCER_FILE.name}")
        return

    gen_healthy = np.load(GEN_HEALTHY_FILE)
    gen_cancer = np.load(GEN_CANCER_FILE)

    print(f"Loaded generated data: {len(gen_healthy)} healthy, {len(gen_cancer)} cancer.")

    # --- 5. Plotting ---
    # Create a figure with 2 subplots, stacked vertically, sharing the x-axis
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 14), sharex=True)

    # --- Plot 1: Generated Data ---
    ax0 = axes[0]
    plot_mean_std(ax0, gen_healthy, wavenumbers, "Gen", "Healthy", color='tab:orange', linestyle='--')
    plot_mean_std(ax0, gen_cancer, wavenumbers, "Gen", "Cancer", color='tab:red', linestyle='--')
    ax0.set_title("Generated FTIR Spectra (2nd Derivative): Mean ± Std", fontsize=16)  # <--- MODIFIED
    ax0.set_ylabel("2nd Derivative (a.u.)", fontsize=12)  # <--- MODIFIED
    ax0.legend()
    ax0.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Real Data (NOW PREPROCESSED) ---
    ax1 = axes[1]

    plot_mean_std(ax1, real_healthy_clean, wavenumbers, "Real", "Healthy", color='tab:blue', linestyle='-')
    plot_mean_std(ax1, real_cancer_clean, wavenumbers, "Real", "Cancer", color='tab:green', linestyle='-')
    ax1.set_title("Real Test Data FTIR Spectra (2nd Derivative): Mean ± Std", fontsize=16)  # <--- MODIFIED
    ax1.set_ylabel("2nd Derivative (a.u.)", fontsize=12)  # <--- MODIFIED
    ax1.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)


    ax1.invert_xaxis()


    plt.tight_layout(pad=2.0)

    # Save the figure
    output_fig_path = OUT_DIR / "generated_and_real_separate.png"
    plt.savefig(output_fig_path, dpi=200)
    print(f"\nSuccessfully saved plot to: {output_fig_path}")

    plt.show()


if __name__ == "__main__":
    main()