#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
from numpy.linalg import norm

# ====== EDIT THESE ======
RAW_CSV = Path(r"ftir_raw_parsed.csv")
OUT_DIR = Path(r"MyDataset")
TRAIN_RATIO = 0.70
# ========================

OUT_DIR.mkdir(parents=True, exist_ok=True)
AVG_OUT = OUT_DIR / "ftir_patient_averaged_hvscancer.csv"
TRAIN_OUT = OUT_DIR / "ftir_raw_train.csv"
TEST_OUT  = OUT_DIR / "ftir_raw_test.csv"

META_COLS = {"groupcodes","groupnumbers","obsnames","classes","class_name"}

def kennard_stone(X: np.ndarray, k: int) -> np.ndarray:
    """Deterministic Kennard–Stone selection (Euclidean). Returns indices of k selected rows."""
    n = X.shape[0]
    if n <= 2:
        return np.arange(min(n, k), dtype=int)

    # farthest pair seeds
    max_d = -1.0
    seed_i, seed_j = 0, 1
    for i in range(n - 1):
        di = norm(X[i+1:] - X[i], axis=1)
        if di.size == 0:
            continue
        j_local = int(np.argmax(di))
        d = float(di[j_local])
        if d > max_d:
            max_d = d
            seed_i = i
            seed_j = i + 1 + j_local

    selected = [seed_i, seed_j]
    remaining = [i for i in range(n) if i not in selected]
    if not remaining or k <= 2:
        return np.array(selected[:k], dtype=int)

    def dist_to(idx, idxs):
        return norm(X[idxs] - X[idx], axis=1)

    min_dists = np.minimum(dist_to(seed_i, remaining), dist_to(seed_j, remaining))

    while len(selected) < k and remaining:
        pos = int(np.argmax(min_dists))
        chosen = remaining[pos]
        selected.append(chosen)

        remaining.pop(pos)
        if not remaining:
            break

        new_d = norm(X[remaining] - X[chosen], axis=1)
        # remove the used entry in min_dists, then update
        min_dists = np.minimum(np.delete(min_dists, pos), new_d)

    return np.array(selected[:k], dtype=int)

def main():
    df = pd.read_csv(RAW_CSV)

    # --- Guard rails ---
    if "classes" not in df.columns or "groupnumbers" not in df.columns:
        raise ValueError("Expected 'classes' and 'groupnumbers' columns in ftir_raw_parsed.csv")

    # --- Healthy vs Cancer by numeric codes (paper mapping confirmed) ---
    # 0=Healthy, 1=Type I, 2=Type II, 3=Mixed, 4=Hyperplasia
    keep_mask = df["classes"].isin([0, 1, 2, 3])  # exclude Hyperplasia from this task
    use_df = df.loc[keep_mask].copy()

    # --- Detect spectral columns: numeric, not in metadata ---
    numeric_cols = [c for c in use_df.columns if pd.api.types.is_numeric_dtype(use_df[c])]
    spectral_cols = [c for c in numeric_cols if c not in META_COLS]
    if not spectral_cols:
        raise ValueError("No spectral columns detected (numeric spectra expected).")

    # Ensure numeric spectra (single pass)
    use_df[spectral_cols] = use_df[spectral_cols].apply(pd.to_numeric, errors="coerce")

    # --- Average 5 replicates per patient ---
    agg_cols = ["groupnumbers", "classes"]
    if "class_name" in use_df.columns:  # optional
        agg_cols.append("class_name")

    avg_df = (
        use_df.groupby(agg_cols, as_index=False)[spectral_cols]
              .mean()
    )

    # Binary label
    avg_df["binary_label"] = np.where(avg_df["classes"].eq(0), "Healthy", "Cancer")

    # --- Clean columns: drop all-NaN or zero-variance spectrals; drop rows with NaNs ---
    all_nan_cols = [c for c in spectral_cols if avg_df[c].isna().all()]
    if all_nan_cols:
        avg_df = avg_df.drop(columns=all_nan_cols, errors="ignore")
        spectral_cols = [c for c in spectral_cols if c not in all_nan_cols]

    zero_var_cols = [c for c in spectral_cols if avg_df[c].nunique(dropna=True) <= 1]
    if zero_var_cols:
        avg_df = avg_df.drop(columns=zero_var_cols, errors="ignore")
        spectral_cols = [c for c in spectral_cols if c not in zero_var_cols]

    before = avg_df.shape[0]
    avg_df = avg_df.dropna(subset=spectral_cols)
    if avg_df.empty:
        raise ValueError("Averaging produced 0 rows after NaN cleanup—check spectral columns in CSV.")

    # Save the averaged Healthy vs Cancer dataset
    avg_df.to_csv(AVG_OUT, index=False)

    # --- Kennard–Stone split (on standardized spectra) ---
    X = avg_df[spectral_cols].to_numpy()
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0); sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    n = Xz.shape[0]
    k = max(1, int(round(TRAIN_RATIO * n)))
    sel = kennard_stone(Xz, k)

    train_mask = np.zeros(n, dtype=bool); train_mask[sel] = True
    train_df = avg_df.loc[train_mask].reset_index(drop=True)
    test_df  = avg_df.loc[~train_mask].reset_index(drop=True)

    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    # Logs
    print("Patients total:", n)
    print("Train/Test:", train_df.shape[0], "/", test_df.shape[0])
    print("Counts (binary):",
          "Train", train_df["binary_label"].value_counts().to_dict(),
          "| Test",  test_df["binary_label"].value_counts().to_dict())
    print(f"\nWrote:\n  {AVG_OUT}\n  {TRAIN_OUT}\n  {TEST_OUT}")

if __name__ == "__main__":
    main()
