#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FTIR IRootLab Parser & Patient-Level Distribution Plot

- Reads the ATR-FTIR dataset exported in IRootLab table format.
- Extracts class labels from header (e.g., "{ 'Healthy', 'Type I', ... }").
- Maps numeric classes 0..4 to names.
- Averages 5 replicate spectra per patient (groupnumbers) -> 1 row per patient.
- Saves intermediate CSVs and plots patient-level class distribution.

"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_class_labels(text: str) -> List[str]:
    """
    Extract class labels from the header chunk.
    The file contains a token like:
        classlabels "{ 'Healthy', 'Type I', 'Type II', 'Mixed', 'Hyperplasia' }"
    """
    m = re.search(r'classlabels\s+"{\s*([^}]*)\s*}"', text, flags=re.IGNORECASE)
    if not m:
        raise ValueError("Could not find classlabels in header.")
    # Split by comma and strip quotes/spaces
    labels = [x.strip().strip("'").strip('"') for x in m.group(1).split(",")]
    if not labels:
        raise ValueError("Parsed an empty class label list.")
    return labels


def find_table_header_row(path: Path) -> int:
    """
    Find the row index where the table header starts (line that begins with 'groupcodes').
    We read the file first as a raw table without header, then locate the index.
    """
    probe = pd.read_csv(path, sep="\t", header=None, engine="python", dtype=str, on_bad_lines="skip")
    candidates = probe.index[probe[0].astype(str).str.lower().eq("groupcodes")]
    if len(candidates) == 0:
        raise ValueError("Could not find table header row: no 'groupcodes' line.")
    return int(candidates[0])


def read_header_text(path: Path, n_lines: int = 60) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = []
        for _ in range(n_lines):
            try:
                lines.append(next(f))
            except StopIteration:
                break
    return "".join(lines)


def load_ir_rootlab_table(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load the IRootLab table, cast key columns, and attach class names.
    Returns:
        df_raw: raw spectra table (per-spectrum rows)
        labels: list of class names by index
    """
    header_text = read_header_text(path)
    labels = extract_class_labels(header_text)

    header_row = find_table_header_row(path)
    df_raw = pd.read_csv(path, sep="\t", header=header_row, engine="python", on_bad_lines="skip")

    # Ensure essential columns
    required_cols = {"groupnumbers", "obsnames", "classes"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Types
    df_raw["classes"] = df_raw["classes"].astype(int)
    df_raw["groupnumbers"] = df_raw["groupnumbers"].astype(int)

    # Map class names
    class_map = {i: labels[i] for i in range(len(labels))}
    df_raw["class_name"] = df_raw["classes"].map(class_map)

    return df_raw, labels


def average_replicates_per_patient(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Average the replicate spectra per patient using the spectral columns
    (everything after 'classes' column).
    """
    cols = df_raw.columns.tolist()
    if "classes" not in cols:
        raise ValueError("'classes' column not found; cannot locate spectral columns.")
    first_spec_idx = cols.index("classes") + 1
    spectral_cols = cols[first_spec_idx:]

    # Sanity check: ensure spectral columns are numeric
    df_num = df_raw.copy()
    df_num[spectral_cols] = df_num[spectral_cols].apply(pd.to_numeric, errors="coerce")

    # Average per patient ID, preserving classes & class_name
    avg_df = (
        df_num.groupby(["groupnumbers", "classes", "class_name"], as_index=False)[spectral_cols]
        .mean()
    )

    return avg_df


def patient_level_distribution(avg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the number of patients per class at the patient-level averaged dataset.
    """
    out = (
        avg_df.groupby(["classes", "class_name"])["groupnumbers"]
        .nunique()
        .reset_index(name="num_patients")
        .sort_values("classes")
        .reset_index(drop=True)
    )
    return out


def plot_distribution(dist_df: pd.DataFrame, title: str, out_path: Path | None = None) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(dist_df["class_name"], dist_df["num_patients"])
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of Patients")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def main(input_path: str, output_dir: str) -> None:
    in_path = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load & map classes
    df_raw, labels = load_ir_rootlab_table(in_path)

    # Basic sanity
    total_spectra = len(df_raw)
    num_patients_raw = df_raw["groupnumbers"].nunique()
    rep_counts = df_raw.groupby("groupnumbers")["obsnames"].count()
    print(f"Parsed labels: {labels}")
    print(f"Total spectra (rows): {total_spectra}")
    print(f"Unique patients (raw): {num_patients_raw}")
    print("Replicates per patient (summary):")
    print(rep_counts.describe())

    # Save raw parsed table
    raw_out = out_dir / "ftir_raw_parsed.csv"
    df_raw.to_csv(raw_out, index=False)
    print(f"Saved raw parsed table: {raw_out}")

    # 2) Average 5 replicates per patient
    avg_df = average_replicates_per_patient(df_raw)
    avg_out = out_dir / "ftir_patient_averaged.csv"
    avg_df.to_csv(avg_out, index=False)
    print(f"Saved patient-averaged table: {avg_out}")

    # 3) Patient-level distribution
    dist_df = patient_level_distribution(avg_df)
    dist_out = out_dir / "ftir_patient_distribution.csv"
    dist_df.to_csv(dist_out, index=False)
    print(f"Saved distribution table: {dist_out}")

    # 4) Plot
    fig_out = out_dir / "ftir_patient_distribution.png"
    plot_distribution(dist_df, "Patient-level Class Distribution (Averaged Replicates)", fig_out)

    print("Done.")


if __name__ == "__main__":
    # === EDIT THESE PATHS FOR YOUR MACHINE ===
    INPUT_PATH = r"Endo Cancer ATIR FTIR.txt"
    OUTPUT_DIR = r""
    main(INPUT_PATH, OUTPUT_DIR)
