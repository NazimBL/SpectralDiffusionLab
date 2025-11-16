#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#RAW_CSV = Path(r"ftir_raw_parsed.csv")      # <- update
RAW_CSV = Path(r"MyDataset\ftir_train_wn.csv")
OUT_PNG = Path(r"patient_counts_per_class.png")  # <- optional

def main():
    df = pd.read_csv(RAW_CSV)

    # Choose class column; prefer names if present
    cls_col = "binary_label" if "binary_label" in df.columns else "classes"

    # Count UNIQUE patients (groupnumbers) per class
    if "groupnumbers" not in df.columns:
        raise ValueError("Column 'groupnumbers' not found in ftir_raw_parsed.csv")

    counts = (
        df.groupby(cls_col)["groupnumbers"]
          .nunique()
          .reset_index(name="num_patients")
    )

    # (Optional) keep a sensible order if numeric classes exist
    if cls_col == "classes":
        counts = counts.sort_values(by=cls_col)
    else:
        # otherwise keep alphabetical order (or use a custom list if you prefer)
        counts = counts.sort_values(by=cls_col)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(counts[cls_col], counts["num_patients"])
    plt.title("Number of Patients per Class")
    plt.xlabel("Class")
    plt.ylabel("Patients")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    # Save & show
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {OUT_PNG}")
    plt.show()

    # Also print the table in the console
    print("\nPatients per class:")
    print(counts.to_string(index=False))

if __name__ == "__main__":
    main()
