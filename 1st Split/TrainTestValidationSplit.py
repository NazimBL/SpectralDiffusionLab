import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from matplotlib import pyplot as plt

# ----------------------
# 1. Load and Parse Data
# ----------------------
with open("Endo Cancer ATIR FTIR.txt", "r") as f:
    lines = f.readlines()

# Parse absorbance (wavenumbers)
fea_line = lines[3].strip().split("\t")[2:]  # Skip "fea_x" and empty tabs
fea_x = [float(val) for val in fea_line if val.replace('.', '', 1).replace('-', '', 1).isdigit()]

# Parse samples
data = []
for line in lines[6:]:
    if line.startswith("plasma"):
        parts = line.strip().split("\t")
        sample_id = parts[0]
        class_label = int(parts[3])
        intensities = [float(val) for val in parts[5:] if val.replace('.', '', 1).replace('-', '', 1).isdigit()]
        data.append([sample_id, class_label] + intensities)

# Create DataFrame
columns = ["sample_id", "class"] + fea_x
df = pd.DataFrame(data, columns=columns)

# ----------------------
# 2. Patient Grouping
# ----------------------
# Extract patient ID from sample_id (e.g., "plasma_EC163.0" → "plasma_EC163")
df["patient_id"] = df["sample_id"].str.split(".", n=1).str[0]

# Group by patient and assign majority class to each patient
grouped = df.groupby("patient_id")["class"].agg(lambda x: x.mode()[0]).reset_index()
grouped.columns = ["patient_id", "class"]


# ----------------------
# 3. Custom Stratified Group Split
# ----------------------
def custom_stratified_group_split(groups, y, test_size, random_state=None):
    """
    Custom stratified group split.
    Ensures that groups are split while preserving class distribution.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # Perform the split on unique groups
    train_idx, test_idx = next(splitter.split(groups, y, groups=groups))

    return train_idx, test_idx


# Extract unique patients and their corresponding classes
unique_patients = grouped["patient_id"].values
patient_classes = grouped["class"].values

# First split: 70% train, 30% temp (test + validation)
train_idx, temp_idx = custom_stratified_group_split(
    groups=unique_patients,
    y=patient_classes,
    test_size=0.3,
    random_state=42
)

# Get patient IDs for train and temp sets
train_patients = unique_patients[train_idx]
temp_patients = unique_patients[temp_idx]

# Second split: Split temp into 50% validation and 50% test (15% each)
temp_patient_classes = patient_classes[temp_idx]
val_idx, test_idx = custom_stratified_group_split(
    groups=temp_patients,
    y=temp_patient_classes,
    test_size=0.5,
    random_state=42
)

# Get patient IDs for validation and test sets
val_patients = temp_patients[val_idx]
test_patients = temp_patients[test_idx]

# ----------------------
# 4. Apply Splits to Original Data
# ----------------------
# Assign set labels to original DataFrame
df["set"] = np.select(
    [
        df["patient_id"].isin(train_patients),
        df["patient_id"].isin(val_patients),
        df["patient_id"].isin(test_patients)
    ],
    ["train", "val", "test"],
    default="unassigned"
)

# Split into DataFrames
train_df = df[df["set"] == "train"].drop(columns=["set"])
val_df = df[df["set"] == "val"].drop(columns=["set"])
test_df = df[df["set"] == "test"].drop(columns=["set"])

# ----------------------
# 5. Verify & Save Splits
# ----------------------
# Check for patient overlap
assert len(set(train_patients) & set(val_patients)) == 0, "Train/Val overlap!"
assert len(set(train_patients) & set(test_patients)) == 0, "Train/Test overlap!"
assert len(set(val_patients) & set(test_patients)) == 0, "Val/Test overlap!"

# Save splits (with all samples)
train_df.to_csv("train_set.csv", index=False)
val_df.to_csv("val_set.csv", index=False)
test_df.to_csv("test_set.csv", index=False)

# Save patient IDs for reference
pd.Series(train_patients).to_csv("train_patients.txt", index=False, header=False)
pd.Series(val_patients).to_csv("val_patients.txt", index=False, header=False)
pd.Series(test_patients).to_csv("test_patients.txt", index=False, header=False)

# ----------------------
# 6. Visualization
# ----------------------
# Class distribution plots
fig, ax = plt.subplots(1, 3, figsize=(18, 4))
for i, (name, data) in enumerate(zip(["Train", "Validation", "Test"], [train_df, val_df, test_df])):
    data["class"].value_counts(normalize=True).sort_index().plot(
        kind="bar",
        title=f"{name} Set Distribution",
        ax=ax[i]
    )
    ax[i].set_xticklabels(['Healthy', 'Type I', 'Type II', 'Mixed', 'Hyperplasia'], rotation=45)
plt.tight_layout()
plt.show()

# Example spectra from training set
plt.figure(figsize=(10, 5))
for label in train_df["class"].unique():
    subset = train_df[train_df["class"] == label]
    plt.plot(fea_x, subset.iloc[0, 2:-1], label=f"Class {label}")  # Exclude patient_id
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Intensity")
plt.title("Training Set Spectral Profiles")
plt.legend()
plt.show()