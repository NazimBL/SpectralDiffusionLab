import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from collections import defaultdict
import matplotlib.pyplot as plt

# ----------------------
# 1. Data Loading & Parsing
# ----------------------
with open("Endo Cancer ATIR FTIR.txt", "r") as f:
    lines = f.readlines()

# Parse feature wavelengths
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
# 2. Patient Grouping & Prep
# ----------------------
# Extract patient ID from sample_id
df["patient_id"] = df["sample_id"].str.split(".", n=1).str[0]

# Create patient-level dataframe (one row per patient)
patient_df = df.groupby("patient_id").agg({
    "class": "first",
    "patient_id": "count"
}).rename(columns={"patient_id": "num_samples"}).reset_index()


# ----------------------
# 3. Custom Stratified Split (70-15-15)
# ----------------------
def stratified_patient_split(patient_df, strat_col, test_size, random_state):
    """Custom stratified split at patient level"""
    rng = np.random.RandomState(random_state)
    groups = defaultdict(list)

    # Group patients by class
    for idx, row in patient_df.iterrows():
        groups[row[strat_col]].append(row["patient_id"])

    # Split each class group proportionally
    train_patients = []
    test_patients = []

    for class_label, patients in groups.items():
        n_test = max(1, int(len(patients) * test_size))  # Ensure at least 1 sample
        test = rng.choice(patients, n_test, replace=False)
        train = list(set(patients) - set(test))
        train_patients.extend(train)
        test_patients.extend(test)

    return train_patients, test_patients


# First split: 70% train, 30% temp
train_patients, temp_patients = stratified_patient_split(
    patient_df,
    strat_col="class",
    test_size=0.3,
    random_state=42
)

# Second split: 15% validation, 15% test
val_patients, test_patients = stratified_patient_split(
    patient_df[patient_df["patient_id"].isin(temp_patients)],
    strat_col="class",
    test_size=0.5,
    random_state=42
)

# ----------------------
# 4. Apply Splits to Original Data
# ----------------------
# Assign set labels
df["set"] = np.select(
    [
        df["patient_id"].isin(train_patients),
        df["patient_id"].isin(val_patients),
        df["patient_id"].isin(test_patients)
    ],
    ["train", "val", "test"],
    default="unassigned"
)

# Split datasets
train_df = df[df["set"] == "train"].drop(columns="set")
val_df = df[df["set"] == "val"].drop(columns="set")
test_df = df[df["set"] == "test"].drop(columns="set")

# ----------------------
# 5. Validation & Saving
# ----------------------
# Verify no patient overlap
assert len(set(train_patients) & set(val_patients)) == 0
assert len(set(train_patients) & set(test_patients)) == 0
assert len(set(val_patients) & set(test_patients)) == 0

# Verify class distributions
print("\nClass distributions:")
for name, dataset in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    dist = dataset["class"].value_counts(normalize=True).sort_index()
    print(f"{name}: {dict(dist.round(2))}")

# Save splits
train_df.to_csv("train_set.csv", index=False)
val_df.to_csv("val_set.csv", index=False)
test_df.to_csv("test_set.csv", index=False)

# ----------------------
# 6. Visualization
# ----------------------
# Class distributions
fig, ax = plt.subplots(1, 3, figsize=(18, 4))
for i, (name, data) in enumerate(zip(["Train", "Validation", "Test"], [train_df, val_df, test_df])):
    data["class"].value_counts(normalize=True).sort_index().plot(
        kind="bar",
        title=f"{name} Set",
        ax=ax[i]
    )
    ax[i].set_xticklabels(['Healthy', 'Type I', 'Type II', 'Mixed', 'Hyperplasia'], rotation=45)
plt.tight_layout()
plt.savefig("class_distributions.png")
plt.show()