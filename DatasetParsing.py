
import pandas as pd
from matplotlib import pyplot as plt

with open("Endo Cancer ATIR FTIR.txt", "r") as f:
    lines = f.readlines()

# Parse feature wavelengths (fea_x)
fea_line = lines[3].strip().split("\t")[2:]  # Skip "fea_x" and empty tabs
fea_x = []
for val in fea_line:
    if val.replace('.', '', 1).replace('-', '', 1).isdigit():
        fea_x.append(float(val))

# Parse samples
data = []
for line in lines[6:]:
    if line.startswith("plasma"):
        parts = line.strip().split("\t")
        sample_id = parts[0]
        class_label = int(parts[3])
        intensities = []
        for val in parts[5:]:
            if val.replace('.', '', 1).replace('-', '', 1).isdigit():
                intensities.append(float(val))
        data.append([sample_id, class_label] + intensities)

# Create DataFrame
columns = ["sample_id", "class"] + fea_x
df = pd.DataFrame(data, columns=columns)
print(df.head())

plt.figure(figsize=(8, 4))
df["class"].value_counts().plot(kind="bar", title="Class Distribution")
plt.xticks(ticks=range(5), labels=['Healthy', 'Type I', 'Type II', 'Mixed', 'Hyperplasia'], rotation=45)
plt.show()

#example spectra
plt.figure(figsize=(10, 5))
for label in df["class"].unique():
    subset = df[df["class"] == label]
    plt.plot(fea_x, subset.iloc[0, 2:], label=f"Class {label}")
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Intensity")
plt.title("Spectral Profiles by Class")
plt.legend()
plt.show()

#patient grouping

