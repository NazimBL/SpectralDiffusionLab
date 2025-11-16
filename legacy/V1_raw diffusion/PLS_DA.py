import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# ------------------------------
# 1. Baseline Correction (Asymmetric Least Squares)
# ------------------------------
def baseline_als(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = np.diff(np.eye(L), 2)
    D = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(niter):
        W = np.diag(w)
        Z = np.linalg.inv(W + D)
        z = Z.dot(w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def apply_baseline_correction(X, lam=1e5, p=0.01, niter=10):
    X_corr = np.zeros_like(X)
    for i in range(X.shape[0]):
        bl = baseline_als(X[i, :], lam=lam, p=p, niter=niter)
        X_corr[i, :] = X[i, :] - bl
    return X_corr

# ------------------------------
# 2. Vector Normalization
# ------------------------------
def vector_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms

# ------------------------------
# 3. Load and Combine Data
# ------------------------------
train_df = pd.read_csv("train_set.csv")
val_df = pd.read_csv("val_set.csv")
test_df = pd.read_csv("test_set.csv")
df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# ------------------------------
# 4. Filter and Binarize Labels
# ------------------------------
df = df[df["class"] != 4].copy()  # Remove Hyperplasia
df["binary_class"] = df["class"].apply(lambda x: 0 if x == 0 else 1)

# ------------------------------
# 5. Select Fingerprint Region (900–1800 cm⁻¹)
# ------------------------------
non_features = {"sample_id", "class", "patient_id", "binary_class"}
feature_cols = [c for c in df.columns if c not in non_features]
wavenumbers = np.array([float(c) for c in feature_cols])
mask = (wavenumbers >= 900) & (wavenumbers <= 1800)
selected_features = list(np.array(feature_cols)[mask])

# ------------------------------
# 6. Preprocess Spectra
# ------------------------------
X_raw = df[selected_features].values
y_raw = df["binary_class"].values
groups_raw = df["patient_id"].values

# (a) Baseline correction
X_bc = apply_baseline_correction(X_raw)

# (b) 2nd derivative Savitzky–Golay (window=5)
X_sg = savgol_filter(X_bc, window_length=5, polyorder=2, deriv=2, axis=1)

# (c) Vector normalization
X_proc = vector_normalize(X_sg)

# ------------------------------
# 7. Average Replicates per Patient
# ------------------------------
proc_df = pd.DataFrame(X_proc, columns=selected_features)
proc_df["patient_id"] = groups_raw
proc_df["binary_class"] = y_raw

patient_avg = proc_df.groupby("patient_id")[selected_features].mean()
patient_avg["binary_class"] = proc_df.groupby("patient_id")["binary_class"].first()

X = patient_avg[selected_features].values
y = patient_avg["binary_class"].values
groups = patient_avg.index.values

# ------------------------------
# 8. Auto-scale Patient-Average Data
# ------------------------------
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 9. PLS-DA with LOPO CV and Component Tuning
# ------------------------------
logo = LeaveOneGroupOut()
component_range = range(2, 11)
aucs = []

for n_comp in component_range:
    all_probs, all_trues = [], []
    for train_idx, test_idx in logo.split(X_scaled, y, groups):
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_scaled[train_idx], y[train_idx])
        probs = pls.predict(X_scaled[test_idx]).ravel()
        all_probs.extend(probs)
        all_trues.extend(y[test_idx])
    aucs.append(roc_auc_score(all_trues, all_probs))

# Plot AUC vs n_components
plt.figure(figsize=(6,4))
plt.plot(list(component_range), aucs, marker='o')
plt.xlabel("Number of PLS Components")
plt.ylabel("LOPO CV ROC AUC")
plt.title("Component Tuning")
plt.tight_layout()
plt.show()

best_ncomp = component_range[np.argmax(aucs)]
best_auc = max(aucs)
print(f"Best PLS components: {best_ncomp}, LOPO CV AUC: {best_auc:.2f}")

# ------------------------------
# 10. Final LOPO Predictions & Metrics using Best Components
# ------------------------------
all_probs, all_trues = [], []
for train_idx, test_idx in logo.split(X_scaled, y, groups):
    pls = PLSRegression(n_components=best_ncomp)
    pls.fit(X_scaled[train_idx], y[train_idx])
    probs = pls.predict(X_scaled[test_idx]).ravel()
    all_probs.extend(probs)
    all_trues.extend(y[test_idx])

# Determine optimal threshold via Youden's J
fpr, tpr, thresholds = roc_curve(all_trues, all_probs)
youden_scores = tpr - fpr
best_thresh = thresholds[np.argmax(youden_scores)]

# Binarize predictions
all_preds = np.array(all_probs) >= best_thresh

# Calculate metrics
auc = roc_auc_score(all_trues, all_probs)
accuracy = accuracy_score(all_trues, all_preds)
cm = confusion_matrix(all_trues, all_preds)
TN, FP, FN, TP = cm.ravel()
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print(f"AUC: {auc:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Sensitivity (Cancer): {sensitivity:.2f}")
print(f"Specificity (Healthy): {specificity:.2f}")
print(f"Decision threshold (Youden's J): {best_thresh:.3f}")

# ------------------------------
# 11. ROC Curve Plot
# ------------------------------
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1], [0,1], 'k--', label="Chance")
plt.scatter(fpr[np.argmax(youden_scores)], tpr[np.argmax(youden_scores)],
            marker='o', color='red', label="Youden's J threshold")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Healthy vs. Cancer")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("ROC.png")
