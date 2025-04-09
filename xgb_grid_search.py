import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             roc_auc_score, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Baseline Correction Function (Asymmetric Least Squares)
# ------------------------------
def baseline_als(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    # Create a second-order difference matrix
    D = np.diff(np.eye(L), 2)
    D = lam * D.dot(D.T)
    w = np.ones(L)
    for i in range(niter):
        W = np.diag(w)
        Z = np.linalg.inv(W + D)
        z = Z.dot(w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def apply_baseline_correction(X, lam=1e5, p=0.01, niter=10):
    X_corrected = np.zeros_like(X)
    for i in range(X.shape[0]):
        baseline = baseline_als(X[i, :], lam=lam, p=p, niter=niter)
        X_corrected[i, :] = X[i, :] - baseline
    return X_corrected

# ------------------------------
# Vector Normalization Function
# ------------------------------
def vector_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    return X / norms

# ------------------------------
# 1. Load Data and Convert Labels to Binary
# ------------------------------
# Load datasets
train_df = pd.read_csv("train_set.csv")
val_df = pd.read_csv("val_set.csv")
test_df = pd.read_csv("test_set.csv")

# Identify spectral feature columns by excluding non-spectral columns
non_feature_cols = ["sample_id", "class", "patient_id"]
feature_cols = [col for col in train_df.columns if col not in non_feature_cols]

# Convert multi-class labels to binary:
# Healthy = 0; any cancer (classes 1,2,3,4) = 1
def to_binary(label):
    return 0 if label == 0 else 1

train_df['binary_class'] = train_df['class'].apply(to_binary)
val_df['binary_class'] = val_df['class'].apply(to_binary)
test_df['binary_class'] = test_df['class'].apply(to_binary)

# Extract features and binary labels
X_train = train_df[feature_cols].values
y_train = train_df["binary_class"].values
X_val = val_df[feature_cols].values
y_val = val_df["binary_class"].values
X_test = test_df[feature_cols].values
y_test = test_df["binary_class"].values

# ------------------------------
# 2. Spectral Pre-processing
# ------------------------------
# (a) Baseline Correction
X_train_bc = apply_baseline_correction(X_train, lam=1e5, p=0.01, niter=10)
X_val_bc = apply_baseline_correction(X_val, lam=1e5, p=0.01, niter=10)
X_test_bc = apply_baseline_correction(X_test, lam=1e5, p=0.01, niter=10)

# (b) 2nd-order Savitzky–Golay Derivative (window=5)
X_train_sg = savgol_filter(X_train_bc, window_length=5, polyorder=2, deriv=2, axis=1)
X_val_sg = savgol_filter(X_val_bc, window_length=5, polyorder=2, deriv=2, axis=1)
X_test_sg = savgol_filter(X_test_bc, window_length=5, polyorder=2, deriv=2, axis=1)

# (c) Vector Normalization
X_train_norm = vector_normalize(X_train_sg)
X_val_norm = vector_normalize(X_val_sg)
X_test_norm = vector_normalize(X_test_sg)

# ------------------------------
# 3. Feature Scaling
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_norm)
X_val_scaled = scaler.transform(X_val_norm)
X_test_scaled = scaler.transform(X_test_norm)

# ------------------------------
# 4. Model Training with Hyperparameter Tuning (XGBoost)
# ------------------------------
param_grid = {
    'max_depth': [3, 4, 5],
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.1, 0.2]
}

xgb = XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, eval_metric='logloss')
grid = GridSearchCV(xgb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train_scaled, y_train)

print("Best parameters found: ", grid.best_params_)
best_model = grid.best_estimator_

# ------------------------------
# 5. Evaluation on Test Set
# ------------------------------
y_pred = best_model.predict(X_test_scaled)
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()
sensitivity = TP / (TP + FN)  # Recall for cancer (positive class)
specificity = TN / (TN + FP)  # Recall for healthy (negative class)

print("Binary Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy: {:.2f}".format(accuracy))
print("Sensitivity (Recall for Cancer): {:.2f}".format(sensitivity))
print("Specificity (Recall for Healthy): {:.2f}".format(specificity))
print("ROC AUC: {:.2f}".format(auc_score))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="ROC curve (AUC = {:.2f})".format(auc_score))
plt.plot([0, 1], [0, 1], 'k--', label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Binary Classification")
plt.legend(loc="lower right")
plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=["Healthy", "Cancer"],
            yticklabels=["Healthy", "Cancer"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Binary Classification")
plt.show()
