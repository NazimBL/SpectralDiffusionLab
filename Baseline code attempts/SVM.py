import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_df = pd.read_csv("train_set.csv")
val_df = pd.read_csv("val_set.csv")
test_df = pd.read_csv("test_set.csv")

# Separate features (spectra) and labels (class)
X_train = train_df.drop(columns=["sample_id", "class", "patient_id"])
y_train = train_df["class"]
X_val = val_df.drop(columns=["sample_id", "class", "patient_id"])
y_val = val_df["class"]
X_test = test_df.drop(columns=["sample_id", "class", "patient_id"])
y_test = test_df["class"]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train an SVM model
svm_model = SVC(kernel='rbf', C=1, gamma='scale')  # You can adjust these parameters
svm_model.fit(X_train_scaled, y_train)

# Evaluate the SVM model
y_pred_svm = svm_model.predict(X_test_scaled)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', xticklabels=svm_model.classes_, yticklabels=svm_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix')
plt.savefig("svm_confusion_matrix.png")
plt.show()