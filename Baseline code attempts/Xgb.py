import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets (same as before)
train_df = pd.read_csv("train_set.csv")
val_df = pd.read_csv("val_set.csv")
test_df = pd.read_csv("test_set.csv")

# Separate features (spectra) and labels (class) (same as before)
X_train = train_df.drop(columns=["sample_id", "class", "patient_id"])
y_train = train_df["class"]
X_val = val_df.drop(columns=["sample_id", "class", "patient_id"])
y_val = val_df["class"]
X_test = test_df.drop(columns=["sample_id", "class", "patient_id"])
y_test = test_df["class"]

# Scale the features (same as before)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train an XGBoost model
xgb_model = XGBClassifier(objective='multi:softmax', num_class=len(y_train.unique()), random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Evaluate the XGBoost model
y_pred_xgb = xgb_model.predict(X_test_scaled)
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Confusion Matrix for XGBoost
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', xticklabels=xgb_model.classes_, yticklabels=xgb_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('XGBoost Confusion Matrix')
plt.savefig("xgboost_confusion_matrix.png")
plt.show()