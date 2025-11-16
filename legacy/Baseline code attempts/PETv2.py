import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from xgboost import XGBClassifier

# 1. Load pre-split datasets
train_df = pd.read_csv("train_set.csv")
val_df = pd.read_csv("val_set.csv")
test_df = pd.read_csv("test_set.csv")


# 2. Data preparation function
def prepare_data(df):
    # Exclude non-feature columns
    X = df.drop(columns=["class", "sample_id", "patient_id"]).values
    y = df["class"].values
    return X, y


# Process all sets
X_train, y_train = prepare_data(train_df)
X_val, y_val = prepare_data(val_df)
X_test, y_test = prepare_data(test_df)

# 3. Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 4. Class weighting analysis
class_counts = np.bincount(y_train)
print("Class distribution in training set:")
for i, count in enumerate(class_counts):
    print(f"Class {i}: {count} samples ({count / len(y_train) * 100:.1f}%)")

# Calculate class weights (inverse frequency)
class_weights = torch.tensor(1.0 / (class_counts + 1e-6))
class_weights /= class_weights.sum()  # Normalize


# 5. PyTorch MLP implementation
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, len(class_counts))
        )

    def forward(self, x):
        return self.model(x)


# 6. Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_train.shape[1]

# Convert to tensors with explicit dtype
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)

# Class weights (now explicitly float32)
class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32)
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


# 7. Training function
def train_model(model, epochs=100):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val_f1 = 0
    patience = 10
    trigger_times = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # Calculate metrics
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        val_f1 = report['macro avg']['f1-score']

        # Early stopping and checkpointing
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            trigger_times = 0
            torch.save(model.state_dict(), "best_mlp.pth")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step(val_f1)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {train_loss:.2f} | Val F1: {val_f1:.3f}")


# 8. XGBoost implementation
def train_xgboost():
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(class_counts),
        max_depth=6,
        learning_rate=0.05,
        n_estimators=200,
        scale_pos_weight=(1 / (class_counts / class_counts.sum())),
        eval_metric='mlogloss'
    )

    # Train on combined train + val data
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    model.fit(X_train_val, y_train_val,
              eval_set=[(X_val, y_val)],
              verbose=10
              )

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nXGBoost Test Results:")
    print(classification_report(y_test, y_pred, zero_division=0))


# 9. Execute training
if __name__ == "__main__":
    # Train MLP
    print("Training MLP model...")
    mlp_model = SimpleMLP(input_dim=input_dim).to(device)
    train_model(mlp_model)

    # Evaluate MLP
    mlp_model.load_state_dict(torch.load("best_mlp.pth"))
    mlp_model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = mlp_model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    print("\nMLP Test Results:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Train XGBoost
    print("\nTraining XGBoost model...")
    train_xgboost()