import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ----------------------
# 1. Load Pre-Split Datasets
# ----------------------
train_df = pd.read_csv("train_set.csv")
val_df = pd.read_csv("val_set.csv")
test_df = pd.read_csv("test_set.csv")

# Extract features and labels
# Exclude non-feature columns like "sample_id", "patient_id", etc.
X_train = train_df.drop(columns=["class", "sample_id", "patient_id"]).values
y_train = train_df["class"].values
X_val = val_df.drop(columns=["class", "sample_id", "patient_id"]).values
y_val = val_df["class"].values
X_test = test_df.drop(columns=["class", "sample_id", "patient_id"]).values
y_test = test_df["class"].values

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert labels to integers
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# ----------------------
# 2. Define Transformer Model
# ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.randn(seq_len, d_model))  # Learnable positional encoding

    def forward(self, x):
        return x + self.encoding  # Add positional encoding to input


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, hidden_dim=128, num_layers=2, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)  # Project input to hidden size
        self.positional_encoding = PositionalEncoding(seq_len=1, d_model=hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, num_classes)  # Classification head
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)  # Project to hidden size
        x = self.positional_encoding(x)  # Add positional encoding
        x = self.transformer_encoder(x)  # Pass through Transformer layers
        x = self.relu(x)  # Activation
        x = self.fc(x.mean(dim=1))  # Global average pooling before classification
        return x


# ----------------------
# 3. Model Initialization
# ----------------------
input_dim = X_train.shape[1]
num_classes = len(np.unique(y_train))
model = TransformerClassifier(input_dim, num_classes)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 200
batch_size = 64

# ----------------------
# 4. Training Loop
# ----------------------
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size(0))

    # Training batches
    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        X_batch, y_batch = X_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(X_batch.unsqueeze(1))  # Adding sequence dimension
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    # Validation step
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor.unsqueeze(1))
        _, val_pred = torch.max(val_outputs, 1)
        val_accuracy = accuracy_score(y_val_tensor.numpy(), val_pred.numpy())

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}")

# ----------------------
# 5. Evaluation on Test Set
# ----------------------
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor.unsqueeze(1))
    _, y_pred = torch.max(test_outputs, 1)
    test_accuracy = accuracy_score(y_test_tensor.numpy(), y_pred.numpy())
    print(f"\nTransformer Model Test Accuracy: {test_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test_tensor.numpy(), y_pred.numpy()))