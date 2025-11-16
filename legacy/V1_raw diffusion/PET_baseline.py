import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt

# ----------------------
# 0. Set manual seed & device
# ----------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------
# 1. Load & Filter Labels
# ----------------------
def load_df(path):
    df = pd.read_csv(path)
    # drop hyperplasia and binarize
    df = df[df['class'] != 4].copy()
    df['binary_class'] = (df['class'] != 0).astype(int)
    return df

train_df = load_df("train_set.csv")
val_df   = load_df("val_set.csv")
test_df  = load_df("test_set.csv")


# ----------------------
# 2. Feature Selection: fingerprint region 900–1800 cm⁻¹
# ----------------------
non_feats = {"sample_id","class","patient_id","binary_class"}
feat_cols = [c for c in train_df.columns if c not in non_feats]
wns = np.array([float(c) for c in feat_cols])
mask = (wns >= 900) & (wns <= 1800)
feat_cols = list(np.array(feat_cols)[mask])


# ----------------------
# 3. Spectral Preprocessing
# ----------------------
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

def preprocess_spectra(X):
    # baseline
    X_bc = np.zeros_like(X)
    for i in range(X.shape[0]):
        bl = baseline_als(X[i])
        X_bc[i] = X[i] - bl
    # 2nd-derivative SG
    X_sg = savgol_filter(X_bc, window_length=5, polyorder=2, deriv=2, axis=1)
    # vector norm
    norms = np.linalg.norm(X_sg, axis=1, keepdims=True)
    norms[norms==0] = 1
    return X_sg / norms

# apply to each split
def prepare(df):
    X = df[feat_cols].values
    y = df["binary_class"].values
    X = preprocess_spectra(X)
    return X, y

X_train, y_train = prepare(train_df)
X_val,   y_val   = prepare(val_df)
X_test,  y_test  = prepare(test_df)


# ----------------------
# 4. Standard Scaling
# ----------------------
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)


# ----------------------
# 5. Convert to PyTorch
# ----------------------
X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
y_train_t = torch.tensor(y_train, dtype=torch.long,   device=DEVICE)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32, device=DEVICE)
y_val_t   = torch.tensor(y_val,   dtype=torch.long,   device=DEVICE)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32, device=DEVICE)
y_test_t  = torch.tensor(y_test,  dtype=torch.long,   device=DEVICE)


# ----------------------
# 6. Positional Transformer
# ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(seq_len, d_model))
    def forward(self, x):
        return x + self.encoding

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4,
                 hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.pos   = PositionalEncoding(seq_len=1, d_model=hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head    = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        # x: [B, F]
        x = self.embed(x).unsqueeze(1)    # [B, 1, H]
        x = self.pos(x)
        x = self.encoder(x)               # [B, 1, H]
        x = x.mean(dim=1)                 # global pool
        return self.head(x)

model = TransformerClassifier(input_dim=len(feat_cols), num_classes=2)
model.to(DEVICE)


# ----------------------
# 7. Train
# ----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs    = 100
batch     = 32

for ep in range(1, epochs+1):
    model.train()
    perm = torch.randperm(len(X_train_t), device=DEVICE)
    total_loss = 0.0
    for i in range(0, len(perm), batch):
        idx = perm[i:i+batch]
        xb, yb = X_train_t[idx], y_train_t[idx]
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)
    # val
    model.eval()
    with torch.no_grad():
        v_logits = model(X_val_t)
        v_pred = v_logits.argmax(dim=1)
        v_acc  = accuracy_score(y_val, v_pred.cpu().numpy())
    print(f"Epoch {ep:03d} | Train Loss: {total_loss/len(y_train):.4f} | Val Acc: {v_acc:.4f}")


# ----------------------
# 8. Evaluate on Test
# ----------------------
model.eval()
with torch.no_grad():
    t_logits = model(X_test_t)
    t_probs  = nn.functional.softmax(t_logits, dim=1)[:,1].cpu().numpy()
    t_pred   = t_logits.argmax(dim=1).cpu().numpy()
    acc      = accuracy_score(y_test, t_pred)
    auc      = roc_auc_score(y_test, t_probs)
    tn, fp, fn, tp = confusion_matrix(y_test, t_pred).ravel()
    sens     = tp/(tp+fn)
    spec     = tn/(tn+fp)

print("\nTest Results:")
print(f"Accuracy: {acc:.2f} | AUC: {auc:.2f}")
print(f"Sensitivity: {sens:.2f} | Specificity: {spec:.2f}")
print(classification_report(y_test, t_pred, target_names=["Healthy","Cancer"]))

# ROC plot
fpr, tpr, _ = roc_curve(y_test, t_probs)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve: Healthy vs. Cancer")
plt.legend()
plt.tight_layout()
plt.savefig("ROC_transformer.png")
