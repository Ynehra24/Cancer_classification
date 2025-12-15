import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
FILE_DIR = "/Users/yatharthnehva/Desktop/CSS_COURSE/drive-download-20251101T110739Z-1-001"

CSV_FILES = [
    "TCGA-BLCARNASeq_Bladder Urothelial Carcinoma_converted_merged.csv",
    "TCGA-BRCARNASeq_Breast Invasive Carcinoma_converted_merged.csv",
    "TCGA-ESCARNASeq_Esophageal Carcinoma_converted_merged.csv",
    "TCGA-HNSCRNASeq_Head and Neck Squamous Cell Carcinoma_converted_merged.csv",
    "TCGA-KICHRNASeq_Kidney Chromophobe_converted_merged.csv",
    "TCGA-KIRCRNASeq_Kidney Renal Clear Cell Carcinoma_converted_merged.csv",
    "TCGA-KIRPRNASeq_Kidney Renal Papillary Cell Carcinoma_converted_merged.csv",
    "TCGA-LIHCRNASeq_Liver Hepatocellular Carcinoma_converted_merged.csv",
    "TCGA-LUADRNASeq_Lung Adenocarcinoma_converted_merged.csv",
    "TCGA-LUSCRNASeq_Lung Squamous Cell Carcinoma_converted_merged.csv",
    "TCGA-PRADRNASeq_Prostate Adenocarcinoma_converted_merged.csv",
    "TCGA-STADRNASeq_Stomach Adenocarcinoma_converted_merged.csv",
    "TCGA-THCARNASeq_Thyroid Carcinoma_converted_merged.csv",
    "TCGA-UCECRNASeq_Uterine Corpus Endometrial Carcinoma_converted_merged.csv",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
BATCH_SIZE = 32
EPOCHS = 25
LR = 5e-4
K_FEATURES = 2000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARTIFACT_DIR = "./artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def parse_class_name_from_filename(fname):
    m = re.search(r"^[^_]+_([^_]+(?: [^_]+)*)_converted", fname)
    return m.group(1).strip()

def extract_sample_type(sample_id):
    try:
        return "Normal" if sample_id.split("-")[3][:2] == "11" else "Tumor"
    except Exception:
        return "Tumor"

def benjamini_hochberg(pvals, alpha=0.05):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    adj = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        adj[i] = prev = min(prev, ranked[i] * n / (i + 1))

    fdr = np.empty(n)
    fdr[order] = np.minimum(adj, 1.0)
    return fdr <= alpha, fdr

def load_and_transpose_csv(path, class_label):
    df = pd.read_csv(path)
    gene_col = df.columns[1]
    genes = df[gene_col].astype(str).tolist()

    df_t = df.drop(df.columns[:2], axis=1).T
    df_t.columns = genes
    df_t = df_t.reset_index().rename(columns={"index": "Sample_ID"})

    df_t["Label"] = np.where(
        df_t["Sample_ID"].apply(extract_sample_type) == "Normal",
        "Normal",
        class_label
    )
    return df_t

dfs = []
for f in tqdm(CSV_FILES, desc="Loading CSVs"):
    dfs.append(
        load_and_transpose_csv(
            os.path.join(FILE_DIR, f),
            parse_class_name_from_filename(f)
        )
    )

common_genes = sorted(
    set.intersection(*[set(df.columns) - {"Sample_ID", "Label"} for df in dfs])
)

df_all = pd.concat(
    [df[["Sample_ID"] + common_genes + ["Label"]] for df in dfs],
    ignore_index=True
)

class_counts = df_all["Label"].value_counts()
class_order = class_counts.index.tolist() 

plt.figure(figsize=(12,5))
plt.bar(class_order, class_counts.values)
plt.xticks(rotation=90)
plt.ylabel("Number of Samples")
plt.title("Class Imbalance in TCGA Dataset")
plt.tight_layout()
plt.savefig(f"{ARTIFACT_DIR}/class_imbalance.png", dpi=300)
plt.show()

X = np.log2(df_all[common_genes].astype(np.float32).values + 1.0)
y_labels = df_all["Label"].values

le = LabelEncoder()
y = le.fit_transform(y_labels)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    stratify=y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

selector = SelectKBest(f_classif, k=min(K_FEATURES, X_tr.shape[1]))
X_tr = selector.fit_transform(X_tr, y_tr)
X_te = selector.transform(X_te)

rej, fdr = benjamini_hochberg(selector.pvalues_)

genes_df = pd.DataFrame({
    "Gene": common_genes,
    "F_score": selector.scores_,
    "p_raw": selector.pvalues_,
    "p_fdr": fdr,
    "FDR_significant": rej
}).sort_values("F_score", ascending=False)

genes_df.to_csv(f"{ARTIFACT_DIR}/genes_ranked_anova_fdr.csv", index=False)
genes_df.head(K_FEATURES).to_csv(f"{ARTIFACT_DIR}/top_{K_FEATURES}_genes.csv", index=False)

with open(f"{ARTIFACT_DIR}/top_genes.txt", "w") as f:
    for g in genes_df.head(K_FEATURES)["Gene"]:
        f.write(f"{g}\n")
cw = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_tr),
    y=y_tr
)
class_to_weight = {
    cls: cw[idx] for idx, cls in enumerate(le.classes_)
}
class_to_weight["Kidney Chromophobe"] *= 1.5

ordered_weights = [class_to_weight[c] for c in class_order]

plt.figure(figsize=(12,5))
plt.bar(class_order, ordered_weights)
plt.xticks(rotation=90)
plt.ylabel("Class Weight")
plt.title("Class Weights Used for Training")
plt.tight_layout()
plt.savefig(f"{ARTIFACT_DIR}/class_weights.png", dpi=300)
plt.show()
weights_tensor = torch.tensor(
    [class_to_weight[c] for c in le.classes_],
    dtype=torch.float32
).to(DEVICE)
train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_tr).float().unsqueeze(1),
        torch.tensor(y_tr)
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_te).float().unsqueeze(1),
        torch.tensor(y_te)
    ),
    batch_size=BATCH_SIZE
)
class MulticlassCNN(nn.Module):
    def __init__(self, n_feats, n_classes):
        super().__init__()
        self.conv = nn.Conv1d(1, 32, kernel_size=9, padding=4)
        self.bn = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * (n_feats // 2), 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn(self.conv(x))))
        x = self.drop(x)
        x = torch.relu(self.fc1(x.flatten(1)))
        return self.fc2(self.drop(x))
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=3.0, smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = torch.log_softmax(logits, dim=1)
        ce = -(true_dist * log_probs).sum(dim=1)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

criterion = FocalLoss(alpha=weights_tensor)
model = MulticlassCNN(X_tr.shape[1], len(le.classes_)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
train_loss, val_loss, train_acc, val_acc = [], [], [], []
best_val = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    tl, tp, ty = 0.0, [], []

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        tl += loss.item()
        tp.append(logits.argmax(1).cpu().numpy())
        ty.append(yb.cpu().numpy())

    tl /= len(train_loader)
    ta = accuracy_score(np.concatenate(ty), np.concatenate(tp))

    model.eval()
    vl, vp, vy = 0.0, [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(DEVICE))
            loss = criterion(logits, yb.to(DEVICE))
            vl += loss.item()
            vp.append(logits.argmax(1).cpu().numpy())
            vy.append(yb.numpy())

    vl /= len(test_loader)
    va = accuracy_score(np.concatenate(vy), np.concatenate(vp))

    train_loss.append(tl); val_loss.append(vl)
    train_acc.append(ta); val_acc.append(va)

    print(f"Epoch {epoch:02d} | Train Acc {ta:.4f} | Val Acc {va:.4f}")

    if vl < best_val:
        best_val = vl
        torch.save(model.state_dict(), f"{ARTIFACT_DIR}/best_model.pth")
plt.figure(figsize=(8,6))
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.legend()
plt.title("Loss Curves")
plt.tight_layout()
plt.savefig(f"{ARTIFACT_DIR}/loss_curves.png", dpi=300)
plt.show()

plt.figure(figsize=(8,6))
plt.plot(train_acc, label="Train Accuracy")
plt.plot(val_acc, label="Val Accuracy")
plt.legend()
plt.title("Accuracy Curves")
plt.tight_layout()
plt.savefig(f"{ARTIFACT_DIR}/accuracy_curves.png", dpi=300)
plt.show()
model.load_state_dict(torch.load(f"{ARTIFACT_DIR}/best_model.pth"))
model.eval()

preds, truth = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
        truth.append(yb.numpy())

preds = np.concatenate(preds)
truth = np.concatenate(truth)

report = classification_report(
    truth, preds, target_names=le.classes_, digits=4
)

macro_f1 = f1_score(truth, preds, average="macro")
acc = accuracy_score(truth, preds)

print("\nCLASS-WISE METRICS (WITH CLASS NAMES)")
print(report)
print(f"Accuracy: {acc:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

with open(f"{ARTIFACT_DIR}/metrics_summary.txt", "w") as f:
    f.write(report)
    f.write(f"\nAccuracy: {acc:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")

cm = confusion_matrix(truth, preds)
plt.figure(figsize=(10,8))
plt.imshow(cm, cmap="Blues")
plt.xticks(range(len(le.classes_)), le.classes_, rotation=90)
plt.yticks(range(len(le.classes_)), le.classes_)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{ARTIFACT_DIR}/confusion_matrix.png", dpi=300)
plt.show()
