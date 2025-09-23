# 📊 Census Income Prediction – PyTorch

Predict whether an individual earns **more than \$50,000 annually** using a binary classification model built with **PyTorch**.

---

## 📝 Overview
This project trains a neural network on a cleaned subset (30,000 rows) of the  
[Census Income (Adult) dataset](https://archive.ics.uci.edu/ml/datasets/adult).  
The model uses:

* **Categorical features**: `sex`, `education`, `marital-status`, `workclass`, `occupation`
* **Continuous features**: `age`, `hours-per-week`
* **Label**: `label` (0 = income ≤ \$50K, 1 = income > \$50K)

Key highlights:
* PyTorch `nn.Embedding` layers for categorical variables  
* Batch-normalized continuous inputs  
* One hidden layer with 50 neurons and dropout (`p=0.4`)  
* Adam optimizer and CrossEntropy loss  

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/thiruking/Census-Income-Prediction-PyTorch
cd census-income-pytorch
```
### PROGRAM :

```pyhton
# ============================================
# 1️⃣ Imports
# ============================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random

# Random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ============================================
# 2️⃣ Load Data
# ============================================
df = pd.read_csv("income.csv")  # adjust path if needed
print(df.head())

# ============================================
# 3️⃣ Identify columns
# ============================================
label_col = "income"   # assume column name is income (<=50K / >50K)
cat_cols = [col for col in df.columns if df[col].dtype == "object" and col != label_col]
cont_cols = [col for col in df.columns if col not in cat_cols + [label_col]]

# ============================================
# 4️⃣ Encode categorical + label
# ============================================
label_enc = LabelEncoder()
df[label_col] = label_enc.fit_transform(df[label_col])  # 0/1

cat_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    cat_encoders[col] = le

# ============================================
# 5️⃣ Split train/test
# ============================================
train_df, test_df = train_test_split(df, train_size=25000, test_size=5000, random_state=SEED)

# categorical tensors
cat_train = torch.tensor(train_df[cat_cols].values, dtype=torch.long)
cat_test  = torch.tensor(test_df[cat_cols].values,  dtype=torch.long)

# continuous tensors (scaled)
scaler = StandardScaler()
cont_train = torch.tensor(scaler.fit_transform(train_df[cont_cols].values), dtype=torch.float)
cont_test  = torch.tensor(scaler.transform(test_df[cont_cols].values), dtype=torch.float)

# labels
y_train = torch.tensor(train_df[label_col].values, dtype=torch.long)
y_test  = torch.tensor(test_df[label_col].values, dtype=torch.long)

train_ds = TensorDataset(cat_train, cont_train, y_train)
test_ds  = TensorDataset(cat_test,  cont_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64)

# ============================================
# 6️⃣ Model
# ============================================
# Embedding sizes
embeddings = []
for col in cat_cols:
    n_unique = df[col].nunique()
    emb_size = min(50, (n_unique + 1)//2)
    embeddings.append((n_unique, emb_size))

class TabularModel(nn.Module):
    def __init__(self, emb_dims, n_cont):
        super().__init__()
        self.emb_layers = nn.ModuleList([nn.Embedding(in_size, out_size) for in_size, out_size in emb_dims])
        self.emb_drop = nn.Dropout(0.4)
        self.bn_cont  = nn.BatchNorm1d(n_cont)

        self.layer1 = nn.Linear(sum(out_size for _, out_size in emb_dims) + n_cont, 50)
        self.bn1    = nn.BatchNorm1d(50)
        self.drop1  = nn.Dropout(0.4)
        self.out    = nn.Linear(50, 2)

    def forward(self, x_cat, x_cont):
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
        x = torch.cat(x, dim=1)
        x = self.emb_drop(x)

        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], dim=1)

        x = F.relu(self.bn1(self.layer1(x)))
        x = self.drop1(x)
        return self.out(x)

model = TabularModel(embeddings, len(cont_cols))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ============================================
# 7️⃣ Training
# ============================================
for epoch in range(300):
    model.train()
    total_loss = 0
    for cat_batch, cont_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(cat_batch, cont_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/300  Loss: {total_loss/len(train_loader):.4f}")

# ============================================
# 8️⃣ Evaluation
# ============================================
model.eval()
test_loss, correct, total = 0, 0, 0
with torch.no_grad():
    for cat_batch, cont_batch, y_batch in test_loader:
        preds = model(cat_batch, cont_batch)
        loss = criterion(preds, y_batch)
        test_loss += loss.item()
        _, predicted = torch.max(preds, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

print(f"Test Loss: {test_loss/len(test_loader):.4f}")
print(f"Test Accuracy: {correct/total:.4f}")

# ============================================
# 9️⃣ BONUS: Predict function
# ============================================
def predict_new(sample_dict):
    """
    sample_dict = { 'marital-status': 'Never-married', 'education': 'Bachelors', ... }
    """
    cat_vals = []
    cont_vals = []
    for c in cat_cols:
        val = cat_encoders[c].transform([sample_dict[c]])[0]
        cat_vals.append(val)
    for c in cont_cols:
        val = (sample_dict[c] - scaler.mean_[cont_cols.index(c)]) / scaler.scale_[cont_cols.index(c)]
        cont_vals.append(val)

    x_cat = torch.tensor([cat_vals], dtype=torch.long)
    x_cont = torch.tensor([cont_vals], dtype=torch.float)

    model.eval()
    with torch.no_grad():
        pred = torch.softmax(model(x_cat, x_cont), dim=1)
    return { "<=50K": float(pred[0][0]), ">50K": float(pred[0][1]) }
```

### OUTPUT :

<img width="582" height="323" alt="image" src="https://github.com/user-attachments/assets/4c99badd-5506-463e-b324-273f75a108b9" />
<img width="673" height="432" alt="image" src="https://github.com/user-attachments/assets/0b926511-2bea-4b63-8b66-ab404863acc4" />

