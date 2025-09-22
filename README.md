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
