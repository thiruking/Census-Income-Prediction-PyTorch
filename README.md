# Binary Classification with Neural Networks on the Census Income Dataset

## Tabular Income Classifier – Neural Network

This project trains a **PyTorch neural network** to guess whether a person earns **more than \$50K** per year based on simple demographic details.

---

## What the Code Does

* **Loads the data** from `income.csv`
* Splits columns into  
  * Categorical: `sex`, `education`, `marital-status`, `workclass`, `occupation`  
  * Continuous: `age`, `hours-per-week`
* Turns each category into a **numeric code** and builds **embedding layers** for them.
* Normalizes the continuous numbers.
* Creates a small neural network:
  * Embeddings → ReLU layers → Dropout → Final 2-class output.
* Trains for about **300 epochs** using Adam + CrossEntropyLoss.
* Shows a training-loss plot and reports test accuracy.
* Lets you **predict a single person’s income group** with an interactive prompt.

---

## Setup

1. **Python 3.9+** recommended.  
2. Install packages:

   ```bash
   pip install torch pandas numpy matplotlib scikit-learn

3. Put your income.csv in the same folder as the script.

## Notes

1. Uses GPU automatically if available.

2. Unknown category inputs default to code 0 and print a warning.

3. Training time depends on your CPU/GPU.