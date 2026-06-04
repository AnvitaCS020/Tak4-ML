Breast Cancer Classification — Logistic Regression
Project Structure

```
breast_cancer_classifier/
│
├── data.csv                          # Breast Cancer Wisconsin dataset
├── logistic_regression_classifier.py # Main classifier script
├── results.png                        # All 6 evaluation plots (auto-generated)
└── README.md
```

---

Objective

Predict whether a tumour is **Malignant (M)** or **Benign (B)** using 30 numeric features derived from a digitized image of a fine needle aspirate (FNA) of a breast mass.

| Label | Class     | Count |
|-------|-----------|-------|
| 1     | Malignant | 212   |
| 0     | Benign    | 357   |

---

What is Logistic Regression?

Logistic Regression is a **linear classifier** that models the probability of a binary outcome using the **sigmoid function**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

where `z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ` is the weighted sum of input features.

- Output is always between **0 and 1** → interpreted as a probability
- If `σ(z) ≥ 0.5` → predict **Malignant (1)**
- If `σ(z) < 0.5`  → predict **Benign (0)**

The model is trained by minimising **Binary Cross-Entropy Loss** using the L-BFGS optimiser.

---

Pipeline Steps

1. Load & Clean Data
```python
df = pd.read_csv("data.csv")
df.drop(columns=["id", "Unnamed: 32"], inplace=True)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
```

2. Train / Test Split (80 / 20, stratified)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```
Stratified split ensures both classes appear proportionally in train and test.

3. Feature Standardization
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```
The scaler is **fit on the training set only** to prevent data leakage.

4. Fit Logistic Regression
```python
model = LogisticRegression(max_iter=10000, random_state=42, solver="lbfgs")
model.fit(X_train_scaled, y_train)
```

5. Evaluate the Model

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 96.49% |
| Precision | 97.50% |
| Recall    | 92.86% |
| F1-Score  | 95.12% |
| ROC-AUC   | 99.60% |

6. Threshold Tuning

The default threshold of 0.5 can be adjusted based on the use case:

| Threshold | Precision | Recall | F1     |
|-----------|-----------|--------|--------|
| 0.3       | 0.9762    | 0.9762 | 0.9762 |
| 0.5       | 0.9750    | 0.9286 | 0.9512 |
| 0.7       | 1.0000    | 0.9048 | 0.9500 |

> In a medical context, **lower threshold (e.g. 0.3)** is preferred to maximise Recall — catching more true Malignant cases at the cost of slightly more false alarms.

---

Visualisations (`results.png`)

| Plot | Description |
|------|-------------|
| **Confusion Matrix** | TP / TN / FP / FN counts at threshold 0.5 |
| **ROC Curve** | Trade-off between TPR and FPR across all thresholds |
| **Sigmoid Function** | Visual explanation of the core logistic function |
| **Precision / Recall / F1 vs Threshold** | Effect of changing the decision boundary |
| **Top 15 Feature Coefficients** | Most influential features (positive = ↑ Malignant risk) |
| **Probability Distribution** | Separation of predicted probabilities by actual class |

---
How to Run

Prerequisites
```bash
pip install numpy pandas matplotlib scikit-learn
```

Execute
```bash
python logistic_regression_classifier.py
```

Output is printed to the console and `results.png` is saved in the same directory.

---

## 📦 Dependencies

| Library      | Purpose                                |
|--------------|----------------------------------------|
| pandas       | Data loading and manipulation          |
| numpy        | Numerical operations                   |
| scikit-learn | Model training, scaling, metrics       |
| matplotlib   | Plotting all evaluation visualisations |

---

Key Concepts Covered

- ✅ Binary classification with Logistic Regression
- ✅ Stratified train/test split
- ✅ Feature standardisation (StandardScaler)
- ✅ Sigmoid function explanation
- ✅ Confusion matrix, Precision, Recall, F1-Score
- ✅ ROC-AUC curve
- ✅ Decision threshold tuning
- ✅ Feature importance via model coefficients

---

Dataset

**Breast Cancer Wisconsin (Diagnostic) Dataset**  
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)  
Features: 30 real-valued features computed from digitised FNA images  
Samples: 569 (357 Benign, 212 Malignant)
