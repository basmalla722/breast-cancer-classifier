# Breast Cancer Classification using Machine Learning

Detecting whether a breast tumour is **benign** or **malignant** from 30 clinical measurements, built with Python and scikit-learn as part of my transition from web development into machine learning.

**Result: 98.25% test accuracy** on a model that is simple enough to explain and strong enough to trust.

## Why this project

Web development was my starting point, but I wanted to work with **data** rather than only interfaces — so I took a real clinical dataset, explored it, trained classification models, and measured how well they actually perform. This is my first hands-on end-to-end machine learning pipeline.

## The dataset

- **Source:** Wisconsin Breast Cancer (Diagnostic) dataset, available through `scikit-learn`
- **Samples:** 569 · **Features:** 30 (radius, texture, perimeter, area, compactness, concavity, symmetry, fractal dimension, …)
- **Target:** `0` = benign (212 samples) · `1` = malignant (357 samples)
- **Missing values:** none

## Approach

1. **Explore** the data with `pandas` — class distribution, descriptive statistics, and a feature correlation heatmap to find features that carry redundant information
2. **Split** into 80% training / 20% test set using `stratify`, so the 37/63 class ratio stays identical in both sets
3. **Scale** features with `StandardScaler` — `mean area` reaches 2501 while `mean smoothness` stays near 0.09, so unscaled data would let a handful of large-magnitude features dominate the model
4. **Train and compare** four classifiers
5. **Evaluate** with accuracy, a classification report (precision / recall / F1) and a confusion matrix

## Results

**Logistic Regression — 98.25% accuracy**

```
              precision    recall   f1-score   support
      Benign       0.98      0.98      0.98        42
   Malignant       0.99      0.99      0.99        72
    accuracy                           0.98       114
```

**Confusion matrix**

```
                 predicted
                 Benign  Malignant
actual Benign        41          1
actual Malignant      1         71
```

Of 114 test samples the model got **113 right**. The two errors are one in each direction — one benign case flagged as malignant and one malignant case missed. In a screening context that trade-off matters: a missed malignant case is more serious than a false alarm, which is exactly why recall and the confusion matrix deserve more attention than a single accuracy number.

## Model comparison

| Model | Accuracy |
|---|---|
| Logistic Regression | **98.25%** |
| Support Vector Machine (RBF) | **98.25%** |
| Random Forest | 95.61% |
| K-Nearest Neighbours (k=5) | 95.61% |

The finding I did not expect: the two simpler models tied at the top, and Random Forest — usually the strongest default — came last. The gap is small, but the direction is informative. On a small, well-separated dataset like this one, a regularised linear boundary and an SVM with a smooth margin are enough, while a fully grown forest fits noise in the training set. The model with the most capacity was not the model that generalised best.

## Which features drive the prediction

| Rank | Feature | Coefficient |
|---|---|---|
| 1 | mean compactness | 0.648 |
| 2 | compactness error | 0.647 |
| 3 | fractal dimension error | 0.438 |
| 4 | symmetry error | 0.360 |
| 5 | texture error | 0.249 |

All top features are *compactness*, *irregularity* and *error* measures rather than simple size measures — the model is picking up how irregular a cell boundary looks, not how large the tumour is. That matches what the clinical literature on this dataset reports, which is a useful check that the model learned something real rather than something spurious.

## What I learned

- Why feature scaling matters when features live on completely different scales
- Reading a confusion matrix instead of trusting accuracy alone, and which error direction actually costs more
- Turning raw coefficients into an interpretable feature ranking instead of treating the model as a black box
- Using `stratify` so a class-imbalanced split doesn't silently corrupt the evaluation
- That higher model complexity does not automatically mean better generalisation

## How to run

```bash
pip install -r requirements.txt
python train.py
```

## Files

| File | Description |
|---|---|
| `train.py` | Full pipeline: load → explore → split → scale → train → evaluate → compare |
| `01_correlation_heatmap.png` | Feature correlation matrix |
| `02_confusion_matrix.png` | Where the model succeeds and fails |
| `03_feature_importance.png` | Top 10 features by influence |
| `04_model_comparison.png` | Accuracy across the four models |
