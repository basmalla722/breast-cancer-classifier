# Breast Cancer Classification

Predicting if a breast tumour is benign or malignant, using the Wisconsin Breast Cancer dataset.

This was my first machine learning project. I wanted to try something different from the web
apps I usually build, so I worked on a real medical dataset instead of a tutorial one.

**Accuracy: 98.25%** (Logistic Regression)

## Dataset

- 569 samples, 30 features
- Target: 0 = benign (212), 1 = malignant (357)
- No missing values
- From `scikit-learn` (`load_breast_cancer`)

## What I did

1. Looked at the data with pandas — class counts, statistics, and a correlation heatmap
2. Split it 80/20 with `stratify` so both sets keep the same class ratio
3. Scaled the features with `StandardScaler`
4. Trained 4 models and compared them
5. Checked the result with a classification report and a confusion matrix

### Why scaling

`mean area` goes up to 2501, but `mean smoothness` is around 0.09. Without scaling the
bigger numbers would just dominate the model, so I put everything on the same scale first.

## Results

**Logistic Regression — 98.25%**

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
actual Benign         41          1
actual Malignant       1         71
```

113 out of 114 correct. One benign case was called malignant, and one malignant case was
missed.

## Model comparison

Accuracy on its own is a weak way to rank models here: the test set is 114 rows,
so one sample is worth 0.88%, and it is 42 benign against 72 malignant, which
means a model can miss malignancies and still look fine. So the table reports
precision, recall and F1 against the malignant class, plus 5-fold
cross-validation next to the holdout score.

| Model | Accuracy | Precision | Recall | F1 | 5-fold CV |
|---|---|---|---|---|---|
| Logistic Regression | 98.25% | 98.61% | 98.61% | 98.61% | **98.07% ± 0.65** |
| Support Vector Machine | 98.25% | 98.61% | 98.61% | 98.61% | 97.36% ± 1.47 |
| K-Nearest Neighbours | 95.61% | 95.89% | 97.22% | 96.55% | 96.49% ± 0.96 |
| Random Forest | 95.61% | 95.89% | 97.22% | 96.55% | 95.61% ± 2.28 |

**Logistic Regression is the one I kept**, and the cross-validation is why. On the
holdout set it ties with the SVM at 98.25%, so that column alone does not separate
them. Across folds the SVM drops to 97.36% with a spread of 1.47, while Logistic
Regression holds at 98.07% with a spread of 0.65. The simpler model is also the
more consistent one here, which is what I would want if this were real.

Random Forest came last, which I did not expect. The dataset is small and fairly
clean, so the simpler models did fine and the forest probably overfitted — note
its ± 2.28 spread, the widest of the four.

## Keeping the number honest

`test_model.py` asserts the model still clears a quality floor, so a refactor that
quietly breaks the split, the scaling or the model itself fails instead of
producing a plausible-looking number:

```bash
python test_model.py
```

The floors sit below what is observed on purpose. Accuracy and F1 are held at 0.95
and malignant recall at 0.97, which tolerates the single missed malignancy but not
two. Recall is asserted separately from accuracy on purpose, because a model can
pass on accuracy while missing the class that actually costs something.

## Most important features

| Feature | Coefficient |
|---|---|
| mean compactness | 0.648 |
| compactness error | 0.647 |
| fractal dimension error | 0.438 |
| symmetry error | 0.360 |
| texture error | 0.249 |

The top features are all about how irregular the cell boundary is, not how big the tumour
is. That matches what I found reading about this dataset.

## Run it

```bash
pip install -r requirements.txt
python train.py
```

## Files

| File | What it shows |
|---|---|
| `train.py` | The whole pipeline |
| `test_model.py` | Quality guards that fail if accuracy drops |
| `.github/workflows/ci.yml` | Runs the guards and the pipeline on every push |
| `01_correlation_heatmap.png` | Which features relate to each other |
| `02_confusion_matrix.png` | Where the model gets it wrong |
| `03_feature_importance.png` | Top 10 features |
| `04_model_comparison.png` | Holdout vs cross-validated accuracy per model |
