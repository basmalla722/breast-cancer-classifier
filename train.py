# ============================================================
#  BREAST CANCER CLASSIFIER  —  Machine Learning project
#  Author: Basmalla Nabil
#  What it does: predicts whether a breast tumour is benign
#  (غير خبيث) or malignant (خبيث) from 30 measured features.
#  Run:  pip install -r requirements.txt   then   python train.py
# ============================================================

# ---------- 1. imports ----------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set_theme(style="whitegrid")

# ---------- 2. load the data ----------
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# 0 = benign (غير خبيث)  |  1 = malignant (خبيث)
class_names = ["Benign", "Malignant"]

print("Dataset shape:", X.shape)
print("Class distribution:")
print(y.value_counts())

# ---------- 3. explore the data ----------
print("\n--- first 5 rows ---")
print(X.head())

print("\n--- missing values ---")
print(X.isnull().sum().sum())

print("\n--- basic statistics of the first 5 features ---")
print(X.iloc[:, :5].describe())

# correlation heatmap of a subset of features
plt.figure(figsize=(10, 7))
sns.heatmap(X.corr().iloc[:12, :12], annot=True, fmt=".1f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("01_correlation_heatmap.png", dpi=150)
plt.close("all")

# ---------- 4. split into train / test ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 5. scale the features ----------
# (needed because features have very different units, e.g. area vs radius)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 6. train the model ----------
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)

# ---------- 7. evaluate ----------
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n=========== RESULTS ===========")
print("Model: Logistic Regression")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("02_confusion_matrix.png", dpi=150)
plt.close("all")

# ---------- 8. which features matter most? ----------
coefficients = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)

print("\n--- top 10 most important features ---")
print(coefficients.head(10))

plt.figure(figsize=(9, 6))
sns.barplot(x=coefficients.head(10).values, y=coefficients.head(10).index, color="#1155cc")
plt.title("Top 10 Features Influencing the Prediction")
plt.xlabel("Coefficient (influence on prediction)")
plt.tight_layout()
plt.savefig("03_feature_importance.png", dpi=150)
plt.close("all")

# ---------- 9. compare against 2 other models ----------
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", random_state=42),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5),
}

print("\n=========== MODEL COMPARISON ===========")
results = {}
for name, clf in models.items():
    clf.fit(X_train_scaled, y_train)
    score = accuracy_score(y_test, clf.predict(X_test_scaled))
    results[name] = round(score * 100, 2)
    print(f"{name}: {round(score * 100, 2)}%")

plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.values()), y=list(results.keys()), color="#0f9d58")
plt.xlabel("Accuracy (%)")
plt.xlim(80, 100)
plt.title("Model Comparison")
plt.tight_layout()
plt.savefig("04_model_comparison.png", dpi=150)
plt.close("all")
