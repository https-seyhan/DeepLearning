# ===== 1. Imports =====
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ===== 2. Create synthetic dataset =====
# 1000 samples, 20 features, 8 classes, balanced
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=8,
    n_clusters_per_class=1,
    random_state=42
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ===== 3. Train model =====
model = RandomForestClassifier(
    n_estimators=200,        # number of trees
    max_depth=None,         # grow trees fully
    random_state=42
)
model.fit(X_train, y_train)

# ===== 4. Evaluate =====
y_pred = model.predict(X_test)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ===== 5. Example predictions =====
sample_input = X_test[:5]
sample_preds = model.predict(sample_input)
print("\nSample Predictions vs Actual:")
for i in range(len(sample_input)):
    print(f"Predicted: {sample_preds[i]}, Actual: {y_test[i]}")
