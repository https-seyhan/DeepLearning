# ===== 1. Imports =====
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
import lightgbm as lgb

# ===== 2. Create synthetic dataset =====
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=8,
    n_clusters_per_class=1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ===== 3. Helper function for benchmarking =====
def benchmark_model(name, model, fit_func=None, predict_func=None):
    results = {}
    print(f"\n=== {name} ===")
    
    # Timing training
    start_train = time.time()
    if fit_func:
        fit_func(model)
    else:
        model.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    # Timing prediction
    start_pred = time.time()
    if predict_func:
        y_pred = predict_func(model)
    else:
        y_pred = model.predict(X_test)
    pred_time = time.time() - start_pred
    
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Training time: {train_time:.3f} sec")
    print(f"Prediction time: {pred_time:.3f} sec")
    print(classification_report(y_test, y_pred))
    
    results['name'] = name
    results['accuracy'] = acc
    results['train_time'] = train_time
    results['pred_time'] = pred_time
    results['y_pred'] = y_pred
    return results

# ===== 4. Models =====
results_all = []

# RandomForest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
results_all.append(benchmark_model("RandomForest", rf_model))

# Logistic Regression (multinomial)
lr_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000)
results_all.append(benchmark_model("LogisticRegression", lr_model))

# XGBoost
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=8,
    eval_metric='mlogloss',
    random_state=42
)
results_all.append(benchmark_model("XGBoost", xgb_model))

# LightGBM
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=8,
    random_state=42
)
results_all.append(benchmark_model("LightGBM", lgb_model))

# TensorFlow Neural Net
def fit_nn(model):
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
def predict_nn(model):
    y_probs = model.predict(X_test, verbose=0)
    return np.argmax(y_probs, axis=1)

nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(8, activation='softmax')
])
nn_model.compile(optimizer=Adam(0.001),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
results_all.append(benchmark_model("NeuralNet (TensorFlow)", nn_model, fit_func=fit_nn, predict_func=predict_nn))

# ===== 5. Plot confusion matrices =====
fig, axes = plt.subplots(1, len(results_all), figsize=(20, 4))
for i, res in enumerate(results_all):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=False, cmap="Blues", ax=axes[i])
    axes[i].set_title(res['name'])
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("True")
plt.tight_layout()
plt.show()

# ===== 6. Summary table =====
summary_df = pd.DataFrame(results_all)[['name', 'accuracy', 'train_time', 'pred_time']]
print("\n=== Benchmark Summary ===")
print(summary_df.sort_values(by="accuracy", ascending=False))
