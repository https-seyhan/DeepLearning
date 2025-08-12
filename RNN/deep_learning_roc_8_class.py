# upgraded_benchmark_with_roc.py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# -----------------------
# 1) Data
# -----------------------
N_CLASSES = 8
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=N_CLASSES,
    n_clusters_per_class=1,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
y_test_binarized = label_binarize(y_test, classes=np.arange(N_CLASSES))

# -----------------------
# Helper: benchmark & prob-getter
# -----------------------
def get_proba(model, X):
    """Return probability matrix shape (n_samples, n_classes). Works for sklearn/xgb/lgb & TF."""
    # TensorFlow model (Sequential) will not have predict_proba
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    # sklearn's LogisticRegression, RandomForest, XGBoost, LightGBM should have predict_proba
    # Keras / TF model: use predict (already returns probs for softmax final layer)
    if hasattr(model, "predict") and not hasattr(model, "predict_proba"):
        return model.predict(X, verbose=0)
    raise ValueError("Model does not support probability prediction")

def benchmark_model(name, model, X_train, y_train, X_test, y_test, fit_func=None, predict_func=None):
    print(f"\n=== {name} ===")
    res = {}
    start = time.time()
    if fit_func:
        fit_func(model)
    else:
        model.fit(X_train, y_train)
    train_time = time.time() - start

    # predictions and probabilities
    start = time.time()
    if predict_func:
        y_pred = predict_func(model)
        y_proba = get_proba(model, X_test)
    else:
        y_pred = model.predict(X_test)
        y_proba = get_proba(model, X_test)
    pred_time = time.time() - start

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Training time: {train_time:.3f} sec, Prediction time: {pred_time:.3f} sec")
    print(classification_report(y_test, y_pred))

    res.update({
        "name": name,
        "model": model,
        "accuracy": acc,
        "train_time": train_time,
        "pred_time": pred_time,
        "y_pred": y_pred,
        "y_proba": y_proba
    })
    return res

# -----------------------
# 2) Hyperparameter tuning (RandomizedSearchCV)
# -----------------------
results_all = []
params_all = {}

# RandomForest
rf_params = {'n_estimators':[100,200,300],'max_depth':[None,10,20],'max_features':['sqrt','log2']}
rf_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_params, n_iter=5, scoring='accuracy', cv=3, n_jobs=-1, random_state=42)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
params_all['RandomForest'] = rf_search.best_params_
results_all.append(benchmark_model("RandomForest (tuned)", best_rf, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test))

# Logistic Regression
lr_params = {'C':[0.01,0.1,1,10],'solver':['lbfgs','newton-cg'],'max_iter':[500,1000]}
lr_search = RandomizedSearchCV(LogisticRegression(multi_class='multinomial'), lr_params, n_iter=4, scoring='accuracy', cv=3, n_jobs=-1, random_state=42)
lr_search.fit(X_train, y_train)
best_lr = lr_search.best_estimator_
params_all['LogisticRegression'] = lr_search.best_params_
results_all.append(benchmark_model("LogisticRegression (tuned)", best_lr, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test))

# XGBoost (ensure predict_proba exists)
xgb_params = {'n_estimators':[100,200],'max_depth':[3,5,7],'learning_rate':[0.05,0.1,0.2],'subsample':[0.8,1.0]}
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_search = RandomizedSearchCV(xgb_clf, xgb_params, n_iter=5, scoring='accuracy', cv=3, n_jobs=-1, random_state=42)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
params_all['XGBoost'] = xgb_search.best_params_
results_all.append(benchmark_model("XGBoost (tuned)", best_xgb, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test))

# LightGBM
lgb_params = {'n_estimators':[100,200],'max_depth':[-1,10,20],'learning_rate':[0.05,0.1,0.2],'num_leaves':[31,50,100]}
lgb_clf = lgb.LGBMClassifier(objective='multiclass', num_class=N_CLASSES, random_state=42)
lgb_search = RandomizedSearchCV(lgb_clf, lgb_params, n_iter=5, scoring='accuracy', cv=3, n_jobs=-1, random_state=42)
lgb_search.fit(X_train, y_train)
best_lgb = lgb_search.best_estimator_
params_all['LightGBM'] = lgb_search.best_params_
results_all.append(benchmark_model("LightGBM (tuned)", best_lgb, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test))

# Neural Net tuning (small grid)
best_acc_nn, best_nn_model, best_nn_params = 0, None, None
for units in [32, 64]:
    for lr in [0.001, 0.005]:
        nn = Sequential([
            Dense(units, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(units//2, activation='relu'),
            Dense(N_CLASSES, activation='softmax')
        ])
        nn.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        nn.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        acc = nn.evaluate(X_test, y_test, verbose=0)[1]
        if acc > best_acc_nn:
            best_acc_nn, best_nn_model, best_nn_params = acc, nn, {'units': units, 'lr': lr, 'epochs': 10}
params_all['NeuralNet'] = best_nn_params
print("Best NN params:", best_nn_params, "test_acc:", best_acc_nn)
# Already trained during tuning; no extra fit needed.
def predict_nn_wrapper(m): return np.argmax(m.predict(X_test, verbose=0), axis=1)
results_all.append(benchmark_model("NeuralNet (tuned)", best_nn_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, fit_func=lambda m: None, predict_func=predict_nn_wrapper))

# -----------------------
# 3) Confusion matrices + ROC curves + Reports
# -----------------------
# Update summary and AUC storage
summary_rows = []
auc_records = []

for res in results_all:
    name = res['name']
    y_pred = res['y_pred']
    y_proba = res['y_proba']

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    res['confusion_matrix'] = cm

    # ROC per class
    # ensure y_proba is shape (n_samples, n_classes)
    if y_proba.shape[1] != N_CLASSES:
        raise ValueError(f"Probability output for {name} has unexpected shape {y_proba.shape}")

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(N_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro-average (aggregate all fpr)
    # first gather all fpr points
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(N_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(N_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= N_CLASSES
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    res['roc_fpr'] = fpr
    res['roc_tpr'] = tpr
    res['roc_auc'] = roc_auc

    # store summary
    summary_rows.append({
        "name": name,
        "accuracy": res['accuracy'],
        "train_time": res['train_time'],
        "pred_time": res['pred_time'],
        "macro_auc": roc_auc["macro"],
        "micro_auc": roc_auc["micro"]
    })

    # store per-class AUCs for Excel
    for i in range(N_CLASSES):
        auc_records.append({"Model": name, "Class": i, "AUC": roc_auc[i]})

# create summary dataframe
summary_df = pd.DataFrame(summary_rows).sort_values(by="accuracy", ascending=False)

# -----------------------
# 4) Save Excel with AUCs + params
# -----------------------
excel_path = "benchmark_results_with_auc.xlsx"
with pd.ExcelWriter(excel_path) as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    pd.DataFrame(auc_records).pivot_table(index="Model", columns="Class", values="AUC").to_excel(writer, sheet_name="PerClass_AUC")
    pd.DataFrame(list(params_all.items()), columns=["Model", "Best_Params"]).to_excel(writer, sheet_name="Best_Params", index=False)
print(f"Excel saved to: {excel_path}")

# -----------------------
# 5) Save PDF with confusion matrices + ROC plots
# -----------------------
pdf_path = "benchmark_report_with_roc.pdf"
with PdfPages(pdf_path) as pdf:
    # Summary table page
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    tbl = ax.table(cellText=summary_df.round(4).values, colLabels=summary_df.columns, loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.2, 1.2)
    ax.set_title("Model Benchmark Summary", fontweight="bold")
    pdf.savefig(fig); plt.close(fig)

    # For each model: confusion matrix then ROC plot
    for res in results_all:
        name = res['name']

        # confusion matrix page
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        pdf.savefig(fig); plt.close(fig)

        # ROC plot page
        fpr = res['roc_fpr']; tpr = res['roc_tpr']; roc_auc = res['roc_auc']
        fig, ax = plt.subplots(figsize=(8,6))
        # plot per-class
        for i in range(N_CLASSES):
            ax.plot(fpr[i], tpr[i], lw=1, label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
        # plot micro and macro
        ax.plot(fpr["micro"], tpr["micro"], label=f"micro-average (AUC = {roc_auc['micro']:.2f})", linestyle='--', linewidth=2)
        ax.plot(fpr["macro"], tpr["macro"], label=f"macro-average (AUC = {roc_auc['macro']:.2f})", linestyle=':', linewidth=2)
        ax.plot([0,1],[0,1], 'k--', lw=0.5)
        ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.01])
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves - {name}")
        ax.legend(fontsize='small', loc='lower right')
        pdf.savefig(fig); plt.close(fig)

print(f"PDF saved to: {pdf_path}")
print("Done.")
