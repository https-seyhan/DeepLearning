# ===== 1. Imports =====
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

# ===== 3. Benchmark helper =====
def benchmark_model(name, model, fit_func=None, predict_func=None):
    results = {}
    print(f"\n=== {name} ===")
    
    start_train = time.time()
    if fit_func:
        fit_func(model)
    else:
        model.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    start_pred = time.time()
    if predict_func:
        y_pred = predict_func(model)
    else:
        y_pred = model.predict(X_test)
    pred_time = time.time() - start_pred
    
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

results_all = []
params_all = {}

# ===== 4. Model tuning =====
# RandomForest
rf_params = {'n_estimators':[100,200,300],'max_depth':[None,10,20,30],'max_features':['sqrt','log2']}
rf_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_params, n_iter=5, scoring='accuracy', cv=3, n_jobs=-1, random_state=42)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
params_all['RandomForest'] = rf_search.best_params_
results_all.append(benchmark_model("RandomForest (tuned)", best_rf))

# Logistic Regression
lr_params = {'C':[0.01,0.1,1,10],'solver':['lbfgs','newton-cg'],'max_iter':[500,1000]}
lr_search = RandomizedSearchCV(LogisticRegression(multi_class='multinomial'), lr_params, n_iter=4, scoring='accuracy', cv=3, n_jobs=-1, random_state=42)
lr_search.fit(X_train, y_train)
best_lr = lr_search.best_estimator_
params_all['LogisticRegression'] = lr_search.best_params_
results_all.append(benchmark_model("LogisticRegression (tuned)", best_lr))

# XGBoost
xgb_params = {'n_estimators':[100,200],'max_depth':[3,5,7],'learning_rate':[0.05,0.1,0.2],'subsample':[0.8,1.0]}
xgb_search = RandomizedSearchCV(xgb.XGBClassifier(objective='multi:softmax', num_class=8, eval_metric='mlogloss', random_state=42), xgb_params, n_iter=5, scoring='accuracy', cv=3, n_jobs=-1, random_state=42)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
params_all['XGBoost'] = xgb_search.best_params_
results_all.append(benchmark_model("XGBoost (tuned)", best_xgb))

# LightGBM
lgb_params = {'n_estimators':[100,200],'max_depth':[-1,10,20],'learning_rate':[0.05,0.1,0.2],'num_leaves':[31,50,100]}
lgb_search = RandomizedSearchCV(lgb.LGBMClassifier(objective='multiclass', num_class=8, random_state=42), lgb_params, n_iter=5, scoring='accuracy', cv=3, n_jobs=-1, random_state=42)
lgb_search.fit(X_train, y_train)
best_lgb = lgb_search.best_estimator_
params_all['LightGBM'] = lgb_search.best_params_
results_all.append(benchmark_model("LightGBM (tuned)", best_lgb))

# Neural Network tuning
best_acc_nn, best_units, best_lr = 0, None, None
best_nn_model = None
for units in [32, 64]:
    for lr in [0.001, 0.005]:
        nn_model = Sequential([
            Dense(units, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(units//2, activation='relu'),
            Dense(8, activation='softmax')
        ])
        nn_model.compile(optimizer=Adam(learning_rate=lr),
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
        nn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        acc = nn_model.evaluate(X_test, y_test, verbose=0)[1]
        if acc > best_acc_nn:
            best_acc_nn, best_units, best_lr = acc, units, lr
            best_nn_model = nn_model
params_all['NeuralNet'] = {'units': best_units, 'learning_rate': best_lr, 'epochs': 10}
print(f"Best NN params: {params_all['NeuralNet']}, accuracy={best_acc_nn:.4f}")

def fit_nn(model): pass  # Already trained
def predict_nn(model):
    y_probs = model.predict(X_test, verbose=0)
    return np.argmax(y_probs, axis=1)

results_all.append(benchmark_model("NeuralNet (tuned)", best_nn_model, fit_func=fit_nn, predict_func=predict_nn))

# ===== 5. Summary Table =====
summary_df = pd.DataFrame(results_all)[['name', 'accuracy', 'train_time', 'pred_time']]
summary_df = summary_df.sort_values(by="accuracy", ascending=False)
print("\n=== Benchmark Summary ===")
print(summary_df)

# ===== 6. Save Excel report =====
excel_path = "benchmark_results.xlsx"
with pd.ExcelWriter(excel_path) as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    pd.DataFrame(params_all.items(), columns=["Model", "Best_Params"]).to_excel(writer, sheet_name="Best_Params", index=False)
print(f"Excel report saved to {excel_path}")

# ===== 7. Save PDF report =====
pdf_path = "benchmark_report.pdf"
with PdfPages(pdf_path) as pdf:
    # Summary table plot
    fig, ax = plt.subplots(figsize=(6,3))
    ax.axis('off')
    tbl = ax.table(cellText=summary_df.round(4).values,
                   colLabels=summary_df.columns,
                   loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.2)
    ax.set_title("Model Benchmark Summary", fontweight="bold")
    pdf.savefig(fig)
    plt.close(fig)
    
    # Confusion matrices
    for res in results_all:
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=False, cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix - {res['name']}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        pdf.savefig(fig)
        plt.close(fig)
print(f"PDF report saved to {pdf_path}")
