import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================================================
# Parameter settings
# ===============================================================
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
FULL_DATASET_PATH = os.path.join(PROJECT_ROOT, "Research_results/step5.2_build_chirality_dataset/full_dataset.pkl")
SAVE_DIR = os.path.join(PROJECT_ROOT, "Research_results/step5.4_chirality_model_benchmark_cv")
TARGET_COORD_DIM = 1000
N_SPLITS = 5
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================================================
# Utility functions
# ===============================================================
def pad_or_truncate(array, target_len):
    if len(array) >= target_len:
        return array[:target_len]
    else:
        return np.pad(array, (0, target_len - len(array)), mode='constant')

# ===============================================================
# Data loading and processing
# ===============================================================
with open(FULL_DATASET_PATH, 'rb') as f:
    dataset = pickle.load(f)

X_coords_only, X_with_tda, X_tda_only, y = [], [], [], []
for sample in dataset:
    if sample['coords'] is not None:
        coords_flat = pad_or_truncate(sample['coords'].flatten(), TARGET_COORD_DIM)
        tda = sample['tda']

        if tda is not None:
            full_vec = np.concatenate([coords_flat, tda])
            X_with_tda.append(full_vec)
            X_tda_only.append(tda)
        else:
            X_with_tda.append(None)
            X_tda_only.append(None)

        X_coords_only.append(coords_flat)
        y.append(sample['label'])

Xc = np.array(X_coords_only)
y = np.array(y)

Xct, Xt, y_ct, y_t = [], [], [], []
for i in range(len(X_with_tda)):
    if X_with_tda[i] is not None:
        Xct.append(X_with_tda[i])
        Xt.append(X_tda_only[i])
        y_ct.append(y[i])
        y_t.append(y[i])

Xct = np.array(Xct)
Xt = np.array(Xt)
y_ct = np.array(y_ct)
y_t = np.array(y_t)

# ===============================================================
# Models
# ===============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# ===============================================================
# Cross-validation evaluation function
# ===============================================================
def cross_validate(name, model, X, y, results_list):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    accs, aucs = [], []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = round(accuracy_score(y_test, preds), 3)
        auc = round(roc_auc_score(y_test, probs), 3) if probs is not None else None

        accs.append(acc)
        aucs.append(auc)

    result = {
        "Model": name,
        "Accuracy Mean": round(np.mean(accs), 3),
        "Accuracy Std": round(np.std(accs), 3),
        "AUC Mean": round(np.mean(aucs), 3),
        "AUC Std": round(np.std(aucs), 3)
    }
    results_list.append(result)
    print(f"✅ {name} - ACC: {result['Accuracy Mean']}±{result['Accuracy Std']}, AUC: {result['AUC Mean']}±{result['AUC Std']}")

# ===============================================================
# Model Evaluation
# ===============================================================
results = []

print("\n🔹 3D Coordinates Only")
for name, model in models.items():
    cross_validate(name + " (Coord)", model, Xc, y, results)

print("\n🔹 Coordinates + TDA")
for name, model in models.items():
    cross_validate(name + " (Coord+TDA)", model, Xct, y_ct, results)

df = pd.DataFrame(results)
df.to_csv(os.path.join(SAVE_DIR, "model_cv_results_final_v7.csv"), index=False)

# ===============================================================
# Ploting
# ===============================================================
plt.figure(figsize=(10, 6))

x = np.arange(0, len(df) * 1.7, 1.7)
bar_width = 0.40

acc_color = "#4C72B0"
auc_color = "#DD8452"

acc_bar = plt.bar(
    x - bar_width / 2, df["Accuracy Mean"], bar_width,
    yerr=df["Accuracy Std"], capsize=5,
    color=acc_color, edgecolor="black", linewidth=0.8, label="Accuracy"
)
auc_bar = plt.bar(
    x + bar_width / 2, df["AUC Mean"], bar_width,
    yerr=df["AUC Std"], capsize=5,
    color=auc_color, edgecolor="black", linewidth=0.8, label="AUC"
)

for i, (bar_acc, bar_auc) in enumerate(zip(acc_bar, auc_bar)):
    height_acc = bar_acc.get_height()
    height_auc = bar_auc.get_height()
    offset_y = 0.035
    offset_x = 0.22  

    plt.text(bar_acc.get_x() + bar_acc.get_width()/2 - offset_x,
             height_acc + offset_y, f"{height_acc:.3f}",
             ha="center", va="bottom", fontsize=13, fontweight="bold")
    plt.text(bar_auc.get_x() + bar_auc.get_width()/2 + offset_x,
             height_auc + offset_y, f"{height_auc:.3f}",
             ha="center", va="bottom", fontsize=13, fontweight="bold")

plt.xticks(x, df["Model"], rotation=25, ha="right", fontsize=13, fontweight="bold")
plt.ylabel("Score", fontsize=15, fontweight="bold")
plt.title("Chirality Classification Benchmark (5-Fold CV)", fontsize=17, fontweight="bold", pad=10)
plt.yticks(fontsize=12)
plt.ylim(0.0, 1.05)

plt.legend(fontsize=13, frameon=False, loc="upper left")

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "model_cv_barplot_final_v7.png"), dpi=400, bbox_inches="tight")
plt.close()

print("✅ Figure saved (v7: bar thicker + wider spacing, top-journal style).")
