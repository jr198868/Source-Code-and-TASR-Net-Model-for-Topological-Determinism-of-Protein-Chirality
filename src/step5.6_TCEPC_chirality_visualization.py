import os, gc, pickle, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

def extract_coord_features(coords):
    coords = np.array(coords)
    return np.array([
        coords.shape[0],                  
        np.mean(coords, axis=0).sum(),    
        np.var(coords),                   
        np.linalg.norm(coords, axis=1).mean()  
    ])

def load_features_from_pkl(pkl_path, max_samples=None):
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)

    features, labels = [], []
    for i, entry in enumerate(dataset):
        feat_coord = extract_coord_features(entry["coords"])
        feat_tda = np.array(entry["tda"])
        feat = np.concatenate([feat_coord, feat_tda])

        features.append(feat)
        labels.append(entry["label"])

        del feat_coord, feat_tda, feat
        if i % 5000 == 0:
            gc.collect()

        if max_samples and len(features) >= max_samples:
            break

    return np.array(features), np.array(labels)

def plot_tsne(X, y, outpath, sample_size=5000):
    """t-SNE ... (... v2)"""
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X, y = X[idx], y[idx]

    # PCA ...
    n_comp = min(30, X.shape[1])
    X_pca = PCA(n_components=n_comp).fit_transform(X)
    X_emb = TSNE(n_components=2, perplexity=50, random_state=42).fit_transform(X_pca)

    label_map = {0: "Original (L-handed)", 1: "Mirror (R-handed)"}
    labels_named = np.vectorize(label_map.get)(y)

    sns.set(style="white", context="talk", font_scale=1.4)
    plt.figure(figsize=(8, 7))

    scatter = sns.scatterplot(
        x=X_emb[:, 0],
        y=X_emb[:, 1],
        hue=labels_named,
        palette=["#1f77b4", "#ff7f0e"],
        alpha=0.55,
        s=70,
        edgecolors='k',
        linewidths=0.3,
        rasterized=True
    )

    plt.xlabel("t-SNE dimension 1", fontsize=18, weight="bold")
    plt.ylabel("t-SNE dimension 2", fontsize=18, weight="bold")
    plt.title("t-SNE Projection of Protein Chirality Space", fontsize=18, weight="bold", pad=15)

    leg = plt.legend(title="Protein Type", fontsize=14, title_fontsize=18, loc="best", frameon=True)
    leg.get_frame().set_edgecolor("lightgray")
    leg.get_frame().set_linewidth(0.8)

    sns.despine()
    plt.tight_layout()
    plt.savefig(outpath, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close()

def plot_roc_curves(models_dict, X_test, y_test, outdir):
    plt.figure(figsize=(6,6))
    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],'k--',lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Chirality Classification")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curves.png"), dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, outdir):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5,5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"confusion_matrix_{model_name}.png"), dpi=300)
    plt.close()


def train_and_plot_models(X, y, outdir):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LogReg": LogisticRegression(max_iter=1000),
        "MLP": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300),
        "XGB": XGBClassifier(n_estimators=200, tree_method="hist", subsample=0.8, eval_metric="logloss")
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        results.append({"Model": name, "Accuracy": acc, "AUC": auc})
        trained_models[name] = model

    df = pd.DataFrame(results)
    plt.figure(figsize=(6,5))
    sns.barplot(data=df, x="Model", y="Accuracy", color="skyblue", edgecolor="black")
    plt.title("Model Performance on Chirality Classification")
    for i, row in df.iterrows():
        plt.text(i, row["Accuracy"]+0.01, f"{row['Accuracy']:.2f}", ha="center", fontsize=18, weight="bold")
    plt.ylim(0,1.1)
    plt.savefig(os.path.join(outdir, "model_performance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plot_roc_curves(trained_models, X_test, y_test, outdir)

    best_model_name = df.sort_values("AUC", ascending=False).iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    y_pred_best = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_best, best_model_name, outdir)

    return df

def plot_xgb_feature_importance(xgb_model, feature_names, outdir):
    importance = xgb_model.feature_importances_
    idx = np.argsort(importance)[::-1]

    sns.set(style="whitegrid", context="talk", font_scale=1.4)
    plt.figure(figsize=(9, 6))

    bars = sns.barplot(
        x=importance[idx][:10],
        y=np.array(feature_names)[idx][:10],
        palette="Blues_r",
        edgecolor="black"
    )

    for i, v in enumerate(importance[idx][:10]):
        plt.text(v + 0.005, i, f"{v:.2f}", color="black", va="center", fontsize=18, weight="bold")

    plt.xlabel("Importance Score", fontsize=18, weight="bold")
    plt.ylabel("Feature", fontsize=18, weight="bold")
    plt.title("Top 10 Feature Importances in Chirality Classification", fontsize=18, weight="bold", pad=15)
    plt.xlim(0, max(importance) * 1.2)

    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tda_feature_importance.png"), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_learning_curve(model, X, y, outdir):
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=3, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
    )
    plt.figure(figsize=(6,5))
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label="Training")
    plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label="Validation")
    plt.title("Learning Curve (XGBoost)")
    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "learning_curve.png"), dpi=300)
    plt.close()


# ========== Main program ==========
def main():
    # =====================
    # Dynamic path configuration
    # =====================
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    pkl_path = os.path.join(PROJECT_ROOT, "Research_results/step5.2_build_chirality_dataset/full_dataset.pkl")
    outdir = os.path.join(PROJECT_ROOT, "Research_results/Step5.6_TCEPC_supplementary_figures")
    os.makedirs(outdir, exist_ok=True)

    print("Loading dataset")
    X, y = load_features_from_pkl(pkl_path)
    print("Features loaded:", X.shape)

    plot_tsne(X, y, os.path.join(outdir, "tsne.png"))

    results_df = train_and_plot_models(X, y, outdir)
    results_df.to_csv(os.path.join(outdir, "model_performance.csv"), index=False)

    best_model = XGBClassifier(n_estimators=200, tree_method="hist", subsample=0.8, eval_metric="logloss")
    best_model.fit(X, y)

    feature_names = [
        "Coord_N_atoms",          # feature 0
        "Coord_mean_sum",         # feature 1
        "Coord_variance",         # feature 2
        "Coord_avg_norm",         # feature 3
        "TDA_H0_count",           # feature 4
        "TDA_H0_mean_persistence",# feature 5
        "TDA_H0_max_persistence", # feature 6
        "TDA_H1_count",           # feature 7
        "TDA_H1_mean_persistence",# feature 8
        "TDA_H1_max_persistence", # feature 9
        "TDA_total_persistence"   # feature 10
    ]

    plot_xgb_feature_importance(best_model, feature_names, outdir)

    plot_learning_curve(best_model, X, y, outdir)

    print("✅ All figures saved in:", outdir)

if __name__ == "__main__":
    main()
