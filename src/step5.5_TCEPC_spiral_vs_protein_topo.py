import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from matplotlib.patheffects import withStroke
from ripser import ripser
from PIL import Image
import warnings

# ===============================================================
# Configuration / constants
# ===============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(PROJECT_ROOT, "Research_results", "step5.5_spiral_vs_protein_topo")
DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "Research_results",
    "step5.2_build_chirality_dataset",
    "full_dataset.pkl",
)
os.makedirs(BASE_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

TSNE_PERPLEX = 50
TSNE_RANDOM_STATE = SEED
PCA_N_COMPONENTS = 5

KMEANS_N_CLUSTERS = 5
KMEANS_BATCH_SIZE = 2048
KMEANS_N_INIT = 10

DOT_SIZE = 55
DOT_ALPHA = 0.55
EDGE_WIDTH = 0.25
EDGE_COLOR = "k"

# ===============================================================
# Unified font control (ONLY NEW DESIGN CHANGE)
# ===============================================================
FONT_BASE = 18

FONT_TITLE = FONT_BASE + 4
FONT_LABEL = FONT_BASE + 2
FONT_TICK = FONT_BASE
FONT_LEGEND = FONT_BASE - 4
FONT_CLUSTER_LABEL = FONT_BASE

FIG_DPI = 300
POINT_SIZE = 10

CLUSTER_NAME_ORDERED = [
    "α-Helix-like",
    "β-Sheet-rich",
    "Loop/Turn-rich",
    "Globular Fold",
    "Complex/Hybrid",
]

CLUSTER_COLOR_MAP = {
    "α-Helix-like": "#2ca02c",
    "β-Sheet-rich": "#d95f02",
    "Loop/Turn-rich": "#e31a1c",
    "Globular Fold": "#1f77b4",
    "Complex/Hybrid": "#9467bd",
}

# ===============================================================
# Utilities
# ===============================================================
def safe_load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_figure(fig, outpath):
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved: {outpath}")

def apply_plot_style(ax, title, xlabel, ylabel, legend_title=None):
    """
    Applies unified font sizes and common aesthetics (title, labels, ticks, spines) 
    to a matplotlib Axes object.
    """
    ax.set_title(title, fontsize=FONT_TITLE, weight="bold")
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL, weight="bold")
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=FONT_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Handle legend only if a title is provided
    if legend_title:
        ax.legend(fontsize=FONT_LEGEND, title=legend_title, loc="lower left")

# ===============================================================
# Figure A
# ===============================================================
def generate_figure_A_from_tda(dataset, outpath):
    tda = np.vstack([
        np.asarray(e["tda"], float).flatten()
        for e in dataset
        if "tda" in e and np.asarray(e["tda"]).size == 7
    ])

    pca = PCA(n_components=min(PCA_N_COMPONENTS, tda.shape[1]), random_state=SEED)
    tda_pca = pca.fit_transform(tda)

    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEX,
        random_state=TSNE_RANDOM_STATE,
        init="pca",
    )
    X_emb = tsne.fit_transform(tda_pca)

    kmeans = MiniBatchKMeans(
        n_clusters=KMEANS_N_CLUSTERS,
        random_state=SEED,
        batch_size=KMEANS_BATCH_SIZE,
        n_init=KMEANS_N_INIT,
    )
    labels = kmeans.fit_predict(tda_pca)

    info = []
    for c in np.unique(labels):
        pts = X_emb[labels == c]
        info.append((c, pts.shape[0], np.median(pts[:, 0]) + np.median(pts[:, 1])))

    info_sorted = sorted(info, key=lambda x: x[1], reverse=True)

    assigned = {
        info_sorted[0][0]: "β-Sheet-rich",
        info_sorted[1][0]: "Globular Fold",
        info_sorted[-1][0]: "Complex/Hybrid",
    }

    remaining = [c for c in np.unique(labels) if c not in assigned]
    alpha = max(
        remaining,
        key=lambda c: np.median(X_emb[labels == c][:, 0]) +
                      np.median(X_emb[labels == c][:, 1]),
    )
    assigned[alpha] = "α-Helix-like"

    for c in remaining:
        if c not in assigned:
            assigned[c] = "Loop/Turn-rich"

    fig, ax = plt.subplots(figsize=(9, 8))

    for c in np.unique(labels):
        mask = labels == c
        name = assigned[c]
        ax.scatter(
            X_emb[mask, 0],
            X_emb[mask, 1],
            s=DOT_SIZE * 0.6,
            alpha=0.75,
            color=CLUSTER_COLOR_MAP[name],
            edgecolor="k",
            linewidth=0.35,
        )

    for name in CLUSTER_NAME_ORDERED:
        # Note: Legend creation must happen before applying the style if using its title
        ax.scatter([], [], color=CLUSTER_COLOR_MAP[name], label=name)

    for c in np.unique(labels):
        pts = X_emb[labels == c]
        x, y = np.median(pts[:, 0]), np.median(pts[:, 1])
        txt = ax.text(
            x, y, assigned[c],
            fontsize=FONT_CLUSTER_LABEL,
            weight="bold",
            ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.3"),
        )
        txt.set_path_effects([withStroke(linewidth=3, foreground="white")])

    apply_plot_style(
        ax,
        title="Global Protein Topological Landscape",
        xlabel="t-SNE Dimension 1",
        ylabel="t-SNE Dimension 2",
        legend_title="Structural Type" # Pass legend title to ensure correct font size
    )

    save_figure(fig, outpath)
    return outpath

# ===============================================================
# Panels B–D (Figure D fully restored)
# ===============================================================
def safe_get_tda_matrix(dataset):
    return np.vstack([np.asarray(e["tda"], float).flatten() for e in dataset if "tda" in e])

def generate_tsne(dataset, mirror=False):
    X = safe_get_tda_matrix(dataset)
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEX,
                random_state=TSNE_RANDOM_STATE, init="pca")
    emb = tsne.fit_transform(X)
    if mirror:
        emb[:, 0] *= -1
    return emb

def generate_spiral():
    t = np.linspace(-2.5, 2.5, 300)
    x = t
    y = t**2
    rng = np.random.default_rng(SEED)
    y += rng.normal(0, 0.15, size=y.shape)
    return np.column_stack([x, y])

def save_scatter(X, title, outpath, color):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(X[:, 0], X[:, 1], s=DOT_SIZE, alpha=DOT_ALPHA,
               color=color, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    
    apply_plot_style(
        ax,
        title=title,
        xlabel="t-SNE Dimension 1",
        ylabel="t-SNE Dimension 2"
    )

    save_figure(fig, outpath)
    return outpath

# ===============================================================
# Figure E (black borders restored)
# ===============================================================
def compute_and_plot_barcode(coords, title, filename):
    dgms = ripser(coords)["dgms"]
    fig, ax = plt.subplots(figsize=(9, 8))
    for i, dgm in enumerate(dgms):
        if len(dgm):
            ax.scatter(dgm[:, 0], dgm[:, 1],
                       s=POINT_SIZE**2.5, alpha=0.75, edgecolor="k", linewidth=0.4, label=f"H{i}")
    lim = max(np.max(d) for d in dgms if len(d))
    ax.plot([0, lim], [0, lim], "--", color="gray")
    
    apply_plot_style(
        ax,
        title=title,
        xlabel="Birth",
        ylabel="Death"
    )
    
    # Note: Legend for barcodes is internal to the loop, so the style must be applied afterward.
    # The legend title is not explicitly needed here, but H0/H1/H2 labels are added by the loop.
    ax.legend(fontsize=FONT_LEGEND, loc="upper right") # Re-adding legend for H0/H1/H2 features
    
    save_figure(fig, os.path.join(BASE_DIR, filename))
    return os.path.join(BASE_DIR, filename)

# ===============================================================
# Combine & Main
# ===============================================================
def make_combined_image(imgs, outpath):
    imgs = [Image.open(p) for p in imgs if p]
    h = max(im.height for im in imgs)
    resized = [im.resize((int(im.width * h / im.height), h)) for im in imgs]
    w = sum(im.width for im in resized)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width
    canvas.save(outpath)
    print(f"✅ Combined figure saved: {outpath}")

def main():
    warnings.filterwarnings("ignore")
    dataset = safe_load_pickle(DATASET_PATH)

    outA = generate_figure_A_from_tda(dataset, os.path.join(BASE_DIR, "FigureA_Global.png"))
    outB = save_scatter(generate_tsne(dataset), "Original Proteins",
                         os.path.join(BASE_DIR, "FigureB_Original.png"), "#1f77b4")
    outC = save_scatter(generate_tsne(dataset, mirror=True), "Mirror Proteins",
                         os.path.join(BASE_DIR, "FigureC_Mirror.png"), "#ff7f0e")
    outD = save_scatter(generate_spiral(), "Theoretical Spiral",
                         os.path.join(BASE_DIR, "FigureD_Spiral.png"), "#ffb347")

    # =======================
    # Figure E1 (LOCKED)
    # =======================
    coords = next(e["coords"] for e in dataset if "coords" in e)
    outE1 = compute_and_plot_barcode(
        coords,
        "Protein (Barcode)",
        "FigureE1_Protein.png"
    )

    # =======================
    # Figure E2 (FIXED — ONLY CHANGE)
    # =======================
    t = np.linspace(-2.5, 2.5, 300)
    x = t
    y = t**2

    rng = np.random.default_rng(SEED)
    y += rng.normal(0, 0.15, size=y.shape)

    spiral_for_barcode = np.column_stack([x, y])

    outE2 = compute_and_plot_barcode(
        spiral_for_barcode,
        "Theoretical Spiral (Barcode)",
        "FigureE2_Spiral.png"
    )


    make_combined_image(
        [outA, outB, outC, outD, outE1, outE2],
        os.path.join(BASE_DIR, "Figure5_All_Panels_A_to_E.png"),
    )

    print("🎯 Step 5.5 complete.")


if __name__ == "__main__":
    main()