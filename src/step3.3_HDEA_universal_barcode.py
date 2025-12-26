import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde  

# =====================
# Dynamic path configuration
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============== Paths and styles ==============
# Define input CSV path and output directory using relative paths
file_path = os.path.join(PROJECT_ROOT, 'Research_results/step3.2_all_barcode_points/step3.2_all_barcode_points.csv')
output_dir = os.path.join(PROJECT_ROOT, 'Research_results/step3.3_universal_barcode/')
os.makedirs(output_dir, exist_ok=True)

# Global plotting style configuration
sns.set(style="ticks", context="notebook")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 16,
    'axes.linewidth': 1.5,
    'axes.edgecolor': 'black',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'savefig.dpi': 300
})

# ============== Read and Preprocess Data ==============
df = pd.read_csv(file_path)
df['lifespan'] = df['death'] - df['birth']
dims = sorted(df['dim'].unique())

# Universal folding scale interval and center constants
scale_min, scale_max, scale_center = 4.32, 4.43, 4.38

def approx_mode(values, bins=200):
    """Approximate the mode using the center of the highest histogram bin"""
    vals = np.asarray(values.dropna())
    if len(vals) == 0:
        return np.nan
    hist, edges = np.histogram(vals, bins=bins)
    idx = np.argmax(hist)
    return 0.5 * (edges[idx] + edges[idx+1])

def add_stat_pad(ax, txt, loc='lower right', box_alpha=0.9):
    """Add an information card (box) to the plot"""
    if loc == 'lower right':
        x, y, ha, va = 0.98, 0.02, 'right', 'bottom'
    elif loc == 'upper right':
        x, y, ha, va = 0.98, 0.98, 'right', 'top'
    elif loc == 'upper left':
        x, y, ha, va = 0.02, 0.98, 'left', 'top'
    elif loc == 'center left':
        x, y, ha, va = 0.02, 0.55, 'left', 'center'
    else:
        x, y, ha, va = 0.02, 0.02, 'left', 'bottom'

    ax.text(
        x, y, txt, transform=ax.transAxes,
        fontsize=11, ha=ha, va=va,
        family='monospace',
        bbox=dict(boxstyle="round,pad=0.35", facecolor='white', edgecolor='black', alpha=box_alpha)
    )

# ============== Main Loop: KDE plots for each dimension + Statistics Pad ==============
for dim in dims:
    dim_df = df[df['dim'] == dim].copy()

    plt.figure(figsize=(7, 7))
    ax = sns.kdeplot(
        data=dim_df, x='birth', y='death',
        fill=True, cmap='coolwarm',
        bw_adjust=0.2, thresh=0.02, levels=120
    )
    max_val = float(max(dim_df['birth'].max(), dim_df['death'].max()))
    ax.plot([0, max_val], [0, max_val], linestyle='--', color='black', linewidth=1.2)

    if dim in [1, 2]:
        ax.axvspan(scale_min, scale_max, color='red', alpha=0.2, label="4.32–4.43 Å")
        ax.axvline(scale_center, color='red', linestyle='--', linewidth=1.5)

    thr = dim_df['lifespan'].quantile(0.9)
    top_df = dim_df[dim_df['lifespan'] >= thr]
    n_all, n_top = len(dim_df), len(top_df)
    birth_mode_all = approx_mode(dim_df['birth'])
    txt = (
        f"dim={dim}\n"
        f"N={n_all:,}\n"
        f"Top10 N={n_top:,}\n"
        f"Top10 birth μ={top_df['birth'].mean():.3f} Å\n"
        f"Top10 death μ={top_df['death'].mean():.3f} Å\n"
        f"Top10 life μ={top_df['lifespan'].mean():.3f} Å\n"
        f"Birth mode (all) ≈ {birth_mode_all:.3f} Å"
    )
    add_stat_pad(ax, txt, loc='upper right')

    ax.set_title(f'Universal Barcode Density (dim={dim})', fontsize=20, fontweight='bold')
    ax.set_xlabel('Birth (Å)', fontsize=18)
    ax.set_ylabel('Death (Å)', fontsize=18)
    ax.set_aspect('equal', adjustable='box')
    if dim in [1, 2]:
        ax.legend(loc='upper left', fontsize=12, frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/universal_barcode_kde_dim{dim}.png')
    plt.close()

# ============== Save Dim=2 Inset (Marginal Distribution + Summary Table) ==============
dim2 = df[df['dim'] == 2].copy()
thr2 = dim2['lifespan'].quantile(0.9)
top2 = dim2[dim2['lifespan'] >= thr2].copy()

# Prepare summary table data
summary_rows = []
for dim in dims:
    dim_df = df[df['dim'] == dim].copy()
    thr = dim_df['lifespan'].quantile(0.9)
    top_df = dim_df[dim_df['lifespan'] >= thr]
    summary_rows.append({
        'dim': dim,
        'Top10_N': len(top_df),
        'Top10_birth_mean': float(top_df['birth'].mean()),
        'Top10_death_mean': float(top_df['death'].mean()),
        'Top10_lifespan_mean': float(top_df['lifespan'].mean())
    })

fig = plt.figure(figsize=(6, 5.5))
gs = GridSpec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4], wspace=0.05, hspace=0.05)
ax_main = fig.add_subplot(gs[1, 0])
ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

sns.kdeplot(
    data=top2, x='birth', y='death',
    fill=True, cmap='Greens', bw_adjust=0.2, thresh=0.02, levels=60, ax=ax_main
)
max_val_main = float(max(top2['birth'].max(), top2['death'].max()))
ax_main.plot([0, max_val_main], [0, max_val_main], linestyle='--', color='black', linewidth=1.0)
ax_main.axvspan(scale_min, scale_max, color='red', alpha=0.2)
ax_main.axvline(scale_center, color='red', linestyle='--', linewidth=1.2)

ax_main.set_xlabel("Birth (Å)", fontsize=10)
ax_main.set_ylabel("Death (Å)", fontsize=10)
ax_main.set_title("Top 10% lifespan (dim=2)", fontsize=12)
sns.despine(ax=ax_main)

sns.kdeplot(data=top2, x='birth', ax=ax_top, bw_adjust=0.5, fill=True)
ax_top.axvspan(scale_min, scale_max, color='red', alpha=0.15)
ax_top.axvline(scale_center, color='red', linestyle='--', linewidth=1.0)
ax_top.tick_params(axis='x', labelbottom=False)
sns.despine(ax=ax_top, bottom=True, left=True)

sns.kdeplot(data=top2, y='death', ax=ax_right, bw_adjust=0.5, fill=True)
ax_right.tick_params(axis='y', labelleft=False)
sns.despine(ax=ax_right, bottom=True, right=True)

# Small table in top-right of main inset
table_text = "dim  Top10_N  birthμ   deathμ   lifeμ\n"
for row in summary_rows:
    table_text += f"{row['dim']}    {row['Top10_N']:<8} {row['Top10_birth_mean']:.3f}  {row['Top10_death_mean']:.3f}  {row['Top10_lifespan_mean']:.3f}\n"

ax_main.text(
    1.05, 1.0, table_text,
    transform=ax_main.transAxes, ha='left', va='top',
    fontsize=9, family='monospace',
    bbox=dict(boxstyle="round,pad=0.35", facecolor='white', edgecolor='black', alpha=0.85)
)

plt.tight_layout()
plt.savefig(f'{output_dir}/dim2_top10_inset.png')
plt.close()

# ============== 1D Birth Distribution (dim=1/2) ==============
for d in [1, 2]:
    dd = df[df['dim'] == d]['birth'].dropna()
    mode_b = approx_mode(dd)
    plt.figure(figsize=(6.5, 4))
    sns.histplot(dd, bins=120, stat='density', alpha=0.35, edgecolor=None)
    sns.kdeplot(dd, bw_adjust=0.5)
    plt.axvspan(scale_min, scale_max, color='red', alpha=0.15, label='4.32–4.43 Å')
    plt.axvline(scale_center, color='red', linestyle='--', linewidth=1.2)
    plt.axvline(mode_b, color='black', linestyle='-', linewidth=1.2, label=f"Mode ≈ {mode_b:.3f} Å")
    plt.title(f'Birth distribution (dim={d})', fontsize=16, fontweight='bold')
    plt.xlabel('Birth (Å)')
    plt.ylabel('Density')
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/birth_distribution_dim{d}.png')
    plt.close()

# ============== Overlay Plot (All Data) ==============
plt.figure(figsize=(7.5,5.5))
ax_overlay = plt.gca()
colors = {1: "#1f77b4", 2: "#d95f02"}
for d in [1, 2]:
    dd = df[df['dim'] == d]['birth'].dropna()
    sns.histplot(dd, bins=80, stat="density", alpha=0.45, color=colors[d], label=f"dim={d}")
plt.axvspan(scale_min, scale_max, color='red', alpha=0.15, label='4.32–4.43 Å')
plt.axvline(scale_center, color='red', linestyle='--', lw=1.8)
plt.title("Universal folding scale across dimensions", fontsize=18, fontweight='bold')
plt.xlabel("Birth (Å)")
plt.ylabel("Density")
plt.legend(frameon=False, fontsize=12, loc="upper left")

# ✅ ADDED: Abstract Statistics box on Overlay Plot
# Placed slightly above center on the left (loc='center left') to avoid legend
abstract_text = "Abstract Statistics (Top 10%):\n"
abstract_text += "dim  Top10_N  birthμ   deathμ   lifeμ\n"
for row in summary_rows:
    abstract_text += f"{row['dim']}    {row['Top10_N']:<8} {row['Top10_birth_mean']:.3f}  {row['Top10_death_mean']:.3f}  {row['Top10_lifespan_mean']:.3f}\n"

add_stat_pad(ax_overlay, abstract_text, loc='center left', box_alpha=0.8)

sns.despine()
plt.tight_layout()
plt.savefig(f'{output_dir}/overlay_dim1_dim2_birth_hist.png')
plt.close()

# ============== New Overlay: Top 10% Filtering ==============
plt.figure(figsize=(7.5,5.5))
for d in [1, 2]:
    sub = df[df['dim'] == d].copy()
    thr = sub['lifespan'].quantile(0.9)
    sub = sub[sub['lifespan'] >= thr]
    sns.histplot(sub['birth'].dropna(), bins=80, stat="density",
                 alpha=0.5, color=colors[d], label=f"dim={d} (Top 10%)")
plt.axvspan(scale_min, scale_max, color='red', alpha=0.15, label='4.32–4.43 Å')
plt.axvline(scale_center, color='red', linestyle='--', lw=1.8)
plt.title("Universal folding scale (Top 10% lifespan)", fontsize=18, fontweight='bold')
plt.xlabel("Birth (Å)")
plt.ylabel("Density")
plt.legend(frameon=False, fontsize=11, loc="upper left")
sns.despine()
plt.tight_layout()
plt.savefig(f'{output_dir}/overlay_dim1_dim2_birth_hist_top10.png')
plt.close()

# ============== New Overlay: Lifespan-Weighted KDE ==============
plt.figure(figsize=(7.5,5.5))
for d, col in [(1, "#1f77b4"), (2, "#d95f02")]:
    sub = df[df['dim'] == d][['birth', 'lifespan']].dropna().copy()
    w = sub['lifespan'].clip(upper=sub['lifespan'].quantile(0.99))
    w = w / w.median()
    x = sub['birth'].values
    grid = np.linspace(max(0, x.min()-0.2), x.max()+0.2, 800)
    kde = gaussian_kde(x, weights=w)
    y = kde(grid)
    plt.plot(grid, y, lw=2.5, color=col, label=f"dim={d} (lifespan-weighted KDE)")
plt.axvspan(scale_min, scale_max, color='red', alpha=0.15, label='4.32–4.43 Å')
plt.axvline(scale_center, color='red', ls='--', lw=1.6)
plt.title("Universal folding scale (lifespan-weighted density)", fontsize=18, fontweight='bold')
plt.xlabel("Birth (Å)")
plt.ylabel("Weighted density")
plt.legend(frameon=False, fontsize=11, loc="upper left")
sns.despine()
plt.tight_layout()
plt.savefig(f'{output_dir}/overlay_dim1_dim2_birth_kde_weighted.png')
plt.close()

# ============== CDF Plot Function ==============
def first_stable_birth_cdf(df, dim, lifespan_q=0.9):
    """Generates a styled CDF plot for stable barcodes"""
    d = df[df['dim'] == dim].copy()
    thr = d['lifespan'].quantile(lifespan_q)
    d = d[d['lifespan'] >= thr]

    births = np.sort(d['birth'].dropna().values)
    y = np.linspace(0, 1, len(births), endpoint=True)

    plt.figure(figsize=(7,5))
    ax = plt.gca()
    ax.set_facecolor('#f7f7f7')
    plt.fill_between(births, 0, y, color='royalblue', alpha=0.3)
    plt.plot(births, y, lw=3, color='royalblue', label="CDF of stable barcodes")
    plt.axvspan(scale_min, scale_max, color='red', alpha=0.15)
    plt.axvline(scale_center, color='red', ls='--', lw=2)
    plt.text(scale_center+0.02, 0.6, "Universal\nfolding scale", color='red',
             fontsize=12, ha='left', va='center', fontweight='bold')

    plt.title(f"First-stable birth CDF (dim={dim})", fontsize=18, fontweight='bold')
    plt.xlabel("Birth (Å)")
    plt.ylabel("Cumulative proportion")
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/first_stable_birth_cdf_dim{dim}_styled.png')
    plt.close()

for d in [1, 2]:
    first_stable_birth_cdf(df, d)

print("✅ Output Complete:")
print("  - universal_barcode_kde_dim{0,1,2}.png")
print("  - dim2_top10_inset.png (Table in top-right)")
print("  - birth_distribution_dim{1,2}.png")
print("  - overlay_dim1_dim2_birth_hist.png (With Abstract Statistics box)")
print("  - overlay_dim1_dim2_birth_hist_top10.png (Top 10% Filter)")
print("  - overlay_dim1_dim2_birth_kde_weighted.png (Weighted KDE)")
print("  - first_stable_birth_cdf_dim{1,2}_styled.png (Styled CDF)")