import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# =====================
# Dynamic path configuration
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- Step 2.1: Read data ----------
csv_path = os.path.join(PROJECT_ROOT, 'Research_results/step3.0_tda_features/step3.0_tda_features.csv')
df = pd.read_csv(csv_path, index_col=0)
output_dir = os.path.join(PROJECT_ROOT, 'Research_results/step3.1_tda_features_with_clusters/')
os.makedirs(output_dir, exist_ok=True) 

def high_dimensional_embedding_analysis(df):
    # ---------- Step 2.2: Standardization ----------
    print(f"✅ Loaded dataset with {df.shape[0]} samples and {df.shape[1]} features")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    
    # ---------- Step 2.3: PCA dimensionality reduction to 3D ----------
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled_features)
    df['pca1'] = pca_result[:, 0]
    df['pca2'] = pca_result[:, 1]
    df['pca3'] = pca_result[:, 2]
    
    # ---------- Step 2.4: KMeans clustering ----------
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_features)
    
    # ---------- Step 2.4.1: Calculate silhouette score ----------
    sil_score = silhouette_score(scaled_features, df['cluster'])
    print(f"✅ Silhouette Score: {sil_score:.3f}")
    
    # Save results to text file
    with open(os.path.join(output_dir, "clustering_metrics.txt"), "w") as f:
        f.write(f"Silhouette Score (n_clusters=2): {sil_score:.3f}\n")
    
    # ---------- Step 2.5: Visualize 2D PCA plots ----------
    cluster_labels = sorted(df['cluster'].unique())
    palette = sns.color_palette("Set2", n_colors=len(cluster_labels))

    def plot_pca(df, x, y, save_path, title):
        plt.figure(figsize=(10, 8))
        sns.set(style="white", font_scale=1.6)
        ax = sns.scatterplot(
            data=df, x=x, y=y, hue='cluster', palette=palette,
            s=120, edgecolor='black', linewidth=0.8
        )
        ax.set_title(f"{title}\nSilhouette Score = {sil_score:.2f}", fontsize=20, weight='bold')
        ax.set_xlabel(x.upper(), fontsize=18)
        ax.set_ylabel(y.upper(), fontsize=18)
        ax.legend(title='Cluster', loc='best', fontsize=14, title_fontsize=16)
        sns.despine()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    plot_pca(df, 'pca1', 'pca2',
             os.path.join(output_dir, "step2.2.1_pca1_vs_pca2.png"),
             "PCA1 vs PCA2 + KMeans Clusters")

    plot_pca(df, 'pca1', 'pca3',
             os.path.join(output_dir, "step2.2.1_pca1_vs_pca3.png"),
             "PCA1 vs PCA3 + KMeans Clusters")

    plot_pca(df, 'pca2', 'pca3',
             os.path.join(output_dir, "step2.2_pca2_vs_pca3.png"),
             "PCA2 vs PCA3 + KMeans Clusters")
    
    # ---------- Step 2.6: Visualize 3D PCA plots ----------
    plotly_colors = ['rgb({}, {}, {})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in palette]
    fig = go.Figure()

    for i, cluster in enumerate(cluster_labels):
        cluster_data = df[df['cluster'] == cluster]
        fig.add_trace(go.Scatter3d(
            x=cluster_data['pca1'],
            y=cluster_data['pca2'],
            z=cluster_data['pca3'],
            mode='markers',
            marker=dict(
                size=6,
                color=plotly_colors[i],
                line=dict(color='black', width=1)
            ),
            name=f'Cluster {cluster}'
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='PC 1', titlefont=dict(size=18), tickfont=dict(size=14)),
            yaxis=dict(title='PC 2', titlefont=dict(size=18), tickfont=dict(size=14)),
            zaxis=dict(title='PC 3', titlefont=dict(size=18), tickfont=dict(size=14)),
        ),
        title=dict(text=f'3D PCA of TDA Barcode Features (Silhouette={sil_score:.2f})', font=dict(size=22)),
        legend=dict(font=dict(size=14)),
        margin=dict(l=0, r=0, b=0, t=50),
        font=dict(family="Arial", size=14)
    )

    fig.write_html(os.path.join(output_dir, "step3.1_tda_features_with_clusters.html"))
    print("✅ Saved step3.1_tda_features_with_clusters.html")
    fig.show()
    
    # ---------- Step 2.7: Save results with cluster labels ----------
    df.to_csv(os.path.join(output_dir, "step3.1_tda_features_with_clusters.csv"))
    print("✅ Step3.1 complete: 3D dimensionality reduction + clustering + visualization, all charts and clustering results saved")

if __name__ == "__main__":
    high_dimensional_embedding_analysis(df)
