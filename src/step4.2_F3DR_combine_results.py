import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn as sns

# =====================
# Dynamic path configuration
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def combine_results(main_output_dir):
    # CSV
    all_csv_files = glob.glob(os.path.join(main_output_dir, "batch*/reconstruction_metrics_*.csv"))
    
    if not all_csv_files:
        print("")
        return
    
    # CSV
    combined_df = pd.concat([pd.read_csv(f) for f in all_csv_files], ignore_index=True)
    
    # 
    final_output_dir = os.path.join(main_output_dir, "final_results")
    os.makedirs(final_output_dir, exist_ok=True)
    combined_path = os.path.join(final_output_dir, "combined_metrics.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f": {combined_path}")
    plot_final_correlation(combined_df, final_output_dir)
    return combined_df

def plot_final_correlation(combined_df, output_dir):
    # ()
    valid_results = combined_df[
        (combined_df['Wasserstein'].notnull()) & 
        (~combined_df['id'].str.contains('mirrored'))
    ]
    
    if valid_results.empty:
        print("")
        return
    
    rmsd_values = valid_results['RMSD'].values
    wasserstein_values = valid_results['Wasserstein'].values

    import seaborn as sns
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=rmsd_values,
        y=wasserstein_values,
        edgecolor='black',
        linewidth=0.3,
        alpha=0.7
    )

    # Adding trend line
    if len(rmsd_values) > 1:
        z = np.polyfit(rmsd_values, wasserstein_values, 1)
        p = np.poly1d(z)
        plt.plot(rmsd_values, p(rmsd_values), "r--")
        correlation = np.corrcoef(rmsd_values, wasserstein_values)[0, 1]
        plt.title(f"Full Dataset Correlation (r = {correlation:.2f})")
    else:
        correlation = None
        plt.title("Full Dataset Correlation")

    plt.xlabel("RMSD (Å)")
    plt.ylabel("Wasserstein Distance")
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, 'full_correlation.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f": {save_path}")
    
    # Adding statistical information
    stats = {
        'mean_RMSD': np.mean(rmsd_values),
        'median_RMSD': np.median(rmsd_values),
        'mean_Wasserstein': np.mean(wasserstein_values),
        'median_Wasserstein': np.median(wasserstein_values),
        'correlation': correlation,
        'num_proteins': len(valid_results)
    }
    
    stats_path = os.path.join(output_dir, 'correlation_stats.txt')
    with open(stats_path, 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f": {stats_path}")


if __name__ == "__main__":
    main_output_dir = os.path.join(PROJECT_ROOT, "Research_results/Step4.1_and_4.2_topological_signature_modeling")
    parser = argparse.ArgumentParser(description='Combine all batch results')
    # parser.add_argument('--main_dir', type=str, required=True, help='Main directory containing all batch results')
    args = parser.parse_args()
    
    combine_results(main_output_dir)

# How to use: python step4.2_F3DR_combine_results.py