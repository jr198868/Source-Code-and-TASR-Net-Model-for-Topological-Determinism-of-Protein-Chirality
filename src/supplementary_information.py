import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.linewidth": 1.5
})

# ========== Main function ==========
def main():
    # =====================
    # Dynamic path configuration
    # =====================
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Input and output
    input_csv = os.path.join(PROJECT_ROOT, "Research_results/Step4.3_analyze_error_distribution_and_fold_class/combined_metrics_with_length.csv")
    output_dir = os.path.join(PROJECT_ROOT, "Research_results/supplementary_information")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    try:
        df = pd.read_csv(input_csv)
        logger.info(f"Successfully loaded data, total {len(df)} records")
    except Exception as e:
        logger.error(f"Loadingfailed: {e}")
        return

    # 
    for col in ["RMSD", "TM-score", "Wasserstein"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 
    metrics = ["RMSD", "TM-score", "Wasserstein"]
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        logger.error(" (RMSD/TM-score/Wasserstein)")
        return

    for metric in available_metrics:
        #  NA
        valid_df = df.dropna(subset=[metric]).copy()
        if valid_df.empty:
            logger.warning(f"{metric} No valid data, skipping")
            continue

        # Top 10 & Bottom 10
        if metric == "TM-score":
            top = valid_df.nlargest(10, metric)
            bottom = valid_df.nsmallest(10, metric)
        else:  # RMSD & Wasserstein 
            top = valid_df.nsmallest(10, metric)
            bottom = valid_df.nlargest(10, metric)

        # 
        top = top.copy()
        bottom = bottom.copy()
        top["Group"] = "Top 10"
        bottom["Group"] = "Bottom 10"
        compare_df = pd.concat([top, bottom])

        # 
        plt.figure(figsize=(8, 6))
        sns.boxplot(x="Group", y=metric, hue="Group", data=compare_df,
                    palette=["#4C72B0", "#DD8452"], legend=False)
        sns.stripplot(x="Group", y=metric, data=compare_df,
                      color="black", size=6, jitter=True, alpha=0.7)
        plt.title(f"{metric} Top 10 vs Bottom 10", weight="bold")
        plt.xlabel("")
        plt.ylabel(metric, weight="bold")
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{metric}_top_vs_bottom.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f": {save_path}")

        # 
        csv_path = os.path.join(output_dir, f"{metric}_top_vs_bottom.csv")
        compare_df.to_csv(csv_path, index=False)
        logger.info(f": {csv_path}")

    logger.info(f" Supplementary : {output_dir}")


if __name__ == "__main__":
    main()
