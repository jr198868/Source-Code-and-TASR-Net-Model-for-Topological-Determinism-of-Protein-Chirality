import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy.stats import pearsonr
import re
from Bio import SeqIO

# Configure logging()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

#  (Science style:  + )
plt.rcParams.update({
    "font.size": 16,
    "font.weight": "bold",
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 2
})

def clean_numeric_column(series):
    """"""
    cleaned = []
    for val in series:
        try:
            cleaned.append(float(val))
            continue
        except:
            pass
        complex_match = re.search(r"\(?([\d.]+)\s*[+-]\s*\d+j\)?", str(val))
        if complex_match:
            try:
                cleaned.append(float(complex_match.group(1)))
                continue
            except:
                pass
        cleaned.append(np.nan)
    return pd.Series(cleaned, index=series.index)


def analyze_error_distribution(metrics_df, output_dir):
    """ + """
    logger.info("...")

    error_dir = os.path.join(output_dir, "error_distribution_analysis")
    os.makedirs(error_dir, exist_ok=True)

    # 
    metrics_df["RMSD"] = clean_numeric_column(metrics_df["RMSD"])
    metrics_df["TM-score"] = clean_numeric_column(metrics_df["TM-score"])
    if "Wasserstein" in metrics_df.columns:
        metrics_df["Wasserstein"] = clean_numeric_column(metrics_df["Wasserstein"])

    valid_df = metrics_df[~metrics_df["id"].str.contains("mirrored", na=False)].copy()
    valid_df = valid_df.dropna(subset=["RMSD", "TM-score"])

    if valid_df.empty:
        logger.warning("")
        return None

    # ---- RMSD ----
    plt.figure(figsize=(10, 6))
    sns.histplot(valid_df["RMSD"], kde=True, bins=30, color="darkred")
    mean_rmsd = valid_df["RMSD"].mean()
    median_rmsd = valid_df["RMSD"].median()
    plt.axvline(mean_rmsd, color="blue", linestyle="--", linewidth=2, label=f"Mean = {mean_rmsd:.2f}")
    plt.axvline(median_rmsd, color="green", linestyle="-.", linewidth=2, label=f"Median = {median_rmsd:.2f}")
    plt.title("RMSD Distribution", weight="bold")
    plt.xlabel("RMSD (Å)", weight="bold")
    plt.ylabel("Frequency", weight="bold")
    plt.legend()
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(error_dir, f"rmsd_distribution.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()

    # ---- TM-score ----
    plt.figure(figsize=(10, 6))
    sns.histplot(valid_df["TM-score"], kde=True, bins=30, color="darkblue")
    mean_tm = valid_df["TM-score"].mean()
    median_tm = valid_df["TM-score"].median()
    plt.axvline(mean_tm, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_tm:.2f}")
    plt.axvline(median_tm, color="green", linestyle="-.", linewidth=2, label=f"Median = {median_tm:.2f}")
    plt.title("TM-score Distribution", weight="bold")
    plt.xlabel("TM-score", weight="bold")
    plt.ylabel("Frequency", weight="bold")
    plt.legend()
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(error_dir, f"tm_score_distribution.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Wasserstein ----
    if "Wasserstein" in valid_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(valid_df["Wasserstein"], kde=True, bins=30, color="darkgreen")
        mean_w = valid_df["Wasserstein"].mean()
        median_w = valid_df["Wasserstein"].median()
        plt.axvline(mean_w, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_w:.2f}")
        plt.axvline(median_w, color="blue", linestyle="-.", linewidth=2, label=f"Median = {median_w:.2f}")
        plt.title("Wasserstein Distance Distribution", weight="bold")
        plt.xlabel("Wasserstein Distance", weight="bold")
        plt.ylabel("Frequency", weight="bold")
        plt.legend()
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            plt.savefig(os.path.join(error_dir, f"wasserstein_distribution.{ext}"), dpi=300, bbox_inches="tight")
        plt.close()

        # ---- Correlation Matrix (Pairplot) ----
    if {"RMSD", "TM-score", "Wasserstein"}.issubset(valid_df.columns):
        corr, _ = pearsonr(valid_df["RMSD"].dropna(), valid_df["TM-score"].dropna())
        g = sns.pairplot(
            valid_df,
            vars=["RMSD", "TM-score", "Wasserstein"],
            kind="scatter",
            diag_kind="kde",
            plot_kws={"alpha": 0.6, "s": 30, "edgecolor": "black", "linewidth": 0.3}
        )
        g.fig.suptitle(f"Correlation between Error Metrics (Pearson r ≈ {corr:.2f})",
                       fontsize=22, weight="bold", y=1.02)

        for ext in ["png", "pdf"]:
            out_path = os.path.join(error_dir, f"error_metrics_correlation.{ext}")
            g.fig.savefig(out_path, dpi=300, bbox_inches="tight")
            logger.info(f": {out_path}")
        plt.close(g.fig)

    # ----  ----
    stats = {
        "mean_RMSD": mean_rmsd,
        "mean_TM-score": mean_tm,
        "mean_Wasserstein": valid_df["Wasserstein"].mean() if "Wasserstein" in valid_df.columns else np.nan
    }
    stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
    stats_df.to_csv(os.path.join(error_dir, "error_statistics.csv"), index=False)

    logger.info(f"RMSD: {mean_rmsd:.4f} Å | TM-score: {mean_tm:.4f} | Wasserstein: {stats['mean_Wasserstein']:.4f}")
    return stats_df


def analyze_original_vs_mirror(metrics_df, output_dir):
    """ vs """
    logger.info("Original Structure vs ...")

    mirror_dir = os.path.join(output_dir, "original_vs_mirror_analysis")
    os.makedirs(mirror_dir, exist_ok=True)

    metrics_df["is_mirror"] = metrics_df["id"].str.contains("mirrored", na=False)
    metrics_df["label"] = metrics_df["is_mirror"].map({False: "Original", True: "Mirror"})

    metrics_df["RMSD"] = clean_numeric_column(metrics_df["RMSD"])
    metrics_df["TM-score"] = clean_numeric_column(metrics_df["TM-score"])
    if "Wasserstein" in metrics_df.columns:
        metrics_df["Wasserstein"] = clean_numeric_column(metrics_df["Wasserstein"])

    for metric in ["RMSD", "TM-score", "Wasserstein"]:
        if metric in metrics_df.columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x="label", y=metric, data=metrics_df,
                        hue="label", palette=["#FF6F61", "#6B5B95"], legend=False)
            grouped = metrics_df.groupby("label")[metric].agg(["mean", "std"]).reset_index()
            for i, row in grouped.iterrows():
                plt.text(i, row["mean"], f"{row['mean']:.2f} ± {row['std']:.2f}",
                         ha="center", va="bottom", fontsize=14, weight="bold", color="black")
            plt.title(f"{metric} Comparison: Original vs Mirror", weight="bold")
            plt.xlabel("")
            plt.ylabel(metric, weight="bold")
            plt.tight_layout()
            for ext in ["png", "pdf"]:
                plt.savefig(os.path.join(mirror_dir, f"{metric.lower()}_original_vs_mirror.{ext}"),
                            dpi=300, bbox_inches="tight")
            plt.close()

    logger.info(f" vs ,in: {mirror_dir}")


def add_protein_length_info(metrics_df, pdb_dir, output_dir):
    """
    (、 .gz、 id ).
    - metrics_df:  'id'  'protein_id'  DataFrame
    - pdb_dir: /FASTA 
    - output_dir:  CSV
    """
    logger.info("( +  .gz)...")
    import gzip  # 

    length_dict = {}
    file_count = 0
    sample_files = []

    # 1) ,
    for root, _, files in os.walk(pdb_dir):
        for fname in sorted(files):
            file_count += 1
            if len(sample_files) < 40:
                sample_files.append(fname)
            full = os.path.join(root, fname)
            low = fname.lower()

            # :.pdb .ent .cif  .gz
            if low.endswith(('.pdb', '.ent', '.cif', '.pdb.gz', '.ent.gz', '.cif.gz')):
                try:
                    opener = gzip.open if low.endswith('.gz') else open
                    ca_count = 0
                    with opener(full, 'rt', encoding='utf-8', errors='ignore') as fh:
                        for line in fh:
                            #  PDB  CA rows
                            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                                ca_count += 1
                    if ca_count <= 0:
                        continue

                    # stem , '10gs'  '10gs-some'
                    stem = re.sub(r'(\.pdb|\.ent|\.cif)(\.gz)?$', '', low)
                    # (/、split 、4-6 / id )
                    variants = set()
                    variants.add(stem)
                    variants.add(stem.replace('.pdb', ''))
                    variants.add(stem.split('_')[0])
                    variants.add(stem.split('-')[0])
                    variants.add(stem.split('.')[0])
                    # ( CSV  protein_id )
                    variants.add(stem + '.pdb')
                    variants.add(stem + '.ent')
                    variants.add(stem + '.cif')
                    # ,
                    if '|' in stem:
                        parts = stem.split('|')
                        variants.add(parts[-1])
                    # 4-6 ( PDB id)
                    m = re.match(r'([a-z0-9]{4,6})', stem)
                    if m:
                        variants.add(m.group(1))

                    #  length_dict(in key in,)
                    variants_list = sorted(list(variants))
                    for v in variants:
                        if v and v not in length_dict:
                            length_dict[v] = ca_count
                except Exception as e:
                    logger.debug(f"failed: {full} -> {e}")
                    continue

            # FASTA (.fasta .fa .fna + .gz)
            elif low.endswith(('.fasta', '.fa', '.fna', '.fasta.gz', '.fa.gz', '.fna.gz')):
                try:
                    opener = gzip.open if low.endswith('.gz') else open
                    with opener(full, 'rt', encoding='utf-8', errors='ignore') as fh:
                        for rec in SeqIO.parse(fh, 'fasta'):
                            rid = rec.id.split()[0].lower()
                            variants = {rid, rid.split('_')[0], rid.split('-')[0], rid + '.pdb', rid.replace('|','')}
                            if '|' in rid:
                                parts = rid.split('|')
                                variants.add(parts[-1])
                                if len(parts) > 1:
                                    variants.add(parts[1])
                            m = re.match(r'([a-z0-9]{4,6})', rid)
                            if m:
                                variants.add(m.group(1))
                            for v in variants:
                                if v and v not in length_dict:
                                    length_dict[v] = len(rec.seq)
                except Exception as e:
                    logger.debug(f" FASTA failed: {full} -> {e}")
                    continue

    logger.info(f" {pdb_dir} ,: {file_count} ")
    logger.info(f"( 40):{sample_files[:10]} ...")
    logger.info(f" length : {len(length_dict)}")

    # 2)  metrics_df  protein_key,
    def resolve_length_for_pid(pid):
        """ length_dict , None"""
        if pd.isna(pid):
            return None
        s = str(pid).lower().strip()
        cand = []
        cand.append(s)
        # 
        cand.append(re.sub(r'\.pdb$|\.ent$|\.cif$|\.fasta$|\.fa$|\.fna$', '', s))
        # / .pdb
        if not s.endswith('.pdb'):
            cand.append(s + '.pdb')
        cand.append(s.split()[0])
        cand.append(s.split('|')[-1] if '|' in s else s)
        cand.append(s.split('_')[0])
        cand.append(s.split('-')[0])
        m = re.match(r'([a-z0-9]{4,6})', s)
        if m:
            cand.append(m.group(1))
        #  chain , '1abc_a' -> '1abc'
        cand.append(re.sub(r'_[a-z0-9]$', '', s))

        #  value
        for c in cand:
            if not c:
                continue
            if c in length_dict:
                return length_dict[c]
        return None

    # metrics_df  protein_id( CSV ), id 
    if "protein_id" in metrics_df.columns:
        pid_series = metrics_df["protein_id"]
    else:
        #  id 
        pid_series = metrics_df["id"]

    # ,
    metrics_df["protein_id_raw_for_length"] = pid_series.astype(str)

    # rows(,)
    lengths = []
    unmatched_set = set()
    for idx, pid in metrics_df["protein_id_raw_for_length"].items():
        ln = resolve_length_for_pid(pid)
        lengths.append(ln)
        if ln is None:
            #  id ,
            unmatched_set.add(str(pid).lower())

    metrics_df["length"] = pd.Series(lengths, index=metrics_df.index)

    matched = metrics_df["length"].notna().sum()
    logger.info(f" {matched} / {len(metrics_df)} ")

    # , 20 ()
    if matched < max(1, len(metrics_df) * 0.01):  #  < 1% ()
        sample_unmatched = list(unmatched_set)[:20]
        logger.warning(f"({matched}/{len(metrics_df)}). ID( 20):{sample_unmatched}")

    # 3)  CSV
    out_path = os.path.join(output_dir, "combined_metrics_with_length.csv")
    try:
        metrics_df.to_csv(out_path, index=False)
        logger.info(f" metrics : {out_path}")
    except Exception as e:
        logger.warning(f" CSV failed: {e}")

    return metrics_df


def analyze_by_protein_length(metrics_df, output_dir):
    """"""
    logger.info("...")

    if "length" not in metrics_df.columns or metrics_df["length"].isna().all():
        logger.error("metrics_df  'length' ")
        return None

    length_dir = os.path.join(output_dir, "length_group_analysis")
    os.makedirs(length_dir, exist_ok=True)

    bins = [0, 200, 400, np.inf]
    labels = ["Small (<200)", "Medium (200-400)", "Large (>400)"]
    metrics_df["length_group"] = pd.cut(metrics_df["length"], bins=bins, labels=labels)

    valid_df = metrics_df[~metrics_df["id"].str.contains("mirrored", na=False)].copy()
    valid_df = valid_df.dropna(subset=["RMSD", "TM-score", "length_group"])

    stats_summary = []
    for metric in ["RMSD", "TM-score", "Wasserstein"]:
        if metric in valid_df.columns:
            plt.figure(figsize=(8, 6))
            ax = sns.boxplot(x="length_group", y=metric, hue="length_group",
                             data=valid_df, palette="Set2", dodge=False)
            if ax.get_legend() is not None:
                ax.get_legend().remove()

            plt.title(f"{metric} by Protein Length Group", fontsize=18, weight="bold")
            plt.xlabel("Protein Length Group", fontsize=16, weight="bold")
            plt.ylabel(metric, fontsize=16, weight="bold")
            plt.xticks(fontsize=14, weight="bold")
            plt.yticks(fontsize=14, weight="bold")

            grouped = valid_df.groupby("length_group", observed=True)[metric].mean().reset_index()
            for i, row in grouped.iterrows():
                plt.text(i, row[metric], f"{row[metric]:.3f}",
                         ha="center", va="bottom", fontsize=12, weight="bold", color="black")

            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(length_dir, f"{metric.lower()}_by_length.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

            grouped_stats = valid_df.groupby("length_group", observed=True)[metric].agg(
                ["mean", "std", "median", "count"]).reset_index()
            grouped_stats = grouped_stats.rename(columns={
                "mean": f"{metric}_mean",
                "std": f"{metric}_std",
                "median": f"{metric}_median",
                "count": f"{metric}_count"
            })
            stats_summary.append(grouped_stats)

    if stats_summary:
        stats_df = stats_summary[0]
        for df in stats_summary[1:]:
            stats_df = pd.merge(stats_df, df, on="length_group", how="outer")
        stats_path = os.path.join(length_dir, "protein_length_group_stats.csv")
        stats_df.to_csv(stats_path, index=False)
        logger.info(f": {stats_path}")
        return stats_df
    else:
        logger.warning("")
        return None


def generate_summary_report(output_dir, error_stats, length_stats):
    """"""
    logger.info("...")

    report_dir = os.path.join(output_dir, "summary_report")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "analysis_summary.txt")

    try:
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Protein Structure Reconstruction Analysis Summary Report\n")
            f.write("="*80 + "\n\n")
            f.write("1. Overall Reconstruction Performance\n")
            f.write("-"*60 + "\n")
            if error_stats is not None:
                f.write(f"  - Average RMSD: {error_stats.loc[error_stats['Metric']=='mean_RMSD','Value'].values[0]:.4f} Å\n")
                f.write(f"  - Average TM-score: {error_stats.loc[error_stats['Metric']=='mean_TM-score','Value'].values[0]:.4f}\n")
                if 'mean_Wasserstein' in error_stats['Metric'].values:
                    f.write(f"  - Average Wasserstein Distance: {error_stats.loc[error_stats['Metric']=='mean_Wasserstein','Value'].values[0]:.4f}\n")
            else:
                f.write("  No error distribution data available\n")

            f.write("\n2. Length Group Analysis Summary\n")
            f.write("-"*60 + "\n")
            if length_stats is not None:
                for _, row in length_stats.iterrows():
                    f.write(f"  - {row['length_group']}: RMSD={row['RMSD_mean']:.4f} Å, TM-score={row['TM-score_mean']:.4f}\n")
            else:
                f.write("  No length group data available\n")

            f.write("\n3. Recommendations\n")
            f.write("-"*60 + "\n")
            f.write("  - Analyze proteins in worst-performing groups\n")
            f.write("  - Consider group-specific model tuning\n")
            f.write("\n" + "="*80 + "\n")
            f.write("End of Report\n")

        logger.info(f": {report_path}")
    except Exception as e:
        logger.warning(f"failed: {e}")

    return report_path


# ========= Main function =========
def main():
    # =====================
    # Dynamic path configuration
    # =====================
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    output_dir = os.path.join(PROJECT_ROOT, "Research_results/Step4.3_analyze_error_distribution_and_fold_class")
    metrics_file = os.path.join(PROJECT_ROOT, "Research_results/Step4.1_and_4.2_topological_signature_modeling/final_results/combined_metrics.csv")
    pdb_dir = os.path.join(PROJECT_ROOT, "Research_results/step1.0_Human_PDB_Data")

    os.makedirs(output_dir, exist_ok=True)

    try:
        metrics_df = pd.read_csv(metrics_file)
        logger.info(f"Loading,  {len(metrics_df)} ")
    except Exception as e:
        logger.error(f"Loadingresults failed: {e}")
        return

    metrics_df = add_protein_length_info(metrics_df, pdb_dir, output_dir)

    error_stats = analyze_error_distribution(metrics_df, output_dir)
    analyze_original_vs_mirror(metrics_df, output_dir)
    length_stats = analyze_by_protein_length(metrics_df, output_dir)
    generate_summary_report(output_dir, error_stats, length_stats)

    logger.info(f"!in: {output_dir}")


if __name__ == "__main__":
    main()
