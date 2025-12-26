import os
import gzip
import gc
import time
import traceback
import argparse
import numpy as np
from Bio import PDB
from ripser import ripser
from sklearn.decomposition import PCA

# Restrict underlying multi-threaded libraries to use only 1 thread
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ===============================================================
# Sampling and diagnostic configuration
# ===============================================================
MAX_CA_POINTS = 1000 
ENABLE_DIAGNOSTIC = True

# ===============================================================
# Core functions
# ===============================================================
def extract_ca_coords(pdb_path):
    """Extract CA atom coordinates from .pdb.gz file"""
    parser = PDB.PDBParser(QUIET=True)
    with gzip.open(pdb_path, "rt") as f:
        struct = parser.get_structure("prot", f)
    coords = [atom.coord for atom in struct.get_atoms() if atom.get_name() == "CA"]
    return np.array(coords)


def compute_persistent_homology(coords):
    """Compute persistent homology (H0, H1 only)"""
    if coords.shape[0] < 3:
        raise ValueError(
            f"Only {coords.shape[0]} CA atoms found; cannot compute persistent homology"
        )
    return ripser(coords, maxdim=1, thresh=6.0)


def normalize_and_align(coords):
    """Normalize via PCA and align to 3D"""
    coords = coords - coords.mean(axis=0)
    norm = np.max(np.linalg.norm(coords, axis=1))
    if norm == 0:
        raise ValueError("After centering, all points coincide; variance is 0")
    coords /= norm
    return PCA(n_components=3).fit_transform(coords)


def save_results(pdb_file, aligned_coords, tda_result, aligned_folder, tda_folder):
    """Save aligned coordinates and TDA results"""
    base = os.path.splitext(pdb_file)[0]

    aligned_path = os.path.join(aligned_folder, f"{base}_aligned_coords.txt")
    np.savetxt(aligned_path, aligned_coords, fmt="%.6f", header="X, Y, Z")

    tda_path = os.path.join(tda_folder, f"{base}_tda_result.txt")
    with open(tda_path, "w", encoding="utf-8") as f:
        for dim, dgms in enumerate(tda_result["dgms"]):
            f.write(f"Dimension {dim}:\n")
            for d in dgms:
                f.write(f"{d}\n")
            f.write("\n")


# ===============================================================
# Main processing logic (FINAL CLEANUP)
# ===============================================================
def main(data_folder, start_idx, end_idx):
    data_folder = data_folder.strip().strip('"').strip("'")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir) 
    
    aligned_folder = os.path.join(
        parent_dir,
        "Research_results/step2.0_Pretreatment_results/step2.0_aligned_coords"
    )
    tda_folder = os.path.join(
        parent_dir,
        "Research_results/step2.0_Pretreatment_results/step2.0_tda_results"
    )

    os.makedirs(aligned_folder, exist_ok=True)
    os.makedirs(tda_folder, exist_ok=True)

    print(f"\n🚀 Data Folder: {data_folder}")
    print(f"🚀 Processing index range: {start_idx} to {end_idx}")

    # List all .pdb.gz files
    try:
        all_pdb_files = sorted(
            f for f in os.listdir(data_folder) if f.endswith(".pdb.gz")
        )
    except OSError as e:
        print(f"❌ Error: Unable to access data directory {data_folder}, Please check the path and permissions.")
        raise e
        
    pdb_files = all_pdb_files[start_idx:end_idx]

    total = len(pdb_files)
    processed = 0
    no_ca = []
    failures = []

    for idx, pdb_file in enumerate(pdb_files, start=1):
        pdb_path = os.path.join(data_folder, pdb_file)
        try:
            t0 = time.perf_counter()
            ca_coords = extract_ca_coords(pdb_path)
            t1 = time.perf_counter()

            if ca_coords.size == 0:
                raise ValueError("No CA atoms found")

            orig_n = ca_coords.shape[0]
            if orig_n > MAX_CA_POINTS:
                sel = np.random.choice(orig_n, MAX_CA_POINTS, replace=False)
                sample_coords = ca_coords[sel]
                print(f"⚙️ Sampling CA: Original {orig_n} → Sampled {MAX_CA_POINTS}")
            else:
                sample_coords = ca_coords
            t2 = time.perf_counter()

            tda_res = compute_persistent_homology(sample_coords)
            t3 = time.perf_counter()

            aligned = normalize_and_align(ca_coords)
            t4 = time.perf_counter()

            save_results(pdb_file, aligned, tda_res, aligned_folder, tda_folder)
            t5 = time.perf_counter()

            processed += 1
            print(f"✅ [{processed}/{total}] {pdb_file} completed")

            if ENABLE_DIAGNOSTIC:
                print(
                    f"    ⌛ parse={(t1-t0):.2f}s, "
                    f"sample={(t2-t1):.2f}s, "
                    f"tda={(t3-t2):.2f}s, "
                    f"PCA={(t4-t3):.2f}s, "
                    f"save={(t5-t4):.2f}s"
                )

        except Exception as e:
            tb = traceback.format_exc()
            if "CA atoms" in str(e) or "Only" in str(e):
                no_ca.append(pdb_file)
                print(f"⚠️ Skipping {pdb_file}: {e}")
            else:
                failures.append((pdb_file, tb))
                print(f"❌ Error processing {pdb_file}:\n{tb}")

        finally:
            gc.collect()

    log_path = os.path.join(
        parent_dir,
        f"Research_results/PDB_pretreatment_results_log_summary_{start_idx}-{end_idx}.txt" 
    )

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"Processed Index Range: {start_idx} to {end_idx}\n")
        log.write(f"✅ Successfully processed: {processed}/{total}\n\n")
        log.write(f"⚠️ Missing CA atoms: {len(no_ca)}\n")
        for fn in no_ca:
            log.write(f" - {fn}\n")
        log.write(f"\n❌ Processing failures: {len(failures)}\n")
        for fn, tb in failures:
            log.write(f" - {fn}\n{tb}\n\n")

    print(f"\n📄 Log saved to: {log_path}")


# ===============================================================
# Argument parsing (UNCHANGED)
# ===============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 2.0 PDB pretreatment with persistent homology"
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Folder containing .pdb.gz files"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        required=True,
        help="Start index (inclusive)"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        required=True,
        help="End index (exclusive)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.start_idx, args.end_idx)