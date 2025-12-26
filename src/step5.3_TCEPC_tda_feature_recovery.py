import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from ripser import ripser
from persim import PersistenceImager
import gc
# TDA Characterization Enhanced Protein Chirality

# =====================
# Dynamic path configuration
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input paths
FULL_DATASET_PATH = os.path.join(PROJECT_ROOT, "Research_results/step5.2_build_chirality_dataset/full_dataset.pkl")
PARTIAL_PATH = FULL_DATASET_PATH.replace(".pkl", "_partial_progress.pkl")

# Load data (prioritize loading partial)
if os.path.exists(PARTIAL_PATH):
    print(f"Detected checkpoint file, resuming from {PARTIAL_PATH}")
    with open(PARTIAL_PATH, 'rb') as f:
        dataset = pickle.load(f)
else:
    with open(FULL_DATASET_PATH, 'rb') as f:
        dataset = pickle.load(f)

# ✅ TDA ()
def extract_tda_features(coords, max_points=1000):
    if coords.shape[0] > max_points:
        idx = np.random.choice(coords.shape[0], max_points, replace=False)
        coords = coords[idx]

    result = ripser(coords, maxdim=1)
    dgms = result['dgms']

    # dim 0
    dim0 = dgms[0]
    dim0_count = len(dim0)
    dim0_lifetime = dim0[:, 1] - dim0[:, 0]
    dim0_lifetime = dim0_lifetime[np.isfinite(dim0_lifetime)]
    dim0_mean = dim0_lifetime.mean() if len(dim0_lifetime) > 0 else 0.0

    # dim 1
    dim1 = dgms[1]
    dim1_count = len(dim1)
    dim1_lifetime = dim1[:, 1] - dim1[:, 0]
    dim1_lifetime = dim1_lifetime[np.isfinite(dim1_lifetime)]
    dim1_mean = dim1_lifetime.mean() if len(dim1_lifetime) > 0 else 0.0

    # PCA-like stats from image
    try:
        pimgr = PersistenceImager()
        dim1_filtered = dim1[np.isfinite(dim1).all(axis=1)]
        if len(dim1_filtered) == 0:
            raise ValueError(" diagram")
        pimgr.fit(dim1_filtered)
        img = pimgr.transform(dim1_filtered)
        pca1 = img.mean()
        pca2 = img.std()
        pca3 = img.max()
    except:
        pca1, pca2, pca3 = 0.0, 0.0, 0.0

    return [dim0_count, dim0_mean, dim1_count, dim1_mean, pca1, pca2, pca3]

# ✅ :
updated_count = 0
save_every = 500

for idx, sample in enumerate(tqdm(dataset, desc="🔍  TDA")):
    if sample['tda'] is None and sample['coords'] is not None:
        try:
            coords = sample['coords']
            if not np.isfinite(coords).all():
                raise ValueError("coords  inf  nan")

            sample['tda'] = extract_tda_features(coords)
            updated_count += 1

        except Exception as e:
            print(f"❌ {sample['id']} failed: {e}")
            continue

    #  N 
    if updated_count > 0 and updated_count % save_every == 0:
        with open(PARTIAL_PATH, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"💾 Saved {updated_count} samples")
        gc.collect()

# ✅ ()
with open(FULL_DATASET_PATH, 'wb') as f:
    pickle.dump(dataset, f)

# ✅ 
if os.path.exists(PARTIAL_PATH):
    os.remove(PARTIAL_PATH)

print(f"\n✅ , {updated_count}  TDA ")
print(f"✅ :{FULL_DATASET_PATH}")
