import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os
# TDA Characterization Enhanced Protein Chirality

# =====================
# Dynamic path configuration
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build_dataset():
    """
    Build protein structure + TDA feature dataset (for chirality classification)
    Output structure unified to save as full_dataset.pkl
    """

    # Input paths
    coords_dir = os.path.join(PROJECT_ROOT, "Research_results/step5.2_build_chirality_dataset")
    output_index_path = os.path.join(PROJECT_ROOT, "Research_results/step5.1_Human_PDB_Data_Coords/coords_index.csv")
    tda_csv_path = os.path.join(PROJECT_ROOT, "Research_results/step2.2.1_tda_features_with_clusters/step2.2.1_tda_features_with_clusters.csv")
    os.makedirs(coords_dir, exist_ok=True)

    # Read TDA features
    print("Reading TDA features")
    try:
        tda_df = pd.read_csv(tda_csv_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"TDA file reading failed: {e}")
        return

    # Extract protein ID (remove .pdb suffix)
    if 'ID' not in tda_df.columns:
        raise ValueError(f"'ID' column not found in TDA file. Current columns: {tda_df.columns.tolist()}")

    tda_df['id'] = tda_df['ID'].str.replace('.pdb', '', regex=False)

    # Select TDA feature columns for modeling
    feature_columns = [
        'dim0_count', 'dim0_mean',
        'dim1_count', 'dim1_mean',
        'pca1', 'pca2', 'pca3'
    ]

    tda_features = {}
    for _, row in tda_df.iterrows():
        pid = row['id']
        try:
            vec = row[feature_columns].values.astype(np.float32)
            tda_features[pid] = vec
        except Exception as e:
            print(f"Feature conversion failed for protein {pid}: {e}")

    print(f"TDA features loaded: {len(tda_features)}")

    # Read coordinate index file
    print("Reading coordinate index table")
    try:
        index_df = pd.read_csv(output_index_path)
        print("Successfully read coords_index.csv")
        print("First few rows:")
        print(index_df.head())
    except Exception as e:
        print(f"Coordinate index file reading failed: {e}")
        return

    # Build dataset
    dataset = []

    for _, row in tqdm(index_df.iterrows(), total=len(index_df), desc="Building samples"):
        pid = row['id']
        label = int(row['label'])
        coords_path = row['coords_path']

        try:
            coords = np.load(coords_path)
            coords = coords - coords.mean(axis=0)

            # Mirror structure has no TDA
            if '_mirror' in pid:
                tda_vec = None
            else:
                tda_vec = tda_features.get(pid, None)

            dataset.append({
                'id': pid,
                'label': label,
                'coords': coords,
                'tda': tda_vec
            })

        except Exception as e:
            print(f"Coordinate loading failed [{pid}]: {e}")
            continue

    # Save result
    output_path = os.path.join(PROJECT_ROOT, "Research_results/step5.2_build_chirality_dataset/full_dataset.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset building complete: {len(dataset)} samples")
    print(f"Saved to: {output_path}")
    return dataset


# Entry point
if __name__ == "__main__":
    build_dataset()
