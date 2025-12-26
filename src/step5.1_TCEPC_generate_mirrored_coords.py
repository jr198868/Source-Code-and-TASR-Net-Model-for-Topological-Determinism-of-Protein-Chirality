import os
import gzip
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm
# TDA Characterization Enhanced Protein Chirality

# =====================
# Dynamic path configuration
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configure paths
pdb_folder = os.path.join(PROJECT_ROOT, 'Human_PDB_Data')
output_folder = os.path.join(PROJECT_ROOT, 'Research_results/step5.1_Human_PDB_Data_Coords')
index_file = os.path.join(output_folder, 'coords_index.csv')
error_log_file = os.path.join(output_folder, 'error.log')

# Auto-create output directory
os.makedirs(output_folder, exist_ok=True)

# Initialize PDB parser
parser = PDBParser(QUIET=True)

# Store coordinate file index records
index_records = []

# Find all compressed PDB files
pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith('.pdb.gz')]

# Process each PDB file
for pdb_file in tqdm(pdb_files, desc='Processing PDBs'):
    try:
        protein_id = pdb_file.replace('.pdb.gz', '')
        pdb_path = os.path.join(pdb_folder, pdb_file)

        # Load PDB structure from compressed file
        with gzip.open(pdb_path, 'rt') as handle:
            structure = parser.get_structure(protein_id, handle)

            # Extract CA atom coordinates
            coords = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            atom = residue['CA']
                            coords.append(atom.coord)

            coords = np.array(coords)
            if coords.shape[0] == 0:
                raise ValueError(f"{protein_id} has no CA atoms.")

            # Save original coordinates
            original_path = os.path.join(output_folder, f"{protein_id}_coords.npy")
            np.save(original_path, coords)

            # Create mirror coordinates (flip X-axis)
            mirror_coords = coords.copy()
            mirror_coords[:, 0] *= -1
            mirror_path = os.path.join(output_folder, f"{protein_id}_mirror_coords.npy")
            np.save(mirror_path, mirror_coords)

            # Record file paths
            index_records.append({
                'id': protein_id,
                'label': 0,
                'coords_path': original_path
            })
            index_records.append({
                'id': protein_id + '_mirror',
                'label': 1,
                'coords_path': mirror_path
            })

    except Exception as e:
        with open(error_log_file, 'a') as log:
            log.write(f"Error processing {pdb_file}: {str(e)}\n")

# Save coordinate index to CSV
df_index = pd.DataFrame(index_records)
df_index.to_csv(index_file, index=False)

print(f"\n✅ Completed: {len(index_records)//2} protein pairs processed")
print(f"   Index saved to: {index_file}")
