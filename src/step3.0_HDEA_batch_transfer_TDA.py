import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

# HDEA: High Dimensional Embedding Analysis

# ==============================================================
# 1. Path configuration
# ==============================================================
# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

tda_folder = os.path.join(project_root, "Research_results", "step2.0_Pretreatment_results", "step2.0_tda_results")
output_dir = os.path.join(project_root, "Research_results", "step3.0_tda_features")
output_csv_path = os.path.join(output_dir, 'step3.0_tda_features.csv')

# Ensure the output folder exists
os.makedirs(output_dir, exist_ok=True)

# ==============================================================
# 2. TDA Feature reading function
# ==============================================================
def read_tda_features(file_path):
    features = {}
    current_dim = None
    barcode = {}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            #  Dimension 0, 1, 2...
            dim_match = re.match(r'Dimension\s+(\d+):', line)
            if dim_match:
                current_dim = int(dim_match.group(1))
                barcode[current_dim] = []
                continue
                
            if current_dim is not None:
                clean_line = line.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
                parts = clean_line.split()
                
                if len(parts) == 2:
                    try:
                        birth = float(parts[0])
                        if 'inf' in parts[1].lower():
                            continue
                        death = float(parts[1])
                        
                        if death > birth:
                            barcode[current_dim].append((birth, death))
                    except ValueError:
                        continue

    # Feature extraction: Dimensions 0-2 (H0, H1, H2)
    for dim in range(3):
        bars = barcode.get(dim, [])
        # compute Persistence = Death - Birth
        persistences = [death - birth for birth, death in bars]
        
        features[f'dim{dim}_count'] = len(persistences)
        features[f'dim{dim}_mean'] = np.mean(persistences) if persistences else 0.0
        features[f'dim{dim}_max'] = np.max(persistences) if persistences else 0.0
        features[f'dim{dim}_sum'] = np.sum(persistences) if persistences else 0.0

    return features

# ==============================================================
# 3. Batch processing logic
# ==============================================================
if __name__ == "__main__":
    data = []

    print(f"📂 Reading data from: {tda_folder}")
    
    if not os.path.exists(tda_folder):
        print(f"❌ Error: Cannot find directory {tda_folder}")
    else:
        all_raw_files = os.listdir(tda_folder)
        all_files = [f for f in all_raw_files if f.endswith('_tda_result.txt') and '.pdb' in f]
        
        print(f"🚀 Found {len(all_files)} valid PDB TDA files. Starting extraction...")

        for filename in tqdm(all_files, desc="Processing progress", unit="files"):
            file_id = filename.replace('_tda_result.txt', '')
            file_path = os.path.join(tda_folder, filename)
            
            try:
                feature_dict = read_tda_features(file_path)
                feature_dict['ID'] = file_id
                data.append(feature_dict)
            except Exception as e:
                print(f" ⚠️ Error processing {filename}: {e}")

        # ---------- Convert to DataFrame and save ----------
        if data:
            tda_df = pd.DataFrame(data)
            tda_df = tda_df.set_index('ID')
            
            tda_df.to_csv(output_csv_path)
            print(f"\n✅ Feature extraction complete!")
            print(f"📊 Number of records processed: {len(tda_df)}")
            print(f"📁 Results saved to: {output_csv_path}")
        else:
            print("❌ No valid data was found, no CSV was generated.")