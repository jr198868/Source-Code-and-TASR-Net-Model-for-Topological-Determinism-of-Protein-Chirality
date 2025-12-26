import os
import re
import csv

# =====================
# Dynamic path configuration
# =====================
current_script_path = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(current_script_path))

# ---------- Configuration (Relative Paths) ----------
output_dir = os.path.join(PROJECT_ROOT, 'Research_results', 'step3.2_all_barcode_points')
tda_folder = os.path.join(PROJECT_ROOT, 'Research_results', 'step2.0_Pretreatment_results', 'step2.0_tda_results')
output_csv_path = os.path.join(output_dir, 'step3.2_all_barcode_points.csv')

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# ---------- Main function ----------
def extract_barcode_points(file_path, file_id):
    barcode_points = []
    current_dim = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dim_match = re.match(r'Dimension\s+(\d+):', line)
            if dim_match:
                current_dim = int(dim_match.group(1))
            elif current_dim is not None:
                parts = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)
                if len(parts) == 2:
                    birth, death = map(float, parts)
                    barcode_points.append((file_id, current_dim, birth, death))
    return barcode_points

# ---------- Scan all files ----------
if __name__ == "__main__":
    all_points = []
    file_count = 0

    for filename in os.listdir(tda_folder):
        if filename.endswith('_tda_result.txt'):
            file_id = filename.replace('_tda_result.txt', '')
            file_path = os.path.join(tda_folder, filename)
            points = extract_barcode_points(file_path, file_id)
            all_points.extend(points)

            file_count += 1
            if file_count % 1000 == 0:
                print(f"✅ Successfully processed {file_count} files")

    # ---------- Write to CSV ----------
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'dim', 'birth', 'death'])
        writer.writerows(all_points)

    print(f"\n✅ Complete! Extracted {len(all_points)} barcode points, saved to: {output_csv_path}")
