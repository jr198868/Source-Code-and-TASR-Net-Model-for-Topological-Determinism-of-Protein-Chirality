import os
import subprocess
import glob
import requests 
import time
import sys

# =====================
# Configuration
# =====================

# 1. Output directory
NEW_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "Research_results", 
    "step1.0_Human_PDB_Data"
)

# 2. Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 3. List directory
LIST_DIR = os.path.join(PROJECT_ROOT, "Download_list_script")

# 4. DOWNLOAD_SCRIPT path
DOWNLOAD_SCRIPT = os.path.join(LIST_DIR, "batch_download.sh") 


# =====================
# Cross-platform download core function (fixed list reading logic)
# =====================

def download_pdb_in_python(pdb_id_list_path, output_dir):
    """
    Downloads PDB files in a cross-platform manner using pure Python.
    Files are downloaded in compressed PDB format (.pdb.gz).
    """
    
    # RCSB PDB download URL
    RCSB_URL_TEMPLATE = "https://files.rcsb.org/download/{}.pdb.gz"
    
    all_pdb_ids = []
    
    try:
        with open(pdb_id_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Separate PDB IDs by commas
                ids_on_line = [
                    pdb_id.strip().upper() 
                    for pdb_id in line.split(',') 
                    if pdb_id.strip() 
                ]
                all_pdb_ids.extend(ids_on_line)
                
    except FileNotFoundError:
        print(f"Error: PDB ID list file not found at {pdb_id_list_path}")
        return 0, 0
    
    pdb_ids = list(set(all_pdb_ids)) # Deduplication ensures that each ID is downloaded only once.
    total_ids = len(pdb_ids)
    success_count = 0
    
    print(f"Attempting to download {total_ids} unique PDB files...")
    
    for i, pdb_id in enumerate(pdb_ids):
        
        # Printing progress to track.
        sys.stdout.write(f"\rDownloading: {i+1}/{total_ids} ({pdb_id})")
        sys.stdout.flush()
        
        url = RCSB_URL_TEMPLATE.format(pdb_id)
        output_file_path = os.path.join(output_dir, f"{pdb_id}.pdb.gz")
        
        # Skip downloaded files to resume interrupted downloads
        if os.path.exists(output_file_path):
            success_count += 1
            continue 

        try:
            # Using requests for streaming download
            # Increase the timeout period because file downloads may take time.
            response = requests.get(url, stream=True, timeout=60) 
            response.raise_for_status() 

            # Write the content to the output directory
            with open(output_file_path, 'wb') as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    handle.write(chunk)
            
            success_count += 1
            # Increase latency to avoid overloading the RCSB server.
            time.sleep(0.1) 

        except requests.exceptions.RequestException as e:
            if '404' in str(e):
                 sys.stdout.write(f"\rWarning: PDB ID {pdb_id} not found (404). Skipping.               ")
                 sys.stdout.flush()
            else:
                sys.stdout.write(f"\rError downloading {pdb_id}: {e}                                      ")
                sys.stdout.flush()

            time.sleep(0.5) 

    print(f"\n✅ Finished processing {os.path.basename(pdb_id_list_path)}.")
    return success_count, total_ids

# =====================
# Main Function (LOCKED)
# =====================
def main():
    """
    Batch download human protein structures from RCSB PDB
    """ 
    # makesure NEW_OUTPUT_DIR exist
    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)
    
    # Find all PDB ID list files
    list_files = sorted(glob.glob(os.path.join(LIST_DIR, "rcsb_pdb_ids_*.txt")))
    
    if not list_files:
        print(f"No PDB ID list files found in {LIST_DIR}")
        return
    
    total_downloaded = 0
    total_attempted = 0
    
    for list_file in list_files:
        print(f"\nProcessing: {os.path.basename(list_file)}")
        
        try:
            success, total = download_pdb_in_python(list_file, NEW_OUTPUT_DIR)
            total_downloaded += success
            total_attempted += total
            
        except Exception as e:
            print(f"Error processing {os.path.basename(list_file)}: {e}")
            continue
            
    print(f"\n--- Summary ---")
    print(f"Total files downloaded successfully: {total_downloaded}")
    print(f"Total IDs processed: {total_attempted}")
    print(f"Download Directory: {NEW_OUTPUT_DIR}")


if __name__ == "__main__":
    main()