import os
import subprocess
import sys
import shlex 

# =====================
# Dynamic path configuration (GLOBAL)
# =====================
# Root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXECUTABLE = sys.executable # Current Python interpreter 

# DATA_FOLDER
DATA_FOLDER = os.path.join(
    PROJECT_ROOT, 
    "Research_results", 
    "step1.0_Human_PDB_Data"
)

os.makedirs(DATA_FOLDER, exist_ok=True)
print(f"Project Root: {PROJECT_ROOT}")
print(f"Data Folder: {DATA_FOLDER}")


def main():
    """
    Pipeline orchestrator: Execute analysis steps sequentially
    Each step can be commented/uncommented as needed
    """
    
    # Helper function to run a step script
    def run_step(script_name, args=""):
        script_path = os.path.join(PROJECT_ROOT, "src", script_name)
        
        cmd = [PYTHON_EXECUTABLE, script_path] + shlex.split(args)
        
        print(f"\n{'='*60}")
        print(f"Running: {script_name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}\n")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running {script_name}: {e}")
            print("🚨 Subscript execution failed. Please check the logs and error stack in the step2.0 file.")
            return False
        except FileNotFoundError:
            print(f"❌ Script not found: Please ensure'{script_name}' in {os.path.join(PROJECT_ROOT, 'src')}")
            return False
        return True
    
    
    # Step 1: Download data
    # run_step("step1.0_Batch_Download_human_protein_PDB.py")
    
    # Step 2: TDA Pretreatment
    #pretreatment_args = (
    #    f"--data_folder \"{DATA_FOLDER}\" "
    #    f"--start_idx 0 --end_idx 72004"
    #)

    #if run_step("step2.0_TDA_Pretreatment.py", pretreatment_args):
    #    print("✅ Step 2.0 completed successfully.")
    #else:
    #    print("Pipeline terminated due to failure in Step 2.0.")
    #    return 
    
    # Step 3: High-Dimensional Embedding Analysis (HDEA)
    # run_step("step3.0_HDEA_batch_transfer_TDA.py")
    # run_step("step3.1_HDEA_dimensionality_reduction_cluster_analysis.py")
    # run_step("step3.2_HDEA_extract_all_barcode_points.py")
    # run_step("step3.3_HDEA_universal_barcode.py")
    
    # Step 4: Topological Signature Modeling (F3DR: Feature-to-3D Regression)
    # run_step("step4.1_F3DR_high_dim_projection_regression.py", "--start_idx 0 --end_idx 69919")
    # run_step("step4.2_F3DR_combine_results.py")  
    # run_step("step4.3_F3DR_analyze_error_distribution_and_fold_class.py")
    
    # Step 5: TDA for Chirality Prediction (TCEPC)
    # run_step("step5.1_TCEPC_generate_mirrored_coords.py")
    # run_step("step5.2_TCEPC_build_chirality_dataset.py")
    # run_step("step5.3_TCEPC_tda_feature_recovery.py")
    # run_step("step5.4_TCEPC_chirality_model_benchmark.py")
    # run_step("step5.5_TCEPC_spiral_vs_protein_topo.py")
    # run_step("step5.6_TCEPC_chirality_visualization.py")
    # run_step("supplementary_information.py") # Optional
    
    print("\n🎉 Pipeline completed!")


if __name__ == "__main__":
    main()