import os
import subprocess
import sys

def main():
    print("=========================================================")
    print("   REDROB HACKATHON - DYNAMIC PIPELINE ORCHESTRATOR      ")
    print("=========================================================")
    
    # Force the working directory to be the 'vrd' folder so relative paths work perfectly
    vrd_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(vrd_dir)
    print(f"Working directory set to: {vrd_dir}")
    
    # ---------------------------------------------------------
    # STEP 1: Run the JD Parser
    # ---------------------------------------------------------
    print("\n>>> [STEP 1/2] Running JD Parser (Extracting brain)...")
    parser_result = subprocess.run([sys.executable, "JD_parser.py"])
    
    if parser_result.returncode != 0:
        print("\n[ERROR] JD_parser.py failed. Pipeline stopped.")
        sys.exit(1)
        
    # Check if the files were actually created
    if not os.path.exists("jd_embeddings.npz") or not os.path.exists("jd_metadata.json"):
        print("\n[ERROR] The parser finished but didn't save the embeddings!")
        print("Please make sure you updated JD_parser.py with the save logic.")
        sys.exit(1)

    # ---------------------------------------------------------
    # STEP 2: Run the Main Ranker
    # ---------------------------------------------------------
    print("\n>>> [STEP 2/2] Running High-Speed Ranker (Scoring candidates)...")
    ranker_result = subprocess.run([sys.executable, "main_ranker.py"])
    
    if ranker_result.returncode != 0:
        print("\n[ERROR] main_ranker.py failed. Pipeline stopped.")
        sys.exit(1)
        
    print("\n=========================================================")
    print("  PIPELINE COMPLETE! Your submission.csv is ready.       ")
    print("=========================================================")

if __name__ == "__main__":
    main()
