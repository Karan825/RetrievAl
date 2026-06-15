"""Robust downloader for the models from Hugging Face.

This script manages downloading:
1. BAAI/bge-small-en-v1.5 (using snapshot_download)
2. Qwen/Qwen2.5-1.5B-Instruct-GGUF (using hf_hub_download)
It handles retries, environment tokens, and relative directory paths.
"""

import os
import time
import traceback
from huggingface_hub import snapshot_download, hf_hub_download

VRD_DIR = os.path.dirname(os.path.abspath(__file__))

def ensure_models_exist(progress_callback=None):
    """
    Checks if BGE and Qwen models exist locally. Downloads them if missing.
    progress_callback is an optional function that accepts a string message (for UI integrations).
    """
    bge_dir = os.path.join(VRD_DIR, "local_bge_model")
    qwen_path = os.path.join(VRD_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    
    # Verify BGE (checks for typical file existence like config.json)
    bge_missing = not os.path.exists(bge_dir) or not os.path.exists(os.path.join(bge_dir, "config.json"))
    # Verify Qwen (checks for existence and minimum size to ensure it's not a corrupted/empty download)
    qwen_missing = not os.path.exists(qwen_path) or os.path.getsize(qwen_path) < 100 * 1024 * 1024
    
    if not bge_missing and not qwen_missing:
        return True

    HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    MAX_ATTEMPTS = 5
    
    if bge_missing:
        msg = "Downloading BGE-small embedding model locally..."
        print(msg)
        if progress_callback:
            progress_callback(msg)
            
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                snapshot_download(
                    repo_id="BAAI/bge-small-en-v1.5",
                    local_dir=bge_dir,
                    token=HF_TOKEN,
                )
                print("BGE model download complete!")
                break
            except Exception as exc:
                print(f"Attempt {attempt} for BGE failed: {exc}")
                if attempt >= MAX_ATTEMPTS:
                    raise exc
                time.sleep(2 ** attempt)
                
    if qwen_missing:
        msg = "Downloading Qwen2.5-1.5B-Instruct-GGUF model locally (~1.1 GB)..."
        print(msg)
        if progress_callback:
            progress_callback(msg)
            
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                # Attempt lowercase filename
                try:
                    hf_hub_download(
                        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
                        local_dir=VRD_DIR,
                        token=HF_TOKEN,
                    )
                except Exception:
                    # Attempt uppercase filename
                    hf_hub_download(
                        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                        filename="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
                        local_dir=VRD_DIR,
                        token=HF_TOKEN,
                    )
                    # Rename to lowercase
                    downloaded_file = os.path.join(VRD_DIR, "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
                    if os.path.exists(downloaded_file):
                        if os.path.exists(qwen_path):
                            os.remove(qwen_path)
                        os.rename(downloaded_file, qwen_path)
                
                print("Qwen model download complete!")
                break
            except Exception as exc:
                print(f"Attempt {attempt} for Qwen failed: {exc}")
                if attempt >= MAX_ATTEMPTS:
                    raise exc
                time.sleep(2 ** attempt)
                
    return True

if __name__ == "__main__":
    try:
        ensure_models_exist()
        print("\nAll models are present and validated!")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        traceback.print_exc()
