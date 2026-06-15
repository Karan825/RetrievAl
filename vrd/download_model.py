"""Robust downloader for a Hugging Face snapshot.

This script wraps `snapshot_download` with retries, an optional token, and
clearer error guidance for common network/TLS issues seen on Windows
(e.g. WinError 10054: connection forcibly closed by remote host).

Usage:
  - Optionally set HF_TOKEN or HUGGINGFACE_HUB_TOKEN in your environment for
    authenticated downloads.
  - If you're behind a proxy, set HTTPS_PROXY / HTTP_PROXY environment vars.
  - To run: python download_model.py
"""

import os
import time
import traceback
from huggingface_hub import snapshot_download

print("Downloading bge-small locally...")

# Config
REPO_ID = "BAAI/bge-small-en-v1.5"
LOCAL_DIR = "./local_bge_model"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
TIMEOUT = 60  # seconds for network operations
MAX_ATTEMPTS = 5

for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        # NOTE: `resume_download` is deprecated and ignored by newer versions of
        # huggingface_hub; downloads resume automatically when possible.
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=LOCAL_DIR,
            token=HF_TOKEN,
        )
        print("Download complete!")
        break
    except Exception as exc:
        print(f"Attempt {attempt} failed with error: {exc}")
        if attempt >= MAX_ATTEMPTS:
            print("\nAll attempts failed. Helpful troubleshooting steps:\n")
            print("1) Check your internet connection and that you can open:")
            print(f"   https://huggingface.co/{REPO_ID}\n")
            print("2) If you're behind a corporate proxy or VPN, set the HTTPS_PROXY and HTTP_PROXY environment variables, e.g. (PowerShell):")
            print("   $env:HTTPS_PROXY='http://proxy:port'; $env:HTTP_PROXY='http://proxy:port'\n")
            print("3) Windows antivirus or TLS-intercepting proxies can close TLS connections. Try disabling them temporarily or use a different network.")
            print("4) Upgrade your client libraries: pip install -U huggingface_hub httpx certifi\n")
            print("5) If the model is private, make sure you set HF_TOKEN or run `huggingface-cli login`.")
            print("6) As an alternative, try downloading via the browser or using the huggingface hub website/download UI.")
            print("\nFull exception traceback:\n")
            traceback.print_exc()
            # Re-raise so the process exits with non-zero code like before
            raise
        backoff = 2 ** attempt
        print(f"Retrying in {backoff} seconds... (attempt {attempt+1}/{MAX_ATTEMPTS})")
        time.sleep(backoff)
