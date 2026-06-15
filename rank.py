#!/usr/bin/env python3
"""
Redrob Hackathon — Candidate Ranking Pipeline Wrapper
=====================================================

This script acts as the root wrapper to execute the main ranker located in the `vrd/` folder.
It satisfies the hackathon CLI requirement:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import sys
import os
import argparse
from pathlib import Path

# Resolve paths
ROOT_DIR = Path(__file__).resolve().parent
VRD_DIR = ROOT_DIR / "vrd"

# Append vrd folder to sys.path so we can import its modules
sys.path.append(str(VRD_DIR))

# Import main ranker run function
try:
    from main_ranker import run as run_ranker
except ImportError as e:
    print(f"ERROR: Could not import main_ranker from vrd folder: {e}", file=sys.stderr)
    sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redrob Hackathon — Candidate Ranking Pipeline"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("candidates.jsonl"),
        help="Path to candidates.jsonl (default: ./candidates.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("submission.csv"),
        help="Output CSV path (default: ./submission.csv)",
    )
    args = parser.parse_args()

    # Resolve paths to absolute paths
    candidates_path = args.candidates.resolve()
    out_path = args.out.resolve()

    if not candidates_path.exists():
        print(f"ERROR: candidates file not found at: {candidates_path}", file=sys.stderr)
        sys.exit(1)

    jd_embed_path = VRD_DIR / "jd_embeddings.npz"
    jd_meta_path = VRD_DIR / "jd_metadata.json"

    # Verify pre-computed files exist
    if not jd_embed_path.exists() or not jd_meta_path.exists():
        print("ERROR: Pre-computed JD embeddings or metadata not found in vrd/ directory.", file=sys.stderr)
        print("Please ensure you have generated them first using JD_parser.py or the UI.", file=sys.stderr)
        sys.exit(1)

    print(f"[rank.py] Launching main ranker...")
    run_ranker(
        candidates_path=str(candidates_path),
        jd_embed_path=str(jd_embed_path),
        jd_meta_path=str(jd_meta_path),
        out_path=str(out_path)
    )

if __name__ == "__main__":
    main()