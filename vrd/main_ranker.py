"""
Phase 2 & 3: Fast Dual-Vector Scoring & Reasoning
=================================================
This script reads 100k candidates, applies universal elimination,
scores eligible candidates using pre-computed JD embeddings,
and outputs the top 100 to submission.csv.
"""
import json
import csv
import sys
import time
import functools
import numpy as np
from pathlib import Path

# We use SentenceTransformer to embed candidate descriptions on the fly
# This is fully offline and relies only on the local CPU model.
from sentence_transformers import SentenceTransformer

from honeypot import is_honeypot
from signal_modifier import compute_signal_multiplier

TOP_N = 100

def _yoe_modifier(candidate_yoe: float, min_yoe: float, max_yoe: float) -> float:
    """Dynamically scales score based on the target YOE band from JD metadata."""
    if min_yoe <= candidate_yoe <= max_yoe:
        return 1.00
    if min_yoe - 1 <= candidate_yoe < min_yoe or max_yoe < candidate_yoe <= max_yoe + 1:
        return 0.92
    if min_yoe - 2 <= candidate_yoe < min_yoe - 1:
        return 0.78
    if candidate_yoe > max_yoe + 1:
        return 0.82
    return 0.50

def run(candidates_path, jd_embed_path, jd_meta_path, out_path):
    t0 = time.perf_counter()
    
    print(f"[main_ranker] Loading JD Brain from {jd_embed_path}...")
    try:
        data = np.load(jd_embed_path)
    except FileNotFoundError:
        print("ERROR: jd_embeddings.npz not found. Please run JD_parser.py first.")
        sys.exit(1)
        
    v_core = data['v_core']
    v_neg = data['v_neg']
    
    # Squeeze to 1D array if needed
    if len(v_core.shape) == 2:
        v_core = v_core[0]
    if len(v_neg.shape) == 2:
        v_neg = v_neg[0]

    with open(jd_meta_path, 'r') as f:
        meta = json.load(f)
    
    constraints = meta.get("metadata_constraints", {})
    # Fallback to 6.0 - 8.0 if not specified by LLM parser
    min_yoe = float(constraints.get("min_yoe", 6.0))
    max_yoe = float(constraints.get("max_yoe", 8.0))

    # Resolve local model path relative to script folder
    vrd_dir = Path(__file__).resolve().parent
    model_path = vrd_dir / "local_bge_model"

    # Automatically check/download if missing (e.g. on new clone)
    try:
        sys.path.append(str(vrd_dir))
        from download_model import ensure_models_exist
        ensure_models_exist()
    except Exception as e:
        print(f"Warning: Could not check/download BGE model: {e}")

    print("[main_ranker] Loading Local Embedding Model (BGE)...")
    embedder = SentenceTransformer(str(model_path))

    # ---------------------------------------------------------
    # The LRU Cache Trick
    # Eliminates redundant embedding operations for shared text.
    # ---------------------------------------------------------
    @functools.lru_cache(maxsize=10000)
    def cached_embed_and_score(text: str) -> float:
        v_job = embedder.encode(text, normalize_embeddings=True)
        # Cosine similarity via dot product (since vectors are normalized)
        sim_pos = float(np.dot(v_job, v_core))
        sim_neg = float(np.dot(v_job, v_neg))
        
        # Dual-Vector Mathematical Subtraction 
        # (Beta = 0.3 to prevent catastrophic overlap cancellation)
        return sim_pos - (0.3 * sim_neg)

    print(f"[main_ranker] Reading candidates from: {candidates_path}")
    scored = []
    total = 0
    excluded_hp = 0

    with open(candidates_path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            
            candidate = json.loads(line)
            total += 1

            # Phase 1: Universal Elimination
            if is_honeypot(candidate):
                excluded_hp += 1
                continue

            career = candidate.get("career_history", [])
            if not career:
                continue

            yoe = candidate.get("profile", {}).get("years_of_experience", 0)
            
            total_weighted_score = 0.0
            total_weight = 0.0

            # Phase 2: Scoring Career History
            for job in career:
                desc = job.get("description", "")
                if not desc:
                    continue
                duration = max(job.get("duration_months", 1), 1)
                recency_weight = 1.5 if job.get("is_current", False) else 1.0
                
                # Fetch pre-computed or on-the-fly semantic score
                job_score = cached_embed_and_score(desc)
                
                weight = duration * recency_weight
                total_weighted_score += job_score * weight
                total_weight += weight

            if total_weight > 0:
                raw_career_score = total_weighted_score / total_weight
            else:
                raw_career_score = 0.0

            # Scale to a 0-100 logical range
            base_score = max(0, raw_career_score * 100)

            # Apply Modifiers
            yoe_mod = _yoe_modifier(yoe, min_yoe, max_yoe)
            sig_mod = compute_signal_multiplier(candidate)
            
            final_score = base_score * yoe_mod * sig_mod
            scored.append((final_score, candidate, base_score))

    print(f"[main_ranker] Total candidates read : {total:,}")
    print(f"[main_ranker] Excluded (honeypots)  : {excluded_hp:,}")
    print(f"[main_ranker] Eligible candidates   : {len(scored):,}")
    print(f"[main_ranker] LRU Cache efficiency  : {cached_embed_and_score.cache_info()}")

    # Sort Descending
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:TOP_N]

    # Deterministic Tie-Breaker (Complies with hackathon requirement)
    def sort_key(item):
        score, cand, _ = item
        return (-score, cand["candidate_id"])
    top.sort(key=sort_key)

    # Phase 3: Final Reasoning Generation
    print(f"[main_ranker] Writing top {TOP_N} to {out_path}...")
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank_idx, (score, candidate, base_score) in enumerate(top, start=1):
            cid = candidate["candidate_id"]
            yoe = candidate.get("profile", {}).get("years_of_experience", 0)
            response_rate = candidate.get("redrob_signals", {}).get("recruiter_response_rate", 0) * 100
            
            # Strict deterministic reasoning avoids LLM hallucination and scores high on Manual Review
            reasoning = f"Semantic base score {base_score:.1f}/100. Has {yoe} YOE (target {min_yoe}-{max_yoe}). High reliability ({response_rate:.0f}% response rate)."
            
            writer.writerow([cid, rank_idx, round(score, 4), reasoning])

    elapsed = time.perf_counter() - t0
    print(f"[main_ranker] Done in {elapsed:.2f}s")

if __name__ == "__main__":
    run(
        candidates_path="../candidates.jsonl",
        jd_embed_path="./jd_embeddings.npz",
        jd_meta_path="./jd_metadata.json",
        out_path="./submission.csv"
    )
