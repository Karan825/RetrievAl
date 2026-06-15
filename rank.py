#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Ranking Pipeline
==========================================================

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints satisfied:
    - ≤ 5 minutes wall-clock time on CPU
    - ≤ 16 GB RAM
    - No GPU required
    - No network calls
    - Deterministic output (same input → same output every run)

Architecture (5 stages):
    1. Honeypot elimination   — removes impossible profiles
    2. Hard constraint filter — removes JD-disqualified candidates
    3. Career relevance score — 44-template description scoring
                                + product company boost
                                + job-hopper penalty
                                + YOE modifier
    4. Signal multiplier      — 23 behavioral signals as a multiplier
    5. Reasoning generation   — fact-grounded 1-2 sentence per candidate
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from ranker.honeypot import is_honeypot
from ranker.filters import passes_hard_constraints
from ranker.career_scorer import compute_career_score
from ranker.signal_modifier import compute_signal_multiplier
from ranker.reasoning import generate_reasoning

# ── Constants ────────────────────────────────────────────────────────────────

TOP_N = 100
EXCLUDE_SCORE = -1.0


# ── Core scoring function ─────────────────────────────────────────────────────

def score_candidate(candidate: dict) -> float:
    """
    Return the composite fit score for a candidate.

    Returns EXCLUDE_SCORE (-1.0) for candidates that must be excluded
    (honeypots or hard-constraint failures).
    """
    if is_honeypot(candidate):
        return EXCLUDE_SCORE

    if not passes_hard_constraints(candidate):
        return EXCLUDE_SCORE

    career_score   = compute_career_score(candidate)
    signal_mult    = compute_signal_multiplier(candidate)

    return career_score * signal_mult


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(candidates_path: Path, out_path: Path) -> None:
    t0 = time.perf_counter()

    print(f"[rank.py] Reading candidates from: {candidates_path}")

    scored: list[tuple[float, dict]] = []
    total = 0
    excluded_honeypot = 0
    excluded_hard = 0

    with candidates_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            candidate = json.loads(line)
            total += 1

            # Fast pre-check: skip if both filters would exclude
            if is_honeypot(candidate):
                excluded_honeypot += 1
                continue

            if not passes_hard_constraints(candidate):
                excluded_hard += 1
                continue

            career_score = compute_career_score(candidate)
            signal_mult  = compute_signal_multiplier(candidate)
            final_score  = career_score * signal_mult

            scored.append((final_score, candidate))

    print(f"[rank.py] Total candidates read       : {total:,}")
    print(f"[rank.py] Excluded (honeypots)        : {excluded_honeypot:,}")
    print(f"[rank.py] Excluded (hard constraints) : {excluded_hard:,}")
    print(f"[rank.py] Eligible candidates         : {len(scored):,}")

    # ── Sort descending by score ───────────────────────────────────────────
    scored.sort(key=lambda x: x[0], reverse=True)

    # ── Take top 100 ──────────────────────────────────────────────────────
    top = scored[:TOP_N]

    # ── Handle score ties: tie-break by candidate_id ascending ────────────
    # Re-sort within tied groups to ensure deterministic, validator-compliant output
    # (validator checks score is non-increasing AND ties break by cand_id asc)
    def sort_key(item):
        score, cand = item
        return (-score, cand["candidate_id"])

    top.sort(key=sort_key)

    # ── Write CSV ──────────────────────────────────────────────────────────
    print(f"[rank.py] Writing top {TOP_N} to: {out_path}")

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank_idx, (score, candidate) in enumerate(top, start=1):
            cid       = candidate["candidate_id"]
            reasoning = generate_reasoning(candidate, score, rank_idx)

            # Round score to 4 decimal places — looks cleaner and avoids
            # floating-point noise in score ordering checks
            writer.writerow([cid, rank_idx, round(score, 4), reasoning])

    elapsed = time.perf_counter() - t0
    print(f"[rank.py] Done in {elapsed:.2f}s  OK")

    # Sanity check - print top 10
    print("\n=== Top 10 Candidates ===")
    print(f"{'Rank':<5} {'Candidate ID':<15} {'Score':<10} {'Name':<20} {'Title':<30} {'YOE'}")
    print("-" * 100)
    for rank_idx, (score, cand) in enumerate(top[:10], start=1):
        p = cand["profile"]
        print(
            f"{rank_idx:<5} {cand['candidate_id']:<15} {score:<10.4f} "
            f"{p.get('anonymized_name',''):<20} "
            f"{p.get('current_title',''):<30} "
            f"{p.get('years_of_experience',0):.1f}"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

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

    if not args.candidates.exists():
        print(f"ERROR: candidates file not found: {args.candidates}", file=sys.stderr)
        sys.exit(1)

    run(args.candidates, args.out)


if __name__ == "__main__":
    main()
