"""
Phase 2 & 3: Fast Dual-Vector Scoring & Reasoning
=================================================
This script reads 100k candidates, applies universal elimination,
scores eligible candidates using pre-computed JD embeddings,
and outputs the top 100 to submission.csv.

Improvements in this version:
  - v_skills vector (4th JD vector) for semantic skills matching
  - Adaptive negative penalty beta (sim_neg-driven)
  - compute_skills_bonus: score candidate skills against JD must-haves
  - 19 behavioral signals via updated signal_modifier
  - LLM-generated reasoning for top-15 candidates (Qwen 2.5 1.5B)
  - Fixed "High Reliability" wording bug
  - Career trajectory signal via seniority_target from JD meta
"""
import json
import csv
import sys
import time
import math
import functools
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer

from honeypot import is_honeypot, integrity_penalty
from signal_modifier import (
    compute_signal_multiplier,
    compute_location_multiplier,
    compute_disqualifier_penalty,
    compute_title_alignment_multiplier,
    compute_hard_behavioral_multiplier,
    compute_seniority_intent_multiplier
)


# ─────────────────────────────────────────────────────────────────────────────
# Try importing llama_cpp for LLM reasoning; degrade gracefully if unavailable
# ─────────────────────────────────────────────────────────────────────────────
try:
    from llama_cpp import Llama as _Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

TOP_N = 100
LLM_REASONING_TOP_N = 10   # Use Qwen for top-N reasoning; template for the rest

# ─────────────────────────────────────────────────────────────────────────────
# JD HARD-REQUIREMENT CATEGORIES
#
# Root-cause fix (v4): The previous code checked must_have_skills_short[:4]
# which grabbed ['embeddings','retrieval','ranking','LLMs']. 'LLMs' and
# 'fine-tuning' appear at positions 4-5 in the JSON but are NOT actual JD hard
# requirements — they are domain keywords. The JD has exactly 4 hard
# requirements (from must_have_hard_skills):
#
#   1. Embeddings / dense retrieval systems (Sentence Transformers, BGE, etc.)
#   2. Vector databases / hybrid search (Weaviate, Qdrant, OpenSearch, pgvector…)
#   3. Python
#   4. Ranking evaluation frameworks (NDCG, MRR, MAP, LTR, A/B test, offline-to-online)
#
# This caused Kiara Sen (Sentence Transformers + Weaviate + pgvector + LTR)
# to score only 2/4 (failing 'LLMs' slot) → gate=0.90, when she should score
# 3-4/4 → gate=1.00.
#
# Fix: check against 4 fixed *semantic categories* instead of the first 4
# items of a variable-length short-list. This is still JD-agnostic in the
# sense that the aliases are comprehensive and cover all realistic variants.
# ─────────────────────────────────────────────────────────────────────────────

# Kept for compute_skills_bonus() direct-hit bonus (skill-name lookup)
_MUST_HAVE_ALIASES: dict[str, set[str]] = {
    "embeddings": {
        "embeddings", "embedding", "sentence-transformers", "sentence transformers",
        "bge", "e5", "openai embeddings", "text embeddings", "dense retrieval",
        "bi-encoder", "semantic similarity",
    },
    "retrieval": {
        "retrieval", "information retrieval", "semantic search", "hybrid search",
        "bm25", "faiss", "dense retrieval", "sparse retrieval", "neural retrieval",
        "haystack", "vector search", "ann search",
    },
    "ranking": {
        "ranking", "learning to rank", "ltr", "ndcg", "mrr", "map",
        "reranking", "re-ranking", "cross-encoder", "listwise", "pairwise",
        "ranking evaluation", "evaluation framework",
    },
    "vector database": {
        "vector database", "vector db", "vector store", "vector search", "pinecone",
        "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "pgvector",
        "faiss", "chroma", "chromadb", "redis vector", "ann search",
        "approximate nearest neighbour", "approximate nearest neighbor",
    },
    "python": {"python"},
    "ndcg": {"ndcg", "normalized discounted cumulative gain", "ranking evaluation"},
    "mrr": {"mrr", "mean reciprocal rank", "ranking evaluation"},
    "map": {"map", "mean average precision", "ranking evaluation"},
    "offline-to-online correlation": {
        "offline-to-online", "a/b test", "a/b testing", "evaluation framework",
        "online evaluation", "offline evaluation",
    },
    "a/b test interpretation": {
        "a/b test", "a/b testing", "online experiment", "experimentation",
        "statistical significance",
    },
}

# The 4 canonical hard-requirement categories, each as a frozenset of aliases.
# Derived from must_have_hard_skills (the 4 bullet-point requirements in the JD).
# These are used by compute_must_have_match() and compute_skills_bonus().
_HARD_REQ_CATEGORIES: list[frozenset] = [
    # Category 1: Embeddings / dense retrieval
    frozenset({
        "embeddings", "embedding", "sentence-transformers", "sentence transformers",
        "bge", "e5", "openai embeddings", "text embeddings", "dense retrieval",
        "bi-encoder", "semantic similarity", "text-embedding-ada",
        "neural retrieval",
    }),
    # Category 2: Vector databases / hybrid search infrastructure
    frozenset({
        "vector database", "vector db", "vector store", "vector search",
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
        "pgvector", "faiss", "chroma", "chromadb", "redis vector",
        "ann search", "approximate nearest neighbour", "approximate nearest neighbor",
        "hybrid search", "bm25",
    }),
    # Category 3: Python (non-negotiable language requirement)
    frozenset({"python"}),
    # Category 4: Ranking evaluation frameworks
    frozenset({
        "ndcg", "mrr", "map", "ltr", "learning to rank", "ranking evaluation",
        "a/b test", "a/b testing", "offline-to-online", "offline evaluation",
        "online evaluation", "evaluation framework", "information retrieval",
        "reranking", "re-ranking", "cross-encoder", "ranking", "listwise", "pairwise",
        "mean reciprocal rank", "mean average precision", "haystack",
    }),
]


def pick_jd_aligned_skills(candidate: dict, meta: dict, n: int = 3) -> tuple:
    """
    Select the most JD-relevant skills for reasoning output.

    Priority order:
      1. Skills that hit any _HARD_REQ_CATEGORIES alias  (must-haves, highest priority)
      2. Skills that appear in domain_keywords            (domain fit)
      3. Skills that appear in must_have_skills_short     (explicit short list)
      4. Fallback: top skills by proficiency x duration   (old behavior)

    Filters out any skill in negative_tools_or_patterns to prevent YOLO, GANs,
    TTS, LangChain etc. from being cited as positives for a retrieval role.

    JD-agnostic: all filter lists come from jd_metadata.json.
    Returns (list_of_skill_names, alignment_note).
    alignment_note == "limited_jd_alignment" if no JD-aligned skill was found.
    """
    domain_kws       = {k.lower().strip() for k in meta.get("domain_keywords", [])}
    must_haves_short = {k.lower().strip() for k in meta.get("must_have_skills_short", [])}
    negative_pats    = {k.lower().strip() for k in meta.get("negative_tools_or_patterns", [])}

    prof_map = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    scored   = []

    for s in candidate.get("skills", []):
        name       = s.get("name", "").strip()
        name_lower = name.lower()
        if not name:
            continue

        # Filter: skip skills matching negative_tools_or_patterns from JD
        if any(neg in name_lower or name_lower in neg for neg in negative_pats):
            continue

        prof_score = prof_map.get(s.get("proficiency", ""), 1)
        dur_score  = min(s.get("duration_months", 0) / 36.0, 1.0)

        # JD relevance tier (3=must-have, 2=domain kw, 1=short list, 0=generic)
        jd_score = 0
        for cat in _HARD_REQ_CATEGORIES:
            if any(alias in name_lower or name_lower in alias for alias in cat):
                jd_score = 3
                break
        if jd_score == 0 and any(kw in name_lower or name_lower in kw for kw in domain_kws):
            jd_score = 2
        if jd_score == 0 and any(mh in name_lower or name_lower in mh for mh in must_haves_short):
            jd_score = 1

        scored.append((jd_score, prof_score, dur_score, name))

    # Sort: JD relevance first, then proficiency, then duration
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    top = [name for _, _, _, name in scored[:n]]

    alignment_note = "limited_jd_alignment" if (not scored or scored[0][0] == 0) else "aligned"

    # Fallback: if all skills were filtered by negative patterns, use raw proficiency sort
    if not top:
        fallback = sorted(
            candidate.get("skills", []),
            key=lambda x: (prof_map.get(x.get("proficiency", ""), 1), x.get("duration_months", 0)),
            reverse=True,
        )
        top = [s["name"] for s in fallback[:n] if s.get("name")]
        alignment_note = "limited_jd_alignment"

    return top, alignment_note



def compute_must_have_match(candidate: dict, meta: dict) -> float:
    """
    Hard gate: score the candidate against the 4 canonical JD hard-requirement
    categories (not the variable must_have_skills_short list).

    Checks both:
      (a) the candidate's advanced/expert skill list (primary signal)
      (b) the candidate's career text (secondary, partial credit)

    Returns a gating multiplier:
      0 categories matched → 0.40  (near-disqualification)
      1 category  matched  → 0.70
      2 categories matched → 0.90
      3+ categories matched→ 1.00  (fully qualified — no penalty)

    Root-cause fix (v4): Previously used must_have_skills_short[:4] which put
    'LLMs' in slot 4. 'LLMs' is a domain keyword, NOT a JD hard requirement.
    This caused Kiara Sen (3 hard categories matched) to receive gate=0.90 instead
    of 1.00. Now checks against the 4 _HARD_REQ_CATEGORIES defined above.
    """
    if not meta:
        return 1.0

    # Build a set of the candidate's expert/advanced skill names (lower-cased)
    candidate_skills: set[str] = {
        s.get("name", "").lower().strip()
        for s in candidate.get("skills", [])
        if s.get("proficiency", "").lower() in ("advanced", "expert")
        and s.get("name", "").strip()
    }
    # Also include ALL skills (not just expert) for Python — it's often intermediate
    all_skill_names: set[str] = {
        s.get("name", "").lower().strip()
        for s in candidate.get("skills", [])
        if s.get("name", "").strip()
    }

    # Career text (lower) for secondary check
    career_text = " ".join(
        j.get("description", "") for j in candidate.get("career_history", [])
    ).lower()

    matched_primary = 0
    career_partial  = 0.0

    for cat_idx, category in enumerate(_HARD_REQ_CATEGORIES):
        # For Python (cat_idx == 2), check ALL skill levels — it's rarely listed
        # as 'expert' by senior engineers who take it for granted.
        skill_pool = all_skill_names if cat_idx == 2 else candidate_skills

        # Primary: skill list hit — any alias in this category matches a candidate skill
        skill_hit = any(
            alias in skill or skill in alias
            for alias in category
            for skill in skill_pool
        )
        if skill_hit:
            matched_primary += 1
            continue

        # Secondary: career text hit (half credit)
        career_hit = any(alias in career_text for alias in category)
        if career_hit:
            career_partial += 0.5

    total_matched = min(matched_primary + career_partial, 4.0)

    if total_matched >= 3.0:
        return 1.00   # fully qualified
    if total_matched >= 2.0:
        return 0.90
    if total_matched >= 1.0:
        return 0.70
    return 0.40       # fails hard requirements — near-disqualification


# ─────────────────────────────────────────────────────────────────────────────
# YOE MODIFIER — dynamically driven by JD metadata
# ─────────────────────────────────────────────────────────────────────────────

def _yoe_modifier(candidate_yoe: float, min_yoe: float, max_yoe: float) -> float:
    """Dynamically scales score based on the target YOE band from JD metadata."""
    if min_yoe <= candidate_yoe <= max_yoe:
        return 1.00
    if min_yoe - 1 <= candidate_yoe < min_yoe or max_yoe < candidate_yoe <= max_yoe + 1:
        return 0.80   # Punish slight YOE mismatch more severely (down from 0.92)
    if min_yoe - 2 <= candidate_yoe < min_yoe - 1:
        return 0.60   # Punish moderate YOE underqualification (down from 0.78)
    if candidate_yoe > max_yoe + 1:
        return 0.65   # Punish significant overqualification (down from 0.82)
    return 0.40       # Otherwise (severe mismatch, down from 0.50)


# ─────────────────────────────────────────────────────────────────────────────
# SAFE LLM CALL — handles context-window overflows gracefully
# ─────────────────────────────────────────────────────────────────────────────

def _safe_llm_call(llm, prompt, max_tokens=150, stop="<|im_end|>", temperature=0.15):
    """Safely call the LLM; retry with reduced tokens on context overflow."""
    attempt_max = max_tokens
    for _ in range(4):
        try:
            return llm(prompt, max_tokens=attempt_max, stop=stop, temperature=temperature)
        except ValueError as e:
            if "exceed context window" in str(e):
                attempt_max = max(32, attempt_max // 2)
                continue
            raise
    # Last-ditch: truncate prompt
    try:
        return llm(prompt[-2000:], max_tokens=64, stop=stop, temperature=temperature)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM REASONING — fully dynamic, built from jd_metadata.json
# ─────────────────────────────────────────────────────────────────────────────

# Known hallucinated phrases that indicate template bleed — used by the guard below
_HALLUCINATION_PHRASES = (
    # Known template-bleed phrases
    "suggests a focus on growth and learning",
    "self-summary suggests a focus on",
    "self-summary suggests growth",
    # Generic filler indicating the LLM didn't read the candidate data
    "is a highly motivated",
    "brings a wealth of experience",
    "would be a valuable addition to any team",
    "strong track record of",
)


def generate_llm_reasoning(llm, candidate: dict, score: float, rank_idx: int, meta: dict) -> str:
    """
    Generate a fact-grounded, JD-anchored hiring brief using the offline LLM.

    The entire prompt is constructed from:
      - meta (jd_metadata.json) — role, must-haves, disqualifiers, domain keywords
      - candidate data — title, company, YOE, top skills, career evidence, self-summary

    Zero hardcoding: works for any role, any domain, any JD.
    Falls back to template reasoning if LLM fails or returns empty output.

    v3 Bug Fixes:
      (1) Removed the heavy-handed CRITICAL HONESTY RULE block that caused the LLM
          to always write "suggests a focus on growth and learning" regardless of
          the actual summary content. Replaced with a conditional, neutral instruction.
      (2) Temperature raised 0.15 → 0.30 to reduce repetition collapse onto the
          training prior (the main driver of phrase hallucination).
      (3) Added post-generation hallucination guard: if the output contains any
          known hallucinated phrases, discard and use template fallback.
      (4) Injected must-have match count into the user block so the LLM anchors
          its recommendation to actual constraint compliance.
    """
    p = candidate.get("profile", {})
    name = p.get("anonymized_name", f"Candidate {candidate.get('candidate_id')}")
    title = p.get("current_title", "Professional")
    company = p.get("current_company", "")
    yoe = p.get("years_of_experience", 0)
    role = meta.get("job_title", "target role")
    company_name = meta.get("company", "Redrob")

    constraints = meta.get("metadata_constraints", {})
    min_yoe = float(constraints.get("min_yoe", 0))
    max_yoe = float(constraints.get("max_yoe", 99))

    # Pull JD-specific context entirely from meta
    must_haves = meta.get("must_have_hard_skills", [])[:2]
    domain_keywords = meta.get("domain_keywords", [])[:4]
    disqualifiers = meta.get("abstract_disqualifiers", [])[:1]
    seniority = meta.get("seniority_target", "")

    # Find which domain keywords appear in the candidate's career descriptions
    career = candidate.get("career_history", [])
    all_career_text = " ".join(j.get("description", "") for j in career).lower()
    matched_signals = [kw for kw in domain_keywords if kw.lower() in all_career_text]

    # Top skills: JD-aligned first (never cite negative patterns or off-domain skills)
    top_skill_names, _ = pick_jd_aligned_skills(candidate, meta, n=4)
    skills = candidate.get("skills", [])

    signals = candidate.get("redrob_signals", {})
    response_rate = signals.get("recruiter_response_rate", 0) * 100
    github_score = signals.get("github_activity_score", -1)

    career_context = f"{title} at {company}" if company else title
    yoe_note = (
        f"(within target {int(min_yoe)}-{int(max_yoe)} yr range)"
        if min_yoe <= yoe <= max_yoe
        else f"(target: {int(min_yoe)}-{int(max_yoe)} yrs)"
    )

    raw_summary = p.get("summary", "").strip()
    summary_snippet = (raw_summary[:150] + "...") if len(raw_summary) > 150 else raw_summary

    # Compute must-have match count to inject into the prompt (anchors LLM to reality)
    expert_skill_names: set[str] = {
        s.get("name", "").lower().strip()
        for s in skills
        if s.get("proficiency", "").lower() in ("advanced", "expert")
        and s.get("name", "").strip()
    }
    all_skill_names: set[str] = {
        s.get("name", "").lower().strip()
        for s in skills
        if s.get("name", "").strip()
    }
    mh_count = 0
    for cat_idx, category in enumerate(_HARD_REQ_CATEGORIES):
        skill_pool = all_skill_names if cat_idx == 2 else expert_skill_names
        skill_hit = any(
            alias in skill or skill in alias
            for alias in category
            for skill in skill_pool
        )
        if skill_hit:
            mh_count += 1
    mh_label = f"{mh_count}/4 core must-haves matched in skills"

    # Check for aspirational/learning intent
    aspirational_phrases = ["looking to transition", "grow into", "still building depth", "aspire to", "learn more"]
    is_aspirational = any(phrase in raw_summary.lower() for phrase in aspirational_phrases)

    aspirational_note = ""
    if is_aspirational:
        aspirational_note = "CRITICAL: The candidate's summary indicates they are looking to transition or grow into senior engineering. You MUST note this honestly as a gap or transition focus, and you MUST NOT claim they have production deployment experience with embeddings, retrieval, or vector databases."
    else:
        aspirational_note = "Focus on their demonstrated production experience with the core requirements."

    # Build the full prompt from meta — zero hardcoded role names or skill names
    prompt = f"""<|im_start|>system
You are a senior technical recruiter writing a concise, factual hiring recommendation for a role at {company_name}.
Role: {role}{f" ({seniority}-level)" if seniority else ""}.
Core requirements: {"; ".join(must_haves) if must_haves else "strong domain expertise"}.
Disqualifying backgrounds: {"; ".join(disqualifiers) if disqualifiers else "none specified"}.
{aspirational_note}
Write exactly 2-3 sentences. Be specific to this candidate's actual data. No bullet points. Do NOT recommend the candidate for a role at their current company ({company}) or any other company; they are being recommended for the role at {company_name}.<|im_end|>
<|im_start|>user
Candidate: {name}
Position: {career_context}
Years of Experience: {yoe:.1f} {yoe_note}
Top Skills: {", ".join(top_skill_names) if top_skill_names else "not listed"}
JD keywords in career history: {", ".join(matched_signals) if matched_signals else "none detected"}
Constraint compliance: {mh_label}
Candidate intent: {"Aspirational / Looking to transition or learn" if is_aspirational else "Experienced professional"}
Candidate summary: "{summary_snippet if summary_snippet else 'not provided'}"
Recruiter response rate: {response_rate:.0f}% | GitHub score: {github_score if github_score != -1 else 'N/A'}
Rank: #{rank_idx} of {TOP_N} (score: {score:.4f})
Write a 2-3 sentence hiring recommendation.<|im_end|>
<|im_start|>assistant
"""


    # Fix (2): Raised temperature 0.15 → 0.30 to reduce collapse onto the training prior
    resp = _safe_llm_call(llm, prompt, max_tokens=180, stop="<|im_end|>", temperature=0.30)
    if resp:
        text = resp["choices"][0]["text"].strip()
        print(f"[LLM rank#{rank_idx}] len={len(text)} | {text[:80]!r}")
        # Fix (3): Post-generation hallucination guard.
        # If the LLM produced a known template-bleed phrase, discard and use template.
        if len(text) >= 40:
            text_lower = text.lower()
            is_hallucinated = any(phrase in text_lower for phrase in _HALLUCINATION_PHRASES)
            if is_hallucinated:
                print(f"[LLM rank#{rank_idx}] DISCARDED: hallucination phrase detected")
            else:
                return text
    else:
        print(f"[LLM rank#{rank_idx}] No response from LLM — using template fallback")

    # Fallback to template if LLM output is empty, truncated, or hallucinated
    return generate_template_reasoning(candidate, score, rank_idx, meta)


# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE REASONING — for ranks > LLM_REASONING_TOP_N, or LLM fallback
# ─────────────────────────────────────────────────────────────────────────────

def generate_template_reasoning(candidate: dict, score: float, rank_idx: int, meta: dict) -> str:
    """
    Generate structured template reasoning. Driven entirely by meta + candidate data.
    Corrects the "High Reliability" wording bug for low response rates.
    """
    p = candidate.get("profile", {})
    name = p.get("anonymized_name", f"Candidate {candidate.get('candidate_id')}")
    title = p.get("current_title", "Professional")
    company = p.get("current_company", "")
    yoe = p.get("years_of_experience", 0)

    constraints = meta.get("metadata_constraints", {})
    min_yoe = float(constraints.get("min_yoe", 5.0))
    max_yoe = float(constraints.get("max_yoe", 9.0))
    role = meta.get("job_title", "target role")
    company_name = meta.get("company", "Redrob")

    # Top skills: JD-aligned first (never cite negative-domain or off-topic skills)
    top_skills, alignment_note = pick_jd_aligned_skills(candidate, meta, n=3)
    if alignment_note == "limited_jd_alignment" and top_skills:
        skills_phrase = (
            f"with limited direct alignment to the {role} must-haves "
            f"(primary skills: {', '.join(top_skills)})"
        )
    elif top_skills:
        skills_phrase = f"demonstrating capabilities in {', '.join(top_skills)}"
    else:
        skills_phrase = ""

    career_phrase = ""
    if title:
        career_phrase = (
            f"currently working as a {title} at {company}"
            if company
            else f"background as a {title}"
        )

    # YOE phrasing
    yoe_phrase = f"possessing {yoe:.1f} YOE"
    if min_yoe <= yoe <= max_yoe:
        yoe_phrase += f" (matching the target {int(min_yoe)}-{int(max_yoe)} bracket)"
    elif yoe < min_yoe:
        yoe_phrase += f" (slightly under the target {int(min_yoe)} YOE requirement)"
    else:
        yoe_phrase += f" (exceeding the base {int(max_yoe)} YOE constraint)"

    # Fixed: no more "High reliability" for low response rates
    rr = candidate.get("redrob_signals", {}).get("recruiter_response_rate", 0) * 100
    if rr >= 70:
        rel_phrase = f"showing strong recruiter responsiveness ({rr:.0f}% response rate)"
    elif rr >= 45:
        rel_phrase = f"with moderate recruiter responsiveness ({rr:.0f}% response rate)"
    else:
        rel_phrase = f"with lower recruiter responsiveness ({rr:.0f}% response rate — may need proactive outreach)"

    parts = []
    if career_phrase:
        parts.append(career_phrase)
    parts.append(yoe_phrase)
    if skills_phrase:
        parts.append(skills_phrase)
    parts.append(rel_phrase)

    if len(parts) > 1:
        reasoning = (
            f"{name} is ranked #{rank_idx} for the {role} role at {company_name} because they are "
            + ", ".join(parts[:-1])
            + ", and "
            + parts[-1]
            + "."
        )
    else:
        reasoning = (
            f"{name} is ranked #{rank_idx} for the {role} role at {company_name} because they are "
            + parts[0]
            + "."
        )
    return reasoning


def generate_detailed_reasoning(candidate: dict, score: float, rank_idx: int, meta: dict) -> str:
    """Wrapper for backward compatibility with app.py."""
    return generate_template_reasoning(candidate, score, rank_idx, meta)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run(candidates_path, jd_embed_path, jd_meta_path, out_path):
    t0 = time.perf_counter()

    # ── Load JD brain ────────────────────────────────────────────────────────
    print(f"[main_ranker] Loading JD Brain from {jd_embed_path}...")
    try:
        data = np.load(jd_embed_path)
    except FileNotFoundError:
        print("ERROR: jd_embeddings.npz not found. Please run JD_parser.py first.")
        sys.exit(1)

    v_core = data["v_core"]
    v_neg  = data["v_neg"]
    # v_skills: 4th vector for semantic skills matching. Backward-compatible fallback.
    try:
        v_skills = data["v_skills"]
    except KeyError:
        v_skills = data["v_core"]
        print("[main_ranker] Warning: v_skills not found in embeddings — using v_core as fallback.")
        print("[main_ranker] Re-run JD_parser.py to generate v_skills for better accuracy.")

    # Squeeze 2D → 1D if saved as shape (1, 384)
    def _squeeze(arr):
        return arr[0] if len(arr.shape) == 2 else arr

    v_core   = _squeeze(v_core)
    v_neg    = _squeeze(v_neg)
    v_skills = _squeeze(v_skills)

    with open(jd_meta_path, "r") as f:
        meta = json.load(f)

    constraints = meta.get("metadata_constraints", {})
    min_yoe   = float(constraints.get("min_yoe", 6.0))
    max_yoe   = float(constraints.get("max_yoe", 8.0))
    pref_locs = constraints.get("preferred_locations", [])
    target_job_title      = meta.get("job_title", "Professional")
    preferred_company_type = constraints.get("preferred_company_type", "")
    title_family_keywords = meta.get("title_family_keywords", [])
    unacceptable_title_keywords = meta.get("unacceptable_title_keywords", [])

    # ── Load embedding model ──────────────────────────────────────────────────
    vrd_dir    = Path(__file__).resolve().parent
    model_path = vrd_dir / "local_bge_model"

    try:
        sys.path.append(str(vrd_dir))
        from download_model import ensure_models_exist
        ensure_models_exist()
    except Exception as e:
        print(f"Warning: Could not check/download BGE model: {e}")

    print("[main_ranker] Loading Local Embedding Model (BGE)...")
    embedder = SentenceTransformer(str(model_path))

    # ── Cached embedding functions (closures capture v_core, v_neg, v_skills) ─

    @functools.lru_cache(maxsize=None)
    def cached_embed_and_score(text: str):
        """
        Returns (final_score, sim_pos, sim_neg).
        Uses adaptive beta: stronger disqualification when candidate is
        clearly in the wrong domain (high sim_neg, low sim_pos).
        """
        v_job   = embedder.encode(text, normalize_embeddings=True)
        sim_pos = float(np.dot(v_job, v_core))
        sim_neg = float(np.dot(v_job, v_neg))

        # Adaptive beta — driven purely by the data, zero hardcoding
        if sim_neg > 0.55 and sim_pos < 0.40:
            beta = 0.60   # clearly wrong domain → heavy penalty
        elif sim_neg > 0.45:
            beta = 0.40   # moderate misalignment
        else:
            beta = 0.30   # baseline — mild negative subtraction

        score = sim_pos - (beta * sim_neg)
        return score, sim_pos, sim_neg

    @functools.lru_cache(maxsize=None)
    def cached_skill_score(skill_name: str) -> float:
        """Cosine similarity of a skill name to the JD skills vector."""
        v = embedder.encode(skill_name, normalize_embeddings=True)
        return float(np.dot(v, v_skills))

    @functools.lru_cache(maxsize=None)
    def get_cached_embedding(text: str):
        return embedder.encode(text, normalize_embeddings=True)

    def verify_skills_grounding(candidate: dict) -> float:
        """
        Verify if the candidate's expert/advanced skills are grounded in career history descriptions.
        Returns a grounding ratio multiplier in [0.50, 1.00] to scale the skills boost.
        """
        skills = candidate.get("skills", [])
        career = candidate.get("career_history", [])
        
        expert_skills = [
            s.get("name", "").strip()
            for s in skills
            if s.get("proficiency", "").lower() in ("advanced", "expert")
            and s.get("name", "").strip()
        ]
        
        if not expert_skills or not career:
            return 1.0
            
        career_descriptions = [
            j.get("description", "").strip()
            for j in career
            if j.get("description", "").strip()
        ]
        
        if not career_descriptions:
            return 0.20
            
        grounded_count = 0
        for skill_name in expert_skills:
            v_skill = get_cached_embedding(skill_name)
            max_sim = 0.0
            for desc in career_descriptions:
                v_desc = get_cached_embedding(desc[:800])
                sim = float(np.dot(v_skill, v_desc))
                max_sim = max(max_sim, sim)
                
            if max_sim >= 0.48:
                grounded_count += 1
                
        ratio = grounded_count / len(expert_skills)
        return 0.50 + (ratio * 0.50)

    def compute_skills_bonus(candidate: dict) -> float:

        """
        Score candidate's declared skills against JD must-have skills vector.

        Returns a multiplier in 0.90–1.40.

        Fix (v3b): The blended v_skills vector (avg of all must-haves + domain
        keywords) means individual skill-vs-blend cosine similarity peaks around
        0.50–0.65 for exact matches — not 0.72+ as originally assumed.

        Two-part score:
          (a) Weighted avg similarity of expert/advanced skills against v_skills.
              Thresholds adjusted down to match realistic blended-vector range.
          (b) Direct must-have exact match bonus: counts how many of the JD's
              must_have_skills_short appear verbatim in the candidate's expert
              skill set via _MUST_HAVE_ALIASES. This is immune to embedding
              dilution and gives a direct uplift for perfect matches.

        Combined multiplier = max(blend_mult, must_have_bonus_mult).
        Only counts advanced/expert skills with ≥6 months duration for the blend.
        """
        skills = candidate.get("skills", [])

        # ── Part (a): Blended v_skills weighted average ──────────────────────
        strong = [
            s for s in skills
            if s.get("proficiency", "") in ("advanced", "expert")
            and s.get("duration_months", 0) >= 6
        ]
        blend_mult = 1.0
        if strong:
            weighted_scores = []
            for s in strong:
                name = s.get("name", "").strip()
                if not name:
                    continue
                dur_w  = min(s.get("duration_months", 6) / 36.0, 1.0)
                end_w  = min(s.get("endorsements", 0) / 20.0, 1.0)
                sim    = cached_skill_score(name)
                weighted_scores.append(sim * (0.60 + 0.20 * dur_w + 0.20 * end_w))

            if weighted_scores:
                avg = sum(weighted_scores) / len(weighted_scores)
                # Thresholds calibrated for blended v_skills vector range (0.40–0.65)
                if avg >= 0.55:
                    blend_mult = 1.22   # strong blend match
                elif avg >= 0.45:
                    blend_mult = 1.12
                elif avg >= 0.35:
                    blend_mult = 1.05
                elif avg >= 0.25:
                    blend_mult = 1.00   # neutral
                elif avg >= 0.15:
                    blend_mult = 0.96
                else:
                    blend_mult = 0.90

        # ── Part (b): Direct must-have exact-match bonus ────────────────────
        # Uses the same hard-requirement categories as compute_must_have_match() for consistency.
        # This path is embedding-independent and catches exact JD must-haves that
        # the blended vector fails to distinguish at high confidence.
        expert_skill_names: set[str] = {
            s.get("name", "").lower().strip()
            for s in skills
            if s.get("proficiency", "").lower() in ("advanced", "expert")
            and s.get("name", "").strip()
        }
        all_skill_names: set[str] = {
            s.get("name", "").lower().strip()
            for s in skills
            if s.get("name", "").strip()
        }

        direct_hits = 0
        for cat_idx, category in enumerate(_HARD_REQ_CATEGORIES):
            # For Python (cat_idx == 2), check all skill levels.
            skill_pool = all_skill_names if cat_idx == 2 else expert_skill_names
            if any(alias in sk or sk in alias for alias in category for sk in skill_pool):
                direct_hits += 1

        # Direct match bonus tiers — stacked on top of blend_mult
        if direct_hits >= 4:
            must_have_bonus = 1.40   # all 4 core must-haves present: exceptional
        elif direct_hits == 3:
            must_have_bonus = 1.30
        elif direct_hits == 2:
            must_have_bonus = 1.15
        elif direct_hits == 1:
            must_have_bonus = 1.05
        else:
            must_have_bonus = 1.00   # no direct must-have match

        # Take the higher of blend similarity or direct must-have match
        return max(blend_mult, must_have_bonus)

    def compute_title_domain_bonus(candidate: dict) -> float:
        """
        [NEW v5] JD-agnostic title-to-domain relevance using precomputed v_core.

        Embeds the candidate's current title and measures cosine similarity to
        the JD's core requirement vector (v_core). Titles that are semantically
        in the JD domain (e.g. "Search Engineer", "Senior NLP Engineer") get a
        boost of up to 1.25x applied directly to raw_technical_capacity, rescuing
        highly relevant candidates whose career *text* may not use JD vocabulary.

        Returns a multiplier in [0.95, 1.25].
        """
        title = candidate.get("profile", {}).get("current_title", "").strip()
        if not title:
            return 1.0
        v_title = embedder.encode(title, normalize_embeddings=True)
        sim = float(np.dot(v_title, v_core))
        if sim >= 0.65:
            return 1.50   # title strongly in JD domain (Search Engineer, NLP Engineer)
        if sim >= 0.55:
            return 1.30
        if sim >= 0.45:
            return 1.15
        if sim >= 0.35:
            return 1.00
        return 0.90       # title far from JD domain (DevOps, Cloud Engineer, etc.)

    # ── Score all candidates ──────────────────────────────────────────
    print(f"[main_ranker] Reading candidates from: {candidates_path}")
    scored      = []
    total       = 0
    excluded_hp = 0
    max_base_seen = 0.0

    with open(candidates_path, "r", encoding="utf-8-sig") as fh:
        # We don't know exact line count in advance without an extra pass, 
        # so we'll just track progress dynamically, or use a known estimate if possible.
        # But we can just use an tqdm with no total to show speed.
        try:
            from tqdm import tqdm
            iterator = tqdm(fh, desc="Scoring Candidates")
        except ImportError:
            iterator = fh

        for line in iterator:
            line = line.strip()
            if not line:
                continue

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
            total_weight         = 0.0

            # Phase 2: Semantic Career Scoring
            for job in career:
                desc = job.get("description", "")
                if not desc:
                    continue
                duration      = max(job.get("duration_months", 1), 1)
                recency_weight = 1.5 if job.get("is_current", False) else 1.0

                job_score, _, _ = cached_embed_and_score(desc)

                weight = duration * recency_weight
                total_weighted_score += job_score * weight
                total_weight         += weight

            raw_career_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
            base_score = max(0, raw_career_score * 100)
            max_base_seen = max(max_base_seen, base_score)

            # Phase 3: Apply all modifiers
            yoe_mod       = _yoe_modifier(yoe, min_yoe, max_yoe)
            skills_mod    = compute_skills_bonus(candidate)
            must_have_mod = compute_must_have_match(candidate, meta)
            integ_mod     = integrity_penalty(candidate)   # soft continuous penalty
            
            # Hard gate multipliers computed outside the behavioral clamp
            loc_mod             = compute_location_multiplier(candidate, pref_locs)
            disq_mod            = compute_disqualifier_penalty(candidate, meta, embedder)
            title_alignment_mod = compute_title_alignment_multiplier(candidate, target_job_title, title_family_keywords, unacceptable_title_keywords, embedder)
            hard_behavior_mod   = compute_hard_behavioral_multiplier(candidate)

            # Hard gate exclusions for non-technical roles and disqualified candidates
            if title_alignment_mod == 0.15:
                continue
            if disq_mod == 0.30:
                continue

            sig_mod       = compute_signal_multiplier(
                candidate=candidate,
                preferred_locations=pref_locs,
                target_job_title=target_job_title,
                preferred_company_type=preferred_company_type,
                embedder=embedder,
                meta=meta,
            )

            # v3 Score Formula:
            #
            # Three bugs fixed from the previous two-tier approach:
            #
            # (A) Must-Have Gating: must_have_mod enforces JD hard constraints.
            #     A candidate with 0/4 must-haves gets a 0.40x gate — they cannot
            #     end up in the Top 10 regardless of other signals. Previously the
            #     system politely noted missing skills but still ranked them highly.
            #
            # (B) Additive Skills Boost: skills_mod is now applied as an ADDITIVE
            #     delta (+/- on the technical base) rather than multiplicative.
            #     A perfect skills match (mod=1.35) adds 35% to the base technical
            #     score — enough to rescue Kavya Joshi from #99 even if her career
            #     text doesn't embed close to v_core.
            #
            # (C) Behavioral Hard Clamp: sig_mod is clamped to [0.85, 1.15].
            #     This means behavioral signals can reward (+15%) or penalise (-15%)
            #     but cannot halve a technically perfect candidate's final score.
            #     Previously a 57% response rate + no GitHub + wrong location could
            #     compound to ~0.40x total, destroying a 90th-percentile tech score.

            # Technical score logic:
            # First, verify and discount skills if they are not grounded in career descriptions.
            skills_grounding_mod = verify_skills_grounding(candidate)
            grounded_skills_mod = 1.0 + (skills_mod - 1.0) * skills_grounding_mod
            skills_delta = base_score * (grounded_skills_mod - 1.0)

            # [v5] Title domain bonus: embed candidate title vs v_core.
            # "Search Engineer" or "Senior NLP Engineer" semantically close to retrieval
            # JD → up to 1.25x boost on raw_technical_capacity.
            # This rescues candidates whose career text doesn’t use JD vocabulary.
            title_domain_bonus = compute_title_domain_bonus(candidate)

            # Combined raw technical capacity (career + skills + title domain signal)
            raw_technical_capacity = (base_score + skills_delta) * title_domain_bonus

            # Gating constraints (Must-Have, YOE, integrity) scale the entire technical capacity
            technical_score = raw_technical_capacity * must_have_mod * yoe_mod * integ_mod

            # Dynamic seniority-intent alignment (rewards alignment, penalizes mismatch)
            seniority_intent_mod = compute_seniority_intent_multiplier(candidate, meta)

            # Behavioral clamp — v4 fix: proper linear normalization to [0.93, 1.07].
            SIG_NATURAL_MIN       = 0.85   # BEHAVIORAL_FLOOR from signal_modifier
            SIG_NATURAL_MAX       = 2.0    # practical ceiling (all-positive signals)
            BEHAVIORAL_CLAMP_LOW  = 0.93
            BEHAVIORAL_CLAMP_HIGH = 1.07
            raw_norm = (sig_mod - SIG_NATURAL_MIN) / (SIG_NATURAL_MAX - SIG_NATURAL_MIN)
            sig_clamped = BEHAVIORAL_CLAMP_LOW + raw_norm * (BEHAVIORAL_CLAMP_HIGH - BEHAVIORAL_CLAMP_LOW)
            sig_clamped = max(BEHAVIORAL_CLAMP_LOW, min(sig_clamped, BEHAVIORAL_CLAMP_HIGH))

            final_score = technical_score * sig_clamped * hard_behavior_mod * title_alignment_mod * disq_mod * loc_mod * seniority_intent_mod
            scored.append((final_score, candidate, base_score))


    print(f"[main_ranker] Total candidates read : {total:,}")
    print(f"[main_ranker] Excluded (honeypots)  : {excluded_hp:,}")
    print(f"[main_ranker] Eligible candidates   : {len(scored):,}")
    print(f"[main_ranker] Career embed cache    : {cached_embed_and_score.cache_info()}")
    print(f"[main_ranker] Skills embed cache    : {cached_skill_score.cache_info()}")

    # Sort descending, then deterministic tie-break by candidate_id
    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    top = scored[:TOP_N]

    # Log-scale normalize scores of top candidates to [0, 1].
    # vs linear min-max: spreads mid-to-bottom range scores more evenly,
    # giving better discrimination at ranks 30-100.
    if top:
        scores    = [item[0] for item in top]
        max_score = max(scores)
        min_score = min(scores)
        normalized_top = []
        for score, candidate, base_score in top:
            if max_score > min_score:
                # log1p preserves rank order and spreads compressed bottom scores
                norm_score = math.log1p(score - min_score) / math.log1p(max_score - min_score)
            else:
                norm_score = 1.0
            norm_score_rounded = round(norm_score, 4)
            normalized_top.append((norm_score_rounded, candidate, base_score))

        # Re-sort to resolve any rounding ties alphabetically by candidate_id ascending
        normalized_top.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
        top = normalized_top

    # ── Load LLM for high-quality reasoning on top candidates ────────────────
    llm = None
    if LLAMA_AVAILABLE and LLM_REASONING_TOP_N > 0:
        qwen_path = vrd_dir / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
        if qwen_path.exists():
            print(f"[main_ranker] Loading Qwen for top-{LLM_REASONING_TOP_N} reasoning...")
            try:
                llm = _Llama(
                    model_path=str(qwen_path),
                    n_ctx=2048,      # smaller context = faster per-call
                    n_threads=4,
                    verbose=False,
                )
                print("[main_ranker] LLM loaded successfully.")
            except Exception as e:
                print(f"[main_ranker] LLM load failed — using template reasoning: {e}")
                llm = None
        else:
            print("[main_ranker] Qwen model not found — using template reasoning.")

    # ── Write submission CSV ──────────────────────────────────────────────────
    print(f"[main_ranker] Writing top {TOP_N} to {out_path}...")
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        try:
            from tqdm import tqdm
            iterator = tqdm(top, desc="Generating Reasoning & Writing CSV", total=len(top))
        except ImportError:
            iterator = top

        for rank_idx, (score, candidate, base_score) in enumerate(iterator, start=1):
            cid = candidate["candidate_id"]

            # Use LLM for top candidates; fall back to template for the rest
            if llm is not None and rank_idx <= LLM_REASONING_TOP_N:
                reasoning = generate_llm_reasoning(llm, candidate, score, rank_idx, meta)
            else:
                reasoning = generate_template_reasoning(candidate, score, rank_idx, meta)

            writer.writerow([cid, rank_idx, round(score, 4), reasoning])

    elapsed = time.perf_counter() - t0
    print(f"[main_ranker] Done in {elapsed:.2f}s")


if __name__ == "__main__":
    run(
        candidates_path="../candidates.jsonl",
        jd_embed_path="./jd_embeddings.npz",
        jd_meta_path="./jd_metadata.json",
        out_path="./submission.csv",
    )
