"""
Redrob Hackathon — Sandbox Demo (Streamlit)
============================================
Accepts a small candidate sample (up to 100 candidates) as JSON/JSONL,
runs the 5-stage ranking pipeline, and displays the ranked results with
downloadable CSV.

Deploy for free at: https://streamlit.io/cloud
"""

import io
import json
import csv
import datetime
import streamlit as st
import pandas as pd

# ── Import the ranker modules ─────────────────────────────────────────────────
from ranker.honeypot import is_honeypot
from ranker.filters import passes_hard_constraints
from ranker.career_scorer import compute_career_score
from ranker.signal_modifier import compute_signal_multiplier
from ranker.reasoning import generate_reasoning


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob Candidate Ranker",

    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #a0a8c0;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
    }

    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #7c83fd;
    }

    .metric-card .label {
        font-size: 0.8rem;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }

    .stage-badge {
        display: inline-block;
        background: rgba(124, 131, 253, 0.15);
        border: 1px solid rgba(124, 131, 253, 0.3);
        color: #7c83fd;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }

    .rank-badge-top {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: #1a1a2e;
        font-weight: 700;
        border-radius: 8px;
        padding: 0.2rem 0.6rem;
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    .upload-zone {
        border: 2px dashed rgba(124,131,253,0.4);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: rgba(124,131,253,0.03);
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎯 Redrob Intelligent Candidate Ranker</h1>
</div>
""", unsafe_allow_html=True)


# ── Pipeline explanation ──────────────────────────────────────────────────────
with st.expander("🏗️ Architecture Overview", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Stage 1 — Honeypot Elimination**
Removes profiles with expert skills at 0 months of use and
severe YOE fabrications (profile vs career history gap > 8 years).

**Stage 2 — Hard Constraint Filter**
Removes candidates whose entire career is at consulting firms
(TCS, Infosys, Wipro, Accenture, etc.) — explicitly disqualified by JD.

**Stage 3 — Career Relevance Scoring**
Maps each of 44 unique career description templates to a base score
(0–100). Applies product company boost (×1.25), consulting penalty
(×0.65), recency weight, YOE modifier, and job-hopper penalty.
        """)
    with col2:
        st.markdown("""
**Stage 4 — Behavioral Signal Multiplier**
All 23 Redrob platform signals applied as a composite multiplier:
last active date, recruiter response rate, notice period, GitHub
activity, platform skill assessments, interview completion rate,
location preference.

**Stage 5 — Dynamic Reasoning**
Generates a 1-2 sentence, fact-grounded reasoning per candidate
using actual profile data — skills, companies, YOE, and concerns.
No hallucination: only states what is in the profile.

**Why no embeddings?**
The 100k pool has descriptions from 44 templates.
Template classification beats embedding cosine similarity — faster,
more accurate, and immune to keyword-stuffed skills sections.
        """)


# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown("### 📁 Upload Candidate Data")
st.caption("Upload a JSON array or JSONL file with up to 100 candidates. Use `sample_candidates.json` from the hackathon bundle.")

uploaded = st.file_uploader(
    "Drop candidates file here",
    type=["json", "jsonl"],
    label_visibility="collapsed",
)


def load_candidates(uploaded_file) -> list[dict]:
    """Parse uploaded file as JSON array or JSONL."""
    content = uploaded_file.read().decode("utf-8")
    # Try JSON array first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
    # Try JSONL
    candidates = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return candidates


def run_pipeline(candidates: list[dict]) -> list[dict]:
    """Run the full 5-stage ranking pipeline on a list of candidates."""
    results = []

    for candidate in candidates:
        cid = candidate.get("candidate_id", "UNKNOWN")

        # Stage 1
        honeypot = is_honeypot(candidate)
        if honeypot:
            results.append({
                "candidate_id": cid,
                "status": "EXCLUDED — Honeypot",
                "score": 0.0,
                "rank": None,
                "reasoning": "Excluded: impossible profile detected (expert skill with 0 months use or fabricated YOE).",
                "name": candidate["profile"].get("anonymized_name", ""),
                "title": candidate["profile"].get("current_title", ""),
                "yoe": candidate["profile"].get("years_of_experience", 0),
                "company": candidate["profile"].get("current_company", ""),
            })
            continue

        # Stage 2
        if not passes_hard_constraints(candidate):
            results.append({
                "candidate_id": cid,
                "status": "EXCLUDED — Hard constraint",
                "score": 0.0,
                "rank": None,
                "reasoning": "Excluded: entire career in consulting-only firms or insufficient experience.",
                "name": candidate["profile"].get("anonymized_name", ""),
                "title": candidate["profile"].get("current_title", ""),
                "yoe": candidate["profile"].get("years_of_experience", 0),
                "company": candidate["profile"].get("current_company", ""),
            })
            continue

        # Stage 3 + 4
        career_score = compute_career_score(candidate)
        signal_mult = compute_signal_multiplier(candidate)
        final_score = career_score * signal_mult

        results.append({
            "candidate_id": cid,
            "status": "ELIGIBLE",
            "score": round(final_score, 4),
            "career_score": round(career_score, 4),
            "signal_mult": round(signal_mult, 4),
            "rank": None,   # assigned after sorting
            "reasoning": "",
            "name": candidate["profile"].get("anonymized_name", ""),
            "title": candidate["profile"].get("current_title", ""),
            "yoe": candidate["profile"].get("years_of_experience", 0),
            "company": candidate["profile"].get("current_company", ""),
            "_candidate": candidate,
        })

    # Sort eligible by score desc, then candidate_id asc (tie-break)
    eligible = [r for r in results if r["status"] == "ELIGIBLE"]
    excluded = [r for r in results if r["status"] != "ELIGIBLE"]

    eligible.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # Stage 5 — reasoning + rank assignment
    for rank_idx, r in enumerate(eligible, start=1):
        r["rank"] = rank_idx
        r["reasoning"] = generate_reasoning(r["_candidate"], r["score"], rank_idx)
        del r["_candidate"]

    return eligible + excluded


# ── Run pipeline on upload ────────────────────────────────────────────────────
if uploaded is not None:
    with st.spinner("Running 5-stage ranking pipeline..."):
        candidates = load_candidates(uploaded)

    if not candidates:
        st.error("Could not parse any candidates from the uploaded file.")
    else:
        st.success(f"Loaded **{len(candidates)}** candidates. Running pipeline...")

        with st.spinner("Scoring..."):
            results = run_pipeline(candidates)

        eligible = [r for r in results if r["status"] == "ELIGIBLE"]
        excluded = [r for r in results if r["status"] != "ELIGIBLE"]

        # ── Metrics ──────────────────────────────────────────────────────────
        st.markdown("### 📊 Pipeline Results")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{len(candidates)}</div>
                <div class="label">Candidates Loaded</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{len(eligible)}</div>
                <div class="label">Eligible</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{len(excluded)}</div>
                <div class="label">Excluded</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            top_score = eligible[0]["score"] if eligible else 0
            st.markdown(f"""<div class="metric-card">
                <div class="value">{top_score:.1f}</div>
                <div class="label">Top Score</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Ranked table ─────────────────────────────────────────────────────
        if eligible:
            st.markdown("### 🏆 Ranked Candidates")

            df_eligible = pd.DataFrame([{
                "Rank":         r["rank"],
                "Candidate ID": r["candidate_id"],
                "Score":        r["score"],
                "Career Score": r.get("career_score", 0),
                "Signal Mult":  r.get("signal_mult", 0),
                "Name":         r["name"],
                "Title":        r["title"],
                "YOE":          r["yoe"],
                "Company":      r["company"],
                "Reasoning":    r["reasoning"],
            } for r in eligible])

            # Top 10 highlight
            st.dataframe(
                df_eligible,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank":         st.column_config.NumberColumn(width="small"),
                    "Score":        st.column_config.NumberColumn(format="%.4f", width="small"),
                    "Career Score": st.column_config.NumberColumn(format="%.4f", width="small"),
                    "Signal Mult":  st.column_config.NumberColumn(format="%.4f", width="small"),
                    "YOE":          st.column_config.NumberColumn(format="%.1f", width="small"),
                    "Reasoning":    st.column_config.TextColumn(width="large"),
                },
            )

            # ── CSV download ──────────────────────────────────────────────────
            st.markdown("### 📥 Download Submission CSV")

            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for r in eligible:
                writer.writerow([r["candidate_id"], r["rank"], r["score"], r["reasoning"]])

            st.download_button(
                label="Download submission.csv",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name="submission.csv",
                mime="text/csv",
            )

        # ── Excluded table ────────────────────────────────────────────────────
        if excluded:
            with st.expander(f"🚫 Excluded Candidates ({len(excluded)})", expanded=False):
                df_excl = pd.DataFrame([{
                    "Candidate ID": r["candidate_id"],
                    "Reason":       r["status"],
                    "Name":         r["name"],
                    "Title":        r["title"],
                    "YOE":          r["yoe"],
                    "Company":      r["company"],
                } for r in excluded])
                st.dataframe(df_excl, use_container_width=True, hide_index=True)

else:
    # ── Landing state ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="upload-zone">
        <h3 style="color:#7c83fd; margin:0">Upload candidates.json or sample_candidates.json</h3>
        <p style="color:#8892a4; margin:0.5rem 0 0 0">
            Accepts JSON array or JSONL format · Up to 100 candidates for the sandbox demo
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Show pipeline stages as info cards
    st.markdown("### How it works")
    s1, s2, s3, s4, s5 = st.columns(5)
    for col, emoji, title, desc in [
        (s1, "🔍", "Stage 1", "Honeypot Elimination"),
        (s2, "🚫", "Stage 2", "Hard Constraint Filter"),
        (s3, "⚡", "Stage 3", "Career Relevance Score"),
        (s4, "📡", "Stage 4", "Behavioral Signals"),
        (s5, "📝", "Stage 5", "Dynamic Reasoning"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.8rem">{emoji}</div>
                <div style="font-size:0.75rem; color:#7c83fd; font-weight:600; margin-top:0.3rem">{title}</div>
                <div style="font-size:0.85rem; color:#dde2f0; margin-top:0.2rem">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.caption("Redrob Hackathon Submission · 5-stage deterministic pipeline · CPU only · No network · ~9s for 100k candidates")
