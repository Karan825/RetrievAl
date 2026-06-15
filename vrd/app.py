import logging
# Suppress noisy streamlit watcher warnings from scanning large packages like transformers
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import streamlit as st
import json
import numpy as np
import functools
import pandas as pd
import time
import os
from sentence_transformers import SentenceTransformer
from honeypot import is_honeypot
from signal_modifier import compute_signal_multiplier

# ==========================================
# 1. PATH RESOLUTION & CONFIG
# ==========================================
# Ensure we look for files in the same directory as this script (vrd folder)
VRD_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(VRD_DIR)

st.set_page_config(
    page_title="Redrob Candidate Ranker Sandbox",
    page_icon="🎯",
    layout="wide"
)

# Automatically download models if missing
try:
    import sys
    sys.path.append(VRD_DIR)
    from download_model import ensure_models_exist
    
    bge_dir = os.path.join(VRD_DIR, "local_bge_model")
    qwen_path = os.path.join(VRD_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    bge_missing = not os.path.exists(bge_dir) or not os.path.exists(os.path.join(bge_dir, "config.json"))
    qwen_missing = not os.path.exists(qwen_path) or os.path.getsize(qwen_path) < 100 * 1024 * 1024
    
    if bge_missing or qwen_missing:
        status_info = st.empty()
        def show_status(msg):
            status_info.info(f"⏳ {msg}")
        
        ensure_models_exist(progress_callback=show_status)
        status_info.success("🎉 Models successfully prepared and ready!")
        time.sleep(2)
        status_info.empty()
except Exception as e:
    st.error(f"Failed to automatically check/download models: {e}")

# Custom CSS for modern typography, animations, and container refinements
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #080A10 !important;
        color: #F8FAFC !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Subtle pulsing effect for processing stages */
    @keyframes pulse {
        0% { transform: scale(0.98); opacity: 0.7; }
        50% { transform: scale(1.02); opacity: 1; }
        100% { transform: scale(0.98); opacity: 0.7; }
    }
    
    /* Native bordered container styling with left accent border and glow */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid rgba(99, 102, 241, 0.08) !important;
        border-left: 4px solid #6366F1 !important;
        border-radius: 16px !important;
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.6) 0%, rgba(11, 15, 26, 0.8) 100%) !important;
        backdrop-filter: blur(12px) !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        padding: 24px !important;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(99, 102, 241, 0.25) !important;
        border-left-color: #818CF8 !important;
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.08), 0 4px 30px rgba(0, 0, 0, 0.3) !important;
        transform: translateY(-2px);
    }

    /* Custom high-contrast action buttons with premium SaaS glow */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #a855f7 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: 0.02em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5) !important;
    }
    
    .stButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Radio button custom look */
    div[data-testid="stRadio"] label p {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #E2E8F0 !important;
    }

    /* File uploader container styling */
    div[data-testid="stFileUploader"] {
        background-color: rgba(17, 24, 39, 0.45) !important;
        border: 1px dashed rgba(99, 102, 241, 0.2) !important;
        border-radius: 12px !important;
        padding: 18px !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(99, 102, 241, 0.5) !important;
        background-color: rgba(17, 24, 39, 0.6) !important;
    }
    
    div[data-testid="stFileUploader"] section {
        background-color: transparent !important;
    }

    /* Dashboard metrics style overrides */
    .dashboard-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }

    .custom-metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.3) 0%, rgba(15, 23, 42, 0.5) 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 20px;
        display: flex;
        align-items: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }

    .custom-metric-card:hover {
        border-color: rgba(99, 102, 241, 0.25);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.05);
        transform: translateY(-2px);
    }

    .metric-icon-wrapper {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        margin-right: 16px;
        background: rgba(99, 102, 241, 0.1);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.15);
    }

    .metric-icon-wrapper.danger {
        background: rgba(239, 68, 68, 0.1);
        color: #F87171;
        border-color: rgba(239, 68, 68, 0.15);
    }

    .metric-icon-wrapper.success {
        background: rgba(16, 185, 129, 0.1);
        color: #34D399;
        border-color: rgba(16, 185, 129, 0.15);
    }

    .metric-content {
        display: flex;
        flex-direction: column;
    }

    .metric-label {
        font-size: 11px;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 700;
        margin-top: 4px;
        color: #F8FAFC;
    }

    .metric-value.primary {
        color: #818CF8;
    }

    .metric-value.danger {
        color: #F87171;
    }

    .metric-value.success {
        color: #34D399;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CACHED MODEL LOADING
# ==========================================
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(os.path.join(VRD_DIR, "local_bge_model"))

@st.cache_data(show_spinner=False)
def load_jd_brain():
    try:
        embed_path = os.path.join(VRD_DIR, "jd_embeddings.npz")
        meta_path = os.path.join(VRD_DIR, "jd_metadata.json")
        
        data = np.load(embed_path)
        v_core, v_neg = data['v_core'], data['v_neg']
        if len(v_core.shape) == 2: v_core = v_core[0]
        if len(v_neg.shape) == 2: v_neg = v_neg[0]
        
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        return v_core, v_neg, meta
    except Exception as e:
        return None, None, None

# ==========================================
# 3. RANKING LOGIC
# ==========================================
def score_candidates(candidates_data, v_core, v_neg, meta, embedder, progress_info, progress_bar):
    constraints = meta.get("metadata_constraints", {})
    min_yoe = float(constraints.get("min_yoe", 5.0))
    max_yoe = float(constraints.get("max_yoe", 9.0))

    def _yoe_modifier(candidate_yoe):
        if min_yoe <= candidate_yoe <= max_yoe: return 1.00
        if min_yoe - 1 <= candidate_yoe < min_yoe or max_yoe < candidate_yoe <= max_yoe + 1: return 0.92
        if min_yoe - 2 <= candidate_yoe < min_yoe - 1: return 0.78
        if candidate_yoe > max_yoe + 1: return 0.82
        return 0.50

    @functools.lru_cache(maxsize=10000)
    def cached_embed_and_score(text: str):
        v_job = embedder.encode(text, normalize_embeddings=True)
        return float(np.dot(v_job, v_core)) - (0.3 * float(np.dot(v_job, v_neg)))

    scored = []
    dropped_hps = 0
    total = len(candidates_data)
    
    for idx, candidate in enumerate(candidates_data):
        if idx % max(1, total // 100) == 0 or idx == total - 1:
            progress_info.markdown(f"<div style='font-size:12px; color:#94A3B8; margin-bottom:5px;'>Scoring progress: <b>{idx+1:,}</b> / <b>{total:,}</b> candidates evaluated...</div>", unsafe_allow_html=True)
            progress_bar.progress(min(1.0, (idx + 1) / total))

        if is_honeypot(candidate):
            dropped_hps += 1
            continue

        career = candidate.get("career_history", [])
        if not career: continue

        yoe = candidate.get("profile", {}).get("years_of_experience", 0)
        total_weighted_score = 0.0
        total_weight = 0.0

        for job in career:
            desc = job.get("description", "")
            if not desc: continue
            weight = max(job.get("duration_months", 1), 1) * (1.5 if job.get("is_current", False) else 1.0)
            total_weighted_score += cached_embed_and_score(desc) * weight
            total_weight += weight

        raw_career_score = (total_weighted_score / total_weight) if total_weight > 0 else 0.0
        base_score = max(0, raw_career_score * 100)
        
        final_score = base_score * _yoe_modifier(yoe) * compute_signal_multiplier(candidate)
        scored.append((final_score, candidate, base_score))

    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    
    from main_ranker import generate_detailed_reasoning
    results = []
    for rank_idx, (score, candidate, base_score) in enumerate(scored[:100], start=1):
        cid = candidate["candidate_id"]
        rounded_score = round(float(score), 4)
        reasoning = generate_detailed_reasoning(candidate, score, rank_idx, meta)
        results.append({
            "candidate_id": cid,
            "rank": rank_idx, 
            "score": rounded_score, 
            "reasoning": reasoning
        })
        
    return results, dropped_hps

# ==========================================
# 4. PIPELINE ANTIMATION DEFINITIONS
# ==========================================
steps = [
    ("Embedder Ingestion", "Initialize and load SentenceTransformer (BGE-small) model in memory."),
    ("Dataset Parsing", "Parse JSONL structure and validate schema fields."),
    ("Universal Elimination", "Identify and purge honeypots, ghost profiles, and fraudulent candidates."),
    ("Dual-Vector Matching", "Compute positive-negative semantic subtraction scores over career history."),
    ("Reasoning & Export", "Inject candidate metrics, build explanation briefs, and format output.")
]

def get_pipeline_html(current_step_idx, step_status):
    html_lines = []
    html_lines.append('<div style="background: rgba(11, 15, 26, 0.6); border: 1px solid rgba(99, 102, 241, 0.12); border-radius: 16px; padding: 24px; margin-bottom: 25px; box-shadow: 0 8px 32px rgba(0,0,0,0.2);">')
    html_lines.append('<h4 style="margin: 0 0 20px 0; color: #F8FAFC; font-size: 16px; font-weight: 700; font-family: \'Space Grotesk\', sans-serif; letter-spacing: -0.01em;">Pipeline Execution Progress</h4>')
    
    for idx, (title, desc) in enumerate(steps):
        if idx < current_step_idx:
            icon = "✓"
            icon_style = "background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.25);"
            card_style = "border: 1px solid rgba(16, 185, 129, 0.12); background: rgba(16, 185, 129, 0.02);"
            badge = '<span style="color: #34D399; font-size: 10px; font-weight: 700; background: rgba(52, 211, 153, 0.12); padding: 3px 8px; border-radius: 6px;">DONE</span>'
        elif idx == current_step_idx:
            if step_status == "running":
                icon = "⚡"
                icon_style = "background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.35); animation: pulse 1.5s infinite;"
                card_style = "border: 1px solid rgba(168, 85, 247, 0.3); background: rgba(168, 85, 247, 0.05); box-shadow: 0 0 15px rgba(168, 85, 247, 0.05);"
                badge = '<span style="color: #C084FC; font-size: 10px; font-weight: 700; background: rgba(168, 85, 247, 0.15); padding: 3px 8px; border-radius: 6px; animation: pulse 1.5s infinite;">PROCESSING</span>'
            elif step_status == "success":
                icon = "✓"
                icon_style = "background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.25);"
                card_style = "border: 1px solid rgba(16, 185, 129, 0.12); background: rgba(16, 185, 129, 0.02);"
                badge = '<span style="color: #34D399; font-size: 10px; font-weight: 700; background: rgba(52, 211, 153, 0.12); padding: 3px 8px; border-radius: 6px;">DONE</span>'
            else:
                icon = "—"
                icon_style = "background: rgba(255,255,255,0.03); color: #475569; border: 1px solid rgba(255,255,255,0.06);"
                card_style = "border: 1px solid rgba(255, 255, 255, 0.03); background: rgba(255, 255, 255, 0.01);"
                badge = '<span style="color: #64748B; font-size: 10px; font-weight: 700; background: rgba(100, 116, 139, 0.08); padding: 3px 8px; border-radius: 6px;">PENDING</span>'
        else:
            icon = "—"
            icon_style = "background: rgba(255,255,255,0.03); color: #475569; border: 1px solid rgba(255,255,255,0.06);"
            card_style = "border: 1px solid rgba(255, 255, 255, 0.03); background: rgba(255, 255, 255, 0.01);"
            badge = '<span style="color: #64748B; font-size: 10px; font-weight: 700; background: rgba(100, 116, 139, 0.08); padding: 3px 8px; border-radius: 6px;">PENDING</span>'
            
        html_lines.append(f'<div style="display: flex; align-items: center; padding: 14px 18px; border-radius: 12px; margin-bottom: 12px; transition: all 0.3s ease; {card_style}">')
        html_lines.append(f'<div style="margin-right: 16px; font-size: 13px; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; border-radius: 50%; font-weight: 700; {icon_style}">{icon}</div>')
        html_lines.append(f'<div style="flex-grow: 1;">')
        html_lines.append(f'<div style="display: flex; justify-content: space-between; align-items: center;">')
        html_lines.append(f'<span style="font-weight: 600; color: #F1F5F9; font-size: 14px; font-family: \'Plus Jakarta Sans\', sans-serif;">{title}</span>')
        html_lines.append(f'{badge}')
        html_lines.append(f'</div>')
        html_lines.append(f'<div style="font-size: 12px; color: #94A3B8; margin-top: 3px; font-family: \'Plus Jakarta Sans\', sans-serif;">{desc}</div>')
        html_lines.append(f'</div>')
        html_lines.append(f'</div>')
    html_lines.append('</div>')
    return "".join(html_lines)

@st.cache_resource(show_spinner=False)
def load_llm():
    from llama_cpp import Llama
    model_path = os.path.join(VRD_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    return Llama(
        model_path=model_path,
        n_ctx=8192,
        n_threads=4,
        verbose=False
    )

def process_custom_jd(jd_text):
    from docx import Document
    import sys
    sys.path.append(VRD_DIR)
    from JD_parser import parse_jd, expand_disqualifiers, create_embeddings
    
    llm = load_llm()
    embedder = load_embedder()
    
    parsed_jd = parse_jd(llm, jd_text)
    expanded_negatives = expand_disqualifiers(llm, parsed_jd.get("abstract_disqualifiers", []), parsed_jd.get("job_title", "Professional"))
    v_core_arr, v_culture_arr, v_neg_arr = create_embeddings(embedder, parsed_jd, expanded_negatives)
    
    v_core_processed = v_core_arr[0] if len(v_core_arr.shape) == 2 else v_core_arr
    v_neg_processed = v_neg_arr[0] if len(v_neg_arr.shape) == 2 else v_neg_arr
    
    return v_core_processed, v_neg_processed, parsed_jd

# ==========================================
# 5. UI LAYOUT (Vertical Spacious Stack)
# ==========================================
# Elegant top banner header (single line HTML list to avoid Markdown block parsing bugs)
st.markdown(
    '<div style="text-align: center; padding: 2rem 0; margin-bottom: 2rem; background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.05) 100%); border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.15);">'
    '<h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; background: linear-gradient(to right, #818CF8, #C084FC); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Redrob Candidate Ranker</h1>',
    unsafe_allow_html=True
)

# Initialize Session State for JD embeddings
if "v_core" not in st.session_state:
    v_core, v_neg, meta = load_jd_brain()
    st.session_state["v_core"] = v_core
    st.session_state["v_neg"] = v_neg
    st.session_state["meta"] = meta

v_core = st.session_state.get("v_core")
v_neg = st.session_state.get("v_neg")
meta = st.session_state.get("meta")

# 1. Job Description & Specification section
st.subheader("💼 Job Description Target Specification")
with st.container(border=True):
    jd_source = st.radio(
        "Select Target Job Description Source:",
        ["Use Pre-computed Hackathon JD (Senior AI Engineer)", "Upload Custom JD File (.docx, .txt)", "Paste Raw JD Text"],
        help="Select whether to use the precomputed Senior AI Engineer JD or specify a new custom Job Description."
    )
    
    uploaded_jd_file = None
    pasted_jd_text = ""
    
    if jd_source == "Upload Custom JD File (.docx, .txt)":
        uploaded_jd_file = st.file_uploader("Upload Job Description Document", type=["docx", "txt"])
    elif jd_source == "Paste Raw JD Text":
        pasted_jd_text = st.text_area("Paste Job Description details here:", height=200, placeholder="Requirements, culture guidelines, YOE limits...")
        
    if jd_source != "Use Pre-computed Hackathon JD (Senior AI Engineer)":
        st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
        process_jd_btn = st.button("⚙️ Parse & Process Job Description", use_container_width=True)
        
        if process_jd_btn:
            jd_text = ""
            if jd_source == "Upload Custom JD File (.docx, .txt)" and uploaded_jd_file is not None:
                if uploaded_jd_file.name.endswith(".docx"):
                    from docx import Document
                    doc = Document(uploaded_jd_file)
                    jd_text = "\n".join([para.text for para in doc.paragraphs])
                else:
                    jd_text = uploaded_jd_file.read().decode("utf-8")
            elif jd_source == "Paste Raw JD Text":
                jd_text = pasted_jd_text
                
            if not jd_text.strip():
                st.error("Please provide job description content first.")
            else:
                with st.spinner("Processing Job Description (Running Qwen LLM & BGE Embedder)..."):
                    try:
                        new_v_core, new_v_neg, new_meta = process_custom_jd(jd_text)
                        st.session_state["v_core"] = new_v_core
                        st.session_state["v_neg"] = new_v_neg
                        st.session_state["meta"] = new_meta
                        
                        v_core = new_v_core
                        v_neg = new_v_neg
                        meta = new_meta
                        st.success("Successfully parsed custom Job Description and generated semantic features!")
                    except Exception as e:
                        st.error(f"Error parsing Job Description: {e}")

    # Active Job profile Details display
    if v_core is not None and meta:
        constraints = meta.get("metadata_constraints", {})
        min_yoe = float(constraints.get("min_yoe", 5.0))
        max_yoe = float(constraints.get("max_yoe", 9.0))
        
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 20px 0;'>", unsafe_allow_html=True)
        
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown(f"**Loaded Role:** `{meta.get('job_title', 'Professional')}`")
        with info_col2:
            st.markdown(f"**Experience Band:** `{int(min_yoe)} - {int(max_yoe)} YOE`")
            
        with st.expander("🔍 View Must-Have Hard Skills & Target Constraints", expanded=False):
            for skill in meta.get("must_have_hard_skills", []):
                st.markdown(f"- {skill}")
            if constraints.get("preferred_locations"):
                locs = ", ".join(constraints["preferred_locations"])
                st.markdown(f"**Preferred Location Bands:** `{locs}`")

# 2. Ingestion Settings & Run Button
st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)
st.subheader("⚙️ Candidate Ingestion Configuration")
with st.container(border=True):
    data_source = st.radio(
        "Select Ingestion Dataset:",
        ["Use Full Hackathon Dataset (100k)", "Upload Custom Subset (JSONL)"],
        help="Choose whether to run candidate ranking on the complete 100,000 candidate dataset or a specific uploaded JSONL subset."
    )
    
    uploaded_file = None
    if data_source == "Upload Custom Subset (JSONL)":
        uploaded_file = st.file_uploader(
            "Drag & drop candidate file",
            type=["jsonl", "json"],
            help="Ensure candidates match the standard schema (supporting both JSONL and standard JSON arrays)."
        )
        
    st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Run Pipeline", use_container_width=True)

# 3. Execution Visualizer
if run_btn:
    if v_core is None:
        st.error("Cannot run ranking: No target Job Description is loaded.")
        st.stop()
        
    candidates_data = []
    st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)
    visualizer_placeholder = st.empty()
    
    # Step 1: Embedder Ingestion
    visualizer_placeholder.markdown(get_pipeline_html(0, "running"), unsafe_allow_html=True)
    embedder = load_embedder()
    time.sleep(0.4)
    
    # Step 2: Dataset Loading
    visualizer_placeholder.markdown(get_pipeline_html(1, "running"), unsafe_allow_html=True)
    if data_source == "Use Full Hackathon Dataset (100k)":
        full_path = os.path.join(ROOT_DIR, "candidates.jsonl")
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        candidates_data.append(json.loads(line))
        except FileNotFoundError:
            st.error(f"Could not find candidates.jsonl in: {full_path}")
            st.stop()
    else:
        if uploaded_file is None:
            st.error("Please upload a candidate file first.")
            st.stop()
        if uploaded_file.name.endswith(".jsonl"):
            for line in uploaded_file:
                decoded_line = line.decode('utf-8').strip()
                if decoded_line:
                    candidates_data.append(json.loads(decoded_line))
        else:
            try:
                content = uploaded_file.read().decode('utf-8')
                data = json.loads(content)
                if isinstance(data, list):
                    candidates_data = data
                elif isinstance(data, dict):
                    candidates_data = [data]
                else:
                    st.error("Invalid JSON format. Expected an array of candidates.")
                    st.stop()
            except Exception as e:
                st.error(f"Error parsing JSON file: {e}")
                st.stop()
    time.sleep(0.4)
    
    # Step 3: Phase 1: Universal Elimination (Honeypot Filter)
    visualizer_placeholder.markdown(get_pipeline_html(2, "running"), unsafe_allow_html=True)
    time.sleep(0.4)
    
    # Step 4: Phase 2: Dual-Vector Scoring
    visualizer_placeholder.markdown(get_pipeline_html(3, "running"), unsafe_allow_html=True)
    
    progress_container = st.container()
    with progress_container:
        progress_info = st.empty()
        progress_bar = st.progress(0)
        
    start_time = time.time()
    results, dropped_hps = score_candidates(candidates_data, v_core, v_neg, meta, embedder, progress_info, progress_bar)
    elapsed = time.time() - start_time
    
    progress_container.empty()
    
    # Step 5: Phase 3: Reasoning & Formatting
    visualizer_placeholder.markdown(get_pipeline_html(4, "running"), unsafe_allow_html=True)
    time.sleep(0.4)
    
    # Complete!
    visualizer_placeholder.markdown(get_pipeline_html(4, "success"), unsafe_allow_html=True)
    
    # Results UI Showcase
    st.markdown("<h3 style='margin-top: 25px; color:#F8FAFC;'>🏆 Ranked Candidate Results</h3>", unsafe_allow_html=True)
    
    # Quick Stats Row
    best_score = results[0]["score"] if results else 0.0
    st.markdown(f"""
    <div class="dashboard-metrics">
        <div class="custom-metric-card">
            <div class="metric-icon-wrapper">👥</div>
            <div class="metric-content">
                <span class="metric-label">Total Ingested</span>
                <span class="metric-value primary">{len(candidates_data):,}</span>
            </div>
        </div>
        <div class="custom-metric-card">
            <div class="metric-icon-wrapper danger">🛡️</div>
            <div class="metric-content">
                <span class="metric-label">Honeypots Blocked</span>
                <span class="metric-value danger">{dropped_hps}</span>
            </div>
        </div>
        <div class="custom-metric-card">
            <div class="metric-icon-wrapper success">🏆</div>
            <div class="metric-content">
                <span class="metric-label">Top Match Score</span>
                <span class="metric-value success">{best_score:.1f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    
    if results:
        df = pd.DataFrame(results)
        
        # Check if installed Streamlit version supports on_select (Streamlit 1.35.0+)
        try:
            parts = [int(p) for p in st.__version__.split(".")]
            supports_selection = (parts[0] > 1) or (parts[0] == 1 and parts[1] >= 35)
        except Exception:
            supports_selection = False

        df_kwargs = {
            "use_container_width": True,
            "hide_index": True,
            "column_config": {
                "candidate_id": st.column_config.TextColumn("Candidate ID", width="medium"),
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "score": st.column_config.NumberColumn("Score", width="small"),
                "reasoning": st.column_config.TextColumn("Reasoning Brief", width=2500)
            }
        }
        
        if supports_selection:
            df_kwargs["on_select"] = "rerun"
            df_kwargs["selection_mode"] = "single-row"
            st.markdown("<div style='font-size: 13px; color: #94A3B8; margin-bottom: 10px;'>💡 <b>Tip:</b> Click on any candidate row in the table below to view their complete, non-truncated detailed explanatory brief.</div>", unsafe_allow_html=True)
            
        event = st.dataframe(df, **df_kwargs)
        
        if supports_selection and event is not None and getattr(event, "selection", None):
            rows = event.selection.get("rows", [])
            if rows:
                selected_row_idx = rows[0]
                selected_cand = df.iloc[selected_row_idx]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(168, 85, 247, 0.08) 100%); 
                            border: 1px solid rgba(99, 102, 241, 0.3); 
                            border-radius: 12px; 
                            padding: 24px; 
                            margin-top: 20px;
                            margin-bottom: 20px;
                            box-shadow: 0 8px 30px rgba(99, 102, 241, 0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 10px;">
                        <span style="font-size: 18px; font-weight: 700; color: #F1F5F9; font-family: 'Space Grotesk', sans-serif;">
                            🔍 Detailed Explanatory Brief — {selected_cand['candidate_id']}
                        </span>
                        <span style="font-size: 13px; font-weight: 600; background: rgba(52, 211, 153, 0.15); color: #34D399; padding: 4px 10px; border-radius: 20px;">
                            Rank #{selected_cand['rank']}
                        </span>
                    </div>
                    <p style="font-size: 15px; line-height: 1.6; color: #F1F5F9; margin: 0; font-family: 'Plus Jakarta Sans', sans-serif;">
                        {selected_cand['reasoning']}
                    </p>
                    <div style="margin-top: 15px; font-size: 13px; color: #94A3B8; font-family: 'Plus Jakarta Sans', sans-serif;">
                        <b>Algorithm Score:</b> <span style="color: #c084fc; font-weight: 700;">{selected_cand['score']:.4f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download submission.csv",
            data=csv,
            file_name='team_LittleBoy.csv',
            mime='text/csv',
            use_container_width=True
        )
    else:
        st.warning("No candidates passed the required filters.")
