# Redrob Hackathon v4 — Candidate Discovery & Ranking Pipeline

> **Team Name:** Little Boy  
> **Team Members:** Karan Dhote (<karan.sdhote@gmail.com>)  

---

## 🚀 Sandbox Environments

As per the hackathon instructions (Section 10.5), we provide two sandbox environments for reproducing and testing our candidate ranking pipeline on small samples:

### Option A: Google Colab (Recommended & Instant)
Run the pipeline end-to-end (cloning, dependency installation, model preparation, ranking, validation, and preview) with a single click.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karan825/RetrievAl/blob/main/sandbox_demo.ipynb)

### Option B: Streamlit Community Cloud
*Note: Due to compilation of the `llama-cpp-python` C extensions from source in Streamlit's free tier, the first container build may take 10-15 minutes ("Your app is in the oven"). Once compiled, subsequent loads are cached and fast.*

- **Sandbox URL:** [Streamlit Cloud Demo App](https://share.streamlit.io/karan825/retrieval/main/vrd/app.py)

---

## 🛠️ Local Installation & Setup

### 1. Clone & Initialize Environment
```bash
git clone https://github.com/Karan825/RetrievAl.git
cd RetrievAl
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pre-fetch Models Programmatically
Our pipeline relies on compact local models served fully offline. Download them to your local directory using:
```bash
python vrd/download_model.py
```

### 3. Run Streamlit UI Locally
```bash
streamlit run vrd/app.py
```

---

## 💻 CLI Reproduction Command

To run the pipeline on the full dataset or any subset as required by the Stage 3 docker test:
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

---

## 🧠 Architecture Overview

Our candidate discovery system is designed for massive datasets (100k+ candidates) operating under a strict **5-minute CPU-only wall-clock budget** without external network dependencies.

```
       [Raw Candidates (100k JSONL)]
                     │
                     ▼
        1. Universal Elimination (Honeypot Filter)
                     │
                     ▼
        2. Dual-Vector Semantic Matching (BGE-small)
                     │
                     ▼
        3. Feature Modifiers (YOE brackets & Recruiter Response)
                     │
                     ▼
        4. Tie-Breaking & Extraction (Top 100)
                     │
                     ▼
        5. Explanatory Brief Generation (Offline Qwen 2.5 1.5B LLM)
                     │
                     ▼
              [submission.csv]
```

### 1. Universal Elimination (Honeypots)
We check each profile against a strict list of behavioral signals to identify and reject fake accounts, impossible experience durations, and keyword-stuffed entries before allocating compute.

### 2. Dual-Vector Semantic Matching
We encode job description targets into two distinct dense vectors:
- **Core Vector**: Represents the must-have capabilities.
- **Negative Vector**: Represents disqualifying aspects (consulting background, generic keywords).

Each candidate's career descriptions are encoded using the local BGE model, and we subtract negative alignment from positive alignment to produce a refined semantic score.

### 3. Feature Modifiers & Constraints
- **Experience Band Scaling**: Scores are scaled depending on target ranges parsed dynamically from the job description (e.g., target bracket = `1.00x` multiplier, minor deviations = `0.92x`, extreme deviations = `0.50x`).
- **Recruiter Response Rate**: Active responsive indicators are multiplied to favor reliable candidates.

### 4. LLM Explanatory Briefs
The top 100 candidates are passed to an offline instance of the **Qwen 2.5 1.5B GGUF** model using the `llama-cpp-python` engine to generate detailed, candidate-specific reasoning matching all Stage 4 review guidelines.
