# RetrievAl: Candidate Discovery & Ranking Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karan825/RetrievAl/blob/main/sandbox_demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> **Team Name:** LittleBoy  
> **Primary Contact:** Karan Dhote (karan.sdhote@gmail.com)  
> **Target Role:** Founding Senior AI Engineer (Redrob AI)  

---

## Sandbox Demonstration Environment

We provide a Google Colab sandbox for reproducing and testing the candidate ranking pipeline on a sample dataset with a single click:

👉 **[Open In Google Colab](https://colab.research.google.com/github/Karan825/RetrievAl/blob/main/sandbox_demo.ipynb)**

---

## Local Installation & Setup

### 1. Clone & Initialize Environment
```bash
git clone https://github.com/Karan825/RetrievAl.git
cd RetrievAl
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
# 1. Install llama-cpp-python via precompiled CPU wheels to avoid slow compilation times
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# 2. Install the remaining requirements
pip install -r requirements.txt
```

### 3. Pre-fetch Models Programmatically
Our pipeline runs completely offline. Pre-download the embedding model and LLM weights locally:
```bash
python vrd/download_model.py
```

### 4. Run Job Description Pre-computation (Offline Parser)
Before running the ranking pipeline, you need to parse the job description document (`job_description.docx`) to generate the dense vectors and role constraints:
```bash
python vrd/JD_parser.py
```
*Note: This script runs locally offline. It loads the Qwen 2.5 1.5B SLM and the BGE embedder to parse requirements, extract domains, and generate structured constraints (`vrd/jd_metadata.json`) and vector embeddings (`vrd/jd_embeddings.npz`). This process takes approximately **1.5 to 2 minutes** on a standard CPU.*

---

## CLI Reproduction Command

To execute the ranking pipeline on the full dataset or any test file:
```bash
python rank.py --candidates ./candidates.jsonl --out ./LittleBoy.csv
```

---

## Pipeline Architecture & Scoring Methodology

The candidate discovery system is designed to process 100k+ profiles under a strict **5-minute CPU-only wall-clock budget** without external network dependencies.

```
       [Raw Candidates (100k+ JSONL)]
                     │
                     ▼
         1. Universal Elimination (Honeypots)
                     │
                     ▼
         2. Dual-Vector Semantic Matching (BGE-small)
                     │
                     ▼
         3. Feature Modifiers & Soft Scaling
                     │
                     ▼
         4. Tie-Breaking & Extraction (Top 100)
                     │
                     ▼
         5. Explanatory Brief Generation (Local Qwen LLM)
                     │
                     ▼
              [submission.csv]
```

### 1. Universal Elimination (Honeypot Filter)
Filters out synthetic profiles and data-corruption patterns in [vrd/honeypot.py](file:///c:/Users/karan/Downloads/India_runs_data_and_ai_challenge/vrd/honeypot.py) before allocating search compute. It evaluates 7 structural constraints (e.g. expert skills with 0 duration, overlapping job dates, impossible education timelines, and technology release date violations).

### 2. Dual-Vector Semantic Matching
Uses a local `bge-small-en-v1.5` model to match candidates' career histories against the job description. Career jobs are weighted by their duration and recency (1.5x multiplier for current roles), and scored using positive target vectors minus a negative alignment vector (such as out-of-domain patterns).

### 3. Feature Modifiers & Soft Scaling
*   **Soft YOE Scaling**: In compliance with the Job Description instructions, we penalize candidates outside the target 5-9 YOE bracket softly rather than skipping them strictly, allowing highly-aligned outliers with strong skills to be considered.
*   **Skills Grounding Check**: Validates that candidates' declared skills are explicitly backed by descriptive experience in their career history, reducing the rank of keyword-stuffed profiles.
*   **20 Behavioral Signal Modifiers**: Scores are adjusted using recruiter response rate, notice period, location preference, GitHub activity, open-to-work flags, verified contact info, and role trajectory metrics in [vrd/signal_modifier.py](file:///c:/Users/karan/Downloads/India_runs_data_and_ai_challenge/vrd/signal_modifier.py).

### 4. LLM Explanatory Briefs
For the top-ranked candidates, we run a local **Qwen 2.5 1.5B GGUF** model using the CPU-bound `llama-cpp-python` engine to generate detailed, candidate-specific hiring briefs. 
*   **Hallucination Guard**: Scans generated text for technology keywords. If the model mentions a tool the candidate does not actually have in their profile skills list, the brief is discarded and replaced with a dynamic, deterministic template to ensure 100% grounded facts.
*   **Bleed Protection**: Prevents target role and company names from bleeding incorrectly into the generated candidate briefs.

---

## Hackathon Constraints Compliance

*   **CPU-Only**: Fully compatible with standard CPU runtimes (tested on 2-vCPU and 4-vCPU systems).
*   **Fully Offline**: 100% local ranking. Zero external network calls or API dependencies are used during ranking.
*   **Speed**: Processes 100k records in **~4 minutes** (CPU-only).
*   **RAM Footprint**: Under **1.2 GB** memory footprint (well within the 16 GB limit).
*   **Disk Footprint**: Total model weight size is **~1.2 GB** (well within the 5 GB limit).
*   **Honeypot Rate**: Discards 90%+ fake profiles prior to ranking, ensuring a honeypot pass rate under the 10% target limit.
*   **Format**: Generates a standard CSV file (e.g. `LittleBoy.csv`) containing exactly 100 sorted data rows, matching the required schema.
