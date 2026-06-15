# Redrob Intelligent Candidate Ranking — Submission

## What this does

A 5-stage deterministic pipeline that ranks 100,000 candidates for a
Senior AI Engineer role. Runs in **~9 seconds on CPU**, no GPU, no
network calls, no external APIs.

## Reproduce the submission

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO
cd YOUR_REPO

# 2. Install dependencies (pure stdlib — nothing to install)
# Python 3.9+ required
python --version

# 3. Place the candidates file in the repo root
#    (candidates.jsonl should be ~487 MB)

# 4. Run
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 5. Validate
python validate_submission.py submission.csv
```

Expected output:
```
[rank.py] Reading candidates from: candidates.jsonl
[rank.py] Total candidates read       : 100,000
[rank.py] Excluded (honeypots)        : 55
[rank.py] Excluded (hard constraints) : 8,931
[rank.py] Eligible candidates         : 91,014
[rank.py] Writing top 100 to: submission.csv
[rank.py] Done in ~9s  OK
```

## Architecture

```
candidates.jsonl (100k)
        |
        v
Stage 1 — Honeypot Elimination
        Removes profiles with expert skills at 0 months duration
        and severe YOE fabrication (profile vs career history > 8yr gap)
        |
        v
Stage 2 — Hard Constraint Filter
        Removes careers entirely at consulting firms (TCS/Infosys/Wipro etc.)
        |
        v
Stage 3 — Career Relevance Scoring
        Maps each of 44 unique career description templates to a
        base relevance score (0-100). Applies:
          - Product company boost (x1.25) for Swiggy, Zomato, CRED, etc.
          - Consulting firm penalty (x0.65)
          - Recency weight (current job x1.5)
          - YOE modifier (target 6-8 years)
          - Job-hopper penalty (>50% jobs < 15 months = 0.70x)
        |
        v
Stage 4 — Behavioral Signal Multiplier
        All 23 Redrob signals applied as a composite multiplier:
          - Last active date (inactive >6 months = 0.75x)
          - Recruiter response rate (<20% = 0.55x)
          - Notice period (<30 days = 1.15x bonus)
          - GitHub activity score
          - Platform skill assessment scores
          - Interview completion rate
          - Location (Pune/Noida = 1.10x)
        |
        v
Stage 5 — Dynamic Reasoning Generation
        Generates a 1-2 sentence, fact-grounded reasoning per candidate
        using actual profile data — never hallucinated.
        |
        v
submission.csv (top 100)
```

## Key design decisions

**Why not embeddings / LLMs?**
The 100k candidates have career descriptions generated from 44 unique
templates. Template classification (O(n) dict lookup) is faster,
more accurate, and fully explainable vs. embedding cosine similarity
which is tricked by keyword-stuffed skills sections.

**Why product company boost?**
The JD explicitly asks for "product company" experience over "services/
consulting" experience. A candidate who ran collaborative filtering at
Swiggy is more relevant than the same description at Wipro.

**Why not score the skills section?**
The dataset contains thousands of non-technical candidates (HR managers,
accountants) with AI keywords (FAISS, Pinecone, RAG) in their skills —
a deliberate trap. We anchor scoring entirely on career descriptions.

## Compute environment

- Python 3.12
- CPU only
- No GPU
- No network during ranking
- Peak RAM: ~1.2 GB (for 100k candidate objects in memory)
- Runtime: ~9 seconds wall-clock

## File structure

```
.
├── rank.py                  # Entry point
├── ranker/
│   ├── __init__.py
│   ├── honeypot.py          # Stage 1
│   ├── filters.py           # Stage 2
│   ├── career_scorer.py     # Stage 3
│   ├── signal_modifier.py   # Stage 4
│   └── reasoning.py         # Stage 5
├── app.py                   # Streamlit sandbox demo
├── requirements.txt
├── submission_metadata.yaml
├── validate_submission.py   # From hackathon bundle
└── README.md
```
