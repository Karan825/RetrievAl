import json
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

# Setup paths
VRD_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(VRD_DIR)
sys.path.append(VRD_DIR)

from honeypot import is_honeypot
from signal_modifier import compute_signal_multiplier

def load_jd_brain():
    embed_path = os.path.join(VRD_DIR, "jd_embeddings.npz")
    meta_path = os.path.join(VRD_DIR, "jd_metadata.json")
    
    data = np.load(embed_path)
    v_core, v_neg = data['v_core'], data['v_neg']
    if len(v_core.shape) == 2: v_core = v_core[0]
    if len(v_neg.shape) == 2: v_neg = v_neg[0]
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    return v_core, v_neg, meta

# Load active JD Brain and Model
print("Loading target embeddings and BGE model...")
v_core, v_neg, meta = load_jd_brain()
embedder = SentenceTransformer(os.path.join(VRD_DIR, "local_bge_model"))

# Define YOE constraints
constraints = meta.get("metadata_constraints", {})
min_yoe = float(constraints.get("min_yoe", 5.0))
max_yoe = float(constraints.get("max_yoe", 9.0))

def _yoe_modifier(candidate_yoe):
    if min_yoe <= candidate_yoe <= max_yoe: return 1.00
    if min_yoe - 1 <= candidate_yoe < min_yoe or max_yoe < candidate_yoe <= max_yoe + 1: return 0.92
    if min_yoe - 2 <= candidate_yoe < min_yoe - 1: return 0.78
    if candidate_yoe > max_yoe + 1: return 0.82
    return 0.50

# Mock Candidates data structure (Complies with 7-digit ID format CAND_XXXXXXX)
candidates = [
    # 1. CAND_0000001: Perfect Match (Senior AI Engineer, 7 YOE, target skills, ideal signals)
    {
        "candidate_id": "CAND_0000001",
        "profile": {"current_title": "Senior AI Engineer", "years_of_experience": 7.0},
        "skills": [
            {"name": "Python", "proficiency": "Expert", "duration_months": 84},
            {"name": "Sentence-Transformers", "proficiency": "Expert", "duration_months": 36},
            {"name": "Pinecone", "proficiency": "Advanced", "duration_months": 24}
        ],
        "career_history": [
            {
                "title": "Senior Machine Learning Engineer",
                "description": "Designed and deployed embeddings-based retrieval systems using sentence-transformers and Pinecone vector database for hybrid search, improving search engagement metrics by 20%. Deployed vector search infrastructure at scale.",
                "duration_months": 36,
                "is_current": True
            },
            {
                "title": "AI Engineer",
                "description": "Developed NLP pipelines, designed offline evaluation frameworks including NDCG and MRR, and conducted offline-to-online correlation analysis.",
                "duration_months": 48,
                "is_current": False
            }
        ],
        "redrob_signals": {
            "last_active_date": "2026-06-05",
            "recruiter_response_rate": 0.95,
            "notice_period_days": 0,
            "open_to_work_flag": True,
            "github_activity_score": 85,
            "skill_assessment_scores": {"Python": 90, "Machine Learning": 95},
            "interview_completion_rate": 1.0,
            "profile_completeness_score": 95
        }
    },
    # 2. CAND_0000002: Strong Match (AI Developer, 6 YOE, good skills, positive signals)
    {
        "candidate_id": "CAND_0000002",
        "profile": {"current_title": "NLP Research Engineer", "years_of_experience": 6.0},
        "skills": [
            {"name": "Python", "proficiency": "Expert", "duration_months": 72},
            {"name": "Vector Search", "proficiency": "Advanced", "duration_months": 36}
        ],
        "career_history": [
            {
                "title": "AI Engineer",
                "description": "Built hybrid search pipelines with BM25 and vector database structures. Set up evaluations via offline NDCG benchmarks.",
                "duration_months": 72,
                "is_current": True
            }
        ],
        "redrob_signals": {
            "last_active_date": "2026-05-20",
            "recruiter_response_rate": 0.75,
            "notice_period_days": 30,
            "open_to_work_flag": True,
            "github_activity_score": 50,
            "skill_assessment_scores": {"Python": 80},
            "interview_completion_rate": 0.90,
            "profile_completeness_score": 85
        }
    },
    # 3. CAND_0000003: Good Match (Senior NLP Developer, 8 YOE, NLP/Search relevance)
    {
        "candidate_id": "CAND_0000003",
        "profile": {"current_title": "Senior NLP Engineer", "years_of_experience": 8.0},
        "skills": [
            {"name": "Python", "proficiency": "Advanced", "duration_months": 96},
            {"name": "NLP", "proficiency": "Expert", "duration_months": 60}
        ],
        "career_history": [
            {
                "title": "NLP Developer",
                "description": "Developed dense retrieval models and semantic search solutions. Conducted A/B testing for query ranking optimization.",
                "duration_months": 96,
                "is_current": True
            }
        ],
        "redrob_signals": {
            "last_active_date": "2026-06-01",
            "recruiter_response_rate": 0.60,
            "notice_period_days": 15,
            "open_to_work_flag": False,
            "github_activity_score": 45,
            "skill_assessment_scores": {"Machine Learning": 75},
            "interview_completion_rate": 0.80,
            "profile_completeness_score": 80
        }
    },
    # 4. CAND_0000004: Medium Match (Python Software Developer, 5 YOE, general developer, basic search)
    {
        "candidate_id": "CAND_0000004",
        "profile": {"current_title": "Python Software Developer", "years_of_experience": 5.0},
        "skills": [
            {"name": "Python", "proficiency": "Advanced", "duration_months": 60}
        ],
        "career_history": [
            {
                "title": "Python Engineer",
                "description": "Wrote complex Python algorithms, implemented basic text search filters and structured data API routes.",
                "duration_months": 60,
                "is_current": True
            }
        ],
        "redrob_signals": {
            "last_active_date": "2026-04-10",
            "recruiter_response_rate": 0.50,
            "notice_period_days": 45,
            "open_to_work_flag": True,
            "github_activity_score": 20,
            "skill_assessment_scores": {"Python": 60},
            "interview_completion_rate": 0.75,
            "profile_completeness_score": 75
        }
    },
    # 5. CAND_0000005: Medium Match (AI Architect, 9 YOE, high end experience, average signals)
    {
        "candidate_id": "CAND_0000005",
        "profile": {"current_title": "Lead AI Engineer", "years_of_experience": 9.0},
        "skills": [
            {"name": "AI Architecture", "proficiency": "Expert", "duration_months": 108}
        ],
        "career_history": [
            {
                "title": "AI Architect",
                "description": "Supervised building of recommendation architectures, set up offline data pipelines, and conducted search indexing audits.",
                "duration_months": 108,
                "is_current": True
            }
        ],
        "redrob_signals": {
            "last_active_date": "2026-03-15",
            "recruiter_response_rate": 0.40,
            "notice_period_days": 60,
            "open_to_work_flag": False,
            "github_activity_score": -1,
            "skill_assessment_scores": {},
            "interview_completion_rate": 0.60,
            "profile_completeness_score": 70
        }
    },
    # 6. CAND_0000006: Out of YOE range (Junior ML Developer, 3 YOE, under minimum limit)
    {
        "candidate_id": "CAND_0000006",
        "profile": {"current_title": "Junior ML Developer", "years_of_experience": 3.0},
        "skills": [
            {"name": "Python", "proficiency": "Advanced", "duration_months": 36}
        ],
        "career_history": [
            {
                "title": "ML Engineer",
                "description": "Trained word2vec embeddings, assisted senior developers in setting up dense vector search indexes.",
                "duration_months": 36,
                "is_current": True
            }
        ],
        "redrob_signals": {
            "last_active_date": "2026-06-08",
            "recruiter_response_rate": 0.90,
            "notice_period_days": 15,
            "open_to_work_flag": True,
            "github_activity_score": 60,
            "skill_assessment_scores": {"Python": 85},
            "interview_completion_rate": 0.95,
            "profile_completeness_score": 90
        }
    },
    # 7. CAND_0000007: Out of YOE range (Director of AI, 12 YOE, above maximum limit)
    {
        "candidate_id": "CAND_0000007",
        "profile": {"current_title": "Director of AI", "years_of_experience": 12.0},
        "skills": [
            {"name": "Machine Learning", "proficiency": "Expert", "duration_months": 144}
        ],
        "career_history": [
            {
                "title": "AI Director",
                "description": "Directed engineering teams building embeddings search engines and vector retrieval infrastructure for products.",
                "duration_months": 144,
                "is_current": True
            }
        ],
        "redrob_signals": {
            "last_active_date": "2026-05-15",
            "recruiter_response_rate": 0.85,
            "notice_period_days": 30,
            "open_to_work_flag": False,
            "github_activity_score": 75,
            "skill_assessment_scores": {"Machine Learning": 92},
            "interview_completion_rate": 1.0,
            "profile_completeness_score": 95
        }
    },
    # 8. CAND_0000008: Poor Match (React Frontend Engineer, 7 YOE, wrong skills)
    {
        "candidate_id": "CAND_0000008",
        "profile": {"current_title": "React Frontend Engineer", "years_of_experience": 7.0},
        "skills": [
            {"name": "React", "proficiency": "Expert", "duration_months": 84}
        ],
        "career_history": [
            {
                "title": "UI Developer",
                "description": "Created interactive user interfaces using React, styled with CSS frameworks, and connected to REST APIs.",
                "duration_months": 84,
                "is_current": True
            }
        ],
        "redrob_signals": {
            "last_active_date": "2026-06-02",
            "recruiter_response_rate": 0.90,
            "notice_period_days": 15,
            "open_to_work_flag": True,
            "github_activity_score": 50,
            "skill_assessment_scores": {"Frontend": 88},
            "interview_completion_rate": 0.95,
            "profile_completeness_score": 90
        }
    },
    # 9. CAND_0000009: Poor Match (Computer Vision Researcher, 7 YOE, wrong AI focus)
    {
        "candidate_id": "CAND_0000009",
        "profile": {"current_title": "Computer Vision Researcher", "years_of_experience": 7.0},
        "skills": [
            {"name": "Computer Vision", "proficiency": "Expert", "duration_months": 84}
        ],
        "career_history": [
            {
                "title": "CV Engineer",
                "description": "Built image classification models, optimized convolutional neural networks (CNNs), and applied object detection with YOLO.",
                "duration_months": 84,
                "is_current": True
            }
        ],
        "redrob_signals": {
            "last_active_date": "2026-06-03",
            "recruiter_response_rate": 0.88,
            "notice_period_days": 30,
            "open_to_work_flag": True,
            "github_activity_score": 65,
            "skill_assessment_scores": {"Computer Vision": 90},
            "interview_completion_rate": 0.95,
            "profile_completeness_score": 88
        }
    },
    # 10. CAND_0000010: Good skills but Ghost Developer (Very low signal multipliers)
    {
        "candidate_id": "CAND_0000010",
        "profile": {"current_title": "Senior AI Specialist", "years_of_experience": 8.0},
        "skills": [
            {"name": "Python", "proficiency": "Expert", "duration_months": 96}
        ],
        "career_history": [
            {
                "title": "AI Lead",
                "description": "Experienced NLP model designer with years of experience building vector database retrieval indexes.",
                "duration_months": 96,
                "is_current": True
            }
        ],
        "redrob_signals": {
            "last_active_date": "2022-01-10",
            "recruiter_response_rate": 0.05,
            "notice_period_days": 90,
            "open_to_work_flag": False,
            "github_activity_score": -1,
            "skill_assessment_scores": {},
            "interview_completion_rate": 0.10,
            "profile_completeness_score": 30
        }
    },
    # 11. CAND_0000011: Honeypot (zero duration expert skills)
    {
        "candidate_id": "CAND_0000011",
        "profile": {"current_title": "Senior AI Engineer", "years_of_experience": 7.0},
        "skills": [
            {"name": "Python", "proficiency": "Expert", "duration_months": 0},
            {"name": "Pinecone", "proficiency": "Advanced", "duration_months": 0}
        ],
        "career_history": [
            {
                "title": "Engineer",
                "description": "Experienced developer.",
                "duration_months": 84,
                "is_current": True
            }
        ],
        "redrob_signals": {
            "last_active_date": "2026-06-01",
            "recruiter_response_rate": 0.80
        }
    },
    # 12. CAND_0000012: Honeypot (Severe YOE discrepancy)
    {
        "candidate_id": "CAND_0000012",
        "profile": {"current_title": "AI Architect", "years_of_experience": 12.0},
        "skills": [],
        "career_history": [
            {
                "title": "ML Engineer",
                "description": "Worked on quick project.",
                "duration_months": 12,
                "is_current": True
            }
        ],
        "redrob_signals": {
            "last_active_date": "2026-06-01",
            "recruiter_response_rate": 0.80
        }
    },
    # 13. CAND_0000013: Honeypot (zero duration expert skills)
    {
        "candidate_id": "CAND_0000013",
        "profile": {"current_title": "Python Developer", "years_of_experience": 6.0},
        "skills": [
            {"name": "Python", "proficiency": "Advanced", "duration_months": 0},
            {"name": "Sentence-Transformers", "proficiency": "Expert", "duration_months": 0}
        ],
        "career_history": [
            {
                "title": "Dev",
                "description": "Some description.",
                "duration_months": 72,
                "is_current": True
            }
        ],
        "redrob_signals": {
            "last_active_date": "2026-06-01",
            "recruiter_response_rate": 0.80
        }
    },
    # 14. CAND_0000014: Honeypot (Severe YOE discrepancy)
    {
        "candidate_id": "CAND_0000014",
        "profile": {"current_title": "AI Specialist", "years_of_experience": 15.0},
        "skills": [],
        "career_history": [
            {
                "title": "Junior Developer",
                "description": "Short gig.",
                "duration_months": 24,
                "is_current": True
            }
        ],
        "redrob_signals": {
            "last_active_date": "2026-06-01",
            "recruiter_response_rate": 0.80
        }
    },
    # 15. CAND_0000015: No career history (Filtered out entirely in score loop)
    {
        "candidate_id": "CAND_0000015",
        "profile": {"current_title": "AI Enthusiast", "years_of_experience": 6.0},
        "skills": [],
        "career_history": [],
        "redrob_signals": {
            "last_active_date": "2026-06-01",
            "recruiter_response_rate": 0.80
        }
    }
]

# Calculate actual pipeline scores and statuses
results = []
jsonl_lines = []

for candidate in candidates:
    cid = candidate["candidate_id"]
    jsonl_lines.append(json.dumps(candidate))
    
    # Check Honeypot
    hp_status = "HONEYPOT (Eliminated)" if is_honeypot(candidate) else "Eligible"
    
    if hp_status == "HONEYPOT (Eliminated)":
        results.append({
            "id": cid,
            "title": candidate["profile"]["current_title"],
            "status": hp_status,
            "score": 0.0,
            "notes": "Purged during Phase 1 universal elimination."
        })
        continue
        
    career = candidate.get("career_history", [])
    if not career:
        results.append({
            "id": cid,
            "title": candidate["profile"]["current_title"],
            "status": "EXCLUDED (No History)",
            "score": 0.0,
            "notes": "Dropped because career history list is empty."
        })
        continue
        
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    total_weighted_score = 0.0
    total_weight = 0.0
    
    # Scoring
    for job in career:
        desc = job.get("description", "")
        if not desc: continue
        weight = max(job.get("duration_months", 1), 1) * (1.5 if job.get("is_current", False) else 1.0)
        
        # Calculate semantic cosine dot product
        v_job = embedder.encode(desc, normalize_embeddings=True)
        job_score = float(np.dot(v_job, v_core)) - (0.3 * float(np.dot(v_job, v_neg)))
        total_weighted_score += job_score * weight
        total_weight += weight
        
    raw_career_score = (total_weighted_score / total_weight) if total_weight > 0 else 0.0
    base_score = max(0, raw_career_score * 100)
    
    # Modifiers
    yoe_mod = _yoe_modifier(yoe)
    sig_mod = compute_signal_multiplier(candidate)
    
    final_score = base_score * yoe_mod * sig_mod
    
    results.append({
        "id": cid,
        "title": candidate["profile"]["current_title"],
        "status": "Eligible",
        "score": round(final_score, 2),
        "notes": f"Base Semantic Match: {base_score:.1f}/100 | YOE Mod: {yoe_mod:.2f} | Signal Mult: {sig_mod:.2f}"
    })

# Sort eligible candidates by score
eligible_sorted = [r for r in results if r["status"] == "Eligible"]
eligible_sorted.sort(key=lambda x: -x["score"])

# Assign ranks
rank_map = {}
for rank_idx, r in enumerate(eligible_sorted, start=1):
    rank_map[r["id"]] = rank_idx

for r in results:
    if r["id"] in rank_map:
        r["rank"] = rank_map[r["id"]]
    else:
        r["rank"] = "N/A"

# Write JSONL test candidates file to root workspace
out_file_path = os.path.join(ROOT_DIR, "test_sandbox_candidates.jsonl")
with open(out_file_path, "w", encoding="utf-8") as f:
    f.write("\n".join(jsonl_lines) + "\n")

print(f"\nSuccessfully wrote 15 test candidates to {out_file_path}")

# Print Markdown Table
print("\n### GROUND TRUTH MATCH RANKINGS:")
print("| Ground Truth Rank | Candidate ID | Profile Title | Score | Status | Math Notes |")
print("|---|---|---|---|---|---|")
for r in sorted(results, key=lambda x: (x["status"] != "Eligible", x.get("rank", 999))):
    rank_str = str(r.get("rank")) if r["status"] == "Eligible" else "N/A"
    print(f"| {rank_str} | {r['id']} | {r['title']} | {r['score']} | {r['status']} | {r['notes']} |")
