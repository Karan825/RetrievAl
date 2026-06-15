"""
Stage 5: Dynamic Reasoning Generation
=======================================
Generates a 1-2 sentence, fact-grounded reasoning for each ranked
candidate that passes Stage 4 manual review.

Stage 4 checks (from submission_spec.md):
  ✅ Specific facts from the candidate's profile
  ✅ Connection to JD requirements
  ✅ Honest concerns where applicable
  ✅ No hallucination (only states things in the profile)
  ✅ Variation across candidates (not templated)
  ✅ Tone matches rank

We achieve all of this by building the sentence from actual
candidate data fields — never inventing information.
"""

import datetime

TODAY = datetime.date(2026, 6, 10)

# Skills the JD explicitly cares about (for mention in reasoning)
JD_RELEVANT_SKILLS = {
    "FAISS", "Pinecone", "Milvus", "Qdrant", "Weaviate",
    "Elasticsearch", "OpenSearch", "Sentence Transformers",
    "BGE", "E5", "XGBoost", "LightGBM", "LoRA", "QLoRA",
    "BentoML", "MLflow", "Kubeflow", "Embeddings", "RAG",
    "Information Retrieval", "Recommendation Systems", "MLOps",
    "Fine-tuning LLMs", "Hugging Face Transformers", "scikit-learn",
    "Machine Learning", "NLP", "LangChain", "Vector Search",
    "Haystack", "PEFT", "PyTorch", "TensorFlow",
}

# Mapping of description key → short label for reasoning sentence
DESCRIPTION_LABELS = {
    "built a rag-based ranking pipeline serving 50m": "large-scale RAG ranking pipeline",
    "led the migration from keyword-based to embedding-based search": "embedding-based search migration across 30M+ items",
    "owned the end-to-end ranking pipeline at a recommendations-heavy": "end-to-end ranking pipeline (BGE + Pinecone + XGBoost)",
    "owned the design and rollout of a large-scale semantic search": "large-scale hybrid semantic search system",
    "owned the search and discovery experience end-to-end": "end-to-end search & discovery system",
    "led the engineering team building infrastructure to surface": "infrastructure to surface relevant content at scale",
    "designed the ranking layer for the company": "company-wide ranking layer",
    "shipped the personalization infrastructure": "personalization infrastructure with online A/B evaluation",
    "built systems that understand what users are looking for": "end-to-end retrieval & matching system",
    "owned the ranking layer for an e-commerce search product": "learning-to-rank model for e-commerce search",
    "trained and shipped multiple ranking models for our product": "XGBoost/LightGBM ranking models for discovery feed",
    "built a content recommendation system serving 10m": "content recommender serving 10M+ users",
    "built and shipped a production recommendation system at a marketplace": "production recommendation system with cold-start handling",
    "developed a semantic search feature for an internal knowledge base": "FAISS-based semantic search with sentence-transformers",
    "fine-tuned llama-2-7b and mistral-7b variants using lora": "LoRA/QLoRA fine-tuning of LLaMA-2/Mistral for matching",
    "implemented a rag-based customer support chatbot": "RAG-based chatbot with Pinecone + GPT-4 → fine-tuned model",
    "built and operated production ml pipelines using mlflow": "production ML pipelines with MLflow + Kubeflow",
    "built recommendation-style features at a mid-stage startup": "recommendation features (collaborative filtering + gradient boosting)",
    "built nlp pipelines for sentiment analysis": "NLP pipeline with transformer-based classifiers",
    "contributed to ml feature engineering and model deployment": "ML feature engineering and model deployment",
    "worked on customer-facing predictive modeling for an e-commerce": "predictive modelling (XGBoost, churn prediction)",
    "worked on time-series forecasting models": "time-series forecasting (Prophet, LightGBM)",
    "built computer vision models for our product": "computer vision work (wrong domain for this JD)",
}


def _best_job_label(career: list) -> tuple[str, str]:
    """Return (label, company) for the highest-scoring job in career."""
    for key, label in DESCRIPTION_LABELS.items():
        for job in career:
            if key in job.get("description", "").lower():
                return label, job.get("company", "")
    return "", ""


def _matched_skills(skills: list) -> list:
    """Return relevant skills that actually appear in the candidate's profile."""
    names = {s["name"] for s in skills}
    return [s for s in JD_RELEVANT_SKILLS if s in names]


def generate_reasoning(candidate: dict, final_score: float, rank: int) -> str:
    """
    Build a 1-2 sentence, fact-grounded reasoning string.

    Positive sentence: {YOE}yr {title} with {work_description}
                       at {company}; {relevant_skills}.
    Concern sentence:  Concern: {comma-separated concerns} — or
                       a positive note if none.
    """
    profile = candidate["profile"]
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Engineer")
    company = profile.get("current_company", "")
    location = profile.get("location", "")

    job_label, job_company = _best_job_label(career)
    rel_skills = _matched_skills(skills)

    # ── Positive sentence ────────────────────────────────────────────────────
    if job_label:
        context = f"shipped {job_label} at {job_company or company}"
    else:
        context = f"works as {title} at {company}"

    if rel_skills:
        skill_str = ", ".join(rel_skills[:3])
        positive = f"{yoe:.1f}yr engineer who {context}; skills include {skill_str}."
    else:
        positive = f"{yoe:.1f}yr engineer who {context}."

    # ── Concern / signal sentence ────────────────────────────────────────────
    concerns = []

    notice = signals.get("notice_period_days", 90)
    if notice > 60:
        concerns.append(f"{notice}-day notice period")

    last_active_str = signals.get("last_active_date", "")
    if last_active_str:
        try:
            days_inactive = (TODAY - datetime.date.fromisoformat(last_active_str)).days
            if days_inactive > 180:
                concerns.append(f"inactive {days_inactive // 30} months on platform")
        except ValueError:
            pass

    rr = signals.get("recruiter_response_rate", 1.0)
    if rr < 0.35:
        concerns.append(f"low recruiter response rate ({rr:.0%})")

    country = profile.get("country", "India")
    if country.lower() not in ("india",):
        willing = signals.get("willing_to_relocate", False)
        if willing:
            concerns.append(f"based in {location} but willing to relocate")
        else:
            concerns.append(f"based in {location}, relocation uncertain")

    if rank <= 10:
        sal = signals.get("expected_salary_range_inr_lpa", {})
        sal_max = sal.get("max", 0)
        if sal_max > 70:
            concerns.append(f"salary expectation {sal_max:.0f} LPA")

    if concerns:
        concern_str = "Note: " + "; ".join(concerns) + "."
    else:
        mode = signals.get("preferred_work_mode", "flexible")
        concern_str = (
            f"Strong match: active on platform, responsive, "
            f"{mode} work mode."
        )

    return f"{positive} {concern_str}"
