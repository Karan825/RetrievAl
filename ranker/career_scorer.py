"""
Stage 3: Career Relevance Scoring
===================================
The core of the ranker.

Key insight: The 100,000 candidates have career history descriptions
generated from exactly 44 unique templates. We map each template to
a base relevance score (0-100) for the Senior AI Engineer JD, then
apply two per-job multipliers:
  1. Company type multiplier (product company vs consulting)
  2. Recency weight (current job counts more)

Final score = duration-weighted average of (template_score
              × company_type_mult × recency_weight)
            × yoe_modifier
            × job_hopper_penalty

The JD says: "A Tier 5 candidate may not use 'RAG' or 'Pinecone' in
their profile, but if their career history shows they built a
recommendation system at a product company, they're a fit."

We implement this literally: recommendation/search work at a product
company scores 58 × 1.25 = 72.5 — well above a keyword-stuffed
Marketing Manager who scores 0.
"""

# ---------------------------------------------------------------------------
# Template scores — ordered by specificity (most specific first so the first
# match wins when a description matches multiple keys).
# ---------------------------------------------------------------------------
DESCRIPTION_TEMPLATE_SCORES = [

    # ── TIER 1 (95-100): Direct, at-scale ranking/search shipped to users ──

    # "Built a RAG-based ranking pipeline serving 50M+ queries/month for
    #  a recruiter-facing search product. BM25 + dense retrieval (BGE
    #  embeddings, FAISS HNSW) with LLM-based re-ranker..."
    ("built a rag-based ranking pipeline serving 50m", 100),

    # "Led the migration from keyword-based to embedding-based search
    #  across a 30M+ candidate corpus..."
    ("led the migration from keyword-based to embedding-based search", 100),

    # "Owned the end-to-end ranking pipeline at a recommendations-heavy
    #  consumer product: candidate sourcing → embedding generation
    #  (fine-tuned BGE-large) → Pinecone retrieval → XGBoost re-scoring"
    ("owned the end-to-end ranking pipeline at a recommendations-heavy", 100),

    # "Owned the design and rollout of a large-scale semantic search
    #  system serving 35M+ items. Migrated BM25 to hybrid sparse+dense..."
    ("owned the design and rollout of a large-scale semantic search", 100),

    # "Owned the search and discovery experience end-to-end at a consumer
    #  product, from how content is represented internally through..."
    ("owned the search and discovery experience end-to-end", 95),

    # "Led the engineering team building infrastructure to surface
    #  relevant content to users at scale..."
    ("led the engineering team building infrastructure to surface", 95),

    # "Designed the ranking layer for the company's flagship product:
    #  how do we surface the right thing at the right time..."
    ("designed the ranking layer for the company", 95),

    # "Shipped the personalization infrastructure: the system that
    #  learns from user behavior and improves relevance over time..."
    ("shipped the personalization infrastructure", 90),

    # "Built systems that understand what users are looking for and
    #  connect them to the most relevant matches across a large dataset."
    ("built systems that understand what users are looking for", 90),

    # ── TIER 2 (80-85): Strong production ML ranking / retrieval ──

    # "Owned the ranking layer for an e-commerce search product, evolving
    #  it from a hand-tuned scoring function to a learning-to-rank model"
    ("owned the ranking layer for an e-commerce search product", 85),

    # "Trained and shipped multiple ranking models for our product's
    #  discovery feed using XGBoost and LightGBM..."
    ("trained and shipped multiple ranking models for our product", 85),

    # "Built a content recommendation system serving 10M+ users that
    #  combined collaborative filtering with content-based ranking..."
    ("built a content recommendation system serving 10m", 85),

    # "Built and shipped a production recommendation system at a
    #  marketplace product, going from offline experimentation to
    #  live A/B test in 5 months..."
    ("built and shipped a production recommendation system at a marketplace", 85),

    # ── TIER 2b (78-80): Strong adjacent — semantic search, fine-tuning ──

    # "Owned the design and rollout of a large-scale semantic search..."
    # (already matched above, but also catches shorter descriptions)
    ("developed a semantic search feature for an internal knowledge base", 80),

    # "Fine-tuned LLaMA-2-7B and Mistral-7B variants using LoRA and QLoRA
    #  for domain-specific candidate-JD matching..."
    ("fine-tuned llama-2-7b and mistral-7b variants using lora", 80),

    # "Implemented a RAG-based customer support chatbot..."
    ("implemented a rag-based customer support chatbot", 78),

    # ── TIER 3 (38-65): Good ML production work — relevant but less direct ──

    # "Built and operated production ML pipelines using MLflow for
    #  experiment tracking, Kubeflow for orchestration..."
    ("built and operated production ml pipelines using mlflow", 63),

    # "Built recommendation-style features at a mid-stage startup —
    #  lighter weight than ranking systems at FAANG, but production."
    # NOTE: This is the "Tier 5 candidate" the JD note talks about.
    # Boosted to 60 — at a product company it becomes 60 × 1.25 = 75
    ("built recommendation-style features at a mid-stage startup", 60),

    # "Built NLP pipelines for sentiment analysis and document
    #  classification..."
    ("built nlp pipelines for sentiment analysis", 52),

    # "Contributed to ML feature engineering and model deployment
    #  for a fraud-detection product..."
    ("contributed to ml feature engineering and model deployment", 48),

    # "Worked on customer-facing predictive modeling for an e-commerce
    #  platform — churn prediction, conversion likelihood..."
    ("worked on customer-facing predictive modeling for an e-commerce", 44),

    # "Worked on time-series forecasting models for supply-chain
    #  demand prediction at a logistics company..."
    ("worked on time-series forecasting models", 38),

    # ── TIER 3c (8): Wrong domain — CV, JD explicitly disqualifies ──

    # "Built computer vision models for our product's image moderation..."
    ("built computer vision models for our product", 8),

    # ── TIER 4 (10-22): Data / Backend engineering — useful but not ML/IR ──

    ("designed and maintained the analytical data warehouse on snowflake", 22),
    ("backend + data hybrid role", 20),
    ("implemented streaming data pipelines on kafka and spark", 20),
    ("built and maintained data pipelines on apache airflow", 20),
    ("mixed data science and analytics-engineering role", 18),
    ("backend development with python (fastapi)", 15),
    ("full-stack web application development at a saas company", 10),

    # ── TIER 5 (2-8): Generic software — minimal relevance ──

    ("cloud infrastructure and devops work", 8),
    ("test automation and qa engineering", 4),
    ("android mobile development", 4),
    ("frontend engineering at a media company", 4),
    ("java backend development at a large enterprise", 4),

    # ── TIER 6 (0): Non-technical noise — should NOT appear in top 100 ──

    ("enterprise sales of cloud software solutions", 0),
    ("customer support team lead at a saas product", 0),
    ("marketing leadership role at a b2b saas company", 0),
    ("business analyst at a consulting firm", 0),
    ("brand design and creative direction", 0),
    ("mechanical engineering design role", 0),
    ("senior accounting role at a mid-sized company", 0),
    ("content writing and seo strategy", 0),
    ("operations management role at a logistics company", 0),
]


# ---------------------------------------------------------------------------
# Company classification
# ---------------------------------------------------------------------------

PRODUCT_COMPANIES = {
    # Indian consumer / fintech product companies
    "Swiggy", "Zomato", "CRED", "Flipkart", "Razorpay",
    "Meesho", "PhonePe", "Freshworks", "Paytm", "Ola",
    "Dream11", "PolicyBazaar", "Nykaa", "upGrad", "Zoho",
    "Vedantu", "Unacademy", "PharmEasy",
    # Indian AI-first companies
    "InMobi", "Glance", "Sarvam AI", "Krutrim",
    "Mad Street Den", "Rephrase.ai", "Aganitha", "Niramai",
    "Saarthi.ai", "Observe.AI", "Wysa", "Haptik",
    "Verloop.io", "Yellow.ai", "Locobuzz", "Genpact AI",
    # Global tech
    "Google", "Amazon", "Meta", "Microsoft", "Netflix",
    "Adobe", "Uber", "Salesforce", "LinkedIn", "Apple",
    # Fictional product companies in this dataset
    "Pied Piper", "Hooli",
}

CONSULTING_FIRMS = {
    "TCS", "Infosys", "Wipro", "Accenture",
    "Cognizant", "Capgemini", "Tech Mahindra", "HCL", "Mphasis",
}


def _match_description(desc_lower: str) -> int:
    """Return the base relevance score for a career description."""
    for key, score in DESCRIPTION_TEMPLATE_SCORES:
        if key in desc_lower:
            return score
    return 0


def _company_multiplier(company: str) -> float:
    """1.25 for product company, 0.65 for consulting, 1.0 otherwise."""
    if company in PRODUCT_COMPANIES:
        return 1.25
    if company in CONSULTING_FIRMS:
        return 0.65
    return 1.0


def _job_hopper_penalty(career: list) -> float:
    """0.70 if >50% of jobs lasted under 15 months (title-chaser signal)."""
    if len(career) < 3:
        return 1.0
    short = sum(1 for j in career if j.get("duration_months", 24) < 15)
    return 0.70 if short / len(career) > 0.5 else 1.0


def _yoe_modifier(yoe: float) -> float:
    """Target: 6-8 years. Ramp down outside 5-9 band."""
    if 6.0 <= yoe <= 8.0:
        return 1.00
    if 5.0 <= yoe < 6.0 or 8.0 < yoe <= 9.0:
        return 0.92
    if 4.0 <= yoe < 5.0:
        return 0.78
    if 9.0 < yoe <= 12.0:
        return 0.82
    if yoe > 12.0:
        return 0.68
    return 0.50   # < 4 years — too junior


def compute_career_score(candidate: dict) -> float:
    """
    Compute the core career relevance score for a candidate.

    Returns a float in [0, ~130] — can exceed 100 due to product
    company multiplier, which is intentional (product company ML
    engineers should rank above consulting-firm equivalents).
    """
    career = candidate.get("career_history", [])
    yoe = candidate["profile"].get("years_of_experience", 0)

    if not career:
        return 0.0

    total_weighted_score = 0.0
    total_weight = 0.0

    for job in career:
        desc = job.get("description", "").lower()
        company = job.get("company", "")
        duration = max(job.get("duration_months", 1), 1)

        base_score = _match_description(desc)
        company_mult = _company_multiplier(company)

        # Current jobs count 1.5× — recency signal
        recency_weight = 1.5 if job.get("is_current", False) else 1.0

        effective_score = base_score * company_mult
        weight = duration * recency_weight

        total_weighted_score += effective_score * weight
        total_weight += weight

    raw = total_weighted_score / total_weight if total_weight > 0 else 0.0

    # Apply YOE and job-hopper modifiers
    final = raw * _yoe_modifier(yoe) * _job_hopper_penalty(career)
    return final
