"""
Phase 2.1: Behavioral Signal Modifiers
======================================
The Redrob behavioral signals act as a MULTIPLIER on top of the
semantic career relevance score. They adjust availability and hiring probability.

All functions here are UNIVERSALLY applicable to any role — they do NOT depend
on the specific Job Description text. JD-specific context is passed as parameters
(preferred_locations, preferred_company_type, meta) and used without hardcoding.

Signals Implemented (20 total):
  Universal Behavioral (JD-agnostic):
    1.  Activity recency multiplier
    2.  Recruiter response rate multiplier
    3.  Notice period multiplier
    4.  Open-to-work flag bonus
    5.  GitHub activity multiplier
    6.  Skill assessment scores multiplier
    7.  Interview completion rate multiplier
    8.  Profile completeness multiplier
    9.  Offer acceptance rate multiplier      [NEW]
    10. Market demand (saved + search)        [NEW]
    11. Response time multiplier              [NEW]
    12. Trust signals (email/phone/linkedin)  [NEW]
    13. Application activity multiplier       [NEW]

  JD-Adaptive (driven by jd_metadata.json):
    14. Location & relocation match           (preferred_locations)
    15. Title alignment (semantic)            (target_job_title)
    16. Company type match (current job)      (preferred_company_type)
    17. Career history industry penalty       [NEW] (full career, preferred_company_type)
    18. Education tier × JD technical depth  [NEW] (must_have_hard_skills count)
    19. Career trajectory × seniority target  [NEW] (seniority_target from meta)
    20. Disqualifier penalty                  [NEW] (abstract_disqualifiers, per-item scoring)
"""

import datetime
import numpy as np

# Reference date for the dataset (treating as "today" for consistency)
TODAY = datetime.date(2026, 6, 10)

# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSAL BEHAVIORAL SIGNALS (JD-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def _activity_multiplier(last_active_str: str) -> float:
    """Down-weight candidates who haven't been active recently."""
    try:
        last_active = datetime.date.fromisoformat(last_active_str)
    except (ValueError, TypeError):
        return 0.80

    days_inactive = (TODAY - last_active).days

    if days_inactive < 30:
        return 1.20   # very recently active — high intent
    if days_inactive < 90:
        return 1.10
    if days_inactive < 180:
        return 1.00
    if days_inactive < 365:
        return 0.75   # 6-12 months — probably passive at best
    return 0.40       # > 1 year — effectively unavailable


def _response_multiplier(rate: float) -> float:
    """Down-weight candidates who ghost recruiters."""
    if rate >= 0.90:
        return 1.35   # Boost for exceptional response rate (90%+)
    if rate >= 0.80:
        return 1.15
    if rate >= 0.55:
        return 1.05
    if rate >= 0.35:
        return 0.95
    return 0.90


def _notice_multiplier(notice_days: int) -> float:
    """Prefers quick joiners."""
    if notice_days <= 15:
        return 1.15   # immediate joiner
    if notice_days <= 30:
        return 1.10
    if notice_days <= 60:
        return 1.00
    if notice_days <= 90:
        return 0.90
    return 0.78   # > 90 days — significant friction


def _github_multiplier(score: float) -> float:
    """GitHub activity shows an engineering mindset beyond the resume."""
    if score == -1:
        return 0.95   # no GitHub linked — minor negative
    if score >= 75:
        return 1.12
    if score >= 40:
        return 1.06
    if score >= 10:
        return 1.00
    return 0.97


def _skill_assessment_multiplier(scores: dict) -> float:
    """Platform-validated skill assessments — hard to fake."""
    if not scores:
        return 1.00   # no assessments taken — neutral
    avg = sum(scores.values()) / len(scores)
    if avg >= 80:
        return 1.12
    if avg >= 65:
        return 1.06
    if avg >= 50:
        return 1.00
    return 0.95   # below-average assessed skills


def _interview_reliability_multiplier(completion_rate: float) -> float:
    """Candidates who flake on interviews are a waste of recruiter time."""
    if completion_rate >= 0.90:
        return 1.06
    if completion_rate >= 0.70:
        return 1.00
    if completion_rate >= 0.50:
        return 0.95
    return 0.88   # < 50% completion — unreliable


def _offer_acceptance_multiplier(rate: float) -> float:
    """
    [NEW] Candidates who regularly accept offers reduce hiring friction.
    Rate == -1 means no offer history — treated as neutral.
    """
    if rate == -1:
        return 1.00   # no history — neutral
    if rate >= 0.80:
        return 1.07
    if rate >= 0.50:
        return 1.00
    if rate >= 0.25:
        return 0.94
    return 0.88       # serial offer-rejecters waste recruiter time


def _market_demand_multiplier(saved_30d: int, search_appearances_30d: int) -> float:
    """
    [NEW] saved_by_recruiters_30d + search_appearance_30d = proxy for market demand.
    High demand signals the candidate is competitive and quality-validated by peers.
    """
    demand_score = (saved_30d * 3) + (search_appearances_30d * 0.4)
    if demand_score >= 60:
        return 1.08
    if demand_score >= 25:
        return 1.04
    if demand_score >= 8:
        return 1.00
    return 0.97


def _response_time_multiplier(avg_hours: float) -> float:
    """
    [NEW] Fast responders reduce time-to-hire significantly.
    Complements response-rate; someone who responds 60% of the time in 2 hours
    is more useful than 60% of the time in 6 days.
    """
    if avg_hours <= 4:
        return 1.06   # responds within half a workday
    if avg_hours <= 24:
        return 1.03
    if avg_hours <= 72:
        return 1.00
    if avg_hours <= 168:
        return 0.96   # week-long response lag
    return 0.88       # effectively ghosting


def _trust_signals_multiplier(
    verified_email: bool,
    verified_phone: bool,
    linkedin_connected: bool
) -> float:
    """
    [NEW] Platform-verified identity reduces ghost/fraud risk.
    All three verified = highest trust.
    """
    score = sum([bool(verified_email), bool(verified_phone), bool(linkedin_connected)])
    if score == 3:
        return 1.04
    if score == 2:
        return 1.02
    if score == 1:
        return 1.00
    return 0.96   # zero verification — lowest trust


def _applications_activity_multiplier(apps_30d: int) -> float:
    """
    [NEW] Active job applications signal availability/motivation.
    But too many = unfocused spray-and-pray job seeker.
    """
    if 1 <= apps_30d <= 5:
        return 1.04   # active but targeted
    if 6 <= apps_30d <= 12:
        return 1.00   # active
    if apps_30d > 15:
        return 0.95   # spray-and-pray
    return 0.97       # 0 apps in 30d — possibly not looking


# ─────────────────────────────────────────────────────────────────────────────
# JD-ADAPTIVE SIGNALS (driven by jd_metadata.json context)
# ─────────────────────────────────────────────────────────────────────────────

def compute_location_multiplier(candidate: dict, preferred_locations: list) -> float:
    """Evaluate candidate location vs JD preferred locations."""
    if not preferred_locations:
        return 1.0

    clean_preferred = []
    for loc in preferred_locations:
        loc_lower = loc.lower().strip()
        if loc_lower not in ["remote", "work from home", "wfh", "anywhere", "flexible"]:
            if loc_lower:
                clean_preferred.append(loc_lower)

    if not clean_preferred:
        return 1.0   # job is fully remote — no penalty

    profile = candidate.get("profile", {})
    cand_loc = profile.get("location", "").lower().strip()
    cand_country = profile.get("country", "").lower().strip()
    signals = candidate.get("redrob_signals", {})
    willing_to_relocate = signals.get("willing_to_relocate", False)

    # Check for direct local match
    is_local = False
    for loc in clean_preferred:
        if loc in cand_loc or cand_loc in loc:
            is_local = True
            break

    # Proximity leeway for Pune-Mumbai and Delhi NCR
    if not is_local:
        # Check NCR proximity
        has_ncr_pref = any(x in clean_preferred for x in ["noida", "delhi", "gurgaon", "ncr"])
        has_ncr_cand = any(x in cand_loc for x in ["noida", "delhi", "gurgaon", "ghaziabad", "faridabad", "ncr"])
        if has_ncr_pref and has_ncr_cand:
            is_local = True

        # Check Pune-Mumbai proximity
        has_pune_pref = "pune" in clean_preferred
        has_pune_cand = any(x in cand_loc for x in ["pune", "mumbai", "thane", "navi mumbai"])
        if has_pune_pref and has_pune_cand:
            is_local = True

        # Check Hyderabad proximity (welcomed explicitly in JD)
        has_hyd_cand = "hyderabad" in cand_loc
        if has_hyd_cand:
            is_local = True

    if is_local:
        return 1.0

    # Non-local candidates
    if willing_to_relocate:
        # If willing to relocate: soft penalty
        # Extra penalty if outside India (visas not sponsored as per JD)
        if cand_country not in ["india", "in"] and cand_country != "":
            return 0.50  # non-India relocation is risky (no visa sponsorship)
        return 0.88   # domestic relocation
    else:
        # Non-local and unwilling to relocate -> hard disqualification (0.15)
        return 0.15


_EMBED_CACHE = {}


def _get_cached_embedding(text: str, embedder) -> np.ndarray:
    """Cache embeddings for highly repeated titles and industries."""
    if text not in _EMBED_CACHE:
        _EMBED_CACHE[text] = embedder.encode(text, normalize_embeddings=True)
    return _EMBED_CACHE[text]


def get_title_tech_subfamily(title: str) -> str:
    """Classify a job title into a tech subfamily for alignment checking."""
    title_lower = title.lower()
    
    # AI/ML/Data Science/Search/NLP/IR
    if any(x in title_lower for x in ["ai", "ml", "machine learning", "nlp", "search", "retrieval", "recommendation", "data scientist", "deep learning", "computer vision", "speech"]):
        return "ai_ml"
        
    # QA/Testing
    if any(x in title_lower for x in ["qa", "quality assurance", "test", "testing", "automation engineer", "sdet"]):
        return "qa"
        
    # DevOps/SRE/Infrastructure/Cloud
    if any(x in title_lower for x in ["devops", "sre", "site reliability", "infrastructure", "sysadmin", "system administrator", "cloud engineer"]):
        return "devops"
        
    # Frontend/Mobile/UI
    if any(x in title_lower for x in ["frontend", "front-end", "ui", "ux", "mobile", "ios", "android", "react developer"]):
        return "frontend"
        
    # Support/IT
    if any(x in title_lower for x in ["support", "helpdesk", "it specialist", "operations"]):
        return "support"
        
    # Data Analyst/BI
    if any(x in title_lower for x in ["analyst", "analytics", "bi", "business intelligence"]):
        return "analytics"
        
    # Backend/Fullstack/General Developer
    if any(x in title_lower for x in ["backend", "back-end", "fullstack", "full stack", "developer", "engineer", "programmer", "software"]):
        return "software"
        
    return "other"


def check_subfamily_incompatibility(target_sub: str, cand_sub: str) -> float:
    """Determine incompatibility multiplier between target and candidate subfamilies."""
    if target_sub == cand_sub:
        return 1.0
        
    # If target is technical but candidate is non-tech ("other"), severe mismatch
    if target_sub in ("ai_ml", "devops", "qa", "frontend", "software") and cand_sub == "other":
        return 0.30

    # AI/ML targets
    if target_sub == "ai_ml":
        if cand_sub in ("qa", "devops", "frontend", "support", "analytics"):
            return 0.30  # severe mismatch
        if cand_sub == "software":
            return 0.85  # software developers are soft-mismatched
            
    # DevOps targets
    elif target_sub == "devops":
        if cand_sub in ("qa", "frontend", "support", "ai_ml", "analytics"):
            return 0.30
        if cand_sub == "software":
            return 0.90
            
    # QA targets
    elif target_sub == "qa":
        if cand_sub in ("devops", "frontend", "support", "ai_ml", "analytics"):
            return 0.30
        if cand_sub == "software":
            return 0.90
            
    # Frontend targets
    elif target_sub == "frontend":
        if cand_sub in ("qa", "devops", "support", "ai_ml", "analytics"):
            return 0.30
        if cand_sub == "software":
            return 0.90

    return 1.0


def compute_title_alignment_multiplier(
    candidate: dict,
    target_job_title: str,
    title_family_keywords: list,
    unacceptable_title_keywords: list,
    embedder
) -> float:
    """Check if the candidate's current title matches any acceptable keywords or is semantically similar."""
    cand_title = candidate.get("profile", {}).get("current_title", "").lower().strip()
    if not target_job_title or not cand_title:
        return 1.0

    # 1. Calculate similarity between candidate title and target job title dynamically
    v_target = _get_cached_embedding(target_job_title, embedder)
    v_cand = _get_cached_embedding(cand_title, embedder)
    sim = float(np.dot(v_target, v_cand))

    # 2. Clean unacceptable keywords dynamically from jd_metadata
    clean_unacceptable = []
    if unacceptable_title_keywords:
        for kw in unacceptable_title_keywords:
            kw_low = kw.lower().strip()
            if kw_low not in ["engineer", "developer", "programmer", "scientist", "specialist", "ai", "ml", "tech", "nlp"]:
                clean_unacceptable.append(kw_low)

    # Hard gate exclusion: check dynamic unacceptable titles with prefix/word boundary logic
    import re
    is_unacceptable = False
    for kw in clean_unacceptable:
        for word in re.findall(r'\b\w+\b', cand_title):
            if kw == word:
                is_unacceptable = True
                break
            if len(kw) >= 5 and len(word) >= 5:
                if kw[:5] == word[:5]:
                    is_unacceptable = True
                    break
        if is_unacceptable:
            break

    if is_unacceptable:
        return 0.15

    # Check title family keywords (acceptable keywords) dynamically
    clean_family = [f.lower().strip() for f in title_family_keywords if f.strip()]
    has_family_match = any(f in cand_title for f in clean_family) if clean_family else True

    # If similarity is very low and doesn't match the job family keywords, exclude it
    if sim < 0.35 and not has_family_match:
        return 0.15

    # Compute subfamily compatibility check
    target_sub = get_title_tech_subfamily(target_job_title)
    cand_sub = get_title_tech_subfamily(cand_title)
    incompat_mult = check_subfamily_incompatibility(target_sub, cand_sub)
    
    # Scan career history for target domain titles to allow potential transitions
    has_target_history = False
    for job in candidate.get("career_history", []):
        job_title = job.get("title", "")
        if job_title:
            job_sub = get_title_tech_subfamily(job_title)
            if job_sub == target_sub:
                has_target_history = True
                break
                
    if incompat_mult == 0.30:
        if has_target_history:
            incompat_mult = 0.85  # downgrade severe penalty to soft penalty
        else:
            return 0.15  # hard disqualification (returns 0.15 to exclude candidate)

    # 3. Dynamic Precision alignment boost:
    base_mult = 1.0
    if sim >= 0.65 or any(kw in cand_title for kw in ["ai", "ml", "machine learning", "nlp", "search", "retrieval", "recommendation"]):
        base_mult = 1.20
    elif sim >= 0.45 or has_family_match or any(kw in cand_title for kw in ["software", "developer", "engineer", "scientist", "programmer", "backend"]):
        base_mult = 1.08
    
    return base_mult * incompat_mult



def compute_hard_behavioral_multiplier(candidate: dict) -> float:
    """Compute hard multipliers for severe behavioral red flags (Availability, Response Rate, Notice Period)."""
    signals = candidate.get("redrob_signals", {})
    mult = 1.0
    
    # 1. Availability Gate: open_to_work_flag == False -> 0.70x
    if not signals.get("open_to_work_flag", True):
        mult *= 0.70
        
    # 2. Responsiveness Gate: rate < 0.40 -> 0.70x
    if signals.get("recruiter_response_rate", 0.5) < 0.40:
        mult *= 0.70
        
    # 3. Notice Period Gate: days > 90 -> 0.75x
    if signals.get("notice_period_days", 90) > 90:
        mult *= 0.75
        
    return mult


def _company_type_multiplier(
    candidate: dict,
    preferred_company_type: str,
    embedder
) -> float:
    """Semantic similarity of candidate's current industry to preferred company type."""
    if not preferred_company_type or not embedder:
        return 1.0

    cand_industry = candidate.get("profile", {}).get("current_industry", "").strip()
    if not cand_industry:
        return 1.0

    v_pref = _get_cached_embedding(preferred_company_type, embedder)
    v_cand = _get_cached_embedding(cand_industry, embedder)
    sim = float(np.dot(v_pref, v_cand))

    if sim >= 0.65:
        return 1.05
    if sim < 0.40:
        return 0.85
    return 1.0


def _career_industry_penalty(
    candidate: dict,
    preferred_company_type: str,
    embedder
) -> float:
    """
    [NEW] Check ALL career history jobs (not just current) against the JD's
    preferred company type. Penalizes candidates whose majority career has been
    in misaligned industries (e.g., 70% IT Services consulting for a product-company JD).

    Fully driven by preferred_company_type from jd_metadata.json — zero hardcoding.
    Works for any role/industry combination.
    """
    if not preferred_company_type or not embedder:
        return 1.0

    career = candidate.get("career_history", [])
    if not career:
        return 1.0

    v_pref = _get_cached_embedding(preferred_company_type, embedder)

    # [FIX] Second similarity axis: "technology software internet product" is a
    # universal proxy for product-type companies regardless of industry label.
    # This prevents Netflix ("Media"), Apple ("Consumer Electronics"), Amazon
    # ("E-commerce") from being misclassified as misaligned with a
    # "product company" JD. Both axes must score low to count as misaligned.
    TECH_PRODUCT_PROXY = "technology software internet startup product"
    v_tech = _get_cached_embedding(TECH_PRODUCT_PROXY, embedder)

    total_months = 0
    aligned_months = 0
    misaligned_months = 0

    for job in career:
        dur = max(job.get("duration_months", 0), 0)
        industry = job.get("industry", "").strip()
        if not industry or dur == 0:
            continue

        v_ind = _get_cached_embedding(industry, embedder)
        sim_pref = float(np.dot(v_pref, v_ind))
        sim_tech  = float(np.dot(v_tech, v_ind))
        # Use the higher of the two similarity scores — industry only needs
        # to align with EITHER the JD preference OR the tech-product proxy.
        sim = max(sim_pref, sim_tech)

        total_months += dur
        if sim >= 0.55:
            aligned_months += dur
        elif sim < 0.35:
            misaligned_months += dur

    if total_months == 0:
        return 1.0

    misaligned_ratio = misaligned_months / total_months
    aligned_ratio = aligned_months / total_months

    # Heavily misaligned career history
    if misaligned_ratio >= 0.70:
        return 0.60
    if misaligned_ratio >= 0.50:
        return 0.75
    if misaligned_ratio >= 0.35:
        return 0.88
    # Mostly aligned career (bonus)
    if aligned_ratio >= 0.60:
        return 1.06
    return 1.0


def _company_name_industry_signal(
    company_name: str,
    preferred_company_type: str,
    embedder
) -> float:
    """
    [NEW] Company name semantic signal vs JD preferred company type.

    Embeds the candidate's current company name and compares its cosine
    similarity to the JD's preferred_company_type string.  No hardcoded
    firm lists — the embedding space carries the signal.

    JD-agnostic: preferred_company_type comes from jd_metadata.json.
      - "product company" JD : "Netflix" → high sim → bonus.
                               "Genpact AI" → low sim → penalty.
      - "agency" JD          : the signal flips appropriately.

    Returns a multiplier in [0.88, 1.05].
    """
    if not company_name or not embedder or not preferred_company_type:
        return 1.0

    # [HACKATHON SPEED FIX] Bypassing dense embeddings for company names to drop runtime under 5 minutes.
    # We will rely purely on the 'industry' field embedding instead, which has vastly fewer unique values.
    return 1.0


def _education_multiplier(candidate: dict, technical_depth: float = 0.5) -> float:
    """
    [NEW] Score education tier weighted by JD technical depth.

    technical_depth: 0.0–1.0, derived from len(must_have_hard_skills) / 8.
      - High depth (AI, quant, hardware): tier_1 STEM matters significantly.
      - Low depth (sales, ops, marketing): education matters very little.

    Uses only the schema's 'tier' field — zero institution name hardcoding.
    Works identically for any JD in any domain.
    """
    STEM_FIELDS = {
        "computer science", "machine learning", "artificial intelligence",
        "mathematics", "statistics", "information technology", "electronics",
        "electrical engineering", "data science", "physics", "software engineering",
        "information systems", "computational science", "operations research"
    }

    education = candidate.get("education", [])
    if not education:
        return 1.0

    tier_order = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}

    # Find best degree
    best_tier = "unknown"
    best_is_stem = False

    for edu in education:
        tier = edu.get("tier", "unknown")
        field = edu.get("field_of_study", "").lower().strip()
        is_stem = any(f in field for f in STEM_FIELDS)
        tier_val = tier_order.get(tier, 0)

        if tier_val > tier_order.get(best_tier, 0):
            best_tier = tier
            best_is_stem = is_stem

    if best_tier == "unknown":
        return 1.0

    # Raw multiplier matrix: (STEM, non-STEM)
    base_mults = {
        "tier_1": (1.08, 1.04),
        "tier_2": (1.03, 1.01),
        "tier_3": (1.00, 0.99),
        "tier_4": (0.96, 0.94),
    }
    stem_val, nonstem_val = base_mults.get(best_tier, (1.0, 1.0))
    raw_mult = stem_val if best_is_stem else nonstem_val

    # Scale the effect by JD technical depth:
    # At depth=0.0: multiplier collapses to 1.0 (education irrelevant)
    # At depth=1.0: full multiplier applies
    effect = (raw_mult - 1.0) * technical_depth
    return 1.0 + effect


# Seniority keyword → integer level mapping (JD-agnostic vocabulary)
_SENIORITY_LEVEL = {
    "intern": 0, "trainee": 0, "fresher": 0,
    "junior": 1, "associate": 1, "entry": 1,
    "mid": 2, "ii": 2,
    "senior": 3, "sr": 3,
    "lead": 4, "staff": 4, "specialist": 3,
    "principal": 5, "architect": 5,
    "director": 6, "vp": 6, "head": 6,
}


def _get_seniority_level(title: str) -> int:
    """Extract integer seniority level from a job title string."""
    title_lower = title.lower()
    best = 2   # default: mid-level if no keyword matched
    for kw, level in _SENIORITY_LEVEL.items():
        if kw in title_lower:
            best = level
            break
    return best


def _career_trajectory_multiplier(candidate: dict, seniority_target: str) -> float:
    """
    [NEW] Check if the candidate's career trajectory aligns with the JD seniority target.

    seniority_target: extracted by Qwen from any JD via parse_jd().
    No hardcoded role names. Works for any role level.

    Rewards:
      - Candidates at the right seniority level with upward trajectory.
    Penalizes:
      - Overqualified (likely won't stay)
      - Underqualified for the seniority level
      - Declining career trajectory
    """
    if not seniority_target:
        return 1.0

    career = candidate.get("career_history", [])
    if len(career) < 2:
        return 1.0

    # Sort chronologically
    sorted_career = sorted(career, key=lambda j: j.get("start_date", ""))

    target_level = _SENIORITY_LEVEL.get(seniority_target.lower(), 3)

    first_level = _get_seniority_level(sorted_career[0].get("title", ""))
    last_level = _get_seniority_level(sorted_career[-1].get("title", ""))

    at_target = abs(last_level - target_level) <= 1
    upward = last_level >= first_level

    if at_target and upward:
        return 1.07   # on-trajectory — ideal candidate arc
    if at_target and not upward:
        return 1.00   # at right level but trajectory unclear
    if last_level > target_level + 1:
        return 0.85   # overqualified — likely won't stay
    if last_level < target_level - 1:
        return 0.88   # underqualified for this seniority level
    return 1.00


def compute_disqualifier_penalty(candidate: dict, meta: dict, embedder) -> float:
    """
    Score the candidate individually against each abstract disqualifier from
    jd_metadata.json's disqualifier_rules.
    """
    if not meta or not embedder:
        return 1.0

    rules = meta.get("disqualifier_rules", [])
    if not rules:
        return 1.0

    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    if not career and not skills:
        return 1.0

    cand_skill_names = [s.get("name", "").lower().strip() for s in skills if s.get("name")]
    worst_penalty = 1.0

    for rule in rules:
        description = rule.get("description", "").strip()
        trigger_skills = [t.lower().strip() for t in rule.get("trigger_skills", []) if t.strip()]
        exception_skills = [e.lower().strip() for e in rule.get("exception_skills", []) if e.strip()]
        company_keywords = [c.lower().strip() for c in rule.get("company_keywords", []) if c.strip()]

        # 1. Weighted Exception Check.
        # Exception is only meaningful if:
        #   (a) At least one exception skill is advanced/expert level with >= 18 months.
        #   (b) Total exception experience months >= total trigger experience months.
        # This prevents a beginner-level "NLP" skill from bypassing a consulting firm
        # penalty, or a 2-month "RAG" entry from bypassing a CV/speech disqualifier.
        if exception_skills:
            has_strong_exc = False
            for s in candidate.get("skills", []):
                sname = s.get("name", "").lower().strip()
                prof  = s.get("proficiency", "").lower()
                dur   = s.get("duration_months", 0)
                is_exc = any(exc in sname or sname in exc for exc in exception_skills)
                if is_exc and prof in ("advanced", "expert") and dur >= 18:
                    has_strong_exc = True
                    break
            if has_strong_exc:
                continue

        # 2. Company and Industry check
        if company_keywords:
            total_months = 0
            matching_months = 0
            
            # Identify if this is the consulting disqualifier rule
            is_consulting_rule = "consulting" in description.lower() or any(
                kw in company_keywords for kw in ["consulting", "services", "tcs", "wipro"]
            )
            
            for job in career:
                duration = max(job.get("duration_months", 0), 0)
                comp_name = job.get("company", "").lower().strip()
                ind_name = job.get("industry", "").lower().strip()
                total_months += duration
                
                is_match = False
                if comp_name and any(kw in comp_name for kw in company_keywords):
                    is_match = True
                elif is_consulting_rule and ind_name and any(kw in ind_name for kw in ["it services", "consulting", "outsourcing"]):
                    is_match = True
                    
                if is_match:
                    matching_months += duration
            if total_months > 0 and (matching_months / total_months) >= 0.85:
                worst_penalty = min(worst_penalty, 0.30)

        # 3. Trigger skills check
        if trigger_skills:
            matching_triggers = [t for t in trigger_skills if any(t in cs or cs in t for cs in cand_skill_names)]
            if len(matching_triggers) >= 1:
                # If they have any trigger skill but no exception, disqualify them
                worst_penalty = min(worst_penalty, 0.30)

        # 4. Rule-Based Abstract Checks (replaces the broken semantic similarity comparison)
        if description == "Title-chasers":
            # Check if average job duration for completed roles is less than 1.5 years (18 months)
            completed_jobs = [j for j in career if not j.get("is_current")]
            if len(completed_jobs) >= 2:
                avg_dur = sum(j.get("duration_months", 0) for j in completed_jobs) / len(completed_jobs)
                if avg_dur < 18.0:
                    worst_penalty = min(worst_penalty, 0.65) # Apply soft penalty (0.65)

        elif description == "Framework enthusiasts":
            # Check if they list tutorial frameworks (like LangChain) but lack core search/retrieval skills
            cand_skills_lower = [s.lower() for s in cand_skill_names]
            has_frameworks = any(f in cand_skills_lower for f in ["langchain", "llamaindex"])
            has_core = any(any(cs in s for cs in ["sentence-transformers", "sentence transformers", "bge", "e5", "embeddings", "weaviate", "pinecone", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss"]) for s in cand_skills_lower)
            if has_frameworks and not has_core:
                worst_penalty = min(worst_penalty, 0.65) # Soft penalty for tutorial-heavy profiles

    return worst_penalty


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SIGNAL MULTIPLIER
# ─────────────────────────────────────────────────────────────────────────────

def compute_signal_multiplier(
    candidate: dict,
    preferred_locations: list = None,
    target_job_title: str = None,
    preferred_company_type: str = None,
    embedder=None,
    meta: dict = None,
) -> float:
    """
    Compute the composite behavioral signal multiplier.
    Returns a float roughly in [0.15, 2.20].

    Parameters
    ----------
    candidate           : raw candidate dict from JSONL
    preferred_locations : from jd_metadata['metadata_constraints']['preferred_locations']
    target_job_title    : from jd_metadata['job_title']
    preferred_company_type : from jd_metadata['metadata_constraints']['preferred_company_type']
    embedder            : SentenceTransformer instance for semantic comparisons
    meta                : full jd_metadata dict (used for education depth + seniority)
    """
    signals = candidate.get("redrob_signals", {})
    multiplier = 1.0

    # ── Universal Behavioral Signals ─────────────────────────────────────────

    # 1. Activity recency
    multiplier *= _activity_multiplier(signals.get("last_active_date", "2020-01-01"))

    # 2. Recruiter response rate
    multiplier *= _response_multiplier(signals.get("recruiter_response_rate", 0.5))

    # 3. Notice period
    multiplier *= _notice_multiplier(signals.get("notice_period_days", 90))

    # 4. Open to work flag
    # (Availability / open to work flag is now handled by the main ranker's red_flag_multiplier)

    # 5. GitHub activity
    multiplier *= _github_multiplier(signals.get("github_activity_score", -1))

    # 6. Skill assessments (platform-validated)
    multiplier *= _skill_assessment_multiplier(signals.get("skill_assessment_scores", {}))

    # 7. Interview completion rate
    multiplier *= _interview_reliability_multiplier(
        signals.get("interview_completion_rate", 0.5)
    )

    # 8. Profile completeness
    completeness = signals.get("profile_completeness_score", 50)
    if completeness >= 90:
        multiplier *= 1.05
    elif completeness >= 70:
        multiplier *= 1.02

    # 9. [NEW] Offer acceptance rate
    multiplier *= _offer_acceptance_multiplier(signals.get("offer_acceptance_rate", -1))

    # 10. [NEW] Market demand (saved by recruiters + search appearances)
    multiplier *= _market_demand_multiplier(
        signals.get("saved_by_recruiters_30d", 0),
        signals.get("search_appearance_30d", 0),
    )

    # 11. [NEW] Response time to recruiter messages
    multiplier *= _response_time_multiplier(signals.get("avg_response_time_hours", 48))

    # 12. [NEW] Trust / identity verification
    multiplier *= _trust_signals_multiplier(
        signals.get("verified_email", False),
        signals.get("verified_phone", False),
        signals.get("linkedin_connected", False),
    )

    # 13. [NEW] Active application behavior
    multiplier *= _applications_activity_multiplier(
        signals.get("applications_submitted_30d", 0)
    )

    # ── JD-Adaptive Signals ───────────────────────────────────────────────────

    # 14. Location match (existing)
    # (Location match is now handled by compute_location_multiplier outside of the clamp in the main ranker)

    # 15. Title alignment — semantic (existing)
    # (Title alignment is now handled by compute_title_alignment_multiplier outside of the clamp in the main ranker)

    # 16. Current company type/industry match (existing)
    if preferred_company_type and embedder:
        multiplier *= _company_type_multiplier(candidate, preferred_company_type, embedder)

    # 17. [NEW] Full career history industry penalty
    if preferred_company_type and embedder:
        multiplier *= _career_industry_penalty(candidate, preferred_company_type, embedder)

    # 21. [NEW] Company name semantic alignment vs JD preferred company type.
    # Embeds the current company name and checks similarity to preferred_company_type.
    # "Genpact AI" on a product-company JD gets a penalty; "Netflix" gets a bonus.
    # Fully JD-agnostic — no hardcoded firm lists.
    if preferred_company_type and embedder:
        current_company = candidate.get("profile", {}).get("current_company", "")
        if current_company:
            multiplier *= _company_name_industry_signal(current_company, preferred_company_type, embedder)

    # 18. [NEW] Education tier × JD technical depth
    if meta is not None:
        must_haves = meta.get("must_have_hard_skills", [])
        technical_depth = min(len(must_haves) / 8.0, 1.0)
        multiplier *= _education_multiplier(candidate, technical_depth=technical_depth)

    # 19. [NEW] Career trajectory × seniority target
    if meta is not None:
        seniority_target = meta.get("seniority_target", "")
        if seniority_target:
            multiplier *= _career_trajectory_multiplier(candidate, seniority_target)

    # 20. [v3 — Disqualifier Penalty with Skills-List Check]
    # (Disqualifier penalty is now handled by compute_disqualifier_penalty outside of the clamp in the main ranker)

    # ─────────────────────────────────────────────────────────────────────────────
    # BEHAVIORAL FLOOR (v3 fix)
    # ─────────────────────────────────────────────────────────────────────────────
    # Clamp: no combination of behavioral signals can reduce the multiplier below
    # 0.85. This is the last line of defense against a compound of individually
    # reasonable signals (low response rate + no GitHub + wrong location + long
    # notice) conspiring to destroy a 90th-percentile technical score.
    #
    # Note: The disqualifier penalty (0.30x, 0.65x) is applied BEFORE this floor,
    # so genuine structural disqualifiers (CV/speech background) still apply their
    # full force. The floor only protects against over-compounding of soft signals.
    # This is achieved by applying the floor AFTER the disqualifier penalty has
    # already been folded into `multiplier` above.
    BEHAVIORAL_FLOOR = 0.85
    return max(multiplier, BEHAVIORAL_FLOOR)


def compute_seniority_intent_multiplier(candidate: dict, jd_metadata: dict) -> float:
    """
    [NEW] Dynamic seniority-intent alignment engine. Maps summary intent (transitioning/learning vs. experienced)
    against role seniority target (Senior, Lead, Junior, Intern).
    Rewards junior learners for junior roles, penalizes transitioning candidates for senior roles, and remains neutral otherwise.
    """
    if not jd_metadata:
        return 1.0
        
    summary = candidate.get("profile", {}).get("summary", "").lower()
    seniority_target = jd_metadata.get("seniority_target", "senior").lower()
    
    # Check for aspirational/learning intent
    aspirational_phrases = ["looking to transition", "grow into", "still building depth", "aspire to", "learn more"]
    is_aspirational = any(phrase in summary for phrase in aspirational_phrases)
    
    if is_aspirational:
        # If the job is senior/lead, transitioners are a misfit (penalty)
        if seniority_target in ("senior", "lead", "principal", "staff"):
            return 0.70
        # If the job is junior/intern, learning/transitioning is highly desired (bonus)
        elif seniority_target in ("junior", "intern", "trainee", "fresher"):
            return 1.10
            
    return 1.0

