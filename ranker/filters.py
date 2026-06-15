"""
Stage 2: Hard Constraint Filters
=================================
These are candidates the JD *explicitly* disqualifies, regardless
of their ML skills or behavioral signals.

JD says:
  - "People who have only worked at consulting firms (TCS, Infosys,
     Wipro, Accenture, Cognizant, Capgemini, etc.) — not a fit."
  - "Computer vision / speech / robotics without NLP/IR — not a fit."
  - "Pure research without production deployment — not a fit."
"""

BANNED_CONSULTING_FIRMS = {
    "TCS", "Infosys", "Wipro", "Accenture",
    "Cognizant", "Capgemini", "Tech Mahindra", "HCL", "Mphasis",
}

# Templates whose primary work is in CV/speech/robotics —
# explicitly disqualified by JD ("you'd be re-learning fundamentals here")
CV_SPEECH_DESC_KEYS = {
    "built computer vision models for our product",
    "android mobile development",
}


def passes_hard_constraints(candidate: dict) -> bool:
    """Return True if the candidate is eligible (not hard-disqualified)."""

    career = candidate.get("career_history", [])

    # ---------------------------------------------------------------
    # Rule 1: Career *entirely* in banned consulting firms → reject
    # ---------------------------------------------------------------
    companies_worked = {j.get("company", "") for j in career}
    if companies_worked and companies_worked.issubset(BANNED_CONSULTING_FIRMS):
        return False

    # ---------------------------------------------------------------
    # Rule 2: No years of experience at all → reject
    # ---------------------------------------------------------------
    yoe = candidate["profile"].get("years_of_experience", 0)
    if yoe < 1:
        return False

    return True
