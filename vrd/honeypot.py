"""
Phase 1: Universal Elimination (Honeypot Detection)
===================================================
Identifies synthetically corrupted "honeypot" profiles with
physically impossible data. 

"""

def is_honeypot(candidate: dict) -> bool:
    """
    Return True if this candidate is a honeypot (must be permanently excluded).
    """
    skills = candidate.get("skills", [])
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    # ---------------------------------------------------------------
    # Signal 1: Expert/Advanced proficiency with 0 months of use
    #
    # You cannot be "expert" at a skill you have never used.
    # We trigger if we find 2 or more of these mathematically impossible
    # skill entries to avoid penalizing a single accidental typo.
    # ---------------------------------------------------------------
    zero_dur_expert_count = 0
    for s in skills:
        if s.get("proficiency", "").lower() in ("advanced", "expert"):
            if s.get("duration_months", -1) == 0:
                zero_dur_expert_count += 1
                
    if zero_dur_expert_count >= 2:
        return True

    # ---------------------------------------------------------------
    # Signal 2: Severe Years of Experience (YOE) Discrepancy
    #
    # If the profile claims 13 years of experience but career history
    # only adds up to 1 year — the profile is fabricated.
    # We use a safe threshold of 8 years to avoid penalising real candidates
    # who have legitimate gaps (career breaks, freelancing, unlisted jobs).
    # ---------------------------------------------------------------
    profile_yoe = profile.get("years_of_experience", 0)
    
    # Calculate YOE from career history
    career_months = sum(max(0, job.get("duration_months", 0)) for job in career)
    career_yoe = career_months / 12.0

    if abs(profile_yoe - career_yoe) > 8.0:
        return True

    return False
