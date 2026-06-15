"""
Stage 1: Honeypot Detection
===========================
Identifies synthetically corrupted "honeypot" profiles with
physically impossible data — expert skills with 0 months of use,
and massive YOE discrepancies between profile and career history.

Any submission that ranks these in the top 100 triggers Stage 3
disqualification (honeypot rate > 10%).
"""


def is_honeypot(candidate: dict) -> bool:
    """Return True if this candidate is a honeypot (must be excluded)."""

    skills = candidate.get("skills", [])

    # ---------------------------------------------------------------
    # Signal 1: Expert/Advanced proficiency with 0 months of use
    #
    # You cannot be "expert" at a skill you have never used.
    # We found exactly 21 such candidates in the 100k pool —
    # all of them honeypots.
    # ---------------------------------------------------------------
    zero_dur_expert = [
        s for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and s.get("duration_months", -1) == 0
    ]
    if len(zero_dur_expert) >= 2:
        return True

    # ---------------------------------------------------------------
    # Signal 2: Severe YOE discrepancy
    #
    # If the profile claims 13 years of experience but career history
    # only adds up to 1 year — the profile is fabricated.
    # We use a threshold of 8 years to avoid penalising real candidates
    # who have gaps (career breaks, freelancing, etc.).
    # ---------------------------------------------------------------
    profile_yoe = candidate["profile"].get("years_of_experience", 0)
    career_months = sum(
        j.get("duration_months", 0)
        for j in candidate.get("career_history", [])
    )
    career_yoe = career_months / 12.0

    if abs(profile_yoe - career_yoe) > 8.0:
        return True

    return False
