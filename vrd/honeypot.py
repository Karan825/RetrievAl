"""
Phase 1: Universal Elimination (Honeypot Detection)
===================================================
Identifies synthetically corrupted "honeypot" profiles with
physically impossible data.

All checks are JD-agnostic data-integrity signals — they apply
to any role, any domain, any job description.

Signals Implemented:
  1. Expert/Advanced skill with zero usage duration (impossible mastery)
  2. Severe YOE discrepancy between profile claim and career history (tightened 8→5 yr)
  3. Physically impossible overlapping simultaneous full-time jobs
  4. Individual skill duration exceeds entire career duration
  5. All skills have identical duration (copy-paste synthetic profile pattern)
"""

from datetime import date


def _parse_date(s):
    """Safely parse an ISO date string. Returns None on failure."""
    try:
        return date.fromisoformat(s) if s else None
    except (ValueError, TypeError):
        return None


def is_honeypot(candidate: dict) -> bool:
    """
    Return True if this candidate is a honeypot (must be permanently excluded).
    Uses only structural data integrity checks — zero JD dependency.
    """
    skills = candidate.get("skills", [])
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    # ---------------------------------------------------------------
    # Signal 1: Expert/Advanced proficiency with 0 months of use
    #
    # You cannot be "expert" at a skill you have never used.
    # We trigger if we find 2 or more of these mathematically impossible
    # skill entries to avoid penalising a single accidental typo.
    # ---------------------------------------------------------------
    zero_dur_expert_count = 0
    for s in skills:
        if s.get("proficiency", "").lower() in ("advanced", "expert"):
            if s.get("duration_months", -1) == 0:
                zero_dur_expert_count += 1

    if zero_dur_expert_count >= 2:
        return True

    # ---------------------------------------------------------------
    # Signal 2: Severe YOE Discrepancy (hard cutoff at 5 years)
    #
    # If the profile claims 10 years of experience but career history
    # only adds up to 2 years — the profile is fabricated.
    # 5-year hard threshold still allows reasonable gaps; borderline
    # discrepancies (0-5 yr) are handled by integrity_penalty() below.
    # ---------------------------------------------------------------
    # freelancing, unlisted consulting stints) while catching fakes.
    # ---------------------------------------------------------------
    profile_yoe = profile.get("years_of_experience", 0)
    career_months = sum(max(0, job.get("duration_months", 0)) for job in career)
    career_yoe = career_months / 12.0

    if abs(profile_yoe - career_yoe) > 5.0:
        return True

    # ---------------------------------------------------------------
    # Signal 3: Physically Impossible Overlapping Full-Time Jobs
    #
    # Two full-time roles cannot genuinely overlap by more than 3 months.
    # (3-month tolerance accounts for transition periods, part-time work,
    #  or minor date entry errors.) Two or more such overlaps = synthetic.
    # ---------------------------------------------------------------
    overlap_violations = 0
    for i in range(len(career)):
        for j in range(i + 1, len(career)):
            j1, j2 = career[i], career[j]
            s1 = _parse_date(j1.get("start_date"))
            e1 = _parse_date(j1.get("end_date")) if j1.get("end_date") else date.today()
            s2 = _parse_date(j2.get("start_date"))
            e2 = _parse_date(j2.get("end_date")) if j2.get("end_date") else date.today()

            if s1 and s2 and e1 and e2:
                overlap_start = max(s1, s2)
                overlap_end = min(e1, e2)
                overlap_months = max(0, (overlap_end - overlap_start).days / 30.0)
                if overlap_months > 3:
                    overlap_violations += 1

    if overlap_violations >= 2:
        return True

    # ---------------------------------------------------------------
    # Signal 4: Skill Duration Exceeds Total Career Duration
    #
    # Hard cutoff: if ANY skill exceeds career total by MORE than 24 months
    # (2 full years) it is physically impossible — even accounting for
    # pre-career education and dataset rounding artifacts.
    #
    # Borderline excess (0-24 months) is handled gracefully by the
    # continuous integrity_penalty() function below — no hard eliminate.
    # ---------------------------------------------------------------
    career_total_months = sum(max(0, j.get("duration_months", 0)) for j in career)
    for s in skills:
        skill_dur = s.get("duration_months", 0)
        if skill_dur > career_total_months + 24:   # truly impossible (2-year buffer)
            return True

    # ---------------------------------------------------------------
    # Signal 5: All Skills Share Identical Duration (Copy-Paste Pattern)
    #
    # Real candidates have varying skill durations reflecting their
    # actual experience trajectory. When 5+ skills all have the exact
    # same duration, the profile was synthetically generated.
    # ---------------------------------------------------------------
    skill_durations = [
        s.get("duration_months", 0)
        for s in skills
        if s.get("duration_months", 0) > 0
    ]
    if len(skill_durations) >= 5 and len(set(skill_durations)) == 1:
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# SOFT INTEGRITY PENALTY
# ─────────────────────────────────────────────────────────────────────────────

def integrity_penalty(candidate: dict) -> float:
    """
    Continuous integrity multiplier in [0.10, 1.0].

    Instead of a binary eliminate/pass for borderline violations, this function
    applies an exponential decay penalty proportional to the severity of the
    data inconsistency. The worse the violation, the heavier the penalty.

    Penalises two borderline signals:

      (A) Skill-duration excess  — any skill whose duration modestly exceeds
          the candidate's total career length (0-24 month range).
          Hard violations (> 24 months) are already caught by is_honeypot().

          Penalty model: exp(-k * total_excess_ratio)
          k = 0.5, calibrated so:
            •  9% total excess  → 0.956 multiplier  (Aarav's case — tiny dip)
            • 25% total excess  → 0.882 multiplier  (moderate concern)
            • 50% total excess  → 0.778 multiplier  (significant concern)
            • 100% total excess → 0.606 multiplier  (near-elimination territory)

      (B) YOE discrepancy — gap between profile's claimed YOE and sum of career
          history (0-5 year range). Hard violations (> 5 yr) are caught by is_honeypot().

          Penalty model: exp(-k * (discrepancy_years / 5))
          k = 1.0, calibrated so:
            • 0.5 yr gap → 0.905 (minor rounding / freelance gap)
            • 1.5 yr gap → 0.741 (notable but possible career break)
            • 3.0 yr gap → 0.549 (suspicious, significant downrank)
            • 4.5 yr gap → 0.407 (very suspicious, near is_honeypot threshold)

    Combined: both penalties multiply together, floored at 0.10
    so a profile is never fully zeroed by this function alone.
    is_honeypot() handles the zero cases.

    JD-agnostic: no role names, skill names, or thresholds depend on the JD.
    """
    import math

    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    career_total = sum(max(0, j.get("duration_months", 0)) for j in career)

    # ── (A) Skill-duration excess penalty ────────────────────────────────────
    #
    # Fix (v3): Added per-skill minimum excess threshold of 12 months before
    # contributing to the penalty. Without this, candidates like Kavya Joshi
    # (4 skills with 3–13 month overages due to normal career-date rounding)
    # accumulate a ~29.5% total_excess_ratio → 0.86x multiplier, unfairly
    # penalising a legitimate profile.
    #
    # Real-world reason for small overages: skills are learned during the last
    # months of one job and the early months of the next, or pre-job in
    # education/side projects. Overages of < 12 months are indistinguishable
    # from data rounding and should not penalise.
    #
    # The is_honeypot() hard cutoff (>24 months) is unchanged — only this
    # soft continuous penalty gains a 12-month minimum floor per skill.
    skill_penalty = 1.0
    if career_total > 0:
        total_excess_ratio = 0.0
        PER_SKILL_NOISE_FLOOR_MONTHS = 12   # ignore overages ≤ 1 year per skill
        for s in skills:
            skill_dur = s.get("duration_months", 0)
            if skill_dur > career_total:
                excess = skill_dur - career_total
                # Subtract the noise floor: only count excess beyond 12 months
                penalizable_excess = max(0, excess - PER_SKILL_NOISE_FLOOR_MONTHS)
                total_excess_ratio += penalizable_excess / career_total

        if total_excess_ratio > 0:
            k_skill = 0.5
            skill_penalty = math.exp(-k_skill * total_excess_ratio)

    # ── (B) YOE discrepancy penalty ──────────────────────────────────────────
    yoe_penalty = 1.0
    profile_yoe = profile.get("years_of_experience", 0)
    career_yoe = career_total / 12.0
    discrepancy = abs(profile_yoe - career_yoe)

    if discrepancy > 0.5:   # ignore sub-6-month rounding noise
        k_yoe = 1.0
        yoe_penalty = math.exp(-k_yoe * (discrepancy / 5.0))

    combined = skill_penalty * yoe_penalty
    return max(combined, 0.10)   # floor: never fully zero from this function alone
