"""
Stage 4: Behavioral Signal Modifiers
======================================
The 23 Redrob behavioral signals act as a MULTIPLIER on top of the
career relevance score — they do not change who is relevant, they
adjust availability and hiring probability.

JD says: "a perfect-on-paper candidate who hasn't logged in for
6 months and has a 5% recruiter response rate is, for hiring
purposes, not actually available. Down-weight them appropriately."

This module implements that literally.
"""

import datetime

# Reference date — treat as "today" for consistency
TODAY = datetime.date(2026, 6, 10)

# JD location preferences
TOP_LOCATIONS = {"pune", "noida", "gurgaon", "delhi", "ncr", "new delhi"}
TIER1_LOCATIONS = {
    "hyderabad", "bangalore", "bengaluru", "mumbai",
    "chennai", "kolkata", "ahmedabad",
}


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
    if rate >= 0.80:
        return 1.15
    if rate >= 0.55:
        return 1.05
    if rate >= 0.35:
        return 0.95
    if rate >= 0.20:
        return 0.80
    return 0.55   # < 20% response rate — practically a ghost


def _notice_multiplier(notice_days: int) -> float:
    """JD says: sub-30 preferred, can buy out 30 days; 30+ bar gets higher."""
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


def _location_multiplier(candidate: dict) -> float:
    """JD: Pune/Noida preferred; Tier-1 Indian cities welcome."""
    location = candidate["profile"].get("location", "").lower()
    country = candidate["profile"].get("country", "").lower()
    willing = candidate["redrob_signals"].get("willing_to_relocate", False)

    if any(city in location for city in TOP_LOCATIONS):
        return 1.10   # already in preferred city
    if any(city in location for city in TIER1_LOCATIONS):
        return 1.05   # Tier-1 India city
    if country == "india" and willing:
        return 0.95   # India but willing to move
    if country == "india":
        return 0.85   # India, not willing to relocate
    if willing:
        return 0.60   # outside India but willing (JD: case-by-case)
    return 0.40       # outside India, not willing to relocate


def compute_signal_multiplier(candidate: dict) -> float:
    """
    Compute the composite behavioral signal multiplier.

    The multiplier is applied directly to the career_score so that
    a great career + great signals scores highest, and a great career
    + terrible signals scores much lower.

    Returns a float roughly in [0.20, 2.00].
    """
    signals = candidate.get("redrob_signals", {})

    multiplier = 1.0

    # Availability / activity
    multiplier *= _activity_multiplier(signals.get("last_active_date", "2020-01-01"))

    # Recruiter responsiveness
    multiplier *= _response_multiplier(signals.get("recruiter_response_rate", 0.5))

    # Notice period
    multiplier *= _notice_multiplier(signals.get("notice_period_days", 90))

    # Open to work flag — small bonus
    if signals.get("open_to_work_flag", False):
        multiplier *= 1.08

    # GitHub activity
    multiplier *= _github_multiplier(signals.get("github_activity_score", -1))

    # Platform skill assessments
    multiplier *= _skill_assessment_multiplier(
        signals.get("skill_assessment_scores", {})
    )

    # Interview reliability
    multiplier *= _interview_reliability_multiplier(
        signals.get("interview_completion_rate", 0.5)
    )

    # Profile completeness
    completeness = signals.get("profile_completeness_score", 50)
    if completeness >= 90:
        multiplier *= 1.05
    elif completeness >= 70:
        multiplier *= 1.02

    # Location
    multiplier *= _location_multiplier(candidate)

    return multiplier
