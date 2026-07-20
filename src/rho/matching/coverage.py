from rapidfuzz import fuzz


def keyword_coverage(req_terms: list[str], resume_skills: list[str]) -> float:
    """Fraction of requirement terms literally present in the resume skills."""
    if not req_terms:
        return 1.0
    blob = " ".join(s.lower() for s in resume_skills)
    hit = sum(1 for t in req_terms if t.lower() in blob)
    return hit / len(req_terms)


def fuzzy_coverage(
    req_terms: list[str], resume_skills: list[str], threshold: int = 85
) -> float:
    """Fraction of requirement terms matched allowing typos (RapidFuzz)."""
    if not req_terms:
        return 1.0
    low = [s.lower() for s in resume_skills]
    hit = sum(
        1
        for t in req_terms
        if any(fuzz.ratio(t.lower(), s) >= threshold or t.lower() in s for s in low)
    )
    return hit / len(req_terms)
