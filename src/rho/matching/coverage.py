from functools import lru_cache

from rapidfuzz import fuzz


@lru_cache(maxsize=1)
def _keybert():
    from keybert import KeyBERT

    from rho.matching.embed import _model

    # reuse the cached mpnet instance rather than loading a second model
    return KeyBERT(model=_model())


def extract_jd_terms(jd_text: str, top_n: int = 15) -> list[str]:
    """KeyBERT-extracted keyphrases from raw JD text, for use as `req_terms`."""
    if not jd_text or not jd_text.strip():
        return []
    pairs = _keybert().extract_keywords(
        jd_text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n
    )
    return [phrase for phrase, _score in pairs]


def resume_text_terms(resume) -> list[str]:
    """All résumé strings a requirement could be evidenced by.

    Coverage originally searched `skills` alone, which assumes requirements are
    short skill tokens. Real JD requirements are phrases ("translating designs
    responsively"), and their evidence usually sits in an experience bullet, so
    matching against skills alone pins both coverage signals at 0.
    """
    terms = list(resume.skills) + list(resume.certifications)
    if resume.headline:
        terms.append(resume.headline)
    if resume.summary:
        terms.append(resume.summary)
    for w in resume.work:
        terms.extend([w.title, w.company, *w.bullets])
    for e in resume.education:
        terms.extend(filter(None, [e.institution, e.degree, e.field]))
    return [t for t in terms if t]


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
