import re

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


# Function words carry no skill signal and would manufacture overlap between
# unrelated phrases ("search and experience" vs "welding and pipefitting").
_STOPWORDS = frozenset(
    "a an the and or of in on for to with at by from as is are be experience "
    "years year strong excellent good ability able must should will".split()
)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9+#.]+", text.lower()) if w not in _STOPWORDS}


def keyword_coverage(req_terms: list[str], resume_skills: list[str]) -> float:
    """Mean per-requirement content-word overlap with the résumé.

    Whole-string containment only fires for single tokens: a phrasal
    requirement like "account project management experience" is absent verbatim
    from every résumé, even one saying "key account management". Scoring each
    requirement by the fraction of its content words present keeps exact
    single-token behaviour (1.0 or 0.0) while giving phrases partial credit.
    """
    if not req_terms:
        return 1.0
    blob_words = _content_words(" ".join(resume_skills))
    scores = []
    for term in req_terms:
        words = _content_words(term)
        if not words:
            continue
        scores.append(len(words & blob_words) / len(words))
    return (sum(scores) / len(scores)) if scores else 0.0


def fuzzy_coverage(
    req_terms: list[str], resume_skills: list[str], threshold: int = 85
) -> float:
    """Mean per-requirement content-word overlap, allowing typos (RapidFuzz).

    Word-level like `keyword_coverage`: comparing a whole phrase against a
    whole skill string never clears the threshold, so phrasal requirements
    scored 0 regardless of how well the résumé matched.
    """
    if not req_terms:
        return 1.0
    resume_words = _content_words(" ".join(resume_skills))
    scores = []
    for term in req_terms:
        words = _content_words(term)
        if not words:
            continue
        hit = sum(
            1
            for w in words
            if any(fuzz.ratio(w, rw) >= threshold for rw in resume_words)
        )
        scores.append(hit / len(words))
    return (sum(scores) / len(scores)) if scores else 0.0


