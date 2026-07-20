from rapidfuzz import fuzz

from rho.matching.coverage import fuzzy_coverage, keyword_coverage
from rho.matching.embed import Embedder
from rho.models.jd import RequirementSet
from rho.models.resume import StructuredResume
from rho.models.scoring import ComponentVector, Gap, MatchResult


def _prov_for(resume: StructuredResume, i: int) -> list[str]:
    """skills_prov may be shorter than skills; missing prov is [], never a dropped skill."""
    return resume.skills_prov[i] if i < len(resume.skills_prov) else []


def _skill_evidence(
    term: str,
    resume: StructuredResume,
    emb: Embedder,
    sem_hi: float = 0.65,
    sem_lo: float = 0.45,
) -> tuple[str, list[str]]:
    """returns (status, prov_ids)"""
    tl = term.lower()
    for i, skill in enumerate(resume.skills):
        sl = skill.lower()
        if tl in sl or sl in tl or fuzz.ratio(tl, sl) >= 85:
            return "present", _prov_for(resume, i)
    if resume.skills:
        tv = emb.encode([term])[0]
        sv = emb.encode(resume.skills)
        best_i = max(range(len(resume.skills)), key=lambda i: emb.cosine(tv, sv[i]))
        best = emb.cosine(tv, sv[best_i])
        if best >= sem_hi:
            return "present", _prov_for(resume, best_i)
        if best >= sem_lo:
            return "weak", _prov_for(resume, best_i)
    return "absent", []


def match(resume: StructuredResume, reqs: RequirementSet) -> MatchResult:
    """fills component_vector + gaps; predicted_score left 0.0 until P4"""
    emb = Embedder()
    req_terms = [r.text for r in reqs.requirements]
    must = [r for r in reqs.requirements if r.priority == "must"]
    nice = [r for r in reqs.requirements if r.priority == "nice"]
    gaps = []
    present_must = present_nice = 0
    for r in reqs.requirements:
        status, prov = _skill_evidence(r.text, resume, emb)
        gaps.append(Gap(requirement=r, status=status, evidence_prov=prov))
        if status in ("present", "weak"):
            if r.priority == "must":
                present_must += 1
            else:
                present_nice += 1
    cv = ComponentVector(
        keyword_coverage=keyword_coverage(req_terms, resume.skills),
        semantic_similarity=sum(1 for g in gaps if g.status in ("present", "weak"))
        / max(len(gaps), 1),
        fuzzy_coverage=fuzzy_coverage(req_terms, resume.skills),
        must_have_coverage=(present_must / len(must)) if must else 1.0,
        nice_have_coverage=(present_nice / len(nice)) if nice else 1.0,
    )
    return MatchResult(component_vector=cv, predicted_score=0.0, gaps=gaps)
