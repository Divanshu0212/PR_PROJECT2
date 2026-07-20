"""Build résumé×JD pairs from the local corpus.

Resume.csv (2484 résumés, 24 `Category` buckets) and training_data.csv (853 JDs
with free-text `position_title`) share no join key, so pairs are constructed.
We deliberately sample both same-domain and cross-domain pairs: calibration
needs a *spread* of match quality, not only good matches, or the fit sees no
contrast and Spearman ρ is uninformative.
"""

import random
import re

import pandas as pd

from rho.models.resume import Education, StructuredResume, WorkExperience

# Coarse résumé Category -> keywords appearing in JD position_title.
CATEGORY_TITLE_HINTS = {
    "INFORMATION-TECHNOLOGY": ["developer", "engineer", "it ", "systems", "software", "web"],
    "ENGINEERING": ["engineer", "mechanical", "electrical", "civil"],
    "SALES": ["sales", "account executive", "retail"],
    "ACCOUNTANT": ["account", "bookkeep", "audit", "financial analyst"],
    "FINANCE": ["financial", "finance", "analyst", "bank"],
    "HR": ["human resources", "recruit", "hr "],
    "HEALTHCARE": ["nurse", "medical", "health", "clinical"],
    "TEACHER": ["teacher", "instructor", "tutor", "professor"],
    "CHEF": ["chef", "cook", "culinary", "kitchen"],
    "DESIGNER": ["designer", "design", "graphic", "ux", "ui"],
}

_BULLET_SPLIT = re.compile(r"\s{2,}|\n")


def _to_structured(resume_str: str, category: str) -> StructuredResume:
    """Cheap heuristic parse of the corpus text into StructuredResume.

    Phase 2's LLM extractor is the real path; for calibration we only need the
    fields ats-screener scores over, and a deterministic parse keeps the whole
    dataset build reproducible.
    """
    lines = [ln.strip() for ln in _BULLET_SPLIT.split(resume_str) if ln.strip()]
    headline = lines[0] if lines else category

    bullets = [ln for ln in lines if 40 <= len(ln) <= 300][:20]
    # Skills: short comma-free fragments that look like technology/skill tokens.
    skills = [ln for ln in lines if 3 <= len(ln) <= 30 and "," not in ln][:25]

    edu_text = " ".join(
        ln for ln in lines if re.search(r"\b(BS|BA|MS|MBA|PhD|Bachelor|Master|University|College)\b", ln)
    )[:400]

    return StructuredResume(
        name="Candidate",
        headline=headline,
        summary=" ".join(lines[1:4])[:600] or None,
        skills=skills,
        work=[WorkExperience(company="Employer", title=headline, bullets=bullets)] if bullets else [],
        education=[Education(institution=edu_text)] if edu_text else [],
    )


def _matches_category(title: str, category: str) -> bool:
    hints = CATEGORY_TITLE_HINTS.get(category, [])
    low = str(title).lower()
    return any(h in low for h in hints)


def build_pairs(
    n_pairs: int = 200,
    resume_csv: str = "Resume.csv",
    jd_csv: str = "training_data.csv",
    same_domain_ratio: float = 0.5,
    seed: int = 0,
) -> list[tuple[StructuredResume, str]]:
    rng = random.Random(seed)
    resumes = pd.read_csv(resume_csv)
    jds = pd.read_csv(jd_csv)

    # Drop length outliers: truncated stubs and multi-résumé blobs.
    resumes = resumes[resumes.Resume_str.str.len().between(800, 12000)]
    jds = jds[jds.job_description.str.len().between(400, 8000)]

    pairs = []
    n_same = int(n_pairs * same_domain_ratio)
    for i in range(n_pairs):
        r = resumes.sample(1, random_state=seed + i).iloc[0]
        candidates = jds[jds.position_title.apply(lambda t: _matches_category(t, r.Category))]
        if i < n_same and len(candidates):
            jd = candidates.sample(1, random_state=seed + i).iloc[0]
        else:
            # Cross-domain: prefer a JD that does NOT match the résumé category.
            others = jds[~jds.position_title.apply(lambda t: _matches_category(t, r.Category))]
            pool = others if len(others) else jds
            jd = pool.sample(1, random_state=seed + i).iloc[0]
        pairs.append((_to_structured(r.Resume_str, r.Category), jd.job_description))

    rng.shuffle(pairs)
    return pairs
