"""Corpus-backed pairs for the fabrication benchmark (C3, option 2).

The synthetic benchmark in `tests/fixtures/fabrication/pairs.json` is 12 curated
résumés carrying skills only, so the gate's work/education/bullet paths never
face real generated text. This module draws instead from the Phase-4 corpus
(`Resume.csv` × `training_data.csv`), giving résumés with populated work history,
bullets, and education — and far more of them.

The one thing the corpus does not give is provenance. `Resume_str` is stored as a
single line with multi-space runs standing in for the original line breaks, so
`ingest()` produces one span covering the whole document. That span "supports"
almost any value by substring match, which would hollow out the gate and break
C1's claim to an *exact* source location. `segment_corpus_text` restores the line
structure first; provenance is then built by the real ingest path over the
segmented text, so offsets stay honest.
"""

import re

from rho.ingestion import ingest
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume

# The corpus flattens line breaks into runs of 2+ spaces. `eval.corpus` splits on
# the same signal to recover bullets, so the two views of the document agree.
_SEGMENT_BREAK = re.compile(r"\s{2,}")


def segment_corpus_text(text: str) -> str:
    """Restore the line structure the corpus flattened into multi-space runs."""
    return "\n".join(seg.strip() for seg in _SEGMENT_BREAK.split(text) if seg.strip())


def corpus_prov(text: str, doc_id: str) -> ProvenanceMap:
    """ProvenanceMap over `text`, one span per recovered line.

    Offsets index the *segmented* text (what `segment_corpus_text` returns), not
    the raw CSV cell, so `raw_text` and the char range stay consistent.
    """
    _, prov = ingest(segment_corpus_text(text).encode(), f"{doc_id}.txt")
    return prov


def build_corpus_pairs(
    n_pairs: int = 30,
    seed: int = 0,
    resume_csv: str = "Resume.csv",
    jd_csv: str = "training_data.csv",
) -> list[dict]:
    """Corpus résumé × JD pairs shaped for `eval.fabrication_ablation.run`.

    Gaps come from the real Phase-3 path — `analyze_jd` then `match` — rather
    than a hand-written "tempting" list, so the pressure on the rewriter is
    whatever the JD actually demands and the résumé actually lacks.
    """
    from eval.corpus import build_pairs
    from rho.jd import analyze_jd

    # Same deviation as Phase 4's calibrator: `rho.jd.llm` needs CUDA, this host
    # has none, so JD analysis goes through the Ollama path.
    from rho.jd.ollama import analyze_jd_schema as _ollama_schema_fn
    from rho.matching import match

    pairs = []
    for i, (resume, jd_text) in enumerate(
        build_pairs(n_pairs=n_pairs, seed=seed, resume_csv=resume_csv, jd_csv=jd_csv)
    ):
        reqs = analyze_jd(jd_text, _schema_fn=_ollama_schema_fn)
        result = match(resume, reqs)
        gaps = [g for g in result.gaps if g.status != "present"]
        pairs.append(
            {
                "id": f"corpus-{seed}-{i}",
                "resume": resume,
                "prov": _prov_for(resume, i),
                "jd": jd_text,
                "gaps": gaps,
                "tempting_absent": [g.requirement.text for g in gaps],
            }
        )
    return pairs


def _prov_for(resume: StructuredResume, idx: int) -> ProvenanceMap:
    """Provenance over the résumé's own values.

    `build_pairs` hands back a parsed `StructuredResume`, not the raw cell, so
    the source document is reconstructed from the values it kept. Anything the
    parse dropped is genuinely absent from the source as far as the gate is
    concerned — which is the conservative direction: the gate can only be
    stricter than reality, never more permissive.
    """
    lines = [resume.name, resume.headline or "", resume.summary or ""]
    lines += resume.skills
    for w in resume.work:
        lines += [w.company, w.title, *w.bullets]
    for e in resume.education:
        lines += [e.institution, e.degree or "", e.field or ""]
    doc = "\n".join(ln.strip() for ln in lines if ln and ln.strip())
    _, prov = ingest(doc.encode(), f"corpus{idx}.txt")
    return prov
