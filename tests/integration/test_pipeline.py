"""End-to-end graph runs with the LLM-backed components monkeypatched out.

The point is the wiring — parallel branches, fan-in, reviewer — not model
quality, so `extract`, `analyze_jd` and `rewrite` are replaced with stubs.
"""

from rho.models.jd import Requirement, RequirementSet
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
from rho.models.rewrite import FabricationReport, TailoredResume
from rho.models.scoring import ComponentVector, MatchResult


def _stub_nodes(monkeypatch):
    import rho.graph.nodes as N

    monkeypatch.setattr(
        N,
        "extract",
        lambda md, prov: StructuredResume(
            name="A", skills=["Python"], skills_prov=[["x"]]
        ),
    )
    monkeypatch.setattr(
        N,
        "analyze_jd",
        lambda jd: RequirementSet(
            requirements=[Requirement(text="Python", kind="skill", priority="must")]
        ),
    )
    monkeypatch.setattr(
        N,
        "rewrite",
        lambda resume, gaps, prov: TailoredResume(
            resume=resume,
            fabrication_report=FabricationReport(
                total_edits=0, verified_edits=0, fabrication_rate=0.0
            ),
        ),
    )


def test_pipeline_end_to_end(monkeypatch):
    _stub_nodes(monkeypatch)
    from rho.graph import run_pipeline

    resp = run_pipeline(b"Alice\nPython", "r.txt", "need python")
    assert resp.structured_resume.name == "A"
    assert resp.match_result.gaps[0].requirement.text == "Python"
    assert isinstance(resp.final_score, float)


def test_pipeline_fans_in_after_both_branches(monkeypatch):
    """`match` must not run until both `extract` and `jd` have landed."""
    _stub_nodes(monkeypatch)
    import rho.graph.nodes as N

    calls = []
    real_match = N.match

    def spy_match(resume, reqs):
        calls.append((resume, reqs))
        return real_match(resume, reqs)

    monkeypatch.setattr(N, "match", spy_match)
    from rho.graph import run_pipeline

    run_pipeline(b"Alice\nPython", "r.txt", "need python")
    # Exactly once: default any-of triggering would fire `match` early on the
    # short jd branch and then again after extract.
    assert len(calls) == 1
    resume, reqs = calls[0]
    assert resume.skills == ["Python"]
    assert reqs.requirements[0].text == "Python"


def test_optimize_endpoint_calls_pipeline(monkeypatch):
    import rho.api.app as A

    cv = ComponentVector(
        keyword_coverage=1,
        semantic_similarity=1,
        fuzzy_coverage=1,
        must_have_coverage=1,
        nice_have_coverage=1,
    )
    resume = StructuredResume(name="A")
    from rho.models.api import PipelineResponse

    monkeypatch.setattr(
        A,
        "run_pipeline",
        lambda fb, fn, jd: PipelineResponse(
            structured_resume=resume,
            provenance_map=ProvenanceMap(doc_id="d"),
            match_result=MatchResult(component_vector=cv, predicted_score=77.0),
            tailored_resume=TailoredResume(
                resume=resume,
                fabrication_report=FabricationReport(
                    total_edits=0, verified_edits=0, fabrication_rate=0.0
                ),
            ),
            final_score=77.0,
        ),
    )
    from fastapi.testclient import TestClient

    c = TestClient(A.app)
    r = c.post(
        "/optimize",
        files={"file": ("r.txt", b"x", "text/plain")},
        data={"jd_text": "jd"},
    )
    assert r.json()["final_score"] == 77.0
