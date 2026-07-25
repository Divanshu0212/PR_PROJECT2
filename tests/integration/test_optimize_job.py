from fastapi.testclient import TestClient

import rho.api.app as appmod
from rho.api.app import app
from rho.models.api import PipelineResponse
from rho.models.resume import StructuredResume
from rho.models.provenance import ProvenanceMap
from rho.models.scoring import MatchResult, ComponentVector
from rho.models.rewrite import TailoredResume, FabricationReport


def _resp():
    return PipelineResponse(
        structured_resume=StructuredResume(name="X"),
        provenance_map=ProvenanceMap(doc_id="d"),
        match_result=MatchResult(component_vector=ComponentVector(
            keyword_coverage=0, semantic_similarity=0, fuzzy_coverage=0,
            must_have_coverage=0, nice_have_coverage=0), predicted_score=80.0),
        tailored_resume=TailoredResume(resume=StructuredResume(name="X"),
            fabrication_report=FabricationReport(total_edits=0, verified_edits=0, fabrication_rate=0.0)),
        final_score=80.0,
    )


def test_optimize_job_lifecycle(monkeypatch):
    # Replace the store's default runner path by patching run_from_structured.
    monkeypatch.setattr("rho.api.jobs.run_from_structured",
                        lambda resume, jd_text, on_stage=None: _resp())
    client = TestClient(app)
    start = client.post("/optimize", json={"resume": {"name": "X"}, "jd_text": "jd"})
    assert start.status_code == 200
    jid = start.json()["id"]

    import time
    for _ in range(200):
        poll = client.get(f"/optimize/{jid}")
        if poll.json()["state"] in ("done", "error"):
            break
        time.sleep(0.01)
    body = poll.json()
    assert body["state"] == "done"
    assert body["result"]["final_score"] == 80.0


def test_optimize_unknown_job_404():
    assert TestClient(app).get("/optimize/nope").status_code == 404
