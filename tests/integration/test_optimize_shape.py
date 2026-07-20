from fastapi.testclient import TestClient

from rho.api.app import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_optimize_returns_pipeline_shape():
    r = client.post(
        "/optimize",
        files={"file": ("r.txt", b"Alice\npython", "text/plain")},
        data={"jd_text": "need python"},
    )
    assert r.status_code == 200
    body = r.json()
    for key in [
        "structured_resume",
        "provenance_map",
        "match_result",
        "tailored_resume",
        "final_score",
    ]:
        assert key in body
