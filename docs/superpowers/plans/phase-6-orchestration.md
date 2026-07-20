# Phase 6 — LangGraph Orchestration + Reviewer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.
> **Read `00-SHARED-CONTEXT.md` first.** Confirm Phases 0–5 done.

**Goal:** Wire all components into one LangGraph pipeline behind `run_pipeline()` and the `/optimize` API. Résumé-parse and JD-analyze branches run in parallel, fan in at the scorer; the calibrator fills `predicted_score`; the rewriter+gate produce the tailored résumé; a **reviewer node re-asserts the end-to-end provenance invariant** and computes `final_score`.

**Architecture:** LangGraph `StateGraph` over a typed state dict. Nodes: `ingest_node`, `extract_node`, `jd_node` (parallel with ingest+extract), `match_node` (fan-in barrier — waits for both branches), `score_node` (apply calibrator), `rewrite_node` (generate + gate), `review_node` (provenance-invariant assertion + final score). Checkpointing via `MemorySaver` (swap `PostgresSaver` in prod). `/optimize` calls `run_pipeline` and returns `PipelineResponse`.

**Tech Stack:** LangGraph, FastAPI (existing).

## Global Constraints
- Implement `rho.graph.run_pipeline(file_bytes, filename, jd_text) -> PipelineResponse` (frozen signature).
- The reviewer node MUST assert the provenance invariant (shared-context Section 7): every hard-content token in the tailored résumé traces to a `prov_id`. On violation, flag in the response (do not crash).
- Parallel branches must fan in correctly — `match_node` runs only after BOTH `extract_node` and `jd_node` complete.
- Calibrator loaded from `eval/calibrator.joblib` if present; else `predicted_score` stays raw-vector-derived fallback (documented).

## This phase consumes
- Every component: `ingest`, `extract`, `analyze_jd`, `match`, `Calibrator`+`score_with_calibrator`, `rewrite`, `verify_against_source` (Phases 1–5).

## This phase produces
- `run_pipeline()`; the LangGraph `build_graph()`; `/optimize` wired to it.
- `rho.graph.review.check_provenance_invariant(tailored, prov) -> tuple[bool, list[str]]`.

---

## File Structure
- Create: `src/rho/graph/state.py` — `PipelineState` TypedDict.
- Create: `src/rho/graph/nodes.py` — node functions.
- Create: `src/rho/graph/review.py` — provenance-invariant check + final score.
- Modify: `src/rho/graph/__init__.py` — `build_graph`, `run_pipeline`.
- Modify: `src/rho/api/app.py` — `/optimize` calls `run_pipeline`.
- Create: `tests/integration/test_pipeline.py`.

---

### Task 1: Pipeline state + reviewer invariant check

**Files:**
- Create: `src/rho/graph/state.py`, `src/rho/graph/review.py`
- Test: `tests/unit/test_review.py`

**Interfaces:**
- Produces: `PipelineState` (TypedDict with `file_bytes, filename, jd_text, markdown, prov, resume, reqs, match_result, tailored, final_score, invariant_ok, invariant_violations`). `check_provenance_invariant(tailored: StructuredResume, prov) -> (bool, list[str])` — returns False + list of unsupported values if any hard-content token lacks a `prov_id`. `compute_final_score(match_result, fabrication_report) -> float`.

- [ ] **Step 1: Write failing test**
```python
# tests/unit/test_review.py
from rho.models.provenance import SourceSpan, ProvenanceMap
from rho.models.resume import StructuredResume
from rho.graph.review import check_provenance_invariant, compute_final_score
from rho.models.scoring import MatchResult, ComponentVector
from rho.models.rewrite import FabricationReport
def _pm():
    pm = ProvenanceMap(doc_id="d")
    pm.add(SourceSpan(doc_id="d", char_start=0, char_end=6, raw_text="Python"))
    return pm
def test_invariant_passes_when_all_sourced():
    ok, viol = check_provenance_invariant(StructuredResume(name="A", skills=["Python"]), _pm())
    assert ok and viol == []
def test_invariant_fails_on_unsourced_token():
    ok, viol = check_provenance_invariant(StructuredResume(name="A", skills=["Rust"]), _pm())
    assert not ok and "Rust" in viol
def test_final_score_penalized_by_fabrication():
    cv = ComponentVector(keyword_coverage=1,semantic_similarity=1,fuzzy_coverage=1,
        must_have_coverage=1,nice_have_coverage=1)
    mr = MatchResult(component_vector=cv, predicted_score=80.0)
    clean = FabricationReport(total_edits=0, verified_edits=0, fabrication_rate=0.0)
    assert compute_final_score(mr, clean) == 80.0
```

- [ ] **Step 2: Run to verify fail** → FAIL.
Run: `pytest tests/unit/test_review.py -v`

- [ ] **Step 3: Implement**
```python
# src/rho/graph/state.py
from typing import TypedDict, Optional
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
from rho.models.jd import RequirementSet
from rho.models.scoring import MatchResult
from rho.models.rewrite import TailoredResume
class PipelineState(TypedDict, total=False):
    file_bytes: bytes; filename: str; jd_text: str
    markdown: str; prov: ProvenanceMap
    resume: StructuredResume; reqs: RequirementSet
    match_result: MatchResult; tailored: TailoredResume
    final_score: float; invariant_ok: bool; invariant_violations: list[str]
```
```python
# src/rho/graph/review.py
from rho.rewrite.tokens import hard_content_tokens
from rho.extraction.provenance_attach import find_prov
def check_provenance_invariant(tailored, prov):
    violations = []
    for value, _path in hard_content_tokens(tailored):
        if not find_prov(value, prov):
            violations.append(value)
    return (len(violations) == 0, violations)
def compute_final_score(match_result, fabrication_report):
    # predicted_score is the calibrated ATS score; fabrication already prevented by gate,
    # so no double penalty. Kept as a hook for future weighting.
    return match_result.predicted_score
```

- [ ] **Step 4: Run to verify pass** → PASS.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: pipeline state + provenance-invariant reviewer"
```

---

### Task 2: Graph nodes

**Files:**
- Create: `src/rho/graph/nodes.py`
- Test: covered by Task 3 end-to-end (nodes are thin wrappers)

**Interfaces:**
- Produces: node functions each `(state: PipelineState) -> dict` (partial state update). `ingest_node`, `extract_node`, `jd_node`, `match_node`, `score_node`, `rewrite_node`, `review_node`. `score_node` loads calibrator if `eval/calibrator.joblib` exists.

- [ ] **Step 1: Implement**
```python
# src/rho/graph/nodes.py
import os
from rho.ingestion import ingest
from rho.extraction import extract
from rho.jd import analyze_jd
from rho.matching import match
from rho.ats import Calibrator, score_with_calibrator
from rho.rewrite import rewrite
from rho.graph.review import check_provenance_invariant, compute_final_score
def ingest_node(state):
    md, prov = ingest(state["file_bytes"], state["filename"])
    return {"markdown": md, "prov": prov}
def extract_node(state):
    return {"resume": extract(state["markdown"], state["prov"])}
def jd_node(state):
    return {"reqs": analyze_jd(state["jd_text"])}
def match_node(state):
    return {"match_result": match(state["resume"], state["reqs"])}
def score_node(state):
    mr = state["match_result"]
    if os.path.exists("eval/calibrator.joblib"):
        cal = Calibrator().load("eval/calibrator.joblib")
        mr = score_with_calibrator(mr, cal)
    return {"match_result": mr}
def rewrite_node(state):
    return {"tailored": rewrite(state["resume"], state["match_result"].gaps, state["prov"])}
def review_node(state):
    ok, viol = check_provenance_invariant(state["tailored"].resume, state["prov"])
    final = compute_final_score(state["match_result"], state["tailored"].fabrication_report)
    return {"invariant_ok": ok, "invariant_violations": viol, "final_score": final}
```

- [ ] **Step 2: Commit**
```bash
git add -A && git commit -m "feat: LangGraph node functions"
```

---

### Task 3: Build graph + `run_pipeline` + parallel fan-in

**Files:**
- Modify: `src/rho/graph/__init__.py`
- Test: `tests/integration/test_pipeline.py`

**Interfaces:**
- Produces: `build_graph()` (compiled LangGraph) and `run_pipeline(file_bytes, filename, jd_text) -> PipelineResponse`. Graph edges: `START → ingest_node → extract_node`; `START → jd_node`; both `extract_node` and `jd_node → match_node` (fan-in); `match_node → score_node → rewrite_node → review_node → END`. For tests without an LLM, `run_pipeline` accepts injectable `_extract_fn`/`_rewrite_fn`/`_jd_fn` OR the test monkeypatches component functions.

- [ ] **Step 1: Add dep**
Add `langgraph>=0.2` to `pyproject.toml`; install.

- [ ] **Step 2: Write integration test (monkeypatched components, no LLM)**
```python
# tests/integration/test_pipeline.py
from rho.models.resume import StructuredResume
from rho.models.jd import RequirementSet, Requirement
def test_pipeline_end_to_end(monkeypatch):
    import rho.graph.nodes as N
    monkeypatch.setattr(N, "extract",
        lambda md, prov: StructuredResume(name="A", skills=["Python"], skills_prov=[["x"]]))
    monkeypatch.setattr(N, "analyze_jd",
        lambda jd: RequirementSet(requirements=[Requirement(text="Python", kind="skill", priority="must")]))
    monkeypatch.setattr(N, "rewrite",
        lambda resume, gaps, prov: __import__("rho.models.rewrite", fromlist=["TailoredResume","FabricationReport"]).TailoredResume(
            resume=resume, fabrication_report=__import__("rho.models.rewrite", fromlist=["FabricationReport"]).FabricationReport(
                total_edits=0, verified_edits=0, fabrication_rate=0.0)))
    from rho.graph import run_pipeline
    resp = run_pipeline(b"Alice\nPython", "r.txt", "need python")
    assert resp.structured_resume.name == "A"
    assert resp.match_result.gaps[0].requirement.text == "Python"
    assert isinstance(resp.final_score, float)
```

- [ ] **Step 3: Run to verify fail** → FAIL.
Run: `pytest tests/integration/test_pipeline.py -v`

- [ ] **Step 4: Implement**
```python
# src/rho/graph/__init__.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from rho.graph.state import PipelineState
from rho.graph import nodes as N
from rho.models.api import PipelineResponse
def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("ingest", N.ingest_node); g.add_node("extract", N.extract_node)
    g.add_node("jd", N.jd_node); g.add_node("match", N.match_node)
    g.add_node("score", N.score_node); g.add_node("rewrite", N.rewrite_node)
    g.add_node("review", N.review_node)
    g.add_edge(START, "ingest"); g.add_edge("ingest", "extract")
    g.add_edge(START, "jd")
    g.add_edge("extract", "match"); g.add_edge("jd", "match")   # fan-in barrier
    g.add_edge("match", "score"); g.add_edge("score", "rewrite")
    g.add_edge("rewrite", "review"); g.add_edge("review", END)
    return g.compile(checkpointer=MemorySaver())
_graph = None
def run_pipeline(file_bytes: bytes, filename: str, jd_text: str) -> PipelineResponse:
    global _graph
    if _graph is None: _graph = build_graph()
    final = _graph.invoke(
        {"file_bytes": file_bytes, "filename": filename, "jd_text": jd_text},
        config={"configurable": {"thread_id": "run"}})
    return PipelineResponse(
        structured_resume=final["resume"],
        provenance_map=final["prov"],
        match_result=final["match_result"],
        tailored_resume=final["tailored"],
        final_score=final["final_score"],
    )
```

- [ ] **Step 5: Run to verify pass** → PASS.

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat: LangGraph pipeline + run_pipeline with parallel fan-in"
```

---

### Task 4: Wire `/optimize` to `run_pipeline`

**Files:**
- Modify: `src/rho/api/app.py`
- Test: `tests/integration/test_pipeline.py` (add API-level, monkeypatched)

**Interfaces:**
- Produces: `/optimize` now calls `run_pipeline` with the uploaded file + `jd_text`, returns the real `PipelineResponse`.

- [ ] **Step 1: Write failing test**
```python
# add to tests/integration/test_pipeline.py
def test_optimize_endpoint_calls_pipeline(monkeypatch):
    from rho.models.api import PipelineResponse
    from rho.models.resume import StructuredResume
    from rho.models.provenance import ProvenanceMap
    from rho.models.scoring import MatchResult, ComponentVector
    from rho.models.rewrite import TailoredResume, FabricationReport
    import rho.api.app as A
    cv = ComponentVector(keyword_coverage=1,semantic_similarity=1,fuzzy_coverage=1,must_have_coverage=1,nice_have_coverage=1)
    resume = StructuredResume(name="A")
    monkeypatch.setattr(A, "run_pipeline", lambda fb, fn, jd: PipelineResponse(
        structured_resume=resume, provenance_map=ProvenanceMap(doc_id="d"),
        match_result=MatchResult(component_vector=cv, predicted_score=77.0),
        tailored_resume=TailoredResume(resume=resume, fabrication_report=FabricationReport(total_edits=0,verified_edits=0,fabrication_rate=0.0)),
        final_score=77.0))
    from fastapi.testclient import TestClient
    c = TestClient(A.app)
    r = c.post("/optimize", files={"file": ("r.txt", b"x", "text/plain")}, data={"jd_text": "jd"})
    assert r.json()["final_score"] == 77.0
```

- [ ] **Step 2: Run to verify fail** → FAIL (still returns placeholder).

- [ ] **Step 3: Implement** — replace placeholder in `app.py`:
```python
from rho.graph import run_pipeline
@app.post("/optimize", response_model=PipelineResponse)
async def optimize(file: UploadFile, jd_text: str = Form(...)):
    data = await file.read()
    return run_pipeline(data, file.filename or "resume", jd_text)
```
(Remove the `_placeholder_response` helper.)

- [ ] **Step 4: Run to verify pass** → PASS.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: wire /optimize to run_pipeline"
```

---

## Self-Review
- [ ] Graph fans in at `match` after both `extract` and `jd`.
- [ ] `run_pipeline` returns a full `PipelineResponse`.
- [ ] Reviewer sets `invariant_ok`/violations; `/optimize` returns real pipeline output.
- [ ] `pytest tests/integration/test_pipeline.py -v` green.

## Results (fill in)
- LangGraph version: ___
- End-to-end latency (mock / real LLM): ___
- Invariant violations observed on real runs: ___
- Tests passing: ___ / ___
