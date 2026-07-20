# Phase 4 — ATS Harness + Calibration (Core Novelty C2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.
> **Read `00-SHARED-CONTEXT.md` first.** Confirm Phases 0–3 done. **This is the paper's hardest, most novel phase (C2).**

**Goal:** Produce a `predicted_score` (0–100) **calibrated against the actual parse+match output of self-hostable ATS engines**, rather than a cosine proxy. Deliver: a harness that runs ≥2 open ATS engines and harvests their real outputs; a fitted `Calibrator` mapping the `ComponentVector` → real-engine outcome; held-out calibration metrics (MAE, Spearman ρ); and an ablation vs the raw cosine score.

**Architecture:** `harvest_ats(file, jd)` runs each configured open ATS engine on a résumé+JD and returns a normalized label dict (per-engine: parse success rate of known fields, match/rank score if the engine exposes one). Aggregate engines into a single ground-truth target `y ∈ [0,100]`. Collect a dataset of `(ComponentVector, y)` over many résumé×JD pairs. Fit a `Calibrator` (start linear/logistic; upgrade to small gradient-boost if it helps held-out fit). `match()` output is passed through `Calibrator.predict` to fill `predicted_score`.

**Tech Stack:** self-hostable ATS engines (survey in Task 0), scikit-learn (calibrator), numpy, scipy (Spearman).

## ⚠️ Risk & Task 0 (do FIRST)
The whole C2 claim depends on getting ≥2 self-hostable ATS engines whose **parse and/or match output is harvestable**. Task 0 is a survey + go/no-go. Fallback if none give a usable match score: use the **controlled parse-injection** ground truth (inject known fields, measure recovery across parsers) as the target `y` — this still yields a real-engine-grounded, reproducible calibration and preserves the novelty, reframed as "parse-ability calibration."

## Global Constraints
- Implement `rho.ats.harvest_ats(file_bytes, filename, jd_text) -> dict` and `rho.ats.Calibrator` (`fit`, `predict`).
- Ground truth must come from **real engine behavior** (parse recovery and/or match score), never a hand-set formula.
- Keep every engine's raw output logged (reproducibility); aggregation to `y` is a documented, single function.
- Missing engine output on a doc → exclude that doc from fit (do not impute).

## This phase consumes
- `ComponentVector` from `match()` (Phase 3) as calibration features.
- `ingest`/`extract`/`analyze_jd`/`match` (Phases 1–3) to build the feature side of the dataset.

## This phase produces
- `harvest_ats()`; `Calibrator`; `rho.ats.dataset.build_calibration_dataset(...)`.
- `rho.ats.aggregate.to_target(engine_outputs) -> float`.
- Fitted calibrator artifact (`eval/calibrator.joblib`) + held-out metrics recorded in Results.

---

## File Structure
- Create: `src/rho/ats/engines/` — one adapter per engine (`base.py`, `<engine>.py`).
- Create: `src/rho/ats/aggregate.py` — `to_target`.
- Create: `src/rho/ats/dataset.py` — build (feature, target) pairs.
- Create: `src/rho/ats/calibrator.py` — `Calibrator`.
- Modify: `src/rho/ats/__init__.py` — export `harvest_ats`, `Calibrator`.
- Create: `tests/unit/test_ats_calibrator.py`, `tests/integration/test_ats_harness.py`.
- Create: `eval/fit_calibrator.py` — script to build dataset + fit + report metrics.

---

### Task 0: Engine survey & go/no-go (research spike, no test)

**Files:** Create `docs/superpowers/plans/phase-4-engine-survey.md` (findings).

- [ ] Survey self-hostable ATS / open résumé-parser + matcher stacks. Candidate directions to evaluate:
  - Open applicant-tracking systems (e.g. OpenCATS-class) — do they expose a résumé parse result and/or a candidate-JD match score via DB/API?
  - Open résumé parsers (spaCy/pyresparser-class, the Qwen2.5-based successor noted in the report) — parse-field recovery as target.
  - Open matcher stacks / `Keywords4CV`-style scorers — a numeric match as target.
- [ ] For each: can you run it headless (Docker/CLI/lib) and read a numeric/structured output programmatically? Record install method + output shape.
- [ ] **Decision:** pick ≥2 engines with harvestable output. If <2 expose a *match score*, adopt the **parse-injection fallback** (Task 0b) as the target and note the reframing ("parse-ability calibration").
- [ ] Write findings + decision into `phase-4-engine-survey.md`. **Do not proceed until decided.**

**Task 0b (fallback only, if adopted): parse-injection target.**
- [ ] Build `inject_known_fields(resume_text) -> (resume_text, known_fields)` and, per engine/parser, `recovery_rate(parsed, known_fields) -> float`. Target `y` = mean recovery across engines × 100. This is real-engine-grounded and reproducible.

---

### Task 1: Engine adapter interface + first engine

**Files:**
- Create: `src/rho/ats/engines/base.py`, `src/rho/ats/engines/<engine1>.py`
- Test: `tests/integration/test_ats_harness.py` (skips unless engine available)

**Interfaces:**
- Produces: `class ATSEngine` (protocol) with `.run(file_bytes, filename, jd_text) -> dict` returning at minimum `{"engine": name, "parse_fields": {...}|None, "match_score": float|None, "raw": ...}`. Concrete engine1 adapter implements it.

- [ ] **Step 1: Write skipping integration test**
```python
# tests/integration/test_ats_harness.py
import os, pytest
pytestmark = pytest.mark.skipif(os.getenv("RHO_ATS_ENABLED") != "1", reason="no ATS engine")
def test_engine1_runs_and_returns_output():
    from rho.ats.engines.engine1 import Engine1        # rename to real engine
    out = Engine1().run(b"Alice\nPython\nAWS", "r.txt", "need python")
    assert out["engine"]
    assert ("parse_fields" in out) or ("match_score" in out)
```

- [ ] **Step 2: Run to verify skip**
Run: `pytest tests/integration/test_ats_harness.py -v` → SKIP.

- [ ] **Step 3: Implement base + engine1**
```python
# src/rho/ats/engines/base.py
from typing import Protocol
class ATSEngine(Protocol):
    name: str
    def run(self, file_bytes: bytes, filename: str, jd_text: str) -> dict: ...
```
Implement `engine1.py` per Task-0 findings (Docker/CLI/lib call → normalized dict). Keep raw output under `out["raw"]`.

- [ ] **Step 4: Commit**
```bash
git add -A && git commit -m "feat: ATS engine adapter interface + first engine"
```

- [ ] **Step 5: Repeat for engine2** (second file, same protocol). Commit.

---

### Task 2: `harvest_ats` + target aggregation

**Files:**
- Create: `src/rho/ats/aggregate.py`; Modify: `src/rho/ats/__init__.py`
- Test: `tests/unit/test_ats_calibrator.py`

**Interfaces:**
- Produces: `harvest_ats(file_bytes, filename, jd_text) -> dict` = `{engine_name: engine_output, ...}` over all configured engines. `to_target(engine_outputs: dict) -> float` = aggregated `y ∈ [0,100]` (mean of available per-engine scores; skip missing). Aggregation is LLM-free and unit-testable with fake engine outputs.

- [ ] **Step 1: Write failing test**
```python
# tests/unit/test_ats_calibrator.py
from rho.ats.aggregate import to_target
def test_to_target_means_available_scores():
    outs = {"e1": {"match_score": 80.0}, "e2": {"match_score": 60.0}}
    assert to_target(outs) == 70.0
def test_to_target_skips_missing():
    outs = {"e1": {"match_score": 80.0}, "e2": {"match_score": None}}
    assert to_target(outs) == 80.0
```

- [ ] **Step 2: Run to verify fail**
Run: `pytest tests/unit/test_ats_calibrator.py -k to_target -v` → FAIL.

- [ ] **Step 3: Implement**
```python
# src/rho/ats/aggregate.py
def to_target(engine_outputs: dict) -> float:
    scores = [o["match_score"] for o in engine_outputs.values()
              if o.get("match_score") is not None]
    if not scores:
        raise ValueError("no engine produced a score; exclude this doc from fit")
    return sum(scores) / len(scores)
```
```python
# src/rho/ats/__init__.py  (add)
def harvest_ats(file_bytes: bytes, filename: str, jd_text: str) -> dict:
    from rho.ats.registry import ENGINES     # list built in Task 1
    return {e.name: e.run(file_bytes, filename, jd_text) for e in ENGINES}
```
Create `src/rho/ats/registry.py` exposing `ENGINES = [Engine1(), Engine2()]` (guard construction so import doesn't require the engines at unit-test time — lazy or try/except).

- [ ] **Step 4: Run to verify pass**
Run: `pytest tests/unit/test_ats_calibrator.py -k to_target -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: harvest_ats + target aggregation"
```

---

### Task 3: Calibrator (fit/predict) with a fittable test

**Files:**
- Create: `src/rho/ats/calibrator.py`; Modify: `src/rho/ats/__init__.py` (export)
- Test: `tests/unit/test_ats_calibrator.py` (add)

**Interfaces:**
- Produces: `Calibrator` with `fit(X: list[ComponentVector], y: list[float]) -> None`, `predict(cv: ComponentVector) -> float` (clamped 0–100), `save(path)`, `load(path)`. Feature order fixed: `[keyword_coverage, semantic_similarity, fuzzy_coverage, must_have_coverage, nice_have_coverage]`.

- [ ] **Step 1: Add dep**
Add `scikit-learn>=1.4`, `scipy>=1.11`, `joblib` to `pyproject.toml`; install.

- [ ] **Step 2: Write failing test**
```python
# add to tests/unit/test_ats_calibrator.py
from rho.ats import Calibrator
from rho.models.scoring import ComponentVector
def _cv(a): return ComponentVector(keyword_coverage=a, semantic_similarity=a,
    fuzzy_coverage=a, must_have_coverage=a, nice_have_coverage=a)
def test_calibrator_learns_monotone_relationship():
    X = [_cv(v/10) for v in range(11)]
    y = [v*10 for v in range(11)]          # perfect linear target 0..100
    c = Calibrator(); c.fit(X, y)
    assert c.predict(_cv(0.0)) < c.predict(_cv(1.0))
    assert 0 <= c.predict(_cv(0.5)) <= 100
```

- [ ] **Step 3: Run to verify fail**
Run: `pytest tests/unit/test_ats_calibrator.py -k calibrator -v` → FAIL.

- [ ] **Step 4: Implement**
```python
# src/rho/ats/calibrator.py
import numpy as np, joblib
from sklearn.linear_model import Ridge
from rho.models.scoring import ComponentVector
FEATURES = ["keyword_coverage","semantic_similarity","fuzzy_coverage",
            "must_have_coverage","nice_have_coverage"]
def _vec(cv: ComponentVector) -> list[float]:
    return [getattr(cv, f) for f in FEATURES]
class Calibrator:
    def __init__(self):
        self.model = Ridge(alpha=1.0)
        self._fitted = False
    def fit(self, X: list[ComponentVector], y: list[float]) -> None:
        self.model.fit(np.array([_vec(x) for x in X]), np.array(y))
        self._fitted = True
    def predict(self, cv: ComponentVector) -> float:
        if not self._fitted:
            raise RuntimeError("calibrator not fitted")
        p = float(self.model.predict([_vec(cv)])[0])
        return max(0.0, min(100.0, p))
    def save(self, path): joblib.dump(self.model, path); 
    def load(self, path):
        self.model = joblib.load(path); self._fitted = True; return self
```
Export in `__init__.py`: `from rho.ats.calibrator import Calibrator`.

- [ ] **Step 5: Run to verify pass**
Run: `pytest tests/unit/test_ats_calibrator.py -k calibrator -v` → PASS.

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat: Calibrator (Ridge) ComponentVector->score"
```

---

### Task 4: Dataset builder + fit script + held-out metrics (the C2 result)

**Files:**
- Create: `src/rho/ats/dataset.py`, `eval/fit_calibrator.py`
- Test: `tests/unit/test_ats_calibrator.py` (add — dataset builder with fakes)

**Interfaces:**
- Produces: `build_calibration_dataset(pairs, harvest_fn, feature_fn) -> tuple[list[ComponentVector], list[float]]` where `pairs = [(file_bytes, filename, jd_text), ...]`, `feature_fn` runs ingest→extract→analyze_jd→match to get the `ComponentVector`, `harvest_fn` = `harvest_ats`. Docs with no engine score are skipped. `eval/fit_calibrator.py` builds dataset, does a train/held-out split, fits, and reports MAE + Spearman ρ, plus the cosine-baseline ablation.

- [ ] **Step 1: Write failing test (fakes, no engines/LLM)**
```python
# add to tests/unit/test_ats_calibrator.py
from rho.ats.dataset import build_calibration_dataset
def test_build_dataset_skips_scoreless_docs():
    pairs = [("a", "a.txt", "jd"), ("b", "b.txt", "jd")]
    def feat(fb, fn, jd): return _cv(0.5)
    def harvest(fb, fn, jd):
        return {"e1": {"match_score": 70.0}} if fb == "a" else {"e1": {"match_score": None}}
    X, y = build_calibration_dataset(pairs, harvest, feat)
    assert len(X) == 1 and y == [70.0]      # scoreless doc skipped
```

- [ ] **Step 2: Run to verify fail**
Run: `pytest tests/unit/test_ats_calibrator.py -k build_dataset -v` → FAIL.

- [ ] **Step 3: Implement dataset builder**
```python
# src/rho/ats/dataset.py
from rho.ats.aggregate import to_target
def build_calibration_dataset(pairs, harvest_fn, feature_fn):
    X, y = [], []
    for file_bytes, filename, jd_text in pairs:
        outs = harvest_fn(file_bytes, filename, jd_text)
        try:
            target = to_target(outs)
        except ValueError:
            continue                       # no engine score -> skip, don't impute
        X.append(feature_fn(file_bytes, filename, jd_text))
        y.append(target)
    return X, y
```

- [ ] **Step 4: Run to verify pass**
Run: `pytest tests/unit/test_ats_calibrator.py -k build_dataset -v` → PASS.

- [ ] **Step 5: Write fit + eval script**
```python
# eval/fit_calibrator.py
"""Build calibration dataset, fit, report held-out MAE + Spearman + cosine ablation."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from rho.ats import Calibrator, harvest_ats
from rho.ats.dataset import build_calibration_dataset
from rho.ingestion import ingest
from rho.extraction import extract
from rho.jd import analyze_jd
from rho.matching import match
def feature_fn(file_bytes, filename, jd_text):
    md, prov = ingest(file_bytes, filename)
    resume = extract(md, prov)
    reqs = analyze_jd(jd_text)
    return match(resume, reqs).component_vector
def main(pairs):
    X, y = build_calibration_dataset(pairs, harvest_ats, feature_fn)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    cal = Calibrator(); cal.fit(Xtr, ytr)
    preds = [cal.predict(x) for x in Xte]
    mae = float(np.mean(np.abs(np.array(preds) - np.array(yte))))
    rho = float(spearmanr(preds, yte).statistic)
    # ablation: raw cosine (semantic_similarity*100) vs calibrated
    cos = [x.semantic_similarity * 100 for x in Xte]
    cos_mae = float(np.mean(np.abs(np.array(cos) - np.array(yte))))
    print(f"calibrated MAE={mae:.2f} Spearman={rho:.3f} | cosine-baseline MAE={cos_mae:.2f}")
    cal.save("eval/calibrator.joblib")
    return {"mae": mae, "spearman": rho, "cosine_mae": cos_mae}
```
Feed `pairs` from your résumé×JD corpus (build in P7 or a small set now).

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat: calibration dataset builder + fit/eval script (C2)"
```

---

### Task 5: Wire calibrator into scoring path

**Files:**
- Modify: `src/rho/matching/__init__.py` (optional calibrator arg) OR keep at graph layer.
- Test: `tests/unit/test_matching.py` (add)

**Interfaces:**
- Produces: a way to fill `predicted_score`. Add `score_with_calibrator(match_result: MatchResult, calibrator: Calibrator) -> MatchResult` in `rho/ats/__init__.py` (keeps matcher LLM/engine-free). Graph (P6) loads the fitted calibrator and calls it.

- [ ] **Step 1: Write failing test**
```python
# add to tests/unit/test_matching.py
from rho.ats import Calibrator, score_with_calibrator
from rho.models.scoring import MatchResult, ComponentVector
def test_score_with_calibrator_fills_predicted_score():
    cv = ComponentVector(keyword_coverage=1,semantic_similarity=1,fuzzy_coverage=1,
        must_have_coverage=1,nice_have_coverage=1)
    cal = Calibrator(); cal.fit([cv, ComponentVector(keyword_coverage=0,semantic_similarity=0,
        fuzzy_coverage=0,must_have_coverage=0,nice_have_coverage=0)], [100.0, 0.0])
    mr = MatchResult(component_vector=cv, predicted_score=0.0)
    mr2 = score_with_calibrator(mr, cal)
    assert mr2.predicted_score > 50
```

- [ ] **Step 2: Run to verify fail** → FAIL.
Run: `pytest tests/unit/test_matching.py -k score_with_calibrator -v`

- [ ] **Step 3: Implement**
```python
# src/rho/ats/__init__.py (add)
def score_with_calibrator(match_result, calibrator):
    mr = match_result.model_copy(deep=True)
    mr.predicted_score = calibrator.predict(mr.component_vector)
    return mr
```

- [ ] **Step 4: Run to verify pass** → PASS.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: apply calibrator to fill predicted_score"
```

---

## Self-Review
- [ ] Task 0 decision recorded; ≥2 engines OR parse-injection fallback adopted.
- [ ] Ground truth comes from real engine behavior, never a hand formula.
- [ ] `Calibrator` fits + predicts, clamped 0–100.
- [ ] `fit_calibrator.py` reports held-out MAE, Spearman, cosine-baseline ablation.
- [ ] Scoreless docs excluded, not imputed.

## Results (fill in — the C2 numbers the paper needs)
- Engines used: ___ (+ versions / commit hashes)
- Target definition: match-score aggregate / parse-injection recovery (circle one)
- Dataset size: ___ résumé×JD pairs (train ___ / held-out ___)
- **Calibrated MAE: ___  | Spearman ρ: ___  | Cosine-baseline MAE: ___**  ← headline C2 result
- Calibrator family (Ridge/other): ___
- Notes / deviations: ___
