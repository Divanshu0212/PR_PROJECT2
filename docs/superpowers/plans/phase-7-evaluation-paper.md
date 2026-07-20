# Phase 7 — Evaluation Harness & Paper Tables — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.
> **Read `00-SHARED-CONTEXT.md` first.** Confirm Phases 0–6 done. This phase produces every number/figure the paper reports.

**Goal:** Assemble the three datasets (gold extraction, ATS calibration, fabrication benchmark), run all metrics and all ablations, and emit reproducible result tables for the paper covering C1, C2, C3.

**Architecture:** Pure evaluation + reporting layer over the built pipeline. Metric functions are unit-tested on tiny inputs; dataset runners produce JSON/CSV tables + a `RESULTS.md` summary. Ablations toggle one component each.

**Tech Stack:** pytest, numpy, scipy, pandas (tables), the built `rho` pipeline.

## Global Constraints
- Every headline number must be reproducible from a script in `eval/` given the datasets.
- Metric functions are deterministic + unit-tested.
- Report per-field F1 with **long-text fields tracked separately** (report finding: long-text is the hardest).
- Ablations required: (A) provenance chain on/off effect on fabrication; (B) calibrated vs cosine score (from P4); (C) rewrite gate on/off (from P5).

## This phase consumes
- Full pipeline (Phases 1–6); P4 `fit_calibrator.py`; P5 `fabrication_ablation.py`.

## This phase produces
- `eval/metrics.py` — F1/precision/recall with entity alignment; long-text scoring.
- `eval/datasets/` — gold set, calibration pairs, fabrication pairs (+ loaders).
- `eval/run_all.py` — runs everything, writes `eval/RESULTS.md` + CSVs.
- Final `eval/RESULTS.md` — the paper's numbers.

---

## File Structure
- Create: `eval/metrics.py`, `eval/datasets/__init__.py` (+ loaders), `eval/run_all.py`.
- Create: `tests/unit/test_metrics.py`.
- Create: `eval/RESULTS.md` (generated).

---

### Task 1: Field-level F1 with entity alignment

**Files:**
- Create: `eval/metrics.py`
- Test: `tests/unit/test_metrics.py`

**Interfaces:**
- Produces: `field_f1(pred: dict, gold: dict, field: str) -> dict` returning `{precision, recall, f1}` for a list-valued field via set/aligned comparison; `long_text_f1(pred: str, gold: str) -> float` (token-overlap F1); `provenance_accuracy(resume, gold_prov_map) -> float` (fraction of fields whose attached `prov_id` points to the correct source span).

- [ ] **Step 1: Write failing test**
```python
# tests/unit/test_metrics.py
from eval.metrics import field_f1, long_text_f1
def test_field_f1_on_skills():
    m = field_f1({"skills":["python","aws","sql"]}, {"skills":["python","aws","gcp"]}, "skills")
    assert round(m["precision"],2) == 0.67 and round(m["recall"],2) == 0.67
    assert round(m["f1"],2) == 0.67
def test_long_text_f1_token_overlap():
    f = long_text_f1("built scalable python api", "built python api service")
    assert 0.0 < f < 1.0
```

- [ ] **Step 2: Run to verify fail** → FAIL.
Run: `pytest tests/unit/test_metrics.py -v`

- [ ] **Step 3: Implement**
```python
# eval/metrics.py
def _prf(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gold_set) if gold_set else 0.0
    f = 2*p*r/(p+r) if (p+r) else 0.0
    return {"precision": p, "recall": r, "f1": f}
def field_f1(pred: dict, gold: dict, field: str) -> dict:
    ps = {str(x).lower() for x in pred.get(field, [])}
    gs = {str(x).lower() for x in gold.get(field, [])}
    return _prf(ps, gs)
def long_text_f1(pred: str, gold: str) -> float:
    return _prf(set(pred.lower().split()), set(gold.lower().split()))["f1"]
def provenance_accuracy(resume, gold_prov: dict) -> float:
    """gold_prov: {field_value: correct_prov_id}. Fraction of attached prov matching gold."""
    from rho.rewrite.tokens import hard_content_tokens
    total = correct = 0
    # caller supplies a resume whose *_prov are populated; compare first prov_id to gold
    # (implement per your gold format; keep deterministic)
    return correct / total if total else 0.0
```
*(Task note: finalize `provenance_accuracy` against the exact gold-span format you choose in Task 2. Keep it deterministic and unit-test it once the format exists.)*

- [ ] **Step 4: Run to verify pass** → PASS.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: evaluation metrics (field F1, long-text F1)"
```

---

### Task 2: Datasets — gold extraction, calibration, fabrication

**Files:**
- Create: `eval/datasets/__init__.py`, `eval/datasets/gold/`, `eval/datasets/calibration/`, `eval/datasets/fabrication/`
- Test: none (data + loaders)

**Interfaces:**
- Produces: loaders `load_gold() -> list[(file_path, gold_json)]`, `load_calibration_pairs() -> list[(file_bytes, filename, jd_text)]`, `load_fabrication_pairs() -> list[(resume, gaps, prov)]`.

- [ ] **Step 1: Assemble gold extraction set** (100–300 résumés across formats; hand-label fields + provenance spans). Sources per report §1.4: Kaggle Resume NER set, HF `yashpwr/resume-ner-training-data`, LiveCareer/Jiechieu; or synthetic (template + substituted content). Store `gold/<id>.pdf` + `gold/<id>.json` (fields + gold prov spans).

- [ ] **Step 2: Assemble calibration pairs** (résumé × JD pairs to run through the ATS harness, P4). Reuse gold résumés × a set of JDs.

- [ ] **Step 3: Assemble fabrication benchmark** (from P5 Task 5; move canonical copy here).

- [ ] **Step 4: Write loaders** — deterministic, path-based, return the shapes above.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "data: gold/calibration/fabrication datasets + loaders"
```

---

### Task 3: `run_all.py` — every table + RESULTS.md

**Files:**
- Create: `eval/run_all.py`
- Output: `eval/RESULTS.md`, CSVs

**Interfaces:**
- Produces: a single script computing and writing:
  - **Table 1 (C1 extraction):** per-field P/R/F1, long-text F1 separately, provenance-attachment accuracy.
  - **Table 2 (C2 calibration):** calibrated MAE, Spearman ρ vs real engines, cosine-baseline MAE (from `fit_calibrator.py`).
  - **Table 3 (C3 fabrication):** unsourced additions gate-OFF vs gate-ON, mean fabrication_rate.
  - **Table 4 (ablations):** provenance on/off, calibrated vs cosine, gate on/off — one row each.
  - Latency + cost-per-successful-task line.

- [ ] **Step 1: Implement runner**
```python
# eval/run_all.py
import json
from eval.metrics import field_f1, long_text_f1
from eval.datasets import load_gold
from rho.ingestion import ingest
from rho.extraction import extract
def eval_extraction():
    rows = []
    for path, gold in load_gold():
        md, prov = ingest(open(path,"rb").read(), path)
        pred = extract(md, prov).model_dump()
        rows.append({
            "id": path,
            "skills_f1": field_f1(pred, gold, "skills")["f1"],
            "summary_longtext_f1": long_text_f1(pred.get("summary") or "", gold.get("summary") or ""),
        })
    return rows
def main():
    ext = eval_extraction()
    # C2 + C3 pulled from eval/fit_calibrator.py and eval/fabrication_ablation.py results
    with open("eval/RESULTS.md", "w") as f:
        f.write("# Results\n\n## Table 1 — Extraction (C1)\n")
        avg = sum(r["skills_f1"] for r in ext)/max(len(ext),1)
        f.write(f"- mean skills F1: {avg:.3f}\n")
        # append C2/C3 tables from their scripts' saved outputs
    print("wrote eval/RESULTS.md")
if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on real datasets** (needs LLM + calibrator fitted). Record numbers.
Run: `RHO_LLM_ENABLED=1 python eval/run_all.py`

- [ ] **Step 3: Commit**
```bash
git add -A && git commit -m "feat: run_all eval harness -> RESULTS.md"
```

---

### Task 4: Ablation runner

**Files:**
- Modify: `eval/run_all.py` (add ablation section) or `eval/ablations.py`

**Interfaces:**
- Produces: Table 4 rows — (A) fabrication rate with provenance chain vs a no-provenance rewrite; (B) calibrated vs cosine MAE (reuse P4); (C) gate on/off shipped-unsourced (reuse P5). Each ablation toggles exactly one component.

- [ ] **Step 1: Implement** the three ablation calls, writing rows into `RESULTS.md`.
- [ ] **Step 2: Run + record.**
- [ ] **Step 3: Commit**
```bash
git add -A && git commit -m "feat: ablation runner (provenance/calibration/gate)"
```

---

## Self-Review
- [ ] Metric functions unit-tested.
- [ ] All three datasets assembled + loadable.
- [ ] `run_all.py` writes Tables 1–4 + latency/cost to `RESULTS.md`.
- [ ] Long-text fields reported separately from named entities.
- [ ] Every headline number reproducible from an `eval/` script.

## Results (fill in — final paper numbers)
- **C1:** mean field F1 ___, long-text F1 ___, provenance-attachment accuracy ___
- **C2:** calibrated MAE ___ / Spearman ___ vs cosine MAE ___
- **C3:** unsourced shipped gate-OFF ___ vs gate-ON ___ ; mean fabrication_rate ___
- **Ablations:** provenance on/off ___ ; calibrated/cosine ___ ; gate on/off ___
- Latency: ___ ; cost-per-successful-task: ___
- Dataset sizes: gold ___ / calibration ___ / fabrication ___
