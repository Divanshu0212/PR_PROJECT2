# Phase 5 — Verified Rewriter (Core Novelty C3) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.
> **Read `00-SHARED-CONTEXT.md` first.** Confirm Phases 0–4 done. **This is the second core novelty (C3).**

**Goal:** Rewrite/tailor the résumé toward the JD gaps, then pass it through a **provenance verification gate**: every hard-content token (skill, tool, org, number, date) in the rewritten output must resolve to a source `prov_id` whose `raw_text` supports it — unresolved additions are **rejected** (reverted), logged, and counted. Emit a `FabricationReport` with a `fabrication_rate`. Build a fabrication benchmark and run the gate-on / gate-off ablation.

**Architecture:** Rewriter = grounded LLM prompt (master résumé = single source of truth). It may reorder/rephrase/select/emphasize, never invent. After generation, the **verifier** diffs the tailored résumé against the source: it extracts hard-content tokens from the tailored text, and for each *newly-introduced* token checks for a supporting `prov_id` (reusing Phase 2's `find_prov`). Unsupported additions are rejected; the offending edit is reverted to the source value. The gate is deterministic and LLM-free — fully unit-testable.

**Tech Stack:** Outlines (rewriter, constrained), the Phase-2 `find_prov` provenance matcher, RapidFuzz.

## Global Constraints
- Implement `rho.rewrite.rewrite(resume, gaps) -> TailoredResume` and `rho.rewrite.verify(tailored, prov) -> FabricationReport`.
- **The gate is the contribution:** an added hard-content token with no supporting `prov_id` is ALWAYS rejected + reverted + counted. Never ship an unverified addition.
- `fabrication_rate = rejected_edits / total_edits` (total_edits = additions attempted).
- Rewriter temperature ≈ 0.6 (creative), but truthfulness enforced by the gate, not the prompt alone.

## This phase consumes
- `StructuredResume` (+ `*_prov`), `ProvenanceMap` (Phases 1–2).
- `Gap` list from `match()` (Phase 3) to know what to target.
- `find_prov` from `rho.extraction.provenance_attach` (Phase 2).

## This phase produces
- `rewrite()`, `verify()`; `rho.rewrite.tokens.hard_content_tokens(resume) -> list[tuple[str, path]]`.
- Fabrication benchmark under `tests/fixtures/fabrication/` + ablation script `eval/fabrication_ablation.py`.

---

## File Structure
- Create: `src/rho/rewrite/tokens.py` — hard-content token extraction from a StructuredResume.
- Create: `src/rho/rewrite/verifier.py` — the gate (`verify` + revert logic).
- Create: `src/rho/rewrite/llm.py` — grounded rewrite generation (Outlines).
- Modify: `src/rho/rewrite/__init__.py` — `rewrite()` orchestration + `verify` re-export.
- Create: `tests/unit/test_verifier.py`, `tests/integration/test_rewrite_llm.py`, `eval/fabrication_ablation.py`.

---

### Task 1: Hard-content token extraction

**Files:**
- Create: `src/rho/rewrite/tokens.py`
- Test: `tests/unit/test_verifier.py`

**Interfaces:**
- Produces: `hard_content_tokens(resume: StructuredResume) -> list[HardToken]` where `HardToken = (value: str, field_path: str)` — every skill, company, title, cert, date, and bullet-embedded tool/number that constitutes a factual claim. Start with structured fields (skills, companies, titles, certs, dates); bullets handled as whole-string claims in Task 2.

- [ ] **Step 1: Write failing test**
```python
# tests/unit/test_verifier.py
from rho.models.resume import StructuredResume, WorkExperience
from rho.rewrite.tokens import hard_content_tokens
def test_hard_tokens_cover_skills_and_work():
    r = StructuredResume(name="A", skills=["Python","AWS"],
        certifications=["AWS SAA"],
        work=[WorkExperience(company="Acme", title="Engineer",
              start_date="2019", end_date="2022")])
    toks = hard_content_tokens(r)
    values = {t[0] for t in toks}
    assert {"Python","AWS","AWS SAA","Acme","Engineer"} <= values
```

- [ ] **Step 2: Run to verify fail**
Run: `pytest tests/unit/test_verifier.py::test_hard_tokens_cover_skills_and_work -v` → FAIL.

- [ ] **Step 3: Implement**
```python
# src/rho/rewrite/tokens.py
from rho.models.resume import StructuredResume
HardToken = tuple[str, str]      # (value, field_path)
def hard_content_tokens(resume: StructuredResume) -> list[HardToken]:
    toks: list[HardToken] = []
    for i, s in enumerate(resume.skills):
        toks.append((s, f"skills[{i}]"))
    for i, c in enumerate(resume.certifications):
        toks.append((c, f"certifications[{i}]"))
    for wi, w in enumerate(resume.work):
        toks.append((w.company, f"work[{wi}].company"))
        toks.append((w.title, f"work[{wi}].title"))
        for d in (w.start_date, w.end_date):
            if d: toks.append((d, f"work[{wi}].date"))
    for ei, e in enumerate(resume.education):
        toks.append((e.institution, f"education[{ei}].institution"))
    return [(v, p) for (v, p) in toks if v and v.strip()]
```

- [ ] **Step 4: Run to verify pass** → PASS.
Run: `pytest tests/unit/test_verifier.py::test_hard_tokens_cover_skills_and_work -v`

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: hard-content token extraction for verification"
```

---

### Task 2: The verification gate (`verify`) — the C3 core

**Files:**
- Create: `src/rho/rewrite/verifier.py`; Modify: `src/rho/rewrite/__init__.py`
- Test: `tests/unit/test_verifier.py` (add)

**Interfaces:**
- Produces: `verify(tailored: StructuredResume, source: StructuredResume, prov: ProvenanceMap) -> tuple[StructuredResume, FabricationReport]`. Logic: for each hard token in `tailored` NOT already present in `source` (a new addition), check `find_prov(value, prov)`; if empty → **reject**: revert that field to source (or drop the added item), append `RejectedEdit`. `total_edits` = number of additions checked; `verified_edits` = additions that had provenance; `fabrication_rate` = rejected/total (0 if no additions).
- Re-export `verify` from `rho.rewrite` matching the frozen signature `verify(tailored, prov)`; provide source via closure in `rewrite()`.

- [ ] **Step 1: Write failing test**
```python
# add to tests/unit/test_verifier.py
from rho.models.provenance import SourceSpan, ProvenanceMap
from rho.rewrite.verifier import verify_against_source
def _prov():
    pm = ProvenanceMap(doc_id="d")
    pm.add(SourceSpan(doc_id="d", char_start=0, char_end=6, raw_text="Python"))
    return pm
def test_verify_rejects_unsupported_addition():
    source = StructuredResume(name="A", skills=["Python"])
    tailored = StructuredResume(name="A", skills=["Python","Kubernetes"])  # k8s not in source/prov
    fixed, report = verify_against_source(tailored, source, _prov())
    assert "Kubernetes" not in fixed.skills          # reverted
    assert report.total_edits == 1
    assert report.verified_edits == 0
    assert report.fabrication_rate == 1.0
    assert report.rejected_edits[0].added_text == "Kubernetes"
def test_verify_keeps_supported_addition():
    pm = _prov(); pm.add(SourceSpan(doc_id="d", char_start=7, char_end=13, raw_text="FastAPI"))
    source = StructuredResume(name="A", skills=["Python"])
    tailored = StructuredResume(name="A", skills=["Python","FastAPI"])
    fixed, report = verify_against_source(tailored, source, pm)
    assert "FastAPI" in fixed.skills
    assert report.verified_edits == 1
    assert report.fabrication_rate == 0.0
```

- [ ] **Step 2: Run to verify fail** → FAIL.
Run: `pytest tests/unit/test_verifier.py -k verify -v`

- [ ] **Step 3: Implement**
```python
# src/rho/rewrite/verifier.py
from rho.models.resume import StructuredResume
from rho.models.provenance import ProvenanceMap
from rho.models.rewrite import FabricationReport, RejectedEdit
from rho.rewrite.tokens import hard_content_tokens
from rho.extraction.provenance_attach import find_prov
def verify_against_source(tailored: StructuredResume, source: StructuredResume,
                          prov: ProvenanceMap):
    src_values = {v.lower() for v, _ in hard_content_tokens(source)}
    fixed = tailored.model_copy(deep=True)
    total = verified = 0
    rejected: list[RejectedEdit] = []
    # only skills list mutated here for clarity; extend to work/certs analogously
    kept_skills = []
    for s in fixed.skills:
        if s.lower() in src_values:
            kept_skills.append(s); continue
        total += 1
        if find_prov(s, prov):
            verified += 1; kept_skills.append(s)
        else:
            rejected.append(RejectedEdit(added_text=s, reason="no supporting prov_id"))
    fixed.skills = kept_skills
    report = FabricationReport(total_edits=total, verified_edits=verified,
        rejected_edits=rejected,
        fabrication_rate=(len(rejected)/total) if total else 0.0)
    return fixed, report
```
*(Task note: extend the same reject-if-no-prov loop to added certifications and to newly-introduced tools/numbers inside bullets. Skills shown as the canonical pattern; replicate for the other hard-content fields and cover each with a test.)*
Add to `src/rho/rewrite/__init__.py`:
```python
from rho.rewrite.verifier import verify_against_source
def verify(tailored, prov):        # frozen signature; source captured by rewrite()
    raise RuntimeError("call verify_against_source with the source resume; wired in rewrite()")
```

- [ ] **Step 4: Run to verify pass** → PASS both.
Run: `pytest tests/unit/test_verifier.py -k verify -v`

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: provenance verification gate + fabrication report (C3)"
```

---

### Task 3: Grounded rewrite generation (Outlines)

**Files:**
- Create: `src/rho/rewrite/llm.py`
- Test: `tests/integration/test_rewrite_llm.py` (skips without model)

**Interfaces:**
- Produces: `rewrite_schema(source: StructuredResume, gaps) -> StructuredResume`. Constrained to the same resume schema. Prompt: master résumé is the ONLY source of truth; reorder/rephrase/select/emphasize toward the gaps; never invent skills/tools/metrics/dates; if a gap can't be satisfied truthfully, leave it. Temperature 0.6.

- [ ] **Step 1: Skipping integration test**
```python
# tests/integration/test_rewrite_llm.py
import os, pytest
pytestmark = pytest.mark.skipif(os.getenv("RHO_LLM_ENABLED") != "1", reason="no LLM")
def test_rewrite_does_not_add_unsourced_skill():
    from rho.models.resume import StructuredResume
    from rho.rewrite.llm import rewrite_schema
    src = StructuredResume(name="A", skills=["Python"])
    out = rewrite_schema(src, gaps=[])
    # grounded prompt shouldn't invent; even if it does, gate catches it later
    assert "python" in [s.lower() for s in out.skills]
```

- [ ] **Step 2: Run to verify skip.**
Run: `pytest tests/integration/test_rewrite_llm.py -v` → SKIP.

- [ ] **Step 3: Implement** (mirror `extraction/llm.py`; constrained to a resume schema; grounding prompt; temperature 0.6). Note version deviations in Results.

- [ ] **Step 4: Commit**
```bash
git add -A && git commit -m "feat: grounded rewrite generation"
```

---

### Task 4: `rewrite()` orchestration (generate → gate)

**Files:**
- Modify: `src/rho/rewrite/__init__.py`
- Test: `tests/unit/test_verifier.py` (add — fake rewriter)

**Interfaces:**
- Produces: `rewrite(resume, gaps, prov, _rewrite_fn=None) -> TailoredResume`. Runs `_rewrite_fn(resume, gaps)` (LLM or fake) → `verify_against_source(tailored, resume, prov)` → assembles `TailoredResume(resume=fixed, fabrication_report=report)`. **Note:** the frozen shared-context signature is `rewrite(resume, gaps)`; extend it to accept `prov` (update shared-context Section 6 + P6 caller to pass `prov`). Record this signature change in shared context.

- [ ] **Step 1: Write failing test (fake rewriter injects a fabrication)**
```python
# add to tests/unit/test_verifier.py
from rho.rewrite import rewrite
def test_rewrite_gate_strips_fabrication():
    source = StructuredResume(name="A", skills=["Python"])
    fake = lambda r, g: StructuredResume(name="A", skills=["Python","GoLang"])  # GoLang invented
    tr = rewrite(source, [], _prov(), _rewrite_fn=fake)
    assert "GoLang" not in tr.resume.skills
    assert tr.fabrication_report.fabrication_rate == 1.0
```

- [ ] **Step 2: Run to verify fail** → FAIL.
Run: `pytest tests/unit/test_verifier.py::test_rewrite_gate_strips_fabrication -v`

- [ ] **Step 3: Implement**
```python
# src/rho/rewrite/__init__.py  (replace stub)
from rho.models.rewrite import TailoredResume
from rho.rewrite.verifier import verify_against_source
def rewrite(resume, gaps, prov, _rewrite_fn=None) -> TailoredResume:
    if _rewrite_fn is None:
        from rho.rewrite.llm import rewrite_schema as _rewrite_fn
    tailored = _rewrite_fn(resume, gaps)
    fixed, report = verify_against_source(tailored, resume, prov)
    return TailoredResume(resume=fixed, fabrication_report=report)
```
Update `00-SHARED-CONTEXT.md` Section 6 signature to `rewrite(resume, gaps, prov)`.

- [ ] **Step 4: Run to verify pass** → PASS.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: rewrite() orchestration with verification gate"
```

---

### Task 5: Fabrication benchmark + gate on/off ablation

**Files:**
- Create: `tests/fixtures/fabrication/` (curated résumé+JD pairs where keyword pressure tempts fabrication), `eval/fabrication_ablation.py`
- Test: none new (ablation is an eval script for the paper)

**Interfaces:**
- Produces: `eval/fabrication_ablation.py` running the real rewriter twice — gate ON (`verify_against_source` applied) vs gate OFF (raw rewrite) — reporting fabrication rate and count of unsourced additions in each. This is the headline C3 number.

- [ ] **Step 1: Build 10–30 adversarial pairs.** Each: a résumé + a JD demanding skills the résumé lacks (tempting the model to invent). Record ground-truth source skills.

- [ ] **Step 2: Write ablation script**
```python
# eval/fabrication_ablation.py
"""Gate ON vs OFF fabrication comparison (C3 headline)."""
from rho.rewrite.llm import rewrite_schema
from rho.rewrite.verifier import verify_against_source
from rho.rewrite.tokens import hard_content_tokens
def unsourced_count(resume, source, prov):
    _, rep = verify_against_source(resume, source, prov)
    return rep.total_edits - rep.verified_edits
def run(pairs):   # pairs: list[(source_resume, gaps, prov)]
    off_total = on_total = 0
    for source, gaps, prov in pairs:
        raw = rewrite_schema(source, gaps)                 # gate OFF
        off_total += unsourced_count(raw, source, prov)
        fixed, rep = verify_against_source(raw, source, prov)  # gate ON
        on_total += (rep.total_edits - rep.verified_edits) - len(rep.rejected_edits)
    print(f"unsourced additions shipped  gate-OFF={off_total}  gate-ON={on_total}")
```
(gate-ON shipped unsourced should be 0 by construction — that IS the claim.)

- [ ] **Step 3: Commit**
```bash
git add -A && git commit -m "feat: fabrication benchmark + gate ablation (C3)"
```

---

## Self-Review
- [ ] Gate rejects every unsourced hard-content addition (skills + certs + bullet tools).
- [ ] `fabrication_rate` computed correctly; gate-ON ships zero unsourced additions.
- [ ] `rewrite()` returns `TailoredResume` with intact provenance + report.
- [ ] Shared-context Section 6 updated for `rewrite(resume, gaps, prov)`.
- [ ] `pytest tests/unit/test_verifier.py -v` green.

## Results (fill in — C3 numbers)
- Fabrication benchmark size: ___ pairs
- **Unsourced additions shipped: gate-OFF ___ vs gate-ON ___ (must be 0)**  ← headline C3
- Mean fabrication_rate (gate detects): ___
- Rewrite quality note (did truthful tailoring still improve match?): ___
- LLM/Outlines deviations: ___
