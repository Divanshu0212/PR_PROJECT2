"""Gate ON vs OFF fabrication comparison (C3 headline).

Each benchmark pair is a résumé whose JD demands skills the résumé does not
have — maximum pressure to invent. The rewriter runs once per pair; we then
count how many unsourced hard-content values would *ship* under each condition:

  gate OFF — the raw rewriter output, prompt-grounding only.
  gate ON  — the same output after `verify_against_source` strips rejections.

Gate-ON is 0 by construction, and that is exactly the claim: the guarantee is
structural, not a model behaviour that happens to hold on this sample. The
number that carries information is gate-OFF — how often a grounded prompt alone
would have shipped a fabrication.

Usage: python -m eval.fabrication_ablation [--limit N] [--out results.json]
"""

import argparse
import json
import time
from pathlib import Path

from rho.ingestion import ingest
from rho.models.jd import Requirement
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
from rho.models.scoring import Gap
from rho.rewrite.llm import rewrite_schema
from rho.rewrite.verifier import verify_against_source

PAIRS_PATH = Path(__file__).parent.parent / "tests/fixtures/fabrication/pairs.json"


def load_pairs(path: Path = PAIRS_PATH) -> list[dict]:
    """Each pair carries a résumé parsed into (resume, prov) via real ingestion.

    Provenance comes from the real ingest path rather than a hand-built map, so
    the gate is exercised against the same span shapes it sees in production.
    """
    raw = json.loads(path.read_text())
    pairs = []
    for item in raw:
        text = item["resume"]
        _, prov = ingest(text.encode(), f"{item['id']}.txt")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        resume = StructuredResume(
            name=lines[0],
            headline=lines[1] if len(lines) > 1 else None,
            skills=list(item["source_skills"]),
        )
        # The absent requirements ARE the adversarial pressure: naming them as
        # targets is what tempts the model to invent. Passing gaps=[] would test
        # the rewriter in an easy condition the benchmark was built to avoid.
        gaps = [
            Gap(
                requirement=Requirement(text=t, kind="skill", priority="must"),
                status="absent",
            )
            for t in item["tempting_absent"]
        ]
        pairs.append(
            {
                "id": item["id"],
                "resume": resume,
                "prov": prov,
                "jd": item["jd"],
                "gaps": gaps,
                "tempting_absent": item["tempting_absent"],
            }
        )
    return pairs


def unsourced_count(resume: StructuredResume, source: StructuredResume, prov: ProvenanceMap) -> int:
    """How many hard-content additions in `resume` lack supporting provenance."""
    _, rep = verify_against_source(resume, source, prov)
    return rep.total_edits - rep.verified_edits


def run(pairs: list[dict], verbose: bool = True) -> dict:
    off_total = on_total = 0
    rates: list[float] = []
    per_pair = []

    for i, pair in enumerate(pairs, 1):
        source, prov = pair["resume"], pair["prov"]
        started = time.monotonic()
        try:
            raw = rewrite_schema(source, pair["gaps"])  # gate OFF: ship as generated
        except Exception as exc:  # a dead model must not look like a clean run
            print(f"  [{i}/{len(pairs)}] {pair['id']}: FAILED ({exc})")
            per_pair.append({"id": pair["id"], "error": str(exc)})
            continue

        off = unsourced_count(raw, source, prov)
        fixed, rep = verify_against_source(raw, source, prov)  # gate ON
        on = unsourced_count(fixed, source, prov)  # what survives the gate

        off_total += off
        on_total += on
        rates.append(rep.fabrication_rate)
        per_pair.append(
            {
                "id": pair["id"],
                "unsourced_off": off,
                "unsourced_on": on,
                "total_edits": rep.total_edits,
                "fabrication_rate": rep.fabrication_rate,
                "rejected": [r.added_text for r in rep.rejected_edits],
            }
        )
        if verbose:
            print(
                f"  [{i}/{len(pairs)}] {pair['id']}: "
                f"OFF={off} ON={on} rate={rep.fabrication_rate:.2f} "
                f"({time.monotonic() - started:.0f}s)"
            )

    mean_rate = sum(rates) / len(rates) if rates else 0.0
    summary = {
        "pairs": len(pairs),
        "pairs_scored": len(rates),
        "unsourced_shipped_gate_off": off_total,
        "unsourced_shipped_gate_on": on_total,
        "mean_fabrication_rate": mean_rate,
        "per_pair": per_pair,
    }
    print(
        f"\nunsourced additions shipped  gate-OFF={off_total}  gate-ON={on_total}"
        f"\nmean fabrication_rate = {mean_rate:.3f}  over {len(rates)} pairs"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "fabrication_results.json")
    args = ap.parse_args()

    pairs = load_pairs()
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"running fabrication ablation on {len(pairs)} pairs...")
    summary = run(pairs)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
