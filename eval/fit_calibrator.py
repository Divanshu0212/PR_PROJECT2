"""Build the calibration dataset, fit, and report held-out metrics (C2).

Reports calibrated MAE + Spearman ρ against the cosine-similarity baseline,
which is the ablation the paper needs: does calibrating against real engine
output beat using raw cosine as the score?
"""

import argparse
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from eval.corpus import build_pairs
from rho.ats import Calibrator, harvest_ats
from rho.ats.dataset import build_calibration_dataset
from rho.jd import analyze_jd
from rho.jd.schema import JDSchema, ReqItem
from rho.matching import match
from rho.matching.coverage import extract_jd_terms

# Deterministic JD analysis: KeyBERT terms instead of the vLLM path, so the
# whole calibration run is reproducible and needs no GPU. Priority is a
# position heuristic — KeyBERT ranks by relevance, so the top third are the
# terms the JD leans on hardest.
_MUST_FRACTION = 3


def _keybert_schema_fn(jd_text: str) -> JDSchema:
    terms = extract_jd_terms(jd_text, top_n=15)
    cutoff = max(1, len(terms) // _MUST_FRACTION)
    return JDSchema(
        reasoning="deterministic KeyBERT extraction (no LLM)",
        title=None,
        requirements=[
            ReqItem(text=t, kind="skill", priority="must" if i < cutoff else "nice")
            for i, t in enumerate(terms)
        ],
    )


def feature_fn(resume, jd_text):
    """resume+jd -> ComponentVector (rho's own raw signals, pre-calibration)."""
    return match(resume, analyze_jd(jd_text, _schema_fn=_keybert_schema_fn)).component_vector


def main(n_pairs: int = 200, seed: int = 0, out: str = "eval/calibrator.joblib") -> dict:
    pairs = build_pairs(n_pairs=n_pairs, seed=seed)
    X, y = build_calibration_dataset(pairs, harvest_ats, feature_fn)
    if len(X) < 10:
        raise SystemExit(f"only {len(X)} usable pairs; need more data to fit")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    cal = Calibrator()
    cal.fit(Xtr, ytr)

    preds = np.array([cal.predict(x) for x in Xte])
    yte_arr = np.array(yte)
    mae = float(np.mean(np.abs(preds - yte_arr)))
    rho = float(spearmanr(preds, yte_arr).statistic)

    # Ablation: raw cosine (semantic_similarity * 100) as the score.
    cos = np.array([x.semantic_similarity * 100 for x in Xte])
    cos_mae = float(np.mean(np.abs(cos - yte_arr)))
    cos_rho = float(spearmanr(cos, yte_arr).statistic)

    cal.save(out)
    metrics = {
        "n_pairs_requested": n_pairs,
        "n_usable": len(X),
        "n_train": len(Xtr),
        "n_heldout": len(Xte),
        "mae": mae,
        "spearman": rho,
        "cosine_mae": cos_mae,
        "cosine_spearman": cos_rho,
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
    }
    print(json.dumps(metrics, indent=2))
    print(
        f"\ncalibrated MAE={mae:.2f} Spearman={rho:.3f} | "
        f"cosine-baseline MAE={cos_mae:.2f} Spearman={cos_rho:.3f}"
    )
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-pairs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="eval/calibrator.joblib")
    a = p.parse_args()
    main(n_pairs=a.n_pairs, seed=a.seed, out=a.out)
