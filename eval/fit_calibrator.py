"""Build the calibration dataset, fit, and report held-out metrics (C2).

Reports calibrated MAE + Spearman ρ against the cosine-similarity baseline,
which is the ablation the paper needs: does calibrating against real engine
output beat using raw cosine as the score?
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from eval.corpus import build_pairs
from rho.ats import Calibrator, harvest_ats
from rho.ats.aggregate import to_match_target
from rho.ats.dataset import build_calibration_dataset
from rho.jd import analyze_jd
from rho.jd.ollama import analyze_jd_schema as _ollama_schema_fn
from rho.matching import match

# JD analysis runs through the LLM path (Ollama, temperature 0). The KeyBERT
# fallback tried earlier extracted company names and boilerplate ("nashville
# office") rather than requirements, leaving keyword_coverage and
# fuzzy_coverage identically 0.0 across every pair.


def feature_fn(resume, jd_text):
    """resume+jd -> ComponentVector (rho's own raw signals, pre-calibration)."""
    return match(resume, analyze_jd(jd_text, _schema_fn=_ollama_schema_fn)).component_vector


PROGRESS_PATH = "eval/progress.json"


def _progress_writer(path: str, started: float):
    """Write run progress to `path` after every pair so a viewer can poll it."""

    def write(index: int, total: int, status: str, kept: int) -> None:
        elapsed = time.time() - started
        rate = elapsed / index if index else 0.0
        payload = {
            "index": index,
            "total": total,
            "kept": kept,
            "skipped": index - kept,
            "last_status": status,
            "elapsed_seconds": round(elapsed, 1),
            "seconds_per_pair": round(rate, 1),
            "eta_seconds": round(rate * (total - index), 1),
            "done": index >= total,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)  # atomic: a poller never reads a half-written file

    return write


def main(
    n_pairs: int = 200,
    seed: int = 0,
    out: str = "eval/calibrator.joblib",
    progress_path: str = PROGRESS_PATH,
) -> dict:
    pairs = build_pairs(n_pairs=n_pairs, seed=seed)
    X, y = build_calibration_dataset(
        pairs,
        harvest_ats,
        feature_fn,
        target_fn=to_match_target,
        on_progress=_progress_writer(progress_path, time.time()),
    )
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

    # Fold the results into the progress file so a viewer shows them on finish.
    try:
        with open(progress_path) as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        payload = {}
    payload.update({"done": True, "metrics": metrics})
    with open(progress_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-pairs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="eval/calibrator.joblib")
    a = p.parse_args()
    main(n_pairs=a.n_pairs, seed=a.seed, out=a.out)
