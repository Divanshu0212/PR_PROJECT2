"""Build (ComponentVector, y) pairs for calibration.

Docs are dropped — never imputed — when no engine scores them or when
featurisation fails, so the fit only sees real observations.
"""

import logging

from rho.ats.aggregate import to_target

logger = logging.getLogger(__name__)


def build_calibration_dataset(pairs, harvest_fn, feature_fn, target_fn=to_target):
    """pairs: [(resume, jd_text), ...] -> (X, y).

    `target_fn` selects the target definition — `to_target` for the engines'
    composite score, `to_match_target` for the JD-dependent dimensions only.
    """
    X, y = [], []
    for resume, jd_text in pairs:
        outs = harvest_fn(resume, jd_text)
        try:
            target = target_fn(outs)
        except ValueError:
            logger.info("no engine score; skipping doc")
            continue
        try:
            features = feature_fn(resume, jd_text)
        except Exception as exc:
            logger.warning("featurisation failed; skipping doc: %s", exc)
            continue
        X.append(features)
        y.append(target)
    return X, y
