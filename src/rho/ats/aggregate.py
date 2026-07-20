"""Aggregate per-engine outputs into the single calibration target `y`.

This is the one documented place where engine outputs collapse to a number, so
the paper can state the target definition precisely.
"""


def to_target(engine_outputs: dict) -> float:
    """-> y in 0..100: mean of the engines that produced a score.

    Raises ValueError when no engine scored, so the caller skips the doc
    rather than imputing a value.
    """
    scores = [
        o["match_score"] for o in engine_outputs.values() if o.get("match_score") is not None
    ]
    if not scores:
        raise ValueError("no engine produced a score; exclude this doc from fit")
    return sum(scores) / len(scores)
