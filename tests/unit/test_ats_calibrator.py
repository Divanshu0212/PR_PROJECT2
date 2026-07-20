import pytest

from rho.ats import Calibrator
from rho.ats.aggregate import to_target
from rho.models.scoring import ComponentVector


def _cv(a: float) -> ComponentVector:
    return ComponentVector(
        keyword_coverage=a,
        semantic_similarity=a,
        fuzzy_coverage=a,
        must_have_coverage=a,
        nice_have_coverage=a,
    )


def test_to_target_means_available_scores():
    outs = {"e1": {"match_score": 80.0}, "e2": {"match_score": 60.0}}
    assert to_target(outs) == 70.0


def test_to_target_skips_missing():
    outs = {"e1": {"match_score": 80.0}, "e2": {"match_score": None}}
    assert to_target(outs) == 80.0


def test_to_target_raises_when_no_engine_scored():
    """No score means exclude the doc from the fit — never impute."""
    with pytest.raises(ValueError):
        to_target({"e1": {"match_score": None}})


def test_calibrator_learns_monotone_relationship():
    X = [_cv(v / 10) for v in range(11)]
    y = [v * 10 for v in range(11)]
    c = Calibrator()
    c.fit(X, y)
    assert c.predict(_cv(0.0)) < c.predict(_cv(1.0))
    assert 0 <= c.predict(_cv(0.5)) <= 100


def test_calibrator_clamps_to_0_100():
    c = Calibrator()
    c.fit([_cv(0.4), _cv(0.5), _cv(0.6)], [0.0, 50.0, 100.0])
    assert c.predict(_cv(-5.0)) >= 0.0
    assert c.predict(_cv(5.0)) <= 100.0


def test_calibrator_raises_before_fit():
    with pytest.raises(RuntimeError):
        Calibrator().predict(_cv(0.5))


def test_build_dataset_skips_scoreless_docs():
    from rho.ats.dataset import build_calibration_dataset

    pairs = [("a", "jd"), ("b", "jd")]

    def feat(resume, jd):
        return _cv(0.5)

    def harvest(resume, jd):
        return {"e1": {"match_score": 70.0}} if resume == "a" else {"e1": {"match_score": None}}

    X, y = build_calibration_dataset(pairs, harvest, feat)
    assert len(X) == 1 and y == [70.0]


def test_build_dataset_skips_docs_whose_features_fail():
    """A doc that can't be featurised is dropped, not imputed."""
    from rho.ats.dataset import build_calibration_dataset

    def feat(resume, jd):
        if resume == "bad":
            raise ValueError("extraction failed")
        return _cv(0.5)

    def harvest(resume, jd):
        return {"e1": {"match_score": 70.0}}

    X, y = build_calibration_dataset([("bad", "jd"), ("ok", "jd")], harvest, feat)
    assert len(X) == 1 and y == [70.0]


def test_calibrator_roundtrips_through_save_load(tmp_path):
    c = Calibrator()
    c.fit([_cv(v / 10) for v in range(11)], [v * 10 for v in range(11)])
    path = tmp_path / "cal.joblib"
    c.save(path)
    assert Calibrator().load(path).predict(_cv(0.7)) == c.predict(_cv(0.7))
