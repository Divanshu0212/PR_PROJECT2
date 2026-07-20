import pytest


def test_stubs_raise_not_implemented():
    from rho.extraction import extract
    from rho.jd import analyze_jd
    from rho.matching import match
    from rho.ats import harvest_ats, Calibrator
    from rho.rewrite import rewrite, verify
    from rho.graph import run_pipeline

    # rho.ingestion.ingest is implemented as of Phase 1 — covered by test_ingestion.py
    with pytest.raises(NotImplementedError):
        extract("", None)
