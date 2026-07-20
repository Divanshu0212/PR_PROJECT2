import pytest


def test_stubs_raise_not_implemented():
    from rho.ingestion import ingest
    from rho.extraction import extract
    from rho.jd import analyze_jd
    from rho.matching import match
    from rho.ats import harvest_ats, Calibrator
    from rho.rewrite import rewrite, verify
    from rho.graph import run_pipeline

    with pytest.raises(NotImplementedError):
        ingest(b"", "x.pdf")
