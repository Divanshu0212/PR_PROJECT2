import pytest


def test_stubs_raise_not_implemented():
    from rho.jd import analyze_jd
    from rho.matching import match
    from rho.ats import harvest_ats, Calibrator
    from rho.rewrite import rewrite, verify
    from rho.graph import run_pipeline

    # Implemented in earlier phases, covered by their own tests:
    #   rho.ingestion.ingest  (P1) -> test_ingestion.py
    #   rho.extraction.extract (P2) -> test_extraction_provenance.py
    with pytest.raises(NotImplementedError):
        analyze_jd("")
