from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
from rho.models.rewrite import FabricationReport, TailoredResume
from rho.models.scoring import Gap
from rho.rewrite.verifier import verify_against_source

__all__ = ["rewrite", "verify", "verify_against_source"]


def rewrite(resume: StructuredResume, gaps: list[Gap]) -> TailoredResume:
    raise NotImplementedError


def verify(tailored: StructuredResume, prov: ProvenanceMap) -> FabricationReport:
    """Frozen Section-6 signature, deliberately not callable.

    The gate must distinguish an *addition* from a reorder of values the source
    already claimed, which needs the source résumé — something this frozen
    signature cannot carry. `rewrite()` therefore calls `verify_against_source`
    directly with the source in hand. Failing loudly beats silently scoring
    every value as new and reporting a meaningless fabrication rate.
    """
    raise RuntimeError(
        "call verify_against_source(tailored, source, prov); wired in rewrite()"
    )
