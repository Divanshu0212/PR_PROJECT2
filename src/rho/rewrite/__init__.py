from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
from rho.models.rewrite import FabricationReport, TailoredResume
from rho.models.scoring import Gap


def rewrite(resume: StructuredResume, gaps: list[Gap]) -> TailoredResume:
    raise NotImplementedError


def verify(tailored: StructuredResume, prov: ProvenanceMap) -> FabricationReport:
    raise NotImplementedError
