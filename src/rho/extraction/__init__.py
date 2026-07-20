from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume


def extract(markdown: str, prov: ProvenanceMap) -> StructuredResume:
    """markdown+prov -> StructuredResume with *_prov filled"""
    raise NotImplementedError
