from rho.extraction.provenance_attach import attach_provenance
from rho.extraction.schema import to_structured
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume


def extract(markdown: str, prov: ProvenanceMap, _schema_fn=None) -> StructuredResume:
    """markdown+prov -> StructuredResume with *_prov filled"""
    if _schema_fn is None:
        from rho.extraction.llm import extract_schema as _schema_fn_default

        _schema_fn = _schema_fn_default
    es = _schema_fn(markdown)  # ExtractionSchema (validated by Pydantic already)
    resume = to_structured(es)
    return attach_provenance(resume, prov)
