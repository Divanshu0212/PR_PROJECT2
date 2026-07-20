from rho.models.provenance import ProvenanceMap


def ingest(file_bytes: bytes, filename: str) -> tuple[str, ProvenanceMap]:
    """file -> (markdown, ProvenanceMap)"""
    raise NotImplementedError
