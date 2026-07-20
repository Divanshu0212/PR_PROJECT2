import os
import tempfile

from rho.models.provenance import ProvenanceMap, SourceSpan


def ingest_docling(file_bytes: bytes, filename: str, doc_id: str) -> tuple[str, ProvenanceMap]:
    """PDF/DOCX/image -> (markdown, ProvenanceMap) via Docling.

    Char offsets index into the exported markdown; page/bbox come from the
    Docling item geometry when present.
    """
    from docling.document_converter import DocumentConverter

    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        result = DocumentConverter().convert(path)
        doc = result.document
        md = doc.export_to_markdown()
    finally:
        os.unlink(path)

    pm = ProvenanceMap(doc_id=doc_id)
    # Locate each text item's string in the exported markdown to get char offsets.
    # search_from advances so repeated strings map to successive occurrences
    # rather than all collapsing onto the first.
    search_from = 0
    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text or not text.strip():
            continue
        content = text.strip()
        idx = md.find(content, search_from)
        if idx == -1:
            idx = md.find(content)
            if idx == -1:
                continue
        else:
            search_from = idx + len(content)
        page = None
        bbox = None
        prov = getattr(item, "prov", None)
        if prov:
            p0 = prov[0]
            page = getattr(p0, "page_no", None)
            bb = getattr(p0, "bbox", None)
            if bb is not None:
                bbox = (bb.l, bb.t, bb.r, bb.b)
        pm.add(
            SourceSpan(
                doc_id=doc_id,
                char_start=idx,
                char_end=idx + len(content),
                page=page,
                bbox=bbox,
                raw_text=content,
            )
        )
    return md, pm
