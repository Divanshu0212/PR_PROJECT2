from rho.models.provenance import SourceSpan, ProvenanceMap


def test_provmap_add_and_get():
    pm = ProvenanceMap(doc_id="d1")
    pid = pm.add(SourceSpan(doc_id="d1", char_start=0, char_end=5, raw_text="Alice"))
    assert pid == "p:d1:0"
    assert pm.get(pid).raw_text == "Alice"
    pid2 = pm.add(SourceSpan(doc_id="d1", char_start=6, char_end=9, raw_text="Bob"))
    assert pid2 == "p:d1:1"
