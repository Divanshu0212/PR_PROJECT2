from pathlib import Path

from rho.ingestion import ingest

FIX = Path(__file__).parent.parent / "fixtures"


def test_text_ingest_offsets_map_back():
    data = (FIX / "clean.txt").read_bytes()
    md, pm = ingest(data, "clean.txt")
    assert "Alice Johnson" in md
    assert len(pm.spans) >= 4
    # every span's raw_text equals the markdown slice at its offsets
    for span in pm.spans.values():
        assert md[span.char_start:span.char_end] == span.raw_text
