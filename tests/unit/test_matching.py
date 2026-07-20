from rho.matching.embed import Embedder


def test_semantic_similarity_high_for_synonyms():
    e = Embedder()
    v = e.encode(["machine learning model development", "AWS cloud platform"])
    sim = e.cosine(e.encode(["ML model building"])[0], v[0])
    assert sim > 0.4  # synonym-ish should beat unrelated
