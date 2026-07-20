from rho.matching.coverage import keyword_coverage, fuzzy_coverage
from rho.matching.embed import Embedder


def test_semantic_similarity_high_for_synonyms():
    e = Embedder()
    v = e.encode(["machine learning model development", "AWS cloud platform"])
    sim = e.cosine(e.encode(["ML model building"])[0], v[0])
    assert sim > 0.4  # synonym-ish should beat unrelated


def test_keyword_and_fuzzy_coverage():
    reqs = ["Python", "Kubernetes", "AWS"]
    skills = ["python", "aws", "kubernets"]  # last is a typo
    assert keyword_coverage(reqs, skills) == 2 / 3  # Python, AWS exact; Kubernetes no
    assert fuzzy_coverage(reqs, skills) == 1.0  # typo caught by fuzzy
