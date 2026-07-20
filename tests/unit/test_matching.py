from rho.matching import match
from rho.models.jd import RequirementSet, Requirement
from rho.models.resume import StructuredResume
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


def test_match_builds_vector_and_prov_gaps():
    resume = StructuredResume(
        name="A", skills=["Python", "AWS"], skills_prov=[["p:d:1"], ["p:d:2"]]
    )
    reqs = RequirementSet(
        requirements=[
            Requirement(text="Python", kind="skill", priority="must"),
            Requirement(text="Kubernetes", kind="skill", priority="must"),
        ]
    )
    mr = match(resume, reqs)
    assert mr.predicted_score == 0.0
    assert 0.0 <= mr.component_vector.must_have_coverage <= 1.0
    py_gap = next(g for g in mr.gaps if g.requirement.text == "Python")
    assert py_gap.status == "present"
    assert py_gap.evidence_prov == ["p:d:1"]  # provenance chain preserved
    k8s_gap = next(g for g in mr.gaps if g.requirement.text == "Kubernetes")
    assert k8s_gap.status == "absent"
    assert k8s_gap.evidence_prov == []
