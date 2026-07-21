"""Phase 5 (C3): hard-content tokens + the provenance verification gate."""

from rho.models.resume import Education, StructuredResume, WorkExperience
from rho.rewrite.tokens import hard_content_tokens


def test_hard_tokens_cover_skills_and_work():
    r = StructuredResume(
        name="A",
        skills=["Python", "AWS"],
        certifications=["AWS SAA"],
        work=[
            WorkExperience(
                company="Acme", title="Engineer", start_date="2019", end_date="2022"
            )
        ],
    )
    toks = hard_content_tokens(r)
    values = {t[0] for t in toks}
    assert {"Python", "AWS", "AWS SAA", "Acme", "Engineer"} <= values


def test_hard_tokens_cover_education_and_dates():
    r = StructuredResume(
        name="A",
        work=[
            WorkExperience(
                company="Acme", title="Engineer", start_date="2019", end_date="2022"
            )
        ],
        education=[Education(institution="MIT")],
    )
    values = {t[0] for t in hard_content_tokens(r)}
    assert {"2019", "2022", "MIT"} <= values


def test_hard_tokens_skip_blank_values():
    r = StructuredResume(name="A", skills=["Python", "", "   "])
    values = [t[0] for t in hard_content_tokens(r)]
    assert values == ["Python"]


def test_hard_tokens_field_paths_are_addressable():
    r = StructuredResume(
        name="A",
        skills=["Python"],
        work=[WorkExperience(company="Acme", title="Engineer")],
    )
    paths = {t[0]: t[1] for t in hard_content_tokens(r)}
    assert paths["Python"] == "skills[0]"
    assert paths["Acme"] == "work[0].company"
    assert paths["Engineer"] == "work[0].title"
