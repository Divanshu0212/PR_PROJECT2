from rho.extraction.schema import ExtractionSchema, to_structured


def test_schema_maps_to_structured_resume():
    es = ExtractionSchema(
        reasoning="found name+skill",
        name="Alice",
        skills=["python"],
        work=[
            {
                "company": "Acme",
                "title": "Eng",
                "start_date": "2019",
                "end_date": "2022",
                "bullets": ["Built API"],
            }
        ],
    )
    sr = to_structured(es)
    assert sr.name == "Alice"
    assert sr.skills == ["python"]
    assert sr.work[0].company == "Acme"
    # prov fields exist but empty until attach step
    assert sr.name_prov == []
