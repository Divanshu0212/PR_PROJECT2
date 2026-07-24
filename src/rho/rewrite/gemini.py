"""Gemini-backed grounded rewriting (`gemini-3.6-flash`).

Same grounded prompt as `rho.rewrite.llm`/`rho.rewrite.groq`, same temperature
(0.6), different backend — via Gemini's `responseSchema` constrained decoding.

The prompt is still not the safety mechanism. Truthfulness is enforced downstream
by `rho.rewrite.verifier`, which is deterministic and LLM-free.
"""

from rho.extraction.schema import EduItem, ExtractionSchema, WorkItem, to_structured
from rho.llm.gemini import GeminiClient
from rho.models.resume import StructuredResume
from rho.models.scoring import Gap

_PROMPT = """You tailor a résumé toward a job's requirements. Rules:
- The MASTER RÉSUMÉ below is the ONLY source of truth.
- You MAY reorder, rephrase, select, and emphasize existing content.
- You MUST NOT invent skills, tools, employers, titles, metrics, dates, or
  certifications. If the résumé does not claim it, it does not go in.
- If a target requirement cannot be satisfied truthfully, leave it unsatisfied.
  An honest gap is the correct output; a fabricated match is a failure.
- Keep the person's name, employers, titles, and dates exactly as given.
- Fill the `reasoning` field first, briefly, then the data fields.
{gaps_block}
MASTER RÉSUMÉ (JSON):
---
{resume_json}
---"""

_GAPS_HEADER = """
TARGET REQUIREMENTS (emphasize existing evidence for these; never invent):
{gaps}
"""

_REWRITE_TEMPERATURE = 0.6

_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "reasoning": {"type": "STRING"},
        "name": {"type": "STRING"},
        "headline": {"type": "STRING", "nullable": True},
        "summary": {"type": "STRING", "nullable": True},
        "emails": {"type": "ARRAY", "items": {"type": "STRING"}},
        "phones": {"type": "ARRAY", "items": {"type": "STRING"}},
        "urls": {"type": "ARRAY", "items": {"type": "STRING"}},
        "work": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "company": {"type": "STRING"},
                    "title": {"type": "STRING"},
                    "start_date": {"type": "STRING", "nullable": True},
                    "end_date": {"type": "STRING", "nullable": True},
                    "bullets": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["company", "title"],
            },
        },
        "education": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "institution": {"type": "STRING"},
                    "degree": {"type": "STRING", "nullable": True},
                    "field": {"type": "STRING", "nullable": True},
                    "end_year": {"type": "STRING", "nullable": True},
                },
                "required": ["institution"],
            },
        },
        "skills": {"type": "ARRAY", "items": {"type": "STRING"}},
        "certifications": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["reasoning", "name", "work", "education", "skills"],
}

_client: GeminiClient | None = None


def _default_client() -> GeminiClient:
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client


def _source_json(resume: StructuredResume) -> str:
    """The résumé as the model sees it — values only, no prov chain noise."""
    return resume.model_dump_json(
        indent=2,
        exclude={
            "name_prov": True,
            "contact_prov": True,
            "skills_prov": True,
            "work": {
                "__all__": {
                    "company_prov": True,
                    "title_prov": True,
                    "date_prov": True,
                    "bullet_prov": True,
                }
            },
            "education": {"__all__": {"institution_prov": True, "edu_prov": True}},
        },
    )


def _gaps_block(gaps: list[Gap]) -> str:
    targets = [g.requirement.text for g in gaps if g.status != "present"]
    if not targets:
        return ""
    return _GAPS_HEADER.format(gaps="\n".join(f"- {t}" for t in targets))


def _coerce(data: dict) -> ExtractionSchema:
    """Validate item-by-item so one malformed entry cannot lose the whole résumé."""
    work = []
    for item in data.get("work") or []:
        try:
            work.append(WorkItem(**item))
        except Exception:
            continue  # No silent fills.
    education = []
    for item in data.get("education") or []:
        try:
            education.append(EduItem(**item))
        except Exception:
            continue
    as_str = lambda xs: [str(x) for x in (xs or []) if str(x).strip()]  # noqa: E731
    return ExtractionSchema(
        reasoning=data.get("reasoning") or "",
        name=str(data.get("name") or ""),
        headline=data.get("headline"),
        summary=data.get("summary"),
        emails=as_str(data.get("emails")),
        phones=as_str(data.get("phones")),
        urls=as_str(data.get("urls")),
        work=work,
        education=education,
        skills=as_str(data.get("skills")),
        certifications=as_str(data.get("certifications")),
    )


def rewrite_schema_gemini(
    resume: StructuredResume, gaps: list[Gap], client: GeminiClient | None = None
) -> StructuredResume:
    """Tailor `resume` toward `gaps` via Gemini. Output is UNVERIFIED — gate it."""
    client = client or _default_client()
    prompt = _PROMPT.format(
        gaps_block=_gaps_block(gaps), resume_json=_source_json(resume)
    )
    data = client.complete_json(
        prompt, response_schema=_SCHEMA, temperature=_REWRITE_TEMPERATURE
    )
    return to_structured(_coerce(data))
