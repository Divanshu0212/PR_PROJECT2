from fastapi import FastAPI, Form, UploadFile

from rho.models.api import PipelineResponse
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
from rho.models.rewrite import FabricationReport, TailoredResume
from rho.models.scoring import ComponentVector, MatchResult

app = FastAPI(title="rho")


@app.get("/health")
def health():
    return {"status": "ok"}


def _placeholder_response() -> PipelineResponse:
    resume = StructuredResume(name="")
    cv = ComponentVector(
        keyword_coverage=0,
        semantic_similarity=0,
        fuzzy_coverage=0,
        must_have_coverage=0,
        nice_have_coverage=0,
    )
    return PipelineResponse(
        structured_resume=resume,
        provenance_map=ProvenanceMap(doc_id="d0"),
        match_result=MatchResult(component_vector=cv, predicted_score=0.0),
        tailored_resume=TailoredResume(
            resume=resume,
            fabrication_report=FabricationReport(
                total_edits=0, verified_edits=0, fabrication_rate=0.0
            ),
        ),
        final_score=0.0,
    )


@app.post("/optimize", response_model=PipelineResponse)
async def optimize(file: UploadFile, jd_text: str = Form(...)):
    _ = await file.read()  # consumed; real wiring in P6
    return _placeholder_response()
