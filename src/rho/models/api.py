from pydantic import BaseModel

from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
from rho.models.rewrite import TailoredResume
from rho.models.scoring import MatchResult


class OptimizeRequest(BaseModel):
    jd_text: str
    # file arrives as multipart upload, not in this model


class PipelineResponse(BaseModel):
    structured_resume: StructuredResume
    provenance_map: ProvenanceMap
    match_result: MatchResult
    tailored_resume: TailoredResume
    final_score: float
