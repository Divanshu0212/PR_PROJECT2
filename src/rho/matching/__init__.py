from rho.models.jd import RequirementSet
from rho.models.resume import StructuredResume
from rho.models.scoring import MatchResult


def match(resume: StructuredResume, reqs: RequirementSet) -> MatchResult:
    """fills component_vector + gaps; predicted_score left 0.0 until P4"""
    raise NotImplementedError
