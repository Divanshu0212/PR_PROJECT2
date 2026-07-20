from rho.models.scoring import ComponentVector


def harvest_ats(file_bytes: bytes, filename: str, jd_text: str) -> dict:
    """run self-hostable ATS engines -> real parse+match labels"""
    raise NotImplementedError


class Calibrator:
    def fit(self, X: list[ComponentVector], y: list[float]) -> None:
        raise NotImplementedError

    def predict(self, cv: ComponentVector) -> float:
        """0..100"""
        raise NotImplementedError
