from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "rho"
    extraction_model: str = "Qwen/Qwen3-0.6B"  # override in P2
    temperature: float = 0.2
    # Matcher semantic bands (P3). Provisional defaults — never swept against a
    # labelled match set. Exposed here so P7 can tune them without code changes.
    sem_hi: float = 0.65  # >= this cosine counts a requirement "present"
    sem_lo: float = 0.45  # >= this (but < sem_hi) counts it "weak"


settings = Settings()
