from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "rho"
    extraction_model: str = "Qwen/Qwen3-0.6B"  # override in P2
    temperature: float = 0.2


settings = Settings()
