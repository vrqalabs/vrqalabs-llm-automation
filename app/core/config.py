from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "VRQALabs LLM Automation"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = True
    openrouter_api_key: str = ""
    hf_api_key: str = ""
    groq_api_key: str = ""
    database_url: str = "sqlite:///./llm_automation.db"
    log_level: str = "INFO"
    huggingface_model: str = "google/gemma-2-2b-it"
    huggingface_base_url: str = "https://router.huggingface.co/hf-inference/models"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
