from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # ==========================
    # Application
    # ==========================

    app_name: str

    app_version: str

    app_env: str

    debug: bool


    # ==========================
    # OpenRouter
    # ==========================

    openrouter_api_key: str

    openrouter_model: str

    openrouter_base_url: str


    # ==========================
    # HuggingFace
    # ==========================

    hf_api_key: str | None = None

    hf_model: str | None = None

    huggingface_base_url: str | None = None


    # ==========================
    # Groq
    # ==========================

    groq_api_key: str | None = None

    groq_model: str | None = None

    groq_base_url: str | None = None


    # ==========================
    # Ollama
    # ==========================

    ollama_base_url: str | None = None

    ollama_model: str | None = None


    # ==========================
    # Database
    # ==========================

    database_url: str


    # ==========================
    # Logging
    # ==========================

    log_level: str


    model_config = SettingsConfigDict(

        env_file=".env",

        case_sensitive=False,

        extra="ignore"

    )


@lru_cache
def get_settings() -> Settings:

    return Settings()