from app.services.adapters.groq import GroqAdapter
from app.services.adapters.huggingface import HuggingFaceAdapter
from app.services.adapters.ollama import OllamaAdapter
from app.services.adapters.openrouter import OpenRouterAdapter
from app.services.exceptions import ProviderNotFoundError


class LLMFactory:
    """Creates LLM provider instances."""

    _providers = {
        "huggingface": HuggingFaceAdapter,
        "openrouter": OpenRouterAdapter,
        "groq": GroqAdapter,
        "ollama": OllamaAdapter,
    }

    @classmethod
    def create(cls, provider: str):
        provider = provider.lower()
        if provider not in cls._providers:
            raise ProviderNotFoundError(f"Unknown provider: {provider}")
        return cls._providers[provider]()
