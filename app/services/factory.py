from app.services.adapters.groq import GroqAdapter
from app.services.adapters.huggingface import HuggingFaceAdapter
from app.services.adapters.ollama import OllamaAdapter
from app.services.adapters.openrouter import OpenRouterAdapter
from app.services.exceptions import ProviderNotFoundError
from app.utils.logging import get_logger


logger = get_logger(__name__)


class LLMFactory:
    """
    Factory Pattern implementation.

    Responsible for creating the correct LLM adapter
    based on the provider name.

    Supported providers:
        - openrouter
        - huggingface
        - groq
        - ollama
    """

    _providers = {
        "openrouter": OpenRouterAdapter,
        "huggingface": HuggingFaceAdapter,
        "groq": GroqAdapter,
        "ollama": OllamaAdapter,
    }


    @classmethod
    def create(cls, provider: str):
        """
        Create and return an LLM adapter instance.

        Example:
            adapter = LLMFactory.create("openrouter")

        """

        provider = provider.lower().strip()

        logger.info(
            f"Factory request received | provider={provider}"
        )


        if provider not in cls._providers:

            logger.error(
                f"Unsupported provider requested | provider={provider}"
            )

            raise ProviderNotFoundError(
                f"Unknown provider: {provider}"
            )


        adapter_class = cls._providers[provider]


        logger.info(
            f"Creating adapter instance | adapter={adapter_class.__name__}"
        )


        adapter = adapter_class()


        logger.info(
            f"Adapter created successfully | provider={provider}"
        )


        return adapter