from app.services.factory import LLMFactory
from app.services.exceptions import ProviderNotFoundError


class LLMService:
    async def generate(self, provider: str, prompt: str) -> str:
        if not provider or not prompt:
            raise ValueError("Provider and prompt are required")

        try:
            adapter = LLMFactory.create(provider)
        except ProviderNotFoundError as exc:
            raise ValueError(f"Unsupported provider: {provider}") from exc

        try:
            result = await adapter.generate(prompt)
        except NotImplementedError as exc:
            raise RuntimeError(str(exc)) from exc

        if isinstance(result, str):
            return result

        if isinstance(result, dict) and "generated_text" in result:
            return str(result["generated_text"])

        return str(result)
