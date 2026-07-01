from app.services.adapters.base import LLMAdapter


class OpenRouterAdapter(LLMAdapter):
    async def generate(self, prompt: str):
        raise NotImplementedError("OpenRouter adapter not implemented yet.")
