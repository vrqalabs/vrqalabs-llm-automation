from app.services.adapters.base import LLMAdapter


class OllamaAdapter(LLMAdapter):
    async def generate(self, prompt: str):
        raise NotImplementedError("Ollama adapter not implemented yet.")
