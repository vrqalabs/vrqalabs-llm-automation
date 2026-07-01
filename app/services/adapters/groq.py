from app.services.adapters.base import LLMAdapter


class GroqAdapter(LLMAdapter):
    async def generate(self, prompt: str):
        raise NotImplementedError("Groq adapter not implemented yet.")
