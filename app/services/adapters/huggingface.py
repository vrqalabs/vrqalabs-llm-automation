from app.services.adapters.base import LLMAdapter


class HuggingFaceAdapter(LLMAdapter):
    async def generate(self, prompt: str):
        raise NotImplementedError("Hugging Face adapter not implemented yet.")
