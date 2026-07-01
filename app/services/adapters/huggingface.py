import httpx

from app.core.config import get_settings
from app.services.adapters.base import LLMAdapter


class HuggingFaceAdapter(LLMAdapter):
    async def generate(self, prompt: str):
        settings = get_settings()
        api_key = settings.hf_api_key.strip()
        if not api_key:
            return f"Mock response for prompt: {prompt}"

        url = f"{settings.huggingface_base_url}/{settings.huggingface_model}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {"inputs": prompt}

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, headers=headers, json=payload)

        if response.status_code == 401:
            raise RuntimeError("Unauthorized Hugging Face request")
        if response.status_code == 429:
            raise RuntimeError("Hugging Face rate limit exceeded")
        if response.status_code >= 400:
            raise RuntimeError(f"Hugging Face request failed: {response.text}")

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Unexpected Hugging Face response format") from exc

        if isinstance(data, list) and data:
            item = data[0]
            if isinstance(item, dict):
                generated = item.get("generated_text") or item.get("text") or ""
                return str(generated).strip()

        if isinstance(data, dict):
            generated = data.get("generated_text") or data.get("text") or ""
            return str(generated).strip()

        raise RuntimeError("Unexpected Hugging Face API response")
