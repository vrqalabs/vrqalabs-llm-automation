import json
import time

import httpx

from app.core.config import get_settings
from app.services.adapters.base import LLMAdapter
from app.utils.logging import get_logger


logger = get_logger(__name__)


class HuggingFaceAdapter(LLMAdapter):
    """
    Hugging Face Inference API Adapter.

    Supports Hugging Face hosted models.
    """

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.base_url = self._get_setting("huggingface_base_url") or "https://router.huggingface.co/hf-inference/models"
        logger.info("HuggingFace adapter initialized")

    async def generate(self, prompt: str):
        start_time = time.time()

        model = self._get_model(self._get_setting("hf_model"), self._get_setting("huggingface_model"))
        if not model:
            model = "offline-fallback"

        api_key = (self.settings.hf_api_key or "").strip()

        if not api_key:
            logger.warning("HuggingFace API key missing; returning offline fallback response")
            return self._build_result(
                provider="huggingface",
                model=model,
                response=f"Offline fallback response for prompt: {prompt}",
                latency_ms=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )

        url = f"{self.base_url}/{model}"

        logger.info("========== HUGGINGFACE REQUEST START ==========")
        logger.info("Provider: HuggingFace")
        logger.info(f"Model: {model}")
        logger.info(f"Endpoint: {url}")
        logger.info(f"Prompt: {prompt}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "inputs": prompt,
            "parameters": {"temperature": 0.2, "max_new_tokens": 256},
        }

        logger.info("Headers: %s", json.dumps(self._sanitize_headers(headers), indent=2))
        logger.info("Payload: %s", json.dumps(payload, indent=2))

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(url, headers=headers, json=payload)
        except Exception as exc:
            logger.exception("HuggingFace request failed")
            return self._build_result(
                provider="huggingface",
                model=model,
                response=f"Offline fallback response for prompt: {prompt}",
                latency_ms=self._get_latency_ms(start_time),
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )

        latency_ms = self._get_latency_ms(start_time)

        logger.info("Response Status: %s", response.status_code)
        logger.info("Latency: %sms", latency_ms)
        response_text = getattr(response, "text", None)
        if response_text is None:
            try:
                response_text = json.dumps(response.json())
            except Exception:
                response_text = ""
        logger.info("Response: %s", response_text)

        if response.status_code != 200:
            logger.error("HuggingFace error response: %s", response.text)
            raise RuntimeError(f"HuggingFace API error {response.status_code}: {response.text}")

        data = response.json()

        try:
            if isinstance(data, list):
                generated_text = data[0].get("generated_text", "")
            else:
                generated_text = data.get("generated_text", "")
        except Exception:
            logger.exception("Unable to parse HuggingFace response")
            raise RuntimeError("Invalid HuggingFace response format")

        logger.info("Generated Answer: %s", generated_text)
        logger.info("========== HUGGINGFACE REQUEST END ==========")

        return self._build_result(
            provider="huggingface",
            model=model,
            response=generated_text,
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )