import json
import time

import httpx

from app.core.config import get_settings
from app.services.adapters.base import LLMAdapter
from app.utils.logging import get_logger


logger = get_logger(__name__)


class GroqAdapter(LLMAdapter):

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.base_url = self._get_setting("groq_base_url") or "https://api.groq.com/openai/v1/chat/completions"
        logger.info("Groq adapter initialized")

    async def generate(self, prompt: str):
        start_time = time.time()

        api_key = (self._get_setting("groq_api_key") or "").strip()
        if not api_key:
            logger.error("Groq API key missing")
            raise RuntimeError("Groq API key is not configured")

        model = self._get_model(self._get_setting("groq_model"))
        if not model:
            logger.error("Groq model missing")
            raise RuntimeError("Groq model is not configured")

        logger.info("========== GROQ REQUEST START ==========")
        logger.info("Provider: Groq")
        logger.info(f"Model: {model}")
        logger.info(f"Endpoint: {self.base_url}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }

        logger.info("Headers: %s", json.dumps(self._sanitize_headers(headers), indent=2))
        logger.info("Payload: %s", json.dumps(payload, indent=2))

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(self.base_url, headers=headers, json=payload)
        except Exception as exc:
            logger.exception("Groq request failed")
            raise RuntimeError(f"Groq request failed: {exc}") from exc

        latency_ms = self._get_latency_ms(start_time)

        logger.info("Response Status: %s", response.status_code)
        logger.info("Latency: %sms", latency_ms)

        if response.status_code != 200:
            logger.error("Groq error response: %s", response.text)
            raise RuntimeError(f"Groq error {response.status_code}: {response.text}")

        data = response.json()
        logger.info("Response: %s", json.dumps(data, indent=2))

        answer = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        prompt_tokens = self._normalize_tokens(usage.get("prompt_tokens"))
        completion_tokens = self._normalize_tokens(usage.get("completion_tokens"))
        total_tokens = self._normalize_tokens(usage.get("total_tokens")) or prompt_tokens + completion_tokens

        logger.info("Generated Answer: %s", answer)
        logger.info("========== GROQ REQUEST END ==========")

        return self._build_result(
            provider="groq",
            model=model,
            response=answer,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
