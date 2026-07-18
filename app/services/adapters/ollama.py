import json
import time

import httpx

from app.core.config import get_settings
from app.services.adapters.base import LLMAdapter
from app.utils.logging import get_logger


logger = get_logger(__name__)


class OllamaAdapter(LLMAdapter):

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.base_url = self._get_setting("ollama_base_url") or "http://localhost:11434"
        logger.info("Ollama adapter initialized")

    async def generate(self, prompt: str):
        start_time = time.time()

        model = self._get_model(self._get_setting("ollama_model"), "llama3.2")
        endpoint = f"{self.base_url.rstrip('/')}/api/generate"

        logger.info("========== OLLAMA REQUEST START ==========")
        logger.info("Provider: Ollama")
        logger.info(f"Model: {model}")
        logger.info(f"Endpoint: {endpoint}")

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

        logger.info("Payload: %s", json.dumps(payload, indent=2))

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(endpoint, json=payload)
        except Exception as exc:
            logger.exception("Ollama request failed")
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        latency_ms = self._get_latency_ms(start_time)

        logger.info("Response Status: %s", response.status_code)
        logger.info("Latency: %sms", latency_ms)

        if response.status_code != 200:
            logger.error("Ollama error response: %s", response.text)
            raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")

        data = response.json()
        logger.info("Response: %s", json.dumps(data, indent=2))

        answer = data.get("response", "")
        prompt_tokens = self._normalize_tokens(data.get("prompt_eval_count"))
        completion_tokens = self._normalize_tokens(data.get("eval_count"))
        total_tokens = prompt_tokens + completion_tokens

        logger.info("Generated Answer: %s", answer)
        logger.info("========== OLLAMA REQUEST END ==========")

        return self._build_result(
            provider="ollama",
            model=model,
            response=answer,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
