import json

import httpx

from app.core.config import get_settings
from app.metrics.cost import CostCalculator
from app.metrics.latency import LatencyTimer
from app.metrics.tokens import TokenCalculator
from app.services.adapters.base import LLMAdapter
from app.utils.logging import get_logger


logger = get_logger(__name__)


class OpenRouterAdapter(LLMAdapter):

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.base_url = self._get_setting("openrouter_base_url") or "https://openrouter.ai/api/v1/chat/completions"
        logger.info("OpenRouter adapter initialized")

    async def generate(self, prompt: str):
        timer = LatencyTimer()
        timer.start()

        api_key = (self._get_setting("openrouter_api_key") or "").strip()

        if not api_key:
            logger.error("OpenRouter API key missing")
            raise RuntimeError("OpenRouter API key is not configured")

        model = self._get_model(self._get_setting("openrouter_model"))
        if not model:
            logger.error("OpenRouter model missing")
            raise RuntimeError("OpenRouter model is not configured")

        logger.info("========== LLM REQUEST START ==========")
        logger.info("Provider: OpenRouter")
        logger.info(f"Model: {model}")
        logger.info(f"Endpoint: {self.base_url}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/vrqalabs",
            "X-Title": "VRQALabs LLM Automation",
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
            logger.exception("OpenRouter request failed")
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

        latency_ms = timer.stop()

        logger.info("Response Status: %s", response.status_code)
        logger.info("Latency: %sms", latency_ms)

        if response.status_code != 200:
            logger.error("OpenRouter error response: %s", response.text)
            raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")

        data = response.json()
        logger.info("Response: %s", json.dumps(data, indent=2))

        answer = data["choices"][0]["message"]["content"]
        tokens = TokenCalculator.from_openrouter(data)
        prompt_tokens = self._normalize_tokens(tokens.prompt_tokens)
        completion_tokens = self._normalize_tokens(tokens.completion_tokens)
        total_tokens = self._normalize_tokens(tokens.total_tokens) or prompt_tokens + completion_tokens
        cost = CostCalculator.estimate_cost(
            provider="openrouter",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        logger.info("Generated Answer: %s", answer)
        logger.info("========== LLM REQUEST END ==========")

        return self._build_result(
            provider="openrouter",
            model=model,
            response=answer,
            latency_ms=int(latency_ms),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=cost,
        )