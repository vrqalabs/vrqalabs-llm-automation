from abc import ABC, abstractmethod
import time
from typing import Any

from app.core.config import get_settings
from app.schemas.llm_response import LLMResponse, TokenUsage


class LLMAdapter(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self):
        self.settings = get_settings()

    @abstractmethod
    async def generate(self, prompt: str) -> Any:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt.

        Returns:
            Provider-specific response.
        """
        raise NotImplementedError

    def _build_result(
        self,
        *,
        provider: str,
        model: str,
        response: str,
        latency_ms: int,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
        estimated_cost: float | None = None,
    ) -> LLMResponse:
        normalized_total = total_tokens
        if normalized_total is None:
            normalized_total = prompt_tokens + completion_tokens

        cost_value = estimated_cost
        if cost_value is None:
            cost_value = self._estimate_cost(provider, model, normalized_total)

        return LLMResponse(
            provider=provider,
            model=model,
            response=response,
            latency_ms=float(max(0, latency_ms)),
            tokens=TokenUsage(
                prompt_tokens=max(0, prompt_tokens),
                completion_tokens=max(0, completion_tokens),
                total_tokens=max(0, normalized_total),
            ),
            estimated_cost=cost_value,
            status="success",
        )

    def _normalize_tokens(self, value: Any) -> int:
        if value is None:
            return 0
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def _estimate_cost(self, provider: str, model: str, total_tokens: int) -> float:
        if total_tokens <= 0:
            return 0.0

        provider_name = provider.lower()
        model_name = (model or "").lower()

        if "free" in model_name or provider_name in {"huggingface", "groq", "ollama"}:
            return 0.0

        if provider_name == "openrouter":
            return round(total_tokens * 0.0000006, 6)

        return 0.0

    def _sanitize_headers(self, headers: dict[str, Any]) -> dict[str, Any]:
        safe_headers: dict[str, Any] = {}
        for key, value in headers.items():
            lowered = key.lower()
            if lowered in {"authorization", "x-api-key", "api-key", "proxy-authorization"}:
                safe_headers[key] = "***"
            elif "secret" in lowered or "token" in lowered:
                safe_headers[key] = "***"
            else:
                safe_headers[key] = value
        return safe_headers

    def _get_model(self, primary: str | None, fallback: str | None = None) -> str:
        return (primary or fallback or "").strip()

    def _get_setting(self, name: str, default: Any = None) -> Any:
        return getattr(self.settings, name, default)

    def _get_latency_ms(self, start_time: float) -> int:
        return int(round((time.time() - start_time) * 1000))
