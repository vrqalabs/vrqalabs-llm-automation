from app.schemas.llm_response import TokenUsage


class TokenCalculator:
    """
    Extract token usage from provider response.
    """

    @staticmethod
    def from_openrouter(data: dict) -> TokenUsage:
        usage = data.get("usage", {})

        return TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
