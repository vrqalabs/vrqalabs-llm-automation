from pydantic import BaseModel


class TokenUsage(BaseModel):
    """
    Token usage information returned by an LLM provider.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """
    Standard response returned by every LLM provider.
    """

    provider: str
    model: str
    response: str
    latency_ms: float
    tokens: TokenUsage
    estimated_cost: float = 0.0
    status: str = "success"
