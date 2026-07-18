from app.database.models import LLMRequest
from app.database.session import SessionLocal
from app.schemas.generate_response import GenerateResponse
from app.schemas.llm_response import LLMResponse, TokenUsage
from app.services.factory import LLMFactory
from app.utils.logging import get_logger

logger = get_logger(__name__)


def save_generation_request(provider: str, prompt: str, response: LLMResponse):
    """
    Persist a completed generation request and response to the database.
    """

    session = SessionLocal()

    try:
        record = LLMRequest(
            provider=provider,
            model=response.model,
            prompt=prompt,
            response=response.response,
            latency_ms=response.latency_ms,
            prompt_tokens=response.tokens.prompt_tokens,
            completion_tokens=response.tokens.completion_tokens,
            total_tokens=response.tokens.total_tokens,
            estimated_cost=response.estimated_cost,
            status=response.status,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record
    except Exception:
        session.rollback()
        logger.exception("Failed to save generation request")
        raise
    finally:
        session.close()


class LLMService:

    async def generate(
        self,
        provider: str,
        prompt: str
    ):

        logger.info(
            f"Selecting provider | {provider}"
        )

        adapter = LLMFactory.create(provider)

        logger.info(
            f"Calling adapter | {provider}"
        )

        result = await adapter.generate(prompt)

        logger.info(
            "Adapter response received"
        )

        if isinstance(result, LLMResponse):
            response = result
        elif isinstance(result, dict):
            tokens = result.get("tokens") or {}
            response = LLMResponse(
                provider=result.get("provider", provider),
                model=result.get("model", ""),
                response=result.get("response", ""),
                latency_ms=float(result.get("latency_ms", 0)),
                tokens=TokenUsage(
                    prompt_tokens=tokens.get("prompt_tokens", 0) if isinstance(tokens, dict) else 0,
                    completion_tokens=tokens.get("completion_tokens", 0) if isinstance(tokens, dict) else 0,
                    total_tokens=tokens.get("total_tokens", 0) if isinstance(tokens, dict) else 0,
                ),
                estimated_cost=float(result.get("estimated_cost", 0.0)),
                status=result.get("status", "success"),
            )
        else:
            response = LLMResponse(
                provider=provider,
                model="",
                response=str(result),
                latency_ms=0.0,
                tokens=TokenUsage(),
            )

        save_generation_request(provider, prompt, response)

        return GenerateResponse(result=response)