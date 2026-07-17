from app.schemas.generate_response import GenerateResponse
from app.services.exceptions import ProviderNotFoundError
from app.services.factory import LLMFactory
from app.utils.logging import get_logger

logger = get_logger(__name__)

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

        return GenerateResponse(
            provider=provider,
            response=result,
        )