from fastapi import APIRouter, HTTPException, status

from app.schemas.generate_request import GenerateRequest
from app.schemas.generate_response import GenerateResponse
from app.services.llm_service import LLMService
from app.utils.logging import get_logger


router = APIRouter(tags=["Generation"])

logger = get_logger(__name__)

service = LLMService()

@router.post(
    "/generate",
    response_model=GenerateResponse
)
async def generate(request: GenerateRequest):

    logger.info(
        f"Request received | provider={request.provider}"
    )

    try:

        result = await service.generate(
            request.provider,
            request.prompt
        )

        logger.info(
            "Generation completed successfully"
        )

        return result

    except Exception:

        logger.exception(
            "Generation failed"
        )

        raise