from fastapi import APIRouter, HTTPException, status

from app.schemas.generate_request import GenerateRequest
from app.schemas.generate_response import GenerateResponse
from app.services.exceptions import ProviderNotFoundError
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

    except ProviderNotFoundError as exc:
        logger.exception("Unsupported provider requested")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported provider: {request.provider}",
        ) from exc

    except Exception:

        logger.exception(
            "Generation failed"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Generation failed",
        )