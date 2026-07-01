from fastapi import APIRouter, HTTPException, status

from app.schemas.generate_request import GenerateRequest
from app.schemas.generate_response import GenerateResponse
from app.services.llm_service import LLMService

router = APIRouter(tags=["Generation"])
service = LLMService()


@router.post("/generate", response_model=GenerateResponse, status_code=status.HTTP_200_OK)
async def generate(request: GenerateRequest):
    try:
        response_text = await service.generate(request.provider, request.prompt)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)) from exc

    return GenerateResponse(provider=request.provider, response=response_text)
