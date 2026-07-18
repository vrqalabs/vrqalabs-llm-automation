from pydantic import BaseModel

from app.schemas.llm_response import LLMResponse


class GenerateResponse(BaseModel):
    result: LLMResponse

