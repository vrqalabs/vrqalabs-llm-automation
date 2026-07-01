from pydantic import BaseModel, Field


class GenerateResponse(BaseModel):
    provider: str = Field(..., description="Provider used for generation")
    response: str = Field(..., description="Generated text returned by the provider")
