from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    provider: str = Field(..., min_length=1, description="LLM provider name")
    prompt: str = Field(..., min_length=1, description="Prompt to send to the provider")
