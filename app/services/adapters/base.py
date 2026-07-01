from abc import ABC, abstractmethod
from typing import Any


class LLMAdapter(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str) -> Any:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt.

        Returns:
            Provider-specific response.
        """
        raise NotImplementedError
