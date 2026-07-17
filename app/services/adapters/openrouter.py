import time
import json

import httpx

from app.core.config import get_settings
from app.services.adapters.base import LLMAdapter
from app.utils.logging import get_logger


logger = get_logger(__name__)


class OpenRouterAdapter(LLMAdapter):

    def __init__(self):

        self.settings = get_settings()

        self.base_url = (
            self.settings.openrouter_base_url
        )

        logger.info(
            "OpenRouter adapter initialized"
        )


    async def generate(self, prompt: str) -> str:

        start_time = time.time()


        api_key = (
            self.settings.openrouter_api_key
            or ""
        ).strip()


        if not api_key:

            logger.error(
                "OpenRouter API key missing"
            )

            raise RuntimeError(
                "OpenRouter API key is not configured"
            )


        logger.info(
            "========== LLM REQUEST START =========="
        )


        logger.info(
            f"Provider: OpenRouter"
        )


        logger.info(
            f"Model: {self.settings.openrouter_model}"
        )


        logger.info(
            f"Prompt: {prompt}"
        )


        headers = {

            "Authorization":
                f"Bearer {api_key}",

            "Content-Type":
                "application/json",

            "HTTP-Referer":
                "https://github.com/vrqalabs",

            "X-Title":
                "VRQALabs LLM Automation"

        }


        payload = {

            "model":
                self.settings.openrouter_model,

            "messages":
                [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

            "temperature": 0.2

        }


        logger.info(
            "Request Payload:"
        )

        logger.info(
            json.dumps(
                payload,
                indent=2
            )
        )


        async with httpx.AsyncClient(
            timeout=60
        ) as client:


            response = await client.post(

                self.base_url,

                headers=headers,

                json=payload

            )


        latency = (
            time.time()
            -
            start_time
        )


        logger.info(
            "OpenRouter Response Received"
        )


        logger.info(
            f"HTTP Status: {response.status_code}"
        )


        logger.info(
            f"Latency: {latency:.2f} seconds"
        )


        if response.status_code != 200:

            logger.error(
                response.text
            )

            raise RuntimeError(
                f"OpenRouter error {response.status_code}: {response.text}"
            )


        data = response.json()


        logger.info(
            "Response JSON:"
        )


        logger.info(
            json.dumps(
                data,
                indent=2
            )
        )


        answer = (
            data["choices"][0]
            ["message"]
            ["content"]
        )


        logger.info(
            "Generated Answer:"
        )


        logger.info(
            answer
        )


        logger.info(
            "========== LLM REQUEST END =========="
        )


        return answer