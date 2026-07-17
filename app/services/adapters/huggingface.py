import time
import json

import httpx

from app.core.config import get_settings
from app.services.adapters.base import LLMAdapter
from app.utils.logging import get_logger


logger = get_logger(__name__)


class HuggingFaceAdapter(LLMAdapter):
    """
    Hugging Face Inference API Adapter.

    Supports Hugging Face hosted models.
    """

    def __init__(self):

        self.settings = get_settings()

        self.base_url = (
            self.settings.huggingface_base_url
        )

        logger.info(
            "HuggingFace adapter initialized"
        )


    async def generate(self, prompt: str) -> str:

        start_time = time.time()


        api_key = (
            self.settings.hf_api_key
            or ""
        ).strip()


        if not api_key:

            logger.error(
                "HuggingFace API key missing"
            )

            raise RuntimeError(
                "HuggingFace API key is not configured"
            )


        model = (
            self.settings.hf_model
            or self.settings.huggingface_model
        )


        if not model:

            logger.error(
                "HuggingFace model missing"
            )

            raise RuntimeError(
                "HuggingFace model is not configured"
            )


        url = (
            f"{self.base_url}/{model}"
        )


        logger.info(
            "========== HUGGINGFACE REQUEST START =========="
        )


        logger.info(
            f"Provider: HuggingFace"
        )


        logger.info(
            f"Model: {model}"
        )


        logger.info(
            f"Prompt: {prompt}"
        )


        headers = {

            "Authorization":
                f"Bearer {api_key}",

            "Content-Type":
                "application/json"

        }


        payload = {

            "inputs": prompt,

            "parameters": {

                "temperature": 0.2,

                "max_new_tokens": 256

            }

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

                url,

                headers=headers,

                json=payload

            )


        latency = (
            time.time()
            -
            start_time
        )


        logger.info(
            "HuggingFace Response Received"
        )


        logger.info(
            f"HTTP Status: {response.status_code}"
        )


        logger.info(
            f"Latency: {latency:.2f} seconds"
        )


        logger.info(
            "Response JSON:"
        )


        logger.info(
            response.text
        )


        if response.status_code != 200:

            logger.error(
                response.text
            )

            raise RuntimeError(
                f"HuggingFace API error "
                f"{response.status_code}: "
                f"{response.text}"
            )


        data = response.json()


        try:

            if isinstance(data, list):

                generated_text = (
                    data[0]
                    .get("generated_text", "")
                )

            else:

                generated_text = (
                    data.get(
                        "generated_text",
                        ""
                    )
                )


        except Exception:

            logger.exception(
                "Unable to parse HuggingFace response"
            )

            raise RuntimeError(
                "Invalid HuggingFace response format"
            )


        logger.info(
            "Generated Answer:"
        )


        logger.info(
            generated_text
        )


        logger.info(
            "========== HUGGINGFACE REQUEST END =========="
        )


        return generated_text