import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app
from app.schemas.generate_response import GenerateResponse
from app.schemas.llm_response import LLMResponse, TokenUsage
from app.services.exceptions import ProviderNotFoundError
from app.services.factory import LLMFactory
from app.services.llm_service import LLMService


client = TestClient(app)


def test_valid_request_returns_200():
    response = client.post(
        "/generate",
        json={"provider": "huggingface", "prompt": "Explain AI"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["provider"] == "huggingface"
    assert body["result"]["model"]
    assert isinstance(body["result"]["response"], str)
    assert body["result"]["response"]
    assert isinstance(body["result"]["latency_ms"], (int, float))
    assert body["result"]["tokens"]["total_tokens"] >= 0
    assert body["result"]["estimated_cost"] >= 0


def test_invalid_request_missing_prompt_returns_422():
    response = client.post(
        "/generate",
        json={"provider": "huggingface"},
    )

    assert response.status_code == 422


def test_unknown_provider_returns_400():
    response = client.post(
        "/generate",
        json={"provider": "unknown", "prompt": "Explain AI"},
    )

    assert response.status_code == 400
    assert "Unsupported provider" in response.json()["detail"]


@pytest.mark.asyncio
async def test_mock_huggingface_response():
    with patch("app.services.adapters.huggingface.httpx.AsyncClient") as mock_client, patch("app.services.adapters.huggingface.get_settings") as mock_settings:
        class MockResponse:
            status_code = 200

            def json(self):
                return [{"generated_text": "Mocked response"}]

        mock_response = MockResponse()
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        mock_settings.return_value = type(
            "Settings",
            (),
            {
                "hf_api_key": "dummy-key",
                "huggingface_base_url": "https://example.test",
                "huggingface_model": "test-model",
                "hf_model": "test-model",
            },
        )()

        from app.services.adapters.huggingface import HuggingFaceAdapter

        adapter = HuggingFaceAdapter()
        result = await adapter.generate("Hello")

    assert result.provider == "huggingface"
    assert result.model == "test-model"
    assert result.response == "Mocked response"
    assert result.latency_ms >= 0
    assert result.tokens.prompt_tokens >= 0


@pytest.mark.asyncio
async def test_mock_openrouter_response():
    with patch("app.services.adapters.openrouter.httpx.AsyncClient") as mock_client, patch("app.services.adapters.openrouter.get_settings") as mock_settings:
        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {"message": {"content": "Mocked OpenRouter response"}}
                    ]
                }

        mock_response = MockResponse()
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        mock_settings.return_value = type(
            "Settings",
            (),
            {
                "openrouter_api_key": "dummy-key",
                "openrouter_base_url": "https://example.test",
                "openrouter_model": "mistral/7b",
            },
        )()

        from app.services.adapters.openrouter import OpenRouterAdapter

        adapter = OpenRouterAdapter()
        result = await adapter.generate("Hello")

    assert result.provider == "openrouter"
    assert result.model == "mistral/7b"
    assert result.response == "Mocked OpenRouter response"
    assert result.latency_ms >= 0
    assert result.tokens.prompt_tokens >= 0


@pytest.mark.asyncio
async def test_openrouter_missing_api_key_raises_runtime_error():
    with patch("app.services.adapters.openrouter.get_settings") as mock_settings:
        mock_settings.return_value = type(
            "Settings",
            (),
            {"openrouter_api_key": "", "openrouter_model": "mistral/7b"},
        )()

        from app.services.adapters.openrouter import OpenRouterAdapter

        adapter = OpenRouterAdapter()

        with pytest.raises(RuntimeError, match="OpenRouter API key is not configured"):
            await adapter.generate("Hello")


@pytest.mark.asyncio
async def test_service_behavior_uses_adapter_result():
    service = LLMService()

    with patch("app.services.factory.LLMFactory.create") as mock_create:
        mock_adapter = AsyncMock()
        mock_adapter.generate.return_value = {
            "provider": "huggingface",
            "model": "test-model",
            "response": "service response",
            "latency_ms": 12,
            "tokens": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
            "estimated_cost": 0.0,
        }
        mock_create.return_value = mock_adapter

        result = await service.generate("huggingface", "Hello")

    assert isinstance(result, GenerateResponse)
    assert result.result.provider == "huggingface"
    assert result.result.response == "service response"
    assert result.result.model == "test-model"
    assert result.result.latency_ms == 12


@pytest.mark.asyncio
async def test_service_persists_generation_record():
    service = LLMService()

    with patch("app.services.factory.LLMFactory.create") as mock_create, patch("app.services.llm_service.save_generation_request") as mock_save:
        mock_adapter = AsyncMock()
        mock_adapter.generate.return_value = LLMResponse(
            provider="huggingface",
            model="test-model",
            response="persisted response",
            latency_ms=8.5,
            tokens=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            estimated_cost=0.01,
            status="success",
        )
        mock_create.return_value = mock_adapter

        result = await service.generate("huggingface", "Hello")

    assert isinstance(result, GenerateResponse)
    mock_save.assert_called_once()


def test_factory_behavior_returns_expected_adapter():
    adapter = LLMFactory.create("openrouter")
    assert adapter.__class__.__name__ == "OpenRouterAdapter"


def test_factory_behavior_unknown_provider_raises():
    with pytest.raises(ProviderNotFoundError):
        LLMFactory.create("not-a-provider")
