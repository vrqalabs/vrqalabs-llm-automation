import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app
from app.services.factory import LLMFactory
from app.services.llm_service import LLMService
from app.services.exceptions import ProviderNotFoundError


client = TestClient(app)


def test_valid_request_returns_200():
    response = client.post(
        "/generate",
        json={"provider": "huggingface", "prompt": "Explain AI"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "huggingface"
    assert isinstance(body["response"], str)
    assert body["response"]


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
            },
        )()

        from app.services.adapters.huggingface import HuggingFaceAdapter

        adapter = HuggingFaceAdapter()
        result = await adapter.generate("Hello")

    assert result == "Mocked response"


@pytest.mark.asyncio
async def test_service_behavior_uses_adapter_result():
    service = LLMService()

    with patch("app.services.factory.LLMFactory.create") as mock_create:
        mock_adapter = AsyncMock()
        mock_adapter.generate.return_value = "service response"
        mock_create.return_value = mock_adapter

        result = await service.generate("huggingface", "Hello")

    assert result == "service response"


def test_factory_behavior_returns_expected_adapter():
    adapter = LLMFactory.create("openrouter")
    assert adapter.__class__.__name__ == "OpenRouterAdapter"


def test_factory_behavior_unknown_provider_raises():
    with pytest.raises(ProviderNotFoundError):
        LLMFactory.create("not-a-provider")
