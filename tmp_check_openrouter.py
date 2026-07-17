import asyncio
from app.services.llm_service import LLMService

async def main():
    service = LLMService()
    try:
        result = await service.generate('openrouter', 'Say hello in one sentence')
        print(result)
    except Exception as exc:
        import traceback
        traceback.print_exc()

asyncio.run(main())
