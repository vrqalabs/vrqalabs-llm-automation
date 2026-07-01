from fastapi import FastAPI

from app.api.routes.generate import router as generate_router
from app.api.routes.health import router as health_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Production-ready LLM automation and benchmarking platform.",
    version=settings.app_version,
)

app.include_router(health_router)
app.include_router(generate_router)


@app.get("/")
async def root():
    return {
        "project": settings.app_name,
        "status": "running",
        "environment": settings.app_env,
        "version": settings.app_version,
    }
