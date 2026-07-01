from fastapi import FastAPI
from app.api.routes.health import router as health_router

app = FastAPI(
    title="VRQALabs LLM Automation",
    description="Production-ready LLM automation and benchmarking platform.",
    version="0.1.0",
)

app.include_router(health_router)


@app.get("/")
async def root():
    return {
        "project": "VRQALabs LLM Automation",
        "status": "running",
        "version": "0.1.0",
    }
