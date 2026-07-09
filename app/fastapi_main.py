from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover - depends on optional runtime deps.
    raise RuntimeError(
        "FastAPI runtime is not installed. Confirm and install fastapi + uvicorn "
        "before running app.fastapi_main:app."
    ) from exc

from app.api.v1 import agent, consult, knowledge, system
from app.api.ws import websocket_endpoint
from app.core.config import settings
from app.services.runtime import consultation_service


ROOT_DIR = Path(__file__).resolve().parent.parent
PUBLIC_DIR = ROOT_DIR / "public"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    print("=" * 50, flush=True)
    print(f"Starting {settings.app_name} FastAPI service", flush=True)
    print(f"Version: {settings.app_version}", flush=True)
    print(f"Loaded consultations: {consultation_service.count()}", flush=True)
    print("=" * 50, flush=True)
    yield
    print(f"Stopping {settings.app_name} FastAPI service", flush=True)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=(
            "AI medical consultation backend powered by a rule-based agent, "
            "local RAG retrieval, and an optional OpenAI-compatible LLM workflow."
        ),
        version=settings.app_version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(system.router, prefix="/api/v1", tags=["system"])
    app.include_router(consult.router, prefix="/api/v1/consultations", tags=["consultations"])
    app.include_router(agent.router, prefix="/api/v1/agent", tags=["agent"])
    app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])

    # Compatibility routes keep the existing frontend working during migration.
    app.include_router(system.router, prefix="/api", tags=["legacy-system"])
    app.include_router(consult.router, prefix="/api/consultations", tags=["legacy-consultations"])
    app.include_router(knowledge.router, prefix="/api/knowledge", tags=["legacy-knowledge"])

    app.websocket_route("/api/ws/chat")(websocket_endpoint)

    if PUBLIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public")

    return app


app = create_app()
