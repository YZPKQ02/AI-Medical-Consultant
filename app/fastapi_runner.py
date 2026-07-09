from __future__ import annotations

from app.core.config import settings


def run() -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised only before deps are installed.
        raise RuntimeError(
            "FastAPI dependencies are missing. Run `python -m pip install -r requirements.txt`."
        ) from exc

    uvicorn.run(
        "app.fastapi_main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
