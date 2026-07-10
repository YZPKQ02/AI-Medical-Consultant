from __future__ import annotations

from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - depends on optional runtime setup.
    raise RuntimeError(
        "MCP SDK is not installed. Run `.venv\\Scripts\\python.exe -m pip install -r requirements.txt`."
    ) from exc

from app.services.hospital_recommender import recommend_hospitals as recommend_hospitals_impl


mcp = FastMCP("hospital-recommender")


@mcp.tool()
def recommend_hospitals(
    city: str,
    department: str,
    urgency_level: int,
    symptoms: list[str] | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Recommend hospital department candidates by city and triage result."""
    return recommend_hospitals_impl(
        city=city,
        department=department,
        urgency_level=urgency_level,
        symptoms=symptoms or [],
        limit=limit,
    )


if __name__ == "__main__":
    mcp.run()
