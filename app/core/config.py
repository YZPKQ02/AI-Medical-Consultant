from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse


@dataclass(frozen=True)
class Settings:
    app_name: str = "AI Medical Consultant"
    app_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 3000
    cors_origins: tuple[str, ...] = ("http://127.0.0.1:3000", "http://localhost:3000")
    rag_top_k: int = 3
    rag_index_path: str = "storage/vector_index.json"
    consultation_store_path: str = "storage/consultations.sqlite3"
    embedding_provider: str = "qwen"
    qwen_text_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    qwen_vision_embedding_model: str = "Qwen/Qwen3-VL-Embedding"
    qwen_enable_local: bool = False
    agent_llm_enabled: bool = False
    llm_model: str = "deepseek-v4-flash"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_temperature: float = 0.3
    llm_top_p: float = 0.9
    llm_max_tokens: int = 1400
    llm_timeout_seconds: int = 30
    hospital_recommender_enabled: bool = False
    amap_mcp_url: str = ""
    amap_web_service_key: str = ""
    hospital_recommender_limit: int = 5
    hospital_recommender_timeout_seconds: int = 5


def get_settings() -> Settings:
    return Settings(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "3000")),
        cors_origins=parse_csv_env(
            os.getenv("CORS_ORIGINS", "http://127.0.0.1:3000,http://localhost:3000")
        ),
        rag_top_k=int(os.getenv("RAG_TOP_K", "3")),
        rag_index_path=os.getenv("RAG_INDEX_PATH", "storage/vector_index.json"),
        consultation_store_path=os.getenv(
            "CONSULTATION_STORE_PATH", "storage/consultations.sqlite3"
        ),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "qwen"),
        qwen_text_embedding_model=os.getenv("QWEN_TEXT_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        qwen_vision_embedding_model=os.getenv("QWEN_VISION_EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding"),
        qwen_enable_local=os.getenv("QWEN_ENABLE_LOCAL", "0").lower() in {"1", "true", "yes"},
        agent_llm_enabled=os.getenv("AGENT_LLM_ENABLED", "0").lower() in {"1", "true", "yes"},
        llm_model=os.getenv("DEEPSEEK_MODEL", os.getenv("LLM_MODEL", "deepseek-v4-flash")),
        llm_base_url=os.getenv("DEEPSEEK_BASE_URL", os.getenv("LLM_BASE_URL", "")),
        llm_api_key=os.getenv("DEEPSEEK_API_KEY", os.getenv("LLM_API_KEY", "")),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        llm_top_p=float(os.getenv("LLM_TOP_P", "0.9")),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1400")),
        llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
        hospital_recommender_enabled=os.getenv("HOSPITAL_RECOMMENDER_ENABLED", "0").lower()
        in {"1", "true", "yes"},
        amap_mcp_url=os.getenv("AMAP_MCP_URL", ""),
        amap_web_service_key=os.getenv("AMAP_WEB_SERVICE_KEY", "")
        or extract_amap_key_from_mcp_url(os.getenv("AMAP_MCP_URL", "")),
        hospital_recommender_limit=int(os.getenv("HOSPITAL_RECOMMENDER_LIMIT", "5")),
        hospital_recommender_timeout_seconds=int(os.getenv("HOSPITAL_RECOMMENDER_TIMEOUT_SECONDS", "5")),
    )


def parse_csv_env(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def extract_amap_key_from_mcp_url(value: str) -> str:
    query = parse_qs(urlparse(str(value or "")).query)
    key = (query.get("key") or [""])[0].strip()
    if not key or key.startswith("<"):
        return ""
    return key


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


load_env_file()
settings = get_settings()
