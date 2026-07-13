from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
from typing import Callable


class HashEmbeddingProvider:
    """Dependency-free embedding provider used as a swappable placeholder.

    The interface mirrors what a real provider should expose:
    - embed_text: text -> vector
    - embed_terms: token list -> vector
    - embed_image: image bytes/path -> vector
    - caption_image: image path + metadata -> searchable text

    In production, replace this class with Qwen-VL/Qwen embedding,
    BioMedCLIP, CLIP, OpenAI embeddings, or another provider.
    """

    name = "hash-multimodal"

    def __init__(self, vector_dim: int = 96):
        self.vector_dim = vector_dim

    def embed_text(self, text: str, tokenizer: Callable[[str], list[str]]) -> tuple[float, ...]:
        return self.embed_terms(tokenizer(text))

    def embed_query(self, text: str, tokenizer: Callable[[str], list[str]]) -> tuple[float, ...]:
        return self.embed_text(text, tokenizer)

    def embed_document(self, text: str, tokenizer: Callable[[str], list[str]]) -> tuple[float, ...]:
        return self.embed_text(text, tokenizer)

    @property
    def embedding_dimension(self) -> int:
        return self.vector_dim

    def embed_terms(self, terms: list[str]) -> tuple[float, ...]:
        vector = [0.0] * self.vector_dim
        counts = Counter(terms)
        for term, count in counts.items():
            self._accumulate(vector, term.encode("utf-8"), 1 + math.log(count))
        return normalize_vector(vector)

    def embed_image(self, image_path: str | Path) -> tuple[float, ...]:
        path = Path(image_path)
        if not path.exists():
            return tuple([0.0] * self.vector_dim)

        data = path.read_bytes()
        vector = [0.0] * self.vector_dim
        self._accumulate(vector, path.suffix.lower().encode("utf-8"), 1.0)
        self._accumulate(vector, path.stem.lower().encode("utf-8"), 1.5)

        for start in range(0, min(len(data), 65536), 1024):
            self._accumulate(vector, data[start : start + 1024], 1.0)

        return normalize_vector(vector)

    def caption_image(self, image_path: str | Path, metadata: dict | None = None) -> str:
        path = Path(image_path)
        meta = metadata or {}
        title = meta.get("title") or path.stem
        body_part = meta.get("body_part") or infer_body_part(path.name)
        image_type = meta.get("image_type") or infer_image_type(path.name)
        description = meta.get("description") or meta.get("caption") or ""
        keywords = meta.get("keywords") or ""

        return " ".join(
            item
            for item in [
                str(title),
                str(image_type),
                str(body_part),
                str(description),
                str(keywords),
            ]
            if item
        )

    def _accumulate(self, vector: list[float], payload: bytes, weight: float) -> None:
        digest = hashlib.sha256(payload).digest()
        index = int.from_bytes(digest[:4], "big") % self.vector_dim
        sign = 1 if digest[4] % 2 == 0 else -1
        vector[index] += sign * weight


class QwenEmbeddingProvider:
    """Qwen-compatible embedding provider with a dependency-free fallback.

    Real Qwen local inference requires extra production dependencies such as
    sentence-transformers/transformers/torch and model weights. This adapter
    keeps the application usable before those dependencies are installed.
    """

    name = "qwen"

    def __init__(
        self,
        vector_dim: int = 96,
        text_model: str = "Qwen/Qwen3-Embedding-0.6B",
        vision_model: str = "Qwen/Qwen3-VL-Embedding",
        enable_local: bool = False,
        query_instruction: str = "",
        local_files_only: bool = False,
    ):
        self.vector_dim = vector_dim
        self.text_model = text_model
        self.vision_model = vision_model
        self.enable_local = enable_local
        self.query_instruction = query_instruction
        self.local_files_only = local_files_only
        self.fallback = HashEmbeddingProvider(vector_dim=vector_dim)
        self.backend = "qwen-local" if enable_local else "qwen-configured-fallback-hash"
        self._text_model_handle = None

    @property
    def is_fallback(self) -> bool:
        return self._text_model_handle is None

    def embed_text(self, text: str, tokenizer: Callable[[str], list[str]]) -> tuple[float, ...]:
        return self.embed_document(text, tokenizer)

    def embed_query(self, text: str, tokenizer: Callable[[str], list[str]]) -> tuple[float, ...]:
        if self._ensure_text_model_loaded():
            encode_kwargs = {"normalize_embeddings": True}
            if self.query_instruction:
                encode_kwargs["prompt"] = f"Instruct: {self.query_instruction}\nQuery: "
            vector = self._text_model_handle.encode([text], **encode_kwargs)[0]
            return tuple(float(value) for value in vector)
        return self.fallback.embed_text(text, tokenizer)

    def embed_document(self, text: str, tokenizer: Callable[[str], list[str]]) -> tuple[float, ...]:
        if self._ensure_text_model_loaded():
            vector = self._text_model_handle.encode([text], normalize_embeddings=True)[0]
            return tuple(float(value) for value in vector)
        return self.fallback.embed_text(text, tokenizer)

    def embed_terms(self, terms: list[str]) -> tuple[float, ...]:
        return self.embed_document(" ".join(terms), lambda value: value.split())

    @property
    def embedding_dimension(self) -> int:
        if self._text_model_handle is not None:
            get_dimension = getattr(self._text_model_handle, "get_embedding_dimension", None)
            if get_dimension is None:
                get_dimension = self._text_model_handle.get_sentence_embedding_dimension
            return int(get_dimension())
        return self.vector_dim

    def embed_image(self, image_path: str | Path) -> tuple[float, ...]:
        # Keep a stable API for future Qwen-VL image embedding. Until the
        # runtime dependency/model is installed, combine image bytes with
        # caption text through the fallback provider.
        return self.fallback.embed_image(image_path)

    def caption_image(self, image_path: str | Path, metadata: dict | None = None) -> str:
        # In a full Qwen-VL setup this method should call the vision-language
        # model to produce findings/caption text. Metadata fallback is explicit
        # so the indexed evidence remains inspectable.
        base_caption = self.fallback.caption_image(image_path, metadata)
        return f"{base_caption} provider:qwen"

    def _ensure_text_model_loaded(self) -> bool:
        if not self.enable_local:
            return False
        if self._text_model_handle is not None:
            return True

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            self.backend = "qwen-missing-sentence-transformers-fallback-hash"
            raise RuntimeError(
                "Qwen local embedding is enabled but sentence-transformers is not installed"
            ) from exc

        try:
            self._text_model_handle = SentenceTransformer(
                self.text_model,
                local_files_only=self.local_files_only,
            )
            self.backend = "qwen-local"
            return True
        except Exception as exc:
            self.backend = "qwen-load-failed-fallback-hash"
            raise RuntimeError(f"Failed to load local embedding model {self.text_model}: {exc}") from exc


def create_embedding_provider(
    provider: str = "qwen",
    vector_dim: int = 96,
    text_model: str = "Qwen/Qwen3-Embedding-0.6B",
    vision_model: str = "Qwen/Qwen3-VL-Embedding",
    enable_local: bool = False,
    query_instruction: str = "",
    local_files_only: bool = False,
):
    normalized = provider.strip().lower()
    if normalized == "qwen":
        return QwenEmbeddingProvider(
            vector_dim=vector_dim,
            text_model=text_model,
            vision_model=vision_model,
            enable_local=enable_local,
            query_instruction=query_instruction,
            local_files_only=local_files_only,
        )
    if normalized in {"hash", "hash-multimodal"}:
        return HashEmbeddingProvider(vector_dim=vector_dim)
    raise ValueError(f"Unsupported embedding provider: {provider}")


def normalize_vector(vector: list[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


def infer_body_part(filename: str) -> str:
    lowered = filename.lower()
    mapping = {
        "chest": "chest",
        "lung": "lung",
        "xray": "chest",
        "ct": "cross-sectional imaging",
        "skin": "skin",
        "rash": "skin",
        "abdomen": "abdomen",
        "brain": "brain",
        "head": "head",
    }
    for key, value in mapping.items():
        if key in lowered:
            return value
    return "unknown body part"


def infer_image_type(filename: str) -> str:
    lowered = filename.lower()
    if "report" in lowered or "lab" in lowered:
        return "document image"
    if "xray" in lowered or "ct" in lowered or "mri" in lowered or "ultrasound" in lowered:
        return "medical imaging"
    if "skin" in lowered or "rash" in lowered:
        return "clinical photo"
    return "medical image"
