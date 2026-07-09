from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from app.core.config import settings
from app.services.embedding_provider import HashEmbeddingProvider, create_embedding_provider
from app.services.knowledge_base import MEDICAL_KNOWLEDGE_BASE, normalize_text, tokenize


DEFAULT_SYSTEM_PROMPT = (
    "你是一位专业、谨慎的医学AI助手。你只能提供健康信息辅助、分诊建议和就医提醒，"
    "不能替代医生面诊、不能给出确定诊断。遇到急症风险必须优先建议线下急诊。"
)

QUERY_EXPANSION_RULES = {
    "发烧": ("发热", "体温升高"),
    "头疼": ("头痛", "头部疼痛"),
    "胃疼": ("胃痛", "胃部疼痛", "腹痛"),
    "肚子疼": ("腹痛", "腹部疼痛"),
    "喘不上气": ("呼吸困难", "气短"),
    "心慌": ("心悸", "胸闷"),
    "拉肚子": ("腹泻", "胃肠不适"),
}

MEDICAL_TERMS = {
    "发热": ("感染", "流感", "病毒", "血常规"),
    "咳嗽": ("呼吸道感染", "肺炎", "咳痰", "胸片"),
    "头痛": ("偏头痛", "神经系统", "血压", "颈项强直"),
    "胸痛": ("心肌梗死", "肺栓塞", "气胸", "心电图"),
    "腹痛": ("胃肠炎", "阑尾炎", "胆囊疾病", "腹部超声"),
    "尿痛": ("泌尿感染", "尿常规", "血尿"),
    "过敏": ("皮疹", "荨麻疹", "喉头水肿", "药物过敏"),
    "用药": ("剂量", "禁忌", "副作用", "相互作用"),
}

LATIN_TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]+")


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    document_id: str
    modality: str
    title: str
    category: str
    department: str
    severity_hint: int
    content: str
    source_path: str | None
    image_path: str | None
    caption: str | None
    keywords: tuple[str, ...]
    aliases: tuple[str, ...]
    red_flags: tuple[str, ...]
    token_counts: Counter
    vector: tuple[float, ...]
    image_vector: tuple[float, ...] | None = None


@dataclass(frozen=True)
class QueryExpansion:
    original: str
    normalized: str
    expanded_terms: tuple[str, ...]
    rewritten_queries: tuple[str, ...]
    extracted_keywords: tuple[str, ...]


@dataclass
class RetrievalCandidate:
    chunk: DocumentChunk
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    matched_terms: set[str] = field(default_factory=set)
    rrf_score: float = 0.0
    final_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.chunk.id,
            "document_id": self.chunk.document_id,
            "modality": self.chunk.modality,
            "title": self.chunk.title,
            "category": self.chunk.category,
            "department": self.chunk.department,
            "severity_hint": self.chunk.severity_hint,
            "content": self.chunk.content,
            "source_path": self.chunk.source_path,
            "image_path": self.chunk.image_path,
            "caption": self.chunk.caption,
            "score": round(self.final_score, 4),
            "rrf_score": round(self.rrf_score, 6),
            "channel_scores": {key: round(value, 4) for key, value in self.scores.items()},
            "channel_ranks": self.ranks,
            "matched_keywords": sorted(self.matched_terms),
            "retrieval_reason": self.retrieval_reason(),
        }

    def retrieval_reason(self) -> str:
        channels = "、".join(sorted(self.ranks))
        terms = "、".join(sorted(self.matched_terms)[:5]) or "语义相似"
        return f"通过{channels}检索命中；匹配：{terms}；关联科室：{self.chunk.department}。"


class RAGService:
    def __init__(
        self,
        top_k: int = 5,
        vector_dim: int = 96,
        rrf_k: int = 60,
        index_path: str | Path | None = None,
        include_builtin: bool = True,
        embedding_provider: HashEmbeddingProvider | None = None,
    ):
        self.top_k = top_k
        self.vector_dim = vector_dim
        self.rrf_k = rrf_k
        self.embedding_provider = embedding_provider or create_embedding_provider(
            provider=settings.embedding_provider,
            vector_dim=vector_dim,
            text_model=settings.qwen_text_embedding_model,
            vision_model=settings.qwen_vision_embedding_model,
            enable_local=settings.qwen_enable_local,
        )
        self.chunks = self._build_chunks() if include_builtin else []
        if index_path:
            self.load_index(index_path)
        self.avg_doc_len = self._average_doc_length()
        self.document_frequency = self._build_document_frequency()

    def add_documents(self, documents: list[dict], chunk_size: int = 220, overlap: int = 40) -> int:
        added = 0
        for doc_index, document in enumerate(documents):
            content = str(document.get("content", "")).strip()
            if not content:
                continue

            document_id = str(document.get("id") or stable_id(content, prefix="doc"))
            title = str(document.get("title") or f"Document {doc_index + 1}")
            category = str(document.get("category") or "external")
            department = str(document.get("department") or "全科 / 普通内科")
            severity_hint = int(document.get("severity_hint") or 1)
            keywords = tuple(document.get("keywords") or tokenize(content))
            aliases = tuple(document.get("aliases") or ())
            red_flags = tuple(document.get("red_flags") or ())

            for chunk_index, part in enumerate(split_text(content, chunk_size=chunk_size, overlap=overlap)):
                chunk_text = f"{title}。{part}"
                tokens = self._terms_for_text(" ".join([chunk_text, *keywords, *aliases, *red_flags]))
                self.chunks.append(
                    DocumentChunk(
                        id=f"{document_id}#chunk-{chunk_index}",
                        document_id=document_id,
                        modality=str(document.get("modality") or "text"),
                        title=title,
                        category=category,
                        department=department,
                        severity_hint=severity_hint,
                        content=part,
                        source_path=document.get("source_path"),
                        image_path=document.get("image_path"),
                        caption=document.get("caption"),
                        keywords=keywords,
                        aliases=aliases,
                        red_flags=red_flags,
                        token_counts=Counter(tokens),
                        vector=self._embed(tokens),
                        image_vector=(
                            self.embedding_provider.embed_image(document["image_path"])
                            if document.get("image_path")
                            else None
                        ),
                    )
                )
                added += 1

        self._refresh_statistics()
        return added

    def save_index(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "embedding": {
                "provider": getattr(self.embedding_provider, "name", "unknown"),
                "backend": getattr(self.embedding_provider, "backend", getattr(self.embedding_provider, "name", "unknown")),
                "dim": self.vector_dim,
                "text_model": getattr(self.embedding_provider, "text_model", None),
                "vision_model": getattr(self.embedding_provider, "vision_model", None),
                "fallback": bool(getattr(self.embedding_provider, "is_fallback", False)),
                "note": "Qwen provider is configured. If fallback=true, install the Qwen runtime dependencies and model weights to enable real local Qwen embeddings.",
            },
            "chunks": [chunk_to_record(chunk) for chunk in self.chunks],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_index(self, path: str | Path, replace: bool = False) -> int:
        source = Path(path)
        if not source.exists():
            return 0

        payload = json.loads(source.read_text(encoding="utf-8"))
        records = payload.get("chunks", [])
        loaded_chunks = [chunk_from_record(record) for record in records]

        if replace:
            self.chunks = loaded_chunks
        else:
            existing_ids = {chunk.id for chunk in self.chunks}
            self.chunks.extend(chunk for chunk in loaded_chunks if chunk.id not in existing_ids)

        self._refresh_statistics()
        return len(loaded_chunks)

    def retrieve(self, query: str, top_k: int | None = None, categories: list[str] | None = None) -> list[dict]:
        context = self.build_context(query=query, top_k=top_k, categories=categories)
        return context["retrieved_docs"]

    def build_context(
        self,
        query: str,
        top_k: int | None = None,
        categories: list[str] | None = None,
        conversation_history: list[dict] | None = None,
        user_context: dict | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> dict[str, Any]:
        expansion = self.expand_query(query)
        filtered_chunks = self._filter_chunks(categories)
        sparse_results = self._sparse_search(expansion, filtered_chunks)
        dense_results = self._dense_search(expansion, filtered_chunks)
        image_results = self._image_search(expansion, filtered_chunks)
        term_results = self._medical_term_search(expansion, filtered_chunks)
        fused = self._rrf_fuse(
            {
                "sparse_bm25": sparse_results,
                "dense_vector": dense_results,
                "image_vector": image_results,
                "medical_terms": term_results,
            }
        )
        selected = self._dedupe_and_select(fused, top_k or self.top_k)
        retrieved_docs = [candidate.to_dict() for candidate in selected]

        return {
            "query_expansion": {
                "original": expansion.original,
                "normalized": expansion.normalized,
                "expanded_terms": list(expansion.expanded_terms),
                "rewritten_queries": list(expansion.rewritten_queries),
                "extracted_keywords": list(expansion.extracted_keywords),
            },
            "retrieved_docs": retrieved_docs,
            "context_text": self._assemble_context_text(
                system_prompt=system_prompt,
                retrieved_docs=retrieved_docs,
                query=query,
                conversation_history=conversation_history or [],
                user_context=user_context or {},
            ),
            "pipeline": {
                "steps": [
                    "query_expansion",
                    "sparse_bm25",
                    "dense_vector",
                    "image_vector",
                    "medical_terms",
                    "rrf_fusion",
                    "dedupe_top_k",
                    "context_assembly",
                ],
                "rrf_formula": "score = sum(1 / (k + rank_i))",
                "rrf_k": self.rrf_k,
            },
        }

    def expand_query(self, query: str) -> QueryExpansion:
        normalized = normalize_text(query)
        extracted_keywords = tuple(tokenize(normalized))
        expanded_terms = []

        for source, targets in QUERY_EXPANSION_RULES.items():
            if source in query or normalize_text(source) in normalized:
                expanded_terms.extend(targets)

        for keyword in extracted_keywords:
            expanded_terms.extend(MEDICAL_TERMS.get(keyword, ()))

        expanded_terms = tuple(dict.fromkeys(term for term in expanded_terms if term))
        rewritten_queries = tuple(
            dict.fromkeys(
                [
                    normalized,
                    " ".join([normalized, *expanded_terms]),
                    self._complete_query(normalized, expanded_terms),
                ]
            )
        )

        return QueryExpansion(
            original=query,
            normalized=normalized,
            expanded_terms=expanded_terms,
            rewritten_queries=rewritten_queries,
            extracted_keywords=extracted_keywords,
        )

    def _build_chunks(self) -> list[DocumentChunk]:
        chunks = []
        for doc in MEDICAL_KNOWLEDGE_BASE:
            parts = split_text(doc.content)
            for index, part in enumerate(parts):
                chunk_text = f"{doc.title}。{part}"
                tokens = self._terms_for_text(
                    " ".join([chunk_text, *doc.keywords, *doc.aliases, *doc.red_flags])
                )
                chunks.append(
                    DocumentChunk(
                        id=f"{doc.id}#chunk-{index}",
                        document_id=doc.id,
                        modality="text",
                        title=doc.title,
                        category=doc.category,
                        department=doc.department,
                        severity_hint=doc.severity_hint,
                        content=part,
                        source_path=None,
                        image_path=None,
                        caption=None,
                        keywords=doc.keywords,
                        aliases=doc.aliases,
                        red_flags=doc.red_flags,
                        token_counts=Counter(tokens),
                        vector=self._embed(tokens),
                        image_vector=None,
                    )
                )
        return chunks

    def _sparse_search(
        self, expansion: QueryExpansion, chunks: list[DocumentChunk]
    ) -> list[tuple[DocumentChunk, float, set[str]]]:
        query_terms = self._query_terms(expansion)
        results = []
        for chunk in chunks:
            score = 0.0
            matches = set()
            doc_len = sum(chunk.token_counts.values()) or 1

            for term in query_terms:
                frequency = chunk.token_counts.get(term, 0)
                if frequency <= 0:
                    continue

                matches.add(term)
                idf = self._idf(term)
                numerator = frequency * 2.2
                denominator = frequency + 1.2 * (1 - 0.75 + 0.75 * doc_len / self.avg_doc_len)
                score += idf * numerator / denominator

            if score > 0:
                results.append((chunk, score, matches))

        return sorted(results, key=lambda item: item[1], reverse=True)

    def _dense_search(
        self, expansion: QueryExpansion, chunks: list[DocumentChunk]
    ) -> list[tuple[DocumentChunk, float, set[str]]]:
        query_terms = self._query_terms(expansion)
        query_vector = self._embed(query_terms)
        results = []

        for chunk in chunks:
            score = cosine_similarity(query_vector, chunk.vector)
            if score <= 0:
                continue

            matches = set(query_terms).intersection(chunk.token_counts)
            results.append((chunk, score, matches))

        return sorted(results, key=lambda item: item[1], reverse=True)

    def _image_search(
        self, expansion: QueryExpansion, chunks: list[DocumentChunk]
    ) -> list[tuple[DocumentChunk, float, set[str]]]:
        query_terms = self._query_terms(expansion)
        query_vector = self._embed(query_terms)
        results = []

        for chunk in chunks:
            if chunk.modality != "image" or not chunk.image_vector:
                continue

            score = cosine_similarity(query_vector, chunk.image_vector)
            caption_matches = set(query_terms).intersection(chunk.token_counts)
            if caption_matches:
                score += min(0.25, len(caption_matches) * 0.05)

            if score <= 0:
                continue

            results.append((chunk, score, caption_matches))

        return sorted(results, key=lambda item: item[1], reverse=True)

    def _medical_term_search(
        self, expansion: QueryExpansion, chunks: list[DocumentChunk]
    ) -> list[tuple[DocumentChunk, float, set[str]]]:
        query_terms = set(self._query_terms(expansion))
        results = []

        for chunk in chunks:
            searchable_terms = set(chunk.keywords + chunk.aliases + chunk.red_flags)
            matches = query_terms.intersection(searchable_terms)
            for red_flag in chunk.red_flags:
                if red_flag in expansion.normalized:
                    matches.add(red_flag)

            if not matches:
                continue

            score = len(matches) * 2 + (chunk.severity_hint if matches.intersection(chunk.red_flags) else 0)
            results.append((chunk, float(score), matches))

        return sorted(results, key=lambda item: item[1], reverse=True)

    def _rrf_fuse(
        self, channels: dict[str, list[tuple[DocumentChunk, float, set[str]]]]
    ) -> list[RetrievalCandidate]:
        candidates: dict[str, RetrievalCandidate] = {}

        for channel_name, results in channels.items():
            for rank, (chunk, score, matches) in enumerate(results, start=1):
                candidate = candidates.setdefault(chunk.id, RetrievalCandidate(chunk=chunk))
                candidate.scores[channel_name] = score
                candidate.ranks[channel_name] = rank
                candidate.matched_terms.update(matches)
                candidate.rrf_score += 1 / (self.rrf_k + rank)

        for candidate in candidates.values():
            weighted_channel_score = (
                candidate.scores.get("sparse_bm25", 0) * 0.45
                + candidate.scores.get("dense_vector", 0) * 0.25
                + candidate.scores.get("image_vector", 0) * 0.15
                + candidate.scores.get("medical_terms", 0) * 0.15
            )
            candidate.final_score = candidate.rrf_score * 100 + weighted_channel_score

        return sorted(candidates.values(), key=lambda item: item.final_score, reverse=True)

    def _dedupe_and_select(self, candidates: list[RetrievalCandidate], top_k: int) -> list[RetrievalCandidate]:
        selected = []
        seen_documents = set()

        for candidate in candidates:
            if candidate.chunk.document_id in seen_documents:
                continue
            if candidate.final_score < 0.5:
                continue

            selected.append(candidate)
            seen_documents.add(candidate.chunk.document_id)

            if len(selected) >= top_k:
                break

        return selected

    def _assemble_context_text(
        self,
        system_prompt: str,
        retrieved_docs: list[dict],
        query: str,
        conversation_history: list[dict],
        user_context: dict,
    ) -> str:
        history_text = "\n".join(
            f"{item.get('role', 'unknown')}: {item.get('content', '')}"
            for item in conversation_history[-6:]
        )
        profile = ", ".join(f"{key}: {value}" for key, value in user_context.items() if value) or "未提供"
        docs_text = "\n".join(
            f"[来源{index}] {doc['title']} | {doc['department']} | {doc['content']}"
            for index, doc in enumerate(retrieved_docs, start=1)
        )

        return "\n\n".join(
            [
                f"System Prompt:\n{system_prompt}",
                f"Patient Context:\n{profile}",
                f"Conversation History:\n{history_text or '无'}",
                f"Retrieved Docs:\n{docs_text or '无'}",
                f"User Question:\n{query}",
            ]
        )

    def _filter_chunks(self, categories: list[str] | None) -> list[DocumentChunk]:
        if not categories:
            return self.chunks

        allowed = set(categories)
        return [chunk for chunk in self.chunks if chunk.category in allowed]

    def _query_terms(self, expansion: QueryExpansion) -> list[str]:
        terms = []
        for query in expansion.rewritten_queries:
            terms.extend(self._terms_for_text(query))
        terms.extend(expansion.extracted_keywords)
        terms.extend(expansion.expanded_terms)
        return list(dict.fromkeys(term for term in terms if term))

    def _terms_for_text(self, text: str) -> list[str]:
        terms = tokenize(text)
        normalized = normalize_text(text)
        for token in LATIN_TOKEN_PATTERN.findall(normalized):
            normalized_token = token.replace("_", " ").replace("-", " ")
            if normalized_token not in terms:
                terms.append(normalized_token)
        for keyword, related_terms in MEDICAL_TERMS.items():
            if keyword in normalized and keyword not in terms:
                terms.append(keyword)
            for related in related_terms:
                if related in normalized and related not in terms:
                    terms.append(related)
        return terms

    def _embed(self, terms: list[str]) -> tuple[float, ...]:
        return self.embedding_provider.embed_terms(terms)

    def _build_document_frequency(self) -> dict[str, int]:
        frequency = defaultdict(int)
        for chunk in self.chunks:
            for term in chunk.token_counts:
                frequency[term] += 1
        return dict(frequency)

    def _refresh_statistics(self) -> None:
        self.avg_doc_len = self._average_doc_length()
        self.document_frequency = self._build_document_frequency()

    def _average_doc_length(self) -> float:
        lengths = [sum(chunk.token_counts.values()) for chunk in self.chunks]
        return sum(lengths) / len(lengths) if lengths else 1.0

    def _idf(self, term: str) -> float:
        doc_count = len(self.chunks)
        frequency = self.document_frequency.get(term, 0)
        return math.log(1 + (doc_count - frequency + 0.5) / (frequency + 0.5))

    def _complete_query(self, normalized_query: str, expanded_terms: tuple[str, ...]) -> str:
        if expanded_terms:
            return f"{normalized_query} {' '.join(expanded_terms[:6])}"
        return normalized_query


def split_text(text: str, chunk_size: int = 220, overlap: int = 40) -> list[str]:
    clean = " ".join(str(text).split())
    if len(clean) <= chunk_size:
        return [clean]

    chunks = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def chunk_to_record(chunk: DocumentChunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "modality": chunk.modality,
        "title": chunk.title,
        "category": chunk.category,
        "department": chunk.department,
        "severity_hint": chunk.severity_hint,
        "content": chunk.content,
        "source_path": chunk.source_path,
        "image_path": chunk.image_path,
        "caption": chunk.caption,
        "keywords": list(chunk.keywords),
        "aliases": list(chunk.aliases),
        "red_flags": list(chunk.red_flags),
        "token_counts": dict(chunk.token_counts),
        "vector": list(chunk.vector),
        "image_vector": list(chunk.image_vector) if chunk.image_vector else None,
    }


def chunk_from_record(record: dict[str, Any]) -> DocumentChunk:
    return DocumentChunk(
        id=str(record["id"]),
        document_id=str(record["document_id"]),
        modality=str(record.get("modality") or "text"),
        title=str(record["title"]),
        category=str(record.get("category") or "external"),
        department=str(record.get("department") or "全科 / 普通内科"),
        severity_hint=int(record.get("severity_hint") or 1),
        content=str(record.get("content") or ""),
        source_path=record.get("source_path"),
        image_path=record.get("image_path"),
        caption=record.get("caption"),
        keywords=tuple(record.get("keywords") or ()),
        aliases=tuple(record.get("aliases") or ()),
        red_flags=tuple(record.get("red_flags") or ()),
        token_counts=Counter(record.get("token_counts") or {}),
        vector=tuple(float(value) for value in record.get("vector", [])),
        image_vector=(
            tuple(float(value) for value in record.get("image_vector", []))
            if record.get("image_vector")
            else None
        ),
    )


def stable_id(value: str, prefix: str = "item") -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def load_documents_from_directory(directory: str | Path) -> list[dict]:
    root = Path(directory)
    documents = []
    image_captioner = HashEmbeddingProvider()

    if not root.exists():
        return documents

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            content = path.read_text(encoding="utf-8")
            metadata, body = parse_front_matter(content)
            documents.append(
                {
                    "id": metadata.get("id") or stable_id(str(path), prefix="file"),
                    "title": metadata.get("title") or path.stem,
                    "category": metadata.get("category") or "external",
                    "department": metadata.get("department") or "全科 / 普通内科",
                    "severity_hint": int(metadata.get("severity_hint") or 1),
                    "keywords": parse_csv(metadata.get("keywords")),
                    "aliases": parse_csv(metadata.get("aliases")),
                    "red_flags": parse_csv(metadata.get("red_flags")),
                    "content": body.strip(),
                    "source_path": str(path),
                    "modality": "text",
                }
            )
        elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            metadata = load_sidecar_metadata(path)
            caption = image_captioner.caption_image(path, metadata)
            documents.append(
                {
                    "id": metadata.get("id") or stable_id(str(path), prefix="image"),
                    "title": metadata.get("title") or path.stem,
                    "category": metadata.get("category") or "image",
                    "department": metadata.get("department") or "全科 / 普通内科",
                    "severity_hint": int(metadata.get("severity_hint") or 1),
                    "keywords": parse_csv(metadata.get("keywords")),
                    "aliases": parse_csv(metadata.get("aliases")),
                    "red_flags": parse_csv(metadata.get("red_flags")),
                    "content": caption,
                    "caption": caption,
                    "source_path": str(path),
                    "image_path": str(path),
                    "modality": "image",
                }
            )

    return documents


def parse_front_matter(content: str) -> tuple[dict[str, str], str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, content

    metadata = {}
    body_start = 0
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body_start = index + 1
            break
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()

    return metadata, "\n".join(lines[body_start:])


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_sidecar_metadata(path: Path) -> dict[str, str]:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.exists():
        sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return {}

    try:
        return {
            str(key): str(value)
            for key, value in json.loads(sidecar.read_text(encoding="utf-8")).items()
        }
    except json.JSONDecodeError:
        return {}
