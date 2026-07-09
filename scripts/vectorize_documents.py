from __future__ import annotations

import argparse
from pathlib import Path

from app.services.rag_service import RAGService, load_documents_from_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorize medical knowledge documents for RAG.")
    parser.add_argument("--input", default="knowledge_base", help="Directory containing .md/.txt documents.")
    parser.add_argument("--output", default="storage/vector_index.json", help="Output vector index JSON path.")
    parser.add_argument("--chunk-size", type=int, default=220, help="Characters per chunk.")
    parser.add_argument("--overlap", type=int, default=40, help="Overlapping characters between chunks.")
    parser.add_argument(
        "--builtin",
        action="store_true",
        help="Also include built-in seed medical knowledge in the saved index.",
    )
    args = parser.parse_args()

    documents = load_documents_from_directory(args.input)
    text_docs = sum(1 for document in documents if document.get("modality") == "text")
    image_docs = sum(1 for document in documents if document.get("modality") == "image")
    rag = RAGService(include_builtin=args.builtin)
    added_chunks = rag.add_documents(documents, chunk_size=args.chunk_size, overlap=args.overlap)
    rag.save_index(args.output)

    print(f"Loaded documents: {len(documents)}")
    print(f"Text documents: {text_docs}")
    print(f"Image documents: {image_docs}")
    print(f"Vectorized chunks: {added_chunks}")
    print(f"Saved index: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
