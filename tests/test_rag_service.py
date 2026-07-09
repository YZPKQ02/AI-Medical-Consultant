import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from app.services.embedding_provider import QwenEmbeddingProvider
from app.services.rag_service import RAGService, load_documents_from_directory


class RAGServiceTests(unittest.TestCase):
    def test_default_embedding_provider_is_qwen_adapter(self):
        rag = RAGService(top_k=3, include_builtin=False)

        self.assertEqual(rag.embedding_provider.name, "qwen")
        self.assertIsInstance(rag.embedding_provider, QwenEmbeddingProvider)

    def test_query_expansion_normalizes_and_expands_symptom(self):
        rag = RAGService(top_k=3)
        expansion = rag.expand_query("我头疼发烧怎么办")

        self.assertEqual(expansion.normalized, "我头痛发热怎么办")
        self.assertIn("头痛", expansion.extracted_keywords)
        self.assertIn("感染", expansion.expanded_terms)

    def test_hybrid_retrieval_returns_channel_scores_and_rrf(self):
        rag = RAGService(top_k=3)
        results = rag.retrieve("胸痛伴呼吸困难和大汗", top_k=3)

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["category"], "emergency")
        self.assertIn("sparse_bm25", results[0]["channel_scores"])
        self.assertIn("dense_vector", results[0]["channel_scores"])
        self.assertIn("medical_terms", results[0]["channel_scores"])
        self.assertGreater(results[0]["rrf_score"], 0)

    def test_context_assembly_contains_prompt_docs_and_question(self):
        rag = RAGService(top_k=2)
        context = rag.build_context(
            query="胃疼怎么办",
            conversation_history=[{"role": "user", "content": "昨天开始不舒服"}],
            user_context={"age": "30"},
        )

        self.assertIn("query_expansion", context)
        self.assertIn("Retrieved Docs", context["context_text"])
        self.assertIn("User Question", context["context_text"])
        self.assertIn("胃疼怎么办", context["context_text"])
        self.assertEqual(context["pipeline"]["steps"][-1], "context_assembly")

    def test_document_vectorization_save_and_load_index(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            doc_path = root / "urinary.md"
            index_path = root / "vector_index.json"
            doc_path.write_text(
                "\n".join(
                    [
                        "---",
                        "id: urinary-test",
                        "title: 尿痛就医建议",
                        "category: urinary",
                        "department: 泌尿外科 / 肾内科",
                        "severity_hint: 2",
                        "keywords: 尿痛,尿频,血尿",
                        "---",
                        "尿痛伴尿频常见于泌尿系统感染，若出现发热、腰痛或血尿，应尽快就医。",
                    ]
                ),
                encoding="utf-8",
            )

            documents = load_documents_from_directory(root)
            rag = RAGService(include_builtin=False)
            chunk_count = rag.add_documents(documents)
            rag.save_index(index_path)

            loaded = RAGService(include_builtin=False, index_path=index_path)
            results = loaded.retrieve("尿痛还有血尿", top_k=1)

            self.assertGreaterEqual(chunk_count, 1)
            self.assertTrue(index_path.exists())
            self.assertEqual(results[0]["document_id"], "urinary-test")

    def test_image_document_vectorization_and_retrieval(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "skin_rash_photo.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake-medical-image-bytes")
            image_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "id": "skin-rash-image",
                        "title": "皮疹照片",
                        "category": "image",
                        "department": "皮肤科",
                        "severity_hint": 1,
                        "keywords": "皮疹,过敏,skin,rash",
                        "image_type": "clinical photo",
                        "body_part": "skin",
                        "description": "患者皮肤红色皮疹照片",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            documents = load_documents_from_directory(root)
            rag = RAGService(include_builtin=False)
            rag.add_documents(documents)
            results = rag.retrieve("皮疹照片 过敏", top_k=1)

            self.assertEqual(documents[0]["modality"], "image")
            self.assertEqual(results[0]["document_id"], "skin-rash-image")
            self.assertEqual(results[0]["modality"], "image")
            self.assertIn("image_vector", results[0]["channel_scores"])
            self.assertTrue(results[0]["image_path"].endswith("skin_rash_photo.png"))


if __name__ == "__main__":
    unittest.main()
