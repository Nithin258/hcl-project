import io
import unittest
import uuid
from unittest.mock import patch

import chromadb
from pypdf import PdfWriter

import rag


class RagTests(unittest.TestCase):
    def test_blank_pdf_reports_missing_searchable_text(self):
        output = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.write(output)

        with self.assertRaisesRegex(RuntimeError, "No extractable text"):
            rag.pdf_to_chunks(output.getvalue(), "blank.pdf")

    def test_index_returns_nearest_chunk_with_source_metadata(self):
        client = chromadb.EphemeralClient()
        chunks = [
            {
                "id": "alpha",
                "text": "alpha content",
                "metadata": {"source": "a.pdf", "page": 1},
            },
            {
                "id": "beta",
                "text": "beta content",
                "metadata": {"source": "b.pdf", "page": 2},
            },
        ]

        def fake_embeddings(texts, model, base_url):
            return [
                [1.0, 0.0]
                if "alpha" in text.lower() or "question" in text.lower()
                else [0.0, 1.0]
                for text in texts
            ]

        with patch.object(rag, "_ollama_embeddings", side_effect=fake_embeddings):
            collection = rag.build_index(
                client,
                f"test_{uuid.uuid4().hex}",
                chunks,
                "mock-embedding-model",
                "http://localhost:11434",
            )
            matches = rag.retrieve_chunks(
                collection,
                "question",
                "mock-embedding-model",
                "http://localhost:11434",
            )

        self.assertEqual(matches[0]["source"], "a.pdf")
        self.assertEqual(matches[0]["page"], 1)
        self.assertEqual(matches[0]["text"], "alpha content")


if __name__ == "__main__":
    unittest.main()