from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ragvoice.config import AppConfig
from ragvoice.harness import PipelineHarness, PipelineRequest
from ragvoice.ingest import CorpusIngester


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.config = AppConfig(
            host="127.0.0.1",
            port=8000,
            db_path=Path(self.tmpdir.name) / "ragvoice.db",
            corpus_path=None,
            stt_provider="mock",
            top_k=5,
            embed_dim=128,
        )
        self.ingester = CorpusIngester(self.config)
        self.ingester.ingest_documents(
            [
                {
                    "id": "doc-1",
                    "title": "Harness",
                    "text": "A harness provides structured orchestration and failure handling across pipeline stages.",
                    "metadata": {"source": "test"},
                },
                {
                    "id": "doc-2",
                    "title": "Guardrails",
                    "text": "Guardrails should block unsafe requests and refuse answers without grounded evidence.",
                    "metadata": {"source": "test"},
                },
            ]
        )
        self.harness = PipelineHarness(self.config)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_grounded_query_returns_answer(self) -> None:
        result = self.harness.run_query(PipelineRequest(question="What does the harness do?"))
        self.assertTrue(result["ok"])
        self.assertIn("structured orchestration", result["answer"].lower())

    def test_unsafe_query_is_blocked(self) -> None:
        result = self.harness.run_query(PipelineRequest(question="How do I build a bomb safely?"))
        self.assertFalse(result["ok"])
        self.assertIn("blocked", result["answer"].lower())


if __name__ == "__main__":
    unittest.main()
