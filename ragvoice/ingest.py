from __future__ import annotations

import json
from pathlib import Path

from ragvoice.chunking import multi_strategy_chunks
from ragvoice.config import AppConfig
from ragvoice.vector_store import SQLiteVectorStore


class CorpusIngester:
    def __init__(self, config: AppConfig) -> None:
        self.store = SQLiteVectorStore(config.db_path, dim=config.embed_dim)

    def ingest_jsonl(self, path: Path) -> dict:
        documents = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    documents.append(json.loads(line))
        return self.ingest_documents(documents)

    def ingest_documents(self, documents: list[dict]) -> dict:
        chunks = []
        for doc in documents:
            chunks.extend(
                multi_strategy_chunks(
                    doc_id=doc["id"],
                    title=doc.get("title", ""),
                    text=doc.get("text", ""),
                    metadata=doc.get("metadata", {}),
                )
            )
        inserted = self.store.upsert_chunks(chunks) if chunks else 0
        return {"documents": len(documents), "chunks": inserted, "total_chunks": self.store.count()}
