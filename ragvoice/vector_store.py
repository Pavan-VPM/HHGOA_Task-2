from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ragvoice.chunking import Chunk
from ragvoice.embed import cosine_similarity, embed_text


class SQLiteVectorStore:
    def __init__(self, db_path: Path, dim: int) -> None:
        self.db_path = db_path
        self.dim = dim
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    vector_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        rows = []
        for chunk in chunks:
            rows.append(
                (
                    chunk.doc_id,
                    chunk.strategy,
                    chunk.text,
                    json.dumps(chunk.metadata, sort_keys=True),
                    json.dumps(embed_text(chunk.text, self.dim)),
                )
            )
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO chunks (doc_id, strategy, text, metadata_json, vector_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            return int(row[0]) if row else 0

    def search(self, query: str, top_k: int) -> list[dict]:
        query_vec = embed_text(query, self.dim)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id, strategy, text, metadata_json, vector_json FROM chunks"
            ).fetchall()
        scored = []
        for doc_id, strategy, text, metadata_json, vector_json in rows:
            score = cosine_similarity(query_vec, json.loads(vector_json))
            scored.append(
                {
                    "doc_id": doc_id,
                    "strategy": strategy,
                    "text": text,
                    "metadata": json.loads(metadata_json),
                    "score": round(score, 5),
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]
