from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    host: str
    port: int
    db_path: Path
    corpus_path: Path | None
    stt_provider: str
    top_k: int
    embed_dim: int

    @classmethod
    def from_env(cls, host: str | None = None, port: int | None = None) -> "AppConfig":
        corpus_env = os.getenv("RAGVOICE_CORPUS_PATH")
        return cls(
            host=host or os.getenv("RAGVOICE_HOST", "127.0.0.1"),
            port=port or int(os.getenv("RAGVOICE_PORT", "8000")),
            db_path=Path(os.getenv("RAGVOICE_DB_PATH", "data/ragvoice.db")),
            corpus_path=Path(corpus_env) if corpus_env else None,
            stt_provider=os.getenv("RAGVOICE_STT_PROVIDER", "mock").lower(),
            top_k=int(os.getenv("RAGVOICE_TOP_K", "5")),
            embed_dim=int(os.getenv("RAGVOICE_EMBED_DIM", "256")),
        )

    def as_public_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "db_path": str(self.db_path),
            "stt_provider": self.stt_provider,
            "top_k": self.top_k,
            "embed_dim": self.embed_dim,
        }
