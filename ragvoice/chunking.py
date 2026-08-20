from __future__ import annotations

import re
from dataclasses import dataclass


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    doc_id: str
    strategy: str
    text: str
    metadata: dict


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sentence_split(text: str) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    return [part.strip() for part in SENTENCE_RE.split(cleaned) if part.strip()]


def fixed_overlap_chunks(text: str, size: int = 280, overlap: int = 60) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + size)
        chunks.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def semantic_window_chunks(text: str, sentences_per_chunk: int = 3) -> list[str]:
    sentences = sentence_split(text)
    if not sentences:
        return []
    results = []
    for idx in range(0, len(sentences), max(1, sentences_per_chunk - 1)):
        window = sentences[idx : idx + sentences_per_chunk]
        if window:
            results.append(" ".join(window))
    return results


def metadata_aware_chunks(title: str, text: str) -> list[str]:
    title = normalize_text(title)
    body_chunks = semantic_window_chunks(text, sentences_per_chunk=2)
    prefix = f"{title}: " if title else ""
    return [prefix + chunk for chunk in body_chunks]


def multi_strategy_chunks(doc_id: str, title: str, text: str, metadata: dict | None = None) -> list[Chunk]:
    metadata = metadata or {}
    chunks: list[Chunk] = []
    for chunk in semantic_window_chunks(text):
        chunks.append(Chunk(doc_id=doc_id, strategy="semantic_window", text=chunk, metadata=metadata))
    for chunk in fixed_overlap_chunks(text):
        chunks.append(Chunk(doc_id=doc_id, strategy="fixed_overlap", text=chunk, metadata=metadata))
    for chunk in metadata_aware_chunks(title, text):
        chunks.append(Chunk(doc_id=doc_id, strategy="metadata_aware", text=chunk, metadata=metadata | {"title": title}))
    return [chunk for chunk in chunks if chunk.text]
