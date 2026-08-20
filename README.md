# Voice-Enabled RAG Demo

This project is a local, runnable implementation of the August 13, 2026 HH Goa Task 2 brief for a voice-enabled RAG pipeline.

It includes:

- voice input via browser recording or uploaded audio
- pluggable speech-to-text adapters for `Sarvam` or `ElevenLabs`
- multiple chunking strategies instead of a single fixed-size splitter
- a SQLite-backed local vector store
- retrieval, guardrails, and structured orchestration
- latency analytics with `p50`, `p70`, and `p100`

## What came from the PDF vs. this repo

The attached PDF defined the product brief:

- build a voice-enabled RAG system
- use either Sarvam or ElevenLabs for speech-to-text
- use a thoughtful chunking strategy
- target sub-200ms pipeline latency
- report `p50`, `p70`, and `p100`
- include harnessing and guardrails

This repository is my implementation of that brief in a self-contained local project.

## Quick start

```bash
python3 app.py --bootstrap-sample-data
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## How it works

1. Audio or transcript enters the HTTP API.
2. A speech-to-text adapter converts audio to text.
3. The ingester creates chunks using several strategies:
   - sentence-aware chunking
   - fixed-size overlapping chunking
   - metadata-aware title-prefixed chunking
4. Chunks are embedded with a lightweight hashed-token embedding.
5. Chunks are stored in SQLite and ranked with cosine similarity.
6. Guardrails screen off-topic and unsafe inputs.
7. The answer stage composes a grounded response only from retrieved evidence.
8. Latency metrics are recorded for each stage and aggregated in benchmark runs.

## Environment variables

- `RAGVOICE_DB_PATH`: SQLite path, default `./data/ragvoice.db`
- `RAGVOICE_CORPUS_PATH`: optional JSONL dataset path
- `RAGVOICE_STT_PROVIDER`: `mock`, `sarvam`, or `elevenlabs`
- `RAGVOICE_SARVAM_API_KEY`: required for `sarvam`
- `RAGVOICE_ELEVENLABS_API_KEY`: required for `elevenlabs`
- `RAGVOICE_TOP_K`: retrieval depth, default `5`
- `RAGVOICE_EMBED_DIM`: embedding dimension, default `256`

## API endpoints

- `GET /health`
- `POST /api/ingest`
- `POST /api/query`
- `POST /api/benchmark`
- `GET /api/config`

## Dataset note

The competition brief points to the `MSMARCO-XI` dataset. Because this workspace has no bundled copy of that dataset and no network download step, the repo ships with a small sample corpus for verification. You can ingest the actual dataset later by converting it to the documented JSONL shape and posting it to `/api/ingest`.

Each JSONL line should look like:

```json
{"id":"doc-1","title":"Example title","text":"Longer document body text here.","metadata":{"source":"custom"}}
```

## Benchmarks

Run a small benchmark with:

```bash
python3 app.py --bootstrap-sample-data --benchmark
```

## Project layout

- `app.py`: server and CLI entrypoint
- `ragvoice/`: pipeline modules
- `data/sample_corpus.jsonl`: sample content for a working demo
- `static/index.html`: simple browser UI
- `tests/test_pipeline.py`: smoke tests
