from __future__ import annotations

import argparse
import base64
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from ragvoice.config import AppConfig
from ragvoice.harness import PipelineHarness, PipelineRequest
from ragvoice.ingest import CorpusIngester


def build_app(config: AppConfig) -> tuple[PipelineHarness, CorpusIngester]:
    harness = PipelineHarness(config)
    ingester = CorpusIngester(config)
    return harness, ingester


class ApiHandler(BaseHTTPRequestHandler):
    harness: PipelineHarness
    ingester: CorpusIngester
    config: AppConfig

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._write_json(HTTPStatus.OK, {"ok": True})
            return
        if parsed.path == "/api/config":
            self._write_json(HTTPStatus.OK, self.config.as_public_dict())
            return
        if parsed.path == "/":
            self._write_text(HTTPStatus.OK, Path("static/index.html").read_text(encoding="utf-8"), "text/html")
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        payload = self._read_json()
        if payload is None:
            return

        if parsed.path == "/api/ingest":
            documents = payload.get("documents", [])
            result = self.ingester.ingest_documents(documents)
            self._write_json(HTTPStatus.OK, result)
            return

        if parsed.path == "/api/query":
            try:
                audio_b64 = payload.get("audio_base64")
                audio_bytes = base64.b64decode(audio_b64) if audio_b64 else None
                request = PipelineRequest(
                    question=payload.get("question", ""),
                    audio_bytes=audio_bytes,
                    audio_filename=payload.get("audio_filename", "query.webm"),
                )
                result = self.harness.run_query(request)
                self._write_json(HTTPStatus.OK, result)
            except Exception as exc:  # noqa: BLE001
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "query_failed", "detail": str(exc)})
            return

        if parsed.path == "/api/benchmark":
            queries = payload.get("queries", [])
            result = self.harness.run_benchmark(queries)
            self._write_json(HTTPStatus.OK, result)
            return

        self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8") or "{}")
        except Exception as exc:  # noqa: BLE001
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json", "detail": str(exc)})
            return None

    def _write_json(self, status: HTTPStatus, data: dict) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_text(self, status: HTTPStatus, data: str, content_type: str) -> None:
        body = data.encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")


def serve(config: AppConfig) -> None:
    harness, ingester = build_app(config)
    ApiHandler.harness = harness
    ApiHandler.ingester = ingester
    ApiHandler.config = config
    server = ThreadingHTTPServer((config.host, config.port), ApiHandler)
    print(f"RAGVoice server listening on http://{config.host}:{config.port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice-enabled RAG demo")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--bootstrap-sample-data", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    config = AppConfig.from_env(host=args.host, port=args.port)
    harness, ingester = build_app(config)

    if args.bootstrap_sample_data:
        sample_path = Path("data/sample_corpus.jsonl")
        if sample_path.exists():
            ingester.ingest_jsonl(sample_path)

    if args.benchmark:
        queries = [
            "What does the system do with speech input?",
            "How are chunking strategies combined?",
            "When should the model refuse to answer?",
            "What is required for latency reporting?",
        ]
        result = harness.run_benchmark(queries)
        print(json.dumps(result, indent=2))
        return

    serve(config)


if __name__ == "__main__":
    main()
