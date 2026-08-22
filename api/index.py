import os
import json
import base64
from pathlib import Path
from ragvoice.config import AppConfig
from ragvoice.harness import PipelineHarness, PipelineRequest
from ragvoice.ingest import CorpusIngester

# Redirect database path to a writable directory on Vercel
if "RAGVOICE_DB_PATH" not in os.environ:
    os.environ["RAGVOICE_DB_PATH"] = "/tmp/ragvoice.db"

config = AppConfig.from_env()
harness = PipelineHarness(config)
ingester = CorpusIngester(config)

# Check if DB is empty and bootstrap it
if ingester.store.count() == 0:
    sample_path = Path("data/sample_corpus.jsonl")
    if sample_path.exists():
        ingester.ingest_jsonl(sample_path)

def app(environ, start_response):
    path = environ.get('PATH_INFO', '')
    method = environ.get('REQUEST_METHOD', 'GET')

    headers = [
        ('Content-Type', 'application/json; charset=utf-8'),
        ('Access-Control-Allow-Origin', '*'),
        ('Access-Control-Allow-Headers', 'Content-Type'),
        ('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    ]

    if method == 'OPTIONS':
        start_response('204 No Content', [
            ('Content-Length', '0'),
            ('Access-Control-Allow-Origin', '*'),
            ('Access-Control-Allow-Headers', 'Content-Type'),
            ('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        ])
        return [b'']

    try:
        if method == 'GET':
            if path == '/health':
                body = json.dumps({"ok": True}).encode('utf-8')
                start_response('200 OK', headers + [('Content-Length', str(len(body)))])
                return [body]
            elif path == '/api/config':
                body = json.dumps(config.as_public_dict()).encode('utf-8')
                start_response('200 OK', headers + [('Content-Length', str(len(body)))])
                return [body]

        elif method == 'POST':
            content_length = int(environ.get('CONTENT_LENGTH') or 0)
            raw_body = environ['wsgi.input'].read(content_length)
            payload = json.loads(raw_body.decode('utf-8') or '{}')

            if path == '/api/ingest':
                documents = payload.get("documents", [])
                result = ingester.ingest_documents(documents)
                body = json.dumps(result).encode('utf-8')
                start_response('200 OK', headers + [('Content-Length', str(len(body)))])
                return [body]

            elif path == '/api/query':
                audio_b64 = payload.get("audio_base64")
                audio_bytes = base64.b64decode(audio_b64) if audio_b64 else None
                request = PipelineRequest(
                    question=payload.get("question", ""),
                    audio_bytes=audio_bytes,
                    audio_filename=payload.get("audio_filename", "query.webm"),
                )
                result = harness.run_query(request)
                body = json.dumps(result).encode('utf-8')
                start_response('200 OK', headers + [('Content-Length', str(len(body)))])
                return [body]

            elif path == '/api/benchmark':
                queries = payload.get("queries", [])
                result = harness.run_benchmark(queries)
                body = json.dumps(result).encode('utf-8')
                start_response('200 OK', headers + [('Content-Length', str(len(body)))])
                return [body]

        # Route fallback
        body = json.dumps({"error": "not_found"}).encode('utf-8')
        start_response('404 Not Found', headers + [('Content-Length', str(len(body)))])
        return [body]

    except Exception as e:
        body = json.dumps({"error": "internal_server_error", "detail": str(e)}).encode('utf-8')
        start_response('500 Internal Server Error', headers + [('Content-Length', str(len(body)))])
        return [body]
