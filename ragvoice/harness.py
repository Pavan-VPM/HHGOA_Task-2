from __future__ import annotations

from dataclasses import dataclass

from ragvoice.answering import answer_from_context
from ragvoice.config import AppConfig
from ragvoice.guardrails import check_query, grounded_enough
from ragvoice.ingest import CorpusIngester
from ragvoice.latency import StageTimer, percentile
from ragvoice.stt import build_stt


@dataclass
class PipelineRequest:
    question: str
    audio_bytes: bytes | None = None
    audio_filename: str = "query.webm"


class PipelineHarness:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.stt = build_stt(config.stt_provider)
        self.ingester = CorpusIngester(config)
        self.store = self.ingester.store

    def _transcribe_if_needed(self, request: PipelineRequest) -> tuple[str, dict]:
        if request.question.strip():
            return request.question.strip(), {"used_audio": False, "provider": self.stt.provider_name}
        timer = StageTimer()
        transcript = self.stt.transcribe(request.audio_bytes, request.audio_filename)
        return transcript.text, {"used_audio": True, "provider": transcript.provider, "latency_ms": timer.stop_ms()}

    def run_query(self, request: PipelineRequest) -> dict:
        total_timer = StageTimer()
        transcript, stt_meta = self._transcribe_if_needed(request)

        guard_timer = StageTimer()
        guard = check_query(transcript)
        guard_ms = guard_timer.stop_ms()
        if not guard.allowed:
            return {
                "ok": False,
                "transcript": transcript,
                "answer": guard.reason,
                "guardrail": guard.reason,
                "retrievals": [],
                "latency_ms": {"guardrail": guard_ms, "total": total_timer.stop_ms()},
                "stt": stt_meta,
            }

        retrieval_timer = StageTimer()
        retrievals = self.store.search(transcript, top_k=self.config.top_k)
        retrieval_ms = retrieval_timer.stop_ms()
        if not grounded_enough(retrievals):
            return {
                "ok": False,
                "transcript": transcript,
                "answer": "I do not have enough grounded evidence in the retrieved context to answer confidently.",
                "guardrail": "grounding_failed",
                "retrievals": retrievals,
                "latency_ms": {
                    "guardrail": guard_ms,
                    "retrieval": retrieval_ms,
                    "total": total_timer.stop_ms(),
                },
                "stt": stt_meta,
            }

        answer_timer = StageTimer()
        answer = answer_from_context(transcript, retrievals)
        answer_ms = answer_timer.stop_ms()
        return {
            "ok": True,
            "transcript": transcript,
            "answer": answer,
            "guardrail": "ok",
            "retrievals": retrievals,
            "latency_ms": {
                "guardrail": guard_ms,
                "retrieval": retrieval_ms,
                "answer": answer_ms,
                "total": total_timer.stop_ms(),
            },
            "stt": stt_meta,
        }

    def run_benchmark(self, queries: list[str]) -> dict:
        runs = [self.run_query(PipelineRequest(question=query)) for query in queries]
        totals = [run["latency_ms"]["total"] for run in runs]
        return {
            "queries": len(queries),
            "p50_ms": percentile(totals, 50),
            "p70_ms": percentile(totals, 70),
            "p100_ms": percentile(totals, 100),
            "runs": runs,
        }
