from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranscriptResult:
    text: str
    provider: str


class BaseSpeechToText:
    provider_name = "base"

    def transcribe(self, audio_bytes: bytes | None, filename: str) -> TranscriptResult:
        raise NotImplementedError


class MockSpeechToText(BaseSpeechToText):
    provider_name = "mock"

    def transcribe(self, audio_bytes: bytes | None, filename: str) -> TranscriptResult:
        if audio_bytes:
            try:
                return MacOSSpeechToText().transcribe(audio_bytes, filename)
            except Exception:
                pass
        return TranscriptResult(
            text="Audio transcription is not configured. Provide a transcript in the question field or configure a real STT provider.",
            provider=self.provider_name,
        )


class MacOSSpeechToText(BaseSpeechToText):
    provider_name = "macos"

    def transcribe(self, audio_bytes: bytes | None, filename: str) -> TranscriptResult:
        if not audio_bytes:
            raise RuntimeError("No audio bytes provided for macOS transcription")

        with tempfile.TemporaryDirectory(prefix="ragvoice-stt-") as tmpdir:
            input_path = Path(tmpdir) / filename
            wav_path = Path(tmpdir) / "query.wav"
            input_path.write_bytes(audio_bytes)
            self._convert_to_wav(input_path, wav_path)
            text = self._transcribe_wav(wav_path)
            if not text:
                raise RuntimeError("macOS Speech returned an empty transcript")
            return TranscriptResult(text=text, provider=self.provider_name)

    def _convert_to_wav(self, input_path: Path, wav_path: Path) -> None:
        ffmpeg = os.getenv("RAGVOICE_FFMPEG_BIN", "/opt/homebrew/bin/ffmpeg")
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=20)
        except FileNotFoundError as exc:
            raise RuntimeError(f"ffmpeg not found at {ffmpeg}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("ffmpeg timed out while converting audio") from exc

        if completed.returncode != 0 or not wav_path.exists():
            detail = completed.stderr.strip() or completed.stdout.strip() or "unknown ffmpeg error"
            raise RuntimeError(f"ffmpeg conversion failed: {detail}")

    def _transcribe_wav(self, wav_path: Path) -> str:
        import AVFoundation
        import Foundation
        import Speech

        status = self._request_authorization(Speech)
        if status != Speech.SFSpeechRecognizerAuthorizationStatusAuthorized:
            raise RuntimeError(f"macOS Speech authorization failed with status {status}")

        recognizer = Speech.SFSpeechRecognizer.alloc().init()
        if recognizer is None:
            raise RuntimeError("Unable to initialize macOS Speech recognizer")
        if hasattr(recognizer, "isAvailable") and not recognizer.isAvailable():
            raise RuntimeError("macOS Speech recognizer is unavailable")

        url = Foundation.NSURL.fileURLWithPath_(str(wav_path))
        request = Speech.SFSpeechURLRecognitionRequest.alloc().initWithURL_(url)
        if hasattr(request, "setShouldReportPartialResults_"):
            request.setShouldReportPartialResults_(False)
        if hasattr(request, "setAddsPunctuation_"):
            request.setAddsPunctuation_(True)
        if hasattr(recognizer, "supportsOnDeviceRecognition") and recognizer.supportsOnDeviceRecognition():
            if hasattr(request, "setRequiresOnDeviceRecognition_"):
                request.setRequiresOnDeviceRecognition_(True)

        done = threading.Event()
        state: dict[str, str | None] = {"text": None, "error": None}

        def handler(result, error) -> None:
            if error is not None:
                state["error"] = str(error)
                done.set()
                return
            if result is not None:
                best = result.bestTranscription()
                if best is not None:
                    state["text"] = str(best.formattedString()).strip()
                if result.isFinal():
                    done.set()

        task = recognizer.recognitionTaskWithRequest_resultHandler_(request, handler)
        try:
            timeout_at = Foundation.NSDate.dateWithTimeIntervalSinceNow_(15.0)
            while not done.is_set():
                if Foundation.NSDate.date().timeIntervalSinceDate_(timeout_at) >= 0:
                    raise RuntimeError("macOS Speech timed out")
                Foundation.NSRunLoop.currentRunLoop().runUntilDate_(
                    Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.1)
                )
        finally:
            if task is not None and hasattr(task, "cancel"):
                task.cancel()

        if state["error"]:
            raise RuntimeError(f"macOS Speech failed: {state['error']}")
        return (state["text"] or "").strip()

    def _request_authorization(self, speech_module) -> int:
        current = speech_module.SFSpeechRecognizer.authorizationStatus()
        if current != speech_module.SFSpeechRecognizerAuthorizationStatusNotDetermined:
            return current

        done = threading.Event()
        state = {"status": current}

        def callback(status) -> None:
            state["status"] = status
            done.set()

        speech_module.SFSpeechRecognizer.requestAuthorization_(callback)
        if not done.wait(timeout=10):
            raise RuntimeError("Timed out waiting for macOS Speech authorization")
        return state["status"]


def _encode_multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    boundary = "----WebKitFormBoundaryRAGVoiceSTT"
    body = []
    for key, val in fields.items():
        body.append(f"--{boundary}\r\n".encode("utf-8"))
        body.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.append(f"{val}\r\n".encode("utf-8"))
    for key, (filename, content, mime_type) in files.items():
        body.append(f"--{boundary}\r\n".encode("utf-8"))
        body.append(f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode("utf-8"))
        body.append(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
        body.append(content)
        body.append(b"\r\n")
    body.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(body), f"multipart/form-data; boundary={boundary}"


def _request_with_retry(request: urllib.request.Request, retries: int = 3, delay: float = 1.0) -> dict:
    import time
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise exc
            time.sleep(delay * (2 ** attempt))
    return {}


class ElevenLabsSpeechToText(BaseSpeechToText):
    provider_name = "elevenlabs"

    def transcribe(self, audio_bytes: bytes | None, filename: str) -> TranscriptResult:
        api_key = os.getenv("RAGVOICE_ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("Missing RAGVOICE_ELEVENLABS_API_KEY")
        if not audio_bytes:
            raise RuntimeError("No audio bytes provided for ElevenLabs transcription")

        fields = {"model_id": "scribe_v2"}
        files = {"file": (filename, audio_bytes, "audio/webm")}
        body, content_type = _encode_multipart(fields, files)

        request = urllib.request.Request(
            url="https://api.elevenlabs.io/v1/speech-to-text",
            data=body,
            method="POST",
            headers={
                "xi-api-key": api_key,
                "Content-Type": content_type,
            },
        )
        try:
            payload = _request_with_retry(request)
        except Exception as exc:
            raise RuntimeError(f"ElevenLabs transcription failed: {exc}") from exc
        return TranscriptResult(text=payload.get("text", "").strip(), provider=self.provider_name)


class SarvamSpeechToText(BaseSpeechToText):
    provider_name = "sarvam"

    def transcribe(self, audio_bytes: bytes | None, filename: str) -> TranscriptResult:
        api_key = os.getenv("RAGVOICE_SARVAM_API_KEY")
        if not api_key:
            raise RuntimeError("Missing RAGVOICE_SARVAM_API_KEY")
        if not audio_bytes:
            raise RuntimeError("No audio bytes provided for Sarvam transcription")

        fields = {"model": "saaras:v1"}
        files = {"file": (filename, audio_bytes, "audio/webm")}
        body, content_type = _encode_multipart(fields, files)

        request = urllib.request.Request(
            url="https://api.sarvam.ai/speech-to-text",
            data=body,
            method="POST",
            headers={
                "api-subscription-key": api_key,
                "Content-Type": content_type,
            },
        )
        try:
            payload = _request_with_retry(request)
        except Exception as exc:
            raise RuntimeError(f"Sarvam transcription failed: {exc}") from exc
        text = payload.get("transcript") or payload.get("text") or ""
        return TranscriptResult(text=text.strip(), provider=self.provider_name)


def build_stt(provider: str) -> BaseSpeechToText:
    mapping = {
        "mock": MockSpeechToText,
        "elevenlabs": ElevenLabsSpeechToText,
        "sarvam": SarvamSpeechToText,
    }
    return mapping.get(provider, MockSpeechToText)()
