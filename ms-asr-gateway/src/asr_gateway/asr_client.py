from __future__ import annotations

import logging

import grpc

import asr_model_pb2
import asr_model_pb2_grpc

from asr_gateway.config import settings

logger = logging.getLogger(__name__)


class ASRClient:
    """gRPC client to the ms-asr-model service."""

    def __init__(self) -> None:
        self._channel: grpc.Channel | None = None
        self._stub: asr_model_pb2_grpc.AsrModelStub | None = None

    def connect(self) -> None:
        addr = f"{settings.asr_model_host}:{settings.asr_model_port}"
        self._channel = grpc.insecure_channel(
            addr,
            options=[
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ],
        )
        self._stub = asr_model_pb2_grpc.AsrModelStub(self._channel)
        logger.info("ASR client connected to %s", addr)

    @property
    def stub(self) -> asr_model_pb2_grpc.AsrModelStub:
        if self._stub is None:
            raise RuntimeError("ASRClient not connected")
        return self._stub

    def transcribe(
        self,
        audio: bytes,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> dict:
        """Send audio to ms-asr-model and return the transcription result.

        Returns dict with keys: text, words, duration, language.
        """
        request = asr_model_pb2.TranscribeRequest(
            audio=audio,
            sample_rate=sample_rate,
            language=language,
        )
        response = self.stub.Transcribe(
            request, timeout=settings.asr_model_timeout
        )
        words = [
            {
                "word": w.word,
                "start_time": w.start_time,
                "end_time": w.end_time,
                "confidence": w.confidence,
            }
            for w in response.words
        ]
        return {
            "text": response.text,
            "words": words,
            "duration": response.duration,
            "language": response.language,
        }

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None


# Singleton
asr_client = ASRClient()
