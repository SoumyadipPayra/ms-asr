from __future__ import annotations

import logging
import sys

import grpc

# Generated stubs are added to sys.path in main.py
import asr_model_pb2
import asr_model_pb2_grpc

from asr_model.model import asr_model

logger = logging.getLogger(__name__)


class AsrModelServicer(asr_model_pb2_grpc.AsrModelServicer):
    """Implements the AsrModel.Transcribe unary RPC."""

    def Transcribe(
        self,
        request: asr_model_pb2.TranscribeRequest,
        context: grpc.ServicerContext,
    ) -> asr_model_pb2.TranscribeResponse:
        audio_len = len(request.audio)
        language = request.language or "en"
        sample_rate = request.sample_rate or 16000

        logger.debug(
            "Transcribe request: %d bytes, sr=%d, lang=%s",
            audio_len,
            sample_rate,
            language,
        )

        if audio_len == 0:
            return asr_model_pb2.TranscribeResponse(
                text="", words=[], duration=0.0, language=language
            )

        try:
            result = asr_model.transcribe(
                audio_bytes=request.audio,
                sample_rate=sample_rate,
                language=language,
            )
        except Exception:
            logger.exception("Transcription failed")
            context.abort(grpc.StatusCode.INTERNAL, "Transcription failed")

        words = [
            asr_model_pb2.WordInfo(
                word=w["word"],
                start_time=w["start_time"],
                end_time=w["end_time"],
                confidence=w["confidence"],
            )
            for w in result["words"]
        ]

        return asr_model_pb2.TranscribeResponse(
            text=result["text"],
            words=words,
            duration=result["duration"],
            language=result["language"],
        )
