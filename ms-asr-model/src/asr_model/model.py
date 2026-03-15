from __future__ import annotations

import io
import logging

import numpy as np
from faster_whisper import WhisperModel

from asr_model.config import settings

logger = logging.getLogger(__name__)


class ASRModel:
    """Wrapper around faster-whisper for transcription."""

    def __init__(self) -> None:
        self._model: WhisperModel | None = None

    def load(self) -> None:
        device = settings.whisper_device
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        compute_type = settings.whisper_compute_type
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        logger.info(
            "Loading faster-whisper model=%s device=%s compute_type=%s",
            settings.whisper_model_size,
            device,
            compute_type,
        )
        self._model = WhisperModel(
            settings.whisper_model_size,
            device=device,
            compute_type=compute_type,
            download_root=settings.whisper_download_root,
        )
        logger.info("Model loaded successfully")

    def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> dict:
        """Transcribe raw PCM 16-bit LE audio bytes.

        Returns dict with keys: text, words, duration, language.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        lang = language or settings.default_language

        # Convert raw PCM s16le bytes to float32 numpy array
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if sample_rate != 16000:
            # faster-whisper expects 16 kHz; resample if needed
            audio_np = self._resample(audio_np, sample_rate, 16000)

        duration = len(audio_np) / 16000.0

        segments, info = self._model.transcribe(
            audio_np,
            language=lang,
            word_timestamps=True,
            vad_filter=False,  # We do our own VAD in the gateway
        )

        words = []
        text_parts = []
        for segment in segments:
            if segment.words:
                for w in segment.words:
                    words.append(
                        {
                            "word": w.word.strip(),
                            "start_time": w.start,
                            "end_time": w.end,
                            "confidence": w.probability,
                        }
                    )
            text_parts.append(segment.text.strip())

        text = " ".join(text_parts)
        return {
            "text": text,
            "words": words,
            "duration": duration,
            "language": info.language,
        }

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear interpolation resampling."""
        ratio = target_sr / orig_sr
        n_samples = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, n_samples)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


# Singleton
asr_model = ASRModel()
