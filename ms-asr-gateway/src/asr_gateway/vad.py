from __future__ import annotations

import copy
import io
import logging
import threading

import numpy as np
import torch

from asr_gateway.config import settings

logger = logging.getLogger(__name__)

# Serialized model bytes — loaded once, used to create per-session copies
_model_buffer: bytes | None = None
_load_lock = threading.Lock()


def load_vad_model() -> None:
    """Load Silero VAD model from torch hub (called once at startup).

    Serializes the model to a buffer so each session can deserialize
    its own independent copy with separate internal state.
    """
    global _model_buffer
    if _model_buffer is not None:
        return

    with _load_lock:
        if _model_buffer is not None:
            return
        logger.info("Loading Silero VAD model...")
        model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        buf = io.BytesIO()
        torch.jit.save(model, buf)
        _model_buffer = buf.getvalue()
        logger.info("Silero VAD model loaded and serialized (%d bytes)", len(_model_buffer))


def _create_model_copy() -> torch.jit.ScriptModule:
    """Create a fresh model instance from the serialized buffer."""
    if _model_buffer is None:
        raise RuntimeError("VAD model not loaded. Call load_vad_model() first.")
    buf = io.BytesIO(_model_buffer)
    return torch.jit.load(buf)


class VADSession:
    """Per-connection VAD state.

    Each instance gets its own model copy so internal hidden state
    is completely independent — no contention between threads.
    """

    def __init__(self, sample_rate: int | None = None) -> None:
        self.sample_rate = sample_rate or settings.vad_sample_rate
        self.threshold = settings.vad_threshold
        self._model = _create_model_copy()
        self._last_prob: float = 0.0

    def process_chunk(self, audio_bytes: bytes) -> float:
        """Process a chunk of raw PCM s16le audio.

        Returns the speech probability (0.0 - 1.0).
        """
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np)

        with torch.no_grad():
            out = self._model(audio_tensor, self.sample_rate)

        self._last_prob = out.item()
        return self._last_prob

    @property
    def last_probability(self) -> float:
        return self._last_prob

    def is_speech(self, prob: float | None = None) -> bool:
        p = prob if prob is not None else self._last_prob
        return p >= self.threshold

    def reset(self) -> None:
        """Reset hidden state for a new utterance."""
        self._model.reset_states()
        self._last_prob = 0.0
