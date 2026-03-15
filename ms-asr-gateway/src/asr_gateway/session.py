from __future__ import annotations

import enum
import logging
import threading
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class SessionState(enum.Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    STOPPED = "STOPPED"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    encoding: str = "pcm_s16le"
    channels: int = 1

    @property
    def bytes_per_sample(self) -> int:
        if self.encoding == "pcm_f32le":
            return 4
        return 2  # pcm_s16le


class Session:
    """Per-connection session managing state, sequencing, and thread lifecycle."""

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex
        self.state = SessionState.IDLE
        self.config = AudioConfig()
        self.seq: int = 0
        self.byte_offset: int = 0
        self.utterance_index: int = 0
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._processing_thread: threading.Thread | None = None

    def start(self, config: AudioConfig | None = None) -> None:
        if config is not None:
            self.config = config
        with self._lock:
            self.state = SessionState.LISTENING
        logger.info("Session %s started (sr=%d)", self.session_id, self.config.sample_rate)

    def next_seq(self) -> int:
        with self._lock:
            self.seq += 1
            return self.seq

    def advance_offset(self, nbytes: int) -> int:
        """Advance byte offset and return the value before advancing."""
        with self._lock:
            current = self.byte_offset
            self.byte_offset += nbytes
            return current

    def next_utterance(self) -> int:
        with self._lock:
            idx = self.utterance_index
            self.utterance_index += 1
            return idx

    def set_processing(self) -> None:
        with self._lock:
            self.state = SessionState.PROCESSING

    def set_listening(self) -> None:
        with self._lock:
            self.state = SessionState.LISTENING

    def stop(self) -> None:
        with self._lock:
            self.state = SessionState.STOPPED
        self._stop_event.set()
        logger.info("Session %s stopped", self.session_id)

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def wait_stop(self, timeout: float | None = None) -> bool:
        return self._stop_event.wait(timeout)

    def set_processing_thread(self, thread: threading.Thread) -> None:
        self._processing_thread = thread

    def join_processing_thread(self, timeout: float = 10.0) -> None:
        if self._processing_thread is not None and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=timeout)
