from __future__ import annotations

import logging
from dataclasses import dataclass, field

from asr_gateway.config import settings
from asr_gateway.vad import VADSession

logger = logging.getLogger(__name__)

# Silero VAD expects 512 samples at 16 kHz (32 ms)
VAD_FRAME_SAMPLES = 512
BYTES_PER_SAMPLE = 2  # PCM s16le


@dataclass
class SpeechSegment:
    """A contiguous segment of speech audio."""

    audio: bytes
    start_offset: int  # byte offset in the session's audio stream
    end_offset: int
    start_time: float  # seconds from session start
    end_time: float


class AudioChunker:
    """VAD-driven speech segmentation.

    Feeds audio frames to Silero VAD and emits SpeechSegments
    when an end-of-utterance silence is detected.
    """

    def __init__(self, sample_rate: int | None = None) -> None:
        self.sample_rate = sample_rate or settings.vad_sample_rate
        self.vad = VADSession(self.sample_rate)

        self._frame_size = VAD_FRAME_SAMPLES * BYTES_PER_SAMPLE  # bytes per VAD frame
        self._buffer = bytearray()  # Partial frame accumulator

        # Speech state
        self._in_speech = False
        self._speech_frames: list[bytes] = []
        self._speech_start_offset: int = 0
        self._current_offset: int = 0  # running byte offset

        # Silence tracking
        min_silence_frames = (
            settings.vad_min_silence_duration_ms * self.sample_rate
        ) // (1000 * VAD_FRAME_SAMPLES)
        self._min_silence_frames = max(1, min_silence_frames)
        self._silence_frame_count = 0

        min_speech_frames = (
            settings.vad_min_speech_duration_ms * self.sample_rate
        ) // (1000 * VAD_FRAME_SAMPLES)
        self._min_speech_frames = max(1, min_speech_frames)

    def feed(self, audio: bytes) -> list[SpeechSegment]:
        """Feed raw PCM audio bytes. Returns any completed speech segments."""
        self._buffer.extend(audio)
        segments: list[SpeechSegment] = []

        while len(self._buffer) >= self._frame_size:
            frame = bytes(self._buffer[: self._frame_size])
            del self._buffer[: self._frame_size]
            seg = self._process_frame(frame)
            if seg is not None:
                segments.append(seg)

        return segments

    def flush(self) -> SpeechSegment | None:
        """Flush any remaining speech when the session ends."""
        if self._in_speech and len(self._speech_frames) >= self._min_speech_frames:
            return self._emit_segment()
        self._speech_frames.clear()
        self._in_speech = False
        return None

    def _process_frame(self, frame: bytes) -> SpeechSegment | None:
        prob = self.vad.process_chunk(frame)
        is_speech = self.vad.is_speech(prob)

        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                self._speech_start_offset = self._current_offset
                self._speech_frames.clear()
                self._silence_frame_count = 0
            self._speech_frames.append(frame)
            self._silence_frame_count = 0
        else:
            if self._in_speech:
                self._speech_frames.append(frame)  # include trailing silence
                self._silence_frame_count += 1

                if self._silence_frame_count >= self._min_silence_frames:
                    if len(self._speech_frames) >= self._min_speech_frames:
                        self._current_offset += self._frame_size
                        return self._emit_segment()
                    else:
                        # Too short, discard
                        self._in_speech = False
                        self._speech_frames.clear()

        self._current_offset += self._frame_size
        return None

    def _emit_segment(self) -> SpeechSegment:
        audio = b"".join(self._speech_frames)
        end_offset = self._speech_start_offset + len(audio)
        bytes_per_sec = self.sample_rate * BYTES_PER_SAMPLE
        seg = SpeechSegment(
            audio=audio,
            start_offset=self._speech_start_offset,
            end_offset=end_offset,
            start_time=self._speech_start_offset / bytes_per_sec,
            end_time=end_offset / bytes_per_sec,
        )
        self._in_speech = False
        self._speech_frames.clear()
        self._silence_frame_count = 0
        logger.debug(
            "Speech segment: %.2fs - %.2fs (%d bytes)",
            seg.start_time,
            seg.end_time,
            len(audio),
        )
        return seg
