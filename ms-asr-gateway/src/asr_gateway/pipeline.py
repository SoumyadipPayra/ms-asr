from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass
from typing import Any, Callable

from asr_gateway.asr_client import asr_client
from asr_gateway.audio_chunker import AudioChunker, BYTES_PER_SAMPLE
from asr_gateway.audio_store import audio_store
from asr_gateway.config import settings
from asr_gateway.post_processor import post_process
from asr_gateway.session import Session, SessionState

logger = logging.getLogger(__name__)


@dataclass
class TranscriptResult:
    """Result pushed back to the I/O handler."""

    text: str
    words: list[dict]
    start_time: float
    end_time: float
    utterance_index: int
    is_final: bool = True


def processing_thread(
    session: Session,
    result_callback: Callable[[TranscriptResult], None],
) -> None:
    """Per-session processing thread.

    Reads audio from Redis Stream, runs VAD segmentation, sends speech
    chunks to the ASR model service, post-processes the transcript,
    acknowledges processed audio in Redis, and pushes results via callback.
    """
    session_id = session.session_id
    consumer_name = f"worker-{session_id}"
    chunker = AudioChunker(sample_rate=session.config.sample_rate)

    # Track entry IDs for acknowledgment
    pending_entry_ids: list[str] = []

    logger.info("Processing thread started for session %s", session_id)

    try:
        while not session.is_stopped:
            # Read from Redis Stream
            entries = audio_store.read_audio(
                session_id,
                consumer_name,
                count=20,
                block_ms=500,
            )

            if not entries:
                continue

            for entry_id, fields in entries:
                audio = fields.get(b"audio", b"")
                pending_entry_ids.append(entry_id)

                # Feed audio to VAD chunker
                segments = chunker.feed(audio)

                for segment in segments:
                    _process_segment(
                        session, segment, pending_entry_ids, result_callback
                    )

        # Flush remaining audio on stop
        final_seg = chunker.flush()
        if final_seg is not None:
            _process_segment(
                session, final_seg, pending_entry_ids, result_callback
            )

        # Ack any remaining entries
        if pending_entry_ids:
            audio_store.ack(session_id, *pending_entry_ids)
            pending_entry_ids.clear()

    except Exception:
        logger.exception("Processing thread error for session %s", session_id)
    finally:
        # Ack any remaining entries so they don't stay in the pending list
        if pending_entry_ids:
            try:
                audio_store.ack(session_id, *pending_entry_ids)
                pending_entry_ids.clear()
            except Exception:
                logger.warning("Failed to ack remaining entries for session %s", session_id)
        logger.info("Processing thread exiting for session %s", session_id)


def _process_segment(
    session: Session,
    segment,
    pending_entry_ids: list[str],
    result_callback: Callable[[TranscriptResult], None],
) -> None:
    """Transcribe a speech segment and deliver the result."""
    session.set_processing()

    try:
        result = asr_client.transcribe(
            audio=segment.audio,
            sample_rate=session.config.sample_rate,
            language=settings.default_language,
        )
    except Exception:
        logger.exception("ASR transcription failed for session %s", session.session_id)
        session.set_listening()
        return

    # Post-process
    audio_duration = segment.end_time - segment.start_time
    result = post_process(result, audio_duration=audio_duration)

    if result["text"]:
        logger.info("Transcript [%s]: %s", session.session_id[:8], result["text"])
        utterance_idx = session.next_utterance()
        transcript = TranscriptResult(
            text=result["text"],
            words=result["words"],
            start_time=segment.start_time,
            end_time=segment.end_time,
            utterance_index=utterance_idx,
        )
        result_callback(transcript)

    # Acknowledge processed audio entries up to this point
    if pending_entry_ids:
        audio_store.ack(session.session_id, *pending_entry_ids)
        pending_entry_ids.clear()

    session.set_listening()
