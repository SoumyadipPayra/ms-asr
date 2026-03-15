from __future__ import annotations

import logging
import queue
import threading
import time

import grpc

import asr_gateway_pb2
import asr_gateway_pb2_grpc

from asr_gateway.audio_store import audio_store
from asr_gateway.pipeline import TranscriptResult, processing_thread
from asr_gateway.session import AudioConfig, Session

logger = logging.getLogger(__name__)


class AsrGatewayServicer(asr_gateway_pb2_grpc.AsrGatewayServicer):
    """Implements AsrGateway.Recognize bidi streaming RPC."""

    def Recognize(self, request_iterator, context: grpc.ServicerContext):
        session: Session | None = None
        result_queue: queue.Queue[TranscriptResult | None] = queue.Queue()

        def on_result(transcript: TranscriptResult) -> None:
            result_queue.put(transcript)

        def response_sender():
            """Yield ServerMessages from the result queue."""
            while True:
                item = result_queue.get()
                if item is None:
                    break
                yield _transcript_to_message(item)

        try:
            for client_msg in request_iterator:
                msg_type = client_msg.WhichOneof("message")

                if msg_type == "start":
                    if session is not None:
                        yield _error_message(400, "Session already started")
                        continue

                    start = client_msg.start
                    config = AudioConfig(
                        sample_rate=start.config.sample_rate or 16000,
                        encoding=start.config.encoding or "pcm_s16le",
                        channels=start.config.channels or 1,
                    )
                    session = Session(session_id=start.session_id or None)
                    session.start(config)

                    # Set up Redis stream
                    audio_store.create_session_stream(session.session_id)

                    # Start processing thread
                    thread = threading.Thread(
                        target=processing_thread,
                        args=(session, on_result),
                        daemon=True,
                        name=f"pipeline-{session.session_id}",
                    )
                    thread.start()
                    session.set_processing_thread(thread)

                    yield asr_gateway_pb2.ServerMessage(
                        started=asr_gateway_pb2.RecognitionStarted(
                            session_id=session.session_id
                        )
                    )

                elif msg_type == "audio":
                    if session is None:
                        yield _error_message(400, "Send StartRecognition first")
                        continue

                    audio_bytes = client_msg.audio.audio
                    if not audio_bytes:
                        continue

                    seq = session.next_seq()
                    offset = session.advance_offset(len(audio_bytes))
                    audio_store.append_audio(
                        session.session_id,
                        seq=seq,
                        audio=audio_bytes,
                        timestamp=time.time(),
                        byte_offset=offset,
                    )

                elif msg_type == "stop":
                    break

            # Drain: signal stop, wait for processing thread, send remaining results
            if session is not None:
                session.stop()
                session.join_processing_thread(timeout=15.0)
                result_queue.put(None)  # sentinel

                # Yield remaining results
                while True:
                    try:
                        item = result_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        break
                    yield _transcript_to_message(item)

                yield asr_gateway_pb2.ServerMessage(
                    stopped=asr_gateway_pb2.RecognitionStopped(
                        session_id=session.session_id
                    )
                )
                audio_store.cleanup_session(session.session_id)

        except Exception:
            logger.exception("Error in Recognize stream")
            if session is not None:
                session.stop()
                session.join_processing_thread(timeout=15.0)
                result_queue.put(None)
                audio_store.cleanup_session(session.session_id)
            yield _error_message(500, "Internal server error")


def _transcript_to_message(t: TranscriptResult) -> asr_gateway_pb2.ServerMessage:
    words = [
        asr_gateway_pb2.WordInfo(
            word=w["word"],
            start_time=w["start_time"],
            end_time=w["end_time"],
            confidence=w["confidence"],
        )
        for w in t.words
    ]
    return asr_gateway_pb2.ServerMessage(
        transcript=asr_gateway_pb2.Transcript(
            text=t.text,
            words=words,
            start_time=t.start_time,
            end_time=t.end_time,
            utterance_index=t.utterance_index,
        )
    )


def _error_message(code: int, message: str) -> asr_gateway_pb2.ServerMessage:
    return asr_gateway_pb2.ServerMessage(
        error=asr_gateway_pb2.Error(code=code, message=message)
    )
