from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection

from asr_gateway.audio_store import audio_store
from asr_gateway.pipeline import TranscriptResult, processing_thread
from asr_gateway.session import AudioConfig, Session

logger = logging.getLogger(__name__)


async def ws_handler(websocket: ServerConnection) -> None:
    """Handle a single WebSocket connection.

    Protocol:
      - Text frame ``{"type": "start", "config": {...}}`` -> StartRecognition
      - Binary frame -> AudioData
      - Text frame ``{"type": "stop"}`` -> StopRecognition
      - Server sends text frames with transcript JSON
    """
    session: Session | None = None
    result_queue: queue.Queue[TranscriptResult | None] = queue.Queue()
    loop = asyncio.get_running_loop()

    def on_result(transcript: TranscriptResult) -> None:
        result_queue.put(transcript)

    async def send_results() -> None:
        """Background task: drain result_queue and send to client."""
        while True:
            # Poll the thread-safe queue from async context
            try:
                item = await loop.run_in_executor(None, result_queue.get, True, 0.2)
            except queue.Empty:
                if session is not None and session.is_stopped:
                    # Drain any remaining
                    while not result_queue.empty():
                        item = result_queue.get_nowait()
                        if item is None:
                            return
                        await websocket.send(json.dumps(_transcript_to_dict(item)))
                    return
                continue
            if item is None:
                return
            await websocket.send(json.dumps(_transcript_to_dict(item)))

    sender_task: asyncio.Task | None = None

    try:
        async for message in websocket:
            if isinstance(message, str):
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "start":
                    if session is not None:
                        await websocket.send(
                            json.dumps({"type": "error", "code": 400, "message": "Already started"})
                        )
                        continue

                    config_data = data.get("config", {})
                    config = AudioConfig(
                        sample_rate=config_data.get("sample_rate", 16000),
                        encoding=config_data.get("encoding", "pcm_s16le"),
                        channels=config_data.get("channels", 1),
                    )
                    session = Session(session_id=data.get("session_id"))
                    session.start(config)

                    audio_store.create_session_stream(session.session_id)

                    thread = threading.Thread(
                        target=processing_thread,
                        args=(session, on_result),
                        daemon=True,
                        name=f"pipeline-{session.session_id}",
                    )
                    thread.start()
                    session.set_processing_thread(thread)

                    sender_task = asyncio.create_task(send_results())

                    await websocket.send(
                        json.dumps({"type": "started", "session_id": session.session_id})
                    )

                elif msg_type == "stop":
                    break

            elif isinstance(message, bytes):
                if session is None:
                    await websocket.send(
                        json.dumps({"type": "error", "code": 400, "message": "Send start first"})
                    )
                    continue

                seq = session.next_seq()
                offset = session.advance_offset(len(message))
                audio_store.append_audio(
                    session.session_id,
                    seq=seq,
                    audio=message,
                    timestamp=time.time(),
                    byte_offset=offset,
                )

    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")
    except Exception:
        logger.exception("WebSocket handler error")
    finally:
        if session is not None:
            session.stop()
            session.join_processing_thread(timeout=15.0)
            result_queue.put(None)

            if sender_task is not None:
                await sender_task

            try:
                await websocket.send(
                    json.dumps({"type": "stopped", "session_id": session.session_id})
                )
            except Exception:
                pass

            audio_store.cleanup_session(session.session_id)


def _transcript_to_dict(t: TranscriptResult) -> dict:
    return {
        "type": "transcript",
        "text": t.text,
        "words": [
            {
                "word": w["word"],
                "start_time": w["start_time"],
                "end_time": w["end_time"],
                "confidence": w["confidence"],
            }
            for w in t.words
        ],
        "start_time": t.start_time,
        "end_time": t.end_time,
        "utterance_index": t.utterance_index,
    }
