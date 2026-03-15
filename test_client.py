#!/usr/bin/env python3
"""Test client for ms-asr services.

Sends audio1.wav via WebSocket to the gateway and prints transcription results.
Converts stereo 44100 Hz to mono 16000 Hz PCM s16le before sending.
"""

import asyncio
import audioop
import json
import struct
import sys
import wave

import websockets


GATEWAY_WS_URL = "ws://localhost:8765"
WAV_PATH = "sample/audio1.wav"
TARGET_SR = 16000
CHUNK_DURATION_MS = 100  # send 100ms chunks to simulate streaming


def load_and_convert_wav(path: str) -> tuple[bytes, int]:
    """Load a WAV file and convert to mono 16-bit PCM at 16 kHz."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    print(f"Input: {n_channels}ch, {framerate} Hz, {sampwidth * 8}-bit, {len(raw)} bytes")

    # Convert to mono if stereo
    if n_channels == 2:
        raw = audioop.tomono(raw, sampwidth, 1, 1)

    # Resample to target sample rate if needed
    if framerate != TARGET_SR:
        raw, _ = audioop.ratecv(raw, sampwidth, 1, framerate, TARGET_SR, None)

    # Ensure 16-bit
    if sampwidth != 2:
        raw = audioop.lin2lin(raw, sampwidth, 2)

    duration = len(raw) / (TARGET_SR * 2)
    print(f"Converted: mono, {TARGET_SR} Hz, 16-bit, {len(raw)} bytes ({duration:.2f}s)")
    return raw, TARGET_SR


async def stream_audio():
    audio_data, sample_rate = load_and_convert_wav(WAV_PATH)

    chunk_size = int(TARGET_SR * 2 * CHUNK_DURATION_MS / 1000)  # bytes per chunk
    print(f"Chunk size: {chunk_size} bytes ({CHUNK_DURATION_MS}ms)")
    print(f"Connecting to {GATEWAY_WS_URL}...")

    async with websockets.connect(GATEWAY_WS_URL) as ws:
        # 1. Send start message
        start_msg = {
            "type": "start",
            "config": {
                "sample_rate": sample_rate,
                "encoding": "pcm_s16le",
                "channels": 1,
            },
        }
        await ws.send(json.dumps(start_msg))
        response = await ws.recv()
        resp = json.loads(response)
        print(f"<< {resp}")

        if resp.get("type") != "started":
            print(f"ERROR: Expected 'started', got {resp}")
            return

        session_id = resp["session_id"]
        print(f"Session started: {session_id}")

        # 2. Stream audio in chunks (simulate real-time)
        offset = 0
        chunk_count = 0
        print("Streaming audio...")

        async def receive_results():
            """Receive and print transcription results."""
            try:
                async for msg in ws:
                    data = json.loads(msg)
                    if data["type"] == "transcript":
                        print(f"\n>> TRANSCRIPT (utterance {data['utterance_index']}):")
                        print(f"   Text: {data['text']}")
                        print(f"   Time: {data['start_time']:.2f}s - {data['end_time']:.2f}s")
                        if data.get("words"):
                            words_str = " | ".join(
                                f"{w['word']}({w['confidence']:.2f})"
                                for w in data["words"][:10]
                            )
                            print(f"   Words: {words_str}")
                            if len(data["words"]) > 10:
                                print(f"   ... and {len(data['words']) - 10} more words")
                    elif data["type"] == "stopped":
                        print(f"\n<< Session stopped: {data['session_id']}")
                        break
                    elif data["type"] == "error":
                        print(f"\n<< ERROR: {data['message']}")
                        break
                    else:
                        print(f"\n<< {data}")
            except websockets.exceptions.ConnectionClosed:
                pass

        # Start receiving in the background
        receiver = asyncio.create_task(receive_results())

        while offset < len(audio_data):
            chunk = audio_data[offset : offset + chunk_size]
            await ws.send(chunk)
            offset += len(chunk)
            chunk_count += 1

            # Simulate real-time streaming pace
            await asyncio.sleep(CHUNK_DURATION_MS / 1000.0)

        print(f"\nSent {chunk_count} chunks ({offset} bytes)")

        # 3. Send stop
        await ws.send(json.dumps({"type": "stop"}))
        print("Stop sent, waiting for final results...")

        # Wait for receiver to finish
        await asyncio.wait_for(receiver, timeout=60.0)

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(stream_audio())
