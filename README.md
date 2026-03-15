# ms-asr

Real-time speech-to-text system built as two microservices communicating over gRPC, with Redis Streams as the audio buffer between them.

## Architecture

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                           ms-asr-gateway                               │
  │                                                                        │
  │  ┌───────────┐    ┌──────────────┐    ┌───────────┐    ┌────────────┐  │
  │  │ WebSocket │    │              │    │ Silero    │    │ Post-      │  │
  │  │ handler   │──▶ │ Redis Stream │──▶ │ VAD       │──▶ │ processor  │  │
  │  │ / gRPC    │    │ (per session)│    │ segmenter │    │            │  │
  │  └───────────┘    └──────────────┘    └─────┬─────┘    └──────┬─────┘  │
  │       ▲                  ▲                  │                 │        │
  │       │                  │ XACK             │ speech          │        │
  │       │                  │ after            │ segment         │        │
  │       │                  │ transcription    ▼                 │        │
  │       │                  │           ┌────────────┐           │        │
  │       │                  └───────────│  gRPC call │           │        │
  │       │                              └──────┬─────┘           │        │
  └───────┼─────────────────────────────────────┼─────────────────┼────────┘
          │                                     │                 │
          │ transcript                          │ audio           │ filter
          │ to client                           ▼                 │ hallucinations,
          │                              ┌─────────────┐          │ fillers
          └──────────────────────────────│ ms-asr-model│◀─────────┘
                                         │ (whisper)   │
                                         └─────────────┘
```

### Services

**ms-asr-gateway** — stateful, client-facing. Accepts audio over gRPC bidi streaming or WebSocket. Each connection gets its own processing thread that:
1. Reads audio chunks from the session's Redis Stream via a consumer group
2. Feeds them through Silero VAD to detect speech boundaries
3. Sends complete speech segments to the model service over gRPC
4. Post-processes the transcript (hallucination filtering, filler removal)
5. Pushes results back to the client

**ms-asr-model** — stateless, runs on GPU or CPU. Wraps [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend). Single unary gRPC endpoint: audio bytes in, transcript with word-level timestamps and confidence scores out. Supports all Whisper model sizes (`tiny` through `large-v3`) and 100+ languages.

**Redis** — durable audio buffer using Streams, one stream per session.

### Data flow

```
client sends audio ──▶ WS/gRPC handler writes to Redis Stream (audio:{session_id})
                             │
                     processing thread reads via consumer group (XREADGROUP)
                             │
                     accumulates entry IDs in pending_entry_ids
                             │
                     feeds audio frames to Silero VAD (512 samples / 32ms per frame)
                             │
                     VAD detects speech end (silence > 700ms)
                             │
                     sends speech segment to ms-asr-model (unary gRPC)
                             │
                     post-processes transcript
                             │
                     XACKs all accumulated entry IDs ──▶ entries leave pending list
                             │
                     pushes transcript to client
```

### Audio acknowledgment strategy

Audio entries in Redis are acknowledged **after each successful transcription**, not based on word timestamps. The flow:

1. As the processing thread reads entries from the stream, it appends their IDs to `pending_entry_ids`
2. After a speech segment is transcribed and post-processed, all accumulated IDs are XACKed and the list is cleared
3. On session stop, any remaining unacked entries are flushed (final segment transcribed, then acked)
4. If the processing thread crashes, the `finally` block acks any remaining pending entries so they don't stay stuck in the consumer group's pending list
5. On session cleanup, the entire stream and consumer group are deleted

This means if the model service goes down mid-transcription, the unacked entries stay in Redis and the stream TTL (default 5 minutes) provides a window for recovery.

### Thread-per-connection model

The gateway uses a thread per session rather than pure async. This works because:
- Silero VAD weights are serialized once at startup; each session deserializes its own copy, so hidden state is independent with zero contention
- The GIL is released during torch inference, gRPC calls, and Redis I/O — threads don't block each other on the expensive parts
- Simpler than async when you need synchronous VAD + gRPC in a tight loop per session

### Graceful shutdown

On SIGTERM/SIGINT the gateway shuts down in order:
1. WebSocket server stops accepting new connections, sends `1001 Going Away` to connected clients
2. Active sessions receive stop signal, processing threads drain and join
3. gRPC server stops with a 5-second grace period for in-flight RPCs
4. ASR client gRPC channel closed
5. Redis connection closed

## Quick start

```bash
# generate proto stubs (needs grpcio-tools)
bash protos/generate.sh

# run everything
docker compose up --build

# or run services individually (need redis-server running):
# terminal 1: cd ms-asr-model && python -m asr_model.main
# terminal 2: cd ms-asr-gateway && python -m asr_gateway.main
```

### Test client

```bash
python test_client.py
```

Sends `sample/audio1.wav` over WebSocket, converts to mono 16kHz PCM, streams in 100ms chunks to simulate real-time, prints transcription results.

## GPU support

The model service Dockerfile uses `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`. Set device and compute type in docker-compose:

```yaml
- ASR_MODEL_WHISPER_DEVICE=cuda
- ASR_MODEL_WHISPER_COMPUTE_TYPE=int8_float16
```

The `deploy.resources.reservations.devices` block in docker-compose.yml handles GPU passthrough. Requires `nvidia-container-toolkit` on the host.

Approximate VRAM usage with `int8_float16`:

| Model | Params | VRAM |
|-------|--------|------|
| `tiny` | 39M | ~1 GB |
| `base` | 74M | ~1 GB |
| `small` | 244M | ~1.5 GB |
| `medium` | 769M | ~3 GB |
| `large-v3` | 1.5B | ~2 GB |

## Language support

Both services accept a `DEFAULT_LANGUAGE` env var (ISO 639-1 code). The gateway passes it to the model service on every transcription request.

```yaml
# Bengali
- ASR_MODEL_WHISPER_MODEL_SIZE=large-v3
- ASR_MODEL_DEFAULT_LANGUAGE=bn
- ASR_GATEWAY_DEFAULT_LANGUAGE=bn
```

For English-only, `base` or `small` is usually enough. The `.en` variants (`base.en`, `small.en`) are slightly better for English.

## Environment variables

### ms-asr-gateway

| Variable | Default | Description |
|---|---|---|
| `ASR_GATEWAY_GRPC_PORT` | 50051 | gRPC listen port |
| `ASR_GATEWAY_WS_PORT` | 8765 | WebSocket listen port |
| `ASR_GATEWAY_REDIS_URL` | redis://localhost:6379/0 | Redis connection |
| `ASR_GATEWAY_ASR_MODEL_HOST` | localhost | Model service hostname |
| `ASR_GATEWAY_ASR_MODEL_PORT` | 50052 | Model service port |
| `ASR_GATEWAY_VAD_THRESHOLD` | 0.5 | Silero VAD speech probability threshold |
| `ASR_GATEWAY_DEFAULT_LANGUAGE` | en | ISO 639-1 language code |

### ms-asr-model

| Variable | Default | Description |
|---|---|---|
| `ASR_MODEL_GRPC_PORT` | 50052 | gRPC listen port |
| `ASR_MODEL_WHISPER_MODEL_SIZE` | base | tiny, base, small, medium, large-v3 |
| `ASR_MODEL_WHISPER_DEVICE` | auto | auto, cpu, cuda |
| `ASR_MODEL_WHISPER_COMPUTE_TYPE` | auto | auto, float16, int8, int8_float16 |
| `ASR_MODEL_DEFAULT_LANGUAGE` | en | ISO 639-1 language code |

## Ports

| Port | Service |
|---|---|
| 50051 | Gateway gRPC (client-facing) |
| 8765 | Gateway WebSocket (client-facing) |
| 50052 | Model gRPC (internal) |
| 6379 | Redis |

## Proto definitions

`protos/asr_gateway.proto` — client-facing bidi streaming. Clients send `StartRecognition` with audio config, stream `AudioData` frames, then `StopRecognition`. Server streams back `Transcript` messages with word-level timestamps and confidence.

`protos/asr_model.proto` — internal unary. Gateway sends a speech segment as raw PCM bytes, model returns text + word timestamps + confidence scores.

Run `bash protos/generate.sh` to regenerate stubs into `protos/generated/`.

## Post-processing

The gateway post-processes every transcript before sending it to the client:

1. **Hallucination filtering** — known Whisper artifacts ("thank you for watching", "please subscribe"), 3+ consecutive repeated words, anomalous word rate (>8 words/sec)
2. **Filler word removal** — "um", "uh", "er", "you know", "I mean", etc.
3. **Text cleanup** — collapse whitespace, re-capitalize after sentence boundaries
