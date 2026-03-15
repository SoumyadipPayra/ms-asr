# ms-asr

Real-time speech-to-text system with two microservices.

## Architecture

- **ms-asr-gateway** — Stateful, client-facing. Accepts audio via gRPC bidi streaming or WebSocket. Buffers in Redis Streams, segments with Silero VAD, sends chunks to model service, post-processes transcripts.
- **ms-asr-model** — Stateless, GPU. Runs faster-whisper (CTranslate2) for ASR inference. Unary gRPC API.

## Quick Start

```bash
# Generate proto stubs (requires grpcio-tools)
bash protos/generate.sh

# Run with Docker Compose
docker compose up --build

# Or run services individually (need Redis running):
# Terminal 1: redis-server
# Terminal 2: cd ms-asr-model && python -m asr_model.main
# Terminal 3: cd ms-asr-gateway && python -m asr_gateway.main
```

## Proto Generation

```bash
bash protos/generate.sh
```

Output goes to `protos/generated/`. Both services add this directory to `sys.path` at startup.

## Environment Variables

### ms-asr-gateway
- `ASR_GATEWAY_GRPC_PORT` (default: 50051)
- `ASR_GATEWAY_WS_PORT` (default: 8765)
- `ASR_GATEWAY_REDIS_URL` (default: redis://localhost:6379/0)
- `ASR_GATEWAY_ASR_MODEL_HOST` (default: localhost)
- `ASR_GATEWAY_ASR_MODEL_PORT` (default: 50052)
- `ASR_GATEWAY_VAD_THRESHOLD` (default: 0.5)

### ms-asr-model
- `ASR_MODEL_GRPC_PORT` (default: 50052)
- `ASR_MODEL_WHISPER_MODEL_SIZE` (default: base)
- `ASR_MODEL_WHISPER_DEVICE` (default: auto)
- `ASR_MODEL_WHISPER_COMPUTE_TYPE` (default: auto)

## Ports
- 50051: Gateway gRPC
- 8765: Gateway WebSocket
- 50052: Model gRPC (internal)
- 6379: Redis

## Key Design Decisions
- Thread-per-connection in gateway (Silero VAD weights shared read-only, hidden state is thread-local, GIL released during torch ops and I/O)
- Redis Streams for durable audio buffering with XACK-based acknowledgment
- Audio acked based on transcript word timestamps mapped back to stream entries
