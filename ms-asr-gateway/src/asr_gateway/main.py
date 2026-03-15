from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from concurrent import futures

import grpc
import websockets

# Add generated proto stubs to path
_PROTO_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "protos", "generated")
_PROTO_DIR = os.path.abspath(_PROTO_DIR)
if _PROTO_DIR not in sys.path:
    sys.path.insert(0, _PROTO_DIR)

import asr_gateway_pb2_grpc

from asr_gateway.asr_client import asr_client
from asr_gateway.audio_store import audio_store
from asr_gateway.config import settings
from asr_gateway.grpc_service import AsrGatewayServicer
from asr_gateway.vad import load_vad_model
from asr_gateway.ws_handler import ws_handler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def _run_ws_server() -> None:
    """Start the WebSocket server. Shuts down cleanly on SIGTERM/SIGINT."""
    loop = asyncio.get_running_loop()
    stop = loop.create_future()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set_result, None)

    async with websockets.serve(
        ws_handler,
        settings.ws_host,
        settings.ws_port,
    ) as server:
        logger.info(
            "WebSocket server listening on ws://%s:%d",
            settings.ws_host,
            settings.ws_port,
        )
        await stop
        logger.info("Shutting down WebSocket server, closing connections...")
        server.close()
        await server.wait_closed()


def _run_grpc_server() -> grpc.Server:
    """Create and start the gRPC server (non-blocking)."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
        ],
    )
    asr_gateway_pb2_grpc.add_AsrGatewayServicer_to_server(
        AsrGatewayServicer(), server
    )
    addr = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(addr)
    server.start()
    logger.info("gRPC server listening on %s", addr)
    return server


def main() -> None:
    # Initialize components
    load_vad_model()
    audio_store.connect()
    asr_client.connect()

    # Start gRPC server in background (uses its own thread pool)
    grpc_server = _run_grpc_server()

    try:
        # Run WebSocket server on the main asyncio loop
        asyncio.run(_run_ws_server())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        grpc_server.stop(grace=5)
        asr_client.close()
        audio_store.close()


if __name__ == "__main__":
    main()
