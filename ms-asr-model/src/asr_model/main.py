from __future__ import annotations

import logging
import os
import sys
from concurrent import futures

import grpc

# Add generated proto stubs to path
_PROTO_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "protos", "generated")
_PROTO_DIR = os.path.abspath(_PROTO_DIR)
if _PROTO_DIR not in sys.path:
    sys.path.insert(0, _PROTO_DIR)

import asr_model_pb2_grpc

from asr_model.config import settings
from asr_model.grpc_service import AsrModelServicer
from asr_model.model import asr_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def serve() -> None:
    asr_model.load()

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=settings.max_workers),
        options=[
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50 MB
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
        ],
    )
    asr_model_pb2_grpc.add_AsrModelServicer_to_server(AsrModelServicer(), server)

    addr = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(addr)
    server.start()
    logger.info("ms-asr-model gRPC server listening on %s", addr)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop(grace=5)


if __name__ == "__main__":
    serve()
