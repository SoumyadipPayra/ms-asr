#!/usr/bin/env bash
# Generate Python gRPC stubs from proto files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/generated"

mkdir -p "$OUT_DIR"

python -m grpc_tools.protoc \
    -I "$SCRIPT_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    "$SCRIPT_DIR/asr_gateway.proto" \
    "$SCRIPT_DIR/asr_model.proto"

# Fix relative imports to absolute (stubs directory is added to sys.path directly)
sed -i 's/^from \. import/import/' "$OUT_DIR/asr_gateway_pb2_grpc.py" 2>/dev/null || true
sed -i 's/^from \. import/import/' "$OUT_DIR/asr_model_pb2_grpc.py" 2>/dev/null || true

echo "Generated stubs in $OUT_DIR"
ls -la "$OUT_DIR"
