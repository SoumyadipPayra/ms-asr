FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    grpcio>=1.62.0 \
    grpcio-tools>=1.62.0 \
    websockets>=12.0 \
    redis>=5.0.0 \
    pydantic-settings>=2.0.0 \
    numpy>=1.26.0 && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir packaging onnxruntime

COPY protos/generated/ /app/protos/generated/
COPY ms-asr-gateway/src/ /app/src/

ENV PYTHONPATH="/app/src:/app/protos/generated"
ENV PYTHONUNBUFFERED=1

EXPOSE 50051 8765

CMD ["python", "-m", "asr_gateway.main"]
