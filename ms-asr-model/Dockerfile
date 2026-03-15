FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        build-essential libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Create venv with Python 3.11
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    grpcio>=1.62.0 \
    grpcio-tools>=1.62.0 \
    faster-whisper>=1.0.0 \
    pydantic-settings>=2.0.0 \
    numpy>=1.26.0

COPY protos/generated/ /app/protos/generated/
COPY ms-asr-model/src/ /app/src/

ENV PYTHONPATH="/app/src:/app/protos/generated"
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "asr_model.main"]
