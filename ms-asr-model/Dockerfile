FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies directly (faster than pip install . which needs src)
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
