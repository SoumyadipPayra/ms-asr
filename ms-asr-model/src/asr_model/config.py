from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_prefix: str = "asr_model"

    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50052
    max_workers: int = 4

    # faster-whisper model settings
    whisper_model_size: str = "base"  # tiny|base|small|medium|large-v3
    whisper_device: str = "auto"  # auto|cpu|cuda
    whisper_compute_type: str = "auto"  # auto|float16|int8|int8_float16
    whisper_download_root: str | None = None

    default_language: str = "en"

    model_config = {"env_prefix": "ASR_MODEL_"}


settings = Settings()
