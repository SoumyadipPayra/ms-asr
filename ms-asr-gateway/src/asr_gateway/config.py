from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_prefix: str = "asr_gateway"

    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_stream_ttl_seconds: int = 300  # 5 minutes

    # ASR model service
    asr_model_host: str = "localhost"
    asr_model_port: int = 50052
    asr_model_timeout: float = 30.0  # seconds

    # VAD
    vad_threshold: float = 0.5
    vad_min_speech_duration_ms: int = 250
    vad_min_silence_duration_ms: int = 700
    vad_speech_pad_ms: int = 30
    vad_sample_rate: int = 16000

    # Audio
    default_sample_rate: int = 16000
    default_encoding: str = "pcm_s16le"
    default_channels: int = 1

    model_config = {"env_prefix": "ASR_GATEWAY_"}


settings = Settings()
