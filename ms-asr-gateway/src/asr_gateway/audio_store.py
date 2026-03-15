from __future__ import annotations

import logging
from typing import Any

import redis

from asr_gateway.config import settings

logger = logging.getLogger(__name__)


class AudioStore:
    """Redis Streams-based durable audio buffer.

    One stream per session: ``audio:{session_id}``
    Each entry: ``{seq, audio, timestamp, byte_offset}``
    Consumer group per session: ``cg:{session_id}``
    """

    def __init__(self) -> None:
        self._redis: redis.Redis | None = None

    def connect(self) -> None:
        self._redis = redis.from_url(
            settings.redis_url,
            decode_responses=False,
        )
        self._redis.ping()
        logger.info("Connected to Redis at %s", settings.redis_url)

    @property
    def client(self) -> redis.Redis:
        if self._redis is None:
            raise RuntimeError("AudioStore not connected")
        return self._redis

    def _stream_key(self, session_id: str) -> str:
        return f"audio:{session_id}"

    def _group_name(self, session_id: str) -> str:
        return f"cg:{session_id}"

    def create_session_stream(self, session_id: str) -> None:
        """Create the stream and consumer group for a new session."""
        key = self._stream_key(session_id)
        try:
            self.client.xgroup_create(
                key, self._group_name(session_id), id="0", mkstream=True
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug("Consumer group already exists for %s", session_id)
            else:
                raise
        # Set TTL for auto-cleanup
        self.client.expire(key, settings.redis_stream_ttl_seconds)

    def append_audio(
        self,
        session_id: str,
        seq: int,
        audio: bytes,
        timestamp: float,
        byte_offset: int,
    ) -> str:
        """Append an audio chunk to the session's stream. Returns the entry ID."""
        key = self._stream_key(session_id)
        entry_id: bytes = self.client.xadd(
            key,
            {
                b"seq": str(seq).encode(),
                b"audio": audio,
                b"timestamp": str(timestamp).encode(),
                b"byte_offset": str(byte_offset).encode(),
            },
        )
        # Refresh TTL on each write
        self.client.expire(key, settings.redis_stream_ttl_seconds)
        return entry_id.decode() if isinstance(entry_id, bytes) else entry_id

    def read_audio(
        self,
        session_id: str,
        consumer_name: str,
        count: int = 10,
        block_ms: int = 1000,
    ) -> list[tuple[str, dict[str, bytes]]]:
        """Read new entries from the session's stream via consumer group.

        Returns list of (entry_id, field_dict) tuples.
        """
        key = self._stream_key(session_id)
        group = self._group_name(session_id)
        try:
            results = self.client.xreadgroup(
                group, consumer_name, {key: ">"}, count=count, block=block_ms
            )
        except redis.ResponseError as e:
            if "NOGROUP" in str(e):
                return []
            raise

        if not results:
            return []

        entries = []
        for _stream_name, messages in results:
            for msg_id, fields in messages:
                mid = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                entries.append((mid, fields))
        return entries

    def ack(self, session_id: str, *entry_ids: str) -> int:
        """Acknowledge processed entries."""
        if not entry_ids:
            return 0
        key = self._stream_key(session_id)
        group = self._group_name(session_id)
        return self.client.xack(key, group, *entry_ids)

    def cleanup_session(self, session_id: str) -> None:
        """Delete stream and consumer group for a finished session."""
        key = self._stream_key(session_id)
        try:
            self.client.xgroup_destroy(key, self._group_name(session_id))
        except redis.ResponseError:
            pass
        self.client.delete(key)
        logger.debug("Cleaned up stream for session %s", session_id)


# Singleton
audio_store = AudioStore()
