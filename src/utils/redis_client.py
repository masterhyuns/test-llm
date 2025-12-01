"""
Redis 클라이언트 유틸리티

Redis 연결 및 관리
"""
from typing import Optional
import redis.asyncio as aioredis
import structlog

from src.config.settings import settings

logger = structlog.get_logger()


class RedisClient:
    """Redis 클라이언트 래퍼"""

    def __init__(self):
        self._client: Optional[aioredis.Redis] = None

    async def connect(self) -> aioredis.Redis:
        """
        Redis 연결

        Returns:
            aioredis.Redis: Redis 클라이언트
        """
        if self._client is not None:
            return self._client

        try:
            # Redis URL 파싱 및 연결
            self._client = await aioredis.from_url(
                settings.redis_url,
                password=settings.redis_password if settings.redis_password else None,
                encoding="utf-8",
                decode_responses=True,
            )

            # 연결 테스트
            await self._client.ping()

            logger.info(
                "redis_connected",
                url=settings.redis_url.split("@")[-1]
                if "@" in settings.redis_url
                else settings.redis_url,  # 비밀번호 숨기기
            )

            return self._client

        except Exception as e:
            logger.error("redis_connection_failed", error=str(e), exc_info=True)
            raise

    async def disconnect(self):
        """Redis 연결 종료"""
        if self._client:
            await self._client.close()
            logger.info("redis_disconnected")
            self._client = None

    async def get_client(self) -> aioredis.Redis:
        """
        Redis 클라이언트 반환 (연결되지 않았으면 연결)

        Returns:
            aioredis.Redis: Redis 클라이언트
        """
        if self._client is None:
            await self.connect()

        return self._client


# 싱글톤 인스턴스
redis_client = RedisClient()
