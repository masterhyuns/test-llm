"""
OpenAI API 클라이언트

LLM 호출을 추상화
"""
from typing import List
import structlog
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.config.settings import settings

logger = structlog.get_logger()


class OpenAIClient:
    """OpenAI API 클라이언트"""

    def __init__(self):
        """클라이언트 초기화"""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.max_tokens = settings.openai_max_tokens
        self.temperature = settings.openai_temperature

        logger.info("openai_client_initialized", model=self.model)

    async def generate(
        self,
        messages: List[ChatCompletionMessageParam],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        채팅 완성 생성

        Args:
            messages: 대화 메시지 리스트
            temperature: Temperature 값 (None이면 기본값 사용)
            max_tokens: 최대 토큰 수 (None이면 기본값 사용)

        Returns:
            str: AI 응답 메시지

        Raises:
            Exception: API 호출 실패 시
        """
        try:
            logger.debug(
                "llm_request",
                model=self.model,
                message_count=len(messages),
                temperature=temperature or self.temperature,
            )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )

            # 사용량 로깅
            usage = response.usage
            logger.info(
                "llm_response",
                model=self.model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            )

            content = response.choices[0].message.content
            return content if content else ""

        except Exception as e:
            logger.error(
                "llm_request_failed", model=self.model, error=str(e), exc_info=True
            )
            raise


# 싱글톤 인스턴스
openai_client = OpenAIClient()
