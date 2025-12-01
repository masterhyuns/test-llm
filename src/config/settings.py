"""
애플리케이션 설정

환경변수를 통해 설정을 관리합니다.
"""
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션 설정"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # Environment
    environment: str = Field(default="dev", description="실행 환경")
    debug: bool = Field(default=True, description="디버그 모드")

    # Server
    host: str = Field(default="0.0.0.0", description="서버 호스트")
    port: int = Field(default=8000, description="서버 포트")

    # OpenAI
    openai_api_key: str = Field(..., description="OpenAI API 키")
    openai_model: str = Field(default="gpt-4o", description="기본 LLM 모델")
    openai_max_tokens: int = Field(default=2000, description="최대 토큰 수")
    openai_temperature: float = Field(default=0.7, description="Temperature")

    # Database
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    redis_password: str = Field(default="", description="Redis 비밀번호")
    mongodb_url: str = Field(
        default="mongodb://localhost:27017/ai_assistant",
        description="MongoDB URL"
    )

    # OpenSearch (Vector Store)
    opensearch_host: str = Field(default="localhost", description="OpenSearch 호스트")
    opensearch_port: int = Field(default=9200, description="OpenSearch 포트")
    opensearch_user: str = Field(default="admin", description="OpenSearch 사용자명")
    opensearch_password: str = Field(default="admin", description="OpenSearch 비밀번호")
    opensearch_use_ssl: bool = Field(default=False, description="SSL 사용 여부")
    opensearch_index: str = Field(default="ai_documents", description="인덱스 이름")

    # Security
    secret_key: str = Field(..., description="JWT Secret Key")
    algorithm: str = Field(default="HS256", description="JWT 알고리즘")

    # CORS
    cors_origins: str = Field(
        default="http://localhost:3000",
        description="CORS 허용 오리진 (쉼표로 구분)"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """CORS 오리진 리스트 반환"""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    # Rate Limiting
    rate_limit_per_minute: int = Field(
        default=100,
        description="분당 요청 제한"
    )

    # Logging
    log_level: str = Field(default="INFO", description="로그 레벨")

    @property
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return self.environment == "prod"


# 싱글톤 인스턴스
settings = Settings()


def get_settings() -> Settings:
    """
    설정 인스턴스 반환

    FastAPI의 Depends와 함께 사용하여 의존성 주입 가능
    테스트 시 설정을 오버라이드할 수 있음

    Returns:
        Settings: 애플리케이션 설정
    """
    return settings
