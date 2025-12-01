"""
Chat 관련 데이터 모델

Pydantic 모델 정의
"""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    대화 요청 모델

    💡 Multi-tenancy (다중 테넌트):
    - organization_id: 조직별 데이터 격리
    - user_id: 사용자별 데이터 격리 (선택)
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="사용자 메시지",
        examples=["안녕하세요", "프로젝트 문서를 찾아주세요"],
    )
    organization_id: str = Field(
        ...,
        description="조직 ID (필수)",
        examples=["org_123"],
    )
    user_id: Optional[str] = Field(
        None,
        description="사용자 ID (선택, 없으면 조직 전체 문서 검색)",
        examples=["user_456"],
    )
    session_id: Optional[str] = Field(
        None, description="세션 ID (없으면 자동 생성)"
    )
    use_rag: bool = Field(
        True,
        description="RAG 사용 여부 (True: 문서 검색 + 답변, False: 일반 LLM 답변)",
    )
    context: Optional[dict] = Field(None, description="추가 컨텍스트")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "프로젝트 A 마감일이 언제야?",
                "organization_id": "org_123",
                "user_id": "user_456",
                "session_id": "sess_abc123",
                "use_rag": True,
                "context": {"project_id": "proj_456"},
            }
        }


class Source(BaseModel):
    """
    참조 문서 출처

    🔍 RAG 검색 결과:
    - text: 문서 내용
    - score: 유사도 점수 (0~1, 높을수록 유사)
    - metadata: 문서 메타데이터 (제목, 작성자, 날짜 등)
    """

    text: str = Field(..., description="문서 내용")
    score: float = Field(..., description="유사도 점수 (0~1)")
    metadata: dict = Field(default_factory=dict, description="문서 메타데이터")


class ChatResponse(BaseModel):
    """대화 응답"""

    session_id: str = Field(..., description="세션 ID")
    message: str = Field(..., description="AI 응답 메시지")
    sources: Optional[List[Source]] = Field(None, description="참조 문서 목록")
    suggestions: Optional[List[str]] = Field(None, description="추천 질문 목록")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="응답 생성 시간"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "message": "안녕하세요! 무엇을 도와드릴까요?",
                "sources": None,
                "suggestions": [
                    "프로젝트 문서 검색하기",
                    "태스크 생성하기",
                    "일정 확인하기",
                ],
                "timestamp": "2025-12-01T12:00:00",
            }
        }
