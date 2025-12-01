"""
Chat API 엔드포인트

AI 대화 기능 (RAG 통합)

🎯 주요 기능:
1. RAG 모드: 문서 검색 + 문서 기반 답변
2. 일반 모드: LLM 일반 지식 기반 답변
3. Multi-tenancy: 조직/사용자별 데이터 격리
"""
import uuid
import structlog
from fastapi import APIRouter, HTTPException

from src.models.chat import ChatRequest, ChatResponse, Source
from src.core.llm.openai_client import openai_client
from src.core.rag import RAGEngine
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

# RAG 엔진 싱글톤 인스턴스
# - 앱 시작 시 한 번만 생성되어 모든 요청에서 재사용
# - Vector Store 연결 등 초기화 비용 절약
rag_engine = RAGEngine()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    AI 채팅 (RAG 통합)

    🎯 동작 방식:
    1. RAG 모드 (use_rag=True, 기본값):
       - 사용자 질문 → Vector DB에서 관련 문서 검색
       - 검색된 문서 + 질문 → LLM에 전달
       - LLM이 문서 기반으로 답변 생성
       - 참고한 문서(sources) 포함하여 반환

    2. 일반 모드 (use_rag=False):
       - 사용자 질문 → 바로 LLM에 전달
       - LLM의 일반 지식으로 답변

    Args:
        request: 채팅 요청
            - message: 사용자 질문
            - organization_id: 조직 ID (필수)
            - user_id: 사용자 ID (선택)
            - use_rag: RAG 사용 여부 (기본 True)

    Returns:
        ChatResponse: AI 응답
            - message: 답변 내용
            - sources: 참고 문서 리스트 (RAG 모드에만 포함)
            - session_id: 세션 ID

    Raises:
        HTTPException: 처리 실패 시

    💡 사용 예시:
    ```json
    POST /api/v1/chat
    {
        "message": "프로젝트 A 마감일이 언제야?",
        "organization_id": "org_123",
        "user_id": "user_456",
        "use_rag": true
    }
    ```

    Response:
    ```json
    {
        "session_id": "sess_abc123",
        "message": "프로젝트 A의 마감일은 2024년 12월 31일입니다.",
        "sources": [
            {
                "text": "프로젝트 A의 마감일은 2024년 12월 31일입니다.",
                "score": 0.92,
                "metadata": {"title": "프로젝트 A 일정"}
            }
        ],
        "timestamp": "2024-12-01T12:00:00"
    }
    ```
    """
    # 세션 ID 생성 또는 사용
    session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"

    logger.info(
        "채팅 요청 수신",
        session_id=session_id,
        message_length=len(request.message),
        organization_id=request.organization_id,
        user_id=request.user_id,
        use_rag=request.use_rag,
    )

    try:
        # RAG 모드: 문서 검색 + 문서 기반 답변
        if request.use_rag:
            logger.info("RAG 모드로 답변 생성 중...")

            # RAG 엔진으로 답변 생성
            # - 내부적으로: 문서 검색 → 컨텍스트 구성 → LLM 답변 생성
            rag_result = rag_engine.generate_answer(
                query=request.message,
                organization_id=request.organization_id,
                user_id=request.user_id,
            )

            # Source 모델로 변환
            sources = [
                Source(
                    text=src["text"],
                    score=src["score"],
                    metadata=src["metadata"],
                )
                for src in rag_result["sources"]
            ]

            logger.info(
                "RAG 답변 생성 완료",
                session_id=session_id,
                answer_length=len(rag_result["answer"]),
                sources_count=len(sources),
            )

            return ChatResponse(
                session_id=session_id,
                message=rag_result["answer"],
                sources=sources if sources else None,
                suggestions=[
                    "관련 문서 더 찾기",
                    "다른 프로젝트 정보 검색",
                    "일정 확인하기",
                ],
            )

        # 일반 모드: LLM 일반 지식 기반 답변
        else:
            logger.info("일반 LLM 모드로 답변 생성 중...")

            # 시스템 프롬프트
            system_prompt = """당신은 Cowexa 협업 플랫폼의 AI 비서입니다.

사용자의 업무를 도와 생산성을 높이는 것이 목표입니다.

지침:
- 친절하고 전문적으로 답변하세요
- 모르는 것은 솔직히 말하세요
- 간결하면서도 도움이 되는 답변을 제공하세요
"""

            # 메시지 구성
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message},
            ]

            # LLM 호출
            response_text = await openai_client.generate(messages)

            logger.info(
                "일반 답변 생성 완료",
                session_id=session_id,
                response_length=len(response_text),
            )

            return ChatResponse(
                session_id=session_id,
                message=response_text,
                sources=None,  # 일반 모드는 참고 문서 없음
                suggestions=[
                    "문서 검색하기",
                    "태스크 생성하기",
                    "일정 확인하기",
                ],
            )

    except Exception as e:
        logger.error(
            "채팅 요청 실패",
            session_id=session_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"답변 생성 실패: {str(e)}"
        )
