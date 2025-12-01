"""
Documents API 엔드포인트

문서 인덱싱 및 관리 기능

🎯 주요 기능:
1. 문서 추가 (Indexing): Vector DB에 문서 저장
2. 문서 검색: 유사한 문서 찾기
3. 문서 삭제: Vector DB에서 문서 제거
4. 통계 조회: 저장된 문서 수 등 확인

💡 용어 설명:
- Indexing (인덱싱): 문서를 검색 가능한 형태로 변환하여 저장하는 과정
  * 텍스트 → 벡터(embedding) 변환 → Vector DB 저장
- Vector DB: 벡터(숫자 배열)를 저장하고 유사도 검색이 가능한 데이터베이스
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.core.rag import RAGEngine
from src.models.chat import Source
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

# RAG 엔진 싱글톤
# - 앱 시작 시 한 번만 생성되어 모든 요청에서 재사용
rag_engine = RAGEngine()


# ============================================================
# 요청/응답 모델 정의
# ============================================================


class DocumentAddRequest(BaseModel):
    """
    문서 추가 요청 모델

    📥 Indexing 요청:
    - 문서를 RAG 시스템에 추가하여 검색 가능하게 만듦
    - 내부적으로 OpenAI API를 사용하여 embedding 생성
    - OpenSearch 사용 시 태그로 문서 분류 가능
    """

    text: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="문서 내용 (최소 10자, 최대 10000자)",
        examples=["프로젝트 A의 마감일은 2024년 12월 31일입니다."],
    )
    metadata: dict = Field(
        default_factory=dict,
        description="문서 메타데이터 (제목, 작성자, 날짜 등)",
        examples=[
            {
                "title": "프로젝트 A 일정",
                "author": "홍길동",
                "created_at": "2024-12-01",
                "project_id": "proj_123",
            }
        ],
    )
    organization_id: str = Field(
        ...,
        description="조직 ID (필수)",
        examples=["org_123"],
    )
    user_id: Optional[str] = Field(
        None,
        description="사용자 ID (선택, 없으면 조직 전체 공유)",
        examples=["user_456"],
    )
    tags: Optional[List[str]] = Field(
        None,
        description="태그 리스트 (선택, OpenSearch 사용 시 유용)",
        examples=[["프로젝트A", "일정", "중요"]],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "프로젝트 A의 마감일은 2024년 12월 31일입니다. 담당자는 홍길동이며, 주요 마일스톤은 다음과 같습니다.",
                "metadata": {
                    "title": "프로젝트 A 일정",
                    "author": "홍길동",
                    "created_at": "2024-12-01",
                    "project_id": "proj_123",
                },
                "organization_id": "org_123",
                "user_id": "user_456",
                "tags": ["프로젝트A", "일정", "중요"],
            }
        }


class DocumentAddResponse(BaseModel):
    """
    문서 추가 응답 모델

    ✅ Indexing 성공:
    - 문서가 Vector DB에 저장됨
    - 이제 Chat API에서 이 문서를 검색하여 답변에 활용 가능
    """

    doc_id: str = Field(..., description="생성된 문서 ID (UUID)")
    message: str = Field(..., description="성공 메시지")

    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": "550e8400-e29b-41d4-a716-446655440000",
                "message": "문서가 성공적으로 추가되었습니다.",
            }
        }


class DocumentSearchRequest(BaseModel):
    """
    문서 검색 요청 모델

    🔍 Semantic Search:
    - 키워드가 정확히 일치하지 않아도 의미가 비슷하면 검색됨
    - 예: "강아지" 검색 → "개", "반려동물" 문서도 검색

    🏷️ 태그 필터링 (OpenSearch):
    - 특정 태그가 있는 문서만 검색
    - 예: tags=["프로젝트A"] → 프로젝트A 태그 문서만
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="검색 질문/키워드",
        examples=["프로젝트 마감일"],
    )
    organization_id: str = Field(
        ...,
        description="조직 ID (필수)",
        examples=["org_123"],
    )
    user_id: Optional[str] = Field(
        None,
        description="사용자 ID (선택)",
        examples=["user_456"],
    )
    tags: Optional[List[str]] = Field(
        None,
        description="태그 필터 (선택, OpenSearch 사용 시 유용)",
        examples=[["프로젝트A"]],
    )
    limit: int = Field(
        5,
        ge=1,
        le=20,
        description="최대 검색 결과 개수 (1~20)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "프로젝트 A 마감일",
                "organization_id": "org_123",
                "user_id": "user_456",
                "tags": ["프로젝트A"],
                "limit": 5,
            }
        }


class DocumentSearchResponse(BaseModel):
    """
    문서 검색 응답 모델

    📊 검색 결과:
    - 유사도가 높은 순서대로 정렬
    - score: 유사도 점수 (0~1, 1에 가까울수록 유사)
    """

    results: List[Source] = Field(..., description="검색 결과 리스트")
    count: int = Field(..., description="검색 결과 개수")

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "text": "프로젝트 A의 마감일은 2024년 12월 31일입니다.",
                        "score": 0.92,
                        "metadata": {"title": "프로젝트 A 일정"},
                    }
                ],
                "count": 1,
            }
        }


class DocumentDeleteResponse(BaseModel):
    """문서 삭제 응답 모델"""

    message: str = Field(..., description="성공 메시지")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "문서가 성공적으로 삭제되었습니다.",
            }
        }


class StatsResponse(BaseModel):
    """
    통계 응답 모델

    📊 RAG 시스템 상태:
    - total_documents: 저장된 총 문서 수
    - vector_store: Vector DB 상세 정보
    - llm_model: 사용 중인 LLM 모델
    """

    total_documents: int = Field(..., description="총 문서 수")
    vector_store: dict = Field(..., description="Vector Store 정보")
    llm_model: str = Field(..., description="LLM 모델명")

    class Config:
        json_schema_extra = {
            "example": {
                "total_documents": 1234,
                "vector_store": {
                    "name": "documents",
                    "vectors_count": 1234,
                },
                "llm_model": "gpt-4o",
            }
        }


# ============================================================
# API 엔드포인트
# ============================================================


@router.post(
    "/documents",
    response_model=DocumentAddResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_document(request: DocumentAddRequest):
    """
    문서를 RAG 시스템에 추가 (Indexing)

    📥 동작 과정:
    1. 요청 받기 (텍스트 + 메타데이터)
    2. OpenAI API로 텍스트를 벡터로 변환 (embedding)
    3. 벡터를 Qdrant Vector DB에 저장
    4. 문서 ID 반환

    💡 이후 사용:
    - 저장된 문서는 Chat API에서 자동으로 검색됨
    - 사용자가 질문하면 관련 문서를 찾아서 답변에 활용

    💰 비용:
    - OpenAI embedding API: $0.00013 / 1K tokens
    - 예: 1000자 문서 → 약 $0.0002

    Args:
        request: 문서 추가 요청
            - text: 문서 내용 (10~10000자)
            - metadata: 문서 메타데이터
            - organization_id: 조직 ID (필수)
            - user_id: 사용자 ID (선택)

    Returns:
        DocumentAddResponse: 생성된 문서 ID

    Raises:
        HTTPException: 추가 실패 시

    💡 사용 예시:
    ```json
    POST /api/v1/documents
    {
        "text": "프로젝트 A의 마감일은 2024년 12월 31일입니다.",
        "metadata": {
            "title": "프로젝트 A 일정",
            "author": "홍길동"
        },
        "organization_id": "org_123",
        "user_id": "user_456"
    }
    ```

    Response:
    ```json
    {
        "doc_id": "550e8400-e29b-41d4-a716-446655440000",
        "message": "문서가 성공적으로 추가되었습니다."
    }
    ```
    """
    logger.info(
        "문서 추가 요청 수신",
        text_length=len(request.text),
        organization_id=request.organization_id,
        user_id=request.user_id,
    )

    try:
        # RAG 엔진으로 문서 추가
        # - 내부적으로: embedding 생성 → Vector DB 저장
        # - OpenSearch 사용 시: 태그도 함께 저장
        doc_id = rag_engine.add_document(
            text=request.text,
            metadata=request.metadata,
            organization_id=request.organization_id,
            user_id=request.user_id,
            tags=request.tags,
        )

        logger.info(
            "문서 추가 완료",
            doc_id=doc_id,
            organization_id=request.organization_id,
        )

        return DocumentAddResponse(
            doc_id=doc_id,
            message="문서가 성공적으로 추가되었습니다.",
        )

    except Exception as e:
        logger.error("문서 추가 실패", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"문서 추가 실패: {str(e)}",
        )


@router.post("/documents/search", response_model=DocumentSearchResponse)
async def search_documents(request: DocumentSearchRequest):
    """
    문서 검색 (Semantic Search)

    🔍 동작 과정:
    1. 검색 쿼리를 벡터로 변환 (embedding)
    2. Vector DB에서 유사한 벡터 찾기
    3. 조직/사용자 필터링 적용
    4. 유사도 높은 순으로 정렬하여 반환

    💡 Semantic Search란?
    - 키워드 일치가 아닌 의미 유사도로 검색
    - 예: "강아지" 검색 → "개", "반려동물" 문서도 검색
    - 동의어, 유사 표현 자동으로 찾아줌

    Args:
        request: 검색 요청
            - query: 검색 질문/키워드
            - organization_id: 조직 ID (필수)
            - user_id: 사용자 ID (선택)
            - limit: 최대 결과 개수 (1~20)

    Returns:
        DocumentSearchResponse: 검색 결과 리스트

    Raises:
        HTTPException: 검색 실패 시

    💡 사용 예시:
    ```json
    POST /api/v1/documents/search
    {
        "query": "프로젝트 마감일",
        "organization_id": "org_123",
        "limit": 5
    }
    ```

    Response:
    ```json
    {
        "results": [
            {
                "text": "프로젝트 A의 마감일은 2024년 12월 31일입니다.",
                "score": 0.92,
                "metadata": {"title": "프로젝트 A"}
            }
        ],
        "count": 1
    }
    ```
    """
    logger.info(
        "문서 검색 요청 수신",
        query=request.query,
        organization_id=request.organization_id,
        user_id=request.user_id,
        limit=request.limit,
    )

    try:
        # RAG 엔진으로 문서 검색
        # - 내부적으로: query embedding → Vector DB 검색
        # - OpenSearch 사용 시: 태그 필터링 지원
        results = rag_engine.search_documents(
            query=request.query,
            organization_id=request.organization_id,
            user_id=request.user_id,
            tags=request.tags,
            limit=request.limit,
        )

        # Source 모델로 변환
        sources = [
            Source(
                text=result["text"],
                score=result["score"],
                metadata=result["metadata"],
            )
            for result in results
        ]

        logger.info(
            "문서 검색 완료",
            query=request.query,
            results_count=len(sources),
        )

        return DocumentSearchResponse(
            results=sources,
            count=len(sources),
        )

    except Exception as e:
        logger.error("문서 검색 실패", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"문서 검색 실패: {str(e)}",
        )


@router.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(doc_id: str):
    """
    문서 삭제

    🗑️ 동작:
    - Vector DB에서 문서 완전히 제거
    - 이후 검색 결과에 나타나지 않음

    Args:
        doc_id: 삭제할 문서 ID (UUID)

    Returns:
        DocumentDeleteResponse: 성공 메시지

    Raises:
        HTTPException: 삭제 실패 시

    💡 사용 예시:
    ```
    DELETE /api/v1/documents/550e8400-e29b-41d4-a716-446655440000
    ```

    Response:
    ```json
    {
        "message": "문서가 성공적으로 삭제되었습니다."
    }
    ```

    ⚠️ 주의:
    - 삭제된 문서는 복구 불가능
    - 프로덕션에서는 soft delete 권장 (metadata에 deleted 플래그 추가)
    """
    logger.info("문서 삭제 요청 수신", doc_id=doc_id)

    try:
        # RAG 엔진으로 문서 삭제
        success = rag_engine.delete_document(doc_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"문서를 찾을 수 없습니다: {doc_id}",
            )

        logger.info("문서 삭제 완료", doc_id=doc_id)

        return DocumentDeleteResponse(
            message="문서가 성공적으로 삭제되었습니다.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("문서 삭제 실패", doc_id=doc_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"문서 삭제 실패: {str(e)}",
        )


@router.get("/documents/stats", response_model=StatsResponse)
async def get_stats():
    """
    RAG 시스템 통계 조회

    📊 제공 정보:
    - 저장된 총 문서 수
    - Vector Store 상세 정보
    - 사용 중인 LLM 모델

    Returns:
        StatsResponse: 통계 정보

    Raises:
        HTTPException: 조회 실패 시

    💡 사용 예시:
    ```
    GET /api/v1/documents/stats
    ```

    Response:
    ```json
    {
        "total_documents": 1234,
        "vector_store": {
            "name": "documents",
            "vectors_count": 1234,
            "indexed_vectors_count": 1234
        },
        "llm_model": "gpt-4o"
    }
    ```

    💡 활용:
    - 대시보드에서 문서 현황 표시
    - 모니터링: 문서 증가 추이 파악
    - 용량 관리: Vector DB 사이즈 확인
    """
    logger.info("통계 조회 요청 수신")

    try:
        # RAG 엔진에서 통계 가져오기
        stats = rag_engine.get_stats()

        logger.info("통계 조회 완료", total_documents=stats["total_documents"])

        return StatsResponse(
            total_documents=stats["total_documents"],
            vector_store=stats["vector_store"],
            llm_model=stats["llm_model"],
        )

    except Exception as e:
        logger.error("통계 조회 실패", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"통계 조회 실패: {str(e)}",
        )
