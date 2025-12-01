"""
Qdrant Vector Store 구현

📚 Vector Store란?
- 텍스트를 숫자 벡터(embedding)로 변환하여 저장하는 데이터베이스입니다
- 일반 DB와 다르게 "의미적 유사도"를 기반으로 검색할 수 있습니다
- 예: "강아지"를 검색하면 "개", "반려동물" 같은 유사한 의미의 문서도 찾아줍니다

🔍 왜 필요한가?
- RAG 시스템에서 사용자 질문과 관련된 문서를 찾기 위해 필요합니다
- 키워드 검색보다 훨씬 똑똑하게 관련 문서를 찾을 수 있습니다

💡 이 파일의 역할:
- Qdrant Vector DB에 문서를 저장하고 검색하는 기능을 제공합니다
- OpenAI의 embedding 모델로 텍스트를 벡터로 변환합니다
"""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
import uuid
from openai import OpenAI

from src.config.settings import get_settings
from src.utils.logger import get_logger

# 설정과 로거 가져오기
settings = get_settings()
logger = get_logger(__name__)


class QdrantStore:
    """
    Qdrant Vector Store 클래스

    🎯 주요 기능:
    1. 문서를 벡터로 변환하여 Qdrant에 저장
    2. 질문과 유사한 문서를 검색
    3. 조직별/사용자별 데이터 격리 (Multi-tenancy)

    📊 Vector Embedding이란?
    - 텍스트를 3072개의 숫자 배열로 변환한 것
    - 예: "안녕하세요" → [0.234, -0.123, 0.456, ..., 0.789] (3072개)
    - 비슷한 의미의 텍스트는 비슷한 숫자 패턴을 가짐
    - OpenAI의 text-embedding-3-large 모델 사용 (가장 성능 좋은 모델)

    🔐 Multi-tenancy (다중 테넌트):
    - organization_id: 회사/조직 단위로 데이터 분리
    - user_id: 사용자 단위로 데이터 분리
    - 각 조직/사용자는 자신의 데이터만 검색 가능
    """

    def __init__(
        self,
        collection_name: str = "documents",
        host: str = "localhost",
        port: int = 6333,
    ):
        """
        Qdrant Store 초기화

        Args:
            collection_name: Qdrant에서 데이터를 저장할 컬렉션 이름
                           (관계형 DB의 '테이블'과 비슷한 개념)
            host: Qdrant 서버 주소 (기본값: localhost)
            port: Qdrant 서버 포트 (기본값: 6333)

        💡 초기화 과정:
        1. Qdrant 클라이언트 연결
        2. OpenAI 클라이언트 연결 (embedding 생성용)
        3. 컬렉션이 없으면 자동 생성
        """
        self.collection_name = collection_name

        # Qdrant 클라이언트 연결
        # - Qdrant는 Vector DB로, 벡터를 저장하고 검색하는 전문 데이터베이스
        self.client = QdrantClient(host=host, port=port)

        # OpenAI 클라이언트 연결
        # - 텍스트를 벡터로 변환(embedding)하는데 사용
        self.openai_client = OpenAI(api_key=settings.openai_api_key)

        # Embedding 모델 설정
        # - text-embedding-3-large: OpenAI의 최신 고성능 embedding 모델
        # - 3072차원의 벡터 생성 (숫자 3072개)
        self.embedding_model = "text-embedding-3-large"
        self.embedding_dimension = 3072  # 벡터의 차원 (크기)

        # 컬렉션 초기화 (없으면 생성)
        self._ensure_collection()

        logger.info(
            "Qdrant Store 초기화 완료",
            collection=collection_name,
            embedding_model=self.embedding_model,
        )

    def _ensure_collection(self) -> None:
        """
        Qdrant 컬렉션이 존재하는지 확인하고, 없으면 생성

        🗂️ 컬렉션(Collection)이란?
        - 관계형 DB의 '테이블'과 비슷한 개념
        - 같은 구조의 벡터 데이터를 모아두는 공간

        ⚙️ 설정 내용:
        - vectors: 벡터의 크기(3072)와 거리 측정 방식(Cosine) 설정
        - Cosine Distance: 벡터 간 유사도를 측정하는 방법
          (0에 가까울수록 유사, 1에 가까울수록 다름)
        """
        try:
            # 기존 컬렉션 목록 가져오기
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            # 컬렉션이 없으면 새로 생성
            if self.collection_name not in collection_names:
                logger.info(f"컬렉션 '{self.collection_name}' 생성 중...")

                # 컬렉션 생성
                # - size: 벡터의 차원 (3072)
                # - distance: 유사도 측정 방식 (COSINE)
                #   * COSINE: 벡터 간 각도로 유사도 측정 (가장 일반적)
                #   * EUCLID: 벡터 간 직선 거리로 측정
                #   * DOT: 내적으로 측정
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )

                logger.info(f"컬렉션 '{self.collection_name}' 생성 완료")
            else:
                logger.info(f"컬렉션 '{self.collection_name}' 이미 존재함")

        except Exception as e:
            logger.error("컬렉션 확인/생성 중 오류", error=str(e))
            raise

    def _create_embedding(self, text: str) -> List[float]:
        """
        텍스트를 벡터(embedding)로 변환

        🔢 Embedding이란?
        - 텍스트를 컴퓨터가 이해할 수 있는 숫자 배열로 변환한 것
        - 비슷한 의미의 텍스트는 비슷한 숫자 패턴을 가짐

        📝 예시:
        입력: "강아지가 귀여워요"
        출력: [0.234, -0.123, 0.456, ..., 0.789] (3072개의 숫자)

        입력: "개가 예뻐요"
        출력: [0.241, -0.119, 0.462, ..., 0.781] (비슷한 패턴!)

        Args:
            text: 벡터로 변환할 텍스트

        Returns:
            3072개의 실수로 이루어진 벡터

        💰 비용:
        - OpenAI API 호출 (유료)
        - text-embedding-3-large: $0.00013 / 1K tokens
        - 예: 1000자 텍스트 → 약 $0.00013
        """
        try:
            # OpenAI API를 통해 embedding 생성
            # - input: 변환할 텍스트
            # - model: 사용할 embedding 모델
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model,
            )

            # API 응답에서 embedding 벡터 추출
            # - data[0]: 첫 번째 (그리고 유일한) 결과
            # - embedding: 실제 벡터 데이터 (3072개 숫자)
            embedding = response.data[0].embedding

            return embedding

        except Exception as e:
            logger.error("Embedding 생성 실패", text=text[:100], error=str(e))
            raise

    def add_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        organization_id: str,
        user_id: Optional[str] = None,
    ) -> str:
        """
        문서를 Vector Store에 추가

        📥 전체 흐름:
        1. 텍스트를 벡터로 변환 (OpenAI API 호출)
        2. 메타데이터에 조직/사용자 정보 추가
        3. Qdrant에 저장

        Args:
            text: 저장할 문서 내용
            metadata: 문서의 메타데이터 (제목, 작성자, 날짜 등)
            organization_id: 조직 ID (필수)
            user_id: 사용자 ID (선택, 없으면 조직 전체 공유)

        Returns:
            생성된 문서의 고유 ID (UUID)

        💡 사용 예시:
        ```python
        store = QdrantStore()
        doc_id = store.add_document(
            text="프로젝트 A의 마감일은 2024년 12월 31일입니다.",
            metadata={
                "title": "프로젝트 A 일정",
                "author": "홍길동",
                "created_at": "2024-12-01",
            },
            organization_id="org_123",
            user_id="user_456",
        )
        print(f"문서 저장 완료: {doc_id}")
        ```

        🔐 Multi-tenancy:
        - organization_id로 조직별 데이터 분리
        - user_id로 사용자별 데이터 분리 (선택)
        - 검색 시 해당 조직/사용자 문서만 검색됨
        """
        try:
            # 1. 텍스트를 벡터로 변환
            logger.info("문서 embedding 생성 중...", text_length=len(text))
            embedding = self._create_embedding(text)

            # 2. 고유 ID 생성
            # - UUID: 전 세계적으로 유일한 ID 생성
            # - 예: "550e8400-e29b-41d4-a716-446655440000"
            doc_id = str(uuid.uuid4())

            # 3. 메타데이터에 필수 정보 추가
            # - organization_id: 조직 식별자 (필수)
            # - user_id: 사용자 식별자 (있으면 추가)
            # - text: 원본 텍스트 (검색 결과에 포함시키기 위함)
            payload = {
                **metadata,  # 기존 메타데이터 유지
                "organization_id": organization_id,
                "text": text,
            }

            # user_id가 있으면 추가
            if user_id:
                payload["user_id"] = user_id

            # 4. Qdrant에 저장
            # - PointStruct: Qdrant의 데이터 단위
            #   * id: 문서 고유 ID
            #   * vector: embedding 벡터 (3072개 숫자)
            #   * payload: 메타데이터 (JSON 형태)
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload=payload,
                    )
                ],
            )

            logger.info(
                "문서 저장 완료",
                doc_id=doc_id,
                organization_id=organization_id,
                user_id=user_id,
            )

            return doc_id

        except Exception as e:
            logger.error("문서 저장 실패", error=str(e))
            raise

    def search(
        self,
        query: str,
        organization_id: str,
        user_id: Optional[str] = None,
        limit: int = 5,
        score_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        질문과 유사한 문서 검색

        🔍 검색 흐름:
        1. 질문을 벡터로 변환
        2. Qdrant에서 유사한 벡터 검색 (Cosine 유사도 기반)
        3. 조직/사용자 필터링 적용
        4. 유사도 점수가 threshold 이상인 결과만 반환

        Args:
            query: 검색할 질문/키워드
            organization_id: 조직 ID (필수)
            user_id: 사용자 ID (선택, 있으면 해당 사용자 문서만 검색)
            limit: 최대 검색 결과 개수 (기본 5개)
            score_threshold: 최소 유사도 점수 (0~1, 기본 0.3)
                           - 1.0: 완전히 동일
                           - 0.7: 상당히 유사
                           - 0.5: 약간 유사
                           - 0.3: 관련 있을 수 있음 (권장)
                           - 0.0: 모든 결과 반환

        Returns:
            검색 결과 리스트 (유사도 높은 순)
            각 결과 형식:
            {
                "id": "문서 ID",
                "score": 0.85,  # 유사도 점수 (0~1)
                "text": "문서 내용",
                "metadata": {...},  # 문서 메타데이터
            }

        💡 사용 예시:
        ```python
        store = QdrantStore()
        results = store.search(
            query="프로젝트 A 마감일이 언제야?",
            organization_id="org_123",
            user_id="user_456",
            limit=3,
            score_threshold=0.3,
        )

        for result in results:
            print(f"유사도: {result['score']:.2f}")
            print(f"내용: {result['text']}")
        ```

        🎯 검색 원리:
        - 질문과 문서를 모두 벡터로 변환
        - Cosine 유사도로 벡터 간 거리 계산
        - 거리가 가까운 (= 의미가 유사한) 문서 반환

        🔐 보안:
        - organization_id 필터: 다른 조직 문서는 절대 검색 안됨
        - user_id 필터: 다른 사용자 문서는 절대 검색 안됨
        """
        try:
            # 1. 질문을 벡터로 변환
            logger.info("검색 쿼리 embedding 생성 중...", query=query)
            query_embedding = self._create_embedding(query)

            # 2. 필터 조건 생성
            # - organization_id는 필수 필터
            # - user_id가 있으면 추가 필터
            filter_conditions = [
                FieldCondition(
                    key="organization_id",
                    match=MatchValue(value=organization_id),
                )
            ]

            # 사용자 ID 필터 추가 (있는 경우)
            if user_id:
                filter_conditions.append(
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id),
                    )
                )

            # 3. Qdrant에서 검색 실행
            # - query: 질문 벡터
            # - limit: 최대 결과 개수
            # - query_filter: 조직/사용자 필터
            # - score_threshold: 최소 유사도 (이보다 낮으면 제외)
            # - with_payload: payload 데이터 포함 (메타데이터, 텍스트 등)
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=limit,
                query_filter=Filter(must=filter_conditions),
                score_threshold=score_threshold,
                with_payload=True,  # payload 포함
            )

            # 4. 결과 정리
            # - Qdrant의 결과를 사용하기 쉬운 형태로 변환
            results = []
            for point in search_result.points:
                result = {
                    "id": point.id,
                    "score": point.score,  # 유사도 점수 (0~1)
                    "text": point.payload.get("text", ""),
                    "metadata": {
                        k: v
                        for k, v in point.payload.items()
                        if k not in ["text", "organization_id", "user_id"]
                    },
                }
                results.append(result)

            logger.info(
                "검색 완료",
                query=query,
                results_count=len(results),
                organization_id=organization_id,
                user_id=user_id,
            )

            return results

        except Exception as e:
            logger.error("검색 실패", query=query, error=str(e))
            raise

    def delete_document(self, doc_id: str) -> bool:
        """
        문서 삭제

        Args:
            doc_id: 삭제할 문서 ID

        Returns:
            삭제 성공 여부

        💡 사용 예시:
        ```python
        store = QdrantStore()
        success = store.delete_document("550e8400-e29b-41d4-a716-446655440000")
        if success:
            print("문서 삭제 완료")
        ```
        """
        try:
            # Qdrant에서 문서 삭제
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[doc_id],
            )

            logger.info("문서 삭제 완료", doc_id=doc_id)
            return True

        except Exception as e:
            logger.error("문서 삭제 실패", doc_id=doc_id, error=str(e))
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """
        컬렉션 정보 조회

        Returns:
            컬렉션 통계 정보
            {
                "name": "documents",
                "vectors_count": 1234,  # 저장된 문서 개수
                "indexed_vectors_count": 1234,
                "points_count": 1234,
            }

        💡 사용 예시:
        ```python
        store = QdrantStore()
        info = store.get_collection_info()
        print(f"저장된 문서 수: {info['vectors_count']}")
        ```
        """
        try:
            # Qdrant에서 컬렉션 정보 가져오기
            collection_info = self.client.get_collection(self.collection_name)

            # vectors_count가 None일 수 있으므로 기본값 0 설정
            # - 빈 컬렉션이거나 인덱싱 전일 때 None 반환될 수 있음
            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count or 0,
                "indexed_vectors_count": collection_info.indexed_vectors_count or 0,
                "points_count": collection_info.points_count or 0,
            }

        except Exception as e:
            logger.error("컬렉션 정보 조회 실패", error=str(e))
            raise
