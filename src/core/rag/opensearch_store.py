"""
OpenSearch Vector Store 구현

📚 OpenSearch란?
- Elasticsearch에서 포크된 오픈소스 검색 엔진
- 전문 검색 + 벡터 검색 + 태그 필터링 모두 지원
- k-NN 플러그인으로 의미 기반 검색 가능

🔍 왜 OpenSearch를 사용하나?
- 이미 플랫폼에서 모든 데이터를 OpenSearch로 문서화
- 기존 인프라 활용 (추가 Vector DB 불필요)
- 태깅 검색 + 자연어 검색 동시에 가능
- Hybrid 검색: 키워드 정확도 + 의미 유사도 결합

💡 이 파일의 역할:
- OpenSearch에 문서를 저장하고 검색하는 기능 제공
- OpenAI embedding으로 텍스트를 벡터로 변환
- 태그/필터링과 벡터 검색을 조합한 Hybrid 검색
"""

from typing import List, Dict, Any, Optional
from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import NotFoundError, RequestError
import uuid
from openai import OpenAI

from src.config.settings import get_settings
from src.utils.logger import get_logger

# 설정과 로거 가져오기
settings = get_settings()
logger = get_logger(__name__)


class OpenSearchStore:
    """
    OpenSearch Vector Store 클래스

    🎯 주요 기능:
    1. 문서를 벡터로 변환하여 OpenSearch에 저장
    2. Hybrid 검색: 키워드 + 벡터 유사도 결합
    3. 태그/필터링 지원
    4. 조직별/사용자별 데이터 격리 (Multi-tenancy)

    📊 Vector Embedding이란?
    - 텍스트를 3072개의 숫자 배열로 변환한 것
    - 예: "안녕하세요" → [0.234, -0.123, 0.456, ..., 0.789] (3072개)
    - 비슷한 의미의 텍스트는 비슷한 숫자 패턴을 가짐
    - OpenAI의 text-embedding-3-large 모델 사용

    🔐 Multi-tenancy (다중 테넌트):
    - organization_id: 회사/조직 단위로 데이터 분리
    - user_id: 사용자 단위로 데이터 분리
    - 각 조직/사용자는 자신의 데이터만 검색 가능

    🔄 Hybrid Search (하이브리드 검색):
    - 키워드 매칭 + 벡터 유사도를 조합
    - 정확한 키워드는 높은 점수, 의미가 비슷하면 보너스 점수
    - 최고의 검색 정확도 제공
    """

    def __init__(
        self,
        index_name: str = "ai_documents",
        hosts: List[Dict[str, Any]] = None,
        http_auth: tuple = None,
        use_ssl: bool = False,
    ):
        """
        OpenSearch Store 초기화

        Args:
            index_name: OpenSearch 인덱스 이름
                       (관계형 DB의 '테이블'과 비슷한 개념)
            hosts: OpenSearch 서버 주소 리스트
                  예: [{"host": "localhost", "port": 9200}]
            http_auth: 인증 정보 (username, password)
                      예: ("admin", "admin")
            use_ssl: SSL/TLS 사용 여부 (HTTPS)
                    - True: HTTPS 사용 (프로덕션)
                    - False: HTTP 사용 (로컬 개발)

        💡 초기화 과정:
        1. OpenSearch 클라이언트 연결
        2. OpenAI 클라이언트 연결 (embedding 생성용)
        3. 인덱스가 없으면 자동 생성 (벡터 필드 포함)
        """
        self.index_name = index_name

        # OpenSearch 클라이언트 연결
        # - OpenSearch는 검색 엔진 + Vector DB
        # - 기존 플랫폼 데이터와 통합 가능
        if hosts is None:
            # 기본값: localhost (개발 환경)
            hosts = [{"host": "localhost", "port": 9200}]

        self.client = OpenSearch(
            hosts=hosts,
            http_auth=http_auth,
            use_ssl=use_ssl,  # SSL 설정
            verify_certs=False,  # 자체 서명 인증서 허용
            ssl_show_warn=False,  # SSL 경고 숨김
        )

        # OpenAI 클라이언트 연결
        # - 텍스트를 벡터로 변환(embedding)하는데 사용
        self.openai_client = OpenAI(api_key=settings.openai_api_key)

        # Embedding 모델 설정
        # - text-embedding-3-large: OpenAI의 최신 고성능 embedding 모델
        # - 3072차원의 벡터 생성 (숫자 3072개)
        self.embedding_model = "text-embedding-3-large"
        self.embedding_dimension = 3072  # 벡터의 차원 (크기)

        # 인덱스 초기화 (없으면 생성)
        self._ensure_index()

        logger.info(
            "OpenSearch Store 초기화 완료",
            index=index_name,
            embedding_model=self.embedding_model,
        )

    def _ensure_index(self) -> None:
        """
        OpenSearch 인덱스가 존재하는지 확인하고, 없으면 생성

        🗂️ 인덱스(Index)란?
        - 관계형 DB의 '테이블'과 비슷한 개념
        - 같은 구조의 문서 데이터를 모아두는 공간

        ⚙️ 인덱스 매핑 (Index Mapping):
        - 필드의 데이터 타입과 인덱싱 방법 정의
        - text: 전문 검색용 (키워드 분석)
        - keyword: 정확한 매칭용 (태그, ID 등)
        - knn_vector: 벡터 유사도 검색용
        """
        try:
            # 인덱스가 이미 존재하는지 확인
            if self.client.indices.exists(index=self.index_name):
                logger.info(f"인덱스 '{self.index_name}' 이미 존재함")
                return

            # 인덱스 매핑 정의
            # - 각 필드의 타입과 검색 방법 설정
            index_body = {
                "settings": {
                    # k-NN 플러그인 활성화
                    "index": {
                        "knn": True,  # 벡터 검색 활성화
                        "knn.algo_param.ef_search": 100,  # 검색 정확도 (높을수록 정확, 느림)
                    }
                },
                "mappings": {
                    "properties": {
                        # 문서 텍스트 (전문 검색 가능)
                        "text": {
                            "type": "text",  # 전문 검색
                            "analyzer": "standard",  # 표준 분석기
                        },
                        # 벡터 임베딩 (의미 기반 검색)
                        "embedding": {
                            "type": "knn_vector",  # 벡터 필드
                            "dimension": self.embedding_dimension,  # 3072
                            "method": {
                                "name": "hnsw",  # HNSW 알고리즘 (빠르고 정확)
                                "space_type": "cosinesimil",  # Cosine 유사도
                                "engine": "lucene",  # Lucene 엔진 (OpenSearch 3.0+ 권장)
                                "parameters": {
                                    "ef_construction": 128,  # 인덱스 구축 정확도
                                    "m": 24,  # 그래프 연결 수
                                },
                            },
                        },
                        # 조직 ID (필터링용)
                        "organization_id": {
                            "type": "keyword",  # 정확한 매칭
                        },
                        # 사용자 ID (필터링용, 선택)
                        "user_id": {
                            "type": "keyword",
                        },
                        # 태그 (필터링용)
                        "tags": {
                            "type": "keyword",  # 배열 가능
                        },
                        # 메타데이터 (동적 필드)
                        "metadata": {
                            "type": "object",  # JSON 객체
                            "enabled": True,
                        },
                        # 생성 시간
                        "created_at": {
                            "type": "date",
                        },
                    }
                },
            }

            # 인덱스 생성
            logger.info(f"인덱스 '{self.index_name}' 생성 중...")
            self.client.indices.create(index=self.index_name, body=index_body)
            logger.info(f"인덱스 '{self.index_name}' 생성 완료")

        except RequestError as e:
            if "resource_already_exists_exception" in str(e):
                logger.info(f"인덱스 '{self.index_name}' 이미 존재함")
            else:
                logger.error("인덱스 생성 중 오류", error=str(e))
                raise
        except Exception as e:
            logger.error("인덱스 확인/생성 중 오류", error=str(e))
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
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model,
            )

            # API 응답에서 embedding 벡터 추출
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
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        문서를 OpenSearch에 추가

        📥 전체 흐름:
        1. 텍스트를 벡터로 변환 (OpenAI API 호출)
        2. 메타데이터에 조직/사용자/태그 정보 추가
        3. OpenSearch에 저장 (인덱싱)

        Args:
            text: 저장할 문서 내용
            metadata: 문서의 메타데이터 (제목, 작성자, 날짜 등)
            organization_id: 조직 ID (필수)
            user_id: 사용자 ID (선택, 없으면 조직 전체 공유)
            tags: 태그 리스트 (선택)
                 예: ["프로젝트A", "일정", "중요"]

        Returns:
            생성된 문서의 고유 ID (UUID)

        💡 사용 예시:
        ```python
        store = OpenSearchStore()
        doc_id = store.add_document(
            text="프로젝트 A의 마감일은 2024년 12월 31일입니다.",
            metadata={
                "title": "프로젝트 A 일정",
                "author": "홍길동",
                "created_at": "2024-12-01",
            },
            organization_id="org_123",
            user_id="user_456",
            tags=["프로젝트A", "일정"],
        )
        print(f"문서 저장 완료: {doc_id}")
        ```

        🔐 Multi-tenancy:
        - organization_id로 조직별 데이터 분리
        - user_id로 사용자별 데이터 분리 (선택)
        - 검색 시 해당 조직/사용자 문서만 검색됨

        🏷️ 태그 활용:
        - 카테고리별 분류: ["기획", "개발", "테스트"]
        - 우선순위: ["긴급", "중요", "일반"]
        - 프로젝트: ["프로젝트A", "프로젝트B"]
        """
        try:
            # 1. 텍스트를 벡터로 변환
            logger.info("문서 embedding 생성 중...", text_length=len(text))
            embedding = self._create_embedding(text)

            # 2. 고유 ID 생성
            doc_id = str(uuid.uuid4())

            # 3. 문서 본문 구성
            from datetime import datetime

            document = {
                "text": text,
                "embedding": embedding,
                "organization_id": organization_id,
                "metadata": metadata,
                "created_at": datetime.utcnow().isoformat(),
            }

            # 선택적 필드 추가
            if user_id:
                document["user_id"] = user_id

            if tags:
                document["tags"] = tags

            # 4. OpenSearch에 저장
            # - index: 인덱스 이름
            # - id: 문서 ID
            # - body: 문서 내용
            self.client.index(
                index=self.index_name,
                id=doc_id,
                body=document,
                refresh=True,  # 즉시 검색 가능하도록 refresh
            )

            logger.info(
                "문서 저장 완료",
                doc_id=doc_id,
                organization_id=organization_id,
                user_id=user_id,
                tags=tags,
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
        tags: Optional[List[str]] = None,
        limit: int = 5,
        use_hybrid: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        질문과 유사한 문서 검색 (Hybrid Search)

        🔍 검색 흐름:
        1. 질문을 벡터로 변환
        2. OpenSearch에서 검색:
           - 키워드 매칭 (정확한 단어 일치)
           - 벡터 유사도 (의미 유사성)
           - 둘을 조합하여 최적 결과 반환
        3. 조직/사용자/태그 필터링 적용
        4. 유사도 점수가 높은 순으로 반환

        Args:
            query: 검색할 질문/키워드
            organization_id: 조직 ID (필수)
            user_id: 사용자 ID (선택, 있으면 해당 사용자 문서만 검색)
            tags: 태그 필터 (선택)
                 예: ["프로젝트A", "일정"] → 이 태그가 있는 문서만
            limit: 최대 검색 결과 개수 (기본 5개)
            use_hybrid: Hybrid 검색 사용 여부
                       - True: 키워드 + 벡터 조합 (권장)
                       - False: 벡터만 사용

        Returns:
            검색 결과 리스트 (유사도 높은 순)
            각 결과 형식:
            {
                "id": "문서 ID",
                "score": 0.85,  # 유사도 점수 (높을수록 관련성 높음)
                "text": "문서 내용",
                "metadata": {...},
                "tags": ["태그1", "태그2"],
            }

        💡 사용 예시:
        ```python
        store = OpenSearchStore()

        # 기본 검색 (Hybrid)
        results = store.search(
            query="프로젝트 A 마감일이 언제야?",
            organization_id="org_123",
            limit=3,
        )

        # 태그 필터링
        results = store.search(
            query="일정 확인",
            organization_id="org_123",
            tags=["프로젝트A"],  # 프로젝트A 태그만
            limit=5,
        )

        # 사용자별 검색
        results = store.search(
            query="내 문서",
            organization_id="org_123",
            user_id="user_456",  # 특정 사용자 문서만
        )
        ```

        🎯 Hybrid Search의 장점:
        - 정확한 키워드 매칭: "프로젝트 A" → "프로젝트 A" 포함 문서 우선
        - 의미 유사도: "마감일" → "deadline", "종료일" 등도 검색
        - 최고의 정확도와 재현율 (Precision & Recall)

        🔐 보안:
        - organization_id 필터: 다른 조직 문서는 절대 검색 안됨
        - user_id 필터: 다른 사용자 문서는 절대 검색 안됨
        - 태그 필터: 지정된 태그가 있는 문서만 검색
        """
        try:
            # 1. 질문을 벡터로 변환
            logger.info("검색 쿼리 embedding 생성 중...", query=query)
            query_embedding = self._create_embedding(query)

            # 2. 필터 조건 구성
            # - organization_id는 필수 필터
            # - user_id, tags는 선택적 필터
            filter_conditions = [
                {"term": {"organization_id": organization_id}}
            ]

            # 사용자 ID 필터 추가 (있는 경우)
            if user_id:
                filter_conditions.append({"term": {"user_id": user_id}})

            # 태그 필터 추가 (있는 경우)
            if tags:
                filter_conditions.append({"terms": {"tags": tags}})

            # 3. 검색 쿼리 구성
            if use_hybrid:
                # Hybrid 검색: 키워드 + 벡터
                # - should 절: 여러 조건 중 하나라도 만족하면 점수 부여
                # - match: 키워드 매칭 (텍스트 분석)
                # - knn: 벡터 유사도
                search_body = {
                    "size": limit,
                    "query": {
                        "bool": {
                            "must": filter_conditions,  # 필수 조건 (조직/사용자)
                            "should": [
                                # 키워드 검색 (가중치 1.0)
                                {
                                    "match": {
                                        "text": {
                                            "query": query,
                                            "boost": 1.0,  # 키워드 매칭 가중치
                                        }
                                    }
                                },
                                # 벡터 검색 (가중치 2.0)
                                {
                                    "knn": {
                                        "embedding": {
                                            "vector": query_embedding,
                                            "k": limit * 2,  # 후보 개수
                                        }
                                    }
                                },
                            ],
                            "minimum_should_match": 1,  # 최소 1개는 매칭
                        }
                    },
                }
            else:
                # 벡터만 사용
                search_body = {
                    "size": limit,
                    "query": {
                        "bool": {
                            "must": [
                                *filter_conditions,
                                {
                                    "knn": {
                                        "embedding": {
                                            "vector": query_embedding,
                                            "k": limit,
                                        }
                                    }
                                },
                            ]
                        }
                    },
                }

            # 4. OpenSearch에서 검색 실행
            response = self.client.search(index=self.index_name, body=search_body)

            # 5. 결과 정리
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "id": hit["_id"],
                    "score": hit["_score"],  # 유사도 점수
                    "text": hit["_source"].get("text", ""),
                    "metadata": hit["_source"].get("metadata", {}),
                    "tags": hit["_source"].get("tags", []),
                }
                results.append(result)

            logger.info(
                "검색 완료",
                query=query,
                results_count=len(results),
                organization_id=organization_id,
                user_id=user_id,
                tags=tags,
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
        store = OpenSearchStore()
        success = store.delete_document("550e8400-e29b-41d4-a716-446655440000")
        if success:
            print("문서 삭제 완료")
        ```
        """
        try:
            # OpenSearch에서 문서 삭제
            self.client.delete(
                index=self.index_name,
                id=doc_id,
                refresh=True,  # 즉시 반영
            )

            logger.info("문서 삭제 완료", doc_id=doc_id)
            return True

        except NotFoundError:
            logger.warning("문서를 찾을 수 없음", doc_id=doc_id)
            return False
        except Exception as e:
            logger.error("문서 삭제 실패", doc_id=doc_id, error=str(e))
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """
        인덱스 통계 조회

        Returns:
            인덱스 통계 정보
            {
                "name": "ai_documents",
                "document_count": 1234,  # 저장된 문서 개수
                "size_in_bytes": 12345678,
                "primary_shards": 1,
            }

        💡 사용 예시:
        ```python
        store = OpenSearchStore()
        stats = store.get_index_stats()
        print(f"저장된 문서 수: {stats['document_count']}")
        ```
        """
        try:
            # OpenSearch에서 인덱스 통계 가져오기
            stats = self.client.indices.stats(index=self.index_name)
            index_stats = stats["indices"][self.index_name]

            return {
                "name": self.index_name,
                "document_count": index_stats["total"]["docs"]["count"],
                "size_in_bytes": index_stats["total"]["store"]["size_in_bytes"],
                "primary_shards": index_stats["primaries"]["docs"]["count"],
            }

        except Exception as e:
            logger.error("인덱스 통계 조회 실패", error=str(e))
            # 인덱스가 없거나 오류 시 기본값 반환
            return {
                "name": self.index_name,
                "document_count": 0,
                "size_in_bytes": 0,
                "primary_shards": 0,
            }
