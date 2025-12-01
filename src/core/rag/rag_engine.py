"""
RAG (Retrieval-Augmented Generation) 엔진 구현

📚 RAG란?
- Retrieval: 관련 문서 검색
- Augmented: 검색된 문서로 보강
- Generation: 보강된 정보로 답변 생성

🎯 왜 RAG를 사용하나?
- LLM(ChatGPT 등)은 학습 시점까지의 정보만 알고 있음
- 최신 정보, 회사 내부 문서 등은 모름
- RAG를 통해 필요한 정보를 실시간으로 제공하여 정확한 답변 생성

💡 RAG 동작 흐름:
1. 사용자 질문 입력
2. 질문과 관련된 문서 검색 (Vector DB에서)
3. 검색된 문서 + 질문을 LLM에 전달
4. LLM이 문서 내용을 바탕으로 답변 생성

예시:
질문: "프로젝트 A 마감일이 언제야?"
→ Vector DB에서 "프로젝트 A 관련 문서" 검색
→ 검색 결과: "프로젝트 A의 마감일은 2024년 12월 31일입니다."
→ LLM에게 전달: "다음 문서를 참고해서 답변해: [검색 결과] 질문: 프로젝트 A 마감일이 언제야?"
→ LLM 답변: "프로젝트 A의 마감일은 2024년 12월 31일입니다."
"""

from typing import List, Dict, Any, Optional, Union
from openai import OpenAI

from src.core.rag.opensearch_store import OpenSearchStore
from src.core.rag.qdrant_store import QdrantStore
from src.config.settings import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class RAGEngine:
    """
    RAG 엔진 클래스

    🎯 주요 기능:
    1. 문서 추가 (Indexing)
    2. 질문-답변 (Query)
    3. 문서 검색 (Search)

    🔧 구성 요소:
    - QdrantStore: Vector DB 관리 (문서 저장/검색)
    - OpenAI Client: LLM 답변 생성
    """

    def __init__(
        self,
        vector_store: Optional[Union[OpenSearchStore, QdrantStore]] = None,
        llm_model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        use_opensearch: bool = True,
    ):
        """
        RAG 엔진 초기화

        Args:
            vector_store: Vector Store 인스턴스 (없으면 자동 생성)
            llm_model: 사용할 LLM 모델
                     - gpt-4o: 최신 고성능 모델 (추천)
                     - gpt-4o-mini: 빠르고 저렴한 모델
                     - gpt-4-turbo: 긴 컨텍스트 처리에 강함
            temperature: 답변의 창의성 (0~1)
                       - 0.0: 항상 같은 답변 (일관성 높음)
                       - 0.7: 적절한 창의성 (기본값, 추천)
                       - 1.0: 매우 창의적 (일관성 낮음)
            max_tokens: 최대 답변 길이
                      - 토큰: 단어의 작은 조각 (한글 1글자 ≈ 2-3토큰)
                      - 2000: 약 700-1000자 정도
            use_opensearch: OpenSearch 사용 여부
                          - True: OpenSearch 사용 (기본, 권장)
                          - False: Qdrant 사용 (레거시)
        """
        # Vector Store 초기화
        # - 문서를 벡터로 저장하고 검색하는 역할
        if vector_store:
            self.vector_store = vector_store
        elif use_opensearch:
            # OpenSearch 사용 (기본)
            # - 기존 플랫폼 데이터 활용
            # - 태깅 + 벡터 검색 지원
            self.vector_store = OpenSearchStore(
                index_name=settings.opensearch_index,
                hosts=[
                    {
                        "host": settings.opensearch_host,
                        "port": settings.opensearch_port,
                    }
                ],
                http_auth=(
                    settings.opensearch_user,
                    settings.opensearch_password,
                ),
                use_ssl=settings.opensearch_use_ssl,
            )
        else:
            # Qdrant 사용 (레거시)
            self.vector_store = QdrantStore()

        # OpenAI 클라이언트 초기화
        # - LLM API 호출용
        self.openai_client = OpenAI(api_key=settings.openai_api_key)

        # LLM 설정
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Vector Store 타입 확인
        store_type = "OpenSearch" if isinstance(self.vector_store, OpenSearchStore) else "Qdrant"

        logger.info(
            "RAG 엔진 초기화 완료",
            vector_store=store_type,
            llm_model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def add_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        organization_id: str,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        문서를 RAG 시스템에 추가

        📥 Indexing이란?
        - 문서를 검색 가능한 형태로 변환하여 저장하는 과정
        - 텍스트 → 벡터 변환 → Vector DB 저장

        Args:
            text: 문서 내용
            metadata: 문서 메타데이터
                    예: {"title": "프로젝트 A", "author": "홍길동"}
            organization_id: 조직 ID
            user_id: 사용자 ID (선택)
            tags: 태그 리스트 (선택, OpenSearch 사용 시 유용)
                 예: ["프로젝트A", "일정", "중요"]

        Returns:
            생성된 문서 ID

        💡 사용 예시:
        ```python
        rag = RAGEngine()
        doc_id = rag.add_document(
            text="프로젝트 A의 마감일은 2024년 12월 31일입니다.",
            metadata={
                "title": "프로젝트 A 일정",
                "created_at": "2024-12-01",
            },
            organization_id="org_123",
            tags=["프로젝트A", "일정"],  # 태그 추가
        )
        ```
        """
        try:
            logger.info(
                "문서 추가 시작",
                text_length=len(text),
                organization_id=organization_id,
                tags=tags,
            )

            # Vector Store에 문서 저장
            # - OpenSearch: embedding + 태그 저장
            # - Qdrant: embedding만 저장 (태그 무시)
            if isinstance(self.vector_store, OpenSearchStore):
                doc_id = self.vector_store.add_document(
                    text=text,
                    metadata=metadata,
                    organization_id=organization_id,
                    user_id=user_id,
                    tags=tags,
                )
            else:
                # Qdrant는 tags 파라미터 미지원
                doc_id = self.vector_store.add_document(
                    text=text,
                    metadata=metadata,
                    organization_id=organization_id,
                    user_id=user_id,
                )

            logger.info("문서 추가 완료", doc_id=doc_id)
            return doc_id

        except Exception as e:
            logger.error("문서 추가 실패", error=str(e))
            raise

    def search_documents(
        self,
        query: str,
        organization_id: str,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        질문과 관련된 문서 검색

        🔍 Semantic Search (의미 기반 검색):
        - 키워드가 정확히 일치하지 않아도 의미가 비슷하면 검색됨
        - 예: "강아지" 검색 → "개", "반려동물" 문서도 검색

        Args:
            query: 검색 질문/키워드
            organization_id: 조직 ID
            user_id: 사용자 ID (선택)
            tags: 태그 필터 (선택, OpenSearch 사용 시 유용)
                 예: ["프로젝트A"] → 프로젝트A 태그가 있는 문서만
            limit: 최대 결과 개수

        Returns:
            검색 결과 리스트
            [
                {
                    "id": "문서 ID",
                    "score": 0.85,  # 유사도
                    "text": "문서 내용",
                    "metadata": {...},
                    "tags": ["태그1", "태그2"],  # OpenSearch만
                },
                ...
            ]

        💡 사용 예시:
        ```python
        rag = RAGEngine()

        # 기본 검색
        results = rag.search_documents(
            query="프로젝트 마감일",
            organization_id="org_123",
            limit=3,
        )

        # 태그 필터링 (OpenSearch)
        results = rag.search_documents(
            query="일정 확인",
            organization_id="org_123",
            tags=["프로젝트A"],  # 프로젝트A 태그만
            limit=5,
        )

        for result in results:
            print(f"{result['score']:.2f} - {result['text']}")
        ```
        """
        try:
            logger.info("문서 검색 시작", query=query, tags=tags)

            # Vector Store에서 유사 문서 검색
            # - OpenSearch: 태그 필터링 지원
            # - Qdrant: 태그 무시
            if isinstance(self.vector_store, OpenSearchStore):
                results = self.vector_store.search(
                    query=query,
                    organization_id=organization_id,
                    user_id=user_id,
                    tags=tags,
                    limit=limit,
                )
            else:
                # Qdrant는 tags 파라미터 미지원
                results = self.vector_store.search(
                    query=query,
                    organization_id=organization_id,
                    user_id=user_id,
                    limit=limit,
                )

            logger.info("문서 검색 완료", results_count=len(results))
            return results

        except Exception as e:
            logger.error("문서 검색 실패", error=str(e))
            raise

    def generate_answer(
        self,
        query: str,
        organization_id: str,
        user_id: Optional[str] = None,
        context_limit: int = 5,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        RAG를 사용하여 질문에 답변 생성

        🎯 전체 RAG 파이프라인:
        1. 질문과 관련된 문서 검색 (Vector DB)
        2. 검색된 문서를 컨텍스트로 정리
        3. 프롬프트 생성 (시스템 메시지 + 컨텍스트 + 질문)
        4. LLM에 전달하여 답변 생성
        5. 답변 + 참고 문서 반환

        Args:
            query: 사용자 질문
            organization_id: 조직 ID
            user_id: 사용자 ID (선택)
            context_limit: 컨텍스트로 사용할 최대 문서 개수
            stream: 스트리밍 모드 (미구현, 추후 확장)

        Returns:
            {
                "answer": "생성된 답변",
                "sources": [참고한 문서 리스트],
                "model": "사용한 LLM 모델",
            }

        💡 사용 예시:
        ```python
        rag = RAGEngine()

        # 먼저 관련 문서들을 추가
        rag.add_document(
            text="프로젝트 A의 마감일은 2024년 12월 31일입니다.",
            metadata={"title": "프로젝트 A"},
            organization_id="org_123",
        )

        # 질문하기
        result = rag.generate_answer(
            query="프로젝트 A 마감일이 언제야?",
            organization_id="org_123",
        )

        print(result["answer"])
        # → "프로젝트 A의 마감일은 2024년 12월 31일입니다."

        print(f"참고 문서: {len(result['sources'])}개")
        ```

        🔍 컨텍스트(Context)란?
        - LLM에게 제공하는 배경 지식/참고 자료
        - 검색된 문서들을 정리하여 LLM에게 전달
        - LLM은 이 컨텍스트를 바탕으로 답변 생성

        💰 비용:
        - Vector Search (embedding): $0.00013 / 1K tokens
        - LLM API (gpt-4o): $2.50 / 1M input tokens, $10.00 / 1M output tokens
        - 예: 질문 1개 + 문서 3개 (각 500자) + 답변 200자
          → embedding: $0.0005 + LLM: $0.005 = 약 $0.0055
        """
        try:
            logger.info("답변 생성 시작", query=query)

            # 1단계: 관련 문서 검색
            # - Vector DB에서 질문과 유사한 문서 찾기
            logger.info("관련 문서 검색 중...")
            search_results = self.search_documents(
                query=query,
                organization_id=organization_id,
                user_id=user_id,
                limit=context_limit,
            )

            # 검색 결과가 없으면 문서 없이 답변
            if not search_results:
                logger.warning("검색 결과 없음 - 일반 LLM 답변으로 대체")
                return self._generate_without_context(query)

            # 2단계: 검색된 문서를 컨텍스트로 정리
            # - 여러 문서를 하나의 문자열로 합치기
            context = self._build_context(search_results)
            logger.info(
                "컨텍스트 생성 완료",
                context_length=len(context),
                sources_count=len(search_results),
            )

            # 3단계: 프롬프트 생성
            # - 시스템 메시지: AI의 역할과 행동 지침
            # - 사용자 메시지: 컨텍스트 + 질문
            messages = [
                {
                    "role": "system",
                    "content": self._get_system_prompt(),
                },
                {
                    "role": "user",
                    "content": self._build_user_prompt(context, query),
                },
            ]

            # 4단계: LLM API 호출하여 답변 생성
            logger.info("LLM 답변 생성 중...", model=self.llm_model)
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # 5단계: 답변 추출
            answer = response.choices[0].message.content

            # 6단계: 결과 정리 및 반환
            result = {
                "answer": answer,
                "sources": [
                    {
                        "text": src["text"],
                        "score": src["score"],
                        "metadata": src["metadata"],
                    }
                    for src in search_results
                ],
                "model": self.llm_model,
            }

            logger.info(
                "답변 생성 완료",
                answer_length=len(answer),
                sources_count=len(search_results),
            )

            return result

        except Exception as e:
            logger.error("답변 생성 실패", error=str(e))
            raise

    def _build_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        검색 결과를 컨텍스트 문자열로 변환

        📝 컨텍스트 형식:
        ```
        [문서 1] (유사도: 0.92)
        프로젝트 A의 마감일은 2024년 12월 31일입니다.

        [문서 2] (유사도: 0.85)
        프로젝트 A 담당자는 홍길동입니다.
        ```

        Args:
            search_results: Vector DB 검색 결과

        Returns:
            정리된 컨텍스트 문자열
        """
        context_parts = []

        for i, result in enumerate(search_results, 1):
            # 각 문서를 "[문서 N] (유사도: 0.XX)" 형식으로 추가
            score = result.get("score", 0.0)
            text = result.get("text", "")

            context_parts.append(
                f"[문서 {i}] (유사도: {score:.2f})\n{text}"
            )

        # 문서들을 빈 줄로 구분하여 합치기
        return "\n\n".join(context_parts)

    def _get_system_prompt(self) -> str:
        """
        시스템 프롬프트 생성

        🎭 System Prompt란?
        - AI의 역할과 행동 방식을 정의하는 지침
        - 답변 스타일, 제약 사항 등을 명시
        - 모든 대화에 일관되게 적용됨

        Returns:
            시스템 프롬프트 문자열

        💡 프롬프트 구성 요소:
        - 역할 정의: "당신은 협업 플랫폼의 AI 어시스턴트입니다"
        - 행동 지침: "제공된 문서를 바탕으로 답변하세요"
        - 제약 사항: "모르면 모른다고 말하세요"
        """
        return """당신은 협업 플랫폼 Cowexa의 AI 어시스턴트입니다.

역할:
- 사용자의 질문에 정확하고 친절하게 답변합니다
- 제공된 문서(컨텍스트)를 바탕으로 답변합니다
- 문서에 없는 내용은 추측하지 않습니다

답변 지침:
1. 제공된 문서의 내용을 우선적으로 사용하세요
2. 문서에 정보가 없으면 "제공된 문서에서 관련 정보를 찾을 수 없습니다"라고 답변하세요
3. 답변은 명확하고 간결하게 작성하세요
4. 필요시 문서 번호를 인용하여 출처를 명시하세요 (예: [문서 1]에 따르면...)

주의사항:
- 개인정보나 민감한 정보는 신중하게 다루세요
- 확실하지 않은 내용은 추측하지 마세요
- 항상 친절하고 전문적인 톤을 유지하세요
"""

    def _build_user_prompt(self, context: str, query: str) -> str:
        """
        사용자 프롬프트 생성

        📝 User Prompt 구조:
        ```
        다음은 참고할 문서들입니다:

        [문서 1] (유사도: 0.92)
        ...

        [문서 2] (유사도: 0.85)
        ...

        질문: 프로젝트 A 마감일이 언제야?
        ```

        Args:
            context: 검색된 문서들의 컨텍스트
            query: 사용자 질문

        Returns:
            완성된 프롬프트
        """
        return f"""다음은 참고할 문서들입니다:

{context}

질문: {query}

위 문서들을 참고하여 질문에 답변해주세요."""

    def _generate_without_context(self, query: str) -> Dict[str, Any]:
        """
        컨텍스트 없이 일반 LLM 답변 생성

        ⚠️ 언제 사용?
        - 검색 결과가 없을 때 (관련 문서가 DB에 없음)
        - fallback 메커니즘

        Args:
            query: 사용자 질문

        Returns:
            답변 결과 (sources는 빈 리스트)

        💡 동작:
        - Vector DB 검색 결과가 없어도 LLM의 일반 지식으로 답변
        - 다만, 회사 내부 정보는 답변 불가
        """
        logger.info("컨텍스트 없이 답변 생성", query=query)

        messages = [
            {
                "role": "system",
                "content": """당신은 협업 플랫폼 Cowexa의 AI 어시스턴트입니다.

사용자의 질문에 답변하되, 관련 문서를 찾을 수 없었음을 알려주세요.
일반적인 정보는 제공할 수 있지만, 회사 내부 정보나 특정 프로젝트 정보는 문서가 필요합니다.""",
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        response = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": [],  # 참고 문서 없음
            "model": self.llm_model,
        }

    def delete_document(self, doc_id: str) -> bool:
        """
        문서 삭제

        Args:
            doc_id: 삭제할 문서 ID

        Returns:
            삭제 성공 여부
        """
        try:
            return self.vector_store.delete_document(doc_id)
        except Exception as e:
            logger.error("문서 삭제 실패", error=str(e))
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        RAG 시스템 통계 조회

        Returns:
            {
                "total_documents": 1234,  # 총 문서 수
                "vector_store": {...},     # Vector Store 정보
                "llm_model": "gpt-4o",     # 사용 중인 LLM 모델
            }
        """
        try:
            # OpenSearch와 Qdrant에서 서로 다른 메서드 사용
            if isinstance(self.vector_store, OpenSearchStore):
                vector_store_info = self.vector_store.get_index_stats()
                total_docs = vector_store_info.get("document_count", 0)
            else:
                vector_store_info = self.vector_store.get_collection_info()
                total_docs = vector_store_info.get("vectors_count", 0)

            return {
                "total_documents": total_docs,
                "vector_store": vector_store_info,
                "llm_model": self.llm_model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        except Exception as e:
            logger.error("통계 조회 실패", error=str(e))
            raise
