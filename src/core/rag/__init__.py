"""
RAG (Retrieval-Augmented Generation) 모듈

📦 제공하는 클래스:
- OpenSearchStore: Vector DB 관리 (문서 저장/검색) - 권장
- QdrantStore: Vector DB 관리 (문서 저장/검색) - 레거시
- RAGEngine: 전체 RAG 파이프라인 (검색 + 답변 생성)

💡 사용 예시:
```python
from src.core.rag import RAGEngine

# RAG 엔진 생성 (OpenSearch 사용)
rag = RAGEngine()

# 문서 추가
rag.add_document(
    text="프로젝트 A의 마감일은 2024년 12월 31일입니다.",
    metadata={"title": "프로젝트 A"},
    organization_id="org_123",
    tags=["프로젝트A", "일정"],  # 태그 추가 가능
)

# 질문하기
result = rag.generate_answer(
    query="프로젝트 A 마감일이 언제야?",
    organization_id="org_123",
)

print(result["answer"])
```
"""

from src.core.rag.opensearch_store import OpenSearchStore
from src.core.rag.qdrant_store import QdrantStore
from src.core.rag.rag_engine import RAGEngine

__all__ = ["OpenSearchStore", "QdrantStore", "RAGEngine"]
