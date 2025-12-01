# AI Assistant 소스코드 학습 가이드

> **목표**: FastAPI + OpenSearch + OpenAI를 활용한 RAG 기반 AI Assistant 시스템을 단계별로 이해하기

---

## 📋 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [사전 지식](#2-사전-지식)
3. [학습 로드맵](#3-학습-로드맵)
4. [단계별 학습 가이드](#4-단계별-학습-가이드)
5. [실습 예제](#5-실습-예제)
6. [트러블슈팅](#6-트러블슈팅)

---

## 1. 프로젝트 개요

### 1.1 시스템 아키텍처

```
┌─────────────┐
│   Client    │
│  (Browser)  │
└──────┬──────┘
       │ HTTP/REST
       ▼
┌─────────────────────────────────────────┐
│         FastAPI Application             │
│  ┌────────────────────────────────┐    │
│  │      API Layer (REST)          │    │
│  │  - chat.py                     │    │
│  │  - documents.py                │    │
│  │  - health.py                   │    │
│  └────────────┬───────────────────┘    │
│               ▼                         │
│  ┌────────────────────────────────┐    │
│  │      Core Business Logic       │    │
│  │  ┌──────────────────────────┐  │    │
│  │  │   RAG Engine             │  │    │
│  │  │  - 문서 검색             │  │    │
│  │  │  - 하이브리드 서치       │  │    │
│  │  └──────────────────────────┘  │    │
│  │  ┌──────────────────────────┐  │    │
│  │  │   LLM Client (OpenAI)    │  │    │
│  │  │  - GPT-4o 호출           │  │    │
│  │  │  - 답변 생성             │  │    │
│  │  └──────────────────────────┘  │    │
│  └────────────────────────────────┘    │
└────────┬────────────┬───────────────────┘
         │            │
         ▼            ▼
┌────────────┐  ┌──────────────┐
│ OpenSearch │  │   OpenAI     │
│  (Vector   │  │   API        │
│   Store)   │  │ (GPT-4o)     │
└────────────┘  └──────────────┘
```

### 1.2 핵심 기능

1. **문서 관리**: 문서 추가/검색/삭제 (조직/사용자별 격리)
2. **벡터 검색**: OpenSearch k-NN을 활용한 의미 기반 검색
3. **하이브리드 검색**: 키워드 매칭 + 벡터 유사도 결합
4. **RAG 질의응답**: 검색된 문서 기반 AI 답변 생성
5. **대화 이력 관리**: Redis 기반 세션 관리

### 1.3 기술 스택

| 영역 | 기술 | 용도 |
|------|------|------|
| **웹 프레임워크** | FastAPI 0.115.5 | REST API 서버 |
| **LLM** | OpenAI GPT-4o | 자연어 생성 |
| **벡터 DB** | OpenSearch 3.3.2 | 문서 저장 + 벡터 검색 |
| **임베딩** | OpenAI text-embedding-3-large | 텍스트 → 벡터 변환 (3072차원) |
| **캐시/세션** | Redis | 대화 이력 저장 |
| **로깅** | Structlog | 구조화된 JSON 로깅 |

---

## 2. 사전 지식

### 2.1 필수 개념

학습 전에 다음 개념을 이해하고 있어야 합니다:

#### Python 기초
- [ ] 타입 힌팅 (Type Hints)
- [ ] 비동기 프로그래밍 (async/await)
- [ ] 데코레이터 (Decorator)
- [ ] Pydantic 모델

#### FastAPI
- [ ] 라우터와 엔드포인트
- [ ] Request/Response 모델
- [ ] 의존성 주입 (Dependency Injection)
- [ ] CORS 설정

#### RAG (Retrieval-Augmented Generation)
- [ ] 임베딩 (Embedding)이란?
- [ ] 벡터 검색 원리
- [ ] k-NN (k-Nearest Neighbors)
- [ ] 코사인 유사도 (Cosine Similarity)

#### OpenSearch
- [ ] 인덱스와 문서 개념
- [ ] k-NN 벡터 검색
- [ ] HNSW 알고리즘
- [ ] 하이브리드 검색 (키워드 + 벡터)

### 2.2 권장 학습 자료

1. **FastAPI 공식 문서**: https://fastapi.tiangolo.com/
2. **OpenSearch k-NN 가이드**: https://opensearch.org/docs/latest/search-plugins/knn/
3. **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings
4. **RAG 개념 이해**: https://www.pinecone.io/learn/retrieval-augmented-generation/

---

## 3. 학습 로드맵

### 3.1 전체 학습 순서 (총 8단계)

```
📚 Level 1: 기초 (1-2일)
└─ Step 1: 프로젝트 구조 파악
└─ Step 2: 환경 설정 이해

🔧 Level 2: 유틸리티 (1일)
└─ Step 3: 로깅과 Redis 클라이언트

📦 Level 3: 데이터 모델 (1일)
└─ Step 4: Pydantic 모델 구조

🧠 Level 4: 핵심 로직 (3-4일) ★ 가장 중요 ★
└─ Step 5: OpenAI 클라이언트
└─ Step 6: OpenSearch 벡터 스토어
└─ Step 7: RAG 엔진

🌐 Level 5: API 레이어 (2일)
└─ Step 8: REST API 엔드포인트

🚀 Level 6: 통합 및 실습 (1-2일)
└─ Step 9: 전체 플로우 이해
└─ Step 10: 실습 예제
```

### 3.2 예상 소요 시간

- **빠른 학습**: 5-7일 (하루 4시간)
- **깊이 있는 학습**: 10-14일 (하루 2-3시간)
- **완벽한 이해**: 3주 이상 (실습 포함)

---

## 4. 단계별 학습 가이드

---

## 📚 Step 1: 프로젝트 구조 파악 (30분)

### 디렉토리 구조

```
ai-assistant/
├── src/
│   ├── main.py                 # ⭐ FastAPI 앱 진입점
│   ├── config/
│   │   └── settings.py         # ⭐ 환경 변수 및 설정
│   ├── utils/
│   │   ├── logger.py           # 구조화된 로깅
│   │   └── redis_client.py     # Redis 연결
│   ├── models/
│   │   └── chat.py             # Pydantic 데이터 모델
│   ├── core/                   # ⭐ 핵심 비즈니스 로직
│   │   ├── llm/
│   │   │   └── openai_client.py    # OpenAI API 호출
│   │   ├── rag/
│   │   │   ├── opensearch_store.py # ⭐ OpenSearch 벡터 DB
│   │   │   └── rag_engine.py       # ⭐ RAG 엔진 (핵심!)
│   │   ├── memory/             # 대화 이력 관리
│   │   └── agent/              # AI 에이전트 (향후 확장)
│   └── api/
│       └── rest/
│           ├── chat.py         # ⭐ 채팅 API
│           ├── documents.py    # ⭐ 문서 관리 API
│           └── health.py       # 헬스 체크
├── .env                        # 환경 변수 (비밀!)
├── requirements.txt
└── test_opensearch_connection.py  # OpenSearch 연결 테스트
```

### 중요도별 파일 우선순위

#### ⭐⭐⭐ 필수 (꼭 이해해야 함)
1. `src/main.py` - 애플리케이션 진입점
2. `src/config/settings.py` - 모든 설정의 중앙 관리
3. `src/core/rag/opensearch_store.py` - 벡터 검색의 핵심
4. `src/core/rag/rag_engine.py` - RAG 로직의 중심
5. `src/api/rest/chat.py` - 사용자 인터페이스
6. `src/api/rest/documents.py` - 문서 관리 API

#### ⭐⭐ 중요 (이해하면 좋음)
7. `src/core/llm/openai_client.py` - LLM 통신
8. `src/utils/logger.py` - 로깅 시스템
9. `src/models/chat.py` - 데이터 모델

#### ⭐ 선택 (필요시 참고)
10. `src/utils/redis_client.py` - 세션 관리

---

## 📚 Step 2: 환경 설정 이해 (1시간)

### 파일: `src/config/settings.py`

**학습 목표**:
- Pydantic Settings 사용법 이해
- 환경 변수 관리 방법 학습

**핵심 코드 읽기**:

```python
# src/config/settings.py

from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """
    환경 변수를 자동으로 로드하는 설정 클래스
    .env 파일의 값을 자동으로 매핑합니다
    """

    # OpenAI 설정
    openai_api_key: str = Field(...)  # 필수 값
    openai_model: str = Field(default="gpt-4o")  # 기본값 지정

    # OpenSearch 설정
    opensearch_host: str = Field(default="localhost")
    opensearch_port: int = Field(default=9200)

    class Config:
        env_file = ".env"  # .env 파일에서 자동 로드
        case_sensitive = False  # 대소문자 구분 안함
```

**학습 포인트**:
1. `BaseSettings` 상속으로 자동 환경 변수 로딩
2. `Field(...)`는 필수, `Field(default=값)`은 선택
3. 타입 힌팅으로 자동 타입 변환 (예: "9200" → 9200)

**실습**:
```bash
# .env 파일 확인
cat .env | grep OPENSEARCH

# Python 콘솔에서 테스트
python3 -c "from src.config.settings import settings; print(settings.opensearch_host)"
```

---

## 🔧 Step 3: 로깅과 유틸리티 (1시간)

### 파일: `src/utils/logger.py`

**학습 목표**:
- Structlog를 사용한 구조화된 로깅
- JSON 형식 로그의 장점 이해

**핵심 개념**:

```python
# src/utils/logger.py

import structlog

# 구조화된 로거 설정
structlog.configure(
    processors=[
        structlog.processors.add_log_level,  # 로그 레벨 추가
        structlog.processors.TimeStamper(fmt="iso"),  # ISO 시간 추가
        structlog.processors.JSONRenderer()  # JSON 형식으로 출력
    ]
)

logger = structlog.get_logger()

# 사용 예
logger.info("사용자 로그인", user_id="user123", organization_id="org456")
# 출력: {"event":"사용자 로그인","user_id":"user123","organization_id":"org456","level":"info","timestamp":"2025-12-01T15:00:00"}
```

**왜 JSON 로그인가?**
- ✅ 검색 가능 (ELK/Splunk 등)
- ✅ 구조화된 데이터
- ✅ 자동 필터링/집계 가능

---

## 📦 Step 4: 데이터 모델 (1-2시간)

### 파일: `src/models/chat.py`

**학습 목표**:
- Pydantic 모델로 요청/응답 검증
- 타입 안전성 확보

**핵심 코드**:

```python
# src/models/chat.py

from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="사용자 메시지"
    )
    organization_id: str = Field(..., description="조직 ID")
    user_id: str = Field(..., description="사용자 ID")
    session_id: Optional[str] = Field(None, description="세션 ID (선택)")

class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    session_id: str
    message: str
    sources: List[dict]
    timestamp: str
```

**Pydantic의 장점**:
1. **자동 검증**: `min_length`, `max_length` 자동 체크
2. **타입 변환**: 문자열 → 정수 자동 변환
3. **문서화**: FastAPI Swagger에 자동 표시
4. **IDE 지원**: 자동완성과 타입 체크

**실습**:
```python
# Python 콘솔에서
from src.models.chat import ChatRequest

# 정상 케이스
req = ChatRequest(message="안녕", organization_id="org1", user_id="user1")
print(req.message)

# 에러 케이스 (자동 검증)
try:
    req = ChatRequest(message="", organization_id="org1")  # 빈 메시지, user_id 누락
except Exception as e:
    print(e)
```

---

## 🧠 Step 5: OpenAI 클라이언트 (2시간)

### 파일: `src/core/llm/openai_client.py`

**학습 목표**:
- OpenAI API 호출 방법
- 임베딩과 채팅 API 차이 이해

**핵심 개념**:

### 5.1 임베딩 생성

```python
# src/core/llm/openai_client.py

from openai import OpenAI

client = OpenAI(api_key=settings.openai_api_key)

def get_embedding(text: str) -> List[float]:
    """
    텍스트를 벡터로 변환

    Args:
        text: 변환할 텍스트 (예: "OpenSearch는 검색 엔진입니다")

    Returns:
        3072차원 벡터 (예: [0.12, -0.34, 0.56, ...])
    """
    response = client.embeddings.create(
        model="text-embedding-3-large",  # 3072차원
        input=text
    )
    return response.data[0].embedding
```

**임베딩이란?**
- 텍스트를 숫자 배열(벡터)로 변환
- 의미가 비슷한 텍스트 → 비슷한 벡터
- 예:
  ```
  "강아지" → [0.8, 0.2, -0.1, ...]
  "개" → [0.79, 0.21, -0.09, ...]  # 비슷함!
  "자동차" → [-0.3, 0.9, 0.5, ...]  # 다름
  ```

### 5.2 채팅 완성

```python
def chat_completion(messages: List[dict]) -> str:
    """
    대화 메시지로 AI 답변 생성

    Args:
        messages: [
            {"role": "system", "content": "너는 친절한 AI야"},
            {"role": "user", "content": "안녕?"}
        ]

    Returns:
        AI 답변 (예: "안녕하세요! 무엇을 도와드릴까요?")
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=2000
    )
    return response.choices[0].message.content
```

**실습**:
```python
# 임베딩 테스트
from src.core.llm.openai_client import OpenAIClient
from src.config.settings import settings

llm = OpenAIClient(api_key=settings.openai_api_key)

# 1. 임베딩 생성
vec1 = llm.get_embedding("강아지")
vec2 = llm.get_embedding("개")
vec3 = llm.get_embedding("자동차")

print(f"벡터 차원: {len(vec1)}")  # 3072
print(f"vec1 처음 5개: {vec1[:5]}")

# 2. 코사인 유사도 계산 (간단 버전)
import numpy as np
similarity_12 = np.dot(vec1, vec2)  # 강아지 vs 개 → 높음
similarity_13 = np.dot(vec1, vec3)  # 강아지 vs 자동차 → 낮음

print(f"강아지-개 유사도: {similarity_12}")
print(f"강아지-자동차 유사도: {similarity_13}")
```

---

## 🧠 Step 6: OpenSearch 벡터 스토어 (4-5시간) ⭐ 가장 중요!

### 파일: `src/core/rag/opensearch_store.py`

**학습 목표**:
- OpenSearch k-NN 벡터 검색 원리
- 하이브리드 검색 구현 방법
- HNSW 알고리즘 이해

### 6.1 인덱스 매핑 (스키마)

```python
# src/core/rag/opensearch_store.py

def _create_index_mapping(self) -> dict:
    """
    OpenSearch 인덱스 구조 정의

    인덱스 = 관계형 DB의 테이블과 유사
    매핑 = 테이블의 스키마 정의
    """
    return {
        "settings": {
            "index": {
                "knn": True,  # k-NN 검색 활성화
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        },
        "mappings": {
            "properties": {
                # 텍스트 필드 (키워드 검색용)
                "text": {
                    "type": "text",
                    "analyzer": "standard"  # 형태소 분석
                },

                # 벡터 필드 (의미 검색용) ⭐ 핵심!
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 3072,  # OpenAI embedding 크기
                    "method": {
                        "name": "hnsw",  # 알고리즘
                        "space_type": "cosinesimil",  # 코사인 유사도
                        "engine": "lucene",  # OpenSearch 3.0+
                        "parameters": {
                            "ef_construction": 128,  # 인덱싱 품질
                            "m": 24  # 그래프 연결 수
                        }
                    }
                },

                # 필터링 필드들
                "organization_id": {"type": "keyword"},
                "user_id": {"type": "keyword"},
                "tags": {"type": "keyword"},  # 배열 저장 가능
                "created_at": {"type": "date"}
            }
        }
    }
```

**핵심 개념**:

1. **knn_vector 타입**: 벡터 저장 + 검색용 특수 타입
2. **HNSW 알고리즘**: Hierarchical Navigable Small World
   - 그래프 기반 근사 최근접 이웃 검색
   - 정확도 vs 속도 트레이드오프
   - `ef_construction` 높을수록 → 정확하지만 느림
   - `m` 높을수록 → 메모리 많이 사용, 정확도 향상

3. **space_type="cosinesimil"**: 코사인 유사도 사용
   ```
   similarity = cos(θ) = (A·B) / (|A||B|)

   범위: -1 ~ 1
   - 1: 완전 같은 방향 (유사함)
   - 0: 직각 (무관)
   - -1: 반대 방향 (반대 의미)
   ```

### 6.2 문서 추가

```python
def add_document(
    self,
    text: str,
    metadata: dict,
    organization_id: str,
    user_id: str,
    tags: Optional[List[str]] = None
) -> str:
    """
    문서를 OpenSearch에 저장

    Process:
    1. 텍스트 → 임베딩 벡터 변환 (OpenAI API 호출)
    2. 메타데이터와 함께 문서 생성
    3. OpenSearch에 인덱싱

    Args:
        text: "OpenSearch는 검색 엔진입니다"
        metadata: {"source": "문서A", "page": 10}
        organization_id: "org-123"
        user_id: "user-456"
        tags: ["기술문서", "검색"]

    Returns:
        문서 ID (UUID)
    """
    # 1. 임베딩 생성
    embedding = self.llm_client.get_embedding(text)

    # 2. 문서 구성
    doc = {
        "text": text,
        "embedding": embedding,  # 3072차원 벡터
        "metadata": metadata,
        "organization_id": organization_id,
        "user_id": user_id,
        "tags": tags or [],
        "created_at": datetime.now().isoformat()
    }

    # 3. OpenSearch에 저장
    doc_id = str(uuid.uuid4())
    self.client.index(
        index=self.index_name,
        id=doc_id,
        body=doc,
        refresh=True  # 즉시 검색 가능하도록
    )

    return doc_id
```

### 6.3 하이브리드 검색 ⭐ 핵심!

```python
def search(
    self,
    query: str,
    organization_id: str,
    user_id: str,
    tags: Optional[List[str]] = None,
    limit: int = 5,
    use_hybrid: bool = True
) -> List[dict]:
    """
    하이브리드 검색: 키워드 + 벡터 검색 결합

    Example:
        query: "벡터 검색이란?"

        → 두 가지 검색 동시 실행:
        1. 키워드 검색: "벡터", "검색" 단어가 포함된 문서
        2. 벡터 검색: 의미적으로 유사한 문서

        → 결과를 점수로 합산하여 정렬
    """
    # 1. 쿼리 임베딩 생성
    query_embedding = self.llm_client.get_embedding(query)

    # 2. 필터 조건 (조직/사용자/태그)
    filter_conditions = [
        {"term": {"organization_id": organization_id}},
        {"term": {"user_id": user_id}}
    ]
    if tags:
        filter_conditions.append({"terms": {"tags": tags}})

    # 3. 하이브리드 검색 쿼리
    search_body = {
        "size": limit,
        "query": {
            "bool": {
                "must": filter_conditions,  # 필수 조건
                "should": [  # 점수 합산 (OR 연산)
                    # (1) 키워드 매칭
                    {
                        "match": {
                            "text": {
                                "query": query,
                                "boost": 1.0  # 가중치
                            }
                        }
                    },
                    # (2) 벡터 유사도
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_embedding,
                                "k": limit * 2  # 후보 개수
                            }
                        }
                    }
                ],
                "minimum_should_match": 1  # 둘 중 하나라도 매칭
            }
        }
    }

    # 4. 검색 실행
    response = self.client.search(
        index=self.index_name,
        body=search_body
    )

    # 5. 결과 변환
    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "text": hit["_source"]["text"],
            "score": hit["_score"],  # 관련도 점수
            "metadata": hit["_source"].get("metadata", {})
        })

    return results
```

**하이브리드 검색의 장점**:

| 검색 방식 | 장점 | 단점 | 예시 |
|----------|------|------|------|
| **키워드** | 정확한 단어 매칭 | 동의어 못찾음 | "OpenSearch" → "OpenSearch" ✅ |
| **벡터** | 의미 이해 | 정확한 단어 못찾음 | "검색 엔진" → "OpenSearch" ✅ |
| **하이브리드** | 양쪽 장점 결합 | 약간 느림 | 최상의 결과! |

**실습**:
```python
# OpenSearch 연결 테스트
from src.core.rag.opensearch_store import OpenSearchStore
from src.config.settings import settings

# 1. 클라이언트 생성
store = OpenSearchStore(
    index_name="test_index",
    hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
    http_auth=(settings.opensearch_user, settings.opensearch_password),
    use_ssl=settings.opensearch_use_ssl
)

# 2. 문서 추가
doc_id = store.add_document(
    text="FastAPI는 Python 웹 프레임워크입니다. 빠르고 현대적입니다.",
    metadata={"source": "공식 문서"},
    organization_id="org-test",
    user_id="user-test",
    tags=["Python", "FastAPI"]
)
print(f"추가된 문서 ID: {doc_id}")

# 3. 검색
results = store.search(
    query="Python 프레임워크",
    organization_id="org-test",
    user_id="user-test",
    limit=3
)

for i, result in enumerate(results, 1):
    print(f"\n{i}. 점수: {result['score']:.2f}")
    print(f"   내용: {result['text'][:50]}...")
```

---

## 🧠 Step 7: RAG 엔진 (3-4시간) ⭐ 가장 중요!

### 파일: `src/core/rag/rag_engine.py`

**학습 목표**:
- RAG(Retrieval-Augmented Generation) 전체 플로우 이해
- 검색 → 프롬프트 구성 → LLM 호출 과정 파악

### 7.1 RAG란?

```
일반 LLM:
User: "우리 회사 프로젝트 A 일정은?"
GPT: "죄송합니다. 그 정보는 모릅니다." ❌

RAG:
User: "우리 회사 프로젝트 A 일정은?"
  ↓
[1단계] 벡터 DB 검색
  → "프로젝트 A의 마감일은 12월 31일..."
  ↓
[2단계] 검색 결과 + 질문 → GPT
  → GPT: "프로젝트 A의 마감일은 12월 31일입니다." ✅
```

### 7.2 핵심 메서드

```python
# src/core/rag/rag_engine.py

class RAGEngine:
    """
    RAG(Retrieval-Augmented Generation) 엔진

    검색 → 컨텍스트 구성 → LLM 답변 생성
    """

    def __init__(
        self,
        vector_store: OpenSearchStore,
        llm_model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        self.vector_store = vector_store
        self.llm_client = OpenAIClient(...)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def query(
        self,
        question: str,
        organization_id: str,
        user_id: str,
        tags: Optional[List[str]] = None,
        top_k: int = 3
    ) -> dict:
        """
        RAG 기반 질의응답

        Process:
        ┌──────────────────────────────────────┐
        │ 1. 질문으로 관련 문서 검색           │
        │    "프로젝트 A 일정은?" → 검색      │
        └──────────────┬───────────────────────┘
                       ▼
        ┌──────────────────────────────────────┐
        │ 2. 검색 결과를 컨텍스트로 구성       │
        │    [문서 1] 프로젝트 A는...         │
        │    [문서 2] 마감일은 12/31...       │
        └──────────────┬───────────────────────┘
                       ▼
        ┌──────────────────────────────────────┐
        │ 3. 시스템 프롬프트 + 컨텍스트 + 질문│
        │    → GPT-4o에게 전달                │
        └──────────────┬───────────────────────┘
                       ▼
        ┌──────────────────────────────────────┐
        │ 4. GPT-4o 답변 생성                 │
        │    "프로젝트 A의 마감일은..."       │
        └──────────────────────────────────────┘

        Args:
            question: 사용자 질문
            organization_id: 조직 ID (데이터 격리)
            user_id: 사용자 ID (데이터 격리)
            tags: 태그 필터
            top_k: 검색할 문서 개수

        Returns:
            {
                "answer": "GPT 답변",
                "sources": [검색된 문서들],
                "query": "원본 질문"
            }
        """

        # ===== 1단계: 관련 문서 검색 =====
        logger.info(
            "RAG 검색 시작",
            question=question,
            organization_id=organization_id,
            user_id=user_id,
            top_k=top_k
        )

        search_results = self.vector_store.search(
            query=question,
            organization_id=organization_id,
            user_id=user_id,
            tags=tags,
            limit=top_k,
            use_hybrid=True  # 하이브리드 검색
        )

        # ===== 2단계: 컨텍스트 구성 =====
        context = self._build_context(search_results)

        # ===== 3단계: 프롬프트 생성 =====
        messages = [
            {
                "role": "system",
                "content": """당신은 회사의 AI 어시스턴트입니다.

주어진 문서를 기반으로 정확하게 답변하세요.
문서에 없는 내용은 "해당 정보를 찾을 수 없습니다"라고 답변하세요.
답변 시 출처 번호 [문서 N]를 반드시 표시하세요."""
            },
            {
                "role": "user",
                "content": f"""다음 문서들을 참고하여 질문에 답변해주세요:

{context}

질문: {question}"""
            }
        ]

        # ===== 4단계: LLM 답변 생성 =====
        answer = self.llm_client.chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        logger.info(
            "RAG 답변 생성 완료",
            question=question,
            sources_count=len(search_results)
        )

        return {
            "answer": answer,
            "sources": search_results,
            "query": question
        }

    def _build_context(self, search_results: List[dict]) -> str:
        """
        검색 결과를 LLM 컨텍스트로 변환

        Args:
            search_results: [
                {"text": "프로젝트 A는...", "score": 0.95},
                {"text": "마감일은...", "score": 0.87}
            ]

        Returns:
            "[문서 1] 프로젝트 A는...\n[문서 2] 마감일은..."
        """
        if not search_results:
            return "관련 문서를 찾을 수 없습니다."

        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(
                f"[문서 {i}] {result['text']}"
            )

        return "\n\n".join(context_parts)
```

**RAG의 핵심 장점**:

1. **최신 정보**: 학습 데이터 외의 정보 활용
2. **환각(Hallucination) 감소**: 문서 기반 답변으로 신뢰성 향상
3. **출처 추적**: 어떤 문서를 참고했는지 명확히 표시
4. **도메인 특화**: 회사 내부 문서로 맞춤형 AI

**실습**:
```python
# RAG 엔진 테스트
from src.core.rag.rag_engine import RAGEngine
from src.core.rag.opensearch_store import OpenSearchStore
from src.config.settings import settings

# 1. Vector Store 초기화
store = OpenSearchStore(
    index_name="ai_documents",
    hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
    http_auth=(settings.opensearch_user, settings.opensearch_password),
    use_ssl=settings.opensearch_use_ssl
)

# 2. RAG 엔진 초기화
rag = RAGEngine(
    vector_store=store,
    llm_model="gpt-4o",
    temperature=0.7
)

# 3. 질의응답
result = rag.query(
    question="OpenSearch의 벡터 검색 기능에 대해 설명해주세요",
    organization_id="org-test-001",
    user_id="user-test-001",
    top_k=3
)

print("=" * 70)
print("질문:", result["query"])
print("=" * 70)
print("\n답변:")
print(result["answer"])
print("\n" + "=" * 70)
print(f"참고 문서 ({len(result['sources'])}개):")
for i, source in enumerate(result["sources"], 1):
    print(f"\n{i}. 점수: {source['score']:.2f}")
    print(f"   내용: {source['text'][:100]}...")
```

---

## 🌐 Step 8: REST API 레이어 (2-3시간)

### 파일: `src/api/rest/chat.py`

**학습 목표**:
- FastAPI 라우터 구조 이해
- 의존성 주입 패턴 학습
- 에러 핸들링 방법

### 8.1 채팅 API 엔드포인트

```python
# src/api/rest/chat.py

from fastapi import APIRouter, HTTPException, Depends
from src.models.chat import ChatRequest, ChatResponse
from src.core.rag.rag_engine import RAGEngine

router = APIRouter(prefix="/api/v1", tags=["Chat"])

# 의존성: RAG 엔진 인스턴스 가져오기
def get_rag_engine() -> RAGEngine:
    """
    RAG 엔진 싱글톤 인스턴스 반환
    FastAPI 의존성 주입 패턴
    """
    # 앱 시작 시 초기화된 전역 인스턴스
    return rag_engine

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag: RAGEngine = Depends(get_rag_engine)  # 의존성 주입
):
    """
    RAG 기반 채팅 API

    Request:
        POST /api/v1/chat
        {
            "message": "프로젝트 A 일정은?",
            "organization_id": "org-123",
            "user_id": "user-456",
            "session_id": "sess_abc"  # 선택
        }

    Response:
        {
            "session_id": "sess_abc",
            "message": "프로젝트 A의 마감일은...",
            "sources": [
                {"text": "...", "score": 0.95}
            ],
            "suggestions": ["관련 질문 1", "관련 질문 2"],
            "timestamp": "2025-12-01T15:00:00"
        }
    """
    try:
        # 1. 세션 ID 생성 또는 재사용
        session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"

        # 2. RAG 쿼리 실행
        result = rag.query(
            question=request.message,
            organization_id=request.organization_id,
            user_id=request.user_id,
            top_k=3
        )

        # 3. 응답 구성
        response = ChatResponse(
            session_id=session_id,
            message=result["answer"],
            sources=result["sources"],
            suggestions=_generate_suggestions(result),
            timestamp=datetime.now().isoformat()
        )

        # 4. 대화 이력 저장 (Redis)
        await _save_chat_history(session_id, request.message, result["answer"])

        logger.info(
            "채팅 응답 완료",
            session_id=session_id,
            user_id=request.user_id,
            sources_count=len(result["sources"])
        )

        return response

    except Exception as e:
        logger.error(
            "채팅 처리 실패",
            error=str(e),
            user_id=request.user_id
        )
        raise HTTPException(
            status_code=500,
            detail=f"채팅 처리 중 오류가 발생했습니다: {str(e)}"
        )

def _generate_suggestions(result: dict) -> List[str]:
    """
    검색 결과 기반 추천 질문 생성
    """
    # 간단한 템플릿 기반 생성
    # 실제로는 LLM으로 생성 가능
    return [
        "관련 문서 더 찾기",
        "다른 프로젝트 정보 검색",
        "일정 확인하기"
    ]

async def _save_chat_history(session_id: str, question: str, answer: str):
    """
    Redis에 대화 이력 저장

    Key: chat:history:{session_id}
    Value: [
        {"role": "user", "content": "질문", "timestamp": "..."},
        {"role": "assistant", "content": "답변", "timestamp": "..."}
    ]
    """
    # Redis 클라이언트 사용
    # 구현 생략 (src/utils/redis_client.py 참조)
    pass
```

**FastAPI 핵심 개념**:

1. **의존성 주입 (Dependency Injection)**:
   ```python
   def get_rag_engine() -> RAGEngine:
       return rag_engine

   async def chat(rag: RAGEngine = Depends(get_rag_engine)):
       # rag를 직접 생성하지 않고 주입받음
       # 테스트 시 Mock으로 교체 가능
   ```

2. **자동 검증**:
   ```python
   @router.post("/chat", response_model=ChatResponse)
   async def chat(request: ChatRequest):
       # Pydantic이 자동으로:
       # - 필수 필드 체크
       # - 타입 변환
       # - 최소/최대 길이 검증
   ```

3. **자동 문서화**:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

### 8.2 문서 관리 API

**파일: `src/api/rest/documents.py`**

```python
# src/api/rest/documents.py

@router.post("/documents", response_model=DocumentAddResponse)
async def add_document(
    request: DocumentAddRequest,
    rag: RAGEngine = Depends(get_rag_engine)
):
    """
    문서 추가 API

    Request:
        POST /api/v1/documents
        {
            "text": "OpenSearch는...",
            "metadata": {"source": "공식 문서"},
            "organization_id": "org-123",
            "user_id": "user-456",
            "tags": ["OpenSearch", "검색"]
        }
    """
    doc_id = rag.add_document(
        text=request.text,
        metadata=request.metadata,
        organization_id=request.organization_id,
        user_id=request.user_id,
        tags=request.tags
    )

    return DocumentAddResponse(
        doc_id=doc_id,
        message="문서가 성공적으로 추가되었습니다."
    )

@router.post("/documents/search", response_model=DocumentSearchResponse)
async def search_documents(
    request: DocumentSearchRequest,
    rag: RAGEngine = Depends(get_rag_engine)
):
    """
    문서 검색 API (RAG 없이 순수 검색)
    """
    results = rag.search_documents(
        query=request.query,
        organization_id=request.organization_id,
        user_id=request.user_id,
        tags=request.tags,
        limit=request.limit
    )

    return DocumentSearchResponse(
        results=results,
        count=len(results)
    )

@router.get("/documents/stats")
async def get_stats(
    organization_id: str,
    user_id: str,
    rag: RAGEngine = Depends(get_rag_engine)
):
    """
    문서 통계 API

    Returns:
        {
            "total_documents": 150,
            "by_tags": {"Python": 50, "FastAPI": 30},
            "recent_uploads": 10
        }
    """
    # OpenSearch aggregation 사용
    # 구현 생략
    pass
```

**실습**:
```bash
# 1. 서버 실행
.venv/bin/python -m src.main

# 2. Swagger UI 접속
# 브라우저: http://localhost:8000/docs

# 3. API 테스트 (curl)
# 문서 추가
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "text": "FastAPI는 빠른 Python 웹 프레임워크입니다",
    "metadata": {"source": "학습자료"},
    "organization_id": "org-test",
    "user_id": "user-test",
    "tags": ["Python", "FastAPI"]
  }'

# 채팅
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "FastAPI에 대해 알려줘",
    "organization_id": "org-test",
    "user_id": "user-test"
  }'
```

---

## 🚀 Step 9: 전체 플로우 이해 (2시간)

### 파일: `src/main.py`

**학습 목표**:
- FastAPI 애플리케이션 초기화 과정
- 라우터 등록 및 CORS 설정
- 시작/종료 이벤트 핸들러

```python
# src/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.config.settings import settings
from src.api.rest import chat, documents, health
from src.core.rag.rag_engine import RAGEngine
from src.utils.logger import logger

# 전역 변수 (싱글톤)
rag_engine: RAGEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 생명주기 관리

    시작 시:
    - OpenSearch 연결
    - RAG 엔진 초기화
    - Redis 연결

    종료 시:
    - 연결 정리
    """
    # ===== 시작 이벤트 =====
    logger.info("애플리케이션 시작", environment=settings.environment)

    # 1. OpenSearch 연결 확인
    logger.info("OpenSearch 연결 중", host=settings.opensearch_host)

    # 2. RAG 엔진 초기화
    global rag_engine
    rag_engine = RAGEngine(
        vector_store=None,  # 자동 생성
        llm_model=settings.openai_model,
        temperature=settings.openai_temperature,
        max_tokens=settings.openai_max_tokens,
        use_opensearch=True  # OpenSearch 사용
    )
    logger.info("RAG 엔진 초기화 완료")

    # 3. Redis 연결 (대화 이력용)
    logger.info("Redis 연결 중", url=settings.redis_url)

    yield  # 여기서 애플리케이션 실행

    # ===== 종료 이벤트 =====
    logger.info("애플리케이션 종료")
    # 연결 정리 로직

# FastAPI 앱 생성
app = FastAPI(
    title="AI Assistant API",
    description="RAG 기반 AI 어시스턴트",
    version="0.1.0",
    lifespan=lifespan  # 생명주기 핸들러 등록
)

# CORS 설정 (프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(health.router)  # /health
app.include_router(chat.router)    # /api/v1/chat
app.include_router(documents.router)  # /api/v1/documents

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "AI Assistant",
        "version": "0.1.0",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug  # 개발 모드에서 자동 리로드
    )
```

**애플리케이션 시작 플로우**:

```
1. main.py 실행
   ↓
2. FastAPI 앱 생성
   ↓
3. lifespan 시작 이벤트
   - OpenSearch 연결
   - RAG 엔진 초기화
   - Redis 연결
   ↓
4. 라우터 등록
   - /health
   - /api/v1/chat
   - /api/v1/documents
   ↓
5. Uvicorn 서버 시작
   - http://localhost:8000
   ↓
6. 요청 대기...
   ↓
7. 종료 신호 (Ctrl+C)
   ↓
8. lifespan 종료 이벤트
   - 연결 정리
   ↓
9. 애플리케이션 종료
```

---

## 5. 실습 예제

### 실습 1: 간단한 RAG 파이프라인 구축 (1시간)

**목표**: 처음부터 끝까지 RAG 시스템 실행해보기

```python
# practice_1_simple_rag.py

"""
실습 1: 간단한 RAG 파이프라인

목표:
1. 문서 3개 추가
2. 검색 테스트
3. RAG 질의응답
"""

from src.core.rag.opensearch_store import OpenSearchStore
from src.core.rag.rag_engine import RAGEngine
from src.config.settings import settings

# ===== Step 1: 데이터 준비 =====
documents = [
    {
        "text": "FastAPI는 Python으로 작성된 현대적인 웹 프레임워크입니다. 타입 힌팅을 기반으로 자동 검증과 문서화를 제공합니다.",
        "tags": ["Python", "FastAPI", "웹개발"]
    },
    {
        "text": "OpenSearch는 Elasticsearch의 오픈소스 포크입니다. 검색 엔진과 분석 도구로 사용됩니다.",
        "tags": ["OpenSearch", "검색", "데이터베이스"]
    },
    {
        "text": "RAG는 검색과 생성을 결합한 AI 기술입니다. 외부 지식을 활용하여 더 정확한 답변을 생성합니다.",
        "tags": ["AI", "RAG", "LLM"]
    }
]

# ===== Step 2: OpenSearch 연결 =====
print("=" * 70)
print("OpenSearch 연결 중...")
print("=" * 70)

store = OpenSearchStore(
    index_name="practice_index",
    hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
    http_auth=(settings.opensearch_user, settings.opensearch_password),
    use_ssl=settings.opensearch_use_ssl
)

# ===== Step 3: 문서 추가 =====
print("\n문서 추가 중...")
doc_ids = []
for i, doc in enumerate(documents, 1):
    doc_id = store.add_document(
        text=doc["text"],
        metadata={"source": f"실습 문서 {i}"},
        organization_id="practice-org",
        user_id="practice-user",
        tags=doc["tags"]
    )
    doc_ids.append(doc_id)
    print(f"  {i}. 문서 추가 완료: {doc_id[:8]}...")

# ===== Step 4: 검색 테스트 =====
print("\n" + "=" * 70)
print("검색 테스트")
print("=" * 70)

queries = [
    "Python 웹 프레임워크",
    "검색 엔진",
    "AI 기술"
]

for query in queries:
    print(f"\n질문: {query}")
    results = store.search(
        query=query,
        organization_id="practice-org",
        user_id="practice-user",
        limit=2
    )

    for i, result in enumerate(results, 1):
        print(f"  {i}. 점수 {result['score']:.2f}: {result['text'][:50]}...")

# ===== Step 5: RAG 질의응답 =====
print("\n" + "=" * 70)
print("RAG 질의응답")
print("=" * 70)

rag = RAGEngine(
    vector_store=store,
    llm_model="gpt-4o",
    temperature=0.7
)

question = "FastAPI와 OpenSearch의 차이점을 설명해주세요"
print(f"\n질문: {question}")

result = rag.query(
    question=question,
    organization_id="practice-org",
    user_id="practice-user",
    top_k=3
)

print("\n답변:")
print(result["answer"])

print("\n참고 문서:")
for i, source in enumerate(result["sources"], 1):
    print(f"  [{i}] {source['text'][:60]}...")

# ===== Step 6: 정리 =====
print("\n" + "=" * 70)
print("인덱스 삭제 (정리)")
print("=" * 70)

# 실습 후 인덱스 삭제 (선택)
# store.client.indices.delete(index="practice_index")
# print("인덱스 'practice_index' 삭제 완료")
```

**실행**:
```bash
.venv/bin/python practice_1_simple_rag.py
```

### 실습 2: 하이브리드 검색 vs 벡터 검색 비교 (1시간)

**목표**: 검색 방식에 따른 결과 차이 이해

```python
# practice_2_search_comparison.py

"""
실습 2: 검색 방식 비교

하이브리드 검색 vs 순수 벡터 검색
"""

from src.core.rag.opensearch_store import OpenSearchStore
from src.config.settings import settings

# OpenSearch 연결
store = OpenSearchStore(
    index_name="ai_documents",  # 기존 인덱스 사용
    hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
    http_auth=(settings.opensearch_user, settings.opensearch_password),
    use_ssl=settings.opensearch_use_ssl
)

# 테스트 쿼리
test_queries = [
    "벡터 검색",  # 정확한 키워드
    "의미 기반 탐색",  # 유사한 의미
    "k-NN algorithm",  # 영문 키워드
]

for query in test_queries:
    print("=" * 70)
    print(f"질문: {query}")
    print("=" * 70)

    # 1. 하이브리드 검색
    print("\n[하이브리드 검색]")
    hybrid_results = store.search(
        query=query,
        organization_id="org-test-001",
        user_id="user-test-001",
        limit=3,
        use_hybrid=True
    )

    for i, result in enumerate(hybrid_results, 1):
        print(f"{i}. 점수 {result['score']:.2f}")
        print(f"   {result['text'][:80]}...")

    print("\n")
```

### 실습 3: 태그 필터링 (30분)

```python
# practice_3_tag_filtering.py

"""
실습 3: 태그 기반 필터링

조직/프로젝트별 문서 격리
"""

from src.core.rag.opensearch_store import OpenSearchStore
from src.config.settings import settings

store = OpenSearchStore(
    index_name="ai_documents",
    hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
    http_auth=(settings.opensearch_user, settings.opensearch_password),
    use_ssl=settings.opensearch_use_ssl
)

# 다양한 태그로 문서 추가
documents = [
    ("프로젝트 A는 웹 개발 프로젝트입니다.", ["프로젝트A", "웹개발"]),
    ("프로젝트 B는 데이터 분석 프로젝트입니다.", ["프로젝트B", "데이터"]),
    ("프로젝트 A의 마감일은 12월입니다.", ["프로젝트A", "일정"]),
]

print("문서 추가 중...")
for text, tags in documents:
    store.add_document(
        text=text,
        metadata={},
        organization_id="org-practice",
        user_id="user-practice",
        tags=tags
    )
    print(f"  추가: {tags}")

# 태그별 검색
print("\n" + "=" * 70)
print("태그 필터링 검색")
print("=" * 70)

# 1. "프로젝트A" 태그만
print("\n[프로젝트A 관련 문서만]")
results = store.search(
    query="프로젝트",
    organization_id="org-practice",
    user_id="user-practice",
    tags=["프로젝트A"],
    limit=5
)

for i, result in enumerate(results, 1):
    print(f"{i}. {result['text']}")

# 2. "일정" 태그만
print("\n[일정 관련 문서만]")
results = store.search(
    query="프로젝트",
    organization_id="org-practice",
    user_id="user-practice",
    tags=["일정"],
    limit=5
)

for i, result in enumerate(results, 1):
    print(f"{i}. {result['text']}")
```

---

## 6. 트러블슈팅

### 6.1 자주 발생하는 오류

#### 오류 1: OpenSearch 연결 실패

```
ConnectionError: Connection to OpenSearch failed
```

**해결책**:
```bash
# 1. OpenSearch 실행 확인
curl -k -u admin:admin https://3.34.20.81:30920

# 2. .env 설정 확인
cat .env | grep OPENSEARCH

# 3. 방화벽 확인
# 포트 30920이 열려있는지 확인
```

#### 오류 2: OpenAI API 키 오류

```
AuthenticationError: Invalid API key
```

**해결책**:
```bash
# .env 파일 확인
echo $OPENAI_API_KEY

# API 키 유효성 테스트
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

#### 오류 3: 인덱스 매핑 오류

```
mapper_parsing_exception: nmslib engine is deprecated
```

**해결책**:
```python
# opensearch_store.py에서 engine 변경
"engine": "lucene",  # nmslib 대신 lucene 사용
```

#### 오류 4: 임베딩 차원 불일치

```
dimension mismatch: expected 3072, got 1536
```

**해결책**:
```python
# 올바른 임베딩 모델 사용
model="text-embedding-3-large"  # 3072차원
# model="text-embedding-3-small"  # 1536차원 (사용X)
```

### 6.2 디버깅 팁

```python
# 1. 로그 레벨 변경
# .env
LOG_LEVEL=DEBUG  # INFO → DEBUG로 변경

# 2. 상세 로그 확인
logger.debug("변수 값", my_var=my_var)

# 3. OpenSearch 쿼리 확인
print(json.dumps(search_body, indent=2))

# 4. 임베딩 벡터 확인
embedding = llm.get_embedding("테스트")
print(f"차원: {len(embedding)}, 처음 5개: {embedding[:5]}")
```

---

## 7. 다음 학습 단계

### 7.1 심화 학습 주제

1. **성능 최적화**
   - OpenSearch HNSW 파라미터 튜닝
   - 임베딩 캐싱 전략
   - 배치 처리

2. **고급 RAG 기법**
   - Re-ranking
   - Query expansion
   - Hybrid fusion 알고리즘

3. **프로덕션 배포**
   - Docker 컨테이너화
   - Kubernetes 배포
   - 모니터링 및 로깅

4. **보안 강화**
   - API 인증/인가
   - Rate limiting
   - 민감 정보 마스킹

### 7.2 추천 리소스

#### 공식 문서
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [OpenSearch 공식 가이드](https://opensearch.org/docs/latest/)
- [OpenAI API 레퍼런스](https://platform.openai.com/docs/api-reference)

#### 튜토리얼
- [LangChain RAG 튜토리얼](https://python.langchain.com/docs/use_cases/question_answering/)
- [Pinecone RAG 가이드](https://www.pinecone.io/learn/retrieval-augmented-generation/)

#### 논문
- [RAG 원논문](https://arxiv.org/abs/2005.11401)
- [HNSW 알고리즘](https://arxiv.org/abs/1603.09320)

---

## 8. 학습 체크리스트

### 기초 (필수)
- [ ] FastAPI 라우터와 엔드포인트 이해
- [ ] Pydantic 모델 사용법
- [ ] 환경 변수 관리 (settings.py)
- [ ] 구조화된 로깅

### 핵심 (중요)
- [ ] OpenAI 임베딩 API 사용법
- [ ] OpenSearch k-NN 벡터 검색 원리
- [ ] HNSW 알고리즘 개념
- [ ] 하이브리드 검색 구현
- [ ] RAG 파이프라인 전체 플로우

### 심화 (선택)
- [ ] Redis 대화 이력 관리
- [ ] 에러 핸들링 및 재시도 로직
- [ ] 성능 모니터링
- [ ] 프로덕션 배포

---

## 9. 마무리

### 학습 목표 달성도 자가 평가

| 항목 | 달성도 | 비고 |
|------|--------|------|
| 프로젝트 구조 이해 | ☐ ☐ ☐ ☐ ☐ | |
| OpenSearch 벡터 검색 | ☐ ☐ ☐ ☐ ☐ | |
| RAG 엔진 구현 | ☐ ☐ ☐ ☐ ☐ | |
| FastAPI 활용 | ☐ ☐ ☐ ☐ ☐ | |
| 전체 플로우 이해 | ☐ ☐ ☐ ☐ ☐ | |

### 다음 액션 아이템

1. [ ] 실습 예제 3개 모두 실행해보기
2. [ ] 자신만의 문서로 RAG 시스템 구축
3. [ ] 프론트엔드 연동 시도
4. [ ] 성능 테스트 및 최적화
5. [ ] 프로덕션 배포 준비

---

**학습에 도움이 되셨기를 바랍니다! 궁금한 점은 언제든 문의해주세요.**

---

## 부록: 용어 사전

| 용어 | 설명 |
|------|------|
| **RAG** | Retrieval-Augmented Generation, 검색 증강 생성 |
| **임베딩 (Embedding)** | 텍스트를 고차원 벡터로 변환한 것 |
| **k-NN** | k-Nearest Neighbors, k-최근접 이웃 알고리즘 |
| **HNSW** | Hierarchical Navigable Small World, 계층적 그래프 기반 검색 |
| **코사인 유사도** | 두 벡터 간 각도로 유사도 측정 (-1 ~ 1) |
| **하이브리드 검색** | 키워드 + 벡터 검색 결합 |
| **Pydantic** | Python 데이터 검증 라이브러리 |
| **의존성 주입** | 객체를 외부에서 주입받는 디자인 패턴 |
| **Uvicorn** | FastAPI 실행용 ASGI 서버 |
