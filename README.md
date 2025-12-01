# AI Assistant Service

Cowexa 협업 플랫폼의 AI 비서 서비스입니다.

## 🎯 주요 기능

### 1. RAG (Retrieval-Augmented Generation)
- **문서 기반 답변**: 사용자가 추가한 문서를 기반으로 정확한 답변 제공
- **Semantic Search**: 의미 기반 문서 검색 (키워드 일치가 아닌 의미 유사도로 검색)
- **Multi-tenancy**: 조직별, 사용자별 데이터 격리

### 2. Chat API
- **RAG 모드**: 문서 검색 + 문서 기반 답변
- **일반 모드**: LLM의 일반 지식으로 답변
- **참고 문서 제공**: 답변에 사용된 문서 출처 표시

### 3. Document Management
- **문서 추가** (Indexing): 텍스트를 Vector DB에 저장
- **문서 검색**: 유사한 문서 찾기
- **문서 삭제**: Vector DB에서 문서 제거
- **통계 조회**: 저장된 문서 수 등 확인

## 🏗️ 기술 스택

- **Framework**: FastAPI (Python 3.11+)
- **LLM**: OpenAI GPT-4o
- **Embeddings**: OpenAI text-embedding-3-large (3072차원)
- **Vector DB**: Qdrant (Docker)
- **Cache**: Redis
- **Logging**: Structlog (JSON 형식)

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 1. Python 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 변수 설정
cp .env.example .env
# .env 파일 열어서 OPENAI_API_KEY 등 설정
```

### 2. Qdrant Vector DB 실행

```bash
# infrastructure/qdrant 폴더로 이동
cd infrastructure/qdrant

# Qdrant 시작
make start

# 상태 확인
make status

# 연결 테스트
make test
```

자세한 내용은 [infrastructure/qdrant/README.md](infrastructure/qdrant/README.md) 참고

### 3. FastAPI 서버 실행

```bash
# 개발 모드 (자동 리로드)
python -m src.main

# 또는 uvicorn 직접 실행
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

서버가 시작되면:
- API 문서: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### 4. RAG 시스템 테스트

```bash
# 전체 테스트 스크립트 실행
python test_rag_api.py
```

이 스크립트는 다음을 테스트합니다:
1. 문서 추가 (Indexing)
2. 문서 검색 (Semantic Search)
3. RAG 채팅 (문서 기반 답변)
4. 일반 채팅 (LLM 일반 지식)
5. 통계 조회

## 📖 API 사용 예시

### 1. 문서 추가 (Indexing)

```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "프로젝트 A의 마감일은 2024년 12월 31일입니다.",
    "metadata": {
      "title": "프로젝트 A 일정",
      "author": "홍길동"
    },
    "organization_id": "org_123",
    "user_id": "user_456"
  }'
```

### 2. 문서 검색

```bash
curl -X POST "http://localhost:8000/api/v1/documents/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "프로젝트 마감일",
    "organization_id": "org_123",
    "limit": 5
  }'
```

### 3. RAG 채팅

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "프로젝트 A 마감일이 언제야?",
    "organization_id": "org_123",
    "user_id": "user_456",
    "use_rag": true
  }'
```

## 🔍 RAG 동작 원리

### Indexing (문서 추가)
```
텍스트 입력
    ↓
OpenAI Embedding API 호출
(text → 3072차원 벡터 변환)
    ↓
Qdrant Vector DB에 저장
(벡터 + 메타데이터)
```

### Retrieval (문서 검색)
```
질문 입력
    ↓
OpenAI Embedding API 호출
(질문 → 3072차원 벡터 변환)
    ↓
Qdrant에서 유사한 벡터 검색
(Cosine 유사도 기반)
    ↓
유사도 높은 문서 반환
```

### Generation (답변 생성)
```
질문 + 검색된 문서
    ↓
프롬프트 생성
(시스템 메시지 + 컨텍스트 + 질문)
    ↓
OpenAI LLM API 호출
(GPT-4o)
    ↓
문서 기반 답변 생성
```

## 📊 주요 개념 설명

### Vector Embedding이란?
- 텍스트를 숫자 배열(벡터)로 변환한 것
- 비슷한 의미의 텍스트는 비슷한 벡터를 가짐
- 예: "강아지" ≈ "개" (벡터가 유사)

### Semantic Search란?
- 의미 기반 검색
- 키워드가 정확히 일치하지 않아도 의미가 비슷하면 검색됨
- 예: "강아지" 검색 → "개", "반려동물" 문서도 검색

### Multi-tenancy란?
- 여러 조직/사용자가 같은 시스템 공유
- 각자의 데이터는 완전히 격리됨
- `organization_id`, `user_id`로 구분

## 🔧 개발 도구

### Swagger UI
- http://localhost:8000/docs
- API 문서 + 테스트 가능

### Qdrant 웹 UI
- http://localhost:6333/dashboard
- Vector DB 상태 확인

### Redis 확인
```bash
redis-cli -h 3.34.20.81 -p 30379 -a redis123!
> KEYS *
```

## 📁 프로젝트 구조

```
ai-assistant/
├── src/
│   ├── api/
│   │   └── rest/
│   │       ├── chat.py          # Chat API (RAG 통합)
│   │       ├── documents.py     # Document Management API
│   │       └── health.py        # Health Check
│   ├── core/
│   │   ├── llm/
│   │   │   └── openai_client.py # OpenAI API 클라이언트
│   │   └── rag/
│   │       ├── qdrant_store.py  # Vector Store 관리
│   │       └── rag_engine.py    # RAG 엔진
│   ├── models/
│   │   └── chat.py              # Pydantic 모델
│   ├── utils/
│   │   ├── logger.py            # 구조화된 로깅
│   │   └── redis_client.py      # Redis 연결
│   ├── config/
│   │   └── settings.py          # 환경 설정
│   └── main.py                  # FastAPI 앱 진입점
├── infrastructure/
│   └── qdrant/
│       ├── docker-compose.yml   # Qdrant Docker 설정
│       ├── Makefile             # 편리한 명령어
│       └── test-connection.py   # 연결 테스트
├── test_rag_api.py              # 전체 테스트 스크립트
├── requirements.txt             # Python 의존성
├── .env.example                 # 환경 변수 예시
└── README.md                    # 이 파일
```

## 💰 비용 예상

### OpenAI API 비용
- **Embedding** (text-embedding-3-large): $0.00013 / 1K tokens
  - 1000자 문서 → 약 $0.0002
- **LLM** (gpt-4o): $2.50 / 1M input tokens, $10.00 / 1M output tokens
  - 질문 + 문서 3개 (각 500자) + 답변 200자 → 약 $0.005

### 예시 (월 1만 건 질문 기준)
- 문서 추가 (1000건): $0.2
- 질문 답변 (10000건): $50
- **총 약 $50/월**

### 비용 절감 방법
- 캐싱: 같은 질문은 캐시된 답변 사용
- 모델 선택: 간단한 질문은 gpt-4o-mini 사용
- 컨텍스트 압축: 검색 결과를 요약하여 전달

## 🐛 트러블슈팅

### Qdrant 연결 실패
```bash
# Qdrant 상태 확인
cd infrastructure/qdrant
make status

# 재시작
make restart

# 로그 확인
make logs
```

### OpenAI API 오류
```bash
# .env 파일 확인
cat .env | grep OPENAI_API_KEY

# API 키 테스트
python -c "from openai import OpenAI; print(OpenAI().models.list())"
```

### Redis 연결 실패
```bash
# Redis 연결 테스트
redis-cli -h 3.34.20.81 -p 30379 -a redis123! PING
```

## 📚 참고 자료

### 공식 문서
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenAI API](https://platform.openai.com/docs)
- [Qdrant](https://qdrant.tech/documentation/)
- [LangChain](https://python.langchain.com/)

### 가이드 문서
- [Vector DB 선택 가이드](docs/VECTOR_DB_GUIDE.md)

## 🤝 기여

버그 리포트, 기능 제안, PR 환영합니다!

## 📄 라이선스

Copyright © 2024 Cowexa. All rights reserved.


  🚀 다음 단계 (선택사항)

  필요하시면 추가로 진행 가능합니다:
  1. 기존 Qdrant 데이터를 OpenSearch로 마이그레이션
  2. 성능 튜닝 (HNSW 파라미터 조정)
  3. 모니터링 설정 (응답 시간, 검색 품질)
  4. README 업데이트 및 API 문서화

  현재 상태로도 프로덕션에서 바로 사용 가능합니다! 🎉
