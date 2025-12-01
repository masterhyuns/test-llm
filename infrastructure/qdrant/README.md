# Qdrant Vector Database 설정

로컬 개발 환경용 Qdrant Docker 설정

## 🚀 빠른 시작

### 방법 1: Makefile 사용 (추천)

```bash
# infrastructure/qdrant 폴더로 이동
cd infrastructure/qdrant

# Qdrant 시작
make start

# 상태 확인
make status

# 연결 테스트
make test

# 웹 UI 열기
make ui
```

### 방법 2: Docker Compose 직접 사용

```bash
# infrastructure/qdrant 폴더로 이동
cd infrastructure/qdrant

# Docker Compose로 실행
docker-compose up -d
```

## 📋 Makefile 명령어 목록

| 명령어 | 설명 |
|--------|------|
| `make help` | 모든 명령어 도움말 보기 |
| `make start` | Qdrant 시작 |
| `make stop` | Qdrant 중지 |
| `make restart` | Qdrant 재시작 |
| `make logs` | 실시간 로그 보기 |
| `make status` | 현재 상태 및 헬스체크 |
| `make test` | Python 연결 테스트 실행 |
| `make clean` | 모든 데이터 삭제 후 재시작 |
| `make backup` | 스냅샷 백업 생성 |
| `make ui` | 웹 UI 브라우저로 열기 |
| `make install-deps` | Python 의존성 설치 |

### Makefile 사용 예시

```bash
# 1. Qdrant 시작
make start

# 출력:
# 🚀 Qdrant 시작 중...
# ✅ Qdrant 시작 완료
# 📊 웹 UI: http://localhost:6333/dashboard

# 2. 상태 확인
make status

# 출력:
# 📊 Qdrant 상태:
# NAME     COMMAND                  SERVICE   STATUS    PORTS
# qdrant   "./qdrant"               qdrant    running   0.0.0.0:6333->6333/tcp

# 3. 연결 테스트
make test

# 출력:
# 🧪 Qdrant 연결 테스트...
# ✅ 모든 테스트 통과!

# 4. 로그 보기 (Ctrl+C로 종료)
make logs

# 5. 중지
make stop
```

### 2. 상태 확인

```bash
# 컨테이너 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs -f qdrant

# 헬스체크
curl http://localhost:6333/healthz
```

### 3. 웹 UI 접속

브라우저에서 접속:
```
http://localhost:6333/dashboard
```

### 4. 중지 및 재시작

#### Makefile 사용
```bash
# 중지
make stop

# 재시작
make restart

# 데이터 삭제 후 재시작 (주의!)
make clean
```

#### Docker Compose 직접 사용
```bash
# 중지
docker-compose down

# 재시작
docker-compose restart

# 데이터 삭제 후 재시작
docker-compose down -v
rm -rf qdrant_storage qdrant_snapshots
docker-compose up -d
```

## 📊 포트 정보

- **6333**: REST API (HTTP)
- **6334**: gRPC API

## 💾 데이터 저장

데이터는 다음 폴더에 영구 저장됩니다:
```
infrastructure/qdrant/
├── qdrant_storage/     # 벡터 데이터
└── qdrant_snapshots/   # 백업 스냅샷
```

## 🔧 설정 커스터마이징

`docker-compose.yml` 파일에서 환경변수를 수정하여 설정 변경 가능:

```yaml
environment:
  - QDRANT__LOG_LEVEL=DEBUG  # 로그 레벨 변경
  - QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=4  # 검색 스레드 수
```

## 📝 Python 클라이언트 연결

```python
from qdrant_client import QdrantClient

# 연결
client = QdrantClient(host="localhost", port=6333)

# 또는 URL 사용
client = QdrantClient(url="http://localhost:6333")

# 헬스체크
print(client.get_collections())
```

## 🔄 일상적인 사용 흐름

### 개발 시작할 때
```bash
cd infrastructure/qdrant
make start      # Qdrant 시작
make status     # 정상 동작 확인
```

### 개발 중
```bash
make logs       # 문제 발생 시 로그 확인
make test       # 연결 테스트
make ui         # 웹 UI에서 데이터 확인
```

### 개발 종료할 때
```bash
make stop       # Qdrant 중지 (데이터는 유지됨)
```

### 데이터 초기화가 필요할 때
```bash
make clean      # 모든 데이터 삭제 후 재시작
# ⚠️ 경고: 모든 벡터 데이터가 삭제됩니다!
```

## 🐛 트러블슈팅

### 포트가 이미 사용 중
```bash
# 포트 사용 확인
lsof -i :6333

# docker-compose.yml에서 포트 변경
ports:
  - "6335:6333"  # 6335로 변경
```

### 권한 오류
```bash
# 데이터 폴더 권한 설정
sudo chown -R $(whoami) qdrant_storage qdrant_snapshots
```

### 컨테이너 실행 안됨
```bash
# 로그 확인
docker-compose logs qdrant

# 컨테이너 재생성
docker-compose down
docker-compose up -d --force-recreate
```

## 🔒 보안 (프로덕션 배포 시)

프로덕션 환경에서는 API 키 설정 권장:

```yaml
environment:
  - QDRANT__SERVICE__API_KEY=your-secret-api-key
```

Python 클라이언트:
```python
client = QdrantClient(
    url="http://localhost:6333",
    api_key="your-secret-api-key"
)
```

## 📚 참고 자료

- [Qdrant 공식 문서](https://qdrant.tech/documentation/)
- [Docker Hub - Qdrant](https://hub.docker.com/r/qdrant/qdrant)
- [Python Client 문서](https://github.com/qdrant/qdrant-client)
