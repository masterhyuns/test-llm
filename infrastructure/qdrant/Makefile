# Qdrant 관리 명령어

.PHONY: help start stop restart logs status test clean

help:  ## 도움말 표시
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

start:  ## Qdrant 시작
	@echo "🚀 Qdrant 시작 중..."
	docker-compose up -d
	@echo "✅ Qdrant 시작 완료"
	@echo "📊 웹 UI: http://localhost:6333/dashboard"

stop:  ## Qdrant 중지
	@echo "⏹️  Qdrant 중지 중..."
	docker-compose down
	@echo "✅ Qdrant 중지 완료"

restart:  ## Qdrant 재시작
	@echo "🔄 Qdrant 재시작 중..."
	docker-compose restart
	@echo "✅ Qdrant 재시작 완료"

logs:  ## 로그 확인
	docker-compose logs -f qdrant

status:  ## 상태 확인
	@echo "📊 Qdrant 상태:"
	@docker-compose ps
	@echo ""
	@echo "헬스체크:"
	@curl -s http://localhost:6333/healthz || echo "❌ Qdrant가 실행 중이지 않습니다"

test:  ## 연결 테스트
	@echo "🧪 Qdrant 연결 테스트..."
	python3 test-connection.py

clean:  ## 데이터 삭제 및 재시작
	@echo "⚠️  모든 데이터가 삭제됩니다. 계속하시겠습니까? [y/N] " && read ans && [ $${ans:-N} = y ]
	@echo "🗑️  데이터 삭제 중..."
	docker-compose down -v
	rm -rf qdrant_storage qdrant_snapshots
	@echo "🚀 Qdrant 재시작 중..."
	docker-compose up -d
	@echo "✅ 완료"

backup:  ## 스냅샷 백업
	@echo "💾 스냅샷 생성 중..."
	@mkdir -p backups
	@DATE=$$(date +%Y%m%d_%H%M%S) && \
		docker-compose exec -T qdrant wget -O- http://localhost:6333/collections/documents/snapshots && \
		echo "✅ 백업 완료: backups/snapshot_$$DATE"

install-deps:  ## Python 의존성 설치
	pip install qdrant-client

ui:  ## 웹 UI 열기
	@echo "📊 웹 UI 열기..."
	@open http://localhost:6333/dashboard || xdg-open http://localhost:6333/dashboard || echo "브라우저에서 http://localhost:6333/dashboard 를 열어주세요"
