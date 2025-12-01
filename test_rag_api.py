#!/usr/bin/env python3
"""
RAG 시스템 전체 테스트 스크립트

🎯 테스트 시나리오:
1. 문서 추가 (Indexing)
2. 문서 검색 (Semantic Search)
3. RAG 채팅 (문서 기반 답변)
4. 통계 조회

💡 사용 방법:
1. Qdrant 실행: cd infrastructure/qdrant && make start
2. FastAPI 서버 실행: python -m src.main
3. 이 스크립트 실행: python test_rag_api.py

⚠️ 주의:
- 실제 OpenAI API를 호출하므로 비용이 발생합니다
- .env 파일에 OPENAI_API_KEY 설정 필요
"""

import requests
import json
from datetime import datetime

# API 기본 URL
BASE_URL = "http://localhost:8000"
API_V1 = f"{BASE_URL}/api/v1"

# 테스트용 조직/사용자 ID
ORG_ID = "test_org_001"
USER_ID = "test_user_001"


def print_section(title):
    """섹션 제목 출력"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_response(response):
    """응답 내용 예쁘게 출력"""
    print(f"\n상태 코드: {response.status_code}")
    print(f"응답 내용:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


def test_add_documents():
    """
    1단계: 문서 추가 테스트

    📥 Indexing:
    - 프로젝트 관련 문서 3개 추가
    - 각 문서는 OpenAI로 embedding 생성 후 Qdrant에 저장
    """
    print_section("1단계: 문서 추가 (Indexing)")

    documents = [
        {
            "text": "프로젝트 A의 마감일은 2024년 12월 31일입니다. 담당자는 홍길동이며, 주요 마일스톤은 기획(11월), 개발(12월), 테스트(12월 말)입니다.",
            "metadata": {
                "title": "프로젝트 A 일정",
                "author": "홍길동",
                "created_at": "2024-12-01",
                "project_id": "proj_a",
                "type": "schedule",
            },
            "organization_id": ORG_ID,
            "user_id": USER_ID,
        },
        {
            "text": "프로젝트 B는 AI 기반 문서 자동 분류 시스템 개발 프로젝트입니다. Python, FastAPI, OpenAI API를 사용하며, 예산은 5000만원입니다.",
            "metadata": {
                "title": "프로젝트 B 개요",
                "author": "김철수",
                "created_at": "2024-11-15",
                "project_id": "proj_b",
                "type": "overview",
            },
            "organization_id": ORG_ID,
            "user_id": USER_ID,
        },
        {
            "text": "회의록: 2024년 12월 1일 주간 회의. 안건: 프로젝트 A 진행 상황 점검. 결론: 일정 준수 중, 추가 인력 1명 필요.",
            "metadata": {
                "title": "주간 회의록",
                "author": "이영희",
                "created_at": "2024-12-01",
                "type": "meeting",
            },
            "organization_id": ORG_ID,
            "user_id": USER_ID,
        },
    ]

    doc_ids = []

    for i, doc in enumerate(documents, 1):
        print(f"\n📄 문서 {i} 추가 중...")
        print(f"제목: {doc['metadata'].get('title', 'N/A')}")
        print(f"내용: {doc['text'][:50]}...")

        response = requests.post(f"{API_V1}/documents", json=doc)
        print_response(response)

        if response.status_code == 201:
            doc_ids.append(response.json()["doc_id"])
            print(f"✅ 성공: {response.json()['doc_id']}")
        else:
            print(f"❌ 실패")

    return doc_ids


def test_search_documents():
    """
    2단계: 문서 검색 테스트

    🔍 Semantic Search:
    - 다양한 검색어로 문서 찾기
    - 유사도 점수 확인
    """
    print_section("2단계: 문서 검색 (Semantic Search)")

    queries = [
        "프로젝트 A 마감일이 언제야?",
        "AI 프로젝트 예산은?",
        "회의에서 뭐 얘기했어?",
    ]

    for query in queries:
        print(f"\n🔍 검색: {query}")

        payload = {
            "query": query,
            "organization_id": ORG_ID,
            "user_id": USER_ID,
            "limit": 3,
        }

        response = requests.post(f"{API_V1}/documents/search", json=payload)
        print_response(response)

        if response.status_code == 200:
            results = response.json()["results"]
            print(f"\n검색 결과: {len(results)}개")
            for i, result in enumerate(results, 1):
                print(f"\n  [{i}] 유사도: {result['score']:.4f}")
                print(f"      내용: {result['text'][:80]}...")
                print(f"      메타: {result['metadata'].get('title', 'N/A')}")


def test_chat_with_rag():
    """
    3단계: RAG 채팅 테스트

    💬 RAG 동작:
    - 질문 → 문서 검색 → 문서 기반 답변 생성
    - 참고한 문서(sources) 확인
    """
    print_section("3단계: RAG 채팅 (문서 기반 답변)")

    questions = [
        "프로젝트 A의 마감일이 언제야?",
        "프로젝트 B는 무슨 기술을 사용해?",
        "최근 회의에서 어떤 결정이 있었어?",
    ]

    for question in questions:
        print(f"\n💬 질문: {question}")

        payload = {
            "message": question,
            "organization_id": ORG_ID,
            "user_id": USER_ID,
            "use_rag": True,  # RAG 모드 활성화
        }

        response = requests.post(f"{API_V1}/chat", json=payload)
        print_response(response)

        if response.status_code == 200:
            data = response.json()
            print(f"\n🤖 답변:")
            print(f"{data['message']}")

            if data.get("sources"):
                print(f"\n📚 참고 문서: {len(data['sources'])}개")
                for i, source in enumerate(data["sources"], 1):
                    print(f"\n  [{i}] 유사도: {source['score']:.4f}")
                    print(f"      내용: {source['text'][:80]}...")


def test_chat_without_rag():
    """
    4단계: 일반 LLM 채팅 테스트 (RAG 미사용)

    💬 일반 모드:
    - LLM의 일반 지식으로만 답변
    - 문서 검색 없음
    """
    print_section("4단계: 일반 LLM 채팅 (RAG 미사용)")

    question = "안녕하세요! 무엇을 도와드릴까요?"
    print(f"\n💬 질문: {question}")

    payload = {
        "message": question,
        "organization_id": ORG_ID,
        "user_id": USER_ID,
        "use_rag": False,  # RAG 모드 비활성화
    }

    response = requests.post(f"{API_V1}/chat", json=payload)
    print_response(response)

    if response.status_code == 200:
        data = response.json()
        print(f"\n🤖 답변:")
        print(f"{data['message']}")
        print(f"\n📚 참고 문서: {data.get('sources') or '없음 (일반 모드)'}")


def test_stats():
    """
    5단계: 통계 조회

    📊 시스템 상태:
    - 저장된 문서 수
    - Vector Store 정보
    """
    print_section("5단계: 통계 조회")

    response = requests.get(f"{API_V1}/documents/stats")
    print_response(response)

    if response.status_code == 200:
        stats = response.json()
        print(f"\n📊 통계 정보:")
        print(f"  총 문서 수: {stats['total_documents']}")
        print(f"  LLM 모델: {stats['llm_model']}")
        print(f"  Vector Store: {stats['vector_store']['name']}")


def test_health():
    """서버 헬스체크"""
    print_section("0단계: 서버 상태 확인")

    try:
        response = requests.get(f"{BASE_URL}/health")
        print_response(response)

        if response.status_code == 200:
            print("\n✅ 서버가 정상 작동 중입니다.")
            return True
        else:
            print("\n❌ 서버 응답이 비정상입니다.")
            return False

    except requests.exceptions.ConnectionError:
        print("\n❌ 서버에 연결할 수 없습니다.")
        print("\n다음을 확인하세요:")
        print("1. FastAPI 서버가 실행 중인가요? (python -m src.main)")
        print("2. Qdrant가 실행 중인가요? (cd infrastructure/qdrant && make start)")
        return False


def main():
    """전체 테스트 실행"""
    print("\n" + "=" * 70)
    print("  RAG 시스템 전체 테스트")
    print("=" * 70)
    print(f"\n시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 0. 서버 상태 확인
    if not test_health():
        print("\n❌ 서버 연결 실패. 테스트를 중단합니다.")
        return

    try:
        # 1. 문서 추가
        doc_ids = test_add_documents()

        # 2. 문서 검색
        test_search_documents()

        # 3. RAG 채팅
        test_chat_with_rag()

        # 4. 일반 채팅
        test_chat_without_rag()

        # 5. 통계 조회
        test_stats()

        # 완료
        print("\n" + "=" * 70)
        print("  ✅ 모든 테스트 완료!")
        print("=" * 70)
        print(f"\n종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n추가된 문서 ID: {len(doc_ids)}개")
        for i, doc_id in enumerate(doc_ids, 1):
            print(f"  {i}. {doc_id}")

        print("\n💡 다음 단계:")
        print("  - Swagger UI에서 API 직접 테스트: http://localhost:8000/docs")
        print("  - Qdrant 웹 UI에서 벡터 확인: http://localhost:6333/dashboard")

    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
