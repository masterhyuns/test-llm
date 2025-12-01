#!/usr/bin/env python3
"""
Qdrant 연결 테스트 스크립트

Qdrant가 정상적으로 실행 중인지 확인합니다.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import sys


def test_connection():
    """Qdrant 연결 테스트"""

    print("=" * 50)
    print("Qdrant 연결 테스트")
    print("=" * 50)

    try:
        # 1. 클라이언트 생성
        print("\n1. Qdrant 클라이언트 생성 중...")
        client = QdrantClient(host="localhost", port=6333)
        print("✅ 클라이언트 생성 성공")

        # 2. 헬스체크
        print("\n2. 헬스체크...")
        # collections를 조회하여 연결 확인
        collections = client.get_collections()
        print(f"✅ 헬스체크 성공 - 현재 컬렉션 수: {len(collections.collections)}")

        # 3. 테스트 컬렉션 생성
        print("\n3. 테스트 컬렉션 생성 중...")
        test_collection = "test_collection"

        # 기존 테스트 컬렉션 삭제 (있으면)
        try:
            client.delete_collection(collection_name=test_collection)
            print(f"   기존 '{test_collection}' 삭제됨")
        except:
            pass

        # 새 컬렉션 생성
        client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=128, distance=Distance.COSINE)
        )
        print(f"✅ 테스트 컬렉션 '{test_collection}' 생성 성공")

        # 4. 테스트 벡터 추가
        print("\n4. 테스트 벡터 추가 중...")
        from qdrant_client.models import PointStruct

        test_points = [
            PointStruct(
                id=1,
                vector=[0.1] * 128,
                payload={"text": "테스트 문서 1", "type": "test"}
            ),
            PointStruct(
                id=2,
                vector=[0.2] * 128,
                payload={"text": "테스트 문서 2", "type": "test"}
            )
        ]

        client.upsert(
            collection_name=test_collection,
            points=test_points
        )
        print(f"✅ {len(test_points)}개 벡터 추가 성공")

        # 5. 검색 테스트
        print("\n5. 검색 테스트...")
        search_result = client.query_points(
            collection_name=test_collection,
            query=[0.15] * 128,
            limit=2
        )
        print(f"✅ 검색 성공 - {len(search_result.points)}개 결과 반환")
        for i, hit in enumerate(search_result.points, 1):
            print(f"   {i}. ID: {hit.id}, Score: {hit.score:.4f}, Text: {hit.payload['text']}")

        # 6. 정리
        print("\n6. 테스트 컬렉션 삭제 중...")
        client.delete_collection(collection_name=test_collection)
        print("✅ 테스트 컬렉션 삭제 성공")

        # 성공
        print("\n" + "=" * 50)
        print("🎉 모든 테스트 통과!")
        print("=" * 50)
        print("\nQdrant가 정상적으로 실행 중입니다.")
        print("웹 UI: http://localhost:6333/dashboard")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("\n해결 방법:")
        print("1. Qdrant가 실행 중인지 확인: docker-compose ps")
        print("2. 포트가 열려있는지 확인: lsof -i :6333")
        print("3. 로그 확인: docker-compose logs qdrant")
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
