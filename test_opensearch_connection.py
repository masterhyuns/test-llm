#!/usr/bin/env python3
"""
OpenSearch 연결 테스트 스크립트

OpenSearch가 정상적으로 연결되는지 확인합니다.
"""
from opensearchpy import OpenSearch
from opensearchpy.exceptions import ConnectionError as OSConnectionError
import sys


def test_connection():
    """OpenSearch 연결 테스트"""

    print("=" * 70)
    print("  OpenSearch 연결 테스트")
    print("=" * 70)

    # 연결 정보
    host = "3.34.20.81"
    port = 30920
    user = "admin"
    password = "admin"
    use_ssl = True  # HTTPS 사용

    print(f"\n연결 정보:")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  User: {user}")
    print(f"  SSL: {use_ssl}")

    try:
        # 1. 클라이언트 생성
        print("\n1. OpenSearch 클라이언트 생성 중...")
        client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=(user, password),
            use_ssl=use_ssl,  # HTTPS 사용
            verify_certs=False,  # 자체 서명 인증서 허용
            ssl_show_warn=False,
            timeout=10,
        )
        print("✅ 클라이언트 생성 성공")

        # 2. 클러스터 정보 확인
        print("\n2. 클러스터 정보 확인 중...")
        info = client.info()
        print("✅ 연결 성공!")
        print(f"\n클러스터 정보:")
        print(f"  Name: {info['cluster_name']}")
        print(f"  Version: {info['version']['number']}")
        print(f"  Lucene Version: {info['version']['lucene_version']}")

        # 3. 클러스터 상태 확인
        print("\n3. 클러스터 상태 확인 중...")
        health = client.cluster.health()
        print(f"✅ 클러스터 상태: {health['status']}")
        print(f"  Nodes: {health['number_of_nodes']}")
        print(f"  Active Shards: {health['active_shards']}")

        # 4. 인덱스 목록 확인
        print("\n4. 기존 인덱스 확인 중...")
        indices = client.cat.indices(format="json")
        print(f"✅ 총 {len(indices)}개 인덱스 존재")

        if indices:
            print("\n기존 인덱스 목록:")
            for idx in indices[:10]:  # 최대 10개만 표시
                print(f"  - {idx['index']} (docs: {idx.get('docs.count', 0)})")
            if len(indices) > 10:
                print(f"  ... 외 {len(indices) - 10}개")

        # 5. k-NN 플러그인 확인
        print("\n5. k-NN 플러그인 확인 중...")
        plugins = client.cat.plugins(format="json")
        knn_installed = any("knn" in p.get("component", "").lower() for p in plugins)

        if knn_installed:
            print("✅ k-NN 플러그인 설치됨 (벡터 검색 가능)")
        else:
            print("⚠️  k-NN 플러그인 미설치 (벡터 검색 불가)")
            print("   OpenSearch 2.x 버전에서는 기본 내장되어야 합니다.")

        # 성공
        print("\n" + "=" * 70)
        print("  🎉 모든 테스트 통과!")
        print("=" * 70)
        print("\n✅ OpenSearch가 정상적으로 연결되었습니다.")
        print("✅ AI Assistant 서비스에서 사용 가능합니다.")
        print("\n다음 단계:")
        print("  1. 서버 실행: .venv/bin/python -m src.main")
        print("  2. Swagger UI: http://localhost:8000/docs")
        print("=" * 70)

        return True

    except OSConnectionError as e:
        print(f"\n❌ 연결 실패: {e}")
        print("\n해결 방법:")
        print("1. OpenSearch가 실행 중인지 확인")
        print("2. 방화벽/보안그룹에서 포트 30920 열려있는지 확인")
        print("3. 네트워크 연결 확인")
        return False

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print(f"\n상세 오류:")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
