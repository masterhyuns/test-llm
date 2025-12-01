"""
Health Check 엔드포인트

서비스 상태 확인
"""
from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    """헬스 체크 응답"""

    status: str
    timestamp: str
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    헬스 체크

    서비스 상태를 확인합니다.

    Returns:
        HealthResponse: 서비스 상태 정보
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="0.1.0",
    )


@router.get("/")
async def root():
    """
    루트 엔드포인트

    Returns:
        dict: 서비스 정보
    """
    return {
        "service": "AI Assistant",
        "status": "running",
        "docs": "/docs",
    }
