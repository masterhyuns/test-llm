"""
AI Assistant Service 메인 애플리케이션

FastAPI 기반 AI 비서 서비스
"""
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config.settings import settings
from src.api.rest import health, chat, documents
from src.utils.logger import setup_logging

# 로깅 설정
setup_logging(settings.log_level)
logger = structlog.get_logger()


def create_app() -> FastAPI:
    """FastAPI 애플리케이션 생성"""

    app = FastAPI(
        title="AI Assistant Service",
        description="Cowexa 협업 플랫폼의 AI 비서 서비스",
        version="0.1.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 라우터 등록
    app.include_router(health.router, tags=["Health"])
    app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
    app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])

    # Startup 이벤트
    @app.on_event("startup")
    async def startup_event():
        logger.info(
            "application_startup",
            environment=settings.environment,
            debug=settings.debug,
            llm_model=settings.openai_model,
        )

    # Shutdown 이벤트
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("application_shutdown")

    # 전역 예외 핸들러
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred"
                if settings.is_production
                else str(exc),
            },
        )

    return app


# 애플리케이션 인스턴스
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
