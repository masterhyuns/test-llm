"""
구조화된 로깅 설정

structlog을 사용한 JSON 로깅
"""
import logging
import structlog


def setup_logging(log_level: str = "INFO"):
    """
    로깅 설정

    Args:
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """

    # structlog 설정
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # 표준 로깅 레벨 설정
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
    )


def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    로거 인스턴스 반환

    Args:
        name: 로거 이름 (보통 __name__ 사용)

    Returns:
        structlog.BoundLogger: 로거 인스턴스

    Example:
        ```python
        logger = get_logger(__name__)
        logger.info("메시지", key="value")
        ```
    """
    return structlog.get_logger(name)
