"""
Structured Logging Configuration.
Sets up Loguru to intercept FastAPI, Uvicorn, and standard library logs,
formatting them cleanly with levels, timestamps, and execution context.
"""
import logging
import sys

from loguru import logger

from app.core.config import settings


def setup_logging() -> None:
    """
    Configure Loguru to override the default system loggers and format
    all output to stdout, which is standard for containerized and Render environments.
    """
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stdout,
        format=log_format,
        level=settings.LOG_LEVEL,
        colorize=True,
        enqueue=True,
    )

    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            frame = logging.currentframe()
            depth = 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
        lib_logger = logging.getLogger(logger_name)
        lib_logger.handlers = [InterceptHandler()]
        lib_logger.propagate = False

    logger.info("Logging infrastructure successfully initialized via Loguru.")
