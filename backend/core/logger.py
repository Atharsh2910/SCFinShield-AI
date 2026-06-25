import sys
from pathlib import Path

from loguru import logger

from backend.core.config import get_settings


def setup_logger():
    settings = get_settings()
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG" if settings.debug else "INFO",
        colorize=True,
    )
    logger.add(
        "logs/scfinshield_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
        format="{time} | {level} | {name}:{line} - {message}",
    )
    return logger
