"""Logging configuration using loguru"""
import sys
import os
from pathlib import Path
from loguru import logger


def setup_logger(log_level: str = "INFO"):
    """
    Configure loguru logger with console and file outputs.

    In Lambda environment, only logs to CloudWatch (stdout).
    In local environment, logs to both console and files.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Remove default handler
    logger.remove()

    # Console handler with colors (always enabled for CloudWatch/local)
    logger.add(
        sys.stdout,
        colorize=False,  # Disable colors for Lambda CloudWatch
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level
    )

    # Only add file handlers if NOT in Lambda environment
    is_lambda = os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None

    if not is_lambda:
        try:
            # Create logs directory (only for local)
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            # File handler for all logs
            logger.add(
                log_dir / "app.log",
                rotation="10 MB",
                retention="7 days",
                level="DEBUG",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
            )

            # Error file handler
            logger.add(
                log_dir / "error.log",
                rotation="10 MB",
                retention="30 days",
                level="ERROR",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
            )
        except Exception as e:
            # Silently fail if can't create log files (e.g., read-only filesystem)
            logger.warning(f"Could not create log files: {e}")

    return logger


# Initialize logger
log = setup_logger()
