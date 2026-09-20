"""
Centralized logging configuration for the MLB Stats MCP project.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file. Nothing here may write to stdout:
# under the stdio transport stdout carries the JSON-RPC stream, and stray output
# corrupts it, so clients fail to parse the handshake and drop the connection.
env_path = Path(__file__).parent.parent.parent / ".env"
print(f"Loading environment variables from {env_path}", file=sys.stderr)
load_dotenv(env_path)


def setup_logging(logger_name: Optional[str] = None) -> logging.Logger:
    """
    Configure logging based on environment variables.

    Args:
        logger_name: Optional name for the logger. If None, returns root logger.

    Returns:
        Configured logger instance
    """
    # Get logging configuration from environment variables
    log_level = os.environ.get("MLB_STATS_LOG_LEVEL", "INFO").upper()
    log_file = os.environ.get("MLB_STATS_LOG_FILE", None)
    log_path = os.environ.get("MLB_STATS_LOG_PATH", "logs/")

    # Get the logger - either named or root
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()

    # Set the log level
    level = getattr(logging, log_level, logging.INFO)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)8s - %(name)s - %(message)s")

    # Clear existing handlers to avoid duplicate logs
    logger.handlers = []

    # Add handlers based on configuration
    if log_file:
        # Create logs directory if it doesn't exist
        project_root = Path(__file__).parent.parent.parent
        log_dir = project_root / log_path
        log_dir.mkdir(parents=True, exist_ok=True)

        # Combine path and filename
        full_log_path = log_dir / log_file

        # File handler if log file is specified
        file_handler = logging.FileHandler(full_log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging configured at {log_level} level, writing to {full_log_path}")
    else:
        # Stream handler if no log file is specified
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logger.info(f"MLB Stats API logging configured at {log_level} level, writing to stderr")

    return logger
