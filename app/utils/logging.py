import sys
import logging
from pathlib import Path

# Create logs directory in base path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"


def setup_logging():
    """Initializes standard Python logging with stdout and file handlers."""
    logger = logging.getLogger("app_logger")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup is called multiple times (e.g. during reload)
    if logger.handlers:
        return logger

    # Log Formatter: [Timestamp] [LogLevel] [File:Line] - Message
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler (writes to stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File Handler (writes to logs/app.log)
    try:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file handler: {e}", file=sys.stderr)

    return logger


# Initialize logger immediately
app_logger = setup_logging()
