"""Logging utility for CCML Engine with safe Unicode and Windows terminal support."""

import logging
import sys
from pathlib import Path
from src.config import REPORTS_DIR

LOG_FILE = REPORTS_DIR / "pipeline.log"

class SafeStreamHandler(logging.StreamHandler):
    """Stream handler that safely writes unicode strings to Windows consoles."""
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                # Fallback to ASCII representation with backslashreplace/replace
                clean_msg = msg.encode(stream.encoding or "ascii", errors="replace").decode(stream.encoding or "ascii")
                stream.write(clean_msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logger(name: str = "CCML") -> logging.Logger:
    """Configures and returns a logger instance with dual outputs (console + file)."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    # Safe Console Handler
    console_handler = SafeStreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File Handler (always UTF-8)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()
