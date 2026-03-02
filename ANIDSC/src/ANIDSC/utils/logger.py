import logging
import os
from pathlib import Path
import sys


class LoggerWriter:
    """Redirects writes to a logger at a given level."""
    
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message: str):
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        if self._buffer:
            self.logger.log(self.level, self._buffer)
            self._buffer = ""

    def isatty(self):
        return False


def setup_logging(log_dir: str) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(f"{log_dir}")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Terminal handler — must be added BEFORE stdout is redirected
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    print("logging to", log_dir)

    # Redirect stdout/stderr AFTER the console handler has captured the real stdout
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)

    return logger