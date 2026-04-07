import os
import time
import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

# Import the custom formatter for milliseconds
from src.core.utils.timestamp_formatter import MillisecondFormatter

# Global variable to store the log directory for the current session
_log_directory = None


def create_log_directory(base_path="logs"):
    """
    Ensures the base log directory exists and returns its path.
    Timestamped subdirectories are no longer created.

    Args:
        base_path (str): Base directory for logs, default is 'logs'

    Returns:
        str: Path to the base log directory
    """
    global _log_directory

    if _log_directory is not None:
        return _log_directory

    # Ensure the base log directory exists
    os.makedirs(base_path, exist_ok=True)

    # Set the log directory to the base path itself
    _log_directory = base_path

    return _log_directory


def get_current_log_directory():
    """
    Gets the current log directory path without creating a new one.
    Useful for referencing the same directory across components.

    Returns:
        str: Path to the current timestamped directory, or None if it hasn't been created yet
    """
    global _log_directory
    return _log_directory


def setup_live_system_logger(base_path="logs"):
    """
    Sets up the system-level logger for live mode.
    Logs are rotated daily and kept for 30 days.
    """
    os.makedirs(base_path, exist_ok=True)
    
    # Use a fixed base name for the active log, rotation will append date
    log_filename = os.path.join(base_path, "live_system.log")
    
    # Get the root logger to capture logs from all modules
    logger = logging.getLogger() # Root logger
    logger.setLevel(logging.INFO)
    
    # Avoid adding handlers multiple times
    if any(isinstance(h, TimedRotatingFileHandler) and h.baseFilename == log_filename for h in logger.handlers):
        return logger

    # Daily rotation, keep 30 days
    handler = TimedRotatingFileHandler(
        log_filename, when="midnight", interval=1, backupCount=30, encoding='utf-8'
    )
    handler.suffix = "%Y-%m-%d"
    
    # Add logger name to format for better context
    formatter = MillisecondFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    # Add console handler (so logs still appear on console)
    # Check if a StreamHandler is already present
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    
    return logger


# --- New Components for Global Logging ---

class StreamToLogger:
    """
    A file-like object that redirects writes to a logger.
    Used to capture stdout and stderr.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        # The logger handlers should flush automatically
        pass

    def isatty(self):
        """
        Pretends to not be a terminal to prevent libraries like uvicorn
        from trying to write color codes to the log file.
        """
        return False

def handle_exception(exc_type, exc_value, exc_traceback):
    """
    A custom exception hook to log all uncaught exceptions.
    """
    logger = logging.getLogger() # Get root logger
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

def setup_global_logging(base_path="logs"):
    """
    Sets up a global logger that captures stdout, stderr, and uncaught exceptions.
    This should be called once at the start of the application in live mode.
    """
    # 1. Setup the base file and console logger on the root logger
    logger = setup_live_system_logger(base_path)

    # 2. Redirect stdout and stderr to our logger
    # Check if redirection has already happened to avoid loops
    if not isinstance(sys.stdout, StreamToLogger):
        sys.stdout = StreamToLogger(logger, logging.INFO)
    
    if not isinstance(sys.stderr, StreamToLogger):
        sys.stderr = StreamToLogger(logger, logging.ERROR)

    # 3. Set the custom exception hook
    sys.excepthook = handle_exception
    
    logger.info("Global logging configured. Stdout, stderr, and exceptions will be captured in live_system.log.")
    
    return logger