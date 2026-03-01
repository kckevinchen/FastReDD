import os
import logging
from datetime import datetime


def get_log_level(level_str):
    """
    Convert string log level to logging level constant.
    
    Args:
        level_str (str): String representation of log level
        
    Returns:
        int: Logging level constant
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    return level_map.get(level_str.upper(), logging.WARNING)


class NoHTTPRequestFilter(logging.Filter):
    def filter(self, record):
        return "HTTP Request" not in record.getMessage()
    

def setup_logging(prefix_str=None, log_dir="logs", console_log_level=logging.WARNING):
    """
    Set up logging configuration with different levels for console and file outputs.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler with configurable level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)  # Configurable console log level
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with all levels (DEBUG and above)
    if prefix_str:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_filename = f"{prefix_str}-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
        log_filepath = os.path.join(log_dir, log_filename)
        file_handler = logging.FileHandler(log_filepath, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)  # Log everything to the file
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logging.info(f"Logs will be saved to: {log_filepath}")
    else:
        logging.info("File logging is disabled")

    # for handler in logger.handlers:
    #     handler.addFilter(NoHTTPRequestFilter())

    # Test logging
    logging.info("Logging initialized")


if __name__ == "__main__":
    log_file = setup_logging()
    logging.info("This is an info message.")
    logging.debug("This is a debug message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")
