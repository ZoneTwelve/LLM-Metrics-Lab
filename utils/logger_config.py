import os
import logging

def setup_logger(name=None):
    # Get log levels from environment variables
    console_log_level = os.getenv("CLL", "INFO").upper()
    file_log_level = os.getenv("FLL", "").upper()  # Default to INFO if not set
    log_file_path = os.getenv("LOG_FILE_PATH", ".")  # Default to current directory if LOG_FILE_PATH is not set
    log_file_path = os.path.join(log_file_path, "app.log")
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Prevent duplicated handler
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Set the global level for the logger from console_log_level (higher priority)
    logger.setLevel(console_log_level)

    # Set logger format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console logging
    if console_log_level:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File logging
    if file_log_level:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
