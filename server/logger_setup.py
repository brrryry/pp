import os
import logging
import datetime

def setup_logger(script_name):
    """
    Configures a logger to write logs to both the console (stdout) 
    and a log file in the `data/logs/` directory.
    Does not clear the root logger's handlers to avoid stripping Uvicorn access logs.
    """
    logs_dir = "data/logs"
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"{script_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        logger.handlers.clear()
        
    # Format of logging output (includes logger name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console Handler (writes to stdout/stderr)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler (writes to data/logs/<script_name>.log)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Mirror Uvicorn logs to our file handler so we can inspect HTTP requests in files
    for uv_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv_logger = logging.getLogger(uv_name)
        has_file = any(isinstance(h, logging.FileHandler) for h in uv_logger.handlers)
        if not has_file:
            uv_logger.addHandler(file_handler)
            
    logger.info(f"--- Logging initialized. Log file: {log_file} ---")
    return logger
