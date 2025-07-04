"""
Logging utilities for VLR_AI Trading System
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

def setup_logging(level=logging.INFO, log_dir="logs"):
    """Setup comprehensive logging for the trading system"""
    
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with UTF-8 encoding
    import sys
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    # Set encoding to handle Unicode characters
    if hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
        except:
            pass
    root_logger.addHandler(console_handler)
    
    # Main log file handler
    main_log_file = log_path / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        main_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Error log file handler
    error_log_file = log_path / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Trading-specific logger
    trading_logger = logging.getLogger('trading_system')
    trading_log_file = log_path / f"trading_system_{datetime.now().strftime('%Y%m%d')}.log"
    trading_handler = logging.handlers.RotatingFileHandler(
        trading_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    trading_handler.setLevel(logging.INFO)
    trading_handler.setFormatter(detailed_formatter)
    trading_logger.addHandler(trading_handler)
    
    logging.info("Logging system initialized")
    return root_logger

def get_logger(name):
    """Get a logger with the specified name"""
    return logging.getLogger(name)