"""
Database Module for VLR_AI Trading System
Handles all database operations for REAL trading data storage
"""

from .redis_cache import RedisCache

__all__ = ['RedisCache']