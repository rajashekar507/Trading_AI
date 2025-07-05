"""
Redis Caching System for VLR_AI Trading System
Implements caching for REAL market data, strategy results, and API responses
"""

import asyncio
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import os

try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("[WARNING] Redis not available. Install with: pip install redis")

logger = logging.getLogger('trading_system.redis_cache')

class RedisCacheManager:
    """Redis cache manager for REAL market data"""
    
    def __init__(self, settings):
        self.settings = settings
        self.redis_client = None
        self.async_redis_client = None
        
        # Redis configuration from environment or settings
        self.host = os.getenv('REDIS_HOST', getattr(settings, 'REDIS_HOST', 'localhost'))
        self.port = int(os.getenv('REDIS_PORT', getattr(settings, 'REDIS_PORT', 6379)))
        self.db = int(os.getenv('REDIS_DB', getattr(settings, 'REDIS_DB', 0)))
        self.password = os.getenv('REDIS_PASSWORD', getattr(settings, 'REDIS_PASSWORD', None))
        
        # Cache settings
        self.default_ttl = 300  # 5 minutes default TTL
        self.market_data_ttl = 30  # 30 seconds for real market data
        self.strategy_results_ttl = 600  # 10 minutes for strategy results
        self.api_response_ttl = 60  # 1 minute for API responses
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        
        if REDIS_AVAILABLE:
            self._initialize_redis()
        else:
            logger.warning("[REDIS_CACHE] Redis not available, caching disabled")
    
    def _initialize_redis(self):
        """Initialize Redis connections for REAL data caching"""
        try:
            # Synchronous Redis client
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            
            # Asynchronous Redis client
            self.async_redis_client = aioredis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False
            )
            
            logger.info(f"[REDIS_CACHE] Connected to Redis at {self.host}:{self.port} for REAL data caching")
            
        except Exception as e:
            logger.error(f"[REDIS_CACHE] Failed to connect to Redis: {e}")
            self.redis_client = None
            self.async_redis_client = None
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        return REDIS_AVAILABLE and self.redis_client is not None
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for Redis storage"""
        try:
            if isinstance(data, (dict, list)):
                return json.dumps(data, default=str).encode('utf-8')
            elif isinstance(data, str):
                return data.encode('utf-8')
            else:
                return pickle.dumps(data)
        except Exception as e:
            logger.error(f"[REDIS_CACHE] Serialization error: {e}")
            return pickle.dumps(data)
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from Redis"""
        try:
            # Try JSON first
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fall back to pickle
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"[REDIS_CACHE] Deserialization error: {e}")
            return None
    
    def _generate_key(self, category: str, identifier: str) -> str:
        """Generate Redis key with namespace"""
        return f"vlr_ai:real_data:{category}:{identifier}"
    
    def set_market_data(self, instrument: str, data: Dict[str, Any]) -> bool:
        """Cache REAL market data"""
        if not self.is_available():
            return False
        
        try:
            key = self._generate_key("market_data", instrument)
            serialized_data = self._serialize_data({
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'data_type': 'REAL_MARKET_DATA'
            })
            
            result = self.redis_client.setex(key, self.market_data_ttl, serialized_data)
            
            if result:
                self.stats['sets'] += 1
                logger.debug(f"[REDIS_CACHE] Cached REAL market data for {instrument}")
            
            return result
            
        except Exception as e:
            logger.error(f"[REDIS_CACHE] Error caching market data: {e}")
            self.stats['errors'] += 1
            return False
    
    def get_market_data(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Get cached REAL market data"""
        if not self.is_available():
            return None
        
        try:
            key = self._generate_key("market_data", instrument)
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                self.stats['hits'] += 1
                data = self._deserialize_data(cached_data)
                
                if data and data.get('data_type') == 'REAL_MARKET_DATA':
                    logger.debug(f"[REDIS_CACHE] Cache hit for REAL market data: {instrument}")
                    return data['data']
                else:
                    logger.warning(f"[REDIS_CACHE] Invalid cached data for {instrument}")
                    self.delete_market_data(instrument)
            else:
                self.stats['misses'] += 1
                logger.debug(f"[REDIS_CACHE] Cache miss for market data: {instrument}")
            
            return None
            
        except Exception as e:
            logger.error(f"[REDIS_CACHE] Error getting market data: {e}")
            self.stats['errors'] += 1
            return None
    
    def set_strategy_result(self, strategy_name: str, instrument: str, result: Dict[str, Any]) -> bool:
        """Cache REAL strategy results"""
        if not self.is_available():
            return False
        
        try:
            key = self._generate_key("strategy", f"{strategy_name}:{instrument}")
            serialized_data = self._serialize_data({
                'result': result,
                'timestamp': datetime.now().isoformat(),
                'strategy': strategy_name,
                'instrument': instrument,
                'data_type': 'REAL_STRATEGY_RESULT'
            })
            
            result = self.redis_client.setex(key, self.strategy_results_ttl, serialized_data)
            
            if result:
                self.stats['sets'] += 1
                logger.debug(f"[REDIS_CACHE] Cached REAL strategy result for {strategy_name}:{instrument}")
            
            return result
            
        except Exception as e:
            logger.error(f"[REDIS_CACHE] Error caching strategy result: {e}")
            self.stats['errors'] += 1
            return False
    
    def get_strategy_result(self, strategy_name: str, instrument: str) -> Optional[Dict[str, Any]]:
        """Get cached REAL strategy result"""
        if not self.is_available():
            return None
        
        try:
            key = self._generate_key("strategy", f"{strategy_name}:{instrument}")
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                self.stats['hits'] += 1
                data = self._deserialize_data(cached_data)
                
                if data and data.get('data_type') == 'REAL_STRATEGY_RESULT':
                    logger.debug(f"[REDIS_CACHE] Cache hit for REAL strategy result: {strategy_name}:{instrument}")
                    return data['result']
                else:
                    logger.warning(f"[REDIS_CACHE] Invalid cached strategy result for {strategy_name}:{instrument}")
                    self.delete_strategy_result(strategy_name, instrument)
            else:
                self.stats['misses'] += 1
                logger.debug(f"[REDIS_CACHE] Cache miss for strategy result: {strategy_name}:{instrument}")
            
            return None
            
        except Exception as e:
            logger.error(f"[REDIS_CACHE] Error getting strategy result: {e}")
            self.stats['errors'] += 1
            return None
    
    def set_api_response(self, api_name: str, endpoint: str, response: Any) -> bool:
        """Cache REAL API response"""
        if not self.is_available():
            return False
        
        try:
            key = self._generate_key("api_response", f"{api_name}:{endpoint}")
            serialized_data = self._serialize_data({
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'api_name': api_name,
                'endpoint': endpoint,
                'data_type': 'REAL_API_RESPONSE'
            })
            
            result = self.redis_client.setex(key, self.api_response_ttl, serialized_data)
            
            if result:
                self.stats['sets'] += 1
                logger.debug(f"[REDIS_CACHE] Cached REAL API response for {api_name}:{endpoint}")
            
            return result
            
        except Exception as e:
            logger.error(f"[REDIS_CACHE] Error caching API response: {e}")
            self.stats['errors'] += 1
            return False
    
    def get_api_response(self, api_name: str, endpoint: str) -> Optional[Any]:
        """Get cached REAL API response"""
        if not self.is_available():
            return None
        
        try:
            key = self._generate_key("api_response", f"{api_name}:{endpoint}")
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                self.stats['hits'] += 1
                data = self._deserialize_data(cached_data)
                
                if data and data.get('data_type') == 'REAL_API_RESPONSE':
                    logger.debug(f"[REDIS_CACHE] Cache hit for REAL API response: {api_name}:{endpoint}")
                    return data['response']
                else:
                    logger.warning(f"[REDIS_CACHE] Invalid cached API response for {api_name}:{endpoint}")
                    self.delete_api_response(api_name, endpoint)
            else:
                self.stats['misses'] += 1
                logger.debug(f"[REDIS_CACHE] Cache miss for API response: {api_name}:{endpoint}")
            
            return None
            
        except Exception as e:
            logger.error(f"[REDIS_CACHE] Error getting API response: {e}")
            self.stats['errors'] += 1
            return None
    
    def delete_market_data(self, instrument: str) -> bool:
        """Delete cached market data"""
        if not self.is_available():
            return False
        
        try:
            key = self._generate_key("market_data", instrument)
            result = self.redis_client.delete(key)
            
            if result:
                self.stats['deletes'] += 1
                logger.debug(f"[REDIS_CACHE] Deleted cached market data for {instrument}")
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"[REDIS_CACHE] Error deleting market data: {e}")
            self.stats['errors'] += 1
            return False
    
    def delete_strategy_result(self, strategy_name: str, instrument: str) -> bool:
        """Delete cached strategy result"""
        if not self.is_available():
            return False
        
        try:
            key = self._generate_key("strategy", f"{strategy_name}:{instrument}")
            result = self.redis_client.delete(key)
            
            if result:
                self.stats['deletes'] += 1
                logger.debug(f"[REDIS_CACHE] Deleted cached strategy result for {strategy_name}:{instrument}")
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"[REDIS_CACHE] Error deleting strategy result: {e}")
            self.stats['errors'] += 1
            return False
    
    def delete_api_response(self, api_name: str, endpoint: str) -> bool:
        """Delete cached API response"""
        if not self.is_available():
            return False
        
        try:
            key = self._generate_key("api_response", f"{api_name}:{endpoint}")
            result = self.redis_client.delete(key)
            
            if result:
                self.stats['deletes'] += 1
                logger.debug(f"[REDIS_CACHE] Deleted cached API response for {api_name}:{endpoint}")
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"[REDIS_CACHE] Error deleting API response: {e}")
            self.stats['errors'] += 1
            return False
    
    def clear_all_cache(self) -> bool:
        """Clear all cached data"""
        if not self.is_available():
            return False
        
        try:
            pattern = self._generate_key("*", "*")
            keys = self.redis_client.keys(pattern)
            
            if keys:
                result = self.redis_client.delete(*keys)
                self.stats['deletes'] += result
                logger.info(f"[REDIS_CACHE] Cleared {result} cached items")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"[REDIS_CACHE] Error clearing cache: {e}")
            self.stats['errors'] += 1
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        cache_info = {
            'available': self.is_available(),
            'host': self.host,
            'port': self.port,
            'db': self.db,
            'stats': self.stats.copy(),
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
        
        if self.is_available():
            try:
                info = self.redis_client.info()
                cache_info['redis_info'] = {
                    'used_memory': info.get('used_memory_human', 'N/A'),
                    'connected_clients': info.get('connected_clients', 0),
                    'total_commands_processed': info.get('total_commands_processed', 0),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                }
            except Exception as e:
                logger.error(f"[REDIS_CACHE] Error getting Redis info: {e}")
        
        return cache_info
    
    async def async_set_market_data(self, instrument: str, data: Dict[str, Any]) -> bool:
        """Async version of set_market_data"""
        if not self.is_available() or not self.async_redis_client:
            return False
        
        try:
            key = self._generate_key("market_data", instrument)
            serialized_data = self._serialize_data({
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'data_type': 'REAL_MARKET_DATA'
            })
            
            result = await self.async_redis_client.setex(key, self.market_data_ttl, serialized_data)
            
            if result:
                self.stats['sets'] += 1
                logger.debug(f"[REDIS_CACHE] Async cached REAL market data for {instrument}")
            
            return result
            
        except Exception as e:
            logger.error(f"[REDIS_CACHE] Error async caching market data: {e}")
            self.stats['errors'] += 1
            return False
    
    async def async_get_market_data(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Async version of get_market_data"""
        if not self.is_available() or not self.async_redis_client:
            return None
        
        try:
            key = self._generate_key("market_data", instrument)
            cached_data = await self.async_redis_client.get(key)
            
            if cached_data:
                self.stats['hits'] += 1
                data = self._deserialize_data(cached_data)
                
                if data and data.get('data_type') == 'REAL_MARKET_DATA':
                    logger.debug(f"[REDIS_CACHE] Async cache hit for REAL market data: {instrument}")
                    return data['data']
                else:
                    logger.warning(f"[REDIS_CACHE] Invalid async cached data for {instrument}")
                    await self.async_delete_market_data(instrument)
            else:
                self.stats['misses'] += 1
                logger.debug(f"[REDIS_CACHE] Async cache miss for market data: {instrument}")
            
            return None
            
        except Exception as e:
            logger.error(f"[REDIS_CACHE] Error async getting market data: {e}")
            self.stats['errors'] += 1
            return None
    
    async def async_delete_market_data(self, instrument: str) -> bool:
        """Async version of delete_market_data"""
        if not self.is_available() or not self.async_redis_client:
            return False
        
        try:
            key = self._generate_key("market_data", instrument)
            result = await self.async_redis_client.delete(key)
            
            if result:
                self.stats['deletes'] += 1
                logger.debug(f"[REDIS_CACHE] Async deleted cached market data for {instrument}")
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"[REDIS_CACHE] Error async deleting market data: {e}")
            self.stats['errors'] += 1
            return False
    
    def close(self):
        """Close Redis connections"""
        try:
            if self.redis_client:
                self.redis_client.close()
            if self.async_redis_client:
                asyncio.create_task(self.async_redis_client.close())
            logger.info("[REDIS_CACHE] Redis connections closed")
        except Exception as e:
            logger.error(f"[REDIS_CACHE] Error closing Redis connections: {e}")

# Global cache instance
_global_cache = None

def initialize_global_cache(settings):
    """Initialize global Redis cache for REAL data"""
    global _global_cache
    _global_cache = RedisCacheManager(settings)
    logger.info("[REDIS_CACHE] Global cache initialized for REAL market data")
    return _global_cache

def get_global_cache() -> Optional[RedisCacheManager]:
    """Get global Redis cache"""
    return _global_cache

# Decorator for automatic caching of REAL data
def cache_result(cache_key: str, ttl: int = 300, cache_type: str = "general"):
    """Decorator to automatically cache function results with REAL data"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_global_cache()
            if not cache or not cache.is_available():
                # No cache available, execute function directly
                return func(*args, **kwargs)
            
            # Try to get from cache
            if cache_type == "market_data":
                cached_result = cache.get_market_data(cache_key)
            elif cache_type == "api_response":
                cached_result = cache.get_api_response("general", cache_key)
            else:
                # Use general caching
                try:
                    key = cache._generate_key("general", cache_key)
                    cached_data = cache.redis_client.get(key)
                    if cached_data:
                        cached_result = cache._deserialize_data(cached_data)
                    else:
                        cached_result = None
                except:
                    cached_result = None
            
            if cached_result is not None:
                logger.debug(f"[REDIS_CACHE] Cache hit for {cache_key}")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Cache the result
            if cache_type == "market_data":
                cache.set_market_data(cache_key, result)
            elif cache_type == "api_response":
                cache.set_api_response("general", cache_key, result)
            else:
                # Use general caching
                try:
                    key = cache._generate_key("general", cache_key)
                    serialized_data = cache._serialize_data(result)
                    cache.redis_client.setex(key, ttl, serialized_data)
                except Exception as e:
                    logger.error(f"[REDIS_CACHE] Error caching result: {e}")
            
            return result
        
        return wrapper
    return decorator