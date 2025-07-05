"""
API Rate Limiting System for VLR_AI Trading System
Implements rate limiting, request queuing, and API usage monitoring for REAL market data
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict, deque
from functools import wraps
import json
from pathlib import Path

logger = logging.getLogger('trading_system.api_rate_limiter')

class RateLimiter:
    """Token bucket rate limiter implementation"""
    
    def __init__(self, max_requests: int, time_window: int, burst_limit: Optional[int] = None):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds
            burst_limit: Maximum burst requests (defaults to max_requests)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.burst_limit = burst_limit or max_requests
        
        self.tokens = self.max_requests
        self.last_refill = time.time()
        self.request_times = deque()
        
        logger.info(f"[RATE_LIMITER] Initialized: {max_requests} requests per {time_window}s")
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens for API request
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens acquired, False if rate limited
        """
        current_time = time.time()
        
        # Refill tokens based on time passed
        self._refill_tokens(current_time)
        
        # Clean old request times
        self._clean_old_requests(current_time)
        
        # Check if we can make the request
        if self.tokens >= tokens and len(self.request_times) < self.burst_limit:
            self.tokens -= tokens
            self.request_times.append(current_time)
            return True
        
        return False
    
    def _refill_tokens(self, current_time: float):
        """Refill tokens based on elapsed time"""
        time_passed = current_time - self.last_refill
        tokens_to_add = int(time_passed * (self.max_requests / self.time_window))
        
        if tokens_to_add > 0:
            self.tokens = min(self.max_requests, self.tokens + tokens_to_add)
            self.last_refill = current_time
    
    def _clean_old_requests(self, current_time: float):
        """Remove old request times outside the time window"""
        cutoff_time = current_time - self.time_window
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()
    
    def get_wait_time(self) -> float:
        """Get estimated wait time until next request can be made"""
        if self.tokens > 0:
            return 0.0
        
        # Calculate time until next token refill
        time_per_token = self.time_window / self.max_requests
        return time_per_token
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status"""
        current_time = time.time()
        self._refill_tokens(current_time)
        self._clean_old_requests(current_time)
        
        return {
            'available_tokens': self.tokens,
            'max_tokens': self.max_requests,
            'requests_in_window': len(self.request_times),
            'burst_limit': self.burst_limit,
            'time_window': self.time_window,
            'wait_time': self.get_wait_time()
        }

class APIRateLimitManager:
    """Manages rate limiting for multiple APIs"""
    
    def __init__(self, settings):
        self.settings = settings
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.request_queues: Dict[str, asyncio.Queue] = {}
        self.usage_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total_requests': 0,
            'successful_requests': 0,
            'rate_limited_requests': 0,
            'failed_requests': 0,
            'last_request_time': None,
            'average_response_time': 0.0
        })
        
        # Initialize rate limiters for known APIs
        self._initialize_rate_limiters()
        
        logger.info("[API_RATE_LIMITER] API Rate Limit Manager initialized for REAL market data APIs")
    
    def _initialize_rate_limiters(self):
        """Initialize rate limiters for different APIs"""
        # Kite Connect API limits
        self.rate_limiters['kite_orders'] = RateLimiter(
            max_requests=10, time_window=1, burst_limit=15  # 10 orders per second
        )
        self.rate_limiters['kite_quotes'] = RateLimiter(
            max_requests=100, time_window=1, burst_limit=150  # 100 quotes per second
        )
        self.rate_limiters['kite_historical'] = RateLimiter(
            max_requests=3, time_window=1, burst_limit=5  # 3 historical requests per second
        )
        
        # Dhan API limits
        self.rate_limiters['dhan_orders'] = RateLimiter(
            max_requests=5, time_window=1, burst_limit=10  # Conservative limit
        )
        self.rate_limiters['dhan_quotes'] = RateLimiter(
            max_requests=50, time_window=1, burst_limit=75  # Conservative limit
        )
        
        # External APIs
        self.rate_limiters['yahoo_finance'] = RateLimiter(
            max_requests=100, time_window=60, burst_limit=120  # 100 per minute
        )
        self.rate_limiters['telegram'] = RateLimiter(
            max_requests=30, time_window=1, burst_limit=40  # 30 messages per second
        )
        self.rate_limiters['google_sheets'] = RateLimiter(
            max_requests=100, time_window=100, burst_limit=120  # 100 per 100 seconds
        )
        
        logger.info(f"[API_RATE_LIMITER] Initialized {len(self.rate_limiters)} rate limiters")
    
    async def make_request(self, api_name: str, request_func: Callable, *args, **kwargs) -> Any:
        """
        Make API request with rate limiting
        
        Args:
            api_name: Name of the API (e.g., 'kite_orders', 'telegram')
            request_func: Function to make the API request
            *args, **kwargs: Arguments for the request function
            
        Returns:
            Result of the API request
        """
        if api_name not in self.rate_limiters:
            logger.warning(f"[API_RATE_LIMITER] No rate limiter for {api_name}, creating default")
            self.rate_limiters[api_name] = RateLimiter(max_requests=10, time_window=1)
        
        rate_limiter = self.rate_limiters[api_name]
        start_time = time.time()
        
        # Wait for rate limit
        while not await rate_limiter.acquire():
            wait_time = rate_limiter.get_wait_time()
            logger.debug(f"[API_RATE_LIMITER] Rate limited for {api_name}, waiting {wait_time:.2f}s")
            self.usage_stats[api_name]['rate_limited_requests'] += 1
            await asyncio.sleep(wait_time + 0.1)  # Small buffer
        
        try:
            # Make the request
            self.usage_stats[api_name]['total_requests'] += 1
            self.usage_stats[api_name]['last_request_time'] = datetime.now().isoformat()
            
            if asyncio.iscoroutinefunction(request_func):
                result = await request_func(*args, **kwargs)
            else:
                result = request_func(*args, **kwargs)
            
            # Update success stats
            self.usage_stats[api_name]['successful_requests'] += 1
            
            # Update average response time
            response_time = time.time() - start_time
            current_avg = self.usage_stats[api_name]['average_response_time']
            total_requests = self.usage_stats[api_name]['successful_requests']
            self.usage_stats[api_name]['average_response_time'] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
            
            logger.debug(f"[API_RATE_LIMITER] {api_name} request completed in {response_time:.2f}s")
            return result
            
        except Exception as e:
            self.usage_stats[api_name]['failed_requests'] += 1
            logger.error(f"[API_RATE_LIMITER] {api_name} request failed: {e}")
            raise e
    
    def get_api_status(self, api_name: str) -> Dict[str, Any]:
        """Get status for specific API"""
        if api_name not in self.rate_limiters:
            return {'error': f'No rate limiter for {api_name}'}
        
        rate_limiter_status = self.rate_limiters[api_name].get_status()
        usage_stats = self.usage_stats[api_name]
        
        return {
            'api_name': api_name,
            'rate_limiter': rate_limiter_status,
            'usage_stats': usage_stats,
            'health_score': self._calculate_health_score(api_name)
        }
    
    def get_all_api_status(self) -> Dict[str, Any]:
        """Get status for all APIs"""
        status = {}
        for api_name in self.rate_limiters.keys():
            status[api_name] = self.get_api_status(api_name)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'apis': status,
            'summary': self._get_summary_stats()
        }
    
    def _calculate_health_score(self, api_name: str) -> float:
        """Calculate health score for API (0-100)"""
        stats = self.usage_stats[api_name]
        
        if stats['total_requests'] == 0:
            return 100.0  # No requests yet, assume healthy
        
        success_rate = stats['successful_requests'] / stats['total_requests']
        rate_limit_penalty = min(stats['rate_limited_requests'] / stats['total_requests'], 0.5)
        
        # Health score based on success rate and rate limiting
        health_score = (success_rate * 100) - (rate_limit_penalty * 50)
        return max(0.0, min(100.0, health_score))
    
    def _get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all APIs"""
        total_requests = sum(stats['total_requests'] for stats in self.usage_stats.values())
        total_successful = sum(stats['successful_requests'] for stats in self.usage_stats.values())
        total_rate_limited = sum(stats['rate_limited_requests'] for stats in self.usage_stats.values())
        total_failed = sum(stats['failed_requests'] for stats in self.usage_stats.values())
        
        avg_health_score = sum(
            self._calculate_health_score(api_name) 
            for api_name in self.rate_limiters.keys()
        ) / len(self.rate_limiters) if self.rate_limiters else 0
        
        return {
            'total_apis': len(self.rate_limiters),
            'total_requests': total_requests,
            'successful_requests': total_successful,
            'rate_limited_requests': total_rate_limited,
            'failed_requests': total_failed,
            'success_rate': (total_successful / total_requests * 100) if total_requests > 0 else 0,
            'average_health_score': avg_health_score
        }
    
    async def save_usage_report(self):
        """Save API usage report to file"""
        try:
            reports_dir = Path("data_storage/api_reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = reports_dir / f"api_usage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report_data = self.get_all_api_status()
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"[API_RATE_LIMITER] Usage report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"[API_RATE_LIMITER] Failed to save usage report: {e}")

# Decorator for automatic rate limiting
def rate_limited(api_name: str):
    """Decorator to automatically apply rate limiting to functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get rate limiter from global instance
            # This would need to be injected or accessed globally
            rate_limiter_manager = get_global_rate_limiter()
            if rate_limiter_manager:
                return await rate_limiter_manager.make_request(api_name, func, *args, **kwargs)
            else:
                # Fallback to direct call if no rate limiter
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        return wrapper
    return decorator

# Global rate limiter instance
_global_rate_limiter = None

def initialize_global_rate_limiter(settings):
    """Initialize global rate limiter"""
    global _global_rate_limiter
    _global_rate_limiter = APIRateLimitManager(settings)
    return _global_rate_limiter

def get_global_rate_limiter() -> Optional[APIRateLimitManager]:
    """Get global rate limiter"""
    return _global_rate_limiter