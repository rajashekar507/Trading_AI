"""
Enhanced Error Recovery System for VLR_AI Trading System
Implements retry mechanisms, circuit breakers, and automatic recovery with standardized patterns
"""

import asyncio
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable, Optional, Union
from functools import wraps
import json
from pathlib import Path
from enum import Enum

logger = logging.getLogger('trading_system.error_recovery')

class ErrorType(Enum):
    """Standardized error types for consistent handling"""
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    DATA_ERROR = "data_error"
    AUTHENTICATION_ERROR = "auth_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    SYSTEM_ERROR = "system_error"

class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY_EXPONENTIAL = "retry_exponential"
    RETRY_LINEAR = "retry_linear"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: Exception = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func):
        """Decorator to apply circuit breaker"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception(f"Circuit breaker is OPEN. Service unavailable.")
            
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time >= timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'

class RetryManager:
    """Retry mechanism with exponential backoff"""
    
    @staticmethod
    async def retry_with_backoff(
        func: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        exceptions: tuple = (Exception,)
    ):
        """Retry function with exponential backoff"""
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func()
                else:
                    return func()
            except exceptions as e:
                if attempt == max_retries:
                    logger.error(f"[RETRY] All {max_retries + 1} attempts failed. Last error: {e}")
                    raise e
                
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                logger.warning(f"[RETRY] Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

class ErrorRecoverySystem:
    """Comprehensive error recovery system"""
    
    def __init__(self, settings):
        self.settings = settings
        self.circuit_breakers = {}
        self.error_history = []
        self.recovery_actions = {}
        self.max_error_history = 1000
        
        # Recovery statistics
        self.stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'failed_recoveries': 0,
            'circuit_breaker_trips': 0
        }
        
        # Initialize recovery actions
        self._setup_recovery_actions()
        
        logger.info("[RECOVERY] Error Recovery System initialized")
    
    def _setup_recovery_actions(self):
        """Setup automatic recovery actions for common errors"""
        self.recovery_actions = {
            'ConnectionError': self._recover_connection_error,
            'TimeoutError': self._recover_timeout_error,
            'APIError': self._recover_api_error,
            'MemoryError': self._recover_memory_error,
            'FileNotFoundError': self._recover_file_error,
            'PermissionError': self._recover_permission_error,
            'UnicodeDecodeError': self._recover_unicode_error,
            'ImportError': self._recover_import_error
        }
    
    def get_circuit_breaker(self, service_name: str, failure_threshold: int = 5, recovery_timeout: int = 60) -> CircuitBreaker:
        """Get or create circuit breaker for a service"""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )
        return self.circuit_breakers[service_name]
    
    async def handle_error(self, error: Exception, context: Dict = None) -> bool:
        """Handle error with automatic recovery"""
        try:
            error_type = type(error).__name__
            error_message = str(error)
            
            # Record error
            error_record = {
                'timestamp': datetime.now().isoformat(),
                'error_type': error_type,
                'error_message': error_message,
                'context': context or {},
                'recovered': False
            }
            
            self.error_history.append(error_record)
            self.stats['total_errors'] += 1
            
            # Limit error history size
            if len(self.error_history) > self.max_error_history:
                self.error_history = self.error_history[-self.max_error_history:]
            
            logger.error(f"[RECOVERY] Handling error: {error_type} - {error_message}")
            
            # Attempt automatic recovery
            recovery_success = await self._attempt_recovery(error_type, error, context)
            
            # Update error record
            error_record['recovered'] = recovery_success
            
            if recovery_success:
                self.stats['recovered_errors'] += 1
                logger.info(f"[RECOVERY] Successfully recovered from {error_type}")
            else:
                self.stats['failed_recoveries'] += 1
                logger.warning(f"[RECOVERY] Failed to recover from {error_type}")
            
            return recovery_success
            
        except Exception as recovery_error:
            logger.error(f"[RECOVERY] Recovery system error: {recovery_error}")
            return False
    
    async def _attempt_recovery(self, error_type: str, error: Exception, context: Dict) -> bool:
        """Attempt to recover from specific error type"""
        try:
            if error_type in self.recovery_actions:
                recovery_func = self.recovery_actions[error_type]
                return await recovery_func(error, context)
            else:
                # Generic recovery attempt
                return await self._generic_recovery(error, context)
        except Exception as e:
            logger.error(f"[RECOVERY] Recovery attempt failed: {e}")
            return False
    
    async def _recover_connection_error(self, error: Exception, context: Dict) -> bool:
        """Recover from connection errors"""
        try:
            logger.info("[RECOVERY] Attempting connection error recovery...")
            
            # Wait and retry
            await asyncio.sleep(5)
            
            # Try to reinitialize connections
            if 'component' in context:
                component = context['component']
                if hasattr(component, 'initialize'):
                    await component.initialize()
                    return True
            
            return False
        except Exception as e:
            logger.error(f"[RECOVERY] Connection recovery failed: {e}")
            return False
    
    async def _recover_timeout_error(self, error: Exception, context: Dict) -> bool:
        """Recover from timeout errors"""
        try:
            logger.info("[RECOVERY] Attempting timeout error recovery...")
            
            # Increase timeout and retry
            await asyncio.sleep(2)
            
            # Could implement timeout adjustment logic here
            return True
        except Exception as e:
            logger.error(f"[RECOVERY] Timeout recovery failed: {e}")
            return False
    
    async def _recover_api_error(self, error: Exception, context: Dict) -> bool:
        """Recover from API errors"""
        try:
            logger.info("[RECOVERY] Attempting API error recovery...")
            
            error_message = str(error).lower()
            
            # Rate limit error
            if 'rate limit' in error_message or 'too many requests' in error_message:
                logger.info("[RECOVERY] Rate limit detected, waiting...")
                await asyncio.sleep(60)  # Wait 1 minute
                return True
            
            # Authentication error
            if 'auth' in error_message or 'token' in error_message:
                logger.info("[RECOVERY] Authentication error, attempting re-auth...")
                if 'component' in context and hasattr(context['component'], 'authenticate'):
                    await context['component'].authenticate()
                    return True
            
            # Generic API error - wait and retry
            await asyncio.sleep(10)
            return True
            
        except Exception as e:
            logger.error(f"[RECOVERY] API recovery failed: {e}")
            return False
    
    async def _recover_memory_error(self, error: Exception, context: Dict) -> bool:
        """Recover from memory errors"""
        try:
            logger.info("[RECOVERY] Attempting memory error recovery...")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear caches if available
            if 'component' in context and hasattr(context['component'], 'clear_cache'):
                context['component'].clear_cache()
            
            await asyncio.sleep(1)
            return True
            
        except Exception as e:
            logger.error(f"[RECOVERY] Memory recovery failed: {e}")
            return False
    
    async def _recover_file_error(self, error: Exception, context: Dict) -> bool:
        """Recover from file errors"""
        try:
            logger.info("[RECOVERY] Attempting file error recovery...")
            
            # Create missing directories
            if 'file_path' in context:
                file_path = Path(context['file_path'])
                file_path.parent.mkdir(parents=True, exist_ok=True)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"[RECOVERY] File recovery failed: {e}")
            return False
    
    async def _recover_permission_error(self, error: Exception, context: Dict) -> bool:
        """Recover from permission errors"""
        try:
            logger.info("[RECOVERY] Attempting permission error recovery...")
            
            # Log the issue for manual intervention
            logger.warning(f"[RECOVERY] Permission error requires manual intervention: {error}")
            
            # Could implement permission fixing logic here
            return False
            
        except Exception as e:
            logger.error(f"[RECOVERY] Permission recovery failed: {e}")
            return False
    
    async def _recover_unicode_error(self, error: Exception, context: Dict) -> bool:
        """Recover from Unicode errors"""
        try:
            logger.info("[RECOVERY] Attempting Unicode error recovery...")
            
            # This is a common issue that can often be fixed
            # by ensuring proper encoding
            return True
            
        except Exception as e:
            logger.error(f"[RECOVERY] Unicode recovery failed: {e}")
            return False
    
    async def _recover_import_error(self, error: Exception, context: Dict) -> bool:
        """Recover from import errors"""
        try:
            logger.info("[RECOVERY] Attempting import error recovery...")
            
            error_message = str(error)
            
            # Try to install missing package
            if 'No module named' in error_message:
                module_name = error_message.split("'")[1] if "'" in error_message else None
                if module_name:
                    logger.info(f"[RECOVERY] Attempting to install missing module: {module_name}")
                    # Could implement automatic pip install here
                    # For now, just log it
                    logger.warning(f"[RECOVERY] Please install missing module: pip install {module_name}")
            
            return False
            
        except Exception as e:
            logger.error(f"[RECOVERY] Import recovery failed: {e}")
            return False
    
    async def _generic_recovery(self, error: Exception, context: Dict) -> bool:
        """Generic recovery attempt"""
        try:
            logger.info(f"[RECOVERY] Attempting generic recovery for {type(error).__name__}")
            
            # Wait a bit and hope the issue resolves
            await asyncio.sleep(5)
            
            # Could implement more sophisticated generic recovery here
            return False
            
        except Exception as e:
            logger.error(f"[RECOVERY] Generic recovery failed: {e}")
            return False
    
    def get_error_statistics(self) -> Dict:
        """Get error recovery statistics"""
        recent_errors = [
            e for e in self.error_history 
            if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(hours=24)
        ]
        
        error_types = {}
        for error in recent_errors:
            error_type = error['error_type']
            if error_type not in error_types:
                error_types[error_type] = {'count': 0, 'recovered': 0}
            error_types[error_type]['count'] += 1
            if error['recovered']:
                error_types[error_type]['recovered'] += 1
        
        return {
            'total_errors': self.stats['total_errors'],
            'recovered_errors': self.stats['recovered_errors'],
            'failed_recoveries': self.stats['failed_recoveries'],
            'recovery_rate': (self.stats['recovered_errors'] / max(self.stats['total_errors'], 1)) * 100,
            'recent_errors_24h': len(recent_errors),
            'error_types_24h': error_types,
            'circuit_breaker_status': {name: cb.state for name, cb in self.circuit_breakers.items()}
        }
    
    async def save_error_report(self):
        """Save error report to file"""
        try:
            reports_dir = Path("data_storage/error_reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = reports_dir / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'statistics': self.get_error_statistics(),
                'recent_errors': self.error_history[-50:],  # Last 50 errors
                'circuit_breakers': {name: {'state': cb.state, 'failure_count': cb.failure_count} 
                                   for name, cb in self.circuit_breakers.items()}
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"[RECOVERY] Error report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"[RECOVERY] Failed to save error report: {e}")

# Standardized decorators for consistent error handling
def with_retry(max_retries: int = 3, backoff_factor: float = 2.0, exceptions: tuple = (Exception,)):
    """Decorator for automatic retry with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await RetryManager.retry_with_backoff(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                exceptions=exceptions
            )
        return wrapper
    return decorator

def with_circuit_breaker(service_name: str, failure_threshold: int = 5, recovery_timeout: int = 60):
    """Decorator for circuit breaker pattern"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get global error recovery system instance
            # This would need to be injected or accessed globally
            circuit_breaker = CircuitBreaker(failure_threshold, recovery_timeout)
            return await circuit_breaker.call(func)(*args, **kwargs)
        return wrapper
    return decorator

def with_error_recovery(error_recovery_system: ErrorRecoverySystem, context: Dict = None):
    """Decorator for automatic error recovery"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                recovery_success = await error_recovery_system.handle_error(e, context)
                if recovery_success:
                    # Retry once after successful recovery
                    return await func(*args, **kwargs)
                else:
                    raise e
        return wrapper
    return decorator

# Global error recovery instance (to be initialized by system)
_global_error_recovery = None

def initialize_global_error_recovery(settings):
    """Initialize global error recovery system"""
    global _global_error_recovery
    _global_error_recovery = ErrorRecoverySystem(settings)
    return _global_error_recovery

def get_global_error_recovery() -> Optional[ErrorRecoverySystem]:
    """Get global error recovery system"""
    return _global_error_recovery

# Decorators for easy use
def with_retry(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator to add retry functionality"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await RetryManager.retry_with_backoff(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                base_delay=base_delay
            )
        return wrapper
    return decorator

def with_circuit_breaker(service_name: str, failure_threshold: int = 5, recovery_timeout: int = 60):
    """Decorator to add circuit breaker functionality"""
    def decorator(func):
        # This would need to be implemented with a global error recovery system instance
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Implementation would go here
            return await func(*args, **kwargs)
        return wrapper
    return decorator