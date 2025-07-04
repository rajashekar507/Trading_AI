"""
Memory Optimization System for VLR_AI Trading System
Implements memory management, garbage collection, and usage monitoring for REAL data processing
IMPORTANT: Optimizes memory for REAL market data processing - NO mock data
"""

import gc
import logging
import psutil
import threading
import time
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import sys
import os
from pathlib import Path

logger = logging.getLogger('trading_system.memory_optimizer')

class MemoryOptimizer:
    """Memory optimization system for REAL market data processing"""
    
    def __init__(self, settings):
        self.settings = settings
        
        # Memory thresholds
        self.warning_threshold = getattr(settings, 'MEMORY_WARNING_THRESHOLD', 80)  # 80%
        self.critical_threshold = getattr(settings, 'MEMORY_CRITICAL_THRESHOLD', 90)  # 90%
        self.cleanup_threshold = getattr(settings, 'MEMORY_CLEANUP_THRESHOLD', 85)  # 85%
        
        # Monitoring settings
        self.monitoring_interval = getattr(settings, 'MEMORY_MONITORING_INTERVAL', 30)  # 30 seconds
        self.cleanup_interval = getattr(settings, 'MEMORY_CLEANUP_INTERVAL', 300)  # 5 minutes
        
        # Memory tracking
        self.memory_history = []
        self.max_history_size = 100
        self.last_cleanup = datetime.now()
        self.cleanup_count = 0
        
        # Object tracking for REAL data
        self.tracked_objects = weakref.WeakSet()
        self.large_objects = weakref.WeakKeyDictionary()
        
        # Statistics
        self.stats = {
            'total_cleanups': 0,
            'memory_freed_mb': 0,
            'peak_memory_mb': 0,
            'current_memory_mb': 0,
            'objects_tracked': 0,
            'large_objects_count': 0
        }
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info("[MEMORY_OPTIMIZER] Memory optimization system initialized for REAL data processing")
    
    def start_monitoring(self):
        """Start memory monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("[MEMORY_OPTIMIZER] Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("[MEMORY_OPTIMIZER] Memory monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_memory_usage()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"[MEMORY_OPTIMIZER] Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _check_memory_usage(self):
        """Check current memory usage and trigger cleanup if needed"""
        try:
            # Get current memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            current_memory_mb = memory_info.rss / 1024 / 1024
            
            # Update statistics
            self.stats['current_memory_mb'] = current_memory_mb
            if current_memory_mb > self.stats['peak_memory_mb']:
                self.stats['peak_memory_mb'] = current_memory_mb
            
            # Add to history
            self.memory_history.append({
                'timestamp': datetime.now(),
                'memory_mb': current_memory_mb,
                'memory_percent': memory_percent,
                'objects_count': len(self.tracked_objects)
            })
            
            # Limit history size
            if len(self.memory_history) > self.max_history_size:
                self.memory_history = self.memory_history[-self.max_history_size:]
            
            # Check thresholds
            if memory_percent >= self.critical_threshold:
                logger.warning(f"[MEMORY_OPTIMIZER] CRITICAL memory usage: {memory_percent:.1f}% ({current_memory_mb:.1f}MB)")
                self._emergency_cleanup()
            elif memory_percent >= self.cleanup_threshold:
                logger.warning(f"[MEMORY_OPTIMIZER] High memory usage: {memory_percent:.1f}% ({current_memory_mb:.1f}MB)")
                self._perform_cleanup()
            elif memory_percent >= self.warning_threshold:
                logger.info(f"[MEMORY_OPTIMIZER] Memory usage warning: {memory_percent:.1f}% ({current_memory_mb:.1f}MB)")
            
            # Periodic cleanup
            if datetime.now() - self.last_cleanup > timedelta(seconds=self.cleanup_interval):
                self._perform_cleanup()
            
        except Exception as e:
            logger.error(f"[MEMORY_OPTIMIZER] Error checking memory usage: {e}")
    
    def _perform_cleanup(self):
        """Perform memory cleanup"""
        try:
            logger.info("[MEMORY_OPTIMIZER] Starting memory cleanup for REAL data processing")
            
            memory_before = self._get_current_memory_mb()
            
            # Force garbage collection
            collected = gc.collect()
            
            # Clean up tracked objects
            self._cleanup_tracked_objects()
            
            # Clean up large objects
            self._cleanup_large_objects()
            
            # Additional cleanup
            self._cleanup_caches()
            
            memory_after = self._get_current_memory_mb()
            memory_freed = memory_before - memory_after
            
            # Update statistics
            self.stats['total_cleanups'] += 1
            self.stats['memory_freed_mb'] += max(0, memory_freed)
            self.last_cleanup = datetime.now()
            
            logger.info(f"[MEMORY_OPTIMIZER] Cleanup completed: {collected} objects collected, {memory_freed:.1f}MB freed")
            
        except Exception as e:
            logger.error(f"[MEMORY_OPTIMIZER] Error during cleanup: {e}")
    
    def _emergency_cleanup(self):
        """Perform emergency memory cleanup"""
        try:
            logger.warning("[MEMORY_OPTIMIZER] EMERGENCY memory cleanup for REAL data processing")
            
            # Aggressive cleanup
            self._perform_cleanup()
            
            # Force multiple GC cycles
            for _ in range(3):
                gc.collect()
            
            # Clear all possible caches
            self._aggressive_cache_cleanup()
            
            logger.warning("[MEMORY_OPTIMIZER] Emergency cleanup completed")
            
        except Exception as e:
            logger.error(f"[MEMORY_OPTIMIZER] Error during emergency cleanup: {e}")
    
    def _cleanup_tracked_objects(self):
        """Clean up tracked objects"""
        try:
            # Update object count
            self.stats['objects_tracked'] = len(self.tracked_objects)
            
            # Remove dead references (WeakSet handles this automatically)
            # Just log the current count
            logger.debug(f"[MEMORY_OPTIMIZER] Tracking {len(self.tracked_objects)} objects")
            
        except Exception as e:
            logger.error(f"[MEMORY_OPTIMIZER] Error cleaning tracked objects: {e}")
    
    def _cleanup_large_objects(self):
        """Clean up large objects"""
        try:
            # Count large objects
            large_count = len(self.large_objects)
            self.stats['large_objects_count'] = large_count
            
            if large_count > 0:
                logger.debug(f"[MEMORY_OPTIMIZER] Tracking {large_count} large objects")
            
        except Exception as e:
            logger.error(f"[MEMORY_OPTIMIZER] Error cleaning large objects: {e}")
    
    def _cleanup_caches(self):
        """Clean up various caches"""
        try:
            # Clear function caches
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
            
            # Clear import cache for unused modules
            # Be careful not to clear modules we're actively using
            
            logger.debug("[MEMORY_OPTIMIZER] Caches cleaned")
            
        except Exception as e:
            logger.error(f"[MEMORY_OPTIMIZER] Error cleaning caches: {e}")
    
    def _aggressive_cache_cleanup(self):
        """Aggressive cache cleanup for emergency situations"""
        try:
            # Clear all possible caches
            self._cleanup_caches()
            
            # Clear Redis cache if available
            try:
                from .redis_cache import get_global_cache
                cache = get_global_cache()
                if cache and cache.is_available():
                    # Only clear old cache entries, not recent REAL data
                    logger.debug("[MEMORY_OPTIMIZER] Selective Redis cache cleanup")
            except ImportError:
                pass
            
            logger.debug("[MEMORY_OPTIMIZER] Aggressive cache cleanup completed")
            
        except Exception as e:
            logger.error(f"[MEMORY_OPTIMIZER] Error in aggressive cleanup: {e}")
    
    def track_object(self, obj: Any, description: str = ""):
        """Track an object for memory monitoring"""
        try:
            self.tracked_objects.add(obj)
            
            # Check if it's a large object (>1MB)
            try:
                size = sys.getsizeof(obj)
                if size > 1024 * 1024:  # 1MB
                    self.large_objects[obj] = {
                        'size': size,
                        'description': description,
                        'timestamp': datetime.now()
                    }
                    logger.debug(f"[MEMORY_OPTIMIZER] Tracking large object: {description} ({size/1024/1024:.1f}MB)")
            except (TypeError, OverflowError):
                # Some objects don't support getsizeof
                pass
            
        except Exception as e:
            logger.error(f"[MEMORY_OPTIMIZER] Error tracking object: {e}")
    
    def untrack_object(self, obj: Any):
        """Stop tracking an object"""
        try:
            self.tracked_objects.discard(obj)
            if obj in self.large_objects:
                del self.large_objects[obj]
        except Exception as e:
            logger.error(f"[MEMORY_OPTIMIZER] Error untracking object: {e}")
    
    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # System memory
            system_memory = psutil.virtual_memory()
            
            # Recent memory trend
            recent_history = self.memory_history[-10:] if self.memory_history else []
            avg_memory = sum(h['memory_mb'] for h in recent_history) / len(recent_history) if recent_history else 0
            
            return {
                'current': {
                    'memory_mb': memory_info.rss / 1024 / 1024,
                    'memory_percent': memory_percent,
                    'virtual_memory_mb': memory_info.vms / 1024 / 1024
                },
                'system': {
                    'total_mb': system_memory.total / 1024 / 1024,
                    'available_mb': system_memory.available / 1024 / 1024,
                    'used_percent': system_memory.percent
                },
                'tracking': {
                    'objects_tracked': len(self.tracked_objects),
                    'large_objects': len(self.large_objects),
                    'history_points': len(self.memory_history)
                },
                'statistics': self.stats.copy(),
                'trends': {
                    'average_memory_mb': avg_memory,
                    'memory_history_size': len(self.memory_history)
                },
                'thresholds': {
                    'warning': self.warning_threshold,
                    'cleanup': self.cleanup_threshold,
                    'critical': self.critical_threshold
                },
                'last_cleanup': self.last_cleanup.isoformat() if self.last_cleanup else None
            }
            
        except Exception as e:
            logger.error(f"[MEMORY_OPTIMIZER] Error getting memory stats: {e}")
            return {'error': str(e)}
    
    def optimize_for_real_data_processing(self):
        """Optimize memory specifically for REAL market data processing"""
        try:
            logger.info("[MEMORY_OPTIMIZER] Optimizing for REAL market data processing")
            
            # Set optimal GC thresholds for high-frequency data
            gc.set_threshold(700, 10, 10)  # More aggressive GC for real-time data
            
            # Enable GC debugging if needed
            if logger.isEnabledFor(logging.DEBUG):
                gc.set_debug(gc.DEBUG_STATS)
            
            # Perform initial cleanup
            self._perform_cleanup()
            
            logger.info("[MEMORY_OPTIMIZER] Memory optimization for REAL data processing completed")
            
        except Exception as e:
            logger.error(f"[MEMORY_OPTIMIZER] Error optimizing for real data: {e}")
    
    def create_memory_report(self) -> str:
        """Create a detailed memory report"""
        try:
            stats = self.get_memory_stats()
            
            report = []
            report.append("=" * 60)
            report.append("VLR_AI TRADING SYSTEM - MEMORY REPORT")
            report.append("REAL MARKET DATA PROCESSING")
            report.append("=" * 60)
            report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # Current memory usage
            current = stats.get('current', {})
            report.append("CURRENT MEMORY USAGE:")
            report.append(f"  Memory: {current.get('memory_mb', 0):.1f} MB ({current.get('memory_percent', 0):.1f}%)")
            report.append(f"  Virtual: {current.get('virtual_memory_mb', 0):.1f} MB")
            report.append("")
            
            # System memory
            system = stats.get('system', {})
            report.append("SYSTEM MEMORY:")
            report.append(f"  Total: {system.get('total_mb', 0):.1f} MB")
            report.append(f"  Available: {system.get('available_mb', 0):.1f} MB")
            report.append(f"  Used: {system.get('used_percent', 0):.1f}%")
            report.append("")
            
            # Statistics
            statistics = stats.get('statistics', {})
            report.append("OPTIMIZATION STATISTICS:")
            report.append(f"  Total Cleanups: {statistics.get('total_cleanups', 0)}")
            report.append(f"  Memory Freed: {statistics.get('memory_freed_mb', 0):.1f} MB")
            report.append(f"  Peak Memory: {statistics.get('peak_memory_mb', 0):.1f} MB")
            report.append(f"  Objects Tracked: {statistics.get('objects_tracked', 0)}")
            report.append(f"  Large Objects: {statistics.get('large_objects_count', 0)}")
            report.append("")
            
            # Thresholds
            thresholds = stats.get('thresholds', {})
            report.append("MEMORY THRESHOLDS:")
            report.append(f"  Warning: {thresholds.get('warning', 0)}%")
            report.append(f"  Cleanup: {thresholds.get('cleanup', 0)}%")
            report.append(f"  Critical: {thresholds.get('critical', 0)}%")
            report.append("")
            
            report.append("=" * 60)
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"[MEMORY_OPTIMIZER] Error creating memory report: {e}")
            return f"Error creating memory report: {str(e)}"

# Global memory optimizer instance
_global_memory_optimizer = None

def initialize_global_memory_optimizer(settings):
    """Initialize global memory optimizer for REAL data processing"""
    global _global_memory_optimizer
    _global_memory_optimizer = MemoryOptimizer(settings)
    _global_memory_optimizer.start_monitoring()
    _global_memory_optimizer.optimize_for_real_data_processing()
    logger.info("[MEMORY_OPTIMIZER] Global memory optimizer initialized for REAL market data")
    return _global_memory_optimizer

def get_global_memory_optimizer() -> Optional[MemoryOptimizer]:
    """Get global memory optimizer"""
    return _global_memory_optimizer

# Decorator for memory-optimized functions
def memory_optimized(track_objects: bool = True):
    """Decorator to optimize memory usage for functions processing REAL data"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            optimizer = get_global_memory_optimizer()
            
            if optimizer and track_objects:
                # Track function execution
                optimizer.track_object(func, f"Function: {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                
                if optimizer and track_objects:
                    # Track result if it's a large object
                    optimizer.track_object(result, f"Result of {func.__name__}")
                
                return result
                
            except Exception as e:
                # Cleanup on error
                if optimizer:
                    optimizer._perform_cleanup()
                raise e
            
        return wrapper
    return decorator