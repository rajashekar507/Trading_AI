"""
System Integration Module for VLR_AI Trading System
Integrates all enhanced components for REAL market data processing
IMPORTANT: Coordinates REAL trading operations - NO mock data, NO simulations
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import all enhanced components
from .api_rate_limiter import initialize_global_rate_limiter, get_global_rate_limiter
from .redis_cache import initialize_global_cache, get_global_cache
from .memory_optimizer import initialize_global_memory_optimizer, get_global_memory_optimizer
from .security_manager import initialize_global_security_manager, get_global_security_manager
from .error_recovery import initialize_global_error_recovery, get_global_error_recovery

logger = logging.getLogger('trading_system.system_integrator')

class SystemIntegrator:
    """System integrator for all enhanced components with REAL data"""
    
    def __init__(self, settings):
        self.settings = settings
        
        # Component status
        self.components = {
            'api_rate_limiter': {'initialized': False, 'instance': None, 'status': 'inactive'},
            'redis_cache': {'initialized': False, 'instance': None, 'status': 'inactive'},
            'memory_optimizer': {'initialized': False, 'instance': None, 'status': 'inactive'},
            'security_manager': {'initialized': False, 'instance': None, 'status': 'inactive'},
            'error_recovery': {'initialized': False, 'instance': None, 'status': 'inactive'}
        }
        
        # Integration statistics
        self.stats = {
            'initialization_time': None,
            'components_initialized': 0,
            'components_failed': 0,
            'total_components': len(self.components),
            'system_health_score': 0,
            'last_health_check': None
        }
        
        logger.info("[SYSTEM_INTEGRATOR] System integrator initialized for REAL trading system")
    
    async def initialize_all_components(self) -> bool:
        """Initialize all enhanced components for REAL data processing"""
        try:
            start_time = datetime.now()
            logger.info("[SYSTEM_INTEGRATOR] Initializing all enhanced components for REAL market data")
            
            # Initialize components in order of dependency
            success_count = 0
            
            # 1. Security Manager (first - needed for credentials)
            if await self._initialize_security_manager():
                success_count += 1
            
            # 2. Error Recovery System
            if await self._initialize_error_recovery():
                success_count += 1
            
            # 3. Memory Optimizer
            if await self._initialize_memory_optimizer():
                success_count += 1
            
            # 4. Redis Cache
            if await self._initialize_redis_cache():
                success_count += 1
            
            # 5. API Rate Limiter (last - depends on others)
            if await self._initialize_api_rate_limiter():
                success_count += 1
            
            # Update statistics
            self.stats['components_initialized'] = success_count
            self.stats['components_failed'] = len(self.components) - success_count
            self.stats['initialization_time'] = (datetime.now() - start_time).total_seconds()
            
            # Calculate system health
            await self._update_system_health()
            
            if success_count == len(self.components):
                logger.info(f"[SYSTEM_INTEGRATOR] All {success_count} components initialized successfully for REAL data")
                return True
            else:
                logger.warning(f"[SYSTEM_INTEGRATOR] {success_count}/{len(self.components)} components initialized")
                return False
                
        except Exception as e:
            logger.error(f"[SYSTEM_INTEGRATOR] Component initialization failed: {e}")
            return False
    
    async def _initialize_security_manager(self) -> bool:
        """Initialize security manager for REAL credentials"""
        try:
            logger.info("[SYSTEM_INTEGRATOR] Initializing security manager for REAL trading credentials")
            
            security_manager = initialize_global_security_manager(self.settings)
            
            if security_manager:
                self.components['security_manager']['initialized'] = True
                self.components['security_manager']['instance'] = security_manager
                self.components['security_manager']['status'] = 'active'
                
                # Load and validate REAL credentials
                credentials = security_manager.load_credentials()
                if credentials:
                    validation_results = security_manager.validate_api_credentials(credentials)
                    logger.info(f"[SYSTEM_INTEGRATOR] Validated {len(validation_results)} credential sets")
                
                logger.info("[SYSTEM_INTEGRATOR] Security manager initialized successfully")
                return True
            else:
                self.components['security_manager']['status'] = 'failed'
                return False
                
        except Exception as e:
            logger.error(f"[SYSTEM_INTEGRATOR] Security manager initialization failed: {e}")
            self.components['security_manager']['status'] = 'failed'
            return False
    
    async def _initialize_error_recovery(self) -> bool:
        """Initialize error recovery system"""
        try:
            logger.info("[SYSTEM_INTEGRATOR] Initializing error recovery system")
            
            error_recovery = initialize_global_error_recovery(self.settings)
            
            if error_recovery:
                self.components['error_recovery']['initialized'] = True
                self.components['error_recovery']['instance'] = error_recovery
                self.components['error_recovery']['status'] = 'active'
                
                logger.info("[SYSTEM_INTEGRATOR] Error recovery system initialized successfully")
                return True
            else:
                self.components['error_recovery']['status'] = 'failed'
                return False
                
        except Exception as e:
            logger.error(f"[SYSTEM_INTEGRATOR] Error recovery initialization failed: {e}")
            self.components['error_recovery']['status'] = 'failed'
            return False
    
    async def _initialize_memory_optimizer(self) -> bool:
        """Initialize memory optimizer for REAL data processing"""
        try:
            logger.info("[SYSTEM_INTEGRATOR] Initializing memory optimizer for REAL data processing")
            
            memory_optimizer = initialize_global_memory_optimizer(self.settings)
            
            if memory_optimizer:
                self.components['memory_optimizer']['initialized'] = True
                self.components['memory_optimizer']['instance'] = memory_optimizer
                self.components['memory_optimizer']['status'] = 'active'
                
                # Get initial memory stats
                memory_stats = memory_optimizer.get_memory_stats()
                logger.info(f"[SYSTEM_INTEGRATOR] Memory optimizer active - Current usage: {memory_stats.get('current', {}).get('memory_mb', 0):.1f}MB")
                return True
            else:
                self.components['memory_optimizer']['status'] = 'failed'
                return False
                
        except Exception as e:
            logger.error(f"[SYSTEM_INTEGRATOR] Memory optimizer initialization failed: {e}")
            self.components['memory_optimizer']['status'] = 'failed'
            return False
    
    async def _initialize_redis_cache(self) -> bool:
        """Initialize Redis cache for REAL market data"""
        try:
            logger.info("[SYSTEM_INTEGRATOR] Initializing Redis cache for REAL market data")
            
            redis_cache = initialize_global_cache(self.settings)
            
            if redis_cache and redis_cache.is_available():
                self.components['redis_cache']['initialized'] = True
                self.components['redis_cache']['instance'] = redis_cache
                self.components['redis_cache']['status'] = 'active'
                
                # Test cache functionality
                test_data = {'test': 'REAL_DATA_TEST', 'timestamp': datetime.now().isoformat()}
                if redis_cache.set_market_data('TEST_INSTRUMENT', test_data):
                    cached_data = redis_cache.get_market_data('TEST_INSTRUMENT')
                    if cached_data:
                        logger.info("[SYSTEM_INTEGRATOR] Redis cache test successful")
                        redis_cache.delete_market_data('TEST_INSTRUMENT')  # Cleanup test data
                
                logger.info("[SYSTEM_INTEGRATOR] Redis cache initialized successfully")
                return True
            else:
                self.components['redis_cache']['status'] = 'failed'
                logger.warning("[SYSTEM_INTEGRATOR] Redis cache not available")
                return False
                
        except Exception as e:
            logger.error(f"[SYSTEM_INTEGRATOR] Redis cache initialization failed: {e}")
            self.components['redis_cache']['status'] = 'failed'
            return False
    
    async def _initialize_api_rate_limiter(self) -> bool:
        """Initialize API rate limiter for REAL APIs"""
        try:
            logger.info("[SYSTEM_INTEGRATOR] Initializing API rate limiter for REAL trading APIs")
            
            rate_limiter = initialize_global_rate_limiter(self.settings)
            
            if rate_limiter:
                self.components['api_rate_limiter']['initialized'] = True
                self.components['api_rate_limiter']['instance'] = rate_limiter
                self.components['api_rate_limiter']['status'] = 'active'
                
                # Get rate limiter status
                status = rate_limiter.get_all_api_status()
                api_count = len(status.get('apis', {}))
                logger.info(f"[SYSTEM_INTEGRATOR] API rate limiter active for {api_count} REAL APIs")
                return True
            else:
                self.components['api_rate_limiter']['status'] = 'failed'
                return False
                
        except Exception as e:
            logger.error(f"[SYSTEM_INTEGRATOR] API rate limiter initialization failed: {e}")
            self.components['api_rate_limiter']['status'] = 'failed'
            return False
    
    async def _update_system_health(self):
        """Update overall system health score"""
        try:
            active_components = sum(1 for comp in self.components.values() if comp['status'] == 'active')
            total_components = len(self.components)
            
            # Base health score
            health_score = (active_components / total_components) * 100
            
            # Adjust based on component importance
            critical_components = ['security_manager', 'error_recovery']
            for comp_name in critical_components:
                if self.components[comp_name]['status'] != 'active':
                    health_score -= 20  # Heavy penalty for critical components
            
            self.stats['system_health_score'] = max(0, min(100, health_score))
            self.stats['last_health_check'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"[SYSTEM_INTEGRATOR] Health update failed: {e}")
            self.stats['system_health_score'] = 0
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            component_status = {}
            
            for name, info in self.components.items():
                component_status[name] = {
                    'initialized': info['initialized'],
                    'status': info['status'],
                    'available': info['instance'] is not None
                }
                
                # Get component-specific status
                if info['instance']:
                    if name == 'redis_cache':
                        component_status[name]['cache_stats'] = info['instance'].get_cache_stats()
                    elif name == 'memory_optimizer':
                        component_status[name]['memory_stats'] = info['instance'].get_memory_stats()
                    elif name == 'security_manager':
                        component_status[name]['security_status'] = info['instance'].get_security_status()
                    elif name == 'api_rate_limiter':
                        component_status[name]['api_status'] = info['instance'].get_all_api_status()
                    elif name == 'error_recovery':
                        component_status[name]['error_stats'] = info['instance'].get_error_statistics()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system_type': 'REAL_TRADING_SYSTEM',
                'components': component_status,
                'statistics': self.stats.copy(),
                'health_summary': {
                    'overall_health': self.stats['system_health_score'],
                    'components_active': sum(1 for comp in self.components.values() if comp['status'] == 'active'),
                    'components_total': len(self.components),
                    'critical_systems_ok': all(
                        self.components[comp]['status'] == 'active' 
                        for comp in ['security_manager', 'error_recovery']
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"[SYSTEM_INTEGRATOR] Error getting system status: {e}")
            return {'error': str(e)}
    
    async def perform_system_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        try:
            logger.info("[SYSTEM_INTEGRATOR] Performing system health check for REAL trading system")
            
            health_results = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'HEALTHY',
                'component_checks': {},
                'recommendations': [],
                'critical_issues': []
            }
            
            # Check each component
            for name, info in self.components.items():
                if info['status'] == 'active' and info['instance']:
                    try:
                        if name == 'redis_cache':
                            cache_stats = info['instance'].get_cache_stats()
                            health_results['component_checks'][name] = {
                                'status': 'HEALTHY' if cache_stats['available'] else 'UNHEALTHY',
                                'details': cache_stats
                            }
                        
                        elif name == 'memory_optimizer':
                            memory_stats = info['instance'].get_memory_stats()
                            current_usage = memory_stats.get('current', {}).get('memory_percent', 0)
                            health_results['component_checks'][name] = {
                                'status': 'HEALTHY' if current_usage < 80 else 'WARNING' if current_usage < 90 else 'CRITICAL',
                                'details': memory_stats
                            }
                            
                            if current_usage > 85:
                                health_results['recommendations'].append(f"High memory usage: {current_usage:.1f}%")
                        
                        elif name == 'security_manager':
                            security_status = info['instance'].get_security_status()
                            violations = security_status.get('violations', [])
                            health_results['component_checks'][name] = {
                                'status': 'HEALTHY' if len(violations) == 0 else 'WARNING',
                                'details': security_status
                            }
                            
                            for violation in violations:
                                if violation.get('severity') == 'HIGH':
                                    health_results['critical_issues'].append(violation['description'])
                        
                        elif name == 'api_rate_limiter':
                            api_status = info['instance'].get_all_api_status()
                            summary = api_status.get('summary', {})
                            success_rate = summary.get('success_rate', 0)
                            health_results['component_checks'][name] = {
                                'status': 'HEALTHY' if success_rate > 95 else 'WARNING' if success_rate > 80 else 'CRITICAL',
                                'details': api_status
                            }
                        
                        elif name == 'error_recovery':
                            error_stats = info['instance'].get_error_statistics()
                            recovery_rate = error_stats.get('recovery_rate', 0)
                            health_results['component_checks'][name] = {
                                'status': 'HEALTHY' if recovery_rate > 80 else 'WARNING',
                                'details': error_stats
                            }
                        
                    except Exception as e:
                        health_results['component_checks'][name] = {
                            'status': 'ERROR',
                            'details': {'error': str(e)}
                        }
                else:
                    health_results['component_checks'][name] = {
                        'status': 'INACTIVE',
                        'details': {'reason': 'Component not initialized or failed'}
                    }
            
            # Determine overall status
            component_statuses = [check['status'] for check in health_results['component_checks'].values()]
            if 'CRITICAL' in component_statuses or len(health_results['critical_issues']) > 0:
                health_results['overall_status'] = 'CRITICAL'
            elif 'WARNING' in component_statuses or 'ERROR' in component_statuses:
                health_results['overall_status'] = 'WARNING'
            
            # Update system health score
            await self._update_system_health()
            
            logger.info(f"[SYSTEM_INTEGRATOR] Health check completed - Status: {health_results['overall_status']}")
            return health_results
            
        except Exception as e:
            logger.error(f"[SYSTEM_INTEGRATOR] Health check failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'ERROR',
                'error': str(e)
            }
    
    def create_system_report(self) -> str:
        """Create comprehensive system report"""
        try:
            status = self.get_system_status()
            
            report = []
            report.append("=" * 80)
            report.append("VLR_AI TRADING SYSTEM - COMPREHENSIVE SYSTEM REPORT")
            report.append("REAL MARKET DATA PROCESSING SYSTEM")
            report.append("=" * 80)
            report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # System health summary
            health = status.get('health_summary', {})
            report.append("SYSTEM HEALTH SUMMARY:")
            report.append(f"  Overall Health: {health.get('overall_health', 0):.1f}%")
            report.append(f"  Active Components: {health.get('components_active', 0)}/{health.get('components_total', 0)}")
            report.append(f"  Critical Systems: {'OK' if health.get('critical_systems_ok') else 'ISSUES DETECTED'}")
            report.append("")
            
            # Component status
            components = status.get('components', {})
            report.append("COMPONENT STATUS:")
            for name, info in components.items():
                status_indicator = "✅" if info['status'] == 'active' else "❌"
                report.append(f"  {status_indicator} {name.upper()}: {info['status'].upper()}")
            report.append("")
            
            # Statistics
            stats = status.get('statistics', {})
            report.append("SYSTEM STATISTICS:")
            report.append(f"  Initialization Time: {stats.get('initialization_time', 0):.2f}s")
            report.append(f"  Components Initialized: {stats.get('components_initialized', 0)}")
            report.append(f"  Components Failed: {stats.get('components_failed', 0)}")
            report.append(f"  Last Health Check: {stats.get('last_health_check', 'Never')}")
            report.append("")
            
            # Component details
            report.append("COMPONENT DETAILS:")
            
            # Redis Cache
            if 'redis_cache' in components and 'cache_stats' in components['redis_cache']:
                cache_stats = components['redis_cache']['cache_stats']
                report.append(f"  REDIS CACHE:")
                report.append(f"    Available: {'YES' if cache_stats.get('available') else 'NO'}")
                report.append(f"    Hit Rate: {cache_stats.get('hit_rate', 0):.1f}%")
                report.append(f"    Total Requests: {cache_stats.get('total_requests', 0)}")
            
            # Memory Optimizer
            if 'memory_optimizer' in components and 'memory_stats' in components['memory_optimizer']:
                memory_stats = components['memory_optimizer']['memory_stats']
                current = memory_stats.get('current', {})
                report.append(f"  MEMORY OPTIMIZER:")
                report.append(f"    Current Usage: {current.get('memory_mb', 0):.1f}MB ({current.get('memory_percent', 0):.1f}%)")
                report.append(f"    Objects Tracked: {memory_stats.get('tracking', {}).get('objects_tracked', 0)}")
            
            # Security Manager
            if 'security_manager' in components and 'security_status' in components['security_manager']:
                security_status = components['security_manager']['security_status']
                report.append(f"  SECURITY MANAGER:")
                report.append(f"    Security Score: {security_status.get('security_score', 0)}/100")
                report.append(f"    Violations: {security_status.get('violation_count', 0)}")
                report.append(f"    Encryption: {'ENABLED' if security_status.get('encryption_available') else 'DISABLED'}")
            
            report.append("")
            report.append("=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"[SYSTEM_INTEGRATOR] Error creating system report: {e}")
            return f"Error creating system report: {str(e)}"
    
    async def shutdown_all_components(self):
        """Shutdown all components gracefully"""
        try:
            logger.info("[SYSTEM_INTEGRATOR] Shutting down all components")
            
            # Shutdown in reverse order
            shutdown_order = ['api_rate_limiter', 'redis_cache', 'memory_optimizer', 'error_recovery', 'security_manager']
            
            for component_name in shutdown_order:
                if self.components[component_name]['instance']:
                    try:
                        instance = self.components[component_name]['instance']
                        
                        if hasattr(instance, 'close'):
                            instance.close()
                        elif hasattr(instance, 'stop'):
                            instance.stop()
                        elif hasattr(instance, 'stop_monitoring'):
                            instance.stop_monitoring()
                        
                        self.components[component_name]['status'] = 'shutdown'
                        logger.info(f"[SYSTEM_INTEGRATOR] {component_name} shutdown completed")
                        
                    except Exception as e:
                        logger.error(f"[SYSTEM_INTEGRATOR] Error shutting down {component_name}: {e}")
            
            logger.info("[SYSTEM_INTEGRATOR] All components shutdown completed")
            
        except Exception as e:
            logger.error(f"[SYSTEM_INTEGRATOR] Error during shutdown: {e}")

# Global system integrator instance
_global_system_integrator = None

def initialize_global_system_integrator(settings):
    """Initialize global system integrator for REAL trading system"""
    global _global_system_integrator
    _global_system_integrator = SystemIntegrator(settings)
    logger.info("[SYSTEM_INTEGRATOR] Global system integrator initialized for REAL trading system")
    return _global_system_integrator

def get_global_system_integrator() -> Optional[SystemIntegrator]:
    """Get global system integrator"""
    return _global_system_integrator

async def initialize_complete_system(settings):
    """Initialize complete enhanced trading system"""
    try:
        logger.info("[SYSTEM_INTEGRATOR] Initializing complete enhanced trading system for REAL market data")
        
        # Initialize system integrator
        integrator = initialize_global_system_integrator(settings)
        
        # Initialize all components
        success = await integrator.initialize_all_components()
        
        if success:
            logger.info("[SYSTEM_INTEGRATOR] Complete enhanced trading system initialized successfully")
            
            # Perform initial health check
            health_check = await integrator.perform_system_health_check()
            logger.info(f"[SYSTEM_INTEGRATOR] Initial health check: {health_check['overall_status']}")
            
            return integrator
        else:
            logger.error("[SYSTEM_INTEGRATOR] Failed to initialize complete system")
            return None
            
    except Exception as e:
        logger.error(f"[SYSTEM_INTEGRATOR] Complete system initialization failed: {e}")
        return None