"""
Comprehensive System Monitor for Google Sheets Integration
Monitors performance, data quality, and system health
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import time
from pathlib import Path
import pytz
import psutil
import statistics

from utils.email_alerts import EmailAlertSystem

logger = logging.getLogger('trading_system.system_monitor')

class SystemMonitor:
    """Comprehensive system monitoring for Google Sheets integration"""
    
    def __init__(self, settings, sheets_service=None):
        self.settings = settings
        self.sheets_service = sheets_service
        self.email_alerts = EmailAlertSystem(settings)
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Monitoring state
        self.monitoring_active = False
        self.performance_metrics = {}
        self.error_tracking = {}
        self.data_quality_issues = []
        self.api_status = {}
        
        # Performance thresholds
        self.thresholds = {
            'api_response_time': 5000,  # 5 seconds
            'sheet_update_time': 30000,  # 30 seconds
            'error_rate': 10,  # 10%
            'data_freshness': 15,  # 15 minutes
            'memory_usage': 80,  # 80%
            'cpu_usage': 90  # 90%
        }
        
        # Monitoring intervals
        self.monitor_intervals = {
            'performance': 300,  # 5 minutes
            'health_check': 300,  # 5 minutes
            'data_quality': 600,  # 10 minutes
            'system_resources': 180,  # 3 minutes
            'daily_summary': 86400  # 24 hours
        }
        
        # Data tracking
        self.performance_history = []
        self.error_history = []
        self.last_checks = {}
        
        logger.info("System Monitor initialized")
    
    async def start_monitoring(self):
        """Start comprehensive system monitoring"""
        try:
            self.monitoring_active = True
            logger.info("Starting comprehensive system monitoring...")
            
            # Start monitoring tasks
            monitoring_tasks = [
                asyncio.create_task(self._performance_monitor()),
                asyncio.create_task(self._health_check_monitor()),
                asyncio.create_task(self._data_quality_monitor()),
                asyncio.create_task(self._system_resource_monitor()),
                asyncio.create_task(self._daily_summary_monitor())
            ]
            
            # Wait for all monitoring tasks
            await asyncio.gather(*monitoring_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error in system monitoring: {e}")
        finally:
            self.monitoring_active = False
    
    async def _performance_monitor(self):
        """Monitor system performance metrics"""
        while self.monitoring_active:
            try:
                # Collect performance metrics
                metrics = await self._collect_performance_metrics()
                
                # Check thresholds
                violations = self._check_performance_thresholds(metrics)
                
                if violations:
                    await self.email_alerts.send_performance_alert({
                        'api_response_time': metrics.get('avg_api_response_time', 0),
                        'sheet_update_time': metrics.get('avg_sheet_update_time', 0),
                        'error_rate': metrics.get('error_rate', 0),
                        'data_freshness': metrics.get('data_freshness_minutes', 0),
                        'threshold_violations': ', '.join(violations)
                    })
                
                # Store metrics
                self.performance_history.append({
                    'timestamp': datetime.now(self.ist),
                    'metrics': metrics
                })
                
                # Keep only last 24 hours of data
                cutoff_time = datetime.now(self.ist) - timedelta(hours=24)
                self.performance_history = [
                    entry for entry in self.performance_history 
                    if entry['timestamp'] > cutoff_time
                ]
                
                # Log to sheets
                if self.sheets_service:
                    await self.sheets_service.sheets_manager._log_system_health(
                        "Performance Monitor",
                        "Online",
                        f"Avg API time: {metrics.get('avg_api_response_time', 0)}ms, Error rate: {metrics.get('error_rate', 0)}%",
                        "Warning" if violations else "Info"
                    )
                
                await asyncio.sleep(self.monitor_intervals['performance'])
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _health_check_monitor(self):
        """Monitor system health and component status"""
        while self.monitoring_active:
            try:
                # Check Google Sheets health
                if self.sheets_service:
                    sheets_health = await self.sheets_service.sheets_service.get_sheet_health_status()
                    
                    if sheets_health['status'] == 'Critical':
                        await self.email_alerts.send_system_health_alert(
                            "Google Sheets Integration",
                            "Critical",
                            f"Error count: {sheets_health['error_count']}, Response time: {sheets_health.get('api_response_time', 0)}ms"
                        )
                    elif sheets_health['status'] == 'Warning':
                        await self.email_alerts.send_system_health_alert(
                            "Google Sheets Integration",
                            "Warning",
                            f"Performance degraded - Error count: {sheets_health['error_count']}"
                        )
                
                # Check API status
                api_health = await self._check_api_health()
                for api_name, status in api_health.items():
                    if status['status'] == 'Failed':
                        failure_duration = status.get('failure_duration', 0)
                        if failure_duration > 15:  # Alert after 15 minutes
                            await self.email_alerts.send_api_failure_alert(
                                api_name,
                                failure_duration,
                                status.get('error', 'Unknown error')
                            )
                
                self.last_checks['health_check'] = datetime.now(self.ist)
                await asyncio.sleep(self.monitor_intervals['health_check'])
                
            except Exception as e:
                logger.error(f"Error in health check monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _data_quality_monitor(self):
        """Monitor data quality and completeness"""
        while self.monitoring_active:
            try:
                # Check data quality
                quality_issues = await self._check_data_quality()
                
                if quality_issues:
                    await self.email_alerts.send_data_quality_alert(quality_issues)
                    
                    # Log issues to sheets
                    if self.sheets_service:
                        for issue in quality_issues:
                            await self.sheets_service.sheets_manager._log_system_health(
                                f"Data Quality - {issue['component']}",
                                "Warning",
                                issue['description'],
                                "Warning"
                            )
                
                self.data_quality_issues = quality_issues
                self.last_checks['data_quality'] = datetime.now(self.ist)
                
                await asyncio.sleep(self.monitor_intervals['data_quality'])
                
            except Exception as e:
                logger.error(f"Error in data quality monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _system_resource_monitor(self):
        """Monitor system resources (CPU, Memory, Disk)"""
        while self.monitoring_active:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Check thresholds
                alerts = []
                if cpu_percent > self.thresholds['cpu_usage']:
                    alerts.append(f"High CPU usage: {cpu_percent}%")
                
                if memory.percent > self.thresholds['memory_usage']:
                    alerts.append(f"High memory usage: {memory.percent}%")
                
                if disk.percent > 90:  # Disk space threshold
                    alerts.append(f"Low disk space: {disk.percent}% used")
                
                # Send alerts if needed
                if alerts:
                    await self.email_alerts.send_critical_alert(
                        "System Resource Alert",
                        f"Resource usage exceeded thresholds:\n" + "\n".join(alerts),
                        "RESOURCES"
                    )
                
                # Log to sheets
                if self.sheets_service:
                    await self.sheets_service.sheets_manager._log_system_health(
                        "System Resources",
                        "Warning" if alerts else "Online",
                        f"CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk.percent}%",
                        "Warning" if alerts else "Info"
                    )
                
                await asyncio.sleep(self.monitor_intervals['system_resources'])
                
            except Exception as e:
                logger.error(f"Error in system resource monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _daily_summary_monitor(self):
        """Send daily summary reports"""
        while self.monitoring_active:
            try:
                # Wait until end of trading day (after 4 PM IST)
                now = datetime.now(self.ist)
                if now.hour >= 16:
                    last_summary = self.last_checks.get('daily_summary')
                    
                    # Send summary once per day
                    if not last_summary or last_summary.date() < now.date():
                        summary_data = await self._generate_daily_summary()
                        await self.email_alerts.send_daily_summary(summary_data)
                        self.last_checks['daily_summary'] = now
                
                # Wait 1 hour before checking again
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in daily summary monitoring: {e}")
                await asyncio.sleep(3600)
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive performance metrics"""
        try:
            metrics = {}
            
            # API response times
            if self.sheets_service:
                start_time = time.time()
                health = await self.sheets_service.get_integration_status()
                api_response_time = (time.time() - start_time) * 1000
                
                metrics['avg_api_response_time'] = api_response_time
                metrics['sheets_status'] = health.get('sheets_status', {}).get('status', 'Unknown')
            
            # Error rate calculation
            recent_errors = [
                entry for entry in self.error_history
                if entry['timestamp'] > datetime.now(self.ist) - timedelta(hours=1)
            ]
            
            total_operations = len(self.performance_history) + len(recent_errors)
            error_rate = (len(recent_errors) / total_operations * 100) if total_operations > 0 else 0
            metrics['error_rate'] = round(error_rate, 2)
            
            # Data freshness
            if self.sheets_service and hasattr(self.sheets_service, 'last_updates'):
                last_updates = self.sheets_service.last_updates
                if last_updates:
                    oldest_update = min(last_updates.values())
                    freshness_minutes = (datetime.now(self.ist) - oldest_update).total_seconds() / 60
                    metrics['data_freshness_minutes'] = round(freshness_minutes, 1)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            return {}
    
    def _check_performance_thresholds(self, metrics: Dict[str, Any]) -> List[str]:
        """Check if performance metrics exceed thresholds"""
        violations = []
        
        try:
            if metrics.get('avg_api_response_time', 0) > self.thresholds['api_response_time']:
                violations.append(f"API response time: {metrics['avg_api_response_time']}ms")
            
            if metrics.get('error_rate', 0) > self.thresholds['error_rate']:
                violations.append(f"Error rate: {metrics['error_rate']}%")
            
            if metrics.get('data_freshness_minutes', 0) > self.thresholds['data_freshness']:
                violations.append(f"Data freshness: {metrics['data_freshness_minutes']} minutes")
            
            return violations
            
        except Exception as e:
            logger.error(f"Error checking performance thresholds: {e}")
            return []
    
    async def _check_api_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health of various APIs"""
        api_health = {}
        
        try:
            # Check Google Sheets API
            if self.sheets_service:
                try:
                    start_time = time.time()
                    await self.sheets_service.get_integration_status()
                    response_time = (time.time() - start_time) * 1000
                    
                    api_health['Google Sheets'] = {
                        'status': 'Online',
                        'response_time': response_time,
                        'last_check': datetime.now(self.ist)
                    }
                except Exception as e:
                    api_health['Google Sheets'] = {
                        'status': 'Failed',
                        'error': str(e),
                        'failure_duration': self._calculate_failure_duration('Google Sheets'),
                        'last_check': datetime.now(self.ist)
                    }
            
            # Add other API checks here (Kite, Yahoo Finance, etc.)
            
            return api_health
            
        except Exception as e:
            logger.error(f"Error checking API health: {e}")
            return {}
    
    async def _check_data_quality(self) -> List[Dict[str, Any]]:
        """Check data quality and completeness"""
        issues = []
        
        try:
            # Check if sheets service is available
            if not self.sheets_service:
                issues.append({
                    'component': 'Google Sheets Service',
                    'description': 'Sheets service not available',
                    'severity': 'Critical'
                })
                return issues
            
            # Check data freshness
            if hasattr(self.sheets_service, 'last_updates'):
                now = datetime.now(self.ist)
                for data_type, last_update in self.sheets_service.last_updates.items():
                    age_minutes = (now - last_update).total_seconds() / 60
                    
                    if age_minutes > 30:  # Data older than 30 minutes
                        issues.append({
                            'component': f'{data_type.title()} Data',
                            'description': f'Data is {age_minutes:.1f} minutes old',
                            'severity': 'Warning' if age_minutes < 60 else 'Critical'
                        })
            
            # Check for missing required data
            # This would be expanded based on specific data requirements
            
            return issues
            
        except Exception as e:
            logger.error(f"Error checking data quality: {e}")
            return []
    
    def _calculate_failure_duration(self, api_name: str) -> int:
        """Calculate how long an API has been failing"""
        try:
            # This would track API failure start times
            # For now, return 0 as placeholder
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating failure duration for {api_name}: {e}")
            return 0
    
    async def _generate_daily_summary(self) -> Dict[str, Any]:
        """Generate daily summary data"""
        try:
            summary = {
                'date': datetime.now(self.ist).strftime('%Y-%m-%d'),
                'signals_generated': 0,
                'signals_executed': 0,
                'signals_rejected': 0,
                'win_rate': 0,
                'daily_pnl': 0,
                'sheets_status': 'Unknown',
                'avg_response_time': 0,
                'error_count': len(self.error_history),
                'data_completeness': 100,
                'nifty_close': 'N/A',
                'banknifty_close': 'N/A',
                'vix_close': 'N/A'
            }
            
            # Get data from sheets service if available
            if self.sheets_service:
                status = await self.sheets_service.get_integration_status()
                summary['sheets_status'] = status.get('sheets_status', {}).get('status', 'Unknown')
                summary['signals_generated'] = status.get('trade_history_count', 0)
                summary['signals_rejected'] = status.get('rejected_signals_count', 0)
            
            # Calculate average response time from performance history
            if self.performance_history:
                response_times = [
                    entry['metrics'].get('avg_api_response_time', 0)
                    for entry in self.performance_history
                    if entry['metrics'].get('avg_api_response_time')
                ]
                if response_times:
                    summary['avg_response_time'] = round(statistics.mean(response_times), 2)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating daily summary: {e}")
            return {}
    
    def log_error(self, component: str, error_message: str, severity: str = "Error"):
        """Log an error for tracking"""
        try:
            self.error_history.append({
                'timestamp': datetime.now(self.ist),
                'component': component,
                'error': error_message,
                'severity': severity
            })
            
            # Keep only last 24 hours of errors
            cutoff_time = datetime.now(self.ist) - timedelta(hours=24)
            self.error_history = [
                entry for entry in self.error_history
                if entry['timestamp'] > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error logging error: {e}")
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        try:
            return {
                'monitoring_active': self.monitoring_active,
                'last_checks': self.last_checks,
                'error_count_24h': len(self.error_history),
                'data_quality_issues': len(self.data_quality_issues),
                'performance_metrics': self.performance_metrics,
                'thresholds': self.thresholds
            }
            
        except Exception as e:
            logger.error(f"Error getting monitoring status: {e}")
            return {}
    
    async def stop_monitoring(self):
        """Stop system monitoring"""
        try:
            self.monitoring_active = False
            logger.info("System monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
    async def test_monitoring_system(self):
        """Test the monitoring system"""
        try:
            logger.info("Testing monitoring system...")
            
            # Test email alerts
            email_test = await self.email_alerts.test_email_system()
            
            # Test performance metrics collection
            metrics = await self._collect_performance_metrics()
            
            # Test data quality check
            quality_issues = await self._check_data_quality()
            
            # Test API health check
            api_health = await self._check_api_health()
            
            test_results = {
                'email_system': email_test,
                'performance_metrics': bool(metrics),
                'data_quality_check': True,
                'api_health_check': bool(api_health),
                'overall_status': 'PASS' if all([email_test, metrics, api_health]) else 'PARTIAL'
            }
            
            logger.info(f"Monitoring system test results: {test_results}")
            return test_results
            
        except Exception as e:
            logger.error(f"Error testing monitoring system: {e}")
            return {'overall_status': 'FAIL', 'error': str(e)}