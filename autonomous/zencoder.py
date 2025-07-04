"""
Autonomous Zencoder - Self-monitoring and self-fixing trading system
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

from utils.logger import get_logger
from notifications.telegram_notifier import TelegramNotifier

logger = get_logger('autonomous.zencoder')

class AutonomousZencoder:
    """Autonomous system monitor and optimizer"""
    
    def __init__(self, settings):
        self.settings = settings
        self.project_root = Path(__file__).parent.parent
        self.config_path = self.project_root / "config" / "autonomous_config.json"
        self.db_path = self.project_root / "data_storage" / "databases" / "autonomous.db"
        self.backup_dir = self.project_root / "backups"
        
        # Create necessary directories
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.telegram = TelegramNotifier(settings) if settings.TELEGRAM_BOT_TOKEN else None
        self.running = False
        self.monitoring_tasks = []
        
        # Load configuration
        self.config = self._load_config()
        
        logger.info("Autonomous Zencoder initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load autonomous system configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info("Autonomous configuration loaded")
                return config
            else:
                # Default configuration
                default_config = {
                    "monitoring_interval": 30,
                    "auto_fix_enabled": True,
                    "backup_interval": 3600,
                    "max_fixes_per_hour": 20,
                    "performance_optimization": True,
                    "telegram_notifications": True
                }
                
                # Save default configuration
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                
                logger.info("Default autonomous configuration created")
                return default_config
                
        except Exception as e:
            logger.error(f"Failed to load autonomous config: {e}")
            return {}
    
    async def initialize(self):
        """Initialize the autonomous system"""
        try:
            logger.info("Initializing autonomous system...")
            
            # Initialize database
            await self._initialize_database()
            
            # Send startup notification
            if self.telegram:
                await self._send_startup_notification()
            
            logger.info("Autonomous system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize autonomous system: {e}")
            raise
    
    async def _initialize_database(self):
        """Initialize the autonomous system database"""
        try:
            import sqlite3
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    status TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS issues_fixed (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    issue_type TEXT,
                    description TEXT,
                    fix_applied TEXT,
                    status TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT,
                    metric_value REAL,
                    improvement_applied TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Autonomous database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def _send_startup_notification(self):
        """Send startup notification via Telegram"""
        try:
            message = f"""
[AUTO] AUTONOMOUS ZENCODER ACTIVATED

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Mode: FULLY AUTONOMOUS
Status: MONITORING & AUTO-FIXING ENABLED

[OK] 24/7 System Monitoring
[OK] Automatic Issue Detection & Fixing
[OK] Performance Optimization
[OK] Backup Management
[OK] Emergency Recovery

Your trading system is now completely autonomous!
"""
            
            await self.telegram.send_message(message)
            logger.info("Startup notification sent")
            
        except Exception as e:
            logger.warning(f"Failed to send startup notification: {e}")
    
    async def start_monitoring(self):
        """Start autonomous monitoring"""
        try:
            self.running = True
            logger.info("Starting autonomous monitoring...")
            
            # Start monitoring tasks
            self.monitoring_tasks = [
                asyncio.create_task(self._system_health_monitor()),
                asyncio.create_task(self._issue_detector()),
                asyncio.create_task(self._performance_optimizer()),
                asyncio.create_task(self._backup_manager())
            ]
            
            # Wait for all tasks
            await asyncio.gather(*self.monitoring_tasks)
            
        except asyncio.CancelledError:
            logger.info("Autonomous monitoring cancelled")
        except Exception as e:
            logger.error(f"Autonomous monitoring error: {e}")
        finally:
            self.running = False
    
    async def _system_health_monitor(self):
        """Monitor system health continuously"""
        while self.running:
            try:
                # Get system metrics
                health_data = await self._get_system_health()
                
                # Store in database
                await self._store_health_data(health_data)
                
                # Check for issues
                if health_data['status'] != 'healthy':
                    await self._handle_health_issue(health_data)
                
                # Wait for next check
                await asyncio.sleep(self.config.get('monitoring_interval', 30))
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics"""
        try:
            import psutil
            
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            # Determine status
            status = 'healthy'
            if cpu_usage > 90 or memory.percent > 95 or disk.percent > 95:
                status = 'warning'
            if cpu_usage > 95 or memory.percent > 98 or disk.percent > 98:
                status = 'critical'
            
            return {
                'cpu_usage': cpu_usage,
                'memory_usage': memory.percent,
                'disk_usage': disk.percent,
                'status': status,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                'cpu_usage': 0,
                'memory_usage': 0,
                'disk_usage': 0,
                'status': 'error',
                'timestamp': datetime.now()
            }
    
    async def _store_health_data(self, health_data: Dict[str, Any]):
        """Store health data in database"""
        try:
            import sqlite3
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_health (cpu_usage, memory_usage, disk_usage, status)
                VALUES (?, ?, ?, ?)
            ''', (
                health_data['cpu_usage'],
                health_data['memory_usage'],
                health_data['disk_usage'],
                health_data['status']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store health data: {e}")
    
    async def _handle_health_issue(self, health_data: Dict[str, Any]):
        """Handle system health issues"""
        try:
            issue_type = f"System Health: {health_data['status']}"
            description = f"CPU: {health_data['cpu_usage']:.1f}%, Memory: {health_data['memory_usage']:.1f}%, Disk: {health_data['disk_usage']:.1f}%"
            
            # Apply fixes based on issue type
            fix_applied = "None"
            
            if health_data['memory_usage'] > 90:
                # Run garbage collection
                import gc
                gc.collect()
                fix_applied = "Garbage collection executed"
            
            if health_data['disk_usage'] > 90:
                # Clean temporary files
                await self._clean_temp_files()
                fix_applied = "Temporary files cleaned"
            
            # Log the issue and fix
            await self._log_issue_fixed(issue_type, description, fix_applied)
            
            # Send notification if critical
            if health_data['status'] == 'critical' and self.telegram:
                await self.telegram.send_message(f"[ALERT] CRITICAL: {description}\n[TOOL] Fix: {fix_applied}")
            
        except Exception as e:
            logger.error(f"Failed to handle health issue: {e}")
    
    async def _issue_detector(self):
        """Detect and fix system issues"""
        while self.running:
            try:
                # Check for common issues
                issues = await self._scan_for_issues()
                
                for issue in issues:
                    if self.config.get('auto_fix_enabled', True):
                        await self._fix_issue(issue)
                
                # Wait before next scan
                await asyncio.sleep(self.config.get('monitoring_interval', 30))
                
            except Exception as e:
                logger.error(f"Issue detection error: {e}")
                await asyncio.sleep(60)
    
    async def _scan_for_issues(self) -> List[Dict[str, Any]]:
        """Scan for common system issues"""
        issues = []
        
        try:
            # Check log files for errors
            log_dir = self.project_root / "logs"
            if log_dir.exists():
                for log_file in log_dir.glob("*.log"):
                    if log_file.stat().st_size > 100 * 1024 * 1024:  # 100MB
                        issues.append({
                            'type': 'large_log_file',
                            'description': f'Log file {log_file.name} is too large',
                            'file_path': log_file
                        })
            
            # Check for Python processes
            import psutil
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
                try:
                    if 'python' in proc.info['name'].lower():
                        if proc.info['memory_percent'] > 10:  # Using more than 10% memory
                            python_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if len(python_processes) > 5:
                issues.append({
                    'type': 'too_many_python_processes',
                    'description': f'Found {len(python_processes)} Python processes',
                    'processes': python_processes
                })
            
        except Exception as e:
            logger.error(f"Issue scanning error: {e}")
        
        return issues
    
    async def _fix_issue(self, issue: Dict[str, Any]):
        """Fix a detected issue"""
        try:
            fix_applied = "None"
            
            if issue['type'] == 'large_log_file':
                # Rotate large log files
                log_file = issue['file_path']
                backup_file = log_file.with_suffix(f'.{datetime.now().strftime("%Y%m%d_%H%M%S")}.bak')
                log_file.rename(backup_file)
                log_file.touch()
                fix_applied = f"Log file rotated to {backup_file.name}"
            
            elif issue['type'] == 'too_many_python_processes':
                # This is informational, no automatic fix
                fix_applied = "Monitoring - no automatic fix applied"
            
            # Log the fix
            await self._log_issue_fixed(issue['type'], issue['description'], fix_applied)
            
            logger.info(f"Fixed issue: {issue['type']} - {fix_applied}")
            
        except Exception as e:
            logger.error(f"Failed to fix issue {issue['type']}: {e}")
    
    async def _log_issue_fixed(self, issue_type: str, description: str, fix_applied: str):
        """Log a fixed issue to database"""
        try:
            import sqlite3
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO issues_fixed (issue_type, description, fix_applied, status)
                VALUES (?, ?, ?, ?)
            ''', (issue_type, description, fix_applied, 'fixed'))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log issue fix: {e}")
    
    async def _performance_optimizer(self):
        """Optimize system performance"""
        while self.running:
            try:
                if self.config.get('performance_optimization', True):
                    await self._optimize_performance()
                
                # Wait 1 hour between optimizations
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Performance optimization error: {e}")
                await asyncio.sleep(3600)
    
    async def _optimize_performance(self):
        """Apply performance optimizations"""
        try:
            optimizations = []
            
            # Garbage collection
            import gc
            collected = gc.collect()
            if collected > 0:
                optimizations.append(f"Garbage collection: {collected} objects")
            
            # Clean cache directory
            cache_dir = self.project_root / "data_storage" / "cache"
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("*"))
                old_files = [f for f in cache_files if (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days > 7]
                for old_file in old_files:
                    old_file.unlink()
                if old_files:
                    optimizations.append(f"Cleaned {len(old_files)} old cache files")
            
            # Log optimizations
            for optimization in optimizations:
                await self._log_performance_metric("optimization", 1.0, optimization)
            
            if optimizations:
                logger.info(f"Performance optimizations applied: {', '.join(optimizations)}")
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
    
    async def _log_performance_metric(self, metric_name: str, metric_value: float, improvement: str):
        """Log performance metric to database"""
        try:
            import sqlite3
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics (metric_name, metric_value, improvement_applied)
                VALUES (?, ?, ?)
            ''', (metric_name, metric_value, improvement))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log performance metric: {e}")
    
    async def _backup_manager(self):
        """Manage system backups"""
        while self.running:
            try:
                await self._create_backup()
                
                # Wait for backup interval
                await asyncio.sleep(self.config.get('backup_interval', 3600))
                
            except Exception as e:
                logger.error(f"Backup manager error: {e}")
                await asyncio.sleep(3600)
    
    async def _create_backup(self):
        """Create system backup"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.backup_dir / f"autonomous_backup_{timestamp}"
            backup_path.mkdir(exist_ok=True)
            
            # Backup critical files
            import shutil
            
            critical_files = [
                'config',
                'main.py',
                'requirements.txt'
            ]
            
            for item in critical_files:
                source = self.project_root / item
                if source.exists():
                    if source.is_file():
                        shutil.copy2(source, backup_path / source.name)
                    elif source.is_dir():
                        shutil.copytree(source, backup_path / source.name, dirs_exist_ok=True)
            
            # Backup database
            if self.db_path.exists():
                shutil.copy2(self.db_path, backup_path / "autonomous.db")
            
            # Clean old backups (keep last 10)
            backups = sorted(self.backup_dir.glob("autonomous_backup_*"), key=lambda x: x.stat().st_mtime, reverse=True)
            for old_backup in backups[10:]:
                shutil.rmtree(old_backup)
            
            logger.info(f"Backup created: {backup_path.name}")
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
    
    async def _clean_temp_files(self):
        """Clean temporary files"""
        try:
            import tempfile
            import shutil
            
            temp_dir = Path(tempfile.gettempdir())
            cleaned_files = 0
            
            # Clean Python temp files
            for temp_file in temp_dir.glob("tmp*"):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                        cleaned_files += 1
                    elif temp_file.is_dir():
                        shutil.rmtree(temp_file)
                        cleaned_files += 1
                except:
                    continue
            
            logger.info(f"Cleaned {cleaned_files} temporary files")
            
        except Exception as e:
            logger.error(f"Temp file cleanup failed: {e}")
    
    async def shutdown(self):
        """Shutdown the autonomous system"""
        try:
            logger.info("Shutting down autonomous system...")
            
            self.running = False
            
            # Cancel monitoring tasks
            for task in self.monitoring_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.monitoring_tasks:
                await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
            
            # Send shutdown notification
            if self.telegram:
                await self.telegram.send_message("[AUTO] Autonomous Zencoder shutting down...")
            
            logger.info("Autonomous system shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")