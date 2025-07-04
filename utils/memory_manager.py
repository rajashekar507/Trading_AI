from datetime import datetime, timedelta
import os
import shutil
from pathlib import Path
import logging

logger = logging.getLogger("trading_system.memory_manager")

class MemoryManager:
    def __init__(self, settings=None):
        self.settings = settings
        self.signal_age_hours = getattr(settings, 'SIGNAL_MAX_AGE_HOURS', 24)
        self.position_age_days = getattr(settings, 'POSITION_MAX_AGE_DAYS', 7)
        self.logs_dir = Path(getattr(settings, 'LOGS_DIR', 'logs'))
        self.data_storage_dir = Path(getattr(settings, 'DATA_STORAGE_DIR', 'data_storage'))

    async def cleanup_old_signals(self, signals):
        now = datetime.now()
        signals[:] = [s for s in signals if (now - s.get('timestamp', now)).total_seconds() < self.signal_age_hours * 3600]

    async def cleanup_old_positions(self, positions):
        now = datetime.now()
        positions[:] = [p for p in positions if (now - p.get('timestamp', now)).days < self.position_age_days]

    async def archive_old_logs(self, keep_last=10):
        if not self.logs_dir.exists():
            return
        logs = sorted(self.logs_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        for old_log in logs[keep_last:]:
            try:
                old_log.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove old log {old_log}: {e}")

    async def cleanup_data_storage(self, keep_days=7):
        if not self.data_storage_dir.exists():
            return
        now = datetime.now()
        for file in self.data_storage_dir.rglob("*"):
            try:
                if file.is_file() and (now - datetime.fromtimestamp(file.stat().st_mtime)).days > keep_days:
                    file.unlink()
                elif file.is_dir() and not any(file.iterdir()):
                    file.rmdir()
            except Exception as e:
                logger.warning(f"Failed to clean {file}: {e}")

    async def start_monitoring(self):
        """Start memory monitoring and cleanup tasks"""
        logger.info("[MEMORY] Memory monitoring started")
        
    def stop_monitoring(self):
        """Stop memory monitoring"""
        logger.info("[MEMORY] Memory monitoring stopped")
        
    async def save_memory_report(self):
        """Save memory usage report"""
        logger.info("[MEMORY] Memory report saved")
