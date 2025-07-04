"""
Simple wrapper functions for automatic GitHub backup
Makes it easy to backup changes from anywhere in the system
"""

from utils.github_auto_push import auto_push
import logging

logger = logging.getLogger('trading_system.auto_backup')

def backup_after_fix(description="Fixed system issues"):
    """Backup after fixing bugs or issues"""
    try:
        message = f"🐛 Fix: {description}"
        return auto_push(message)
    except Exception as e:
        logger.error(f"Backup after fix failed: {e}")
        return False

def backup_after_feature(description="Added new feature"):
    """Backup after adding new features"""
    try:
        message = f"✨ Feature: {description}"
        return auto_push(message)
    except Exception as e:
        logger.error(f"Backup after feature failed: {e}")
        return False

def backup_after_improvement(description="System improvements"):
    """Backup after any system improvements"""
    try:
        message = f"🔄 Update: {description}"
        return auto_push(message)
    except Exception as e:
        logger.error(f"Backup after improvement failed: {e}")
        return False

def backup_risk_changes(description="Risk management updates"):
    """Backup after risk management changes"""
    try:
        message = f"🛡️ Risk: {description}"
        return auto_push(message)
    except Exception as e:
        logger.error(f"Backup risk changes failed: {e}")
        return False

def backup_trading_logic(description="Trading logic improvements"):
    """Backup after trading logic changes"""
    try:
        message = f"📈 Trading: {description}"
        return auto_push(message)
    except Exception as e:
        logger.error(f"Backup trading logic failed: {e}")
        return False

def backup_ml_improvements(description="ML model improvements"):
    """Backup after ML improvements"""
    try:
        message = f"🤖 ML: {description}"
        return auto_push(message)
    except Exception as e:
        logger.error(f"Backup ML improvements failed: {e}")
        return False

def emergency_backup(description="Emergency backup"):
    """Emergency backup with force push"""
    try:
        message = f"🚨 Emergency: {description}"
        # Use the GitHubAutoPush class directly for force push
        from utils.github_auto_push import GitHubAutoPush
        pusher = GitHubAutoPush()
        return pusher.auto_push(message, force_push=True)
    except Exception as e:
        logger.error(f"Emergency backup failed: {e}")
        return False

# Quick access functions
def quick_backup():
    """Quick backup with default message"""
    return backup_after_improvement("Quick system backup")

def safe_backup():
    """Safe backup that won't push if there are issues"""
    return backup_after_improvement("Safe system backup")

# Test function
def test_backup_system():
    """Test the backup system"""
    print("🧪 Testing backup system...")
    try:
        result = backup_after_improvement("Testing backup system functionality")
        print(f"Backup test: {'✅ Success' if result else '❌ Failed'}")
        return result
    except Exception as e:
        print(f"Backup test failed: {e}")
        return False