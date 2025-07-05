"""
Auto-commit system that triggers after any code changes
This ensures every improvement is automatically backed up to GitHub
"""

import os
import sys
import time
from pathlib import Path
from utils.github_auto_push import GitHubAutoPush
import logging

logger = logging.getLogger('trading_system.auto_commit')

class AutoCommitSystem:
    """Automatically commits changes after system improvements"""
    
    def __init__(self):
        self.github_pusher = GitHubAutoPush()
        self.last_commit_time = time.time()
        self.min_commit_interval = 300  # 5 minutes minimum between commits
    
    def should_commit(self):
        """Check if enough time has passed since last commit"""
        return (time.time() - self.last_commit_time) > self.min_commit_interval
    
    def commit_improvements(self, improvement_type="System improvement", details=""):
        """Commit improvements with intelligent messages"""
        
        if not self.should_commit():
            logger.info("Skipping commit - too soon since last commit")
            return False
        
        try:
            # Generate commit message based on improvement type
            commit_messages = {
                "bug_fix": "🐛 Fix: Resolved system issues and bugs",
                "feature_add": "✨ Feature: Added new functionality",
                "performance": "⚡ Performance: Optimized system performance", 
                "security": "🔒 Security: Enhanced security measures",
                "risk_management": "🛡️ Risk: Improved risk management",
                "trading_logic": "📈 Trading: Enhanced trading logic",
                "ml_improvement": "🤖 ML: Machine learning improvements",
                "config_update": "⚙️ Config: Updated system configuration",
                "data_handling": "📊 Data: Improved data processing",
                "api_integration": "🔌 API: Enhanced API integration"
            }
            
            message = commit_messages.get(improvement_type, "🔄 Update: System improvements")
            
            if details:
                message += f"\n\n📝 Details: {details}"
            
            # Attempt to push
            success = self.github_pusher.auto_push(message)
            
            if success:
                self.last_commit_time = time.time()
                logger.info(f"✅ Successfully committed: {improvement_type}")
                return True
            else:
                logger.warning(f"❌ Failed to commit: {improvement_type}")
                return False
                
        except Exception as e:
            logger.error(f"Auto-commit failed: {e}")
            return False

# Global instance for easy access
auto_commit = AutoCommitSystem()

def commit_after_fix(fix_type="bug_fix", details=""):
    """Quick function to commit after fixing issues"""
    return auto_commit.commit_improvements(fix_type, details)

def commit_after_feature(details=""):
    """Quick function to commit after adding features"""
    return auto_commit.commit_improvements("feature_add", details)

def commit_after_improvement(improvement_type="System improvement", details=""):
    """General function to commit any improvements"""
    return auto_commit.commit_improvements(improvement_type, details)

if __name__ == "__main__":
    # Test the auto-commit system
    print("🧪 Testing auto-commit system...")
    success = commit_after_improvement("testing", "Auto-commit system test")
    print(f"Test result: {'✅ Success' if success else '❌ Failed'}")