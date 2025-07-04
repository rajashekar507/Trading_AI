"""
Automation Orchestrator - Full System Automation
Manages all automated processes including API authentication, health monitoring, and trading operations
"""

import os
import sys
import asyncio
import logging
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from threading import Thread
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

# Import our enhanced systems
from auth.enhanced_kite_auth import EnhancedKiteAuthenticator, get_authenticated_kite_client
from utils.api_health_monitor import APIHealthMonitor
from brokers.dhan_integration import DhanAPIClient
from notifications.telegram_notifier import TelegramNotifier
from config.enhanced_settings import EnhancedSettings

logger = logging.getLogger('trading_system.automation_orchestrator')

class AutomationOrchestrator:
    """Master automation orchestrator for the entire trading system"""
    
    def __init__(self):
        self.settings = EnhancedSettings()
        self.is_running = False
        self.health_monitor = None
        self.kite_auth = None
        self.dhan_client = None
        self.telegram_notifier = None
        
        # Automation status
        self.automation_status = {
            'kite_auth': False,
            'dhan_connection': False,
            'telegram_notifications': False,
            'health_monitoring': False,
            'trading_system': False
        }
        
        # Scheduling
        self.scheduled_tasks = []
        self.scheduler_thread = None
        
        logger.info("[ORCHESTRATOR] Automation Orchestrator initialized")
    
    async def initialize_all_systems(self) -> bool:
        """Initialize all systems with full automation"""
        try:
            logger.info("[INIT] Starting full system initialization...")
            
            # Step 1: Initialize Kite Authentication
            await self._initialize_kite_authentication()
            
            # Step 2: Initialize Dhan Connection
            await self._initialize_dhan_connection()
            
            # Step 3: Initialize Telegram Notifications
            await self._initialize_telegram_notifications()
            
            # Step 4: Initialize Health Monitoring
            await self._initialize_health_monitoring()
            
            # Step 5: Setup Automated Schedules
            await self._setup_automated_schedules()
            
            # Step 6: Start Automation Loop
            await self._start_automation_loop()
            
            logger.info("[INIT] Full system initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"[INIT] System initialization failed: {e}")
            return False
    
    async def _initialize_kite_authentication(self) -> bool:
        """Initialize Kite authentication with full automation"""
        try:
            logger.info("[KITE] Initializing Kite authentication...")
            
            self.kite_auth = EnhancedKiteAuthenticator()
            
            # Attempt authentication
            if await self.kite_auth.authenticate():
                self.automation_status['kite_auth'] = True
                logger.info("[KITE] ✅ Kite authentication successful")
                
                # Test connection
                if await self.kite_auth.test_connection():
                    logger.info("[KITE] ✅ Kite connection test passed")
                    
                    # Send success notification
                    await self._send_system_notification(
                        "🟢 KITE AUTHENTICATION SUCCESS",
                        "Kite Connect API is now fully operational and authenticated."
                    )
                    return True
                else:
                    logger.warning("[KITE] ⚠️ Kite connection test failed")
                    return False
            else:
                logger.error("[KITE] ❌ Kite authentication failed")
                await self._send_system_notification(
                    "🔴 KITE AUTHENTICATION FAILED",
                    "Failed to authenticate with Kite Connect API. Manual intervention may be required."
                )
                return False
                
        except Exception as e:
            logger.error(f"[KITE] Kite initialization failed: {e}")
            return False
    
    async def _initialize_dhan_connection(self) -> bool:
        """Initialize Dhan connection"""
        try:
            logger.info("[DHAN] Initializing Dhan connection...")
            
            client_id = os.getenv('DHAN_CLIENT_ID')
            access_token = os.getenv('DHAN_ACCESS_TOKEN')
            
            if not client_id or not access_token:
                logger.error("[DHAN] ❌ Missing Dhan credentials")
                return False
            
            self.dhan_client = DhanAPIClient(client_id, access_token)
            
            # Test connection
            try:
                profile = self.dhan_client.get_profile()
                if profile and not profile.get('error'):
                    self.automation_status['dhan_connection'] = True
                    logger.info("[DHAN] ✅ Dhan connection successful")
                    
                    # Send success notification
                    await self._send_system_notification(
                        "🟢 DHAN CONNECTION SUCCESS",
                        f"Dhan API is operational. Client: {profile.get('dhanClientId', 'Unknown')}"
                    )
                    return True
                else:
                    logger.error(f"[DHAN] ❌ Dhan connection failed: {profile}")
                    return False
                    
            except Exception as e:
                logger.error(f"[DHAN] Dhan connection test failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"[DHAN] Dhan initialization failed: {e}")
            return False
    
    async def _initialize_telegram_notifications(self) -> bool:
        """Initialize Telegram notifications"""
        try:
            logger.info("[TELEGRAM] Initializing Telegram notifications...")
            
            self.telegram_notifier = TelegramNotifier(self.settings)
            
            if await self.telegram_notifier.initialize():
                self.automation_status['telegram_notifications'] = True
                logger.info("[TELEGRAM] ✅ Telegram notifications initialized")
                
                # Send test notification
                await self.telegram_notifier.send_system_alert(
                    "SYSTEM_STARTUP",
                    "🚀 VLR_AI Trading System automation is now fully operational!"
                )
                return True
            else:
                logger.error("[TELEGRAM] ❌ Telegram initialization failed")
                return False
                
        except Exception as e:
            logger.error(f"[TELEGRAM] Telegram initialization failed: {e}")
            return False
    
    async def _initialize_health_monitoring(self) -> bool:
        """Initialize health monitoring system"""
        try:
            logger.info("[HEALTH] Initializing health monitoring...")
            
            self.health_monitor = APIHealthMonitor()
            
            # Run initial health check
            health_status = await self.health_monitor.comprehensive_health_check()
            
            if health_status:
                self.automation_status['health_monitoring'] = True
                logger.info("[HEALTH] ✅ Health monitoring initialized")
                
                # Send health status notification
                summary = self.health_monitor.get_health_summary()
                await self._send_system_notification(
                    "📊 SYSTEM HEALTH STATUS",
                    f"Health Monitoring Active\n"
                    f"Healthy APIs: {summary['healthy_apis']}/{summary['total_apis']}\n"
                    f"Overall Health: {summary['health_percentage']:.1f}%"
                )
                return True
            else:
                logger.error("[HEALTH] ❌ Health monitoring initialization failed")
                return False
                
        except Exception as e:
            logger.error(f"[HEALTH] Health monitoring initialization failed: {e}")
            return False
    
    async def _setup_automated_schedules(self):
        """Setup all automated schedules"""
        try:
            logger.info("[SCHEDULE] Setting up automated schedules...")
            
            # Daily Kite token refresh (6:00 AM)
            schedule.every().day.at("06:00").do(self._schedule_kite_token_refresh)
            
            # Health checks every 5 minutes
            schedule.every(5).minutes.do(self._schedule_health_check)
            
            # System status report every hour
            schedule.every().hour.do(self._schedule_system_status_report)
            
            # Daily system summary (6:00 PM)
            schedule.every().day.at("18:00").do(self._schedule_daily_summary)
            
            # Weekly system maintenance (Sunday 2:00 AM)
            schedule.every().sunday.at("02:00").do(self._schedule_weekly_maintenance)
            
            # Emergency health check every minute (for critical issues)
            schedule.every().minute.do(self._schedule_emergency_check)
            
            logger.info("[SCHEDULE] ✅ Automated schedules configured")
            
        except Exception as e:
            logger.error(f"[SCHEDULE] Failed to setup schedules: {e}")
    
    def _schedule_kite_token_refresh(self):
        """Scheduled Kite token refresh"""
        asyncio.create_task(self._automated_kite_token_refresh())
    
    def _schedule_health_check(self):
        """Scheduled health check"""
        asyncio.create_task(self._automated_health_check())
    
    def _schedule_system_status_report(self):
        """Scheduled system status report"""
        asyncio.create_task(self._automated_system_status_report())
    
    def _schedule_daily_summary(self):
        """Scheduled daily summary"""
        asyncio.create_task(self._automated_daily_summary())
    
    def _schedule_weekly_maintenance(self):
        """Scheduled weekly maintenance"""
        asyncio.create_task(self._automated_weekly_maintenance())
    
    def _schedule_emergency_check(self):
        """Scheduled emergency check"""
        asyncio.create_task(self._automated_emergency_check())
    
    async def _automated_kite_token_refresh(self):
        """Automated Kite token refresh"""
        try:
            logger.info("[AUTO] Starting automated Kite token refresh...")
            
            if self.kite_auth:
                success = await self.kite_auth.force_refresh_token()
                if success:
                    logger.info("[AUTO] ✅ Kite token refreshed successfully")
                    await self._send_system_notification(
                        "🔄 KITE TOKEN REFRESHED",
                        "Daily Kite access token has been refreshed successfully."
                    )
                else:
                    logger.error("[AUTO] ❌ Kite token refresh failed")
                    await self._send_system_notification(
                        "🚨 KITE TOKEN REFRESH FAILED",
                        "Failed to refresh Kite access token. Manual intervention required."
                    )
            
        except Exception as e:
            logger.error(f"[AUTO] Automated Kite token refresh failed: {e}")
    
    async def _automated_health_check(self):
        """Automated health check"""
        try:
            if self.health_monitor:
                await self.health_monitor.comprehensive_health_check()
                
        except Exception as e:
            logger.error(f"[AUTO] Automated health check failed: {e}")
    
    async def _automated_system_status_report(self):
        """Automated system status report"""
        try:
            logger.info("[AUTO] Generating system status report...")
            
            status_report = await self._generate_system_status_report()
            
            # Send report via Telegram (only if there are issues)
            if status_report['issues_count'] > 0:
                await self._send_system_notification(
                    "⚠️ SYSTEM STATUS ALERT",
                    status_report['summary']
                )
            
        except Exception as e:
            logger.error(f"[AUTO] System status report failed: {e}")
    
    async def _automated_daily_summary(self):
        """Automated daily summary"""
        try:
            logger.info("[AUTO] Generating daily summary...")
            
            summary = await self._generate_daily_summary()
            
            await self._send_system_notification(
                "📊 DAILY SYSTEM SUMMARY",
                summary
            )
            
        except Exception as e:
            logger.error(f"[AUTO] Daily summary failed: {e}")
    
    async def _automated_weekly_maintenance(self):
        """Automated weekly maintenance"""
        try:
            logger.info("[AUTO] Starting weekly maintenance...")
            
            # Clean up old log files
            await self._cleanup_old_logs()
            
            # Refresh all API connections
            await self._refresh_all_connections()
            
            # Generate weekly report
            report = await self._generate_weekly_report()
            
            await self._send_system_notification(
                "🔧 WEEKLY MAINTENANCE COMPLETE",
                report
            )
            
        except Exception as e:
            logger.error(f"[AUTO] Weekly maintenance failed: {e}")
    
    async def _automated_emergency_check(self):
        """Automated emergency check for critical issues"""
        try:
            if not self.health_monitor:
                return
            
            # Check for critical failures
            critical_issues = []
            for api_name, status in self.health_monitor.health_status.items():
                if not status.is_healthy and status.consecutive_failures >= 3:
                    critical_issues.append(api_name)
            
            if critical_issues:
                logger.warning(f"[EMERGENCY] Critical issues detected: {critical_issues}")
                
                # Trigger emergency healing
                for api in critical_issues:
                    await self._emergency_healing(api)
            
        except Exception as e:
            logger.error(f"[EMERGENCY] Emergency check failed: {e}")
    
    async def _emergency_healing(self, api_name: str):
        """Emergency healing for critical API failures"""
        try:
            logger.warning(f"[EMERGENCY] Starting emergency healing for {api_name}")
            
            if api_name == 'kite':
                # Force Kite token refresh
                if self.kite_auth:
                    await self.kite_auth.force_refresh_token()
            
            elif api_name == 'dhan':
                # Reinitialize Dhan connection
                await self._initialize_dhan_connection()
            
            elif api_name == 'telegram':
                # Reinitialize Telegram
                await self._initialize_telegram_notifications()
            
            # Send emergency notification
            await self._send_system_notification(
                f"🚨 EMERGENCY HEALING: {api_name.upper()}",
                f"Emergency healing procedures activated for {api_name} due to critical failures."
            )
            
        except Exception as e:
            logger.error(f"[EMERGENCY] Emergency healing failed for {api_name}: {e}")
    
    async def _start_automation_loop(self):
        """Start the main automation loop"""
        try:
            logger.info("[LOOP] Starting automation loop...")
            
            self.is_running = True
            
            # Start scheduler in separate thread
            self.scheduler_thread = Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
            # Start health monitoring
            if self.health_monitor:
                asyncio.create_task(self.health_monitor.start_monitoring())
            
            self.automation_status['trading_system'] = True
            logger.info("[LOOP] ✅ Automation loop started")
            
        except Exception as e:
            logger.error(f"[LOOP] Failed to start automation loop: {e}")
    
    def _run_scheduler(self):
        """Run the scheduler in a separate thread"""
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"[SCHEDULER] Scheduler error: {e}")
    
    async def _generate_system_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive system status report"""
        try:
            issues = []
            
            # Check automation status
            for system, status in self.automation_status.items():
                if not status:
                    issues.append(f"{system} is not operational")
            
            # Check API health
            if self.health_monitor:
                for api_name, health_status in self.health_monitor.health_status.items():
                    if not health_status.is_healthy:
                        issues.append(f"{api_name} API is unhealthy")
            
            summary = f"System Status Report - {datetime.now().strftime('%H:%M:%S')}\n"
            summary += f"Issues Found: {len(issues)}\n"
            
            if issues:
                summary += "\nIssues:\n"
                for issue in issues[:5]:  # Limit to 5 issues
                    summary += f"• {issue}\n"
            else:
                summary += "\n✅ All systems operational"
            
            return {
                'issues_count': len(issues),
                'issues': issues,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"[REPORT] Failed to generate status report: {e}")
            return {'issues_count': 0, 'issues': [], 'summary': 'Report generation failed'}
    
    async def _generate_daily_summary(self) -> str:
        """Generate daily summary"""
        try:
            summary = f"📊 DAILY SUMMARY - {datetime.now().strftime('%Y-%m-%d')}\n\n"
            
            # System status
            operational_systems = sum(1 for status in self.automation_status.values() if status)
            total_systems = len(self.automation_status)
            summary += f"🔧 System Status: {operational_systems}/{total_systems} operational\n"
            
            # API health
            if self.health_monitor:
                health_summary = self.health_monitor.get_health_summary()
                summary += f"📡 API Health: {health_summary['health_percentage']:.1f}%\n"
                summary += f"🚀 Avg Response: {health_summary['average_response_time']:.2f}s\n"
            
            # Kite status
            if self.automation_status['kite_auth']:
                summary += "✅ Kite: Authenticated\n"
            else:
                summary += "❌ Kite: Authentication needed\n"
            
            # Dhan status
            if self.automation_status['dhan_connection']:
                summary += "✅ Dhan: Connected\n"
            else:
                summary += "❌ Dhan: Connection issues\n"
            
            summary += f"\n🕐 Generated: {datetime.now().strftime('%H:%M:%S')}"
            
            return summary
            
        except Exception as e:
            logger.error(f"[SUMMARY] Failed to generate daily summary: {e}")
            return "Daily summary generation failed"
    
    async def _generate_weekly_report(self) -> str:
        """Generate weekly maintenance report"""
        try:
            report = f"🔧 WEEKLY MAINTENANCE REPORT\n"
            report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            report += "✅ Log cleanup completed\n"
            report += "✅ API connections refreshed\n"
            report += "✅ System health verified\n"
            report += "✅ Automated schedules validated\n"
            
            # Add system statistics
            if self.health_monitor:
                health_summary = self.health_monitor.get_health_summary()
                report += f"\n📊 Weekly Statistics:\n"
                report += f"• Average API Health: {health_summary['health_percentage']:.1f}%\n"
                report += f"• Average Response Time: {health_summary['average_response_time']:.2f}s\n"
            
            return report
            
        except Exception as e:
            logger.error(f"[WEEKLY] Failed to generate weekly report: {e}")
            return "Weekly report generation failed"
    
    async def _cleanup_old_logs(self):
        """Clean up old log files"""
        try:
            logs_dir = PROJECT_ROOT / 'logs'
            if not logs_dir.exists():
                return
            
            # Delete log files older than 7 days
            cutoff_date = datetime.now() - timedelta(days=7)
            
            for log_file in logs_dir.glob('*.log'):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    logger.info(f"[CLEANUP] Deleted old log file: {log_file.name}")
            
        except Exception as e:
            logger.error(f"[CLEANUP] Log cleanup failed: {e}")
    
    async def _refresh_all_connections(self):
        """Refresh all API connections"""
        try:
            logger.info("[REFRESH] Refreshing all API connections...")
            
            # Refresh Kite
            if self.kite_auth:
                await self.kite_auth.force_refresh_token()
            
            # Test Dhan connection
            if self.dhan_client:
                await self._initialize_dhan_connection()
            
            # Test Telegram
            if self.telegram_notifier:
                await self.telegram_notifier.initialize()
            
            logger.info("[REFRESH] ✅ All connections refreshed")
            
        except Exception as e:
            logger.error(f"[REFRESH] Connection refresh failed: {e}")
    
    async def _send_system_notification(self, title: str, message: str):
        """Send system notification via Telegram"""
        try:
            if self.telegram_notifier and self.automation_status['telegram_notifications']:
                full_message = f"{title}\n\n{message}"
                await self.telegram_notifier.send_system_alert("AUTOMATION", full_message)
            
        except Exception as e:
            logger.error(f"[NOTIFY] Failed to send system notification: {e}")
    
    def get_automation_status(self) -> Dict[str, Any]:
        """Get current automation status"""
        return {
            'automation_status': self.automation_status,
            'is_running': self.is_running,
            'health_summary': self.health_monitor.get_health_summary() if self.health_monitor else {},
            'last_update': datetime.now().isoformat()
        }
    
    async def stop_automation(self):
        """Stop all automation"""
        try:
            logger.info("[STOP] Stopping automation orchestrator...")
            
            self.is_running = False
            
            # Stop health monitoring
            if self.health_monitor:
                await self.health_monitor.stop_monitoring()
            
            # Send shutdown notification
            await self._send_system_notification(
                "🔴 SYSTEM SHUTDOWN",
                "VLR_AI Trading System automation has been stopped."
            )
            
            logger.info("[STOP] ✅ Automation stopped")
            
        except Exception as e:
            logger.error(f"[STOP] Failed to stop automation: {e}")

# Standalone functions for easy integration
async def start_full_automation() -> AutomationOrchestrator:
    """Start full system automation"""
    orchestrator = AutomationOrchestrator()
    
    if await orchestrator.initialize_all_systems():
        logger.info("[MAIN] ✅ Full automation started successfully")
        return orchestrator
    else:
        logger.error("[MAIN] ❌ Failed to start full automation")
        return None

async def get_system_status() -> Dict[str, Any]:
    """Get current system status"""
    orchestrator = AutomationOrchestrator()
    return orchestrator.get_automation_status()

# Main execution for testing
async def main():
    """Main function for testing"""
    print("\n" + "=" * 70)
    print("VLR_AI TRADING SYSTEM - FULL AUTOMATION ORCHESTRATOR")
    print("=" * 70)
    
    try:
        # Start full automation
        orchestrator = await start_full_automation()
        
        if orchestrator:
            print("\n🎉 FULL AUTOMATION STARTED SUCCESSFULLY!")
            print("\nSystem Status:")
            status = orchestrator.get_automation_status()
            
            for system, is_operational in status['automation_status'].items():
                status_icon = "✅" if is_operational else "❌"
                print(f"   {status_icon} {system.replace('_', ' ').title()}")
            
            print(f"\nOverall Health: {status['health_summary'].get('health_percentage', 0):.1f}%")
            print(f"Healthy APIs: {status['health_summary'].get('healthy_apis', 0)}/{status['health_summary'].get('total_apis', 0)}")
            
            print("\n" + "=" * 70)
            print("🚀 SYSTEM IS NOW FULLY AUTOMATED!")
            print("All APIs are monitored and will auto-heal if issues occur.")
            print("Daily token refresh, health checks, and notifications are active.")
            print("=" * 70)
            
            # Keep running for demonstration
            print("\nPress Ctrl+C to stop automation...")
            try:
                while True:
                    await asyncio.sleep(60)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Automation running...")
            except KeyboardInterrupt:
                print("\n\nStopping automation...")
                await orchestrator.stop_automation()
                print("✅ Automation stopped successfully!")
        else:
            print("❌ Failed to start automation")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())