"""
API Health Monitoring System
Comprehensive monitoring and auto-healing for all API integrations
"""

import os
import sys
import json
import asyncio
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import schedule
import time
from threading import Thread

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

# Import our enhanced authentication
from auth.enhanced_kite_auth import EnhancedKiteAuthenticator
from brokers.dhan_integration import DhanAPIClient

logger = logging.getLogger('trading_system.api_health_monitor')

@dataclass
class APIHealthStatus:
    """API health status data structure"""
    api_name: str
    is_healthy: bool
    last_check: datetime
    response_time: float
    error_message: Optional[str] = None
    consecutive_failures: int = 0
    uptime_percentage: float = 100.0
    last_success: Optional[datetime] = None

class APIHealthMonitor:
    """Comprehensive API health monitoring and auto-healing system"""
    
    def __init__(self):
        self.health_status: Dict[str, APIHealthStatus] = {}
        self.health_history: List[Dict[str, Any]] = []
        self.monitoring_active = False
        self.auto_healing_enabled = True
        self.notification_enabled = True
        
        # Health check intervals (seconds)
        self.check_intervals = {
            'kite': 300,      # 5 minutes
            'dhan': 180,      # 3 minutes
            'telegram': 600,  # 10 minutes
            'openai': 900,    # 15 minutes
            'perplexity': 900, # 15 minutes
            'alpha_vantage': 1800  # 30 minutes
        }
        
        # Failure thresholds for auto-healing
        self.failure_thresholds = {
            'kite': 3,
            'dhan': 2,
            'telegram': 5,
            'openai': 3,
            'perplexity': 3,
            'alpha_vantage': 2
        }
        
        # Initialize health status
        self._initialize_health_status()
        
        logger.info("[MONITOR] API Health Monitor initialized")
    
    def _initialize_health_status(self):
        """Initialize health status for all APIs"""
        apis = ['kite', 'dhan', 'telegram', 'openai', 'perplexity', 'alpha_vantage']
        
        for api in apis:
            self.health_status[api] = APIHealthStatus(
                api_name=api,
                is_healthy=False,
                last_check=datetime.now(),
                response_time=0.0,
                last_success=None
            )
    
    async def start_monitoring(self):
        """Start continuous API health monitoring"""
        try:
            logger.info("[MONITOR] Starting API health monitoring...")
            self.monitoring_active = True
            
            # Initial health check
            await self.comprehensive_health_check()
            
            # Schedule periodic checks
            self._schedule_health_checks()
            
            # Start monitoring loop
            await self._monitoring_loop()
            
        except Exception as e:
            logger.error(f"[MONITOR] Failed to start monitoring: {e}")
    
    def _schedule_health_checks(self):
        """Schedule periodic health checks for each API"""
        try:
            # Schedule Kite checks every 5 minutes
            schedule.every(5).minutes.do(lambda: asyncio.create_task(self.check_kite_health()))
            
            # Schedule Dhan checks every 3 minutes
            schedule.every(3).minutes.do(lambda: asyncio.create_task(self.check_dhan_health()))
            
            # Schedule Telegram checks every 10 minutes
            schedule.every(10).minutes.do(lambda: asyncio.create_task(self.check_telegram_health()))
            
            # Schedule AI API checks every 15 minutes
            schedule.every(15).minutes.do(lambda: asyncio.create_task(self.check_ai_apis_health()))
            
            # Schedule Alpha Vantage checks every 30 minutes
            schedule.every(30).minutes.do(lambda: asyncio.create_task(self.check_alpha_vantage_health()))
            
            # Schedule comprehensive check every hour
            schedule.every().hour.do(lambda: asyncio.create_task(self.comprehensive_health_check()))
            
            logger.info("[SCHEDULE] Health check schedules configured")
            
        except Exception as e:
            logger.error(f"[SCHEDULE] Failed to schedule health checks: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.monitoring_active:
                # Run scheduled checks
                schedule.run_pending()
                
                # Check for auto-healing triggers
                await self._check_auto_healing_triggers()
                
                # Sleep for 1 minute
                await asyncio.sleep(60)
                
        except Exception as e:
            logger.error(f"[LOOP] Monitoring loop error: {e}")
    
    async def comprehensive_health_check(self) -> Dict[str, APIHealthStatus]:
        """Perform comprehensive health check on all APIs"""
        try:
            logger.info("[HEALTH] Starting comprehensive health check...")
            
            # Check all APIs concurrently
            tasks = [
                self.check_kite_health(),
                self.check_dhan_health(),
                self.check_telegram_health(),
                self.check_ai_apis_health(),
                self.check_alpha_vantage_health()
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Generate health report
            await self._generate_health_report()
            
            # Send notifications if needed
            if self.notification_enabled:
                await self._send_health_notifications()
            
            logger.info("[HEALTH] Comprehensive health check completed")
            return self.health_status
            
        except Exception as e:
            logger.error(f"[HEALTH] Comprehensive health check failed: {e}")
            return self.health_status
    
    async def check_kite_health(self) -> bool:
        """Check Kite API health"""
        start_time = time.time()
        api_name = 'kite'
        
        try:
            logger.debug("[KITE] Checking Kite API health...")
            
            # Try to authenticate and test connection
            auth = EnhancedKiteAuthenticator()
            
            if await auth.authenticate():
                # Test basic API calls
                profile = await auth.get_profile()
                funds = await auth.get_funds()
                
                if profile and funds:
                    response_time = time.time() - start_time
                    await self._update_health_status(api_name, True, response_time)
                    logger.debug("[KITE] Health check passed")
                    return True
                else:
                    await self._update_health_status(api_name, False, 0, "API calls failed")
                    return False
            else:
                await self._update_health_status(api_name, False, 0, "Authentication failed")
                return False
                
        except Exception as e:
            response_time = time.time() - start_time
            await self._update_health_status(api_name, False, response_time, str(e))
            logger.error(f"[KITE] Health check failed: {e}")
            return False
    
    async def check_dhan_health(self) -> bool:
        """Check Dhan API health with retry/backoff"""
        from utils.error_recovery import RetryManager
        async def _check():
        """Check Dhan API health"""
        start_time = time.time()
        api_name = 'dhan'
        
        try:
            logger.debug("[DHAN] Checking Dhan API health...")
            return await RetryManager.retry_with_backoff(_check, max_retries=3)
        except Exception as e:
            response_time = time.time() - time.time()
            await self._update_health_status('dhan', False, response_time, str(e))
            logger.error(f"[DHAN] Health check failed: {e}")
            return False

        # --- original code below, now inside _check() ---
        #
            
            client_id = os.getenv('DHAN_CLIENT_ID')
            access_token = os.getenv('DHAN_ACCESS_TOKEN')
            
            if not client_id or not access_token:
                await self._update_health_status(api_name, False, 0, "Missing credentials")
                return False
            
            # Test Dhan API
            dhan_client = DhanAPIClient(client_id, access_token)
            
            # Test profile endpoint
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'access-token': access_token
            }
            
            response = requests.get(
                'https://api.dhan.co/v2/profile',
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                response_time = time.time() - start_time
                await self._update_health_status(api_name, True, response_time)
                logger.debug("[DHAN] Health check passed")
                return True
            else:
                await self._update_health_status(api_name, False, 0, f"API error: {response.status_code}")
                return False
                
        except Exception as e:
            response_time = time.time() - start_time
            await self._update_health_status(api_name, False, response_time, str(e))
            logger.error(f"[DHAN] Health check failed: {e}")
            return False
    
    async def check_telegram_health(self) -> bool:
        """Check Telegram Bot API health"""
        start_time = time.time()
        api_name = 'telegram'
        
        try:
            logger.debug("[TELEGRAM] Checking Telegram API health...")
            
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if not bot_token or not chat_id:
                await self._update_health_status(api_name, False, 0, "Missing credentials")
                return False
            
            # Test bot info
            response = requests.get(
                f'https://api.telegram.org/bot{bot_token}/getMe',
                timeout=10
            )
            
            if response.status_code == 200:
                bot_info = response.json()
                if bot_info.get('ok'):
                    response_time = time.time() - start_time
                    await self._update_health_status(api_name, True, response_time)
                    logger.debug("[TELEGRAM] Health check passed")
                    return True
                else:
                    await self._update_health_status(api_name, False, 0, "Bot not responding")
                    return False
            else:
                await self._update_health_status(api_name, False, 0, f"API error: {response.status_code}")
                return False
                
        except Exception as e:
            response_time = time.time() - start_time
            await self._update_health_status(api_name, False, response_time, str(e))
            logger.error(f"[TELEGRAM] Health check failed: {e}")
            return False
    
    async def check_ai_apis_health(self) -> bool:
        """Check AI APIs (OpenAI and Perplexity) health"""
        openai_healthy = await self._check_openai_health()
        perplexity_healthy = await self._check_perplexity_health()
        
        return openai_healthy or perplexity_healthy  # At least one should work
    
    async def _check_openai_health(self) -> bool:
        """Check OpenAI API health"""
        start_time = time.time()
        api_name = 'openai'
        
        try:
            logger.debug("[OPENAI] Checking OpenAI API health...")
            
            openai_key = os.getenv('OPENAI_API_KEY')
            if not openai_key:
                await self._update_health_status(api_name, False, 0, "Missing API key")
                return False
            
            headers = {
                'Authorization': f'Bearer {openai_key}',
                'Content-Type': 'application/json'
            }
            
            # Test with minimal request
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json={
                    'model': 'gpt-3.5-turbo',
                    'messages': [{'role': 'user', 'content': 'Hi'}],
                    'max_tokens': 1
                },
                timeout=15
            )
            
            if response.status_code == 200:
                response_time = time.time() - start_time
                await self._update_health_status(api_name, True, response_time)
                logger.debug("[OPENAI] Health check passed")
                return True
            elif response.status_code == 429:
                # Rate limited but API is working
                response_time = time.time() - start_time
                await self._update_health_status(api_name, True, response_time, "Rate limited")
                logger.debug("[OPENAI] Health check passed (rate limited)")
                return True
            else:
                await self._update_health_status(api_name, False, 0, f"API error: {response.status_code}")
                return False
                
        except Exception as e:
            response_time = time.time() - start_time
            await self._update_health_status(api_name, False, response_time, str(e))
            logger.error(f"[OPENAI] Health check failed: {e}")
            return False
    
    async def _check_perplexity_health(self) -> bool:
        """Check Perplexity API health"""
        start_time = time.time()
        api_name = 'perplexity'
        
        try:
            logger.debug("[PERPLEXITY] Checking Perplexity API health...")
            
            perplexity_key = os.getenv('PERPLEXITY_API_KEY')
            if not perplexity_key:
                await self._update_health_status(api_name, False, 0, "Missing API key")
                return False
            
            headers = {
                'Authorization': f'Bearer {perplexity_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                'https://api.perplexity.ai/chat/completions',
                headers=headers,
                json={
                    'model': 'llama-3.1-sonar-small-128k-online',
                    'messages': [{'role': 'user', 'content': 'Hi'}],
                    'max_tokens': 1
                },
                timeout=15
            )
            
            if response.status_code == 200:
                response_time = time.time() - start_time
                await self._update_health_status(api_name, True, response_time)
                logger.debug("[PERPLEXITY] Health check passed")
                return True
            else:
                await self._update_health_status(api_name, False, 0, f"API error: {response.status_code}")
                return False
                
        except Exception as e:
            response_time = time.time() - start_time
            await self._update_health_status(api_name, False, response_time, str(e))
            logger.error(f"[PERPLEXITY] Health check failed: {e}")
            return False
    
    async def check_alpha_vantage_health(self) -> bool:
        """Check Alpha Vantage API health"""
        start_time = time.time()
        api_name = 'alpha_vantage'
        
        try:
            logger.debug("[ALPHA] Checking Alpha Vantage API health...")
            
            api_key = os.getenv('ALPHA_VANTAGE_KEY')
            if not api_key:
                await self._update_health_status(api_name, False, 0, "Missing API key")
                return False
            
            response = requests.get(
                f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey={api_key}',
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'Global Quote' in data:
                    response_time = time.time() - start_time
                    await self._update_health_status(api_name, True, response_time)
                    logger.debug("[ALPHA] Health check passed")
                    return True
                elif 'Note' in data:
                    # Rate limited but API is working
                    response_time = time.time() - start_time
                    await self._update_health_status(api_name, True, response_time, "Rate limited")
                    logger.debug("[ALPHA] Health check passed (rate limited)")
                    return True
                else:
                    await self._update_health_status(api_name, False, 0, "Invalid response")
                    return False
            else:
                await self._update_health_status(api_name, False, 0, f"API error: {response.status_code}")
                return False
                
        except Exception as e:
            response_time = time.time() - start_time
            await self._update_health_status(api_name, False, response_time, str(e))
            logger.error(f"[ALPHA] Health check failed: {e}")
            return False
    
    async def _update_health_status(self, api_name: str, is_healthy: bool, response_time: float, error_message: str = None):
        """Update health status for an API"""
        try:
            current_status = self.health_status.get(api_name)
            if not current_status:
                return
            
            # Update basic status
            current_status.is_healthy = is_healthy
            current_status.last_check = datetime.now()
            current_status.response_time = response_time
            current_status.error_message = error_message
            
            if is_healthy:
                current_status.consecutive_failures = 0
                current_status.last_success = datetime.now()
            else:
                current_status.consecutive_failures += 1
            
            # Calculate uptime percentage (last 24 hours)
            await self._calculate_uptime(api_name)
            
            # Log status change
            status_text = "HEALTHY" if is_healthy else "UNHEALTHY"
            logger.info(f"[STATUS] {api_name.upper()}: {status_text} (Response: {response_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"[UPDATE] Failed to update health status for {api_name}: {e}")
    
    async def _calculate_uptime(self, api_name: str):
        """Calculate uptime percentage for an API"""
        try:
            # This is a simplified calculation
            # In production, you'd store historical data
            current_status = self.health_status.get(api_name)
            if not current_status:
                return
            
            if current_status.consecutive_failures == 0:
                current_status.uptime_percentage = min(100.0, current_status.uptime_percentage + 1.0)
            else:
                current_status.uptime_percentage = max(0.0, current_status.uptime_percentage - 2.0)
                
        except Exception as e:
            logger.error(f"[UPTIME] Failed to calculate uptime for {api_name}: {e}")
    
    async def _check_auto_healing_triggers(self):
        """Check if auto-healing should be triggered"""
        try:
            if not self.auto_healing_enabled:
                return
            
            for api_name, status in self.health_status.items():
                threshold = self.failure_thresholds.get(api_name, 3)
                
                if status.consecutive_failures >= threshold:
                    logger.warning(f"[HEALING] Auto-healing triggered for {api_name}")
                    await self._trigger_auto_healing(api_name)
                    
        except Exception as e:
            logger.error(f"[HEALING] Auto-healing check failed: {e}")
    
    async def _trigger_auto_healing(self, api_name: str):
        """Trigger auto-healing for a specific API"""
        try:
            logger.info(f"[HEALING] Starting auto-healing for {api_name}...")
            
            if api_name == 'kite':
                # Force refresh Kite token
                auth = EnhancedKiteAuthenticator()
                success = await auth.force_refresh_token()
                if success:
                    logger.info("[HEALING] Kite token refreshed successfully")
                    # Reset failure count
                    self.health_status[api_name].consecutive_failures = 0
                else:
                    logger.error("[HEALING] Kite token refresh failed")
            
            elif api_name == 'dhan':
                # For Dhan, just wait and retry (token might be temporarily invalid)
                logger.info("[HEALING] Waiting before retrying Dhan...")
                await asyncio.sleep(60)
                await self.check_dhan_health()
            
            elif api_name == 'telegram':
                # Test Telegram connection
                logger.info("[HEALING] Testing Telegram connection...")
                await self.check_telegram_health()
            
            # Add more auto-healing strategies as needed
            
        except Exception as e:
            logger.error(f"[HEALING] Auto-healing failed for {api_name}: {e}")
    
    async def _generate_health_report(self):
        """Generate comprehensive health report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'overall_health': self._calculate_overall_health(),
                'api_status': {name: asdict(status) for name, status in self.health_status.items()},
                'summary': self._generate_health_summary()
            }
            
            # Save report to file
            report_file = PROJECT_ROOT / 'logs' / f'health_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            report_file.parent.mkdir(exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"[REPORT] Health report saved to: {report_file}")
            
        except Exception as e:
            logger.error(f"[REPORT] Failed to generate health report: {e}")
    
    def _calculate_overall_health(self) -> float:
        """Calculate overall system health percentage"""
        try:
            if not self.health_status:
                return 0.0
            
            healthy_apis = sum(1 for status in self.health_status.values() if status.is_healthy)
            total_apis = len(self.health_status)
            
            return (healthy_apis / total_apis) * 100.0
            
        except Exception as e:
            logger.error(f"[HEALTH] Failed to calculate overall health: {e}")
            return 0.0
    
    def _generate_health_summary(self) -> Dict[str, Any]:
        """Generate health summary"""
        try:
            healthy_count = sum(1 for status in self.health_status.values() if status.is_healthy)
            total_count = len(self.health_status)
            
            return {
                'healthy_apis': healthy_count,
                'total_apis': total_count,
                'health_percentage': self._calculate_overall_health(),
                'critical_issues': [
                    name for name, status in self.health_status.items()
                    if not status.is_healthy and status.consecutive_failures >= 3
                ],
                'average_response_time': sum(
                    status.response_time for status in self.health_status.values()
                ) / total_count if total_count > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"[SUMMARY] Failed to generate health summary: {e}")
            return {}
    
    async def _send_health_notifications(self):
        """Send health notifications via Telegram"""
        try:
            # Only send notifications for critical issues
            critical_issues = [
                name for name, status in self.health_status.items()
                if not status.is_healthy and status.consecutive_failures >= 2
            ]
            
            if critical_issues:
                message = f"🚨 API HEALTH ALERT\n\nCritical Issues:\n"
                for api in critical_issues:
                    status = self.health_status[api]
                    message += f"• {api.upper()}: {status.consecutive_failures} failures\n"
                
                message += f"\nOverall Health: {self._calculate_overall_health():.1f}%"
                message += f"\nTime: {datetime.now().strftime('%H:%M:%S')}"
                
                # Send via Telegram (if available)
                await self._send_telegram_notification(message)
                
        except Exception as e:
            logger.error(f"[NOTIFY] Failed to send health notifications: {e}")
    
    async def _send_telegram_notification(self, message: str):
        """Send notification via Telegram"""
        try:
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if not bot_token or not chat_id:
                return
            
            response = requests.post(
                f'https://api.telegram.org/bot{bot_token}/sendMessage',
                json={
                    'chat_id': chat_id,
                    'text': message
                },
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("[NOTIFY] Health notification sent via Telegram")
            else:
                logger.error(f"[NOTIFY] Failed to send Telegram notification: {response.status_code}")
                
        except Exception as e:
            logger.error(f"[NOTIFY] Telegram notification failed: {e}")
    
    def get_health_status(self) -> Dict[str, APIHealthStatus]:
        """Get current health status"""
        return self.health_status
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary"""
        return self._generate_health_summary()
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        try:
            logger.info("[MONITOR] Stopping API health monitoring...")
            self.monitoring_active = False
            
        except Exception as e:
            logger.error(f"[MONITOR] Failed to stop monitoring: {e}")

# Standalone functions for easy integration
async def run_health_check() -> Dict[str, APIHealthStatus]:
    """Run a one-time comprehensive health check"""
    monitor = APIHealthMonitor()
    return await monitor.comprehensive_health_check()

async def start_continuous_monitoring():
    """Start continuous API monitoring"""
    monitor = APIHealthMonitor()
    await monitor.start_monitoring()

# Main execution for testing
async def main():
    """Main function for testing"""
    print("\n" + "=" * 60)
    print("API HEALTH MONITORING SYSTEM")
    print("=" * 60)
    
    try:
        monitor = APIHealthMonitor()
        
        print("\n1. Running comprehensive health check...")
        health_status = await monitor.comprehensive_health_check()
        
        print(f"\n2. Health Summary:")
        summary = monitor.get_health_summary()
        print(f"   Healthy APIs: {summary['healthy_apis']}/{summary['total_apis']}")
        print(f"   Health Percentage: {summary['health_percentage']:.1f}%")
        print(f"   Average Response Time: {summary['average_response_time']:.2f}s")
        
        print(f"\n3. Individual API Status:")
        for api_name, status in health_status.items():
            status_icon = "✅" if status.is_healthy else "❌"
            print(f"   {status_icon} {api_name.upper()}: {status.response_time:.2f}s")
            if status.error_message:
                print(f"      Error: {status.error_message}")
        
        print("\n" + "=" * 60)
        print("🎉 API HEALTH MONITORING COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())