"""
Google Sheets Integration Service for VLR_AI Trading System
Main service that coordinates data collection and sheet updates
"""

import asyncio
import logging
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import pytz

from utils.google_sheets_manager import GoogleSheetsManager
from utils.data_collector import EnhancedDataCollector
from notifications.telegram_notifier import TelegramNotifier

logger = logging.getLogger('trading_system.sheets_integration')

class SheetsIntegrationService:
    """Main service for Google Sheets integration"""
    
    def __init__(self, settings, kite_client=None):
        self.settings = settings
        self.kite_client = kite_client
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Initialize components
        self.sheets_manager = GoogleSheetsManager(settings)
        self.data_collector = EnhancedDataCollector(settings, kite_client)
        self.telegram = TelegramNotifier(settings) if settings.TELEGRAM_BOT_TOKEN else None
        
        # Service state
        self.running = False
        self.update_tasks = []
        
        # Update intervals (in seconds)
        self.update_intervals = {
            'market_hours': 300,      # 5 minutes during market hours
            'pre_post_market': 900,   # 15 minutes during pre/post market
            'off_market': 7200,       # 2 hours when market is closed
            'weekend': 14400          # 4 hours on weekends
        }
        
        # Data tracking
        self.last_updates = {}
        self.trade_history = []
        self.rejected_signals = []
        
        logger.info("Sheets Integration Service initialized")
    
    async def initialize(self):
        """Initialize the sheets integration service"""
        try:
            logger.info("Initializing Sheets Integration Service...")
            
            # Initialize Google Sheets manager
            sheets_success = await self.sheets_manager.initialize()
            if not sheets_success:
                logger.info("Google Sheets integration disabled - continuing without it")
                return True  # Return True to continue without sheets
            
            # Send startup notification
            if self.telegram:
                await self._send_startup_notification()
            
            logger.info("Sheets Integration Service initialized successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Sheets Integration Service failed - continuing without it: {e}")
            return True  # Return True to continue without sheets
    
    async def start_continuous_updates(self):
        """Start continuous data updates to Google Sheets"""
        try:
            self.running = True
            logger.info("Starting continuous Google Sheets updates...")
            
            # Start update tasks
            self.update_tasks = [
                asyncio.create_task(self._market_context_updater()),
                asyncio.create_task(self._options_data_updater()),
                asyncio.create_task(self._institutional_flow_updater()),
                asyncio.create_task(self._news_sentiment_updater()),
                asyncio.create_task(self._performance_analytics_updater()),
                asyncio.create_task(self._system_health_monitor())
            ]
            
            # Wait for all tasks
            await asyncio.gather(*self.update_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error in continuous updates: {e}")
        finally:
            self.running = False
    
    async def _market_context_updater(self):
        """Continuously update market context data"""
        while self.running:
            try:
                # Collect market context data
                market_data = await self.data_collector.collect_market_context()
                
                if market_data:
                    # Log to Google Sheets
                    success = await self.sheets_manager.log_market_context(market_data)
                    
                    if success:
                        self.last_updates['market_context'] = datetime.now(self.ist)
                        logger.info("Market context updated successfully")
                    else:
                        logger.warning("Failed to update market context")
                
                # Wait for next update
                await asyncio.sleep(self._get_update_interval())
                
            except Exception as e:
                logger.error(f"Error in market context updater: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _options_data_updater(self):
        """Continuously update options chain data"""
        while self.running:
            try:
                # Only update during market hours for options data
                if self.data_collector.is_market_hours():
                    options_data = await self.data_collector.collect_options_data()
                    
                    if options_data:
                        success = await self.sheets_manager.log_options_data(options_data)
                        
                        if success:
                            self.last_updates['options_data'] = datetime.now(self.ist)
                            logger.info(f"Options data updated: {len(options_data)} entries")
                        else:
                            logger.warning("Failed to update options data")
                
                # Wait for next update
                await asyncio.sleep(self._get_update_interval())
                
            except Exception as e:
                logger.error(f"Error in options data updater: {e}")
                await asyncio.sleep(60)
    
    async def _institutional_flow_updater(self):
        """Update institutional flow data (daily)"""
        while self.running:
            try:
                # Update once per day
                last_update = self.last_updates.get('institutional_flow')
                now = datetime.now(self.ist)
                
                if not last_update or (now - last_update).days >= 1:
                    flow_data = await self.data_collector.collect_institutional_flow()
                    
                    if flow_data:
                        success = await self.sheets_manager.log_institutional_flow(flow_data)
                        
                        if success:
                            self.last_updates['institutional_flow'] = now
                            logger.info("Institutional flow data updated")
                
                # Wait 1 hour before checking again
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in institutional flow updater: {e}")
                await asyncio.sleep(3600)
    
    async def _news_sentiment_updater(self):
        """Update news and sentiment data"""
        while self.running:
            try:
                news_data = await self.data_collector.collect_news_sentiment()
                
                for news_item in news_data:
                    success = await self.sheets_manager.log_news_sentiment(news_item)
                    
                    if success:
                        logger.info("News sentiment updated")
                
                self.last_updates['news_sentiment'] = datetime.now(self.ist)
                
                # Wait for next update (every 30 minutes)
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"Error in news sentiment updater: {e}")
                await asyncio.sleep(1800)
    
    async def _performance_analytics_updater(self):
        """Update daily performance analytics"""
        while self.running:
            try:
                # Update once per day at market close
                now = datetime.now(self.ist)
                
                # Check if it's after market hours (after 4 PM)
                if now.hour >= 16:
                    last_update = self.last_updates.get('performance_analytics')
                    
                    if not last_update or last_update.date() < now.date():
                        performance_data = await self.data_collector.calculate_performance_analytics(self.trade_history)
                        
                        if performance_data:
                            success = await self.sheets_manager.log_performance_analytics(performance_data)
                            
                            if success:
                                self.last_updates['performance_analytics'] = now
                                logger.info("Performance analytics updated")
                
                # Wait 1 hour before checking again
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in performance analytics updater: {e}")
                await asyncio.sleep(3600)
    
    async def _system_health_monitor(self):
        """Monitor and log system health"""
        while self.running:
            try:
                # Get system health status
                health_status = await self.sheets_manager.get_sheet_health_status()
                
                # Log system health
                await self.sheets_manager._log_system_health(
                    "Sheets Integration Service",
                    health_status['status'],
                    f"Error count: {health_status['error_count']}, Response time: {health_status.get('api_response_time', 0)}ms",
                    "Info" if health_status['status'] == 'Healthy' else "Warning"
                )
                
                # Send alert if critical
                if health_status['status'] == 'Critical' and self.telegram:
                    await self.telegram.send_message(
                        f"🚨 CRITICAL: Google Sheets Integration\n"
                        f"Status: {health_status['status']}\n"
                        f"Error Count: {health_status['error_count']}\n"
                        f"Time: {datetime.now(self.ist).strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                
                # Wait 5 minutes before next health check
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in system health monitor: {e}")
                await asyncio.sleep(300)
    
    def _get_update_interval(self) -> int:
        """Get appropriate update interval based on market session"""
        try:
            now = datetime.now(self.ist)
            
            # Check if it's weekend
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return self.update_intervals['weekend']
            
            # Check market session
            current_time = now.time()
            market_start = datetime.strptime("09:15", "%H:%M").time()
            market_end = datetime.strptime("15:30", "%H:%M").time()
            pre_market_start = datetime.strptime("09:00", "%H:%M").time()
            post_market_end = datetime.strptime("16:00", "%H:%M").time()
            
            if market_start <= current_time <= market_end:
                return self.update_intervals['market_hours']
            elif pre_market_start <= current_time < market_start or market_end < current_time <= post_market_end:
                return self.update_intervals['pre_post_market']
            else:
                return self.update_intervals['off_market']
                
        except Exception as e:
            logger.warning(f"Failed to determine update interval: {e}")
            return self.update_intervals['market_hours']  # Default
    
    async def log_trade_signal(self, signal_data: Dict[str, Any]):
        """Log a new trade signal"""
        try:
            # Add to trade history
            self.trade_history.append(signal_data)
            
            # Log to Google Sheets
            success = await self.sheets_manager.log_trade_signal(signal_data)
            
            if success:
                logger.info(f"Trade signal logged: {signal_data.get('signal_id')}")
                
                # Send Telegram notification for high-confidence signals
                if (signal_data.get('confidence_score', 0) >= 80 and self.telegram):
                    await self._send_signal_notification(signal_data)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to log trade signal: {e}")
            return False
    
    async def log_rejected_signal(self, rejected_data: Dict[str, Any]):
        """Log a rejected trade signal"""
        try:
            # Add to rejected signals
            self.rejected_signals.append(rejected_data)
            
            # Log to Google Sheets
            success = await self.sheets_manager.log_rejected_signal(rejected_data)
            
            if success:
                logger.info(f"Rejected signal logged: {rejected_data.get('signal_id')}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to log rejected signal: {e}")
            return False
    
    async def update_signal_status(self, signal_id: str, status: str, exit_time: str = None, pnl: float = None):
        """Update the status of an existing signal"""
        try:
            # Update in trade history
            for trade in self.trade_history:
                if trade.get('signal_id') == signal_id:
                    trade['status'] = status
                    if exit_time:
                        trade['exit_time'] = exit_time
                    if pnl is not None:
                        trade['pnl'] = pnl
                    break
            
            # Update in Google Sheets
            success = await self.sheets_manager.update_signal_status(signal_id, status, exit_time, pnl)
            
            if success:
                logger.info(f"Signal {signal_id} status updated to {status}")
                
                # Send notification for completed trades
                if status in ['Hit Target 1', 'Hit Target 2', 'Hit SL'] and self.telegram:
                    await self._send_trade_completion_notification(signal_id, status, pnl)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update signal status: {e}")
            return False
    
    async def _send_startup_notification(self):
        """Send startup notification"""
        try:
            message = f"""
📊 GOOGLE SHEETS INTEGRATION ACTIVATED

🕐 Time: {datetime.now(self.ist).strftime('%Y-%m-%d %H:%M:%S')} IST
📈 Sheet: VLR_AI Live Trade Dashboard
🔗 URL: {self.sheets_manager.sheet_url}

✅ Real-time Data Logging:
• Trade Signals & Status Updates
• Live Market Context (NIFTY, BANKNIFTY, Global)
• Options Chain Data (Top 5 strikes each side)
• Institutional Flow (FII/DII)
• News Sentiment Analysis
• Performance Analytics
• System Health Monitoring

🔄 Update Schedule:
• Market Hours: Every 5 minutes
• Pre/Post Market: Every 15 minutes
• Off Market: Every 2 hours

Your trading data is now being logged automatically!
"""
            
            await self.telegram.send_message(message)
            logger.info("Startup notification sent")
            
        except Exception as e:
            logger.warning(f"Failed to send startup notification: {e}")
    
    async def _send_signal_notification(self, signal_data: Dict[str, Any]):
        """Send notification for high-confidence signals"""
        try:
            message = f"""
🎯 HIGH CONFIDENCE SIGNAL LOGGED

📊 Signal ID: {signal_data.get('signal_id')}
📈 Instrument: {signal_data.get('instrument')}
🎯 Direction: {signal_data.get('direction')}
💰 Strike: {signal_data.get('strike_price')}
📅 Expiry: {signal_data.get('expiry_date')}

💡 Confidence: {signal_data.get('confidence_score')}%
⚠️ Risk Score: {signal_data.get('risk_score')}
📝 Reason: {signal_data.get('reason_summary')}

✅ Logged to Google Sheets automatically
"""
            
            await self.telegram.send_message(message)
            
        except Exception as e:
            logger.warning(f"Failed to send signal notification: {e}")
    
    async def _send_trade_completion_notification(self, signal_id: str, status: str, pnl: float):
        """Send notification when trade is completed"""
        try:
            pnl_emoji = "💰" if pnl and pnl > 0 else "📉"
            
            message = f"""
{pnl_emoji} TRADE COMPLETED

📊 Signal ID: {signal_id}
📈 Status: {status}
💰 P&L: ₹{pnl if pnl else 'N/A'}

✅ Updated in Google Sheets automatically
"""
            
            await self.telegram.send_message(message)
            
        except Exception as e:
            logger.warning(f"Failed to send trade completion notification: {e}")
    
    async def cleanup_old_data(self):
        """Clean up old data from sheets"""
        try:
            success = await self.sheets_manager.cleanup_old_data(days_to_keep=90)
            
            if success:
                logger.info("Old data cleanup completed")
                
                if self.telegram:
                    await self.telegram.send_message(
                        f"🧹 DATA CLEANUP COMPLETED\n"
                        f"Removed data older than 90 days\n"
                        f"Time: {datetime.now(self.ist).strftime('%Y-%m-%d %H:%M:%S')}"
                    )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return False
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get overall integration status"""
        try:
            sheets_health = await self.sheets_manager.get_sheet_health_status()
            
            status = {
                'service_running': self.running,
                'sheets_status': sheets_health,
                'last_updates': self.last_updates,
                'trade_history_count': len(self.trade_history),
                'rejected_signals_count': len(self.rejected_signals),
                'update_interval': self._get_update_interval()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get integration status: {e}")
            return {'error': str(e)}
    
    async def stop(self):
        """Stop the sheets integration service"""
        try:
            logger.info("Stopping Sheets Integration Service...")
            
            self.running = False
            
            # Cancel all update tasks
            for task in self.update_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.update_tasks:
                await asyncio.gather(*self.update_tasks, return_exceptions=True)
            
            # Send shutdown notification
            if self.telegram:
                await self.telegram.send_message(
                    f"📊 Google Sheets Integration Service stopped\n"
                    f"Time: {datetime.now(self.ist).strftime('%Y-%m-%d %H:%M:%S')}"
                )
            
            # Shutdown sheets manager
            await self.sheets_manager.shutdown()
            
            logger.info("Sheets Integration Service stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping Sheets Integration Service: {e}")