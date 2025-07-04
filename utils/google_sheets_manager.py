"""
Enhanced Google Sheets Integration for VLR_AI Trading System
Comprehensive data logging and real-time updates to Google Sheets
"""

import asyncio
import logging
import json
import gspread
from google.auth.exceptions import RefreshError
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time
import traceback
from pathlib import Path
import pytz

logger = logging.getLogger('trading_system.google_sheets')

class GoogleSheetsManager:
    """Enhanced Google Sheets Manager for VLR_AI Trading System"""
    
    def __init__(self, settings):
        self.settings = settings
        self.project_root = Path(__file__).parent.parent
        self.credentials_path = self.project_root / "google_credentials.json"
        
        # Sheet configuration
        self.sheet_url = "https://docs.google.com/spreadsheets/d/1cRUP3VnM5JcjFyFZTholuQfmEXInTAXz_14V_OZP67M/edit?gid=365548845#gid=365548845"
        self.sheet_id = "1cRUP3VnM5JcjFyFZTholuQfmEXInTAXz_14V_OZP67M"
        
        # Initialize Google Sheets client
        self.gc = None
        self.spreadsheet = None
        self.worksheets = {}
        
        # IST timezone
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Rate limiting
        self.last_api_call = 0
        self.min_api_interval = 1.0  # Minimum 1 second between API calls
        
        # Error tracking
        self.error_count = 0
        self.max_errors = 10
        
        # Tab names mapping
        self.tab_names = {
            'trade_signals': 'Trade Signals',
            'market_context': 'Market Context Snapshot',
            'options_data': 'Options Data',
            'institutional_flow': 'Institutional Flow (FII/DII)',
            'news_sentiment': 'News Sentiment',
            'rejected_signals': 'Rejected Trade Signals',
            'performance_analytics': 'Performance Analytics',
            'system_health': 'System Health & Alerts'
        }
        
        logger.info("GoogleSheetsManager initialized")
    
    async def initialize(self):
        """Initialize Google Sheets connection and setup worksheets"""
        try:
            logger.info("Initializing Google Sheets connection...")
            
            # Load credentials
            if not self.credentials_path.exists():
                logger.warning(f"Google credentials file not found: {self.credentials_path}")
                logger.info("Google Sheets integration disabled - continuing without it")
                return False
            
            # Setup Google Sheets API
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            
            credentials = Credentials.from_service_account_file(
                self.credentials_path, scopes=scope
            )
            
            self.gc = gspread.authorize(credentials)
            
            # Open the spreadsheet
            self.spreadsheet = self.gc.open_by_key(self.sheet_id)
            
            # Setup all worksheets
            await self._setup_worksheets()
            
            # Log system startup
            await self._log_system_health("Google Sheets", "Online", "Successfully initialized", "Info")
            
            logger.info("Google Sheets integration initialized successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Google Sheets initialization failed: {e}")
            logger.info("Continuing without Google Sheets integration")
            return False
    
    async def _setup_worksheets(self):
        """Setup all required worksheets with proper headers"""
        try:
            # Get existing worksheets
            existing_sheets = {ws.title: ws for ws in self.spreadsheet.worksheets()}
            
            # Setup each tab
            for tab_key, tab_name in self.tab_names.items():
                if tab_name not in existing_sheets:
                    # Create new worksheet
                    worksheet = self.spreadsheet.add_worksheet(title=tab_name, rows=1000, cols=30)
                    logger.info(f"Created new worksheet: {tab_name}")
                else:
                    worksheet = existing_sheets[tab_name]
                
                self.worksheets[tab_key] = worksheet
                
                # Setup headers for each tab
                await self._setup_tab_headers(tab_key, worksheet)
            
            logger.info("All worksheets setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup worksheets: {e}")
            raise
    
    async def _setup_tab_headers(self, tab_key: str, worksheet):
        """Setup headers for each specific tab"""
        try:
            headers = self._get_tab_headers(tab_key)
            
            # Check if headers already exist
            try:
                existing_headers = worksheet.row_values(1)
                if existing_headers == headers:
                    logger.info(f"Headers already exist for {tab_key}")
                    return
            except:
                pass
            
            # Set headers
            await self._rate_limited_call(worksheet.update, 'A1', [headers])
            
            # Apply formatting
            await self._apply_tab_formatting(tab_key, worksheet)
            
            logger.info(f"Headers setup completed for {tab_key}")
            
        except Exception as e:
            logger.error(f"Failed to setup headers for {tab_key}: {e}")
    
    def _get_tab_headers(self, tab_key: str) -> List[str]:
        """Get headers for each tab"""
        headers_map = {
            'trade_signals': [
                'Timestamp (IST)', 'Signal ID', 'Instrument', 'Signal Direction',
                'Strike Price', 'Expiry Date', 'Entry Price', 'Stop Loss',
                'Target 1', 'Target 2', 'Confidence Score (%)', 'Risk Score',
                'Reason Summary', 'Status', 'Entry Time', 'Exit Time',
                'P&L (₹)', 'Risk-Reward Ratio'
            ],
            'market_context': [
                'Timestamp (IST)', 'NIFTY Spot', 'BANKNIFTY Spot', 'India VIX',
                'NIFTY PE Ratio', 'NIFTY PB Ratio', 'SGX Nifty', 'Dow Jones',
                'Nasdaq', 'S&P 500', 'Dollar Index (DXY)', 'Crude Oil (Brent)',
                'Gold Price (₹/10g)', 'USDINR', '10Y Bond Yield (India)',
                'Global Sentiment', 'Market Session'
            ],
            'options_data': [
                'Timestamp (IST)', 'Instrument', 'Strike Price', 'Option Type',
                'LTP', 'Volume', 'OI (Open Interest)', 'Change in OI',
                'OI Change %', 'IV (Implied Volatility)', 'Delta', 'Gamma',
                'Theta', 'Vega', 'PCR (Put-Call Ratio)', 'Bid-Ask Spread'
            ],
            'institutional_flow': [
                'Date', 'FII Equity Buy Value (₹ Cr)', 'FII Equity Sell Value (₹ Cr)',
                'FII Equity Net', 'FII Derivative Buy Value (₹ Cr)', 'FII Derivative Sell Value (₹ Cr)',
                'FII Derivative Net', 'DII Equity Buy Value (₹ Cr)', 'DII Equity Sell Value (₹ Cr)',
                'DII Equity Net', 'Total FII+DII Net Flow', 'Cumulative Monthly Flow (FII)',
                'Cumulative Monthly Flow (DII)'
            ],
            'news_sentiment': [
                'Timestamp', 'Source', 'Category', 'Headline', 'Summary',
                'Sentiment Score (-100 to +100)', 'Market Impact', 'Confidence Level',
                'Related Stocks/Sectors'
            ],
            'rejected_signals': [
                'Timestamp (IST)', 'Signal ID', 'Instrument', 'Signal Direction',
                'Strike Price', 'Expiry Date', 'Proposed Entry Price', 'Proposed Stop Loss',
                'Proposed Target 1', 'Proposed Target 2', 'Risk Score', 'Confidence Score (%)',
                'Rejection Reason', 'Risk Cost (₹)', 'Max Drawdown Risk (₹)', 'Position Size (Lots)'
            ],
            'performance_analytics': [
                'Date', 'Total Signals Generated', 'Total Signals Executed', 'Total Signals Rejected',
                'Win Rate (%)', 'Average P&L per Trade (₹)', 'Max Profit (₹)', 'Max Loss (₹)',
                'Daily P&L (₹)', 'Cumulative P&L (₹)', 'Sharpe Ratio (Monthly)', 'Max Drawdown (₹)',
                'Success Rate by Instrument', 'Average Hold Time (minutes)'
            ],
            'system_health': [
                'Timestamp', 'System Component', 'Status', 'Error Message',
                'Response Time (ms)', 'Data Freshness', 'Alert Level', 'Action Required'
            ]
        }
        
        return headers_map.get(tab_key, [])
    
    async def _apply_tab_formatting(self, tab_key: str, worksheet):
        """Apply conditional formatting and styling to tabs"""
        try:
            # Freeze header row
            worksheet.freeze(rows=1)
            
            # Apply conditional formatting based on tab type
            if tab_key == 'trade_signals':
                # Color code P&L column (green for profit, red for loss)
                pass  # Will implement conditional formatting rules
            elif tab_key == 'news_sentiment':
                # Color code sentiment scores
                pass
            
            logger.info(f"Formatting applied to {tab_key}")
            
        except Exception as e:
            logger.warning(f"Failed to apply formatting to {tab_key}: {e}")
    
    async def _rate_limited_call(self, func, *args, **kwargs):
        """Make rate-limited API calls to avoid hitting Google Sheets limits"""
        try:
            # Ensure minimum interval between API calls
            current_time = time.time()
            time_since_last_call = current_time - self.last_api_call
            
            if time_since_last_call < self.min_api_interval:
                await asyncio.sleep(self.min_api_interval - time_since_last_call)
            
            # Make the API call
            result = func(*args, **kwargs)
            self.last_api_call = time.time()
            
            # Reset error count on successful call
            self.error_count = 0
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Rate limited API call failed: {e}")
            
            if self.error_count >= self.max_errors:
                logger.critical("Too many API errors, stopping Google Sheets updates")
                raise
            
            # Exponential backoff
            backoff_time = min(60, 2 ** self.error_count)
            await asyncio.sleep(backoff_time)
            raise
    
    async def log_trade_signal(self, signal_data: Dict[str, Any]):
        """Log trade signal to Google Sheets"""
        try:
            if not self.gc or 'trade_signals' not in self.worksheets:
                logger.debug("Trade signals worksheet not available - Google Sheets disabled")
                return False
            
            worksheet = self.worksheets['trade_signals']
            
            # Prepare row data
            row_data = [
                self._get_ist_timestamp(),
                signal_data.get('signal_id', ''),
                signal_data.get('instrument', ''),
                signal_data.get('direction', ''),
                signal_data.get('strike_price', ''),
                signal_data.get('expiry_date', ''),
                signal_data.get('entry_price', ''),
                signal_data.get('stop_loss', ''),
                signal_data.get('target_1', ''),
                signal_data.get('target_2', ''),
                signal_data.get('confidence_score', ''),
                signal_data.get('risk_score', ''),
                signal_data.get('reason_summary', ''),
                signal_data.get('status', 'Pending'),
                signal_data.get('entry_time', ''),
                signal_data.get('exit_time', ''),
                signal_data.get('pnl', ''),
                signal_data.get('risk_reward_ratio', '')
            ]
            
            # Append row
            await self._rate_limited_call(worksheet.append_row, row_data)
            
            logger.info(f"Trade signal logged: {signal_data.get('signal_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log trade signal: {e}")
            await self._log_system_health("Trade Signal Logging", "Error", str(e), "Warning")
            return False
    
    async def log_market_context(self, market_data: Dict[str, Any]):
        """Log market context snapshot to Google Sheets"""
        try:
            if not self.gc or 'market_context' not in self.worksheets:
                logger.debug("Market context worksheet not available - Google Sheets disabled")
                return False
            
            worksheet = self.worksheets['market_context']
            
            # Prepare row data
            row_data = [
                self._get_ist_timestamp(),
                market_data.get('nifty_spot', ''),
                market_data.get('banknifty_spot', ''),
                market_data.get('india_vix', ''),
                market_data.get('nifty_pe', ''),
                market_data.get('nifty_pb', ''),
                market_data.get('sgx_nifty', ''),
                market_data.get('dow_jones', ''),
                market_data.get('nasdaq', ''),
                market_data.get('sp500', ''),
                market_data.get('dxy', ''),
                market_data.get('crude_oil', ''),
                market_data.get('gold_price', ''),
                market_data.get('usdinr', ''),
                market_data.get('bond_yield_10y', ''),
                market_data.get('global_sentiment', ''),
                market_data.get('market_session', '')
            ]
            
            # Append row
            await self._rate_limited_call(worksheet.append_row, row_data)
            
            logger.info("Market context logged successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log market context: {e}")
            await self._log_system_health("Market Context Logging", "Error", str(e), "Warning")
            return False
    
    async def log_options_data(self, options_data: List[Dict[str, Any]]):
        """Log options chain data to Google Sheets"""
        try:
            if not self.gc or 'options_data' not in self.worksheets:
                logger.debug("Options data worksheet not available - Google Sheets disabled")
                return False
            
            worksheet = self.worksheets['options_data']
            
            # Prepare batch data
            batch_data = []
            timestamp = self._get_ist_timestamp()
            
            for option in options_data:
                row_data = [
                    timestamp,
                    option.get('instrument', ''),
                    option.get('strike_price', ''),
                    option.get('option_type', ''),
                    option.get('ltp', ''),
                    option.get('volume', ''),
                    option.get('oi', ''),
                    option.get('change_in_oi', ''),
                    option.get('oi_change_percent', ''),
                    option.get('iv', ''),
                    option.get('delta', ''),
                    option.get('gamma', ''),
                    option.get('theta', ''),
                    option.get('vega', ''),
                    option.get('pcr', ''),
                    option.get('bid_ask_spread', '')
                ]
                batch_data.append(row_data)
            
            # Batch append for efficiency
            if batch_data:
                await self._rate_limited_call(worksheet.append_rows, batch_data)
                logger.info(f"Options data logged: {len(batch_data)} entries")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log options data: {e}")
            await self._log_system_health("Options Data Logging", "Error", str(e), "Warning")
            return False
    
    async def log_institutional_flow(self, flow_data: Dict[str, Any]):
        """Log FII/DII institutional flow data"""
        try:
            if not self.gc or 'institutional_flow' not in self.worksheets:
                logger.debug("Institutional flow worksheet not available - Google Sheets disabled")
                return False
            
            worksheet = self.worksheets['institutional_flow']
            
            # Prepare row data
            row_data = [
                flow_data.get('date', self._get_ist_date()),
                flow_data.get('fii_equity_buy', ''),
                flow_data.get('fii_equity_sell', ''),
                flow_data.get('fii_equity_net', ''),
                flow_data.get('fii_derivative_buy', ''),
                flow_data.get('fii_derivative_sell', ''),
                flow_data.get('fii_derivative_net', ''),
                flow_data.get('dii_equity_buy', ''),
                flow_data.get('dii_equity_sell', ''),
                flow_data.get('dii_equity_net', ''),
                flow_data.get('total_net_flow', ''),
                flow_data.get('fii_monthly_cumulative', ''),
                flow_data.get('dii_monthly_cumulative', '')
            ]
            
            # Append row
            await self._rate_limited_call(worksheet.append_row, row_data)
            
            logger.info("Institutional flow data logged successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log institutional flow: {e}")
            await self._log_system_health("Institutional Flow Logging", "Error", str(e), "Warning")
            return False
    
    async def log_news_sentiment(self, news_data: Dict[str, Any]):
        """Log news sentiment data"""
        try:
            if not self.gc or 'news_sentiment' not in self.worksheets:
                logger.debug("News sentiment worksheet not available - Google Sheets disabled")
                return False
            
            worksheet = self.worksheets['news_sentiment']
            
            # Prepare row data
            row_data = [
                self._get_ist_timestamp(),
                news_data.get('source', ''),
                news_data.get('category', ''),
                news_data.get('headline', ''),
                news_data.get('summary', ''),
                news_data.get('sentiment_score', ''),
                news_data.get('market_impact', ''),
                news_data.get('confidence_level', ''),
                news_data.get('related_stocks', '')
            ]
            
            # Append row
            await self._rate_limited_call(worksheet.append_row, row_data)
            
            logger.info("News sentiment logged successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log news sentiment: {e}")
            await self._log_system_health("News Sentiment Logging", "Error", str(e), "Warning")
            return False
    
    async def log_rejected_signal(self, rejected_data: Dict[str, Any]):
        """Log rejected trade signal"""
        try:
            if not self.gc or 'rejected_signals' not in self.worksheets:
                logger.debug("Rejected signals worksheet not available - Google Sheets disabled")
                return False
            
            worksheet = self.worksheets['rejected_signals']
            
            # Prepare row data
            row_data = [
                self._get_ist_timestamp(),
                rejected_data.get('signal_id', ''),
                rejected_data.get('instrument', ''),
                rejected_data.get('direction', ''),
                rejected_data.get('strike_price', ''),
                rejected_data.get('expiry_date', ''),
                rejected_data.get('proposed_entry_price', ''),
                rejected_data.get('proposed_stop_loss', ''),
                rejected_data.get('proposed_target_1', ''),
                rejected_data.get('proposed_target_2', ''),
                rejected_data.get('risk_score', ''),
                rejected_data.get('confidence_score', ''),
                rejected_data.get('rejection_reason', ''),
                rejected_data.get('risk_cost', ''),
                rejected_data.get('max_drawdown_risk', ''),
                rejected_data.get('position_size', '')
            ]
            
            # Append row
            await self._rate_limited_call(worksheet.append_row, row_data)
            
            logger.info(f"Rejected signal logged: {rejected_data.get('signal_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log rejected signal: {e}")
            await self._log_system_health("Rejected Signal Logging", "Error", str(e), "Warning")
            return False
    
    async def log_performance_analytics(self, performance_data: Dict[str, Any]):
        """Log daily performance analytics"""
        try:
            if not self.gc or 'performance_analytics' not in self.worksheets:
                logger.debug("Performance analytics worksheet not available - Google Sheets disabled")
                return False
            
            worksheet = self.worksheets['performance_analytics']
            
            # Prepare row data
            row_data = [
                performance_data.get('date', self._get_ist_date()),
                performance_data.get('total_signals_generated', ''),
                performance_data.get('total_signals_executed', ''),
                performance_data.get('total_signals_rejected', ''),
                performance_data.get('win_rate', ''),
                performance_data.get('avg_pnl_per_trade', ''),
                performance_data.get('max_profit', ''),
                performance_data.get('max_loss', ''),
                performance_data.get('daily_pnl', ''),
                performance_data.get('cumulative_pnl', ''),
                performance_data.get('sharpe_ratio', ''),
                performance_data.get('max_drawdown', ''),
                performance_data.get('success_rate_by_instrument', ''),
                performance_data.get('avg_hold_time', '')
            ]
            
            # Append row
            await self._rate_limited_call(worksheet.append_row, row_data)
            
            logger.info("Performance analytics logged successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log performance analytics: {e}")
            await self._log_system_health("Performance Analytics Logging", "Error", str(e), "Warning")
            return False
    
    async def _log_system_health(self, component: str, status: str, message: str, alert_level: str):
        """Log system health and alerts"""
        try:
            if 'system_health' not in self.worksheets:
                return False
            
            worksheet = self.worksheets['system_health']
            
            # Prepare row data
            row_data = [
                self._get_ist_timestamp(),
                component,
                status,
                message,
                '',  # Response time - will be filled by monitoring
                self._get_ist_timestamp(),  # Data freshness
                alert_level,
                'Yes' if alert_level == 'Critical' else 'No'
            ]
            
            # Append row
            await self._rate_limited_call(worksheet.append_row, row_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log system health: {e}")
            return False
    
    def _get_ist_timestamp(self) -> str:
        """Get current timestamp in IST"""
        return datetime.now(self.ist).strftime('%Y-%m-%d %H:%M:%S')
    
    def _get_ist_date(self) -> str:
        """Get current date in IST"""
        return datetime.now(self.ist).strftime('%Y-%m-%d')
    
    async def update_signal_status(self, signal_id: str, status: str, exit_time: str = None, pnl: float = None):
        """Update existing signal status"""
        try:
            if 'trade_signals' not in self.worksheets:
                return False
            
            worksheet = self.worksheets['trade_signals']
            
            # Find the signal row
            all_records = worksheet.get_all_records()
            
            for i, record in enumerate(all_records, start=2):  # Start from row 2 (after header)
                if record.get('Signal ID') == signal_id:
                    # Update status
                    worksheet.update_cell(i, 14, status)  # Status column
                    
                    if exit_time:
                        worksheet.update_cell(i, 16, exit_time)  # Exit time column
                    
                    if pnl is not None:
                        worksheet.update_cell(i, 17, pnl)  # P&L column
                    
                    logger.info(f"Updated signal {signal_id} status to {status}")
                    return True
            
            logger.warning(f"Signal {signal_id} not found for status update")
            return False
            
        except Exception as e:
            logger.error(f"Failed to update signal status: {e}")
            return False
    
    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to maintain sheet performance"""
        try:
            cutoff_date = datetime.now(self.ist) - timedelta(days=days_to_keep)
            
            for tab_key, worksheet in self.worksheets.items():
                if tab_key in ['performance_analytics', 'system_health']:
                    continue  # Keep all performance and health data
                
                # Get all records
                all_records = worksheet.get_all_records()
                
                # Find rows to delete (older than cutoff)
                rows_to_delete = []
                for i, record in enumerate(all_records, start=2):
                    timestamp_str = record.get('Timestamp (IST)', record.get('Timestamp', ''))
                    if timestamp_str:
                        try:
                            record_date = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                            if record_date < cutoff_date:
                                rows_to_delete.append(i)
                        except:
                            continue
                
                # Delete old rows (in reverse order to maintain row numbers)
                for row_num in reversed(rows_to_delete):
                    worksheet.delete_rows(row_num)
                
                if rows_to_delete:
                    logger.info(f"Cleaned up {len(rows_to_delete)} old records from {tab_key}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return False
    
    async def get_sheet_health_status(self) -> Dict[str, Any]:
        """Get overall health status of Google Sheets integration"""
        try:
            health_status = {
                'status': 'Healthy',
                'last_update': self._get_ist_timestamp(),
                'error_count': self.error_count,
                'worksheets_available': len(self.worksheets),
                'api_response_time': 0
            }
            
            # Test API response time
            start_time = time.time()
            test_worksheet = self.worksheets.get('system_health')
            if test_worksheet:
                test_worksheet.row_count  # Simple API call to test response
            health_status['api_response_time'] = round((time.time() - start_time) * 1000, 2)
            
            if self.error_count > 5:
                health_status['status'] = 'Warning'
            elif self.error_count >= self.max_errors:
                health_status['status'] = 'Critical'
            
            return health_status
            
        except Exception as e:
            logger.error(f"Failed to get sheet health status: {e}")
            return {
                'status': 'Error',
                'last_update': self._get_ist_timestamp(),
                'error_count': self.error_count,
                'error_message': str(e)
            }
    
    async def shutdown(self):
        """Graceful shutdown of Google Sheets manager"""
        try:
            await self._log_system_health("Google Sheets", "Offline", "System shutdown", "Info")
            logger.info("Google Sheets manager shutdown completed")
        except Exception as e:
            logger.error(f"Error during Google Sheets shutdown: {e}")