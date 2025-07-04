"""
Email Alert System for Google Sheets Integration
Sends email notifications for critical system events
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import asyncio
from typing import List, Dict, Any
import pytz

logger = logging.getLogger('trading_system.email_alerts')

class EmailAlertSystem:
    """Email alert system for critical notifications"""
    
    def __init__(self, settings):
        self.settings = settings
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Email configuration
        self.smtp_server = getattr(settings, 'SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = getattr(settings, 'SMTP_PORT', 587)
        self.email_user = getattr(settings, 'EMAIL_USER', None)
        self.email_password = getattr(settings, 'EMAIL_PASSWORD', None)
        self.alert_recipients = getattr(settings, 'ALERT_RECIPIENTS', [])
        
        # Alert tracking
        self.last_alerts = {}
        self.alert_cooldown = 900  # 15 minutes cooldown for same alert type
        
        logger.info("Email Alert System initialized")
    
    async def send_critical_alert(self, subject: str, message: str, alert_type: str = "CRITICAL"):
        """Send critical alert email"""
        try:
            if not self._should_send_alert(alert_type):
                return False
            
            if not self.email_user or not self.email_password or not self.alert_recipients:
                logger.warning("Email configuration incomplete - cannot send alerts")
                return False
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = ', '.join(self.alert_recipients)
            msg['Subject'] = f"[VLR_AI TRADING] {subject}"
            
            # Email body
            body = f"""
VLR_AI Trading System Alert

Alert Type: {alert_type}
Time: {datetime.now(self.ist).strftime('%Y-%m-%d %H:%M:%S')} IST

{message}

---
This is an automated alert from VLR_AI Trading System.
Google Sheets Dashboard: https://docs.google.com/spreadsheets/d/1cRUP3VnM5JcjFyFZTholuQfmEXInTAXz_14V_OZP67M/edit

Please check the system immediately if this is a critical alert.
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            await self._send_email(msg)
            
            # Update alert tracking
            self.last_alerts[alert_type] = datetime.now()
            
            logger.info(f"Critical alert sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send critical alert: {e}")
            return False
    
    async def send_system_health_alert(self, component: str, status: str, error_message: str):
        """Send system health alert"""
        try:
            if status == "Critical":
                subject = f"CRITICAL: {component} System Failure"
                message = f"""
CRITICAL SYSTEM ALERT

Component: {component}
Status: {status}
Error: {error_message}

IMMEDIATE ACTION REQUIRED:
1. Check Google Sheets integration
2. Verify API connections
3. Review system logs
4. Restart services if necessary

This alert indicates a critical system failure that may affect trading operations.
"""
                await self.send_critical_alert(subject, message, f"HEALTH_{component}")
                
            elif status == "Warning":
                subject = f"WARNING: {component} Issues Detected"
                message = f"""
SYSTEM WARNING

Component: {component}
Status: {status}
Issue: {error_message}

RECOMMENDED ACTIONS:
1. Monitor system performance
2. Check for recurring issues
3. Review error logs
4. Consider preventive maintenance

This is a warning that may indicate developing issues.
"""
                await self.send_critical_alert(subject, message, f"WARNING_{component}")
            
        except Exception as e:
            logger.error(f"Failed to send system health alert: {e}")
    
    async def send_api_failure_alert(self, api_name: str, failure_duration: int, error_details: str):
        """Send API failure alert"""
        try:
            subject = f"API FAILURE: {api_name} Down for {failure_duration} minutes"
            message = f"""
API FAILURE ALERT

API: {api_name}
Failure Duration: {failure_duration} minutes
Error Details: {error_details}

IMPACT:
- Data collection may be incomplete
- Trading signals may be affected
- Sheet updates may be delayed

ACTIONS TAKEN:
- Automatic retry attempts in progress
- Fallback data sources activated (if available)
- System continues with available data

MANUAL INTERVENTION MAY BE REQUIRED if failure persists.
"""
            
            await self.send_critical_alert(subject, message, f"API_{api_name}")
            
        except Exception as e:
            logger.error(f"Failed to send API failure alert: {e}")
    
    async def send_data_quality_alert(self, data_issues: List[Dict[str, Any]]):
        """Send data quality alert"""
        try:
            if not data_issues:
                return
            
            subject = f"DATA QUALITY ISSUES: {len(data_issues)} problems detected"
            
            issues_text = "\n".join([
                f"- {issue['component']}: {issue['description']}" 
                for issue in data_issues
            ])
            
            message = f"""
DATA QUALITY ALERT

{len(data_issues)} data quality issues detected:

{issues_text}

POTENTIAL IMPACTS:
- Inaccurate trading signals
- Incorrect risk calculations
- Misleading performance metrics

RECOMMENDED ACTIONS:
1. Verify data sources
2. Check API connections
3. Review data validation rules
4. Consider manual data verification

Data quality issues can significantly impact trading performance.
"""
            
            await self.send_critical_alert(subject, message, "DATA_QUALITY")
            
        except Exception as e:
            logger.error(f"Failed to send data quality alert: {e}")
    
    async def send_performance_alert(self, performance_metrics: Dict[str, Any]):
        """Send performance degradation alert"""
        try:
            subject = "PERFORMANCE ALERT: System Performance Degraded"
            message = f"""
PERFORMANCE DEGRADATION ALERT

Performance Metrics:
- API Response Time: {performance_metrics.get('api_response_time', 'N/A')}ms
- Sheet Update Time: {performance_metrics.get('sheet_update_time', 'N/A')}ms
- Error Rate: {performance_metrics.get('error_rate', 'N/A')}%
- Data Freshness: {performance_metrics.get('data_freshness', 'N/A')} minutes

PERFORMANCE THRESHOLDS EXCEEDED:
{performance_metrics.get('threshold_violations', 'Multiple thresholds exceeded')}

RECOMMENDED ACTIONS:
1. Check system resources
2. Monitor network connectivity
3. Review API rate limits
4. Consider system optimization

Poor performance can affect trading execution and data accuracy.
"""
            
            await self.send_critical_alert(subject, message, "PERFORMANCE")
            
        except Exception as e:
            logger.error(f"Failed to send performance alert: {e}")
    
    async def send_daily_summary(self, summary_data: Dict[str, Any]):
        """Send daily system summary (non-critical)"""
        try:
            if not self.alert_recipients:
                return False
            
            subject = f"Daily Summary - {datetime.now(self.ist).strftime('%Y-%m-%d')}"
            
            message = f"""
VLR_AI Trading System - Daily Summary

Date: {datetime.now(self.ist).strftime('%Y-%m-%d')}

TRADING ACTIVITY:
- Signals Generated: {summary_data.get('signals_generated', 0)}
- Signals Executed: {summary_data.get('signals_executed', 0)}
- Signals Rejected: {summary_data.get('signals_rejected', 0)}
- Win Rate: {summary_data.get('win_rate', 0)}%
- Daily P&L: Rs.{summary_data.get('daily_pnl', 0):.2f}

SYSTEM HEALTH:
- Google Sheets Status: {summary_data.get('sheets_status', 'Unknown')}
- API Response Time: {summary_data.get('avg_response_time', 0)}ms
- Error Count: {summary_data.get('error_count', 0)}
- Data Completeness: {summary_data.get('data_completeness', 0)}%

MARKET DATA:
- NIFTY Close: {summary_data.get('nifty_close', 'N/A')}
- BANKNIFTY Close: {summary_data.get('banknifty_close', 'N/A')}
- VIX Close: {summary_data.get('vix_close', 'N/A')}

Google Sheets Dashboard: https://docs.google.com/spreadsheets/d/1cRUP3VnM5JcjFyFZTholuQfmEXInTAXz_14V_OZP67M/edit

---
This is an automated daily summary from VLR_AI Trading System.
"""
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = ', '.join(self.alert_recipients)
            msg['Subject'] = f"[VLR_AI TRADING] {subject}"
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email
            await self._send_email(msg)
            
            logger.info("Daily summary email sent")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")
            return False
    
    def _should_send_alert(self, alert_type: str) -> bool:
        """Check if alert should be sent based on cooldown"""
        try:
            last_alert = self.last_alerts.get(alert_type)
            if last_alert:
                time_since_last = (datetime.now() - last_alert).total_seconds()
                if time_since_last < self.alert_cooldown:
                    logger.info(f"Alert {alert_type} in cooldown period")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking alert cooldown: {e}")
            return True  # Send alert if unsure
    
    async def _send_email(self, msg: MIMEMultipart):
        """Send email using SMTP"""
        try:
            # Use asyncio to run SMTP in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._send_smtp_email, msg)
            
        except Exception as e:
            logger.error(f"Failed to send email via SMTP: {e}")
            raise
    
    def _send_smtp_email(self, msg: MIMEMultipart):
        """Send email using SMTP (synchronous)"""
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            
            text = msg.as_string()
            server.sendmail(self.email_user, self.alert_recipients, text)
            server.quit()
            
        except Exception as e:
            logger.error(f"SMTP email sending failed: {e}")
            raise
    
    async def test_email_system(self):
        """Test email system functionality"""
        try:
            subject = "Email System Test"
            message = """
This is a test email from VLR_AI Trading System.

If you receive this email, the email alert system is working correctly.

Test performed at: {datetime.now(self.ist).strftime('%Y-%m-%d %H:%M:%S')} IST

Email system is ready for critical alerts.
"""
            
            success = await self.send_critical_alert(subject, message, "TEST")
            
            if success:
                logger.info("Email system test successful")
            else:
                logger.warning("Email system test failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Email system test error: {e}")
            return False