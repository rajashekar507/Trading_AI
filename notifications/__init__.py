"""
Notifications Module for VLR_AI Trading System
Handles all notification systems for REAL trading alerts
"""

from .telegram_notifier import TelegramNotifier
from .email_alerts import EmailAlertSystem

__all__ = ['TelegramNotifier', 'EmailAlertSystem']