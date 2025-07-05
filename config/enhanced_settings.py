import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EnhancedSettings:
    def __init__(self):
        self.RSI_OVERBOUGHT_THRESHOLD = float(os.getenv('RSI_OVERBOUGHT_THRESHOLD', 70))
        self.RSI_OVERSOLD_THRESHOLD = float(os.getenv('RSI_OVERSOLD_THRESHOLD', 30))
        self.MACD_SIGNAL_THRESHOLD = float(os.getenv('MACD_SIGNAL_THRESHOLD', 0))
        self.MACD_FAST = int(os.getenv('MACD_FAST', 12))
        self.MACD_SLOW = int(os.getenv('MACD_SLOW', 26))
        self.MACD_SIGNAL = int(os.getenv('MACD_SIGNAL', 9))
        self.SR_LOOKBACK_DAYS = int(os.getenv('SR_LOOKBACK_DAYS', 30))
        self.SR_MIN_TOUCHES = int(os.getenv('SR_MIN_TOUCHES', 2))
        self.SR_MAX_DISTANCE = float(os.getenv('SR_MAX_DISTANCE', 0.01))
        self.ORB_WINDOW_MINUTES = int(os.getenv('ORB_WINDOW_MINUTES', 15))
        self.ORB_BREAKOUT_THRESHOLD = float(os.getenv('ORB_BREAKOUT_THRESHOLD', 0.5))
        self.ORB_MIN_VOLUME = int(os.getenv('ORB_MIN_VOLUME', 1000))
        self.ORB_MIN_RANGE = float(os.getenv('ORB_MIN_RANGE', 20))
        self.ORB_MAX_RANGE = float(os.getenv('ORB_MAX_RANGE', 200))
        self.ORB_CONFIRMATION_MINUTES = int(os.getenv('ORB_CONFIRMATION_MINUTES', 5))
        self.ORB_CONFIRMATION_VOLUME = int(os.getenv('ORB_CONFIRMATION_VOLUME', 500))
        self.ORB_CONFIRMATION_RANGE = float(os.getenv('ORB_CONFIRMATION_RANGE', 10))

        self.NIFTY_LOT_SIZE = int(os.getenv('NIFTY_LOT_SIZE', 50))
        self.BANKNIFTY_LOT_SIZE = int(os.getenv('BANKNIFTY_LOT_SIZE', 15))
        self.FINNIFTY_LOT_SIZE = int(os.getenv('FINNIFTY_LOT_SIZE', 40))
        self.SIGNAL_COOLDOWN_MINUTES = int(os.getenv('SIGNAL_COOLDOWN_MINUTES', 30))  # FIXED: 30 minutes minimum
        self.SIGNAL_MIN_CONFIDENCE = float(os.getenv('SIGNAL_MIN_CONFIDENCE', 65))  # FIXED: Professional minimum 65%
        self.SIGNAL_MAX_AGE_MINUTES = int(os.getenv('SIGNAL_MAX_AGE_MINUTES', 30))
        self.SIGNAL_MAX_PER_CYCLE = int(os.getenv('SIGNAL_MAX_PER_CYCLE', 5))
        self.SIGNAL_MAX_TOTAL = int(os.getenv('SIGNAL_MAX_TOTAL', 20))
        self.SIGNAL_MIN_VOLUME = int(os.getenv('SIGNAL_MIN_VOLUME', 500))
        self.SIGNAL_MIN_OI = int(os.getenv('SIGNAL_MIN_OI', 1000))
        self.SIGNAL_COOLDOWN_SECONDS = int(os.getenv('SIGNAL_COOLDOWN_SECONDS', 1800))  # FIXED: 30 minutes = 1800 seconds
        # Removed duplicate SIGNAL_COOLDOWN_MINUTES (already defined above)
        self.SIGNAL_MAX_LOSS = float(os.getenv('SIGNAL_MAX_LOSS', -5000))
        self.SIGNAL_MAX_RISK_SCORE = float(os.getenv('SIGNAL_MAX_RISK_SCORE', 70))
        self.SIGNAL_MAX_DRAWDOWN = float(os.getenv('SIGNAL_MAX_DRAWDOWN', -10000))
        self.SIGNAL_MAX_TRADES_PER_DAY = int(os.getenv('SIGNAL_MAX_TRADES_PER_DAY', 10))
        self.SIGNAL_MAX_POSITIONS = int(os.getenv('SIGNAL_MAX_POSITIONS', 10))
        self.SIGNAL_MAX_POSITIONS_PER_SYMBOL = int(os.getenv('SIGNAL_MAX_POSITIONS_PER_SYMBOL', 3))
        self.SIGNAL_MAX_SAME_EXPIRY_POSITIONS = int(os.getenv('SIGNAL_MAX_SAME_EXPIRY_POSITIONS', 5))

        self.PAPER_TRADING = os.getenv('PAPER_TRADING', 'True').lower() == 'true'
        self.LIVE_TRADING_ENABLED = os.getenv('LIVE_TRADING_ENABLED', 'False').lower() == 'true'
        self.MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', -10000))
        self.MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 100000))
        self.MAX_TOTAL_POSITIONS = int(os.getenv('MAX_TOTAL_POSITIONS', 3))
        self.KITE_API_KEY = os.getenv('KITE_API_KEY', '')
        self.KITE_API_SECRET = os.getenv('KITE_API_SECRET', '')
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
        self.DATA_REFRESH_INTERVAL = int(os.getenv('DATA_REFRESH_INTERVAL', 60))
        self.CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 60))
        self.LOGS_DIR = os.getenv('LOGS_DIR', 'logs')
        self.DATA_STORAGE_DIR = os.getenv('DATA_STORAGE_DIR', 'data_storage')
        # Add more as needed

    def get_lot_size(self, symbol):
        symbol = symbol.upper()
        if symbol == 'NIFTY':
            return getattr(self, 'NIFTY_LOT_SIZE', 50)
        elif symbol == 'BANKNIFTY':
            return getattr(self, 'BANKNIFTY_LOT_SIZE', 15)
        elif symbol == 'FINNIFTY':
            return getattr(self, 'FINNIFTY_LOT_SIZE', 40)
        return 1

    def get_trading_mode_status(self):
        return 'PAPER_TRADING' if self.PAPER_TRADING else 'LIVE_TRADING'

    def get_risk_limits(self):
        return {
            'max_daily_loss': getattr(self, 'MAX_DAILY_LOSS', -10000),
            'max_position_size': getattr(self, 'MAX_POSITION_SIZE', 100000),
            'max_total_positions': getattr(self, 'MAX_TOTAL_POSITIONS', 3),
        }

    def validate(self):
        assert self.KITE_API_KEY, 'KITE_API_KEY is required'
        assert self.KITE_API_SECRET, 'KITE_API_SECRET is required'
        # Add more validation as needed
