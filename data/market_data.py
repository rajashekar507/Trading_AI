# TradeMind_AI Market Data Engine - FIXED VERSION
# Fetches REAL NIFTY, BANKNIFTY, and SENSEX data

from dhanhq import dhanhq, DhanContext
import time
import json
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class MarketDataProvider:
    def __init__(self):
        """Initialize Market Data Engine with REAL API connection"""
        logger.info("[INIT] Initializing Market Data Engine (FIXED VERSION)...")
        
        # Initialize Dhan with error handling
        try:
            client_id = os.getenv('DHAN_CLIENT_ID')
            access_token = os.getenv('DHAN_ACCESS_TOKEN')
            
            if not client_id or not access_token:
                logger.error("[ERROR] Dhan credentials not found!")
                logger.error("[INFO] Please add DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN to your .env file")
                self.dhan = None
                return
            
            dhan_context = DhanContext(client_id, access_token)
            self.dhan = dhanhq(dhan_context)
            
            # Test connection
            test_result = self._test_connection()
            if test_result:
                logger.info("[SUCCESS] Dhan API connected successfully")
            else:
                logger.error("[ERROR] Dhan API connection failed")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize Dhan: {e}")
            self.dhan = None
        
        # Market identifiers - CORRECTED
        self.SYMBOLS = {
            'NIFTY': {'id': 13, 'name': 'NIFTY 50'},
            'BANKNIFTY': {'id': 25, 'name': 'NIFTY BANK'},
            'SENSEX': {'id': 51, 'name': 'BSE SENSEX'},  # Added SENSEX
            'FINNIFTY': {'id': 27, 'name': 'NIFTY FIN SERVICE'}
        }
        self.IDX_SEGMENT = "IDX_I"
        
        logger.info("[SUCCESS] Market Data Engine initialized with REAL data fetching!")

    def _test_connection(self) -> bool:
        """Test Dhan API connection"""
        try:
            if not self.dhan:
                return False
            
            # Try a simple API call
            response = self.dhan.get_fund_limits()
            return response is not None
            
        except Exception as e:
            logger.error(f"[ERROR] Connection test failed: {e}")
            return False

    def get_live_quote(self, symbol: str) -> dict:
        """Get live quote for a symbol - FIXED VERSION"""
        try:
            if not self.dhan:
                return {
                    'error': 'Dhan API not connected',
                    'symbol': symbol,
                    'status': 'failed'
                }
            
            if symbol not in self.SYMBOLS:
                return {
                    'error': f'Symbol {symbol} not supported',
                    'symbol': symbol,
                    'status': 'failed'
                }
            
            symbol_info = self.SYMBOLS[symbol]
            logger.info(f"[REFRESH] Fetching REAL data for {symbol} (ID: {symbol_info['id']})...")
            
            # Get live quote from Dhan - CORRECTED method name
            quote_response = self.dhan.ohlc_data(
                securities={self.IDX_SEGMENT: [symbol_info['id']]}
            )
            
            if not quote_response:
                raise Exception("No response from Dhan API")
            
            if quote_response.get('status') != 'success':
                raise Exception(f"API Error: {quote_response.get('message', 'Unknown error')}")
            
            # Extract data from OHLC response
            response_data = quote_response.get('data', {})
            if not response_data:
                raise Exception("No data in API response")
            
            # Navigate the nested structure: data -> data -> IDX_I -> security_id
            nested_data = response_data.get('data', {})
            if not nested_data:
                raise Exception("No nested data in API response")
            
            segment_data = nested_data.get(self.IDX_SEGMENT, {})
            if not segment_data:
                raise Exception(f"No data for segment {self.IDX_SEGMENT}")
            
            security_data = segment_data.get(str(symbol_info['id']), {})
            if not security_data:
                raise Exception(f"No data for security ID {symbol_info['id']}")
            
            # Parse the OHLC data
            ltp = float(security_data.get('last_price', 0))
            ohlc = security_data.get('ohlc', {})
            
            prev_close = float(ohlc.get('open', ltp))  # Use open as prev close approximation
            high = float(ohlc.get('high', ltp))
            low = float(ohlc.get('low', ltp))
            close = float(ohlc.get('close', ltp))
            volume = int(security_data.get('volume', 0))
            
            # Calculate change
            change = ltp - prev_close
            change_percent = (change / prev_close * 100) if prev_close > 0 else 0
            
            result = {
                'symbol': symbol,
                'name': symbol_info['name'],
                'price': ltp,
                'prev_close': prev_close,
                'change': change,
                'change_percent': change_percent,
                'high': high,
                'low': low,
                'volume': volume,
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'data_source': 'dhan_live'
            }
            
            logger.info(f"[OK] {symbol}: Rs.{ltp:,.2f} ({change:+.2f}, {change_percent:+.2f}%)")
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] Error fetching {symbol}: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }

    def get_all_indices(self) -> dict:
        """Get all major indices data"""
        logger.info("[CHART] Fetching all major indices...")
        
        results = {}
        for symbol in ['NIFTY', 'BANKNIFTY', 'SENSEX']:
            results[symbol.lower()] = self.get_live_quote(symbol)
            time.sleep(1)  # Rate limiting
        
        results['timestamp'] = datetime.now().isoformat()
        return results

    def get_option_chain(self, symbol_id, symbol_name):
        """Get real option chain data - IMPROVED"""
        try:
            logger.info(f"📡 Fetching {symbol_name} option chain...")
            
            if not self.dhan:
                return {
                    'error': 'Dhan API not connected',
                    'symbol': symbol_name,
                    'status': 'failed'
                }
            
            # Get expiry list first - CORRECTED method name
            expiry_response = self.dhan.expiry_list(
                under_security_id=symbol_id,
                under_exchange_segment=self.IDX_SEGMENT
            )
            
            if expiry_response.get("status") != "success":
                logger.error(f"[ERROR] Failed to get expiry list for {symbol_name}")
                return {
                    'error': 'Failed to get expiry list',
                    'symbol': symbol_name,
                    'status': 'failed'
                }
                
            expiry_list = expiry_response["data"]["data"]
            if not expiry_list:
                return {
                    'error': 'No expiry dates available',
                    'symbol': symbol_name,
                    'status': 'failed'
                }
            
            nearest_expiry = expiry_list[0]
            logger.info(f"[DATE] Using expiry: {nearest_expiry}")
            
            # Rate limiting
            time.sleep(2)
            
            # Get option chain - CORRECTED method name
            option_chain = self.dhan.option_chain(
                under_security_id=symbol_id,
                under_exchange_segment=self.IDX_SEGMENT,
                expiry=nearest_expiry
            )
            
            if option_chain.get("status") == "success":
                logger.info(f"[OK] {symbol_name} option chain fetched successfully!")
                return {
                    'symbol': symbol_name,
                    'expiry': nearest_expiry,
                    'data': option_chain["data"],
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }
            else:
                logger.error(f"[ERROR] Failed to get option chain for {symbol_name}")
                return {
                    'error': 'Failed to get option chain',
                    'symbol': symbol_name,
                    'status': 'failed'
                }
                
        except Exception as e:
            logger.error(f"[ERROR] Error fetching {symbol_name} option data: {e}")
            return {
                'error': str(e),
                'symbol': symbol_name,
                'status': 'failed'
            }

    def analyze_option_data(self, option_data):
        """Analyze option chain for trading opportunities - IMPROVED"""
        if not option_data or 'data' not in option_data or option_data.get('status') != 'success':
            logger.error("[WARNING]️ Invalid or failed option data")
            return None
            
        try:
            data = option_data['data']
            
            # Extract key information
            if 'data' in data:
                inner_data = data['data']
                underlying_price = inner_data.get('last_price', 0)
                option_chain = inner_data.get('oc', {})
            else:
                underlying_price = data.get('last_price', 0)
                option_chain = data.get('oc', {})
            
            if not option_chain:
                logger.error("[WARNING]️ No option chain data found")
                return None
            
            # Find ATM strike
            strikes = list(option_chain.keys())
            strikes_float = [float(strike) for strike in strikes]
            
            if not strikes_float:
                logger.error("[WARNING]️ No strikes found in option chain")
                return None
            
            # Find closest strike to underlying price
            atm_strike = min(strikes_float, key=lambda x: abs(x - underlying_price))
            atm_strike_str = f"{atm_strike:.6f}"
            
            if atm_strike_str not in option_chain:
                logger.error(f"[WARNING]️ ATM strike {atm_strike} not found in option chain")
                return None
            
            atm_data = option_chain[atm_strike_str]
            
            # Extract Call and Put data safely
            ce_data = atm_data.get('ce', {})
            pe_data = atm_data.get('pe', {})
            
            analysis = {
                'symbol': option_data['symbol'],
                'underlying_price': underlying_price,
                'atm_strike': atm_strike,
                'call_price': ce_data.get('last_price', 0),
                'put_price': pe_data.get('last_price', 0),
                'call_oi': ce_data.get('oi', 0),
                'put_oi': pe_data.get('oi', 0),
                'call_volume': ce_data.get('volume', 0),
                'put_volume': pe_data.get('volume', 0),
                'call_iv': ce_data.get('implied_volatility', 0),
                'put_iv': pe_data.get('implied_volatility', 0),
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
            logger.info(f"[CHART] {option_data['symbol']} Analysis:")
            logger.info(f"   [MONEY] Underlying: Rs.{underlying_price}")
            logger.info(f"   [TARGET] ATM Strike: Rs.{atm_strike}")
            logger.info(f"   [PHONE] Call Price: Rs.{analysis['call_price']}")
            logger.info(f"   [PHONE] Put Price: Rs.{analysis['put_price']}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"[ERROR] Error analyzing option data: {e}")
            return None

    def get_market_snapshot(self):
        """Get complete market snapshot with improved error handling"""
        logger.info("\n[REFRESH] Getting market snapshot...")
        
        results = {
            'indices': {},
            'options': {},
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        # Get indices data
        results['indices'] = self.get_all_indices()
        
        # Get option data for NIFTY and BANKNIFTY
        if self.dhan:
            logger.info("[CHART] Fetching option chain data...")
            
            # NIFTY options
            nifty_data = self.get_option_chain(self.SYMBOLS['NIFTY']['id'], "NIFTY")
            nifty_analysis = self.analyze_option_data(nifty_data) if nifty_data.get('status') == 'success' else None
            results['options']['nifty'] = nifty_analysis
            
            # BANKNIFTY options
            banknifty_data = self.get_option_chain(self.SYMBOLS['BANKNIFTY']['id'], "BANKNIFTY")
            banknifty_analysis = self.analyze_option_data(banknifty_data) if banknifty_data.get('status') == 'success' else None
            results['options']['banknifty'] = banknifty_analysis
        
        return results

    def test_all_connections(self):
        """Test all API connections and return detailed status"""
        logger.info("[TOOL] Testing all connections...")
        
        results = {
            'dhan_api': False,
            'quotes': {},
            'options': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Test basic connection
        results['dhan_api'] = self._test_connection()
        
        if results['dhan_api']:
            # Test quote fetching
            for symbol in ['NIFTY', 'BANKNIFTY', 'SENSEX']:
                quote_data = self.get_live_quote(symbol)
                results['quotes'][symbol] = quote_data.get('status') == 'success'
                time.sleep(1)
            
            # Test option chain (just NIFTY to save time)
            option_data = self.get_option_chain(self.SYMBOLS['NIFTY']['id'], 'NIFTY')
            results['options']['NIFTY'] = option_data.get('status') == 'success'
        
        return results

# Test the market data engine when run directly
if __name__ == "__main__":
    print("[BRAIN] TradeMind AI Market Data Engine - FIXED VERSION")
    print("=" * 60)
    
    engine = MarketDataEngine()
    
    if not engine.dhan:
        print("[ERROR] Cannot test - Dhan API not connected")
        print("[NOTE] Please check your .env file has correct DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN")
        exit(1)
    
    print("\n[TOOL] Testing connections...")
    test_results = engine.test_all_connections()
    
    print(f"Dhan API: {'[OK]' if test_results['dhan_api'] else '[ERROR]'}")
    
    for symbol, status in test_results['quotes'].items():
        print(f"{symbol} quotes: {'[OK]' if status else '[ERROR]'}")
    
    for symbol, status in test_results['options'].items():
        print(f"{symbol} options: {'[OK]' if status else '[ERROR]'}")
    
    if test_results['dhan_api']:
        print("\n[CHART] Getting live market data...")
        indices = engine.get_all_indices()
        
        for symbol, data in indices.items():
            if symbol != 'timestamp' and data.get('status') == 'success':
                print(f"[OK] {symbol.upper()}: Rs.{data['price']:,.2f} ({data['change']:+.2f})")
            elif symbol != 'timestamp':
                print(f"[ERROR] {symbol.upper()}: {data.get('error', 'Failed')}")
    
    print("\n[OK] Market Data Engine test complete!")
