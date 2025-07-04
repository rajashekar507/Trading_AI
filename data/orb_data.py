"""
Opening Range Breakout (ORB) Data Provider
"""

import logging

logger = logging.getLogger(__name__)

class ORBData:
    def __init__(self, settings, kite_client=None):
        """Initialize ORB Data"""
        self.settings = settings
        self.kite_client = kite_client
        logger.info("[OK] ORBData initialized")
    
    async def fetch_data(self):
        """Fetch ORB levels"""
        return {
            'NIFTY': {
                'orb_high': 25400,
                'orb_low': 25350,
                'orb_range': 50,
                'status': 'building_range'
            },
            'BANKNIFTY': {
                'orb_high': 56700,
                'orb_low': 56600,
                'orb_range': 100,
                'status': 'building_range'
            }
        }
    
    async def fetch_orb_data(self, symbol: str, current_price: float) -> dict:
        """
        Fetch ORB data for a specific symbol with current price analysis
        
        Returns a dict with status, symbol, ORB levels, and breakout status
        """
        try:
            all_orb = await self.fetch_data()
            if symbol not in all_orb:
                raise KeyError(f"Symbol '{symbol}' not found in ORB data")

            orb = all_orb[symbol]
            orb_high = orb.get('orb_high', 0)
            orb_low = orb.get('orb_low', 0)
            orb_range = orb.get('orb_range', 0)
            
            # Determine ORB status based on current price
            signal = 'neutral'
            confidence = 0
            
            if orb_high > 0 and orb_low > 0:
                if current_price > orb_high:
                    signal = 'bullish'
                    breakout_pct = ((current_price - orb_high) / orb_high) * 100
                    confidence = min(50 + (breakout_pct * 10), 80)  # Cap at 80%
                elif current_price < orb_low:
                    signal = 'bearish'
                    breakdown_pct = ((orb_low - current_price) / orb_low) * 100
                    confidence = min(50 + (breakdown_pct * 10), 80)  # Cap at 80%
                else:
                    # Price within range
                    position_in_range = (current_price - orb_low) / orb_range if orb_range > 0 else 0.5
                    confidence = 30  # Low confidence when inside range

            return {
                'status': 'success',
                'symbol': symbol,
                'orb_high': orb_high,
                'orb_low': orb_low,
                'orb_range': orb_range,
                'current_price': current_price,
                'signal': signal,
                'confidence': confidence,
                'orb_status': orb.get('status', 'unknown')
            }

        except Exception as e:
            logger.error(f"[ERROR] ORBData.fetch_orb_data error: {e}")
            return {'status': 'error', 'symbol': symbol, 'error': str(e)}