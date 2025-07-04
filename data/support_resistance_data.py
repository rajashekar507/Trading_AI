"""
Support & Resistance Data Provider
"""

import logging

logger = logging.getLogger(__name__)

class SupportResistanceData:
    def __init__(self, settings, kite_client=None):
        """Initialize Support & Resistance Data"""
        self.settings = settings
        self.kite_client = kite_client
        logger.info("[OK] SupportResistanceData initialized")
    
    async def fetch_data(self):
        """Fetch support and resistance levels"""
        return {
            'NIFTY': {
                'support_levels': [25300, 25250, 25200],
                'resistance_levels': [25450, 25500, 25550]
            },
            'BANKNIFTY': {
                'support_levels': [56500, 56400, 56300],
                'resistance_levels': [56800, 56900, 57000]
            }
        }
    
    async def fetch_sr_data(self, symbol: str, current_price: float) -> dict:
        """
        Fetch S/R data for a specific symbol with current price analysis
        
        Returns a dict with status, symbol, support/resistance levels, and current level
        """
        try:
            all_sr = await self.fetch_data()
            if symbol not in all_sr:
                raise KeyError(f"Symbol '{symbol}' not found in SR data")

            sr_data = all_sr[symbol]
            supports = sr_data.get('support_levels', [])
            resistances = sr_data.get('resistance_levels', [])

            # Determine where current price sits relative to S/R levels
            current_level = 'between'
            if supports and current_price <= min(supports):
                current_level = 'below'
            elif supports and abs(current_price - max(supports)) / current_price < 0.005:  # Within 0.5%
                current_level = 'near_support'
            elif resistances and current_price >= max(resistances):
                current_level = 'above'
            elif resistances and abs(current_price - min(resistances)) / current_price < 0.005:  # Within 0.5%
                current_level = 'near_resistance'

            return {
                'status': 'success',
                'symbol': symbol,
                'support_levels': [{'level': level, 'distance': abs(current_price - level) / current_price * 100} 
                                  for level in supports],
                'resistance_levels': [{'level': level, 'distance': abs(current_price - level) / current_price * 100} 
                                     for level in resistances],
                'current_level': current_level,
                'current_price': current_price,
                'strength': 0.7  # Default strength, can be calculated based on touches
            }

        except Exception as e:
            logger.error(f"[ERROR] SupportResistanceData.fetch_sr_data error: {e}")
            return {'status': 'error', 'symbol': symbol, 'error': str(e)}