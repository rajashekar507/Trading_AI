"""
Enhanced News & Sentiment Analyzer - Real-time Web Intelligence
Replaces placeholder with advanced stealth data gathering
"""

import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.stealth_web_intelligence import EnhancedNewsSentimentAnalyzer

logger = logging.getLogger(__name__)

class NewsSentimentAnalyzer:
    """Enhanced News & Sentiment Analyzer with real web intelligence"""
    
    def __init__(self, settings=None):
        """Initialize Enhanced News & Sentiment Analyzer"""
        logger.info("[INIT] Initializing Enhanced News & Sentiment Analyzer...")
        
        try:
            self.enhanced_analyzer = EnhancedNewsSentimentAnalyzer(settings)
            self.initialized = False
            logger.info("[SUCCESS] Enhanced News & Sentiment Analyzer ready!")
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize enhanced analyzer: {e}")
            self.enhanced_analyzer = None
            self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the analyzer"""
        try:
            if self.enhanced_analyzer:
                self.initialized = await self.enhanced_analyzer.initialize()
                if self.initialized:
                    logger.info("[INIT] Enhanced analyzer initialized successfully")
                else:
                    logger.warning("[INIT] Enhanced analyzer initialization failed")
                return self.initialized
            return False
        except Exception as e:
            logger.error(f"[INIT] Initialization error: {e}")
            return False
    
    async def fetch_data(self):
        """Fetch real-time news and sentiment data"""
        try:
            if not self.initialized and self.enhanced_analyzer:
                await self.initialize()
            
            if self.initialized and self.enhanced_analyzer:
                logger.info("[FETCH] Fetching real-time market intelligence...")
                data = await self.enhanced_analyzer.fetch_data()
                
                logger.info(f"[FETCH] Retrieved {data.get('news_count', 0)} news items, "
                           f"sentiment: {data.get('news_sentiment', 'unknown')}")
                
                return data
            else:
                logger.warning("[FETCH] Enhanced analyzer not available, using fallback")
                return self._get_fallback_data()
                
        except Exception as e:
            logger.error(f"[FETCH] Error fetching enhanced data: {e}")
            return self._get_fallback_data()
    
    def _get_fallback_data(self):
        """Fallback data when enhanced system is unavailable"""
        return {
            'news_sentiment': 'neutral',
            'market_sentiment': 50,
            'news_count': 0,
            'sentiment_score': 0.0,
            'market_mood': 'uncertain',
            'risk_factors': [],
            'important_events': []
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.enhanced_analyzer:
                await self.enhanced_analyzer.cleanup()
        except Exception as e:
            logger.error(f"[CLEANUP] Error during cleanup: {e}")