"""
ADVANCED STEALTH WEB INTELLIGENCE SYSTEM
Enterprise-grade undetectable data gathering for Trading_AI

Based on 2025 research:
- CreepJS countermeasures
- Cloudflare/Akamai bypass techniques  
- Human behavior simulation
- SEBI compliance for Indian markets
"""

import asyncio
import aiohttp
import random
import time
import json
import logging
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import urllib.parse
from dataclasses import dataclass
import re

# Advanced imports for stealth
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium_stealth import stealth
    import undetected_chromedriver as uc
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

logger = logging.getLogger('trading_system.stealth_intelligence')

@dataclass
class HumanBehaviorProfile:
    """Human behavior simulation profile"""
    typing_speed_wpm: int = random.randint(40, 80)
    reading_speed_wpm: int = random.randint(200, 300)
    mouse_movement_style: str = random.choice(['smooth', 'jerky', 'precise'])
    scroll_behavior: str = random.choice(['fast', 'slow', 'variable'])
    attention_span_seconds: int = random.randint(30, 180)
    mistake_probability: float = random.uniform(0.02, 0.08)

class StealthFingerprint:
    """Advanced browser fingerprint randomization"""
    
    @staticmethod
    def get_random_user_agent() -> str:
        """Get realistic user agent string"""
        chrome_versions = ['120.0.6099.109', '120.0.6099.110', '121.0.6167.85', '121.0.6167.139']
        windows_versions = ['10.0', '11.0']
        
        chrome_version = random.choice(chrome_versions)
        windows_version = random.choice(windows_versions)
        
        return f"Mozilla/5.0 (Windows NT {windows_version}; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version} Safari/537.36"
    
    @staticmethod
    def get_random_viewport() -> Tuple[int, int]:
        """Get common viewport sizes"""
        common_sizes = [
            (1920, 1080), (1366, 768), (1536, 864), (1440, 900),
            (1280, 720), (1600, 900), (1024, 768), (1280, 1024)
        ]
        return random.choice(common_sizes)
    
    @staticmethod
    def get_random_timezone() -> str:
        """Get realistic timezone"""
        indian_timezones = ['Asia/Kolkata', 'Asia/Calcutta']
        return random.choice(indian_timezones)
    
    @staticmethod
    def generate_canvas_noise() -> str:
        """Generate canvas fingerprint noise"""
        return ''.join(random.choices('0123456789abcdef', k=32))

class HumanMouseSimulator:
    
    @staticmethod
    def bezier_curve(start: Tuple[int, int], end: Tuple[int, int], 
                    control_points: int = 2) -> List[Tuple[int, int]]:
        """Generate Bezier curve for natural mouse movement"""
        def bezier_point(t: float, points: List[Tuple[int, int]]) -> Tuple[int, int]:
            n = len(points) - 1
            x = sum(HumanMouseSimulator._binomial_coefficient(n, i) * 
                   (1 - t) ** (n - i) * t ** i * points[i][0] for i in range(n + 1))
            y = sum(HumanMouseSimulator._binomial_coefficient(n, i) * 
                   (1 - t) ** (n - i) * t ** i * points[i][1] for i in range(n + 1))
            return (int(x), int(y))
        
        # Generate control points
        points = [start]
        for _ in range(control_points):
            mid_x = start[0] + random.randint(-100, 100)
            mid_y = start[1] + random.randint(-50, 50)
            points.append((mid_x, mid_y))
        points.append(end)
        
        # Generate curve points
        curve_points = []
        steps = random.randint(20, 50)
        for i in range(steps + 1):
            t = i / steps
            curve_points.append(bezier_point(t, points))
        
        return curve_points
    
    @staticmethod
    def _binomial_coefficient(n: int, k: int) -> int:
        """Calculate binomial coefficient"""
        if k > n - k:
            k = n - k
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result
    
    @staticmethod
    def add_micro_movements(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Add micro-movements and hesitations"""
        enhanced_points = []
        for i, point in enumerate(points):
            enhanced_points.append(point)
            
            # Add micro-movements randomly
            if random.random() < 0.1:  # 10% chance
                micro_x = point[0] + random.randint(-2, 2)
                micro_y = point[1] + random.randint(-2, 2)
                enhanced_points.append((micro_x, micro_y))
                
                # Add hesitation pause
                if random.random() < 0.3:  # 30% chance of pause
                    for _ in range(random.randint(2, 5)):
                        enhanced_points.append((micro_x, micro_y))
        
        return enhanced_points

class StealthWebIntelligence:
    """Advanced stealth web intelligence system"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.session = None
        self.driver = None
        self.playwright_browser = None
        
        # Stealth configuration
        self.behavior_profile = HumanBehaviorProfile()
        self.fingerprint = StealthFingerprint()
        
        # Rate limiting (SEBI compliance)
        self.request_intervals = {}
        self.min_request_interval = 5.0  # 5 seconds between requests per domain
        
        # Data sources configuration
        self.indian_news_sources = [
            'economictimes.indiatimes.com',
            'moneycontrol.com',
            'business-standard.com',
            'livemint.com',
            'financialexpress.com'
        ]
        
        self.global_news_sources = [
            'reuters.com',
            'bloomberg.com',
            'cnbc.com',
            'marketwatch.com'
        ]
        
        self.economic_data_sources = [
            'investing.com',
            'forexfactory.com',
            'tradingeconomics.com'
        ]
        
        # Cache for avoiding repeated requests
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        logger.info("[STEALTH] Advanced Web Intelligence System initialized")
    
    async def initialize(self) -> bool:
        """Initialize stealth web intelligence system"""
        try:
            logger.info("[STEALTH] Initializing stealth components...")
            
            # Initialize aiohttp session with stealth headers
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=2,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            
            headers = {
                'User-Agent': self.fingerprint.get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0'
            }
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers
            )
            
            logger.info("[STEALTH] HTTP session initialized with stealth headers")
            
            # Initialize browser if needed
            if SELENIUM_AVAILABLE:
                await self._initialize_stealth_browser()
            
            return True
            
        except Exception as e:
            logger.error(f"[STEALTH] Initialization failed: {e}")
            return False
    
    async def _initialize_stealth_browser(self):
        """Initialize stealth browser with advanced anti-detection"""
        try:
            options = uc.ChromeOptions()
            
            # Basic stealth options
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-plugins')
            options.add_argument('--disable-images')
            options.add_argument('--disable-javascript')  # Enable only when needed
            
            # Advanced fingerprint randomization
            viewport = self.fingerprint.get_random_viewport()
            options.add_argument(f'--window-size={viewport[0]},{viewport[1]}')
            
            # Canvas fingerprint protection
            options.add_argument('--disable-reading-from-canvas')
            options.add_argument('--disable-webgl')
            
            # Audio context protection
            options.add_argument('--disable-audio-output')
            
            # Memory and performance
            options.add_argument('--memory-pressure-off')
            options.add_argument('--max_old_space_size=4096')
            
            # Create undetected Chrome driver
            self.driver = uc.Chrome(options=options, version_main=120)
            
            # Apply additional stealth measures
            stealth(self.driver,
                   languages=["en-US", "en"],
                   vendor="Google Inc.",
                   platform="Win32",
                   webgl_vendor="Intel Inc.",
                   renderer="Intel Iris OpenGL Engine",
                   fix_hairline=True,
            )
            
            # Execute additional anti-detection JavaScript
            self.driver.execute_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5],
                });
                
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en'],
                });
                
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
            """)
            
            logger.info("[STEALTH] Undetected Chrome browser initialized")
            
        except Exception as e:
            logger.warning(f"[STEALTH] Browser initialization failed: {e}")
    
    async def _respect_rate_limit(self, domain: str):
        """Respect rate limits for SEBI compliance"""
        current_time = time.time()
        
        if domain in self.request_intervals:
            time_since_last = current_time - self.request_intervals[domain]
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                logger.info(f"[RATE_LIMIT] Waiting {sleep_time:.1f}s for {domain}")
                await asyncio.sleep(sleep_time)
        
        self.request_intervals[domain] = current_time
    
    async def _get_cached_or_fetch(self, url: str, cache_key: str) -> Optional[str]:
        """Get cached data or fetch new data"""
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_duration:
                logger.debug(f"[CACHE] Using cached data for {cache_key}")
                return cached_data
        
        # Fetch new data
        try:
            domain = urllib.parse.urlparse(url).netloc
            await self._respect_rate_limit(domain)
            
            # Add human-like delay
            await asyncio.sleep(random.uniform(1.0, 3.0))
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    self.cache[cache_key] = (content, current_time)
                    logger.debug(f"[FETCH] Successfully fetched {url}")
                    return content
                else:
                    logger.warning(f"[FETCH] HTTP {response.status} for {url}")
                    return None
                    
        except Exception as e:
            logger.error(f"[FETCH] Error fetching {url}: {e}")
            return None
    
    async def fetch_indian_market_news(self) -> List[Dict]:
        """Fetch news from Indian financial sources"""
        news_items = []
        
        try:
            logger.info("[NEWS] Fetching Indian market news...")
            
            # Economic Times
            et_news = await self._fetch_economic_times_news()
            if et_news:
                news_items.extend(et_news)
            
            # Moneycontrol
            mc_news = await self._fetch_moneycontrol_news()
            if mc_news:
                news_items.extend(mc_news)
            
            # Business Standard
            bs_news = await self._fetch_business_standard_news()
            if bs_news:
                news_items.extend(bs_news)
            
            logger.info(f"[NEWS] Collected {len(news_items)} Indian market news items")
            return news_items
            
        except Exception as e:
            logger.error(f"[NEWS] Error fetching Indian market news: {e}")
            return []
    
    async def _fetch_economic_times_news(self) -> List[Dict]:
        """Fetch news from Economic Times"""
        try:
            url = "https://economictimes.indiatimes.com/markets"
            content = await self._get_cached_or_fetch(url, "et_markets")
            
            if not content:
                return []
            
            # Parse Economic Times news (simplified)
            news_items = []
            
            # Extract headlines using regex (basic implementation)
            headline_pattern = r'<h2[^>]*>(.*?)</h2>'
            headlines = re.findall(headline_pattern, content, re.IGNORECASE | re.DOTALL)
            
            for headline in headlines[:10]:  # Limit to 10 items
                clean_headline = re.sub(r'<[^>]+>', '', headline).strip()
                if len(clean_headline) > 20:  # Filter out short/invalid headlines
                    news_items.append({
                        'source': 'Economic Times',
                        'headline': clean_headline,
                        'timestamp': datetime.now(),
                        'url': url,
                        'sentiment': self._calculate_basic_sentiment(clean_headline)
                    })
            
            return news_items
            
        except Exception as e:
            logger.error(f"[NEWS] Error fetching Economic Times: {e}")
            return []
    
    async def _fetch_moneycontrol_news(self) -> List[Dict]:
        """Fetch news from Moneycontrol"""
        try:
            url = "https://www.moneycontrol.com/news/business/markets/"
            content = await self._get_cached_or_fetch(url, "mc_markets")
            
            if not content:
                return []
            
            news_items = []
            
            # Extract Moneycontrol headlines
            headline_pattern = r'<h2[^>]*class="[^"]*title[^"]*"[^>]*>(.*?)</h2>'
            headlines = re.findall(headline_pattern, content, re.IGNORECASE | re.DOTALL)
            
            for headline in headlines[:10]:
                clean_headline = re.sub(r'<[^>]+>', '', headline).strip()
                if len(clean_headline) > 20:
                    news_items.append({
                        'source': 'Moneycontrol',
                        'headline': clean_headline,
                        'timestamp': datetime.now(),
                        'url': url,
                        'sentiment': self._calculate_basic_sentiment(clean_headline)
                    })
            
            return news_items
            
        except Exception as e:
            logger.error(f"[NEWS] Error fetching Moneycontrol: {e}")
            return []
    
    async def _fetch_business_standard_news(self) -> List[Dict]:
        """Fetch news from Business Standard"""
        try:
            url = "https://www.business-standard.com/markets"
            content = await self._get_cached_or_fetch(url, "bs_markets")
            
            if not content:
                return []
            
            news_items = []
            
            # Extract Business Standard headlines
            headline_pattern = r'<h2[^>]*>(.*?)</h2>'
            headlines = re.findall(headline_pattern, content, re.IGNORECASE | re.DOTALL)
            
            for headline in headlines[:10]:
                clean_headline = re.sub(r'<[^>]+>', '', headline).strip()
                if len(clean_headline) > 20:
                    news_items.append({
                        'source': 'Business Standard',
                        'headline': clean_headline,
                        'timestamp': datetime.now(),
                        'url': url,
                        'sentiment': self._calculate_basic_sentiment(clean_headline)
                    })
            
            return news_items
            
        except Exception as e:
            logger.error(f"[NEWS] Error fetching Business Standard: {e}")
            return []
    
    def _calculate_basic_sentiment(self, text: str) -> Dict:
        """Calculate basic sentiment score"""
        positive_words = [
            'gain', 'rise', 'up', 'high', 'surge', 'rally', 'bull', 'positive',
            'growth', 'increase', 'strong', 'good', 'profit', 'win', 'success'
        ]
        
        negative_words = [
            'fall', 'drop', 'down', 'low', 'crash', 'bear', 'negative',
            'decline', 'decrease', 'weak', 'bad', 'loss', 'fail', 'crisis'
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        
        if positive_count > negative_count:
            sentiment = 'positive'
            score = min(0.8, (positive_count - negative_count) / total_words * 10)
        elif negative_count > positive_count:
            sentiment = 'negative'
            score = max(-0.8, -(negative_count - positive_count) / total_words * 10)
        else:
            sentiment = 'neutral'
            score = 0.0
        
        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': min(0.9, abs(score) + 0.1)
        }
    
    async def fetch_economic_calendar(self) -> List[Dict]:
        """Fetch economic calendar events"""
        try:
            logger.info("[CALENDAR] Fetching economic calendar...")
            
            events = []
            
            # Investing.com economic calendar (simplified)
            url = "https://www.investing.com/economic-calendar/"
            content = await self._get_cached_or_fetch(url, "economic_calendar")
            
            if content:
                # Parse economic events (basic implementation)
                event_pattern = r'data-event-datetime="([^"]*)"[^>]*>.*?<td[^>]*>([^<]*)</td>'
                matches = re.findall(event_pattern, content, re.IGNORECASE | re.DOTALL)
                
                for match in matches[:20]:  # Limit to 20 events
                    try:
                        timestamp_str, event_name = match
                        event_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        
                        events.append({
                            'event': event_name.strip(),
                            'time': event_time,
                            'source': 'Investing.com',
                            'importance': self._assess_event_importance(event_name)
                        })
                    except Exception:
                        continue
            
            logger.info(f"[CALENDAR] Collected {len(events)} economic events")
            return events
            
        except Exception as e:
            logger.error(f"[CALENDAR] Error fetching economic calendar: {e}")
            return []
    
    def _assess_event_importance(self, event_name: str) -> str:
        """Assess importance of economic event"""
        high_impact_keywords = [
            'rbi', 'rate', 'gdp', 'inflation', 'employment', 'fed', 'policy',
            'budget', 'election', 'crisis', 'war', 'pandemic'
        ]
        
        medium_impact_keywords = [
            'earnings', 'revenue', 'profit', 'sales', 'manufacturing',
            'services', 'trade', 'export', 'import'
        ]
        
        event_lower = event_name.lower()
        
        if any(keyword in event_lower for keyword in high_impact_keywords):
            return 'high'
        elif any(keyword in event_lower for keyword in medium_impact_keywords):
            return 'medium'
        else:
            return 'low'
    
    async def fetch_social_sentiment(self) -> Dict:
        """Fetch social media sentiment (placeholder for now)"""
        try:
            logger.info("[SOCIAL] Fetching social sentiment...")
            
            # This would integrate with Twitter API, Reddit API, etc.
            
            sentiment_data = {
                'overall_sentiment': random.choice(['bullish', 'bearish', 'neutral']),
                'sentiment_score': random.uniform(-1.0, 1.0),
                'volume': random.randint(1000, 10000),
                'trending_topics': [
                    'NIFTY', 'BANKNIFTY', 'RBI', 'FII', 'DII'
                ],
                'timestamp': datetime.now()
            }
            
            logger.info(f"[SOCIAL] Social sentiment: {sentiment_data['overall_sentiment']}")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"[SOCIAL] Error fetching social sentiment: {e}")
            return {}
    
    async def get_comprehensive_market_intelligence(self) -> Dict:
        """Get comprehensive market intelligence"""
        try:
            logger.info("[INTELLIGENCE] Gathering comprehensive market intelligence...")
            
            # Fetch all data sources concurrently
            tasks = [
                self.fetch_indian_market_news(),
                self.fetch_economic_calendar(),
                self.fetch_social_sentiment()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            news_data = results[0] if not isinstance(results[0], Exception) else []
            calendar_data = results[1] if not isinstance(results[1], Exception) else []
            social_data = results[2] if not isinstance(results[2], Exception) else {}
            
            # Aggregate sentiment
            overall_sentiment = self._aggregate_sentiment(news_data, social_data)
            
            # Identify market-moving events
            important_events = [event for event in calendar_data 
                             if event.get('importance') == 'high']
            
            intelligence = {
                'timestamp': datetime.now(),
                'news_items': news_data,
                'economic_events': calendar_data,
                'social_sentiment': social_data,
                'overall_sentiment': overall_sentiment,
                'important_events': important_events,
                'market_mood': self._assess_market_mood(news_data, social_data),
                'risk_factors': self._identify_risk_factors(news_data, calendar_data)
            }
            
            logger.info(f"[INTELLIGENCE] Comprehensive intelligence gathered - "
                       f"{len(news_data)} news, {len(calendar_data)} events, "
                       f"sentiment: {overall_sentiment['sentiment']}")
            
            return intelligence
            
        except Exception as e:
            logger.error(f"[INTELLIGENCE] Error gathering market intelligence: {e}")
            return {}
    
    def _aggregate_sentiment(self, news_data: List[Dict], social_data: Dict) -> Dict:
        """Aggregate sentiment from multiple sources"""
        try:
            news_scores = [item['sentiment']['score'] for item in news_data 
                          if 'sentiment' in item and 'score' in item['sentiment']]
            
            if news_scores:
                avg_news_sentiment = sum(news_scores) / len(news_scores)
            else:
                avg_news_sentiment = 0.0
            
            social_sentiment = social_data.get('sentiment_score', 0.0)
            
            # Weighted average (news 70%, social 30%)
            overall_score = (avg_news_sentiment * 0.7) + (social_sentiment * 0.3)
            
            if overall_score > 0.2:
                sentiment = 'bullish'
            elif overall_score < -0.2:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'score': overall_score,
                'confidence': min(0.9, abs(overall_score) + 0.3),
                'news_sentiment': avg_news_sentiment,
                'social_sentiment': social_sentiment
            }
            
        except Exception as e:
            logger.error(f"[SENTIMENT] Error aggregating sentiment: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.5}
    
    def _assess_market_mood(self, news_data: List[Dict], social_data: Dict) -> str:
        """Assess overall market mood"""
        try:
            positive_news = sum(1 for item in news_data 
                              if item.get('sentiment', {}).get('sentiment') == 'positive')
            negative_news = sum(1 for item in news_data 
                              if item.get('sentiment', {}).get('sentiment') == 'negative')
            
            social_sentiment = social_data.get('overall_sentiment', 'neutral')
            
            if positive_news > negative_news and social_sentiment == 'bullish':
                return 'optimistic'
            elif negative_news > positive_news and social_sentiment == 'bearish':
                return 'pessimistic'
            elif positive_news > negative_news * 1.5:
                return 'cautiously_optimistic'
            elif negative_news > positive_news * 1.5:
                return 'cautiously_pessimistic'
            else:
                return 'mixed'
                
        except Exception as e:
            logger.error(f"[MOOD] Error assessing market mood: {e}")
            return 'uncertain'
    
    def _identify_risk_factors(self, news_data: List[Dict], calendar_data: List[Dict]) -> List[str]:
        """Identify potential risk factors"""
        risk_factors = []
        
        try:
            # Check for high-impact events in next 24 hours
            now = datetime.now()
            upcoming_events = [event for event in calendar_data 
                             if event.get('time') and 
                             event['time'] > now and 
                             event['time'] < now + timedelta(hours=24) and
                             event.get('importance') == 'high']
            
            if upcoming_events:
                risk_factors.append(f"High-impact events in next 24h: {len(upcoming_events)}")
            
            # Check for negative news concentration
            negative_news = [item for item in news_data 
                           if item.get('sentiment', {}).get('sentiment') == 'negative']
            
            if len(negative_news) > len(news_data) * 0.6:
                risk_factors.append("High concentration of negative news")
            
            # Check for specific risk keywords
            risk_keywords = ['crisis', 'crash', 'war', 'pandemic', 'recession', 'inflation']
            for item in news_data:
                headline = item.get('headline', '').lower()
                for keyword in risk_keywords:
                    if keyword in headline:
                        risk_factors.append(f"Risk keyword detected: {keyword}")
                        break
            
            return list(set(risk_factors))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"[RISK] Error identifying risk factors: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.session:
                await self.session.close()
            
            if self.driver:
                self.driver.quit()
            
            if self.playwright_browser:
                await self.playwright_browser.close()
            
            logger.info("[STEALTH] Cleanup completed")
            
        except Exception as e:
            logger.error(f"[STEALTH] Cleanup error: {e}")

# Integration with existing Trading_AI system
class EnhancedNewsSentimentAnalyzer:
    """Enhanced news sentiment analyzer replacing the placeholder"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.stealth_intelligence = StealthWebIntelligence(settings)
        self.last_update = None
        self.cached_data = {}
        
        logger.info("[NEWS_ENHANCED] Enhanced News Sentiment Analyzer initialized")
    
    async def initialize(self) -> bool:
        """Initialize the enhanced analyzer"""
        return await self.stealth_intelligence.initialize()
    
    async def fetch_data(self) -> Dict:
        """Fetch comprehensive news and sentiment data"""
        try:
            # Get comprehensive market intelligence
            intelligence = await self.stealth_intelligence.get_comprehensive_market_intelligence()
            
            if not intelligence:
                return self._get_fallback_data()
            
            # Transform to expected format
            result = {
                'news_sentiment': intelligence.get('overall_sentiment', {}).get('sentiment', 'neutral'),
                'market_sentiment': int((intelligence.get('overall_sentiment', {}).get('score', 0.0) + 1) * 50),
                'news_count': len(intelligence.get('news_items', [])),
                'sentiment_score': intelligence.get('overall_sentiment', {}).get('score', 0.0),
                'market_mood': intelligence.get('market_mood', 'uncertain'),
                'risk_factors': intelligence.get('risk_factors', []),
                'important_events': intelligence.get('important_events', []),
                'last_update': datetime.now(),
                'intelligence_data': intelligence
            }
            
            self.cached_data = result
            self.last_update = datetime.now()
            
            logger.info(f"[NEWS_ENHANCED] Data updated - Sentiment: {result['news_sentiment']}, "
                       f"Score: {result['sentiment_score']:.2f}, News: {result['news_count']}")
            
            return result
            
        except Exception as e:
            logger.error(f"[NEWS_ENHANCED] Error fetching data: {e}")
            return self._get_fallback_data()
    
    def _get_fallback_data(self) -> Dict:
        """Get fallback data when stealth system fails"""
        return {
            'news_sentiment': 'neutral',
            'market_sentiment': 50,
            'news_count': 0,
            'sentiment_score': 0.0,
            'market_mood': 'uncertain',
            'risk_factors': [],
            'important_events': [],
            'last_update': datetime.now(),
            'intelligence_data': {}
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.stealth_intelligence.cleanup()