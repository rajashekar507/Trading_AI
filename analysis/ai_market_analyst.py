"""
AI-Powered Market Analyst
Uses GPT-4 and Perplexity for advanced market analysis and insights
"""

import asyncio
import aiohttp
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import re

logger = logging.getLogger('trading_system.ai_market_analyst')

class AIMarketAnalyst:
    """AI-powered market analysis using GPT-4 and Perplexity"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
        
        # API endpoints
        self.openai_endpoint = "https://api.openai.com/v1/chat/completions"
        self.perplexity_endpoint = "https://api.perplexity.ai/chat/completions"
        
        # Rate limiting
        self.last_openai_request = 0
        self.last_perplexity_request = 0
        self.min_request_interval = 2.0  # 2 seconds between requests
        
        # Analysis cache
        self.analysis_cache = {}
        self.cache_duration = 600  # 10 minutes
        
        logger.info("[AI_ANALYST] AI Market Analyst initialized")
    
    async def _make_openai_request(self, messages: List[Dict], model: str = "gpt-4") -> Optional[str]:
        """Make request to OpenAI API"""
        try:
            # Rate limiting
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self.last_openai_request
            if time_since_last < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last)
            
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 1500,
                "temperature": 0.3,
                "top_p": 0.9
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.openai_endpoint, 
                                      headers=headers, 
                                      json=payload) as response:
                    
                    self.last_openai_request = asyncio.get_event_loop().time()
                    
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content']
                    else:
                        error_text = await response.text()
                        logger.error(f"[AI_ANALYST] OpenAI API error {response.status}: {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"[AI_ANALYST] OpenAI request failed: {e}")
            return None
    
    async def _make_perplexity_request(self, query: str) -> Optional[str]:
        """Make request to Perplexity API"""
        try:
            # Rate limiting
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self.last_perplexity_request
            if time_since_last < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last)
            
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial market research assistant. Provide accurate, up-to-date information about Indian and global financial markets."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.2
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.perplexity_endpoint, 
                                      headers=headers, 
                                      json=payload) as response:
                    
                    self.last_perplexity_request = asyncio.get_event_loop().time()
                    
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content']
                    else:
                        error_text = await response.text()
                        logger.error(f"[AI_ANALYST] Perplexity API error {response.status}: {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"[AI_ANALYST] Perplexity request failed: {e}")
            return None
    
    async def analyze_market_conditions(self, market_data: Dict, news_data: Dict) -> Dict:
        """Comprehensive AI-powered market analysis"""
        try:
            logger.info("[AI_ANALYST] Starting comprehensive market analysis...")
            
            # Check cache
            cache_key = f"market_analysis_{datetime.now().strftime('%Y%m%d_%H')}"
            if cache_key in self.analysis_cache:
                cached_analysis, timestamp = self.analysis_cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self.cache_duration):
                    logger.info("[AI_ANALYST] Using cached analysis")
                    return cached_analysis
            
            # Prepare market context
            market_context = self._prepare_market_context(market_data, news_data)
            
            # Run multiple AI analyses concurrently
            tasks = [
                self._analyze_technical_patterns(market_context),
                self._analyze_market_sentiment(news_data),
                self._research_market_events(),
                self._generate_trading_insights(market_context, news_data)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Compile comprehensive analysis
            analysis = {
                'timestamp': datetime.now(),
                'technical_analysis': results[0] if not isinstance(results[0], Exception) else {},
                'sentiment_analysis': results[1] if not isinstance(results[1], Exception) else {},
                'market_research': results[2] if not isinstance(results[2], Exception) else {},
                'trading_insights': results[3] if not isinstance(results[3], Exception) else {},
                'overall_recommendation': None,
                'confidence_score': 0.0,
                'risk_assessment': {}
            }
            
            # Generate overall recommendation
            analysis['overall_recommendation'] = await self._generate_overall_recommendation(analysis)
            analysis['confidence_score'] = self._calculate_confidence_score(analysis)
            analysis['risk_assessment'] = self._assess_risks(analysis)
            
            # Cache the analysis
            self.analysis_cache[cache_key] = (analysis, datetime.now())
            
            logger.info(f"[AI_ANALYST] Analysis complete - Recommendation: {analysis['overall_recommendation']}")
            return analysis
            
        except Exception as e:
            logger.error(f"[AI_ANALYST] Market analysis failed: {e}")
            return self._get_fallback_analysis()
    
    def _prepare_market_context(self, market_data: Dict, news_data: Dict) -> str:
        """Prepare market context for AI analysis"""
        try:
            context_parts = []
            
            # Market data summary
            if market_data:
                context_parts.append("CURRENT MARKET DATA:")
                if 'nifty' in market_data:
                    nifty_data = market_data['nifty']
                    context_parts.append(f"NIFTY 50: {nifty_data.get('ltp', 'N/A')} ({nifty_data.get('change_percent', 'N/A')}%)")
                
                if 'banknifty' in market_data:
                    bn_data = market_data['banknifty']
                    context_parts.append(f"BANK NIFTY: {bn_data.get('ltp', 'N/A')} ({bn_data.get('change_percent', 'N/A')}%)")
            
            # News summary
            if news_data and 'news_items' in news_data:
                context_parts.append("\nRECENT NEWS HEADLINES:")
                for item in news_data['news_items'][:5]:  # Top 5 news items
                    headline = item.get('headline', '')
                    source = item.get('source', '')
                    context_parts.append(f"- {headline} ({source})")
            
            # Market sentiment
            if news_data:
                sentiment = news_data.get('news_sentiment', 'neutral')
                score = news_data.get('sentiment_score', 0.0)
                context_parts.append(f"\nMARKET SENTIMENT: {sentiment} (Score: {score:.2f})")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"[AI_ANALYST] Error preparing market context: {e}")
            return "Market context unavailable"
    
    async def _analyze_technical_patterns(self, market_context: str) -> Dict:
        """AI-powered technical pattern analysis"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert technical analyst specializing in Indian stock markets. 
                    Analyze the provided market data and identify key technical patterns, support/resistance levels, 
                    and potential trading opportunities. Focus on NIFTY and BANK NIFTY."""
                },
                {
                    "role": "user",
                    "content": f"""Analyze the following market data and provide technical insights:

{market_context}

Please provide:
1. Key technical patterns identified
2. Support and resistance levels
3. Momentum indicators assessment
4. Short-term outlook (1-3 days)
5. Key levels to watch

Format your response as structured analysis."""
                }
            ]
            
            response = await self._make_openai_request(messages)
            
            if response:
                return {
                    'analysis': response,
                    'patterns_identified': self._extract_patterns(response),
                    'key_levels': self._extract_levels(response),
                    'outlook': self._extract_outlook(response)
                }
            else:
                return {'analysis': 'Technical analysis unavailable', 'patterns_identified': [], 'key_levels': {}, 'outlook': 'neutral'}
                
        except Exception as e:
            logger.error(f"[AI_ANALYST] Technical analysis failed: {e}")
            return {'analysis': 'Technical analysis failed', 'patterns_identified': [], 'key_levels': {}, 'outlook': 'neutral'}
    
    async def _analyze_market_sentiment(self, news_data: Dict) -> Dict:
        """AI-powered sentiment analysis"""
        try:
            if not news_data or not news_data.get('news_items'):
                return {'analysis': 'No news data available', 'sentiment': 'neutral', 'key_themes': []}
            
            # Prepare news summary
            news_summary = []
            for item in news_data['news_items'][:10]:
                headline = item.get('headline', '')
                source = item.get('source', '')
                news_summary.append(f"- {headline} ({source})")
            
            news_text = "\n".join(news_summary)
            
            messages = [
                {
                    "role": "system",
                    "content": """You are a financial sentiment analyst. Analyze news headlines and market sentiment 
                    to determine the overall market mood and identify key themes affecting Indian markets."""
                },
                {
                    "role": "user",
                    "content": f"""Analyze the sentiment of these recent market news headlines:

{news_text}

Please provide:
1. Overall market sentiment (bullish/bearish/neutral)
2. Key themes and concerns
3. Potential market impact
4. Sentiment strength (1-10 scale)
5. Risk factors identified

Focus on Indian market context."""
                }
            ]
            
            response = await self._make_openai_request(messages)
            
            if response:
                return {
                    'analysis': response,
                    'sentiment': self._extract_sentiment(response),
                    'key_themes': self._extract_themes(response),
                    'risk_factors': self._extract_risk_factors(response)
                }
            else:
                return {'analysis': 'Sentiment analysis unavailable', 'sentiment': 'neutral', 'key_themes': [], 'risk_factors': []}
                
        except Exception as e:
            logger.error(f"[AI_ANALYST] Sentiment analysis failed: {e}")
            return {'analysis': 'Sentiment analysis failed', 'sentiment': 'neutral', 'key_themes': [], 'risk_factors': []}
    
    async def _research_market_events(self) -> Dict:
        """Research current market events using Perplexity"""
        try:
            query = """What are the most important financial market events, policy decisions, 
            and economic developments affecting Indian stock markets (NSE, BSE) in the last 24 hours? 
            Include any RBI announcements, government policies, FII/DII flows, and global market impacts."""
            
            response = await self._make_perplexity_request(query)
            
            if response:
                return {
                    'research': response,
                    'key_events': self._extract_events(response),
                    'policy_updates': self._extract_policies(response),
                    'global_impact': self._extract_global_factors(response)
                }
            else:
                return {'research': 'Market research unavailable', 'key_events': [], 'policy_updates': [], 'global_impact': []}
                
        except Exception as e:
            logger.error(f"[AI_ANALYST] Market research failed: {e}")
            return {'research': 'Market research failed', 'key_events': [], 'policy_updates': [], 'global_impact': []}
    
    async def _generate_trading_insights(self, market_context: str, news_data: Dict) -> Dict:
        """Generate AI-powered trading insights"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are a professional trading strategist specializing in Indian options and equity markets. 
                    Provide actionable trading insights based on market data and news analysis. Focus on risk management 
                    and SEBI-compliant strategies."""
                },
                {
                    "role": "user",
                    "content": f"""Based on the following market conditions and news, provide trading insights:

MARKET CONTEXT:
{market_context}

NEWS SENTIMENT: {news_data.get('news_sentiment', 'neutral')}
MARKET MOOD: {news_data.get('market_mood', 'uncertain')}

Please provide:
1. Trading opportunities (if any)
2. Risk management recommendations
3. Key levels for entry/exit
4. Time horizon for trades
5. Position sizing suggestions
6. Market regime assessment

Focus on NIFTY and BANK NIFTY options strategies."""
                }
            ]
            
            response = await self._make_openai_request(messages)
            
            if response:
                return {
                    'insights': response,
                    'opportunities': self._extract_opportunities(response),
                    'risk_management': self._extract_risk_management(response),
                    'key_levels': self._extract_trading_levels(response)
                }
            else:
                return {'insights': 'Trading insights unavailable', 'opportunities': [], 'risk_management': [], 'key_levels': {}}
                
        except Exception as e:
            logger.error(f"[AI_ANALYST] Trading insights generation failed: {e}")
            return {'insights': 'Trading insights failed', 'opportunities': [], 'risk_management': [], 'key_levels': {}}
    
    async def _generate_overall_recommendation(self, analysis: Dict) -> str:
        """Generate overall trading recommendation"""
        try:
            # Combine all analyses
            combined_analysis = f"""
TECHNICAL ANALYSIS: {analysis.get('technical_analysis', {}).get('outlook', 'neutral')}
SENTIMENT ANALYSIS: {analysis.get('sentiment_analysis', {}).get('sentiment', 'neutral')}
MARKET RESEARCH: Available
TRADING INSIGHTS: Available
"""
            
            messages = [
                {
                    "role": "system",
                    "content": """You are a senior portfolio manager. Based on comprehensive market analysis, 
                    provide a clear, actionable overall recommendation. Be conservative and focus on risk management."""
                },
                {
                    "role": "user",
                    "content": f"""Based on this comprehensive analysis, what is your overall recommendation?

{combined_analysis}

Provide a single clear recommendation: BULLISH, BEARISH, NEUTRAL, or WAIT_AND_WATCH with brief reasoning."""
                }
            ]
            
            response = await self._make_openai_request(messages)
            
            if response:
                # Extract recommendation
                if 'BULLISH' in response.upper():
                    return 'BULLISH'
                elif 'BEARISH' in response.upper():
                    return 'BEARISH'
                elif 'WAIT' in response.upper() or 'WATCH' in response.upper():
                    return 'WAIT_AND_WATCH'
                else:
                    return 'NEUTRAL'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"[AI_ANALYST] Overall recommendation generation failed: {e}")
            return 'NEUTRAL'
    
    def _calculate_confidence_score(self, analysis: Dict) -> float:
        """Calculate confidence score for the analysis"""
        try:
            score = 0.5  # Base score
            
            # Technical analysis available
            if analysis.get('technical_analysis', {}).get('analysis'):
                score += 0.15
            
            # Sentiment analysis available
            if analysis.get('sentiment_analysis', {}).get('analysis'):
                score += 0.15
            
            # Market research available
            if analysis.get('market_research', {}).get('research'):
                score += 0.1
            
            # Trading insights available
            if analysis.get('trading_insights', {}).get('insights'):
                score += 0.1
            
            return min(0.9, score)  # Cap at 90%
            
        except Exception as e:
            logger.error(f"[AI_ANALYST] Confidence calculation failed: {e}")
            return 0.5
    
    def _assess_risks(self, analysis: Dict) -> Dict:
        """Assess overall risks"""
        try:
            risks = []
            
            # Collect risk factors from different analyses
            sentiment_risks = analysis.get('sentiment_analysis', {}).get('risk_factors', [])
            risks.extend(sentiment_risks)
            
            # Market research risks
            if 'policy' in str(analysis.get('market_research', {})).lower():
                risks.append('Policy uncertainty')
            
            # Technical risks
            technical_outlook = analysis.get('technical_analysis', {}).get('outlook', 'neutral')
            if technical_outlook == 'bearish':
                risks.append('Negative technical outlook')
            
            return {
                'risk_factors': risks,
                'risk_level': 'HIGH' if len(risks) > 3 else 'MEDIUM' if len(risks) > 1 else 'LOW',
                'risk_count': len(risks)
            }
            
        except Exception as e:
            logger.error(f"[AI_ANALYST] Risk assessment failed: {e}")
            return {'risk_factors': [], 'risk_level': 'MEDIUM', 'risk_count': 0}
    
    # Helper methods for extracting information from AI responses
    def _extract_patterns(self, text: str) -> List[str]:
        """Extract technical patterns from analysis"""
        patterns = []
        pattern_keywords = ['triangle', 'flag', 'pennant', 'head and shoulders', 'double top', 'double bottom', 'support', 'resistance']
        text_lower = text.lower()
        for keyword in pattern_keywords:
            if keyword in text_lower:
                patterns.append(keyword.title())
        return patterns
    
    def _extract_levels(self, text: str) -> Dict:
        """Extract key levels from analysis"""
        levels = {}
        # Simple regex to find numbers that might be levels
        numbers = re.findall(r'\b\d{4,5}\b', text)
        if numbers:
            levels['support'] = min([int(n) for n in numbers])
            levels['resistance'] = max([int(n) for n in numbers])
        return levels
    
    def _extract_outlook(self, text: str) -> str:
        """Extract outlook from analysis"""
        text_lower = text.lower()
        if 'bullish' in text_lower or 'positive' in text_lower:
            return 'bullish'
        elif 'bearish' in text_lower or 'negative' in text_lower:
            return 'bearish'
        else:
            return 'neutral'
    
    def _extract_sentiment(self, text: str) -> str:
        """Extract sentiment from analysis"""
        text_lower = text.lower()
        if 'bullish' in text_lower:
            return 'bullish'
        elif 'bearish' in text_lower:
            return 'bearish'
        else:
            return 'neutral'
    
    def _extract_themes(self, text: str) -> List[str]:
        """Extract key themes from analysis"""
        themes = []
        theme_keywords = ['inflation', 'interest rates', 'earnings', 'policy', 'global', 'fii', 'dii', 'banking', 'it', 'pharma']
        text_lower = text.lower()
        for keyword in theme_keywords:
            if keyword in text_lower:
                themes.append(keyword.title())
        return themes
    
    def _extract_risk_factors(self, text: str) -> List[str]:
        """Extract risk factors from analysis"""
        risks = []
        risk_keywords = ['risk', 'concern', 'uncertainty', 'volatility', 'pressure', 'weakness']
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in risk_keywords):
                risks.append(line.strip())
        return risks[:5]  # Limit to 5 risks
    
    def _extract_events(self, text: str) -> List[str]:
        """Extract key events from research"""
        events = []
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['rbi', 'sebi', 'budget', 'policy', 'announcement']):
                events.append(line.strip())
        return events[:5]
    
    def _extract_policies(self, text: str) -> List[str]:
        """Extract policy updates from research"""
        policies = []
        lines = text.split('\n')
        for line in lines:
            if 'policy' in line.lower() or 'regulation' in line.lower():
                policies.append(line.strip())
        return policies[:3]
    
    def _extract_global_factors(self, text: str) -> List[str]:
        """Extract global factors from research"""
        factors = []
        global_keywords = ['fed', 'us', 'china', 'global', 'international', 'world']
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in global_keywords):
                factors.append(line.strip())
        return factors[:3]
    
    def _extract_opportunities(self, text: str) -> List[str]:
        """Extract trading opportunities"""
        opportunities = []
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['buy', 'sell', 'opportunity', 'trade', 'position']):
                opportunities.append(line.strip())
        return opportunities[:5]
    
    def _extract_risk_management(self, text: str) -> List[str]:
        """Extract risk management recommendations"""
        risk_mgmt = []
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['stop', 'risk', 'position size', 'hedge', 'protect']):
                risk_mgmt.append(line.strip())
        return risk_mgmt[:5]
    
    def _extract_trading_levels(self, text: str) -> Dict:
        """Extract trading levels"""
        levels = {}
        numbers = re.findall(r'\b\d{4,5}\b', text)
        if numbers:
            levels['entry'] = numbers[0] if numbers else None
            levels['target'] = numbers[1] if len(numbers) > 1 else None
            levels['stop_loss'] = numbers[2] if len(numbers) > 2 else None
        return levels
    
    def _get_fallback_analysis(self) -> Dict:
        """Fallback analysis when AI systems fail"""
        return {
            'timestamp': datetime.now(),
            'technical_analysis': {'analysis': 'Technical analysis unavailable', 'outlook': 'neutral'},
            'sentiment_analysis': {'analysis': 'Sentiment analysis unavailable', 'sentiment': 'neutral'},
            'market_research': {'research': 'Market research unavailable'},
            'trading_insights': {'insights': 'Trading insights unavailable'},
            'overall_recommendation': 'NEUTRAL',
            'confidence_score': 0.3,
            'risk_assessment': {'risk_level': 'MEDIUM', 'risk_factors': ['AI analysis unavailable']}
        }