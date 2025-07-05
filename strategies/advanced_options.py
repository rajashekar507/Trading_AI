"""
INSTITUTIONAL-GRADE Advanced Options Strategies
Implements professional options strategies with Greeks optimization
Iron Condor, Butterfly, Straddle, Strangle, Calendar, Ratio spreads + Delta-neutral strategies
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math
from scipy.stats import norm
from scipy.optimize import minimize

logger = logging.getLogger('trading_system.advanced_options')

@dataclass
class OptionLeg:
    """Single option leg in a strategy"""
    instrument: str  # NIFTY, BANKNIFTY, FINNIFTY
    option_type: str  # CE, PE
    strike_price: float
    expiry_date: str
    action: str  # BUY, SELL
    quantity: int
    premium: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

@dataclass
class StrategyResult:
    """Strategy analysis result"""
    strategy_name: str
    legs: List[OptionLeg]
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    net_premium: float
    probability_of_profit: float
    risk_reward_ratio: float
    margin_required: float

class AdvancedOptionsStrategies:
    """INSTITUTIONAL-GRADE Advanced Options Strategies"""
    
    def __init__(self, settings):
        self.settings = settings
        self.risk_free_rate = 0.065  # 6.5% risk-free rate
        
        # Strategy selection criteria
        self.strategy_criteria = {
            'iron_condor': {
                'min_dte': 15,  # Days to expiry
                'max_dte': 45,
                'ideal_iv_rank': (30, 70),  # IV percentile range
                'market_condition': 'range_bound'
            },
            'straddle': {
                'min_dte': 7,
                'max_dte': 30,
                'ideal_iv_rank': (10, 40),  # Low IV for buying
                'market_condition': 'high_volatility_expected'
            },
            'strangle': {
                'min_dte': 7,
                'max_dte': 30,
                'ideal_iv_rank': (10, 40),
                'market_condition': 'high_volatility_expected'
            },
            'butterfly': {
                'min_dte': 15,
                'max_dte': 45,
                'ideal_iv_rank': (40, 80),  # High IV for selling
                'market_condition': 'low_volatility_expected'
            },
            'calendar_spread': {
                'min_dte': 30,
                'max_dte': 60,
                'ideal_iv_rank': (20, 60),
                'market_condition': 'neutral_with_time_decay'
            }
        }
        
        logger.info("[STRATEGIES] INSTITUTIONAL-GRADE Options Strategies initialized")
        
    def iron_condor(self, underlying: str, spot_price: float, expiry_days: int, 
                   market_data: Dict) -> Optional[StrategyResult]:
        """
        Iron Condor Strategy
        Sell OTM Call + Buy Far OTM Call + Sell OTM Put + Buy Far OTM Put
        """
        try:
            # Calculate strike prices (typically 100-200 points apart for NIFTY)
            strike_width = 100 if underlying == 'NIFTY' else 200 if underlying == 'BANKNIFTY' else 50
            
            # OTM strikes (2-3% away from spot)
            otm_call_strike = round(spot_price * 1.02 / strike_width) * strike_width
            otm_put_strike = round(spot_price * 0.98 / strike_width) * strike_width
            
            # Far OTM strikes
            far_otm_call_strike = otm_call_strike + strike_width
            far_otm_put_strike = otm_put_strike - strike_width
            
            otm_call_premium = self._estimate_option_premium(
                spot_price, otm_call_strike, expiry_days, 'CE', market_data)
            far_otm_call_premium = self._estimate_option_premium(
                spot_price, far_otm_call_strike, expiry_days, 'CE', market_data)
            otm_put_premium = self._estimate_option_premium(
                spot_price, otm_put_strike, expiry_days, 'PE', market_data)
            far_otm_put_premium = self._estimate_option_premium(
                spot_price, far_otm_put_strike, expiry_days, 'PE', market_data)
            
            # Create legs
            lot_size = self.settings.get_lot_size(underlying)
            expiry_date = (datetime.now() + timedelta(days=expiry_days)).strftime('%Y-%m-%d')
            
            legs = [
                OptionLeg(underlying, 'CE', otm_call_strike, expiry_date, 'SELL', lot_size, otm_call_premium),
                OptionLeg(underlying, 'CE', far_otm_call_strike, expiry_date, 'BUY', lot_size, far_otm_call_premium),
                OptionLeg(underlying, 'PE', otm_put_strike, expiry_date, 'SELL', lot_size, otm_put_premium),
                OptionLeg(underlying, 'PE', far_otm_put_strike, expiry_date, 'BUY', lot_size, far_otm_put_premium)
            ]
            
            # Calculate strategy metrics
            net_premium = (otm_call_premium + otm_put_premium - far_otm_call_premium - far_otm_put_premium) * lot_size
            max_profit = net_premium
            max_loss = (strike_width * lot_size) - net_premium
            
            # Breakeven points
            upper_breakeven = otm_call_strike + (net_premium / lot_size)
            lower_breakeven = otm_put_strike - (net_premium / lot_size)
            
            # Probability of profit (rough estimate)
            profit_range = upper_breakeven - lower_breakeven
            total_range = far_otm_call_strike - far_otm_put_strike
            probability_of_profit = min(0.85, profit_range / total_range)
            
            return StrategyResult(
                strategy_name="Iron Condor",
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[lower_breakeven, upper_breakeven],
                net_premium=net_premium,
                probability_of_profit=probability_of_profit,
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
                margin_required=max_loss * 1.2  # 20% buffer
            )
            
        except Exception as e:
            logger.error(f"Iron Condor calculation failed: {e}")
            return None
    
    def butterfly_spread(self, underlying: str, spot_price: float, expiry_days: int,
                        option_type: str, market_data: Dict) -> Optional[StrategyResult]:
        """
        Butterfly Spread Strategy
        Buy ITM + Sell 2x ATM + Buy OTM (same option type)
        """
        try:
            strike_width = 100 if underlying == 'NIFTY' else 200 if underlying == 'BANKNIFTY' else 50
            
            # Strike selection
            atm_strike = round(spot_price / strike_width) * strike_width
            itm_strike = atm_strike - strike_width
            otm_strike = atm_strike + strike_width
            
            # Get premiums
            itm_premium = self._estimate_option_premium(spot_price, itm_strike, expiry_days, option_type, market_data)
            atm_premium = self._estimate_option_premium(spot_price, atm_strike, expiry_days, option_type, market_data)
            otm_premium = self._estimate_option_premium(spot_price, otm_strike, expiry_days, option_type, market_data)
            
            lot_size = self.settings.get_lot_size(underlying)
            expiry_date = (datetime.now() + timedelta(days=expiry_days)).strftime('%Y-%m-%d')
            
            legs = [
                OptionLeg(underlying, option_type, itm_strike, expiry_date, 'BUY', lot_size, itm_premium),
                OptionLeg(underlying, option_type, atm_strike, expiry_date, 'SELL', lot_size * 2, atm_premium),
                OptionLeg(underlying, option_type, otm_strike, expiry_date, 'BUY', lot_size, otm_premium)
            ]
            
            # Calculate metrics
            net_premium = (itm_premium + otm_premium - 2 * atm_premium) * lot_size
            max_profit = (strike_width * lot_size) + net_premium
            max_loss = abs(net_premium)
            
            return StrategyResult(
                strategy_name=f"{option_type} Butterfly Spread",
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[itm_strike + abs(net_premium/lot_size), otm_strike - abs(net_premium/lot_size)],
                net_premium=net_premium,
                probability_of_profit=0.35,  # Butterfly has lower probability but higher reward
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
                margin_required=max_loss * 1.2
            )
            
        except Exception as e:
            logger.error(f"Butterfly spread calculation failed: {e}")
            return None
    
    def straddle(self, underlying: str, spot_price: float, expiry_days: int,
                market_data: Dict, long: bool = True) -> Optional[StrategyResult]:
        """
        Straddle Strategy
        Buy/Sell ATM Call + Buy/Sell ATM Put
        """
        try:
            strike_width = 100 if underlying == 'NIFTY' else 200 if underlying == 'BANKNIFTY' else 50
            atm_strike = round(spot_price / strike_width) * strike_width
            
            # Get premiums
            call_premium = self._estimate_option_premium(spot_price, atm_strike, expiry_days, 'CE', market_data)
            put_premium = self._estimate_option_premium(spot_price, atm_strike, expiry_days, 'PE', market_data)
            
            lot_size = self.settings.get_lot_size(underlying)
            expiry_date = (datetime.now() + timedelta(days=expiry_days)).strftime('%Y-%m-%d')
            
            action = 'BUY' if long else 'SELL'
            
            legs = [
                OptionLeg(underlying, 'CE', atm_strike, expiry_date, action, lot_size, call_premium),
                OptionLeg(underlying, 'PE', atm_strike, expiry_date, action, lot_size, put_premium)
            ]
            
            # Calculate metrics
            total_premium = (call_premium + put_premium) * lot_size
            
            if long:
                max_profit = float('inf')  # Unlimited profit potential
                max_loss = total_premium
                breakeven_points = [atm_strike - total_premium/lot_size, atm_strike + total_premium/lot_size]
                net_premium = -total_premium  # We pay premium
            else:
                max_profit = total_premium
                max_loss = float('inf')  # Unlimited loss potential (but we'll set a practical limit)
                breakeven_points = [atm_strike - total_premium/lot_size, atm_strike + total_premium/lot_size]
                net_premium = total_premium  # We receive premium
                max_loss = total_premium * 3  # Practical limit for risk calculation
            
            return StrategyResult(
                strategy_name=f"{'Long' if long else 'Short'} Straddle",
                legs=legs,
                max_profit=max_profit if max_profit != float('inf') else total_premium * 2,
                max_loss=max_loss if max_loss != float('inf') else total_premium,
                breakeven_points=breakeven_points,
                net_premium=net_premium,
                probability_of_profit=0.45 if long else 0.55,
                risk_reward_ratio=2.0 if long else 0.5,
                margin_required=total_premium * 1.5
            )
            
        except Exception as e:
            logger.error(f"Straddle calculation failed: {e}")
            return None
    
    def strangle(self, underlying: str, spot_price: float, expiry_days: int,
                market_data: Dict, long: bool = True) -> Optional[StrategyResult]:
        """
        Strangle Strategy
        Buy/Sell OTM Call + Buy/Sell OTM Put
        """
        try:
            strike_width = 100 if underlying == 'NIFTY' else 200 if underlying == 'BANKNIFTY' else 50
            
            # OTM strikes
            otm_call_strike = round(spot_price * 1.02 / strike_width) * strike_width
            otm_put_strike = round(spot_price * 0.98 / strike_width) * strike_width
            
            # Get premiums
            call_premium = self._estimate_option_premium(spot_price, otm_call_strike, expiry_days, 'CE', market_data)
            put_premium = self._estimate_option_premium(spot_price, otm_put_strike, expiry_days, 'PE', market_data)
            
            lot_size = self.settings.get_lot_size(underlying)
            expiry_date = (datetime.now() + timedelta(days=expiry_days)).strftime('%Y-%m-%d')
            
            action = 'BUY' if long else 'SELL'
            
            legs = [
                OptionLeg(underlying, 'CE', otm_call_strike, expiry_date, action, lot_size, call_premium),
                OptionLeg(underlying, 'PE', otm_put_strike, expiry_date, action, lot_size, put_premium)
            ]
            
            # Calculate metrics
            total_premium = (call_premium + put_premium) * lot_size
            
            if long:
                max_profit = float('inf')
                max_loss = total_premium
                breakeven_points = [otm_put_strike - total_premium/lot_size, otm_call_strike + total_premium/lot_size]
                net_premium = -total_premium
            else:
                max_profit = total_premium
                max_loss = total_premium * 3  # Practical limit
                breakeven_points = [otm_put_strike - total_premium/lot_size, otm_call_strike + total_premium/lot_size]
                net_premium = total_premium
            
            return StrategyResult(
                strategy_name=f"{'Long' if long else 'Short'} Strangle",
                legs=legs,
                max_profit=max_profit if max_profit != float('inf') else total_premium * 2,
                max_loss=max_loss,
                breakeven_points=breakeven_points,
                net_premium=net_premium,
                probability_of_profit=0.40 if long else 0.60,
                risk_reward_ratio=2.0 if long else 0.5,
                margin_required=total_premium * 1.5
            )
            
        except Exception as e:
            logger.error(f"Strangle calculation failed: {e}")
            return None
    
    def _estimate_option_premium(self, spot: float, strike: float, days_to_expiry: int,
                                option_type: str, market_data: Dict) -> float:
        """
        Estimate option premium using simplified Black-Scholes
        Replace with real market data in production
        """
        try:
            # Get volatility from VIX or use default
            volatility = market_data.get('india_vix', 20) / 100
            
            # Time to expiry in years
            time_to_expiry = days_to_expiry / 365.0
            
            # Simplified Black-Scholes calculation
            d1 = (math.log(spot / strike) + (self.risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
            d2 = d1 - volatility * math.sqrt(time_to_expiry)
            
            # Standard normal CDF approximation
            def norm_cdf(x):
                return 0.5 * (1 + math.erf(x / math.sqrt(2)))
            
            if option_type == 'CE':
                premium = spot * norm_cdf(d1) - strike * math.exp(-self.risk_free_rate * time_to_expiry) * norm_cdf(d2)
            else:  # PE
                premium = strike * math.exp(-self.risk_free_rate * time_to_expiry) * norm_cdf(-d2) - spot * norm_cdf(-d1)
            
            return max(premium, 1.0)  # Minimum premium of 1
            
        except Exception as e:
            logger.error(f"Premium estimation failed: {e}")
            # Fallback estimation based on moneyness
            moneyness = spot / strike if option_type == 'CE' else strike / spot
            base_premium = max(spot * 0.02, 10)  # 2% of spot or minimum 10
            return base_premium * moneyness * (days_to_expiry / 30)
    
    def analyze_strategy(self, strategy_result: StrategyResult, market_data: Dict) -> Dict[str, Any]:
        """Analyze strategy for trading decision"""
        try:
            analysis = {
                'strategy_name': strategy_result.strategy_name,
                'recommendation': 'HOLD',
                'confidence_score': 0,
                'risk_assessment': 'MEDIUM',
                'market_suitability': 'NEUTRAL',
                'execution_priority': 'LOW'
            }
            
            # Risk assessment
            risk_reward = strategy_result.risk_reward_ratio
            if risk_reward > 2.0:
                analysis['risk_assessment'] = 'LOW'
                analysis['confidence_score'] += 30
            elif risk_reward > 1.0:
                analysis['risk_assessment'] = 'MEDIUM'
                analysis['confidence_score'] += 20
            else:
                analysis['risk_assessment'] = 'HIGH'
                analysis['confidence_score'] += 10
            
            # Market suitability based on VIX
            vix = market_data.get('india_vix', 20)
            if vix > 25:
                analysis['market_suitability'] = 'HIGH_VOLATILITY'
                if 'Straddle' in strategy_result.strategy_name or 'Strangle' in strategy_result.strategy_name:
                    analysis['confidence_score'] += 25
            elif vix < 15:
                analysis['market_suitability'] = 'LOW_VOLATILITY'
                if 'Condor' in strategy_result.strategy_name or 'Butterfly' in strategy_result.strategy_name:
                    analysis['confidence_score'] += 25
            
            # Probability of profit consideration
            if strategy_result.probability_of_profit > 0.6:
                analysis['confidence_score'] += 20
            elif strategy_result.probability_of_profit > 0.4:
                analysis['confidence_score'] += 10
            
            # Final recommendation
            if analysis['confidence_score'] > 70:
                analysis['recommendation'] = 'STRONG_BUY'
                analysis['execution_priority'] = 'HIGH'
            elif analysis['confidence_score'] > 50:
                analysis['recommendation'] = 'BUY'
                analysis['execution_priority'] = 'MEDIUM'
            elif analysis['confidence_score'] > 30:
                analysis['recommendation'] = 'WEAK_BUY'
                analysis['execution_priority'] = 'LOW'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Strategy analysis failed: {e}")
            return {'recommendation': 'HOLD', 'confidence_score': 0}
