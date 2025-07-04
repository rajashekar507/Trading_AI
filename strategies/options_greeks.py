"""
Options Greeks-Based Trading Strategies
Implements Delta-neutral, Gamma scalping, and Volatility trading
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm
import math

logger = logging.getLogger('trading_system.options_greeks')

class OptionsGreeksCalculator:
    """Advanced Options Greeks Calculator"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.risk_free_rate = 0.06  # 6% risk-free rate
        self.dividend_yield = 0.0   # No dividend for indices
        
        logger.info("[GREEKS] Options Greeks Calculator initialized")
    
    def calculate_all_greeks(self, spot_price: float, strike_price: float, 
                           time_to_expiry: float, volatility: float, 
                           option_type: str = 'CE') -> Dict:
        """Calculate all Greeks for an option"""
        try:
            # Convert time to expiry to years
            T = time_to_expiry / 365.0
            
            if T <= 0:
                return self._zero_greeks()
            
            # Black-Scholes parameters
            d1 = self._calculate_d1(spot_price, strike_price, T, volatility)
            d2 = d1 - volatility * math.sqrt(T)
            
            # Calculate Greeks
            greeks = {
                'delta': self._calculate_delta(d1, option_type),
                'gamma': self._calculate_gamma(spot_price, d1, T, volatility),
                'theta': self._calculate_theta(spot_price, strike_price, d1, d2, T, volatility, option_type),
                'vega': self._calculate_vega(spot_price, d1, T),
                'rho': self._calculate_rho(strike_price, d2, T, option_type),
                'option_price': self._calculate_option_price(spot_price, strike_price, d1, d2, T, option_type),
                'implied_volatility': volatility,
                'time_to_expiry': T,
                'moneyness': spot_price / strike_price
            }
            
            # Add derived metrics
            greeks['delta_dollars'] = greeks['delta'] * spot_price
            greeks['gamma_dollars'] = greeks['gamma'] * spot_price * spot_price / 100
            greeks['theta_dollars'] = greeks['theta']
            greeks['vega_dollars'] = greeks['vega'] / 100
            
            return greeks
            
        except Exception as e:
            logger.error(f"[GREEKS] Calculation failed: {e}")
            return self._zero_greeks()
    
    def _calculate_d1(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate d1 parameter for Black-Scholes"""
        return (math.log(S/K) + (self.risk_free_rate + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    
    def _calculate_delta(self, d1: float, option_type: str) -> float:
        """Calculate Delta"""
        if option_type == 'CE':
            return norm.cdf(d1)
        else:  # PE
            return norm.cdf(d1) - 1
    
    def _calculate_gamma(self, S: float, d1: float, T: float, sigma: float) -> float:
        """Calculate Gamma"""
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))
    
    def _calculate_theta(self, S: float, K: float, d1: float, d2: float, T: float, sigma: float, option_type: str) -> float:
        """Calculate Theta (per day)"""
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
        term2 = self.risk_free_rate * K * math.exp(-self.risk_free_rate * T)
        
        if option_type == 'CE':
            theta = term1 - term2 * norm.cdf(d2)
        else:  # PE
            theta = term1 + term2 * norm.cdf(-d2)
        
        return theta / 365  # Convert to per day
    
    def _calculate_vega(self, S: float, d1: float, T: float) -> float:
        """Calculate Vega"""
        return S * norm.pdf(d1) * math.sqrt(T)
    
    def _calculate_rho(self, K: float, d2: float, T: float, option_type: str) -> float:
        """Calculate Rho"""
        if option_type == 'CE':
            return K * T * math.exp(-self.risk_free_rate * T) * norm.cdf(d2)
        else:  # PE
            return -K * T * math.exp(-self.risk_free_rate * T) * norm.cdf(-d2)
    
    def _calculate_option_price(self, S: float, K: float, d1: float, d2: float, T: float, option_type: str) -> float:
        """Calculate theoretical option price using Black-Scholes"""
        if option_type == 'CE':
            return S * norm.cdf(d1) - K * math.exp(-self.risk_free_rate * T) * norm.cdf(d2)
        else:  # PE
            return K * math.exp(-self.risk_free_rate * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    def _zero_greeks(self) -> Dict:
        """Return zero Greeks for expired options"""
        return {
            'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0,
            'option_price': 0.0, 'implied_volatility': 0.0, 'time_to_expiry': 0.0,
            'moneyness': 1.0, 'delta_dollars': 0.0, 'gamma_dollars': 0.0,
            'theta_dollars': 0.0, 'vega_dollars': 0.0
        }

class DeltaNeutralStrategy:
    """Delta-Neutral Trading Strategy"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.greeks_calculator = OptionsGreeksCalculator(settings)
        self.delta_threshold = 0.05  # Rebalance when delta exceeds ±5%
        self.positions = {}
        
        logger.info("[DELTA_NEUTRAL] Strategy initialized")
    
    def create_delta_neutral_position(self, options_data: Dict, spot_price: float) -> Dict:
        """Create a delta-neutral position using options"""
        try:
            recommendations = []
            
            # Find ATM options for both CE and PE
            atm_strike = self._find_atm_strike(options_data, spot_price)
            
            if not atm_strike:
                return {'status': 'failed', 'message': 'No ATM options found'}
            
            # Get option data for ATM strike
            ce_data = self._get_option_data(options_data, atm_strike, 'CE')
            pe_data = self._get_option_data(options_data, atm_strike, 'PE')
            
            if not ce_data or not pe_data:
                return {'status': 'failed', 'message': 'ATM option data not available'}
            
            # Calculate Greeks for both options
            ce_greeks = self._calculate_option_greeks(ce_data, spot_price, 'CE')
            pe_greeks = self._calculate_option_greeks(pe_data, spot_price, 'PE')
            
            # Strategy 1: Long Straddle (if expecting high volatility)
            straddle_delta = ce_greeks['delta'] + pe_greeks['delta']
            straddle_gamma = ce_greeks['gamma'] + pe_greeks['gamma']
            straddle_theta = ce_greeks['theta'] + pe_greeks['theta']
            straddle_vega = ce_greeks['vega'] + pe_greeks['vega']
            
            recommendations.append({
                'strategy': 'LONG_STRADDLE',
                'positions': [
                    {'type': 'BUY', 'option_type': 'CE', 'strike': atm_strike, 'quantity': 1},
                    {'type': 'BUY', 'option_type': 'PE', 'strike': atm_strike, 'quantity': 1}
                ],
                'net_delta': straddle_delta,
                'net_gamma': straddle_gamma,
                'net_theta': straddle_theta,
                'net_vega': straddle_vega,
                'max_profit': 'Unlimited',
                'max_loss': ce_data['ltp'] + pe_data['ltp'],
                'breakeven_upper': atm_strike + ce_data['ltp'] + pe_data['ltp'],
                'breakeven_lower': atm_strike - (ce_data['ltp'] + pe_data['ltp']),
                'confidence': self._calculate_strategy_confidence(ce_greeks, pe_greeks, 'straddle')
            })
            
            # Strategy 2: Short Strangle (if expecting low volatility)
            otm_ce_strike = self._find_otm_strike(options_data, spot_price, 'CE', 0.02)  # 2% OTM
            otm_pe_strike = self._find_otm_strike(options_data, spot_price, 'PE', 0.02)  # 2% OTM
            
            if otm_ce_strike and otm_pe_strike:
                otm_ce_data = self._get_option_data(options_data, otm_ce_strike, 'CE')
                otm_pe_data = self._get_option_data(options_data, otm_pe_strike, 'PE')
                
                if otm_ce_data and otm_pe_data:
                    otm_ce_greeks = self._calculate_option_greeks(otm_ce_data, spot_price, 'CE')
                    otm_pe_greeks = self._calculate_option_greeks(otm_pe_data, spot_price, 'PE')
                    
                    strangle_delta = -otm_ce_greeks['delta'] - otm_pe_greeks['delta']
                    strangle_gamma = -otm_ce_greeks['gamma'] - otm_pe_greeks['gamma']
                    strangle_theta = -otm_ce_greeks['theta'] - otm_pe_greeks['theta']
                    strangle_vega = -otm_ce_greeks['vega'] - otm_pe_greeks['vega']
                    
                    recommendations.append({
                        'strategy': 'SHORT_STRANGLE',
                        'positions': [
                            {'type': 'SELL', 'option_type': 'CE', 'strike': otm_ce_strike, 'quantity': 1},
                            {'type': 'SELL', 'option_type': 'PE', 'strike': otm_pe_strike, 'quantity': 1}
                        ],
                        'net_delta': strangle_delta,
                        'net_gamma': strangle_gamma,
                        'net_theta': strangle_theta,
                        'net_vega': strangle_vega,
                        'max_profit': otm_ce_data['ltp'] + otm_pe_data['ltp'],
                        'max_loss': 'Unlimited',
                        'breakeven_upper': otm_ce_strike + otm_ce_data['ltp'] + otm_pe_data['ltp'],
                        'breakeven_lower': otm_pe_strike - (otm_ce_data['ltp'] + otm_pe_data['ltp']),
                        'confidence': self._calculate_strategy_confidence(otm_ce_greeks, otm_pe_greeks, 'strangle')
                    })
            
            # Select best strategy based on market conditions
            best_strategy = self._select_best_strategy(recommendations, spot_price)
            
            return {
                'status': 'success',
                'recommended_strategy': best_strategy,
                'all_strategies': recommendations,
                'market_analysis': self._analyze_market_conditions(ce_greeks, pe_greeks),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"[DELTA_NEUTRAL] Strategy creation failed: {e}")
            return {'status': 'failed', 'message': str(e)}
    
    def _find_atm_strike(self, options_data: Dict, spot_price: float) -> Optional[float]:
        """Find At-The-Money strike price"""
        try:
            strikes = []
            for instrument, data in options_data.items():
                if 'strike' in data:
                    strikes.append(data['strike'])
            
            if not strikes:
                return None
            
            # Find closest strike to spot price
            strikes = sorted(set(strikes))
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            
            return atm_strike
            
        except Exception as e:
            logger.error(f"[ATM_STRIKE] Calculation failed: {e}")
            return None
    
    def _find_otm_strike(self, options_data: Dict, spot_price: float, option_type: str, otm_pct: float) -> Optional[float]:
        """Find Out-of-The-Money strike price"""
        try:
            strikes = []
            for instrument, data in options_data.items():
                if 'strike' in data and instrument.endswith(option_type):
                    strikes.append(data['strike'])
            
            if not strikes:
                return None
            
            strikes = sorted(set(strikes))
            
            if option_type == 'CE':
                # For calls, OTM is above spot price
                target_strike = spot_price * (1 + otm_pct)
                otm_strikes = [s for s in strikes if s > spot_price]
            else:  # PE
                # For puts, OTM is below spot price
                target_strike = spot_price * (1 - otm_pct)
                otm_strikes = [s for s in strikes if s < spot_price]
            
            if not otm_strikes:
                return None
            
            # Find closest OTM strike to target
            otm_strike = min(otm_strikes, key=lambda x: abs(x - target_strike))
            
            return otm_strike
            
        except Exception as e:
            logger.error(f"[OTM_STRIKE] Calculation failed: {e}")
            return None
    
    def _get_option_data(self, options_data: Dict, strike: float, option_type: str) -> Optional[Dict]:
        """Get option data for specific strike and type"""
        try:
            for instrument, data in options_data.items():
                if (data.get('strike') == strike and 
                    instrument.endswith(option_type)):
                    return data
            return None
        except Exception:
            return None
    
    def _calculate_option_greeks(self, option_data: Dict, spot_price: float, option_type: str) -> Dict:
        """Calculate Greeks for an option"""
        try:
            strike = option_data['strike']
            current_price = option_data['ltp']
            
            # Estimate time to expiry (simplified - should use actual expiry)
            time_to_expiry = 7  # Assume 7 days for weekly options
            
            # Estimate implied volatility (simplified)
            implied_vol = self._estimate_implied_volatility(current_price, spot_price, strike, time_to_expiry, option_type)
            
            return self.greeks_calculator.calculate_all_greeks(
                spot_price, strike, time_to_expiry, implied_vol, option_type
            )
            
        except Exception as e:
            logger.error(f"[OPTION_GREEKS] Calculation failed: {e}")
            return self.greeks_calculator._zero_greeks()
    
    def _estimate_implied_volatility(self, option_price: float, spot_price: float, 
                                   strike: float, time_to_expiry: float, option_type: str) -> float:
        """Estimate implied volatility (simplified method)"""
        try:
            # Simple approximation - in practice, use Newton-Raphson method
            T = time_to_expiry / 365.0
            
            if T <= 0:
                return 0.2  # Default 20% volatility
            
            # Rough approximation based on option price and moneyness
            moneyness = spot_price / strike
            
            if option_type == 'CE':
                if moneyness > 1.05:  # Deep ITM
                    base_vol = 0.15
                elif moneyness > 0.95:  # ATM
                    base_vol = 0.25
                else:  # OTM
                    base_vol = 0.35
            else:  # PE
                if moneyness < 0.95:  # Deep ITM
                    base_vol = 0.15
                elif moneyness < 1.05:  # ATM
                    base_vol = 0.25
                else:  # OTM
                    base_vol = 0.35
            
            # Adjust based on option price
            price_factor = option_price / (spot_price * 0.02)  # Normalize by 2% of spot
            adjusted_vol = base_vol * max(0.5, min(2.0, price_factor))
            
            return max(0.1, min(1.0, adjusted_vol))  # Cap between 10% and 100%
            
        except Exception:
            return 0.2  # Default 20% volatility

    def _calculate_strategy_confidence(self, ce_greeks: Dict, pe_greeks: Dict, strategy_type: str) -> float:
        """Calculate confidence score for strategy"""
        try:
            confidence = 50.0  # Base confidence
            
            # Adjust based on Greeks
            if strategy_type == 'straddle':
                # Higher confidence if high gamma and vega
                if ce_greeks['gamma'] > 0.01 and pe_greeks['gamma'] > 0.01:
                    confidence += 15
                if ce_greeks['vega'] > 10 and pe_greeks['vega'] > 10:
                    confidence += 10
                # Lower confidence if high theta decay
                if abs(ce_greeks['theta']) > 5 or abs(pe_greeks['theta']) > 5:
                    confidence -= 10
            
            elif strategy_type == 'strangle':
                # Higher confidence if positive theta
                if ce_greeks['theta'] < -2 and pe_greeks['theta'] < -2:
                    confidence += 15
                # Lower confidence if high vega (volatility risk)
                if ce_greeks['vega'] > 15 or pe_greeks['vega'] > 15:
                    confidence -= 10
            
            return max(10, min(90, confidence))
            
        except Exception:
            return 50.0
    
    def _select_best_strategy(self, strategies: List[Dict], spot_price: float) -> Dict:
        """Select best strategy based on market conditions"""
        try:
            if not strategies:
                return {}
            
            # Simple selection based on confidence
            best_strategy = max(strategies, key=lambda x: x.get('confidence', 0))
            
            logger.info(f"[STRATEGY_SELECTION] Selected {best_strategy['strategy']} with {best_strategy['confidence']:.1f}% confidence")
            
            return best_strategy
            
        except Exception as e:
            logger.error(f"[STRATEGY_SELECTION] Failed: {e}")
            return strategies[0] if strategies else {}
    
    def _analyze_market_conditions(self, ce_greeks: Dict, pe_greeks: Dict) -> Dict:
        """Analyze current market conditions"""
        try:
            analysis = {
                'volatility_regime': 'NORMAL',
                'trend_bias': 'NEUTRAL',
                'time_decay_impact': 'MODERATE',
                'recommended_approach': 'BALANCED'
            }
            
            # Volatility analysis
            avg_vega = (ce_greeks['vega'] + abs(pe_greeks['vega'])) / 2
            if avg_vega > 20:
                analysis['volatility_regime'] = 'HIGH'
                analysis['recommended_approach'] = 'VOLATILITY_SELLING'
            elif avg_vega < 10:
                analysis['volatility_regime'] = 'LOW'
                analysis['recommended_approach'] = 'VOLATILITY_BUYING'
            
            # Trend analysis based on delta
            net_delta = ce_greeks['delta'] + pe_greeks['delta']
            if net_delta > 0.1:
                analysis['trend_bias'] = 'BULLISH'
            elif net_delta < -0.1:
                analysis['trend_bias'] = 'BEARISH'
            
            # Time decay analysis
            avg_theta = abs(ce_greeks['theta']) + abs(pe_greeks['theta'])
            if avg_theta > 10:
                analysis['time_decay_impact'] = 'HIGH'
            elif avg_theta < 3:
                analysis['time_decay_impact'] = 'LOW'
            
            return analysis
            
        except Exception as e:
            logger.error(f"[MARKET_ANALYSIS] Failed: {e}")
            return {'error': str(e)}

class GammaScalpingStrategy:
    """Gamma Scalping Strategy Implementation"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.greeks_calculator = OptionsGreeksCalculator(settings)
        self.gamma_threshold = 0.01  # Minimum gamma for scalping
        self.delta_rebalance_threshold = 0.05  # Rebalance when delta exceeds ±5%
        
        logger.info("[GAMMA_SCALPING] Strategy initialized")
    
    def identify_gamma_scalping_opportunities(self, options_data: Dict, spot_price: float) -> Dict:
        """Identify options suitable for gamma scalping"""
        try:
            opportunities = []
            
            for instrument, data in options_data.items():
                if 'strike' in data and 'ltp' in data:
                    option_type = 'CE' if instrument.endswith('CE') else 'PE'
                    greeks = self._calculate_option_greeks(data, spot_price, option_type)
                    
                    # Check if suitable for gamma scalping
                    if self._is_gamma_scalping_candidate(greeks, data, spot_price):
                        opportunity = {
                            'instrument': instrument,
                            'strike': data['strike'],
                            'option_type': option_type,
                            'current_price': data['ltp'],
                            'greeks': greeks,
                            'scalping_score': self._calculate_scalping_score(greeks, data, spot_price),
                            'recommended_quantity': self._calculate_scalping_quantity(greeks, data),
                            'hedge_ratio': greeks['delta'],
                            'expected_gamma_pnl': self._estimate_gamma_pnl(greeks, spot_price)
                        }
                        opportunities.append(opportunity)
            
            # Sort by scalping score
            opportunities.sort(key=lambda x: x['scalping_score'], reverse=True)
            
            return {
                'status': 'success',
                'opportunities': opportunities[:5],  # Top 5 opportunities
                'market_gamma_profile': self._analyze_market_gamma_profile(options_data, spot_price),
                'scalping_conditions': self._assess_scalping_conditions(spot_price),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"[GAMMA_SCALPING] Opportunity identification failed: {e}")
            return {'status': 'failed', 'message': str(e)}
    
    def _is_gamma_scalping_candidate(self, greeks: Dict, option_data: Dict, spot_price: float) -> bool:
        """Check if option is suitable for gamma scalping"""
        try:
            # High gamma requirement
            if greeks['gamma'] < self.gamma_threshold:
                return False
            
            # Reasonable time to expiry (not too close to expiry)
            if greeks['time_to_expiry'] < 0.02:  # Less than 7 days
                return False
            
            # Not too far OTM
            moneyness = spot_price / option_data['strike']
            if option_data['strike'] > 0:
                if abs(1 - moneyness) > 0.1:  # More than 10% away from ATM
                    return False
            
            # Sufficient liquidity (simplified check)
            if option_data.get('volume', 0) < 1000:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_scalping_score(self, greeks: Dict, option_data: Dict, spot_price: float) -> float:
        """Calculate gamma scalping score"""
        try:
            score = 0.0
            
            # Gamma contribution (40% weight)
            gamma_score = min(100, greeks['gamma'] * 1000)  # Scale gamma
            score += gamma_score * 0.4
            
            # Vega contribution (20% weight) - lower is better for scalping
            vega_score = max(0, 100 - greeks['vega'])
            score += vega_score * 0.2
            
            # Theta contribution (20% weight) - moderate theta is good
            theta_score = max(0, 100 - abs(greeks['theta']) * 10)
            score += theta_score * 0.2
            
            # Moneyness contribution (20% weight) - ATM is best
            moneyness = spot_price / option_data['strike'] if option_data['strike'] > 0 else 1
            moneyness_score = max(0, 100 - abs(1 - moneyness) * 500)
            score += moneyness_score * 0.2
            
            return min(100, max(0, score))
            
        except Exception:
            return 0.0
    
    def _calculate_scalping_quantity(self, greeks: Dict, option_data: Dict) -> int:
        """Calculate recommended quantity for gamma scalping"""
        try:
            # Base quantity calculation
            base_qty = 100  # Base 100 options
            
            # Adjust based on gamma
            gamma_factor = min(2.0, max(0.5, greeks['gamma'] * 100))
            
            # Adjust based on option price
            price_factor = min(2.0, max(0.5, 50 / option_data['ltp']))
            
            recommended_qty = int(base_qty * gamma_factor * price_factor)
            
            # Round to nearest 25 (typical lot size)
            recommended_qty = round(recommended_qty / 25) * 25
            
            return max(25, recommended_qty)
            
        except Exception:
            return 100
    
    def _estimate_gamma_pnl(self, greeks: Dict, spot_price: float) -> Dict:
        """Estimate potential P&L from gamma scalping"""
        try:
            # Assume 1% move in underlying
            move_1pct = spot_price * 0.01
            
            # Gamma P&L = 0.5 * Gamma * (Move)^2 * Quantity
            gamma_pnl_1pct = 0.5 * greeks['gamma'] * (move_1pct ** 2) * 100  # For 100 options
            
            # Daily estimates
            daily_moves = [0.005, 0.01, 0.015, 0.02]  # 0.5%, 1%, 1.5%, 2%
            daily_estimates = {}
            
            for move_pct in daily_moves:
                move = spot_price * move_pct
                gamma_pnl = 0.5 * greeks['gamma'] * (move ** 2) * 100
                theta_cost = greeks['theta'] * 100  # Theta cost for 100 options
                net_pnl = gamma_pnl + theta_cost  # Theta is usually negative
                
                daily_estimates[f'{move_pct*100:.1f}%_move'] = {
                    'gamma_pnl': gamma_pnl,
                    'theta_cost': theta_cost,
                    'net_pnl': net_pnl
                }
            
            return {
                'daily_estimates': daily_estimates,
                'breakeven_move': self._calculate_breakeven_move(greeks, spot_price)
            }
            
        except Exception as e:
            logger.error(f"[GAMMA_PNL] Estimation failed: {e}")
            return {}
    
    def _calculate_breakeven_move(self, greeks: Dict, spot_price: float) -> float:
        """Calculate breakeven move for gamma scalping"""
        try:
            # Breakeven when Gamma P&L = Theta cost
            # 0.5 * Gamma * Move^2 = |Theta|
            # Move = sqrt(2 * |Theta| / Gamma)
            
            if greeks['gamma'] <= 0:
                return float('inf')
            
            breakeven_move = math.sqrt(2 * abs(greeks['theta']) / greeks['gamma'])
            breakeven_pct = (breakeven_move / spot_price) * 100
            
            return breakeven_pct
            
        except Exception:
            return float('inf')
    
    def _analyze_market_gamma_profile(self, options_data: Dict, spot_price: float) -> Dict:
        """Analyze overall market gamma profile"""
        try:
            total_gamma = 0
            gamma_by_strike = {}
            
            for instrument, data in options_data.items():
                if 'strike' in data and 'ltp' in data:
                    option_type = 'CE' if instrument.endswith('CE') else 'PE'
                    greeks = self._calculate_option_greeks(data, spot_price, option_type)
                    
                    strike = data['strike']
                    if strike not in gamma_by_strike:
                        gamma_by_strike[strike] = 0
                    
                    gamma_by_strike[strike] += greeks['gamma'] * data.get('oi', 1000)  # Weight by OI
                    total_gamma += greeks['gamma']
            
            # Find gamma concentration points
            max_gamma_strike = max(gamma_by_strike.items(), key=lambda x: x[1]) if gamma_by_strike else (spot_price, 0)
            
            return {
                'total_market_gamma': total_gamma,
                'max_gamma_strike': max_gamma_strike[0],
                'max_gamma_value': max_gamma_strike[1],
                'gamma_distribution': gamma_by_strike,
                'gamma_regime': 'HIGH' if total_gamma > 1.0 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"[GAMMA_PROFILE] Analysis failed: {e}")
            return {}
    
    def _assess_scalping_conditions(self, spot_price: float) -> Dict:
        """Assess current market conditions for gamma scalping"""
        try:
            conditions = {
                'overall_rating': 'MODERATE',
                'volatility_environment': 'NORMAL',
                'trend_strength': 'WEAK',
                'time_of_day': 'REGULAR',
                'recommended_approach': 'CONSERVATIVE'
            }
            
            # Time-based assessment
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 11 or 14 <= current_hour <= 15:
                conditions['time_of_day'] = 'HIGH_ACTIVITY'
                conditions['overall_rating'] = 'GOOD'
            elif 11 <= current_hour <= 14:
                conditions['time_of_day'] = 'LUNCH_LULL'
                conditions['overall_rating'] = 'POOR'
            
            return conditions
            
        except Exception as e:
            logger.error(f"[SCALPING_CONDITIONS] Assessment failed: {e}")
            return {'error': str(e)}
    
    def _calculate_option_greeks(self, option_data: Dict, spot_price: float, option_type: str) -> Dict:
        """Calculate Greeks for an option (reuse from DeltaNeutralStrategy)"""
        try:
            strike = option_data['strike']
            current_price = option_data['ltp']
            
            # Estimate time to expiry (simplified)
            time_to_expiry = 7  # Assume 7 days
            
            # Estimate implied volatility (simplified)
            implied_vol = self._estimate_implied_volatility(current_price, spot_price, strike, time_to_expiry, option_type)
            
            return self.greeks_calculator.calculate_all_greeks(
                spot_price, strike, time_to_expiry, implied_vol, option_type
            )
            
        except Exception as e:
            logger.error(f"[OPTION_GREEKS] Calculation failed: {e}")
            return self.greeks_calculator._zero_greeks()
    
    def _estimate_implied_volatility(self, option_price: float, spot_price: float, 
                                   strike: float, time_to_expiry: float, option_type: str) -> float:
        """Estimate implied volatility (simplified method)"""
        try:
            T = time_to_expiry / 365.0
            if T <= 0:
                return 0.2
            
            moneyness = spot_price / strike
            
            if option_type == 'CE':
                if moneyness > 1.05:
                    base_vol = 0.15
                elif moneyness > 0.95:
                    base_vol = 0.25
                else:
                    base_vol = 0.35
            else:
                if moneyness < 0.95:
                    base_vol = 0.15
                elif moneyness < 1.05:
                    base_vol = 0.25
                else:
                    base_vol = 0.35
            
            price_factor = option_price / (spot_price * 0.02)
            adjusted_vol = base_vol * max(0.5, min(2.0, price_factor))
            
            return max(0.1, min(1.0, adjusted_vol))
            
        except Exception:
            return 0.2

class VolatilityTradingStrategy:
    """Volatility-based trading using Greeks"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.greeks_calculator = OptionsGreeksCalculator(settings)
        
        logger.info("[VOLATILITY_TRADING] Strategy initialized")
    
    def analyze_volatility_opportunities(self, options_data: Dict, spot_price: float, current_vix: float = 20) -> Dict:
        """Analyze volatility trading opportunities"""
        try:
            opportunities = []
            
            # Calculate implied volatility for all options
            iv_data = []
            for instrument, data in options_data.items():
                if 'strike' in data and 'ltp' in data:
                    option_type = 'CE' if instrument.endswith('CE') else 'PE'
                    greeks = self._calculate_option_greeks(data, spot_price, option_type)
                    
                    iv_data.append({
                        'instrument': instrument,
                        'strike': data['strike'],
                        'option_type': option_type,
                        'implied_vol': greeks['implied_volatility'],
                        'vega': greeks['vega'],
                        'current_price': data['ltp'],
                        'greeks': greeks
                    })
            
            if not iv_data:
                return {'status': 'failed', 'message': 'No option data available'}
            
            # Calculate average implied volatility
            avg_iv = np.mean([d['implied_vol'] for d in iv_data])
            
            # Identify volatility opportunities
            for option in iv_data:
                iv_percentile = (option['implied_vol'] - avg_iv) / avg_iv * 100
                
                opportunity = {
                    'instrument': option['instrument'],
                    'strike': option['strike'],
                    'option_type': option['option_type'],
                    'current_price': option['current_price'],
                    'implied_vol': option['implied_vol'],
                    'iv_percentile': iv_percentile,
                    'vega': option['vega'],
                    'strategy_recommendation': self._recommend_volatility_strategy(option, iv_percentile, current_vix),
                    'confidence': self._calculate_volatility_confidence(option, iv_percentile, current_vix)
                }
                
                opportunities.append(opportunity)
            
            # Sort by confidence
            opportunities.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'status': 'success',
                'opportunities': opportunities[:10],  # Top 10
                'market_iv_analysis': {
                    'average_iv': avg_iv,
                    'iv_range': [min(d['implied_vol'] for d in iv_data), max(d['implied_vol'] for d in iv_data)],
                    'current_vix': current_vix,
                    'iv_regime': 'HIGH' if avg_iv > 0.3 else 'LOW' if avg_iv < 0.15 else 'NORMAL'
                },
                'recommended_approach': self._recommend_overall_approach(avg_iv, current_vix),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"[VOLATILITY_TRADING] Analysis failed: {e}")
            return {'status': 'failed', 'message': str(e)}
    
    def _recommend_volatility_strategy(self, option: Dict, iv_percentile: float, current_vix: float) -> str:
        """Recommend volatility strategy based on IV analysis"""
        try:
            if iv_percentile > 20:  # High IV
                if option['vega'] > 15:
                    return 'SELL_VOLATILITY'  # Sell high IV options
                else:
                    return 'NEUTRAL'
            elif iv_percentile < -20:  # Low IV
                if option['vega'] > 10:
                    return 'BUY_VOLATILITY'  # Buy low IV options
                else:
                    return 'NEUTRAL'
            else:
                return 'NEUTRAL'
                
        except Exception:
            return 'NEUTRAL'
    
    def _calculate_volatility_confidence(self, option: Dict, iv_percentile: float, current_vix: float) -> float:
        """Calculate confidence in volatility strategy"""
        try:
            confidence = 50.0
            
            # IV percentile contribution
            confidence += min(25, abs(iv_percentile) * 0.5)
            
            # Vega contribution
            if option['vega'] > 15:
                confidence += 15
            elif option['vega'] > 10:
                confidence += 10
            
            # VIX consideration
            if current_vix > 25 and iv_percentile > 0:
                confidence += 10  # High VIX + High IV = good selling opportunity
            elif current_vix < 15 and iv_percentile < 0:
                confidence += 10  # Low VIX + Low IV = good buying opportunity
            
            return max(10, min(90, confidence))
            
        except Exception:
            return 50.0
    
    def _recommend_overall_approach(self, avg_iv: float, current_vix: float) -> str:
        """Recommend overall volatility trading approach"""
        try:
            if avg_iv > 0.3 or current_vix > 25:
                return 'VOLATILITY_SELLING'
            elif avg_iv < 0.15 or current_vix < 15:
                return 'VOLATILITY_BUYING'
            else:
                return 'NEUTRAL_STRATEGIES'
                
        except Exception:
            return 'NEUTRAL_STRATEGIES'
    
    def _calculate_option_greeks(self, option_data: Dict, spot_price: float, option_type: str) -> Dict:
        """Calculate Greeks for an option"""
        try:
            strike = option_data['strike']
            current_price = option_data['ltp']
            time_to_expiry = 7
            implied_vol = self._estimate_implied_volatility(current_price, spot_price, strike, time_to_expiry, option_type)
            
            return self.greeks_calculator.calculate_all_greeks(
                spot_price, strike, time_to_expiry, implied_vol, option_type
            )
            
        except Exception as e:
            logger.error(f"[OPTION_GREEKS] Calculation failed: {e}")
            return self.greeks_calculator._zero_greeks()
    
    def _estimate_implied_volatility(self, option_price: float, spot_price: float, 
                                   strike: float, time_to_expiry: float, option_type: str) -> float:
        """Estimate implied volatility"""
        try:
            T = time_to_expiry / 365.0
            if T <= 0:
                return 0.2
            
            moneyness = spot_price / strike
            
            if option_type == 'CE':
                if moneyness > 1.05:
                    base_vol = 0.15
                elif moneyness > 0.95:
                    base_vol = 0.25
                else:
                    base_vol = 0.35
            else:
                if moneyness < 0.95:
                    base_vol = 0.15
                elif moneyness < 1.05:
                    base_vol = 0.25
                else:
                    base_vol = 0.35
            
            price_factor = option_price / (spot_price * 0.02)
            adjusted_vol = base_vol * max(0.5, min(2.0, price_factor))
            
            return max(0.1, min(1.0, adjusted_vol))
            
        except Exception:
            return 0.2