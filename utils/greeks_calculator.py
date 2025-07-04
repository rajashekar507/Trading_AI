"""
Real-time Options Greeks Calculator for VLR_AI Trading System
Implements Delta, Gamma, Theta, Vega calculations and hedging strategies
"""

import logging
import numpy as np
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import norm

logger = logging.getLogger('trading_system.greeks_calculator')

@dataclass
class GreeksData:
    """Options Greeks data structure"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    implied_volatility: float
    intrinsic_value: float
    time_value: float
    moneyness: float

@dataclass
class PortfolioGreeks:
    """Portfolio-level Greeks"""
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    delta_exposure: float
    gamma_risk: float
    theta_decay: float
    vega_risk: float

class GreeksCalculator:
    """Real-time Options Greeks Calculator"""
    
    def __init__(self, settings):
        self.settings = settings
        self.risk_free_rate = 0.065  # 6.5% risk-free rate
        
    def calculate_greeks(self, spot_price: float, strike_price: float, 
                        time_to_expiry: float, volatility: float, 
                        option_type: str, dividend_yield: float = 0.0) -> GreeksData:
        """
        Calculate all Greeks for an option using Black-Scholes model
        
        Args:
            spot_price: Current price of underlying
            strike_price: Strike price of option
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility (as decimal, e.g., 0.20 for 20%)
            option_type: 'CE' for Call, 'PE' for Put
            dividend_yield: Dividend yield (default 0 for index options)
        """
        try:
            # Ensure minimum time to expiry to avoid division by zero
            time_to_expiry = max(time_to_expiry, 1/365)  # Minimum 1 day
            
            # Calculate d1 and d2
            d1 = (math.log(spot_price / strike_price) + 
                  (self.risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (
                  volatility * math.sqrt(time_to_expiry))
            
            d2 = d1 - volatility * math.sqrt(time_to_expiry)
            
            # Standard normal PDF and CDF
            nd1 = norm.pdf(d1)
            nd2 = norm.pdf(d2)
            Nd1 = norm.cdf(d1)
            Nd2 = norm.cdf(d2)
            N_minus_d1 = norm.cdf(-d1)
            N_minus_d2 = norm.cdf(-d2)
            
            # Discount factors
            discount_factor = math.exp(-self.risk_free_rate * time_to_expiry)
            dividend_discount = math.exp(-dividend_yield * time_to_expiry)
            
            if option_type.upper() == 'CE':
                # Call option calculations
                option_price = (spot_price * dividend_discount * Nd1 - 
                               strike_price * discount_factor * Nd2)
                delta = dividend_discount * Nd1
                theta = ((-spot_price * dividend_discount * nd1 * volatility) / (2 * math.sqrt(time_to_expiry)) -
                        self.risk_free_rate * strike_price * discount_factor * Nd2 +
                        dividend_yield * spot_price * dividend_discount * Nd1) / 365
                
            else:  # Put option
                option_price = (strike_price * discount_factor * N_minus_d2 - 
                               spot_price * dividend_discount * N_minus_d1)
                delta = -dividend_discount * N_minus_d1
                theta = ((-spot_price * dividend_discount * nd1 * volatility) / (2 * math.sqrt(time_to_expiry)) +
                        self.risk_free_rate * strike_price * discount_factor * N_minus_d2 -
                        dividend_yield * spot_price * dividend_discount * N_minus_d1) / 365
            
            # Greeks that are same for calls and puts
            gamma = (dividend_discount * nd1) / (spot_price * volatility * math.sqrt(time_to_expiry))
            vega = (spot_price * dividend_discount * nd1 * math.sqrt(time_to_expiry)) / 100
            
            # Rho (different for calls and puts)
            if option_type.upper() == 'CE':
                rho = (strike_price * time_to_expiry * discount_factor * Nd2) / 100
            else:
                rho = (-strike_price * time_to_expiry * discount_factor * N_minus_d2) / 100
            
            # Additional metrics
            intrinsic_value = max(0, spot_price - strike_price) if option_type.upper() == 'CE' else max(0, strike_price - spot_price)
            time_value = option_price - intrinsic_value
            moneyness = spot_price / strike_price
            
            return GreeksData(
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho,
                implied_volatility=volatility,
                intrinsic_value=intrinsic_value,
                time_value=time_value,
                moneyness=moneyness
            )
            
        except Exception as e:
            logger.error(f"Greeks calculation failed: {e}")
            return GreeksData(0, 0, 0, 0, 0, volatility, 0, 0, 1)
    
    def calculate_portfolio_greeks(self, positions: List[Dict[str, Any]], 
                                  market_data: Dict[str, Any]) -> PortfolioGreeks:
        """Calculate portfolio-level Greeks"""
        try:
            total_delta = 0
            total_gamma = 0
            total_theta = 0
            total_vega = 0
            
            for position in positions:
                # Get position details
                instrument = position.get('instrument', 'NIFTY')
                strike_price = position.get('strike_price', 0)
                option_type = position.get('option_type', 'CE')
                quantity = position.get('quantity', 0)
                expiry_date = position.get('expiry_date', '')
                
                # Get current spot price
                spot_price = market_data.get(f'{instrument.lower()}_spot', 25000)
                
                # Calculate time to expiry
                if expiry_date:
                    try:
                        expiry = datetime.strptime(expiry_date, '%Y-%m-%d')
                        time_to_expiry = (expiry - datetime.now()).days / 365.0
                    except:
                        time_to_expiry = 7 / 365.0  # Default 1 week
                else:
                    time_to_expiry = 7 / 365.0
                
                # Get volatility
                volatility = market_data.get('india_vix', 20) / 100
                
                # Calculate Greeks for this position
                greeks = self.calculate_greeks(
                    spot_price, strike_price, time_to_expiry, volatility, option_type
                )
                
                # Add to portfolio totals (considering quantity and sign)
                position_multiplier = quantity
                total_delta += greeks.delta * position_multiplier
                total_gamma += greeks.gamma * position_multiplier
                total_theta += greeks.theta * position_multiplier
                total_vega += greeks.vega * position_multiplier
            
            # Calculate risk metrics
            delta_exposure = abs(total_delta) * market_data.get('nifty_spot', 25000)
            gamma_risk = abs(total_gamma) * (market_data.get('nifty_spot', 25000) ** 2) * 0.01  # 1% move
            theta_decay = abs(total_theta)  # Daily decay
            vega_risk = abs(total_vega) * 0.01  # 1% volatility change
            
            return PortfolioGreeks(
                total_delta=total_delta,
                total_gamma=total_gamma,
                total_theta=total_theta,
                total_vega=total_vega,
                delta_exposure=delta_exposure,
                gamma_risk=gamma_risk,
                theta_decay=theta_decay,
                vega_risk=vega_risk
            )
            
        except Exception as e:
            logger.error(f"Portfolio Greeks calculation failed: {e}")
            return PortfolioGreeks(0, 0, 0, 0, 0, 0, 0, 0)
    
    def suggest_delta_hedge(self, portfolio_greeks: PortfolioGreeks, 
                           market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest delta hedging strategy"""
        try:
            hedge_suggestion = {
                'hedge_required': False,
                'hedge_type': 'NONE',
                'hedge_quantity': 0,
                'hedge_instrument': 'NIFTY_FUT',
                'current_delta': portfolio_greeks.total_delta,
                'target_delta': 0,
                'hedge_cost_estimate': 0
            }
            
            # Check if hedging is needed (delta > 50 or < -50)
            if abs(portfolio_greeks.total_delta) > 50:
                hedge_suggestion['hedge_required'] = True
                
                # Determine hedge type
                if portfolio_greeks.total_delta > 0:
                    hedge_suggestion['hedge_type'] = 'SHORT_FUTURES'
                    hedge_suggestion['hedge_quantity'] = -int(portfolio_greeks.total_delta)
                else:
                    hedge_suggestion['hedge_type'] = 'LONG_FUTURES'
                    hedge_suggestion['hedge_quantity'] = -int(portfolio_greeks.total_delta)
                
                # Estimate hedge cost (futures margin requirement)
                nifty_price = market_data.get('nifty_spot', 25000)
                lot_size = self.settings.get_lot_size('NIFTY')
                hedge_suggestion['hedge_cost_estimate'] = abs(hedge_suggestion['hedge_quantity']) * nifty_price * 0.1  # 10% margin
            
            return hedge_suggestion
            
        except Exception as e:
            logger.error(f"Delta hedge suggestion failed: {e}")
            return {'hedge_required': False, 'error': str(e)}
    
    def gamma_scalping_opportunity(self, portfolio_greeks: PortfolioGreeks,
                                  market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify gamma scalping opportunities"""
        try:
            scalping_analysis = {
                'opportunity_exists': False,
                'gamma_level': portfolio_greeks.total_gamma,
                'scalping_potential': 'LOW',
                'recommended_action': 'HOLD',
                'profit_potential': 0
            }
            
            # High gamma positions can benefit from scalping
            if abs(portfolio_greeks.total_gamma) > 0.1:
                scalping_analysis['opportunity_exists'] = True
                
                # Determine scalping potential based on gamma level
                gamma_level = abs(portfolio_greeks.total_gamma)
                if gamma_level > 0.5:
                    scalping_analysis['scalping_potential'] = 'HIGH'
                    scalping_analysis['recommended_action'] = 'ACTIVE_SCALP'
                elif gamma_level > 0.2:
                    scalping_analysis['scalping_potential'] = 'MEDIUM'
                    scalping_analysis['recommended_action'] = 'MODERATE_SCALP'
                else:
                    scalping_analysis['scalping_potential'] = 'LOW'
                    scalping_analysis['recommended_action'] = 'PASSIVE_SCALP'
                
                # Estimate profit potential (simplified)
                nifty_price = market_data.get('nifty_spot', 25000)
                expected_move = nifty_price * 0.005  # 0.5% move
                scalping_analysis['profit_potential'] = gamma_level * (expected_move ** 2) * 0.5
            
            return scalping_analysis
            
        except Exception as e:
            logger.error(f"Gamma scalping analysis failed: {e}")
            return {'opportunity_exists': False, 'error': str(e)}
    
    def theta_decay_analysis(self, portfolio_greeks: PortfolioGreeks) -> Dict[str, Any]:
        """Analyze theta decay impact"""
        try:
            decay_analysis = {
                'daily_decay': portfolio_greeks.total_theta,
                'weekly_decay': portfolio_greeks.total_theta * 7,
                'decay_impact': 'NEUTRAL',
                'recommendation': 'MONITOR'
            }
            
            daily_decay = abs(portfolio_greeks.total_theta)
            
            if daily_decay > 1000:
                decay_analysis['decay_impact'] = 'HIGH'
                if portfolio_greeks.total_theta < 0:  # Losing money to decay
                    decay_analysis['recommendation'] = 'CLOSE_POSITIONS'
                else:  # Benefiting from decay
                    decay_analysis['recommendation'] = 'HOLD_FOR_DECAY'
            elif daily_decay > 500:
                decay_analysis['decay_impact'] = 'MEDIUM'
                decay_analysis['recommendation'] = 'MONITOR_CLOSELY'
            else:
                decay_analysis['decay_impact'] = 'LOW'
                decay_analysis['recommendation'] = 'NORMAL_MONITORING'
            
            return decay_analysis
            
        except Exception as e:
            logger.error(f"Theta decay analysis failed: {e}")
            return {'daily_decay': 0, 'error': str(e)}
    
    def vega_risk_assessment(self, portfolio_greeks: PortfolioGreeks,
                           market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess vega risk from volatility changes"""
        try:
            vega_assessment = {
                'total_vega': portfolio_greeks.total_vega,
                'vega_risk_level': 'LOW',
                'volatility_sensitivity': 0,
                'hedge_recommendation': 'NONE'
            }
            
            total_vega = abs(portfolio_greeks.total_vega)
            current_vix = market_data.get('india_vix', 20)
            
            # Calculate sensitivity to 1% volatility change
            vega_assessment['volatility_sensitivity'] = total_vega * 0.01
            
            # Assess risk level
            if total_vega > 5000:
                vega_assessment['vega_risk_level'] = 'HIGH'
                vega_assessment['hedge_recommendation'] = 'HEDGE_VEGA'
            elif total_vega > 2000:
                vega_assessment['vega_risk_level'] = 'MEDIUM'
                vega_assessment['hedge_recommendation'] = 'MONITOR_VOLATILITY'
            else:
                vega_assessment['vega_risk_level'] = 'LOW'
                vega_assessment['hedge_recommendation'] = 'NO_ACTION_NEEDED'
            
            # Special consideration for high VIX environments
            if current_vix > 30 and portfolio_greeks.total_vega > 0:
                vega_assessment['hedge_recommendation'] = 'CONSIDER_VEGA_HEDGE'
            
            return vega_assessment
            
        except Exception as e:
            logger.error(f"Vega risk assessment failed: {e}")
            return {'total_vega': 0, 'error': str(e)}
    
    def comprehensive_greeks_report(self, positions: List[Dict[str, Any]],
                                  market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive Greeks analysis report"""
        try:
            # Calculate portfolio Greeks
            portfolio_greeks = self.calculate_portfolio_greeks(positions, market_data)
            
            # Get all analyses
            delta_hedge = self.suggest_delta_hedge(portfolio_greeks, market_data)
            gamma_scalp = self.gamma_scalping_opportunity(portfolio_greeks, market_data)
            theta_analysis = self.theta_decay_analysis(portfolio_greeks)
            vega_assessment = self.vega_risk_assessment(portfolio_greeks, market_data)
            
            # Compile comprehensive report
            report = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_greeks': {
                    'delta': portfolio_greeks.total_delta,
                    'gamma': portfolio_greeks.total_gamma,
                    'theta': portfolio_greeks.total_theta,
                    'vega': portfolio_greeks.total_vega,
                    'delta_exposure': portfolio_greeks.delta_exposure,
                    'gamma_risk': portfolio_greeks.gamma_risk,
                    'theta_decay': portfolio_greeks.theta_decay,
                    'vega_risk': portfolio_greeks.vega_risk
                },
                'risk_analysis': {
                    'overall_risk_level': 'MEDIUM',
                    'primary_risks': [],
                    'recommendations': []
                },
                'hedging_suggestions': delta_hedge,
                'gamma_scalping': gamma_scalp,
                'theta_analysis': theta_analysis,
                'vega_assessment': vega_assessment
            }
            
            # Determine overall risk level and recommendations
            risk_factors = []
            recommendations = []
            
            if abs(portfolio_greeks.total_delta) > 100:
                risk_factors.append('HIGH_DELTA_EXPOSURE')
                recommendations.append('Consider delta hedging')
            
            if abs(portfolio_greeks.total_gamma) > 0.5:
                risk_factors.append('HIGH_GAMMA_RISK')
                recommendations.append('Monitor for gamma scalping opportunities')
            
            if abs(portfolio_greeks.total_theta) > 1000:
                risk_factors.append('HIGH_THETA_DECAY')
                recommendations.append('Monitor time decay impact')
            
            if abs(portfolio_greeks.total_vega) > 5000:
                risk_factors.append('HIGH_VEGA_RISK')
                recommendations.append('Consider volatility hedging')
            
            # Set overall risk level
            if len(risk_factors) >= 3:
                report['risk_analysis']['overall_risk_level'] = 'HIGH'
            elif len(risk_factors) >= 1:
                report['risk_analysis']['overall_risk_level'] = 'MEDIUM'
            else:
                report['risk_analysis']['overall_risk_level'] = 'LOW'
            
            report['risk_analysis']['primary_risks'] = risk_factors
            report['risk_analysis']['recommendations'] = recommendations
            
            return report
            
        except Exception as e:
            logger.error(f"Comprehensive Greeks report failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
