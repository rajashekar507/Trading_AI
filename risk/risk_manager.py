"""
Advanced risk management system for institutional trading
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

logger = logging.getLogger('trading_system.risk_manager')

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, settings=None):
        self.settings = settings
        
        # Use configurable values from settings or defaults
        self.risk_limits = {
            'max_daily_loss': getattr(settings, 'MAX_DAILY_LOSS', -50000),
            'max_position_size': getattr(settings, 'MAX_POSITION_SIZE', 500000),
            'max_portfolio_exposure': getattr(settings, 'MAX_PORTFOLIO_EXPOSURE', 2000000),
            'max_single_trade_risk': getattr(settings, 'MAX_SINGLE_TRADE_RISK', 100000),
            'max_correlation_exposure': getattr(settings, 'MAX_CORRELATION_EXPOSURE', 0.7),
            'max_vix_threshold': getattr(settings, 'MAX_VIX_THRESHOLD', 35),
            'min_liquidity_threshold': getattr(settings, 'MIN_LIQUIDITY_THRESHOLD', 500)
        }
        
        self.position_limits = {
            'max_positions_per_symbol': getattr(settings, 'MAX_POSITIONS_PER_SYMBOL', 3),
            'max_total_positions': getattr(settings, 'MAX_TOTAL_POSITIONS', 10),
            'max_same_expiry_positions': getattr(settings, 'MAX_SAME_EXPIRY_POSITIONS', 5)
        }
        
        self.daily_stats = {
            'trades_count': 0,
            'pnl': 0,
            'max_drawdown': 0,
            'risk_violations': []
        }
        
        # Circuit breakers
        self.circuit_breakers = {
            'trading_halted': False,
            'halt_reason': None,
            'halt_timestamp': None,
            'consecutive_losses': 0,
            'max_consecutive_losses': getattr(settings, 'MAX_CONSECUTIVE_LOSSES', 5)
        }
    
    def validate_trade_risk(self, signal: Dict, current_positions: Dict, market_data: Dict) -> Dict:
        """Comprehensive trade risk validation with circuit breakers"""
        risk_assessment = {
            'timestamp': datetime.now(),
            'signal': signal,
            'approved': False,
            'risk_score': 0,
            'violations': [],
            'recommendations': []
        }
        
        # Check circuit breakers first
        if self.circuit_breakers['trading_halted']:
            risk_assessment.update({
                'approved': False,
                'violations': [f"Trading halted: {self.circuit_breakers['halt_reason']}"],
                'circuit_breaker_triggered': True
            })
            logger.warning(f"[CIRCUIT BREAKER] Trading halted: {self.circuit_breakers['halt_reason']}")
            return risk_assessment
        
        try:
            violations = []
            risk_score = 0
            
            position_risk = self._assess_position_risk(signal, current_positions)
            violations.extend(position_risk['violations'])
            risk_score += position_risk['score']
            
            market_risk = self._assess_market_risk(signal, market_data)
            violations.extend(market_risk['violations'])
            risk_score += market_risk['score']
            
            liquidity_risk = self._assess_liquidity_risk(signal, market_data)
            violations.extend(liquidity_risk['violations'])
            risk_score += liquidity_risk['score']
            
            correlation_risk = self._assess_correlation_risk(signal, current_positions, market_data)
            violations.extend(correlation_risk['violations'])
            risk_score += correlation_risk['score']
            
            concentration_risk = self._assess_concentration_risk(signal, current_positions)
            violations.extend(concentration_risk['violations'])
            risk_score += concentration_risk['score']
            
            risk_assessment.update({
                'approved': len(violations) == 0 and risk_score <= 70,
                'risk_score': min(risk_score, 100),
                'violations': violations,
                'recommendations': self._generate_risk_recommendations(violations, risk_score)
            })
            
            if risk_assessment['approved']:
                logger.info(f"[OK] Trade risk approved: {signal['instrument']} (Risk Score: {risk_score})")
            else:
                logger.warning(f"[WARNING] Trade risk rejected: {len(violations)} violations (Risk Score: {risk_score})")
            
        except Exception as e:
            logger.error(f"[ERROR] Risk validation failed: {e}")
            risk_assessment['error'] = str(e)
        
        return risk_assessment
    
    def _assess_position_risk(self, signal: Dict, current_positions: Dict) -> Dict:
        """Assess position-related risks"""
        violations = []
        score = 0
        
        try:
            trade_value = signal['entry_price'] * self._get_quantity(signal)
            
            if trade_value > self.risk_limits['max_single_trade_risk']:
                violations.append(f"Trade value {trade_value} exceeds single trade limit {self.risk_limits['max_single_trade_risk']}")
                score += 30
            
            symbol_positions = sum(1 for pos in current_positions.values() if pos.get('symbol', pos.get('instrument', '')) == signal['instrument'])
            if symbol_positions >= self.position_limits['max_positions_per_symbol']:
                violations.append(f"Maximum positions per symbol ({self.position_limits['max_positions_per_symbol']}) reached for {signal['instrument']}")
                score += 25
            
            total_positions = len(current_positions)
            if total_positions >= self.position_limits['max_total_positions']:
                violations.append(f"Maximum total positions ({self.position_limits['max_total_positions']}) reached")
                score += 35
            
            total_portfolio_value = sum(pos['entry_price'] * pos['quantity'] for pos in current_positions.values())
            if total_portfolio_value + trade_value > self.risk_limits['max_portfolio_exposure']:
                violations.append(f"Portfolio exposure limit exceeded")
                score += 40
            
            if self.daily_stats['pnl'] <= self.risk_limits['max_daily_loss']:
                violations.append(f"Daily loss limit reached: {self.daily_stats['pnl']}")
                score += 50
            
        except Exception as e:
            logger.warning(f"[WARNING] Position risk assessment failed: {e}")
            score += 20
        
        return {'violations': violations, 'score': score}
    
    def _assess_market_risk(self, signal: Dict, market_data: Dict) -> Dict:
        """Assess market-related risks"""
        violations = []
        score = 0
        
        try:
            vix_data = market_data.get('vix_data', {})
            if vix_data.get('status') == 'success':
                vix = vix_data.get('vix', 16)
                if vix > self.risk_limits['max_vix_threshold']:
                    violations.append(f"VIX too high: {vix} > {self.risk_limits['max_vix_threshold']}")
                    score += 25
                elif vix > 25:
                    score += 10
            
            global_data = market_data.get('global_data', {})
            if global_data.get('status') == 'success':
                indices = global_data.get('indices', {})
                
                negative_global_count = 0
                for index_name, index_data in indices.items():
                    if isinstance(index_data, dict) and index_data.get('change_pct', 0) < -1:
                        negative_global_count += 1
                    elif isinstance(index_data, (int, float)) and index_data < -1:
                        negative_global_count += 1
                
                if negative_global_count >= 3:
                    violations.append("Multiple global indices showing significant weakness")
                    score += 15
            
            fii_dii_data = market_data.get('fii_dii_data', {})
            if fii_dii_data.get('status') == 'success':
                net_flow = fii_dii_data.get('net_flow', 0)
                if net_flow < -2000:
                    violations.append(f"Heavy FII selling: {net_flow} Cr")
                    score += 20
            
            if signal['confidence'] < 70:
                score += 15
            elif signal['confidence'] < 60:
                score += 25
            
        except Exception as e:
            logger.warning(f"[WARNING] Market risk assessment failed: {e}")
            score += 10
        
        return {'violations': violations, 'score': score}
    
    def _assess_liquidity_risk(self, signal: Dict, market_data: Dict) -> Dict:
        """Assess liquidity-related risks"""
        violations = []
        score = 0
        
        try:
            options_data = market_data.get('options_data', {}).get(signal['instrument'], {})
            
            if options_data.get('status') == 'success':
                chain = options_data.get('chain', [])
                
                target_strike = signal['strike']
                target_option = None
                
                for option in chain:
                    if (option.get('strike') == target_strike and 
                        option.get(signal['option_type'].lower())):
                        target_option = option.get(signal['option_type'].lower())
                        break
                
                if target_option:
                    volume = target_option.get('volume', 0)
                    if volume < self.risk_limits['min_liquidity_threshold']:
                        violations.append(f"Low liquidity: {volume} < {self.risk_limits['min_liquidity_threshold']}")
                        score += 30
                    
                    oi = target_option.get('oi', 0)
                    if oi < 100:
                        violations.append(f"Low open interest: {oi}")
                        score += 20
                else:
                    # Don't penalize for missing options during paper trading
                    logger.debug(f"[DEBUG] Target option not found in chain for {signal['instrument']} {target_strike}")
                    score += 0  # No penalty for paper trading
            
        except Exception as e:
            logger.warning(f"[WARNING] Liquidity risk assessment failed: {e}")
            score += 15
        
        return {'violations': violations, 'score': score}
    
    def _assess_correlation_risk(self, signal: Dict, current_positions: Dict, market_data: Dict) -> Dict:
        """Assess correlation-related risks"""
        violations = []
        score = 0
        
        try:
            same_symbol_positions = [
                pos for pos in current_positions.values() 
                if pos.get('symbol', pos.get('instrument', '')) == signal['instrument']
            ]
            
            if len(same_symbol_positions) >= 2:
                same_direction_count = sum(
                    1 for pos in same_symbol_positions 
                    if pos['option_type'] == signal['option_type']
                )
                
                if same_direction_count >= 2:
                    violations.append(f"High correlation risk: {same_direction_count} positions in same direction")
                    score += 25
            
            similar_strikes = [
                pos for pos in current_positions.values()
                if (pos.get('symbol', pos.get('instrument', '')) == signal['instrument'] and 
                    abs(pos['strike'] - signal['strike']) <= 100)
            ]
            
            if len(similar_strikes) >= 2:
                violations.append("Multiple positions with similar strikes")
                score += 20
            
        except Exception as e:
            logger.warning(f"[WARNING] Correlation risk assessment failed: {e}")
            score += 10
        
        return {'violations': violations, 'score': score}
    
    def _assess_concentration_risk(self, signal: Dict, current_positions: Dict) -> Dict:
        """Assess concentration-related risks"""
        violations = []
        score = 0
        
        try:
            total_portfolio_value = sum(
                pos['entry_price'] * pos['quantity'] 
                for pos in current_positions.values()
            )
            
            # FIXED: Use base portfolio value instead of current positions
            base_portfolio_value = self.risk_limits.get('max_portfolio_exposure', 1000000)  # 10 lakh default
            trade_value = signal['entry_price'] * self._get_quantity(signal)
            
            # Calculate concentration as percentage of base portfolio
            concentration_pct = (trade_value / base_portfolio_value) * 100
            
            if concentration_pct > 30:
                violations.append(f"High concentration risk: {concentration_pct:.1f}% of portfolio")
                score += 35
            elif concentration_pct > 20:
                score += 15
            
            logger.info(f"[RISK] Trade value: Rs.{trade_value:,.0f}, Portfolio: Rs.{base_portfolio_value:,.0f}, Concentration: {concentration_pct:.1f}%")
            
            symbol_exposure = sum(
                pos['entry_price'] * pos['quantity']
                for pos in current_positions.values()
                if pos.get('symbol', pos.get('instrument', '')) == signal['instrument']
            )
            
            if total_portfolio_value > 0:
                symbol_concentration = (symbol_exposure / total_portfolio_value) * 100
                if symbol_concentration > 50:
                    violations.append(f"High symbol concentration: {symbol_concentration:.1f}%")
                    score += 30
            
        except Exception as e:
            logger.warning(f"[WARNING] Concentration risk assessment failed: {e}")
            score += 10
        
        return {'violations': violations, 'score': score}
    
    def _get_quantity(self, signal: Dict) -> int:
        """Get quantity for risk calculations"""
        try:
            symbol = signal['instrument']
            if symbol == 'NIFTY':
                return 25
            elif symbol == 'BANKNIFTY':
                return 75
            else:
                return 25
        except Exception:
            return 25
    
    def _generate_risk_recommendations(self, violations: List[str], risk_score: float) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        try:
            if risk_score > 80:
                recommendations.append("REJECT TRADE - Risk score too high")
            elif risk_score > 60:
                recommendations.append("Reduce position size by 50%")
                recommendations.append("Consider tighter stop loss")
            elif risk_score > 40:
                recommendations.append("Monitor position closely")
                recommendations.append("Consider partial profit booking")
            
            if any("liquidity" in v.lower() for v in violations):
                recommendations.append("Wait for better liquidity before entering")
            
            if any("concentration" in v.lower() for v in violations):
                recommendations.append("Diversify across different strikes/symbols")
            
            if any("correlation" in v.lower() for v in violations):
                recommendations.append("Avoid similar positions")
            
            if any("vix" in v.lower() for v in violations):
                recommendations.append("Wait for VIX to normalize")
            
            if any("loss limit" in v.lower() for v in violations):
                recommendations.append("Stop trading for the day")
            
        except Exception as e:
            logger.warning(f"[WARNING]️ Risk recommendations generation failed: {e}")
            recommendations.append("Manual review required")
        
        return recommendations
    
    def update_daily_stats(self, trade_result: Dict):
        """Update daily statistics"""
        try:
            self.daily_stats['trades_count'] += 1
            
            if 'pnl' in trade_result:
                self.daily_stats['pnl'] += trade_result['pnl']
            
            current_drawdown = min(0, self.daily_stats['pnl'])
            self.daily_stats['max_drawdown'] = min(self.daily_stats['max_drawdown'], current_drawdown)
            
        except Exception as e:
            logger.error(f"[ERROR] Daily stats update failed: {e}")
    
    def add_risk_violation(self, violation: str):
        """Add risk violation to daily tracking"""
        try:
            self.daily_stats['risk_violations'].append({
                'timestamp': datetime.now(),
                'violation': violation
            })
            
        except Exception as e:
            logger.error(f"[ERROR] Risk violation tracking failed: {e}")
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary"""
        try:
            return {
                'timestamp': datetime.now(),
                'risk_limits': self.risk_limits,
                'position_limits': self.position_limits,
                'daily_stats': self.daily_stats,
                'risk_utilization': {
                    'daily_loss_used': abs(self.daily_stats['pnl'] / self.risk_limits['max_daily_loss']) * 100,
                    'max_drawdown_pct': abs(self.daily_stats['max_drawdown'] / self.risk_limits['max_daily_loss']) * 100
                }
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Risk summary generation failed: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    def reset_daily_limits(self):
        """Reset daily risk tracking"""
        try:
            self.daily_stats = {
                'trades_count': 0,
                'pnl': 0,
                'max_drawdown': 0,
                'risk_violations': []
            }
            
            logger.info("[OK] Daily risk limits reset")
            
        except Exception as e:
            logger.error(f"[ERROR] Daily risk reset failed: {e}")
    
    def trigger_circuit_breaker(self, reason: str):
        """Trigger emergency circuit breaker"""
        self.circuit_breakers.update({
            'trading_halted': True,
            'halt_reason': reason,
            'halt_timestamp': datetime.now()
        })
        logger.critical(f"[CIRCUIT BREAKER TRIGGERED] {reason}")
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker (manual intervention required)"""
        self.circuit_breakers.update({
            'trading_halted': False,
            'halt_reason': None,
            'halt_timestamp': None,
            'consecutive_losses': 0
        })
        logger.info("[CIRCUIT BREAKER RESET] Trading resumed")
    
    def check_circuit_breakers(self, market_data: Dict) -> Dict:
        """Check market-wide circuit breakers"""
        circuit_status = {
            'timestamp': datetime.now(),
            'triggered': False,
            'breakers': [],
            'action': 'continue'
        }
        
        try:
            vix_data = market_data.get('vix_data', {})
            if isinstance(vix_data, dict) and vix_data.get('status') == 'success':
                vix = vix_data.get('vix', 16)
                if isinstance(vix, (int, float)) and vix > 40:
                    circuit_status['triggered'] = True
                    circuit_status['breakers'].append(f"VIX circuit breaker: {vix}")
                    circuit_status['action'] = 'halt_trading'
            
            global_data = market_data.get('global_data', {})
            if isinstance(global_data, dict) and global_data.get('status') == 'success':
                indices = global_data.get('indices', {})
                
                major_decline_count = 0
                if isinstance(indices, dict):
                    for index_name, index_data in indices.items():
                        if isinstance(index_data, dict) and index_data.get('change_pct', 0) < -3:
                            major_decline_count += 1
                
                if major_decline_count >= 4:
                    circuit_status['triggered'] = True
                    circuit_status['breakers'].append("Global market circuit breaker")
                    circuit_status['action'] = 'reduce_exposure'
            
            fii_dii_data = market_data.get('fii_dii_data', {})
            if isinstance(fii_dii_data, dict) and fii_dii_data.get('status') == 'success':
                net_flow = fii_dii_data.get('net_flow', 0)
                if isinstance(net_flow, (int, float)) and net_flow < -5000:
                    circuit_status['triggered'] = True
                    circuit_status['breakers'].append(f"FII selling circuit breaker: {net_flow} Cr")
                    circuit_status['action'] = 'defensive_mode'
            
        except Exception as e:
            logger.error(f"[ERROR] Circuit breaker check failed: {e}")
            circuit_status['error'] = str(e)
        
        return circuit_status
