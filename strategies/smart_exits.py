"""
Smart Exit Strategies for VLR_AI Trading System
Implements Trailing Stop-loss, Time-based Exits, Profit Scaling, Greeks-based Exits
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math

logger = logging.getLogger('trading_system.smart_exits')

@dataclass
class ExitSignal:
    """Exit signal information"""
    exit_type: str  # STOP_LOSS, TAKE_PROFIT, TIME_BASED, GREEKS_BASED, TRAILING_STOP
    urgency: str   # LOW, MEDIUM, HIGH, CRITICAL
    reason: str
    recommended_action: str  # PARTIAL_EXIT, FULL_EXIT, ADJUST_STOP
    exit_price: float
    confidence: float

class SmartExitManager:
    """Advanced exit strategy management"""
    
    def __init__(self, settings):
        self.settings = settings
        
    def trailing_stop_loss(self, position: Dict[str, Any], current_price: float,
                          market_data: Dict[str, Any]) -> Optional[ExitSignal]:
        """
        Implement trailing stop-loss strategy
        
        Args:
            position: Current position details
            current_price: Current market price
            market_data: Current market data
        """
        try:
            entry_price = position.get('entry_price', 0)
            position_type = position.get('position_type', 'LONG')  # LONG or SHORT
            trailing_pct = position.get('trailing_stop_pct', 0.02)  # 2% default
            
            if entry_price <= 0:
                return None
            
            # Calculate current P&L percentage
            if position_type == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
                # Update trailing stop if position is profitable
                if pnl_pct > 0:
                    trailing_stop_price = current_price * (1 - trailing_pct)
                    
                    # Check if current price hits trailing stop
                    if current_price <= trailing_stop_price:
                        return ExitSignal(
                            exit_type="TRAILING_STOP",
                            urgency="HIGH",
                            reason=f"Price {current_price} hit trailing stop at {trailing_stop_price:.2f}",
                            recommended_action="FULL_EXIT",
                            exit_price=trailing_stop_price,
                            confidence=0.9
                        )
            else:  # SHORT position
                pnl_pct = (entry_price - current_price) / entry_price
                if pnl_pct > 0:
                    trailing_stop_price = current_price * (1 + trailing_pct)
                    
                    if current_price >= trailing_stop_price:
                        return ExitSignal(
                            exit_type="TRAILING_STOP",
                            urgency="HIGH",
                            reason=f"Price {current_price} hit trailing stop at {trailing_stop_price:.2f}",
                            recommended_action="FULL_EXIT",
                            exit_price=trailing_stop_price,
                            confidence=0.9
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Trailing stop-loss calculation failed: {e}")
            return None
    
    def time_based_exit(self, position: Dict[str, Any], current_time: datetime) -> Optional[ExitSignal]:
        """
        Implement time-based exit strategies
        
        Args:
            position: Current position details
            current_time: Current timestamp
        """
        try:
            entry_time = position.get('entry_time')
            if not entry_time:
                return None
            
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
            
            time_in_position = current_time - entry_time
            max_hold_time = timedelta(hours=position.get('max_hold_hours', 24))
            
            # Check if maximum hold time exceeded
            if time_in_position >= max_hold_time:
                return ExitSignal(
                    exit_type="TIME_BASED",
                    urgency="MEDIUM",
                    reason=f"Position held for {time_in_position}, exceeds max hold time of {max_hold_time}",
                    recommended_action="FULL_EXIT",
                    exit_price=0,  # Use market price
                    confidence=0.7
                )
            
            # Check for end-of-day exit (for intraday strategies)
            if position.get('intraday_only', False):
                market_close_time = current_time.replace(hour=15, minute=20, second=0, microsecond=0)
                if current_time >= market_close_time:
                    return ExitSignal(
                        exit_type="TIME_BASED",
                        urgency="CRITICAL",
                        reason="End of trading day - intraday position must be closed",
                        recommended_action="FULL_EXIT",
                        exit_price=0,
                        confidence=1.0
                    )
            
            # Check for expiry-based exit (for options)
            expiry_date = position.get('expiry_date')
            if expiry_date:
                if isinstance(expiry_date, str):
                    expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d')
                
                days_to_expiry = (expiry_date - current_time).days
                
                # Exit 1 day before expiry to avoid assignment risk
                if days_to_expiry <= 1:
                    return ExitSignal(
                        exit_type="TIME_BASED",
                        urgency="CRITICAL",
                        reason=f"Option expires in {days_to_expiry} days - avoiding assignment risk",
                        recommended_action="FULL_EXIT",
                        exit_price=0,
                        confidence=0.95
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Time-based exit calculation failed: {e}")
            return None
    
    def profit_target_scaling(self, position: Dict[str, Any], current_price: float) -> Optional[ExitSignal]:
        """
        Implement profit target scaling (partial exits at different profit levels)
        
        Args:
            position: Current position details
            current_price: Current market price
        """
        try:
            entry_price = position.get('entry_price', 0)
            position_type = position.get('position_type', 'LONG')
            quantity = position.get('quantity', 0)
            
            if entry_price <= 0 or quantity <= 0:
                return None
            
            # Calculate current P&L percentage
            if position_type == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Define profit scaling levels
            profit_levels = [
                (0.02, 0.25, "Take 25% profit at 2% gain"),    # 2% profit, exit 25%
                (0.04, 0.50, "Take 50% profit at 4% gain"),    # 4% profit, exit 50%
                (0.06, 0.75, "Take 75% profit at 6% gain"),    # 6% profit, exit 75%
                (0.10, 1.00, "Take full profit at 10% gain")   # 10% profit, exit all
            ]
            
            for profit_threshold, exit_percentage, reason in profit_levels:
                if pnl_pct >= profit_threshold:
                    # Check if this level hasn't been triggered yet
                    exits_taken = position.get('profit_exits_taken', [])
                    if profit_threshold not in exits_taken:
                        return ExitSignal(
                            exit_type="TAKE_PROFIT",
                            urgency="MEDIUM",
                            reason=reason,
                            recommended_action="PARTIAL_EXIT" if exit_percentage < 1.0 else "FULL_EXIT",
                            exit_price=current_price,
                            confidence=0.8
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Profit target scaling failed: {e}")
            return None
    
    def greeks_based_exit(self, position: Dict[str, Any], greeks_data: Dict[str, Any],
                         market_data: Dict[str, Any]) -> Optional[ExitSignal]:
        """
        Implement Greeks-based exit strategies
        
        Args:
            position: Current position details
            greeks_data: Current Greeks values
            market_data: Current market data
        """
        try:
            # Delta-based exits
            delta = greeks_data.get('delta', 0)
            if abs(delta) < 0.1:  # Very low delta
                return ExitSignal(
                    exit_type="GREEKS_BASED",
                    urgency="MEDIUM",
                    reason=f"Delta too low ({delta:.3f}) - option losing directional sensitivity",
                    recommended_action="FULL_EXIT",
                    exit_price=0,
                    confidence=0.7
                )
            
            # Theta-based exits (time decay)
            theta = greeks_data.get('theta', 0)
            days_to_expiry = position.get('days_to_expiry', 30)
            
            if days_to_expiry <= 7 and theta < -50:  # High time decay near expiry
                return ExitSignal(
                    exit_type="GREEKS_BASED",
                    urgency="HIGH",
                    reason=f"High theta decay ({theta:.2f}) with {days_to_expiry} days to expiry",
                    recommended_action="FULL_EXIT",
                    exit_price=0,
                    confidence=0.85
                )
            
            # Vega-based exits (volatility risk)
            vega = greeks_data.get('vega', 0)
            current_vix = market_data.get('india_vix', 20)
            
            if abs(vega) > 100 and current_vix > 35:  # High vega risk in high volatility
                return ExitSignal(
                    exit_type="GREEKS_BASED",
                    urgency="MEDIUM",
                    reason=f"High vega exposure ({vega:.2f}) in high volatility environment (VIX: {current_vix})",
                    recommended_action="PARTIAL_EXIT",
                    exit_price=0,
                    confidence=0.75
                )
            
            # Gamma-based exits
            gamma = greeks_data.get('gamma', 0)
            if abs(gamma) > 0.01 and position.get('position_type') == 'SHORT':  # High gamma risk for short positions
                return ExitSignal(
                    exit_type="GREEKS_BASED",
                    urgency="HIGH",
                    reason=f"High gamma risk ({gamma:.4f}) for short position",
                    recommended_action="FULL_EXIT",
                    exit_price=0,
                    confidence=0.8
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Greeks-based exit calculation failed: {e}")
            return None
    
    def volatility_based_exit(self, position: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[ExitSignal]:
        """
        Implement volatility-based exit strategies
        
        Args:
            position: Current position details
            market_data: Current market data including VIX
        """
        try:
            current_vix = market_data.get('india_vix', 20)
            entry_vix = position.get('entry_vix', current_vix)
            
            # Exit if volatility has changed significantly
            vix_change_pct = (current_vix - entry_vix) / entry_vix if entry_vix > 0 else 0
            
            # High volatility spike - exit long volatility positions
            if vix_change_pct > 0.5 and position.get('volatility_exposure', 'NEUTRAL') == 'LONG':
                return ExitSignal(
                    exit_type="VOLATILITY_BASED",
                    urgency="HIGH",
                    reason=f"VIX spiked {vix_change_pct*100:.1f}% from entry - take volatility profits",
                    recommended_action="PARTIAL_EXIT",
                    exit_price=0,
                    confidence=0.8
                )
            
            # Volatility crush - exit short volatility positions
            if vix_change_pct < -0.3 and position.get('volatility_exposure', 'NEUTRAL') == 'SHORT':
                return ExitSignal(
                    exit_type="VOLATILITY_BASED",
                    urgency="HIGH",
                    reason=f"VIX dropped {abs(vix_change_pct)*100:.1f}% from entry - volatility crush profits",
                    recommended_action="PARTIAL_EXIT",
                    exit_price=0,
                    confidence=0.8
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Volatility-based exit calculation failed: {e}")
            return None
    
    def comprehensive_exit_analysis(self, position: Dict[str, Any], current_price: float,
                                  market_data: Dict[str, Any], greeks_data: Dict[str, Any] = None) -> List[ExitSignal]:
        """
        Perform comprehensive exit analysis using all strategies
        
        Args:
            position: Current position details
            current_price: Current market price
            market_data: Current market data
            greeks_data: Current Greeks data (optional)
        """
        try:
            exit_signals = []
            current_time = datetime.now()
            
            # Check all exit strategies
            strategies = [
                self.trailing_stop_loss(position, current_price, market_data),
                self.time_based_exit(position, current_time),
                self.profit_target_scaling(position, current_price),
                self.volatility_based_exit(position, market_data)
            ]
            
            # Add Greeks-based exit if Greeks data available
            if greeks_data:
                strategies.append(self.greeks_based_exit(position, greeks_data, market_data))
            
            # Filter out None results
            exit_signals = [signal for signal in strategies if signal is not None]
            
            # Sort by urgency and confidence
            urgency_order = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
            exit_signals.sort(key=lambda x: (urgency_order.get(x.urgency, 0), x.confidence), reverse=True)
            
            return exit_signals
            
        except Exception as e:
            logger.error(f"Comprehensive exit analysis failed: {e}")
            return []
    
    def execute_exit_recommendation(self, position: Dict[str, Any], exit_signal: ExitSignal) -> Dict[str, Any]:
        """
        Generate execution plan for exit recommendation
        
        Args:
            position: Current position details
            exit_signal: Exit signal to execute
        """
        try:
            execution_plan = {
                'action': exit_signal.recommended_action,
                'exit_type': exit_signal.exit_type,
                'urgency': exit_signal.urgency,
                'reason': exit_signal.reason,
                'confidence': exit_signal.confidence,
                'execution_details': {}
            }
            
            current_quantity = position.get('quantity', 0)
            
            if exit_signal.recommended_action == 'FULL_EXIT':
                execution_plan['execution_details'] = {
                    'quantity_to_exit': current_quantity,
                    'exit_percentage': 100,
                    'order_type': 'MARKET' if exit_signal.urgency in ['HIGH', 'CRITICAL'] else 'LIMIT',
                    'limit_price': exit_signal.exit_price if exit_signal.exit_price > 0 else None
                }
            
            elif exit_signal.recommended_action == 'PARTIAL_EXIT':
                # Default to 50% exit for partial exits
                exit_percentage = 0.5
                if 'Take 25% profit' in exit_signal.reason:
                    exit_percentage = 0.25
                elif 'Take 50% profit' in exit_signal.reason:
                    exit_percentage = 0.5
                elif 'Take 75% profit' in exit_signal.reason:
                    exit_percentage = 0.75
                
                execution_plan['execution_details'] = {
                    'quantity_to_exit': int(current_quantity * exit_percentage),
                    'exit_percentage': exit_percentage * 100,
                    'order_type': 'LIMIT',
                    'limit_price': exit_signal.exit_price if exit_signal.exit_price > 0 else None
                }
            
            elif exit_signal.recommended_action == 'ADJUST_STOP':
                execution_plan['execution_details'] = {
                    'new_stop_price': exit_signal.exit_price,
                    'stop_type': 'TRAILING' if exit_signal.exit_type == 'TRAILING_STOP' else 'FIXED'
                }
            
            return execution_plan
            
        except Exception as e:
            logger.error(f"Exit execution plan generation failed: {e}")
            return {'error': str(e)}
