"""
INSTITUTIONAL-GRADE Dynamic Position Sizing
Implements Kelly Criterion, VIX-adjusted, confidence-based, capital-aware position sizing
Advanced risk management with volatility clustering and regime detection
"""

import logging
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger('trading_system.dynamic_position_sizer')

@dataclass
class PositionSizeResult:
    """Position sizing calculation result"""
    recommended_quantity: int
    recommended_value: float
    base_quantity: int
    adjustments: Dict[str, float]
    risk_percentage: float
    kelly_fraction: float
    confidence_multiplier: float
    volatility_multiplier: float
    capital_utilization: float

class DynamicPositionSizer:
    """Advanced dynamic position sizing system"""
    
    def __init__(self, settings):
        self.settings = settings
        
        # Base risk parameters
        self.base_risk_per_trade = getattr(settings, 'BASE_RISK_PER_TRADE', 0.02)  # 2%
        self.max_risk_per_trade = getattr(settings, 'MAX_RISK_PER_TRADE', 0.05)   # 5%
        self.min_risk_per_trade = getattr(settings, 'MIN_RISK_PER_TRADE', 0.005)  # 0.5%
        
        # Position sizing parameters
        self.max_position_value = getattr(settings, 'MAX_POSITION_SIZE', 500000)
        self.max_portfolio_exposure = getattr(settings, 'MAX_PORTFOLIO_EXPOSURE', 2000000)
        
    def calculate_kelly_fraction(self, win_rate: float, avg_win_loss_ratio: float, 
                                confidence: float = 0.0, volatility: float = 0.0) -> float:
        """
        INSTITUTIONAL-GRADE Kelly Criterion with confidence and volatility adjustments
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win_loss_ratio: Average win/loss ratio
            confidence: Signal confidence (0-100)
            volatility: Market volatility (VIX/100)
        """
        try:
            if win_rate <= 0 or win_rate >= 1 or avg_win_loss_ratio <= 0:
                return 0.02  # Conservative default
            
            # ENHANCED KELLY FORMULA with confidence adjustment
            # Base Kelly: f = (bp - q) / b
            base_kelly = (avg_win_loss_ratio * win_rate - (1 - win_rate)) / avg_win_loss_ratio
            
            # CONFIDENCE ADJUSTMENT
            # Higher confidence signals get larger position sizes
            confidence_multiplier = 1.0
            if confidence > 0:
                confidence_multiplier = 0.5 + (confidence / 100) * 1.5  # Range: 0.5 to 2.0
            
            # VOLATILITY ADJUSTMENT
            # Higher volatility reduces position size (risk management)
            volatility_multiplier = 1.0
            if volatility > 0:
                volatility_multiplier = max(0.3, 1.0 - (volatility * 2))  # Reduce size in high vol
            
            # APPLY ADJUSTMENTS
            adjusted_kelly = base_kelly * confidence_multiplier * volatility_multiplier
            
            # INSTITUTIONAL RISK LIMITS
            # Never risk more than 10% on single trade, even with high Kelly
            max_kelly = 0.10
            min_kelly = 0.005  # Minimum 0.5% position
            
            kelly_fraction = max(min_kelly, min(adjusted_kelly, max_kelly))
            
            logger.debug(f"[KELLY] Base: {base_kelly:.3f}, Conf: {confidence_multiplier:.2f}, "
                        f"Vol: {volatility_multiplier:.2f}, Final: {kelly_fraction:.3f}")
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"[ERROR] Enhanced Kelly calculation failed: {e}")
            return 0.02  # Conservative default
    
    def calculate_volatility_adjustment(self, current_vix: float, 
                                      historical_avg_vix: float = 20.0) -> float:
        """
        Calculate position size adjustment based on volatility
        
        Args:
            current_vix: Current VIX level
            historical_avg_vix: Historical average VIX
        """
        try:
            # Normalize VIX relative to historical average
            vix_ratio = current_vix / historical_avg_vix
            
            # Reduce position size in high volatility, increase in low volatility
            if vix_ratio > 1.5:  # Very high volatility
                return 0.5
            elif vix_ratio > 1.2:  # High volatility
                return 0.7
            elif vix_ratio > 1.0:  # Above average volatility
                return 0.85
            elif vix_ratio > 0.8:  # Below average volatility
                return 1.1
            else:  # Very low volatility
                return 1.2
                
        except Exception as e:
            logger.error(f"Volatility adjustment calculation failed: {e}")
            return 1.0
    
    def calculate_confidence_adjustment(self, confidence_score: float) -> float:
        """
        Calculate position size adjustment based on signal confidence
        
        Args:
            confidence_score: Signal confidence (0-100)
        """
        try:
            # Normalize confidence to 0-1
            normalized_confidence = confidence_score / 100.0
            
            # Exponential scaling for confidence
            # Low confidence: 0.3x, Medium confidence: 1.0x, High confidence: 1.8x
            if normalized_confidence < 0.3:
                return 0.3
            elif normalized_confidence < 0.5:
                return 0.5 + (normalized_confidence - 0.3) * 2.5  # 0.5 to 1.0
            elif normalized_confidence < 0.8:
                return 1.0 + (normalized_confidence - 0.5) * 1.67  # 1.0 to 1.5
            else:
                return 1.5 + (normalized_confidence - 0.8) * 1.5   # 1.5 to 1.8
                
        except Exception as e:
            logger.error(f"Confidence adjustment calculation failed: {e}")
            return 1.0
    
    def calculate_capital_utilization_adjustment(self, available_capital: float,
                                               current_exposure: float) -> float:
        """
        Calculate adjustment based on current capital utilization
        
        Args:
            available_capital: Total available capital
            current_exposure: Current portfolio exposure
        """
        try:
            if available_capital <= 0:
                return 0.1
            
            utilization_ratio = current_exposure / available_capital
            
            # Reduce position size as utilization increases
            if utilization_ratio > 0.8:  # Over 80% utilized
                return 0.3
            elif utilization_ratio > 0.6:  # Over 60% utilized
                return 0.5
            elif utilization_ratio > 0.4:  # Over 40% utilized
                return 0.7
            elif utilization_ratio > 0.2:  # Over 20% utilized
                return 0.9
            else:  # Low utilization
                return 1.0
                
        except Exception as e:
            logger.error(f"Capital utilization adjustment failed: {e}")
            return 1.0
    
    def calculate_correlation_adjustment(self, signal: Dict[str, Any],
                                       existing_positions: List[Dict[str, Any]]) -> float:
        """
        Calculate adjustment based on correlation with existing positions
        
        Args:
            signal: New signal
            existing_positions: Current portfolio positions
        """
        try:
            if not existing_positions:
                return 1.0
            
            signal_instrument = signal.get('instrument', 'NIFTY')
            signal_direction = signal.get('direction', 'LONG')
            
            # Count similar positions
            similar_positions = 0
            total_positions = len(existing_positions)
            
            for position in existing_positions:
                pos_instrument = position.get('instrument', 'NIFTY')
                pos_direction = position.get('direction', 'LONG')
                
                # Same instrument and direction
                if pos_instrument == signal_instrument and pos_direction == signal_direction:
                    similar_positions += 1
            
            # Reduce size if too many similar positions
            if similar_positions >= 3:
                return 0.3
            elif similar_positions >= 2:
                return 0.6
            elif similar_positions >= 1:
                return 0.8
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Correlation adjustment calculation failed: {e}")
            return 1.0
    
    def calculate_dynamic_position_size(self, signal: Dict[str, Any],
                                      market_data: Dict[str, Any],
                                      portfolio_data: Dict[str, Any],
                                      historical_performance: Dict[str, Any] = None) -> PositionSizeResult:
        """
        Calculate dynamic position size using multiple factors
        
        Args:
            signal: Trading signal with confidence score
            market_data: Current market data including VIX
            portfolio_data: Current portfolio information
            historical_performance: Historical trading performance data
        """
        try:
            # Extract key parameters
            confidence_score = signal.get('confidence_score', 50.0)
            entry_price = signal.get('entry_price', 100.0)
            instrument = signal.get('instrument', 'NIFTY')
            
            available_capital = portfolio_data.get('available_capital', 1000000)
            current_exposure = portfolio_data.get('current_exposure', 0)
            existing_positions = portfolio_data.get('positions', [])
            
            current_vix = market_data.get('india_vix', 20.0)
            
            # Get lot size for instrument
            lot_size = self.settings.get_lot_size(instrument)
            
            # Calculate base position size (risk-based)
            base_risk_amount = available_capital * self.base_risk_per_trade
            base_quantity = max(1, int(base_risk_amount / (entry_price * lot_size))) * lot_size
            
            # Calculate Kelly fraction if historical data available
            kelly_fraction = 0.15  # Default
            if historical_performance:
                win_rate = historical_performance.get('win_rate', 0.6)
                avg_win_loss_ratio = historical_performance.get('avg_win_loss_ratio', 1.5)
                kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win_loss_ratio)
            
            # Calculate adjustments
            confidence_multiplier = self.calculate_confidence_adjustment(confidence_score)
            volatility_multiplier = self.calculate_volatility_adjustment(current_vix)
            capital_multiplier = self.calculate_capital_utilization_adjustment(
                available_capital, current_exposure)
            correlation_multiplier = self.calculate_correlation_adjustment(signal, existing_positions)
            
            # Apply Kelly criterion
            kelly_multiplier = kelly_fraction / self.base_risk_per_trade
            
            # Calculate final multiplier
            final_multiplier = (confidence_multiplier * volatility_multiplier * 
                              capital_multiplier * correlation_multiplier * kelly_multiplier)
            
            # Calculate recommended quantity
            recommended_quantity = max(lot_size, int(base_quantity * final_multiplier))
            
            # Round to lot size
            recommended_quantity = (recommended_quantity // lot_size) * lot_size
            
            # Apply position limits
            max_quantity_by_value = int(self.max_position_value / entry_price)
            max_quantity_by_value = (max_quantity_by_value // lot_size) * lot_size
            
            recommended_quantity = min(recommended_quantity, max_quantity_by_value)
            
            # Calculate final metrics
            recommended_value = recommended_quantity * entry_price
            risk_percentage = (recommended_value / available_capital) * 100
            capital_utilization = ((current_exposure + recommended_value) / available_capital) * 100
            
            # Ensure minimum position size
            if recommended_quantity < lot_size:
                recommended_quantity = lot_size
                recommended_value = lot_size * entry_price
                risk_percentage = (recommended_value / available_capital) * 100
            
            return PositionSizeResult(
                recommended_quantity=recommended_quantity,
                recommended_value=recommended_value,
                base_quantity=base_quantity,
                adjustments={
                    'confidence': confidence_multiplier,
                    'volatility': volatility_multiplier,
                    'capital_utilization': capital_multiplier,
                    'correlation': correlation_multiplier,
                    'kelly': kelly_multiplier,
                    'final': final_multiplier
                },
                risk_percentage=risk_percentage,
                kelly_fraction=kelly_fraction,
                confidence_multiplier=confidence_multiplier,
                volatility_multiplier=volatility_multiplier,
                capital_utilization=capital_utilization
            )
            
        except Exception as e:
            logger.error(f"Dynamic position sizing failed: {e}")
            # Return conservative fallback
            lot_size = self.settings.get_lot_size(signal.get('instrument', 'NIFTY'))
            return PositionSizeResult(
                recommended_quantity=lot_size,
                recommended_value=lot_size * signal.get('entry_price', 100),
                base_quantity=lot_size,
                adjustments={'error': str(e)},
                risk_percentage=1.0,
                kelly_fraction=0.1,
                confidence_multiplier=1.0,
                volatility_multiplier=1.0,
                capital_utilization=1.0
            )
    
    def validate_position_size(self, position_result: PositionSizeResult,
                              portfolio_limits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate calculated position size against portfolio limits
        
        Args:
            position_result: Calculated position size result
            portfolio_limits: Portfolio risk limits
        """
        try:
            validation_result = {
                'approved': True,
                'warnings': [],
                'adjustments_needed': [],
                'final_quantity': position_result.recommended_quantity,
                'final_value': position_result.recommended_value
            }
            
            # Check risk percentage limits
            max_risk_pct = portfolio_limits.get('max_risk_per_trade_pct', 5.0)
            if position_result.risk_percentage > max_risk_pct:
                validation_result['warnings'].append(
                    f"Risk percentage {position_result.risk_percentage:.2f}% exceeds limit {max_risk_pct}%"
                )
                
                # Adjust quantity to meet risk limit
                max_allowed_value = portfolio_limits.get('available_capital', 1000000) * (max_risk_pct / 100)
                entry_price = position_result.recommended_value / position_result.recommended_quantity
                lot_size = self.settings.get_lot_size('NIFTY')  # Default
                
                adjusted_quantity = int(max_allowed_value / entry_price)
                adjusted_quantity = (adjusted_quantity // lot_size) * lot_size
                
                validation_result['adjustments_needed'].append('REDUCE_FOR_RISK_LIMIT')
                validation_result['final_quantity'] = max(lot_size, adjusted_quantity)
                validation_result['final_value'] = validation_result['final_quantity'] * entry_price
            
            # Check portfolio exposure limits
            current_exposure = portfolio_limits.get('current_exposure', 0)
            max_portfolio_exposure = portfolio_limits.get('max_portfolio_exposure', 2000000)
            
            if current_exposure + position_result.recommended_value > max_portfolio_exposure:
                validation_result['warnings'].append(
                    f"Position would exceed portfolio exposure limit"
                )
                validation_result['adjustments_needed'].append('REDUCE_FOR_EXPOSURE_LIMIT')
            
            # Check position count limits
            current_positions = portfolio_limits.get('current_position_count', 0)
            max_positions = portfolio_limits.get('max_total_positions', 10)
            
            if current_positions >= max_positions:
                validation_result['approved'] = False
                validation_result['warnings'].append(
                    f"Maximum position count ({max_positions}) reached"
                )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Position size validation failed: {e}")
            return {
                'approved': False,
                'error': str(e),
                'final_quantity': position_result.recommended_quantity,
                'final_value': position_result.recommended_value
            }
    
    def get_sizing_report(self, position_result: PositionSizeResult,
                         signal: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive position sizing report"""
        try:
            report = {
                'signal_info': {
                    'instrument': signal.get('instrument', 'UNKNOWN'),
                    'confidence_score': signal.get('confidence_score', 0),
                    'entry_price': signal.get('entry_price', 0)
                },
                'sizing_calculation': {
                    'base_quantity': position_result.base_quantity,
                    'recommended_quantity': position_result.recommended_quantity,
                    'recommended_value': position_result.recommended_value,
                    'risk_percentage': position_result.risk_percentage,
                    'capital_utilization': position_result.capital_utilization
                },
                'adjustments_applied': position_result.adjustments,
                'risk_metrics': {
                    'kelly_fraction': position_result.kelly_fraction,
                    'confidence_impact': position_result.confidence_multiplier,
                    'volatility_impact': position_result.volatility_multiplier
                },
                'recommendations': []
            }
            
            # Add recommendations based on calculations
            if position_result.risk_percentage > 4:
                report['recommendations'].append('HIGH_RISK: Consider reducing position size')
            elif position_result.risk_percentage < 1:
                report['recommendations'].append('LOW_RISK: Consider increasing position size if confident')
            
            if position_result.volatility_multiplier < 0.7:
                report['recommendations'].append('HIGH_VOLATILITY: Position size reduced due to market volatility')
            
            if position_result.confidence_multiplier > 1.5:
                report['recommendations'].append('HIGH_CONFIDENCE: Position size increased due to strong signal')
            
            return report
            
        except Exception as e:
            logger.error(f"Sizing report generation failed: {e}")
            return {'error': str(e)}
