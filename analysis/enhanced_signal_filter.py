"""
Enhanced Signal Filtering for VLR_AI Trading System
Implements volume confirmation, spread validation, liquidity checks, time-based filters
"""

import logging
from datetime import datetime, time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger('trading_system.enhanced_signal_filter')

@dataclass
class FilterResult:
    """Signal filter result"""
    passed: bool
    filter_name: str
    reason: str
    confidence_adjustment: float  # -1.0 to +1.0

@dataclass
class EnhancedSignal:
    """Enhanced signal with filtering results"""
    original_signal: Dict[str, Any]
    filter_results: List[FilterResult]
    final_confidence: float
    quality_score: float
    recommendation: str  # EXECUTE, SKIP, WAIT

class EnhancedSignalFilter:
    """Advanced signal filtering system"""
    
    def __init__(self, settings):
        self.settings = settings
        
        # Filter thresholds (configurable)
        self.min_volume = getattr(settings, 'MIN_SIGNAL_VOLUME', 500)
        self.max_bid_ask_spread_pct = getattr(settings, 'MAX_BID_ASK_SPREAD_PCT', 2.0)
        self.min_open_interest = getattr(settings, 'MIN_OPEN_INTEREST', 1000)
        self.avoid_first_minutes = getattr(settings, 'AVOID_FIRST_MINUTES', 15)
        self.avoid_last_minutes = getattr(settings, 'AVOID_LAST_MINUTES', 15)
        
        # Market timing
        self.market_open = time(9, 15)
        self.market_close = time(15, 30)
    
    def volume_confirmation_filter(self, signal: Dict[str, Any], 
                                  options_data: List[Dict[str, Any]]) -> FilterResult:
        """
        Filter based on volume confirmation
        
        Args:
            signal: Trading signal
            options_data: Current options chain data
        """
        try:
            strike_price = signal.get('strike_price', 0)
            option_type = signal.get('option_type', 'CE')
            
            # Find matching option in options data
            matching_option = None
            for option in options_data:
                if (option.get('strike_price') == strike_price and 
                    option.get('option_type') == option_type):
                    matching_option = option
                    break
            
            if not matching_option:
                return FilterResult(
                    passed=False,
                    filter_name="VOLUME_CONFIRMATION",
                    reason="Option data not found in chain",
                    confidence_adjustment=-0.3
                )
            
            volume = matching_option.get('volume', 0)
            
            if volume >= self.min_volume:
                # High volume gives confidence boost
                if volume >= self.min_volume * 3:
                    confidence_boost = 0.2
                elif volume >= self.min_volume * 2:
                    confidence_boost = 0.1
                else:
                    confidence_boost = 0.0
                
                return FilterResult(
                    passed=True,
                    filter_name="VOLUME_CONFIRMATION",
                    reason=f"Volume {volume} meets minimum requirement of {self.min_volume}",
                    confidence_adjustment=confidence_boost
                )
            else:
                return FilterResult(
                    passed=False,
                    filter_name="VOLUME_CONFIRMATION",
                    reason=f"Volume {volume} below minimum requirement of {self.min_volume}",
                    confidence_adjustment=-0.2
                )
                
        except Exception as e:
            logger.error(f"Volume confirmation filter failed: {e}")
            return FilterResult(
                passed=False,
                filter_name="VOLUME_CONFIRMATION",
                reason=f"Filter error: {e}",
                confidence_adjustment=-0.1
            )
    
    def spread_validation_filter(self, signal: Dict[str, Any],
                                options_data: List[Dict[str, Any]]) -> FilterResult:
        """
        Filter based on bid-ask spread validation
        
        Args:
            signal: Trading signal
            options_data: Current options chain data
        """
        try:
            strike_price = signal.get('strike_price', 0)
            option_type = signal.get('option_type', 'CE')
            
            # Find matching option
            matching_option = None
            for option in options_data:
                if (option.get('strike_price') == strike_price and 
                    option.get('option_type') == option_type):
                    matching_option = option
                    break
            
            if not matching_option:
                return FilterResult(
                    passed=False,
                    filter_name="SPREAD_VALIDATION",
                    reason="Option data not found for spread calculation",
                    confidence_adjustment=-0.2
                )
            
            bid_price = matching_option.get('bid', 0)
            ask_price = matching_option.get('ask', 0)
            ltp = matching_option.get('ltp', 0)
            
            if bid_price <= 0 or ask_price <= 0 or ltp <= 0:
                return FilterResult(
                    passed=False,
                    filter_name="SPREAD_VALIDATION",
                    reason="Invalid bid/ask/ltp prices",
                    confidence_adjustment=-0.3
                )
            
            spread = ask_price - bid_price
            spread_pct = (spread / ltp) * 100
            
            if spread_pct <= self.max_bid_ask_spread_pct:
                # Tight spreads give confidence boost
                if spread_pct <= self.max_bid_ask_spread_pct * 0.5:
                    confidence_boost = 0.15
                else:
                    confidence_boost = 0.05
                
                return FilterResult(
                    passed=True,
                    filter_name="SPREAD_VALIDATION",
                    reason=f"Bid-ask spread {spread_pct:.2f}% within acceptable range",
                    confidence_adjustment=confidence_boost
                )
            else:
                return FilterResult(
                    passed=False,
                    filter_name="SPREAD_VALIDATION",
                    reason=f"Bid-ask spread {spread_pct:.2f}% exceeds maximum {self.max_bid_ask_spread_pct}%",
                    confidence_adjustment=-0.25
                )
                
        except Exception as e:
            logger.error(f"Spread validation filter failed: {e}")
            return FilterResult(
                passed=False,
                filter_name="SPREAD_VALIDATION",
                reason=f"Filter error: {e}",
                confidence_adjustment=-0.1
            )
    
    def liquidity_check_filter(self, signal: Dict[str, Any],
                              options_data: List[Dict[str, Any]]) -> FilterResult:
        """
        Filter based on open interest (liquidity) checks
        
        Args:
            signal: Trading signal
            options_data: Current options chain data
        """
        try:
            strike_price = signal.get('strike_price', 0)
            option_type = signal.get('option_type', 'CE')
            
            # Find matching option
            matching_option = None
            for option in options_data:
                if (option.get('strike_price') == strike_price and 
                    option.get('option_type') == option_type):
                    matching_option = option
                    break
            
            if not matching_option:
                return FilterResult(
                    passed=False,
                    filter_name="LIQUIDITY_CHECK",
                    reason="Option data not found for liquidity check",
                    confidence_adjustment=-0.2
                )
            
            open_interest = matching_option.get('oi', 0)
            
            if open_interest >= self.min_open_interest:
                # High OI gives confidence boost
                if open_interest >= self.min_open_interest * 5:
                    confidence_boost = 0.2
                elif open_interest >= self.min_open_interest * 2:
                    confidence_boost = 0.1
                else:
                    confidence_boost = 0.0
                
                return FilterResult(
                    passed=True,
                    filter_name="LIQUIDITY_CHECK",
                    reason=f"Open Interest {open_interest} meets minimum requirement",
                    confidence_adjustment=confidence_boost
                )
            else:
                return FilterResult(
                    passed=False,
                    filter_name="LIQUIDITY_CHECK",
                    reason=f"Open Interest {open_interest} below minimum {self.min_open_interest}",
                    confidence_adjustment=-0.3
                )
                
        except Exception as e:
            logger.error(f"Liquidity check filter failed: {e}")
            return FilterResult(
                passed=False,
                filter_name="LIQUIDITY_CHECK",
                reason=f"Filter error: {e}",
                confidence_adjustment=-0.1
            )
    
    def time_based_filter(self, signal: Dict[str, Any], current_time: datetime = None) -> FilterResult:
        """
        Filter based on time-based rules (avoid first/last minutes)
        
        Args:
            signal: Trading signal
            current_time: Current timestamp (optional, defaults to now)
        """
        try:
            if current_time is None:
                current_time = datetime.now()
            
            current_time_only = current_time.time()
            
            # Check if market is open
            if not (self.market_open <= current_time_only <= self.market_close):
                return FilterResult(
                    passed=False,
                    filter_name="TIME_BASED",
                    reason="Market is closed",
                    confidence_adjustment=-0.5
                )
            
            # Calculate minutes from market open and close
            market_open_dt = datetime.combine(current_time.date(), self.market_open)
            market_close_dt = datetime.combine(current_time.date(), self.market_close)
            
            minutes_from_open = (current_time - market_open_dt).total_seconds() / 60
            minutes_to_close = (market_close_dt - current_time).total_seconds() / 60
            
            # Avoid first few minutes (market volatility)
            if minutes_from_open < self.avoid_first_minutes:
                return FilterResult(
                    passed=False,
                    filter_name="TIME_BASED",
                    reason=f"Too close to market open ({minutes_from_open:.1f} min), avoiding initial volatility",
                    confidence_adjustment=-0.2
                )
            
            # Avoid last few minutes (liquidity issues)
            if minutes_to_close < self.avoid_last_minutes:
                return FilterResult(
                    passed=False,
                    filter_name="TIME_BASED",
                    reason=f"Too close to market close ({minutes_to_close:.1f} min), avoiding liquidity issues",
                    confidence_adjustment=-0.2
                )
            
            # Optimal trading hours (10:00 AM to 2:30 PM) get confidence boost
            optimal_start = time(10, 0)
            optimal_end = time(14, 30)
            
            if optimal_start <= current_time_only <= optimal_end:
                return FilterResult(
                    passed=True,
                    filter_name="TIME_BASED",
                    reason="Signal generated during optimal trading hours",
                    confidence_adjustment=0.1
                )
            else:
                return FilterResult(
                    passed=True,
                    filter_name="TIME_BASED",
                    reason="Signal timing acceptable",
                    confidence_adjustment=0.0
                )
                
        except Exception as e:
            logger.error(f"Time-based filter failed: {e}")
            return FilterResult(
                passed=True,  # Don't block on filter errors
                filter_name="TIME_BASED",
                reason=f"Filter error: {e}",
                confidence_adjustment=-0.05
            )
    
    def volatility_regime_filter(self, signal: Dict[str, Any], 
                                market_data: Dict[str, Any]) -> FilterResult:
        """
        Filter based on current volatility regime
        
        Args:
            signal: Trading signal
            market_data: Current market data including VIX
        """
        try:
            current_vix = market_data.get('india_vix', 20)
            signal_type = signal.get('strategy_type', 'DIRECTIONAL')
            
            # High volatility regime (VIX > 30)
            if current_vix > 30:
                if signal_type in ['STRADDLE', 'STRANGLE']:
                    # Volatility strategies benefit from high VIX
                    return FilterResult(
                        passed=True,
                        filter_name="VOLATILITY_REGIME",
                        reason=f"High VIX ({current_vix}) favorable for volatility strategies",
                        confidence_adjustment=0.25
                    )
                else:
                    # Directional strategies suffer in high volatility
                    return FilterResult(
                        passed=True,  # Don't block, but reduce confidence
                        filter_name="VOLATILITY_REGIME",
                        reason=f"High VIX ({current_vix}) challenging for directional strategies",
                        confidence_adjustment=-0.15
                    )
            
            # Low volatility regime (VIX < 15)
            elif current_vix < 15:
                if signal_type in ['IRON_CONDOR', 'BUTTERFLY']:
                    # Range-bound strategies benefit from low VIX
                    return FilterResult(
                        passed=True,
                        filter_name="VOLATILITY_REGIME",
                        reason=f"Low VIX ({current_vix}) favorable for range-bound strategies",
                        confidence_adjustment=0.2
                    )
                else:
                    return FilterResult(
                        passed=True,
                        filter_name="VOLATILITY_REGIME",
                        reason=f"Normal VIX ({current_vix}) - neutral impact",
                        confidence_adjustment=0.0
                    )
            
            # Normal volatility regime
            else:
                return FilterResult(
                    passed=True,
                    filter_name="VOLATILITY_REGIME",
                    reason=f"Normal VIX ({current_vix}) - favorable for most strategies",
                    confidence_adjustment=0.05
                )
                
        except Exception as e:
            logger.error(f"Volatility regime filter failed: {e}")
            return FilterResult(
                passed=True,
                filter_name="VOLATILITY_REGIME",
                reason=f"Filter error: {e}",
                confidence_adjustment=0.0
            )
    
    def comprehensive_signal_filter(self, signal: Dict[str, Any], 
                                   options_data: List[Dict[str, Any]],
                                   market_data: Dict[str, Any],
                                   current_time: datetime = None) -> EnhancedSignal:
        """
        Apply comprehensive signal filtering
        
        Args:
            signal: Original trading signal
            options_data: Current options chain data
            market_data: Current market data
            current_time: Current timestamp
        """
        try:
            filter_results = []
            
            # Apply all filters
            filters = [
                self.volume_confirmation_filter(signal, options_data),
                self.spread_validation_filter(signal, options_data),
                self.liquidity_check_filter(signal, options_data),
                self.time_based_filter(signal, current_time),
                self.volatility_regime_filter(signal, market_data)
            ]
            
            filter_results.extend(filters)
            
            # Calculate final confidence and quality score
            original_confidence = signal.get('confidence_score', 50.0)
            confidence_adjustments = sum(f.confidence_adjustment for f in filter_results)
            final_confidence = max(0, min(100, original_confidence + confidence_adjustments * 10))
            
            # Calculate quality score (0-100)
            passed_filters = sum(1 for f in filter_results if f.passed)
            total_filters = len(filter_results)
            base_quality = (passed_filters / total_filters) * 100
            
            # Adjust quality based on confidence adjustments
            quality_adjustment = sum(f.confidence_adjustment for f in filter_results) * 20
            quality_score = max(0, min(100, base_quality + quality_adjustment))
            
            # Determine recommendation
            critical_filters_passed = all(f.passed for f in filter_results 
                                        if f.filter_name in ['VOLUME_CONFIRMATION', 'LIQUIDITY_CHECK', 'TIME_BASED'])
            
            if not critical_filters_passed:
                recommendation = 'SKIP'
            elif final_confidence >= 60 and quality_score >= 70:
                recommendation = 'EXECUTE'
            elif final_confidence >= 40 and quality_score >= 50:
                recommendation = 'WAIT'  # Wait for better conditions
            else:
                recommendation = 'SKIP'
            
            return EnhancedSignal(
                original_signal=signal,
                filter_results=filter_results,
                final_confidence=final_confidence,
                quality_score=quality_score,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Comprehensive signal filtering failed: {e}")
            return EnhancedSignal(
                original_signal=signal,
                filter_results=[FilterResult(False, "ERROR", str(e), -0.5)],
                final_confidence=0,
                quality_score=0,
                recommendation='SKIP'
            )
    
    def get_filter_summary(self, enhanced_signal: EnhancedSignal) -> Dict[str, Any]:
        """Get summary of filter results"""
        try:
            summary = {
                'original_confidence': enhanced_signal.original_signal.get('confidence_score', 0),
                'final_confidence': enhanced_signal.final_confidence,
                'quality_score': enhanced_signal.quality_score,
                'recommendation': enhanced_signal.recommendation,
                'filters_passed': sum(1 for f in enhanced_signal.filter_results if f.passed),
                'total_filters': len(enhanced_signal.filter_results),
                'filter_details': []
            }
            
            for filter_result in enhanced_signal.filter_results:
                summary['filter_details'].append({
                    'name': filter_result.filter_name,
                    'passed': filter_result.passed,
                    'reason': filter_result.reason,
                    'confidence_impact': filter_result.confidence_adjustment
                })
            
            return summary
            
        except Exception as e:
            logger.error(f"Filter summary generation failed: {e}")
            return {'error': str(e)}
