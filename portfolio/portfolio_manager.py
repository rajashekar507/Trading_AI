"""
Integrated Portfolio Management System
Combines all optimization strategies and provides unified interface
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

from portfolio.portfolio_optimizer import ModernPortfolioTheory, BlackLittermanModel, RiskParityOptimizer

logger = logging.getLogger('trading_system.portfolio_manager')

class PortfolioManager:
    """Unified Portfolio Management System"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.mpt_optimizer = ModernPortfolioTheory(settings)
        self.bl_optimizer = BlackLittermanModel(settings)
        self.rp_optimizer = RiskParityOptimizer(settings)
        
        # Portfolio configuration
        self.rebalance_frequency = getattr(settings, 'REBALANCE_FREQUENCY_DAYS', 7)  # Weekly
        self.max_position_size = getattr(settings, 'MAX_POSITION_SIZE', 0.3)  # 30% max
        self.min_position_size = getattr(settings, 'MIN_POSITION_SIZE', 0.05)  # 5% min
        
        # Current portfolio state
        self.current_portfolio = {}
        self.last_rebalance_date = None
        self.portfolio_history = []
        
        logger.info("[PORTFOLIO_MANAGER] Integrated portfolio management system initialized")
    
    def optimize_portfolio(self, market_data: Dict, strategy: str = 'adaptive', 
                          risk_tolerance: str = 'moderate') -> Dict:
        """Main portfolio optimization method"""
        try:
            logger.info(f"[PORTFOLIO] Starting optimization with strategy: {strategy}")
            
            # Extract and prepare data
            returns_data = self._prepare_returns_data(market_data)
            if returns_data.empty:
                return {'status': 'failed', 'message': 'Insufficient market data for optimization'}
            
            # Run optimization based on strategy
            if strategy == 'mpt':
                result = self._run_mpt_optimization(returns_data, risk_tolerance)
            elif strategy == 'black_litterman':
                result = self._run_bl_optimization(returns_data, market_data)
            elif strategy == 'risk_parity':
                result = self._run_rp_optimization(returns_data)
            elif strategy == 'adaptive':
                result = self._run_adaptive_optimization(returns_data, market_data, risk_tolerance)
            else:
                return {'status': 'failed', 'message': f'Unknown strategy: {strategy}'}
            
            # Post-process and validate results
            if result.get('status') == 'success':
                result = self._post_process_optimization(result, market_data)
                
                # Update portfolio state
                self._update_portfolio_state(result)
                
                logger.info(f"[PORTFOLIO] Optimization completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"[PORTFOLIO] Optimization failed: {e}")
            return {'status': 'failed', 'message': str(e)}
    
    def _prepare_returns_data(self, market_data: Dict) -> pd.DataFrame:
        """Prepare returns data from market data"""
        try:
            returns_data = pd.DataFrame()
            
            # Extract spot data
            spot_data = market_data.get('spot_data', {})
            if spot_data.get('status') == 'success':
                # Create synthetic returns data for available instruments
                instruments = ['NIFTY', 'BANKNIFTY']
                
                for instrument in instruments:
                    if instrument in spot_data:
                        current_price = spot_data[instrument]['ltp']
                        
                        # Generate synthetic historical returns (in production, use real data)
                        dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
                        
                        # Simulate returns with realistic parameters
                        if instrument == 'NIFTY':
                            daily_returns = np.random.normal(0.0005, 0.015, 60)  # 0.05% mean, 1.5% vol
                        else:  # BANKNIFTY
                            daily_returns = np.random.normal(0.0006, 0.018, 60)  # 0.06% mean, 1.8% vol
                        
                        returns_data[instrument] = daily_returns
                
                returns_data.index = dates[:len(returns_data)]
            
            # Add VIX as a defensive asset (inverse correlation)
            if len(returns_data.columns) > 0:
                vix_returns = -returns_data.mean(axis=1) * 0.5 + np.random.normal(0, 0.02, len(returns_data))
                returns_data['VIX_HEDGE'] = vix_returns
            
            logger.info(f"[PORTFOLIO] Prepared returns data for {len(returns_data.columns)} assets over {len(returns_data)} periods")
            return returns_data
            
        except Exception as e:
            logger.error(f"[PORTFOLIO] Returns data preparation failed: {e}")
            return pd.DataFrame()
    
    def _run_mpt_optimization(self, returns_data: pd.DataFrame, risk_tolerance: str) -> Dict:
        """Run Modern Portfolio Theory optimization"""
        try:
            result = self.mpt_optimizer.optimize_portfolio(returns_data, risk_tolerance=risk_tolerance)
            if result.get('status') == 'success':
                result['optimization_method'] = 'Modern Portfolio Theory'
                result['strategy_details'] = {
                    'objective': 'Maximize risk-adjusted returns',
                    'constraints': 'Long-only, fully invested',
                    'risk_tolerance': risk_tolerance
                }
            return result
            
        except Exception as e:
            logger.error(f"[MPT] Optimization failed: {e}")
            return {'status': 'failed', 'message': str(e)}
    
    def _run_bl_optimization(self, returns_data: pd.DataFrame, market_data: Dict) -> Dict:
        """Run Black-Litterman optimization"""
        try:
            # Create market cap data (simplified)
            market_caps = {
                'NIFTY': 100000000,  # 100M
                'BANKNIFTY': 80000000,  # 80M
                'VIX_HEDGE': 20000000   # 20M
            }
            
            # Create investor views based on current market conditions
            views = self._generate_market_views(market_data, returns_data.columns)
            
            result = self.bl_optimizer.optimize_with_views(returns_data, market_caps, views)
            if result.get('status') == 'success':
                result['optimization_method'] = 'Black-Litterman'
                result['strategy_details'] = {
                    'objective': 'Incorporate market views into equilibrium',
                    'views_count': len(views),
                    'market_caps': market_caps
                }
            return result
            
        except Exception as e:
            logger.error(f"[BLACK_LITTERMAN] Optimization failed: {e}")
            return {'status': 'failed', 'message': str(e)}
    
    def _run_rp_optimization(self, returns_data: pd.DataFrame) -> Dict:
        """Run Risk Parity optimization"""
        try:
            result = self.rp_optimizer.optimize_risk_parity(returns_data, target_vol=0.15)
            if result.get('status') == 'success':
                result['optimization_method'] = 'Risk Parity'
                result['strategy_details'] = {
                    'objective': 'Equal risk contribution from all assets',
                    'target_volatility': '15%',
                    'leverage_applied': result.get('leverage_factor', 1.0)
                }
            return result
            
        except Exception as e:
            logger.error(f"[RISK_PARITY] Optimization failed: {e}")
            return {'status': 'failed', 'message': str(e)}
    
    def _run_adaptive_optimization(self, returns_data: pd.DataFrame, market_data: Dict, 
                                 risk_tolerance: str) -> Dict:
        """Run adaptive optimization combining multiple strategies"""
        try:
            logger.info("[ADAPTIVE] Running multi-strategy optimization")
            
            # Run all optimization methods
            mpt_result = self._run_mpt_optimization(returns_data, risk_tolerance)
            bl_result = self._run_bl_optimization(returns_data, market_data)
            rp_result = self._run_rp_optimization(returns_data)
            
            # Combine results based on market conditions
            market_regime = self._assess_market_regime(market_data, returns_data)
            
            # Weight strategies based on market regime
            if market_regime == 'HIGH_VOLATILITY':
                # Favor risk parity in high volatility
                strategy_weights = {'mpt': 0.2, 'black_litterman': 0.3, 'risk_parity': 0.5}
            elif market_regime == 'LOW_VOLATILITY':
                # Favor MPT in low volatility
                strategy_weights = {'mpt': 0.5, 'black_litterman': 0.3, 'risk_parity': 0.2}
            else:  # NORMAL
                # Balanced approach
                strategy_weights = {'mpt': 0.4, 'black_litterman': 0.4, 'risk_parity': 0.2}
            
            # Combine portfolio weights
            combined_weights = self._combine_portfolio_weights([
                (mpt_result, strategy_weights['mpt']),
                (bl_result, strategy_weights['black_litterman']),
                (rp_result, strategy_weights['risk_parity'])
            ])
            
            # Calculate combined metrics
            combined_metrics = self._calculate_combined_metrics(combined_weights, returns_data)
            
            return {
                'status': 'success',
                'optimization_method': 'Adaptive Multi-Strategy',
                'weights': combined_weights,
                'portfolio_metrics': combined_metrics,
                'strategy_details': {
                    'market_regime': market_regime,
                    'strategy_weights': strategy_weights,
                    'component_strategies': ['MPT', 'Black-Litterman', 'Risk Parity']
                },
                'individual_results': {
                    'mpt': mpt_result,
                    'black_litterman': bl_result,
                    'risk_parity': rp_result
                },
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Optimization failed: {e}")
            return {'status': 'failed', 'message': str(e)}
    
    def _generate_market_views(self, market_data: Dict, assets: pd.Index) -> List[Dict]:
        """Generate investor views based on current market conditions"""
        try:
            views = []
            
            # VIX-based view
            vix_data = market_data.get('vix_data', {})
            if vix_data.get('status') == 'success':
                current_vix = vix_data.get('current_vix', 20)
                
                if current_vix > 25:
                    # High VIX - expect mean reversion
                    views.append({
                        'type': 'absolute',
                        'asset': 'NIFTY',
                        'expected_return': 0.12,  # 12% expected return
                        'confidence': 0.7
                    })
                elif current_vix < 15:
                    # Low VIX - expect continued stability
                    views.append({
                        'type': 'absolute',
                        'asset': 'BANKNIFTY',
                        'expected_return': 0.15,  # 15% expected return
                        'confidence': 0.6
                    })
            
            # FII/DII flow-based view
            fii_dii_data = market_data.get('fii_dii_data', {})
            if fii_dii_data.get('status') == 'success':
                fii_flow = fii_dii_data.get('fii_flow_trend', 'neutral')
                
                if fii_flow == 'positive':
                    views.append({
                        'type': 'relative',
                        'asset': 'NIFTY',
                        'relative_to': 'VIX_HEDGE',
                        'expected_return': 0.05,  # 5% outperformance
                        'confidence': 0.8
                    })
            
            logger.info(f"[VIEWS] Generated {len(views)} market views")
            return views
            
        except Exception as e:
            logger.error(f"[VIEWS] Generation failed: {e}")
            return []
    
    def _assess_market_regime(self, market_data: Dict, returns_data: pd.DataFrame) -> str:
        """Assess current market regime"""
        try:
            # Calculate recent volatility
            recent_vol = returns_data.tail(20).std().mean() * np.sqrt(252)
            
            # VIX level
            vix_data = market_data.get('vix_data', {})
            current_vix = vix_data.get('current_vix', 20) if vix_data.get('status') == 'success' else 20
            
            # Determine regime
            if recent_vol > 0.25 or current_vix > 25:
                return 'HIGH_VOLATILITY'
            elif recent_vol < 0.15 or current_vix < 15:
                return 'LOW_VOLATILITY'
            else:
                return 'NORMAL'
                
        except Exception:
            return 'NORMAL'
    
    def _combine_portfolio_weights(self, strategy_results: List[Tuple[Dict, float]]) -> Dict:
        """Combine weights from multiple strategies"""
        try:
            combined_weights = {}
            total_weight = 0
            
            for result, strategy_weight in strategy_results:
                if result.get('status') == 'success':
                    weights = result.get('weights', {})
                    if isinstance(weights, dict):
                        portfolio_weights = weights
                    else:
                        # Handle different result formats
                        portfolio_weights = result.get('recommended_portfolio', {}).get('weights', {})
                    
                    for asset, weight in portfolio_weights.items():
                        if asset not in combined_weights:
                            combined_weights[asset] = 0
                        combined_weights[asset] += weight * strategy_weight
                    
                    total_weight += strategy_weight
            
            # Normalize weights
            if total_weight > 0:
                for asset in combined_weights:
                    combined_weights[asset] /= total_weight
            
            # Apply position size constraints
            combined_weights = self._apply_position_constraints(combined_weights)
            
            return combined_weights
            
        except Exception as e:
            logger.error(f"[COMBINE_WEIGHTS] Failed: {e}")
            return {}
    
    def _apply_position_constraints(self, weights: Dict) -> Dict:
        """Apply position size constraints"""
        try:
            constrained_weights = weights.copy()
            
            # Apply maximum position size
            for asset in constrained_weights:
                if constrained_weights[asset] > self.max_position_size:
                    constrained_weights[asset] = self.max_position_size
            
            # Remove positions below minimum size
            constrained_weights = {
                asset: weight for asset, weight in constrained_weights.items() 
                if weight >= self.min_position_size
            }
            
            # Renormalize
            total_weight = sum(constrained_weights.values())
            if total_weight > 0:
                for asset in constrained_weights:
                    constrained_weights[asset] /= total_weight
            
            return constrained_weights
            
        except Exception as e:
            logger.error(f"[POSITION_CONSTRAINTS] Failed: {e}")
            return weights
    
    def _calculate_combined_metrics(self, weights: Dict, returns_data: pd.DataFrame) -> Dict:
        """Calculate metrics for combined portfolio"""
        try:
            if not weights or returns_data.empty:
                return {}
            
            # Convert weights to array
            weight_array = np.array([weights.get(col, 0) for col in returns_data.columns])
            
            # Calculate portfolio metrics
            expected_returns = returns_data.mean() * 252
            cov_matrix = returns_data.cov() * 252
            
            portfolio_return = np.dot(weight_array, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weight_array.T, np.dot(cov_matrix, weight_array)))
            sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            
            return {
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'max_weight': max(weights.values()) if weights else 0,
                'min_weight': min(weights.values()) if weights else 0,
                'num_positions': len([w for w in weights.values() if w > 0.01]),
                'concentration_ratio': sum(w**2 for w in weights.values())
            }
            
        except Exception as e:
            logger.error(f"[COMBINED_METRICS] Calculation failed: {e}")
            return {}
    
    def _post_process_optimization(self, result: Dict, market_data: Dict) -> Dict:
        """Post-process optimization results"""
        try:
            # Add implementation recommendations
            result['implementation'] = self._generate_implementation_plan(result, market_data)
            
            # Add risk analysis
            result['risk_analysis'] = self._analyze_portfolio_risks(result)
            
            # Add rebalancing recommendations
            result['rebalancing'] = self._generate_rebalancing_plan(result)
            
            return result
            
        except Exception as e:
            logger.error(f"[POST_PROCESS] Failed: {e}")
            return result
    
    def _generate_implementation_plan(self, result: Dict, market_data: Dict) -> Dict:
        """Generate implementation plan for portfolio"""
        try:
            weights = result.get('weights', {})
            
            # Calculate trade sizes
            portfolio_value = getattr(self.settings, 'PORTFOLIO_VALUE', 1000000)  # 10L default
            
            trades = []
            for asset, target_weight in weights.items():
                current_weight = self.current_portfolio.get(asset, 0)
                weight_diff = target_weight - current_weight
                
                if abs(weight_diff) > 0.01:  # 1% threshold
                    trade_value = weight_diff * portfolio_value
                    trades.append({
                        'asset': asset,
                        'action': 'BUY' if weight_diff > 0 else 'SELL',
                        'target_weight': target_weight,
                        'current_weight': current_weight,
                        'weight_change': weight_diff,
                        'trade_value': abs(trade_value),
                        'priority': 'HIGH' if abs(weight_diff) > 0.05 else 'MEDIUM'
                    })
            
            return {
                'total_trades': len(trades),
                'trades': sorted(trades, key=lambda x: abs(x['weight_change']), reverse=True),
                'estimated_turnover': sum(abs(t['weight_change']) for t in trades) / 2,
                'implementation_cost': self._estimate_implementation_cost(trades)
            }
            
        except Exception as e:
            logger.error(f"[IMPLEMENTATION_PLAN] Failed: {e}")
            return {}
    
    def _analyze_portfolio_risks(self, result: Dict) -> Dict:
        """Analyze portfolio risks"""
        try:
            weights = result.get('weights', {})
            portfolio_metrics = result.get('portfolio_metrics', {})
            
            risks = {
                'concentration_risk': 'LOW',
                'volatility_risk': 'MODERATE',
                'correlation_risk': 'LOW',
                'liquidity_risk': 'LOW'
            }
            
            # Concentration risk
            max_weight = max(weights.values()) if weights else 0
            if max_weight > 0.4:
                risks['concentration_risk'] = 'HIGH'
            elif max_weight > 0.3:
                risks['concentration_risk'] = 'MODERATE'
            
            # Volatility risk
            portfolio_vol = portfolio_metrics.get('volatility', 0)
            if portfolio_vol > 0.25:
                risks['volatility_risk'] = 'HIGH'
            elif portfolio_vol < 0.15:
                risks['volatility_risk'] = 'LOW'
            
            return risks
            
        except Exception as e:
            logger.error(f"[RISK_ANALYSIS] Failed: {e}")
            return {}
    
    def _generate_rebalancing_plan(self, result: Dict) -> Dict:
        """Generate rebalancing recommendations"""
        try:
            return {
                'next_rebalance_date': (datetime.now() + timedelta(days=self.rebalance_frequency)).strftime('%Y-%m-%d'),
                'rebalance_frequency': f'{self.rebalance_frequency} days',
                'rebalance_threshold': '5% weight deviation',
                'monitoring_frequency': 'Daily'
            }
            
        except Exception as e:
            logger.error(f"[REBALANCING_PLAN] Failed: {e}")
            return {}
    
    def _estimate_implementation_cost(self, trades: List[Dict]) -> Dict:
        """Estimate implementation costs"""
        try:
            total_value = sum(trade['trade_value'] for trade in trades)
            
            # Simplified cost estimation
            brokerage_rate = 0.0005  # 0.05%
            impact_rate = 0.001     # 0.1% market impact
            
            brokerage_cost = total_value * brokerage_rate
            market_impact = total_value * impact_rate
            total_cost = brokerage_cost + market_impact
            
            return {
                'brokerage_cost': brokerage_cost,
                'market_impact': market_impact,
                'total_cost': total_cost,
                'cost_percentage': (total_cost / total_value * 100) if total_value > 0 else 0
            }
            
        except Exception:
            return {'total_cost': 0, 'cost_percentage': 0}
    
    def _update_portfolio_state(self, result: Dict):
        """Update current portfolio state"""
        try:
            if result.get('status') == 'success':
                weights = result.get('weights', {})
                if isinstance(weights, dict):
                    self.current_portfolio = weights
                else:
                    # Handle different result formats
                    self.current_portfolio = result.get('recommended_portfolio', {}).get('weights', {})
                
                self.last_rebalance_date = datetime.now()
                
                # Add to history
                self.portfolio_history.append({
                    'timestamp': datetime.now(),
                    'weights': self.current_portfolio.copy(),
                    'optimization_method': result.get('optimization_method', 'Unknown'),
                    'portfolio_metrics': result.get('portfolio_metrics', {})
                })
                
                # Keep only last 30 records
                if len(self.portfolio_history) > 30:
                    self.portfolio_history = self.portfolio_history[-30:]
                
                logger.info(f"[PORTFOLIO_STATE] Updated with {len(self.current_portfolio)} positions")
                
        except Exception as e:
            logger.error(f"[PORTFOLIO_STATE] Update failed: {e}")
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        try:
            return {
                'current_portfolio': self.current_portfolio,
                'last_rebalance_date': self.last_rebalance_date.isoformat() if self.last_rebalance_date else None,
                'days_since_rebalance': (datetime.now() - self.last_rebalance_date).days if self.last_rebalance_date else None,
                'next_rebalance_due': self.last_rebalance_date + timedelta(days=self.rebalance_frequency) if self.last_rebalance_date else None,
                'portfolio_history_count': len(self.portfolio_history),
                'total_positions': len([w for w in self.current_portfolio.values() if w > 0.01]),
                'max_position_size': max(self.current_portfolio.values()) if self.current_portfolio else 0,
                'portfolio_concentration': sum(w**2 for w in self.current_portfolio.values()) if self.current_portfolio else 0
            }
            
        except Exception as e:
            logger.error(f"[PORTFOLIO_STATUS] Failed: {e}")
            return {'error': str(e)}