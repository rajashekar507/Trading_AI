"""
Advanced Portfolio Optimization
Implements Modern Portfolio Theory, Black-Litterman, and Risk Parity
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
from scipy.linalg import inv
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('trading_system.portfolio_optimizer')

class ModernPortfolioTheory:
    """Modern Portfolio Theory Implementation"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.risk_free_rate = 0.06  # 6% annual risk-free rate
        self.trading_days = 252
        
        logger.info("[MPT] Modern Portfolio Theory optimizer initialized")
    
    def optimize_portfolio(self, returns_data: pd.DataFrame, target_return: Optional[float] = None,
                          risk_tolerance: str = 'moderate') -> Dict:
        """Optimize portfolio using MPT"""
        try:
            if returns_data.empty or len(returns_data.columns) < 2:
                return {'status': 'failed', 'message': 'Insufficient data for optimization'}
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_data.mean() * self.trading_days
            cov_matrix = returns_data.cov() * self.trading_days
            
            n_assets = len(expected_returns)
            
            # Risk tolerance mapping
            risk_multipliers = {
                'conservative': 0.5,
                'moderate': 1.0,
                'aggressive': 2.0
            }
            risk_multiplier = risk_multipliers.get(risk_tolerance, 1.0)
            
            # Optimize for different objectives
            optimizations = {}
            
            # 1. Maximum Sharpe Ratio Portfolio
            max_sharpe_weights = self._optimize_max_sharpe(expected_returns, cov_matrix)
            optimizations['max_sharpe'] = self._calculate_portfolio_metrics(
                max_sharpe_weights, expected_returns, cov_matrix, 'Maximum Sharpe Ratio'
            )
            
            # 2. Minimum Variance Portfolio
            min_var_weights = self._optimize_min_variance(cov_matrix)
            optimizations['min_variance'] = self._calculate_portfolio_metrics(
                min_var_weights, expected_returns, cov_matrix, 'Minimum Variance'
            )
            
            # 3. Target Return Portfolio (if specified)
            if target_return:
                target_weights = self._optimize_target_return(expected_returns, cov_matrix, target_return)
                if target_weights is not None:
                    optimizations['target_return'] = self._calculate_portfolio_metrics(
                        target_weights, expected_returns, cov_matrix, f'Target Return ({target_return:.2%})'
                    )
            
            # 4. Risk Parity Portfolio
            risk_parity_weights = self._optimize_risk_parity(cov_matrix)
            optimizations['risk_parity'] = self._calculate_portfolio_metrics(
                risk_parity_weights, expected_returns, cov_matrix, 'Risk Parity'
            )
            
            # 5. Efficient Frontier
            efficient_frontier = self._calculate_efficient_frontier(expected_returns, cov_matrix)
            
            # Select recommended portfolio based on risk tolerance
            recommended_portfolio = self._select_recommended_portfolio(optimizations, risk_tolerance)
            
            return {
                'status': 'success',
                'recommended_portfolio': recommended_portfolio,
                'all_optimizations': optimizations,
                'efficient_frontier': efficient_frontier,
                'market_analysis': {
                    'expected_returns': expected_returns.to_dict(),
                    'volatilities': np.sqrt(np.diag(cov_matrix)).tolist(),
                    'correlation_matrix': returns_data.corr().to_dict(),
                    'sharpe_ratios': ((expected_returns - self.risk_free_rate) / np.sqrt(np.diag(cov_matrix))).to_dict()
                },
                'risk_tolerance': risk_tolerance,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"[MPT] Portfolio optimization failed: {e}")
            return {'status': 'failed', 'message': str(e)}
    
    def _optimize_max_sharpe(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Optimize for maximum Sharpe ratio"""
        try:
            n_assets = len(expected_returns)
            
            def negative_sharpe(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -(portfolio_return - self.risk_free_rate) / portfolio_vol
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(n_assets))
            initial_guess = np.array([1/n_assets] * n_assets)
            
            result = minimize(negative_sharpe, initial_guess, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            return result.x if result.success else initial_guess
            
        except Exception as e:
            logger.error(f"[MAX_SHARPE] Optimization failed: {e}")
            return np.array([1/len(expected_returns)] * len(expected_returns))
    
    def _optimize_min_variance(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Optimize for minimum variance"""
        try:
            n_assets = len(cov_matrix)
            
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(n_assets))
            initial_guess = np.array([1/n_assets] * n_assets)
            
            result = minimize(portfolio_variance, initial_guess, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            return result.x if result.success else initial_guess
            
        except Exception as e:
            logger.error(f"[MIN_VARIANCE] Optimization failed: {e}")
            return np.array([1/len(cov_matrix)] * len(cov_matrix))
    
    def _optimize_target_return(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, 
                              target_return: float) -> Optional[np.ndarray]:
        """Optimize for target return"""
        try:
            n_assets = len(expected_returns)
            
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(x * expected_returns) - target_return}
            ]
            bounds = tuple((0, 1) for _ in range(n_assets))
            initial_guess = np.array([1/n_assets] * n_assets)
            
            result = minimize(portfolio_variance, initial_guess, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            return result.x if result.success else None
            
        except Exception as e:
            logger.error(f"[TARGET_RETURN] Optimization failed: {e}")
            return None
    
    def _optimize_risk_parity(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Optimize for risk parity (equal risk contribution)"""
        try:
            n_assets = len(cov_matrix)
            
            def risk_parity_objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                target_contrib = portfolio_vol / n_assets
                return np.sum((contrib - target_contrib) ** 2)
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0.01, 0.99) for _ in range(n_assets))  # Avoid zero weights
            initial_guess = np.array([1/n_assets] * n_assets)
            
            result = minimize(risk_parity_objective, initial_guess, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            return result.x if result.success else initial_guess
            
        except Exception as e:
            logger.error(f"[RISK_PARITY] Optimization failed: {e}")
            return np.array([1/len(cov_matrix)] * len(cov_matrix))
    
    def _calculate_efficient_frontier(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, 
                                    num_points: int = 50) -> List[Dict]:
        """Calculate efficient frontier points"""
        try:
            min_ret = expected_returns.min()
            max_ret = expected_returns.max()
            target_returns = np.linspace(min_ret, max_ret, num_points)
            
            efficient_portfolios = []
            
            for target_ret in target_returns:
                weights = self._optimize_target_return(expected_returns, cov_matrix, target_ret)
                if weights is not None:
                    portfolio_return = np.sum(weights * expected_returns)
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
                    
                    efficient_portfolios.append({
                        'return': portfolio_return,
                        'volatility': portfolio_vol,
                        'sharpe_ratio': sharpe_ratio,
                        'weights': weights.tolist()
                    })
            
            return efficient_portfolios
            
        except Exception as e:
            logger.error(f"[EFFICIENT_FRONTIER] Calculation failed: {e}")
            return []
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray, expected_returns: pd.Series, 
                                   cov_matrix: pd.DataFrame, strategy_name: str) -> Dict:
        """Calculate portfolio performance metrics"""
        try:
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            # Risk contributions
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            risk_contrib_pct = risk_contrib / np.sum(risk_contrib) * 100
            
            return {
                'strategy_name': strategy_name,
                'weights': dict(zip(expected_returns.index, weights)),
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'risk_contributions': dict(zip(expected_returns.index, risk_contrib_pct)),
                'max_weight': np.max(weights),
                'min_weight': np.min(weights),
                'concentration_ratio': np.sum(weights**2),  # Herfindahl index
                'diversification_ratio': self._calculate_diversification_ratio(weights, expected_returns, cov_matrix)
            }
            
        except Exception as e:
            logger.error(f"[PORTFOLIO_METRICS] Calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_diversification_ratio(self, weights: np.ndarray, expected_returns: pd.Series, 
                                       cov_matrix: pd.DataFrame) -> float:
        """Calculate diversification ratio"""
        try:
            individual_vols = np.sqrt(np.diag(cov_matrix))
            weighted_avg_vol = np.sum(weights * individual_vols)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            return weighted_avg_vol / portfolio_vol
            
        except Exception:
            return 1.0
    
    def _select_recommended_portfolio(self, optimizations: Dict, risk_tolerance: str) -> Dict:
        """Select recommended portfolio based on risk tolerance"""
        try:
            if risk_tolerance == 'conservative':
                # Prefer minimum variance or risk parity
                if 'min_variance' in optimizations:
                    return optimizations['min_variance']
                elif 'risk_parity' in optimizations:
                    return optimizations['risk_parity']
            
            elif risk_tolerance == 'aggressive':
                # Prefer maximum Sharpe ratio
                if 'max_sharpe' in optimizations:
                    return optimizations['max_sharpe']
            
            else:  # moderate
                # Balance between Sharpe and risk parity
                if 'max_sharpe' in optimizations and 'risk_parity' in optimizations:
                    max_sharpe = optimizations['max_sharpe']
                    risk_parity = optimizations['risk_parity']
                    
                    # Blend the two strategies
                    blended_weights = {}
                    for asset in max_sharpe['weights']:
                        blended_weights[asset] = (max_sharpe['weights'][asset] * 0.6 + 
                                                risk_parity['weights'][asset] * 0.4)
                    
                    return {
                        'strategy_name': 'Blended (60% Max Sharpe + 40% Risk Parity)',
                        'weights': blended_weights,
                        'expected_return': (max_sharpe['expected_return'] * 0.6 + 
                                          risk_parity['expected_return'] * 0.4),
                        'volatility': (max_sharpe['volatility'] * 0.6 + 
                                     risk_parity['volatility'] * 0.4),
                        'sharpe_ratio': ((max_sharpe['expected_return'] * 0.6 + 
                                        risk_parity['expected_return'] * 0.4) - self.risk_free_rate) / 
                                       (max_sharpe['volatility'] * 0.6 + risk_parity['volatility'] * 0.4)
                    }
            
            # Fallback to first available optimization
            return list(optimizations.values())[0] if optimizations else {}
            
        except Exception as e:
            logger.error(f"[PORTFOLIO_SELECTION] Failed: {e}")
            return list(optimizations.values())[0] if optimizations else {}

class BlackLittermanModel:
    """Black-Litterman Portfolio Optimization"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.risk_aversion = 3.0  # Typical risk aversion parameter
        self.tau = 0.025  # Scaling factor for uncertainty of prior
        
        logger.info("[BLACK_LITTERMAN] Model initialized")
    
    def optimize_with_views(self, returns_data: pd.DataFrame, market_caps: Dict, 
                           views: List[Dict]) -> Dict:
        """Optimize portfolio using Black-Litterman with investor views"""
        try:
            if returns_data.empty:
                return {'status': 'failed', 'message': 'No returns data provided'}
            
            # Calculate market-implied returns (equilibrium returns)
            cov_matrix = returns_data.cov() * 252
            market_weights = self._calculate_market_weights(market_caps, returns_data.columns)
            implied_returns = self._calculate_implied_returns(market_weights, cov_matrix)
            
            # Process investor views
            if not views:
                # No views - use equilibrium returns
                bl_returns = implied_returns
                bl_cov = cov_matrix
            else:
                # Incorporate views using Black-Litterman
                P, Q, Omega = self._process_views(views, returns_data.columns)
                bl_returns, bl_cov = self._black_litterman_optimization(
                    implied_returns, cov_matrix, P, Q, Omega
                )
            
            # Optimize portfolio with Black-Litterman inputs
            optimal_weights = self._optimize_bl_portfolio(bl_returns, bl_cov)
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_bl_metrics(optimal_weights, bl_returns, bl_cov, implied_returns)
            
            return {
                'status': 'success',
                'optimal_weights': dict(zip(returns_data.columns, optimal_weights)),
                'bl_returns': dict(zip(returns_data.columns, bl_returns)),
                'implied_returns': dict(zip(returns_data.columns, implied_returns)),
                'portfolio_metrics': portfolio_metrics,
                'views_incorporated': len(views),
                'market_weights': market_weights,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"[BLACK_LITTERMAN] Optimization failed: {e}")
            return {'status': 'failed', 'message': str(e)}
    
    def _calculate_market_weights(self, market_caps: Dict, assets: pd.Index) -> np.ndarray:
        """Calculate market capitalization weights"""
        try:
            weights = []
            total_cap = sum(market_caps.get(asset, 1.0) for asset in assets)
            
            for asset in assets:
                weight = market_caps.get(asset, 1.0) / total_cap
                weights.append(weight)
            
            return np.array(weights)
            
        except Exception:
            # Equal weights if market cap data not available
            return np.array([1/len(assets)] * len(assets))
    
    def _calculate_implied_returns(self, market_weights: np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Calculate market-implied equilibrium returns"""
        try:
            # Implied returns = risk_aversion * covariance_matrix * market_weights
            implied_returns = self.risk_aversion * np.dot(cov_matrix.values, market_weights)
            return implied_returns
            
        except Exception as e:
            logger.error(f"[IMPLIED_RETURNS] Calculation failed: {e}")
            return np.zeros(len(market_weights))
    
    def _process_views(self, views: List[Dict], assets: pd.Index) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process investor views into P, Q, Omega matrices"""
        try:
            n_assets = len(assets)
            n_views = len(views)
            
            P = np.zeros((n_views, n_assets))  # Picking matrix
            Q = np.zeros(n_views)  # View returns
            Omega = np.eye(n_views) * 0.01  # View uncertainty (1% default)
            
            asset_to_idx = {asset: idx for idx, asset in enumerate(assets)}
            
            for i, view in enumerate(views):
                view_type = view.get('type', 'absolute')
                asset = view.get('asset')
                expected_return = view.get('expected_return', 0)
                confidence = view.get('confidence', 0.5)  # 0 to 1
                
                if asset in asset_to_idx:
                    asset_idx = asset_to_idx[asset]
                    
                    if view_type == 'absolute':
                        # Absolute view: Asset will return X%
                        P[i, asset_idx] = 1.0
                        Q[i] = expected_return
                    
                    elif view_type == 'relative' and 'relative_to' in view:
                        # Relative view: Asset A will outperform Asset B by X%
                        relative_asset = view['relative_to']
                        if relative_asset in asset_to_idx:
                            relative_idx = asset_to_idx[relative_asset]
                            P[i, asset_idx] = 1.0
                            P[i, relative_idx] = -1.0
                            Q[i] = expected_return
                    
                    # Adjust uncertainty based on confidence
                    Omega[i, i] = (1 - confidence) * 0.05  # Higher confidence = lower uncertainty
            
            return P, Q, Omega
            
        except Exception as e:
            logger.error(f"[PROCESS_VIEWS] Failed: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def _black_litterman_optimization(self, implied_returns: np.ndarray, cov_matrix: pd.DataFrame,
                                    P: np.ndarray, Q: np.ndarray, Omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Black-Litterman formula"""
        try:
            if P.size == 0:
                return implied_returns, cov_matrix.values
            
            # Black-Litterman formula
            tau_cov = self.tau * cov_matrix.values
            
            # Calculate new expected returns
            M1 = inv(tau_cov)
            M2 = np.dot(P.T, np.dot(inv(Omega), P))
            M3 = np.dot(inv(tau_cov), implied_returns)
            M4 = np.dot(P.T, np.dot(inv(Omega), Q))
            
            bl_returns = np.dot(inv(M1 + M2), M3 + M4)
            
            # Calculate new covariance matrix
            bl_cov = inv(M1 + M2)
            
            return bl_returns, bl_cov
            
        except Exception as e:
            logger.error(f"[BL_OPTIMIZATION] Failed: {e}")
            return implied_returns, cov_matrix.values
    
    def _optimize_bl_portfolio(self, bl_returns: np.ndarray, bl_cov: np.ndarray) -> np.ndarray:
        """Optimize portfolio using Black-Litterman inputs"""
        try:
            # Mean-variance optimization with Black-Litterman inputs
            inv_cov = inv(bl_cov)
            ones = np.ones((len(bl_returns), 1))
            
            # Calculate optimal weights
            numerator = np.dot(inv_cov, bl_returns)
            denominator = np.dot(ones.T, np.dot(inv_cov, ones))[0, 0]
            
            optimal_weights = numerator / denominator
            
            # Ensure weights sum to 1 and are non-negative
            optimal_weights = np.maximum(optimal_weights, 0)
            optimal_weights = optimal_weights / np.sum(optimal_weights)
            
            return optimal_weights
            
        except Exception as e:
            logger.error(f"[BL_PORTFOLIO] Optimization failed: {e}")
            return np.array([1/len(bl_returns)] * len(bl_returns))
    
    def _calculate_bl_metrics(self, weights: np.ndarray, bl_returns: np.ndarray, 
                            bl_cov: np.ndarray, implied_returns: np.ndarray) -> Dict:
        """Calculate Black-Litterman portfolio metrics"""
        try:
            portfolio_return = np.dot(weights, bl_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(bl_cov, weights)))
            sharpe_ratio = portfolio_return / portfolio_vol
            
            # Compare with equilibrium portfolio
            eq_weights = np.array([1/len(weights)] * len(weights))
            eq_return = np.dot(eq_weights, implied_returns)
            eq_vol = np.sqrt(np.dot(eq_weights.T, np.dot(bl_cov, eq_weights)))
            
            return {
                'portfolio_return': portfolio_return,
                'portfolio_volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'equilibrium_return': eq_return,
                'equilibrium_volatility': eq_vol,
                'return_improvement': portfolio_return - eq_return,
                'risk_reduction': eq_vol - portfolio_vol,
                'information_ratio': (portfolio_return - eq_return) / abs(portfolio_vol - eq_vol) if abs(portfolio_vol - eq_vol) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"[BL_METRICS] Calculation failed: {e}")
            return {'error': str(e)}

class RiskParityOptimizer:
    """Risk Parity Portfolio Optimization"""
    
    def __init__(self, settings=None):
        self.settings = settings
        logger.info("[RISK_PARITY] Optimizer initialized")
    
    def optimize_risk_parity(self, returns_data: pd.DataFrame, target_vol: float = 0.15) -> Dict:
        """Optimize portfolio for equal risk contribution"""
        try:
            if returns_data.empty:
                return {'status': 'failed', 'message': 'No returns data provided'}
            
            cov_matrix = returns_data.cov() * 252
            n_assets = len(returns_data.columns)
            
            # Optimize for risk parity
            weights = self._solve_risk_parity(cov_matrix)
            
            # Scale to target volatility
            current_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            leverage = target_vol / current_vol
            scaled_weights = weights * leverage
            
            # Calculate metrics
            expected_returns = returns_data.mean() * 252
            portfolio_metrics = self._calculate_rp_metrics(scaled_weights, expected_returns, cov_matrix)
            
            # Risk contributions
            risk_contributions = self._calculate_risk_contributions(scaled_weights, cov_matrix)
            
            return {
                'status': 'success',
                'weights': dict(zip(returns_data.columns, scaled_weights)),
                'risk_contributions': dict(zip(returns_data.columns, risk_contributions)),
                'portfolio_metrics': portfolio_metrics,
                'leverage_factor': leverage,
                'target_volatility': target_vol,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"[RISK_PARITY] Optimization failed: {e}")
            return {'status': 'failed', 'message': str(e)}
    
    def _solve_risk_parity(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Solve for risk parity weights"""
        try:
            n_assets = len(cov_matrix)
            
            def risk_parity_objective(weights):
                # Calculate risk contributions
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                
                # Target equal risk contribution
                target_contrib = portfolio_vol / n_assets
                
                # Sum of squared deviations from equal risk contribution
                return np.sum((contrib - target_contrib) ** 2)
            
            # Constraints and bounds
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0.01, 0.99) for _ in range(n_assets))
            initial_guess = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(risk_parity_objective, initial_guess, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            return result.x if result.success else initial_guess
            
        except Exception as e:
            logger.error(f"[RP_SOLVE] Failed: {e}")
            return np.array([1/len(cov_matrix)] * len(cov_matrix))
    
    def _calculate_risk_contributions(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Calculate risk contributions for each asset"""
        try:
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            
            # Convert to percentages
            return (risk_contrib / np.sum(risk_contrib)) * 100
            
        except Exception as e:
            logger.error(f"[RISK_CONTRIB] Calculation failed: {e}")
            return np.zeros(len(weights))
    
    def _calculate_rp_metrics(self, weights: np.ndarray, expected_returns: pd.Series, 
                            cov_matrix: pd.DataFrame) -> Dict:
        """Calculate risk parity portfolio metrics"""
        try:
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Risk-adjusted metrics
            sharpe_ratio = portfolio_return / portfolio_vol
            max_drawdown = self._estimate_max_drawdown(portfolio_vol)
            calmar_ratio = portfolio_return / max_drawdown if max_drawdown > 0 else 0
            
            return {
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'estimated_max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'diversification_ratio': self._calculate_diversification_ratio(weights, cov_matrix)
            }
            
        except Exception as e:
            logger.error(f"[RP_METRICS] Calculation failed: {e}")
            return {'error': str(e)}
    
    def _estimate_max_drawdown(self, volatility: float) -> float:
        """Estimate maximum drawdown based on volatility"""
        # Simple approximation: max drawdown ≈ 2.5 * volatility for normal distribution
        return 2.5 * volatility
    
    def _calculate_diversification_ratio(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
        """Calculate diversification ratio"""
        try:
            individual_vols = np.sqrt(np.diag(cov_matrix))
            weighted_avg_vol = np.dot(weights, individual_vols)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            return weighted_avg_vol / portfolio_vol
            
        except Exception:
            return 1.0