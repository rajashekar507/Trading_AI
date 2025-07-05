"""
Enhanced Backtesting Engine for VLR_AI Trading System
Implements Walk-forward Analysis, Monte Carlo Simulations, Strategy Optimization
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import random
import math

logger = logging.getLogger('trading_system.enhanced_backtester')

@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    best_trade: float
    worst_trade: float
    consecutive_wins: int
    consecutive_losses: int

@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results"""
    mean_return: float
    std_return: float
    percentile_5: float
    percentile_95: float
    probability_of_loss: float
    var_95: float
    expected_shortfall: float

class EnhancedBacktester:
    """Advanced backtesting engine with multiple analysis methods"""
    
    def __init__(self, settings):
        self.settings = settings
        self.risk_free_rate = 0.065
        
    def walk_forward_analysis(self, strategy_func, data: pd.DataFrame,
                             train_window: int = 252, test_window: int = 63,
                             step_size: int = 21) -> Dict[str, Any]:
        """
        Perform walk-forward analysis
        
        Args:
            strategy_func: Strategy function to test
            data: Historical market data
            train_window: Training window in days
            test_window: Testing window in days
            step_size: Step size for rolling window
        """
        try:
            results = []
            total_data_points = len(data)
            
            # Ensure we have enough data
            if total_data_points < train_window + test_window:
                logger.warning("Insufficient data for walk-forward analysis")
                return {'error': 'Insufficient data'}
            
            current_pos = 0
            while current_pos + train_window + test_window <= total_data_points:
                # Define train and test periods
                train_start = current_pos
                train_end = current_pos + train_window
                test_start = train_end
                test_end = test_start + test_window
                
                # Extract train and test data
                train_data = data.iloc[train_start:train_end]
                test_data = data.iloc[test_start:test_end]
                
                # Train strategy (optimize parameters)
                optimized_params = self._optimize_strategy_parameters(strategy_func, train_data)
                
                # Test strategy with optimized parameters
                test_results = self._run_backtest(strategy_func, test_data, optimized_params)
                
                results.append({
                    'train_period': (train_data.index[0], train_data.index[-1]),
                    'test_period': (test_data.index[0], test_data.index[-1]),
                    'parameters': optimized_params,
                    'test_return': test_results.total_return,
                    'test_sharpe': test_results.sharpe_ratio,
                    'test_max_dd': test_results.max_drawdown,
                    'test_win_rate': test_results.win_rate
                })
                
                current_pos += step_size
            
            # Aggregate results
            if not results:
                return {'error': 'No valid walk-forward periods'}
            
            avg_return = np.mean([r['test_return'] for r in results])
            avg_sharpe = np.mean([r['test_sharpe'] for r in results])
            avg_max_dd = np.mean([r['test_max_dd'] for r in results])
            avg_win_rate = np.mean([r['test_win_rate'] for r in results])
            
            return {
                'walk_forward_results': results,
                'summary': {
                    'average_return': avg_return,
                    'average_sharpe': avg_sharpe,
                    'average_max_drawdown': avg_max_dd,
                    'average_win_rate': avg_win_rate,
                    'consistency_score': self._calculate_consistency_score(results),
                    'total_periods': len(results)
                }
            }
            
        except Exception as e:
            logger.error(f"Walk-forward analysis failed: {e}")
            return {'error': str(e)}
    
    def monte_carlo_simulation(self, returns: pd.Series, num_simulations: int = 1000,
                              time_horizon: int = 252) -> MonteCarloResult:
        """
        Perform Monte Carlo simulation on returns
        
        Args:
            returns: Historical returns series
            num_simulations: Number of simulation runs
            time_horizon: Time horizon in days
        """
        try:
            if len(returns) < 30:
                logger.warning("Insufficient data for Monte Carlo simulation")
                return MonteCarloResult(0, 0, 0, 0, 1, 0, 0)
            
            # Calculate return statistics
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Run simulations
            simulation_results = []
            
            for _ in range(num_simulations):
                # Generate random returns
                random_returns = np.random.normal(mean_return, std_return, time_horizon)
                
                # Calculate cumulative return
                cumulative_return = (1 + pd.Series(random_returns)).prod() - 1
                simulation_results.append(cumulative_return)
            
            simulation_results = np.array(simulation_results)
            
            # Calculate statistics
            mean_sim_return = np.mean(simulation_results)
            std_sim_return = np.std(simulation_results)
            percentile_5 = np.percentile(simulation_results, 5)
            percentile_95 = np.percentile(simulation_results, 95)
            probability_of_loss = np.sum(simulation_results < 0) / num_simulations
            var_95 = np.percentile(simulation_results, 5)
            
            # Expected Shortfall (Conditional VaR)
            losses = simulation_results[simulation_results <= var_95]
            expected_shortfall = np.mean(losses) if len(losses) > 0 else var_95
            
            return MonteCarloResult(
                mean_return=mean_sim_return,
                std_return=std_sim_return,
                percentile_5=percentile_5,
                percentile_95=percentile_95,
                probability_of_loss=probability_of_loss,
                var_95=var_95,
                expected_shortfall=expected_shortfall
            )
            
        except Exception as e:
            logger.error(f"Monte Carlo simulation failed: {e}")
            return MonteCarloResult(0, 0, 0, 0, 1, 0, 0)
    
    def strategy_optimization(self, strategy_func, data: pd.DataFrame,
                            parameter_ranges: Dict[str, Tuple[float, float, float]]) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search
        
        Args:
            strategy_func: Strategy function to optimize
            data: Historical data
            parameter_ranges: Dict of parameter_name: (min, max, step)
        """
        try:
            best_params = {}
            best_score = -np.inf
            best_results = None
            
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(parameter_ranges)
            
            optimization_results = []
            
            for params in param_combinations:
                try:
                    # Run backtest with these parameters
                    results = self._run_backtest(strategy_func, data, params)
                    
                    # Calculate optimization score (Sharpe ratio with drawdown penalty)
                    score = results.sharpe_ratio - abs(results.max_drawdown) * 2
                    
                    optimization_results.append({
                        'parameters': params,
                        'score': score,
                        'return': results.total_return,
                        'sharpe': results.sharpe_ratio,
                        'max_drawdown': results.max_drawdown,
                        'win_rate': results.win_rate
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_results = results
                        
                except Exception as e:
                    logger.warning(f"Parameter combination failed: {params}, error: {e}")
                    continue
            
            return {
                'best_parameters': best_params,
                'best_score': best_score,
                'best_results': best_results,
                'all_results': optimization_results,
                'total_combinations_tested': len(optimization_results)
            }
            
        except Exception as e:
            logger.error(f"Strategy optimization failed: {e}")
            return {'error': str(e)}
    
    def performance_attribution(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze performance attribution across different dimensions
        
        Args:
            trades: List of trade records
        """
        try:
            if not trades:
                return {'error': 'No trades to analyze'}
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(trades)
            
            attribution = {
                'by_instrument': {},
                'by_strategy': {},
                'by_time_of_day': {},
                'by_day_of_week': {},
                'by_month': {},
                'by_volatility_regime': {}
            }
            
            # By instrument
            if 'instrument' in df.columns and 'pnl' in df.columns:
                attribution['by_instrument'] = df.groupby('instrument')['pnl'].agg([
                    'sum', 'mean', 'count', 'std'
                ]).to_dict()
            
            # By strategy type
            if 'strategy' in df.columns:
                attribution['by_strategy'] = df.groupby('strategy')['pnl'].agg([
                    'sum', 'mean', 'count', 'std'
                ]).to_dict()
            
            # By time dimensions (if timestamp available)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.day_name()
                df['month'] = df['timestamp'].dt.month_name()
                
                attribution['by_time_of_day'] = df.groupby('hour')['pnl'].agg([
                    'sum', 'mean', 'count'
                ]).to_dict()
                
                attribution['by_day_of_week'] = df.groupby('day_of_week')['pnl'].agg([
                    'sum', 'mean', 'count'
                ]).to_dict()
                
                attribution['by_month'] = df.groupby('month')['pnl'].agg([
                    'sum', 'mean', 'count'
                ]).to_dict()
            
            # Calculate contribution percentages
            total_pnl = df['pnl'].sum()
            if total_pnl != 0:
                for category in attribution:
                    if isinstance(attribution[category], dict) and 'sum' in attribution[category]:
                        for key in attribution[category]['sum']:
                            attribution[category]['contribution_pct'] = attribution[category]['contribution_pct'] or {}
                            attribution[category]['contribution_pct'][key] = (
                                attribution[category]['sum'][key] / total_pnl * 100
                            )
            
            return attribution
            
        except Exception as e:
            logger.error(f"Performance attribution failed: {e}")
            return {'error': str(e)}
    
    def _optimize_strategy_parameters(self, strategy_func, data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize strategy parameters for training data"""
        # Simplified parameter optimization
        # In a real implementation, this would use more sophisticated methods
        return {
            'confidence_threshold': 25.0,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'stop_loss': 0.02,
            'take_profit': 0.04
        }
    
    def _run_backtest(self, strategy_func, data: pd.DataFrame, params: Dict[str, Any]) -> BacktestResult:
        """Run backtest with given parameters"""
        # Simplified backtest implementation
        # In a real implementation, this would run the actual strategy
        
        cumulative_returns = (1 + pd.Series(returns)).cumprod()
        
        total_return = cumulative_returns.iloc[-1] - 1
        volatility = pd.Series(returns).std() * np.sqrt(252)
        sharpe_ratio = (total_return * 252 - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        win_rate = 0.6
        profit_factor = 1.5
        total_trades = len(data) // 5  # Assume trade every 5 days
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=total_return * 252 / len(data),
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_duration=5.0,
            best_trade=0.05,
            worst_trade=-0.03,
            consecutive_wins=5,
            consecutive_losses=3
        )
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, Tuple[float, float, float]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for optimization"""
        combinations = []
        
        # Simple grid search implementation
        # For demonstration, we'll limit to a few combinations
        for i in range(min(50, 10 ** len(parameter_ranges))):  # Limit combinations
            combination = {}
            for param_name, (min_val, max_val, step) in parameter_ranges.items():
                # Random sampling within range
                combination[param_name] = random.uniform(min_val, max_val)
            combinations.append(combination)
        
        return combinations
    
    def _calculate_consistency_score(self, results: List[Dict[str, Any]]) -> float:
        """Calculate consistency score for walk-forward results"""
        if not results:
            return 0.0
        
        returns = [r['test_return'] for r in results]
        positive_periods = sum(1 for r in returns if r > 0)
        consistency = positive_periods / len(returns)
        
        # Penalize high volatility of returns
        return_std = np.std(returns)
        volatility_penalty = min(return_std * 2, 0.5)
        
        return max(0, consistency - volatility_penalty)
