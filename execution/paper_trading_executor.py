"""
Paper Trading Executor for VLR_AI Trading System
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import os
from pathlib import Path

logger = logging.getLogger('trading_system.paper_trading_executor')

class PaperTradingExecutor:
    """Paper trading executor for safe strategy testing"""
    
    def __init__(self, settings):
        self.settings = settings
        self.virtual_balance = float(getattr(settings, 'PAPER_TRADING_BALANCE', 1000000))  # 10L default
        self.initial_balance = self.virtual_balance
        self.paper_positions = {}
        self.trade_history = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Paper trading data file
        self.data_dir = Path("data_storage/paper_trading")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_file = self.data_dir / "paper_trading_data.json"
        
        # Load existing data if available
        self._load_paper_trading_data()
        
        logger.info(f"[PAPER] Paper Trading Executor initialized with virtual balance: Rs.{self.virtual_balance:,.2f}")
    
    def _load_paper_trading_data(self):
        """Load existing paper trading data"""
        try:
            if self.data_file.exists():
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.virtual_balance = data.get('virtual_balance', self.virtual_balance)
                    self.paper_positions = data.get('paper_positions', {})
                    self.trade_history = data.get('trade_history', [])
                    self.daily_pnl = data.get('daily_pnl', 0.0)
                    self.total_pnl = data.get('total_pnl', 0.0)
                    self.total_trades = data.get('total_trades', 0)
                    self.winning_trades = data.get('winning_trades', 0)
                    self.losing_trades = data.get('losing_trades', 0)
                    logger.info("[PAPER] Loaded existing paper trading data")
        except Exception as e:
            logger.warning(f"[PAPER] Could not load paper trading data: {e}")
    
    def _save_paper_trading_data(self):
        """Save paper trading data to file"""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_positions = {}
            for pos_id, pos in self.paper_positions.items():
                serializable_pos = pos.copy()
                if 'signal' in serializable_pos and 'timestamp' in serializable_pos['signal']:
                    if hasattr(serializable_pos['signal']['timestamp'], 'isoformat'):
                        serializable_pos['signal']['timestamp'] = serializable_pos['signal']['timestamp'].isoformat()
                serializable_positions[pos_id] = serializable_pos
            
            data = {
                'virtual_balance': self.virtual_balance,
                'paper_positions': serializable_positions,
                'trade_history': self.trade_history[-100:],  # Keep last 100 trades
                'daily_pnl': self.daily_pnl,
                'total_pnl': self.total_pnl,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[PAPER] Could not save paper trading data: {e}")
    
    async def execute_trade(self, signal: Dict) -> Dict:
        """Execute trade - wrapper for execute_paper_trade"""
        return await self.execute_paper_trade(signal)
    
    async def monitor_positions(self) -> Dict:
        """Monitor positions - wrapper for monitor_paper_positions"""
        return await self.monitor_paper_positions()
    
    async def execute_paper_trade(self, signal: Dict) -> Dict:
        """Execute a paper trade based on signal"""
        try:
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            
            if position_size <= 0:
                return {
                    'status': 'rejected',
                    'reason': 'Insufficient virtual balance',
                    'signal': signal
                }
            
            
            # Calculate total cost
            total_cost = position_size * entry_price
            
            if total_cost > self.virtual_balance:
                return {
                    'status': 'rejected',
                    'reason': 'Insufficient virtual balance for trade',
                    'signal': signal,
                    'required': total_cost,
                    'available': self.virtual_balance
                }
            
            # Get instrument name first
            instrument = signal.get('instrument', signal.get('symbol', 'NIFTY'))
            
            # Create position
            position_id = f"{instrument}_{signal.get('strike', 'SPOT')}_{signal.get('option_type', 'EQUITY')}_{datetime.now().strftime('%H%M%S')}"
            
            position = {
                'id': position_id,
                'signal': signal,
                'entry_price': entry_price,
                'quantity': position_size,
                'entry_time': datetime.now().isoformat(),
                'total_cost': total_cost,
                'stop_loss': signal.get('stop_loss'),
                'target': signal.get('target'),
                'status': 'open',
                'current_pnl': 0.0
            }
            
            # Update virtual balance
            self.virtual_balance -= total_cost
            
            # Store position
            self.paper_positions[position_id] = position
            
            # Record trade
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'action': signal.get('action', 'BUY'),
                'instrument': instrument,
                'quantity': position_size,
                'price': entry_price,
                'total_cost': total_cost,
                'position_id': position_id,
                'signal_confidence': signal.get('confidence', 0)
            }
            
            self.trade_history.append(trade_record)
            self.total_trades += 1
            
            # Save data
            self._save_paper_trading_data()
            
            logger.info(f"[PAPER] Paper trade executed: {instrument} @ Rs.{entry_price:.2f}, Qty: {position_size}")
            
            return {
                'status': 'success',
                'position_id': position_id,
                'entry_price': entry_price,
                'quantity': position_size,
                'total_cost': total_cost,
                'signal': signal,
                'virtual_balance': self.virtual_balance
            }
            
        except Exception as e:
            logger.error(f"[PAPER] Paper trade execution failed: {e}")
            return {
                'status': 'error',
                'reason': str(e),
                'signal': signal
            }
    
    def _calculate_position_size(self, signal: Dict) -> int:
        """Calculate position size for paper trading"""
        try:
            # Use 2% of virtual balance per trade (configurable)
            risk_per_trade = self.virtual_balance * 0.02
            
            # Get lot size based on instrument
            instrument = signal.get('instrument', signal.get('symbol', 'NIFTY'))
            lot_size = self._get_lot_size(instrument)
            
            # Calculate maximum lots we can afford
            cost_per_lot = lot_size * estimated_price
            
            if cost_per_lot <= 0:
                return 0
            
            max_lots = int(risk_per_trade / cost_per_lot)
            
            # Ensure at least 1 lot if we have enough balance
            if max_lots == 0 and self.virtual_balance > cost_per_lot:
                max_lots = 1
            
            return max_lots * lot_size
            
        except Exception as e:
            logger.error(f"[PAPER] Position size calculation failed: {e}")
            return 0
    
    def _get_lot_size(self, instrument: str) -> int:
        """Get lot size for instrument"""
        lot_sizes = {
            'NIFTY': getattr(self.settings, 'NIFTY_LOT_SIZE', 75),
            'BANKNIFTY': getattr(self.settings, 'BANKNIFTY_LOT_SIZE', 30),
            'FINNIFTY': getattr(self.settings, 'FINNIFTY_LOT_SIZE', 65),
            'MIDCPNIFTY': getattr(self.settings, 'MIDCPNIFTY_LOT_SIZE', 100)
        }
        
        for key in lot_sizes:
            if key in instrument.upper():
                return lot_sizes[key]
        
        return 1  # Default for equity
    
    def _get_realistic_entry_price(self, signal: Dict[str, Any]) -> float:
        """Get realistic entry price from signal data"""
        try:
            # Use actual market price from signal
            price = signal.get('current_price', signal.get('ltp', signal.get('price', 0)))
            
            if price and price > 0:
                return float(price)
            
            # If no price available, log error and return 0
            logger.error(f"[PAPER] No valid price found in signal: {signal}")
            return 0.0
                
        except Exception as e:
            logger.error(f"[PAPER] Price extraction failed: {e}")
            return 0.0
    
    async def monitor_paper_positions(self) -> Dict:
        try:
            actions_taken = []
            total_unrealized_pnl = 0.0
            
            for position_id, position in list(self.paper_positions.items()):
                if position['status'] != 'open':
                    continue
                
                
                # Calculate P&L
                action = position['signal'].get('action', 'BUY')
                if action == 'BUY':
                    pnl = (current_price - position['entry_price']) * position['quantity']
                else:
                    pnl = (position['entry_price'] - current_price) * position['quantity']
                
                position['current_pnl'] = pnl
                total_unrealized_pnl += pnl
                
                # Check stop loss and target
                should_exit = False
                exit_reason = ""
                
                if position['stop_loss'] and current_price <= position['stop_loss']:
                    should_exit = True
                    exit_reason = "Stop Loss Hit"
                elif position['target'] and current_price >= position['target']:
                    should_exit = True
                    exit_reason = "Target Achieved"
                
                # Exit position if needed
                if should_exit:
                    exit_result = await self._exit_paper_position(position_id, current_price, exit_reason)
                    actions_taken.append(exit_result)
            
            # Save updated data
            self._save_paper_trading_data()
            
            return {
                'total_unrealized_pnl': total_unrealized_pnl,
                'active_positions': len([p for p in self.paper_positions.values() if p['status'] == 'open']),
                'actions_taken': actions_taken
            }
            
        except Exception as e:
            logger.error(f"[PAPER] Position monitoring failed: {e}")
            return {'total_unrealized_pnl': 0.0, 'active_positions': 0, 'actions_taken': []}
    
    def _get_current_market_price(self, position: Dict[str, Any]) -> float:
        """Get current market price for position (should integrate with real market data)"""
        try:
            # In a real implementation, this would fetch current market price
            # For now, return entry price (no P&L simulation)
            # This should be connected to your market data provider
            
            instrument = position['signal'].get('instrument', position['signal'].get('symbol', 'NIFTY'))
            
            # TODO: Integrate with market data provider to get real current price
            # For now, return entry price to avoid simulation
            return position['entry_price']
            
        except Exception as e:
            logger.error(f"[PAPER] Failed to get current market price: {e}")
            return position['entry_price']
    
    async def _exit_paper_position(self, position_id: str, exit_price: float, reason: str) -> Dict:
        """Exit a paper position"""
        try:
            position = self.paper_positions[position_id]
            
            # Calculate final P&L
            action = position['signal'].get('action', 'BUY')
            if action == 'BUY':
                pnl = (exit_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - exit_price) * position['quantity']
            
            # Update virtual balance
            exit_value = position['quantity'] * exit_price
            self.virtual_balance += exit_value
            
            # Update position
            position['status'] = 'closed'
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now().isoformat()
            position['exit_reason'] = reason
            position['final_pnl'] = pnl
            
            # Update statistics
            self.total_pnl += pnl
            self.daily_pnl += pnl
            
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Record exit trade
            exit_record = {
                'timestamp': datetime.now().isoformat(),
                'action': 'EXIT',
                'instrument': position['signal']['instrument'],
                'quantity': position['quantity'],
                'price': exit_price,
                'pnl': pnl,
                'reason': reason,
                'position_id': position_id
            }
            
            self.trade_history.append(exit_record)
            
            logger.info(f"[PAPER] Position closed: {position_id}, P&L: Rs.{pnl:.2f}, Reason: {reason}")
            
            return {
                'position': position_id,
                'action': 'closed',
                'pnl': pnl,
                'reason': reason,
                'exit_price': exit_price
            }
            
        except Exception as e:
            logger.error(f"[PAPER] Position exit failed: {e}")
            return {'position': position_id, 'action': 'error', 'reason': str(e)}
    
    def get_paper_trading_stats(self) -> Dict:
        """Get comprehensive paper trading statistics"""
        try:
            total_return = ((self.virtual_balance + sum(p.get('current_pnl', 0) for p in self.paper_positions.values() if p['status'] == 'open')) / self.initial_balance - 1) * 100
            
            win_rate = (self.winning_trades / max(self.winning_trades + self.losing_trades, 1)) * 100
            
            return {
                'virtual_balance': self.virtual_balance,
                'initial_balance': self.initial_balance,
                'total_pnl': self.total_pnl,
                'daily_pnl': self.daily_pnl,
                'total_return_pct': total_return,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate_pct': win_rate,
                'active_positions': len([p for p in self.paper_positions.values() if p['status'] == 'open']),
                'unrealized_pnl': sum(p.get('current_pnl', 0) for p in self.paper_positions.values() if p['status'] == 'open')
            }
            
        except Exception as e:
            logger.error(f"[PAPER] Stats calculation failed: {e}")
            return {}
    
    async def reset_paper_trading(self):
        """Reset paper trading data"""
        try:
            self.virtual_balance = self.initial_balance
            self.paper_positions = {}
            self.trade_history = []
            self.daily_pnl = 0.0
            self.total_pnl = 0.0
            self.total_trades = 0
            self.winning_trades = 0
            self.losing_trades = 0
            
            self._save_paper_trading_data()
            logger.info("[PAPER] Paper trading data reset successfully")
            
        except Exception as e:
            logger.error(f"[PAPER] Reset failed: {e}")