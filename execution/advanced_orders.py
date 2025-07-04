"""
Advanced Order Types for Institutional Trading
Implements Iceberg, TWAP, VWAP, Bracket, and Conditional Orders
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from enum import Enum
import time

logger = logging.getLogger('trading_system.advanced_orders')

class OrderType(Enum):
    """Advanced order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"
    BRACKET = "BRACKET"
    CONDITIONAL = "CONDITIONAL"
    TRAILING_STOP = "TRAILING_STOP"

class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"

class AdvancedOrderManager:
    """Advanced Order Management System"""
    
    def __init__(self, settings=None, kite_client=None):
        self.settings = settings
        self.kite_client = kite_client
        self.active_orders = {}
        self.order_history = []
        self.execution_tasks = {}
        
        # TWAP/VWAP parameters
        self.default_twap_duration = 300  # 5 minutes
        self.default_slice_size = 0.1  # 10% of total quantity per slice
        self.min_slice_interval = 5  # 5 seconds between slices
        
        # Iceberg parameters
        self.default_iceberg_size = 0.2  # 20% visible quantity
        self.iceberg_refresh_threshold = 0.1  # Refresh when 10% remaining
        
        logger.info("[ORDERS] Advanced Order Manager initialized")
    
    async def place_iceberg_order(self, symbol: str, quantity: int, price: float, 
                                side: str, visible_qty_pct: float = 0.2) -> str:
        """
        Place Iceberg Order - Large order split into smaller visible portions
        """
        try:
            order_id = f"ICE_{int(time.time())}_{symbol}"
            visible_qty = max(1, int(quantity * visible_qty_pct))
            
            order_data = {
                'order_id': order_id,
                'type': OrderType.ICEBERG,
                'symbol': symbol,
                'total_quantity': quantity,
                'visible_quantity': visible_qty,
                'price': price,
                'side': side,
                'filled_quantity': 0,
                'remaining_quantity': quantity,
                'status': OrderStatus.PENDING,
                'created_time': datetime.now(),
                'child_orders': []
            }
            
            self.active_orders[order_id] = order_data
            
            # Start iceberg execution
            task = asyncio.create_task(self._execute_iceberg_order(order_id))
            self.execution_tasks[order_id] = task
            
            logger.info(f"[ICEBERG] Order placed: {symbol} {quantity}@{price} (Visible: {visible_qty})")
            return order_id
            
        except Exception as e:
            logger.error(f"[ICEBERG] Order placement failed: {e}")
            return None
    
    async def place_twap_order(self, symbol: str, quantity: int, side: str,
                             duration_seconds: int = 300, slice_pct: float = 0.1) -> str:
        """
        Place TWAP Order - Time Weighted Average Price execution
        """
        try:
            order_id = f"TWAP_{int(time.time())}_{symbol}"
            slice_qty = max(1, int(quantity * slice_pct))
            num_slices = max(1, quantity // slice_qty)
            slice_interval = duration_seconds / num_slices
            
            order_data = {
                'order_id': order_id,
                'type': OrderType.TWAP,
                'symbol': symbol,
                'total_quantity': quantity,
                'slice_quantity': slice_qty,
                'side': side,
                'duration': duration_seconds,
                'slice_interval': slice_interval,
                'filled_quantity': 0,
                'remaining_quantity': quantity,
                'status': OrderStatus.PENDING,
                'created_time': datetime.now(),
                'slices_executed': 0,
                'total_slices': num_slices,
                'child_orders': []
            }
            
            self.active_orders[order_id] = order_data
            
            # Start TWAP execution
            task = asyncio.create_task(self._execute_twap_order(order_id))
            self.execution_tasks[order_id] = task
            
            logger.info(f"[TWAP] Order placed: {symbol} {quantity} over {duration_seconds}s ({num_slices} slices)")
            return order_id
            
        except Exception as e:
            logger.error(f"[TWAP] Order placement failed: {e}")
            return None
    
    async def place_vwap_order(self, symbol: str, quantity: int, side: str,
                             duration_seconds: int = 300) -> str:
        """
        Place VWAP Order - Volume Weighted Average Price execution
        """
        try:
            order_id = f"VWAP_{int(time.time())}_{symbol}"
            
            order_data = {
                'order_id': order_id,
                'type': OrderType.VWAP,
                'symbol': symbol,
                'total_quantity': quantity,
                'side': side,
                'duration': duration_seconds,
                'filled_quantity': 0,
                'remaining_quantity': quantity,
                'status': OrderStatus.PENDING,
                'created_time': datetime.now(),
                'volume_profile': [],
                'child_orders': []
            }
            
            self.active_orders[order_id] = order_data
            
            # Start VWAP execution
            task = asyncio.create_task(self._execute_vwap_order(order_id))
            self.execution_tasks[order_id] = task
            
            logger.info(f"[VWAP] Order placed: {symbol} {quantity} over {duration_seconds}s")
            return order_id
            
        except Exception as e:
            logger.error(f"[VWAP] Order placement failed: {e}")
            return None
    
    async def place_bracket_order(self, symbol: str, quantity: int, entry_price: float,
                                stop_loss: float, target: float, side: str,
                                trailing_stop: bool = False) -> str:
        """
        Place Bracket Order - Entry with automatic stop-loss and target
        """
        try:
            order_id = f"BRK_{int(time.time())}_{symbol}"
            
            order_data = {
                'order_id': order_id,
                'type': OrderType.BRACKET,
                'symbol': symbol,
                'quantity': quantity,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target': target,
                'side': side,
                'trailing_stop': trailing_stop,
                'status': OrderStatus.PENDING,
                'created_time': datetime.now(),
                'entry_filled': False,
                'exit_order_placed': False,
                'child_orders': []
            }
            
            self.active_orders[order_id] = order_data
            
            # Start bracket execution
            task = asyncio.create_task(self._execute_bracket_order(order_id))
            self.execution_tasks[order_id] = task
            
            logger.info(f"[BRACKET] Order placed: {symbol} {quantity}@{entry_price} SL:{stop_loss} TGT:{target}")
            return order_id
            
        except Exception as e:
            logger.error(f"[BRACKET] Order placement failed: {e}")
            return None
    
    async def place_conditional_order(self, symbol: str, quantity: int, price: float,
                                    side: str, condition_func: Callable, 
                                    condition_params: Dict) -> str:
        """
        Place Conditional Order - Execute when custom condition is met
        """
        try:
            order_id = f"COND_{int(time.time())}_{symbol}"
            
            order_data = {
                'order_id': order_id,
                'type': OrderType.CONDITIONAL,
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'side': side,
                'condition_func': condition_func,
                'condition_params': condition_params,
                'status': OrderStatus.PENDING,
                'created_time': datetime.now(),
                'condition_met': False
            }
            
            self.active_orders[order_id] = order_data
            
            # Start conditional monitoring
            task = asyncio.create_task(self._monitor_conditional_order(order_id))
            self.execution_tasks[order_id] = task
            
            logger.info(f"[CONDITIONAL] Order placed: {symbol} {quantity}@{price} with custom condition")
            return order_id
            
        except Exception as e:
            logger.error(f"[CONDITIONAL] Order placement failed: {e}")
            return None
    
    async def _execute_iceberg_order(self, order_id: str):
        """Execute iceberg order logic"""
        try:
            order = self.active_orders[order_id]
            
            while order['remaining_quantity'] > 0 and order['status'] == OrderStatus.PENDING:
                # Calculate current visible quantity
                visible_qty = min(order['visible_quantity'], order['remaining_quantity'])
                
                # Place child order
                child_order_id = await self._place_market_order(
                    order['symbol'], visible_qty, order['price'], order['side']
                )
                
                if child_order_id:
                    order['child_orders'].append(child_order_id)
                    
                    # Simulate partial fill (in real implementation, monitor actual fills)
                    filled_qty = visible_qty  # Assume full fill for simulation
                    order['filled_quantity'] += filled_qty
                    order['remaining_quantity'] -= filled_qty
                    
                    logger.info(f"[ICEBERG] {order_id} slice filled: {filled_qty} (Remaining: {order['remaining_quantity']})")
                    
                    if order['remaining_quantity'] > 0:
                        # Wait before next slice
                        await asyncio.sleep(self.min_slice_interval)
                else:
                    logger.error(f"[ICEBERG] {order_id} child order failed")
                    break
            
            # Update final status
            if order['remaining_quantity'] == 0:
                order['status'] = OrderStatus.COMPLETED
                logger.info(f"[ICEBERG] {order_id} completed successfully")
            else:
                order['status'] = OrderStatus.PARTIAL
                logger.warning(f"[ICEBERG] {order_id} partially completed")
                
        except Exception as e:
            logger.error(f"[ICEBERG] Execution failed for {order_id}: {e}")
            self.active_orders[order_id]['status'] = OrderStatus.FAILED
    
    async def _execute_twap_order(self, order_id: str):
        """Execute TWAP order logic"""
        try:
            order = self.active_orders[order_id]
            start_time = datetime.now()
            
            while (order['remaining_quantity'] > 0 and 
                   order['status'] == OrderStatus.PENDING and
                   (datetime.now() - start_time).total_seconds() < order['duration']):
                
                # Calculate slice quantity
                slice_qty = min(order['slice_quantity'], order['remaining_quantity'])
                
                # Get current market price for TWAP calculation
                current_price = await self._get_current_price(order['symbol'])
                
                # Place slice order at market price
                child_order_id = await self._place_market_order(
                    order['symbol'], slice_qty, current_price, order['side']
                )
                
                if child_order_id:
                    order['child_orders'].append(child_order_id)
                    order['filled_quantity'] += slice_qty
                    order['remaining_quantity'] -= slice_qty
                    order['slices_executed'] += 1
                    
                    logger.info(f"[TWAP] {order_id} slice {order['slices_executed']}/{order['total_slices']} executed: {slice_qty}@{current_price}")
                    
                    # Wait for next slice interval
                    await asyncio.sleep(order['slice_interval'])
                else:
                    logger.error(f"[TWAP] {order_id} slice execution failed")
                    break
            
            # Update final status
            if order['remaining_quantity'] == 0:
                order['status'] = OrderStatus.COMPLETED
                logger.info(f"[TWAP] {order_id} completed successfully")
            else:
                order['status'] = OrderStatus.PARTIAL
                logger.warning(f"[TWAP] {order_id} time expired or partially completed")
                
        except Exception as e:
            logger.error(f"[TWAP] Execution failed for {order_id}: {e}")
            self.active_orders[order_id]['status'] = OrderStatus.FAILED
    
    async def _execute_vwap_order(self, order_id: str):
        """Execute VWAP order logic"""
        try:
            order = self.active_orders[order_id]
            start_time = datetime.now()
            
            # Collect volume profile data
            volume_data = []
            
            while (order['remaining_quantity'] > 0 and 
                   order['status'] == OrderStatus.PENDING and
                   (datetime.now() - start_time).total_seconds() < order['duration']):
                
                # Get current market data
                current_price = await self._get_current_price(order['symbol'])
                current_volume = await self._get_current_volume(order['symbol'])
                
                volume_data.append({
                    'timestamp': datetime.now(),
                    'price': current_price,
                    'volume': current_volume
                })
                
                # Calculate VWAP-based slice size
                if len(volume_data) > 1:
                    recent_volume = sum([d['volume'] for d in volume_data[-5:]])  # Last 5 periods
                    volume_ratio = current_volume / (recent_volume / len(volume_data[-5:]) + 1)
                    
                    # Adjust slice size based on volume
                    base_slice = order['total_quantity'] * 0.1  # 10% base
                    volume_adjusted_slice = int(base_slice * min(2.0, max(0.5, volume_ratio)))
                    slice_qty = min(volume_adjusted_slice, order['remaining_quantity'])
                else:
                    slice_qty = min(int(order['total_quantity'] * 0.1), order['remaining_quantity'])
                
                if slice_qty > 0:
                    # Place volume-weighted slice
                    child_order_id = await self._place_market_order(
                        order['symbol'], slice_qty, current_price, order['side']
                    )
                    
                    if child_order_id:
                        order['child_orders'].append(child_order_id)
                        order['filled_quantity'] += slice_qty
                        order['remaining_quantity'] -= slice_qty
                        
                        logger.info(f"[VWAP] {order_id} volume-weighted slice executed: {slice_qty}@{current_price}")
                
                # Wait before next evaluation
                await asyncio.sleep(10)  # 10 second intervals
            
            # Update final status
            if order['remaining_quantity'] == 0:
                order['status'] = OrderStatus.COMPLETED
                logger.info(f"[VWAP] {order_id} completed successfully")
            else:
                order['status'] = OrderStatus.PARTIAL
                logger.warning(f"[VWAP] {order_id} time expired or partially completed")
                
        except Exception as e:
            logger.error(f"[VWAP] Execution failed for {order_id}: {e}")
            self.active_orders[order_id]['status'] = OrderStatus.FAILED
    
    async def _execute_bracket_order(self, order_id: str):
        """Execute bracket order logic"""
        try:
            order = self.active_orders[order_id]
            
            # Place entry order
            entry_order_id = await self._place_limit_order(
                order['symbol'], order['quantity'], order['entry_price'], order['side']
            )
            
            if not entry_order_id:
                order['status'] = OrderStatus.FAILED
                logger.error(f"[BRACKET] {order_id} entry order failed")
                return
            
            order['child_orders'].append(entry_order_id)
            
            # Monitor entry order fill
            entry_filled = await self._wait_for_order_fill(entry_order_id, timeout=300)
            
            if entry_filled:
                order['entry_filled'] = True
                logger.info(f"[BRACKET] {order_id} entry filled")
                
                # Place exit orders (stop-loss and target)
                exit_side = 'SELL' if order['side'] == 'BUY' else 'BUY'
                
                # Place stop-loss order
                sl_order_id = await self._place_stop_order(
                    order['symbol'], order['quantity'], order['stop_loss'], exit_side
                )
                
                # Place target order
                target_order_id = await self._place_limit_order(
                    order['symbol'], order['quantity'], order['target'], exit_side
                )
                
                if sl_order_id and target_order_id:
                    order['child_orders'].extend([sl_order_id, target_order_id])
                    order['exit_order_placed'] = True
                    order['status'] = OrderStatus.COMPLETED
                    
                    logger.info(f"[BRACKET] {order_id} bracket setup completed")
                else:
                    logger.error(f"[BRACKET] {order_id} exit orders failed")
                    order['status'] = OrderStatus.PARTIAL
            else:
                order['status'] = OrderStatus.FAILED
                logger.error(f"[BRACKET] {order_id} entry order not filled")
                
        except Exception as e:
            logger.error(f"[BRACKET] Execution failed for {order_id}: {e}")
            self.active_orders[order_id]['status'] = OrderStatus.FAILED
    
    async def _monitor_conditional_order(self, order_id: str):
        """Monitor conditional order until condition is met"""
        try:
            order = self.active_orders[order_id]
            
            while order['status'] == OrderStatus.PENDING:
                # Check condition
                condition_met = await self._check_condition(
                    order['condition_func'], 
                    order['condition_params'],
                    order['symbol']
                )
                
                if condition_met:
                    order['condition_met'] = True
                    
                    # Execute the order
                    child_order_id = await self._place_market_order(
                        order['symbol'], order['quantity'], order['price'], order['side']
                    )
                    
                    if child_order_id:
                        order['child_orders'].append(child_order_id)
                        order['status'] = OrderStatus.COMPLETED
                        logger.info(f"[CONDITIONAL] {order_id} condition met and executed")
                    else:
                        order['status'] = OrderStatus.FAILED
                        logger.error(f"[CONDITIONAL] {order_id} execution failed")
                    break
                
                # Wait before next check
                await asyncio.sleep(5)
                
        except Exception as e:
            logger.error(f"[CONDITIONAL] Monitoring failed for {order_id}: {e}")
            self.active_orders[order_id]['status'] = OrderStatus.FAILED
    
    async def _place_market_order(self, symbol: str, quantity: int, price: float, side: str) -> str:
        """Place a market order (simulation)"""
        try:
            # In real implementation, this would use the broker API
            order_id = f"MKT_{int(time.time())}_{symbol}"
            
            # Simulate order placement
            logger.info(f"[MARKET] Order placed: {side} {quantity} {symbol} @ {price}")
            
            # Add to order history
            self.order_history.append({
                'order_id': order_id,
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'side': side,
                'type': 'MARKET',
                'timestamp': datetime.now(),
                'status': 'FILLED'
            })
            
            return order_id
            
        except Exception as e:
            logger.error(f"[MARKET] Order failed: {e}")
            return None
    
    async def _place_limit_order(self, symbol: str, quantity: int, price: float, side: str) -> str:
        """Place a limit order (simulation)"""
        try:
            order_id = f"LMT_{int(time.time())}_{symbol}"
            
            # Simulate order placement
            logger.info(f"[LIMIT] Order placed: {side} {quantity} {symbol} @ {price}")
            
            self.order_history.append({
                'order_id': order_id,
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'side': side,
                'type': 'LIMIT',
                'timestamp': datetime.now(),
                'status': 'PENDING'
            })
            
            return order_id
            
        except Exception as e:
            logger.error(f"[LIMIT] Order failed: {e}")
            return None
    
    async def _place_stop_order(self, symbol: str, quantity: int, price: float, side: str) -> str:
        """Place a stop order (simulation)"""
        try:
            order_id = f"STP_{int(time.time())}_{symbol}"
            
            # Simulate order placement
            logger.info(f"[STOP] Order placed: {side} {quantity} {symbol} @ {price}")
            
            self.order_history.append({
                'order_id': order_id,
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'side': side,
                'type': 'STOP',
                'timestamp': datetime.now(),
                'status': 'PENDING'
            })
            
            return order_id
            
        except Exception as e:
            logger.error(f"[STOP] Order failed: {e}")
            return None
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price (simulation)"""
        # In real implementation, fetch from market data
        base_prices = {'NIFTY': 25400, 'BANKNIFTY': 56800}
        base_price = base_prices.get(symbol, 100)
        # Add small random variation
        return base_price * (1 + np.random.normal(0, 0.001))
    
    async def _get_current_volume(self, symbol: str) -> int:
        """Get current volume (simulation)"""
        # Simulate volume data
        return np.random.randint(1000, 10000)
    
    async def _wait_for_order_fill(self, order_id: str, timeout: int = 300) -> bool:
        """Wait for order to be filled (simulation)"""
        # Simulate order fill after random delay
        await asyncio.sleep(np.random.uniform(1, 5))
        return True  # Assume orders get filled for simulation
    
    async def _check_condition(self, condition_func: Callable, params: Dict, symbol: str) -> bool:
        """Check if custom condition is met"""
        try:
            # Get current market data for condition evaluation
            current_price = await self._get_current_price(symbol)
            params['current_price'] = current_price
            params['timestamp'] = datetime.now()
            
            # Evaluate condition
            return condition_func(params)
            
        except Exception as e:
            logger.error(f"[CONDITIONAL] Condition check failed: {e}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        try:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                order['status'] = OrderStatus.CANCELLED
                
                # Cancel execution task
                if order_id in self.execution_tasks:
                    self.execution_tasks[order_id].cancel()
                    del self.execution_tasks[order_id]
                
                logger.info(f"[CANCEL] Order {order_id} cancelled")
                return True
            else:
                logger.warning(f"[CANCEL] Order {order_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"[CANCEL] Failed to cancel order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get order status and details"""
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        else:
            return {'error': 'Order not found'}
    
    def get_active_orders(self) -> Dict:
        """Get all active orders"""
        return {oid: order for oid, order in self.active_orders.items() 
                if order['status'] in [OrderStatus.PENDING, OrderStatus.PARTIAL]}
    
    def get_order_history(self) -> List[Dict]:
        """Get order execution history"""
        return self.order_history.copy()

# Predefined condition functions for conditional orders
def price_above_condition(params: Dict) -> bool:
    """Condition: Current price above threshold"""
    return params['current_price'] > params['threshold_price']

def price_below_condition(params: Dict) -> bool:
    """Condition: Current price below threshold"""
    return params['current_price'] < params['threshold_price']

def time_condition(params: Dict) -> bool:
    """Condition: Current time after specified time"""
    return params['timestamp'] >= params['trigger_time']

def rsi_condition(params: Dict) -> bool:
    """Condition: RSI above/below threshold"""
    # This would require real RSI calculation
    # For simulation, return random condition
    return np.random.random() > 0.7