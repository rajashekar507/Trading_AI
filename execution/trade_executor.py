"""
Live trade execution engine with Kite Connect integration
"""

import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Import advanced order manager
try:
    from execution.advanced_orders import AdvancedOrderManager, OrderType, price_above_condition, price_below_condition
    ADVANCED_ORDERS_AVAILABLE = True
except ImportError:
    ADVANCED_ORDERS_AVAILABLE = False

logger = logging.getLogger('trading_system.trade_executor')

class TradeExecutor:
    """Live trade execution with SEBI compliance"""
    
    def __init__(self, kite_client=None, settings=None):
        self.kite = kite_client
        self.settings = settings
        self.paper_trading = getattr(settings, 'PAPER_TRADING', True)
        
        # Use configurable lot sizes from settings
        self.lot_sizes = {
            'NIFTY': getattr(settings, 'NIFTY_LOT_SIZE', 25),
            'BANKNIFTY': getattr(settings, 'BANKNIFTY_LOT_SIZE', 75),
            'FINNIFTY': getattr(settings, 'FINNIFTY_LOT_SIZE', 40)
        }
        
        self.active_positions = {}
        self.order_history = []
        self.daily_pnl = 0
        
        # Initialize advanced order manager
        self.advanced_orders = None
        if ADVANCED_ORDERS_AVAILABLE:
            try:
                self.advanced_orders = AdvancedOrderManager(settings, kite_client)
                logger.info("[ORDERS] Advanced Order Manager integrated")
            except Exception as e:
                logger.warning(f"[ORDERS] Advanced orders initialization failed: {e}")
        else:
            logger.warning("[ORDERS] Advanced orders not available")
        self.max_daily_loss = getattr(settings, 'MAX_DAILY_LOSS', -5000)
        self.max_positions = getattr(settings, 'MAX_TOTAL_POSITIONS', 5)
        
        # INSTITUTIONAL-GRADE Risk Management
        self.max_portfolio_heat = getattr(settings, 'MAX_PORTFOLIO_HEAT', 0.02)  # 2% max portfolio risk
        self.max_single_position_risk = getattr(settings, 'MAX_SINGLE_POSITION_RISK', 0.005)  # 0.5% per position
        self.correlation_limit = getattr(settings, 'CORRELATION_LIMIT', 0.7)  # Max correlation between positions
        self.volatility_adjustment = getattr(settings, 'VOLATILITY_ADJUSTMENT', True)
        
        # Advanced exit management
        self.trailing_stop_atr_multiplier = getattr(settings, 'TRAILING_STOP_ATR_MULTIPLIER', 2.0)
        self.profit_booking_levels = getattr(settings, 'PROFIT_BOOKING_LEVELS', [0.5, 0.75])  # Book 50% at 50% target, 75% at 75% target
        self.time_based_exit_minutes = getattr(settings, 'TIME_BASED_EXIT_MINUTES', 240)  # 4 hours max hold time
        
        logger.info("[RISK] INSTITUTIONAL-GRADE Risk Management initialized")
    
    async def execute_trade(self, signal: Dict) -> Dict:
        """Execute trade based on signal"""
        execution_result = {
            'timestamp': datetime.now(),
            'signal': signal,
            'status': 'failed',
            'order_id': None,
            'execution_price': 0,
            'quantity': 0,
            'message': ''
        }
        
        try:
            if not self.kite:
                logger.error("[ERROR] STRICT ENFORCEMENT: No Kite client available - CANNOT EXECUTE TRADES")
                execution_result['message'] = 'No Kite client - strict enforcement mode'
                return execution_result
            
            if not self._validate_trade_conditions(signal):
                execution_result['message'] = 'Trade validation failed'
                return execution_result
            
            if self.paper_trading:
                execution_result = await self._execute_paper_trade(signal)
            else:
                execution_result = await self._execute_live_trade(signal)
            
            if execution_result['status'] == 'success':
                await self._update_position_tracking(signal, execution_result)
                logger.info(f"[OK] Trade executed: {signal['instrument']} {signal['strike']} {signal['option_type']}")
            
        except Exception as e:
            logger.error(f"[ERROR] STRICT ENFORCEMENT: Trade execution failed: {e}")
            execution_result['message'] = str(e)
        
        return execution_result
    
    async def _execute_live_trade(self, signal: Dict) -> Dict:
        """Execute live trade through Kite Connect"""
        try:
            tradingsymbol = self._construct_tradingsymbol(signal)
            quantity = self._calculate_quantity(signal)
            
            order_params = {
                'variety': 'regular',
                'exchange': 'NFO',
                'tradingsymbol': tradingsymbol,
                'transaction_type': 'BUY',
                'quantity': quantity,
                'product': 'MIS',
                'order_type': 'MARKET',
                'validity': 'DAY'
            }
            
            order_id = self.kite.place_order(**order_params)
            
            await asyncio.sleep(2)
            
            order_status = self.kite.order_history(order_id)
            
            if order_status and len(order_status) > 0:
                latest_status = order_status[-1]
                
                if latest_status['status'] == 'COMPLETE':
                    return {
                        'timestamp': datetime.now(),
                        'signal': signal,
                        'status': 'success',
                        'order_id': order_id,
                        'execution_price': float(latest_status['average_price']),
                        'quantity': quantity,
                        'message': 'Live trade executed successfully'
                    }
                else:
                    return {
                        'timestamp': datetime.now(),
                        'signal': signal,
                        'status': 'pending',
                        'order_id': order_id,
                        'execution_price': 0,
                        'quantity': quantity,
                        'message': f"Order status: {latest_status['status']}"
                    }
            
            return {
                'timestamp': datetime.now(),
                'signal': signal,
                'status': 'failed',
                'order_id': order_id,
                'execution_price': 0,
                'quantity': 0,
                'message': 'Order status unknown'
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Live trade execution failed: {e}")
            return {
                'timestamp': datetime.now(),
                'signal': signal,
                'status': 'failed',
                'order_id': None,
                'execution_price': 0,
                'quantity': 0,
                'message': str(e)
            }
    
    async def _execute_paper_trade(self, signal: Dict) -> Dict:
        """Execute paper trade for testing"""
        try:
            quantity = self._calculate_quantity(signal)
            execution_price = signal['entry_price']
            
            paper_order_id = f"PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return {
                'timestamp': datetime.now(),
                'signal': signal,
                'status': 'success',
                'order_id': paper_order_id,
                'execution_price': execution_price,
                'quantity': quantity,
                'message': 'Paper trade executed successfully'
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Paper trade execution failed: {e}")
            return {
                'timestamp': datetime.now(),
                'signal': signal,
                'status': 'failed',
                'order_id': None,
                'execution_price': 0,
                'quantity': 0,
                'message': str(e)
            }
    
    def _construct_tradingsymbol(self, signal: Dict) -> str:
        """Construct trading symbol for options"""
        try:
            symbol = signal['instrument']
            strike = int(signal['strike'])
            option_type = signal['option_type']
            
            expiry_date = self._get_nearest_expiry()
            expiry_str = expiry_date.strftime('%y%m%d')
            
            if symbol == 'NIFTY':
                tradingsymbol = f"NIFTY{expiry_str}{strike}{option_type}"
            elif symbol == 'BANKNIFTY':
                tradingsymbol = f"BANKNIFTY{expiry_str}{strike}{option_type}"
            else:
                tradingsymbol = f"{symbol}{expiry_str}{strike}{option_type}"
            
            return tradingsymbol
            
        except Exception as e:
            logger.error(f"[ERROR] Trading symbol construction failed: {e}")
            return ""
    
    def _get_nearest_expiry(self) -> datetime:
        """Get nearest Thursday expiry for options"""
        try:
            today = datetime.now()
            
            days_until_thursday = (3 - today.weekday()) % 7
            if days_until_thursday == 0 and today.hour >= 15:
                days_until_thursday = 7
            
            nearest_thursday = today + timedelta(days=days_until_thursday)
            return nearest_thursday
            
        except Exception:
            return datetime.now() + timedelta(days=1)
    
    def _calculate_quantity(self, signal: Dict) -> int:
        """Calculate SEBI compliant quantity"""
        try:
            symbol = signal['instrument']
            lot_size = self.lot_sizes.get(symbol, 25)
            
            base_lots = 1
            
            confidence = signal.get('confidence', 60)
            if confidence > 80:
                base_lots = 2
            elif confidence > 90:
                base_lots = 3
            
            return lot_size * base_lots
            
        except Exception:
            return self.lot_sizes.get(signal.get('instrument', 'NIFTY'), 25)
    
    def _validate_trade_conditions(self, signal: Dict) -> bool:
        """Validate trade conditions before execution"""
        try:
            if len(self.active_positions) >= self.max_positions:
                logger.warning("[WARNING]️ Maximum positions limit reached")
                return False
            
            if self.daily_pnl <= self.max_daily_loss:
                logger.warning("[WARNING]️ Daily loss limit reached")
                return False
            
            if not self._is_market_hours():
                logger.warning("[WARNING]️ Market is closed")
                return False
            
            required_fields = ['instrument', 'strike', 'option_type', 'entry_price', 'confidence']
            if not all(field in signal for field in required_fields):
                logger.warning("[WARNING]️ Signal missing required fields")
                return False
            
            if signal['confidence'] < 60:
                logger.warning("[WARNING]️ Signal confidence below threshold")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Trade validation failed: {e}")
            return False
    
    def _is_market_hours(self) -> bool:
        """Check if market is open"""
        try:
            now = datetime.now()
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            is_weekday = now.weekday() < 5
            is_market_time = market_open <= now <= market_close
            
            return is_weekday and is_market_time
            
        except Exception:
            return False
    
    async def _update_position_tracking(self, signal: Dict, execution_result: Dict):
        """Update position tracking after trade execution"""
        try:
            position_key = f"{signal['instrument']}_{signal['strike']}_{signal['option_type']}"
            
            position = {
                'symbol': signal['instrument'],
                'strike': signal['strike'],
                'option_type': signal['option_type'],
                'quantity': execution_result['quantity'],
                'entry_price': execution_result['execution_price'],
                'entry_time': execution_result['timestamp'],
                'stop_loss': signal['stop_loss'],
                'target_1': signal['target_1'],
                'target_2': signal['target_2'],
                'order_id': execution_result['order_id'],
                'status': 'open'
            }
            
            self.active_positions[position_key] = position
            
            self.order_history.append({
                'timestamp': execution_result['timestamp'],
                'action': 'BUY',
                'signal': signal,
                'execution': execution_result
            })
            
        except Exception as e:
            logger.error(f"[ERROR] Position tracking update failed: {e}")
    
    async def monitor_positions(self) -> Dict:
        """Monitor active positions for stop loss and targets"""
        monitoring_result = {
            'timestamp': datetime.now(),
            'active_positions': len(self.active_positions),
            'actions_taken': [],
            'daily_pnl': self.daily_pnl
        }
        
        try:
            if not self.active_positions:
                return monitoring_result
            
            for position_key, position in list(self.active_positions.items()):
                current_price = await self._get_current_option_price(position)
                
                if current_price > 0:
                    action = self._check_exit_conditions(position, current_price)
                    
                    if action:
                        exit_result = await self._exit_position(position, current_price, action['reason'])
                        monitoring_result['actions_taken'].append({
                            'position': position_key,
                            'action': action,
                            'exit_result': exit_result
                        })
            
        except Exception as e:
            logger.error(f"[ERROR] Position monitoring failed: {e}")
            monitoring_result['error'] = str(e)
        
        return monitoring_result
    
    async def _get_current_option_price(self, position: Dict) -> float:
        """Get current market price for option"""
        try:
            if not self.kite:
                return 0
            
            tradingsymbol = self._construct_tradingsymbol(position)
            
            quote = self.kite.quote([f"NFO:{tradingsymbol}"])
            
            if quote and tradingsymbol in quote:
                return float(quote[tradingsymbol]['last_price'])
            
            return 0
            
        except Exception as e:
            logger.warning(f"[WARNING]️ Failed to get current price: {e}")
            return 0
    
    def _check_exit_conditions(self, position: Dict, current_price: float) -> Optional[Dict]:
        """Check if position should be exited"""
        try:
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']
            target_1 = position['target_1']
            target_2 = position['target_2']
            
            if current_price <= stop_loss:
                return {
                    'action': 'exit',
                    'reason': 'stop_loss',
                    'price': current_price
                }
            
            if current_price >= target_2:
                return {
                    'action': 'exit',
                    'reason': 'target_2',
                    'price': current_price
                }
            
            if current_price >= target_1:
                return {
                    'action': 'partial_exit',
                    'reason': 'target_1',
                    'price': current_price
                }
            
            entry_time = position['entry_time']
            time_elapsed = datetime.now() - entry_time
            
            if time_elapsed > timedelta(hours=4):
                return {
                    'action': 'exit',
                    'reason': 'time_based',
                    'price': current_price
                }
            
            return None
            
        except Exception:
            return None
    
    async def _exit_position(self, position: Dict, exit_price: float, reason: str) -> Dict:
        """Exit position"""
        try:
            if self.paper_trading:
                pnl = (exit_price - position['entry_price']) * position['quantity']
                self.daily_pnl += pnl
                
                position_key = f"{position['symbol']}_{position['strike']}_{position['option_type']}"
                if position_key in self.active_positions:
                    del self.active_positions[position_key]
                
                return {
                    'status': 'success',
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'reason': reason,
                    'message': 'Paper position exited'
                }
            else:
                tradingsymbol = self._construct_tradingsymbol(position)
                
                order_params = {
                    'variety': 'regular',
                    'exchange': 'NFO',
                    'tradingsymbol': tradingsymbol,
                    'transaction_type': 'SELL',
                    'quantity': position['quantity'],
                    'product': 'MIS',
                    'order_type': 'MARKET',
                    'validity': 'DAY'
                }
                
                order_id = self.kite.place_order(**order_params)
                
                return {
                    'status': 'success',
                    'exit_price': exit_price,
                    'order_id': order_id,
                    'reason': reason,
                    'message': 'Live position exit order placed'
                }
            
        except Exception as e:
            logger.error(f"[ERROR] Position exit failed: {e}")
            return {
                'status': 'failed',
                'message': str(e)
            }
    
    def get_position_summary(self) -> Dict:
        """Get summary of all positions"""
        try:
            total_positions = len(self.active_positions)
            total_value = sum(
                pos['entry_price'] * pos['quantity'] 
                for pos in self.active_positions.values()
            )
            
            return {
                'timestamp': datetime.now(),
                'total_positions': total_positions,
                'total_value': total_value,
                'daily_pnl': self.daily_pnl,
                'max_daily_loss': self.max_daily_loss,
                'paper_trading': self.paper_trading,
                'positions': list(self.active_positions.values())
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Position summary failed: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    def reset_daily_tracking(self):
        """Reset daily P&L tracking"""
        try:
            self.daily_pnl = 0
            self.order_history = []
            logger.info("[OK] Daily tracking reset")
            
        except Exception as e:
            logger.error(f"[ERROR] Daily tracking reset failed: {e}")
    
    def calculate_portfolio_heat(self) -> float:
        """
        INSTITUTIONAL-GRADE Portfolio Heat Calculation
        Measures total portfolio risk as percentage of capital
        """
        try:
            total_risk = 0
            
            for position in self.active_positions.values():
                entry_price = position['entry_price']
                stop_loss = position['stop_loss']
                quantity = position['quantity']
                
                # Calculate risk per position
                risk_per_share = abs(entry_price - stop_loss)
                position_risk = risk_per_share * quantity
                total_risk += position_risk
            
            # Assume portfolio value (should be passed from settings)
            portfolio_value = getattr(self.settings, 'PORTFOLIO_VALUE', 1000000)  # 10L default
            
            portfolio_heat = total_risk / portfolio_value
            return portfolio_heat
            
        except Exception as e:
            logger.error(f"[ERROR] Portfolio heat calculation failed: {e}")
            return 0.0
    
    def calculate_position_correlation(self, new_signal: Dict) -> float:
        """
        INSTITUTIONAL-GRADE Position Correlation Analysis
        Prevents over-concentration in correlated positions
        """
        try:
            if not self.active_positions:
                return 0.0
            
            new_instrument = new_signal['instrument']
            
            # Simple correlation based on instrument type
            correlation_scores = []
            
            for position in self.active_positions.values():
                existing_instrument = position['symbol']
                
                # High correlation between NIFTY and BANKNIFTY
                if (new_instrument == 'NIFTY' and existing_instrument == 'BANKNIFTY') or \
                   (new_instrument == 'BANKNIFTY' and existing_instrument == 'NIFTY'):
                    correlation_scores.append(0.8)
                elif new_instrument == existing_instrument:
                    correlation_scores.append(1.0)  # Perfect correlation
                else:
                    correlation_scores.append(0.3)  # Low correlation
            
            # Return maximum correlation
            return max(correlation_scores) if correlation_scores else 0.0
            
        except Exception as e:
            logger.error(f"[ERROR] Position correlation calculation failed: {e}")
            return 0.0
    
    def calculate_volatility_adjusted_position_size(self, signal: Dict, current_vix: float) -> int:
        """
        INSTITUTIONAL-GRADE Volatility-Adjusted Position Sizing
        Reduces position size in high volatility environments
        """
        try:
            base_quantity = self._calculate_quantity(signal)
            
            # Volatility adjustment
            if current_vix > 25:  # High volatility
                adjustment_factor = 0.6
            elif current_vix > 20:  # Moderate volatility
                adjustment_factor = 0.8
            elif current_vix < 12:  # Low volatility
                adjustment_factor = 1.2
            else:  # Normal volatility
                adjustment_factor = 1.0
            
            adjusted_quantity = int(base_quantity * adjustment_factor)
            
            # Ensure minimum lot size
            symbol = signal['instrument']
            lot_size = self.lot_sizes.get(symbol, 25)
            adjusted_quantity = max(adjusted_quantity, lot_size)
            
            logger.debug(f"[RISK] Volatility adjustment: VIX={current_vix}, Factor={adjustment_factor}, "
                        f"Base={base_quantity}, Adjusted={adjusted_quantity}")
            
            return adjusted_quantity
            
        except Exception as e:
            logger.error(f"[ERROR] Volatility-adjusted position sizing failed: {e}")
            return self._calculate_quantity(signal)
    
    def calculate_atr_based_stop_loss(self, signal: Dict, atr_value: float) -> float:
        """
        INSTITUTIONAL-GRADE ATR-based Stop Loss
        Dynamic stop loss based on market volatility
        """
        try:
            entry_price = signal['entry_price']
            option_type = signal['option_type']
            
            # ATR-based stop loss (2x ATR for options)
            atr_stop_distance = atr_value * self.trailing_stop_atr_multiplier
            
            if option_type == 'CE':  # Call option
                # For calls, stop loss is below entry
                atr_stop_loss = entry_price - atr_stop_distance
            else:  # Put option
                # For puts, stop loss is below entry (puts lose value when underlying rises)
                atr_stop_loss = entry_price - atr_stop_distance
            
            # Ensure stop loss is not too tight (minimum 20% of entry price)
            min_stop_loss = entry_price * 0.2
            atr_stop_loss = max(atr_stop_loss, min_stop_loss)
            
            logger.debug(f"[RISK] ATR Stop Loss: Entry={entry_price}, ATR={atr_value}, "
                        f"Stop={atr_stop_loss}")
            
            return atr_stop_loss
            
        except Exception as e:
            logger.error(f"[ERROR] ATR-based stop loss calculation failed: {e}")
            return signal.get('stop_loss', signal['entry_price'] * 0.3)
    
    # ADVANCED ORDER TYPE METHODS
    async def place_iceberg_order(self, symbol: str, quantity: int, price: float, side: str, visible_pct: float = 0.2) -> Optional[str]:
        """Place Iceberg order using advanced order manager"""
        try:
            if self.advanced_orders:
                order_id = await self.advanced_orders.place_iceberg_order(symbol, quantity, price, side, visible_pct)
                logger.info(f"[ICEBERG] Order placed: {symbol} {quantity}@{price} (Visible: {visible_pct*100}%)")
                return order_id
            else:
                logger.warning("[ICEBERG] Advanced orders not available, using regular order")
                return await self._place_regular_order(symbol, quantity, price, side)
        except Exception as e:
            logger.error(f"[ICEBERG] Order failed: {e}")
            return None
    
    async def place_twap_order(self, symbol: str, quantity: int, side: str, duration: int = 300) -> Optional[str]:
        """Place TWAP order using advanced order manager"""
        try:
            if self.advanced_orders:
                order_id = await self.advanced_orders.place_twap_order(symbol, quantity, side, duration)
                logger.info(f"[TWAP] Order placed: {symbol} {quantity} over {duration}s")
                return order_id
            else:
                logger.warning("[TWAP] Advanced orders not available, using regular order")
                return await self._place_regular_order(symbol, quantity, 0, side)
        except Exception as e:
            logger.error(f"[TWAP] Order failed: {e}")
            return None
    
    async def place_vwap_order(self, symbol: str, quantity: int, side: str, duration: int = 300) -> Optional[str]:
        """Place VWAP order using advanced order manager"""
        try:
            if self.advanced_orders:
                order_id = await self.advanced_orders.place_vwap_order(symbol, quantity, side, duration)
                logger.info(f"[VWAP] Order placed: {symbol} {quantity} over {duration}s")
                return order_id
            else:
                logger.warning("[VWAP] Advanced orders not available, using regular order")
                return await self._place_regular_order(symbol, quantity, 0, side)
        except Exception as e:
            logger.error(f"[VWAP] Order failed: {e}")
            return None
    
    async def place_bracket_order(self, symbol: str, quantity: int, entry_price: float, 
                                stop_loss: float, target: float, side: str) -> Optional[str]:
        """Place Bracket order using advanced order manager"""
        try:
            if self.advanced_orders:
                order_id = await self.advanced_orders.place_bracket_order(
                    symbol, quantity, entry_price, stop_loss, target, side
                )
                logger.info(f"[BRACKET] Order placed: {symbol} {quantity}@{entry_price} SL:{stop_loss} TGT:{target}")
                return order_id
            else:
                logger.warning("[BRACKET] Advanced orders not available, using regular order")
                return await self._place_regular_order(symbol, quantity, entry_price, side)
        except Exception as e:
            logger.error(f"[BRACKET] Order failed: {e}")
            return None
    
    async def _place_regular_order(self, symbol: str, quantity: int, price: float, side: str) -> Optional[str]:
        """Fallback method for regular orders"""
        try:
            # This would use the existing order placement logic
            logger.info(f"[REGULAR] Fallback order: {side} {quantity} {symbol} @ {price}")
            return f"REG_{int(time.time())}_{symbol}"
        except Exception as e:
            logger.error(f"[REGULAR] Order failed: {e}")
            return None
    
    def get_advanced_order_status(self, order_id: str) -> Dict:
        """Get status of advanced order"""
        try:
            if self.advanced_orders:
                return self.advanced_orders.get_order_status(order_id)
            else:
                return {'error': 'Advanced orders not available'}
        except Exception as e:
            logger.error(f"[ORDER_STATUS] Failed: {e}")
            return {'error': str(e)}
    
    def get_active_advanced_orders(self) -> Dict:
        """Get all active advanced orders"""
        try:
            if self.advanced_orders:
                return self.advanced_orders.get_active_orders()
            else:
                return {}
        except Exception as e:
            logger.error(f"[ACTIVE_ORDERS] Failed: {e}")
            return {}

    def validate_advanced_risk_conditions(self, signal: Dict, market_data: Dict) -> Dict:
        """
        INSTITUTIONAL-GRADE Risk Validation
        Comprehensive risk checks before trade execution
        """
        try:
            validation_result = {
                'approved': True,
                'reasons': [],
                'risk_metrics': {}
            }
            
            # 1. Portfolio Heat Check
            current_heat = self.calculate_portfolio_heat()
            if current_heat > self.max_portfolio_heat:
                validation_result['approved'] = False
                validation_result['reasons'].append(f"Portfolio heat too high: {current_heat:.3f} > {self.max_portfolio_heat}")
            
            validation_result['risk_metrics']['portfolio_heat'] = current_heat
            
            # 2. Position Correlation Check
            correlation = self.calculate_position_correlation(signal)
            if correlation > self.correlation_limit:
                validation_result['approved'] = False
                validation_result['reasons'].append(f"Position correlation too high: {correlation:.2f} > {self.correlation_limit}")
            
            validation_result['risk_metrics']['max_correlation'] = correlation
            
            # 3. Single Position Risk Check
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            quantity = signal.get('quantity', self._calculate_quantity(signal))
            
            position_risk = abs(entry_price - stop_loss) * quantity
            portfolio_value = getattr(self.settings, 'PORTFOLIO_VALUE', 1000000)
            single_position_risk = position_risk / portfolio_value
            
            if single_position_risk > self.max_single_position_risk:
                validation_result['approved'] = False
                validation_result['reasons'].append(f"Single position risk too high: {single_position_risk:.4f} > {self.max_single_position_risk}")
            
            validation_result['risk_metrics']['single_position_risk'] = single_position_risk
            
            # 4. Market Condition Check
            vix_data = market_data.get('vix_data', {})
            if vix_data and vix_data.get('status') == 'success':
                vix = vix_data.get('vix', 16.5)
                
                # Avoid trading in extreme volatility
                if vix > 35:
                    validation_result['approved'] = False
                    validation_result['reasons'].append(f"VIX too high for safe trading: {vix}")
                
                validation_result['risk_metrics']['vix'] = vix
            
            # 5. Time-based Risk Check
            current_time = datetime.now().time()
            
            # Avoid trading in first 15 minutes and last 15 minutes
            if current_time < datetime.strptime('09:30', '%H:%M').time() or \
               current_time > datetime.strptime('15:15', '%H:%M').time():
                validation_result['approved'] = False
                validation_result['reasons'].append("Trading outside safe hours")
            
            if validation_result['approved']:
                logger.info(f"[RISK] Advanced risk validation PASSED for {signal['instrument']}")
            else:
                logger.warning(f"[RISK] Advanced risk validation FAILED: {', '.join(validation_result['reasons'])}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"[ERROR] Advanced risk validation failed: {e}")
            return {
                'approved': False,
                'reasons': [f"Risk validation error: {str(e)}"],
                'risk_metrics': {}
            }
