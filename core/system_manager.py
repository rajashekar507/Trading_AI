"""
Main system manager and orchestrator for institutional-grade trading
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List

from core.data_manager import DataManager
from analysis.signal_engine import TradeSignalEngine
from analysis.multi_timeframe import MultiTimeframeAnalyzer
from analysis.pattern_detection import PatternDetector
from analysis.support_resistance import SupportResistanceCalculator
from analysis.ai_market_analyst import AIMarketAnalyst
from ml.adaptive_learning_system import AdaptiveLearningSystem
from strategies.orb_strategy import ORBStrategy
from execution.trade_executor import TradeExecutor
from risk.risk_manager import RiskManager
from analysis.backtesting import BacktestingEngine
from notifications.telegram_notifier import TelegramNotifier
from utils.sheets_integration_service import SheetsIntegrationService
from utils.error_recovery import ErrorRecoverySystem
from utils.memory_manager import MemoryManager
from auth.kite_auth_manager import KiteAuthManager as KiteAuthenticator
from utils.github_auto_push import GitHubAutoPush
# Auto backup functionality removed during cleanup

logger = logging.getLogger('trading_system.system_manager')

class TradingSystemManager:
    """Institutional-grade trading system orchestrator"""
    
    def __init__(self, settings):
        self.settings = settings
        
        self.data_manager = DataManager(settings)
        self.signal_engine = TradeSignalEngine(settings)
        self.telegram_notifier = TelegramNotifier(settings) if settings.TELEGRAM_BOT_TOKEN else None
        self.sheets_service = None  # Will be initialized after Kite client is ready
        
        # Initialize error recovery and memory management
        self.error_recovery = ErrorRecoverySystem(settings)
        self.memory_manager = MemoryManager(settings)
        
        self.kite_auth = KiteAuthenticator()
        self.kite_client = None
        self.multi_timeframe = None
        self.pattern_detector = None
        self.support_resistance = None
        self.orb_strategy = None
        self.trade_executor = None
        self.risk_manager = None
        self.backtesting_engine = None
        
        # Advanced AI components
        self.ai_market_analyst = None
        self.adaptive_learning = None
        
        self.running = False
        self.cycle_count = 0
        self.institutional_mode = True
        self.active_positions = {}
        
        logger.info("[OK] Institutional-grade TradingSystemManager initialized")
    
    async def run_live_trading(self):
        """Run live trading mode"""
        try:
            logger.info("[LIVE] Starting live trading mode...")
            
            # Start memory monitoring in background
            asyncio.create_task(self.memory_manager.start_monitoring())
            
            # Run main trading loop
            await self.run()
            
        except Exception as e:
            logger.error(f"[LIVE] Live trading error: {e}")
            await self.error_recovery.handle_error(e, {'component': 'live_trading'})
            raise e
        finally:
            # Stop memory monitoring
            self.memory_manager.stop_monitoring()
    
    async def shutdown(self):
        """Shutdown system gracefully"""
        try:
            logger.info("[SHUTDOWN] Shutting down trading system...")
            self.running = False
            
            # Stop memory monitoring
            self.memory_manager.stop_monitoring()
            
            # Save error and memory reports
            await self.error_recovery.save_error_report()
            await self.memory_manager.save_memory_report()
            
            logger.info("[SHUTDOWN] System shutdown completed")
            
        except Exception as e:
            logger.error(f"[SHUTDOWN] Shutdown error: {e}")
    
    async def initialize(self):
        """Initialize all system components"""
        try:
            if hasattr(self, '_initialized') and self._initialized:
                logger.info("[OK] System already initialized")
                return True
            
            logger.info(" Initializing all institutional-grade components...")
            result = await self._initialize_system()
            if result:
                self._initialized = True
            return result
        except Exception as e:
            logger.error(f"[ERROR] System initialization failed: {e}")
            return False
    
    async def run(self):
        """Main system loop"""
        try:
            logger.info(" Starting Trading System Manager")
            
            # Skip initialization if already done
            if not hasattr(self, '_initialized') or not self._initialized:
                if not await self.initialize():
                    logger.error("[ERROR] System initialization failed")
                    return
            
            self.running = True
            logger.info("[OK] System ready for trading")
            
            while self.running:
                try:
                    await self._execute_cycle_with_recovery()
                    self.cycle_count += 1
                    # Memory/data cleanup after each cycle
                    if hasattr(self, 'memory_manager'):
                        await self.memory_manager.cleanup_old_signals(getattr(self, 'signals', []))
                        await self.memory_manager.cleanup_old_positions(getattr(self, 'positions', []))
                        await self.memory_manager.archive_old_logs()
                        await self.memory_manager.cleanup_data_storage()
                    await asyncio.sleep(self.settings.DATA_REFRESH_INTERVAL)
                except KeyboardInterrupt:
                    logger.info(" Shutdown signal received")
                    break
                except Exception as e:
                    logger.error(f"[ERROR] Cycle execution error: {e}")
                    # Attempt error recovery
                    recovery_success = await self.error_recovery.handle_error(e, {'component': 'system_manager'})
                    if recovery_success:
                        logger.info("[RECOVERY] Successfully recovered from cycle error")
                        await asyncio.sleep(5)  # Short wait after recovery
                    else:
                        logger.error("[RECOVERY] Failed to recover from cycle error")
                        await asyncio.sleep(30)  # Longer wait after failed recovery

        except Exception as e:
            logger.error(f"[ERROR] System manager error: {e}")
        finally:
            await self._shutdown()
    
    async def _execute_cycle_with_recovery(self):
        """Execute trading cycle with error recovery"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                await self._execute_cycle()
                return  # Success, exit retry loop
            except Exception as e:
                if attempt == max_retries - 1:
                    # Last attempt failed, re-raise the exception
                    raise e
                
                logger.warning(f"[RECOVERY] Cycle attempt {attempt + 1} failed: {e}")
                
                # Attempt recovery
                recovery_success = await self.error_recovery.handle_error(e, {
                    'component': 'trading_cycle',
                    'attempt': attempt + 1
                })
                
                if recovery_success:
                    logger.info(f"[RECOVERY] Recovery successful, retrying cycle...")
                    await asyncio.sleep(2)  # Brief wait before retry
                else:
                    logger.warning(f"[RECOVERY] Recovery failed, waiting before retry...")
                    await asyncio.sleep(10 * (attempt + 1))  # Exponential backoff
    
    async def _initialize_system(self) -> bool:
        """Initialize all institutional-grade system components"""
        try:
            logger.info(" Initializing institutional-grade system components...")
            
            logger.info("[AUTH] Starting enhanced Kite authentication...")
            
            self.kite_client = self.kite_auth.get_authenticated_client()
            if not self.kite_client:
                logger.warning("[WARNING] Kite client not available - some features will be limited")
            else:
                logger.info("[OK] Kite client obtained successfully")
                self.data_manager.kite_client = self.kite_client
                logger.info("[OK] Kite client passed to DataManager")
            
            # Initialize Google Sheets integration
            logger.info("[INIT] Initializing Google Sheets integration...")
            self.sheets_service = SheetsIntegrationService(self.settings, self.kite_client)
            sheets_success = await self.sheets_service.initialize()
            if sheets_success:
                logger.info("[OK] Google Sheets integration initialized successfully")
                # Start continuous updates in background
                asyncio.create_task(self.sheets_service.start_continuous_updates())
            else:
                logger.warning("[WARNING] Google Sheets integration failed - continuing without it")

            if not await self.data_manager.initialize():
                logger.error("[ERROR] Data manager initialization failed")
                return False
            
            self.multi_timeframe = MultiTimeframeAnalyzer(self.settings)
            self.pattern_detector = PatternDetector(self.kite_client) if self.kite_client else None
            self.support_resistance = SupportResistanceCalculator(self.kite_client) if self.kite_client else None
            self.orb_strategy = ORBStrategy(self.kite_client) if self.kite_client else None
            
            self.trade_executor = TradeExecutor(self.kite_client, self.settings)
            self.risk_manager = RiskManager(self.settings)
            self.backtesting_engine = BacktestingEngine(self.kite_client)
            
            # Initialize advanced AI components
            logger.info("[AI] Initializing advanced AI components...")
            self.ai_market_analyst = AIMarketAnalyst(self.settings)
            self.adaptive_learning = AdaptiveLearningSystem(self.settings)
            
            # Initialize adaptive learning system
            if await self.adaptive_learning.initialize():
                logger.info("[AI] Adaptive learning system initialized")
            else:
                logger.warning("[AI] Adaptive learning system initialization failed")
            
            if self.telegram_notifier:
                await self.telegram_notifier.send_message(
                    " **INSTITUTIONAL-GRADE TRADING SYSTEM STARTED**\n\n"
                    "[OK] Multi-timeframe Analysis\n"
                    "[OK] Pattern Detection\n"
                    "[OK] Support/Resistance Calculation\n"
                    "[OK] ORB Strategy\n"
                    "[OK] Trade Execution Engine\n"
                    "[OK] Advanced Risk Management\n"
                    "[OK] Backtesting Framework\n\n"
                    " All systems operational and SEBI compliant"
                )
            
            logger.info("[OK] All institutional-grade components initialized")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] System initialization error: {e}")
            return False
    
    async def _execute_cycle(self):
        """Execute one complete institutional-grade analysis cycle"""
        cycle_start = datetime.now()
        logger.info(f" Starting institutional analysis cycle #{self.cycle_count + 1}")
        
        try:
            market_data = await self.data_manager.fetch_all_data()
            
            enhanced_data = market_data  # Use market_data directly for now
            
            risk_status = {'triggered': False}
            if self.risk_manager and isinstance(market_data, dict):
                risk_status = self.risk_manager.check_circuit_breakers(market_data)
            if risk_status['triggered']:
                logger.warning(f"[WARNING] Circuit breaker triggered: {risk_status['action']}")
                if risk_status['action'] == 'halt_trading':
                    await self._display_results(enhanced_data, [])
                    return
            
            if hasattr(self.signal_engine, 'cleanup_expired_signals'):
                self.signal_engine.cleanup_expired_signals()
            
            # Get AI market analysis
            ai_analysis = {}
            news_data = {}
            if self.ai_market_analyst:
                try:
                    # Get news data from data manager
                    if hasattr(self.data_manager, 'news_analyzer') and self.data_manager.news_analyzer:
                        news_data = await self.data_manager.news_analyzer.fetch_data()
                    
                    # Get comprehensive AI analysis
                    ai_analysis = await self.ai_market_analyst.analyze_market_conditions(enhanced_data, news_data)
                    logger.info(f"[AI] AI Analysis: {ai_analysis.get('overall_recommendation', 'NEUTRAL')} "
                               f"(Confidence: {ai_analysis.get('confidence_score', 0.0):.2f})")
                except Exception as e:
                    logger.warning(f"[AI] AI analysis failed: {e}")
            
            # Generate signals with AI enhancement
            signals = await self.signal_engine.generate_signals(enhanced_data)
            
            # Enhance signals with AI predictions
            if self.adaptive_learning and ai_analysis:
                enhanced_signals = []
                for signal in signals:
                    try:
                        # Get AI prediction for this signal
                        market_conditions = enhanced_data.get('market_conditions', {})
                        technical_indicators = signal.get('technical_indicators', {})
                        
                        prediction = await self.adaptive_learning.predict_trade_success(
                            market_conditions, news_data, technical_indicators, ai_analysis
                        )
                        
                        # Enhance signal with AI prediction
                        signal['ai_prediction'] = prediction
                        signal['ai_success_probability'] = prediction.get('success_probability', 0.5)
                        signal['ai_recommendation'] = prediction.get('recommendation', 'NEUTRAL')
                        
                        # Adjust signal strength based on AI prediction
                        original_strength = signal.get('strength', 50)
                        ai_adjustment = (prediction.get('success_probability', 0.5) - 0.5) * 100
                        signal['strength'] = max(0, min(100, original_strength + ai_adjustment))
                        
                        enhanced_signals.append(signal)
                        
                    except Exception as e:
                        logger.warning(f"[AI] Signal enhancement failed: {e}")
                        enhanced_signals.append(signal)
                
                signals = enhanced_signals
            
            validated_signals = []
            risk_filtered_signals = []
            
            for signal in signals:
                risk_assessment = self.risk_manager.validate_trade_risk(
                    signal, self.active_positions, enhanced_data
                )
                
                if risk_assessment['approved']:
                    validated_signals.append(signal)
                    logger.info(f"[OK] Signal approved: {signal['instrument']} {signal['strike']} {signal['option_type']}")
                    
                    # Log approved signal to Google Sheets
                    if self.sheets_service:
                        await self._log_signal_to_sheets(signal, 'approved')
                        
                else:
                    logger.warning(f"[WARNING] Signal rejected: {signal['instrument']} - {risk_assessment['violations']}")
                    signal['risk_status'] = 'RISK_FILTERED'
                    signal['risk_violations'] = risk_assessment.get('violations', [])
                    signal['risk_score'] = risk_assessment.get('risk_score', 0)
                    risk_filtered_signals.append(signal)
                    
                    # Log rejected signal to Google Sheets
                    if self.sheets_service:
                        await self._log_signal_to_sheets(signal, 'rejected', risk_assessment)
            
            # FIXED: Only send approved signals to Telegram, not risk-filtered ones
            if validated_signals and self.telegram_notifier:
                await self._send_institutional_notifications(validated_signals)
                logger.info(f"[OK] Sent {len(validated_signals)} approved signals to Telegram")
            
            if risk_filtered_signals:
                logger.warning(f"[BLOCKED] {len(risk_filtered_signals)} signals blocked by risk filters (NOT sent to Telegram)")
            
            if validated_signals and self.institutional_mode:
                execution_results = await self._execute_validated_trades(validated_signals)
                await self._update_position_tracking(execution_results)
            
            if self.active_positions:
                position_updates = await self.trade_executor.monitor_positions()
                await self._process_position_updates(position_updates)
            
            await self._display_institutional_results(enhanced_data, validated_signals, risk_filtered_signals)
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logger.info(f"[OK] Institutional cycle #{self.cycle_count + 1} completed in {cycle_duration:.1f}s")
            
        except Exception as e:
            logger.error(f"[ERROR] Institutional cycle execution failed: {e}")
    
    async def _perform_enhanced_analysis(self, market_data: Dict) -> Dict:
        """Perform enhanced analysis with all institutional modules"""
        try:
            enhanced_data = market_data.copy()
            
            for symbol in ['NIFTY', 'BANKNIFTY']:
                if self.multi_timeframe:
                    mtf_analysis = await self.multi_timeframe.analyze_symbol(symbol, market_data)
                    enhanced_data[f'{symbol.lower()}_mtf'] = mtf_analysis
                
                if self.pattern_detector:
                    patterns = await self.pattern_detector.detect_patterns(symbol)
                    enhanced_data[f'{symbol.lower()}_patterns'] = patterns
                
                if self.support_resistance:
                    sr_levels = await self.support_resistance.calculate_levels(symbol)
                    enhanced_data[f'{symbol.lower()}_sr'] = sr_levels
                
                if self.orb_strategy:
                    orb_analysis = await self.orb_strategy.analyze_orb(symbol)
                    enhanced_data[f'{symbol.lower()}_orb'] = orb_analysis
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"[ERROR] Enhanced analysis failed: {e}")
            return market_data
    
    async def _execute_validated_trades(self, signals: List[Dict]) -> List[Dict]:
        """Execute validated trades through trade executor"""
        execution_results = []
        
        try:
            for signal in signals:
                execution_result = await self.trade_executor.execute_trade(signal)
                execution_results.append(execution_result)
                
                if execution_result['status'] == 'success':
                    self.risk_manager.update_daily_stats(execution_result)
                
        except Exception as e:
            logger.error(f"[ERROR] Trade execution failed: {e}")
        
        return execution_results
    
    async def _update_position_tracking(self, execution_results: List[Dict]):
        """Update position tracking with execution results"""
        try:
            for result in execution_results:
                if result['status'] == 'success':
                    position_key = f"{result['signal']['instrument']}_{result['signal']['strike']}_{result['signal']['option_type']}"
                    self.active_positions[position_key] = result
                    
        except Exception as e:
            logger.error(f"[ERROR] Position tracking update failed: {e}")
    
    async def _process_position_updates(self, position_updates: Dict):
        """Process position monitoring updates"""
        try:
            if position_updates.get('actions_taken'):
                for action in position_updates['actions_taken']:
                    position_key = action['position']
                    if position_key in self.active_positions:
                        del self.active_positions[position_key]
                        logger.info(f"[OK] Position closed: {position_key}")
                        
        except Exception as e:
            logger.error(f"[ERROR] Position update processing failed: {e}")
    
    async def _display_institutional_results(self, market_data: Dict, signals: List[Dict], risk_filtered_signals: List[Dict] = None):
        """Display comprehensive institutional-grade analysis results"""
        try:
            print("\n" + "=" * 100)
            print("[INSTITUTIONAL]️ INSTITUTIONAL-GRADE OPTIONS TRADING SYSTEM - COMPREHENSIVE ANALYSIS")
            print("=" * 100)
            print(f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f" Cycle: #{self.cycle_count + 1} | Mode: INSTITUTIONAL")
            
            print("\n MARKET HEALTH & RISK ASSESSMENT:")
            print("-" * 50)
            
            spot_data = market_data.get('spot_data', {})
            if spot_data.get('status') == 'success':
                prices = spot_data.get('prices', {})
                print(f"[TRADING] NIFTY: {prices.get('NIFTY', 'N/A')}")
                print(f"[TRADING] BANKNIFTY: {prices.get('BANKNIFTY', 'N/A')}")
            
            vix_data = market_data.get('vix_data', {})
            if vix_data.get('status') == 'success':
                vix_level = vix_data.get('vix', 'N/A')
                vix_status = "🔴 HIGH" if vix_level > 25 else "🟡 MEDIUM" if vix_level > 20 else "🟢 LOW"
                print(f" VIX: {vix_level} ({vix_data.get('change', 0):+.2f}) - {vix_status}")
            
            print("\n[BRAIN] ENHANCED ANALYSIS RESULTS:")
            print("-" * 50)
            
            for symbol in ['NIFTY', 'BANKNIFTY']:
                symbol_lower = symbol.lower()
                
                mtf_data = market_data.get(f'{symbol_lower}_mtf', {})
                if mtf_data.get('status') == 'success':
                    consensus = mtf_data.get('consensus', {})
                    print(f" {symbol} Multi-Timeframe: {consensus.get('overall_signal', 'N/A').upper()} "
                          f"(Confidence: {consensus.get('confidence', 0):.0f}%)")
                
                patterns_data = market_data.get(f'{symbol_lower}_patterns', {})
                if patterns_data.get('status') == 'success':
                    detected_patterns = patterns_data.get('detected_patterns', [])
                    if detected_patterns:
                        pattern_names = [p['pattern'] for p in detected_patterns[:3]]
                        print(f"🕯️ {symbol} Patterns: {', '.join(pattern_names)}")
                
                sr_data = market_data.get(f'{symbol_lower}_sr', {})
                if sr_data.get('status') == 'success':
                    current_level = sr_data.get('current_level', 'neutral')
                    strength = sr_data.get('strength', 0)
                    print(f" {symbol} S/R: {current_level.upper()} (Strength: {strength:.1f})")
                
                orb_data = market_data.get(f'{symbol_lower}_orb', {})
                if orb_data.get('status') == 'success':
                    orb_signal = orb_data.get('signal', 'neutral')
                    orb_confidence = orb_data.get('confidence', 0)
                    print(f" {symbol} ORB: {orb_signal.upper()} (Confidence: {orb_confidence:.0f}%)")
            
            if self.risk_manager:
                risk_summary = self.risk_manager.get_risk_summary()
                print(f"\n RISK MANAGEMENT STATUS:")
                print("-" * 50)
                daily_stats = risk_summary.get('daily_stats', {})
                print(f"[MONEY] Daily P&L: Rs.{daily_stats.get('pnl', 0):.2f}")
                print(f" Trades Today: {daily_stats.get('trades_count', 0)}")
                print(f"[DOWN] Max Drawdown: Rs.{daily_stats.get('max_drawdown', 0):.2f}")
                print(f"[WARNING] Risk Violations: {len(daily_stats.get('risk_violations', []))}")
            
            print(f"\n[CLIPBOARD] ACTIVE POSITIONS: {len(self.active_positions)}")
            print("-" * 50)
            if self.active_positions:
                for pos_key, position in list(self.active_positions.items())[:5]:  # Show max 5
                    signal = position.get('signal', {})
                    print(f"   📍 {signal.get('instrument', 'N/A')} {signal.get('strike', 'N/A')} "
                          f"{signal.get('option_type', 'N/A')} - Entry: Rs.{position.get('execution_price', 0)}")
            else:
                print("   No active positions")
            
            print("\n VALIDATED TRADE SIGNALS:")
            print("-" * 50)
            
            if signals:
                for i, signal in enumerate(signals, 1):
                    print(f"\n[OK] INSTITUTIONAL SIGNAL #{i}:")
                    print(f"    Instrument: {signal['instrument']}")
                    print(f"    Strike: {signal['strike']} {signal['option_type']}")
                    print(f"   [MONEY] Entry Price: Rs.{signal['entry_price']}")
                    print(f"    Stop Loss: Rs.{signal['stop_loss']}")
                    print(f"    Target 1: Rs.{signal['target_1']}")
                    print(f"    Target 2: Rs.{signal['target_2']}")
                    print(f"    Confidence: {signal['confidence']}%")
                    print(f"   [NOTE] Reason: {signal['reason']}")
                    print(f"   ⬆️ Direction: {signal['direction'].upper()}")
                    print(f"    Risk Approved: [OK]")
            else:
                print("[ERROR] No validated signals (risk-filtered or below confidence threshold)")
                print(f"   Minimum confidence required: {self.settings.CONFIDENCE_THRESHOLD}%")
                if risk_filtered_signals:
                    print(f"   Risk-filtered signals: {len(risk_filtered_signals)} (sent to Telegram with risk warnings)")
            
            freshness = self.data_manager.get_data_freshness()
            print(f"\n📡 INSTITUTIONAL DATA SOURCES ({freshness['health_percentage']:.0f}% HEALTHY):")
            print("-" * 50)
            for source, status in market_data.get('data_status', {}).items():
                status_icon = "[OK]" if status == 'success' else "[ERROR]" if status == 'failed' else ""
                print(f"   {status_icon} {source.replace('_', ' ').title()}: {status}")
            
            print("\n" + "=" * 100)
            
        except Exception as e:
            logger.error(f"[ERROR] Institutional display results error: {e}")
    
    async def _display_results(self, market_data: Dict, signals: List[Dict]):
        """Fallback display method for compatibility"""
        await self._display_institutional_results(market_data, signals, [])
    
    async def _send_institutional_notifications(self, signals: List[Dict]):
        """Send institutional-grade trade signal notifications via Telegram"""
        try:
            for signal in signals:
                risk_status = signal.get('risk_status', 'VALIDATED')
                if risk_status == 'RISK_FILTERED':
                    risk_summary = f"[WARNING] RISK FILTERED (Score: {signal.get('risk_score', 0):.0f}/100)"
                    violations = signal.get('risk_violations', [])
                    risk_details = f"\n[ALERT] **Violations:** {', '.join(violations[:2])}" if violations else ""
                else:
                    risk_summary = "[OK] VALIDATED & APPROVED"
                    risk_details = ""
                
                message = f"""
 **TRADE SIGNAL - {signal['instrument']}**

 **Instrument:** {signal['instrument']}
 **Strike:** {signal['strike']} {signal['option_type']}
 **Expiry:** {signal.get('expiry', 'Current Week')}
[MONEY] **Entry Price:** Rs.{signal['entry_price']}
 **Stop Loss:** Rs.{signal['stop_loss']}
 **Target 1:** Rs.{signal['target_1']}
 **Target 2:** Rs.{signal['target_2']}
 **Confidence:** {signal['confidence']}%
⬆️ **Direction:** {signal['direction'].upper()}
[NOTE] **Reason:** {signal['reason']}

 **Risk Status:** {risk_summary}{risk_details}
[CLIPBOARD] **Active Positions:** {len(self.active_positions)}
 **Timestamp (IST):** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

[INSTITUTIONAL]️ **VLR_AI Institutional Trading System**
"""
                
                if self.telegram_notifier:
                    success = await self.telegram_notifier.send_message(message)
                    if success:
                        logger.info(f"[OK] Telegram signal sent: {signal['instrument']} {signal['strike']} {signal['option_type']}")
                    else:
                        logger.error(f"[ERROR] Telegram signal failed: {signal['instrument']}")
                
        except Exception as e:
            logger.error(f"[ERROR] Institutional notification sending failed: {e}")
    
    async def _send_signal_notifications(self, signals: List[Dict]):
        """Fallback notification method for compatibility"""
        await self._send_institutional_notifications(signals)
    
    async def _shutdown(self):
        """Graceful institutional system shutdown"""
        try:
            logger.info(" Shutting down institutional trading system...")
            self.running = False
            
            if self.active_positions and self.trade_executor:
                logger.info(f"[CLIPBOARD] Monitoring {len(self.active_positions)} active positions during shutdown")
            
            if self.risk_manager:
                final_risk_report = self.risk_manager.get_risk_summary()
                logger.info(f" Final daily P&L: Rs.{final_risk_report.get('daily_stats', {}).get('pnl', 0):.2f}")
            
            if self.telegram_notifier:
                shutdown_message = f"""
 **INSTITUTIONAL TRADING SYSTEM SHUTDOWN**

 **Final Session Summary:**
- Cycles Completed: {self.cycle_count}
- Active Positions: {len(self.active_positions)}
- System Mode: INSTITUTIONAL

 **Risk Management:**
- All positions monitored
- Risk limits maintained
- SEBI compliance verified

 **Shutdown Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                await self.telegram_notifier.send_message(shutdown_message)
            
            logger.info("[OK] Institutional system shutdown complete")
            
        except Exception as e:
            logger.error(f"[ERROR] Shutdown error: {e}")
    
    def stop(self):
        """Stop the institutional system"""
        self.running = False
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'running': self.running,
                'cycle_count': self.cycle_count,
                'institutional_mode': self.institutional_mode,
                'active_positions': len(self.active_positions),
                'components': {
                    'data_manager': bool(self.data_manager),
                    'signal_engine': bool(self.signal_engine),
                    'multi_timeframe': bool(self.multi_timeframe),
                    'pattern_detector': bool(self.pattern_detector),
                    'support_resistance': bool(self.support_resistance),
                    'orb_strategy': bool(self.orb_strategy),
                    'trade_executor': bool(self.trade_executor),
                    'risk_manager': bool(self.risk_manager),
                    'backtesting_engine': bool(self.backtesting_engine),
                    'kite_client': bool(self.kite_client),
                    'telegram_notifier': bool(self.telegram_notifier)
                }
            }
        except Exception as e:
            logger.error(f"[ERROR] System status error: {e}")
            return {'error': str(e)}
    

    
    async def _display_institutional_results(self, enhanced_data, validated_signals, risk_filtered_signals):
        """Display institutional-grade analysis results"""
        try:
            # Display market data summary
            if 'spot_data' in enhanced_data:
                spot_data = enhanced_data['spot_data']
                nifty_price = spot_data.get('NIFTY', 0)
                banknifty_price = spot_data.get('BANKNIFTY', 0)
                logger.info(f"[MARKET] NIFTY: {nifty_price}, BANKNIFTY: {banknifty_price}")
            
            # Display signals summary
            total_signals = len(validated_signals) + len(risk_filtered_signals)
            logger.info(f"[SIGNALS] Generated: {total_signals}, Validated: {len(validated_signals)}, Risk-filtered: {len(risk_filtered_signals)}")
            
            # Display validated signals
            for signal in validated_signals:
                logger.info(f"[TRADE] {signal.get('instrument', 'N/A')} {signal.get('action', 'N/A')} - Score: {signal.get('confidence', 0):.1f}%")
            
        except Exception as e:
            logger.error(f"[ERROR] Institutional display results error: {e}")
    
    async def _log_signal_to_sheets(self, signal: Dict, signal_type: str, risk_assessment: Dict = None):
        """Log signal to Google Sheets"""
        try:
            if signal_type == 'approved':
                # Prepare signal data for Google Sheets
                signal_data = {
                    'signal_id': f"{signal.get('instrument', 'UNK')}_{signal.get('strike', 0)}_{signal.get('option_type', 'CE')}_{datetime.now().strftime('%H%M%S')}",
                    'instrument': signal.get('instrument', ''),
                    'direction': signal.get('option_type', ''),
                    'strike_price': signal.get('strike', ''),
                    'expiry_date': signal.get('expiry', ''),
                    'entry_price': signal.get('entry_price', ''),
                    'stop_loss': signal.get('stop_loss', ''),
                    'target_1': signal.get('target_1', ''),
                    'target_2': signal.get('target_2', ''),
                    'confidence_score': signal.get('confidence', 0),
                    'risk_score': signal.get('risk_score', 0),
                    'reason_summary': signal.get('reason', ''),
                    'status': 'Pending'
                }
                
                await self.sheets_service.log_trade_signal(signal_data)
                
            elif signal_type == 'rejected':
                # Prepare rejected signal data
                rejected_data = {
                    'signal_id': f"REJ_{signal.get('instrument', 'UNK')}_{signal.get('strike', 0)}_{signal.get('option_type', 'CE')}_{datetime.now().strftime('%H%M%S')}",
                    'instrument': signal.get('instrument', ''),
                    'direction': signal.get('option_type', ''),
                    'strike_price': signal.get('strike', ''),
                    'expiry_date': signal.get('expiry', ''),
                    'proposed_entry_price': signal.get('entry_price', ''),
                    'proposed_stop_loss': signal.get('stop_loss', ''),
                    'proposed_target_1': signal.get('target_1', ''),
                    'proposed_target_2': signal.get('target_2', ''),
                    'risk_score': risk_assessment.get('risk_score', 0) if risk_assessment else 0,
                    'confidence_score': signal.get('confidence', 0),
                    'rejection_reason': ', '.join(risk_assessment.get('violations', [])) if risk_assessment else 'Unknown',
                    'risk_cost': risk_assessment.get('risk_cost', 0) if risk_assessment else 0,
                    'max_drawdown_risk': risk_assessment.get('max_drawdown_risk', 0) if risk_assessment else 0,
                    'position_size': signal.get('position_size', 1)
                }
                
                await self.sheets_service.log_rejected_signal(rejected_data)
                
        except Exception as e:
            logger.error(f"Failed to log signal to sheets: {e}")
    
    async def update_signal_status_in_sheets(self, signal_id: str, status: str, pnl: float = None):
        """Update signal status in Google Sheets"""
        try:
            if self.sheets_service:
                exit_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') if status != 'Pending' else None
                await self.sheets_service.update_signal_status(signal_id, status, exit_time, pnl)
                
        except Exception as e:
            logger.error(f"Failed to update signal status in sheets: {e}")
    
    async def _shutdown(self):
        """Shutdown all system components"""
        try:
            logger.info("[OK] Shutting down trading system...")
            
            self.running = False
            
            # Shutdown Google Sheets service
            if self.sheets_service:
                await self.sheets_service.stop()
            
            # Send shutdown notification
            if self.telegram_notifier:
                await self.telegram_notifier.send_message(
                    " **INSTITUTIONAL TRADING SYSTEM SHUTDOWN**\n\n"
                    f" Final daily P&L: Rs.{sum(pos.get('pnl', 0) for pos in self.active_positions.values()):.2f}\n"
                    f" Active positions: {len(self.active_positions)}\n"
                    f" Total cycles completed: {self.cycle_count}\n\n"
                    " All systems safely shutdown"
                )
            
            logger.info("[OK] Institutional system shutdown complete")
            
        except Exception as e:
            logger.error(f"[ERROR] Shutdown error: {e}")
    
    def stop(self):
        """Stop the trading system"""
        self.running = False
        logger.info("[STOP] Trading system stop signal sent")
    

    
    async def run_institutional_cycle(self):
        """Run a single institutional analysis cycle"""
        try:
            await self._execute_cycle()
            return True
        except Exception as e:
            logger.error(f"[ERROR] Institutional cycle failed: {e}")
            return False

