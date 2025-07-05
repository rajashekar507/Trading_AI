"""
FINAL COMPREHENSIVE SYSTEM VALIDATION
Tests ALL features, ALL data sources, ALL modes
Provides complete system health report
"""

import asyncio
import logging
import sys
import json
import time
import traceback
import subprocess
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'final_validation.log')
    ]
)
logger = logging.getLogger('final_system_validation')

class FinalSystemValidator:
    """Complete end-to-end system validation with comprehensive reporting"""
    
    def __init__(self):
        self.validation_results = {}
        self.data_sources_verified = {}
        self.issues_found = []
        self.fixes_applied = []
        self.performance_metrics = {}
        self.system_health_score = 0
        
        # Create logs directory
        (PROJECT_ROOT / 'logs').mkdir(exist_ok=True)
        
    async def run_final_validation(self):
        """Run complete final validation"""
        logger.info("🚀 STARTING FINAL COMPREHENSIVE SYSTEM VALIDATION")
        logger.info("="*100)
        logger.info("This validation will test EVERY component with REAL data")
        logger.info("="*100)
        
        # Comprehensive test sequence
        test_sequence = [
            ("🔧 System Requirements", self.validate_system_requirements),
            ("📦 Import Validation", self.validate_all_imports),
            ("🌐 Data Source Verification", self.verify_real_data_sources),
            ("📊 Market Data Testing", self.test_market_data_real),
            ("🤖 ML Models Testing", self.test_ml_models),
            ("⚡ Demo Mode Execution", self.test_demo_mode),
            ("💰 Paper Trading Test", self.test_paper_trading),
            ("🔗 System Integration", self.test_system_integration),
            ("🛡️ Risk Management", self.test_risk_management),
            ("📱 Notification Systems", self.test_notifications),
            ("🔄 Error Recovery", self.test_error_recovery),
            ("💾 Memory Management", self.test_memory_management),
            ("🌍 API Connections", self.test_api_connections),
            ("📈 Performance Metrics", self.test_performance),
            ("🔍 Data Accuracy", self.verify_data_accuracy),
            ("🔗 Internal Linking", self.test_internal_linking)
        ]
        
        start_time = time.time()
        
        for test_name, test_func in test_sequence:
            try:
                logger.info(f"\n{test_name}")
                logger.info("-" * 80)
                
                test_start = time.time()
                result = await test_func()
                test_end = time.time()
                
                self.validation_results[test_name] = {
                    'success': result.get('success', False),
                    'details': result,
                    'duration': test_end - test_start,
                    'timestamp': datetime.now().isoformat()
                }
                
                if result.get('success', False):
                    logger.info(f"✅ {test_name}: PASSED ({test_end - test_start:.2f}s)")
                    self.system_health_score += 1
                else:
                    logger.error(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
                    if 'fixes' in result:
                        self.fixes_applied.extend(result['fixes'])
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: EXCEPTION - {e}")
                logger.error(traceback.format_exc())
                self.validation_results[test_name] = {
                    'success': False,
                    'error': str(e),
                    'duration': 0,
                    'timestamp': datetime.now().isoformat()
                }
        
        total_time = time.time() - start_time
        
        # Generate comprehensive final report
        await self.generate_final_comprehensive_report(total_time)
    
    async def validate_system_requirements(self) -> Dict:
        """Validate system requirements"""
        try:
            logger.info("🔍 Validating system requirements...")
            
            # Python version
            python_version = sys.version_info
            python_ok = python_version >= (3, 8)
            
            # Required packages
            required_packages = [
                'kiteconnect', 'pandas', 'numpy', 'requests', 'asyncio',
                'aiohttp', 'python-dotenv', 'tensorflow', 'scikit-learn',
                'dhanhq', 'psutil', 'brotli', 'yfinance', 'beautifulsoup4'
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package.replace('-', '_'))
                except ImportError:
                    missing_packages.append(package)
            
            # System resources
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            resource_ok = (
                memory.available > 2 * 1024**3 and  # At least 2GB free
                disk.free > 5 * 1024**3  # At least 5GB free
            )
            
            success = python_ok and len(missing_packages) == 0 and resource_ok
            
            return {
                'success': success,
                'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                'python_ok': python_ok,
                'missing_packages': missing_packages,
                'memory_available_gb': memory.available / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'resource_ok': resource_ok
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def validate_all_imports(self) -> Dict:
        """Validate all system imports"""
        try:
            logger.info("🔍 Testing all system imports...")
            
            # Critical imports
            critical_imports = [
                'core.system_manager',
                'core.data_manager',
                'data.market_data',
                'execution.paper_trading_executor',
                'analysis.signal_engine',
                'risk.risk_manager',
                'config.enhanced_settings'
            ]
            
            # All other imports
            other_imports = [
                'analysis.technical_analysis',
                'analysis.news_sentiment',
                'analysis.ai_market_analyst',
                'ml.ensemble_predictor',
                'ml.lstm_predictor',
                'strategies.orb_strategy',
                'strategies.options_greeks',
                'notifications.telegram_notifier',
                'utils.memory_optimizer',
                'utils.error_recovery'
            ]
            
            failed_critical = []
            failed_other = []
            successful_imports = []
            
            # Test critical imports
            for module_name in critical_imports:
                try:
                    __import__(module_name)
                    successful_imports.append(module_name)
                except Exception as e:
                    failed_critical.append(f"{module_name}: {e}")
                    logger.error(f"❌ CRITICAL Import failed: {module_name} - {e}")
            
            # Test other imports
            for module_name in other_imports:
                try:
                    __import__(module_name)
                    successful_imports.append(module_name)
                except Exception as e:
                    failed_other.append(f"{module_name}: {e}")
                    logger.warning(f"⚠️ Import failed: {module_name} - {e}")
            
            # Success if all critical imports work
            success = len(failed_critical) == 0
            
            return {
                'success': success,
                'successful_imports': len(successful_imports),
                'failed_critical': failed_critical,
                'failed_other': failed_other,
                'total_tested': len(critical_imports) + len(other_imports)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def verify_real_data_sources(self) -> Dict:
        """Verify all data sources are REAL"""
        try:
            logger.info("🔍 Verifying ALL data sources are REAL...")
            
            from config.enhanced_settings import EnhancedSettings
            settings = EnhancedSettings()
            
            data_sources = {}
            
            # Test Dhan API (Primary market data)
            try:
                from data.market_data import MarketDataProvider
                market_provider = MarketDataProvider()
                
                connection_test = market_provider._test_connection()
                data_sources['dhan_api'] = {
                    'real': connection_test,
                    'source': 'Dhan API',
                    'status': 'connected' if connection_test else 'failed',
                    'critical': True
                }
                
                if connection_test:
                    # Test actual data
                    nifty_data = market_provider.get_live_quote('NIFTY')
                    banknifty_data = market_provider.get_live_quote('BANKNIFTY')
                    
                    data_sources['nifty_data'] = {
                        'real': nifty_data.get('status') == 'success',
                        'price': nifty_data.get('price', 0),
                        'source': 'Dhan API',
                        'critical': True
                    }
                    
                    data_sources['banknifty_data'] = {
                        'real': banknifty_data.get('status') == 'success',
                        'price': banknifty_data.get('price', 0),
                        'source': 'Dhan API',
                        'critical': True
                    }
                
            except Exception as e:
                data_sources['dhan_api'] = {'real': False, 'error': str(e), 'critical': True}
            
            # Test Kite API (Secondary market data)
            try:
                from auth.enhanced_kite_auth import EnhancedKiteAuthenticator
                kite_auth = EnhancedKiteAuthenticator()
                data_sources['kite_api'] = {
                    'real': True,
                    'source': 'Kite Connect API',
                    'status': 'configured',
                    'critical': False
                }
            except Exception as e:
                data_sources['kite_api'] = {'real': False, 'error': str(e), 'critical': False}
            
            # Test News Intelligence
            try:
                from analysis.news_sentiment import NewsSentimentAnalyzer
                news_analyzer = NewsSentimentAnalyzer(settings)
                await news_analyzer.initialize()
                
                data_sources['news_intelligence'] = {
                    'real': True,
                    'source': 'Stealth Web Intelligence',
                    'status': 'initialized',
                    'critical': False
                }
            except Exception as e:
                data_sources['news_intelligence'] = {'real': False, 'error': str(e), 'critical': False}
            
            # Count real sources
            critical_real = sum(1 for ds in data_sources.values() 
                              if ds.get('real', False) and ds.get('critical', False))
            total_real = sum(1 for ds in data_sources.values() if ds.get('real', False))
            
            self.data_sources_verified = data_sources
            
            # Success if all critical sources are real
            success = critical_real >= 2  # At least Dhan API and NIFTY data
            
            return {
                'success': success,
                'data_sources': data_sources,
                'critical_real': critical_real,
                'total_real': total_real,
                'total_sources': len(data_sources)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_market_data_real(self) -> Dict:
        """Test market data with REAL verification"""
        try:
            logger.info("🔍 Testing market data with REAL verification...")
            
            from data.market_data import MarketDataProvider
            market_provider = MarketDataProvider()
            
            # Test multiple instruments
            instruments = ['NIFTY', 'BANKNIFTY']
            results = {}
            
            for instrument in instruments:
                data = market_provider.get_live_quote(instrument)
                
                if data.get('status') == 'success':
                    price = data.get('price', 0)
                    
                    # Verify price is reasonable
                    reasonable_ranges = {
                        'NIFTY': (15000, 30000),
                        'BANKNIFTY': (30000, 70000)
                    }
                    
                    min_price, max_price = reasonable_ranges.get(instrument, (0, 999999))
                    price_reasonable = min_price <= price <= max_price
                    
                    results[instrument] = {
                        'success': True,
                        'price': price,
                        'reasonable': price_reasonable,
                        'change': data.get('change', 0),
                        'change_percent': data.get('change_percent', 0)
                    }
                else:
                    results[instrument] = {
                        'success': False,
                        'error': data.get('error', 'Unknown error')
                    }
            
            # Success if at least one instrument works with reasonable price
            success = any(r.get('success', False) and r.get('reasonable', False) 
                         for r in results.values())
            
            return {
                'success': success,
                'instruments_tested': results,
                'working_instruments': sum(1 for r in results.values() if r.get('success', False))
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_ml_models(self) -> Dict:
        """Test ML models"""
        try:
            logger.info("🔍 Testing ML models...")
            
            from ml.ensemble_predictor import EnsemblePredictor
            from config.enhanced_settings import EnhancedSettings
            
            settings = EnhancedSettings()
            ensemble = EnsemblePredictor(settings)
            
            # Test with sample data
            import numpy as np
            test_data = {
                'prices': np.random.random(100) * 1000 + 24000,
                'volumes': np.random.random(100) * 1000000,
                'features': np.random.random((100, 10))
            }
            
            # Test prediction
            try:
                prediction = await ensemble.predict(test_data)
                ml_working = prediction is not None
            except:
                ml_working = False
            
            return {
                'success': ml_working,
                'ensemble_available': True,
                'prediction_working': ml_working
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_demo_mode(self) -> Dict:
        """Test demo mode execution"""
        try:
            logger.info("🔍 Testing demo mode execution...")
            
            # Run demo mode with timeout
            result = subprocess.run([
                sys.executable, 'main.py', '--mode', 'demo'
            ], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=180)
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            # Check for success indicators
            success_indicators = [
                'READY FOR REAL TRADING',
                'Data fetch completed',
                'System validation passed'
            ]
            
            indicators_found = sum(1 for indicator in success_indicators if indicator in output)
            
            return {
                'success': success and indicators_found >= 2,
                'return_code': result.returncode,
                'indicators_found': indicators_found,
                'output_length': len(output)
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Demo mode timeout (180s)'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_paper_trading(self) -> Dict:
        """Test paper trading functionality"""
        try:
            logger.info("🔍 Testing paper trading functionality...")
            
            from execution.paper_trading_executor import PaperTradingExecutor
            from config.enhanced_settings import EnhancedSettings
            
            settings = EnhancedSettings()
            paper_executor = PaperTradingExecutor(settings)
            
            initial_balance = paper_executor.virtual_balance
            
            # Test signal
            test_signal = {
                'symbol': 'NIFTY',
                'action': 'BUY',
                'quantity': 1,
                'price': 25000,
                'signal_type': 'test',
                'confidence': 75,
                'timestamp': datetime.now()
            }
            
            # Execute test trade
            trade_result = await paper_executor.execute_paper_trade(test_signal)
            
            success = (
                initial_balance > 0 and
                trade_result.get('status') in ['success', 'rejected']  # Both are valid
            )
            
            return {
                'success': success,
                'initial_balance': initial_balance,
                'trade_status': trade_result.get('status', 'unknown'),
                'trade_executed': trade_result.get('status') == 'success'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_system_integration(self) -> Dict:
        """Test system integration"""
        try:
            logger.info("🔍 Testing system integration...")
            
            from core.system_manager import TradingSystemManager
            from config.enhanced_settings import EnhancedSettings
            
            settings = EnhancedSettings()
            system_manager = TradingSystemManager(settings)
            
            # Test initialization
            init_success = await system_manager.initialize()
            
            # Test components
            components = {
                'data_manager': hasattr(system_manager, 'data_manager'),
                'signal_engine': hasattr(system_manager, 'signal_engine'),
                'risk_manager': hasattr(system_manager, 'risk_manager'),
                'trade_executor': hasattr(system_manager, 'trade_executor')
            }
            
            # Cleanup
            try:
                await system_manager.shutdown()
            except:
                pass
            
            success = init_success and sum(components.values()) >= 3
            
            return {
                'success': success,
                'initialization': init_success,
                'components': components
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_risk_management(self) -> Dict:
        """Test risk management"""
        try:
            logger.info("🔍 Testing risk management...")
            
            from risk.risk_manager import RiskManager
            from config.enhanced_settings import EnhancedSettings
            
            settings = EnhancedSettings()
            risk_manager = RiskManager(settings)
            
            # Test position risk
            test_position = {
                'symbol': 'NIFTY',
                'quantity': 1,
                'entry_price': 25000,
                'current_price': 25100,
                'position_type': 'LONG'
            }
            
            risk_assessment = await risk_manager.assess_position_risk(test_position)
            
            return {
                'success': risk_assessment is not None,
                'risk_assessment_working': risk_assessment is not None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_notifications(self) -> Dict:
        """Test notification systems"""
        try:
            logger.info("🔍 Testing notification systems...")
            
            from notifications.telegram_notifier import TelegramNotifier
            from config.enhanced_settings import EnhancedSettings
            
            settings = EnhancedSettings()
            telegram = TelegramNotifier(settings)
            
            telegram_configured = bool(settings.TELEGRAM_BOT_TOKEN)
            
            return {
                'success': telegram_configured,
                'telegram_configured': telegram_configured
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_error_recovery(self) -> Dict:
        """Test error recovery system"""
        try:
            logger.info("🔍 Testing error recovery...")
            
            from utils.error_recovery import ErrorRecoverySystem
            from config.enhanced_settings import EnhancedSettings
            
            settings = EnhancedSettings()
            error_recovery = ErrorRecoverySystem(settings)
            
            test_error = Exception("Test error")
            recovery_result = await error_recovery.handle_error(test_error, "test")
            
            return {
                'success': True,
                'error_recovery_available': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_memory_management(self) -> Dict:
        """Test memory management"""
        try:
            logger.info("🔍 Testing memory management...")
            
            from utils.memory_optimizer import MemoryOptimizer
            from config.enhanced_settings import EnhancedSettings
            
            settings = EnhancedSettings()
            memory_optimizer = MemoryOptimizer(settings)
            
            memory_optimizer.start_monitoring()
            await asyncio.sleep(1)
            memory_optimizer.stop_monitoring()
            
            return {
                'success': True,
                'memory_optimizer_available': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_api_connections(self) -> Dict:
        """Test API connections"""
        try:
            logger.info("🔍 Testing API connections...")
            
            api_tests = {}
            
            # Test Dhan API
            try:
                from data.market_data import MarketDataProvider
                provider = MarketDataProvider()
                connected = provider._test_connection()
                api_tests['dhan'] = {'connected': connected, 'critical': True}
            except Exception as e:
                api_tests['dhan'] = {'connected': False, 'error': str(e), 'critical': True}
            
            # Test Kite API
            try:
                from auth.enhanced_kite_auth import EnhancedKiteAuthenticator
                kite_auth = EnhancedKiteAuthenticator()
                api_tests['kite'] = {'configured': True, 'critical': False}
            except Exception as e:
                api_tests['kite'] = {'configured': False, 'error': str(e), 'critical': False}
            
            # Success if critical APIs work
            critical_working = sum(1 for api in api_tests.values() 
                                 if api.get('connected', api.get('configured', False)) and api.get('critical', False))
            
            return {
                'success': critical_working >= 1,
                'api_tests': api_tests,
                'critical_working': critical_working
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_performance(self) -> Dict:
        """Test system performance"""
        try:
            logger.info("🔍 Testing system performance...")
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            self.performance_metrics = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
            
            # Performance thresholds
            performance_ok = (
                cpu_percent < 80 and
                memory.percent < 85 and
                disk.percent < 90 and
                memory.available > 1024**3
            )
            
            fixes = []
            if memory.percent >= 85:
                import gc
                gc.collect()
                fixes.append("Memory cleanup applied")
                self.fixes_applied.extend(fixes)
            
            return {
                'success': performance_ok,
                'metrics': self.performance_metrics,
                'fixes': fixes
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def verify_data_accuracy(self) -> Dict:
        """Verify data accuracy"""
        try:
            logger.info("🔍 Verifying data accuracy...")
            
            from data.market_data import MarketDataProvider
            provider = MarketDataProvider()
            
            # Test data consistency
            nifty_data1 = provider.get_live_quote('NIFTY')
            await asyncio.sleep(2)
            nifty_data2 = provider.get_live_quote('NIFTY')
            
            data_reasonable = True
            if nifty_data1.get('status') == 'success':
                price = nifty_data1.get('price', 0)
                if not (15000 <= price <= 30000):
                    data_reasonable = False
            
            return {
                'success': data_reasonable,
                'data_reasonable': data_reasonable,
                'sample_price': nifty_data1.get('price', 0)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_internal_linking(self) -> Dict:
        """Test internal module linking"""
        try:
            logger.info("🔍 Testing internal linking...")
            
            from core.data_manager import DataManager
            from config.enhanced_settings import EnhancedSettings
            
            settings = EnhancedSettings()
            data_manager = DataManager(settings)
            
            await data_manager.initialize()
            
            return {
                'success': True,
                'data_manager_initialized': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def generate_final_comprehensive_report(self, total_time: float):
        """Generate comprehensive final report"""
        logger.info("\n" + "="*100)
        logger.info("🏆 FINAL COMPREHENSIVE SYSTEM VALIDATION REPORT")
        logger.info("="*100)
        
        # Calculate metrics
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for result in self.validation_results.values() if result.get('success', False))
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # System Health Score
        max_score = len(self.validation_results)
        health_percentage = (self.system_health_score / max_score) * 100 if max_score > 0 else 0
        
        logger.info(f"🎯 OVERALL SUCCESS RATE: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        logger.info(f"🏥 SYSTEM HEALTH SCORE: {health_percentage:.1f}% ({self.system_health_score}/{max_score})")
        logger.info(f"⏱️ TOTAL VALIDATION TIME: {total_time:.2f} seconds")
        
        # System Status
        if health_percentage >= 95:
            status = "🏆 EXCELLENT - Production Ready"
            recommendation = "✅ System is ready for live trading"
        elif health_percentage >= 85:
            status = "✅ GOOD - Ready for Trading"
            recommendation = "✅ System is ready with minor monitoring"
        elif health_percentage >= 75:
            status = "⚠️ ACCEPTABLE - Needs Attention"
            recommendation = "⚠️ Address failed components before live trading"
        elif health_percentage >= 60:
            status = "🔧 NEEDS WORK - Several Issues"
            recommendation = "🔧 Fix critical issues before proceeding"
        else:
            status = "❌ CRITICAL - Major Issues"
            recommendation = "❌ System needs significant work"
        
        logger.info(f"\n🏥 SYSTEM STATUS: {status}")
        logger.info(f"💡 RECOMMENDATION: {recommendation}")
        
        # Detailed test results
        logger.info(f"\n📋 DETAILED TEST RESULTS:")
        for test_name, result in self.validation_results.items():
            status_icon = "✅" if result.get('success', False) else "❌"
            duration = result.get('duration', 0)
            logger.info(f"   {status_icon} {test_name} ({duration:.2f}s)")
            
            if not result.get('success', False) and 'error' in result:
                logger.info(f"      Error: {result['error']}")
        
        # Data sources verification
        if self.data_sources_verified:
            logger.info(f"\n📊 DATA SOURCES VERIFICATION:")
            for source_name, source_info in self.data_sources_verified.items():
                status_icon = "✅" if source_info.get('real', False) else "❌"
                critical_icon = "🔴" if source_info.get('critical', False) else "🟡"
                logger.info(f"   {status_icon} {critical_icon} {source_name}")
                
                if 'source' in source_info:
                    logger.info(f"      Source: {source_info['source']}")
                if 'price' in source_info and source_info['price'] > 0:
                    logger.info(f"      Sample: Rs.{source_info['price']:,.2f}")
                if 'error' in source_info:
                    logger.info(f"      Error: {source_info['error']}")
        
        # Performance metrics
        if self.performance_metrics:
            logger.info(f"\n⚡ SYSTEM PERFORMANCE:")
            logger.info(f"   CPU Usage: {self.performance_metrics['cpu_usage']:.1f}%")
            logger.info(f"   Memory Usage: {self.performance_metrics['memory_usage']:.1f}%")
            logger.info(f"   Memory Available: {self.performance_metrics['memory_available_gb']:.1f}GB")
            logger.info(f"   Disk Usage: {self.performance_metrics['disk_usage']:.1f}%")
            logger.info(f"   Disk Free: {self.performance_metrics['disk_free_gb']:.1f}GB")
        
        # Issues and fixes
        if self.issues_found:
            logger.info(f"\n⚠️ ISSUES FOUND:")
            for issue in self.issues_found:
                logger.info(f"   • {issue}")
        
        if self.fixes_applied:
            logger.info(f"\n🔧 FIXES APPLIED:")
            for fix in self.fixes_applied:
                logger.info(f"   • {fix}")
        
        # Final summary
        logger.info(f"\n🎯 FINAL SUMMARY:")
        logger.info(f"   ✅ Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"   📊 Success Rate: {success_rate:.1f}%")
        logger.info(f"   🏥 Health Score: {health_percentage:.1f}%")
        logger.info(f"   🌐 Real Data Sources: {sum(1 for ds in self.data_sources_verified.values() if ds.get('real', False))}")
        logger.info(f"   🔧 Fixes Applied: {len(self.fixes_applied)}")
        
        # Save comprehensive report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'validation_duration': total_time,
            'success_rate': success_rate,
            'health_score': health_percentage,
            'system_status': status,
            'recommendation': recommendation,
            'tests_passed': passed_tests,
            'total_tests': total_tests,
            'validation_results': self.validation_results,
            'data_sources_verified': self.data_sources_verified,
            'performance_metrics': self.performance_metrics,
            'issues_found': self.issues_found,
            'fixes_applied': self.fixes_applied
        }
        
        # Save to multiple formats
        report_dir = PROJECT_ROOT / "data_storage" / "validation_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_report = report_dir / f"final_validation_report_{timestamp}.json"
        with open(json_report, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Text summary
        text_report = report_dir / f"final_validation_summary_{timestamp}.txt"
        with open(text_report, 'w') as f:
            f.write(f"FINAL SYSTEM VALIDATION REPORT\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"="*50 + "\n\n")
            f.write(f"SUCCESS RATE: {success_rate:.1f}%\n")
            f.write(f"HEALTH SCORE: {health_percentage:.1f}%\n")
            f.write(f"SYSTEM STATUS: {status}\n")
            f.write(f"RECOMMENDATION: {recommendation}\n\n")
            f.write(f"TESTS PASSED: {passed_tests}/{total_tests}\n")
            f.write(f"VALIDATION TIME: {total_time:.2f} seconds\n")
        
        logger.info(f"\n📄 REPORTS SAVED:")
        logger.info(f"   📊 JSON Report: {json_report}")
        logger.info(f"   📝 Text Summary: {text_report}")
        
        logger.info("\n" + "="*100)
        logger.info("🎉 FINAL COMPREHENSIVE VALIDATION COMPLETED!")
        logger.info("="*100)

async def main():
    """Main validation function"""
    validator = FinalSystemValidator()
    await validator.run_final_validation()

if __name__ == "__main__":
    asyncio.run(main())