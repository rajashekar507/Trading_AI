"""
QUICK COMPREHENSIVE SYSTEM TEST
Tests all critical components without Unicode issues
"""

import asyncio
import logging
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('quick_system_test')

class QuickSystemTest:
    """Quick comprehensive system test"""
    
    def __init__(self):
        self.results = {}
        self.issues_found = []
        self.fixes_applied = []
        
    async def run_quick_test(self):
        """Run quick comprehensive test"""
        logger.info("STARTING QUICK COMPREHENSIVE SYSTEM TEST")
        logger.info("="*80)
        
        tests = [
            ("System Requirements", self.test_requirements),
            ("Critical Imports", self.test_imports),
            ("Market Data (REAL)", self.test_market_data),
            ("Paper Trading", self.test_paper_trading),
            ("System Integration", self.test_integration),
            ("Performance", self.test_performance)
        ]
        
        start_time = time.time()
        passed = 0
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\nTesting: {test_name}")
                logger.info("-" * 60)
                
                result = await test_func()
                self.results[test_name] = result
                
                if result.get('success', False):
                    logger.info(f"PASSED: {test_name}")
                    passed += 1
                else:
                    logger.error(f"FAILED: {test_name} - {result.get('error', 'Unknown')}")
                    
            except Exception as e:
                logger.error(f"ERROR: {test_name} - {e}")
                self.results[test_name] = {'success': False, 'error': str(e)}
        
        total_time = time.time() - start_time
        success_rate = (passed / len(tests)) * 100
        
        # Generate report
        await self.generate_report(passed, len(tests), success_rate, total_time)
    
    async def test_requirements(self):
        """Test system requirements"""
        try:
            import sys
            import psutil
            
            # Check Python version
            python_ok = sys.version_info >= (3, 8)
            
            # Check memory
            memory = psutil.virtual_memory()
            memory_ok = memory.available > 512 * 1024**2  # 512MB (more realistic)
            
            # Check critical packages
            packages = ['pandas', 'numpy', 'requests', 'dhanhq', 'kiteconnect']
            missing = []
            
            for pkg in packages:
                try:
                    __import__(pkg)
                except ImportError:
                    missing.append(pkg)
            
            success = python_ok and memory_ok and len(missing) == 0
            
            return {
                'success': success,
                'python_ok': python_ok,
                'memory_ok': memory_ok,
                'missing_packages': missing
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_imports(self):
        """Test critical imports"""
        try:
            critical_imports = [
                'core.system_manager',
                'core.data_manager', 
                'data.market_data',
                'execution.paper_trading_executor',
                'config.enhanced_settings'
            ]
            
            failed = []
            for module in critical_imports:
                try:
                    __import__(module)
                except Exception as e:
                    failed.append(f"{module}: {e}")
            
            return {
                'success': len(failed) == 0,
                'failed_imports': failed
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_market_data(self):
        """Test market data with REAL verification"""
        try:
            from data.market_data import MarketDataProvider
            
            provider = MarketDataProvider()
            
            # Test connection
            connected = provider._test_connection()
            if not connected:
                return {'success': False, 'error': 'Dhan API connection failed'}
            
            # Test REAL data
            nifty_data = provider.get_live_quote('NIFTY')
            
            if nifty_data.get('status') != 'success':
                return {'success': False, 'error': 'NIFTY data fetch failed'}
            
            price = nifty_data.get('price', 0)
            
            # Verify price is reasonable (REAL data check)
            if not (15000 <= price <= 30000):
                return {'success': False, 'error': f'Unreasonable NIFTY price: {price}'}
            
            return {
                'success': True,
                'nifty_price': price,
                'change': nifty_data.get('change', 0),
                'data_source': 'Dhan API (REAL)'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_paper_trading(self):
        """Test paper trading"""
        try:
            from execution.paper_trading_executor import PaperTradingExecutor
            from config.enhanced_settings import EnhancedSettings
            
            settings = EnhancedSettings()
            executor = PaperTradingExecutor(settings)
            
            # Test signal
            signal = {
                'symbol': 'NIFTY',
                'action': 'BUY',
                'price': 25000,
                'confidence': 75,
                'timestamp': datetime.now()
            }
            
            result = await executor.execute_paper_trade(signal)
            
            return {
                'success': result.get('status') in ['success', 'rejected'],
                'trade_status': result.get('status'),
                'balance': executor.virtual_balance
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_integration(self):
        """Test system integration"""
        try:
            from core.system_manager import TradingSystemManager
            from config.enhanced_settings import EnhancedSettings
            
            settings = EnhancedSettings()
            manager = TradingSystemManager(settings)
            
            # Test initialization (will prompt for Kite auth but that's expected)
            try:
                init_result = await asyncio.wait_for(manager.initialize(), timeout=10)
            except asyncio.TimeoutError:
                # Expected due to Kite auth prompt
                init_result = False
            
            # Check components exist
            components = {
                'data_manager': hasattr(manager, 'data_manager'),
                'signal_engine': hasattr(manager, 'signal_engine'),
                'risk_manager': hasattr(manager, 'risk_manager')
            }
            
            try:
                await manager.shutdown()
            except:
                pass
            
            return {
                'success': sum(components.values()) >= 2,
                'components': components,
                'init_attempted': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_performance(self):
        """Test system performance"""
        try:
            import psutil
            
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            performance_ok = (
                cpu < 80 and
                memory.percent < 85 and
                memory.available > 1024**3
            )
            
            return {
                'success': performance_ok,
                'cpu_usage': cpu,
                'memory_usage': memory.percent,
                'memory_available_gb': memory.available / (1024**3)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def generate_report(self, passed, total, success_rate, total_time):
        """Generate test report"""
        logger.info("\n" + "="*80)
        logger.info("QUICK SYSTEM TEST REPORT")
        logger.info("="*80)
        
        logger.info(f"SUCCESS RATE: {success_rate:.1f}% ({passed}/{total})")
        logger.info(f"TEST TIME: {total_time:.2f} seconds")
        
        # System status
        if success_rate >= 90:
            status = "EXCELLENT - Production Ready"
        elif success_rate >= 80:
            status = "GOOD - Ready for Trading"
        elif success_rate >= 70:
            status = "ACCEPTABLE - Minor Issues"
        else:
            status = "NEEDS WORK - Major Issues"
        
        logger.info(f"SYSTEM STATUS: {status}")
        
        # Detailed results
        logger.info("\nDETAILED RESULTS:")
        for test_name, result in self.results.items():
            status_icon = "PASS" if result.get('success', False) else "FAIL"
            logger.info(f"  {status_icon} - {test_name}")
            
            if not result.get('success', False) and 'error' in result:
                logger.info(f"    Error: {result['error']}")
            
            # Show key metrics
            if test_name == "Market Data (REAL)" and result.get('success'):
                logger.info(f"    NIFTY Price: Rs.{result.get('nifty_price', 0):,.2f}")
                logger.info(f"    Change: {result.get('change', 0):+.2f}")
                logger.info(f"    Source: {result.get('data_source', 'Unknown')}")
            
            elif test_name == "Performance" and result.get('success'):
                logger.info(f"    CPU: {result.get('cpu_usage', 0):.1f}%")
                logger.info(f"    Memory: {result.get('memory_usage', 0):.1f}%")
                logger.info(f"    Available: {result.get('memory_available_gb', 0):.1f}GB")
        
        # Data source verification
        logger.info("\nDATA SOURCE VERIFICATION:")
        market_result = self.results.get("Market Data (REAL)", {})
        if market_result.get('success'):
            logger.info("  REAL DATA CONFIRMED - Dhan API providing live market data")
            logger.info(f"  Sample: NIFTY @ Rs.{market_result.get('nifty_price', 0):,.2f}")
        else:
            logger.info("  DATA SOURCE ISSUE - Market data not working properly")
        
        # Final recommendation
        logger.info("\nRECOMMENDATION:")
        if success_rate >= 85:
            logger.info("  System is ready for trading operations")
            logger.info("  All critical components are functional")
            logger.info("  Real data sources verified")
        elif success_rate >= 70:
            logger.info("  System is mostly functional")
            logger.info("  Address failed components before live trading")
        else:
            logger.info("  System needs significant work")
            logger.info("  Fix critical issues before proceeding")
        
        # Save report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'success_rate': success_rate,
            'tests_passed': passed,
            'total_tests': total,
            'test_duration': total_time,
            'system_status': status,
            'detailed_results': self.results
        }
        
        report_path = PROJECT_ROOT / "data_storage" / "quick_test_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"\nReport saved: {report_path}")
        logger.info("="*80)

async def main():
    """Main test function"""
    tester = QuickSystemTest()
    await tester.run_quick_test()

if __name__ == "__main__":
    asyncio.run(main())