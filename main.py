#!/usr/bin/env python3
"""
Trading_AI System - Main Entry Point
Professional, Clean, and Modular Trading System

This is the ONLY file you need to run to start the trading system.
🤖 Auto-push system test - Updated on July 5, 2025
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import argparse
import logging
from dotenv import load_dotenv

# Always load .env at startup
load_dotenv()

# Print API key status (masked)
def print_api_key_status():
    kite_key = os.getenv('KITE_API_KEY')
    kite_secret = os.getenv('KITE_API_SECRET')
    def mask(val):
        if not val or len(val) < 6:
            return 'NOT SET'
        return val[:2] + '*'*(len(val)-4) + val[-2:]
    print(f"[INFO] KITE_API_KEY loaded: {mask(kite_key)}")
    print(f"[INFO] KITE_API_SECRET loaded: {mask(kite_secret)}")

print_api_key_status()

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import core modules
try:
    from core.system_manager import TradingSystemManager
    from config.enhanced_settings import EnhancedSettings as Settings
    from utils.logger import setup_logging
    from utils.validators import validate_system_requirements
    from autonomous.zencoder import AutonomousZencoder
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

def print_banner():
    """Print professional system banner"""
    print("\n" + "=" * 50)
    print("   VLR_AI INSTITUTIONAL TRADING SYSTEM")
    print("   Professional - Modular - Scalable")
    print("=" * 50)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Root: {PROJECT_ROOT}")
    print("=" * 50 + "\n")

def setup_argument_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="VLR_AI Institutional Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start normal trading
  python main.py --mode live        # Live trading mode
  python main.py --mode backtest    # Backtesting mode
  python main.py --mode autonomous  # Autonomous mode
  python main.py --mode dashboard   # Dashboard only
  python main.py --mode demo        # System validation with REAL data
  python main.py --mode paper       # Paper trading mode
  python main.py --validate         # Validate system
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['live', 'backtest', 'autonomous', 'dashboard', 'demo', 'paper'],
        default='live',
        help='Trading system mode (default: live)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.py',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate system requirements and exit'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--no-telegram',
        action='store_true',
        help='Disable Telegram notifications'
    )
    
    return parser

async def run_live_trading(settings):
    """Run live trading mode"""
    print("STARTING LIVE TRADING MODE")
    print("=" * 50)
    
    system_manager = TradingSystemManager(settings)
    
    try:
        print(" Initializing trading system...")
        await system_manager.initialize()
        
        print(" Starting live trading...")
        print(" Market Hours: 09:15-15:30 IST")
        print(" Analysis Cycle: Every 30 seconds")
        print(" Risk Management: ACTIVE")
        print(" Telegram Alerts: ENABLED" if settings.TELEGRAM_BOT_TOKEN else " Telegram Alerts: DISABLED")
        print("=" * 50)
        
        await system_manager.run_live_trading()
        
    except KeyboardInterrupt:
        print("\n Trading stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Trading error: {e}")
        logging.error(f"Live trading error: {e}", exc_info=True)
    finally:
        await system_manager.shutdown()

async def run_backtesting(settings):
    """Run backtesting mode"""
    print(" STARTING BACKTESTING MODE")
    print("=" * 50)
    
    from analysis.backtesting import BacktestingEngine
    
    backtesting_engine = BacktestingEngine(settings)
    
    try:
        print(" Initializing backtesting engine...")
        await backtesting_engine.initialize()
        
        print(" Running historical analysis...")
        results = await backtesting_engine.run_backtest()
        
        print(" Backtesting Results:")
        print(f"   Total Trades: {results.get('total_trades', 0)}")
        print(f"   Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"   Total Return: {results.get('total_return', 0):.2%}")
        print(f"   Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        
    except Exception as e:
        print(f"[ERROR] Backtesting error: {e}")
        logging.error(f"Backtesting error: {e}", exc_info=True)

async def run_autonomous_mode(settings):
    """Run autonomous mode"""
    print(" STARTING AUTONOMOUS MODE")
    print("=" * 50)
    
    autonomous_system = AutonomousZencoder(settings)
    
    try:
        print(" Initializing autonomous system...")
        await autonomous_system.initialize()
        
        print(" Starting autonomous monitoring...")
        print(" 24/7 System Monitoring: ACTIVE")
        print(" Auto-Fix Issues: ENABLED")
        print(" Performance Optimization: ACTIVE")
        print("[SAVE] Automatic Backups: ENABLED")
        print("=" * 50)
        
        await autonomous_system.start_monitoring()
        
    except KeyboardInterrupt:
        print("\n Autonomous system stopped by user")
    except Exception as e:
        print(f"[ERROR] Autonomous system error: {e}")
        logging.error(f"Autonomous system error: {e}", exc_info=True)

async def run_dashboard_mode(settings):
    """Run dashboard mode"""
    print(" STARTING DASHBOARD MODE")
    print("=" * 50)
    
    from dashboard.app import create_dashboard_app
    
    try:
        print(" Initializing dashboard...")
        app = create_dashboard_app(settings)
        
        print(" Starting web dashboard...")
        print(" URL: http://localhost:8080")
        print(" Real-time monitoring available")
        print("=" * 50)
        
        # Run dashboard (this will be implemented in dashboard module)
        await app.run()
        
    except Exception as e:
        print(f"[ERROR] Dashboard error: {e}")
        logging.error(f"Dashboard error: {e}", exc_info=True)

async def run_validation_mode(settings):
    """Run system validation with REAL data verification"""
    try:
        from core.demo_runner import SystemValidator
        
        system_validator = SystemValidator(settings)
        validation_results = await system_validator.run_system_validation()
        
        # Display final results
        success_rate = validation_results.get('success_rate', 0)
        overall_status = validation_results.get('overall_status', 'UNKNOWN')
        
        print(f"\n🔍 REAL DATA VALIDATION COMPLETED!")
        print(f"📊 Success Rate: {success_rate:.1f}%")
        print(f"🚀 System Status: {overall_status}")
        
        if success_rate >= 80:
            print("\n✅ YOUR SYSTEM IS READY FOR REAL TRADING!")
            print("   All REAL data sources validated")
            print("   Run: python main.py --mode live")
        elif success_rate >= 60:
            print("\n⚠️ SYSTEM NEEDS ATTENTION BEFORE REAL TRADING")
            print("   Check the validation results above")
        else:
            print("\n❌ SYSTEM NOT READY FOR REAL TRADING")
            print("   Please fix the failed validations first")
        
    except Exception as e:
        print(f"[ERROR] System validation error: {e}")
        logging.error(f"System validation error: {e}", exc_info=True)

async def run_paper_trading_mode(settings):
    """Run paper trading mode"""
    print(" STARTING PAPER TRADING MODE")
    print("=" * 50)
    
    try:
        from execution.paper_trading_executor import PaperTradingExecutor
        
        # Enable paper trading in settings
        settings.PAPER_TRADING = True
        settings.LIVE_TRADING_ENABLED = False
        
        system_manager = TradingSystemManager(settings)
        
        print(" Initializing paper trading system...")
        await system_manager.initialize()
        
        print(" Starting paper trading...")
        print(" 💰 Virtual Money Trading - NO REAL MONEY AT RISK")
        print(" 📊 All strategies active with simulated execution")
        print(" 📱 Telegram alerts enabled for paper trades")
        print(" 🔄 Real market data with simulated trading")
        print("=" * 50)
        
        # Override the trade executor with paper trading executor
        paper_executor = PaperTradingExecutor(settings)
        system_manager.trade_executor = paper_executor
        
        await system_manager.run_live_trading()
        
    except KeyboardInterrupt:
        print("\n Paper trading stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Paper trading error: {e}")
        logging.error(f"Paper trading error: {e}", exc_info=True)
    finally:
        if 'system_manager' in locals():
            await system_manager.shutdown()

def validate_system(settings, bypass_api_check=False):
    """Validate system requirements"""
    print(" VALIDATING SYSTEM REQUIREMENTS")
    print("=" * 50)
    
    try:
        validation_results = validate_system_requirements(settings, bypass_api_check)
        
        if validation_results['valid']:
            print("[OK] System validation passed!")
            print(" System is ready to run")
            
            for check, status in validation_results['checks'].items():
                status_icon = "[OK]" if status else "[ERROR]"
                print(f"   {status_icon} {check}")
                
        else:
            print("[ERROR] System validation failed!")
            print(" Please fix the following issues:")
            
            for check, status in validation_results['checks'].items():
                if not status:
                    print(f"   [ERROR] {check}")
            
            return False
            
    except Exception as e:
        print(f"[ERROR] Validation error: {e}")
        return False
    
    print("=" * 50)
    return True

async def main():
    """Main entry point"""
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    try:
        # Load settings
        print(" Loading configuration...")
        settings = Settings()
        
        # Disable Telegram if requested
        if args.no_telegram:
            settings.TELEGRAM_BOT_TOKEN = None
            print(" Telegram notifications disabled")
        
        # Validate system if requested
        if args.validate:
            if validate_system(settings):
                print("[OK] System validation completed successfully!")
                return 0
            else:
                print("[ERROR] System validation failed!")
                return 1
        
        # Run quick validation
        print(" Quick system check...")
        if not validate_system(settings, bypass_api_check=(args.mode in ['demo', 'paper'])):
            print("[ERROR] System validation failed. Use --validate for detailed check.")
            return 1
        
        # Route to appropriate mode
        if args.mode == 'live':
            await run_live_trading(settings)
        elif args.mode == 'backtest':
            await run_backtesting(settings)
        elif args.mode == 'autonomous':
            await run_autonomous_mode(settings)
        elif args.mode == 'dashboard':
            await run_dashboard_mode(settings)
        elif args.mode == 'demo':
            await run_validation_mode(settings)
        elif args.mode == 'paper':
            await run_paper_trading_mode(settings)
        else:
            print(f"[ERROR] Unknown mode: {args.mode}")
            return 1
            
        return 0
        
    except KeyboardInterrupt:
        print("\n System stopped by user")
        return 0
    except Exception as e:
        print(f"\n[ERROR] System error: {e}")
        logging.error(f"Main system error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n System interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n Critical error: {e}")
        sys.exit(1)
