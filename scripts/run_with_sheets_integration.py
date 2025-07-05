#!/usr/bin/env python3
"""
VLR_AI Trading System with Google Sheets Integration
Main entry point for running the complete system with real-time Google Sheets logging
"""

import asyncio
import sys
import signal
import os
from pathlib import Path
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.enhanced_settings import EnhancedSettings as Settings
from core.system_manager import TradingSystemManager
from utils.sheets_integration_service import SheetsIntegrationService
from utils.system_monitor import SystemMonitor
from utils.logger import setup_logging

class VLRTradingSystemWithSheets:
    """VLR_AI Trading System with Google Sheets Integration"""
    
    def __init__(self):
        self.settings = Settings()
        self.system_manager = None
        self.sheets_service = None
        self.system_monitor = None
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("[VLR_AI] Initializing Trading System with Google Sheets Integration...")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n[SHUTDOWN] Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def initialize(self):
        """Initialize all system components"""
        try:
            print("[INIT] Setting up logging system...")
            setup_logging()
            
            print("[INIT] Initializing core trading system...")
            self.system_manager = TradingSystemManager(self.settings)
            
            # Initialize core system first
            core_success = await self.system_manager.initialize()
            if not core_success:
                print("[ERROR] Core trading system initialization failed")
                return False
            
            print("[INIT] Initializing Google Sheets integration...")
            self.sheets_service = SheetsIntegrationService(
                self.settings, 
                self.system_manager.kite_client
            )
            
            sheets_success = await self.sheets_service.initialize()
            if not sheets_success:
                print("[WARNING] Google Sheets integration failed - continuing without it")
                self.sheets_service = None
            else:
                print("[OK] Google Sheets integration initialized successfully")
                
                # Connect sheets service to system manager
                self.system_manager.sheets_service = self.sheets_service
            
            print("[INIT] Initializing system monitoring...")
            self.system_monitor = SystemMonitor(self.settings, self.sheets_service)
            
            print("[SUCCESS] All systems initialized successfully!")
            return True
            
        except Exception as e:
            print(f"[ERROR] System initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_live_trading(self):
        """Run live trading with Google Sheets integration"""
        try:
            if not await self.initialize():
                print("[ERROR] System initialization failed")
                return False
            
            self.running = True
            
            print("\n" + "=" * 80)
            print("VLR_AI TRADING SYSTEM WITH GOOGLE SHEETS INTEGRATION")
            print("=" * 80)
            print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Google Sheets: {'ENABLED' if self.sheets_service else 'DISABLED'}")
            print(f"System Monitor: {'ENABLED' if self.system_monitor else 'DISABLED'}")
            
            # Get dashboard URL from environment or config
            dashboard_url = os.getenv('GOOGLE_SHEETS_DASHBOARD_URL', 
                                    getattr(self.settings, 'GOOGLE_SHEETS_DASHBOARD_URL', 'Not configured'))
            if dashboard_url != 'Not configured':
                print(f"Dashboard: {dashboard_url}")
            else:
                print("Dashboard: Configure GOOGLE_SHEETS_DASHBOARD_URL in .env file")
            print("=" * 80)
            
            # Start background tasks
            background_tasks = []
            
            # Start Google Sheets continuous updates
            if self.sheets_service:
                print("[START] Starting Google Sheets continuous updates...")
                background_tasks.append(
                    asyncio.create_task(self.sheets_service.start_continuous_updates())
                )
            
            # Start system monitoring
            if self.system_monitor:
                print("[START] Starting system monitoring...")
                background_tasks.append(
                    asyncio.create_task(self.system_monitor.start_monitoring())
                )
            
            # Start main trading system
            print("[START] Starting main trading system...")
            background_tasks.append(
                asyncio.create_task(self.system_manager.run_live_trading())
            )
            
            print("\n[RUNNING] All systems operational - Press Ctrl+C to stop")
            print("-" * 80)
            
            # Wait for all tasks or shutdown signal
            try:
                await asyncio.gather(*background_tasks, return_exceptions=True)
            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Keyboard interrupt received")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Live trading error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            await self._shutdown()
    
    async def run_single_cycle(self):
        """Run a single trading cycle for testing"""
        try:
            if not await self.initialize():
                print("[ERROR] System initialization failed")
                return False
            
            print("\n[TEST] Running single trading cycle with Google Sheets logging...")
            
            # Run one cycle
            success = await self.system_manager.run_institutional_cycle()
            
            if success:
                print("[SUCCESS] Single cycle completed successfully")
                
                # Show integration status
                if self.sheets_service:
                    status = await self.sheets_service.get_integration_status()
                    print(f"[INFO] Sheets Status: {status.get('sheets_status', {}).get('status', 'Unknown')}")
                    print(f"[INFO] Trade History: {status.get('trade_history_count', 0)} signals")
                    print(f"[INFO] Rejected Signals: {status.get('rejected_signals_count', 0)} signals")
            
            return success
            
        except Exception as e:
            print(f"[ERROR] Single cycle error: {e}")
            return False
        finally:
            await self._shutdown()
    
    async def test_integration(self):
        """Test Google Sheets integration without running trading"""
        try:
            print("[TEST] Testing Google Sheets integration...")
            
            # Initialize only sheets service
            setup_logging()
            self.sheets_service = SheetsIntegrationService(self.settings)
            
            success = await self.sheets_service.initialize()
            if not success:
                print("[ERROR] Google Sheets integration test failed")
                return False
            
            # Test logging sample data
            print("[TEST] Logging sample data...")
            
            sample_signal = {
                'signal_id': f'TEST_{datetime.now().strftime("%H%M%S")}',
                'instrument': 'NIFTY',
                'direction': 'CE',
                'strike_price': 25400,
                'expiry_date': '2025-06-26',
                'entry_price': 150.0,
                'stop_loss': 120.0,
                'target_1': 180.0,
                'target_2': 200.0,
                'confidence_score': 85,
                'risk_score': 25,
                'reason_summary': 'Integration test signal',
                'status': 'Pending'
            }
            
            await self.sheets_service.log_trade_signal(sample_signal)
            print("[OK] Sample signal logged successfully")
            
            # Get status
            status = await self.sheets_service.get_integration_status()
            print(f"[INFO] Integration Status: {status.get('sheets_status', {}).get('status', 'Unknown')}")
            
            print("[SUCCESS] Google Sheets integration test completed")
            return True
            
        except Exception as e:
            print(f"[ERROR] Integration test failed: {e}")
            return False
        finally:
            if self.sheets_service:
                await self.sheets_service.stop()
    
    async def _shutdown(self):
        """Graceful shutdown of all components"""
        try:
            print("\n[SHUTDOWN] Initiating graceful shutdown...")
            
            # Stop system monitor
            if self.system_monitor:
                await self.system_monitor.stop_monitoring()
                print("[OK] System monitor stopped")
            
            # Stop sheets service
            if self.sheets_service:
                await self.sheets_service.stop()
                print("[OK] Google Sheets service stopped")
            
            # Stop system manager
            if self.system_manager:
                self.system_manager.stop()
                print("[OK] Trading system stopped")
            
            print("[SHUTDOWN] All systems shutdown completed")
            
        except Exception as e:
            print(f"[ERROR] Shutdown error: {e}")

async def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_with_sheets_integration.py live     # Run live trading")
        print("  python run_with_sheets_integration.py test     # Run single cycle test")
        print("  python run_with_sheets_integration.py sheets   # Test sheets integration only")
        return
    
    mode = sys.argv[1].lower()
    system = VLRTradingSystemWithSheets()
    
    if mode == "live":
        print("[MODE] Live Trading with Google Sheets Integration")
        success = await system.run_live_trading()
    elif mode == "test":
        print("[MODE] Single Cycle Test")
        success = await system.run_single_cycle()
    elif mode == "sheets":
        print("[MODE] Google Sheets Integration Test")
        success = await system.test_integration()
    else:
        print(f"[ERROR] Unknown mode: {mode}")
        return
    
    if success:
        print("\n[COMPLETE] System execution completed successfully")
    else:
        print("\n[FAILED] System execution failed")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] System interrupted by user")
    except Exception as e:
        print(f"\n[CRITICAL] System crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)