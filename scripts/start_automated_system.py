#!/usr/bin/env python3
"""
VLR_AI Trading System - Automated Startup Script
One-click startup for the fully automated trading system
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

# Import automation orchestrator
from core.automation_orchestrator import AutomationOrchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print system banner"""
    print("\n" + "=" * 80)
    print("VLR_AI TRADING SYSTEM - FULLY AUTOMATED")
    print("Professional Algorithmic Trading Platform")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Mode: {os.getenv('TRADING_MODE', 'PAPER')}")
    print(f"Primary Broker: {os.getenv('PRIMARY_BROKER', 'DHAN')}")
    print("=" * 80)

async def start_automated_system():
    """Start the fully automated trading system"""
    try:
        print_banner()
        
        print("\n[STARTUP] Initializing VLR_AI Trading System...")
        print("[INFO] All systems will be automatically managed")
        print("[INFO] No manual intervention required")
        
        # Create orchestrator
        orchestrator = AutomationOrchestrator()
        
        print("\n[INIT] Starting full system automation...")
        if await orchestrator.initialize_all_systems():
            print("[SUCCESS] All systems initialized successfully!")
            
            # Get system status
            status = orchestrator.get_automation_status()
            
            print("\n[STATUS] System Status:")
            for system, is_operational in status['automation_status'].items():
                status_text = "OPERATIONAL" if is_operational else "ERROR"
                print(f"   [{status_text}] {system.replace('_', ' ').title()}")
            
            health_summary = status.get('health_summary', {})
            print(f"\n[HEALTH] Overall Health: {health_summary.get('health_percentage', 0):.1f}%")
            print(f"[HEALTH] Healthy APIs: {health_summary.get('healthy_apis', 0)}/{health_summary.get('total_apis', 0)}")
            
            print("\n" + "=" * 80)
            print("SYSTEM IS NOW FULLY AUTOMATED AND OPERATIONAL!")
            print("=" * 80)
            print("Features Active:")
            print("  - Automated Kite authentication (daily refresh)")
            print("  - 24/7 API health monitoring")
            print("  - Auto-healing on failures")
            print("  - Real-time Telegram notifications")
            print("  - Scheduled maintenance")
            print("  - Emergency recovery procedures")
            print("\nThe system will now run autonomously.")
            print("Check Telegram for real-time updates and alerts.")
            print("=" * 80)
            
            # Keep system running
            print("\n[RUNNING] System is now running autonomously...")
            print("[INFO] Press Ctrl+C to stop the system")
            
            try:
                # Run indefinitely
                while True:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    current_time = datetime.now().strftime('%H:%M:%S')
                    print(f"[{current_time}] System running autonomously...")
                    
                    # Quick health check
                    status = orchestrator.get_automation_status()
                    health = status.get('health_summary', {}).get('health_percentage', 0)
                    if health < 80:
                        print(f"[WARNING] System health: {health:.1f}% - Auto-healing in progress...")
                    
            except KeyboardInterrupt:
                print("\n\n[SHUTDOWN] Shutdown signal received...")
                print("[INFO] Stopping automated systems...")
                
                await orchestrator.stop_automation()
                
                print("[SUCCESS] System shutdown completed successfully!")
                print("Thank you for using VLR_AI Trading System!")
                
        else:
            print("[ERROR] Failed to initialize systems")
            print("Please check the logs and fix any issues")
            return False
            
    except Exception as e:
        print(f"[ERROR] System startup failed: {e}")
        logger.error(f"System startup failed: {e}")
        return False
    
    return True

async def quick_health_check():
    """Perform a quick health check before startup"""
    try:
        print("\n[HEALTH] Performing pre-startup health check...")
        
        # Check environment variables
        required_vars = [
            'KITE_API_KEY', 'KITE_API_SECRET', 'KITE_USER_ID', 
            'KITE_PASSWORD', 'KITE_TOTP_SECRET',
            'DHAN_CLIENT_ID', 'DHAN_ACCESS_TOKEN',
            'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"[ERROR] Missing environment variables: {missing_vars}")
            return False
        
        print("[OK] All required environment variables present")
        
        # Check if logs directory exists
        logs_dir = PROJECT_ROOT / 'logs'
        logs_dir.mkdir(exist_ok=True)
        
        print("[OK] Log directory ready")
        
        # Check if auth directory exists
        auth_dir = PROJECT_ROOT / 'auth'
        auth_dir.mkdir(exist_ok=True)
        
        print("[OK] Auth directory ready")
        
        print("[SUCCESS] Pre-startup health check passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Health check failed: {e}")
        return False

def show_help():
    """Show help information"""
    print("\n" + "=" * 80)
    print("VLR_AI TRADING SYSTEM - HELP")
    print("=" * 80)
    print("\nUsage:")
    print("  python start_automated_system.py")
    print("\nFeatures:")
    print("  - Fully automated Kite authentication")
    print("  - 24/7 API health monitoring")
    print("  - Auto-healing capabilities")
    print("  - Real-time notifications")
    print("  - Scheduled maintenance")
    print("\nConfiguration:")
    print("  - Edit .env file for credentials")
    print("  - Set TRADING_MODE=PAPER for testing")
    print("  - Set TRADING_MODE=LIVE for real trading")
    print("\nSupport:")
    print("  - Check logs/ directory for detailed logs")
    print("  - Monitor Telegram for real-time alerts")
    print("  - System auto-heals most issues")
    print("=" * 80)

async def main():
    """Main execution"""
    try:
        # Check command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] in ['--help', '-h', 'help']:
                show_help()
                return
        
        # Perform health check
        if not await quick_health_check():
            print("\n[ERROR] Pre-startup health check failed!")
            print("Please fix the issues and try again.")
            return
        
        # Start automated system
        success = await start_automated_system()
        
        if success:
            print("\n[SUCCESS] System completed successfully!")
        else:
            print("\n[ERROR] System encountered errors!")
            
    except Exception as e:
        print(f"\n[ERROR] Execution failed: {e}")
        logger.error(f"Execution failed: {e}")

if __name__ == "__main__":
    # Ensure logs directory exists
    (PROJECT_ROOT / 'logs').mkdir(exist_ok=True)
    
    # Run the system
    asyncio.run(main())