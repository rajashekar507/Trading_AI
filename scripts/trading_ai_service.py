
import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import sys
import os
import subprocess
import time
import logging
from pathlib import Path

class Trading_AI_Service(win32serviceutil.ServiceFramework):
    _svc_name_ = "Trading_AI_Service"
    _svc_display_name_ = "Trading AI Professional Trading System"
    _svc_description_ = "Institutional-grade algorithmic trading system running 24/7"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.is_alive = True
        self.trading_ai_path = r"C:\Users\RAJASHEKAR REDDY\OneDrive\Desktop\Trading_AI"
        
        # Setup logging
        logging.basicConfig(
            filename=os.path.join(self.trading_ai_path, 'logs', 'service.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('Trading_AI_Service')

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.is_alive = False
        self.logger.info("Trading_AI Service stopped")

    def SvcDoRun(self):
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        self.logger.info("Trading_AI Service started")
        self.main()

    def main(self):
        while self.is_alive:
            try:
                # Check if market hours (9:15 AM - 3:30 PM IST)
                from datetime import datetime, time
                now = datetime.now()
                market_start = time(9, 15)
                market_end = time(15, 30)
                current_time = now.time()
                
                # Only run during market hours on weekdays
                if (now.weekday() < 5 and 
                    market_start <= current_time <= market_end):
                    
                    self.logger.info("Market hours - Starting trading system")
                    
                    # Run trading system
                    cmd = [
                        sys.executable, 
                        os.path.join(self.trading_ai_path, 'main.py'),
                        '--mode', 'live',
                        '--service'
                    ]
                    
                    process = subprocess.Popen(
                        cmd,
                        cwd=self.trading_ai_path,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    # Monitor process
                    while process.poll() is None and self.is_alive:
                        time.sleep(30)  # Check every 30 seconds
                    
                    if process.returncode != 0:
                        self.logger.error(f"Trading system exited with code {process.returncode}")
                        # Auto-restart after 60 seconds
                        time.sleep(60)
                    
                else:
                    self.logger.info("Outside market hours - Sleeping")
                    time.sleep(300)  # Sleep 5 minutes outside market hours
                    
            except Exception as e:
                self.logger.error(f"Service error: {e}")
                time.sleep(60)  # Wait 1 minute before retry
                
            # Check stop event
            if win32event.WaitForSingleObject(self.hWaitStop, 1000) == win32event.WAIT_OBJECT_0:
                break

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(Trading_AI_Service)
