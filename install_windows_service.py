"""
Windows Service Installation for Trading_AI System
Creates a 24/7 background service that runs independently
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def create_windows_service():
    """Create Windows Service for 24/7 Trading_AI operation"""
    
    service_script = f"""
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
        self.trading_ai_path = r"{os.getcwd()}"
        
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
                        self.logger.error(f"Trading system exited with code {{process.returncode}}")
                        # Auto-restart after 60 seconds
                        time.sleep(60)
                    
                else:
                    self.logger.info("Outside market hours - Sleeping")
                    time.sleep(300)  # Sleep 5 minutes outside market hours
                    
            except Exception as e:
                self.logger.error(f"Service error: {{e}}")
                time.sleep(60)  # Wait 1 minute before retry
                
            # Check stop event
            if win32event.WaitForSingleObject(self.hWaitStop, 1000) == win32event.WAIT_OBJECT_0:
                break

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(Trading_AI_Service)
"""
    
    # Write service script
    service_file = Path("trading_ai_service.py")
    with open(service_file, 'w') as f:
        f.write(service_script)
    
    print("✅ Windows Service script created")
    
    # Create installation batch file
    install_script = f"""
@echo off
echo Installing Trading_AI Windows Service...
python trading_ai_service.py install
echo Starting Trading_AI Service...
python trading_ai_service.py start
echo ✅ Trading_AI Service installed and started
pause
"""
    
    with open("install_service.bat", 'w') as f:
        f.write(install_script)
    
    print("✅ Service installation script created")
    print("📋 To install: Run 'install_service.bat' as Administrator")

def create_file_watcher():
    """Create file system watcher for automatic commits"""
    
    watcher_script = f"""
import time
import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from datetime import datetime

class TradingAIFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_commit = time.time()
        self.commit_cooldown = 300  # 5 minutes between commits
        
        logging.basicConfig(
            filename='logs/file_watcher.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('FileWatcher')

    def on_modified(self, event):
        if event.is_directory:
            return
            
        # Only watch Python files
        if not event.src_path.endswith('.py'):
            return
            
        # Skip log files and temporary files
        if any(skip in event.src_path for skip in ['logs/', '__pycache__/', '.git/', 'temp']):
            return
            
        current_time = time.time()
        if current_time - self.last_commit > self.commit_cooldown:
            self.auto_commit(event.src_path)
            self.last_commit = current_time

    def auto_commit(self, file_path):
        try:
            # Get file name for commit message
            file_name = os.path.basename(file_path)
            
            # Auto-commit changes
            subprocess.run(['git', 'add', file_path], check=True)
            
            commit_msg = f"🤖 Auto-commit: Updated {{file_name}} - {{datetime.now().strftime('%Y-%m-%d %H:%M')}}"
            
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
            subprocess.run(['git', 'push', 'origin', 'master'], check=True)
            
            self.logger.info(f"Auto-committed: {{file_name}}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Auto-commit failed: {{e}}")

def start_file_watcher():
    event_handler = TradingAIFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path='{os.getcwd()}', recursive=True)
    observer.start()
    
    print("🔍 File watcher started - Monitoring for changes...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("🛑 File watcher stopped")
    
    observer.join()

if __name__ == "__main__":
    start_file_watcher()
"""
    
    with open("file_watcher.py", 'w') as f:
        f.write(watcher_script)
    
    print("✅ File watcher created")

def create_scheduled_tasks():
    """Create Windows scheduled tasks"""
    
    # Daily backup task
    backup_task = f"""
schtasks /create /tn "Trading_AI_Daily_Backup" /tr "python {os.getcwd()}\\utils\\auto_backup.py" /sc daily /st 18:00 /f
schtasks /create /tn "Trading_AI_System_Monitor" /tr "python {os.getcwd()}\\utils\\system_monitor.py" /sc minute /mo 5 /f
schtasks /create /tn "Trading_AI_Health_Check" /tr "python {os.getcwd()}\\main.py --validate" /sc hourly /f
"""
    
    with open("create_scheduled_tasks.bat", 'w') as f:
        f.write(backup_task)
    
    print("✅ Scheduled tasks script created")

if __name__ == "__main__":
    print("🚀 Installing Trading_AI Background Automation...")
    
    try:
        create_windows_service()
        create_file_watcher()
        create_scheduled_tasks()
        
        print("\n" + "="*60)
        print("✅ BACKGROUND AUTOMATION INSTALLED SUCCESSFULLY!")
        print("="*60)
        print("📋 Next Steps:")
        print("1. Run 'install_service.bat' as Administrator")
        print("2. Run 'create_scheduled_tasks.bat' as Administrator") 
        print("3. Start file watcher: python file_watcher.py")
        print("4. System will now run 24/7 automatically!")
        
    except Exception as e:
        print(f"❌ Installation failed: {e}")