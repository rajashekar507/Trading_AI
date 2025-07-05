# 🚀 VLR_AI Trading System - Complete Setup Guide

## 📋 **TABLE OF CONTENTS**

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Broker Setup](#broker-setup)
4. [Configuration](#configuration)
5. [First Run](#first-run)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## 🔧 **PREREQUISITES**

### **System Requirements**
- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: 3.11 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space
- **Internet**: Stable broadband connection

### **Required Accounts**
- **Zerodha Kite** account (for live trading)
- **Dhan** account (alternative broker)
- **Telegram** account (for notifications)
- **News API** account (optional, for news sentiment)

---

## 📦 **INSTALLATION**

### **Step 1: Clone Repository**
```bash
git clone https://github.com/yourusername/VLR_AI_Trading.git
cd VLR_AI_Trading
```

### **Step 2: Create Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Verify Installation**
```bash
python -c "import tensorflow, pandas, numpy; print('All packages installed successfully!')"
```

---

## 🏦 **BROKER SETUP**

### **Zerodha Kite Setup**

#### **1. Create Kite Connect App**
1. Login to [Kite Connect](https://kite.trade/)
2. Go to "My Apps" → "Create New App"
3. Fill in app details:
   - **App Name**: VLR_AI_Trading
   - **App Type**: Connect
   - **Redirect URL**: http://localhost:8080/callback
4. Note down **API Key** and **API Secret**

#### **2. Enable TOTP (Time-based OTP)**
1. Install Google Authenticator or similar app
2. Go to Kite web → Settings → Account → Two-factor authentication
3. Scan QR code and note down the **secret key**

### **Dhan Setup**

#### **1. Get API Credentials**
1. Login to [Dhan](https://dhan.co/)
2. Go to API section
3. Generate **Client ID** and **Access Token**

### **Telegram Setup**

#### **1. Create Telegram Bot**
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot` command
3. Follow instructions to create bot
4. Note down **Bot Token**

#### **2. Get Chat ID**
1. Start conversation with your bot
2. Send any message
3. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
4. Find your **Chat ID** in the response

---

## ⚙️ **CONFIGURATION**

### **Step 1: Environment Setup**
```bash
# Copy environment template
cp .env.example .env

# Edit configuration file
# Windows:
notepad .env
# Linux/macOS:
nano .env
```

### **Step 2: Fill in Credentials**
```env
# Zerodha Kite
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here
KITE_USER_ID=your_user_id_here
KITE_PASSWORD=your_password_here
KITE_TOTP_SECRET=your_totp_secret_here

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
TELEGRAM_ENABLED=true

# Trading Settings
TRADING_MODE=paper
AUTO_TRADING_ENABLED=false
MAX_POSITION_SIZE=10000
```

### **Step 3: Validate Configuration**
```bash
python -c "from config.enhanced_settings import EnhancedSettings; settings = EnhancedSettings(); print('Configuration loaded successfully!')"
```

---

## 🚀 **FIRST RUN**

### **Step 1: Demo Mode (Recommended)**
```bash
# Run in demo mode (no real trades)
python main.py --mode demo
```

### **Step 2: Paper Trading Mode**
```bash
# Run in paper trading mode (virtual money)
python main.py --mode paper
```

### **Step 3: System Health Check**
```bash
# Run comprehensive system test
python utils/complete_system_test.py
```

### **Step 4: Authentication Test**
```bash
# Test broker authentication
python -c "
from auth.enhanced_kite_auth import EnhancedKiteAuth
from config.enhanced_settings import EnhancedSettings
settings = EnhancedSettings()
auth = EnhancedKiteAuth(settings)
print('Authentication test completed')
"
```

---

## ✅ **VERIFICATION**

### **Check System Components**
```bash
# 1. Test market data
python -c "from data.market_data import MarketDataProvider; provider = MarketDataProvider(); print('Market data: OK')"

# 2. Test notifications
python -c "
import asyncio
from notifications.telegram_notifier import TelegramNotifier
from config.enhanced_settings import EnhancedSettings
async def test():
    settings = EnhancedSettings()
    telegram = TelegramNotifier(settings)
    result = await telegram.send_system_alert('TEST', 'System setup completed successfully!')
    print(f'Telegram test: {result}')
asyncio.run(test())
"

# 3. Test AI models
python -c "from ml.lstm_predictor import LSTMPredictor; predictor = LSTMPredictor(); print('AI models: OK')"
```

### **Verify File Structure**
```bash
# Check if all directories exist
python -c "
import os
dirs = ['analysis', 'auth', 'config', 'core', 'data', 'execution', 'ml', 'notifications', 'risk', 'strategies', 'utils']
for d in dirs:
    if os.path.exists(d):
        print(f'✅ {d}/')
    else:
        print(f'❌ {d}/ missing')
"
```

---

## 🔧 **TROUBLESHOOTING**

### **Common Issues**

#### **1. Import Errors**
```bash
# Problem: ModuleNotFoundError
# Solution: Ensure virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Reinstall dependencies
pip install -r requirements.txt
```

#### **2. Authentication Failures**
```bash
# Problem: Kite authentication fails
# Solution: Check credentials in .env file
# Ensure TOTP secret is correct
# Try manual authentication first
```

#### **3. TensorFlow Issues**
```bash
# Problem: TensorFlow warnings/errors
# Solution: Set environment variables
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
```

#### **4. Memory Issues**
```bash
# Problem: High memory usage
# Solution: Adjust settings in .env
MAX_MEMORY_USAGE_MB=512
CLEANUP_INTERVAL_HOURS=2
```

### **Debug Mode**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py --mode demo --debug
```

### **Log Analysis**
```bash
# Check system logs
tail -f logs/trading_system.log

# Check error logs
tail -f logs/errors.log
```

---

## 📊 **PERFORMANCE OPTIMIZATION**

### **System Optimization**
```env
# Add to .env for better performance
MAX_MEMORY_USAGE_MB=1024
CLEANUP_INTERVAL_HOURS=6
API_RATE_LIMIT_PER_MINUTE=60
```

### **Database Optimization**
```bash
# Clean up old data
python utils/system_cleanup.py

# Optimize database
python -c "
from database.redis_cache import RedisCache
cache = RedisCache()
cache.cleanup_expired()
print('Database optimized')
"
```

---

## 🔐 **SECURITY CHECKLIST**

### **Before Going Live**
- [ ] All credentials in .env file (not in code)
- [ ] .env file added to .gitignore
- [ ] Strong passwords used
- [ ] 2FA enabled on broker accounts
- [ ] API keys have minimal required permissions
- [ ] Regular backup of configuration
- [ ] System monitoring enabled

### **Ongoing Security**
- [ ] Regular password changes
- [ ] Monitor account activity
- [ ] Keep software updated
- [ ] Review logs regularly
- [ ] Backup trading data

---

## 📞 **SUPPORT**

### **Getting Help**
1. **Documentation**: Check all files in `docs/` folder
2. **GitHub Issues**: Report bugs and request features
3. **Community**: Join discussions
4. **Email**: support@vlr-ai.com

### **Before Asking for Help**
1. Check this setup guide
2. Review error logs
3. Try debug mode
4. Search existing issues
5. Provide detailed error information

---

## 🎯 **NEXT STEPS**

### **After Successful Setup**
1. **Paper Trading**: Test strategies with virtual money
2. **Performance Analysis**: Monitor system metrics
3. **Strategy Optimization**: Tune parameters
4. **Risk Management**: Set appropriate limits
5. **Live Trading**: Start with small positions

### **Recommended Learning Path**
1. Understand the system architecture
2. Study the trading strategies
3. Learn risk management principles
4. Practice with paper trading
5. Gradually increase position sizes

---

**🚀 Congratulations! Your VLR_AI Trading System is ready to use.**

*For the latest updates and documentation, visit our GitHub repository.*
