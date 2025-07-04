# VLR_AI Trading System - Setup Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_new.txt
```

### 2. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# Add your Kite Connect API credentials
# Add Telegram bot token (optional)
```

### 3. Run the System
```bash
# Start live trading
python main.py

# Or run in different modes
python main.py --mode demo        # Safe demo mode
python main.py --mode backtest    # Backtesting
python main.py --mode autonomous  # Autonomous mode
python main.py --mode dashboard   # Dashboard only
```

## 📁 Directory Structure

```
VLR_AI_Trading_System/
├── main.py                 # Main entry point - RUN THIS FILE
├── requirements_new.txt    # Dependencies
├── config/                 # Configuration files
├── core/                   # Core business logic
├── strategies/             # Trading strategies
├── analysis/               # Market analysis
├── execution/              # Trade execution
├── risk/                   # Risk management
├── data/                   # Data handling
├── auth/                   # Authentication
├── utils/                  # Utilities
├── autonomous/             # Autonomous system
├── dashboard/              # Web dashboard
├── logs/                   # Log files
├── data_storage/           # Data storage
├── backups/                # System backups
└── docs/                   # Documentation
```

## 🎯 Entry Points

### Main Entry Point
```bash
python main.py
```
This is the ONLY file you need to run!

### Available Modes
- `--mode live` (default): Live trading
- `--mode demo`: Safe demo mode
- `--mode backtest`: Historical backtesting
- `--mode autonomous`: 24/7 autonomous mode
- `--mode dashboard`: Web dashboard only

### Validation
```bash
python main.py --validate
```

## 🔧 Configuration

### Environment Variables (.env)
```
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret
KITE_ACCESS_TOKEN=your_access_token
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Trading Configuration (config/settings.py)
- Risk management parameters
- Position sizing rules
- Strategy parameters
- Market hours and timing

## 🛡️ Safety Features

### Demo Mode
- No real trades executed
- Safe testing environment
- All components tested

### Risk Management
- Maximum position size limits
- Stop-loss protection
- Portfolio-level risk controls
- Real-time monitoring

### Autonomous Mode
- 24/7 system monitoring
- Automatic issue fixing
- Performance optimization
- Emergency recovery

## 📊 Monitoring

### Logs
- `logs/trading_YYYYMMDD.log` - Main trading logs
- `logs/errors_YYYYMMDD.log` - Error logs
- `logs/trading_system_YYYYMMDD.log` - System logs

### Database
- `data_storage/databases/` - SQLite databases
- System health metrics
- Trade history
- Performance data

### Telegram Notifications
- Real-time trade alerts
- System status updates
- Error notifications
- Performance reports

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements_new.txt
   ```

2. **API Connection Issues**
   - Check .env file
   - Verify API credentials
   - Check internet connection

3. **Permission Errors**
   - Ensure write permissions for logs/ and data_storage/
   - Run as administrator if needed

4. **System Validation**
   ```bash
   python main.py --validate
   ```

### Getting Help
1. Check logs in `logs/` directory
2. Run validation: `python main.py --validate`
3. Try demo mode: `python main.py --mode demo`

## 🎯 Next Steps

1. **Start with Demo Mode**
   ```bash
   python main.py --mode demo
   ```

2. **Run Backtesting**
   ```bash
   python main.py --mode backtest
   ```

3. **Go Live**
   ```bash
   python main.py --mode live
   ```

4. **Enable Autonomous Mode**
   ```bash
   python main.py --mode autonomous
   ```

## ⚠️ Important Notes

- Always test in demo mode first
- Ensure proper risk management settings
- Monitor system logs regularly
- Keep backups of configuration
- Use autonomous mode for 24/7 operation

## 🏆 Success Indicators

✅ System validation passes  
✅ Demo mode runs successfully  
✅ Telegram notifications working  
✅ Logs being generated  
✅ Database initialized  
✅ All modules importing correctly  

Your VLR_AI Trading System is ready for professional trading!