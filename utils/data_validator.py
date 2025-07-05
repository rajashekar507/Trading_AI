from datetime import datetime, timedelta

class DataValidationError(Exception):
    pass

class DataValidator:
    def __init__(self, settings=None, max_age_minutes=5):
        self.settings = settings
        from datetime import timedelta
        self.max_age = timedelta(minutes=max_age_minutes)

    def is_safe_to_trade(self, market_data, current_positions=None, risk_metrics=None):
        try:
            errors = []
            
            # Check market hours (9:15 AM to 3:30 PM IST)
            now = datetime.now()
            market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            if not (market_start <= now <= market_end):
                # Allow trading in extended hours for testing
                if now.weekday() >= 5:  # Weekend
                    errors.append("Market closed - Weekend")
            
            # Validate market data freshness
            if market_data:
                timestamp = market_data.get('timestamp')
                if timestamp:
                    if isinstance(timestamp, str):
                        try:
                            from dateutil.parser import parse
                            timestamp = parse(timestamp)
                        except:
                            errors.append("Invalid timestamp format")
                    
                    if isinstance(timestamp, datetime):
                        age = (now - timestamp).total_seconds()
                        if age > 300:  # 5 minutes
                            errors.append(f"Market data is stale ({age:.0f}s old)")
            
            # Check for extreme market conditions
            if market_data:
                nifty_price = market_data.get('nifty_spot') or market_data.get('nifty', {}).get('ltp')
                if nifty_price:
                    if nifty_price < 15000 or nifty_price > 30000:
                        errors.append(f"NIFTY price unusual: {nifty_price}")
                
                banknifty_price = market_data.get('banknifty_spot') or market_data.get('banknifty', {}).get('ltp')
                if banknifty_price:
                    if banknifty_price < 30000 or banknifty_price > 70000:
                        errors.append(f"BANKNIFTY price unusual: {banknifty_price}")
                
                vix = market_data.get('india_vix') or market_data.get('vix')
                if vix and (vix < 8 or vix > 80):
                    errors.append(f"VIX unusual: {vix}")
            
            # Check position limits
            if current_positions and len(current_positions) > 10:
                errors.append(f"Too many positions: {len(current_positions)}")
            
            # Check risk metrics
            if risk_metrics:
                if risk_metrics.get('portfolio_risk', 0) > 0.05:  # 5% max risk
                    errors.append("Portfolio risk too high")
            
            is_safe = len(errors) == 0
            message = "Safe to trade" if is_safe else "; ".join(errors)
            
            return is_safe, message
            
        except Exception as e:
            return False, f"Safety check error: {e}"

    def validate_market_data(self, data):
        errors = []
        now = datetime.now()
        # Example: NIFTY/BANKNIFTY spot validation
        for key, minval, maxval in [("nifty_spot", 10000, 26000), ("banknifty_spot", 20000, 60000)]:
            spot = data.get(key)
            if spot is None or not (minval <= spot <= maxval):
                errors.append(f"{key} value out of range: {spot}")
        vix = data.get("india_vix")
        if vix is None or not (10 < vix < 60):
            errors.append(f"VIX value out of range: {vix}")
        # Timestamp check (if present)
        ts = data.get("timestamp")
        if ts:
            if isinstance(ts, str):
                try:
                    from dateutil.parser import parse
                    ts = parse(ts)
                except Exception:
                    errors.append("Timestamp is not a valid datetime")
            if isinstance(ts, datetime) and (now - ts) > self.max_age:
                errors.append("Market data is stale")
        return (len(errors) == 0, errors)

    def validate_options_data(self, options_chain):
        errors = []
        for option in options_chain:
            if "strike_price" not in option or "option_type" not in option:
                errors.append("Incomplete option data")
            if option.get("ltp") is None or option.get("volume") is None or option.get("oi") is None:
                errors.append(f"Missing ltp/volume/oi for strike {option.get('strike_price')}")
        return (len(errors) == 0, errors)
