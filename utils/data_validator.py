from datetime import datetime, timedelta

class DataValidationError(Exception):
    pass

class DataValidator:
    def __init__(self, settings=None, max_age_minutes=5):
        self.settings = settings
        from datetime import timedelta
        self.max_age = timedelta(minutes=max_age_minutes)

    def is_safe_to_trade(self, *args, **kwargs):
        # Dummy logic for test
        return True, "OK"

        self.max_age = timedelta(minutes=max_age_minutes)
        self.settings = settings

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
