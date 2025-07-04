class TradingSystemError(Exception): pass
class DataFetchError(TradingSystemError): pass
class RiskViolationError(TradingSystemError): pass
class APIConnectionError(TradingSystemError): pass
class DataValidationError(TradingSystemError): pass
