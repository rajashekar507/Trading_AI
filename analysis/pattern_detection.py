"""
INSTITUTIONAL-GRADE Pattern Recognition with Machine Learning
Advanced candlestick patterns + ML-based pattern detection
"""

import logging
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('trading_system.pattern_detection')

class PatternDetector:
    """INSTITUTIONAL-GRADE Pattern Recognition with Machine Learning"""
    
    def __init__(self, kite_client=None):
        self.kite = kite_client
        
        # Traditional candlestick patterns
        self.patterns = {
            'doji': self._detect_doji,
            'hammer': self._detect_hammer,
            'shooting_star': self._detect_shooting_star,
            'engulfing_bullish': self._detect_bullish_engulfing,
            'engulfing_bearish': self._detect_bearish_engulfing,
            'morning_star': self._detect_morning_star,
            'evening_star': self._detect_evening_star,
            'piercing_line': self._detect_piercing_line,
            'dark_cloud': self._detect_dark_cloud
        }
        
        # Machine Learning Models
        self.ml_models = {
            'price_direction': None,
            'volatility_regime': None,
            'pattern_classifier': None
        }
        
        # Feature engineering parameters
        self.lookback_periods = [5, 10, 20, 50]
        self.scaler = StandardScaler()
        
        logger.info("[ML] INSTITUTIONAL-GRADE Pattern Detector initialized with ML capabilities")
    
    async def detect_patterns(self, symbol: str, timeframe: str = 'day') -> Dict:
        """Detect candlestick patterns for given symbol"""
        pattern_data = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'failed',
            'patterns': {},
            'signal': 'neutral',
            'confidence': 0
        }
        
        try:
            if not self.kite:
                logger.error("[ERROR] STRICT ENFORCEMENT: No Kite client available - CANNOT DETECT PATTERNS")
                pattern_data['error'] = 'No Kite client - strict enforcement mode'
                return pattern_data
            
            df = await self._fetch_historical_data(symbol, timeframe)
            if df is None or len(df) < 10:
                logger.error(f"[ERROR] STRICT ENFORCEMENT: Insufficient data for pattern detection - {symbol}")
                pattern_data['error'] = 'Insufficient historical data'
                return pattern_data
            
            detected_patterns = {}
            for pattern_name, detector_func in self.patterns.items():
                try:
                    result = detector_func(df)
                    if result:
                        detected_patterns[pattern_name] = result
                except Exception as e:
                    logger.warning(f"[WARNING]️ Pattern detection failed for {pattern_name}: {e}")
            
            pattern_data['patterns'] = detected_patterns
            pattern_data['signal'], pattern_data['confidence'] = self._calculate_pattern_signal(detected_patterns)
            pattern_data['status'] = 'success'
            
            logger.info(f"[OK] Pattern detection completed for {symbol}: {len(detected_patterns)} patterns found")
            
        except Exception as e:
            logger.error(f"[ERROR] STRICT ENFORCEMENT: Pattern detection failed for {symbol}: {e}")
            pattern_data['error'] = str(e)
        
        return pattern_data
    
    async def _fetch_historical_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch historical data for pattern analysis"""
        try:
            instruments = self.kite.instruments("NSE")
            instrument_token = None
            
            for instrument in instruments:
                if instrument['name'] == f"{symbol} 50" if symbol == 'NIFTY' else f"{symbol} BANK":
                    if instrument['segment'] == 'INDICES':
                        instrument_token = instrument['instrument_token']
                        break
            
            if not instrument_token:
                logger.error(f"[ERROR] Instrument token not found for {symbol}")
                return None
            
            from_date = datetime.now() - timedelta(days=30)
            to_date = datetime.now()
            
            historical_data = self.kite.historical_data(
                instrument_token,
                from_date,
                to_date,
                timeframe
            )
            
            if not historical_data:
                return None
            
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"[ERROR] Historical data fetch failed: {e}")
            return None
    
    def _detect_doji(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Doji patterns"""
        try:
            latest = df.iloc[-1]
            open_price = latest['open']
            close_price = latest['close']
            high_price = latest['high']
            low_price = latest['low']
            
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            
            if total_range == 0:
                return None
            
            body_ratio = body_size / total_range
            
            if body_ratio <= 0.1:
                return {
                    'type': 'doji',
                    'signal': 'reversal',
                    'strength': 1 - body_ratio,
                    'description': 'Market indecision, potential reversal'
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_hammer(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Hammer patterns"""
        try:
            latest = df.iloc[-1]
            open_price = latest['open']
            close_price = latest['close']
            high_price = latest['high']
            low_price = latest['low']
            
            body_size = abs(close_price - open_price)
            lower_shadow = min(open_price, close_price) - low_price
            upper_shadow = high_price - max(open_price, close_price)
            total_range = high_price - low_price
            
            if total_range == 0:
                return None
            
            if (lower_shadow >= 2 * body_size and 
                upper_shadow <= body_size * 0.5 and
                body_size / total_range >= 0.1):
                
                return {
                    'type': 'hammer',
                    'signal': 'bullish',
                    'strength': lower_shadow / total_range,
                    'description': 'Bullish reversal pattern'
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_shooting_star(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Shooting Star patterns"""
        try:
            latest = df.iloc[-1]
            open_price = latest['open']
            close_price = latest['close']
            high_price = latest['high']
            low_price = latest['low']
            
            body_size = abs(close_price - open_price)
            lower_shadow = min(open_price, close_price) - low_price
            upper_shadow = high_price - max(open_price, close_price)
            total_range = high_price - low_price
            
            if total_range == 0:
                return None
            
            if (upper_shadow >= 2 * body_size and 
                lower_shadow <= body_size * 0.5 and
                body_size / total_range >= 0.1):
                
                return {
                    'type': 'shooting_star',
                    'signal': 'bearish',
                    'strength': upper_shadow / total_range,
                    'description': 'Bearish reversal pattern'
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_bullish_engulfing(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Bullish Engulfing patterns"""
        try:
            if len(df) < 2:
                return None
            
            prev_candle = df.iloc[-2]
            curr_candle = df.iloc[-1]
            
            prev_bearish = prev_candle['close'] < prev_candle['open']
            curr_bullish = curr_candle['close'] > curr_candle['open']
            
            if (prev_bearish and curr_bullish and
                curr_candle['open'] < prev_candle['close'] and
                curr_candle['close'] > prev_candle['open']):
                
                engulfing_ratio = (curr_candle['close'] - curr_candle['open']) / (prev_candle['open'] - prev_candle['close'])
                
                return {
                    'type': 'bullish_engulfing',
                    'signal': 'bullish',
                    'strength': min(engulfing_ratio, 2.0) / 2.0,
                    'description': 'Strong bullish reversal pattern'
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_bearish_engulfing(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Bearish Engulfing patterns"""
        try:
            if len(df) < 2:
                return None
            
            prev_candle = df.iloc[-2]
            curr_candle = df.iloc[-1]
            
            prev_bullish = prev_candle['close'] > prev_candle['open']
            curr_bearish = curr_candle['close'] < curr_candle['open']
            
            if (prev_bullish and curr_bearish and
                curr_candle['open'] > prev_candle['close'] and
                curr_candle['close'] < prev_candle['open']):
                
                engulfing_ratio = (curr_candle['open'] - curr_candle['close']) / (prev_candle['close'] - prev_candle['open'])
                
                return {
                    'type': 'bearish_engulfing',
                    'signal': 'bearish',
                    'strength': min(engulfing_ratio, 2.0) / 2.0,
                    'description': 'Strong bearish reversal pattern'
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_morning_star(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Morning Star patterns"""
        try:
            if len(df) < 3:
                return None
            
            first = df.iloc[-3]
            second = df.iloc[-2]
            third = df.iloc[-1]
            
            first_bearish = first['close'] < first['open']
            second_small = abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3
            third_bullish = third['close'] > third['open']
            
            gap_down = second['high'] < first['close']
            gap_up = third['open'] > second['high']
            
            if (first_bearish and second_small and third_bullish and gap_down and gap_up):
                return {
                    'type': 'morning_star',
                    'signal': 'bullish',
                    'strength': 0.8,
                    'description': 'Strong bullish reversal pattern'
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_evening_star(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Evening Star patterns"""
        try:
            if len(df) < 3:
                return None
            
            first = df.iloc[-3]
            second = df.iloc[-2]
            third = df.iloc[-1]
            
            first_bullish = first['close'] > first['open']
            second_small = abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3
            third_bearish = third['close'] < third['open']
            
            gap_up = second['low'] > first['close']
            gap_down = third['open'] < second['low']
            
            if (first_bullish and second_small and third_bearish and gap_up and gap_down):
                return {
                    'type': 'evening_star',
                    'signal': 'bearish',
                    'strength': 0.8,
                    'description': 'Strong bearish reversal pattern'
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_piercing_line(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Piercing Line patterns"""
        try:
            if len(df) < 2:
                return None
            
            prev_candle = df.iloc[-2]
            curr_candle = df.iloc[-1]
            
            prev_bearish = prev_candle['close'] < prev_candle['open']
            curr_bullish = curr_candle['close'] > curr_candle['open']
            
            midpoint = (prev_candle['open'] + prev_candle['close']) / 2
            
            if (prev_bearish and curr_bullish and
                curr_candle['open'] < prev_candle['close'] and
                curr_candle['close'] > midpoint):
                
                penetration = (curr_candle['close'] - midpoint) / (prev_candle['open'] - prev_candle['close'])
                
                return {
                    'type': 'piercing_line',
                    'signal': 'bullish',
                    'strength': min(penetration, 1.0),
                    'description': 'Bullish reversal pattern'
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_dark_cloud(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Dark Cloud Cover patterns"""
        try:
            if len(df) < 2:
                return None
            
            prev_candle = df.iloc[-2]
            curr_candle = df.iloc[-1]
            
            prev_bullish = prev_candle['close'] > prev_candle['open']
            curr_bearish = curr_candle['close'] < curr_candle['open']
            
            midpoint = (prev_candle['open'] + prev_candle['close']) / 2
            
            if (prev_bullish and curr_bearish and
                curr_candle['open'] > prev_candle['close'] and
                curr_candle['close'] < midpoint):
                
                penetration = (midpoint - curr_candle['close']) / (prev_candle['close'] - prev_candle['open'])
                
                return {
                    'type': 'dark_cloud',
                    'signal': 'bearish',
                    'strength': min(penetration, 1.0),
                    'description': 'Bearish reversal pattern'
                }
            
            return None
            
        except Exception:
            return None
    
    def _calculate_pattern_signal(self, patterns: Dict) -> tuple:
        """Calculate overall pattern signal and confidence"""
        if not patterns:
            return 'neutral', 0
        
        bullish_strength = 0
        bearish_strength = 0
        total_patterns = 0
        
        for pattern_data in patterns.values():
            strength = pattern_data.get('strength', 0)
            signal = pattern_data.get('signal', 'neutral')
            
            if signal == 'bullish':
                bullish_strength += strength
            elif signal == 'bearish':
                bearish_strength += strength
            
            total_patterns += 1
        
        if total_patterns == 0:
            return 'neutral', 0
        
        if bullish_strength > bearish_strength:
            confidence = (bullish_strength / total_patterns) * 100
            return 'bullish', min(confidence, 100)
        elif bearish_strength > bullish_strength:
            confidence = (bearish_strength / total_patterns) * 100
            return 'bearish', min(confidence, 100)
        else:
            return 'neutral', 0
    
    def extract_ml_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        INSTITUTIONAL-GRADE Feature Engineering for ML Models
        Extract comprehensive features from price data
        """
        try:
            features = []
            
            # Price-based features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['high_low_ratio'] = df['high'] / df['low']
            df['open_close_ratio'] = df['open'] / df['close']
            
            # Volatility features
            for period in self.lookback_periods:
                df[f'volatility_{period}'] = df['returns'].rolling(period).std()
                df[f'price_range_{period}'] = (df['high'] - df['low']).rolling(period).mean()
            
            # Technical indicators as features
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            df['macd'] = ta.trend.MACD(df['close']).macd()
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.volatility.BollingerBands(df['close']).bollinger_hband(), ta.volatility.BollingerBands(df['close']).bollinger_mavg(), ta.volatility.BollingerBands(df['close']).bollinger_lband()
            
            # Volume features (if available)
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
                for period in [5, 10, 20]:
                    df[f'volume_trend_{period}'] = df['volume'].rolling(period).mean() / df['volume'].rolling(period*2).mean()
            
            # Price pattern features
            for period in self.lookback_periods:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'price_vs_sma_{period}'] = df['close'] / df[f'sma_{period}']
                df[f'momentum_{period}'] = df['close'] / df['close'].shift(period)
            
            # Candlestick pattern features
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            df['total_range'] = df['high'] - df['low']
            
            # Relative features
            df['body_to_range'] = df['body_size'] / (df['total_range'] + 1e-8)
            df['upper_shadow_to_body'] = df['upper_shadow'] / (df['body_size'] + 1e-8)
            df['lower_shadow_to_body'] = df['lower_shadow'] / (df['body_size'] + 1e-8)
            
            # Market structure features
            df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
            df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
            df['lower_high'] = (df['high'] < df['high'].shift(1)).astype(int)
            df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
            
            # Select feature columns (exclude NaN rows)
            feature_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            feature_df = df[feature_columns].dropna()
            
            if len(feature_df) == 0:
                return np.array([])
            
            return feature_df.values
            
        except Exception as e:
            logger.error(f"[ERROR] Feature extraction failed: {e}")
            return np.array([])
    
    def train_price_direction_model(self, df: pd.DataFrame) -> bool:
        """
        Train ML model to predict price direction
        """
        try:
            if len(df) < 100:  # Need sufficient data
                logger.warning("[ML] Insufficient data for training price direction model")
                return False
            
            # Extract features
            features = self.extract_ml_features(df.copy())
            if len(features) == 0:
                return False
            
            # Create target variable (next day's direction)
            df['next_return'] = df['close'].shift(-1) / df['close'] - 1
            df['direction'] = (df['next_return'] > 0).astype(int)  # 1 for up, 0 for down
            
            # Align features with targets
            target = df['direction'].dropna().values
            min_len = min(len(features), len(target))
            
            if min_len < 50:
                return False
            
            X = features[:min_len]
            y = target[:min_len]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train ensemble model
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6)
            
            rf_model.fit(X_train_scaled, y_train)
            gb_model.fit(X_train_scaled, y_train)
            
            # Evaluate models
            rf_score = rf_model.score(X_test_scaled, y_test)
            gb_score = gb_model.score(X_test_scaled, y_test)
            
            # Use best performing model
            if rf_score > gb_score:
                self.ml_models['price_direction'] = rf_model
                logger.info(f"[ML] Random Forest model trained - Accuracy: {rf_score:.3f}")
            else:
                self.ml_models['price_direction'] = gb_model
                logger.info(f"[ML] Gradient Boosting model trained - Accuracy: {gb_score:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Price direction model training failed: {e}")
            return False
    
    def predict_price_direction(self, df: pd.DataFrame) -> Dict:
        """
        Predict price direction using trained ML model
        """
        try:
            if self.ml_models['price_direction'] is None:
                # Try to train model first
                if not self.train_price_direction_model(df):
                    return {'prediction': 'neutral', 'confidence': 0, 'error': 'Model not available'}
            
            # Extract features for latest data
            features = self.extract_ml_features(df.copy())
            if len(features) == 0:
                return {'prediction': 'neutral', 'confidence': 0, 'error': 'Feature extraction failed'}
            
            # Use latest features for prediction
            latest_features = features[-1:].reshape(1, -1)
            scaled_features = self.scaler.transform(latest_features)
            
            # Make prediction
            model = self.ml_models['price_direction']
            prediction = model.predict(scaled_features)[0]
            prediction_proba = model.predict_proba(scaled_features)[0]
            
            # Convert to readable format
            direction = 'bullish' if prediction == 1 else 'bearish'
            confidence = max(prediction_proba) * 100
            
            return {
                'prediction': direction,
                'confidence': confidence,
                'probability_up': prediction_proba[1] * 100,
                'probability_down': prediction_proba[0] * 100,
                'model_type': type(model).__name__
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Price direction prediction failed: {e}")
            return {'prediction': 'neutral', 'confidence': 0, 'error': str(e)}
    
    def detect_volatility_regime(self, df: pd.DataFrame) -> Dict:
        """
        INSTITUTIONAL-GRADE Volatility Regime Detection
        Identifies high/low volatility periods for strategy selection
        """
        try:
            if len(df) < 50:
                return {'regime': 'unknown', 'confidence': 0}
            
            # Calculate rolling volatility
            df['returns'] = df['close'].pct_change()
            df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252)  # Annualized
            df['volatility_50'] = df['returns'].rolling(50).std() * np.sqrt(252)
            
            current_vol = df['volatility_20'].iloc[-1]
            long_term_vol = df['volatility_50'].iloc[-1]
            
            # Historical volatility percentiles
            vol_percentile = df['volatility_20'].rank(pct=True).iloc[-1]
            
            # Determine regime
            if vol_percentile > 0.8:
                regime = 'high_volatility'
                confidence = (vol_percentile - 0.8) * 500  # Scale to 0-100
            elif vol_percentile < 0.2:
                regime = 'low_volatility'
                confidence = (0.2 - vol_percentile) * 500
            else:
                regime = 'normal_volatility'
                confidence = 100 - abs(vol_percentile - 0.5) * 200
            
            return {
                'regime': regime,
                'confidence': min(confidence, 100),
                'current_volatility': current_vol,
                'long_term_volatility': long_term_vol,
                'volatility_percentile': vol_percentile * 100
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Volatility regime detection failed: {e}")
            return {'regime': 'unknown', 'confidence': 0}
    
    async def advanced_pattern_analysis(self, symbol: str) -> Dict:
        """
        INSTITUTIONAL-GRADE Pattern Analysis combining traditional + ML
        """
        try:
            # Get traditional pattern analysis
            traditional_patterns = await self.detect_patterns(symbol)
            
            # Get historical data for ML analysis
            df = await self._fetch_historical_data(symbol, 'day')
            if df is None or len(df) < 50:
                return traditional_patterns
            
            # ML-based predictions
            ml_prediction = self.predict_price_direction(df)
            volatility_regime = self.detect_volatility_regime(df)
            
            # Combine traditional and ML analysis
            combined_analysis = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'traditional_patterns': traditional_patterns,
                'ml_prediction': ml_prediction,
                'volatility_regime': volatility_regime,
                'status': 'success'
            }
            
            # Generate final signal combining both approaches
            traditional_signal = traditional_patterns.get('signal', 'neutral')
            traditional_confidence = traditional_patterns.get('confidence', 0)
            
            ml_signal = ml_prediction.get('prediction', 'neutral')
            ml_confidence = ml_prediction.get('confidence', 0)
            
            # Weighted combination (60% ML, 40% traditional)
            if traditional_signal == ml_signal and traditional_signal != 'neutral':
                # Both agree - high confidence
                final_signal = traditional_signal
                final_confidence = (ml_confidence * 0.6 + traditional_confidence * 0.4) * 1.2
            elif traditional_signal != 'neutral' and ml_signal != 'neutral':
                # Disagreement - lower confidence, prefer ML
                final_signal = ml_signal
                final_confidence = ml_confidence * 0.7
            elif ml_signal != 'neutral':
                # Only ML has signal
                final_signal = ml_signal
                final_confidence = ml_confidence * 0.8
            elif traditional_signal != 'neutral':
                # Only traditional has signal
                final_signal = traditional_signal
                final_confidence = traditional_confidence * 0.6
            else:
                # Both neutral
                final_signal = 'neutral'
                final_confidence = 0
            
            combined_analysis.update({
                'final_signal': final_signal,
                'final_confidence': min(final_confidence, 100),
                'analysis_method': 'traditional_ml_combined'
            })
            
            logger.info(f"[ML_PATTERN] {symbol}: {final_signal} ({final_confidence:.1f}% confidence)")
            return combined_analysis
            
        except Exception as e:
            logger.error(f"[ERROR] Advanced pattern analysis failed: {e}")
            return {'status': 'failed', 'error': str(e)}
