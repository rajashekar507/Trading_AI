"""
Ensemble Machine Learning Predictor
Combines multiple ML models for robust predictions
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ml.lstm_predictor import LSTMPredictor

logger = logging.getLogger('trading_system.ensemble_predictor')

class EnsemblePredictor:
    """Advanced Ensemble ML Predictor combining multiple models"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.models = {}
        self.scalers = {}
        self.model_weights = {}
        self.is_trained = False
        self.last_training_time = None
        
        # Feature engineering parameters
        self.lookback_periods = [5, 10, 20, 50]
        self.technical_indicators = ['rsi', 'macd', 'bb_position', 'volume_ratio']
        
        if not SKLEARN_AVAILABLE:
            logger.warning("[WARNING] Scikit-learn not available. Install with: pip install scikit-learn")
            return
        
        # Initialize models
        self._initialize_models()
        
        # Initialize LSTM predictor
        self.lstm_predictor = LSTMPredictor(settings)
        
        logger.info("[ML] Ensemble Predictor initialized with multiple ML models")
    
    def _initialize_models(self):
        """Initialize all ML models in the ensemble"""
        try:
            self.models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'linear_regression': LinearRegression(),
                'svr': SVR(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale'
                )
            }
            
            # Initialize scalers for each model
            for model_name in self.models.keys():
                self.scalers[model_name] = StandardScaler()
            
            # Initial equal weights
            num_models = len(self.models)
            for model_name in self.models.keys():
                self.model_weights[model_name] = 1.0 / num_models
            
            logger.info(f"[ML] Initialized {len(self.models)} models in ensemble")
            
        except Exception as e:
            logger.error(f"[ML] Model initialization failed: {e}")
    
    def engineer_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering for ML models"""
        try:
            data = market_data.copy()
            
            # Price-based features
            for period in self.lookback_periods:
                if len(data) > period:
                    data[f'return_{period}'] = data['close'].pct_change(period)
                    data[f'volatility_{period}'] = data['close'].rolling(period).std()
                    data[f'sma_{period}'] = data['close'].rolling(period).mean()
                    data[f'price_position_{period}'] = (data['close'] - data[f'sma_{period}']) / data[f'sma_{period}']
            
            # Technical indicators
            if 'rsi' not in data.columns:
                data['rsi'] = self._calculate_rsi(data['close'])
            
            if 'macd' not in data.columns:
                data['macd'] = self._calculate_macd(data['close'])
            
            # Bollinger Bands position
            bb_upper, bb_lower = self._calculate_bollinger_bands(data['close'])
            bb_middle = (bb_upper + bb_lower) / 2
            data['bb_position'] = (data['close'] - bb_middle) / (bb_upper - bb_lower)
            
            # Volume indicators
            if 'volume' in data.columns:
                data['volume_sma'] = data['volume'].rolling(20).mean()
                data['volume_ratio'] = data['volume'] / data['volume_sma']
            else:
                data['volume_ratio'] = 1.0
            
            # Price momentum
            data['momentum_5'] = data['close'] / data['close'].shift(5) - 1
            data['momentum_10'] = data['close'] / data['close'].shift(10) - 1
            
            # High-Low spread
            if 'high' in data.columns and 'low' in data.columns:
                data['hl_spread'] = (data['high'] - data['low']) / data['close']
            else:
                data['hl_spread'] = 0.01
            
            # Target variable (next period return)
            data['target'] = data['close'].shift(-1) / data['close'] - 1
            
            # Fill NaN values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"[ML] Feature engineering completed. Features: {len(data.columns)}")
            return data
            
        except Exception as e:
            logger.error(f"[ML] Feature engineering failed: {e}")
            return market_data
    
    def train_ensemble(self, market_data: pd.DataFrame) -> bool:
        """Train all models in the ensemble"""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("[ML] Scikit-learn not available for training")
                return False
            
            logger.info("[ML] Starting ensemble model training...")
            
            # Engineer features
            engineered_data = self.engineer_features(market_data)
            
            # Select feature columns (exclude target and non-numeric columns)
            feature_columns = [col for col in engineered_data.columns 
                             if col not in ['target', 'timestamp', 'symbol'] 
                             and engineered_data[col].dtype in ['float64', 'int64']]
            
            if len(feature_columns) < 5:
                logger.warning("[ML] Insufficient features for training")
                return False
            
            # Prepare training data
            X = engineered_data[feature_columns].dropna()
            y = engineered_data.loc[X.index, 'target'].dropna()
            
            # Align X and y
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            if len(X) < 100:
                logger.warning("[ML] Insufficient data for training")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            model_performances = {}
            
            # Train each model
            for model_name, model in self.models.items():
                try:
                    # Scale features
                    X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                    X_test_scaled = self.scalers[model_name].transform(X_test)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate model
                    y_pred = model.predict(X_test_scaled)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    model_performances[model_name] = {
                        'mse': mse,
                        'r2': r2,
                        'accuracy': max(0, r2)  # Use R² as accuracy measure
                    }
                    
                    logger.info(f"[ML] {model_name} trained - MSE: {mse:.6f}, R²: {r2:.4f}")
                    
                except Exception as e:
                    logger.warning(f"[ML] Failed to train {model_name}: {e}")
                    model_performances[model_name] = {'mse': 1.0, 'r2': 0.0, 'accuracy': 0.0}
            
            # Update model weights based on performance
            self._update_model_weights(model_performances)
            
            # Train LSTM model
            lstm_trained = self.lstm_predictor.train_model(market_data)
            if lstm_trained:
                logger.info("[ML] LSTM model training completed")
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            logger.info("[ML] Ensemble training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"[ML] Ensemble training failed: {e}")
            return False
    
    def _update_model_weights(self, performances: Dict):
        """Update model weights based on performance"""
        try:
            # Calculate weights based on R² scores
            total_performance = sum(max(0, perf['r2']) for perf in performances.values())
            
            if total_performance > 0:
                for model_name, perf in performances.items():
                    self.model_weights[model_name] = max(0, perf['r2']) / total_performance
            else:
                # Equal weights if no model performs well
                num_models = len(self.models)
                for model_name in self.models.keys():
                    self.model_weights[model_name] = 1.0 / num_models
            
            logger.info(f"[ML] Model weights updated: {self.model_weights}")
            
        except Exception as e:
            logger.error(f"[ML] Weight update failed: {e}")
    
    def predict_ensemble(self, recent_data: pd.DataFrame) -> Optional[Dict]:
        """Generate ensemble prediction combining all models"""
        try:
            if not self.is_trained:
                logger.warning("[ML] Ensemble not trained. Cannot make predictions.")
                return None
            
            # Engineer features for recent data
            engineered_data = self.engineer_features(recent_data)
            
            # Select feature columns
            feature_columns = [col for col in engineered_data.columns 
                             if col not in ['target', 'timestamp', 'symbol'] 
                             and engineered_data[col].dtype in ['float64', 'int64']]
            
            if len(feature_columns) < 5:
                logger.warning("[ML] Insufficient features for prediction")
                return None
            
            # Get latest features
            latest_features = engineered_data[feature_columns].iloc[-1:].fillna(0)
            
            predictions = {}
            valid_predictions = []
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                try:
                    # Scale features
                    features_scaled = self.scalers[model_name].transform(latest_features)
                    
                    # Make prediction
                    pred = model.predict(features_scaled)[0]
                    predictions[model_name] = pred
                    
                    # Weight the prediction
                    weighted_pred = pred * self.model_weights[model_name]
                    valid_predictions.append(weighted_pred)
                    
                except Exception as e:
                    logger.warning(f"[ML] {model_name} prediction failed: {e}")
            
            if not valid_predictions:
                logger.warning("[ML] No valid predictions from ensemble models")
                return None
            
            # Calculate ensemble prediction
            ensemble_prediction = sum(valid_predictions)
            
            # Get LSTM prediction
            lstm_prediction = self.lstm_predictor.predict_price(recent_data)
            
            # Combine ensemble and LSTM predictions
            if lstm_prediction:
                lstm_change = lstm_prediction['predicted_change_pct'] / 100
                final_prediction = (ensemble_prediction * 0.6) + (lstm_change * 0.4)
                confidence = (70 + lstm_prediction['confidence']) / 2
            else:
                final_prediction = ensemble_prediction
                confidence = 60
            
            current_price = recent_data['close'].iloc[-1]
            predicted_price = current_price * (1 + final_prediction)
            predicted_change_pct = final_prediction * 100
            
            # Determine signal
            if predicted_change_pct > 0.5:
                signal = 'BUY'
            elif predicted_change_pct < -0.5:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            # Adjust confidence based on prediction magnitude
            confidence = min(85, max(15, confidence - abs(predicted_change_pct) * 1.5))
            
            result = {
                'predicted_price': float(predicted_price),
                'current_price': float(current_price),
                'predicted_change_pct': float(predicted_change_pct),
                'confidence': float(confidence),
                'signal': signal,
                'timestamp': datetime.now(),
                'model_type': 'ENSEMBLE',
                'individual_predictions': predictions,
                'model_weights': self.model_weights
            }
            
            logger.info(f"[ML] Ensemble Prediction: {predicted_price:.2f} ({predicted_change_pct:+.2f}%) - {signal}")
            
            return result
            
        except Exception as e:
            logger.error(f"[ML] Ensemble prediction failed: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd.fillna(0)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band.fillna(prices), lower_band.fillna(prices)
    
    def get_ensemble_info(self) -> Dict:
        """Get ensemble information and status"""
        return {
            'is_trained': self.is_trained,
            'last_training_time': self.last_training_time,
            'sklearn_available': SKLEARN_AVAILABLE,
            'models_count': len(self.models),
            'model_weights': self.model_weights,
            'lstm_info': self.lstm_predictor.get_model_info()
        }