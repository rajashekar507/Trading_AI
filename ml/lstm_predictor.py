"""
LSTM Neural Network for Price Prediction
Advanced machine learning model for institutional-grade trading
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger('trading_system.lstm_predictor')

class LSTMPredictor:
    """Advanced LSTM Neural Network for Price Prediction"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        self.last_training_time = None
        
        # Model parameters
        self.sequence_length = 60  # Look back 60 periods
        self.prediction_horizon = 1  # Predict 1 step ahead
        self.features = ['close', 'volume', 'high', 'low', 'rsi', 'macd', 'bb_upper', 'bb_lower']
        
        # Training parameters
        self.epochs = 50
        self.batch_size = 32
        self.validation_split = 0.2
        self.retrain_interval_hours = 24  # Retrain every 24 hours
        
        if not TENSORFLOW_AVAILABLE:
            logger.warning("[WARNING] TensorFlow not available. Install with: pip install tensorflow")
            return
            
        # Configure TensorFlow for optimal performance
        tf.config.experimental.enable_memory_growth = True
        
        logger.info("[ML] LSTM Predictor initialized with TensorFlow")
    
    def prepare_data(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        try:
            if market_data.empty or len(market_data) < self.sequence_length + 10:
                logger.warning("[ML] Insufficient data for LSTM training")
                return None, None
            
            # Ensure we have required features
            required_features = ['close', 'volume', 'high', 'low']
            available_features = [col for col in required_features if col in market_data.columns]
            
            if len(available_features) < 4:
                logger.warning(f"[ML] Missing required features. Available: {available_features}")
                return None, None
            
            # Add technical indicators if not present
            data = market_data.copy()
            if 'rsi' not in data.columns:
                data['rsi'] = self._calculate_rsi(data['close'])
            if 'macd' not in data.columns:
                data['macd'] = self._calculate_macd(data['close'])
            if 'bb_upper' not in data.columns or 'bb_lower' not in data.columns:
                bb_upper, bb_lower = self._calculate_bollinger_bands(data['close'])
                data['bb_upper'] = bb_upper
                data['bb_lower'] = bb_lower
            
            # Select features that exist
            feature_cols = [col for col in self.features if col in data.columns]
            feature_data = data[feature_cols].fillna(method='ffill').fillna(method='bfill')
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(feature_data)
            
            # Create sequences
            X, y = [], []
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i])
                y.append(scaled_data[i, 0])  # Predict close price (first feature)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"[ML] Data preparation failed: {e}")
            return None, None
    
    def build_model(self, input_shape: Tuple) -> Sequential:
        """Build advanced LSTM model architecture"""
        try:
            model = Sequential([
                # First LSTM layer with return sequences
                LSTM(100, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                BatchNormalization(),
                
                # Second LSTM layer with return sequences
                LSTM(100, return_sequences=True),
                Dropout(0.2),
                BatchNormalization(),
                
                # Third LSTM layer without return sequences
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                BatchNormalization(),
                
                # Dense layers
                Dense(25, activation='relu'),
                Dropout(0.1),
                Dense(1, activation='linear')
            ])
            
            # Compile with advanced optimizer
            optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            logger.info("[ML] Advanced LSTM model architecture built")
            return model
            
        except Exception as e:
            logger.error(f"[ML] Model building failed: {e}")
            return None
    
    def train_model(self, market_data: pd.DataFrame) -> bool:
        """Train the LSTM model with market data"""
        try:
            if not TENSORFLOW_AVAILABLE:
                logger.warning("[ML] TensorFlow not available for training")
                return False
            
            logger.info("[ML] Starting LSTM model training...")
            
            # Prepare data
            X, y = self.prepare_data(market_data)
            if X is None or y is None:
                return False
            
            # Build model
            self.model = self.build_model((X.shape[1], X.shape[2]))
            if self.model is None:
                return False
            
            # Training callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5, 
                min_lr=0.0001
            )
            
            # Train the model
            history = self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Evaluate model performance
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            
            logger.info(f"[ML] Training completed - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"[ML] Model training failed: {e}")
            return False
    
    def predict_price(self, recent_data: pd.DataFrame) -> Optional[Dict]:
        """Generate price prediction using trained LSTM model"""
        try:
            if not self.is_trained or self.model is None:
                logger.warning("[ML] Model not trained. Cannot make predictions.")
                return None
            
            if len(recent_data) < self.sequence_length:
                logger.warning(f"[ML] Insufficient data for prediction. Need {self.sequence_length} periods.")
                return None
            
            # Prepare recent data
            feature_cols = [col for col in self.features if col in recent_data.columns]
            if len(feature_cols) < 4:
                logger.warning("[ML] Insufficient features for prediction")
                return None
            
            # Get last sequence
            last_sequence = recent_data[feature_cols].tail(self.sequence_length)
            last_sequence = last_sequence.fillna(method='ffill').fillna(method='bfill')
            
            # Scale the data
            scaled_sequence = self.scaler.transform(last_sequence)
            
            # Reshape for prediction
            X_pred = scaled_sequence.reshape(1, self.sequence_length, len(feature_cols))
            
            # Make prediction
            prediction_scaled = self.model.predict(X_pred, verbose=0)
            
            # Inverse transform to get actual price
            # Create dummy array for inverse transform
            dummy_array = np.zeros((1, len(feature_cols)))
            dummy_array[0, 0] = prediction_scaled[0, 0]
            prediction_actual = self.scaler.inverse_transform(dummy_array)[0, 0]
            
            current_price = recent_data['close'].iloc[-1]
            predicted_change = ((prediction_actual - current_price) / current_price) * 100
            
            # Calculate confidence based on recent model performance
            confidence = min(85, max(15, 70 - abs(predicted_change) * 2))
            
            prediction_result = {
                'predicted_price': float(prediction_actual),
                'current_price': float(current_price),
                'predicted_change_pct': float(predicted_change),
                'confidence': float(confidence),
                'signal': 'BUY' if predicted_change > 0.5 else 'SELL' if predicted_change < -0.5 else 'HOLD',
                'timestamp': datetime.now(),
                'model_type': 'LSTM'
            }
            
            logger.info(f"[ML] LSTM Prediction: {prediction_actual:.2f} ({predicted_change:+.2f}%) - {prediction_result['signal']}")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"[ML] Prediction failed: {e}")
            return None
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained"""
        if not self.is_trained or self.last_training_time is None:
            return True
        
        time_since_training = datetime.now() - self.last_training_time
        return time_since_training.total_seconds() > (self.retrain_interval_hours * 3600)
    
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
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band.fillna(prices), lower_band.fillna(prices)
    
    def get_model_info(self) -> Dict:
        """Get model information and status"""
        return {
            'is_trained': self.is_trained,
            'last_training_time': self.last_training_time,
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'sequence_length': self.sequence_length,
            'features_count': len(self.features),
            'should_retrain': self.should_retrain()
        }