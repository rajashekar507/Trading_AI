"""
Adaptive Learning System
Learns from trade outcomes and continuously improves strategies
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sqlite3
from dataclasses import dataclass, asdict
import pickle

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger('trading_system.adaptive_learning')

@dataclass
class TradeOutcome:
    """Trade outcome data structure"""
    trade_id: str
    timestamp: datetime
    instrument: str
    strategy: str
    entry_price: float
    exit_price: float
    quantity: int
    profit_loss: float
    profit_loss_pct: float
    duration_minutes: int
    market_conditions: Dict
    news_sentiment: Dict
    technical_indicators: Dict
    ai_analysis: Dict
    success: bool  # True if profitable, False if loss

@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    last_updated: datetime

class AdaptiveLearningSystem:
    """Adaptive learning system that improves from trade outcomes"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.project_root = Path(__file__).parent.parent
        self.db_path = self.project_root / "data_storage" / "adaptive_learning.db"
        self.models_path = self.project_root / "data_storage" / "ml_models"
        
        # Create directories
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Learning models
        self.success_predictor = None
        self.strategy_selector = None
        self.risk_assessor = None
        self.scaler = StandardScaler()
        
        # Performance tracking
        self.strategy_performance = {}
        self.market_regime_detector = None
        
        # Learning parameters
        self.min_trades_for_learning = 50
        self.retraining_interval_hours = 24
        self.last_training_time = None
        
        # Feature importance tracking
        self.feature_importance = {}
        
        logger.info("[ADAPTIVE] Adaptive Learning System initialized")
    
    async def initialize(self) -> bool:
        """Initialize the adaptive learning system"""
        try:
            logger.info("[ADAPTIVE] Initializing adaptive learning system...")
            
            # Initialize database
            await self._initialize_database()
            
            # Load existing models
            await self._load_models()
            
            # Load strategy performance
            await self._load_strategy_performance()
            
            logger.info("[ADAPTIVE] Adaptive learning system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Initialization failed: {e}")
            return False
    
    async def _initialize_database(self):
        """Initialize SQLite database for trade outcomes"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Trade outcomes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    trade_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    instrument TEXT,
                    strategy TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity INTEGER,
                    profit_loss REAL,
                    profit_loss_pct REAL,
                    duration_minutes INTEGER,
                    market_conditions TEXT,
                    news_sentiment TEXT,
                    technical_indicators TEXT,
                    ai_analysis TEXT,
                    success INTEGER
                )
            ''')
            
            # Strategy performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    strategy_name TEXT PRIMARY KEY,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    avg_profit REAL,
                    avg_loss REAL,
                    profit_factor REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    last_updated TEXT
                )
            ''')
            
            # Feature importance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_importance (
                    feature_name TEXT PRIMARY KEY,
                    importance_score REAL,
                    last_updated TEXT
                )
            ''')
            
            # Market regime table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_regimes (
                    timestamp TEXT PRIMARY KEY,
                    regime TEXT,
                    volatility REAL,
                    trend_strength REAL,
                    market_conditions TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("[ADAPTIVE] Database initialized successfully")
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Database initialization failed: {e}")
            raise
    
    async def record_trade_outcome(self, trade_outcome: TradeOutcome):
        """Record a trade outcome for learning"""
        try:
            logger.info(f"[ADAPTIVE] Recording trade outcome: {trade_outcome.trade_id}")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO trade_outcomes 
                (trade_id, timestamp, instrument, strategy, entry_price, exit_price, 
                 quantity, profit_loss, profit_loss_pct, duration_minutes, 
                 market_conditions, news_sentiment, technical_indicators, ai_analysis, success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_outcome.trade_id,
                trade_outcome.timestamp.isoformat(),
                trade_outcome.instrument,
                trade_outcome.strategy,
                trade_outcome.entry_price,
                trade_outcome.exit_price,
                trade_outcome.quantity,
                trade_outcome.profit_loss,
                trade_outcome.profit_loss_pct,
                trade_outcome.duration_minutes,
                json.dumps(trade_outcome.market_conditions),
                json.dumps(trade_outcome.news_sentiment),
                json.dumps(trade_outcome.technical_indicators),
                json.dumps(trade_outcome.ai_analysis),
                1 if trade_outcome.success else 0
            ))
            
            conn.commit()
            conn.close()
            
            # Update strategy performance
            await self._update_strategy_performance(trade_outcome)
            
            # Check if we should retrain models
            if await self._should_retrain():
                await self._retrain_models()
            
            logger.info(f"[ADAPTIVE] Trade outcome recorded successfully")
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Failed to record trade outcome: {e}")
    
    async def _update_strategy_performance(self, trade_outcome: TradeOutcome):
        """Update strategy performance metrics"""
        try:
            strategy_name = trade_outcome.strategy
            
            if strategy_name not in self.strategy_performance:
                self.strategy_performance[strategy_name] = StrategyPerformance(
                    strategy_name=strategy_name,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    win_rate=0.0,
                    avg_profit=0.0,
                    avg_loss=0.0,
                    profit_factor=0.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0,
                    last_updated=datetime.now()
                )
            
            perf = self.strategy_performance[strategy_name]
            perf.total_trades += 1
            
            if trade_outcome.success:
                perf.winning_trades += 1
                perf.avg_profit = ((perf.avg_profit * (perf.winning_trades - 1)) + 
                                 trade_outcome.profit_loss_pct) / perf.winning_trades
            else:
                perf.losing_trades += 1
                perf.avg_loss = ((perf.avg_loss * (perf.losing_trades - 1)) + 
                               abs(trade_outcome.profit_loss_pct)) / perf.losing_trades
            
            perf.win_rate = perf.winning_trades / perf.total_trades
            
            if perf.avg_loss > 0:
                perf.profit_factor = (perf.avg_profit * perf.winning_trades) / (perf.avg_loss * perf.losing_trades)
            
            perf.last_updated = datetime.now()
            
            # Save to database
            await self._save_strategy_performance(perf)
            
            logger.info(f"[ADAPTIVE] Updated performance for {strategy_name}: "
                       f"Win Rate: {perf.win_rate:.2%}, Trades: {perf.total_trades}")
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Failed to update strategy performance: {e}")
    
    async def _save_strategy_performance(self, perf: StrategyPerformance):
        """Save strategy performance to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO strategy_performance 
                (strategy_name, total_trades, winning_trades, losing_trades, win_rate,
                 avg_profit, avg_loss, profit_factor, max_drawdown, sharpe_ratio, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                perf.strategy_name, perf.total_trades, perf.winning_trades, perf.losing_trades,
                perf.win_rate, perf.avg_profit, perf.avg_loss, perf.profit_factor,
                perf.max_drawdown, perf.sharpe_ratio, perf.last_updated.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Failed to save strategy performance: {e}")
    
    async def _should_retrain(self) -> bool:
        """Check if models should be retrained"""
        try:
            # Check if enough time has passed
            if self.last_training_time:
                time_since_training = datetime.now() - self.last_training_time
                if time_since_training.total_seconds() < (self.retraining_interval_hours * 3600):
                    return False
            
            # Check if we have enough new data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM trade_outcomes')
            total_trades = cursor.fetchone()[0]
            
            conn.close()
            
            return total_trades >= self.min_trades_for_learning
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Error checking retrain condition: {e}")
            return False
    
    async def _retrain_models(self):
        """Retrain machine learning models with new data"""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("[ADAPTIVE] Scikit-learn not available for retraining")
                return
            
            logger.info("[ADAPTIVE] Starting model retraining...")
            
            # Load training data
            training_data = await self._load_training_data()
            
            if len(training_data) < self.min_trades_for_learning:
                logger.warning(f"[ADAPTIVE] Insufficient data for training: {len(training_data)} trades")
                return
            
            # Prepare features and targets
            X, y_success, y_strategy = await self._prepare_training_data(training_data)
            
            if X is None or len(X) == 0:
                logger.warning("[ADAPTIVE] No valid training data prepared")
                return
            
            # Train success predictor
            await self._train_success_predictor(X, y_success)
            
            # Train strategy selector
            await self._train_strategy_selector(X, y_strategy)
            
            # Update feature importance
            await self._update_feature_importance(X.columns if hasattr(X, 'columns') else None)
            
            # Save models
            await self._save_models()
            
            self.last_training_time = datetime.now()
            
            logger.info("[ADAPTIVE] Model retraining completed successfully")
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Model retraining failed: {e}")
    
    async def _load_training_data(self) -> List[Dict]:
        """Load training data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM trade_outcomes 
                ORDER BY timestamp DESC 
                LIMIT 1000
            ''')
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            conn.close()
            
            training_data = []
            for row in rows:
                trade_dict = dict(zip(columns, row))
                
                # Parse JSON fields
                try:
                    trade_dict['market_conditions'] = json.loads(trade_dict['market_conditions'])
                    trade_dict['news_sentiment'] = json.loads(trade_dict['news_sentiment'])
                    trade_dict['technical_indicators'] = json.loads(trade_dict['technical_indicators'])
                    trade_dict['ai_analysis'] = json.loads(trade_dict['ai_analysis'])
                except json.JSONDecodeError:
                    continue
                
                training_data.append(trade_dict)
            
            logger.info(f"[ADAPTIVE] Loaded {len(training_data)} training samples")
            return training_data
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Failed to load training data: {e}")
            return []
    
    async def _prepare_training_data(self, training_data: List[Dict]) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data for machine learning"""
        try:
            features = []
            success_labels = []
            strategy_labels = []
            
            for trade in training_data:
                try:
                    # Extract features
                    feature_dict = {}
                    
                    # Basic trade features
                    feature_dict['duration_minutes'] = trade.get('duration_minutes', 0)
                    feature_dict['quantity'] = trade.get('quantity', 0)
                    
                    # Market conditions features
                    market_conditions = trade.get('market_conditions', {})
                    feature_dict['market_volatility'] = market_conditions.get('volatility', 0.0)
                    feature_dict['market_trend'] = market_conditions.get('trend_strength', 0.0)
                    feature_dict['market_volume'] = market_conditions.get('volume_ratio', 1.0)
                    
                    # News sentiment features
                    news_sentiment = trade.get('news_sentiment', {})
                    feature_dict['news_sentiment_score'] = news_sentiment.get('sentiment_score', 0.0)
                    feature_dict['news_count'] = news_sentiment.get('news_count', 0)
                    
                    # Technical indicators features
                    technical = trade.get('technical_indicators', {})
                    feature_dict['rsi'] = technical.get('rsi', 50.0)
                    feature_dict['macd'] = technical.get('macd', 0.0)
                    feature_dict['bb_position'] = technical.get('bb_position', 0.0)
                    
                    # AI analysis features
                    ai_analysis = trade.get('ai_analysis', {})
                    feature_dict['ai_confidence'] = ai_analysis.get('confidence_score', 0.5)
                    feature_dict['ai_sentiment'] = 1.0 if ai_analysis.get('overall_recommendation') == 'BULLISH' else -1.0 if ai_analysis.get('overall_recommendation') == 'BEARISH' else 0.0
                    
                    # Time-based features
                    timestamp = datetime.fromisoformat(trade['timestamp'])
                    feature_dict['hour_of_day'] = timestamp.hour
                    feature_dict['day_of_week'] = timestamp.weekday()
                    
                    features.append(feature_dict)
                    success_labels.append(trade['success'])
                    strategy_labels.append(trade['strategy'])
                    
                except Exception as e:
                    logger.warning(f"[ADAPTIVE] Error processing trade data: {e}")
                    continue
            
            if not features:
                return None, None, None
            
            # Convert to DataFrame
            X = pd.DataFrame(features)
            y_success = np.array(success_labels)
            y_strategy = np.array(strategy_labels)
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            logger.info(f"[ADAPTIVE] Prepared training data: {len(X)} samples, {len(X.columns)} features")
            return X, y_success, y_strategy
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Failed to prepare training data: {e}")
            return None, None, None
    
    async def _train_success_predictor(self, X: pd.DataFrame, y: np.ndarray):
        """Train model to predict trade success"""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Random Forest classifier
            self.success_predictor = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            
            self.success_predictor.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.success_predictor.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"[ADAPTIVE] Success predictor trained - Accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Success predictor training failed: {e}")
    
    async def _train_strategy_selector(self, X: pd.DataFrame, y: np.ndarray):
        """Train model to select best strategy"""
        try:
            # Only train if we have multiple strategies
            unique_strategies = np.unique(y)
            if len(unique_strategies) < 2:
                logger.info("[ADAPTIVE] Only one strategy available, skipping strategy selector training")
                return
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train Random Forest classifier
            self.strategy_selector = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=5,
                random_state=42
            )
            
            self.strategy_selector.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.strategy_selector.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"[ADAPTIVE] Strategy selector trained - Accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Strategy selector training failed: {e}")
    
    async def _update_feature_importance(self, feature_names: Optional[List[str]]):
        """Update feature importance scores"""
        try:
            if not self.success_predictor or not feature_names:
                return
            
            importance_scores = self.success_predictor.feature_importances_
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for feature_name, importance in zip(feature_names, importance_scores):
                cursor.execute('''
                    INSERT OR REPLACE INTO feature_importance 
                    (feature_name, importance_score, last_updated)
                    VALUES (?, ?, ?)
                ''', (feature_name, float(importance), datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            # Update in-memory cache
            self.feature_importance = dict(zip(feature_names, importance_scores))
            
            logger.info(f"[ADAPTIVE] Updated feature importance for {len(feature_names)} features")
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Failed to update feature importance: {e}")
    
    async def predict_trade_success(self, market_conditions: Dict, news_sentiment: Dict, 
                                  technical_indicators: Dict, ai_analysis: Dict) -> Dict:
        """Predict probability of trade success"""
        try:
            if not self.success_predictor:
                return {'success_probability': 0.5, 'confidence': 0.0, 'recommendation': 'NEUTRAL'}
            
            # Prepare features
            features = {
                'market_volatility': market_conditions.get('volatility', 0.0),
                'market_trend': market_conditions.get('trend_strength', 0.0),
                'market_volume': market_conditions.get('volume_ratio', 1.0),
                'news_sentiment_score': news_sentiment.get('sentiment_score', 0.0),
                'news_count': news_sentiment.get('news_count', 0),
                'rsi': technical_indicators.get('rsi', 50.0),
                'macd': technical_indicators.get('macd', 0.0),
                'bb_position': technical_indicators.get('bb_position', 0.0),
                'ai_confidence': ai_analysis.get('confidence_score', 0.5),
                'ai_sentiment': 1.0 if ai_analysis.get('overall_recommendation') == 'BULLISH' else -1.0 if ai_analysis.get('overall_recommendation') == 'BEARISH' else 0.0,
                'hour_of_day': datetime.now().hour,
                'day_of_week': datetime.now().weekday(),
                'duration_minutes': 60,  # Default expected duration
                'quantity': 1  # Default quantity
            }
            
            # Convert to DataFrame and scale
            X = pd.DataFrame([features])
            X_scaled = self.scaler.transform(X)
            
            # Predict
            success_prob = self.success_predictor.predict_proba(X_scaled)[0][1]  # Probability of success
            confidence = max(self.success_predictor.predict_proba(X_scaled)[0]) - 0.5  # Confidence measure
            
            # Generate recommendation
            if success_prob > 0.7:
                recommendation = 'STRONG_BUY'
            elif success_prob > 0.6:
                recommendation = 'BUY'
            elif success_prob < 0.3:
                recommendation = 'STRONG_SELL'
            elif success_prob < 0.4:
                recommendation = 'SELL'
            else:
                recommendation = 'NEUTRAL'
            
            result = {
                'success_probability': float(success_prob),
                'confidence': float(confidence),
                'recommendation': recommendation,
                'feature_importance': dict(list(self.feature_importance.items())[:5]) if self.feature_importance else {}
            }
            
            logger.info(f"[ADAPTIVE] Trade success prediction: {success_prob:.3f} ({recommendation})")
            return result
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Trade success prediction failed: {e}")
            return {'success_probability': 0.5, 'confidence': 0.0, 'recommendation': 'NEUTRAL'}
    
    async def recommend_best_strategy(self, market_conditions: Dict) -> Dict:
        """Recommend best strategy based on current conditions"""
        try:
            # Get strategy performance
            best_strategy = None
            best_score = 0.0
            
            for strategy_name, perf in self.strategy_performance.items():
                if perf.total_trades < 10:  # Need minimum trades
                    continue
                
                # Calculate composite score
                score = (perf.win_rate * 0.4) + (perf.profit_factor * 0.3) + (perf.sharpe_ratio * 0.3)
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy_name
            
            if not best_strategy:
                best_strategy = 'ORB_STRATEGY'  # Default strategy
                best_score = 0.5
            
            return {
                'recommended_strategy': best_strategy,
                'confidence_score': min(0.9, best_score),
                'strategy_performance': asdict(self.strategy_performance.get(best_strategy)) if best_strategy in self.strategy_performance else {},
                'all_strategies': {name: {'win_rate': perf.win_rate, 'total_trades': perf.total_trades} 
                                 for name, perf in self.strategy_performance.items()}
            }
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Strategy recommendation failed: {e}")
            return {'recommended_strategy': 'ORB_STRATEGY', 'confidence_score': 0.5}
    
    async def get_learning_insights(self) -> Dict:
        """Get insights from the learning system"""
        try:
            insights = {
                'total_trades_learned': sum(perf.total_trades for perf in self.strategy_performance.values()),
                'best_performing_strategy': None,
                'worst_performing_strategy': None,
                'overall_win_rate': 0.0,
                'key_success_factors': [],
                'improvement_suggestions': [],
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None
            }
            
            if self.strategy_performance:
                # Find best and worst strategies
                best_strategy = max(self.strategy_performance.values(), key=lambda x: x.win_rate)
                worst_strategy = min(self.strategy_performance.values(), key=lambda x: x.win_rate)
                
                insights['best_performing_strategy'] = {
                    'name': best_strategy.strategy_name,
                    'win_rate': best_strategy.win_rate,
                    'total_trades': best_strategy.total_trades
                }
                
                insights['worst_performing_strategy'] = {
                    'name': worst_strategy.strategy_name,
                    'win_rate': worst_strategy.win_rate,
                    'total_trades': worst_strategy.total_trades
                }
                
                # Calculate overall win rate
                total_wins = sum(perf.winning_trades for perf in self.strategy_performance.values())
                total_trades = sum(perf.total_trades for perf in self.strategy_performance.values())
                insights['overall_win_rate'] = total_wins / total_trades if total_trades > 0 else 0.0
            
            # Key success factors from feature importance
            if self.feature_importance:
                top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                insights['key_success_factors'] = [{'feature': name, 'importance': importance} 
                                                 for name, importance in top_features]
            
            # Generate improvement suggestions
            insights['improvement_suggestions'] = await self._generate_improvement_suggestions()
            
            return insights
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Failed to get learning insights: {e}")
            return {}
    
    async def _generate_improvement_suggestions(self) -> List[str]:
        """Generate improvement suggestions based on learning"""
        suggestions = []
        
        try:
            if not self.strategy_performance:
                suggestions.append("Need more trade data to generate meaningful suggestions")
                return suggestions
            
            # Analyze win rates
            low_win_rate_strategies = [perf for perf in self.strategy_performance.values() 
                                     if perf.win_rate < 0.4 and perf.total_trades > 20]
            
            if low_win_rate_strategies:
                suggestions.append(f"Consider reviewing or disabling strategies with low win rates: {[s.strategy_name for s in low_win_rate_strategies]}")
            
            # Analyze feature importance
            if self.feature_importance:
                top_feature = max(self.feature_importance.items(), key=lambda x: x[1])
                suggestions.append(f"Focus on optimizing '{top_feature[0]}' as it has the highest impact on success")
            
            # Check for insufficient data
            low_data_strategies = [perf for perf in self.strategy_performance.values() if perf.total_trades < 50]
            if low_data_strategies:
                suggestions.append("Some strategies need more trades for reliable performance assessment")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Failed to generate improvement suggestions: {e}")
            return ["Unable to generate suggestions due to system error"]
    
    async def _load_models(self):
        """Load saved machine learning models"""
        try:
            success_predictor_path = self.models_path / "success_predictor.pkl"
            strategy_selector_path = self.models_path / "strategy_selector.pkl"
            scaler_path = self.models_path / "scaler.pkl"
            
            if success_predictor_path.exists():
                with open(success_predictor_path, 'rb') as f:
                    self.success_predictor = pickle.load(f)
                logger.info("[ADAPTIVE] Success predictor model loaded")
            
            if strategy_selector_path.exists():
                with open(strategy_selector_path, 'rb') as f:
                    self.strategy_selector = pickle.load(f)
                logger.info("[ADAPTIVE] Strategy selector model loaded")
            
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("[ADAPTIVE] Feature scaler loaded")
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Failed to load models: {e}")
    
    async def _save_models(self):
        """Save machine learning models"""
        try:
            if self.success_predictor:
                success_predictor_path = self.models_path / "success_predictor.pkl"
                with open(success_predictor_path, 'wb') as f:
                    pickle.dump(self.success_predictor, f)
            
            if self.strategy_selector:
                strategy_selector_path = self.models_path / "strategy_selector.pkl"
                with open(strategy_selector_path, 'wb') as f:
                    pickle.dump(self.strategy_selector, f)
            
            scaler_path = self.models_path / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            logger.info("[ADAPTIVE] Models saved successfully")
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Failed to save models: {e}")
    
    async def _load_strategy_performance(self):
        """Load strategy performance from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM strategy_performance')
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            conn.close()
            
            for row in rows:
                perf_dict = dict(zip(columns, row))
                perf_dict['last_updated'] = datetime.fromisoformat(perf_dict['last_updated'])
                
                perf = StrategyPerformance(**perf_dict)
                self.strategy_performance[perf.strategy_name] = perf
            
            logger.info(f"[ADAPTIVE] Loaded performance data for {len(self.strategy_performance)} strategies")
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Failed to load strategy performance: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Save models before cleanup
            await self._save_models()
            logger.info("[ADAPTIVE] Adaptive learning system cleanup completed")
        except Exception as e:
            logger.error(f"[ADAPTIVE] Cleanup error: {e}")