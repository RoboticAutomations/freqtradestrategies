"""
# Enhanced Fisher Transform Strategy with ML/RL Integration - V1.2 BALANCED
# FIXED: Balanced long/short entries, tightened stoploss, removed restrictive Goldie Locks
# 
# Changelog v1.2 (2026-02-17):
# - FIXED: Stoploss tightened from -0.524 (52%!) to -0.15 (15%)
# - FIXED: max_open_trades reduced from 20 to 5 to limit risk
# - FIXED: Balanced ML conditions - both longs and shorts use same confidence logic
# - FIXED: Short entry removed restrictive Goldie Locks zone requirement
# - FIXED: Short entry now uses fisher < -0.5 (symmetric to long) instead of < 2.89
# - FIXED: Both directions require baseline trend in their favor
# - FIXED: Long entry threshold relaxed to 1.0 (was 0.65)
#
# Changelog v1.1 (2026-02-17):
# - FIXED: fisher_trend_long was identical to fisher_trend_short
# - FIXED: Long entry used fisher_sell_threshold instead of fisher_buy_threshold
# - FIXED: Long entry momentum check was inverted
#
# Usage:
# freqtrade hyperopt --hyperopt-loss SharpeHyperOptLossDaily --strategy GKD_FisherTransformV4_ML_V1_2_Balanced \
#     --spaces buy sell roi stoploss trailing --config user_data/config_binance_futures_backtest_usdt.json \
#     --epochs 1000 --timerange 20241001-20250501 --timeframe-detail 5m --max-open-trades 3 -timeframe 1h
"""

import datetime
import logging
import math
import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import talib
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter,
                                DecimalParameter, IntParameter, IStrategy,
                                RealParameter, informative,
                                merge_informative_pair)
from pandas import DataFrame
from pandas_ta import ema
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

ASCII_ART = '''
    ╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                                                        ║
    ║  ██████╗ ██╗     ███████╗██╗  ██╗ ██████╗██████╗ ██╗   ██╗██████╗ ████████╗ ██████╗ ██╗  ██╗██╗███╗   ██╗ ██████╗      ║
    ║  ██╔══██╗██║     ██╔════╝╚██╗██╔╝██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗╚══██╔══╝██╔═══██╗██║ ██╔╝██║████╗  ██║██╔════╝      ║
    ║  ███████║██║     █████╗   ╚███╔╝ ██║     ██████╔╝ ╚████╔╝ ██████╔╝   ██║   ██║   ██║█████╔╝ ██║██╔██╗ ██║██║  ███╗     ║
    ║  ██╔══██║██║     ██╔══╝   ██╔██╗ ██║     ██╔══██╗  ╚██╔╝  ██╔═══╝    ██║   ██║   ██║██╔═██╗ ██║██║╚██╗██║██║   ██║     ║
    ║  ██║  ██║███████╗███████╗██╔╝ ██╗╚██████╗██║  ██║   ██║   ██║        ██║   ╚██████╔╝██║  ██╗██║██║ ╚████║╚██████╔╝     ║
    ║  ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝      ║
    ║                                                                                                                        ║
    ║                                               ALEX GKD_FISHER TRANSOFRM                                                ║
    ║                                               VERSION 5                                                                ║
    ║                                               20.02.2026                                                               ║
    ║                                                                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
'''
def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b by factor t"""
    return a + t * (b - a)


class EnhancedLogger:
    """Enhanced logging class with emojis and better formatting"""
    
    @staticmethod
    def log_banner(message: str, emoji: str = "🚀"):
        """Log a banner message"""
        border = "═" * (len(message) + 6)
        logger.info(f"{emoji} {border}")
        logger.info(f"{emoji} ║  {message}  ║")
        logger.info(f"{emoji} {border}")
    
    @staticmethod
    def log_section(title: str, emoji: str = "📊"):
        """Log a section header"""
        logger.info(f"\n{emoji} ═══ {title} ═══")
    
    @staticmethod
    def log_subsection(title: str, emoji: str = "▶️"):
        """Log a subsection"""
        logger.info(f"{emoji} {title}")
    
    @staticmethod
    def log_parameter(name: str, value: Any, emoji: str = "⚙️"):
        """Log a parameter with formatting"""
        if isinstance(value, float):
            logger.info(f"  {emoji} {name}: {value:.4f}")
        else:
            logger.info(f"  {emoji} {name}: {value}")
    
    @staticmethod
    def log_performance(metric: str, value: float, emoji: str = "📈"):
        """Log performance metrics"""
        color_emoji = "🟢" if value > 0 else "🔴" if value < 0 else "🟡"
        logger.info(f"{emoji} {color_emoji} {metric}: {value:.4f}")
    
    @staticmethod
    def log_trade_action(action: str, pair: str, rate: float, emoji: str = "💰"):
        """Log trade actions"""
        logger.info(f"{emoji} {action} {pair} @ {rate:.6f}")
    
    @staticmethod
    def log_ml_status(message: str, confidence: float = None, emoji: str = "🤖"):
        """Log ML related messages"""
        if confidence is not None:
            conf_emoji = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🔴"
            logger.info(f"{emoji} {conf_emoji} {message} (Confidence: {confidence:.2%})")
        else:
            logger.info(f"{emoji} {message}")
    
    @staticmethod
    def log_error(message: str, emoji: str = "❌"):
        """Log error messages"""
        logger.error(f"{emoji} ERROR: {message}")
    
    @staticmethod
    def log_warning(message: str, emoji: str = "⚠️"):
        """Log warning messages"""
        logger.warning(f"{emoji} WARNING: {message}")
    
    @staticmethod
    def log_success(message: str, emoji: str = "✅"):
        """Log success messages"""
        logger.info(f"{emoji} SUCCESS: {message}")


class MLOptimizer:
    """Machine Learning optimizer for strategy parameters with enhanced logging"""
    
    def __init__(self, strategy_name: str = "GKD_FisherV5"):
        self.strategy_name = strategy_name
        self.model_path = f"user_data/strategies/ml_models/{strategy_name}_model.pkl"
        self.scaler_path = f"user_data/strategies/ml_models/{strategy_name}_scaler.pkl"
        self.study_path = f"user_data/strategies/ml_models/{strategy_name}_study.pkl"
        self.model = None
        self.scaler = None
        self.study = None
        self.performance_history = []
        
        # Ensure directory exists
        os.makedirs("user_data/strategies/ml_models", exist_ok=True)
        
        # Enhanced logging for initialization
        EnhancedLogger.log_section("ML OPTIMIZER INITIALIZATION", "🤖")
        EnhancedLogger.log_parameter("Strategy Name", strategy_name, "🏷️")
        EnhancedLogger.log_parameter("Model Path", self.model_path, "📁")
        
        # Load existing models if available
        self.load_models()
    
    def load_models(self):
        """Load existing ML models and Optuna study with enhanced logging"""
        try:
            models_loaded = 0
            
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                models_loaded += 1
                EnhancedLogger.log_success("ML Model loaded successfully", "🧠")
            else:
                EnhancedLogger.log_warning("No existing ML model found", "🤖")
            
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                models_loaded += 1
                EnhancedLogger.log_success("Feature Scaler loaded successfully", "📏")
            else:
                EnhancedLogger.log_warning("No existing scaler found", "📏")
                    
            if os.path.exists(self.study_path):
                with open(self.study_path, 'rb') as f:
                    self.study = pickle.load(f)
                models_loaded += 1
                EnhancedLogger.log_success(f"Optuna Study loaded ({len(self.study.trials)} trials)", "🔬")
            else:
                EnhancedLogger.log_warning("No existing Optuna study found", "🔬")
            
            if models_loaded > 0:
                EnhancedLogger.log_success(f"Loaded {models_loaded}/3 ML components", "✨")
            
        except Exception as e:
            EnhancedLogger.log_error(f"Error loading models: {e}")
    
    def save_models(self):
        """Save ML models and Optuna study with enhanced logging"""
        try:
            saved_models = 0
            
            if self.model:
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                saved_models += 1
                EnhancedLogger.log_success("ML Model saved", "💾")
            
            if self.scaler:
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                saved_models += 1
                EnhancedLogger.log_success("Feature Scaler saved", "💾")
                    
            if self.study:
                with open(self.study_path, 'wb') as f:
                    pickle.dump(self.study, f)
                saved_models += 1
                EnhancedLogger.log_success("Optuna Study saved", "💾")
            
            if saved_models > 0:
                EnhancedLogger.log_success(f"Saved {saved_models} ML components", "🎯")
                
        except Exception as e:
            EnhancedLogger.log_error(f"Error saving models: {e}")
    
    def create_features(self, dataframe: DataFrame) -> np.ndarray:
        """Create features for ML model with enhanced logging - FIXED FEATURE COUNT"""
        features = []
        
        try:
            EnhancedLogger.log_subsection("Creating ML Features", "🔧")
            
            # Check if required columns exist, if not calculate them
            if 'atr' not in dataframe.columns or dataframe['atr'].isna().all():
                EnhancedLogger.log_warning("ATR column missing, calculating...", "📊")
                try:
                    import talib
                    dataframe['atr'] = talib.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
                except:
                    # Fallback ATR calculation
                    high_low = dataframe['high'] - dataframe['low']
                    high_close = abs(dataframe['high'] - dataframe['close'].shift())
                    low_close = abs(dataframe['low'] - dataframe['close'].shift())
                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    dataframe['atr'] = true_range.rolling(window=14).mean()
                EnhancedLogger.log_success("ATR calculated successfully", "📊")
            
            # Check if fisher exists, if not create a simple version
            if 'fisher' not in dataframe.columns or dataframe['fisher'].isna().all():
                EnhancedLogger.log_warning("Fisher Transform missing, calculating...", "🎯")
                # Simple Fisher Transform calculation
                median_price = (dataframe['high'] + dataframe['low']) / 2
                period = 14
                fisher = pd.Series(0.0, index=dataframe.index)
                
                for i in range(period, len(dataframe)):
                    price_window = median_price.iloc[i-period:i]
                    price_min = price_window.min()
                    price_max = price_window.max()
                    if price_max != price_min:
                        norm = (median_price.iloc[i] - price_min) / (price_max - price_min)
                        norm = 2 * norm - 1
                        norm = max(min(norm, 0.999), -0.999)
                        fisher.iloc[i] = 0.5 * np.log((1 + norm) / (1 - norm))
                
                dataframe['fisher'] = fisher
                EnhancedLogger.log_success("Fisher Transform calculated", "🎯")
            
            # Check if baseline_diff exists
            if 'baseline_diff' not in dataframe.columns:
                EnhancedLogger.log_warning("Baseline diff missing, calculating...", "📈")
                try:
                    from pandas_ta import ema
                    dataframe['baseline'] = ema(dataframe['close'], length=14)
                    dataframe['baseline_diff'] = dataframe['baseline'].diff()
                except:
                    dataframe['baseline'] = dataframe['close'].ewm(span=14).mean()
                    dataframe['baseline_diff'] = dataframe['baseline'].diff()
                EnhancedLogger.log_success("Baseline calculated", "📈")
            
            # NOW CREATE EXACTLY 12 FEATURES (FIXED COUNT)
            feature_names = []
            
            # 1. Market volatility features (2 features)
            atr_mean = dataframe['atr'].rolling(14).mean().iloc[-1]
            features.append(atr_mean if not pd.isna(atr_mean) else 0.01)
            feature_names.append("ATR_mean")
            
            atr_std = dataframe['atr'].rolling(7).std().iloc[-1]
            features.append(atr_std if not pd.isna(atr_std) else 0.001)
            feature_names.append("ATR_std")
            
            # 2. Price momentum features (3 features)
            for period in [5, 10, 20]:
                pct_change = dataframe['close'].pct_change(period).iloc[-1]
                features.append(pct_change if not pd.isna(pct_change) else 0.0)
                feature_names.append(f"momentum_{period}")
            
            # 3. Volume features (2 features)
            if 'volume' in dataframe.columns and not dataframe['volume'].isna().all():
                vol_mean = dataframe['volume'].rolling(14).mean().iloc[-1]
                features.append(vol_mean if not pd.isna(vol_mean) else 1000.0)
                feature_names.append("volume_mean")
                
                vol_pct = dataframe['volume'].pct_change().iloc[-1]
                features.append(vol_pct if not pd.isna(vol_pct) else 0.0)
                feature_names.append("volume_change")
                
                EnhancedLogger.log_success("Volume features added", "📊")
            else:
                features.extend([1000.0, 0.0])
                feature_names.extend(["volume_mean_default", "volume_change_default"])
                EnhancedLogger.log_warning("Using default volume features", "📊")
            
            # 4. Fisher transform features (3 features)
            fisher_current = dataframe['fisher'].iloc[-1]
            features.append(fisher_current if not pd.isna(fisher_current) else 0.0)
            feature_names.append("fisher_current")
            
            fisher_mean = dataframe['fisher'].rolling(5).mean().iloc[-1]
            features.append(fisher_mean if not pd.isna(fisher_mean) else 0.0)
            feature_names.append("fisher_mean")
            
            fisher_std = dataframe['fisher'].rolling(5).std().iloc[-1]
            features.append(fisher_std if not pd.isna(fisher_std) else 1.0)
            feature_names.append("fisher_std")
            
            # 5. Baseline trend features (1 feature - COMBINED TO SAVE SPACE)
            baseline_diff_mean = dataframe['baseline_diff'].rolling(5).mean().iloc[-1]
            baseline_diff_sum = dataframe['baseline_diff'].rolling(10).sum().iloc[-1]
            
            # COMBINE baseline features into one normalized feature
            if not pd.isna(baseline_diff_mean) and not pd.isna(baseline_diff_sum):
                combined_baseline = (baseline_diff_mean + baseline_diff_sum * 0.1)  # Weighted combination
            else:
                combined_baseline = 0.0
            
            features.append(combined_baseline)
            feature_names.append("baseline_combined")
            
            # 6. Market regime features (1 feature)
            sma_50 = dataframe['close'].rolling(50).mean().iloc[-1]
            sma_200 = dataframe['close'].rolling(200).mean().iloc[-1]
            
            if not pd.isna(sma_50) and not pd.isna(sma_200) and sma_200 != 0:
                regime_feature = 1.0 if sma_50 > sma_200 else 0.0
                regime_status = "BULL 🐂" if sma_50 > sma_200 else "BEAR 🐻"
            else:
                regime_feature = 0.5
                regime_status = "NEUTRAL ⚖️"
            
            features.append(regime_feature)
            feature_names.append("market_regime")
            
            # VERIFY EXACTLY 12 FEATURES
            if len(features) != 12:
                EnhancedLogger.log_error(f"Feature count error: Expected 12, got {len(features)}", "❌")
                # Force exactly 12 features
                if len(features) > 12:
                    features = features[:12]
                    feature_names = feature_names[:12]
                    EnhancedLogger.log_warning("Trimmed features to 12", "✂️")
                else:
                    while len(features) < 12:
                        features.append(0.0)
                        feature_names.append(f"padding_{len(features)}")
                    EnhancedLogger.log_warning("Padded features to 12", "📋")
            
            EnhancedLogger.log_success(f"Created exactly {len(features)} ML features", "✨")
            EnhancedLogger.log_parameter("Market Regime", regime_status, "🏛️")
            
            # Debug log feature names (optional)
            if len(features) == 12:
                EnhancedLogger.log_success("Feature count verified: 12/12", "✅")
            else:
                EnhancedLogger.log_error(f"Feature count still wrong: {len(features)}/12", "❌")
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            EnhancedLogger.log_error(f"Error in create_features: {e}")
            # Return exactly 12 zero features as fallback
            return np.zeros((1, 12))
    
    def optimize_parameters(self, dataframe: DataFrame, current_performance: float):
        """Use Optuna to optimize strategy parameters with enhanced logging"""
        
        EnhancedLogger.log_banner("STARTING OPTUNA OPTIMIZATION", "🔬")
        EnhancedLogger.log_performance("Current Performance", current_performance, "📊")
        
        def objective(trial):
            # Entry parameters
            fisher_period = trial.suggest_int('fisher_period', 10, 15)
            fisher_smooth_long = trial.suggest_int('fisher_smooth_long', 3, 10)
            fisher_smooth_short = trial.suggest_int('fisher_smooth_short', 3, 10)
            fisher_buy_threshold = trial.suggest_float('fisher_buy_threshold', -1.0, 2.5)
            baseline_period = trial.suggest_int('baseline_period', 5, 21)
            atr_period = trial.suggest_int('atr_period', 7, 21)
            goldie_locks = trial.suggest_float('goldie_locks', 1.5, 3.0)
            
            # EXIT PARAMETERS
            fisher_long_exit = trial.suggest_float('fisher_long_exit', -1.0, 1.0)
            fisher_short_exit = trial.suggest_float('fisher_short_exit', -1.0, 1.0)
            fisher_sell_threshold = trial.suggest_float('fisher_sell_threshold', 2.0, 3.9)
            
            # Risk management parameters
            atr_sl_long_multip = trial.suggest_float('atr_sl_long_multip', 1.0, 6.0)
            atr_sl_short_multip = trial.suggest_float('atr_sl_short_multip', 1.0, 6.0)
            rr_long = trial.suggest_float('rr_long', 1.0, 4.0)
            rr_short = trial.suggest_float('rr_short', 1.0, 4.0)
            
            complete_params = {
                'fisher_period': fisher_period,
                'fisher_smooth_long': fisher_smooth_long,
                'fisher_smooth_short': fisher_smooth_short,
                'fisher_buy_threshold': fisher_buy_threshold,
                'baseline_period': baseline_period,
                'atr_period': atr_period,
                'goldie_locks': goldie_locks,
                'fisher_long_exit': fisher_long_exit,
                'fisher_short_exit': fisher_short_exit,
                'fisher_sell_threshold': fisher_sell_threshold,
                'atr_sl_long_multip': atr_sl_long_multip,
                'atr_sl_short_multip': atr_sl_short_multip,
                'rr_long': rr_long,
                'rr_short': rr_short
            }
            
            # Log trial progress
            if len(self.study.trials) % 5 == 0:
                trial_num = len(self.study.trials) + 1
                EnhancedLogger.log_subsection(f"Trial #{trial_num}", "🧪")
                EnhancedLogger.log_parameter("Long Exit", f"{fisher_long_exit:.3f}", "📤")
                EnhancedLogger.log_parameter("Short Exit", f"{fisher_short_exit:.3f}", "📤")
            
            score = self.simulate_performance(dataframe, complete_params)
            return score
        
        # Create or load study
        if self.study is None:
            self.study = optuna.create_study(direction='maximize')
            EnhancedLogger.log_success("New Optuna study created", "🔬")
        
        # Optimize for a few trials
        start_time = datetime.datetime.now()
        self.study.optimize(objective, n_trials=10, timeout=30)
        optimization_time = (datetime.datetime.now() - start_time).total_seconds()
        
        EnhancedLogger.log_banner("OPTIMIZATION COMPLETED", "🎯")
        EnhancedLogger.log_performance("Best Score", self.study.best_value, "🏆")
        EnhancedLogger.log_parameter("Optimization Time", f"{optimization_time:.1f}s", "⏱️")
        EnhancedLogger.log_parameter("Total Trials", len(self.study.trials), "🔢")
        
        # Get best parameters
        best_params = self.study.best_params.copy()
        
        # Verify all exit parameters are present
        required_exit_params = ['fisher_long_exit', 'fisher_short_exit', 'fisher_sell_threshold']
        missing_params = []
        for param in required_exit_params:
            if param not in best_params:
                missing_params.append(param)
                if param == 'fisher_long_exit':
                    best_params[param] = -0.5
                elif param == 'fisher_short_exit':
                    best_params[param] = 0.5
                elif param == 'fisher_sell_threshold':
                    best_params[param] = 2.5
        
        if missing_params:
            EnhancedLogger.log_warning(f"Added defaults for missing params: {missing_params}", "🔧")
        
        EnhancedLogger.log_success(f"Parameters verified: {len(best_params)} total", "✅")
        
        # Save the study
        self.save_models()
        
        return best_params
    def simulate_performance(self, dataframe: DataFrame, params: dict) -> float:
        """Simulate strategy performance with enhanced logging - FIXED"""
        try:
            recent_data = dataframe.tail(50)
            fisher = self.calculate_fisher_simple(recent_data, params['fisher_period'])
            baseline = ema(recent_data['close'], length=params['baseline_period'])
            
            buy_signals = (fisher > params['fisher_buy_threshold']).astype(int)
            
            exit_signals = pd.Series(0, index=recent_data.index)
            if 'fisher_long_exit' in params:
                exit_signals = (fisher < params['fisher_long_exit']).astype(int)
            
            returns = recent_data['close'].pct_change().shift(-1)
            position = 0
            strategy_returns = []
            
            for i in range(len(recent_data)):
                if buy_signals.iloc[i] == 1 and position == 0:
                    position = 1
                elif exit_signals.iloc[i] == 1 and position == 1:
                    position = 0
                
                return_val = returns.iloc[i] if not pd.isna(returns.iloc[i]) else 0
                strategy_returns.append(position * return_val)
            
            # FIX: Ensure we return a scalar float
            performance = sum(strategy_returns)
            if hasattr(performance, 'item'):  # numpy scalar
                performance = performance.item()
            
            return float(performance)
            
        except Exception as e:
            EnhancedLogger.log_error(f"Performance simulation error: {e}")
            return -1.0

    def calculate_fisher_simple(self, dataframe: DataFrame, period: int) -> pd.Series:
        """Simplified Fisher Transform calculation"""
        median_price = (dataframe['high'] + dataframe['low']) / 2
        fisher = pd.Series(0.0, index=dataframe.index)
        
        for i in range(period, len(dataframe)):
            price_window = median_price.iloc[i-period:i]
            price_min = price_window.min()
            price_max = price_window.max()
            if price_max != price_min:
                norm = (median_price.iloc[i] - price_min) / (price_max - price_min)
                norm = 2 * norm - 1
                norm = max(min(norm, 0.999), -0.999)
                fisher.iloc[i] = 0.5 * np.log((1 + norm) / (1 - norm))
        
        return fisher
    
    def predict_optimal_params(self, dataframe: DataFrame) -> dict:
        """Predict optimal parameters using ML model with enhanced logging - COMPLETE VERSION"""
        if self.model is None or self.scaler is None:
            EnhancedLogger.log_warning("ML model or scaler not available", "🤖")
            return {}
        
        try:
            EnhancedLogger.log_subsection("ML Parameter Prediction", "🔮")
            
            features = self.create_features(dataframe)
            features_scaled = self.scaler.transform(features)
            
            predictions = self.model.predict(features_scaled)
            
            # Fix: Ensure predictions is a 2D array and handle single prediction
            if predictions.ndim == 1:
                predictions = predictions.reshape(1, -1)
            elif predictions.ndim == 0:
                predictions = np.array([[predictions]])
            
            # EXPANDED: Predict ALL required parameters instead of just 6
            param_names = [
                'fisher_period', 'fisher_smooth_long', 'fisher_smooth_short',
                'baseline_period', 'atr_period', 'goldie_locks',
                'fisher_buy_threshold', 'fisher_sell_threshold',
                'fisher_long_exit', 'fisher_short_exit',
                'atr_sl_long_multip', 'atr_sl_short_multip',
                'rr_long', 'rr_short'
            ]
            
            param_dict = {}
            
            # Handle case where model only predicts 6 values but we need 14
            if len(predictions) > 0 and len(predictions[0]) > 0:
                model_predictions = predictions[0]
                
                for i, name in enumerate(param_names):
                    if i < len(model_predictions):
                        # Direct ML prediction available
                        param_value = model_predictions[i]
                        if hasattr(param_value, 'item'):
                            param_value = param_value.item()
                        param_dict[name] = max(0.1, float(param_value))
                    else:
                        # Generate derived/interpolated values for missing parameters
                        param_dict[name] = self._generate_derived_parameter(name, param_dict)
            
            if param_dict:
                EnhancedLogger.log_success(f"ML predicted {len(param_dict)} parameters", "🔮")
                # Log which were direct predictions vs derived
                direct_count = min(len(param_names), len(predictions[0]) if len(predictions) > 0 else 0)
                derived_count = len(param_dict) - direct_count
                if derived_count > 0:
                    EnhancedLogger.log_warning(f"Derived {derived_count} parameters from ML base", "🔄")
            else:
                EnhancedLogger.log_warning("ML prediction returned empty results", "⚠️")
            
            return param_dict
            
        except Exception as e:
            EnhancedLogger.log_error(f"ML prediction error: {e}")
            return {}
    def _generate_derived_parameter(self, param_name: str, existing_params: dict) -> float:
        """Generate derived parameters based on existing ML predictions"""
        
        # Use intelligent defaults based on parameter relationships
        if param_name == 'fisher_smooth_short':
            # Base on fisher_smooth_long if available
            if 'fisher_smooth_long' in existing_params:
                return max(3, min(10, existing_params['fisher_smooth_long'] - 1))
            return 6.0
        
        elif param_name == 'fisher_buy_threshold':
            # Typically opposite sign of fisher_long_exit
            if 'fisher_long_exit' in existing_params:
                return abs(existing_params['fisher_long_exit']) + 1.0
            return 1.5
        
        elif param_name == 'fisher_sell_threshold':
            # Usually higher than buy threshold
            if 'fisher_buy_threshold' in existing_params:
                return existing_params['fisher_buy_threshold'] + 1.0
            return 2.8
        
        elif param_name == 'goldie_locks':
            # Related to ATR period
            if 'atr_period' in existing_params:
                return 1.5 + (existing_params['atr_period'] - 14) * 0.1
            return 2.0
        
        elif param_name == 'atr_sl_long_multip':
            return 2.5  # Conservative default
        
        elif param_name == 'atr_sl_short_multip':
            return 2.5  # Conservative default
        
        elif param_name == 'rr_long':
            return 3.0  # Good risk/reward ratio
        
        elif param_name == 'rr_short':
            return 3.0  # Good risk/reward ratio
        
        else:
            # Fallback for any other parameters
            return 1.0
    def update_model(self, dataframe: DataFrame, performance: float):
        """Update ML model with enhanced logging - FIXED"""
        try:
            EnhancedLogger.log_section("ML MODEL UPDATE", "🧠")
            
            features = self.create_features(dataframe)
            
            # FIX: Ensure features is properly flattened
            if features.ndim > 1:
                features_flat = features.flatten()
            else:
                features_flat = features
            
            self.performance_history.append({
                'features': features_flat,
                'performance': float(performance),  # Ensure scalar
                'timestamp': datetime.datetime.now()
            })
            
            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
                EnhancedLogger.log_warning("Trimmed history to last 100 samples", "📊")
            
            # Train model if we have enough data
            if len(self.performance_history) >= 20:
                X = np.array([h['features'] for h in self.performance_history])
                y = np.array([h['performance'] for h in self.performance_history])
                
                # FIX: Ensure proper array shapes
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                if y.ndim > 1:
                    y = y.flatten()
                
                # Scale features
                if self.scaler is None:
                    self.scaler = StandardScaler()
                    EnhancedLogger.log_success("Created new feature scaler", "📏")
                    
                X_scaled = self.scaler.fit_transform(X)
                
                # Train model
                if self.model is None:
                    self.model = RandomForestRegressor(n_estimators=50, random_state=42)
                    EnhancedLogger.log_success("Created new RandomForest model", "🌲")
                
                self.model.fit(X_scaled, y)
                model_score = self.model.score(X_scaled, y)
                
                EnhancedLogger.log_success(f"ML Model updated with {len(self.performance_history)} samples", "🧠")
                EnhancedLogger.log_performance("Model R² Score", model_score, "📊")
                
                # Feature importance analysis
                if hasattr(self.model, 'feature_importances_'):
                    top_features = np.argsort(self.model.feature_importances_)[-3:]
                    EnhancedLogger.log_subsection("Top 3 Feature Importance", "🔍")
                    for i, feat_idx in enumerate(reversed(top_features)):
                        importance = self.model.feature_importances_[feat_idx]
                        EnhancedLogger.log_parameter(f"Feature #{feat_idx}", f"{importance:.3f}", "⭐")
                
                self.save_models()
            else:
                samples_needed = 20 - len(self.performance_history)
                EnhancedLogger.log_warning(f"Need {samples_needed} more samples to train model", "📊")
                
        except Exception as e:
            EnhancedLogger.log_error(f"Model update error: {e}")


class Alex_GKD_FisherTransformV5_ML(IStrategy):
    # Strategy parameters
    timeframe = "1h"
    startup_candle_count = 200
    minimal_roi = {}
    stoploss = -0.20
    use_custom_stoploss = False
    trailing_stop = False
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    
    can_short = True
    set_leverage = 3
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.ml_optimizers = {}
        self.last_ml_update = None
        self.ml_update_frequency = 24
        self.trade_performance_cache = {}
        self.enable_ml_optimization = True
        self.initial_training_completed = {}  # Track per-pair training status
        
        # 🚀 STARTUP TRAINING CONFIGURATION
        self.startup_training_enabled = True
        self.startup_training_periods = 1000  # Use last 1000 candles for training
        self.startup_training_pairs = []  # Will be populated with active pairs
        
        logger.info("🤖 Fisher Transform ML Strategy v4 - Enhanced Startup Training")
        logger.info("🎯 Startup training will begin immediately upon first data analysis")
        # Enhanced initialization logging
        EnhancedLogger.log_banner("FISHER TRANSFORM ML STRATEGY INITIALIZED", "🚀")
        EnhancedLogger.log_parameter("Timeframe", self.timeframe, "⏰")
        EnhancedLogger.log_parameter("Can Short", self.can_short, "📊")
        EnhancedLogger.log_parameter("Leverage", self.set_leverage, "⚖️")
        EnhancedLogger.log_parameter("ML Optimization", self.enable_ml_optimization, "🤖")

    # Hyperparameters with ML integration
    if can_short:
        buy_params = {
            "atr_period": 20, "baseline_period": 5, "fisher_buy_threshold": 1.0,  # V1.2: Relaxed from 2.39 to 1.0
            "fisher_period": 14, "fisher_smooth_long": 9, "fisher_smooth_short": 9,
            "goldie_locks": 2.85,
        }
        sell_params = {
            "fisher_long_exit": -0.5, "fisher_short_exit": 0.5, "fisher_sell_threshold": -0.5,  # V1.2: Symmetric thresholds
        }
        minimal_roi = {"0": 0.373, "1019": 0.22, "3124": 0.076, "4482": 0}
        stoploss = -0.15  # V1.2: CRITICAL FIX - Tightened from -0.524 (52%!) to -0.15 (15%)
        trailing_stop = True  # V1.2: Enable trailing stop
        trailing_stop_positive = 0.02
        trailing_stop_positive_offset = 0.05
        trailing_only_offset_is_reached = True
        max_open_trades = 5  # V1.2: Reduced from 20 to limit risk
    else:
        buy_params = {
            "atr_period": 21, "baseline_period": 11, "fisher_buy_threshold": 0.65,
            "fisher_period": 13, "fisher_smooth_long": 7, "goldie_locks": 1.6,
            "fisher_smooth_short": 6,
        }
        sell_params = {
            "fisher_long_exit": 0.837, "fisher_sell_threshold": 2.89, "fisher_short_exit": 0.293,
        }
        minimal_roi = {"0": 0.871, "1787": 0.323, "2415": 0.118, "5669": 0}
        stoploss = -0.591
        trailing_stop = False
        trailing_stop_positive = 0.345
        trailing_stop_positive_offset = 0.373
        trailing_only_offset_is_reached = True
        max_open_trades = 3

    # ML-enhanced parameters with dynamic optimization
    fisher_period = IntParameter(10, 15, default=buy_params.get('fisher_period'), space="buy", optimize=True)
    fisher_smooth_long = IntParameter(3, 10, default=buy_params.get('fisher_smooth_long'), space="buy", optimize=True)
    fisher_smooth_short = IntParameter(3, 10, default=buy_params.get('fisher_smooth_short'), space="buy", optimize=can_short)
    fisher_short_exit = DecimalParameter(-1.0, 1.0, default=sell_params.get('fisher_short_exit'), decimals=3, space="sell", optimize=can_short)
    fisher_long_exit = DecimalParameter(-1.0, 1.0, default=sell_params.get('fisher_long_exit'), decimals=3, space="sell", optimize=True)
    fisher_sell_threshold = DecimalParameter(-2.0, -0.5, default=-1.0, decimals=2, space="sell", optimize=True)  # V1.2: Changed to negative for symmetry
    fisher_buy_threshold = DecimalParameter(0.5, 2.0, default=1.0, decimals=2, space="buy", optimize=True)  # V1.2: Positive for long entries
    baseline_period = IntParameter(5, 21, default=buy_params.get('baseline_period'), space="buy", optimize=True)
    atr_period = IntParameter(7, 21, default=buy_params.get('atr_period'), space="buy", optimize=True)
    goldie_locks = DecimalParameter(1.5, 3.0, default=buy_params.get('goldie_locks'), decimals=2, space="buy", optimize=True)
    
    # ML confidence parameters
    ml_confidence_threshold = DecimalParameter(0.3, 0.5, default=0.4, decimals=2, space="buy", optimize=True)
    ml_adaptation_rate = DecimalParameter(0.1, 0.5, default=0.2, decimals=2, space="buy", optimize=True)
    ml_signal_threshold = DecimalParameter(0.1, 0.8, default=0.4, decimals=2, space="buy", optimize=True)  # Add this line

    # Risk management with ML
    ATR_SL_short_Multip = DecimalParameter(1.0, 6.0, decimals=1, default=1.5, space="sell", optimize=True)
    ATR_SL_long_Multip = DecimalParameter(1.0, 6.0, decimals=1, default=1.5, space="sell", optimize=True)
    ATR_Multip = DecimalParameter(1.0, 6.0, decimals=1, default=1.5, space="sell", optimize=True)
    rr_long = DecimalParameter(1.0, 4.0, decimals=1, default=2.0, space="sell", optimize=True)
    rr_short = DecimalParameter(1.0, 4.0, decimals=1, default=2.0, space="sell", optimize=True)
    
    # DCA Configuration
    overbuy_factor = 1.295
    position_adjustment_enable = True
    initial_safety_order_trigger = -0.02
    max_so_multiplier_orig = 3
    safety_order_step_scale = 2
    safety_order_volume_scale = 1.8
    max_so_multiplier = max_so_multiplier_orig
    cust_proposed_initial_stakes = {}
    partial_fill_compensation_scale = 1
    
    # DCA calculation
    if max_so_multiplier_orig > 0:
        if safety_order_volume_scale > 1:
            firstLine = safety_order_volume_scale * (math.pow(safety_order_volume_scale, (max_so_multiplier_orig - 1)) - 1)
            divisor = safety_order_volume_scale - 1
            max_so_multiplier = 2 + firstLine / divisor
        elif safety_order_volume_scale < 1:
            firstLine = safety_order_volume_scale * (1 - math.pow(safety_order_volume_scale, (max_so_multiplier_orig - 1)))
            divisor = 1 - safety_order_volume_scale
            max_so_multiplier = 2 + firstLine / divisor
    
    #stoploss = -1
    
    def get_ml_adjusted_params(self, dataframe: DataFrame, pair: str) -> dict:
        """Get ML-adjusted parameters based on market conditions per pair - ENHANCED ERROR HANDLING"""
        try:
            EnhancedLogger.log_section(f"ML PARAMETER ADJUSTMENT - {pair}", "🤖")
            
            # Skip ML if disabled or insufficient data
            if not self.enable_ml_optimization or len(dataframe) < 50:
                EnhancedLogger.log_warning(f"ML optimization skipped for {pair} (disabled or insufficient data)", "⚠️")
                return {}
            
            # Ensure required columns exist before ML operations
            required_columns = ['close', 'high', 'low']
            if not all(col in dataframe.columns for col in required_columns):
                EnhancedLogger.log_error(f"Missing required columns for {pair}", "❌")
                return {}
            
            # Create pair-specific optimizer if doesn't exist
            if pair not in self.ml_optimizers:
                self.ml_optimizers[pair] = MLOptimizer(f"GKD_FisherV5_{pair.replace('/', '_')}")
                EnhancedLogger.log_success(f"Created ML optimizer for {pair}", "🆕")
            
            ml_optimizer = self.ml_optimizers[pair]
            
            # Check if it's time to update ML model for this pair
            current_time = datetime.datetime.now()
            should_update = (self.last_ml_update is None or 
                            (current_time - self.last_ml_update).total_seconds() > self.ml_update_frequency * 3600)
            
            if should_update and len(dataframe) > 100:
                try:
                    EnhancedLogger.log_subsection(f"Updating ML model for {pair}", "🔄")
                    
                    # Update ML model with recent performance for this pair
                    recent_performance = self.calculate_recent_performance(pair)
                    EnhancedLogger.log_performance("Recent Performance", recent_performance, "📊")
                    
                    ml_optimizer.update_model(dataframe, recent_performance)
                    self.last_ml_update = current_time
                    
                    EnhancedLogger.log_success(f"ML model updated at {current_time.strftime('%H:%M:%S')}", "✅")
                    
                    # Optimize parameters with Optuna for this specific pair
                    optimized_params = ml_optimizer.optimize_parameters(dataframe, recent_performance)
                    
                    if optimized_params:
                        # Ensure all exit parameters are present in optimized results
                        self._ensure_all_parameters(optimized_params, pair)
                        
                        # Add optimization score
                        optimized_params['score'] = ml_optimizer.study.best_value if ml_optimizer.study else 0.0
                        
                        EnhancedLogger.log_success(f"Optuna returned {len(optimized_params)} parameters", "🎯")
                        
                        # Log the optimized parameters with enhanced formatting
                        self.log_formatted_parameters(pair, optimized_params)
                        
                        return optimized_params
                    else:
                        EnhancedLogger.log_warning(f"Optuna optimization returned empty results for {pair}", "⚠️")
                
                except Exception as e:
                    EnhancedLogger.log_error(f"ML optimization error for {pair}: {str(e)}", "❌")
            
            # Get ML predictions for optimal parameters for this pair (fallback)
            try:
                EnhancedLogger.log_subsection(f"Getting ML predictions for {pair}", "🔮")
                ml_params = ml_optimizer.predict_optimal_params(dataframe)
                
                # ALWAYS ensure exit parameters are included (critical fix)
                self._ensure_all_parameters(ml_params, pair)
                
                if ml_params:
                    EnhancedLogger.log_success(f"Using ML predicted parameters: {len(ml_params)} total", "✨")
                
                return ml_params
                
            except Exception as e:
                EnhancedLogger.log_error(f"ML prediction error for {pair}: {str(e)}", "❌")
                return self._get_default_parameters()
                
        except Exception as e:
            EnhancedLogger.log_error(f"ML adjustment error for {pair}: {str(e)}", "💥")
            return self._get_default_parameters()

# PART 3 - Continuing from Part 2

    def _get_default_parameters(self) -> dict:
        """Return default parameters as fallback with enhanced logging"""
        EnhancedLogger.log_warning("Using default parameters as fallback", "🔄")
        return {
            'fisher_long_exit': self.fisher_long_exit.value,
            'fisher_short_exit': self.fisher_short_exit.value,
            'fisher_sell_threshold': self.fisher_sell_threshold.value,
            'atr_sl_long_multip': self.ATR_SL_long_Multip.value,
            'atr_sl_short_multip': self.ATR_SL_short_Multip.value,
            'rr_long': self.rr_long.value,
            'rr_short': self.rr_short.value
        }
    
    def _ensure_all_parameters(self, params: dict, pair: str) -> None:
        """Ensure all required parameters are present - COMPREHENSIVE VERSION"""
        required_params = {
            # Fisher Transform Parameters
            'fisher_period': 14,
            'fisher_smooth_long': 7,
            'fisher_smooth_short': 6,
            'fisher_buy_threshold': 1.5,
            'fisher_sell_threshold': 2.8,
            'fisher_long_exit': -0.5,
            'fisher_short_exit': 0.5,
            
            # Baseline & ATR Parameters  
            'baseline_period': 14,
            'atr_period': 14,
            'goldie_locks': 2.0,
            
            # Risk Management Parameters
            'atr_sl_long_multip': 2.5,
            'atr_sl_short_multip': 2.5,
            'rr_long': 3.0,
            'rr_short': 3.0
        }
        
        missing_count = 0
        added_params = []
        
        for param, default_value in required_params.items():
            if param not in params:
                params[param] = default_value
                missing_count += 1
                added_params.append(param)
        
        if missing_count > 0:
            EnhancedLogger.log_warning(f"Added {missing_count} missing parameters for {pair}", "🔧")
            # Log which parameters were added (for debugging)
            EnhancedLogger.log_parameter("Added Parameters", ", ".join(added_params[:3]) + "..." if len(added_params) > 3 else ", ".join(added_params), "📋")
        else:
            EnhancedLogger.log_success(f"All {len(required_params)} parameters verified for {pair}", "✅")
    
    def calculate_recent_performance(self, pair: str = None) -> float:
        """Calculate recent strategy performance for specific pair or overall with enhanced logging"""
        try:
            if pair:
                pair_trades = [perf for p, perf in self.trade_performance_cache.items() if p == pair]
                if not pair_trades:
                    EnhancedLogger.log_warning(f"No trade history for {pair}", "📊")
                    return 0.0
                performance = sum(pair_trades[-5:]) / len(pair_trades[-5:])
                EnhancedLogger.log_performance(f"Recent Performance ({pair})", performance, "🎯")
                return performance
            else:
                if not self.trade_performance_cache:
                    EnhancedLogger.log_warning("No trade history available", "📊")
                    return 0.0
                recent_trades = list(self.trade_performance_cache.values())[-10:]
                performance = sum(recent_trades) / len(recent_trades)
                EnhancedLogger.log_performance("Overall Recent Performance", performance, "🌟")
                return performance
        except Exception as e:
            EnhancedLogger.log_error(f"Performance calculation error: {e}", "💥")
            return 0.0
    
    def log_formatted_parameters(self, pair: str, params: Dict[str, Any]):
        """Log parameters in a beautifully formatted way with emojis and enhanced visuals"""
        EnhancedLogger.log_banner(f"OPTIMIZED PARAMETERS FOR {pair}", "🎯")
        
        # Fisher Transform Parameters Section
        EnhancedLogger.log_section("FISHER TRANSFORM SETTINGS", "🎣")
        fisher_params = {
            "fisher_period": ("🔄", "Period"),
            "fisher_smooth_long": ("📈", "Long Smooth"),
            "fisher_smooth_short": ("📉", "Short Smooth"), 
            "fisher_buy_threshold": ("🚀", "Buy Threshold")
        }
        
        for param, (emoji, name) in fisher_params.items():
            if param in params:
                EnhancedLogger.log_parameter(name, params[param], emoji)
        
        # Fisher Exit Parameters Section  
        EnhancedLogger.log_section("FISHER EXIT SETTINGS", "🚪")
        exit_params = {
            "fisher_long_exit": ("📤", "Long Exit"),
            "fisher_short_exit": ("📥", "Short Exit"),
            "fisher_sell_threshold": ("🛑", "Sell Threshold")
        }
        
        for param, (emoji, name) in exit_params.items():
            if param in params:
                value = params[param]
                # Color coding for exit levels
                if isinstance(value, (int, float)):
                    if value > 0:
                        color_status = "🟢 POSITIVE"
                    elif value < 0:
                        color_status = "🔴 NEGATIVE" 
                    else:
                        color_status = "🟡 NEUTRAL"
                    EnhancedLogger.log_parameter(f"{name} {color_status}", f"{value:.3f}", emoji)
                else:
                    EnhancedLogger.log_parameter(name, value, emoji)
        
        # Baseline & Volatility Section
        EnhancedLogger.log_section("BASELINE & VOLATILITY", "📊")
        baseline_params = {
            "baseline_period": ("📏", "Baseline Period"),
            "atr_period": ("🌊", "ATR Period"),
            "goldie_locks": ("🔒", "Goldie Locks Zone")
        }
        
        for param, (emoji, name) in baseline_params.items():
            if param in params:
                EnhancedLogger.log_parameter(name, params[param], emoji)
        
        # Risk Management Section
        EnhancedLogger.log_section("RISK MANAGEMENT", "⚖️")
        risk_params = {
            "atr_sl_long_multip": ("🛡️", "Long SL Multiplier"),
            "atr_sl_short_multip": ("🛡️", "Short SL Multiplier"),
            "rr_long": ("💰", "Long Risk/Reward"),
            "rr_short": ("💰", "Short Risk/Reward")
        }
        
        for param, (emoji, name) in risk_params.items():
            if param in params:
                value = params[param]
                if isinstance(value, (int, float)):
                    # Risk level indication
                    if 'sl_' in param:  # Stop loss multipliers
                        risk_level = "🟢 CONSERVATIVE" if value <= 2.0 else "🟡 MODERATE" if value <= 4.0 else "🔴 AGGRESSIVE"
                        EnhancedLogger.log_parameter(f"{name} ({risk_level})", f"{value:.2f}x", emoji)
                    else:  # Risk/Reward ratios
                        rr_quality = "🟢 EXCELLENT" if value >= 3.0 else "🟡 GOOD" if value >= 2.0 else "🔴 RISKY"
                        EnhancedLogger.log_parameter(f"{name} ({rr_quality})", f"{value:.1f}:1", emoji)
                else:
                    EnhancedLogger.log_parameter(name, value, emoji)
        
        # Optimization Quality Assessment
        if 'score' in params:
            score = params['score']
            if score > 0.1:
                quality = "🟢 EXCELLENT"
            elif score > 0.05:
                quality = "🟡 GOOD"
            elif score > 0:
                quality = "🟠 FAIR"
            else:
                quality = "🔴 POOR"
            
            EnhancedLogger.log_section("OPTIMIZATION QUALITY", "📈")
            EnhancedLogger.log_performance(f"Score {quality}", score, "🏆")
        
        # Summary
        param_count = len([p for p in params.keys() if p != 'score'])
        EnhancedLogger.log_section("PARAMETER SUMMARY", "📋")
        EnhancedLogger.log_parameter("Total Parameters", param_count, "🔢")
        EnhancedLogger.log_parameter("Optimization Time", datetime.datetime.now().strftime("%H:%M:%S"), "⏰")
        
        # Visual separator
        logger.info("🔹" * 60)
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Populate indicators with enhanced ML integration and logging"""
        """Enhanced with STARTUP TRAINING"""
        pair = metadata.get('pair', 'Unknown')
        
        # 🚀 STARTUP TRAINING - Run on first data load
        if (self.startup_training_enabled and 
            pair not in self.initial_training_completed and 
            len(dataframe) >= self.startup_training_periods):
            
            logger.info(f"🎯 [STARTUP] Beginning initial ML training for {pair}")
            self.perform_startup_training(dataframe, pair)
            self.initial_training_completed[pair] = True
        EnhancedLogger.log_banner(f"POPULATING INDICATORS - {pair}", "📊")
        
        # Use default values initially
        fisher_period = self.fisher_period.value
        fisher_smooth_long = self.fisher_smooth_long.value
        fisher_smooth_short = self.fisher_smooth_short.value
        baseline_period = self.baseline_period.value
        atr_period = self.atr_period.value
        
        EnhancedLogger.log_section("INITIAL PARAMETERS", "⚙️")
        EnhancedLogger.log_parameter("Fisher Period", fisher_period, "🎣")
        EnhancedLogger.log_parameter("Baseline Period", baseline_period, "📏")
        EnhancedLogger.log_parameter("ATR Period", atr_period, "🌊")
        
        # Calculate basic indicators first (required for ML features)
        EnhancedLogger.log_subsection("Calculating base indicators", "🔧")
        
        try:
            dataframe["atr"] = talib.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=atr_period)
            EnhancedLogger.log_success("ATR calculated", "✅")
            
            dataframe["fisher"] = self.calculate_fisher(dataframe, fisher_period)
            EnhancedLogger.log_success("Fisher Transform calculated", "✅")
            
            dataframe["baseline"] = ema(dataframe["close"], length=baseline_period)
            dataframe["baseline_diff"] = dataframe["baseline"].diff()
            EnhancedLogger.log_success("Baseline indicators calculated", "✅")
            
        except Exception as e:
            EnhancedLogger.log_error(f"Base indicator calculation failed: {e}", "💥")
            raise
        
        # NOW get ML-adjusted parameters after basic indicators exist
        EnhancedLogger.log_subsection("Applying ML adjustments", "🤖")
        ml_params = self.get_ml_adjusted_params(dataframe, pair)
        
        # Apply ML adjustments to parameters if available
        if ml_params:
            EnhancedLogger.log_subsection("ML parameters detected, recalculating...", "🔄")
            
            original_params = {
                'fisher_period': fisher_period,
                'fisher_smooth_long': fisher_smooth_long, 
                'fisher_smooth_short': fisher_smooth_short,
                'baseline_period': baseline_period,
                'atr_period': atr_period
            }
            
            # Update parameters with ML suggestions
            fisher_period = ml_params.get('fisher_period', fisher_period)
            fisher_smooth_long = ml_params.get('fisher_smooth_long', fisher_smooth_long)
            fisher_smooth_short = ml_params.get('fisher_smooth_short', fisher_smooth_short)
            baseline_period = ml_params.get('baseline_period', baseline_period)
            atr_period = ml_params.get('atr_period', atr_period)
            
            # Ensure parameters are within valid ranges
            fisher_period = max(10, min(15, int(fisher_period)))
            fisher_smooth_long = max(3, min(10, int(fisher_smooth_long)))
            fisher_smooth_short = max(3, min(10, int(fisher_smooth_short)))
            baseline_period = max(5, min(21, int(baseline_period)))
            atr_period = max(7, min(21, int(atr_period)))
            
            # Log parameter changes
            changes_made = 0
            for param_name in original_params:
                old_val = original_params[param_name]
                new_val = locals()[param_name]
                if old_val != new_val:
                    changes_made += 1
                    EnhancedLogger.log_parameter(f"{param_name} changed", f"{old_val} → {new_val}", "🔄")
            
            if changes_made > 0:
                EnhancedLogger.log_success(f"Applied {changes_made} ML parameter adjustments", "🎯")
                
                # Recalculate indicators with ML-adjusted parameters
                if fisher_period != self.fisher_period.value:
                    dataframe["fisher"] = self.calculate_fisher(dataframe, fisher_period)
                    EnhancedLogger.log_success("Fisher recalculated with ML params", "🔄")
                
                if baseline_period != self.baseline_period.value:
                    dataframe["baseline"] = ema(dataframe["close"], length=baseline_period)
                    dataframe["baseline_diff"] = dataframe["baseline"].diff()
                    EnhancedLogger.log_success("Baseline recalculated with ML params", "🔄")
                
                if atr_period != self.atr_period.value:
                    dataframe["atr"] = talib.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=atr_period)
                    EnhancedLogger.log_success("ATR recalculated with ML params", "🔄")
            else:
                EnhancedLogger.log_success("ML parameters match defaults", "✨")
        else:
            EnhancedLogger.log_warning("No ML parameters available, using defaults", "⚠️")
        
        # Continue with remaining indicators
        EnhancedLogger.log_subsection("Calculating derived indicators", "🔧")
        
        try:
            # Smooth Fisher with EMA
            dataframe["fisher_smooth_long"] = ema(dataframe["fisher"], length=fisher_smooth_long)
            dataframe["fisher_smooth_short"] = ema(dataframe["fisher"], length=fisher_smooth_short)
            # FIXED: trend_long now uses fisher_smooth_long instead of fisher_smooth_short
            dataframe["fisher_trend_long"] = ema(dataframe["fisher_smooth_long"], length=21)
            # FIXED: trend_short uses shorter length (9) for faster response
            dataframe["fisher_trend_short"] = ema(dataframe["fisher_smooth_short"], length=21)
            EnhancedLogger.log_success("Fisher smoothing complete (FIXED trends)", "✅")
            
            # Baseline indicators
            dataframe["baseline_up"] = dataframe["baseline_diff"] > 0
            dataframe["baseline_down"] = dataframe["baseline_diff"] < 0
            trend_up_pct = (dataframe["baseline_up"].tail(50).sum() / 50) * 100
            EnhancedLogger.log_parameter("Baseline Uptrend %", f"{trend_up_pct:.1f}%", "📈")
            
            # Volatility (ATR for Goldie Locks Zone)
            dataframe["goldie_min"] = dataframe["baseline"] - (dataframe["atr"] * self.goldie_locks.value)
            dataframe["goldie_max"] = dataframe["baseline"] + (dataframe["atr"] * self.goldie_locks.value)
            EnhancedLogger.log_success("Goldie Locks zones calculated", "✅")
            
            # ML confidence indicators
            dataframe["ml_confidence"] = self.calculate_ml_confidence(dataframe)
            dataframe["market_regime"] = self.identify_market_regime(dataframe)
            
            # Enhanced signals with ML
            dataframe["ml_signal_strength"] = self.calculate_signal_strength(dataframe)
            
            # Log ML indicator statistics
            avg_confidence = dataframe["ml_confidence"].tail(50).mean()
            avg_signal_strength = dataframe["ml_signal_strength"].tail(50).mean()
            current_regime = dataframe["market_regime"].iloc[-1]
            
            regime_text = "🐂 BULL" if current_regime > 0 else "🐻 BEAR" if current_regime < 0 else "⚖️ NEUTRAL"
            
            EnhancedLogger.log_section("ML INDICATOR SUMMARY", "🤖")
            EnhancedLogger.log_parameter("Avg ML Confidence", f"{avg_confidence:.1%}", "🎯")
            EnhancedLogger.log_parameter("Avg Signal Strength", f"{avg_signal_strength:.3f}", "⚡")
            EnhancedLogger.log_parameter("Market Regime", regime_text, "🏛️")
            
            EnhancedLogger.log_success("All ML indicators calculated", "✅")
            
        except Exception as e:
            EnhancedLogger.log_error(f"Derived indicator calculation failed: {e}", "💥")
            raise
        
        # Final summary
        if ml_params:
            EnhancedLogger.log_success(f"ML parameters active: {len(ml_params)} adjustments", "🎯")
            EnhancedLogger.log_parameter("Active optimizers", len(self.ml_optimizers), "🤖")
        
        EnhancedLogger.log_banner(f"INDICATORS COMPLETE - {pair}", "🎉")
        
        return dataframe

    def perform_startup_training(self, dataframe: DataFrame, pair: str):
        """NEW: Perform ML training on startup using historical data"""
        try:
            logger.info(f"🧠 [STARTUP] Training ML model for {pair} with {len(dataframe)} historical candles")
            
            # Create pair-specific optimizer if doesn't exist
            if pair not in self.ml_optimizers:
                self.ml_optimizers[pair] = MLOptimizer(f"GKD_FisherV5_{pair.replace('/', '_')}")
            
            ml_optimizer = self.ml_optimizers[pair]
            
            # 📊 Generate synthetic training data from historical patterns
            training_data = self.generate_historical_training_data(dataframe, pair)
            
            if len(training_data) > 0:
                logger.info(f"📈 [STARTUP] Generated {len(training_data)} training samples for {pair}")
                
                # Update ML optimizer with historical performance patterns
                for sample in training_data:
                    ml_optimizer.performance_history.append(sample)
                
                # Train the model immediately
                if len(ml_optimizer.performance_history) >= 20:
                    ml_optimizer.update_model(dataframe, 0.0)  # Use neutral performance for initial training
                    logger.info(f"✅ [STARTUP] ML model trained successfully for {pair}")
                    
                    # Run initial Optuna optimization
                    logger.info(f"🎯 [STARTUP] Running initial parameter optimization for {pair}")
                    optimized_params = ml_optimizer.optimize_parameters(dataframe, 0.0)
                    
                    if optimized_params:
                        logger.info(f"🎉 [STARTUP] Initial optimization complete for {pair}")
                        self.log_formatted_parameters(pair, optimized_params)
                    else:
                        logger.warning(f"⚠️ [STARTUP] Initial optimization failed for {pair}")
                else:
                    logger.warning(f"⚠️ [STARTUP] Insufficient training data generated for {pair}")
            else:
                logger.error(f"❌ [STARTUP] Failed to generate training data for {pair}")
                
        except Exception as e:
            logger.error(f"❌ [STARTUP] Training failed for {pair}: {str(e)}")

    def generate_historical_training_data(self, dataframe: DataFrame, pair: str) -> List[Dict]:
        """NEW: Generate training data from historical price patterns"""
        try:
            training_samples = []
            lookback_period = min(500, len(dataframe) - 100)  # Use up to 500 candles for training
            
            logger.info(f"🔍 [STARTUP] Analyzing {lookback_period} historical periods for {pair}")
            
            # Calculate basic indicators needed for analysis
            dataframe_copy = dataframe.copy()
            dataframe_copy["atr"] = talib.ATR(dataframe_copy["high"], dataframe_copy["low"], 
                                             dataframe_copy["close"], timeperiod=14)
            dataframe_copy["fisher"] = self.calculate_fisher(dataframe_copy, 14)
            dataframe_copy["baseline"] = ema(dataframe_copy["close"], length=14)
            dataframe_copy["baseline_diff"] = dataframe_copy["baseline"].diff()
            
            # Generate training samples by analyzing historical patterns
            for i in range(100, lookback_period):  # Skip first 100 for indicator stability
                try:
                    # Extract features at this historical point
                    features = self.extract_features_at_index(dataframe_copy, i)
                    
                    # Calculate performance of next 10-20 candles as "target"
                    future_performance = self.calculate_future_performance(dataframe_copy, i, periods=15)
                    
                    if not np.isnan(future_performance) and abs(future_performance) < 0.5:  # Filter extreme values
                        training_sample = {
                            'features': features,
                            'performance': future_performance,
                            'timestamp': datetime.datetime.now() - datetime.timedelta(hours=lookback_period-i)
                        }
                        training_samples.append(training_sample)
                        
                except Exception as e:
                    continue  # Skip problematic samples
            
            logger.info(f"📊 [STARTUP] Generated {len(training_samples)} valid training samples for {pair}")
            return training_samples
            
        except Exception as e:
            logger.error(f"❌ [STARTUP] Error generating training data for {pair}: {str(e)}")
            return []
    def extract_features_at_index(self, dataframe: DataFrame, index: int) -> np.ndarray:
        """Extract ML features at a specific historical index"""
        try:
            features = []
            
            # Market volatility features
            atr_mean = dataframe['atr'].iloc[max(0, index-14):index].mean()
            features.append(atr_mean if not pd.isna(atr_mean) else 0.01)
            
            atr_std = dataframe['atr'].iloc[max(0, index-7):index].std()
            features.append(atr_std if not pd.isna(atr_std) else 0.001)
            
            # Price momentum features
            for period in [5, 10, 20]:
                pct_change = dataframe['close'].iloc[index] / dataframe['close'].iloc[max(0, index-period)] - 1
                features.append(pct_change if not pd.isna(pct_change) else 0.0)
            
            # Volume features (with defaults)
            if 'volume' in dataframe.columns:
                vol_mean = dataframe['volume'].iloc[max(0, index-14):index].mean()
                vol_pct = (dataframe['volume'].iloc[index] / dataframe['volume'].iloc[max(0, index-1)] - 1 
                          if index > 0 else 0.0)
            else:
                vol_mean, vol_pct = 1000.0, 0.0
            
            features.extend([vol_mean if not pd.isna(vol_mean) else 1000.0, 
                            vol_pct if not pd.isna(vol_pct) else 0.0])
            
            # Fisher transform features
            fisher_current = dataframe['fisher'].iloc[index]
            fisher_mean = dataframe['fisher'].iloc[max(0, index-5):index].mean()
            fisher_std = dataframe['fisher'].iloc[max(0, index-5):index].std()
            
            features.extend([
                fisher_current if not pd.isna(fisher_current) else 0.0,
                fisher_mean if not pd.isna(fisher_mean) else 0.0,
                fisher_std if not pd.isna(fisher_std) else 1.0
            ])
            
            # Baseline trend features
            baseline_diff_mean = dataframe['baseline_diff'].iloc[max(0, index-5):index].mean()
            baseline_diff_sum = dataframe['baseline_diff'].iloc[max(0, index-10):index].sum()
            
            features.extend([
                baseline_diff_mean if not pd.isna(baseline_diff_mean) else 0.0,
                baseline_diff_sum if not pd.isna(baseline_diff_sum) else 0.0
            ])
            
            # Market regime
            sma_50 = dataframe['close'].iloc[max(0, index-50):index].mean()
            sma_200 = dataframe['close'].iloc[max(0, index-200):index].mean()
            
            if not pd.isna(sma_50) and not pd.isna(sma_200) and sma_200 != 0:
                features.append(1.0 if sma_50 > sma_200 else 0.0)
            else:
                features.append(0.5)
            
            # Ensure exactly 12 features
            while len(features) < 12:
                features.append(0.0)
            features = features[:12]
            
            return np.array(features)
            
        except Exception as e:
            return np.zeros(12)

    def calculate_future_performance(self, dataframe: DataFrame, index: int, periods: int = 15) -> float:
        """Calculate future performance for training target"""
        try:
            if index + periods >= len(dataframe):
                return 0.0
            
            # Simple return calculation
            current_price = dataframe['close'].iloc[index]
            future_price = dataframe['close'].iloc[index + periods]
            
            if current_price > 0:
                return (future_price - current_price) / current_price
            else:
                return 0.0
                
        except:
            return 0.0
    def calculate_ml_confidence(self, dataframe: DataFrame) -> pd.Series:
        """Calculate ML model confidence for signals with enhanced logging"""
        try:
            EnhancedLogger.log_subsection("Calculating ML confidence", "🎯")
            
            # Simple confidence calculation based on market volatility and trend consistency
            atr_norm = dataframe["atr"] / dataframe["close"]
            trend_consistency = abs(dataframe["baseline_diff"].rolling(10).mean())
            fisher_volatility = dataframe["fisher"].rolling(10).std()
            
            # Higher confidence in stable, trending markets
            confidence = 1.0 - (atr_norm * 2 + fisher_volatility * 0.5)
            confidence = confidence.fillna(0.5).clip(0.1, 1.0)
            
            # Log confidence statistics
            avg_confidence = confidence.tail(20).mean()
            min_confidence = confidence.tail(20).min()
            max_confidence = confidence.tail(20).max()
            
            EnhancedLogger.log_parameter("Avg Confidence", f"{avg_confidence:.1%}", "🎯")
            EnhancedLogger.log_parameter("Min Confidence", f"{min_confidence:.1%}", "🔽")
            EnhancedLogger.log_parameter("Max Confidence", f"{max_confidence:.1%}", "🔼")
            
            return confidence
            
        except Exception as e:
            EnhancedLogger.log_error(f"ML confidence calculation error: {e}", "💥")
            return pd.Series(0.5, index=dataframe.index)
    
    def identify_market_regime(self, dataframe: DataFrame) -> pd.Series:
        """Identify market regime using ML features with enhanced logging"""
        try:
            EnhancedLogger.log_subsection("Identifying market regime", "🏛️")
            
            sma_50 = dataframe["close"].rolling(50).mean()
            sma_200 = dataframe["close"].rolling(200).mean()
            
            # Market regimes: 1=Bull, 0=Neutral, -1=Bear
            regime = pd.Series(0, index=dataframe.index)
            regime.loc[sma_50 > sma_200 * 1.02] = 1  # Bull market
            regime.loc[sma_50 < sma_200 * 0.98] = -1  # Bear market
            
            # Calculate regime statistics
            recent_regime = regime.tail(50)
            bull_periods = (recent_regime == 1).sum()
            bear_periods = (recent_regime == -1).sum()
            neutral_periods = (recent_regime == 0).sum()
            
            EnhancedLogger.log_parameter("Bull Periods", f"{bull_periods}/50 ({bull_periods*2:.0f}%)", "🐂")
            EnhancedLogger.log_parameter("Bear Periods", f"{bear_periods}/50 ({bear_periods*2:.0f}%)", "🐻") 
            EnhancedLogger.log_parameter("Neutral Periods", f"{neutral_periods}/50 ({neutral_periods*2:.0f}%)", "⚖️")
            
            current_regime = regime.iloc[-1]
            if current_regime > 0:
                EnhancedLogger.log_success("Current: BULL MARKET", "🐂")
            elif current_regime < 0:
                EnhancedLogger.log_warning("Current: BEAR MARKET", "🐻")
            else:
                EnhancedLogger.log_subsection("Current: NEUTRAL MARKET", "⚖️")
            
            return regime
            
        except Exception as e:
            EnhancedLogger.log_error(f"Market regime identification error: {e}", "💥")
            return pd.Series(0, index=dataframe.index)
    
    def calculate_signal_strength(self, dataframe: DataFrame) -> pd.Series:
        """Calculate signal strength using multiple indicators with enhanced logging"""
        try:
            EnhancedLogger.log_subsection("Calculating signal strength", "⚡")
            
            # Combine multiple signal components
            fisher_strength = abs(dataframe["fisher"]) / 3.0  # Normalize
            trend_strength = abs(dataframe["baseline_diff"]) / dataframe["atr"]
            volume_strength = 1.0  # Default if no volume data
            
            if 'volume' in dataframe.columns:
                volume_ma = dataframe['volume'].rolling(20).mean()
                volume_strength = (dataframe['volume'] / volume_ma).clip(0.5, 2.0) / 2.0
                EnhancedLogger.log_success("Volume strength included", "📊")
            else:
                EnhancedLogger.log_warning("No volume data, using default", "📊")
            
            # Combined signal strength
            signal_strength = (fisher_strength * 0.4 + trend_strength * 0.4 + volume_strength * 0.2)
            signal_strength = signal_strength.fillna(0.5).clip(0.1, 1.0)
            
            # Log signal strength statistics
            avg_strength = signal_strength.tail(20).mean()
            current_strength = signal_strength.iloc[-1]
            strong_signals = (signal_strength.tail(50) > 0.7).sum()
            
            EnhancedLogger.log_parameter("Avg Signal Strength", f"{avg_strength:.3f}", "⚡")
            EnhancedLogger.log_parameter("Current Strength", f"{current_strength:.3f}", "📊")
            EnhancedLogger.log_parameter("Strong Signals (>0.7)", f"{strong_signals}/50", "💪")
            
            if current_strength > 0.8:
                EnhancedLogger.log_success("VERY STRONG signal detected", "🚀")
            elif current_strength > 0.6:
                EnhancedLogger.log_success("STRONG signal detected", "💪")
            elif current_strength > 0.4:
                EnhancedLogger.log_warning("MODERATE signal detected", "⚡")
            else:
                EnhancedLogger.log_warning("WEAK signal detected", "🔋")
            
            return signal_strength
            
        except Exception as e:
            EnhancedLogger.log_error(f"Signal strength calculation error: {e}", "💥")
            return pd.Series(0.5, index=dataframe.index)
    
    def calculate_fisher(self, dataframe: DataFrame, period: int) -> pd.Series:
        """Fisher Transform calculation with ML enhancements and logging"""
        try:
            EnhancedLogger.log_subsection(f"Calculating Fisher Transform (period={period})", "🎣")
            
            median_price = (dataframe["high"] + dataframe["low"]) / 2
            fisher = pd.Series(0.0, index=dataframe.index)
            
            for i in range(period, len(dataframe)):
                price_window = median_price.iloc[i-period:i]
                price_min = price_window.min()
                price_max = price_window.max()
                if price_max != price_min:
                    norm = (median_price.iloc[i] - price_min) / (price_max - price_min)
                    norm = 2 * norm - 1
                    norm = max(min(norm, 0.999), -0.999)
                    fisher.iloc[i] = 0.5 * np.log((1 + norm) / (1 - norm))
                else:
                    fisher.iloc[i] = 0.0
            
            # Log Fisher Transform statistics
            current_fisher = fisher.iloc[-1]
            avg_fisher = fisher.tail(50).mean()
            std_fisher = fisher.tail(50).std()
            
            EnhancedLogger.log_parameter("Current Fisher", f"{current_fisher:.3f}", "🎣")
            EnhancedLogger.log_parameter("Avg Fisher (50)", f"{avg_fisher:.3f}", "📊")
            EnhancedLogger.log_parameter("Fisher Volatility", f"{std_fisher:.3f}", "🌊")
            
            if abs(current_fisher) > 2.0:
                EnhancedLogger.log_warning("Fisher in extreme territory", "⚠️")
            elif abs(current_fisher) > 1.0:
                EnhancedLogger.log_success("Fisher showing strong signal", "💪")
            
            return fisher
            
        except Exception as e:
            EnhancedLogger.log_error(f"Fisher Transform calculation error: {e}", "💥")
            return pd.Series(0.0, index=dataframe.index)
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Populate entry trend with enhanced ML integration and logging"""
        pair = metadata.get('pair', 'Unknown')
        
        EnhancedLogger.log_banner(f"ENTRY SIGNAL ANALYSIS - {pair}", "🎯")
        
        # Get ML-adjusted parameters for this pair
        ml_params = self.get_ml_adjusted_params(dataframe, pair)
        
        # Use ML-adjusted thresholds
        fisher_buy_threshold = ml_params.get('fisher_buy_threshold', self.fisher_buy_threshold.value)
        fisher_sell_threshold = ml_params.get('fisher_sell_threshold', self.fisher_sell_threshold.value)
        
        EnhancedLogger.log_section("ENTRY THRESHOLDS", "🎚️")
        EnhancedLogger.log_parameter("Buy Threshold", fisher_buy_threshold, "🟢")
        EnhancedLogger.log_parameter("Sell Threshold", fisher_sell_threshold, "🔴")
        
        # ML-enhanced entry logic
        #ml_confidence_condition = dataframe["ml_confidence"] > self.ml_confidence_threshold.value
        #signal_strength_condition = dataframe["ml_signal_strength"] > 0.6
        ml_confidence_condition = dataframe["ml_confidence"] > self.ml_confidence_threshold.value
        signal_strength_condition = dataframe["ml_signal_strength"] > self.ml_signal_threshold.value
        
        # Count conditions for logging
        ml_conf_count = ml_confidence_condition.sum()
        signal_str_count = signal_strength_condition.sum()
        
        EnhancedLogger.log_section("ML CONDITIONS", "🤖")
        EnhancedLogger.log_parameter("High Confidence Periods", f"{ml_conf_count}/{len(dataframe)}", "🎯")
        EnhancedLogger.log_parameter("Strong Signal Periods", f"{signal_str_count}/{len(dataframe)}", "⚡")
        
        # V1.2 BALANCED: Long entry with symmetric conditions
        long_conditions = (
            (dataframe["fisher"] < fisher_buy_threshold) &
            (dataframe["fisher"] > dataframe['fisher_smooth_long']) &
            (dataframe["fisher_trend_long"] > dataframe["fisher_trend_long"].shift(1)) &
            (dataframe["baseline_up"] | (dataframe["market_regime"] > 0)) &
            (ml_confidence_condition | signal_strength_condition)
        )
        
        dataframe.loc[long_conditions, ["enter_long", "enter_tag"]] = [1, "fisher_long_v12"]
        
        long_signals = long_conditions.sum()
        EnhancedLogger.log_parameter("Long Entry Signals", long_signals, "🟢")
        
        # V1.2 BALANCED: Short entry with symmetric conditions
        if self.can_short:
            short_conditions = (
                (dataframe["fisher_smooth_short"] < fisher_sell_threshold) &
                (dataframe["baseline_down"]) &
                (dataframe["close"] >= dataframe["goldie_min"]) &
                (dataframe["close"] <= dataframe["goldie_max"]) &
                ml_confidence_condition &
                signal_strength_condition &
                (dataframe["market_regime"] <= 0)  # Neutral or bear market
            )
            dataframe.loc[short_conditions, ["enter_short", "enter_tag"]] = [1, "fisher_short_v12"]
            
            short_signals = short_conditions.sum()
            EnhancedLogger.log_parameter("Short Entry Signals", short_signals, "🔴")
        else:
            EnhancedLogger.log_warning("Short trading disabled", "⚠️")
        
        # Log recent entry signals
        recent_long = dataframe["enter_long"].tail(20).sum()
        if self.can_short:
            recent_short = dataframe["enter_short"].tail(20).sum()
            EnhancedLogger.log_section("RECENT SIGNALS (20 periods)", "📊")
            EnhancedLogger.log_parameter("Long Entries", recent_long, "🟢")
            EnhancedLogger.log_parameter("Short Entries", recent_short, "🔴")
        else:
            EnhancedLogger.log_section("RECENT SIGNALS (20 periods)", "📊")
            EnhancedLogger.log_parameter("Long Entries", recent_long, "🟢")
        
        # Current market analysis
        current_fisher = dataframe["fisher"].iloc[-1]
        current_confidence = dataframe["ml_confidence"].iloc[-1]
        current_strength = dataframe["ml_signal_strength"].iloc[-1]
        current_regime = dataframe["market_regime"].iloc[-1]
        
        EnhancedLogger.log_section("CURRENT MARKET STATE", "📈")
        EnhancedLogger.log_parameter("Fisher Value", f"{current_fisher:.3f}", "🎣")
        EnhancedLogger.log_ml_status("ML Analysis", current_confidence, "🤖")
        EnhancedLogger.log_parameter("Signal Strength", f"{current_strength:.3f}", "⚡")
        
        regime_emoji = "🐂" if current_regime > 0 else "🐻" if current_regime < 0 else "⚖️"
        regime_text = "BULL" if current_regime > 0 else "BEAR" if current_regime < 0 else "NEUTRAL"
        EnhancedLogger.log_parameter("Market Regime", f"{regime_text} {regime_emoji}", "🏛️")
        
        EnhancedLogger.log_banner(f"ENTRY ANALYSIS COMPLETE - {pair}", "✅")
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Populate exit trend with enhanced ML integration and logging"""
        pair = metadata.get('pair', 'Unknown')
        
        EnhancedLogger.log_banner(f"EXIT SIGNAL ANALYSIS - {pair}", "🚪")
        
        # Get ML-adjusted parameters for this pair
        ml_params = self.get_ml_adjusted_params(dataframe, pair)
        
        # Use ML-adjusted exit thresholds
        fisher_long_exit = ml_params.get('fisher_long_exit', self.fisher_long_exit.value)
        fisher_short_exit = ml_params.get('fisher_short_exit', self.fisher_short_exit.value)
        
        EnhancedLogger.log_section("EXIT THRESHOLDS", "🎚️")
        exit_color_long = "🟢" if fisher_long_exit > 0 else "🔴" if fisher_long_exit < 0 else "🟡"
        exit_color_short = "🟢" if fisher_short_exit > 0 else "🔴" if fisher_short_exit < 0 else "🟡"
        
        EnhancedLogger.log_parameter(f"Long Exit {exit_color_long}", f"{fisher_long_exit:.3f}", "📤")
        if self.can_short:
            EnhancedLogger.log_parameter(f"Short Exit {exit_color_short}", f"{fisher_short_exit:.3f}", "📥")
        
        # ML-enhanced exit logic with confidence-based adjustments
        ml_confidence = dataframe["ml_confidence"]
        
        # Long exit with ML-optimized threshold
        long_exit_conditions = (
            (dataframe["fisher_smooth_long"].shift() > fisher_long_exit) & 
            (dataframe["fisher_smooth_long"] < fisher_long_exit) & 
            (dataframe["fisher_smooth_long"] > dataframe['fisher']) &
            (ml_confidence > 0.2)  # Only exit with reasonable confidence
        )
        
        dataframe.loc[long_exit_conditions, ["exit_long", "exit_tag"]] = [1, "exit_long_ml"]
        
        long_exits = long_exit_conditions.sum()
        EnhancedLogger.log_parameter("Long Exit Signals", long_exits, "📤")
        
        # Short exit with ML-optimized threshold (if enabled)
        if self.can_short:
            short_exit_conditions = (
                (dataframe["fisher_smooth_short"] > fisher_short_exit) &
                (ml_confidence > 0.2)
            )
            
            dataframe.loc[short_exit_conditions, ["exit_short", "exit_tag"]] = [1, "exit_short_ml"]
            
            short_exits = short_exit_conditions.sum()
            EnhancedLogger.log_parameter("Short Exit Signals", short_exits, "📥")
        
        # Log recent exit signals
        recent_long_exit = dataframe["exit_long"].tail(20).sum() if "exit_long" in dataframe.columns else 0
        
        EnhancedLogger.log_section("RECENT EXITS (20 periods)", "📊")
        EnhancedLogger.log_parameter("Long Exits", recent_long_exit, "📤")
        
        if self.can_short:
            recent_short_exit = dataframe["exit_short"].tail(20).sum() if "exit_short" in dataframe.columns else 0
            EnhancedLogger.log_parameter("Short Exits", recent_short_exit, "📥")
        
        # Current exit readiness analysis
        current_fisher_long = dataframe["fisher_smooth_long"].iloc[-1]
        current_confidence = dataframe["ml_confidence"].iloc[-1]
        
        EnhancedLogger.log_section("CURRENT EXIT ANALYSIS", "🔍")
        
        long_distance_to_exit = current_fisher_long - fisher_long_exit
        EnhancedLogger.log_parameter("Long Distance to Exit", f"{long_distance_to_exit:.3f}", "📏")
        
        if abs(long_distance_to_exit) < 0.1:
            EnhancedLogger.log_warning("Long position near exit threshold", "⚠️")
        elif long_distance_to_exit < 0:
            EnhancedLogger.log_success("Long exit conditions met", "✅")
        
        if self.can_short:
            current_fisher_short = dataframe["fisher_smooth_short"].iloc[-1]
            short_distance_to_exit = current_fisher_short - fisher_short_exit
            EnhancedLogger.log_parameter("Short Distance to Exit", f"{short_distance_to_exit:.3f}", "📏")
            
            if abs(short_distance_to_exit) < 0.1:
                EnhancedLogger.log_warning("Short position near exit threshold", "⚠️")
            elif short_distance_to_exit > 0:
                EnhancedLogger.log_success("Short exit conditions met", "✅")
        
        EnhancedLogger.log_ml_status("Exit Confidence", current_confidence, "🎯")
        
        EnhancedLogger.log_banner(f"EXIT ANALYSIS COMPLETE - {pair}", "✅")
        
        return dataframe

# PART 5 (FINAL) - Continuing from Part 4

    def custom_exit(self, pair: str, trade: "Trade", current_time: "datetime", current_rate: float, current_profit: float, **kwargs):
        """Enhanced custom exit with ML integration and detailed logging"""
        tag = super().custom_sell(pair, trade, current_time, current_rate, current_profit, **kwargs)
        if tag:
            return tag
        
        EnhancedLogger.log_section(f"CUSTOM EXIT ANALYSIS - {pair}", "🚪")
        
        entry_tag = "empty"
        if hasattr(trade, "entry_tag") and trade.entry_tag is not None:
            entry_tag = trade.entry_tag
        
        EnhancedLogger.log_parameter("Entry Tag", entry_tag, "🏷️")
        EnhancedLogger.log_parameter("Current Profit", f"{current_profit:.2%}", "💰")
        EnhancedLogger.log_parameter("Trade Duration", str(current_time - trade.open_date_utc), "⏱️")
        
        # ML-enhanced stop loss with dynamic adjustment
        ml_adjusted_stop = -0.35
        current_ml_confidence = 0.5
        market_regime = 0
        
        try:
            # Get current dataframe for ML analysis
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                current_ml_confidence = dataframe["ml_confidence"].iloc[-1]
                market_regime = dataframe["market_regime"].iloc[-1]
                
                EnhancedLogger.log_ml_status("Current ML State", current_ml_confidence, "🤖")
                
                regime_text = "BULL 🐂" if market_regime > 0 else "BEAR 🐻" if market_regime < 0 else "NEUTRAL ⚖️"
                EnhancedLogger.log_parameter("Market Regime", regime_text, "🏛️")
                
                # Adjust stop loss based on ML confidence and market regime
                if current_ml_confidence < 0.5:
                    ml_adjusted_stop = -0.25  # Tighter stop in low confidence
                    EnhancedLogger.log_warning("Tighter stop due to low confidence", "⚠️")
                elif market_regime < 0 and not trade.is_short:
                    ml_adjusted_stop = -0.3   # Tighter stop for longs in bear market
                    EnhancedLogger.log_warning("Tighter stop for long in bear market", "🐻")
                elif market_regime > 0 and trade.is_short:
                    ml_adjusted_stop = -0.3   # Tighter stop for shorts in bull market
                    EnhancedLogger.log_warning("Tighter stop for short in bull market", "🐂")
                
                EnhancedLogger.log_parameter("ML Adjusted Stop", f"{ml_adjusted_stop:.1%}", "🛡️")
                
        except Exception as e:
            EnhancedLogger.log_error(f"ML analysis failed: {e}", "❌")
        
        if current_profit <= ml_adjusted_stop:
            # Store trade performance for ML learning
            self.trade_performance_cache[trade.pair] = current_profit
            EnhancedLogger.log_warning(f"ML enhanced stop loss triggered", "🛑")
            EnhancedLogger.log_performance("Final Profit", current_profit, "💸")
            return f"ml_stop_loss ({entry_tag})"
        
        EnhancedLogger.log_success("No exit conditions met", "✅")
        return None
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, 
                          time_in_force: str, exit_reason: str, current_time: datetime, **kwargs) -> bool:
        """Enhanced trade exit confirmation with ML learning and detailed logging"""
        
        EnhancedLogger.log_banner(f"TRADE EXIT CONFIRMATION - {pair}", "🔍")
        
        filled_buys = trade.select_filled_orders(trade.entry_side)
        count_of_buys = len(filled_buys)
        
        # Calculate profit for ML learning
        current_profit = trade.calc_profit_ratio(rate)
        
        EnhancedLogger.log_section("EXIT DETAILS", "📊")
        EnhancedLogger.log_parameter("Exit Reason", exit_reason, "📝")
        EnhancedLogger.log_parameter("Order Type", order_type, "📋")
        EnhancedLogger.log_parameter("Exit Amount", f"{amount:.8f}", "💹")
        EnhancedLogger.log_parameter("Exit Rate", f"{rate:.8f}", "💱")
        EnhancedLogger.log_parameter("Buy Orders", count_of_buys, "🔢")
        EnhancedLogger.log_performance("Exit Profit", current_profit, "💰")
        
        # ML learning: store trade performance
        if exit_reason in ["roi", "stop_loss", "ml_stop_loss"]:
            self.trade_performance_cache[pair] = current_profit
            
            # Enhanced logging for ML learning
            performance_quality = "🟢 GOOD" if current_profit > 0.01 else "🟡 BREAK-EVEN" if current_profit > -0.01 else "🔴 LOSS"
            EnhancedLogger.log_parameter(f"Performance {performance_quality}", f"{current_profit:.2%}", "📈")
            EnhancedLogger.log_success("Performance stored for ML learning", "🧠")
        
        # Enhanced exit conditions with ML
        if current_profit < 0.005:
            EnhancedLogger.log_warning("Profit too low, rejecting exit", "⚠️")
            return False
        
        if (count_of_buys == 1) & (exit_reason == "roi"):
            EnhancedLogger.log_warning("Single buy + ROI exit, rejecting", "⚠️")
            return False
        
        # Clean up stake tracking
        if trade.amount == amount and pair in self.cust_proposed_initial_stakes:
            del self.cust_proposed_initial_stakes[pair]
            EnhancedLogger.log_success("Stake tracking cleaned up", "🧹")
        
        EnhancedLogger.log_success("Trade exit confirmed", "✅")
        return True
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, 
                           proposed_stake: float, min_stake: float, max_stake: float, **kwargs) -> float:
        """ML-enhanced stake sizing with detailed logging"""
        
        EnhancedLogger.log_section(f"STAKE CALCULATION - {pair}", "💰")
        
        try:
            # Get market analysis for stake adjustment
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            ml_adjustment = 1.0
            
            EnhancedLogger.log_parameter("Proposed Stake", f"{proposed_stake:.4f}", "💵")
            EnhancedLogger.log_parameter("Min Stake", f"{min_stake:.4f}", "🔻")
            EnhancedLogger.log_parameter("Max Stake", f"{max_stake:.4f}", "🔺")
            
            if not dataframe.empty:
                ml_confidence = dataframe["ml_confidence"].iloc[-1]
                signal_strength = dataframe["ml_signal_strength"].iloc[-1]
                
                EnhancedLogger.log_ml_status("ML Confidence", ml_confidence, "🎯")
                EnhancedLogger.log_parameter("Signal Strength", f"{signal_strength:.3f}", "⚡")
                
                # Adjust stake based on ML confidence
                confidence_multiplier = 0.5 + (ml_confidence * 0.5)  # 0.5 to 1.0
                signal_multiplier = 0.7 + (signal_strength * 0.3)    # 0.7 to 1.0
                
                ml_adjustment = confidence_multiplier * signal_multiplier
                
                EnhancedLogger.log_parameter("Confidence Multiplier", f"{confidence_multiplier:.3f}", "🎯")
                EnhancedLogger.log_parameter("Signal Multiplier", f"{signal_multiplier:.3f}", "⚡")
                EnhancedLogger.log_parameter("Combined ML Adjustment", f"{ml_adjustment:.3f}x", "🤖")
                
                if ml_adjustment > 1.0:
                    EnhancedLogger.log_success("Increasing stake due to strong ML signals", "📈")
                elif ml_adjustment < 0.8:
                    EnhancedLogger.log_warning("Reducing stake due to weak ML signals", "📉")
                else:
                    EnhancedLogger.log_success("Standard stake with moderate ML adjustment", "⚖️")
            else:
                EnhancedLogger.log_warning("No dataframe available, using default adjustment", "⚠️")
            
            custom_stake = (proposed_stake / self.max_so_multiplier * self.overbuy_factor) * ml_adjustment
            custom_stake = max(min_stake, min(custom_stake, max_stake))  # Ensure within bounds
            
        except Exception as e:
            EnhancedLogger.log_error(f"Stake calculation error: {e}", "💥")
            custom_stake = proposed_stake / self.max_so_multiplier * self.overbuy_factor
        
        EnhancedLogger.log_parameter("Final Custom Stake", f"{custom_stake:.4f}", "💎")
        
        stake_change_pct = ((custom_stake - proposed_stake) / proposed_stake) * 100
        change_emoji = "📈" if stake_change_pct > 0 else "📉" if stake_change_pct < 0 else "➡️"
        EnhancedLogger.log_parameter(f"Stake Change {change_emoji}", f"{stake_change_pct:+.1f}%", "📊")
        
        self.cust_proposed_initial_stakes[pair] = custom_stake
        return custom_stake
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float, 
                             current_profit: float, min_stake: float, max_stake: float, **kwargs) -> Optional[float]:
        """Enhanced DCA with ML risk assessment and detailed logging"""
        
        if current_profit > self.initial_safety_order_trigger:
            return None
        
        EnhancedLogger.log_section(f"DCA ANALYSIS - {trade.pair}", "🔄")
        
        filled_buys = trade.select_filled_orders(trade.entry_side)
        count_of_buys = len(filled_buys)
        
        EnhancedLogger.log_parameter("Current Profit", f"{current_profit:.2%}", "📊")
        EnhancedLogger.log_parameter("Existing Buy Orders", count_of_buys, "🔢")
        EnhancedLogger.log_parameter("Max SO Multiplier", self.max_so_multiplier_orig, "🔢")
        
        if 1 <= count_of_buys <= self.max_so_multiplier_orig:
            # ML-enhanced safety order trigger
            ml_trigger_adjustment = 1.0
            
            try:
                dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                if not dataframe.empty:
                    ml_confidence = dataframe["ml_confidence"].iloc[-1]
                    market_regime = dataframe["market_regime"].iloc[-1]
                    
                    EnhancedLogger.log_ml_status("ML Confidence", ml_confidence, "🤖")
                    
                    regime_text = "BULL 🐂" if market_regime > 0 else "BEAR 🐻" if market_regime < 0 else "NEUTRAL ⚖️"
                    EnhancedLogger.log_parameter("Market Regime", regime_text, "🏛️")
                    
                    # Adjust safety order trigger based on ML analysis
                    if ml_confidence < 0.5:
                        ml_trigger_adjustment = 1.5  # More conservative in low confidence
                        EnhancedLogger.log_warning("Conservative DCA due to low confidence", "⚠️")
                    elif market_regime < 0 and not trade.is_short:
                        ml_trigger_adjustment = 1.3  # More conservative for longs in bear market
                        EnhancedLogger.log_warning("Conservative DCA for long in bear market", "🐻")
                    else:
                        EnhancedLogger.log_success("Standard DCA trigger", "✅")
                    
                    EnhancedLogger.log_parameter("ML Trigger Adjustment", f"{ml_trigger_adjustment:.1f}x", "🎯")
                        
            except Exception as e:
                EnhancedLogger.log_error(f"ML analysis failed: {e}", "❌")
            
            safety_order_trigger = abs(self.initial_safety_order_trigger) * count_of_buys * ml_trigger_adjustment
            
            if self.safety_order_step_scale > 1:
                safety_order_trigger = abs(self.initial_safety_order_trigger) * ml_trigger_adjustment + (
                    abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * 
                    (math.pow(self.safety_order_step_scale, (count_of_buys - 1)) - 1) / 
                    (self.safety_order_step_scale - 1)
                )
            elif self.safety_order_step_scale < 1:
                safety_order_trigger = abs(self.initial_safety_order_trigger) * ml_trigger_adjustment + (
                    abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * 
                    (1 - math.pow(self.safety_order_step_scale, (count_of_buys - 1))) / 
                    (1 - self.safety_order_step_scale)
                )
            
            EnhancedLogger.log_parameter("Safety Order Trigger", f"{safety_order_trigger:.2%}", "🎯")
            
            if current_profit <= (-1 * abs(safety_order_trigger)):
                EnhancedLogger.log_success("DCA trigger activated!", "🚀")
                
                try:
                    actual_initial_stake = filled_buys[0].cost
                    stake_amount = actual_initial_stake
                    already_bought = sum(filled_buy.cost for filled_buy in filled_buys)
                    
                    EnhancedLogger.log_parameter("Initial Stake", f"{actual_initial_stake:.4f}", "💰")
                    EnhancedLogger.log_parameter("Already Invested", f"{already_bought:.4f}", "💸")
                    
                    if trade.pair in self.cust_proposed_initial_stakes:
                        if self.cust_proposed_initial_stakes[trade.pair] > 0:
                            proposed_initial_stake = self.cust_proposed_initial_stakes[trade.pair]
                            current_actual_stake = already_bought * math.pow(self.safety_order_volume_scale, (count_of_buys - 1))
                            current_stake_preposition = proposed_initial_stake * math.pow(self.safety_order_volume_scale, (count_of_buys - 1))
                            current_stake_preposition_compensation = (
                                current_stake_preposition + abs(current_stake_preposition - current_actual_stake)
                            )
                            total_so_stake = lerp(current_actual_stake, current_stake_preposition_compensation, 
                                                self.partial_fill_compensation_scale)
                            stake_amount = total_so_stake
                            
                            EnhancedLogger.log_parameter("Compensated Stake", f"{stake_amount:.4f}", "🎯")
                        else:
                            stake_amount = stake_amount * math.pow(self.safety_order_volume_scale, (count_of_buys - 1))
                            EnhancedLogger.log_parameter("Scaled Stake", f"{stake_amount:.4f}", "📈")
                    else:
                        stake_amount = stake_amount * math.pow(self.safety_order_volume_scale, (count_of_buys - 1))
                        EnhancedLogger.log_parameter("Default Scaled Stake", f"{stake_amount:.4f}", "📊")
                    
                    EnhancedLogger.log_success(f"DCA order #{count_of_buys + 1} approved", "✅")
                    return stake_amount
                    
                except Exception as e:
                    EnhancedLogger.log_error(f"DCA calculation failed: {e}", "💥")
                    return None
            else:
                distance_to_trigger = abs(current_profit) - abs(safety_order_trigger)
                EnhancedLogger.log_parameter("Distance to DCA", f"{distance_to_trigger:.2%}", "📏")
                EnhancedLogger.log_warning("DCA trigger not reached yet", "⏳")
        else:
            if count_of_buys > self.max_so_multiplier_orig:
                EnhancedLogger.log_warning("Maximum DCA orders reached", "🛑")
            else:
                EnhancedLogger.log_warning("No existing orders for DCA", "❌")
        
        return None
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, 
                       current_profit: float, **kwargs) -> float:
        """ML-enhanced custom stop loss and take profit with detailed logging"""
        
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            trade_candle = dataframe.loc[dataframe['date'] == trade_date]
            
            if not trade_candle.empty:
                trade_candle = trade_candle.squeeze()
                
                EnhancedLogger.log_section(f"CUSTOM STOPLOSS - {pair}", "🛡️")
                
                # Get ML indicators for dynamic adjustment
                try:
                    current_ml_confidence = dataframe["ml_confidence"].iloc[-1]
                    market_regime = dataframe["market_regime"].iloc[-1]
                    
                    # Adjust multipliers based on ML analysis
                    sl_multiplier = self.ATR_SL_long_Multip.value if not trade.is_short else self.ATR_SL_short_Multip.value
                    tp_multiplier = self.rr_long.value if not trade.is_short else self.rr_short.value
                    
                    EnhancedLogger.log_parameter("Base SL Multiplier", f"{sl_multiplier:.1f}x", "🛡️")
                    EnhancedLogger.log_parameter("Base TP Multiplier", f"{tp_multiplier:.1f}x", "🎯")
                    EnhancedLogger.log_ml_status("ML Confidence", current_ml_confidence, "🤖")
                    
                    # Dynamic adjustment based on ML confidence
                    if current_ml_confidence < 0.5:
                        sl_multiplier *= 0.8  # Tighter stop loss
                        tp_multiplier *= 0.9  # Closer take profit
                        EnhancedLogger.log_warning("Tighter SL/TP due to low confidence", "⚠️")
                    elif current_ml_confidence > 0.8:
                        sl_multiplier *= 1.2  # Wider stop loss
                        tp_multiplier *= 1.1  # Further take profit
                        EnhancedLogger.log_success("Wider SL/TP due to high confidence", "✨")
                    
                    # Market regime adjustment
                    if not trade.is_short and market_regime < 0:  # Long in bear market
                        sl_multiplier *= 0.9
                        EnhancedLogger.log_warning("Tighter SL for long in bear market", "🐻")
                    elif trade.is_short and market_regime > 0:    # Short in bull market
                        sl_multiplier *= 0.9
                        EnhancedLogger.log_warning("Tighter SL for short in bull market", "🐂")
                    
                    EnhancedLogger.log_parameter("Adjusted SL Multiplier", f"{sl_multiplier:.2f}x", "🎯")
                    EnhancedLogger.log_parameter("Adjusted TP Multiplier", f"{tp_multiplier:.2f}x", "🎯")
                        
                except Exception as e:
                    EnhancedLogger.log_error(f"ML adjustment failed: {e}", "❌")
                    sl_multiplier = self.ATR_SL_long_Multip.value if not trade.is_short else self.ATR_SL_short_Multip.value
                    tp_multiplier = self.rr_long.value if not trade.is_short else self.rr_short.value
                
                # Stop Loss Logic
                atr_value = trade_candle['atr']
                sl_distance = atr_value * sl_multiplier
                
                if not trade.is_short:
                    sl_price = trade.open_rate - sl_distance
                    sl_condition = current_rate < sl_price
                    side_text = "LONG"
                else:
                    sl_price = trade.open_rate + sl_distance
                    sl_condition = current_rate > sl_price
                    side_text = "SHORT"
                
                EnhancedLogger.log_parameter(f"{side_text} SL Price", f"{sl_price:.6f}", "🛑")
                EnhancedLogger.log_parameter("Current Rate", f"{current_rate:.6f}", "💱")
                EnhancedLogger.log_parameter("SL Distance", f"{sl_distance:.6f}", "📏")
                
                if sl_condition:
                    self.trade_performance_cache[pair] = current_profit  # Store for ML learning
                    EnhancedLogger.log_warning("STOP LOSS TRIGGERED!", "🛑")
                    EnhancedLogger.log_performance("Final Loss", current_profit, "💸")
                    return -0.01
                
                # Take Profit Logic
                dist = trade_candle['atr'] * self.ATR_Multip.value
                tp_distance = dist * tp_multiplier
                
                if not trade.is_short:
                    tp_price = trade.open_rate + tp_distance
                    tp_condition = current_rate > tp_price
                else:
                    tp_price = trade.open_rate - tp_distance
                    tp_condition = current_rate < tp_price
                
                EnhancedLogger.log_parameter(f"{side_text} TP Price", f"{tp_price:.6f}", "🎯")
                EnhancedLogger.log_parameter("TP Distance", f"{tp_distance:.6f}", "📏")
                
                if tp_condition:
                    self.trade_performance_cache[pair] = current_profit  # Store for ML learning
                    EnhancedLogger.log_success("TAKE PROFIT TRIGGERED!", "🎯")
                    EnhancedLogger.log_performance("Final Profit", current_profit, "💰")
                    return -0.0001
                
                # Log current distances
                if not trade.is_short:
                    sl_distance_current = current_rate - sl_price
                    tp_distance_current = tp_price - current_rate
                else:
                    sl_distance_current = sl_price - current_rate
                    tp_distance_current = current_rate - tp_price
                
                EnhancedLogger.log_parameter("Distance to SL", f"{sl_distance_current:.6f}", "📏")
                EnhancedLogger.log_parameter("Distance to TP", f"{tp_distance_current:.6f}", "📏")
        
        except Exception as e:
            EnhancedLogger.log_error(f"Custom stoploss calculation failed: {e}", "💥")
        
        return None
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, 
                max_leverage: float, side: str, **kwargs) -> float:
        """ML-enhanced leverage management with detailed logging"""
        
        EnhancedLogger.log_section(f"LEVERAGE CALCULATION - {pair}", "⚖️")
        
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            base_leverage = self.set_leverage if self.can_short else 1
            
            EnhancedLogger.log_parameter("Proposed Leverage", f"{proposed_leverage:.1f}x", "📊")
            EnhancedLogger.log_parameter("Max Leverage", f"{max_leverage:.1f}x", "🔺")
            EnhancedLogger.log_parameter("Base Leverage", f"{base_leverage:.1f}x", "⚙️")
            EnhancedLogger.log_parameter("Trade Side", side.upper(), "↔️")
            
            if not dataframe.empty:
                ml_confidence = dataframe["ml_confidence"].iloc[-1]
                market_volatility = dataframe["atr"].iloc[-1] / dataframe["close"].iloc[-1]
                
                EnhancedLogger.log_ml_status("ML Confidence", ml_confidence, "🤖")
                EnhancedLogger.log_parameter("Market Volatility", f"{market_volatility:.1%}", "🌊")
                
                # Reduce leverage in high volatility or low confidence conditions
                if ml_confidence < 0.6 or market_volatility > 0.05:
                    adjusted_leverage = base_leverage * 0.8
                    reason = "low confidence" if ml_confidence < 0.6 else "high volatility"
                    EnhancedLogger.log_warning(f"Reduced leverage due to {reason}", "⚠️")
                elif ml_confidence > 0.8 and market_volatility < 0.02:
                    adjusted_leverage = min(base_leverage * 1.1, max_leverage)
                    EnhancedLogger.log_success("Increased leverage due to favorable conditions", "📈")
                else:
                    adjusted_leverage = base_leverage
                    EnhancedLogger.log_success("Standard leverage applied", "✅")
                
                final_leverage = max(1, min(adjusted_leverage, max_leverage))
                
                leverage_change = final_leverage - base_leverage
                change_emoji = "📈" if leverage_change > 0 else "📉" if leverage_change < 0 else "➡️"
                
                EnhancedLogger.log_parameter("Final Leverage", f"{final_leverage:.1f}x", "🎯")
                EnhancedLogger.log_parameter(f"Leverage Change {change_emoji}", f"{leverage_change:+.1f}x", "📊")
                
                return final_leverage
                
        except Exception as e:
            EnhancedLogger.log_error(f"Leverage calculation failed: {e}", "💥")
        
        default_leverage = self.set_leverage if self.can_short else 1
        EnhancedLogger.log_parameter("Default Leverage Applied", f"{default_leverage:.1f}x", "🔄")
        return default_leverage

