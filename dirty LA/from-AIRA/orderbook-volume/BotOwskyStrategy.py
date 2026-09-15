"""
# BOT-OWSKY Strategy - Advanced Multi-Phase ML Trading Strategy
#
# VERSION: 1.0.0 - Initial Implementation
#
# ARCHITECTURE:
# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Mathematical Tools & Pattern Loading
#   - Local Max/Min + Adaptive VWAP (fair value determination)
#   - Fibonacci Retracements (support/resistance levels)
#   - ATR for Volatility (trend and entry/exit zones)
#   - FFT for Cycle Detection (filter false signals)
#   - Kelly Criterion for Position Sizing (optimal position size)
#   - Markov Chains for State Transition Probabilities
#   - Candlestick Patterns (Encyclopedia of Candlestick Charts - Bulkowski)
#   - Clustering (pattern-strategy matching)
#   - Gaussian Processes (probabilistic predictions with uncertainty)
#   - Bayesian Models (probabilistic market sentiment)
#
# Phase 2: LSTM Neural Network
#   - Multivariate LSTM with dual input channels:
#     1. Mathematical/Statistical indicators
#     2. Clustered pattern recognition
#   - Temporal sequence capture
#   - Pattern recognition and risk management
#
# Phase 3: Decision System
#   - Random Forest scoring/ranking system
#   - Final buy/sell decision making
#   - Market regime detection (bullish/bearish/sideways)
#
# ═══════════════════════════════════════════════════════════════════════════════
#
# Author: Based on BOT-OWSKY concept
# Created: 2025-10-05
# Freqtrade Version: 2025.9+
# Python Version: 3.11+
"""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import talib.abstract as ta
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
import logging
import pickle
import json
from pathlib import Path
from typing import Any
from datetime import datetime, timedelta

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    CategoricalParameter,
    BooleanParameter,
    informative,
    merge_informative_pair
)
from freqtrade.persistence import Trade

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class BotOwskyStrategy(IStrategy):
    """
    BOT-OWSKY: Advanced Multi-Phase Machine Learning Strategy
    
    This strategy implements a three-phase approach:
    1. Mathematical analysis and pattern recognition
    2. LSTM-based temporal modeling (placeholder for future implementation)
    3. Random Forest decision system
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY METADATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    INTERFACE_VERSION = 3
    
    # Strategy configuration
    timeframe = '1h'
    startup_candle_count: int = 200
    
    # Trading mode
    can_short = True
    
    # ROI - Take profit configuration
    minimal_roi = {
        "0": 0.15,      # 15% at any time
        "30": 0.10,     # 10% after 30 minutes
        "60": 0.05,     # 5% after 1 hour
        "120": 0.025    # 2.5% after 2 hours
    }
    
    # Stoploss
    stoploss = -0.08  # 8% stoploss
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True
    
    # Order types
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }
    
    # Order time in force
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }
    
    # Position adjustment
    position_adjustment_enable = True
    max_entry_position_adjustment = 2
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HYPEROPTABLE PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Phase 1: Mathematical Indicators
    atr_period = IntParameter(10, 30, default=14, space='buy', optimize=True)
    atr_multiplier = DecimalParameter(1.0, 3.0, default=2.0, space='buy', optimize=True)
    
    # Fibonacci levels
    use_fibonacci = BooleanParameter(default=True, space='buy', optimize=True)
    fib_lookback = IntParameter(20, 100, default=50, space='buy', optimize=True)
    
    # VWAP
    vwap_period = IntParameter(10, 50, default=20, space='buy', optimize=True)
    vwap_deviation_threshold = DecimalParameter(0.01, 0.05, default=0.02, space='buy', optimize=True)
    
    # FFT Cycle Detection
    use_fft_filter = BooleanParameter(default=True, space='buy', optimize=True)
    fft_threshold = DecimalParameter(0.3, 0.8, default=0.5, space='buy', optimize=True)
    
    # Kelly Criterion
    use_kelly = BooleanParameter(default=True, space='buy', optimize=True)
    kelly_fraction = DecimalParameter(0.1, 0.5, default=0.25, space='buy', optimize=True)
    
    # Machine Learning
    ml_confidence_threshold = DecimalParameter(0.5, 0.9, default=0.7, space='buy', optimize=True)
    use_gaussian_process = BooleanParameter(default=True, space='buy', optimize=True)
    use_clustering = BooleanParameter(default=True, space='buy', optimize=True)
    
    # Random Forest Decision
    rf_signal_threshold = DecimalParameter(0.5, 0.9, default=0.65, space='buy', optimize=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        # Model storage paths
        self.model_dir = Path(config.get('user_data_dir', 'user_data')) / 'models' / 'bot_owsky'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.random_forest_long = None
        self.random_forest_short = None
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.gp_model = None
        
        # Pattern library (simplified - expandable with Bulkowski patterns)
        self.candlestick_patterns = [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
            'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS',
            'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBELTHOLD',
            'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL',
            'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
            'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING',
            'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE',
            'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN',
            'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE',
            'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON',
            'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER',
            'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM',
            'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU',
            'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR',
            'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING',
            'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES',
            'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP',
            'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI',
            'CDLTASUKIGAP', 'CDLTHRUSTING', 'CDLTRISTAR',
            'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'
        ]
        
        # Markov chain transition matrix (will be learned from data)
        self.markov_states = ['bullish', 'bearish', 'sideways']
        self.transition_matrix = None
        
        # Model loaded flags
        self.models_loaded = False
        
        logger.info("🚀 BOT-OWSKY Strategy initialized")
    
    def load_models(self):
        """Load pre-trained models if available"""
        try:
            rf_long_path = self.model_dir / 'random_forest_long.pkl'
            rf_short_path = self.model_dir / 'random_forest_short.pkl'
            scaler_path = self.model_dir / 'scaler.pkl'
            cluster_path = self.model_dir / 'cluster_model.pkl'
            
            if rf_long_path.exists():
                with open(rf_long_path, 'rb') as f:
                    self.random_forest_long = pickle.load(f)
                logger.info("✅ Loaded Random Forest LONG model")
            
            if rf_short_path.exists():
                with open(rf_short_path, 'rb') as f:
                    self.random_forest_short = pickle.load(f)
                logger.info("✅ Loaded Random Forest SHORT model")
            
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("✅ Loaded scaler")
            
            if cluster_path.exists():
                with open(cluster_path, 'rb') as f:
                    self.cluster_model = pickle.load(f)
                logger.info("✅ Loaded clustering model")
            
            self.models_loaded = True
            
        except Exception as e:
            logger.warning(f"⚠️  Could not load models: {e}")
            self.models_loaded = False
    
    def save_models(self):
        """Save trained models"""
        try:
            if self.random_forest_long:
                with open(self.model_dir / 'random_forest_long.pkl', 'wb') as f:
                    pickle.dump(self.random_forest_long, f)
            
            if self.random_forest_short:
                with open(self.model_dir / 'random_forest_short.pkl', 'wb') as f:
                    pickle.dump(self.random_forest_short, f)
            
            if self.scaler:
                with open(self.model_dir / 'scaler.pkl', 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            if self.cluster_model:
                with open(self.model_dir / 'cluster_model.pkl', 'wb') as f:
                    pickle.dump(self.cluster_model, f)
            
            logger.info("💾 Models saved successfully")
        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1: MATHEMATICAL TOOLS & INDICATORS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_local_extrema(self, dataframe: DataFrame, column: str = 'close', order: int = 5) -> tuple[Series, Series]:
        """
        Calculate local maxima and minima using scipy
        """
        data = dataframe[column].values
        
        # Find local maxima
        max_indices = signal.argrelextrema(data, np.greater, order=order)[0]
        local_max = pd.Series(np.nan, index=dataframe.index)
        if len(max_indices) > 0:
            local_max.iloc[max_indices] = data[max_indices]
        
        # Find local minima
        min_indices = signal.argrelextrema(data, np.less, order=order)[0]
        local_min = pd.Series(np.nan, index=dataframe.index)
        if len(min_indices) > 0:
            local_min.iloc[min_indices] = data[min_indices]
        
        # Forward fill for easier reference
        local_max_filled = local_max.ffill()
        local_min_filled = local_min.ffill()
        
        return local_max_filled, local_min_filled
    
    def calculate_adaptive_vwap(self, dataframe: DataFrame, period: int = 20) -> Series:
        """
        Calculate adaptive VWAP (Volume Weighted Average Price)
        """
        typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        vwap = (typical_price * dataframe['volume']).rolling(window=period).sum() / dataframe['volume'].rolling(window=period).sum()
        return vwap
    
    def calculate_fibonacci_levels(self, dataframe: DataFrame, lookback: int = 50) -> dict[str, Series]:
        """
        Calculate Fibonacci retracement levels
        """
        rolling_max = dataframe['high'].rolling(window=lookback).max()
        rolling_min = dataframe['low'].rolling(window=lookback).min()
        diff = rolling_max - rolling_min
        
        fib_levels = {
            'fib_0': rolling_min,
            'fib_236': rolling_min + 0.236 * diff,
            'fib_382': rolling_min + 0.382 * diff,
            'fib_500': rolling_min + 0.500 * diff,
            'fib_618': rolling_min + 0.618 * diff,
            'fib_786': rolling_min + 0.786 * diff,
            'fib_100': rolling_max,
        }
        
        return fib_levels
    
    def calculate_fft_cycles(self, dataframe: DataFrame, column: str = 'close', threshold: float = 0.5) -> Series:
        """
        Fast Fourier Transform for cycle detection
        Filters out high-frequency noise to identify underlying trends
        """
        if len(dataframe) < 50:
            return pd.Series(0, index=dataframe.index)
        
        # Get price data
        prices = dataframe[column].values
        
        # Apply FFT
        fft_values = fft(prices)
        frequencies = fftfreq(len(prices))
        
        # Filter out high frequencies (noise)
        fft_filtered = fft_values.copy()
        fft_filtered[np.abs(frequencies) > threshold] = 0
        
        # Inverse FFT to get filtered signal
        filtered_signal = np.fft.ifft(fft_filtered).real
        
        # Calculate cycle strength (difference between original and filtered)
        cycle_strength = np.abs(prices - filtered_signal)
        
        return pd.Series(cycle_strength, index=dataframe.index)
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing
        
        Formula: f* = (p * b - q) / b
        where:
        - p = win_rate (probability of winning)
        - q = 1 - p (probability of losing)
        - b = avg_win / avg_loss (win/loss ratio)
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        b = avg_win / avg_loss
        q = 1 - win_rate
        
        kelly = (win_rate * b - q) / b
        
        # Return conservative Kelly (usually use 1/4 to 1/2 Kelly)
        return max(0, min(kelly * self.kelly_fraction.value, 1.0))
    
    def calculate_markov_probabilities(self, dataframe: DataFrame) -> dict[str, float]:
        """
        Calculate Markov chain state transition probabilities
        States: bullish, bearish, sideways
        """
        # Simplified state detection based on trend
        returns = dataframe['close'].pct_change()
        
        # Define states
        bullish_threshold = 0.005  # 0.5% per candle
        bearish_threshold = -0.005
        
        states = pd.Series('sideways', index=dataframe.index)
        states[returns > bullish_threshold] = 'bullish'
        states[returns < bearish_threshold] = 'bearish'
        
        # Count transitions
        if len(states) < 2:
            return {'bullish': 0.33, 'bearish': 0.33, 'sideways': 0.34}
        
        current_state = states.iloc[-1]
        
        # Look at last N candles for transition probabilities
        recent_states = states.iloc[-50:] if len(states) >= 50 else states
        
        # Count state occurrences
        state_counts = recent_states.value_counts()
        total = len(recent_states)
        
        probabilities = {
            'bullish': state_counts.get('bullish', 0) / total,
            'bearish': state_counts.get('bearish', 0) / total,
            'sideways': state_counts.get('sideways', 0) / total
        }
        
        return probabilities
    
    def detect_candlestick_patterns(self, dataframe: DataFrame) -> DataFrame:
        """
        Detect candlestick patterns using TA-Lib
        Based on Encyclopedia of Candlestick Charts (Thomas N. Bulkowski)
        """
        df = dataframe.copy()
        
        # Detect all patterns
        for pattern in self.candlestick_patterns:
            try:
                df[f'pattern_{pattern}'] = getattr(ta, pattern)(df)
            except Exception:
                df[f'pattern_{pattern}'] = 0
        
        # Create aggregated pattern signals
        pattern_cols = [col for col in df.columns if col.startswith('pattern_')]
        df['bullish_pattern_count'] = df[pattern_cols].apply(lambda x: (x > 0).sum(), axis=1)
        df['bearish_pattern_count'] = df[pattern_cols].apply(lambda x: (x < 0).sum(), axis=1)
        df['pattern_strength'] = df['bullish_pattern_count'] - df['bearish_pattern_count']
        
        return df
    
    def calculate_gaussian_process_prediction(self, dataframe: DataFrame, feature_cols: list[str]) -> tuple[Series, Series]:
        """
        Gaussian Process for probabilistic predictions with uncertainty
        Returns: (mean_prediction, uncertainty)
        """
        if len(dataframe) < 50:
            return pd.Series(0, index=dataframe.index), pd.Series(1, index=dataframe.index)
        
        try:
            # Prepare training data (use last 100 candles)
            train_size = min(100, len(dataframe) - 1)
            X_train = dataframe[feature_cols].iloc[-train_size:-1].values
            y_train = dataframe['close'].pct_change().iloc[-train_size:].values[1:]
            
            # Handle NaN values
            X_train = np.nan_to_num(X_train, nan=0.0)
            y_train = np.nan_to_num(y_train, nan=0.0)
            
            # Define kernel
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            
            # Create and fit GP model
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-6)
            gp.fit(X_train, y_train)
            
            # Predict on current data
            X_current = dataframe[feature_cols].iloc[-1:].values
            X_current = np.nan_to_num(X_current, nan=0.0)
            
            mean_pred, std_pred = gp.predict(X_current, return_std=True)
            
            # Create series for full dataframe
            mean_series = pd.Series(0, index=dataframe.index)
            mean_series.iloc[-1] = mean_pred[0]
            
            std_series = pd.Series(1, index=dataframe.index)
            std_series.iloc[-1] = std_pred[0]
            
            return mean_series, std_series
            
        except Exception as e:
            logger.warning(f"GP prediction failed: {e}")
            return pd.Series(0, index=dataframe.index), pd.Series(1, index=dataframe.index)
    
    def calculate_bayesian_sentiment(self, dataframe: DataFrame) -> Series:
        """
        Bayesian model for market sentiment combining multiple indicators
        Returns posterior probability of upward movement
        """
        # Prior: neutral 50/50
        prior_up = 0.5
        
        # Likelihood from RSI
        rsi = dataframe.get('rsi', pd.Series(50, index=dataframe.index))
        p_rsi_up = (100 - rsi) / 100  # Probability of up given oversold
        
        # Likelihood from volume
        volume_ratio = dataframe['volume'] / dataframe['volume'].rolling(20).mean()
        p_volume_up = (volume_ratio - 1).clip(0, 1)  # Higher volume suggests momentum
        
        # Likelihood from trend (EMA)
        if 'ema_20' in dataframe.columns and 'ema_50' in dataframe.columns:
            trend_up = (dataframe['ema_20'] > dataframe['ema_50']).astype(float)
            p_trend_up = trend_up * 0.7 + 0.15  # 70% when uptrend, 15% when downtrend
        else:
            p_trend_up = pd.Series(0.5, index=dataframe.index)
        
        # Bayesian update (simplified)
        # P(up|evidence) = P(evidence|up) * P(up) / P(evidence)
        # Using log odds for numerical stability
        
        log_odds = np.log(prior_up / (1 - prior_up))
        log_odds += np.log(p_rsi_up / (1 - p_rsi_up + 1e-10))
        log_odds += np.log(p_trend_up / (1 - p_trend_up + 1e-10))
        
        # Convert back to probability
        posterior_up = 1 / (1 + np.exp(-log_odds))
        
        return pd.Series(posterior_up, index=dataframe.index)
    
    def perform_clustering(self, dataframe: DataFrame, feature_cols: list[str], n_clusters: int = 5) -> Series:
        """
        Cluster market conditions to identify similar patterns
        """
        if len(dataframe) < n_clusters * 2:
            return pd.Series(0, index=dataframe.index)
        
        try:
            # Prepare data
            X = dataframe[feature_cols].values
            X = np.nan_to_num(X, nan=0.0)
            
            # Fit or use existing cluster model
            if self.cluster_model is None:
                self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                self.cluster_model.fit(X)
            
            # Predict clusters
            clusters = self.cluster_model.predict(X)
            
            return pd.Series(clusters, index=dataframe.index)
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return pd.Series(0, index=dataframe.index)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: LSTM NEURAL NETWORK (Placeholder for future implementation)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def lstm_prediction(self, dataframe: DataFrame, features: DataFrame) -> tuple[Series, Series]:
        """
        LSTM-based temporal sequence prediction
        
        TODO: Implement LSTM with:
        - Dual input channels (mathematical indicators + clustered patterns)
        - Sequence length: 50-100 candles
        - Output: Buy/sell probability + confidence
        
        For now, returns placeholder values
        """
        # Placeholder: return neutral predictions
        lstm_signal = pd.Series(0.5, index=dataframe.index)
        lstm_confidence = pd.Series(0.5, index=dataframe.index)
        
        return lstm_signal, lstm_confidence
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3: RANDOM FOREST DECISION SYSTEM
    # ═══════════════════════════════════════════════════════════════════════════
    
    def train_random_forest(self, dataframe: DataFrame, feature_cols: list[str]):
        """
        Train Random Forest models for buy/sell decisions
        """
        if len(dataframe) < 200:
            logger.warning("Not enough data to train Random Forest")
            return
        
        try:
            # Prepare features
            X = dataframe[feature_cols].values
            X = np.nan_to_num(X, nan=0.0)
            
            # Create labels (future returns)
            future_returns = dataframe['close'].pct_change(3).shift(-3)  # 3-candle forward return
            
            # Binary classification: up vs down
            y_long = (future_returns > 0.01).astype(int).values  # 1% threshold for long
            y_short = (future_returns < -0.01).astype(int).values  # -1% threshold for short
            
            # Remove NaN values
            valid_mask = ~np.isnan(future_returns.values)
            X = X[valid_mask]
            y_long = y_long[valid_mask]
            y_short = y_short[valid_mask]
            
            if len(X) < 50:
                return
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train LONG model
            self.random_forest_long = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            self.random_forest_long.fit(X_scaled, y_long)
            
            # Train SHORT model
            self.random_forest_short = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            self.random_forest_short.fit(X_scaled, y_short)
            
            logger.info("✅ Random Forest models trained successfully")
            
        except Exception as e:
            logger.error(f"❌ Random Forest training failed: {e}")
    
    def random_forest_predict(self, dataframe: DataFrame, feature_cols: list[str]) -> tuple[Series, Series]:
        """
        Get Random Forest predictions for current data
        Returns: (long_probability, short_probability)
        """
        if self.random_forest_long is None or self.random_forest_short is None:
            return pd.Series(0.5, index=dataframe.index), pd.Series(0.5, index=dataframe.index)
        
        try:
            # Prepare features
            X = dataframe[feature_cols].values
            X = np.nan_to_num(X, nan=0.0)
            X_scaled = self.scaler.transform(X)
            
            # Get predictions
            long_proba = self.random_forest_long.predict_proba(X_scaled)[:, 1]
            short_proba = self.random_forest_short.predict_proba(X_scaled)[:, 1]
            
            return pd.Series(long_proba, index=dataframe.index), pd.Series(short_proba, index=dataframe.index)
            
        except Exception as e:
            logger.warning(f"RF prediction failed: {e}")
            return pd.Series(0.5, index=dataframe.index), pd.Series(0.5, index=dataframe.index)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FREQTRADE INTERFACE METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add all indicators to the dataframe
        This is where all three phases are computed
        """
        
        # ═══ PHASE 1: MATHEMATICAL TOOLS ═══
        
        # Basic indicators
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=21)
        
        # Moving averages
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        
        # ATR for volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
        dataframe['atr_percent'] = (dataframe['atr'] / dataframe['close']) * 100
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Stochastic
        stoch = ta.STOCH(dataframe)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']
        
        # Volume indicators
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # Adaptive VWAP
        dataframe['vwap'] = self.calculate_adaptive_vwap(dataframe, period=self.vwap_period.value)
        dataframe['vwap_distance'] = (dataframe['close'] - dataframe['vwap']) / dataframe['vwap']
        
        # Local extrema
        dataframe['local_max'], dataframe['local_min'] = self.calculate_local_extrema(dataframe)
        dataframe['distance_to_max'] = (dataframe['close'] - dataframe['local_max']) / dataframe['local_max']
        dataframe['distance_to_min'] = (dataframe['close'] - dataframe['local_min']) / dataframe['local_min']
        
        # Fibonacci levels
        if self.use_fibonacci.value:
            fib_levels = self.calculate_fibonacci_levels(dataframe, lookback=self.fib_lookback.value)
            for key, series in fib_levels.items():
                dataframe[key] = series
            
            # Distance to key Fibonacci levels
            dataframe['distance_to_fib_618'] = (dataframe['close'] - dataframe['fib_618']) / dataframe['fib_618']
            dataframe['distance_to_fib_382'] = (dataframe['close'] - dataframe['fib_382']) / dataframe['fib_382']
        
        # FFT cycle detection
        if self.use_fft_filter.value:
            dataframe['fft_cycle_strength'] = self.calculate_fft_cycles(dataframe, threshold=self.fft_threshold.value)
        
        # Markov probabilities
        markov_probs = self.calculate_markov_probabilities(dataframe)
        dataframe['markov_bullish'] = markov_probs['bullish']
        dataframe['markov_bearish'] = markov_probs['bearish']
        dataframe['markov_sideways'] = markov_probs['sideways']
        
        # Candlestick patterns
        dataframe = self.detect_candlestick_patterns(dataframe)
        
        # Bayesian sentiment
        dataframe['bayesian_sentiment'] = self.calculate_bayesian_sentiment(dataframe)
        
        # ═══ PHASE 1: CLUSTERING ═══
        
        if self.use_clustering.value:
            cluster_features = [
                'rsi', 'atr_percent', 'bb_width', 'volume_ratio',
                'vwap_distance', 'macd', 'stoch_k'
            ]
            dataframe['market_cluster'] = self.perform_clustering(dataframe, cluster_features, n_clusters=5)
        else:
            dataframe['market_cluster'] = 0
        
        # ═══ PHASE 1: GAUSSIAN PROCESS ═══
        
        if self.use_gaussian_process.value:
            gp_features = ['rsi', 'atr_percent', 'volume_ratio', 'macd', 'bb_width']
            dataframe['gp_prediction'], dataframe['gp_uncertainty'] = self.calculate_gaussian_process_prediction(
                dataframe, gp_features
            )
            
            # High confidence signal: positive prediction with low uncertainty
            dataframe['gp_confidence'] = dataframe['gp_prediction'] / (dataframe['gp_uncertainty'] + 1e-6)
        else:
            dataframe['gp_prediction'] = 0
            dataframe['gp_uncertainty'] = 1
            dataframe['gp_confidence'] = 0
        
        # ═══ PHASE 2: LSTM (Placeholder) ═══
        
        # TODO: Implement actual LSTM
        dataframe['lstm_signal'], dataframe['lstm_confidence'] = 0.5, 0.5
        
        # ═══ PHASE 3: RANDOM FOREST DECISION ═══
        
        # Define feature set for RF
        rf_features = [
            'rsi', 'rsi_slow', 'atr_percent', 'bb_width', 'volume_ratio',
            'vwap_distance', 'macd', 'macdhist', 'stoch_k', 'stoch_d',
            'distance_to_max', 'distance_to_min',
            'pattern_strength', 'bullish_pattern_count', 'bearish_pattern_count',
            'markov_bullish', 'markov_bearish', 'markov_sideways',
            'bayesian_sentiment', 'gp_confidence', 'market_cluster'
        ]
        
        # Add Fibonacci features if enabled
        if self.use_fibonacci.value:
            rf_features.extend(['distance_to_fib_618', 'distance_to_fib_382'])
        
        # Add FFT feature if enabled
        if self.use_fft_filter.value:
            rf_features.append('fft_cycle_strength')
        
        # Train RF models (in backtesting mode)
        if not self.models_loaded and len(dataframe) > 200:
            self.train_random_forest(dataframe, rf_features)
            self.save_models()
            self.models_loaded = True
        
        # Get RF predictions
        dataframe['rf_long_signal'], dataframe['rf_short_signal'] = self.random_forest_predict(dataframe, rf_features)
        
        # ═══ FINAL COMPOSITE SIGNALS ═══
        
        # Combine all signals into composite score
        dataframe['composite_long_score'] = (
            dataframe['rf_long_signal'] * 0.5 +  # 50% RF
            dataframe['bayesian_sentiment'] * 0.2 +  # 20% Bayesian
            dataframe['gp_confidence'].clip(-1, 1) * 0.15 +  # 15% GP
            (dataframe['pattern_strength'] / 10).clip(-1, 1) * 0.15  # 15% Patterns
        )
        
        dataframe['composite_short_score'] = (
            dataframe['rf_short_signal'] * 0.5 +  # 50% RF
            (1 - dataframe['bayesian_sentiment']) * 0.2 +  # 20% Bayesian (inverted)
            (-dataframe['gp_confidence']).clip(-1, 1) * 0.15 +  # 15% GP (inverted)
            (-dataframe['pattern_strength'] / 10).clip(-1, 1) * 0.15  # 15% Patterns (inverted)
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry signals based on Phase 3 Random Forest decisions
        """
        
        # Initialize entry columns
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        # ═══ LONG ENTRY CONDITIONS - RF DECISION BASED ═══
        
        # Phase 3: Random Forest makes the decision
        # Entry when RF and composite models agree
        dataframe.loc[
            (
                # Primary: Moderate composite score from all models (relaxed)
                (dataframe['composite_long_score'] > 0.25)
                # Secondary: Either strong RF signal OR moderate signal with RSI confirmation
                & (
                    (dataframe['rf_long_signal'] > 0.5)  # Strong RF signal alone
                    | (
                        (dataframe['rf_long_signal'] > 0.25)  # Moderate RF signal (relaxed)
                        & (dataframe['rsi'] < 45)  # with RSI support
                    )
                )
                # Tertiary: Decent volume
                & (dataframe['volume_ratio'] > 0.6)
            ),
            'enter_long'
        ] = 1
        
        # ═══ SHORT ENTRY CONDITIONS - RF DECISION BASED ═══
        
        # Phase 3: Random Forest makes the decision
        # Entry when RF and composite models agree
        dataframe.loc[
            (
                # Primary: Moderate composite score from all models (relaxed)
                (dataframe['composite_short_score'] > 0.25)
                # Secondary: Either strong RF signal OR moderate signal with RSI confirmation
                & (
                    (dataframe['rf_short_signal'] > 0.5)  # Strong RF signal alone
                    | (
                        (dataframe['rf_short_signal'] > 0.25)  # Moderate RF signal (relaxed)
                        & (dataframe['rsi'] > 55)  # with RSI support
                    )
                )
                # Tertiary: Decent volume
                & (dataframe['volume_ratio'] > 0.6)
            ),
            'enter_short'
        ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit signals
        """
        
        # Initialize exit columns
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        # ═══ LONG EXIT CONDITIONS - STRICT ═══
        
        # Exit only when multiple conditions confirm trend reversal
        dataframe.loc[
            (
                # Exit when composite score turns clearly negative (strict)
                (dataframe['composite_long_score'] < 0.2)
                # AND price is overbought (strict)
                & (dataframe['rsi'] > 75)
                # AND RF confirms exit
                & (dataframe['rf_long_signal'] < 0.4)
                # Confirm with volume
                & (dataframe['volume_ratio'] > 0.7)
            ),
            'exit_long'
        ] = 1
        
        # ═══ SHORT EXIT CONDITIONS - STRICT ═══
        
        # Exit only when multiple conditions confirm trend reversal
        dataframe.loc[
            (
                # Exit when composite score turns clearly negative (strict)
                (dataframe['composite_short_score'] < 0.2)
                # AND price is oversold (strict)
                & (dataframe['rsi'] < 25)
                # AND RF confirms exit
                & (dataframe['rf_short_signal'] < 0.4)
                # Confirm with volume
                & (dataframe['volume_ratio'] > 0.7)
            ),
            'exit_short'
        ] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs) -> float:
        """
        Dynamic leverage based on Kelly Criterion
        """
        if not self.use_kelly.value:
            return 1.0
        
        # Estimate win rate from recent trades (simplified)
        # In production, this should use actual trade history
        estimated_win_rate = 0.55
        estimated_avg_win = 0.03
        estimated_avg_loss = 0.02
        
        kelly_fraction = self.calculate_kelly_criterion(estimated_win_rate, estimated_avg_win, estimated_avg_loss)
        
        # Convert Kelly fraction to leverage (max 5x)
        dynamic_leverage = 1.0 + (kelly_fraction * 4)  # 1x to 5x
        
        return min(dynamic_leverage, max_leverage, 5.0)
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float | None, max_stake: float,
                           leverage: float, entry_tag: str | None, side: str,
                           **kwargs) -> float:
        """
        Custom stake amount based on Kelly Criterion
        """
        if not self.use_kelly.value:
            return proposed_stake
        
        # Get Kelly fraction
        estimated_win_rate = 0.55
        estimated_avg_win = 0.03
        estimated_avg_loss = 0.02
        
        kelly_fraction = self.calculate_kelly_criterion(estimated_win_rate, estimated_avg_win, estimated_avg_loss)
        
        # Adjust stake amount
        kelly_stake = proposed_stake * (kelly_fraction / self.kelly_fraction.value)
        
        return max(min_stake if min_stake else 0, min(kelly_stake, max_stake))
