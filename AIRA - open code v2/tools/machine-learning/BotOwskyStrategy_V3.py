# BOT-OWSKY Strategy V3.0 - Pattern-First ML Trading Strategy
#
# KEY CHANGES FROM V2:
# ═══════════════════════════════════════════════════════════════════════════════
#   - Pattern-First approach (35% weight vs 15% in V2)
#   - 3-Gate Entry System (Pattern → Trend → ML)
#   - Stricter thresholds for 15m timeframe
#   - ATR-based dynamic stoploss
#   - Declining DCA (not exponential)
#   - Faster Bayesian learning (2-3 trades vs 5+)
#   - Reduced leverage (2x max vs 5x)
#   - Simplified ML features (10 vs 14)
#
"""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import talib.abstract as ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
import warnings
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    BooleanParameter,
)
from freqtrade.persistence import Trade

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# COIN CONFIDENCE MANAGER V3 - Bayesian Learning
# ═══════════════════════════════════════════════════════════════════════════════

class CoinConfidenceManagerV3:
    """
    Bayesian confidence manager - learns meaningfully after 2-3 trades.
    """

    def __init__(self, storage_path: str = None):
        if storage_path is None:
            storage_path = Path("user_data/models/bot_owsky_v3")
        else:
            storage_path = Path(storage_path)

        storage_path.mkdir(parents=True, exist_ok=True)

        self.storage_path = storage_path
        self.confidence_file = storage_path / "coin_confidence_v3.json"
        self.patterns_file = storage_path / "pattern_performance_v3.json"
        self.regime_file = storage_path / "regime_performance_v3.json"

        self.coin_data = self._load_json(self.confidence_file)
        self.pattern_data = self._load_json(self.patterns_file)
        self.regime_data = self._load_json(self.regime_file)

        # Bayesian priors
        self.default_prior = {
            'win_rate': 0.50,
            'profit_factor': 1.0,
            'confidence': 50.0
        }

        logger.info(f"CoinConfidenceManagerV3 initialized with {len(self.coin_data)} coins")

    def _load_json(self, filepath: Path) -> Dict:
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load {filepath}: {e}")
        return {}

    def save(self):
        try:
            with open(self.confidence_file, 'w') as f:
                json.dump(self.coin_data, f, indent=2)
            with open(self.patterns_file, 'w') as f:
                json.dump(self.pattern_data, f, indent=2)
            with open(self.regime_file, 'w') as f:
                json.dump(self.regime_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def get_coin_confidence(self, pair: str) -> float:
        if pair not in self.coin_data:
            return 50.0
        return self.coin_data[pair].get('confidence', 50.0)

    def get_regime_performance(self, pair: str, regime: str) -> Dict:
        key = f"{pair}_{regime}"
        if key not in self.regime_data:
            return {'win_rate': 0.5, 'avg_profit': 0.0, 'trades': 0}
        return self.regime_data[key]

    def update_trade_result(self, pair: str, profit: float,
                           duration_minutes: int, entry_patterns: list,
                           exit_reason: str, regime: str = 'unknown'):
        if pair not in self.coin_data:
            self.coin_data[pair] = self._create_new_coin_entry(pair)

        data = self.coin_data[pair]
        data['total_trades'] = data.get('total_trades', 0) + 1
        data['total_profit'] = data.get('total_profit', 0.0) + profit

        is_win = profit > 0
        if is_win:
            data['wins'] = data.get('wins', 0) + 1
            data['total_win_profit'] = data.get('total_win_profit', 0.0) + profit
        else:
            data['losses'] = data.get('losses', 0) + 1
            data['total_loss_profit'] = data.get('total_loss_profit', 0.0) + abs(profit)

        self._update_pattern_performance(pair, entry_patterns, is_win, profit)
        self._update_regime_performance(pair, regime, is_win, profit)

        # Bayesian confidence calculation
        data['confidence'] = self._calculate_bayesian_confidence(data)

        self.save()

    def _create_new_coin_entry(self, pair: str) -> Dict:
        return {
            'pair': pair,
            'confidence': 50.0,
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_profit': 0.0,
            'total_win_profit': 0.0,
            'total_loss_profit': 0.0,
        }

    def _calculate_bayesian_confidence(self, data: Dict) -> float:
        """
        Bayesian confidence - meaningful after 2-3 trades instead of 5+
        """
        total = data.get('total_trades', 0)
        if total == 0:
            return 50.0

        # Prior weight decreases quickly
        # After 2 trades: 0.42, After 3: 0.32, After 5: 0.22
        prior_weight = 1 / (1 + total * 0.7)

        # Observed win rate
        wins = data.get('wins', 0)
        observed_win_rate = wins / total

        # Bayesian posterior for win rate
        posterior_win_rate = (
            prior_weight * self.default_prior['win_rate'] +
            (1 - prior_weight) * observed_win_rate
        )

        # Base confidence from win rate
        confidence = posterior_win_rate * 100

        # Profit factor adjustment (avoid division by zero)
        gross_profit = data.get('total_win_profit', 0.0)
        gross_loss = max(data.get('total_loss_profit', 0.0), 0.001)  # Min 0.001 to prevent div/0
        profit_factor = gross_profit / gross_loss

        if profit_factor > 1.5:
            confidence += 10
        elif profit_factor > 1.2:
            confidence += 5
        elif profit_factor < 0.8:
            confidence -= 10
        elif profit_factor < 1.0:
            confidence -= 5

        return max(0, min(100, confidence))

    def _update_pattern_performance(self, pair: str, patterns: list, is_win: bool, profit: float):
        for pattern in patterns:
            if pattern not in self.pattern_data:
                self.pattern_data[pattern] = {'wins': 0, 'losses': 0, 'total_profit': 0.0}
            if is_win:
                self.pattern_data[pattern]['wins'] += 1
            else:
                self.pattern_data[pattern]['losses'] += 1
            self.pattern_data[pattern]['total_profit'] += profit

    def _update_regime_performance(self, pair: str, regime: str, is_win: bool, profit: float):
        key = f"{pair}_{regime}"
        if key not in self.regime_data:
            self.regime_data[key] = {'wins': 0, 'losses': 0, 'total_profit': 0.0, 'trades': 0}

        self.regime_data[key]['trades'] += 1
        self.regime_data[key]['total_profit'] += profit
        if is_win:
            self.regime_data[key]['wins'] += 1
        else:
            self.regime_data[key]['losses'] += 1

        total = self.regime_data[key]['trades']
        self.regime_data[key]['win_rate'] = self.regime_data[key]['wins'] / total
        self.regime_data[key]['avg_profit'] = self.regime_data[key]['total_profit'] / total


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN ANALYZER V3
# ═══════════════════════════════════════════════════════════════════════════════

class PatternAnalyzerV3:
    """
    Pattern analysis with Bulkowski reliability ratings.
    Now primary decision maker (35% weight).
    """

    HIGH_RELIABILITY = {
        'CDL3WHITESOLDIERS': 0.78,
        'CDL3BLACKCROWS': 0.78,
        'CDLENGULFING': 0.63,
        'CDLMORNINGSTAR': 0.76,
        'CDLEVENINGSTAR': 0.72,
        'CDLPIERCING': 0.64,
        'CDLDARKCLOUDCOVER': 0.60,
    }

    MEDIUM_RELIABILITY = {
        'CDLHAMMER': 0.60,
        'CDLHANGINGMAN': 0.59,
        'CDLDOJI': 0.51,
        'CDLHARAMI': 0.53,
        'CDLINVERTEDHAMMER': 0.55,
        'CDLSHOOTINGSTAR': 0.58,
    }

    def __init__(self):
        self.all_patterns = list(self.HIGH_RELIABILITY.keys()) + list(self.MEDIUM_RELIABILITY.keys())

    def analyze(self, dataframe: DataFrame) -> DataFrame:
        df = dataframe.copy()

        pattern_signals = {}
        for pattern in self.all_patterns:
            try:
                pattern_signals[pattern] = getattr(ta, pattern)(df)
            except Exception:
                pattern_signals[pattern] = pd.Series(0, index=df.index)

        for pattern, sig in pattern_signals.items():
            df[f'pattern_{pattern}'] = sig

        df['pattern_confidence'] = self._calc_confidence(df, pattern_signals)
        df['bullish_pattern_count'] = sum((s > 0).fillna(False).astype(int) for s in pattern_signals.values())
        df['bearish_pattern_count'] = sum((s < 0).fillna(False).astype(int) for s in pattern_signals.values())
        df['pattern_strength'] = df['bullish_pattern_count'] - df['bearish_pattern_count']
        df['weighted_pattern_strength'] = self._calc_weighted_strength(pattern_signals)

        # Normalized pattern score (0-1)
        max_strength = len(self.all_patterns) * 0.78
        df['pattern_score'] = (df['weighted_pattern_strength'].clip(-max_strength, max_strength) + max_strength) / (2 * max_strength)

        return df

    def _calc_confidence(self, df: DataFrame, signals: Dict) -> Series:
        confidence = pd.Series(50.0, index=df.index)

        for i in range(len(df)):
            active = [(p, s.iloc[i]) for p, s in signals.items() if s.iloc[i] != 0]
            if active:
                confs = []
                for p, _ in active:
                    if p in self.HIGH_RELIABILITY:
                        confs.append(self.HIGH_RELIABILITY[p] * 100)
                    elif p in self.MEDIUM_RELIABILITY:
                        confs.append(self.MEDIUM_RELIABILITY[p] * 100)
                if confs:
                    confidence.iloc[i] = np.mean(confs)

        return confidence

    def _calc_weighted_strength(self, signals: Dict) -> Series:
        weighted = pd.Series(0.0, index=list(signals.values())[0].index)
        for p, s in signals.items():
            w = self.HIGH_RELIABILITY.get(p, self.MEDIUM_RELIABILITY.get(p, 0.5))
            weighted += s * w
        return weighted

    def get_active_patterns(self, row) -> List[str]:
        """Extract active pattern names from a dataframe row."""
        patterns = []
        for pattern in self.all_patterns:
            col = f'pattern_{pattern}'
            if col in row.index and row[col] != 0:
                patterns.append(pattern)
        return patterns


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN STRATEGY CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class BotOwskyStrategy_V3(IStrategy):
    """
    BOT-OWSKY V3: Pattern-First ML Strategy for 15m Futures Trading

    3-Gate Entry System:
    1. Pattern Gate: Min 55% confidence
    2. Trend Gate: Never trade against strong trends
    3. ML Gate: Ensemble prediction > 0.58
    """

    INTERFACE_VERSION = 3

    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════

    timeframe = '15m'
    startup_candle_count: int = 300  # More for 15m
    can_short = True

    # ROI - Faster exits for 15m
    minimal_roi = {
        "0": 0.08,
        "30": 0.04,
        "60": 0.025,
        "120": 0.015,
        "240": 0.008
    }

    # Stoploss - Default, ATR overrides via custom_stoploss
    stoploss = -0.04

    # Trailing stop - Tighter for 15m
    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    # Order configuration
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    # DCA - Declining, not exponential
    position_adjustment_enable = True
    max_entry_position_adjustment = 2

    # ═══════════════════════════════════════════════════════════════════════════
    # HYPEROPTABLE PARAMETERS - STRICTER THAN V2
    # ═══════════════════════════════════════════════════════════════════════════

    # Indicator periods
    atr_period = IntParameter(10, 20, default=14, space='buy', optimize=True)
    rsi_period = IntParameter(10, 21, default=14, space='buy', optimize=True)
    ema_fast = IntParameter(8, 21, default=12, space='buy', optimize=True)
    ema_slow = IntParameter(21, 55, default=30, space='buy', optimize=True)

    # Entry thresholds - STRICTER for 15m
    composite_threshold = DecimalParameter(0.30, 0.55, default=0.40, decimals=2, space='buy', optimize=True)
    pattern_min_confidence = IntParameter(50, 70, default=55, space='buy', optimize=True)
    rf_threshold = DecimalParameter(0.50, 0.70, default=0.58, decimals=2, space='buy', optimize=True)
    volume_threshold = DecimalParameter(0.8, 1.5, default=1.0, decimals=1, space='buy', optimize=True)

    # Trend filter
    adx_strong_threshold = IntParameter(25, 40, default=30, space='buy', optimize=True)
    adx_weak_threshold = IntParameter(15, 25, default=20, space='buy', optimize=True)

    # Divergence
    use_divergence = BooleanParameter(default=True, space='buy', optimize=True)
    divergence_lookback = IntParameter(10, 30, default=15, space='buy', optimize=True)
    divergence_boost = DecimalParameter(0.05, 0.15, default=0.10, decimals=2, space='buy', optimize=True)

    # ATR stoploss
    atr_stoploss_mult = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space='buy', optimize=True)
    max_stoploss_pct = DecimalParameter(0.03, 0.06, default=0.04, decimals=2, space='buy', optimize=True)

    # Exit parameters
    exit_rsi_high = IntParameter(65, 80, default=70, space='sell', optimize=True)
    exit_rsi_low = IntParameter(20, 35, default=30, space='sell', optimize=True)

    # RSI guards for entries - STRICTER
    rsi_long_max = IntParameter(45, 60, default=55, space='buy', optimize=True)
    rsi_short_min = IntParameter(40, 55, default=45, space='buy', optimize=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # ML CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════

    MIN_CANDLES_FOR_TRAINING = 300  # Down from 500
    RETRAIN_INTERVAL = 100

    ML_FEATURES = [
        'rsi', 'adx', 'atr_pct', 'bb_pct', 'volume_ratio',
        'macd_hist', 'pattern_strength', 'trend_strength',
        'momentum_quality', 'squeeze'
    ]

    # Component weights
    ML_WEIGHT = 0.35
    PATTERN_WEIGHT = 0.35
    TREND_WEIGHT = 0.20
    MOMENTUM_WEIGHT = 0.10

    # ═══════════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        self.model_dir = Path(config.get('user_data_dir', 'user_data')) / 'models' / 'bot_owsky_v3'
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # ML models
        self.rf_long = None
        self.rf_short = None
        self.gb_long = None
        self.gb_short = None
        self.scaler = RobustScaler()
        self.models_loaded = False
        self.last_train_candle = 0

        # Managers
        self.coin_confidence = CoinConfidenceManagerV3(str(self.model_dir))
        self.pattern_analyzer = PatternAnalyzerV3()

        # Current regime tracking
        self.current_regimes = {}

        logger.info("BOT-OWSKY V3.0 initialized - Pattern-First ML Strategy")

    # ═══════════════════════════════════════════════════════════════════════════
    # LEVERAGE
    # ═══════════════════════════════════════════════════════════════════════════

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Conservative leverage: max 2.0x (down from 5x in VVRP)
        """
        # Base leverage
        base_leverage = 2.0

        # Reduce for low confidence coins
        confidence = self.coin_confidence.get_coin_confidence(pair)
        if confidence < 40:
            base_leverage = 1.5
        elif confidence < 30:
            base_leverage = 1.0

        return min(base_leverage, max_leverage)

    # ═══════════════════════════════════════════════════════════════════════════
    # ATR-BASED CUSTOM STOPLOSS
    # ═══════════════════════════════════════════════════════════════════════════

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        ATR-based dynamic stoploss instead of fixed percentage.
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) == 0:
                return -self.max_stoploss_pct.value

            atr = dataframe['atr'].iloc[-1]
            entry_price = trade.open_rate

            # Base: 2x ATR
            atr_stop = (atr * self.atr_stoploss_mult.value) / entry_price

            # Tighten as profit increases
            if current_profit > 0.03:  # 3%+
                atr_stop *= 0.5  # 1x ATR
            elif current_profit > 0.015:  # 1.5%+
                atr_stop *= 0.7  # 1.4x ATR

            # Cap at max stoploss
            final_stop = min(atr_stop, self.max_stoploss_pct.value)

            return -final_stop

        except Exception as e:
            logger.warning(f"Error in custom_stoploss: {e}")
            return -self.max_stoploss_pct.value

    # ═══════════════════════════════════════════════════════════════════════════
    # DECLINING DCA
    # ═══════════════════════════════════════════════════════════════════════════

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Declining DCA: Each addition is 50% of previous (not exponential growth).
        """
        # Only DCA if down 2%+
        if current_profit > -0.02:
            return None

        filled_entries = trade.nr_of_successful_entries
        if filled_entries >= 3:  # Max 3 entries total
            return None

        # Declining stake: 50% of previous
        # Entry 1: 100%, Entry 2: 50%, Entry 3: 25%
        dca_multiplier = 0.5 ** filled_entries
        dca_stake = trade.stake_amount * dca_multiplier

        # Minimum viable stake
        if min_stake and dca_stake < min_stake:
            return None

        return dca_stake

    # ═══════════════════════════════════════════════════════════════════════════
    # CUSTOM STAKE AMOUNT
    # ═══════════════════════════════════════════════════════════════════════════

    def custom_stake_amount(self, pair: str, current_time: datetime,
                            current_rate: float, proposed_stake: float,
                            min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str],
                            side: str, **kwargs) -> float:
        """
        Confidence-based position sizing.
        """
        # Base: 80% of proposed
        base_stake = proposed_stake * 0.8

        # Confidence multiplier: 0.6x to 1.2x
        confidence = self.coin_confidence.get_coin_confidence(pair)
        conf_multiplier = 0.6 + (confidence / 100) * 0.6

        final_stake = base_stake * conf_multiplier

        return max(min_stake or 0, min(final_stake, max_stake))

    # ═══════════════════════════════════════════════════════════════════════════
    # DIVERGENCE DETECTION
    # ═══════════════════════════════════════════════════════════════════════════

    def detect_divergence(self, dataframe: DataFrame, lookback: int = 15) -> Tuple[Series, Series]:
        """Detect bullish and bearish RSI divergences."""
        bullish_div = pd.Series(0, index=dataframe.index)
        bearish_div = pd.Series(0, index=dataframe.index)

        if len(dataframe) < lookback + 5:
            return bullish_div, bearish_div

        close = dataframe['close'].values
        rsi = dataframe['rsi'].values

        for i in range(lookback, len(dataframe)):
            price_window = close[i-lookback:i+1]
            rsi_window = rsi[i-lookback:i+1]

            price_min_idx = np.argmin(price_window)
            price_max_idx = np.argmax(price_window)

            # Bullish: Price lower low, RSI higher low
            if price_min_idx > lookback // 2:
                prev_low_idx = np.argmin(price_window[:lookback//2])
                if price_window[price_min_idx] < price_window[prev_low_idx]:
                    if rsi_window[price_min_idx] > rsi_window[prev_low_idx]:
                        bullish_div.iloc[i] = 1

            # Bearish: Price higher high, RSI lower high
            if price_max_idx > lookback // 2:
                prev_high_idx = np.argmax(price_window[:lookback//2])
                if price_window[price_max_idx] > price_window[prev_high_idx]:
                    if rsi_window[price_max_idx] < rsi_window[prev_high_idx]:
                        bearish_div.iloc[i] = 1

        return bullish_div, bearish_div

    # ═══════════════════════════════════════════════════════════════════════════
    # MARKET REGIME DETECTION (Simplified to 3 states)
    # ═══════════════════════════════════════════════════════════════════════════

    def detect_market_regime(self, dataframe: DataFrame) -> Series:
        """
        Simplified 3-state regime:
        - trending: ADX > 30
        - weak_trend: ADX 20-30
        - ranging: ADX < 20
        """
        regime = pd.Series('ranging', index=dataframe.index)

        adx = dataframe['adx']
        ema_diff = (dataframe['ema_fast'] - dataframe['ema_slow']) / dataframe['ema_slow'] * 100

        # Strong trend
        strong_up = (adx > self.adx_strong_threshold.value) & (ema_diff > 0.5)
        strong_down = (adx > self.adx_strong_threshold.value) & (ema_diff < -0.5)

        # Weak trend
        weak_up = (adx > self.adx_weak_threshold.value) & (adx <= self.adx_strong_threshold.value) & (ema_diff > 0)
        weak_down = (adx > self.adx_weak_threshold.value) & (adx <= self.adx_strong_threshold.value) & (ema_diff < 0)

        regime.loc[strong_up] = 'strong_uptrend'
        regime.loc[strong_down] = 'strong_downtrend'
        regime.loc[weak_up] = 'weak_uptrend'
        regime.loc[weak_down] = 'weak_downtrend'

        return regime

    # ═══════════════════════════════════════════════════════════════════════════
    # ML TRAINING
    # ═══════════════════════════════════════════════════════════════════════════

    def train_ml_models(self, dataframe: DataFrame) -> bool:
        """Train ML models on historical data."""
        if len(dataframe) < self.MIN_CANDLES_FOR_TRAINING:
            return False

        try:
            # Prepare features
            features = self._prepare_ml_features(dataframe)
            if features is None or len(features) < self.MIN_CANDLES_FOR_TRAINING:
                return False

            # Create labels: +1.5% within 12 candles = success
            labels_long = self._create_labels(dataframe, direction='long', threshold=0.015, horizon=12)
            labels_short = self._create_labels(dataframe, direction='short', threshold=0.015, horizon=12)

            # Align data
            valid_idx = ~(features.isna().any(axis=1) | labels_long.isna() | labels_short.isna())
            X = features.loc[valid_idx]
            y_long = labels_long.loc[valid_idx]
            y_short = labels_short.loc[valid_idx]

            if len(X) < 200:
                return False

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train models
            self.rf_long = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10, n_jobs=-1, random_state=42)
            self.rf_short = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10, n_jobs=-1, random_state=42)
            self.gb_long = GradientBoostingClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
            self.gb_short = GradientBoostingClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)

            self.rf_long.fit(X_scaled, y_long)
            self.rf_short.fit(X_scaled, y_short)
            self.gb_long.fit(X_scaled, y_long)
            self.gb_short.fit(X_scaled, y_short)

            self.models_loaded = True
            self.last_train_candle = len(dataframe)
            logger.info(f"V3 ML models trained on {len(X)} samples")
            return True

        except Exception as e:
            logger.error(f"Error training ML models: {e}")
            return False

    def _prepare_ml_features(self, dataframe: DataFrame) -> Optional[DataFrame]:
        """Prepare feature matrix for ML."""
        required = ['rsi', 'adx', 'atr', 'bb_pct', 'volume_ratio', 'macd_hist', 'pattern_strength', 'ema_fast', 'ema_slow']
        if not all(col in dataframe.columns for col in required):
            return None

        features = pd.DataFrame(index=dataframe.index)
        features['rsi'] = dataframe['rsi']
        features['adx'] = dataframe['adx']
        features['atr_pct'] = dataframe['atr'] / dataframe['close'] * 100
        features['bb_pct'] = dataframe['bb_pct']
        features['volume_ratio'] = dataframe['volume_ratio']
        features['macd_hist'] = dataframe['macd_hist']
        features['pattern_strength'] = dataframe['pattern_strength']
        features['trend_strength'] = (dataframe['ema_fast'] - dataframe['ema_slow']) / dataframe['ema_slow'] * 100
        features['momentum_quality'] = dataframe.get('momentum_quality', 50)
        features['squeeze'] = dataframe.get('squeeze', 0).astype(float)

        return features

    def _create_labels(self, dataframe: DataFrame, direction: str, threshold: float, horizon: int) -> Series:
        """Create binary labels for ML training."""
        labels = pd.Series(0, index=dataframe.index)
        close = dataframe['close'].values

        for i in range(len(close) - horizon):
            future_max = np.max(close[i+1:i+horizon+1])
            future_min = np.min(close[i+1:i+horizon+1])

            if direction == 'long':
                if (future_max - close[i]) / close[i] >= threshold:
                    labels.iloc[i] = 1
            else:  # short
                if (close[i] - future_min) / close[i] >= threshold:
                    labels.iloc[i] = 1

        return labels

    def _get_ml_predictions(self, dataframe: DataFrame) -> Tuple[Series, Series]:
        """Get ML ensemble predictions."""
        if not self.models_loaded:
            return pd.Series(0.5, index=dataframe.index), pd.Series(0.5, index=dataframe.index)

        try:
            features = self._prepare_ml_features(dataframe)
            if features is None:
                return pd.Series(0.5, index=dataframe.index), pd.Series(0.5, index=dataframe.index)

            X_scaled = self.scaler.transform(features.fillna(0))

            # RF predictions (50% weight)
            rf_long = self.rf_long.predict_proba(X_scaled)[:, 1]
            rf_short = self.rf_short.predict_proba(X_scaled)[:, 1]

            # GB predictions (30% weight)
            gb_long = self.gb_long.predict_proba(X_scaled)[:, 1]
            gb_short = self.gb_short.predict_proba(X_scaled)[:, 1]

            # Ensemble
            ml_long = rf_long * 0.6 + gb_long * 0.4
            ml_short = rf_short * 0.6 + gb_short * 0.4

            return pd.Series(ml_long, index=dataframe.index), pd.Series(ml_short, index=dataframe.index)

        except Exception as e:
            logger.warning(f"ML prediction error: {e}")
            return pd.Series(0.5, index=dataframe.index), pd.Series(0.5, index=dataframe.index)

    # ═══════════════════════════════════════════════════════════════════════════
    # POPULATE INDICATORS
    # ═══════════════════════════════════════════════════════════════════════════

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate all indicators."""
        pair = metadata['pair']

        # Basic indicators
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        # EMAs
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)

        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        dataframe['bb_pct'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])

        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['macd_hist'] = macd['macdhist']

        # Volume
        dataframe['volume_ma'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma']

        # Keltner for squeeze detection
        dataframe['kc_upper'] = ta.EMA(dataframe, timeperiod=20) + ta.ATR(dataframe, timeperiod=14) * 1.5
        dataframe['kc_lower'] = ta.EMA(dataframe, timeperiod=20) - ta.ATR(dataframe, timeperiod=14) * 1.5
        dataframe['squeeze'] = (dataframe['bb_lower'] > dataframe['kc_lower']) & (dataframe['bb_upper'] < dataframe['kc_upper'])

        # Momentum quality
        roc = ta.ROC(dataframe, timeperiod=10)
        dataframe['momentum_quality'] = (
            dataframe['rsi'].clip(0, 100) * 0.4 +
            (dataframe['macd_hist'] / dataframe['close'] * 1000).clip(-50, 50) + 50 * 0.3 +
            (roc.clip(-10, 10) + 10) * 5 * 0.3
        )

        # Pattern analysis
        dataframe = self.pattern_analyzer.analyze(dataframe)

        # Divergence
        if self.use_divergence.value:
            bullish_div, bearish_div = self.detect_divergence(dataframe, self.divergence_lookback.value)
            dataframe['bullish_div'] = bullish_div
            dataframe['bearish_div'] = bearish_div
        else:
            dataframe['bullish_div'] = 0
            dataframe['bearish_div'] = 0

        # Market regime
        dataframe['regime'] = self.detect_market_regime(dataframe)
        self.current_regimes[pair] = dataframe['regime'].iloc[-1] if len(dataframe) > 0 else 'unknown'

        # Trend strength (normalized 0-1)
        trend_diff = (dataframe['ema_fast'] - dataframe['ema_slow']) / dataframe['ema_slow'] * 100
        dataframe['trend_strength'] = trend_diff
        dataframe['trend_score'] = (trend_diff.clip(-5, 5) + 5) / 10

        # ML predictions
        need_retrain = (
            not self.models_loaded or
            len(dataframe) - self.last_train_candle > self.RETRAIN_INTERVAL
        )
        if need_retrain and len(dataframe) >= self.MIN_CANDLES_FOR_TRAINING:
            self.train_ml_models(dataframe)

        ml_long, ml_short = self._get_ml_predictions(dataframe)
        dataframe['ml_long'] = ml_long
        dataframe['ml_short'] = ml_short

        # Composite scores
        dataframe['composite_long'] = self._calculate_composite_score(dataframe, 'long')
        dataframe['composite_short'] = self._calculate_composite_score(dataframe, 'short')

        return dataframe

    def _calculate_composite_score(self, dataframe: DataFrame, direction: str) -> Series:
        """
        Calculate composite entry score with V3 weights:
        - ML: 35%
        - Pattern: 35%
        - Trend: 20%
        - Momentum: 10%
        """
        if direction == 'long':
            ml_score = dataframe['ml_long']
            pattern_score = dataframe['pattern_score']
            trend_score = dataframe['trend_score']
        else:
            ml_score = dataframe['ml_short']
            pattern_score = 1 - dataframe['pattern_score']  # Invert for shorts
            trend_score = 1 - dataframe['trend_score']

        # Momentum score (normalized 0-1)
        momentum_score = dataframe['momentum_quality'] / 100

        composite = (
            ml_score * self.ML_WEIGHT +
            pattern_score * self.PATTERN_WEIGHT +
            trend_score * self.TREND_WEIGHT +
            momentum_score * self.MOMENTUM_WEIGHT
        )

        # Divergence bonus
        if direction == 'long':
            composite = composite + dataframe['bullish_div'] * self.divergence_boost.value
        else:
            composite = composite + dataframe['bearish_div'] * self.divergence_boost.value

        return composite.clip(0, 1)

    # ═══════════════════════════════════════════════════════════════════════════
    # 3-GATE ENTRY SYSTEM
    # ═══════════════════════════════════════════════════════════════════════════

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        3-Gate Entry System:
        1. Pattern Gate: Min 55% confidence
        2. Trend Gate: Never against strong trends
        3. ML Gate: Ensemble > 0.58
        """
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0

        # Gate 1: Pattern confidence
        pattern_ok = dataframe['pattern_confidence'] >= self.pattern_min_confidence.value

        # Gate 2: Trend alignment (never against strong trends)
        long_trend_ok = ~(
            (dataframe['regime'] == 'strong_downtrend') |
            ((dataframe['adx'] > self.adx_strong_threshold.value) & (dataframe['ema_fast'] < dataframe['ema_slow']))
        )
        short_trend_ok = ~(
            (dataframe['regime'] == 'strong_uptrend') |
            ((dataframe['adx'] > self.adx_strong_threshold.value) & (dataframe['ema_fast'] > dataframe['ema_slow']))
        )

        # Gate 3: ML ensemble (pattern-first fallback when ML not useful)
        # If ML predictions are near 0 or 1, they're not calibrated properly
        ml_avg = (dataframe['ml_long'].mean() + dataframe['ml_short'].mean()) / 2
        ml_is_useful = self.models_loaded and 0.2 < ml_avg < 0.8

        if ml_is_useful:
            ml_long_ok = dataframe['ml_long'] >= self.rf_threshold.value
            ml_short_ok = dataframe['ml_short'] >= self.rf_threshold.value
        else:
            # Pattern-first mode: ML gate always passes, rely on pattern + trend + composite
            ml_long_ok = True
            ml_short_ok = True

        # Additional filters
        volume_ok = dataframe['volume_ratio'] >= self.volume_threshold.value
        rsi_long_ok = dataframe['rsi'] < self.rsi_long_max.value
        rsi_short_ok = dataframe['rsi'] > self.rsi_short_min.value
        composite_long_ok = dataframe['composite_long'] >= self.composite_threshold.value
        composite_short_ok = dataframe['composite_short'] >= self.composite_threshold.value

        # Long entries
        long_conditions = (
            pattern_ok &
            long_trend_ok &
            ml_long_ok &
            volume_ok &
            rsi_long_ok &
            composite_long_ok
        )

        # Short entries
        short_conditions = (
            pattern_ok &
            short_trend_ok &
            ml_short_ok &
            volume_ok &
            rsi_short_ok &
            composite_short_ok
        )

        dataframe.loc[long_conditions, 'enter_long'] = 1
        dataframe.loc[short_conditions, 'enter_short'] = 1

        # Entry tags
        dataframe.loc[long_conditions, 'enter_tag'] = dataframe.apply(
            lambda row: self._create_entry_tag(row, 'long'), axis=1
        )
        dataframe.loc[short_conditions, 'enter_tag'] = dataframe.apply(
            lambda row: self._create_entry_tag(row, 'short'), axis=1
        )

        return dataframe

    def _create_entry_tag(self, row, direction: str) -> str:
        """Create informative entry tag."""
        import math
        dir_code = 'L' if direction == 'long' else 'S'
        regime = str(row.get('regime', 'unknown'))[:2].upper()
        rsi_val = row.get('rsi', 50)
        adx_val = row.get('adx', 20)
        conf_val = row.get('pattern_confidence', 50)
        rsi = int(rsi_val) if not (isinstance(rsi_val, float) and math.isnan(rsi_val)) else 50
        adx = int(adx_val) if not (isinstance(adx_val, float) and math.isnan(adx_val)) else 20
        conf = int(conf_val) if not (isinstance(conf_val, float) and math.isnan(conf_val)) else 50

        patterns = self.pattern_analyzer.get_active_patterns(row)
        pattern_code = '_'.join(patterns[:2]) if patterns else 'NP'

        return f"{dir_code}_{regime}_r{rsi}_a{adx}_c{conf}_{pattern_code}_v3"

    # ═══════════════════════════════════════════════════════════════════════════
    # EXIT LOGIC
    # ═══════════════════════════════════════════════════════════════════════════

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit based on RSI extremes and trend reversal."""
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0

        # RSI-based exits
        dataframe.loc[
            (dataframe['rsi'] > self.exit_rsi_high.value) &
            (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1)),
            'exit_long'
        ] = 1

        dataframe.loc[
            (dataframe['rsi'] < self.exit_rsi_low.value) &
            (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1)),
            'exit_short'
        ] = 1

        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """Priority-ordered exit system."""
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) == 0:
                return None

            last = dataframe.iloc[-1]
            is_long = not trade.is_short

            # Priority 1: Regime reversal
            if is_long and last['regime'] == 'strong_downtrend':
                return 'regime_reversal'
            if not is_long and last['regime'] == 'strong_uptrend':
                return 'regime_reversal'

            # Priority 2: Pattern reversal with high confidence
            if is_long and last['pattern_strength'] < -2 and last['pattern_confidence'] > 65:
                return 'bearish_pattern'
            if not is_long and last['pattern_strength'] > 2 and last['pattern_confidence'] > 65:
                return 'bullish_pattern'

            # Priority 3: Time-based (stagnant trade)
            trade_hours = (current_time - trade.open_date_utc).total_seconds() / 3600
            if trade_hours > 24 and abs(current_profit) < 0.005:
                return 'time_exit_stagnant'

            return None

        except Exception as e:
            logger.warning(f"Error in custom_exit: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════════
    # TRADE RESULT CALLBACK
    # ═══════════════════════════════════════════════════════════════════════════

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str,
                           amount: float, rate: float, time_in_force: str,
                           exit_reason: str, current_time: datetime, **kwargs) -> bool:
        """Update confidence manager on trade exit."""
        try:
            profit = trade.calc_profit_ratio(rate)
            duration = (current_time - trade.open_date_utc).total_seconds() / 60

            # Extract patterns and regime from entry tag
            patterns = []
            regime = 'unknown'
            if trade.enter_tag:
                parts = trade.enter_tag.split('_')
                for part in parts:
                    if part.startswith('CDL'):
                        patterns.append(part)
                if len(parts) >= 2:
                    regime_code = parts[1]
                    regime_map = {'ST': 'strong_trend', 'WE': 'weak_trend', 'RA': 'ranging'}
                    regime = regime_map.get(regime_code, 'unknown')

            self.coin_confidence.update_trade_result(
                pair=pair,
                profit=profit,
                duration_minutes=int(duration),
                entry_patterns=patterns,
                exit_reason=exit_reason,
                regime=regime
            )

        except Exception as e:
            logger.warning(f"Error updating confidence: {e}")

        return True
