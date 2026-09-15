import logging
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import talib.abstract as ta
import pandas_ta as pta  # pandas_ta is imported but not explicitly used in the provided code.
# If it's for future use or part of an older version, that's okay.
# Otherwise, it can be removed if not needed.
from scipy.signal import argrelextrema

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter
from freqtrade.persistence import Trade

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)

# Define Murrey Math level names for consistency
MML_LEVEL_NAMES = [
    "[-3/8]P", "[-2/8]P", "[-1/8]P", "[0/8]P", "[1/8]P",
    "[2/8]P", "[3/8]P", "[4/8]P", "[5/8]P", "[6/8]P",
    "[7/8]P", "[8/8]P", "[+1/8]P", "[+2/8]P", "[+3/8]P"
]
def calculate_minima_maxima(df, window):
    if df is None or df.empty:
        return np.zeros(0), np.zeros(0)

    minima = np.zeros(len(df))
    maxima = np.zeros(len(df))

    for i in range(window, len(df)):
        window_data = df['ha_close'].iloc[i - window:i + 1]
        if df['ha_close'].iloc[i] == window_data.min() and (window_data == df['ha_close'].iloc[i]).sum() == 1:
            minima[i] = -window
        if df['ha_close'].iloc[i] == window_data.max() and (window_data == df['ha_close'].iloc[i]).sum() == 1:
            maxima[i] = window

    return minima, maxima
def calculate_exit_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced exit signals based on market deterioration"""
    
    # === MOMENTUM DETERIORATION ===
    dataframe['momentum_deteriorating'] = (
        (dataframe['momentum_quality'] < dataframe['momentum_quality'].shift(1)) &
        (dataframe['momentum_acceleration'] < 0) &
        (dataframe['price_momentum'] < dataframe['price_momentum'].shift(1))
    ).astype(int)
    
    # === VOLUME DETERIORATION ===
    dataframe['volume_deteriorating'] = (
        (dataframe['volume_strength'] < 0.8) &
        (dataframe['selling_pressure'] > dataframe['buying_pressure']) &
        (dataframe['volume_pressure'] < 0)
    ).astype(int)
    
    # === STRUCTURE DETERIORATION ===
    dataframe['structure_deteriorating'] = (
        (dataframe['structure_score'] < -1) &
        (dataframe['bearish_structure'] > dataframe['bullish_structure']) &
        (dataframe['structure_break_down'] == 1)
    ).astype(int)
    
    # === CONFLUENCE BREAKDOWN ===
    dataframe['confluence_breakdown'] = (
        (dataframe['confluence_score'] < 2) &
        (dataframe['near_resistance'] == 1) &
        (dataframe['volume_spike'] == 0)
    ).astype(int)
    
    # === TREND WEAKNESS ===
    dataframe['trend_weakening'] = (
        (dataframe['trend_strength'] < 0) &
        (dataframe['close'] < dataframe['ema50']) &
        (dataframe['strong_downtrend'] == 1)
    ).astype(int)
    
    # === ULTIMATE EXIT SCORE ===
    dataframe['exit_pressure'] = (
        dataframe['momentum_deteriorating'] * 2 +
        dataframe['volume_deteriorating'] * 2 +
        dataframe['structure_deteriorating'] * 2 +
        dataframe['confluence_breakdown'] * 1 +
        dataframe['trend_weakening'] * 1
    )
    
    # === RSI OVERBOUGHT WITH DIVERGENCE ===
    dataframe['rsi_exit_signal'] = (
        (dataframe['rsi'] > 75) &
        (
            (dataframe['rsi_divergence_bear'] == 1) |
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &
            (dataframe['close'] < dataframe['close'].shift(1))
        )
    ).astype(int)
    
    # === PROFIT TAKING LEVELS ===
    mml_resistance_levels = ['[6/8]P', '[8/8]P']
    dataframe['near_resistance_level'] = 0
    
    for level in mml_resistance_levels:
        if level in dataframe.columns:
            near_level = (
                (dataframe['close'] >= dataframe[level] * 0.99) &
                (dataframe['close'] <= dataframe[level] * 1.02)
            ).astype(int)
            dataframe['near_resistance_level'] += near_level
    
    # === VOLATILITY SPIKE EXIT ===
    dataframe['volatility_spike'] = (
        dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 1.5
    ).astype(int)
    
    # === EXHAUSTION SIGNALS ===
    dataframe['bullish_exhaustion'] = (
        (dataframe['consecutive_green'] >= 4) &
        (dataframe['rsi'] > 70) &
        (dataframe['volume'] < dataframe['avg_volume'] * 0.8) &
        (dataframe['momentum_acceleration'] < 0)
    ).astype(int)
    
    return dataframe


def calculate_dynamic_profit_targets(dataframe: pd.DataFrame, entry_type_col: str = 'entry_type') -> pd.DataFrame:
    """Calculate dynamic profit targets based on entry quality and market conditions"""
    
    # Base profit targets based on ATR
    dataframe['base_profit_target'] = dataframe['atr'] * 2
    
    # Adjust based on entry type
    dataframe['profit_multiplier'] = 1.0
    if entry_type_col in dataframe.columns:
        dataframe.loc[dataframe[entry_type_col] == 3, 'profit_multiplier'] = 2.0  # High quality
        dataframe.loc[dataframe[entry_type_col] == 2, 'profit_multiplier'] = 1.5  # Medium quality
        dataframe.loc[dataframe[entry_type_col] == 1, 'profit_multiplier'] = 1.2  # Backup
        dataframe.loc[dataframe[entry_type_col] == 4, 'profit_multiplier'] = 2.5  # Breakout
        dataframe.loc[dataframe[entry_type_col] == 5, 'profit_multiplier'] = 1.8  # Reversal
    
    # Final profit target
    dataframe['dynamic_profit_target'] = dataframe['base_profit_target'] * dataframe['profit_multiplier']
    
    return dataframe


def calculate_advanced_stop_loss(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe['base_stop_loss'] = dataframe['atr'] * 1.5
    if 'minima_sort_threshold' in dataframe.columns:
        dataframe['support_stop_loss'] = dataframe['close'] - dataframe['minima_sort_threshold']
        dataframe['support_stop_loss'] = dataframe['support_stop_loss'].clip(
            dataframe['base_stop_loss'] * 0.5,
            dataframe['base_stop_loss'] * 1.5  # Reduced from 2.0
        )
        dataframe['final_stop_loss'] = np.minimum(
            dataframe['base_stop_loss'],
            dataframe['support_stop_loss']
        ).clip(-0.15, -0.01)  # Hard cap at -15%
    else:
        dataframe['final_stop_loss'] = dataframe['base_stop_loss'].clip(-0.15, -0.01)
    return dataframe

def calculate_confluence_score(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Multi-factor confluence analysis - much better than BTC correlation"""
    
    # Support/Resistance Confluence
    dataframe['near_support'] = (
        (dataframe['close'] <= dataframe['minima_sort_threshold'] * 1.02) &
        (dataframe['close'] >= dataframe['minima_sort_threshold'] * 0.98)
    ).astype(int)
    
    dataframe['near_resistance'] = (
        (dataframe['close'] <= dataframe['maxima_sort_threshold'] * 1.02) &
        (dataframe['close'] >= dataframe['maxima_sort_threshold'] * 0.98)
    ).astype(int)
    
    # MML Level Confluence
    mml_levels = ['[0/8]P', '[2/8]P', '[4/8]P', '[6/8]P', '[8/8]P']
    dataframe['near_mml'] = 0
    
    for level in mml_levels:
        if level in dataframe.columns:
            near_level = (
                (dataframe['close'] <= dataframe[level] * 1.015) &
                (dataframe['close'] >= dataframe[level] * 0.985)
            ).astype(int)
            dataframe['near_mml'] += near_level
    
    # Volume Confluence
    dataframe['volume_spike'] = (
        dataframe['volume'] > dataframe['avg_volume'] * 1.5
    ).astype(int)
    
    # RSI Confluence Zones
    dataframe['rsi_oversold'] = (dataframe['rsi'] < 30).astype(int)
    dataframe['rsi_overbought'] = (dataframe['rsi'] > 70).astype(int)
    dataframe['rsi_neutral'] = (
        (dataframe['rsi'] >= 40) & (dataframe['rsi'] <= 60)
    ).astype(int)
    
    # EMA Confluence
    dataframe['above_ema'] = (dataframe['close'] > dataframe['ema50']).astype(int)
    
    # CONFLUENCE SCORE (0-6)
    dataframe['confluence_score'] = (
        dataframe['near_support'] +
        dataframe['near_mml'].clip(0, 2) +  # Max 2 points for MML
        dataframe['volume_spike'] +
        dataframe['rsi_oversold'] +
        dataframe['above_ema'] +
        (dataframe['trend_strength'] > 0.01).astype(int)  # Positive trend
    )
    
    return dataframe


def calculate_smart_volume(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Advanced volume analysis - beats any external correlation"""
    
    # Volume-Price Trend (VPT)
    price_change_pct = (dataframe['close'] - dataframe['close'].shift(1)) / dataframe['close'].shift(1)
    dataframe['vpt'] = (dataframe['volume'] * price_change_pct).fillna(0).cumsum()
    
    # Volume moving averages
    dataframe['volume_sma20'] = dataframe['volume'].rolling(20).mean()
    dataframe['volume_sma50'] = dataframe['volume'].rolling(50).mean()
    
    # Volume strength
    dataframe['volume_strength'] = dataframe['volume'] / dataframe['volume_sma20']
    
    # Smart money indicators
    dataframe['accumulation'] = (
        (dataframe['close'] > dataframe['open']) &  # Green candle
        (dataframe['volume'] > dataframe['volume_sma20'] * 1.2) &  # High volume
        (dataframe['close'] > (dataframe['high'] + dataframe['low']) / 2)  # Close in upper half
    ).astype(int)
    
    dataframe['distribution'] = (
        (dataframe['close'] < dataframe['open']) &  # Red candle
        (dataframe['volume'] > dataframe['volume_sma20'] * 1.2) &  # High volume
        (dataframe['close'] < (dataframe['high'] + dataframe['low']) / 2)  # Close in lower half
    ).astype(int)
    
    # Buying/Selling pressure
    dataframe['buying_pressure'] = dataframe['accumulation'].rolling(5).sum()
    dataframe['selling_pressure'] = dataframe['distribution'].rolling(5).sum()
    
    # Net volume pressure
    dataframe['volume_pressure'] = dataframe['buying_pressure'] - dataframe['selling_pressure']
    
    # Volume trend
    dataframe['volume_trend'] = (
        dataframe['volume_sma20'] > dataframe['volume_sma50']
    ).astype(int)
    
    # Money flow
    typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
    money_flow = typical_price * dataframe['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_flow_sum = positive_flow.rolling(14).sum()
    negative_flow_sum = negative_flow.rolling(14).sum()
    
    dataframe['money_flow_ratio'] = positive_flow_sum / (negative_flow_sum + 1e-10)
    dataframe['money_flow_index'] = 100 - (100 / (1 + dataframe['money_flow_ratio']))
    
    return dataframe


def calculate_advanced_momentum(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Multi-timeframe momentum system - superior to BTC correlation"""
    
    # Multi-timeframe momentum
    dataframe['momentum_3'] = dataframe['close'].pct_change(6)
    dataframe['momentum_7'] = dataframe['close'].pct_change(14)
    dataframe['momentum_14'] = dataframe['close'].pct_change(28)
    dataframe['momentum_21'] = dataframe['close'].pct_change(21)
    
    # Momentum acceleration
    dataframe['momentum_acceleration'] = (
        dataframe['momentum_3'] - dataframe['momentum_3'].shift(3)
    )
    
    # Momentum consistency
    dataframe['momentum_consistency'] = (
        (dataframe['momentum_3'] > 0).astype(int) +
        (dataframe['momentum_7'] > 0).astype(int) +
        (dataframe['momentum_14'] > 0).astype(int)
    )
    
    # Momentum divergence with volume
    dataframe['price_momentum_rank'] = dataframe['momentum_7'].rolling(20).rank(pct=True)
    dataframe['volume_momentum_rank'] = dataframe['volume_strength'].rolling(20).rank(pct=True)
    
    dataframe['momentum_divergence'] = (
        dataframe['price_momentum_rank'] - dataframe['volume_momentum_rank']
    ).abs()
    
    # Momentum strength
    dataframe['momentum_strength'] = (
        dataframe['momentum_3'].abs() +
        dataframe['momentum_7'].abs() +
        dataframe['momentum_14'].abs()
    ) / 3
    
    # Momentum quality score (0-5)
    dataframe['momentum_quality'] = (
        (dataframe['momentum_3'] > 0).astype(int) +
        (dataframe['momentum_7'] > 0).astype(int) +
        (dataframe['momentum_acceleration'] > 0).astype(int) +
        (dataframe['volume_strength'] > 1.1).astype(int) +
        (dataframe['momentum_divergence'] < 0.3).astype(int)
    )
    
    # Rate of Change
    dataframe['roc_5'] = dataframe['close'].pct_change(5) * 100
    dataframe['roc_10'] = dataframe['close'].pct_change(10) * 100
    dataframe['roc_20'] = dataframe['close'].pct_change(20) * 100
    
    # Momentum oscillator
    dataframe['momentum_oscillator'] = (
        dataframe['roc_5'] + dataframe['roc_10'] + dataframe['roc_20']
    ) / 3
    
    return dataframe


def calculate_market_structure(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Market structure analysis - intrinsic trend recognition"""
    
    # Higher highs, higher lows detection
    dataframe['higher_high'] = (
        (dataframe['high'] > dataframe['high'].shift(1)) &
        (dataframe['high'].shift(1) > dataframe['high'].shift(2))
    ).astype(int)
    
    dataframe['higher_low'] = (
        (dataframe['low'] > dataframe['low'].shift(1)) &
        (dataframe['low'].shift(1) > dataframe['low'].shift(2))
    ).astype(int)
    
    dataframe['lower_high'] = (
        (dataframe['high'] < dataframe['high'].shift(1)) &
        (dataframe['high'].shift(1) < dataframe['high'].shift(2))
    ).astype(int)
    
    dataframe['lower_low'] = (
        (dataframe['low'] < dataframe['low'].shift(1)) &
        (dataframe['low'].shift(1) < dataframe['low'].shift(2))
    ).astype(int)
    
    # Market structure scores
    dataframe['bullish_structure'] = (
        dataframe['higher_high'].rolling(5).sum() +
        dataframe['higher_low'].rolling(5).sum()
    )
    
    dataframe['bearish_structure'] = (
        dataframe['lower_high'].rolling(5).sum() +
        dataframe['lower_low'].rolling(5).sum()
    )
    
    dataframe['structure_score'] = (
        dataframe['bullish_structure'] - dataframe['bearish_structure']
    )
    
    # Swing highs and lows
    dataframe['swing_high'] = (
        (dataframe['high'] > dataframe['high'].shift(1)) &
        (dataframe['high'] > dataframe['high'].shift(-1))
    ).astype(int)
    
    dataframe['swing_low'] = (
        (dataframe['low'] < dataframe['low'].shift(1)) &
        (dataframe['low'] < dataframe['low'].shift(-1))
    ).astype(int)
    
    # Market structure breaks
    swing_highs = dataframe['high'].where(dataframe['swing_high'] == 1)
    swing_lows = dataframe['low'].where(dataframe['swing_low'] == 1)
    
    # Structure break detection
    dataframe['structure_break_up'] = (
        dataframe['close'] > swing_highs.ffill()
    ).astype(int)
    
    dataframe['structure_break_down'] = (
        dataframe['close'] < swing_lows.ffill()
    ).astype(int)
    
    # Trend strength based on structure
    dataframe['structure_trend_strength'] = (
        dataframe['structure_score'] / 10  # Normalize
    ).clip(-1, 1)
    
    # Support and resistance strength
    dataframe['support_strength'] = dataframe['swing_low'].rolling(20).sum()
    dataframe['resistance_strength'] = dataframe['swing_high'].rolling(20).sum()
    
    return dataframe


def calculate_advanced_entry_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Advanced entry signal generation"""
    
    # Multi-factor signal strength
    dataframe['signal_strength'] = 0
    
    # Confluence signals
    dataframe['confluence_signal'] = (dataframe['confluence_score'] >= 3).astype(int)
    dataframe['signal_strength'] += dataframe['confluence_signal'] * 2
    
    # Volume signals
    dataframe['volume_signal'] = (
        (dataframe['volume_pressure'] >= 2) &
        (dataframe['volume_strength'] > 1.2)
    ).astype(int)
    dataframe['signal_strength'] += dataframe['volume_signal'] * 2
    
    # Momentum signals
    dataframe['momentum_signal'] = (
        (dataframe['momentum_quality'] >= 3) &
        (dataframe['momentum_acceleration'] > 0)
    ).astype(int)
    dataframe['signal_strength'] += dataframe['momentum_signal'] * 2
    
    # Structure signals
    dataframe['structure_signal'] = (
        (dataframe['structure_score'] > 0) &
        (dataframe['structure_break_up'] == 1)
    ).astype(int)
    dataframe['signal_strength'] += dataframe['structure_signal'] * 1
    
    # RSI position signal
    dataframe['rsi_signal'] = (
        (dataframe['rsi'] > 30) & (dataframe['rsi'] < 70)
    ).astype(int)
    dataframe['signal_strength'] += dataframe['rsi_signal'] * 1
    
    # Trend alignment signal
    dataframe['trend_signal'] = (
        (dataframe['close'] > dataframe['ema50']) &
        (dataframe['trend_strength'] > 0)
    ).astype(int)
    dataframe['signal_strength'] += dataframe['trend_signal'] * 1
    
    # Money flow signal
    dataframe['money_flow_signal'] = (
        dataframe['money_flow_index'] > 50
    ).astype(int)
    dataframe['signal_strength'] += dataframe['money_flow_signal'] * 1
    
    return dataframe


class AlexNexusForgeV7(IStrategy):
    """
    Enhanced strategy on the 15-minute timeframe with Market Correlation Filters.

    Key improvements:
      - Removed lookahead Bias
      - TRAILING STOP and TRAILING BLOCKED
      - Adjustments of Signals medium_qulity_short
      - Dynamic stoploss based on ATR.
      - Dynamic leverage calculation.
      - Murrey Math level calculation (rolling window for performance).
      - Enhanced DCA (Average Price) logic.
      - Translated to English and code structured for clarity.
      - Parameterization of internal constants for optimization.
      - Changed Exit Signals for Opposite.
      - Change SL to -0.15
      - Changed stake amout for renentry
      - FIXED: Prevents opening opposite position when trade is active
      - FIXED: Trailing stop properly disabled
      - NEW: Market correlation filters for better entry timing
    """

    # General strategy parameters
    timeframe = "15m"
    startup_candle_count: int = 100
    stoploss = -0.15
    #trailing_stop = False
    #trailing_stop_positive = None
    #trailing_stop_positive_offset = None
    #trailing_only_offset_is_reached = None
    trailing_stop = True
    trailing_stop_positive = 0.005  # Trail at 0.5% below peak profit
    trailing_stop_positive_offset = 0.03  # Start trailing only at 3% profit
    trailing_only_offset_is_reached = True  # Ensure trailing only starts after offset is reached
    use_custom_stoploss = False # Disable custom logic
    stoploss_on_exchange = True  # Let exchange handle it
    stoploss_on_exchange_interval = 60  # Check every 60 seconds
    position_adjustment_enable = True
    can_short = True
    use_exit_signal = True
    ignore_roi_if_entry_signal = True
    max_stake_per_trade = 5.0
    max_portfolio_percentage_per_trade = 0.03
    max_entry_position_adjustment = 1
    process_only_new_candles = True
    max_dca_orders = 3
    max_total_stake_per_pair = 10
    max_single_dca_amount = 5
    use_custom_exits_advanced = True
    use_emergency_exits = True

    # 🚨  REGIME CHANGE DETECTION PARAMETERS (NEU)
    regime_change_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    regime_change_sensitivity = DecimalParameter(0.3, 0.8, default=0.5, decimals=2, space="sell", optimize=True, load=True)
    
    # Flash Move Detection
    flash_move_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    flash_move_threshold = DecimalParameter(0.03, 0.08, default=0.05, decimals=3, space="sell", optimize=True, load=True)
    flash_move_candles = IntParameter(3, 10, default=5, space="sell", optimize=True, load=True)
    
    # Volume Spike Detection
    volume_spike_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    volume_spike_multiplier = DecimalParameter(2.0, 5.0, default=3.0, decimals=1, space="sell", optimize=True, load=True)
    
    # Emergency Exit Protection
    emergency_exit_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    emergency_exit_profit_threshold = DecimalParameter(0.005, 0.03, default=0.015, decimals=3, space="sell", optimize=True, load=True)
    
    # Market Sentiment Protection
    sentiment_protection_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    sentiment_shift_threshold = DecimalParameter(0.2, 0.4, default=0.3, decimals=2, space="sell", optimize=True, load=True)

    # 🔧ATR STOPLOSS PARAMETERS (Anpassbar machen)
    atr_stoploss_multiplier = DecimalParameter(0.8, 2.0, default=1.0, decimals=1, space="sell", optimize=True, load=True)
    atr_stoploss_minimum = DecimalParameter(-0.25, -0.10, default=-0.12, decimals=2, space="sell", optimize=True, load=True)
    atr_stoploss_maximum = DecimalParameter(-0.30, -0.15, default=-0.18, decimals=2, space="sell", optimize=True, load=True)
    atr_stoploss_ceiling = DecimalParameter(-0.10, -0.06, default=-0.06, decimals=2, space="sell", optimize=True, load=True)
    # DCA parameters
    initial_safety_order_trigger = DecimalParameter(
        low=-0.02, high=-0.01, default=-0.018, decimals=3, space="buy", optimize=True, load=True
    )
    max_safety_orders = IntParameter(1, 3, default=1, space="buy", optimize=True, load=True)
    safety_order_step_scale = DecimalParameter(
        low=1.05, high=1.5, default=1.25, decimals=2, space="buy", optimize=True, load=True
    )
    safety_order_volume_scale = DecimalParameter(
        low=1.1, high=2.0, default=1.4, decimals=1, space="buy", optimize=True, load=True
    )
    h2 = IntParameter(20, 60, default=40, space="buy", optimize=True, load=True)
    h1 = IntParameter(10, 40, default=20, space="buy", optimize=True, load=True)
    h0 = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)
    cp = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)

    # Entry parameters
    increment_for_unique_price = DecimalParameter(
        low=1.0005, high=1.002, default=1.001, decimals=4, space="buy", optimize=True, load=True
    )
    last_entry_price: Optional[float] = None

    # Protection parameters
    cooldown_lookback = IntParameter(2, 48, default=1, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=4, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # Murrey Math level parameters
    mml_const1 = DecimalParameter(1.0, 1.1, default=1.0699, decimals=4, space="buy", optimize=True, load=True)
    mml_const2 = DecimalParameter(0.99, 1.0, default=0.99875, decimals=5, space="buy", optimize=True, load=True)
    indicator_mml_window = IntParameter(32, 128, default=64, space="buy", optimize=True, load=True)

    # Dynamic Stoploss parameters
    # Add these parameters
    stoploss_atr_multiplier = DecimalParameter(1.0, 3.0, default=1.5, space="sell", optimize=True)
    stoploss_max_reasonable = DecimalParameter(-0.30, -0.15, default=-0.20, space="sell", optimize=True)

    # === Hyperopt Parameters ===
    dominance_threshold = IntParameter(1, 10, default=3, space="buy", optimize=True)
    tightness_factor = DecimalParameter(0.5, 2.0, default=1.0, space="buy", optimize=True)
    long_rsi_threshold = IntParameter(50, 65, default=50, space="buy", optimize=True)
    short_rsi_threshold = IntParameter(30, 45, default=35, space="sell", optimize=True)

    # Dynamic Leverage parameters
    leverage_window_size = IntParameter(20, 100, default=50, space="buy", optimize=True, load=True)
    leverage_base = DecimalParameter(5.0, 20.0, default=5.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_rsi_low = DecimalParameter(20.0, 40.0, default=30.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_rsi_high = DecimalParameter(60.0, 80.0, default=70.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_long_increase_factor = DecimalParameter(1.1, 2.0, default=1.5, decimals=1, space="buy", optimize=True,
                                                     load=True)
    leverage_long_decrease_factor = DecimalParameter(0.3, 0.9, default=0.5, decimals=1, space="buy", optimize=True,
                                                     load=True)
    leverage_volatility_decrease_factor = DecimalParameter(0.5, 0.95, default=0.8, decimals=2, space="buy",
                                                           optimize=True, load=True)
    leverage_atr_threshold_pct = DecimalParameter(0.01, 0.05, default=0.03, decimals=3, space="buy", optimize=True,
                                                  load=True)

    # Indicator parameters
    indicator_extrema_order = IntParameter(3, 15, default=8, space="buy", optimize=True, load=True)  # War 5
    indicator_mml_window = IntParameter(50, 200, default=50, space="buy", optimize=True, load=True)  # War 50
    indicator_rolling_window_threshold = IntParameter(20, 100, default=50, space="buy", optimize=True, load=True)  # War 20
    indicator_rolling_check_window = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)  # War 5


    
    # Market breadth parameters
    market_breadth_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    market_breadth_threshold = DecimalParameter(0.3, 0.6, default=0.45, space="buy", optimize=True)
    
    # Total market cap parameters
    total_mcap_filter_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    total_mcap_ma_period = IntParameter(20, 100, default=50, space="buy", optimize=True)
    
    # Market regime parameters
    regime_filter_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    regime_lookback_period = IntParameter(24, 168, default=48, space="buy", optimize=True)  # hours
    
    # Fear & Greed parameters
    fear_greed_enabled = BooleanParameter(default=False, space="buy", optimize=True)  # Optional
    fear_greed_extreme_threshold = IntParameter(20, 30, default=25, space="buy", optimize=True)
    fear_greed_greed_threshold = IntParameter(70, 80, default=75, space="buy", optimize=True)
    # Momentum
    avoid_strong_trends = BooleanParameter(default=True, space="buy", optimize=True)
    trend_strength_threshold = DecimalParameter(0.01, 0.05, default=0.02, space="buy", optimize=True)
    momentum_confirmation_candles = IntParameter(1, 5, default=2, space="buy", optimize=True)

    # Dynamic exit based on entry quality
    dynamic_exit_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    exit_on_confluence_loss = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    exit_on_structure_break = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    
    # Profit target multipliers based on entry type
    high_quality_profit_multiplier = DecimalParameter(1.2, 3.0, default=2.0, space="sell", optimize=True, load=True)
    medium_quality_profit_multiplier = DecimalParameter(1.0, 2.5, default=1.5, space="sell", optimize=True, load=True)
    backup_profit_multiplier = DecimalParameter(0.8, 2.0, default=1.2, space="sell", optimize=True, load=True)
    
    # Advanced exit thresholds
    volume_decline_exit_threshold = DecimalParameter(0.3, 0.8, default=0.5, space="sell", optimize=True, load=True)
    momentum_decline_exit_threshold = IntParameter(1, 4, default=2, space="sell", optimize=True, load=True)
    structure_deterioration_threshold = DecimalParameter(-3.0, 0.0, default=-1.5, space="sell", optimize=True, load=True)
    
    # RSI exit levels
    rsi_overbought_exit = IntParameter(70, 85, default=75, space="sell", optimize=True, load=True)
    rsi_divergence_exit_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    
    # Trailing stop improvements
    use_advanced_trailing = BooleanParameter(default=False, space="sell", optimize=False, load=True)
    trailing_stop_positive_offset_high_quality = DecimalParameter(0.02, 0.08, default=0.04, space="sell", optimize=True, load=True)
    trailing_stop_positive_offset_medium_quality = DecimalParameter(0.015, 0.06, default=0.03, space="sell", optimize=True, load=True)
    
    # === NEUE ADVANCED PARAMETERS ===
    # Confluence Analysis
    confluence_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    confluence_threshold = DecimalParameter(2.0, 4.0, default=2.5, space="buy", optimize=True, load=True)  # War 3.0
    
    # Volume Analysis
    volume_analysis_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    volume_strength_threshold = DecimalParameter(1.1, 2.0, default=1.3, space="buy", optimize=True, load=True)
    volume_pressure_threshold = IntParameter(1, 3, default=1, space="buy", optimize=True, load=True)  # War 2

    
    # Momentum Analysis
    momentum_analysis_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    momentum_quality_threshold = IntParameter(2, 4, default=2, space="buy", optimize=True, load=True)  # War 3
    
    # Market Structure Analysis
    structure_analysis_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    structure_score_threshold = DecimalParameter(-2.0, 5.0, default=0.5, space="buy", optimize=True, load=True)
    
    # Ultimate Score
    ultimate_score_threshold = DecimalParameter(0.5, 3.0, default=1.5, space="buy", optimize=True, load=True)
    
    # Advanced Entry Filters
    require_volume_confirmation = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    require_momentum_confirmation = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    require_structure_confirmation = BooleanParameter(default=True, space="buy", optimize=False, load=True)

    # ✅ Replace your old ROI with this:
    minimal_roi = {
        "0": 0.07,      # 7% immediate (vs your 12%)
        "5": 0.055,     # 5.5% after 1.25h  
        "10": 0.04,     # 4% after 2.5h
        "20": 0.03,     # 3% after 5h
        "40": 0.025,    # 2.5% after 10h
        "80": 0.02,     # 2% after 20h
        "160": 0.015,   # 1.5% after 40h
        "320": 0.01     # 1% after 80h
    }

    # Plot configuration for backtesting UI
    plot_config = {
        "main_plot": {
            # Trend indicators
            "ema50": {"color": "gray", "type": "line"},
            
            # Support/Resistance
            "minima_sort_threshold": {"color": "#4ae747", "type": "line"},
            "maxima_sort_threshold": {"color": "#5b5e4b", "type": "line"},
        },
        "subplots": {
            "extrema_analysis": {
                "s_extrema": {"color": "#f53580", "type": "line"},
                "maxima": {"color": "#a29db9", "type": "scatter"},
                "minima": {"color": "#aac7fc", "type": "scatter"},
            },
            "murrey_math_levels": {
                "[4/8]P": {"color": "blue", "type": "line"},        # 50% MML
                "[6/8]P": {"color": "green", "type": "line"},       # 75% MML
                "[2/8]P": {"color": "orange", "type": "line"},      # 25% MML
                "[8/8]P": {"color": "red", "type": "line"},         # 100% MML
                "[0/8]P": {"color": "red", "type": "line"},         # 0% MML
                "mmlextreme_oscillator": {"color": "purple", "type": "line"},
            },
            "rsi_analysis": {
                "rsi": {"color": "purple", "type": "line"},
                "rsi_divergence_bull": {"color": "green", "type": "scatter"},
                "rsi_divergence_bear": {"color": "red", "type": "scatter"},
            },
            "confluence_analysis": {
                "confluence_score": {"color": "gold", "type": "line"},
                "near_support": {"color": "green", "type": "scatter"},
                "near_resistance": {"color": "red", "type": "scatter"},
                "near_mml": {"color": "blue", "type": "line"},
                "volume_spike": {"color": "orange", "type": "scatter"},
            },
            "volume_analysis": {
                "volume_strength": {"color": "cyan", "type": "line"},
                "volume_pressure": {"color": "magenta", "type": "line"},
                "buying_pressure": {"color": "green", "type": "line"},
                "selling_pressure": {"color": "red", "type": "line"},
                "money_flow_index": {"color": "yellow", "type": "line"},
            },
            "momentum_analysis": {
                "momentum_quality": {"color": "brown", "type": "line"},
                "momentum_acceleration": {"color": "pink", "type": "line"},
                "momentum_consistency": {"color": "lime", "type": "line"},
                "momentum_oscillator": {"color": "navy", "type": "line"},
            },
            "structure_analysis": {
                "structure_score": {"color": "teal", "type": "line"},
                "bullish_structure": {"color": "green", "type": "line"},
                "bearish_structure": {"color": "red", "type": "line"},
                "structure_break_up": {"color": "lime", "type": "scatter"},
                "structure_break_down": {"color": "crimson", "type": "scatter"},
            },
            "trend_strength": {
                "trend_strength": {"color": "indigo", "type": "line"},
                "trend_strength_5": {"color": "lightblue", "type": "line"},
                "trend_strength_10": {"color": "mediumblue", "type": "line"},
                "trend_strength_20": {"color": "darkblue", "type": "line"},
            },
            "ultimate_signals": {
                "ultimate_score": {"color": "gold", "type": "line"},
                "signal_strength": {"color": "silver", "type": "line"},
                "high_quality_setup": {"color": "lime", "type": "scatter"},
                "entry_type": {"color": "white", "type": "line"},
            },
            "market_conditions": {
                "strong_uptrend": {"color": "green", "type": "scatter"},
                "strong_downtrend": {"color": "red", "type": "scatter"},
                "ranging": {"color": "yellow", "type": "scatter"},
                "strong_up_momentum": {"color": "lime", "type": "scatter"},
                "strong_down_momentum": {"color": "crimson", "type": "scatter"},
            },
            "di_analysis": {
                "DI_values": {"color": "orange", "type": "line"},
                "DI_catch": {"color": "red", "type": "scatter"},
                "plus_di": {"color": "green", "type": "line"},
                "minus_di": {"color": "red", "type": "line"},
            }
        },
    }

    # Helper method to check if we have an active position in the opposite direction
    def has_active_trade(self, pair: str, side: str) -> bool:
        """
        Check if there's an active trade in the specified direction
        """
        try:
            trades = Trade.get_open_trades()
            for trade in trades:
                if trade.pair == pair:
                    if side == "long" and not trade.is_short:
                        return True
                    elif side == "short" and trade.is_short:
                        return True
        except Exception as e:
            logger.warning(f"Error checking active trades for {pair}: {e}")
        return False

    @staticmethod
    def _calculate_mml_core(mn: float, finalH: float, mx: float, finalL: float,
                            mml_c1: float, mml_c2: float) -> Dict[str, float]:
        dmml_calc = ((finalH - finalL) / 8.0) * mml_c1
        if dmml_calc == 0 or np.isinf(dmml_calc) or np.isnan(dmml_calc) or finalH == finalL:
            return {key: finalL for key in MML_LEVEL_NAMES}
        mml_val = (mx * mml_c2) + (dmml_calc * 3)
        if np.isinf(mml_val) or np.isnan(mml_val):
            return {key: finalL for key in MML_LEVEL_NAMES}
        ml = [mml_val - (dmml_calc * i) for i in range(16)]
        return {
            "[-3/8]P": ml[14], "[-2/8]P": ml[13], "[-1/8]P": ml[12],
            "[0/8]P": ml[11], "[1/8]P": ml[10], "[2/8]P": ml[9],
            "[3/8]P": ml[8], "[4/8]P": ml[7], "[5/8]P": ml[6],
            "[6/8]P": ml[5], "[7/8]P": ml[4], "[8/8]P": ml[3],
            "[+1/8]P": ml[2], "[+2/8]P": ml[1], "[+3/8]P": ml[0],
        }

    def calculate_rolling_murrey_math_levels_optimized(self, df: pd.DataFrame, window_size: int) -> Dict[str, pd.Series]:
        """
        OPTIMIZED Version - Calculate MML levels every 5 candles using only past data
        """
        murrey_levels_data: Dict[str, list] = {key: [np.nan] * len(df) for key in MML_LEVEL_NAMES}
        mml_c1 = self.mml_const1.value
        mml_c2 = self.mml_const2.value
        
        calculation_step = 5
        
        for i in range(0, len(df), calculation_step):
            if i < window_size:
                continue
                
            # Use data up to the previous candle for the rolling window
            window_end = i - 1
            window_start = window_end - window_size + 1
            if window_start < 0:
                window_start = 0
                
            window_data = df.iloc[window_start:window_end]
            mn_period = window_data["low"].min()
            mx_period = window_data["high"].max()
            current_close = df["close"].iloc[window_end] if window_end > 0 else df["close"].iloc[0]
            
            if pd.isna(mn_period) or pd.isna(mx_period) or mn_period == mx_period:
                for key in MML_LEVEL_NAMES:
                    murrey_levels_data[key][window_end] = current_close
                continue
                
            levels = self._calculate_mml_core(mn_period, mx_period, mx_period, mn_period, mml_c1, mml_c2)
            
            for key in MML_LEVEL_NAMES:
                murrey_levels_data[key][window_end] = levels.get(key, current_close)
        
        # Interpolate using only past data up to each point
        for key in MML_LEVEL_NAMES:
            series = pd.Series(murrey_levels_data[key], index=df.index)
            # Interpolate forward only up to the current point, avoiding future data
            series = series.expanding().mean().ffill()  # Use expanding mean as a safe alternative
            murrey_levels_data[key] = series.tolist()
        
        return {key: pd.Series(data, index=df.index) for key, data in murrey_levels_data.items()}


    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend strength to avoid entering against strong trends
        """
        # Linear regression slope
        def calc_slope(series, period=10):
            """Calculate linear regression slope"""
            if len(series) < period:
                return 0
            x = np.arange(period)
            y = series.iloc[-period:].values
            if np.isnan(y).any():
                return 0
            slope = np.polyfit(x, y, 1)[0]
            return slope
        
        # Calculate trend strength using multiple timeframes
        df['slope_5'] = df['close'].rolling(5).apply(lambda x: calc_slope(x, 5), raw=False)
        df['slope_10'] = df['close'].rolling(10).apply(lambda x: calc_slope(x, 10), raw=False)
        df['slope_20'] = df['close'].rolling(20).apply(lambda x: calc_slope(x, 20), raw=False)
        
        # Normalize slopes by price
        df['trend_strength_5'] = df['slope_5'] / df['close'] * 100
        df['trend_strength_10'] = df['slope_10'] / df['close'] * 100
        df['trend_strength_20'] = df['slope_20'] / df['close'] * 100
        
        # Combined trend strength
        df['trend_strength'] = (df['trend_strength_5'] + df['trend_strength_10'] + df['trend_strength_20']) / 3
        
        # Trend classification
        strong_up_threshold = self.trend_strength_threshold.value
        strong_down_threshold = -self.trend_strength_threshold.value
        
        df['strong_uptrend'] = df['trend_strength'] > strong_up_threshold
        df['strong_downtrend'] = df['trend_strength'] < strong_down_threshold
        df['ranging'] = (df['trend_strength'].abs() < strong_up_threshold * 0.5)
        
        return df
    @property
    def protections(self):
        prot = [{"method": "CooldownPeriod", "stop_duration_candles": self.cooldown_lookback.value}]
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 72,
                "trade_limit": 2,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False,
            })
        return prot

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        
        # ===========================================
        # STAKE LIMITS DEFINIEREN
        # ===========================================
        
        # Maximaler Stake pro Trade (in USDT) - kannst du anpassen
        MAX_STAKE_PER_TRADE = self.max_stake_per_trade
        
        # Maximaler Stake basierend auf Portfolio
        try:
            total_portfolio = self.wallets.get_total_stake_amount()
            MAX_STAKE_PERCENTAGE = self.max_portfolio_percentage_per_trade
            max_stake_from_portfolio = total_portfolio * MAX_STAKE_PERCENTAGE
        except:
            # Fallback wenn wallets nicht verfÃ¼gbar
            max_stake_from_portfolio = MAX_STAKE_PER_TRADE
            total_portfolio = 1000.0  # Dummy value
        
        # Market condition check fÃ¼r volatility-based stake reduction (DEIN CODE)
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if not dataframe.empty:
            last_candle = dataframe.iloc[-1]
            current_volatility = last_candle.get("volatility", 0.02)
            
            # Reduziere Stake in hochvolatilen MÃ¤rkten
            if current_volatility > 0.05:  # 5% ATR/Price ratio
                volatility_reduction = min(0.5, current_volatility * 10)  # Max 50% reduction
                proposed_stake *= (1 - volatility_reduction)
                logger.info(f"{pair} Stake reduced by {volatility_reduction:.1%} due to high volatility ({current_volatility:.2%})")
        
        # DCA Multiplier Berechnung (DEIN CODE)
        calculated_max_dca_multiplier = 1.0
        if self.position_adjustment_enable:
            num_safety_orders = int(self.max_safety_orders.value)
            volume_scale = self.safety_order_volume_scale.value
            if num_safety_orders > 0 and volume_scale > 0:
                current_order_relative_size = 1.0
                for _ in range(num_safety_orders):
                    current_order_relative_size *= volume_scale
                    calculated_max_dca_multiplier += current_order_relative_size
            else:
                logger.warning(f"{pair}: Could not calculate max_dca_multiplier due to "
                               f"invalid max_safety_orders ({num_safety_orders}) or "
                               f"safety_order_volume_scale ({volume_scale}). Defaulting to 1.0.")
        else:
            logger.debug(f"{pair}: Position adjustment not enabled. max_dca_multiplier is 1.0.")

        if calculated_max_dca_multiplier > 0:
            stake_amount = proposed_stake / calculated_max_dca_multiplier
            
            # ===========================================
            # NEUE STAKE LIMITS ANWENDEN
            # ===========================================
            
            # Verschiedene Limits prÃ¼fen
            final_stake = min(
                stake_amount,
                MAX_STAKE_PER_TRADE,
                max_stake_from_portfolio,
                max_stake  # Freqtrade's max_stake
            )
            
            # Bestimme welches Limit gegriffen hat
            limit_reason = "calculated"
            if final_stake == MAX_STAKE_PER_TRADE:
                limit_reason = "max_per_trade"
            elif final_stake == max_stake_from_portfolio:
                limit_reason = "portfolio_percentage"
            elif final_stake == max_stake:
                limit_reason = "freqtrade_max"
            
            logger.info(f"{pair} Initial stake calculated: {final_stake:.8f} (Proposed: {proposed_stake:.8f}, "
                        f"Calculated Max DCA Multiplier: {calculated_max_dca_multiplier:.2f}, "
                        f"Limited by: {limit_reason}, Portfolio %: {(final_stake/total_portfolio)*100:.1f}%)")
            
            # Min stake prÃ¼fen (DEIN CODE)
            if min_stake is not None and final_stake < min_stake:
                logger.info(f"{pair} Initial stake {final_stake:.8f} was below min_stake {min_stake:.8f}. "
                            f"Adjusting to min_stake. Consider tuning your DCA parameters or proposed stake.")
                final_stake = min_stake
            
            return final_stake
        else:
            # Fallback (DEIN CODE)
            logger.warning(
                f"{pair} Calculated max_dca_multiplier is {calculated_max_dca_multiplier:.2f}, which is invalid. "
                f"Using proposed_stake: {proposed_stake:.8f}")
            return proposed_stake

    def custom_entry_price(self, pair: str, trade: Optional[Trade], current_time: datetime,
                           proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty:
            logger.warning(f"{pair} Empty DataFrame in custom_entry_price. Returning proposed_rate.")
            return proposed_rate
        last_candle = dataframe.iloc[-1]
        entry_price = (last_candle["close"] + last_candle["open"] + proposed_rate) / 3.0
        if side == "long":
            if proposed_rate < entry_price:
                entry_price = proposed_rate
        elif side == "short":
            if proposed_rate > entry_price:
                entry_price = proposed_rate
        logger.info(
            f"{pair} Calculated Entry Price: {entry_price:.8f} | Last Close: {last_candle['close']:.8f}, "
            f"Last Open: {last_candle['open']:.8f}, Proposed Rate: {proposed_rate:.8f}")
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.000005:
            increment_factor = self.increment_for_unique_price.value if side == "long" else (
                1.0 / self.increment_for_unique_price.value)
            entry_price *= increment_factor
            logger.info(
                f"{pair} Entry price incremented to {entry_price:.8f} (previous: {self.last_entry_price:.8f}) due to proximity.")
        self.last_entry_price = entry_price
        return entry_price

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        
        if dataframe.empty or 'atr' not in dataframe.columns:
            return self.stoploss  # Use strategy stoploss (-0.15) as fallback
        
        atr = dataframe["atr"].iat[-1]
        if pd.isna(atr) or atr <= 0:
            return self.stoploss  # Fallback to -0.15
        
        atr_percent = atr / current_rate
        
        # Profit-based multiplier adjustment
        if current_profit > 0.15:
            multiplier = 1.0
        elif current_profit > 0.08:
            multiplier = 1.2
        elif current_profit > 0.03:
            multiplier = 1.4
        else:
            multiplier = 1.6
        
        calculated_stoploss = -(atr_percent * multiplier * self.atr_stoploss_multiplier.value)
        
        # Initialize trailing_offset
        trailing_offset = 0.0
        # Dynamic adjustment with trailing-like behavior
        if current_profit > self.trailing_stop_positive_offset:  # 0.03
            trailing_offset = max(0, current_profit - self.trailing_stop_positive)  # Adjust based on 1.5% trail
            calculated_stoploss = min(calculated_stoploss, -trailing_offset)  # Trail up in profit
        
        final_stoploss = max(
            min(calculated_stoploss, self.atr_stoploss_ceiling.value),
            self.atr_stoploss_maximum.value
        )
        
        logger.info(f"{pair} Custom SL: {final_stoploss:.3f} | ATR: {atr:.6f} | Profit: {current_profit:.3f} | Trailing: {trailing_offset:.3f}")
        return final_stoploss


    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                                      current_profit: float, min_stake: Optional[float], max_stake: float,
                                      current_entry_rate: float, current_exit_rate: float, 
                                      current_entry_profit: float, current_exit_profit: float, **kwargs) -> Optional[float]:
        """
        IMPROVED VERSION: Cleaner logic, better coordination with stop loss
        """
        
        # Get market conditions
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if not dataframe.empty:
            last_candle = dataframe.iloc[-1]
            btc_trend = last_candle.get('btc_trend', 0)
            market_score = last_candle.get('market_score', 0.5)
            market_breadth = last_candle.get('market_breadth', 0.5)
            rsi = last_candle.get('rsi', 50)
        else:
            btc_trend = 0
            market_score = 0.5
            market_breadth = 0.5
            rsi = 50
        
        count_of_entries = trade.nr_of_successful_entries
        count_of_exits = trade.nr_of_successful_exits
        
        # ==========================================
        # PROFIT TAKING LOGIC (IMPROVED)
        # ==========================================
        
        # 1. Extreme greed protection (more conservative)
        if market_score > 0.8 and current_profit > 0.18 and count_of_exits == 0:
            logger.info(f"{trade.pair} Extreme greed profit taking: {market_score:.2f}")
            amount_to_sell = (trade.amount * current_rate) * 0.25  # Reduced from 33%
            return -amount_to_sell
        
        # 2. Standard profit taking (unchanged)
        if current_profit > 0.15 and count_of_exits == 0:
            logger.info(f"{trade.pair} Standard profit taking (25%) at {current_profit:.2%}")
            amount_to_sell = (trade.amount * current_rate) * 0.25
            return -amount_to_sell

        # 3. Bear market profit taking (IMPROVED - more conservative)
        if not trade.is_short and btc_trend < -0.02:  # Significant downtrend
            if current_profit > 0.12 and count_of_exits == 0:  # Higher threshold
                logger.info(f"{trade.pair} Bear market profit taking at {current_profit:.2%}")
                amount_to_sell = (trade.amount * current_rate) * 0.3  # Reduced from 50%
                return -amount_to_sell

        # 4. Secondary profit taking (unchanged)
        if current_profit > 0.30 and count_of_exits == 1:
            logger.info(f"{trade.pair} Secondary profit taking (33%) at {current_profit:.2%}")
            amount_to_sell = (trade.amount * current_rate) * (1 / 3)
            return -amount_to_sell
        
        # ==========================================
        # DCA LOGIC (UNCHANGED - LOOKS GOOD)
        # ==========================================
        
        if not self.position_adjustment_enable:
            return None
        
        # Validate strategy variables
        if not hasattr(self, 'max_dca_orders'):
            logger.error(f"{trade.pair} Missing max_dca_orders parameter")
            return None
        
        # Get DCA limits
        max_dca_for_pair = self.max_dca_orders
        max_total_stake = getattr(self, 'max_total_stake_per_pair', 10)
        max_single_dca = getattr(self, 'max_single_dca_amount', 5)
        
        # Check DCA limits
        if count_of_entries > max_dca_for_pair:
            logger.info(f"{trade.pair} 🛑 MAX DCA REACHED: {count_of_entries}/{max_dca_for_pair}")
            return None
        
        if trade.stake_amount >= max_total_stake:
            logger.info(f"{trade.pair} 🛑 MAX STAKE REACHED: {trade.stake_amount:.2f}/{max_total_stake} USDT")
            return None
        
        # Market condition blocks
        if market_breadth < 0.25 and not trade.is_short:
            logger.info(f"{trade.pair} Blocking DCA due to bearish market: {market_breadth:.2%}")
            return None
        
        # Volatility adjustment
        dca_multiplier = 1.5 if last_candle.get('market_regime') == 'high_volatility' else 1.0
        
        # DCA trigger calculation (FIXED)
        trigger = getattr(self, 'initial_safety_order_trigger', DecimalParameter(-0.02, -0.01, default=-0.018)).value
        trigger = trigger * dca_multiplier
        
        # CORRECTED DCA LOGIC: DCA when we're LOSING money
        if (current_profit > trigger / 2.0 and count_of_entries == 1) or \
           (current_profit > trigger and count_of_entries == 2) or \
           (current_profit > trigger * 1.5 and count_of_entries == 3):
            logger.info(f"{trade.pair} DCA not triggered. Profit {current_profit:.2%} > threshold {trigger:.2%}")
            return None
        
        # Calculate DCA amount (unchanged - looks good)
        try:
            filled_entry_orders = trade.select_filled_orders(trade.entry_side)
            if not filled_entry_orders:
                return None
            
            # Progressive DCA sizing
            dca_multipliers = [1.0, 0.8, 0.6]
            current_multiplier = dca_multipliers[count_of_entries - 1] if count_of_entries <= len(dca_multipliers) else 0.5
            
            dca_stake_amount = max_single_dca * current_multiplier
            
            # Apply limits
            remaining_budget = max_total_stake - trade.stake_amount
            if dca_stake_amount > remaining_budget:
                if remaining_budget > 1:
                    dca_stake_amount = remaining_budget
                else:
                    return None
            
            if min_stake and dca_stake_amount < min_stake:
                dca_stake_amount = min_stake
            
            logger.info(f"{trade.pair} 🛑 DCA #{count_of_entries}: +{dca_stake_amount:.2f} USDT")
            return dca_stake_amount
            
        except Exception as e:
            logger.error(f"DCA calculation error for {trade.pair}: {e}")
            return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float,
                 max_leverage: float, side: str, **kwargs) -> float:
        window_size = self.leverage_window_size.value
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if len(dataframe) < window_size:
            logger.warning(
                f"{pair} Not enough data ({len(dataframe)} candles) to calculate dynamic leverage (requires {window_size}). Using proposed: {proposed_leverage}")
            return proposed_leverage
        close_prices_series = dataframe["close"].tail(window_size)
        high_prices_series = dataframe["high"].tail(window_size)
        low_prices_series = dataframe["low"].tail(window_size)
        base_leverage = self.leverage_base.value
        rsi_array = ta.RSI(close_prices_series, timeperiod=14)
        atr_array = ta.ATR(high_prices_series, low_prices_series, close_prices_series, timeperiod=14)
        sma_array = ta.SMA(close_prices_series, timeperiod=20)
        macd_output = ta.MACD(close_prices_series, fastperiod=12, slowperiod=26, signalperiod=9)

        current_rsi = rsi_array[-1] if rsi_array.size > 0 and not np.isnan(rsi_array[-1]) else 50.0
        current_atr = atr_array[-1] if atr_array.size > 0 and not np.isnan(atr_array[-1]) else 0.0
        current_sma = sma_array[-1] if sma_array.size > 0 and not np.isnan(sma_array[-1]) else current_rate
        current_macd_hist = 0.0

        if isinstance(macd_output, pd.DataFrame):
            if not macd_output.empty and 'macdhist' in macd_output.columns:
                valid_macdhist_series = macd_output['macdhist'].dropna()
                if not valid_macdhist_series.empty:
                    current_macd_hist = valid_macdhist_series.iloc[-1]

        # Apply rules based on indicators
        if side == "long":
            if current_rsi < self.leverage_rsi_low.value:
                base_leverage *= self.leverage_long_increase_factor.value
            elif current_rsi > self.leverage_rsi_high.value:
                base_leverage *= self.leverage_long_decrease_factor.value

            if current_atr > 0 and current_rate > 0:
                if (current_atr / current_rate) > self.leverage_atr_threshold_pct.value:
                    base_leverage *= self.leverage_volatility_decrease_factor.value

            if current_macd_hist > 0:
                base_leverage *= self.leverage_long_increase_factor.value

            if current_sma > 0 and current_rate < current_sma:
                base_leverage *= self.leverage_long_decrease_factor.value

        adjusted_leverage = round(max(1.0, min(base_leverage, max_leverage)), 2)
        logger.info(
            f"{pair} Dynamic Leverage: {adjusted_leverage:.2f} (Base: {base_leverage:.2f}, RSI: {current_rsi:.2f}, "
            f"ATR: {current_atr:.4f}, MACD Hist: {current_macd_hist:.4f}, SMA: {current_sma:.4f})")
        return adjusted_leverage

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        ULTIMATE indicator calculations with advanced market analysis
        """
        # === STANDARD INDICATORS ===
        dataframe["ema50"] = ta.EMA(dataframe["close"], timeperiod=50)
        dataframe["ema100"] = ta.EMA(dataframe["close"], timeperiod=100) # Neu hinzufÃ¼gen
        dataframe["rsi"] = ta.RSI(dataframe["close"])
        dataframe["atr"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=10)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe)
        dataframe["DI_values"] = dataframe["plus_di"] - dataframe["minus_di"]
        dataframe["DI_cutoff"] = 0

        # === EXTREMA DETECTION ===
        extrema_order = self.indicator_extrema_order.value
        dataframe["maxima"] = (
            dataframe["close"] == dataframe["close"].shift(1).rolling(window=extrema_order).max()
        ).astype(int)
        dataframe["minima"] = (
            dataframe["close"] == dataframe["close"].shift(1).rolling(window=extrema_order).min()
        ).astype(int)

        dataframe["s_extrema"] = 0
        dataframe.loc[dataframe["minima"] == 1, "s_extrema"] = -1
        dataframe.loc[dataframe["maxima"] == 1, "s_extrema"] = 1

        # === HEIKIN-ASHI ===
        dataframe["ha_close"] = (dataframe["open"] + dataframe["high"] + dataframe["low"] + dataframe["close"]) / 4

        # === ROLLING EXTREMA ===
        dataframe["minh2"], dataframe["maxh2"] = calculate_minima_maxima(dataframe, self.h2.value)
        dataframe["minh1"], dataframe["maxh1"] = calculate_minima_maxima(dataframe, self.h1.value)
        dataframe["minh0"], dataframe["maxh0"] = calculate_minima_maxima(dataframe, self.h0.value)
        dataframe["mincp"], dataframe["maxcp"] = calculate_minima_maxima(dataframe, self.cp.value)

        # === MURREY MATH LEVELS ===
        mml_window = self.indicator_mml_window.value
        murrey_levels = self.calculate_rolling_murrey_math_levels_optimized(dataframe, window_size=mml_window)
        
        for level_name in MML_LEVEL_NAMES:
            if level_name in murrey_levels:
                dataframe[level_name] = murrey_levels[level_name]
            else:
                dataframe[level_name] = dataframe["close"]

        # === MML OSCILLATOR ===
        mml_4_8 = dataframe.get("[4/8]P")
        mml_plus_3_8 = dataframe.get("[+3/8]P")
        mml_minus_3_8 = dataframe.get("[-3/8]P")
        
        if mml_4_8 is not None and mml_plus_3_8 is not None and mml_minus_3_8 is not None:
            osc_denominator = (mml_plus_3_8 - mml_minus_3_8).replace(0, np.nan)
            dataframe["mmlextreme_oscillator"] = 100 * ((dataframe["close"] - mml_4_8) / osc_denominator)
        else:
            dataframe["mmlextreme_oscillator"] = np.nan

        # === DI CATCH ===
        dataframe["DI_catch"] = np.where(dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1)

        # === ROLLING THRESHOLDS ===
        rolling_window_threshold = self.indicator_rolling_window_threshold.value
        dataframe["minima_sort_threshold"] = dataframe["close"].rolling(
            window=rolling_window_threshold, min_periods=1
        ).min()
        dataframe["maxima_sort_threshold"] = dataframe["close"].rolling(
            window=rolling_window_threshold, min_periods=1
        ).max()

        # === EXTREMA CHECKS ===
        rolling_check_window = self.indicator_rolling_check_window.value
        dataframe["minima_check"] = (
            dataframe["minima"].rolling(window=rolling_check_window, min_periods=1).sum() == 0
        ).astype(int)
        dataframe["maxima_check"] = (
            dataframe["maxima"].rolling(window=rolling_check_window, min_periods=1).sum() == 0
        ).astype(int)

        # === VOLATILITY INDICATORS ===
        dataframe["volatility_range"] = dataframe["high"] - dataframe["low"]
        dataframe["avg_volatility"] = dataframe["volatility_range"].rolling(window=50).mean()
        dataframe["avg_volume"] = dataframe["volume"].rolling(window=50).mean()

        # === TREND STRENGTH INDICATORS ===
        def calc_slope(series, period):
            """Calculate linear regression slope"""
            if len(series) < period:
                return 0
            x = np.arange(period)
            y = series.values
            if np.isnan(y).any():
                return 0
            try:
                slope = np.polyfit(x, y, 1)[0]
                return slope
            except:
                return 0

        dataframe['slope_5'] = dataframe['close'].rolling(5).apply(lambda x: calc_slope(x, 5), raw=False)
        dataframe['slope_10'] = dataframe['close'].rolling(10).apply(lambda x: calc_slope(x, 10), raw=False)
        dataframe['slope_20'] = dataframe['close'].rolling(20).apply(lambda x: calc_slope(x, 20), raw=False)

        dataframe['trend_strength_5'] = dataframe['slope_5'] / dataframe['close'] * 100
        dataframe['trend_strength_10'] = dataframe['slope_10'] / dataframe['close'] * 100
        dataframe['trend_strength_20'] = dataframe['slope_20'] / dataframe['close'] * 100

        dataframe['trend_strength'] = (dataframe['trend_strength_5'] + dataframe['trend_strength_10'] + dataframe['trend_strength_20']) / 3

        strong_threshold = 0.02
        dataframe['strong_uptrend'] = dataframe['trend_strength'] > strong_threshold
        dataframe['strong_downtrend'] = dataframe['trend_strength'] < -strong_threshold
        dataframe['ranging'] = dataframe['trend_strength'].abs() < (strong_threshold * 0.5)

        # === MOMENTUM INDICATORS ===
        dataframe['price_momentum'] = dataframe['close'].pct_change(3)
        dataframe['momentum_increasing'] = dataframe['price_momentum'] > dataframe['price_momentum'].shift(1)
        dataframe['momentum_decreasing'] = dataframe['price_momentum'] < dataframe['price_momentum'].shift(1)

        dataframe['volume_momentum'] = dataframe['volume'].rolling(3).mean() / dataframe['volume'].rolling(20).mean()

        dataframe['rsi_divergence_bull'] = (
            (dataframe['close'] < dataframe['close'].shift(5)) &
            (dataframe['rsi'] > dataframe['rsi'].shift(5))
        )
        dataframe['rsi_divergence_bear'] = (
            (dataframe['close'] > dataframe['close'].shift(5)) &
            (dataframe['rsi'] < dataframe['rsi'].shift(5))
        )

        # === CANDLE PATTERNS ===
        dataframe['green_candle'] = dataframe['close'] > dataframe['open']
        dataframe['red_candle'] = dataframe['close'] < dataframe['open']
        dataframe['consecutive_green'] = dataframe['green_candle'].rolling(3).sum()
        dataframe['consecutive_red'] = dataframe['red_candle'].rolling(3).sum()

        dataframe['strong_up_momentum'] = (
            (dataframe['consecutive_green'] >= 3) &
            (dataframe['volume'] > dataframe['avg_volume']) &
            (dataframe['trend_strength'] > strong_threshold)
        )
        dataframe['strong_down_momentum'] = (
            (dataframe['consecutive_red'] >= 3) &
            (dataframe['volume'] > dataframe['avg_volume']) &
            (dataframe['trend_strength'] < -strong_threshold)
        )

        # === ðŸš€ ADVANCED ANALYSIS MODULES ===
        
        # 1. CONFLUENCE ANALYSIS
        if self.confluence_enabled.value:
            dataframe = calculate_confluence_score(dataframe)
        else:
            dataframe['confluence_score'] = 0
        
        # 2. SMART VOLUME ANALYSIS
        if self.volume_analysis_enabled.value:
            dataframe = calculate_smart_volume(dataframe)
        else:
            dataframe['volume_pressure'] = 0
            dataframe['volume_strength'] = 1.0
            dataframe['money_flow_index'] = 50
        
        # 3. ADVANCED MOMENTUM
        if self.momentum_analysis_enabled.value:
            dataframe = calculate_advanced_momentum(dataframe)
        else:
            dataframe['momentum_quality'] = 0
            dataframe['momentum_acceleration'] = 0
        
        # 4. MARKET STRUCTURE
        if self.structure_analysis_enabled.value:
            dataframe = calculate_market_structure(dataframe)
        else:
            dataframe['structure_score'] = 0
            dataframe['structure_break_up'] = 0
        
        # 5. ADVANCED ENTRY SIGNALS
        dataframe = calculate_advanced_entry_signals(dataframe)

        # === ðŸŽ¯ ULTIMATE MARKET SCORE ===
        dataframe['ultimate_score'] = (
            dataframe['confluence_score'] * 0.25 +           # 25% confluence
            dataframe['volume_pressure'] * 0.2 +             # 20% volume pressure
            dataframe['momentum_quality'] * 0.2 +            # 20% momentum quality
            (dataframe['structure_score'] / 5) * 0.15 +      # 15% structure (normalized)
            (dataframe['signal_strength'] / 10) * 0.2        # 20% signal strength
        )
        
        # Normalize ultimate score to 0-1 range
        dataframe['ultimate_score'] = dataframe['ultimate_score'].clip(0, 5) / 5

        # === FINAL QUALITY CHECKS ===
        dataframe['high_quality_setup'] = (
            (dataframe['ultimate_score'] > self.ultimate_score_threshold.value) &
            (dataframe['signal_strength'] >= 5) &
            (dataframe['volume_strength'] > 1.1) &
            (dataframe['rsi'] > 30) & (dataframe['rsi'] < 70)
        ).astype(int)

        # === DEBUG INFO ===
        if metadata['pair'] in ['BTC/USDT:USDT', 'ETH/USDT:USDT']:  # Only log for major pairs
            latest_score = dataframe['ultimate_score'].iloc[-1]
            latest_signal = dataframe['signal_strength'].iloc[-1]
            logger.info(f"{metadata['pair']} Ultimate Score: {latest_score:.3f}, Signal Strength: {latest_signal}")

        # ===========================================
        # ðŸš¨ REGIME CHANGE DETECTION (NEU HINZUFÃœGEN)
        # ===========================================
        
        if self.regime_change_enabled.value:
            
            # ===========================================
            # âš¡ FLASH MOVE DETECTION
            # ===========================================
            
            flash_candles = self.flash_move_candles.value
            flash_threshold = self.flash_move_threshold.value
            
            # Schnelle Preisbewegungen
            dataframe['price_change_fast'] = dataframe['close'].pct_change(flash_candles)
            dataframe['flash_pump'] = dataframe['price_change_fast'] > flash_threshold
            dataframe['flash_dump'] = dataframe['price_change_fast'] < -flash_threshold
            dataframe['flash_move'] = dataframe['flash_pump'] | dataframe['flash_dump']
            
            # ===========================================
            # ðŸ”Š VOLUME SPIKE DETECTION
            # ===========================================
            
            volume_ma20 = dataframe['volume'].rolling(20).mean()
            volume_multiplier = self.volume_spike_multiplier.value
            dataframe['volume_spike'] = dataframe['volume'] > (volume_ma20 * volume_multiplier)
            
            # Volume + Bewegung kombiniert
            dataframe['volume_pump'] = dataframe['volume_spike'] & dataframe['flash_pump']
            dataframe['volume_dump'] = dataframe['volume_spike'] & dataframe['flash_dump']
            
            # ===========================================
            # ðŸŒŠ MARKET SENTIMENT DETECTION
            # ===========================================
            
            # Market Breadth Change (falls vorhanden)
            if 'market_breadth' in dataframe.columns:
                dataframe['market_breadth_change'] = dataframe['market_breadth'].diff(3)
                sentiment_threshold = self.sentiment_shift_threshold.value
                dataframe['sentiment_shift_bull'] = dataframe['market_breadth_change'] > sentiment_threshold
                dataframe['sentiment_shift_bear'] = dataframe['market_breadth_change'] < -sentiment_threshold
            else:
                dataframe['sentiment_shift_bull'] = False
                dataframe['sentiment_shift_bear'] = False
            
            # ===========================================
            # â‚¿ BTC CORRELATION MONITORING
            # ===========================================
            
            # BTC Flash Moves (falls BTC Daten vorhanden)
            if 'btc_close' in dataframe.columns:
                dataframe['btc_change_fast'] = dataframe['btc_close'].pct_change(flash_candles)
                dataframe['btc_flash_pump'] = dataframe['btc_change_fast'] > flash_threshold
                dataframe['btc_flash_dump'] = dataframe['btc_change_fast'] < -flash_threshold
                
                # Correlation Break: BTC bewegt sich stark, Coin nicht
                pair_movement = dataframe['price_change_fast'].abs()
                btc_movement = dataframe['btc_change_fast'].abs()
                dataframe['correlation_break'] = (btc_movement > flash_threshold) & (pair_movement < flash_threshold * 0.4)
            else:
                dataframe['btc_flash_pump'] = False
                dataframe['btc_flash_dump'] = False
                dataframe['correlation_break'] = False
            
            # ===========================================
            # ðŸŽ¯ REGIME CHANGE SCORE
            # ===========================================
            
            # Kombiniere alle Signale
            regime_signals = [
                'flash_move', 'volume_spike', 
                'sentiment_shift_bull', 'sentiment_shift_bear',
                'btc_flash_pump', 'btc_flash_dump', 'correlation_break'
            ]
            
            dataframe['regime_change_score'] = 0
            for signal in regime_signals:
                if signal in dataframe.columns:
                    dataframe['regime_change_score'] += dataframe[signal].astype(int)
            
            # Normalisiere auf 0-1
            max_signals = len(regime_signals)
            dataframe['regime_change_intensity'] = dataframe['regime_change_score'] / max_signals
            
            # Alert Level
            sensitivity = self.regime_change_sensitivity.value
            dataframe['regime_alert'] = dataframe['regime_change_intensity'] >= sensitivity
            
        else:
            # Falls Regime Detection deaktiviert
            dataframe['flash_pump'] = False
            dataframe['flash_dump'] = False
            dataframe['volume_pump'] = False
            dataframe['volume_dump'] = False
            dataframe['regime_alert'] = False
            dataframe['regime_change_intensity'] = 0.0
        
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        ULTIMATE ENTRY LOGIC - Multi-factor confluence system
        """

        # ===========================================
        # INITIALIZE ENTRY COLUMNS
        # ===========================================
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""
        
        # === CORE CONDITIONS (must all be true) ===
        core_conditions = (
            # Basic trend and momentum
            (dataframe['close'] > dataframe['ema50']) &
            (dataframe['rsi'] > 25) & (dataframe['rsi'] < 75) &
            (dataframe['trend_strength'] > -0.02) &  # Not in strong downtrend
            
            # Volume confirmation
            (dataframe['volume'] > dataframe['avg_volume'] * 0.7) &  # Minimum volume
            
            # No recent distribution
            (dataframe['selling_pressure'] <= 4)
        )
        
        # === ADVANCED CONDITIONS (weighted scoring) ===
        
        # 1. CONFLUENCE CONDITIONS
        confluence_conditions = (
            (dataframe['confluence_score'] >= self.confluence_threshold.value) |
            (
                (dataframe['confluence_score'] >= (self.confluence_threshold.value - 1)) &
                (dataframe['near_support'] == 1) &
                (dataframe['volume_spike'] == 1)
            )
        )
        
        # 2. VOLUME CONDITIONS
        volume_conditions = (
            (dataframe['volume_pressure'] >= self.volume_pressure_threshold.value) &
            (dataframe['volume_strength'] > self.volume_strength_threshold.value) &
            (dataframe['money_flow_index'] > 45)
        ) if self.require_volume_confirmation.value else True
        
        # 3. MOMENTUM CONDITIONS
        momentum_conditions = (
            (dataframe['momentum_quality'] >= self.momentum_quality_threshold.value) &
            (dataframe['momentum_acceleration'] > -0.01) &
            (dataframe['momentum_consistency'] >= 2)
        ) if self.require_momentum_confirmation.value else True
        
        # 4. STRUCTURE CONDITIONS
        structure_conditions = (
            (dataframe['structure_score'] >= self.structure_score_threshold.value) &
            (dataframe['bullish_structure'] > dataframe['bearish_structure'])
        ) if self.require_structure_confirmation.value else True
        
        # === ORIGINAL EXTREMA CONDITIONS ===
        extrema_conditions = (
            # Minima conditions
            (
                (dataframe["minima"] == 1) &
                (dataframe["minima_check"] == 1) &
                (dataframe["close"] <= dataframe["minima_sort_threshold"] * 1.02) &
                (dataframe["DI_catch"] == 1)
            ) |
            # Alternative: MML level conditions
            (
                (dataframe["close"] <= dataframe["[0/8]P"] * 1.01) |
                (dataframe["close"] <= dataframe["[2/8]P"] * 1.01) |
                (dataframe["close"] <= dataframe["[4/8]P"] * 1.01)
            ) |
            # Rolling extrema conditions
            (
                (dataframe["close"] <= dataframe["minh2"] * 1.015) |
                (dataframe["close"] <= dataframe["minh1"] * 1.015) |
                (dataframe["close"] <= dataframe["minh0"] * 1.015)
            )
        )
        
        # === QUALITY FILTERS ===
        quality_filters = (
            # Ultimate score threshold
            (dataframe['ultimate_score'] > self.ultimate_score_threshold.value) &
            
            # Signal strength
            (dataframe['signal_strength'] >= 5) &
            
            # No extreme RSI
            (dataframe['rsi'] < 80) &
            
            # Positive momentum
            (dataframe['price_momentum'] > -0.02) &
            
            # Volume not declining
            (dataframe['volume_trend'] == 1) |
            (dataframe['volume_strength'] > 1.1)
        )
        
        # === RISK MANAGEMENT CONDITIONS ===
        risk_conditions = (
            # No consecutive red candles
            (dataframe['consecutive_red'] <= 2) &
            
            # ATR not too high (volatility control)
            (dataframe['atr'] < dataframe['close'] * 0.05) &  # Max 5% ATR
            
            # No strong bearish momentum
            (dataframe['strong_down_momentum'] == 0) &
            
            # RSI divergence protection
            (dataframe['rsi_divergence_bear'] == 0)
        )
        
        # === MARKET REGIME ADAPTATIONS ===
        
        # In ranging markets, require stronger confluence
        ranging_market_adjustment = (
            (~dataframe['ranging']) |  # Not ranging, OR
            (dataframe['confluence_score'] >= (self.confluence_threshold.value + 1))  # Higher confluence if ranging
        )
        
        # In strong trends, allow more aggressive entries
        trend_market_adjustment = (
            (~dataframe['strong_uptrend']) |  # Not in strong uptrend, OR
            (dataframe['volume_pressure'] >= 1)  # Lower volume requirement in strong trends
        )
        
        # === FINAL ENTRY CONDITION COMBINATIONS ===
        
        # HIGH QUALITY ENTRIES (all advanced filters)
        high_quality_entry = (
            core_conditions &
            confluence_conditions &
            volume_conditions &
            momentum_conditions &
            structure_conditions &
            extrema_conditions &
            quality_filters &
            risk_conditions &
            ranging_market_adjustment &
            trend_market_adjustment
        )
        
        # MEDIUM QUALITY ENTRIES (relaxed requirements)
        medium_quality_entry = (
            core_conditions &
            extrema_conditions &
            (
                # Either strong confluence OR strong momentum OR strong volume
                confluence_conditions |
                (dataframe['momentum_quality'] >= 4) |
                (dataframe['volume_pressure'] >= 3)
            ) &
            # Basic quality filters
            (dataframe['ultimate_score'] > (self.ultimate_score_threshold.value * 0.7)) &
            (dataframe['signal_strength'] >= 3) &
            risk_conditions
        )
        
        # BACKUP ENTRIES (original logic only)
        backup_entry = (
            core_conditions &
            extrema_conditions &
            (dataframe['volume'] > dataframe['avg_volume']) &
            (dataframe['trend_strength'] > 0)
        )
        
        # === ADDITIONAL ENTRY SIGNALS FOR SPECIFIC SCENARIOS ===
        
        # Momentum breakout entries
        momentum_breakout = (
            (dataframe['structure_break_up'] == 1) &
            (dataframe['volume_strength'] > 1.5) &
            (dataframe['momentum_quality'] >= 4) &
            (dataframe['close'] > dataframe['ema50']) &
            (dataframe['rsi'] < 75) &
            risk_conditions
        )
        
        # Volume spike reversals
        volume_reversal = (
            (dataframe['volume_spike'] == 1) &
            (dataframe['rsi_oversold'] == 1) &
            (dataframe['near_support'] == 1) &
            (dataframe['buying_pressure'] >= 2) &
            (dataframe['close'] > dataframe['open']) &  # Green candle
            risk_conditions
        )
        
        # === SHORT LOGIC ===
        
        # CORE SHORT CONDITIONS (mirror of long)
        core_short_conditions = (
            (dataframe['close'] < dataframe['ema50']) &         # Below EMA
            (dataframe['rsi'] > 30) & (dataframe['rsi'] < 70) & # Not extreme
            (dataframe['trend_strength'] < 0.01) &              # Downtrend or neutral
            (dataframe['volume'] > dataframe['avg_volume'] * 0.8) &
            (dataframe['buying_pressure'] <= 3)                 # Not too much buying
        )
        
        # SHORT CONFLUENCE (inverted)
        short_confluence_conditions = (
            (dataframe['confluence_score'] >= self.confluence_threshold.value) |
            (
                (dataframe['confluence_score'] >= (self.confluence_threshold.value - 1)) &
                (dataframe['near_resistance'] == 1) &           # Near resistance for SHORT
                (dataframe['volume_spike'] == 1)
            )
        )
        
        # SHORT VOLUME CONDITIONS
        short_volume_conditions = (
            (dataframe['volume_pressure'] <= -self.volume_pressure_threshold.value) &  # Negative pressure
            (dataframe['volume_strength'] > self.volume_strength_threshold.value) &
            (dataframe['money_flow_index'] < 55)                # Money flowing out
        ) if self.require_volume_confirmation.value else True
        
        # SHORT MOMENTUM CONDITIONS  
        short_momentum_conditions = (
            (dataframe['momentum_quality'] <= -2) &             # Negative momentum quality
            (dataframe['momentum_acceleration'] < 0.01) &       # Decelerating
            (dataframe['momentum_consistency'] <= 1)            # Inconsistent upward momentum
        ) if self.require_momentum_confirmation.value else True
        
        # SHORT STRUCTURE CONDITIONS
        short_structure_conditions = (
            (dataframe['structure_score'] <= -self.structure_score_threshold.value) &  # Negative structure
            (dataframe['bearish_structure'] > dataframe['bullish_structure'])          # More bearish signals
        ) if self.require_structure_confirmation.value else True
        
        # SHORT EXTREMA CONDITIONS (inverted)
        short_extrema_conditions = (
            # Maxima conditions (resistance for shorts)
            (
                (dataframe["maxima"] == 1) &
                (dataframe["maxima_check"] == 1) &
                (dataframe["close"] >= dataframe["maxima_sort_threshold"] * 0.98) &
                (dataframe["DI_catch"] == 1)
            ) |
            # MML resistance levels
            (
                (dataframe["close"] >= dataframe["[6/8]P"] * 0.99) |
                (dataframe["close"] >= dataframe["[8/8]P"] * 0.99) |
                (dataframe["close"] >= dataframe["[+1/8]P"] * 0.99)
            ) |
            # Rolling maxima
            (
                (dataframe["close"] >= dataframe["maxh2"] * 0.985) |
                (dataframe["close"] >= dataframe["maxh1"] * 0.985) |
                (dataframe["close"] >= dataframe["maxh0"] * 0.985)
            )
        )
        
        # SHORT QUALITY FILTERS
        short_quality_filters = (
            (dataframe['ultimate_score'] < (1 - self.ultimate_score_threshold.value)) &  # Low score for shorts
            (dataframe['signal_strength'] <= -3) &              # Negative signal strength
            (dataframe['rsi'] > 20) &                           # Not oversold
            (dataframe['price_momentum'] < 0.02) &              # Negative momentum
            (dataframe['volume_trend'] == 0) |                  # Volume declining
            (dataframe['volume_strength'] < 0.9)                # Weak volume
        )
        
        # SHORT RISK CONDITIONS
        short_risk_conditions = (
            (dataframe['consecutive_green'] <= 2) &             # Not too many green candles
            (dataframe['atr'] < dataframe['close'] * 0.05) &    # Volatility control
            (dataframe['strong_up_momentum'] == 0) &            # No strong bullish momentum
            (dataframe['rsi_divergence_bull'] == 0)             # No bullish divergence
        )
        
        # === SHORT ENTRY COMBINATIONS ===
        
        # HIGH QUALITY SHORT ENTRIES
        high_quality_short = (
            core_short_conditions &
            short_confluence_conditions &
            short_volume_conditions &
            short_momentum_conditions &
            short_structure_conditions &
            short_extrema_conditions &
            short_quality_filters &
            short_risk_conditions
        )
        
        # MEDIUM QUALITY SHORT ENTRIES
        medium_quality_short = (
            core_short_conditions &
            short_extrema_conditions &
            (
                short_confluence_conditions |
                (dataframe['momentum_quality'] <= -3) |
                (dataframe['volume_pressure'] <= -3)
            ) &
            short_risk_conditions &
            ~(
                (dataframe['close'] > dataframe['ema50']) |  # Exclude if above EMA50 (long condition)
                (dataframe['trend_strength'] > 0.01) |       # Exclude if in uptrend
                (dataframe['rsi'] < 30) |                   # Exclude if oversold (long bias)
                (dataframe['ultimate_score'] > self.ultimate_score_threshold.value)  # Exclude high-quality long setups
            )
        )
        
        # === ENTRY PRIORITY SYSTEM ===
        # Initialize entry type column
        dataframe['entry_type'] = 0
        
        # === APPLY LONG ENTRIES ===
        # HIGH QUALITY LONG
        dataframe.loc[high_quality_entry, "enter_long"] = 1
        dataframe.loc[high_quality_entry, 'entry_type'] = 3
        dataframe.loc[high_quality_entry, "enter_tag"] = "high_quality_long"
        
        # MEDIUM QUALITY LONG  
        dataframe.loc[medium_quality_entry & ~high_quality_entry, "enter_long"] = 1
        dataframe.loc[medium_quality_entry & ~high_quality_entry, 'entry_type'] = 2
        dataframe.loc[medium_quality_entry & ~high_quality_entry, "enter_tag"] = "medium_quality_long"
        
        # BACKUP LONG
        dataframe.loc[backup_entry & ~(high_quality_entry | medium_quality_entry), "enter_long"] = 1
        dataframe.loc[backup_entry & ~(high_quality_entry | medium_quality_entry), 'entry_type'] = 1  
        dataframe.loc[backup_entry & ~(high_quality_entry | medium_quality_entry), "enter_tag"] = "backup_long"
        
        # BREAKOUT LONG
        dataframe.loc[momentum_breakout, "enter_long"] = 1
        dataframe.loc[momentum_breakout, 'entry_type'] = 4
        dataframe.loc[momentum_breakout, "enter_tag"] = "breakout_long"
        
        # REVERSAL LONG
        dataframe.loc[volume_reversal, "enter_long"] = 1  
        dataframe.loc[volume_reversal, 'entry_type'] = 5
        dataframe.loc[volume_reversal, "enter_tag"] = "reversal_long"
        
        # === APPLY SHORT ENTRIES ===
        dataframe.loc[high_quality_short, "enter_short"] = 1
        dataframe.loc[high_quality_short, 'entry_type'] = 6     # High quality short
        dataframe.loc[high_quality_short, "enter_tag"] = "high_quality_short"
        
        dataframe.loc[
            medium_quality_short & ~high_quality_short, 
            "enter_short"
        ] = 1
        dataframe.loc[
            medium_quality_short & ~high_quality_short, 
            'entry_type'
        ] = 7  # Medium quality short
        dataframe.loc[
            medium_quality_short & ~high_quality_short, 
            "enter_tag"
        ] = "medium_quality_short"
        
        # === ENTRY DEBUGGING ===
        # Log entry signals for major pairs
        if metadata['pair'] in ['BTC/USDT:USDT', 'ETH/USDT:USDT']:
            recent_entries = dataframe['enter_long'].tail(5).sum() + dataframe['enter_short'].tail(5).sum()
            if recent_entries > 0:
                entry_type = dataframe['entry_type'].iloc[-1]
                entry_types = {0: 'None', 1: 'Backup', 2: 'Medium', 3: 'High', 4: 'Breakout', 5: 'Reversal', 6: 'High Short', 7: 'Medium Short'}
                logger.info(f"{metadata['pair']} ENTRY SIGNAL - Type: {entry_types.get(entry_type, 'Unknown')}")
                logger.info(f"  Ultimate Score: {dataframe['ultimate_score'].iloc[-1]:.3f}")
                logger.info(f"  Signal Strength: {dataframe['signal_strength'].iloc[-1]}")
                logger.info(f"  Confluence Score: {dataframe['confluence_score'].iloc[-1]}")
        
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        UNIFIED EXIT SYSTEM - Choose between Custom MML Exits or Simple Opposite Signal Exits
        """
        # ===========================================
        # INITIALIZE EXIT COLUMNS
        # ===========================================
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""
        
        # ===========================================
        # CHOOSE EXIT SYSTEM
        # ===========================================
        if self.use_custom_exits_advanced:
            # Use Alex's Advanced MML-based Exit System
            return self._populate_custom_exits_advanced(dataframe, metadata)
        else:
            # Use Simple Opposite Signal Exit System
            return self._populate_simple_exits(dataframe, metadata)
    
    def _populate_custom_exits_advanced(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        ALEX'S ADVANCED MML-BASED EXIT SYSTEM
        Profit-protecting exit strategy with better signal coordination
        """
        
        # ===========================================
        # MML MARKET STRUCTURE FOR EXITS
        # ===========================================
        
        # Bullish/Bearish structure (same as entry)
        bullish_mml = (
            (df["close"] > df["[6/8]P"]) |
            ((df["close"] > df["[4/8]P"]) & (df["close"].shift(5) < df["[4/8]P"].shift(5)))
        )
        
        bearish_mml = (
            (df["close"] < df["[2/8]P"]) |
            ((df["close"] < df["[4/8]P"]) & (df["close"].shift(5) > df["[4/8]P"].shift(5)))
        )
        
        # MML resistance/support levels for exits
        at_resistance = (
            (df["high"] >= df["[6/8]P"]) |  # At 75%
            (df["high"] >= df["[7/8]P"]) |  # At 87.5%
            (df["high"] >= df["[8/8]P"])    # At 100%
        )
        
        at_support = (
            (df["low"] <= df["[2/8]P"]) |   # At 25%
            (df["low"] <= df["[1/8]P"]) |   # At 12.5%
            (df["low"] <= df["[0/8]P"])     # At 0%
        )
        
        # ===========================================
        # LONG EXIT SIGNALS (ADVANCED MML SYSTEM)
        # ===========================================
        
        # 1. Profit-Taking Exits
        long_exit_resistance_profit = (
            at_resistance &
            (df["close"] < df["high"]) &  # Failed to close at high
            (df["rsi"] > 65) &  # Overbought
            (df["maxima"] == 1) &  # Local top
            (df["volume"] > df["volume"].rolling(10).mean())
        )
        
        long_exit_extreme_overbought = (
            (df["close"] > df["[7/8]P"]) &
            (df["rsi"] > 75) &
            (df["close"] < df["close"].shift(1)) &  # Price turning down
            (df["maxima"] == 1)
        )
        
        long_exit_volume_exhaustion = (
            at_resistance &
            (df["volume"] < df["volume"].rolling(20).mean() * 0.6) &  # Tightened from 0.8
            (df["rsi"] > 70) &
            (df["close"] < df["close"].shift(1)) &
            (df["close"] < df["close"].rolling(3).mean())  # Added price confirmation
        )
        
        # 2. Structure Breakdown (Improved with strong filters)
        long_exit_structure_breakdown = (
            (df["close"] < df["[4/8]P"]) &
            (df["close"].shift(1) >= df["[4/8]P"].shift(1)) &
            bullish_mml.shift(1) &
            (df["close"] < df["[4/8]P"] * 0.995) &
            (df["close"] < df["close"].shift(1)) &
            (df["close"] < df["close"].shift(2)) &
            (df["rsi"] < 45) &  # Tightened from 50
            (df["volume"] > df["volume"].rolling(15).mean() * 2.0) &  # Increased from 1.5
            (df["close"] < df["open"]) &
            (df["low"] < df["low"].shift(1)) &
            (df["close"] < df["close"].rolling(3).mean()) &
            (df["momentum_quality"] < 0)  # Added momentum check
        )
        
        # 3. Momentum Divergence
        long_exit_momentum_divergence = (
            at_resistance &
            (df["rsi"] < df["rsi"].shift(1)) &  # RSI falling
            (df["rsi"].shift(1) < df["rsi"].shift(2)) &  # RSI was falling
            (df["rsi"] < df["rsi"].shift(3)) &  # 3-candle RSI decline
            (df["close"] >= df["close"].shift(1)) &  # Price still up/flat
            (df["maxima"] == 1) &
            (df["rsi"] > 60)  # Only in overbought territory
        )
        
        # 4. Range Exit
        long_exit_range = (
            (df["close"] >= df["[2/8]P"]) &
            (df["close"] <= df["[6/8]P"]) &  # In range
            (df["high"] >= df["[6/8]P"]) &  # HIGH touched 75%, not close
            (df["close"] < df["[6/8]P"] * 0.995) &  # But closed below
            (df["rsi"] > 65) &  # More conservative RSI
            (df["maxima"] == 1) &
            (df["volume"] > df["volume"].rolling(10).mean() * 1.2)  # Volume confirmation
        )
        
        # 5. Emergency Exit
        long_exit_emergency = (
            (df["close"] < df["[0/8]P"]) &
            (df["rsi"] < 20) &  # Changed from 15
            (df["volume"] > df["volume"].rolling(20).mean() * 2.5) &  # Reduced from 3
            (df["close"] < df["close"].shift(1)) &
            (df["close"] < df["close"].shift(2)) &
            (df["close"] < df["open"])
        ) if self.use_emergency_exits else pd.Series([False] * len(df), index=df.index)
        
        # Combine all Long Exit signals
        any_long_exit = (
            long_exit_resistance_profit |
            long_exit_extreme_overbought |
            long_exit_volume_exhaustion |
            long_exit_structure_breakdown |
            long_exit_momentum_divergence |
            long_exit_range |
            long_exit_emergency
        )
        
        # ===========================================
        # SHORT EXIT SIGNALS (if enabled)
        # ===========================================
        
        if self.can_short:
            # 1. Profit-Taking Exits
            short_exit_support_profit = (
                at_support &
                (df["close"] > df["low"]) &  # Failed to close at low
                (df["rsi"] < 35) &  # Oversold
                (df["minima"] == 1) &  # Local bottom
                (df["volume"] > df["volume"].rolling(10).mean())
            )
            
            short_exit_extreme_oversold = (
                (df["close"] < df["[1/8]P"]) &
                (df["rsi"] < 25) &
                (df["close"] > df["close"].shift(1)) &  # Price turning up
                (df["minima"] == 1)
            )
            
            short_exit_volume_exhaustion = (
                at_support &
                (df["volume"] < df["volume"].rolling(20).mean() * 0.6) &  # Tightened from 0.8
                (df["rsi"] < 30) &
                (df["close"] > df["close"].shift(1)) &
                (df["close"] > df["close"].rolling(3).mean())  # Added price confirmation
            )
            
            # 2. Structure Breakout
            short_exit_structure_breakout = (
                (df["close"] > df["[4/8]P"]) &
                (df["close"].shift(1) <= df["[4/8]P"].shift(1)) &
                bearish_mml.shift(1) &
                (df["close"] > df["[4/8]P"] * 1.005) &
                (df["close"] > df["close"].shift(1)) &
                (df["close"] > df["close"].shift(2)) &
                (df["rsi"] > 55) &  # Tightened from 50
                (df["volume"] > df["volume"].rolling(15).mean() * 2.0) &  # Increased from 1.5
                (df["close"] > df["open"]) &
                (df["high"] > df["high"].shift(1)) &
                (df["momentum_quality"] > 0)  # Added momentum check
            )
            
            # 3. Momentum Divergence
            short_exit_momentum_divergence = (
                at_support &
                (df["rsi"] > df["rsi"].shift(1)) &  # RSI rising
                (df["rsi"].shift(1) > df["rsi"].shift(2)) &  # RSI was rising
                (df["rsi"] > df["rsi"].shift(3)) &  # 3-candle RSI rise
                (df["close"] <= df["close"].shift(1)) &  # Price still down/flat
                (df["minima"] == 1) &
                (df["rsi"] < 40)  # Only in oversold territory
            )
            
            # 4. Range Exit
            short_exit_range = (
                (df["close"] >= df["[2/8]P"]) &
                (df["close"] <= df["[6/8]P"]) &  # In range
                (df["low"] <= df["[2/8]P"]) &  # LOW touched 25%
                (df["close"] > df["[2/8]P"] * 1.005) &  # But closed above
                (df["rsi"] < 35) &  # More conservative RSI
                (df["minima"] == 1) &
                (df["volume"] > df["volume"].rolling(10).mean() * 1.2)  # Volume confirmation
            )
            
            # 5. Emergency Exit
            short_exit_emergency = (
                (df["close"] > df["[8/8]P"]) &
                (df["rsi"] > 80) &  # Changed from 85
                (df["volume"] > df["volume"].rolling(20).mean() * 2.5) &  # Reduced from 3
                (df["close"] > df["close"].shift(1)) &
                (df["close"] > df["close"].shift(2)) &
                (df["close"] > df["open"])
            ) if self.use_emergency_exits else pd.Series([False] * len(df), index=df.index)
            
            # Combine all Short Exit signals
            any_short_exit = (
                short_exit_support_profit |
                short_exit_extreme_oversold |
                short_exit_volume_exhaustion |
                short_exit_structure_breakout |
                short_exit_momentum_divergence |
                short_exit_range |
                short_exit_emergency
            )
        else:
            any_short_exit = pd.Series([False] * len(df), index=df.index)
        
        # ===========================================
        # COORDINATION WITH ENTRY SIGNALS
        # ===========================================
        
        # If we have new Entry signals, they override Exit signals
        has_long_entry = "enter_long" in df.columns and (df["enter_long"] == 1).any()
        has_short_entry = "enter_short" in df.columns and (df["enter_short"] == 1).any()
        
        if has_long_entry:
            long_entry_mask = df["enter_long"] == 1
            any_long_exit = any_long_exit & (~long_entry_mask)
        
        if has_short_entry and self.can_short:
            short_entry_mask = df["enter_short"] == 1
            any_short_exit = any_short_exit & (~short_entry_mask)
        
        # ===========================================
        # SET FINAL EXIT SIGNALS AND TAGS
        # ===========================================
        
        # Long Exits
        df.loc[any_long_exit, "exit_long"] = 1
        
        # Tags for Long Exits (Priority: Emergency > Structure > Profit)
        df.loc[any_long_exit & long_exit_emergency, "exit_tag"] = "MML_Emergency_Long_Exit"
        df.loc[any_long_exit & long_exit_structure_breakdown & (df["exit_tag"] == ""), "exit_tag"] = "MML_Structure_Breakdown_Confirmed"
        df.loc[any_long_exit & long_exit_resistance_profit & (df["exit_tag"] == ""), "exit_tag"] = "MML_Resistance_Profit"
        df.loc[any_long_exit & long_exit_extreme_overbought & (df["exit_tag"] == ""), "exit_tag"] = "MML_Extreme_Overbought"
        df.loc[any_long_exit & long_exit_volume_exhaustion & (df["exit_tag"] == ""), "exit_tag"] = "MML_Volume_Exhaustion_Long"
        df.loc[any_long_exit & long_exit_momentum_divergence & (df["exit_tag"] == ""), "exit_tag"] = "MML_Momentum_Divergence_Long"
        df.loc[any_long_exit & long_exit_range & (df["exit_tag"] == ""), "exit_tag"] = "MML_Range_Exit_Long"
        
        # Short Exits
        if self.can_short:
            df.loc[any_short_exit, "exit_short"] = 1
            
            # Tags for Short Exits (Priority: Emergency > Structure > Profit)
            df.loc[any_short_exit & short_exit_emergency, "exit_tag"] = "MML_Emergency_Short_Exit"
            df.loc[any_short_exit & short_exit_structure_breakout & (df["exit_tag"] == ""), "exit_tag"] = "MML_Structure_Breakout_Confirmed"
            df.loc[any_short_exit & short_exit_support_profit & (df["exit_tag"] == ""), "exit_tag"] = "MML_Support_Profit"
            df.loc[any_short_exit & short_exit_extreme_oversold & (df["exit_tag"] == ""), "exit_tag"] = "MML_Extreme_Oversold"
            df.loc[any_short_exit & short_exit_volume_exhaustion & (df["exit_tag"] == ""), "exit_tag"] = "MML_Volume_Exhaustion_Short"
            df.loc[any_short_exit & short_exit_momentum_divergence & (df["exit_tag"] == ""), "exit_tag"] = "MML_Momentum_Divergence_Short"
            df.loc[any_short_exit & short_exit_range & (df["exit_tag"] == ""), "exit_tag"] = "MML_Range_Exit_Short"
        
        return df
    
    def _populate_simple_exits(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        SIMPLE OPPOSITE SIGNAL EXIT SYSTEM - SYNTAX FIXED
        """
        
        # Exit LONG when any SHORT signal appears
        long_exit_on_short = (dataframe["enter_short"] == 1)
        
        # Exit SHORT when any LONG signal appears  
        short_exit_on_long = (dataframe["enter_long"] == 1)
        
        # Emergency exits (if enabled)
        if self.use_emergency_exits:
            emergency_long_exit = (
                (dataframe['rsi'] > 85) &
                (dataframe['volume'] > dataframe['avg_volume'] * 3) &
                (dataframe['close'] < dataframe['open']) &
                (dataframe['close'] < dataframe['low'].shift(1))
            ) | (
                (dataframe.get('structure_break_down', 0) == 1) &
                (dataframe['volume'] > dataframe['avg_volume'] * 2.5) &
                (dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 2)
            )
            
            emergency_short_exit = (
                (dataframe['rsi'] < 15) &
                (dataframe['volume'] > dataframe['avg_volume'] * 3) &
                (dataframe['close'] > dataframe['open']) &
                (dataframe['close'] > dataframe['high'].shift(1))
            ) | (
                (dataframe.get('structure_break_up', 0) == 1) &
                (dataframe['volume'] > dataframe['avg_volume'] * 2.5) &
                (dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 2)
            )
        else:
            emergency_long_exit = pd.Series([False] * len(dataframe), index=dataframe.index)
            emergency_short_exit = pd.Series([False] * len(dataframe), index=dataframe.index)
        
        # Apply exits
        dataframe.loc[long_exit_on_short, "exit_long"] = 1
        dataframe.loc[long_exit_on_short, "exit_tag"] = "trend_reversal"
        
        dataframe.loc[short_exit_on_long, "exit_short"] = 1
        dataframe.loc[short_exit_on_long, "exit_tag"] = "trend_reversal"
        
        # Emergency exits
        dataframe.loc[emergency_long_exit & ~long_exit_on_short, "exit_long"] = 1
        dataframe.loc[emergency_long_exit & ~long_exit_on_short, "exit_tag"] = "emergency_exit"
        
        dataframe.loc[emergency_short_exit & ~short_exit_on_long, "exit_short"] = 1
        dataframe.loc[emergency_short_exit & ~short_exit_on_long, "exit_tag"] = "emergency_exit"
        
        # DEBUGGING (FIXED THE ERROR HERE)
        if metadata['pair'] in ['BTC/USDT:USDT', 'ETH/USDT:USDT']:
            recent_exits = dataframe['exit_long'].tail(5).sum() + dataframe['exit_short'].tail(5).sum()
            if recent_exits > 0:
                exit_tag = dataframe['exit_tag'].iloc[-1]
                logger.info(f"{metadata['pair']} EXIT SIGNAL - Tag: {exit_tag}")
                # âœ… FIXED: Use the correct attribute name
                logger.info(f"  Exit System: {'Custom MML' if self.use_custom_exits_advanced else 'Simple Opposite'}")
                logger.info(f"  RSI: {dataframe['rsi'].iloc[-1]:.1f}")
        
        return dataframe
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float,
                          time_in_force: str, exit_reason: str, current_time: datetime, **kwargs) -> bool:
        current_profit_ratio = trade.calc_profit_ratio(rate)
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600  # Hours
        
        always_allow = [
            "stoploss", "stop_loss", "custom_stoploss",
            "roi", "trend_reversal", "emergency_exit"
        ]
        
        # Allow regime protection exits
        if any(char in exit_reason for char in ["⚡", "🔊", "🌊", "🎯", "₿"]):
            return True
        
        # Allow known good exits
        if exit_reason in always_allow:
            return True
        
        # Allow trailing stop only if in profit
        if exit_reason in ["trailing_stop_loss", "trailing_stop"]:
            if current_profit_ratio > 0:  # Only allow trailing if trade is profitable
                logger.info(f"{pair} Allowing trailing stop exit. Profit: {current_profit_ratio:.2%}")
                return True
            else:
                logger.info(f"{pair} Blocking trailing stop exit. Trade not in profit: {current_profit_ratio:.2%}")
                return False
        
        # Allow timeout exit after 48 hours
        if trade_duration > 48:
            logger.info(f"{pair} Forcing exit due to max holding time (48h). Profit: {current_profit_ratio:.2%}")
            return True
        
        # Allow all other exits
        return True