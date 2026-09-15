# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
from typing import Optional, Union
import talib.abstract as ta
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)
from freqtrade.persistence import Trade
import sqlite3
import logging

logger = logging.getLogger(__name__)

class HybridOrderbookStrategy15m(IStrategy):
    """
    Hybrid 15m Scalping Strategy mit Orderbook-Filter

    Kombiniert Trend-Following und Mean-Reversion basierend auf Regime Detection.
    Nutzt Orderbook/CVD-Daten als Entry-Filter zur Bestätigung von TA-Signalen.

    Architektur:
    1. REGIME DETECTION (ADX + ATR) → Trending vs Ranging
    2. TREND-FOLLOWING → EMA Structure + MACD + Orderbook
    3. MEAN-REVERSION → RSI + Stochastic + BB + CVD Reversal
    4. ORDERBOOK FILTER → CVD Slope + Delta Ratio + Divergence
    """

    INTERFACE_VERSION = 3

    # ==================== CONFIGURATION ====================

    timeframe = '15m'

    # ROI: Realistisch für 15m Scalping
    minimal_roi = {
        "0": 0.025,    # 2.5% sofort
        "15": 0.015,   # 1.5% nach 1 Candle (15 min)
        "30": 0.010,   # 1.0% nach 2 Candles (30 min)
        "45": 0.005    # 0.5% nach 3 Candles (45 min)
    }

    # Stop-Loss: Nur für Flash Crashes
    stoploss = -0.10  # -10% (absolute Notbremse)

    # Trailing Stop: Früher aktiviert, enger am Preis
    trailing_stop = True
    trailing_stop_positive = 0.003       # Start bei +0.3% (war +0.5%)
    trailing_stop_positive_offset = 0.008  # Trail 0.8% unter Peak (war 1.2%)
    trailing_only_offset_is_reached = True

    # DCA: DISABLED für Scalping
    position_adjustment_enable = False
    max_entry_position_adjustment = 0

    # Order Types
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Order Time in Force
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    # Startup candle count - nicht mehr nötig, da historische Daten vorhanden
    startup_candle_count: int = 0

    # Can short
    can_short: bool = True

    # Exit signal - DISABLED (hurts performance, rely on ROI + Trailing Stop)
    use_exit_signal: bool = False

    # ==================== PROTECTIONS ====================

    @property
    def protections(self):
        return [
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 96,   # 24h
                "trade_limit": 10,
                "stop_duration_candles": 12,     # 3h pause
                "max_allowed_drawdown": 0.12     # 12% max DD
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 12,   # 3h
                "trade_limit": 3,                # 3 losses
                "stop_duration_candles": 8,      # 2h pause
                "only_per_pair": False
            }
        ]

    # ==================== ORDERBOOK/CVD METHODS ====================

    def load_cvd_history(self, pair: str, limit: int = 500) -> pd.DataFrame:
        """
        Load CVD history from Rust feeder SQLite database
        Returns 1m candles with delta, buy_vol, sell_vol, whale metrics
        """
        try:
            db_path = '/freqtrade/user_data/orderbook_cache.db'

            # Convert pair format: BTC/USDC:USDC → BTC/USDT
            symbol = pair.split('/')[0] + '/USDT'

            conn = sqlite3.connect(db_path, timeout=10)

            query = f"""
                SELECT timestamp, delta, volume, buy_vol, sell_vol,
                       buy_count, sell_count, whale_buy_vol, whale_sell_vol, whale_delta
                FROM trade_candles
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """

            df = pd.read_sql_query(query, conn, params=[symbol, limit])
            conn.close()

            if df.empty:
                logger.warning(f"No CVD data found for {symbol}")
                return pd.DataFrame()

            # Sort ascending by time
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Error loading CVD for {pair}: {e}")
            return pd.DataFrame()

    def resample_cvd_to_15m(self, cvd_df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 1m CVD data to 15m by summing deltas over 15 candles
        """
        if cvd_df.empty:
            return pd.DataFrame()

        try:
            # Convert timestamp to datetime
            cvd_df['datetime'] = pd.to_datetime(cvd_df['timestamp'], unit='s')
            cvd_df.set_index('datetime', inplace=True)

            # Resample to 15m
            resampled = cvd_df.resample('15min').agg({
                'delta': 'sum',
                'volume': 'sum',
                'buy_vol': 'sum',
                'sell_vol': 'sum',
                'buy_count': 'sum',
                'sell_count': 'sum',
                'whale_buy_vol': 'sum',
                'whale_sell_vol': 'sum',
                'whale_delta': 'sum',
                'timestamp': 'first'
            })

            resampled.reset_index(drop=True, inplace=True)

            return resampled

        except Exception as e:
            logger.error(f"Error resampling CVD: {e}")
            return pd.DataFrame()

    def calculate_cvd_metrics(self, dataframe: pd.DataFrame, cvd_15m: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CVD-based metrics for orderbook filtering
        """
        if cvd_15m.empty:
            # Fill with zeros if no CVD data
            dataframe['cvd_absolute'] = 0
            dataframe['cvd_ema'] = 0
            dataframe['delta_ratio'] = 0
            dataframe['cvd_slope'] = 0
            dataframe['whale_ratio'] = 0
            dataframe['whale_delta'] = 0
            return dataframe

        # Align CVD data with dataframe length
        cvd_len = len(cvd_15m)
        df_len = len(dataframe)

        if cvd_len < df_len:
            # Pad with zeros at the beginning
            pad_len = df_len - cvd_len
            for col in ['delta', 'volume', 'buy_vol', 'sell_vol', 'whale_delta']:
                if col in cvd_15m.columns:
                    padded = np.concatenate([np.zeros(pad_len), cvd_15m[col].values])
                    dataframe[f'cvd_{col}'] = padded
        else:
            # Take last df_len rows
            for col in ['delta', 'volume', 'buy_vol', 'sell_vol', 'whale_delta']:
                if col in cvd_15m.columns:
                    dataframe[f'cvd_{col}'] = cvd_15m[col].values[-df_len:]

        # Calculate metrics
        dataframe['cvd_absolute'] = dataframe['cvd_delta'].cumsum()
        dataframe['cvd_ema'] = ta.EMA(dataframe['cvd_absolute'], timeperiod=8)

        # Delta ratio: % of volume that is buy pressure
        dataframe['delta_ratio'] = np.where(
            dataframe['cvd_volume'] > 0,
            dataframe['cvd_delta'] / dataframe['cvd_volume'],
            0
        )

        # CVD slope: 3-candle rate of change
        dataframe['cvd_slope'] = dataframe['cvd_ema'].diff(3)

        # Whale metrics
        dataframe['whale_delta'] = dataframe['cvd_whale_delta']
        dataframe['whale_ratio'] = np.where(
            dataframe['cvd_volume'] > 0,
            dataframe['cvd_whale_delta'] / dataframe['cvd_volume'],
            0
        )

        return dataframe

    # ==================== INDICATORS ====================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate all technical indicators and CVD metrics
        """

        # ========== ORDERBOOK / CVD DATA ==========

        cvd_df = self.load_cvd_history(metadata['pair'], limit=500)
        cvd_15m = self.resample_cvd_to_15m(cvd_df)
        dataframe = self.calculate_cvd_metrics(dataframe, cvd_15m)

        # ========== REGIME DETECTION ==========

        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['is_trending'] = dataframe['adx'] > 25

        # ========== TREND-FOLLOWING INDICATORS ==========

        # EMA Structure: 8/16/48 periods = 2h/4h/12h on 15m
        dataframe['ema_fast'] = ta.EMA(dataframe['close'], timeperiod=8)
        dataframe['ema_slow'] = ta.EMA(dataframe['close'], timeperiod=16)
        dataframe['ema_vslow'] = ta.EMA(dataframe['close'], timeperiod=48)

        # MACD Momentum
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['macd_hist'] = macd['macdhist']

        # ========== MEAN-REVERSION INDICATORS ==========

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Stochastic
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']

        # Bollinger Bands
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'] = bb['upperband']
        dataframe['bb_middle'] = bb['middleband']
        dataframe['bb_lower'] = bb['lowerband']
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']

        # ========== VOLATILITY & SUPPORT/RESISTANCE ==========

        # ATR-based support/resistance
        dataframe['atr_high'] = dataframe['close'] + (dataframe['atr'] * 1.5)
        dataframe['atr_low'] = dataframe['close'] - (dataframe['atr'] * 1.5)

        return dataframe

    # ==================== ENTRY SIGNALS ====================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Hybrid entry logic: Trend-Following OR Mean-Reversion based on regime
        """

        # ========== ORDERBOOK-FILTERED STRATEGY ==========
        # CVD als BESTÄTIGUNG, nicht als harte Blockierung

        # CVD-Filter: Nur blockieren wenn CVD STARK gegen das Signal spricht
        cvd_allows_long = (
            (dataframe['cvd_slope'] >= -5) |  # CVD nicht stark fallend
            (dataframe['delta_ratio'] >= -0.15) |  # Nicht 15%+ Verkaufsdruck
            (dataframe['cvd_absolute'].isna())  # Oder keine CVD-Daten (nicht blockieren)
        )

        cvd_allows_short = (
            (dataframe['cvd_slope'] <= 5) |  # CVD nicht stark steigend
            (dataframe['delta_ratio'] <= 0.15) |  # Nicht 15%+ Kaufdruck
            (dataframe['cvd_absolute'].isna())  # Oder keine CVD-Daten (nicht blockieren)
        )

        # LONG: Stark oversold + Uptrend + CVD erlaubt es
        simple_long = (
            (dataframe['rsi'] < 30) &  # Sehr oversold
            (dataframe['ema_fast'] > dataframe['ema_vslow']) &  # Langfristiger Uptrend
            (dataframe['volume'] > 0) &
            cvd_allows_long  # CVD blockiert nicht
        )

        # SHORT: Stark overbought + Downtrend + CVD erlaubt es
        simple_short = (
            (dataframe['rsi'] > 70) &  # Sehr overbought
            (dataframe['ema_fast'] < dataframe['ema_vslow']) &  # Langfristiger Downtrend
            (dataframe['volume'] > 0) &
            cvd_allows_short  # CVD blockiert nicht
        )

        # TREND-FOLLOWING mit CVD-Bestätigung
        trend_long = (
            (dataframe['ema_fast'] > dataframe['ema_slow']) &  # Uptrend
            (dataframe['ema_slow'] > dataframe['ema_vslow']) &  # Alle EMAs aligned
            (dataframe['close'] < dataframe['ema_fast']) &  # Pullback zum EMA
            (dataframe['close'] > dataframe['ema_slow']) &  # Nicht zu tief
            (dataframe['rsi'] > 40) &  # Nicht oversold (Stärke zeigen)
            (dataframe['rsi'] < 60) &  # Nicht overbought
            (dataframe['volume'] > 0) &
            cvd_allows_long  # CVD blockiert nicht
        )

        trend_short = (
            (dataframe['ema_fast'] < dataframe['ema_slow']) &  # Downtrend
            (dataframe['ema_slow'] < dataframe['ema_vslow']) &  # Alle EMAs aligned
            (dataframe['close'] > dataframe['ema_fast']) &  # Pullback zum EMA
            (dataframe['close'] < dataframe['ema_slow']) &  # Nicht zu hoch
            (dataframe['rsi'] < 60) &  # Nicht overbought (Schwäche zeigen)
            (dataframe['rsi'] > 40) &  # Nicht oversold
            (dataframe['volume'] > 0) &
            cvd_allows_short  # CVD blockiert nicht
        )

        # ========== COMBINED ENTRY SIGNALS ==========

        dataframe.loc[simple_long | trend_long, 'enter_long'] = 1
        dataframe.loc[simple_short | trend_short, 'enter_short'] = 1

        # Entry tags for tracking
        dataframe.loc[simple_long, 'enter_tag'] = 'rsi_long'
        dataframe.loc[trend_long & (dataframe.get('enter_tag', '') == ''), 'enter_tag'] = 'trend_long'
        dataframe.loc[simple_short, 'enter_tag'] = 'rsi_short'
        dataframe.loc[trend_short & (dataframe.get('enter_tag', '') == ''), 'enter_tag'] = 'trend_short'

        return dataframe

    # ==================== EXIT SIGNALS ====================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Dynamic exit logic based on trend reversal or profit targets
        """

        # ========== SIMPLIFIED EXIT LONG ==========

        exit_long = (
            # RSI back to neutral/overbought (mean reversion complete)
            (dataframe['rsi'] > 60) |

            # Trend reversal (simple EMA cross)
            (dataframe['ema_fast'] < dataframe['ema_slow'])
        )

        # ========== SIMPLIFIED EXIT SHORT ==========

        exit_short = (
            # RSI back to neutral/oversold (mean reversion complete)
            (dataframe['rsi'] < 40) |

            # Trend reversal (simple EMA cross)
            (dataframe['ema_fast'] > dataframe['ema_slow'])
        )

        dataframe.loc[exit_long, 'exit_long'] = 1
        dataframe.loc[exit_short, 'exit_short'] = 1

        return dataframe

    # ==================== CUSTOM EXIT ====================

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """
        Custom exit logic for risk management and time-based exits
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe.empty or len(dataframe) < 1:
            return None

        last_candle = dataframe.iloc[-1].squeeze()

        # Trade duration in minutes
        duration_min = (current_time - trade.open_date_utc).total_seconds() / 60

        # ========== TIME DECAY EXIT ==========

        # Force exit after 90 minutes (6 candles) if in profit
        if duration_min > 90 and current_profit > 0:
            return 'time_decay_90min'

        # Force exit after 120 minutes (8 candles) regardless
        if duration_min > 120:
            return 'time_decay_120min'

        # ========== MEAN-REVERSION QUICK EXIT ==========

        # For mean-reversion entries, exit quickly at 1.5% profit
        if trade.enter_tag in ['mr_long', 'mr_short']:
            if current_profit > 0.015:
                return 'mr_target_reached'

        # ========== TREND-FOLLOWING PROFIT PROTECTION ==========

        # For trend entries, protect profits if CVD reverses
        if trade.enter_tag == 'trend_long':
            if current_profit > 0.01 and last_candle['cvd_slope'] < 0 and last_candle['delta_ratio'] < -0.10:
                return 'cvd_reversal_long'

        if trade.enter_tag == 'trend_short':
            if current_profit > 0.01 and last_candle['cvd_slope'] > 0 and last_candle['delta_ratio'] > 0.10:
                return 'cvd_reversal_short'

        # ========== WHALE ACTIVITY EXIT ==========

        # Exit if whale activity moves against position
        if trade.enter_tag in ['trend_long', 'mr_long']:
            if last_candle['whale_ratio'] < -0.20:  # Heavy whale selling
                return 'whale_selling'

        if trade.enter_tag in ['trend_short', 'mr_short']:
            if last_candle['whale_ratio'] > 0.20:  # Heavy whale buying
                return 'whale_buying'

        return None

    # ==================== LEVERAGE ====================

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Conservative leverage for 15m scalping
        """
        return 3.0  # 3x leverage für Futures
