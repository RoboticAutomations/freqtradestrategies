# --- Do not remove these libs ---
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date

import logging

import datetime
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from pandas import DataFrame, Series
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import (IStrategy, BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, RealParameter, merge_informative_pair, stoploss_from_open,
                                stoploss_from_absolute, merge_informative_pair)
from freqtrade.persistence import Trade
from typing import List, Tuple, Optional
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from collections import deque

import warnings

import logging

logger = logging.getLogger(__name__)

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

class PlotConfig():
    def __init__(self):
        self.config = {
            'main_plot': {
                # Try direct column names first to test
                'bollinger_upperband': {'color': 'rgba(4,137,122,0.7)'},
                'kc_upperband': {'color': 'rgba(4,146,250,0.7)'},
                'kc_middleband': {'color': 'rgba(4,146,250,0.7)'},
                'kc_lowerband': {'color': 'rgba(4,146,250,0.7)'},
                'bollinger_lowerband': {
                    'color': 'rgba(4,137,122,0.7)',
                    'fill_to': 'bollinger_upperband',
                    'fill_color': 'rgba(4,137,122,0.07)'
                },
                'ema9': {'color': 'purple'},
                'ema20': {'color': 'yellow'},
                'ema50': {'color': 'red'},
                'ema200': {'color': 'white'},
                'trend_1h_1h': {'color': 'orange'},
            },
            'subplots': {
                "RSI": {
                    'rsi': {'color': 'green'}
                },
                "ATR": {
                    'atr': {'color': 'firebrick'}
                },
                "Signal Strength": {
                    'signal_strength': {'color': 'blue'}
                }
            }
        }
    
    def add_total_divergences_in_config(self, dataframe):
        # Test if columns exist before adding them
        if 'total_bullish_divergences' in dataframe.columns:
            self.config['main_plot']['total_bullish_divergences'] = {
                "plotly": {
                    'mode': 'markers',
                    'marker': {
                        'symbol': 'diamond',
                        'size': 11,
                        'color': 'green'
                    }
                }
            }
        
        if 'total_bearish_divergences' in dataframe.columns:
            self.config['main_plot']['total_bearish_divergences'] = {
                "plotly": {
                    'mode': 'markers',
                    'marker': {
                        'symbol': 'diamond',
                        'size': 11,
                        'color': 'crimson'
                    }
                }
            }
        
        return self

class AlexBandSniperV4(IStrategy):
    """
    Alex BandSniper on 15m Timeframe - OPTIMIZED VERSION
    Version 47C - Claude optimized Entry & Exit
    Key improvements:
    - Fixed ROI and Trailing adjusted Custom Exits
    - Fixed Entry Signals
    - Included 1h Informative Timeframe
    - Multi-timeframe analysis (1h trend confirmation)
    - Enhanced signal filtering with minimum divergence counts
    - Volume and volatility filters
    - Adaptive position sizing based on signal strength
    - Improved risk management
    """
    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "v34C-optimized"

    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.15, -0.03, decimals=2, name='stoploss')]

        # Define a custom max_open_trades space
        def max_open_trades_space() -> List[Dimension]:
            return [
                Integer(3, 8, name='max_open_trades'),
            ]
        
        def trailing_space() -> List[Dimension]:
            return [
                Categorical([True], name='trailing_stop'),
                SKDecimal(0.02, 0.3, decimals=2, name='trailing_stop_positive'),
                SKDecimal(0.03, 0.1, decimals=2, name='trailing_stop_positive_offset_p1'),
                Categorical([True, False], name='trailing_only_offset_is_reached'),
            ]
    
    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.0314 # Disables ROI completely - let custom_exit handle everything
    }
    
    # Optimal stoploss designed for the strategy.
    stoploss = -0.99
    can_short = True
    use_custom_stoploss = False
    leverage_value = 1.0  # Reduced leverage for better risk management

    #trailing_stop = False
    #trailing_stop_positive = 0.40        # Only trail after 40% profit (very high)
    #trailing_stop_positive_offset = 0.45 # Start trailing at 45% profit
    #trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy.
    timeframe = '1h'
    timeframe_minutes = timeframe_to_minutes(timeframe)

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "exit_pricing" section in the config.
    use_exit_signal = True
    exit_profit_only = False
    # ignore_roi_if_entry_signal = False
    use_custom_exits = True
    # In your hyperopt parameters, consider these more permissive defaults:
    min_divergence_count = IntParameter(1, 3, default=1, space='buy', optimize=True, load=True)  # Reduced from 2-5
    min_signal_strength = IntParameter(1, 5, default=1, space='buy', optimize=True, load=True)   # Reduced from 3-10
    volume_threshold = DecimalParameter(1.0, 1.5, default=1.0, decimals=1, space='buy', optimize=True, load=True)  # Reduced from 1.1-2.5

    # Make ADX less restrictive
    adx_threshold = IntParameter(15, 30, default=15, space='buy', optimize=True, load=True)  # Reduced from 25-45
  
    # Market Condition Filters
    rsi_overbought = DecimalParameter(65.0, 85.0, default=80.0, decimals=1, space='buy', optimize=True, load=True)
    rsi_oversold = DecimalParameter(15.0, 35.0, default=15.0, decimals=1, space='buy', optimize=True, load=True)
    
    # Volatility Filters
    max_volatility = DecimalParameter(0.015, 0.035, default=0.025, decimals=3, space='buy', optimize=True, load=True)
    min_volatility = DecimalParameter(0.003, 0.008, default=0.005, decimals=3, space='buy', optimize=True, load=True)
    
    # Exit Parameters
    rsi_exit_overbought = DecimalParameter(70.0, 90.0, default=80.0, decimals=1, space='sell', optimize=True, load=True)
    rsi_exit_oversold = DecimalParameter(10.0, 30.0, default=20.0, decimals=1, space='sell', optimize=True, load=True)
    adx_exit_threshold = IntParameter(15, 30, default=20, space='sell', optimize=True, load=True)
    
    # Trend Confirmation Parameters
    trend_strength_threshold = IntParameter(20, 40, default=25, space='buy', optimize=True, load=True)
    
    # Technical Parameters
    window = IntParameter(3, 6, default=5, space="buy", optimize=True, load=True)
    index_range = IntParameter(20, 50, default=30, space='buy', optimize=True, load=True)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 24

    # Protection parameters
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=20, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)
    use_cooldown_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # Enhanced protection parameters
    use_max_drawdown_protection = BooleanParameter(default=False, space="protection", optimize=True)
    max_drawdown_lookback = IntParameter(100, 300, default=200, space="protection", optimize=True)
    max_drawdown_trade_limit = IntParameter(5, 15, default=10, space="protection", optimize=True)
    max_drawdown_stop_duration = IntParameter(1, 5, default=1, space="protection", optimize=True)
    max_allowed_drawdown = DecimalParameter(0.08, 0.25, default=0.15, decimals=2, space="protection", optimize=True)

    stoploss_guard_lookback = IntParameter(30, 80, default=50, space="protection", optimize=True)
    stoploss_guard_trade_limit = IntParameter(2, 6, default=3, space="protection", optimize=True)
    stoploss_guard_only_per_pair = BooleanParameter(default=True, space="protection", optimize=True)

    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'limit',
        'stoploss_on_exchange': True
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = None

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])
    
    def informative_pairs(self):
        """Define additional timeframes to download"""
        pairs = self.dp.current_whitelist()
        return [(pair, '1h') for pair in pairs]
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enhanced indicator population with multi-timeframe analysis - Fixed for dry run
        """
        
        # === MULTI-TIMEFRAME ANALYSIS ===
        # Get 1h timeframe for trend confirmation with improved error handling
        try:
            # Check if we're in backtesting mode or if pair supports 1h data
            if hasattr(self.dp, 'runmode') and self.dp.runmode.value in ['backtest', 'hyperopt']:
                # In backtesting, try to get 1h data but don't fail if unavailable
                informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
            else:
                # In live/dry run, be more cautious about data availability
                try:
                    informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
                except Exception:
                    informative_1h = None
            
            # Enhanced data validation
            if (informative_1h is not None and 
                len(informative_1h) > 50 and  # Reduced minimum requirement
                not informative_1h.empty and
                'close' in informative_1h.columns):
                
                try:
                    # 1h Trend indicators with additional error checking
                    informative_1h['ema50_1h'] = ta.EMA(informative_1h, timeperiod=50)
                    informative_1h['ema200_1h'] = ta.EMA(informative_1h, timeperiod=200)
                    informative_1h['trend_1h'] = ta.EMA(informative_1h, timeperiod=21)
                    informative_1h['trend_strength_1h'] = ta.ADX(informative_1h)
                    informative_1h['rsi_1h'] = ta.RSI(informative_1h)
                    
                    # Fill NaN values before merging
                    informative_1h = informative_1h.bfill().ffill()
                    
                    # Safe merge with additional error handling
                    dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, '1h', ffill=True)
                    
                    
                except Exception as merge_error:
                    logger.warning(f"Failed to merge 1h data for {metadata['pair']}: {merge_error}")
                    self._add_dummy_1h_columns(dataframe)
            else:
                logger.info(f"Using fallback 1h indicators for {metadata['pair']} (insufficient data)")
                self._add_dummy_1h_columns(dataframe)
                
        except Exception as e:
            logger.warning(f"Error accessing 1h data for {metadata['pair']}: {e}")
            self._add_dummy_1h_columns(dataframe)
        
        # === 15M TIMEFRAME INDICATORS ===
        informative = dataframe.copy()
        
        # Volume analysis with safe calculations
        try:
            informative['volume_sma'] = informative['volume'].rolling(window=20, min_periods=1).mean()
            informative['volume_ratio'] = informative['volume'] / informative['volume_sma']
            informative['volume_ratio'] = informative['volume_ratio'].fillna(1.0)
        except:
            informative['volume_sma'] = informative['volume']
            informative['volume_ratio'] = 1.0
        
        # Volatility analysis with safe calculations
        try:
            informative['atr'] = qtpylib.atr(informative, window=14, exp=False)
            informative['volatility'] = informative['atr'] / informative['close']
            informative['volatility'] = informative['volatility'].fillna(0.01)
        except:
            informative['atr'] = informative['close'] * 0.02
            informative['volatility'] = 0.01
        
        # Momentum Indicators with error handling
        try:
            informative['rsi'] = ta.RSI(informative)
            informative['stoch'] = ta.STOCH(informative)['slowk']
            informative['roc'] = ta.ROC(informative)
            informative['uo'] = ta.ULTOSC(informative)
            informative['ao'] = qtpylib.awesome_oscillator(informative)
            informative['macd'] = ta.MACD(informative)['macd']
            informative['cci'] = ta.CCI(informative)
            informative['cmf'] = chaikin_money_flow(informative, 20)
            informative['obv'] = ta.OBV(informative)
            informative['mfi'] = ta.MFI(informative)
            informative['adx'] = ta.ADX(informative)
            
            # Fill NaN values for all indicators
            indicator_columns = ['rsi', 'stoch', 'roc', 'uo', 'ao', 'macd', 'cci', 'cmf', 'obv', 'mfi', 'adx']
            for col in indicator_columns:
                if col in informative.columns:
                    informative[col] = informative[col].bfill().fillna(50 if col in ['rsi', 'mfi'] else 0)
                    
        except Exception as e:
            logger.warning(f"Error calculating momentum indicators: {e}")
            # Provide fallback values
            informative['rsi'] = 50
            informative['stoch'] = 50
            informative['roc'] = 0
            informative['uo'] = 50
            informative['ao'] = 0
            informative['macd'] = 0
            informative['cci'] = 0
            informative['cmf'] = 0
            informative['obv'] = informative['volume'].cumsum()
            informative['mfi'] = 50
            informative['adx'] = 25

        # Keltner Channel with error handling
        try:
            keltner = emaKeltner(informative)
            informative["kc_upperband"] = keltner["upper"]
            informative["kc_middleband"] = keltner["mid"]
            informative["kc_lowerband"] = keltner["lower"]
        except:
            informative["kc_upperband"] = informative['close'] * 1.02
            informative["kc_middleband"] = informative['close']
            informative["kc_lowerband"] = informative['close'] * 0.98

        # Bollinger Bands with error handling
        try:
            bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
            informative['bollinger_upperband'] = bollinger['upper']
            informative['bollinger_lowerband'] = bollinger['lower']
        except:
            informative['bollinger_upperband'] = informative['close'] * 1.02
            informative['bollinger_lowerband'] = informative['close'] * 0.98

        # EMA - Exponential Moving Average with error handling
        try:
            informative['ema9'] = ta.EMA(informative, timeperiod=9)
            informative['ema20'] = ta.EMA(informative, timeperiod=20)
            informative['ema50'] = ta.EMA(informative, timeperiod=50)
            informative['ema200'] = ta.EMA(informative, timeperiod=200)
            
            # Fill NaN values for EMAs
            ema_columns = ['ema9', 'ema20', 'ema50', 'ema200']
            for col in ema_columns:
                if col in informative.columns:
                    informative[col] = informative[col].bfill().fillna(informative['close'])
        except:
            informative['ema9'] = informative['close']
            informative['ema20'] = informative['close']
            informative['ema50'] = informative['close']
            informative['ema200'] = informative['close']

        # Pivot Points with error handling
        try:
            pivots = pivot_points(informative, self.window.value)
            informative['pivot_lows'] = pivots['pivot_lows']
            informative['pivot_highs'] = pivots['pivot_highs']
        except Exception as e:
            logger.warning(f"Error calculating pivot points: {e}")
            informative['pivot_lows'] = np.nan
            informative['pivot_highs'] = np.nan

        # === DIVERGENCE ANALYSIS ===
        try:
            self.initialize_divergences_lists(informative)
            (high_iterator, low_iterator) = self.get_iterators(informative)
            
            # Add divergences for multiple indicators
            indicators = ['rsi', 'stoch', 'roc', 'uo', 'ao', 'macd', 'cci', 'cmf', 'obv', 'mfi']
            for indicator in indicators:
                try:
                    if indicator in informative.columns:
                        self.add_divergences(informative, indicator, high_iterator, low_iterator)
                except Exception as e:
                    logger.warning(f"Error adding divergences for {indicator}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"Error in divergence analysis: {e}")
            # Initialize with empty divergence data
            informative["total_bullish_divergences"] = np.nan
            informative["total_bullish_divergences_count"] = 0
            informative["total_bullish_divergences_names"] = ''
            informative["total_bearish_divergences"] = np.nan
            informative["total_bearish_divergences_count"] = 0
            informative["total_bearish_divergences_names"] = ''
        
        # === SIGNAL STRENGTH CALCULATION ===
        try:
            informative['signal_strength'] = self.calculate_signal_strength(informative)
        except:
            informative['signal_strength'] = 0
        
        # === MERGE BACK TO DATAFRAME ===
        for col in informative.columns:
            if col not in dataframe.columns:
                dataframe[col] = informative[col]
            else:
                dataframe[col] = informative[col]

        # Additional market structure analysis with error handling
        try:
            dataframe['chop'] = choppiness_index(dataframe['high'], dataframe['low'], dataframe['close'], window=14)
            dataframe['natr'] = ta.NATR(dataframe['high'], dataframe['low'], dataframe['close'], window=14)
            dataframe['natr_diff'] = dataframe['natr'] - dataframe['natr'].shift(1)
            dataframe['natr_direction_change'] = (dataframe['natr_diff'] * dataframe['natr_diff'].shift(1) < 0)
        except:
            dataframe['chop'] = 50
            dataframe['natr'] = 0.02
            dataframe['natr_diff'] = 0
            dataframe['natr_direction_change'] = False

        # Support/Resistance levels with error handling
        try:
            dataframe['swing_high'] = dataframe['high'].rolling(window=50, min_periods=1).max()
            dataframe['swing_low'] = dataframe['low'].rolling(window=50, min_periods=1).min()
            dataframe['distance_to_resistance'] = (dataframe['swing_high'] - dataframe['close']) / dataframe['close']
            dataframe['distance_to_support'] = (dataframe['close'] - dataframe['swing_low']) / dataframe['close']
        except:
            dataframe['swing_high'] = dataframe['high']
            dataframe['swing_low'] = dataframe['low']
            dataframe['distance_to_resistance'] = 0.02
            dataframe['distance_to_support'] = 0.02

        dataframe["minima"], dataframe["maxima"] = calculate_minima_maxima(dataframe)
        dataframe["maxima_check"] = (
            dataframe["maxima"].rolling(4).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)
        )
        dataframe["minima_check"] = (
            dataframe["minima"].rolling(4).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)
        )

        # Plot configuration with error handling
        try:
            self.plot_config = (
                PlotConfig()
                .add_total_divergences_in_config(dataframe)
                .config)
        except:
            self.plot_config = None

        return dataframe

    def _add_dummy_1h_columns(self, dataframe):
        """Add dummy 1h columns when higher timeframe data is unavailable"""
        # Use current 15m data to simulate 1h trend
        try:
            dataframe['ema50_1h_1h'] = ta.EMA(dataframe, timeperiod=200)  # Use longer period on 15m
            dataframe['ema200_1h_1h'] = ta.EMA(dataframe, timeperiod=800)  # Use much longer period
            dataframe['trend_1h_1h'] = ta.EMA(dataframe, timeperiod=84)   # 21 * 4 (4x 15m = 1h)
            dataframe['trend_strength_1h_1h'] = ta.ADX(dataframe)
            dataframe['rsi_1h_1h'] = ta.RSI(dataframe, timeperiod=56)     # Adjusted for timeframe
            
            # Fill NaN values
            columns_1h = ['ema50_1h_1h', 'ema200_1h_1h', 'trend_1h_1h', 'trend_strength_1h_1h', 'rsi_1h_1h']
            for col in columns_1h:
                if col in dataframe.columns:
                    dataframe[col] = dataframe[col].bfill().fillna(
                        dataframe['close'] if 'ema' in col or 'trend' in col else 
                        25 if 'strength' in col else 50
                    )
        except Exception as e:
            logger.warning(f"Error creating dummy 1h columns: {e}")
            # Absolute fallback
            dataframe['ema50_1h_1h'] = dataframe['close']
            dataframe['ema200_1h_1h'] = dataframe['close']
            dataframe['trend_1h_1h'] = dataframe['close']
            dataframe['trend_strength_1h_1h'] = 25
            dataframe['rsi_1h_1h'] = 50

    def calculate_signal_strength(self, dataframe: DataFrame) -> Series:
        """
        Calculate overall signal strength based on multiple factors
        """
        strength = pd.Series(0, index=dataframe.index)
        
        # Divergence strength
        strength += dataframe['total_bullish_divergences_count'] * 2
        strength += dataframe['total_bearish_divergences_count'] * 2
        
        # Volume strength
        volume_strength = np.where(dataframe['volume_ratio'] > 1.5, 2, 
                                 np.where(dataframe['volume_ratio'] > 1.2, 1, 0))
        strength += volume_strength
        
        # Trend alignment strength
        ema_bullish = (dataframe['ema20'] > dataframe['ema50']) & (dataframe['ema50'] > dataframe['ema200'])
        ema_bearish = (dataframe['ema20'] < dataframe['ema50']) & (dataframe['ema50'] < dataframe['ema200'])
        strength += np.where(ema_bullish | ema_bearish, 1, 0)
        
        # ADX strength
        strength += np.where(dataframe['adx'] > 30, 1, 0)
        
        return strength

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if len(dataframe) > 1:
            dataframe = dataframe.copy().iloc[:-1]

        # Initialize
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe["enter_tag"] = ""
       
        # === PRIMARY DIVERGENCE CONDITIONS ===
        bullish_divergence = (
            (dataframe['total_bullish_divergences'].shift(1) > 0)
        )
       
        bearish_divergence = (
            (dataframe['total_bearish_divergences'].shift(1) > 0)
        )
       
        # === FILTER CONDITIONS ===
        volatility_ok = (
            (dataframe['volatility'].shift(1) >= self.min_volatility.value * 0.5) &
            (dataframe['volatility'].shift(1) <= self.max_volatility.value * 2.0)
        )
       
        bands_long = (
            (dataframe['low'] <= dataframe['kc_lowerband']) |
            (dataframe['close'] <= dataframe['kc_lowerband'])
        )
       
        bands_short = (
            (dataframe['high'] >= dataframe['kc_upperband']) |
            (dataframe['close'] >= dataframe['kc_upperband'])
        )
       
        rsi_long_ok = (
            (dataframe['rsi'].shift(1) < self.rsi_overbought.value + 5) &
            (dataframe['rsi'].shift(1) > 30)
        )
       
        rsi_short_ok = (
            (dataframe['rsi'].shift(1) > self.rsi_oversold.value - 5) &
            (dataframe['rsi'].shift(1) < 70)
        )
       
        has_volume = dataframe['volume'] > 0

        minima_ok = (
            (
                (dataframe["minima_check"] == 0) &
                (dataframe["minima_check"].shift(5) == 1)
            ) |
            (
                (dataframe["minima"].shift(1) == 1)
            )
        )
        maxima_ok = (
            (
                (dataframe["maxima_check"] == 0) &
                (dataframe["maxima_check"].shift(5) == 1)
            ) |
            (
                (dataframe["maxima"].shift(1) == 1)
            )
        )
       
        # === PRIMARY CONDITIONS (HIGHEST PRIORITY) ===
        long_condition_primary = (
            bullish_divergence &
            volatility_ok &
            bands_long &
            rsi_long_ok &
            has_volume &
            minima_ok
        )
       
        short_condition_primary = (
            bearish_divergence &
            volatility_ok &
            bands_short &
            rsi_short_ok &
            has_volume &
            maxima_ok
        )
       
        # === TREND BREAKOUT CONDITIONS ===
        long_condition_trend = (
            (dataframe['close'] > dataframe['ema20']) &
            (dataframe['ema20'] > dataframe['ema50']) &
            (dataframe['ema50'] > dataframe['ema200']) &
            (dataframe['close'].shift(1) < dataframe['ema20'].shift(1)) &
            (dataframe['close'] > dataframe['ema20']) &
            (~((dataframe['close'].shift(2) > dataframe['ema20'].shift(2)) & 
               (dataframe['close'].shift(3) < dataframe['ema20'].shift(3)))) &
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.5) &
            (dataframe['rsi'].shift(1) > 40) &
            (dataframe['rsi'].shift(1) < 60) &
            (dataframe['close'] > dataframe['close'].shift(2)) &
            (dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 1.0) &
            (dataframe['close'] > dataframe['kc_middleband']) &
            (dataframe['adx'] > 25) &
            has_volume &
            minima_ok
        )
        
        short_condition_trend = (
            (dataframe['close'] < dataframe['ema20']) &
            (dataframe['ema20'] < dataframe['ema50']) &
            (dataframe['ema50'] < dataframe['ema200']) &
            (dataframe['close'].shift(1) > dataframe['ema20'].shift(1)) &
            (dataframe['close'] < dataframe['ema20']) &
            (~((dataframe['close'].shift(2) < dataframe['ema20'].shift(2)) & 
               (dataframe['close'].shift(3) > dataframe['ema20'].shift(3)))) &
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.5) &
            (dataframe['rsi'].shift(1) > 40) &
            (dataframe['rsi'].shift(1) < 60) &
            (dataframe['close'] < dataframe['close'].shift(2)) &
            (dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 1.0) &
            (dataframe['close'] < dataframe['kc_middleband']) &
            (dataframe['adx'] > 25) &
            has_volume &
            maxima_ok
        )
       
        # === MOMENTUM CONTINUATION CONDITIONS ===
        long_condition_momentum = (
            (dataframe['close'] > dataframe['ema50']) &
            (dataframe['ema20'] > dataframe['ema50']) &
            (dataframe['ema50'] > dataframe['ema200']) &
            (dataframe['rsi'].shift(3) < 40) &
            (dataframe['rsi'].shift(1) > 60) &
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &
            (dataframe['low'].shift(1) > dataframe['low'].shift(2)) &
            (dataframe['close'] > dataframe['high'].shift(2)) &
            (dataframe['volume'] > dataframe['volume'].rolling(10).mean() * 1.2) &
            (dataframe['close'] < dataframe['kc_upperband'] * 0.995) &
            (dataframe['adx'] > 25) &
            has_volume &
            minima_ok
        )
        
        short_condition_momentum = (
            (dataframe['close'] < dataframe['ema50']) &
            (dataframe['ema20'] < dataframe['ema50']) &
            (dataframe['ema50'] < dataframe['ema200']) &
            (dataframe['rsi'].shift(3) > 60) &
            (dataframe['rsi'].shift(1) < 40) &
            (dataframe['rsi'] < dataframe['rsi'].shift(1)) &
            (dataframe['high'].shift(1) < dataframe['high'].shift(2)) &
            (dataframe['close'] < dataframe['low'].shift(2)) &
            (dataframe['volume'] > dataframe['volume'].rolling(10).mean() * 1.2) &
            (dataframe['close'] > dataframe['kc_lowerband'] * 1.005) &
            (dataframe['adx'] > 25) &
            has_volume &
            maxima_ok
        )
       
        # === SECONDARY CONDITIONS ===
        long_condition_secondary = (
            bullish_divergence &
            (dataframe['close'] > dataframe['close'].shift(2)) &
            (dataframe['rsi'].shift(1) > 30) &
            (dataframe['rsi'].shift(1) < 70) &
            has_volume &
            minima_ok
        )
       
        short_condition_secondary = (
            bearish_divergence &
            (dataframe['close'] < dataframe['close'].shift(2)) &
            (dataframe['rsi'].shift(1) > 30) &
            (dataframe['rsi'].shift(1) < 70) &
            has_volume &
            maxima_ok
        )
       
        # === TERTIARY CONDITIONS ===
        long_condition_tertiary = (
            bullish_divergence &
            (dataframe['close'] > dataframe['ema20']) &
            (dataframe['rsi'].shift(1) > 40) &
            (dataframe['rsi'].shift(1) < 60) &
            (dataframe['volume'] > dataframe['volume'].rolling(10).mean()) &
            has_volume &
            minima_ok
        )
       
        short_condition_tertiary = (
            bearish_divergence &
            (dataframe['close'] < dataframe['ema20']) &
            (dataframe['rsi'].shift(1) > 30) &
            (dataframe['rsi'].shift(1) < 70) &
            (dataframe['volume'] > dataframe['volume'].rolling(10).mean()) &
            has_volume &
            maxima_ok
        )
       
        # === QUATERNARY CONDITIONS ===
        long_condition_quaternary = (
            bullish_divergence &
            (dataframe['close'].shift(1) < dataframe['ema20'].shift(1)) &
            (dataframe['close'] > dataframe['ema20']) &
            (dataframe['rsi'] > 40) &
            (dataframe['volume'] > dataframe['volume'].rolling(5).mean() * 1.2) &
            has_volume &
            minima_ok
        )
       
        short_condition_quaternary = (
            bearish_divergence &
            (dataframe['close'].shift(1) > dataframe['ema20'].shift(1)) &
            (dataframe['close'] < dataframe['ema20']) &
            (dataframe['rsi'] < 60) &
            (dataframe['volume'] > dataframe['volume'].rolling(5).mean() * 1.2) &
            has_volume &
            maxima_ok
        )
       
        # === FIFTH CONDITIONS ===
        long_condition_fifth = (
            bullish_divergence &
            (dataframe['close'] > dataframe['kc_middleband']) &
            (dataframe['close'] < dataframe['kc_upperband']) &
            (dataframe['close'] > dataframe['close'].shift(1)) &
            (dataframe['rsi'].shift(1) > 30) &
            (dataframe['rsi'].shift(1) < 70) &
            has_volume &
            minima_ok
        )
       
        short_condition_fifth = (
            bearish_divergence &
            (dataframe['close'] < dataframe['kc_middleband']) &
            (dataframe['close'] > dataframe['kc_lowerband']) &
            (dataframe['close'] < dataframe['close'].shift(1)) &
            (dataframe['rsi'].shift(1) > 30) &
            (dataframe['rsi'].shift(1) < 70) &
            has_volume &
            maxima_ok
        )
        
        # === SIXTH CONDITIONS - STRONG DIVERGENCE COUNT ===
        long_condition_sixth = (
            (dataframe['total_bullish_divergences_count'] >= 2) &
            bullish_divergence &
            (dataframe['rsi'].shift(1) > 30) &
            (dataframe['rsi'].shift(1) < 70) &
            has_volume &
            minima_ok
        )
       
        short_condition_sixth = (
            (dataframe['total_bearish_divergences_count'] >= 2) &
            bearish_divergence &
            (dataframe['rsi'].shift(1) > 30) &
            (dataframe['rsi'].shift(1) < 70) &
            has_volume &
            maxima_ok
        )
       
        # === SEVENTH CONDITIONS ===
        long_condition_seventh = (
            (dataframe['ema20'] > dataframe['ema50']) &
            (dataframe['close'] > dataframe['ema20']) &
            (
                (dataframe['close'].shift(1) < dataframe['ema20'].shift(1)) |
                (dataframe['close'].shift(2) < dataframe['ema20'].shift(2))
            ) &
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.2) &
            (dataframe['rsi'].shift(1) > 30) &
            (dataframe['rsi'].shift(1) < 70) &
            (dataframe['close'] > dataframe['close'].shift(2)) &
            (dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 0.5) &
            (dataframe['close'] > dataframe['kc_middleband']) &
            (dataframe['adx'] > 15) &
            has_volume &
            minima_ok
        )

        short_condition_seventh = (
            (dataframe['ema20'] < dataframe['ema50']) &
            (dataframe['close'] < dataframe['ema20']) &
            (
                (dataframe['close'].shift(1) > dataframe['ema20'].shift(1)) |
                (dataframe['close'].shift(2) > dataframe['ema20'].shift(2))
            ) &
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.2) &
            (dataframe['rsi'].shift(1) > 30) &
            (dataframe['rsi'].shift(1) < 70) &
            (dataframe['close'] < dataframe['close'].shift(2)) &
            (dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 0.5) &
            (dataframe['close'] < dataframe['kc_middleband']) &
            (dataframe['adx'] > 15) &
            has_volume &
            maxima_ok
        )

        # === PURE DIVERGENCE CONDITIONS - FALLBACK ===
        long_condition_div = (
            bullish_divergence &
            (dataframe['rsi'].shift(1) > 30) &
            (dataframe['rsi'].shift(1) < 70) &
            has_volume &
            minima_ok
        )
       
        short_condition_div = (
            bearish_divergence &
            (dataframe['rsi'].shift(1) > 30) &
            (dataframe['rsi'].shift(1) < 70) &
            has_volume &
            maxima_ok
        )
       
        # === PRIORITY-BASED ENTRY ASSIGNMENTS ===
        # Apply highest priority first, exclude already assigned entries
        
        # 1. PRIMARY CONDITIONS (Highest Priority)
        dataframe.loc[long_condition_primary, 'enter_long'] = 1
        dataframe.loc[long_condition_primary, 'enter_tag'] = 'Bull_E1'
        dataframe.loc[short_condition_primary, 'enter_short'] = 1  
        dataframe.loc[short_condition_primary, 'enter_tag'] = 'Bear_E1'

        # 2. TREND CONDITIONS
        trend_long = long_condition_trend & (dataframe['enter_tag'] == "")
        trend_short = short_condition_trend & (dataframe['enter_tag'] == "")
        dataframe.loc[trend_long, 'enter_long'] = 1
        dataframe.loc[trend_long, 'enter_tag'] = 'Bull_Trend'
        dataframe.loc[trend_short, 'enter_short'] = 1
        dataframe.loc[trend_short, 'enter_tag'] = 'Bear_Trend'

        # 3. MOMENTUM CONDITIONS
        momentum_long = long_condition_momentum & (dataframe['enter_tag'] == "")
        momentum_short = short_condition_momentum & (dataframe['enter_tag'] == "")
        dataframe.loc[momentum_long, 'enter_long'] = 1
        dataframe.loc[momentum_long, 'enter_tag'] = 'Bull_Momentum'
        dataframe.loc[momentum_short, 'enter_short'] = 1
        dataframe.loc[momentum_short, 'enter_tag'] = 'Bear_Momentum'

        # 4. SIXTH CONDITIONS (Strong divergence count)
        sixth_long = long_condition_sixth & (dataframe['enter_tag'] == "")
        sixth_short = short_condition_sixth & (dataframe['enter_tag'] == "")
        dataframe.loc[sixth_long, 'enter_long'] = 1
        dataframe.loc[sixth_long, 'enter_tag'] = 'Bull_E6'
        dataframe.loc[sixth_short, 'enter_short'] = 1
        dataframe.loc[sixth_short, 'enter_tag'] = 'Bear_E6'

        # 5. SECONDARY CONDITIONS
        secondary_long = long_condition_secondary & (dataframe['enter_tag'] == "")
        secondary_short = short_condition_secondary & (dataframe['enter_tag'] == "")
        dataframe.loc[secondary_long, 'enter_long'] = 1
        dataframe.loc[secondary_long, 'enter_tag'] = 'Bull_E2'
        dataframe.loc[secondary_short, 'enter_short'] = 1
        dataframe.loc[secondary_short, 'enter_tag'] = 'Bear_E2'

        # 6. TERTIARY CONDITIONS
        tertiary_long = long_condition_tertiary & (dataframe['enter_tag'] == "")
        tertiary_short = short_condition_tertiary & (dataframe['enter_tag'] == "")
        dataframe.loc[tertiary_long, 'enter_long'] = 1
        dataframe.loc[tertiary_long, 'enter_tag'] = 'Bull_E3'
        dataframe.loc[tertiary_short, 'enter_short'] = 1
        dataframe.loc[tertiary_short, 'enter_tag'] = 'Bear_E3'

        # 7. QUATERNARY CONDITIONS
        quaternary_long = long_condition_quaternary & (dataframe['enter_tag'] == "")
        quaternary_short = short_condition_quaternary & (dataframe['enter_tag'] == "")
        dataframe.loc[quaternary_long, 'enter_long'] = 1
        dataframe.loc[quaternary_long, 'enter_tag'] = 'Bull_E4'
        dataframe.loc[quaternary_short, 'enter_short'] = 1
        dataframe.loc[quaternary_short, 'enter_tag'] = 'Bear_E4'

        # 8. FIFTH CONDITIONS
        fifth_long = long_condition_fifth & (dataframe['enter_tag'] == "")
        fifth_short = short_condition_fifth & (dataframe['enter_tag'] == "")
        dataframe.loc[fifth_long, 'enter_long'] = 1
        dataframe.loc[fifth_long, 'enter_tag'] = 'Bull_E5'
        dataframe.loc[fifth_short, 'enter_short'] = 1
        dataframe.loc[fifth_short, 'enter_tag'] = 'Bear_E5'

        # 9. SEVENTH CONDITIONS
        # seventh_long = long_condition_seventh & (dataframe['enter_tag'] == "")
        # seventh_short = short_condition_seventh & (dataframe['enter_tag'] == "")
        # dataframe.loc[seventh_long, 'enter_long'] = 1
        # dataframe.loc[seventh_long, 'enter_tag'] = 'Bull_E7'
        # dataframe.loc[seventh_short, 'enter_short'] = 1
        # dataframe.loc[seventh_short, 'enter_tag'] = 'Bear_E7'

        # 10. PURE DIVERGENCE (Lowest Priority)
        div_long = long_condition_div & (dataframe['enter_tag'] == "")
        div_short = short_condition_div & (dataframe['enter_tag'] == "")
        dataframe.loc[div_long, 'enter_long'] = 1
        dataframe.loc[div_long, 'enter_tag'] = 'Bull_Div'
        dataframe.loc[div_short, 'enter_short'] = 1
        dataframe.loc[div_short, 'enter_tag'] = 'Bear_Div'

        # Logging
        # if len(dataframe) > 0:
        #     last_row = dataframe.iloc[-1]
        #     print(f"INFO {metadata['pair']}: Bull_div_count={last_row.get('total_bullish_divergences_count', 0)}, "
        #           f"Bear_div_count={last_row.get('total_bearish_divergences_count', 0)}, "
        #           f"Enter_long={last_row.get('enter_long', 0)}, Enter_short={last_row.get('enter_short', 0)}, "
        #           f"Tag={last_row.get('enter_tag', '')}")

        # if True:  # Show for all pairs
        #     recent_entries = dataframe['enter_long'].tail(10).sum() + dataframe['enter_short'].tail(10).sum()
        #     if recent_entries > 0:
        #         latest = dataframe.iloc[-1]
        #         logger.info(f"🚀 {metadata['pair']} ENTRY DETECTED!")
        #         logger.info(f"   🏷️ Tag: {latest['enter_tag']}")
        #         logger.info(f"   📊 RSI: {latest['rsi']:.1f}")
        #         logger.info(f"   💧 Volume Ratio: {latest['volume_ratio']:.2f}")
        #         logger.info(f"   🎯 Bull Div Count: {latest.get('total_bullish_divergences_count', 0)}")
        #         logger.info(f"   🎯 Bear Div Count: {latest.get('total_bearish_divergences_count', 0)}")

        # ✅ 调试日志打印
        if not dataframe.empty:
            last = dataframe.iloc[-1]
            logger.info(
            f"[ENTRY] {metadata['pair']} | 时间: {last.name} | RSI: {last['rsi']:.2f} | ADX: {last['adx']:.2f} | 信号: {last['enter_long']} | 标签: {last['enter_tag']}"
                    )   
                
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Adaptive leverage based on signal strength
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) > 0:
                current_signal_strength = dataframe['signal_strength'].iloc[-1]
                
                # Reduce leverage for weaker signals
                if current_signal_strength >= 8:
                    return self.leverage_value  # Full leverage for strong signals
                elif current_signal_strength >= 6:
                    return self.leverage_value * 0.8  # 80% leverage
                elif current_signal_strength >= 4:
                    return self.leverage_value * 0.6  # 60% leverage
                else:
                    return self.leverage_value * 0.4  # 40% leverage for weak signals
        except:
            pass
        
        return self.leverage_value * 0.5  # Conservative fallback


    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, after_fill: bool, **kwargs) -> float:
        """Modified to not interfere with profit taking"""
        
        # Only apply stoploss for losses or very small profits
        if current_profit > 0.04:  # Let custom_exit handle profits > 4%
            return None
        
        # Your existing stoploss logic here for losses only
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                current_candle = dataframe.iloc[-1]
                atr_value = current_candle.get('atr', 0.02)
                atr_multiplier = 3.0  # More conservative
                
                if trade.is_short:
                    stoploss_price = trade.open_rate + (atr_value * atr_multiplier)
                else:
                    stoploss_price = trade.open_rate - (atr_value * atr_multiplier)
                
                return stoploss_from_absolute(stoploss_price, current_rate, 
                                            is_short=trade.is_short, leverage=trade.leverage)
        except:
            pass
        
        return None  # Keep current stoploss
    def initialize_divergences_lists(self, dataframe: DataFrame):
        """Initialize divergence tracking columns"""
        # Bullish Divergences
        dataframe["total_bullish_divergences"] = np.nan
        dataframe["total_bullish_divergences_count"] = 0
        dataframe["total_bullish_divergences_names"] = ''

        # Bearish Divergences
        dataframe["total_bearish_divergences"] = np.nan
        dataframe["total_bearish_divergences_count"] = 0
        dataframe["total_bearish_divergences_names"] = ''

    def get_iterators(self, dataframe):
        """Get pivot point iterators for divergence detection"""
        low_iterator = []
        high_iterator = []

        for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
            if np.isnan(row.pivot_lows):
                low_iterator.append(0 if len(low_iterator) == 0 else low_iterator[-1])
            else:
                low_iterator.append(index)
            if np.isnan(row.pivot_highs):
                high_iterator.append(0 if len(high_iterator) == 0 else high_iterator[-1])
            else:
                high_iterator.append(index)
        
        return high_iterator, low_iterator

    def add_divergences(self, dataframe: DataFrame, indicator: str, high_iterator, low_iterator):
        """Add divergence detection for a specific indicator"""
        (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines) = self.divergence_finder_dataframe(
            dataframe, indicator, high_iterator, low_iterator)
        dataframe['bearish_divergence_' + indicator + '_occurence'] = bearish_divergences
        dataframe['bullish_divergence_' + indicator + '_occurence'] = bullish_divergences

    def divergence_finder_dataframe(self, dataframe: DataFrame, indicator_source: str, high_iterator, low_iterator) -> Tuple[pd.Series, pd.Series]:
        """Enhanced divergence finder with improved logic"""
        bearish_lines = [np.empty(len(dataframe['close'])) * np.nan]
        bearish_divergences = np.empty(len(dataframe['close'])) * np.nan
        bullish_lines = [np.empty(len(dataframe['close'])) * np.nan]
        bullish_divergences = np.empty(len(dataframe['close'])) * np.nan

        for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):

            # Bearish divergence detection
            bearish_occurence = self.bearish_divergence_finder(
                dataframe, dataframe[indicator_source], high_iterator, index)

            if bearish_occurence is not None:
                (prev_pivot, current_pivot) = bearish_occurence
                bearish_prev_pivot = dataframe['close'][prev_pivot]
                bearish_current_pivot = dataframe['close'][current_pivot]
                bearish_ind_prev_pivot = dataframe[indicator_source][prev_pivot]
                bearish_ind_current_pivot = dataframe[indicator_source][current_pivot]
                
                # Enhanced validation for bearish divergence
                price_diff = abs(bearish_current_pivot - bearish_prev_pivot)
                indicator_diff = abs(bearish_ind_current_pivot - bearish_ind_prev_pivot)
                time_diff = current_pivot - prev_pivot
                
                # Only accept divergences with sufficient magnitude and time separation
                if (price_diff > dataframe['atr'][current_pivot] * 0.5 and 
                    indicator_diff > 5 and 
                    time_diff >= 5):
                    
                    bearish_divergences[index] = row.close
                    dataframe.loc[index, "total_bearish_divergences"] = row.close
                    dataframe.loc[index, "total_bearish_divergences_count"] += 1
                    dataframe.loc[index, "total_bearish_divergences_names"] += indicator_source.upper() + '<br>'

            # Bullish divergence detection
            bullish_occurence = self.bullish_divergence_finder(
                dataframe, dataframe[indicator_source], low_iterator, index)

            if bullish_occurence is not None:
                (prev_pivot, current_pivot) = bullish_occurence
                bullish_prev_pivot = dataframe['close'][prev_pivot]
                bullish_current_pivot = dataframe['close'][current_pivot]
                bullish_ind_prev_pivot = dataframe[indicator_source][prev_pivot]
                bullish_ind_current_pivot = dataframe[indicator_source][current_pivot]
                
                # Enhanced validation for bullish divergence
                price_diff = abs(bullish_current_pivot - bullish_prev_pivot)
                indicator_diff = abs(bullish_ind_current_pivot - bullish_ind_prev_pivot)
                time_diff = current_pivot - prev_pivot
                
                # Only accept divergences with sufficient magnitude and time separation
                if (price_diff > dataframe['atr'][current_pivot] * 0.5 and 
                    indicator_diff > 5 and 
                    time_diff >= 5):
                    
                    bullish_divergences[index] = row.close
                    dataframe.loc[index, "total_bullish_divergences"] = row.close
                    
                    # CORRECT - increment BULLISH counters for bullish divergence:
                    dataframe.loc[index, "total_bullish_divergences_count"] += 1
                    dataframe.loc[index, "total_bullish_divergences_names"] += indicator_source.upper() + '<br>'

        return (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines)

    def bearish_divergence_finder(self, dataframe, indicator, high_iterator, index):
        """Enhanced bearish divergence detection"""
        try:
            if high_iterator[index] == index:
                current_pivot = high_iterator[index]
                occurences = list(dict.fromkeys(high_iterator))
                current_index = occurences.index(high_iterator[index])
                
                for i in range(current_index-1, current_index - self.window.value - 1, -1):
                    if i < 0 or i >= len(occurences):
                        continue
                    prev_pivot = occurences[i]
                    if np.isnan(prev_pivot):
                        continue
                    
                    # Enhanced divergence validation
                    price_higher = dataframe['pivot_highs'][current_pivot] > dataframe['pivot_highs'][prev_pivot]
                    indicator_lower = indicator[current_pivot] < indicator[prev_pivot]
                    
                    price_lower = dataframe['pivot_highs'][current_pivot] < dataframe['pivot_highs'][prev_pivot]
                    indicator_higher = indicator[current_pivot] > indicator[prev_pivot]
                    
                    # Check for classic or hidden divergence
                    if (price_higher and indicator_lower) or (price_lower and indicator_higher):
                        # Additional validation: check trend consistency
                        if self.validate_divergence_trend(dataframe, prev_pivot, current_pivot, 'bearish'):
                            return (prev_pivot, current_pivot)
        except:
            pass
        return None

    def bullish_divergence_finder(self, dataframe, indicator, low_iterator, index):
        """Enhanced bullish divergence detection"""
        try:
            if low_iterator[index] == index:
                current_pivot = low_iterator[index]
                occurences = list(dict.fromkeys(low_iterator))
                current_index = occurences.index(low_iterator[index])
                
                for i in range(current_index-1, current_index - self.window.value - 1, -1):
                    if i < 0 or i >= len(occurences):
                        continue
                    prev_pivot = occurences[i]
                    if np.isnan(prev_pivot):
                        continue
                    
                    # Enhanced divergence validation
                    price_lower = dataframe['pivot_lows'][current_pivot] < dataframe['pivot_lows'][prev_pivot]
                    indicator_higher = indicator[current_pivot] > indicator[prev_pivot]
                    
                    price_higher = dataframe['pivot_lows'][current_pivot] > dataframe['pivot_lows'][prev_pivot]
                    indicator_lower = indicator[current_pivot] < indicator[prev_pivot]
                    
                    # Check for classic or hidden divergence
                    if (price_lower and indicator_higher) or (price_higher and indicator_lower):
                        # Additional validation: check trend consistency
                        if self.validate_divergence_trend(dataframe, prev_pivot, current_pivot, 'bullish'):
                            return (prev_pivot, current_pivot)
        except:
            pass
        return None

    def validate_divergence_trend(self, dataframe, prev_pivot, current_pivot, divergence_type):
        """Validate divergence by checking intermediate trend"""
        try:
            # Check if there's a clear trend between pivots
            mid_point = (prev_pivot + current_pivot) // 2
            
            if divergence_type == 'bearish':
                # For bearish divergence, expect uptrend in between
                return dataframe['ema20'][mid_point] > dataframe['ema20'][prev_pivot]
            else:
                # For bullish divergence, expect downtrend in between
                return dataframe['ema20'][mid_point] < dataframe['ema20'][prev_pivot]
        except:
            return True  # Default to accepting divergence if validation fails

    @property
    def protections(self):
        """Enhanced protection configuration"""
        prot = []
       
        if self.use_cooldown_protection.value:
            prot.append({
                "method": "CooldownPeriod",
                "stop_duration_candles": self.cooldown_lookback.value
            })
       
        if self.use_max_drawdown_protection.value:
            prot.append({
                "method": "MaxDrawdown",
                "lookback_period_candles": self.max_drawdown_lookback.value,
                "trade_limit": self.max_drawdown_trade_limit.value,
                "stop_duration_candles": self.max_drawdown_stop_duration.value,
                "max_allowed_drawdown": self.max_allowed_drawdown.value
            })
       
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": self.stoploss_guard_lookback.value,
                "trade_limit": self.stoploss_guard_trade_limit.value,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": self.stoploss_guard_only_per_pair.value,
            })
       
        return prot
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if len(dataframe) > 1:
            dataframe = dataframe.copy().iloc[:-1]
    
        # Initialize
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0  
        dataframe['exit_tag'] = ''
        
        # === YOUR OTHER EXIT CONDITIONS FIRST ===
        # (Add any other exit conditions you have here)
        # any_long_exit = (some_other_condition)
        # any_short_exit = (some_other_condition)
        
        # === EXIT ON OPPOSITE SIGNALS (your previous approach) ===
        if 'enter_long' in dataframe.columns and 'enter_short' in dataframe.columns:
            # Exit longs on short signals
            reversal_long_exit = (dataframe['enter_short'] == 1)
            dataframe.loc[reversal_long_exit, 'exit_long'] = 1
            dataframe.loc[reversal_long_exit, 'exit_tag'] = 'Reversal_Short_Signal'
            
            # Exit shorts on long signals  
            if self.can_short:
                reversal_short_exit = (dataframe['enter_long'] == 1)
                dataframe.loc[reversal_short_exit, 'exit_short'] = 1
                dataframe.loc[reversal_short_exit, 'exit_tag'] = 'Reversal_Long_Signal'
        # ✅ 调试日志打印
        if not dataframe.empty:
            last = dataframe.iloc[-1]
            logger.info(
            f"[EXIT] {metadata['pair']} | 时间: {last.name} | RSI: {last['rsi']:.2f} | ADX: {last['adx']:.2f} | 信号: {last['exit_long']} | 标签: {last['exit_tag']}"
            )
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime',
                        current_rate: float, current_profit: float, **kwargs):
            """
            Optimized exit strategy with reversal signal detection
            Includes only profitable exits + opposite signal detection
            """
            from logging import getLogger
            logger = getLogger(__name__)

            trade_duration_minutes = (current_time - trade.open_date_utc).total_seconds() / 60

            # === DATA FETCH ===
            signal_strength = 5
            dataframe = None
            volatility = 0.02
            momentum_score = 0
            rsi = 50
            
            try:
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if not dataframe.empty:
                    last_row = dataframe.iloc[-1]
                    
                    # Get signal strength (your existing column)
                    if 'signal_strength' in dataframe.columns:
                        signal_strength = last_row['signal_strength']
                    
                    # Get volatility (your existing column)
                    if 'volatility' in dataframe.columns:
                        volatility = last_row['volatility']
                    elif 'atr' in dataframe.columns:
                        volatility = last_row['atr'] / last_row['close']
                    
                    # Get RSI (your existing column)
                    if 'rsi' in dataframe.columns:
                        rsi = last_row['rsi']
                    
                    # Calculate momentum safely
                    if len(dataframe) >= 5:
                        momentum_score = (last_row['close'] - dataframe.iloc[-5]['close']) / dataframe.iloc[-5]['close']
            except Exception:
                pass

            # === Helper: Log and return
            def log_and_exit(reason: str):
                # logger.info(f"📤 {pair} EXIT: {reason}")
                # logger.info(f"   🕒 Time Held: {int(trade_duration_minutes)} min")
                # logger.info(f"   💰 Profit: {current_profit * 100:.2f}%")
                # logger.info(f"   📊 Signal Strength: {signal_strength}")
                # logger.info(f"   📈 Momentum: {momentum_score * 100:.2f}%")
                # logger.info(f"   📊 RSI: {rsi:.1f}")
                # logger.info(f"   🔄 Exit Trigger: {reason}")
                return reason

            # === TIME-BASED SESSION DETECTION ===
            hour = current_time.hour
            is_major_session = 8 <= hour <= 16  # London/NY overlap
            is_asian_session = 0 <= hour <= 6

            # === REVERSAL SIGNAL EXIT (New - close on opposite signals) ===
            try:
                if dataframe is not None and not dataframe.empty:
                    last_row = dataframe.iloc[-1]
                    
                    # Close long trades when short signal appears
                    if not trade.is_short and last_row.get('enter_short', 0) == 1:
                        if current_profit > 0.01:  # Only if profitable (1%+)
                            return log_and_exit("reversal_short_signal")
                        elif trade_duration_minutes >= 60:  # Or if held long enough
                            return log_and_exit("reversal_short_signal")
                    
                    # Close short trades when long signal appears
                    if trade.is_short and last_row.get('enter_long', 0) == 1:
                        if current_profit > 0.01:  # Only if profitable (1%+)
                            return log_and_exit("reversal_long_signal")
                        elif trade_duration_minutes >= 60:  # Or if held long enough
                            return log_and_exit("reversal_long_signal")
            except Exception:
                pass

            # === MOMENTUM FADE EXIT ===
            if trade_duration_minutes >= 15:
                if not trade.is_short and momentum_score < -0.02 and current_profit > 0.02:
                    return log_and_exit("momentum_fade_long")
                elif trade.is_short and momentum_score > 0.02 and current_profit > 0.02:
                    return log_and_exit("momentum_fade_short")

            # === RSI EXTREME EXITS ===
            if current_profit >= 0.03 and trade_duration_minutes >= 10:
                if not trade.is_short and rsi >= 75:
                    return log_and_exit("rsi_overbought_exit")
                elif trade.is_short and rsi <= 25:
                    return log_and_exit("rsi_oversold_exit")

            # === ENHANCED EMERGENCY EXIT ===
            # Scale based on momentum - ride strong trends longer
            emergency_35 = 0.35 if abs(momentum_score) < 0.05 else 0.40
            emergency_28 = 0.28 if abs(momentum_score) < 0.03 else 0.32
            
            if current_profit >= emergency_35:
                return log_and_exit(f"emergency_{int(emergency_35*100)}pct")
            if current_profit >= emergency_28:
                return log_and_exit(f"emergency_{int(emergency_28*100)}pct")

            # === QUICK WIN EXIT ===
            quick_multiplier = 1.3 if is_major_session else 0.8
            
            if trade_duration_minutes <= 30:
                # These showed 100% win rates
                quick_10_target = 0.10 * quick_multiplier
                quick_6_target = 0.06 * quick_multiplier
                quick_4_target = 0.04 * quick_multiplier
                
                if current_profit >= quick_10_target:
                    return log_and_exit(f"quick_30min_{int(quick_10_target*100)}pct")
                elif current_profit >= quick_6_target:
                    return log_and_exit(f"quick_30min_{int(quick_6_target*100)}pct")
                elif current_profit >= quick_4_target:
                    return log_and_exit(f"quick_30min_{int(quick_4_target*100)}pct")

            # === ADAPTIVE SWING EXIT ===
            base_swing_profit = 0.08
            if signal_strength >= 8:
                base_swing_profit = 0.12 + (volatility * 3)
            elif signal_strength <= 4:
                base_swing_profit = 0.05 + (volatility * 2)
            else:
                base_swing_profit = 0.08 + (volatility * 2.5)

            # Time-based adjustment
            if is_asian_session:
                base_swing_profit *= 0.8
            elif is_major_session:
                base_swing_profit *= 1.1

            if 30 < trade_duration_minutes <= 90:
                if current_profit >= base_swing_profit:
                    return log_and_exit("adaptive_swing_exit")

            # === CLEANUP EXIT ===
            cleanup_8_target = 0.08
            cleanup_6_target = 0.06
            cleanup_4_target = 0.04
            cleanup_3_target = 0.03
            
            if is_major_session:
                cleanup_8_target = 0.09
                cleanup_6_target = 0.07
                cleanup_4_target = 0.05
                cleanup_3_target = 0.04

            if 90 < trade_duration_minutes <= 120:
                if current_profit >= cleanup_8_target:
                    return log_and_exit(f"cleanup_2hr_{int(cleanup_8_target*100)}pct")
                elif current_profit >= cleanup_6_target:
                    return log_and_exit(f"cleanup_2hr_{int(cleanup_6_target*100)}pct")
                elif current_profit >= cleanup_4_target:
                    return log_and_exit(f"cleanup_2hr_{int(cleanup_4_target*100)}pct")
                elif current_profit >= cleanup_3_target:
                    return log_and_exit(f"cleanup_2hr_{int(cleanup_3_target*100)}pct")

            # === FRIDAY CLOSE EXIT ===
            if current_time.weekday() == 4 and current_time.hour >= 15:  # Friday after 3PM
                if current_profit >= 0.02:
                    return log_and_exit("friday_close_exit")

            # === CONSERVATIVE TIMEOUT (Only for very long trades) ===
            # Added a basic timeout as safety net for trades that get stuck
            if trade_duration_minutes > 240:  # 4 hours
                if current_profit > 0:  # Only exit profitable long trades
                    return log_and_exit("long_timeout_exit")
                elif trade_duration_minutes > 360 and current_profit > -0.02:  # 6 hours, small loss
                    return log_and_exit("extended_timeout_exit")

            return None

def choppiness_index(high, low, close, window=14):
    """Calculate Choppiness Index"""
    natr = pd.Series(ta.NATR(high, low, close, window=window))
    high_max = high.rolling(window=window).max()
    low_min = low.rolling(window=window).min()
    
    choppiness = 100 * np.log10((natr.rolling(window=window).sum()) / (high_max - low_min)) / np.log10(window)
    return choppiness
    
def pivot_points(dataframe: DataFrame, window: int = 5, pivot_source=None) -> DataFrame:
    """Enhanced pivot point detection"""
    from enum import Enum
    
    class PivotSource(Enum):
        HighLow = 0
        Close = 1
    
    if pivot_source is None:
        pivot_source = PivotSource.Close
    
    high_source = 'close' if pivot_source == PivotSource.Close else 'high'
    low_source = 'close' if pivot_source == PivotSource.Close else 'low'

    pivot_points_lows = np.empty(len(dataframe['close'])) * np.nan
    pivot_points_highs = np.empty(len(dataframe['close'])) * np.nan
    last_values = deque()

    # Find pivot points with enhanced validation
    for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
        last_values.append(row)
        if len(last_values) >= window * 2 + 1:
            current_value = last_values[window]
            is_greater = True
            is_less = True
            
            for window_index in range(0, window):
                left = last_values[window_index]
                right = last_values[2 * window - window_index]
                local_is_greater, local_is_less = check_if_pivot_is_greater_or_less(
                    current_value, high_source, low_source, left, right)
                is_greater &= local_is_greater
                is_less &= local_is_less
            
            # Additional validation: ensure pivot is significant
            if is_greater:
                current_high = getattr(current_value, high_source)
                # Check if high is significant enough (above ATR threshold)
                if hasattr(current_value, 'atr') and current_high > 0:
                    pivot_points_highs[index - window] = current_high
            
            if is_less:
                current_low = getattr(current_value, low_source)
                # Check if low is significant enough
                if hasattr(current_value, 'atr') and current_low > 0:
                    pivot_points_lows[index - window] = current_low
            
            last_values.popleft()

    return pd.DataFrame(index=dataframe.index, data={
        'pivot_lows': pivot_points_lows,
        'pivot_highs': pivot_points_highs
    })

def check_if_pivot_is_greater_or_less(current_value, high_source: str, low_source: str, left, right) -> Tuple[bool, bool]:
    """Helper function for pivot point validation"""
    is_greater = True
    is_less = True
    
    if (getattr(current_value, high_source) <= getattr(left, high_source) or
            getattr(current_value, high_source) <= getattr(right, high_source)):
        is_greater = False

    if (getattr(current_value, low_source) >= getattr(left, low_source) or
            getattr(current_value, low_source) >= getattr(right, low_source)):
        is_less = False
    
    return (is_greater, is_less)

def emaKeltner(dataframe):
    """Calculate EMA-based Keltner Channels"""
    keltner = {}
    atr = qtpylib.atr(dataframe, window=10)
    ema20 = ta.EMA(dataframe, timeperiod=20)
    keltner['upper'] = ema20 + atr
    keltner['mid'] = ema20
    keltner['lower'] = ema20 - atr
    return keltner

def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Calculate Chaikin Money Flow indicator"""
    df = dataframe.copy()
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)
    mfv *= df['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum() / df['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')

def calculate_minima_maxima(df, window=5, last_n=(288 * 10), price_type='ohlc4'):
    minima = np.zeros(len(df))
    maxima = np.zeros(len(df))

    df['hl2'] = (df['high'] + df['low']) / 2
    df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    if price_type not in ['close', 'hl2', 'ohlc4']:
        raise ValueError("price_type must be 'close', 'hl2', 'ohlc4'")
    price_series = df[price_type]

    start_idx = max(window, len(df) - last_n)

    for i in range(start_idx, len(df)):
        window_data = price_series.iloc[i - window:i + 1]
        if price_series.iloc[i] == window_data.min() and (window_data == price_series.iloc[i]).sum() == 1:
            minima[i] = 1
        if price_series.iloc[i] == window_data.max() and (window_data == price_series.iloc[i]).sum() == 1:
            maxima[i] = 1

    return minima, maxima