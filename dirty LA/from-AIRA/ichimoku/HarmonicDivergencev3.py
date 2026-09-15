# --- Do NOT remove these libs ---
import datetime
from datetime import timedelta
from typing import Optional, Tuple, Union, List
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from pandas import DataFrame, Series
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import IStrategy, merge_informative_pair, timeframe_to_prev_date
from freqtrade.strategy import stoploss_from_open, stoploss_from_absolute, informative
from freqtrade.strategy import DecimalParameter, IntParameter, BooleanParameter, CategoricalParameter
from freqtrade.persistence import Trade
from freqtrade.optimize.hyperopt import IHyperOptLoss
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.exchange import date_minus_candles
# --------------------------------
# Add your lib to import here
from functools import reduce
import talib.abstract as ta
from collections import deque
import math
from enum import Enum
import logging
from ta import add_all_ta_features
from ta.utils import dropna
logger = logging.getLogger(__name__)

class PlotConfig:

    def __init__(self):
        self.config = {'main_plot': {'bollinger_upperband': {'color': 'rgba(4,137,122,0.7)'}, 'kc_upperband': {'color': 'rgba(4,146,250,0.7)'}, 'kc_middleband': {'color': 'rgba(4,146,250,0.7)'}, 'kc_lowerband': {'color': 'rgba(4,146,250,0.7)'}, 'bollinger_lowerband': {'color': 'rgba(4,137,122,0.7)', 'fill_to': 'bollinger_upperband', 'fill_color': 'rgba(4,137,122,0.07)'}, 'ema9': {'color': 'purple'}, 'ema20': {'color': 'yellow'}, 'ema50': {'color': 'red'}, 'ema200': {'color': 'white'}}, 'subplots': {'ATR': {'atr': {'color': 'firebrick'}}}}
        self.config['main_plot']['pivot_highs'] = {'plotly': {'mode': 'markers', 'marker': {'symbol': 'diamond-open', 'size': 11, 'line': {'width': 2}, 'color': 'violet'}}}
        self.config['main_plot']['pivot_lows'] = {'plotly': {'mode': 'markers', 'marker': {'symbol': 'diamond-open', 'size': 11, 'line': {'width': 2}, 'color': 'magenta'}}}
        self.config['main_plot']['bull_d'] = {'plotly': {'mode': 'markers', 'marker': {'symbol': 'triangle-up', 'size': 13, 'line': {'width': 3}, 'color': 'green'}}}
        self.config['main_plot']['bear_d'] = {'plotly': {'mode': 'markers', 'marker': {'symbol': 'triangle-down', 'size': 13, 'line': {'width': 3}, 'color': 'red'}}}
        # self.config['main_plot'] = {'sub_plot': {'total_bear_d': {'color': 'red', 'type': 'scatter'}, 'total_bull_d': {'color': 'green', 'type': 'scatter'}}}
        return None

    def add_divergence_in_config(self, indicator: str):
        # Add bullish and bearish divergence markers to the main plot configuration
        self.config['main_plot']['bull_d' + indicator + '_occurrence'] = {'type': 'scatter', 'plotly': {'mode': 'markers', 'marker': {'symbol': 'triangle-up', 'color': 'green', 'size': 10}}}
        self.config['main_plot']['bear_d' + indicator + '_occurrence'] = {'type': 'scatter', 'plotly': {'mode': 'markers', 'marker': {'symbol': 'triangle-down', 'color': 'red', 'size': 10}}}
        for i in range(3):
            self.config['main_plot']['bull_d' + indicator + '_line_' + str(i)] = {'plotly': {'mode': 'lines', 'line': {'color': 'green', 'dash': 'dash'}}}
            self.config['main_plot']['bear_d' + indicator + '_line_' + str(i)] = {'plotly': {'mode': 'lines', 'line': {'color': 'red', 'dash': 'dash'}}}
        return self

    def add_total_divergences_in_config(self, dataframe):
        total_bull_d_count = dataframe[resample('total_bull_d_count')]
        total_bull_d_names = dataframe[resample('total_bull_d_names')]
        self.config['main_plot'][resample('total_bull_d')] = {'plotly': {'mode': 'markers+text', 'text': total_bull_d_count, 'hovertext': total_bull_d_names, 'textfont': {'size': 11, 'color': 'green'}, 'textposition': 'bottom center', 'marker': {'symbol': 'diamond', 'size': 11, 'line': {'width': 2}, 'color': 'green'}}}
        total_bear_d_count = dataframe[resample('total_bear_d_count')]
        total_bear_d_names = dataframe[resample('total_bear_d_names')]
        self.config['main_plot'][resample('total_bear_d')] = {'plotly': {'mode': 'markers+text', 'text': total_bear_d_count, 'hovertext': total_bear_d_names, 'textfont': {'size': 11, 'color': 'red'}, 'textposition': 'top center', 'marker': {'symbol': 'diamond', 'size': 11, 'line': {'width': 2}, 'color': 'red'}}}
        return self

class HarmonicDivergencev3(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '15m'
    can_short = True
    
    minimal_roi = {'0': 0.01, '60': 0.005, '75': 0}
    stoploss = -0.03
    use_custom_stoploss = True
    
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.025
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True
    process_only_new_candles = False
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.03
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 512  # or 256
    
    # Strategy parameters
    buy_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space='sell', optimize=True, load=True)
    buy_long_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    short_rsi = IntParameter(low=51, high=100, default=70, space='sell', optimize=True, load=True)
    exit_long_rsi = IntParameter(low=51, high=100, default=70, space='sell', optimize=True, load=True)
    exit_short_rsi = IntParameter(low=1, high=50, default=30, space='sell', optimize=True, load=True)
    atr_window = IntParameter(low=1, high=50, default=14, space='buy', optimize=True, load=True)
    atr_exp = BooleanParameter(default=False, space='buy', optimize=True, load=True)
    atr_multiplier = DecimalParameter(low=1, high=10, default=2, space='buy', optimize=True, load=True)
    
    # DCA
    position_adjustment_enable = True
    initial_safety_order_trigger = DecimalParameter(low=-0.02, high=-0.01, default=-0.015, decimals=3, space='buy', optimize=True, load=True)
    max_safety_orders = IntParameter(1, 6, default=2, space='buy', optimize=True)
    safety_order_step_scale = DecimalParameter(low=1.05, high=1.5, default=1.25, decimals=2, space='buy', optimize=True, load=True)
    safety_order_volume_scale = DecimalParameter(low=1.1, high=2.0, default=1.5, decimals=1, space='buy', optimize=True, load=True)
    safety_order_reserve = IntParameter(1, 10, default=1, space='buy', optimize=True)
    
    ## Trailing params
    is_optimize_trailing = True
    
    # Hard stoploss profit
    pHSL = DecimalParameter(-0.2, -0.04, default=-0.08, decimals=3, space='sell', optimize=is_optimize_trailing, load=True)
    
    # Profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.02, default=0.016, decimals=3, space='sell', optimize=is_optimize_trailing, load=True)
    pSL_1 = DecimalParameter(0.008, 0.02, default=0.011, decimals=3, space='sell', optimize=is_optimize_trailing, load=True)
    
    # Profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.04, 0.1, default=0.08, decimals=3, space='sell', optimize=is_optimize_trailing, load=True)
    pSL_2 = DecimalParameter(0.02, 0.07, default=0.04, decimals=3, space='sell', optimize=is_optimize_trailing, load=True)
    
    # Unclog Function
    days = IntParameter(1, 8, default=5, space='sell', optimize=True)
    loss = DecimalParameter(-0.05, -0.015, default=-0.03, space='sell', optimize=True)

    @property
    def protections(self):
        return [{'method': 'CooldownPeriod', 'stop_duration_candles': 4}, {'method': 'MaxDrawdown', 'lookback_period_candles': 48, 'trade_limit': 20, 'stop_duration_candles': 4, 'max_allowed_drawdown': 0.2}]
    # Optional order type mapping.
    order_types = {'entry': 'limit', 'exit': 'market', 'emergency_exit': 'market', 'force_exit': 'market', 'force_entry': 'market', 'stoploss': 'market', 'stoploss_on_exchange': True, 'stoploss_on_exchange_interval': 120}
    # Optional order time in force.
    order_time_in_force = {'entry': 'gtc', 'exit': 'gtc'}
    # # Example specific variables
    max_entry_position_adjustment = 1

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])
    # @informative('15m', 'BTC/USDT:USDT')
    # @informative('15m', 'ETH/USDT:USDT')

    @informative('4h', 'BTC/USDT:USDT')
    @informative('4h', 'ETH/USDT:USDT')
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # Momentum Indicators
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['stoch'] = ta.STOCH(dataframe)['slowk']
        dataframe['roc'] = ta.ROC(dataframe)
        dataframe['uo'] = ta.ULTOSC(dataframe)
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)
        dataframe['macd'] = ta.MACD(dataframe)['macd']
        dataframe['cci'] = ta.CCI(dataframe)
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)
        dataframe['obv'] = ta.OBV(dataframe['close'], dataframe['volume'])
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['atr'] = qtpylib.atr(dataframe, window=14, exp=False)
        
        # Keltner Channel
        keltner = qtpylib.keltner_channel(dataframe, window=20, atrs=1)
        dataframe['kc_upperband'] = keltner['upper']
        dataframe['kc_middleband'] = keltner['mid']
        dataframe['kc_lowerband'] = keltner['lower']
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bollinger_upperband'] = bollinger['upper']
        dataframe['bollinger_middleband'] = bollinger['mid']
        dataframe['bollinger_lowerband'] = bollinger['lower']
        
        # EMA - Exponential Moving Average
        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        
        # TEMA - Triple Exponential Moving Average
        
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        
        # SAR - Parabolic SAR
        dataframe['sar'] = ta.SAR(dataframe)
        
        #Pivots
        pivots = pivot_points(dataframe)
        dataframe['pivot_lows'] = pivots['pivot_lows']
        dataframe['pivot_highs'] = pivots['pivot_highs']
        
        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        # # Hammer: values [0, 100]
        dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        # # Inverted Hammer: values [0, 100]
        dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
        # # Dragonfly Doji: values [0, 100]
        dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
        # # Piercing Line: values [0, 100]
        dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe)  # values [0, 100]
        # # Morningstar: values [0, 100]
        dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe)  # values [0, 100]
        # # Three White Soldiers: values [0, 100]
        dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe)  # values [0, 100]
        
        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        # # Hanging Man: values [0, 100]
        dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
        # # Shooting Star: values [0, 100]
        dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
        # # Gravestone Doji: values [0, 100]
        dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
        # # Dark Cloud Cover: values [0, 100]
        dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
        # # Evening Doji Star: values [0, 100]
        dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
        # # Evening Star: values [0, 100]
        dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)
        
        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------
        # # Three Line Strike: values [0, -100, 100]
        dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
        # # Spinning Top: values [0, -100, 100]
        dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe)  # values [0, -100, 100]
        # # Engulfing: values [0, -100, 100]
        dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe)  # values [0, -100, 100]
        # # Harami: values [0, -100, 100]
        dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe)  # values [0, -100, 100]
        # # Three Outside Up/Down: values [0, -100, 100]
        dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe)  # values [0, -100, 100]
        # # Three Inside Up/Down: values [0, -100, 100]
        dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe)  # values [0, -100, 100]
        
        # Add Ichimoku Cloud divergences
        add_divergences_ichimoku(dataframe)
        add_fibonacci_retracement_levels(dataframe)
        
        # Add Divergences
        initialize_divergences_lists(dataframe)
        add_divergences(dataframe, 'rsi')
        add_divergences(dataframe, 'stoch')
        add_divergences(dataframe, 'roc')
        add_divergences(dataframe, 'uo')
        add_divergences(dataframe, 'ao')
        add_divergences(dataframe, 'macd')
        add_divergences(dataframe, 'cci')
        add_divergences(dataframe, 'cmf')
        add_divergences(dataframe, 'obv')
        add_divergences(dataframe, 'mfi')
        add_divergences(dataframe, 'adx')
        return dataframe
    # https://www.freqtrade.io/en/stable/advanced-hyperopt/#overriding-pre-defined-spaces

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe
        assert self.dp, 'DataProvider is required for backtesting.'
        
        # Get the informative pair
        inf_tf = '4h'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        
        # Momentum Indicators
        informative['rsi'] = ta.RSI(informative, timeperiod=14)
        informative['stoch'] = ta.STOCH(informative)['slowk']
        informative['roc'] = ta.ROC(informative['close'], timeperiod=9)
        informative['uo'] = ta.ULTOSC(informative['high'], informative['low'], informative['close'])
        informative['ao'] = qtpylib.awesome_oscillator(informative)
        informative['macd'] = ta.MACD(informative)['macd']
        informative['cci'] = ta.CCI(informative)
        informative['cmf'] = chaikin_money_flow(informative, 20)
        informative['obv'] = ta.OBV(informative['close'], informative['volume'])
        informative['mfi'] = ta.MFI(informative)
        informative['adx'] = ta.ADX(informative)
        informative['atr'] = qtpylib.atr(informative, window=14, exp=False)
        
        # Keltner Channel
        keltner = qtpylib.keltner_channel(informative, window=20, atrs=1)
        informative['kc_upperband'] = keltner['upper']
        informative['kc_middleband'] = keltner['mid']
        informative['kc_lowerband'] = keltner['lower']
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
        informative['bollinger_upperband'] = bollinger['upper']
        informative['bollinger_middleband'] = bollinger['mid']
        informative['bollinger_lowerband'] = bollinger['lower']
        
        # EMA - Exponential Moving Average
        informative['ema9'] = ta.EMA(informative['close'], 9)
        informative['ema20'] = ta.EMA(informative['close'], 20)
        informative['ema50'] = ta.EMA(informative['close'], 50)
        informative['ema200'] = ta.EMA(informative['close'], 200)
        
        # TEMA - Triple Exponential Moving Average
        informative['tema'] = ta.TEMA(informative, timeperiod=9)
        
        # SAR - Parabolic SAR
        informative['sar'] = ta.SAR(informative)
        
        #Pivots
        pivots = pivot_points(informative)
        informative['pivot_lows'] = pivots['pivot_lows']
        informative['pivot_highs'] = pivots['pivot_highs']
        
        # Add Ichimoku Cloud divergences
        add_divergences_ichimoku(informative)
        add_fibonacci_retracement_levels(informative)
        initialize_divergences_lists(informative)
        add_divergences(informative, 'rsi')
        add_divergences(informative, 'stoch')
        add_divergences(informative, 'roc')
        add_divergences(informative, 'uo')
        add_divergences(informative, 'ao')
        add_divergences(informative, 'macd')
        add_divergences(informative, 'cci')
        add_divergences(informative, 'cmf')
        add_divergences(informative, 'obv')
        add_divergences(informative, 'mfi')
        add_divergences(informative, 'adx')
        
        # Momentum Indicators
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['stoch'] = ta.STOCH(dataframe)['slowk']
        dataframe['roc'] = ta.ROC(dataframe)
        dataframe['uo'] = ta.ULTOSC(dataframe)
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)
        dataframe['macd'] = ta.MACD(dataframe)['macd']
        dataframe['cci'] = ta.CCI(dataframe)
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)
        dataframe['obv'] = ta.OBV(dataframe['close'], dataframe['volume'])
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['atr'] = qtpylib.atr(dataframe, window=14, exp=False)
        
        # Keltner Channel
        keltner = qtpylib.keltner_channel(dataframe, window=20, atrs=1)
        dataframe['kc_upperband'] = keltner['upper']
        dataframe['kc_middleband'] = keltner['mid']
        dataframe['kc_lowerband'] = keltner['lower']
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bollinger_upperband'] = bollinger['upper']
        dataframe['bollinger_middleband'] = bollinger['mid']
        dataframe['bollinger_lowerband'] = bollinger['lower']
        
        # EMA - Exponential Moving Average
        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        
        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        
        # SAR - Parabolic SAR
        dataframe['sar'] = ta.SAR(dataframe)
        
        #Pivots
        pivots = pivot_points(dataframe)
        dataframe['pivot_lows'] = pivots['pivot_lows']
        dataframe['pivot_highs'] = pivots['pivot_highs']
        
        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        # # Hammer: values [0, 100]
        dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        # # Inverted Hammer: values [0, 100]
        dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
        # # Dragonfly Doji: values [0, 100]
        dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
        # # Piercing Line: values [0, 100]
        dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe)  # values [0, 100]
        # # Morningstar: values [0, 100]
        dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe)  # values [0, 100]
        # # Three White Soldiers: values [0, 100]
        dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe)  # values [0, 100]
        
        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        # # Hanging Man: values [0, 100]
        dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
        # # Shooting Star: values [0, 100]
        dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
        # # Gravestone Doji: values [0, 100]
        dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
        # # Dark Cloud Cover: values [0, 100]
        dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
        # # Evening Doji Star: values [0, 100]
        dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
        # # Evening Star: values [0, 100]
        dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)
        
        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------
        # # Three Line Strike: values [0, -100, 100]
        dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
        # # Spinning Top: values [0, -100, 100]
        dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe)  # values [0, -100, 100]
        # # Engulfing: values [0, -100, 100]
        dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe)  # values [0, -100, 100]
        # # Harami: values [0, -100, 100]
        dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe)  # values [0, -100, 100]
        # # Three Outside Up/Down: values [0, -100, 100]
        dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe)  # values [0, -100, 100]
        # # Three Inside Up/Down: values [0, -100, 100]
        dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe)  # values [0, -100, 100]
        
        # Add Ichimoku Cloud divergences
        add_divergences_ichimoku(dataframe)
        add_fibonacci_retracement_levels(dataframe)
        
        # Add Divergences
        initialize_divergences_lists(dataframe)
        add_divergences(dataframe, 'rsi')
        add_divergences(dataframe, 'stoch')
        add_divergences(dataframe, 'roc')
        add_divergences(dataframe, 'uo')
        add_divergences(dataframe, 'ao')
        add_divergences(dataframe, 'macd')
        add_divergences(dataframe, 'cci')
        add_divergences(dataframe, 'cmf')
        add_divergences(dataframe, 'obv')
        add_divergences(dataframe, 'mfi')
        add_divergences(dataframe, 'adx')
        add_confirmation_signals(dataframe)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)
        dataframe.to_csv('BackTestDF.csv', index=False)
        HarmonicDivergencev3.plot_config = PlotConfig().config
        return dataframe
    # https://www.freqtrade.io/en/stable/freqai-configuration/#setting-up-the-configuration-file
    # https://www.freqtrade.io/en/stable/advanced-hyperopt/

    # CHECK THE FIBONACCI VALUES AND APPLY CORRECT COMPARISON FOR ENTRY SIGNALS
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # Calculate EMA crosses for the entire dataframe
        ema_bullish = ~ema_cross_check(dataframe)  # Inverse of the bearish cross check for bullish confirmation
        
        # Additional checks
        keltner_middleband_bullish = keltner_middleband_check(dataframe)
        keltner_lowerband_bullish = keltner_lowerband_check(dataframe)
        bollinger_lowerband_bullish = bollinger_lowerband_check(dataframe)
        bollinger_keltner_bullish = bollinger_keltner_check(dataframe)
        ema_trend_bullish = ema_check(dataframe)
        
        # New entry trend condition for a long entry with a green candle, bullish EMA crosses, and other bullish indicators
        # Sufficient trading volume
        dataframe.loc[green_candle(dataframe) & ema_bullish & keltner_middleband_bullish & keltner_lowerband_bullish & bollinger_lowerband_bullish & bollinger_keltner_bullish & ema_trend_bullish & (dataframe['volume'] > 0), ['enter_long', 'enter_tag']] = (1, 'Green Candle Bullish')
        
        # New entry trend condition for a short entry with a red candle, bearish EMA crosses, and other bearish indicators
        # Sufficient trading volume
        dataframe.loc[~green_candle(dataframe) & ema_cross_check(dataframe) & ~keltner_middleband_bullish & ~keltner_lowerband_bullish & ~bollinger_lowerband_bullish & ~bollinger_keltner_bullish & ~ema_trend_bullish & (dataframe['volume'] > 0), ['enter_short', 'enter_tag']] = (1, 'Red Candle Bearish')
        
        # Above and Below
        dataframe.loc[qtpylib.crossed_above(dataframe['rsi_4h'], self.buy_long_rsi.value) & (dataframe['tema_4h'] <= dataframe['bollinger_middleband_4h']) & (dataframe['tema_4h'] > dataframe['tema'].shift(1)) & (dataframe['volume'] > 0), ['enter_long', 'enter_tag']] = (1, 'Above')
        dataframe.loc[qtpylib.crossed_below(dataframe['rsi_4h'], self.short_rsi.value) & (dataframe['tema_4h'] > dataframe['bollinger_middleband_4h']) & (dataframe['tema_4h'] < dataframe['tema'].shift(1)) & (dataframe['volume'] > 0), ['enter_short', 'enter_tag']] = (1, 'Below')
        
        # # Detect bullish trend entries with Fibonacci conditions on both timeframes
        dataframe.loc[
            (dataframe['high'] > dataframe['high_4h'].shift()) &
            (dataframe['total_bull_d'].shift() > 0) &
            two_bands_check(dataframe) &
            (dataframe['volume'] > 0) &
            ((dataframe['low'] <= dataframe['fib_61.8%']) | (dataframe['low'] <= dataframe['fib_38.2%'])) &
            ((dataframe['low_4h'] <= dataframe['fib_61.8%_4h']) | (dataframe['low_4h'] <= dataframe['fib_38.2%_4h'])) &
            (dataframe['bullish_confirmation_signal'] == 1), ['enter_long', 'enter_tag']] = (1, 'Bullish Divergence with Fibonacci')

        # Detect bearish trend entries with Fibonacci conditions on both timeframes
        dataframe.loc[
            (dataframe['low'] < dataframe['low_4h'].shift()) &
            (dataframe['total_bear_d'].shift() > 0) &
            two_bands_check(dataframe) &
            (dataframe['volume'] > 0) &
            ((dataframe['high'] >= dataframe['fib_61.8%']) | (dataframe['high'] >= dataframe['fib_38.2%'])) &
            ((dataframe['high_4h'] >= dataframe['fib_61.8%_4h']) | (dataframe['high_4h'] >= dataframe['fib_38.2%_4h'])) &
            (dataframe['bearish_confirmation_signal'] == 1), ['enter_short', 'enter_tag']] = (1, 'Bearish Divergence with Fibonacci')

        # Detect bullish trend entries
        dataframe.loc[
            (dataframe['high'] > dataframe['high_4h'].shift()) &
            (dataframe['total_bull_d'].shift() > 0) &
            (dataframe['total_bull_d_4h'].shift() > 0) &
            two_bands_check(dataframe) &
            (dataframe['volume'] > 0) &
            (dataframe['bullish_confirmation_signal'] == 1), ['enter_long', 'enter_tag']] = (1, 'Bullish Divergence 4h')

        # Detect bearish trend entries
        dataframe.loc[
            (dataframe['low'] < dataframe['low_4h'].shift()) &
            (dataframe['total_bear_d'].shift() > 0) &
            (dataframe['total_bear_d_4h'].shift() > 0) &
            two_bands_check(dataframe) &
            (dataframe['volume'] > 0) &
            (dataframe['bearish_confirmation_signal'] == 1), ['enter_short', 'enter_tag']] = (1, 'Bearish Divergence 4h')

        # Detect bullish trend entries when both standard and 4-hour timeframe divergences are present
        dataframe.loc[(dataframe['high'] > dataframe['high_4h'].shift()) & (dataframe['total_bull_d'].shift() > 0) & (dataframe['total_bull_d_4h'].shift() > 0) & (dataframe['eth_usdt_total_bull_d_4h'].shift() > 0) & (dataframe['btc_usdt_total_bull_d_4h'].shift() > 0) & two_bands_check(dataframe) & (dataframe['volume'] > 0), ['enter_long', 'enter_tag']] = (1, 'Harmonic Bullish Divergence Both Timeframes')
        
        # Detect bearish trend entries when both standard and 4-hour timeframe divergences are present
        dataframe.loc[(dataframe['low'] < dataframe['low_4h'].shift()) & (dataframe['total_bear_d'].shift() > 0) & (dataframe['total_bear_d_4h'].shift() > 0) & (dataframe['eth_usdt_total_bear_d_4h'].shift() > 0) & (dataframe['btc_usdt_total_bear_d_4h'].shift() > 0) & two_bands_check(dataframe) & (dataframe['volume'] > 0), ['enter_short', 'enter_tag']] = (1, 'Harmonic Bearish Divergence Both Timeframes')
        
        # Detect Ichimoku bullish trend entries when all three divergences are present on both standard and 4-hour timeframes
        # Sufficient trading volume
        dataframe.loc[(dataframe[['kijun_bullish_divergence', 'senkou_span_a_bullish_divergence', 'senkou_span_b_bullish_divergence']].sum(axis=1) == 3) & (dataframe[['kijun_bullish_divergence_4h', 'senkou_span_a_bullish_divergence_4h', 'senkou_span_b_bullish_divergence_4h']].sum(axis=1) == 3) & (dataframe['volume'] > 0), ['enter_long', 'enter_tag']] = (1, 'Ichimoku Bullish Divergence Both Timeframes')
        
        # Detect Ichimoku bullish trend entries when all three divergences are present on both standard and 4-hour timeframes
        # Sufficient trading volume
        dataframe.loc[(dataframe[['kijun_bearish_divergence', 'senkou_span_a_bearish_divergence', 'senkou_span_b_bearish_divergence']].sum(axis=1) == 3) & (dataframe[['kijun_bearish_divergence_4h', 'senkou_span_a_bearish_divergence_4h', 'senkou_span_b_bearish_divergence_4h']].sum(axis=1) == 3) & (dataframe['volume'] > 0), ['enter_short', 'enter_tag']] = (1, 'Ichimoku Bearish Divergence Both Timeframes')
        
        # Define bearish conditions for Stochastic Oscillator and ADX
        # A bearish signal for the Stochastic Oscillator might be when it is falling from overbought levels
        stoch_bearish = (dataframe['stoch'] < 20) & (dataframe['btc_usdt_stoch_4h'] < 20) & (dataframe['eth_usdt_stoch_4h'] < 20)
        stoch_bearish_4h = (dataframe['btc_usdt_stoch_4h'] < 20) & (dataframe['eth_usdt_stoch_4h'] < 20)
        
        # A strong but not extremely strong trend for the ADX is indicated by a value above 25 but below 50
        adx_strong_trend = (dataframe['adx'] < 50) & (dataframe['adx_4h'] < 50) & (dataframe['btc_usdt_adx_4h'] < 50) & (dataframe['eth_usdt_adx_4h'] < 50)
        adx_strong_trend_4h = (dataframe['btc_usdt_adx_4h'] < 50) & (dataframe['eth_usdt_adx_4h'] < 50)
        
        # Define bulishh conditions for Stochastic Oscillator and ADX
        # A bullish signal for the Stochastic Oscillator
        stoch_bullish = (dataframe['stoch'] > 80) & (dataframe['btc_usdt_stoch_4h'] > 80) & (dataframe['eth_usdt_stoch_4h'] > 80)
        stoch_bullish_4h = (dataframe['btc_usdt_stoch_4h'] > 80) & (dataframe['eth_usdt_stoch_4h'] > 80)
        
        # A weak but not extremely weak trend for the ADX
        adx_weak_trend = (dataframe['adx'] < 25) & (dataframe['adx_4h'] < 25)
        adx_weak_trend_4h = (dataframe['btc_usdt_adx_4h'] < 25) & (dataframe['eth_usdt_adx_4h'] < 25)
        
        # Detect Ichimoku bearish trend entries when divergences are present on both standard and 4-hour timeframes
        # and when Stochastic Oscillator and ADX confirm the bearish trend
        # Sufficient trading volume
        dataframe.loc[(dataframe['eth_usdt_kijun_bearish_divergence_4h'] & dataframe['eth_usdt_senkou_span_a_bearish_divergence_4h'] & dataframe['eth_usdt_senkou_span_b_bearish_divergence_4h'] & dataframe['eth_usdt_chikou_bearish_divergence_4h'] & dataframe['btc_usdt_kijun_bearish_divergence_4h'] & dataframe['btc_usdt_senkou_span_a_bearish_divergence_4h'] & dataframe['btc_usdt_senkou_span_b_bearish_divergence_4h'] & dataframe['btc_usdt_chikou_bearish_divergence_4h'] & dataframe['kijun_bearish_divergence'] & dataframe['kijun_bearish_divergence_4h'] | dataframe['senkou_span_a_bearish_divergence'] & dataframe['senkou_span_a_bearish_divergence_4h'] | dataframe['senkou_span_b_bearish_divergence'] & dataframe['senkou_span_b_bearish_divergence_4h'] | dataframe['chikou_bearish_divergence'] & dataframe['chikou_bearish_divergence_4h']) & stoch_bearish & stoch_bearish_4h & adx_strong_trend & adx_strong_trend_4h & (dataframe['volume'] > 0), ['enter_short', 'enter_tag']] = (1, 'Ichimoku Bearish Divergence with Stoch and ADX Both Timeframes')  # Sufficient trading volume
        dataframe.loc[(dataframe['eth_usdt_kijun_bullish_divergence_4h'] & dataframe['eth_usdt_senkou_span_a_bullish_divergence_4h'] & dataframe['eth_usdt_senkou_span_b_bullish_divergence_4h'] & dataframe['eth_usdt_chikou_bullish_divergence_4h'] & dataframe['btc_usdt_kijun_bullish_divergence_4h'] & dataframe['btc_usdt_senkou_span_a_bullish_divergence_4h'] & dataframe['btc_usdt_senkou_span_b_bullish_divergence_4h'] & dataframe['btc_usdt_chikou_bullish_divergence_4h'] & dataframe['kijun_bullish_divergence'] & dataframe['kijun_bullish_divergence_4h'] | dataframe['senkou_span_a_bullish_divergence'] & dataframe['senkou_span_a_bullish_divergence_4h'] | dataframe['senkou_span_b_bullish_divergence'] & dataframe['senkou_span_b_bullish_divergence_4h'] | dataframe['chikou_bullish_divergence'] & dataframe['chikou_bullish_divergence_4h']) & stoch_bullish & stoch_bullish_4h & adx_weak_trend & adx_weak_trend_4h & (dataframe['volume'] > 0), ['enter_long', 'enter_tag']] = (1, 'Ichimoku Bullish Divergence with Stoch and ADX Both Timeframes')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit condition for long positions when RSI is too high
        dataframe.loc[qtpylib.crossed_above(dataframe['rsi'], self.buy_rsi.value) & (dataframe['tema'] > dataframe['bollinger_middleband']) & (dataframe['tema'] < dataframe['tema'].shift(1)) & (dataframe['volume'] > 0), ['exit_long', 'exit_tag']] = (1, 'RSI too high')
        
        # Exit condition for short positions when RSI is too low
        dataframe.loc[qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi.value) & (dataframe['tema'] <= dataframe['bollinger_middleband']) & (dataframe['tema'] > dataframe['tema'].shift(1)) & (dataframe['volume'] > 0), ['exit_short', 'exit_tag']] = (1, 'RSI too low')
        
        # Dynamic exit condition for long positions based on RSI
        dataframe.loc[qtpylib.crossed_above(dataframe['rsi'], self.exit_long_rsi.value) & (dataframe['tema'] > dataframe['bollinger_middleband']) & (dataframe['tema'] < dataframe['tema'].shift(1)) & (dataframe['volume'] > 0), ['exit_short', 'exit_tag']] = (1, 'Dynamic exit condition for short positions based on RSI')
        
        # Dynamic exit condition for short positions based on RSI
        dataframe.loc[qtpylib.crossed_above(dataframe['rsi'], self.exit_short_rsi.value) & (dataframe['tema'] <= dataframe['bollinger_middleband']) & (dataframe['tema'] > dataframe['tema'].shift(1)) & (dataframe['volume'] > 0), ['exit_long', 'exit_tag']] = (1, 'Dynamic exit condition for long positions based on RSI')
        
        # Exit long positions when a bearish divergence is detected on both timeframes
        # Bearish divergence on standard timeframe
        # Bearish divergence on 4-hour timeframe
        # Sufficient trading volume
        dataframe.loc[(dataframe['total_bear_d'].shift() > 0) & (dataframe['total_bear_d_4h'].shift() > 0) & (dataframe['volume'] > 0), ['exit_long', 'exit_tag']] = (1, 'Counter Bearish Divergence Detected')
        
        # Exit short positions when a bullish divergence is detected on both timeframes
        # Bullish divergence on standard timeframe
        # Bullish divergence on 4-hour timeframe
        # Sufficient trading volume
        dataframe.loc[(dataframe['total_bull_d'].shift() > 0) & (dataframe['total_bull_d_4h'].shift() > 0) & (dataframe['volume'] > 0), ['exit_short', 'exit_tag']] = (1, 'Counter Bullish Divergence Detected')
        return dataframe
    # https://www.freqtrade.io/en/stable/strategy-callbacks/#adjust-trade-position
    # Define the parameter spaces
    max_epa = CategoricalParameter([-1, 0, 1, 3, 5, 10], default=1, space='buy', optimize=True)

    @property
    def max_entry_position_adjustment(self):
        return self.max_epa.value

    # def confirm_trade_entry(
    #     self,
    #     pair: str,
    #     order_type: str,
    #     amount: float,
    #     rate: float,
    #     time_in_force: str,
    #     current_time: datetime,
    #     entry_tag: Optional[str],
    #     side: str,
    #     **kwargs
    # ) -> bool:

    #     open_trades = Trade.get_trades(trade_filter=Trade.is_open.is_(True))

    #     num_shorts, num_longs = 0, 0
    #     for trade in open_trades:
    #         if "short" in trade.enter_tag:
    #             num_shorts += 1
    #         elif "long" in trade.enter_tag:
    #             num_longs += 1

    #     if side == "long" and num_longs >= 5:
    #         return False

    #     if side == "short" and num_shorts >= 5:
    #         return False

    #     df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     last_candle = df.iloc[-1].squeeze()

    #     if side == "long":
    #         if rate > (last_candle["close"] * (1 + 0.0025)):
    #             return False
    #     else:
    #         if rate < (last_candle["close"] * (1 - 0.0025)):
    #             return False

    #     return True

# THE ADJUSTED TRADE POSITION BASED ON NEW = SIGNALS SHOULD BE INCREASING LEVERAGE AND STAKE AMOUNT - RE-CHECK
    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, min_stake: Optional[float], max_stake: float, current_entry_rate: float, current_exit_rate: float, current_entry_profit: float, current_exit_profit: float, **kwargs) -> Optional[float]:
        """
        Adjust the position of a futures trade based on a new entry signal.
        """
        # Get the analyzed dataframe for the trade's pair
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # Check if there are at least two candles to compare
        if len(dataframe) > 2:
            last_candle = dataframe.iloc[-1].squeeze()
            previous_candle = dataframe.iloc[-2].squeeze()
            signal_name = 'enter_long' if not trade.is_short else 'enter_short'
            prior_date = date_minus_candles(self.timeframe, 1, current_time)
            # Check for a new entry signal and other conditions
            if last_candle[signal_name] == 1 and previous_candle[signal_name] != 1 and (trade.nr_of_successful_entries < 2) and (trade.orders[-1].order_date_utc < prior_date):
                # Calculate the adjusted leverage
                proposed_leverage = trade.leverage  # Current leverage of the trade
                side = 'long' if not trade.is_short else 'short'
                # Define a default max_leverage if it's not provided
                default_max_leverage = 50  # This is an example value; adjust as needed for your strategy
                adjusted_leverage = self.leverage(trade.pair, current_time, current_rate, proposed_leverage, default_max_leverage, None, side)
                # Calculate the new position size based on the adjusted leverage
                new_position_size = trade.stake_amount * (adjusted_leverage / proposed_leverage)
                # Ensure the new position size does not exceed the maximum stake
                new_position_size = min(new_position_size, max_stake)
                print(new_position_size)
                # Return the new position size
                return new_position_size
        # If no adjustments are to be made, return None
        return None

    ### Trailing Stop ###
    ## Custom Trailing stoploss ( credit to Perkmeister for this custom stoploss to help the strategy ride a green candle )

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        # Fetch hyperparameters 
        hsl = self.pHSL.value
        pf_1 = self.pPF_1.value
        sl_1 = self.pSL_1.value
        pf_2 = self.pPF_2.value
        sl_2 = self.pSL_2.value
        # 1. Profit-Based Trailing Stop-Loss
        if current_profit > pf_2:
            sl_profit = sl_2 + (current_profit - pf_2)
        elif current_profit > pf_1:
            sl_profit = sl_1 + (current_profit - pf_1) * (sl_2 - sl_1) / (pf_2 - pf_1)
        else:
            sl_profit = hsl
        if sl_profit >= current_profit:
            return -0.50
        # 2. Time-Based Tiered Trailing (Fallback)
        four_hour_threshold = trade.open_date_utc + timedelta(minutes=240)
        eight_hour_threshold = trade.open_date_utc + timedelta(minutes=480)
        if current_time >= four_hour_threshold:
            return current_profit - 0.02  # Tight trailing
        elif current_time >= eight_hour_threshold:
            return current_profit - 0.05  # Intermediate trailing
        else:
            # Initial or Profit-Based will handle stop-loss
            return stoploss_from_open(sl_profit, current_profit)

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        # Sell any positions at a loss if they are held for more than 7 days.
        if current_profit < self.loss.value and (current_time - trade.open_date_utc).days >= self.days.value:
            return 'unclog'

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        # Ensure max_leverage is not None and is a float
        if max_leverage is None:
            raise ValueError('Max leverage cannot be None')
        if not isinstance(max_leverage, (int, float)):
            raise ValueError('Max leverage must be a number')
        base_leverage = 3
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        bullish_patterns = ['CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLPIERCING', 'CDLMORNINGSTAR', 'CDL3WHITESOLDIERS']
        bearish_patterns = ['CDLHANGINGMAN', 'CDLSHOOTINGSTAR', 'CDLGRAVESTONEDOJI', 'CDLDARKCLOUDCOVER', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR']
        bullish_bearish_patterns = ['CDL3LINESTRIKE', 'CDLSPINNINGTOP', 'CDLENGULFING', 'CDLHARAMI', 'CDL3OUTSIDE', 'CDL3INSIDE']
        pattern_found = False  # Flag to check if any pattern is found
        
        if side == 'long':
            for pattern in bullish_patterns:
                pattern_value = current_candle.get(pattern)
                if pattern_value is not None and pattern_value == 100:
                    # Bullish signal
                    base_leverage = max(1.0, min(base_leverage * 1.5, max_leverage))
                    print(f'Bullish Pattern: {pattern} Value: {pattern_value} Adjusted Leverage: {base_leverage}')
                    pattern_found = True
                    break
        
        elif side == 'short':
            for pattern in bearish_patterns:
                pattern_value = current_candle.get(pattern)
                if pattern_value is not None and pattern_value == 100:
                    # Bearish signal
                    base_leverage = max(1.0, min(base_leverage * 1.5, max_leverage))
                    print(f'Bearish Pattern: {pattern} Value: {pattern_value} Adjusted Leverage: {base_leverage}')
                    pattern_found = True
                    break
        
        if not pattern_found and side == 'long':
            for pattern in bullish_bearish_patterns:
                pattern_value = current_candle.get(pattern)
                if pattern_value is not None and pattern_value == 100:
                    # Bullish signal
                    base_leverage = max(1.0, min(base_leverage * 2, max_leverage))
                    print(f'Bullish Pattern: {pattern} Value: {pattern_value} Adjusted Leverage: {base_leverage}')
                    pattern_found = True
                    break
        
        elif not pattern_found and side == 'short':
            for pattern in bullish_bearish_patterns:
                pattern_value = current_candle.get(pattern)
                if pattern_value is not None and pattern_value == -100:
                    # Bearish signal
                    base_leverage = max(1.0, min(base_leverage * 2, max_leverage))
                    print(f'Bearish Pattern: {pattern} Value: {pattern_value} Adjusted Leverage: {base_leverage}')
                    pattern_found = True
                    break
        
        # Apply maximum and minimum limits if any pattern is found
        if pattern_found:
            adjusted_leverage = max(min(base_leverage, max_leverage), 1.0)  # Apply max and min limits
            print(f'Base Leverage: {proposed_leverage}')
            print(f'Adjusted Leverage: {adjusted_leverage}')
            return adjusted_leverage  # Return the adjusted leverage
        
        # If no pattern matches, return the proposed_leverage
        print(f'No Pattern Matched. Using Proposed Leverage: {proposed_leverage}')
        return proposed_leverage

def resample(indicator):
    return indicator

def two_bands_check(dataframe):
    check = (dataframe['low'] < dataframe['kc_lowerband']) & (dataframe['high'] > dataframe['kc_upperband'])
    return ~check

def ema_cross_check(dataframe):
    dataframe['ema20_50_cross'] = qtpylib.crossed_below(dataframe['ema20'], dataframe['ema50'])
    dataframe['ema20_200_cross'] = qtpylib.crossed_below(dataframe['ema20'], dataframe['ema200'])
    dataframe['ema50_200_cross'] = qtpylib.crossed_below(dataframe['ema50'], dataframe['ema200'])
    return ~(dataframe['ema20_50_cross'] | dataframe['ema20_200_cross'] | dataframe['ema50_200_cross'])

def green_candle(dataframe):
    return dataframe['open_4h'] < dataframe['close_4h']

def keltner_middleband_check(dataframe):
    return (dataframe['low_4h'] < dataframe['kc_middleband_4h']) & (dataframe['high_4h'] > dataframe['kc_middleband_4h'])

def keltner_lowerband_check(dataframe):
    return (dataframe['low_4h'] < dataframe['kc_lowerband_4h']) & (dataframe['high_4h'] > dataframe['kc_lowerband_4h'])

def bollinger_lowerband_check(dataframe):
    return (dataframe['low_4h'] < dataframe['bollinger_lowerband_4h']) & (dataframe['high_4h'] > dataframe['bollinger_lowerband_4h'])

def bollinger_keltner_check(dataframe):
    return (dataframe['bollinger_lowerband_4h'] < dataframe['kc_lowerband_4h']) & (dataframe['bollinger_upperband_4h'] > dataframe['kc_upperband_4h'])

def ema_check(dataframe):
    check = (dataframe['ema9_4h'] < dataframe['ema20_4h']) & (dataframe['ema20_4h'] < dataframe['ema50_4h']) & (dataframe['ema50_4h'] < dataframe['ema200_4h'])
    return ~check

class PivotSource(Enum):
    HighLow = 0
    Close = 1

def pivot_points(dataframe: pd.DataFrame, window: int = 5, pivot_source: PivotSource = PivotSource.Close) -> pd.DataFrame:
    if pivot_source == PivotSource.Close:
        high_source = 'close'
        low_source = 'close'
    elif pivot_source == PivotSource.HighLow:
        high_source = 'high'
        low_source = 'low'
    pivot_points_lows = np.empty(len(dataframe)) * np.nan
    pivot_points_highs = np.empty(len(dataframe)) * np.nan
    last_values = deque()

    # Find pivot points
    for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
        last_values.append(row)
        if len(last_values) >= window * 2 + 1:
            current_value = last_values[window]
            is_greater, is_less = True, True
            for window_index in range(window):
                left = last_values[window_index]
                right = last_values[2 * window - window_index]
                is_greater = is_greater and getattr(current_value, high_source) > max(getattr(left, high_source), getattr(right, high_source))
                is_less = is_less and getattr(current_value, low_source) < min(getattr(left, low_source), getattr(right, low_source))
            if is_greater:
                pivot_points_highs[index - window] = getattr(current_value, high_source)
            if is_less:
                pivot_points_lows[index - window] = getattr(current_value, low_source)
            last_values.popleft()

    # Find last one
    if len(last_values) >= window + 2:
        current_value = last_values[-2]
        is_greater, is_less = True, True
        for window_index in range(window):
            left = last_values[-2 - window_index - 1]
            right = last_values[-1]
            is_greater = is_greater and getattr(current_value, high_source) > max(getattr(left, high_source), getattr(right, high_source))
            is_less = is_less and getattr(current_value, low_source) < min(getattr(left, low_source), getattr(right, low_source))
        if is_greater:
            pivot_points_highs[index - 1] = getattr(current_value, high_source)
        if is_less:
            pivot_points_lows[index - 1] = getattr(current_value, low_source)
    return pd.DataFrame(index=dataframe.index, data={'pivot_lows': pivot_points_lows, 'pivot_highs': pivot_points_highs})

def check_if_pivot_is_greater_or_less(current_value, high_source: str, low_source: str, left, right) -> Tuple[bool, bool]:
    is_greater = getattr(current_value, high_source) > max(getattr(left, high_source), getattr(right, high_source))
    is_less = getattr(current_value, low_source) < min(getattr(left, low_source), getattr(right, low_source))
    return is_greater, is_less

def initialize_divergences_lists(dataframe: pd.DataFrame):
    columns = ['total_bull_d', 'total_bear_d']
    for col in columns:
        dataframe[col] = np.zeros(len(dataframe['close']))
        count_col = col + '_count'
        dataframe[count_col] = np.zeros(len(dataframe['close']), dtype=int)
        dataframe[col + '_names'] = ['' for _ in range(len(dataframe['close']))]

def add_divergences(dataframe: pd.DataFrame, indicator: str):
    bearish_divergences, bearish_lines, bullish_divergences, bullish_lines = divergence_finder_dataframe(dataframe, indicator)
    dataframe['bear_d_' + indicator + '_occurrence'] = bearish_divergences
    dataframe['bull_d_' + indicator + '_occurrence'] = bullish_divergences

def divergence_finder_dataframe(dataframe: pd.DataFrame, indicator_source: str) -> Tuple[pd.Series, list, pd.Series, list]:
    bearish_divergences = np.zeros(dataframe.shape[0])
    bullish_divergences = np.zeros(dataframe.shape[0])
    bearish_lines = [[] for _ in range(dataframe.shape[0])]
    bullish_lines = [[] for _ in range(dataframe.shape[0])]
    total_bear_d = 'total_bear_d'
    total_bear_d_count = 'total_bear_d_count'
    total_bear_d_names = 'total_bear_d_names'
    total_bull_d = 'total_bull_d'
    total_bull_d_count = 'total_bull_d_count'
    total_bull_d_names = 'total_bull_d_names'
    
    # Ensure the necessary columns exist in the dataframe
    for col in [total_bear_d, total_bear_d_count, total_bear_d_names, total_bull_d, total_bull_d_count, total_bull_d_names]:
        if col not in dataframe.columns:
            dataframe[col] = np.nan
            
    low_iterator = [0 if np.isnan(value) else index for index, value in enumerate(dataframe['pivot_lows'])]
    high_iterator = [0 if np.isnan(value) else index for index, value in enumerate(dataframe['pivot_highs'])]

    rsi = dataframe['rsi']
    macd = dataframe['macd']
    stoch = dataframe['stoch']
    cci = dataframe['cci']
    roc = dataframe['roc']
    uo = dataframe['uo']
    ao = dataframe['ao']
    obv = dataframe['obv']
    mfi = dataframe['mfi']
    adx = dataframe['adx']
    
    for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
        bearish_occurrence = bearish_divergence_finder(dataframe, indicator_source, high_iterator, index, rsi, macd, stoch, cci, roc, uo, ao, obv, mfi, adx)
        if bearish_occurrence:
            prev_pivot, current_pivot = bearish_occurrence
            bearish_prev_pivot = dataframe['close'].iloc[prev_pivot]
            bearish_current_pivot = dataframe['close'].iloc[current_pivot]
            bearish_ind_prev_pivot = dataframe[indicator_source].iloc[prev_pivot]
            bearish_ind_current_pivot = dataframe[indicator_source].iloc[current_pivot]
            length = current_pivot - prev_pivot
            can_exist = True
            for i in range(length + 1):
                point = bearish_prev_pivot + (bearish_current_pivot - bearish_prev_pivot) * i / length
                indicator_point = bearish_ind_prev_pivot + (bearish_ind_current_pivot - bearish_ind_prev_pivot) * i / length
                if i != 0 and i != length:
                    if point <= dataframe['close'].iloc[prev_pivot + i] or indicator_point <= dataframe[indicator_source].iloc[prev_pivot + i]:
                        can_exist = False
                if can_exist:
                    bearish_lines[index].append(point)
            if can_exist:
                bearish_divergences[index] = row.close  # Set divergence value
                dataframe.at[index, total_bear_d] = row.close
                dataframe.at[index, total_bear_d_count] += 1
                dataframe.at[index, total_bear_d_names] += indicator_source.upper() + '-'
        bullish_occurrence = bullish_divergence_finder(dataframe, indicator_source, low_iterator, index, rsi, macd, stoch, cci, roc, uo, ao, obv, mfi, adx)
        if bullish_occurrence:
            prev_pivot, current_pivot = bullish_occurrence
            bullish_prev_pivot = dataframe['close'].iloc[prev_pivot]
            bullish_current_pivot = dataframe['close'].iloc[current_pivot]
            bullish_ind_prev_pivot = dataframe[indicator_source].iloc[prev_pivot]
            bullish_ind_current_pivot = dataframe[indicator_source].iloc[current_pivot]
            length = current_pivot - prev_pivot
            can_exist = True
            for i in range(length + 1):
                point = bullish_prev_pivot + (bullish_current_pivot - bullish_prev_pivot) * i / length
                indicator_point = bullish_ind_prev_pivot + (bullish_ind_current_pivot - bullish_ind_prev_pivot) * i / length
                if i != 0 and i != length:
                    if point >= dataframe['close'].iloc[prev_pivot + i] or indicator_point >= dataframe[indicator_source].iloc[prev_pivot + i]:
                        can_exist = False
                if can_exist:
                    bullish_lines[index].append(point)
            if can_exist:
                bullish_divergences[index] = row.close
                dataframe.at[index, total_bull_d] = row.close
                dataframe.at[index, total_bull_d_count] += 1
                dataframe.at[index, total_bull_d_names] += indicator_source.upper() + '-'
    return (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines)

def bearish_divergence_finder(dataframe, indicator, high_iterator, index, rsi, macd, stoch, cci, roc, uo, ao, obv, mfi, adx):
    if high_iterator[index] == index:
        current_pivot = high_iterator[index]
        occurrences = list(dict.fromkeys(high_iterator))
        current_index = occurrences.index(high_iterator[index])
        strongest_divergence = None
        max_discrepancy = 0
        for i in range(current_index - 1, max(current_index - 5, -1), -1):
            prev_pivot = occurrences[i]
            if np.isnan(prev_pivot):
                continue
            price_difference = dataframe['high'].iloc[current_pivot] - dataframe['high'].iloc[prev_pivot]
            discrepancies = {'rsi': abs(rsi.iloc[current_pivot] - rsi.iloc[prev_pivot]), 
                            'macd': abs(macd.iloc[current_pivot] - macd.iloc[prev_pivot]), 
                            'stoch': abs(stoch.iloc[current_pivot] - stoch.iloc[prev_pivot]), 
                            'cci': abs(cci.iloc[current_pivot] - cci.iloc[prev_pivot]),
                            'roc': abs(roc.iloc[current_pivot] - roc.iloc[prev_pivot]),
                            'uo': abs(uo.iloc[current_pivot] - uo.iloc[prev_pivot]),
                            'ao': abs(ao.iloc[current_pivot] - ao.iloc[prev_pivot]),
                            'obv': abs(obv.iloc[current_pivot] - obv.iloc[prev_pivot]),
                            'mfi': abs(mfi.iloc[current_pivot] - mfi.iloc[prev_pivot]),
                            'adx': abs(adx.iloc[current_pivot] - adx.iloc[prev_pivot])}
            for indicator, discrepancy in discrepancies.items():
                if discrepancy > max_discrepancy:
                    max_discrepancy = discrepancy
                    strongest_divergence = (prev_pivot, current_pivot)
        return strongest_divergence
    return None

def bullish_divergence_finder(dataframe, indicator, low_iterator, index, rsi, macd, stoch, cci, roc, uo, ao, obv, mfi, adx):
    if low_iterator[index] == index:
        current_pivot = low_iterator[index]
        occurrences = list(dict.fromkeys(low_iterator))
        current_index = occurrences.index(low_iterator[index])
        strongest_divergence = None
        max_discrepancy = 0
        for i in range(current_index - 1, max(current_index - 7, -1), -1):
            prev_pivot = occurrences[i]
            if np.isnan(prev_pivot):
                continue
            price_difference = dataframe['low'].iloc[prev_pivot] - dataframe['low'].iloc[current_pivot]
            discrepancies = {'rsi': abs(rsi.iloc[prev_pivot] - rsi.iloc[current_pivot]), 
                            'macd': abs(macd.iloc[prev_pivot] - macd.iloc[current_pivot]), 
                            'stoch': abs(stoch.iloc[prev_pivot] - stoch.iloc[current_pivot]), 
                            'cci': abs(cci.iloc[prev_pivot] - cci.iloc[current_pivot]),
                            'roc': abs(roc.iloc[prev_pivot] - roc.iloc[current_pivot]),
                            'uo': abs(uo.iloc[prev_pivot] - uo.iloc[current_pivot]),
                            'ao': abs(ao.iloc[prev_pivot] - ao.iloc[current_pivot]),
                            'obv': abs(obv.iloc[prev_pivot] - obv.iloc[current_pivot]),
                            'mfi': abs(mfi.iloc[prev_pivot] - mfi.iloc[current_pivot]),
                            'adx': abs(adx.iloc[prev_pivot] - adx.iloc[current_pivot])}
            for indicator, discrepancy in discrepancies.items():
                if discrepancy > max_discrepancy:
                    max_discrepancy = discrepancy
                    strongest_divergence = (prev_pivot, current_pivot)
        return strongest_divergence
    return None

def bearish_divergence_finder(dataframe, indicator, high_iterator, index, rsi, macd, stoch, cci, roc, uo, ao, obv, mfi, adx):
    if high_iterator[index] == index:
        current_pivot = high_iterator[index]
        occurrences = list(dict.fromkeys(high_iterator))
        current_index = occurrences.index(high_iterator[index])
        strongest_divergence = None
        max_discrepancy = 0
        for i in range(current_index - 1, max(current_index - 5, -1), -1):
            prev_pivot = occurrences[i]
            if np.isnan(prev_pivot):
                continue
            price_difference = dataframe['high'].iloc[current_pivot] - dataframe['high'].iloc[prev_pivot]
            discrepancies = {'rsi': abs(rsi.iloc[current_pivot] - rsi.iloc[prev_pivot]), 
                            'macd': abs(macd.iloc[current_pivot] - macd.iloc[prev_pivot]), 
                            'stoch': abs(stoch.iloc[current_pivot] - stoch.iloc[prev_pivot]), 
                            'cci': abs(cci.iloc[current_pivot] - cci.iloc[prev_pivot]),
                            'roc': abs(roc.iloc[current_pivot] - roc.iloc[prev_pivot]),
                            'uo': abs(uo.iloc[current_pivot] - uo.iloc[prev_pivot]),
                            'ao': abs(ao.iloc[current_pivot] - ao.iloc[prev_pivot]),
                            'obv': abs(obv.iloc[current_pivot] - obv.iloc[prev_pivot]),
                            'mfi': abs(mfi.iloc[current_pivot] - mfi.iloc[prev_pivot]),
                            'adx': abs(adx.iloc[current_pivot] - adx.iloc[prev_pivot])}
            for indicator, discrepancy in discrepancies.items():
                if discrepancy > max_discrepancy:
                    max_discrepancy = discrepancy
                    strongest_divergence = (prev_pivot, current_pivot)
        return strongest_divergence
    return None

def bullish_divergence_finder(dataframe, indicator, low_iterator, index, rsi, macd, stoch, cci, roc, uo, ao, obv, mfi, adx):
    if low_iterator[index] == index:
        current_pivot = low_iterator[index]
        occurrences = list(dict.fromkeys(low_iterator))
        current_index = occurrences.index(low_iterator[index])
        strongest_divergence = None
        max_discrepancy = 0
        for i in range(current_index - 1, max(current_index - 7, -1), -1):
            prev_pivot = occurrences[i]
            if np.isnan(prev_pivot):
                continue
            price_difference = dataframe['low'].iloc[prev_pivot] - dataframe['low'].iloc[current_pivot]
            discrepancies = {'rsi': abs(rsi.iloc[prev_pivot] - rsi.iloc[current_pivot]), 
                            'macd': abs(macd.iloc[prev_pivot] - macd.iloc[current_pivot]), 
                            'stoch': abs(stoch.iloc[prev_pivot] - stoch.iloc[current_pivot]), 
                            'cci': abs(cci.iloc[prev_pivot] - cci.iloc[current_pivot]),
                            'roc': abs(roc.iloc[prev_pivot] - roc.iloc[current_pivot]),
                            'uo': abs(uo.iloc[prev_pivot] - uo.iloc[current_pivot]),
                            'ao': abs(ao.iloc[prev_pivot] - ao.iloc[current_pivot]),
                            'obv': abs(obv.iloc[prev_pivot] - obv.iloc[current_pivot]),
                            'mfi': abs(mfi.iloc[prev_pivot] - mfi.iloc[current_pivot]),
                            'adx': abs(adx.iloc[prev_pivot] - adx.iloc[current_pivot])}
            for indicator, discrepancy in discrepancies.items():
                if discrepancy > max_discrepancy:
                    max_discrepancy = discrepancy
                    strongest_divergence = (prev_pivot, current_pivot)
        return strongest_divergence
    return None

def emaKeltner(dataframe):
    keltner = {}
    atr = qtpylib.atr(dataframe, window=10)
    ema20 = ta.EMA(dataframe, timeperiod=20)
    keltner['upper'] = ema20 + atr
    keltner['mid'] = ema20
    keltner['lower'] = ema20 - atr
    return keltner

def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    df = dataframe.copy()
    
    # Calculate the Money Flow Multiplier
    mfm = (df['close'] - df['low'] - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfm = mfm.replace([np.inf, -np.inf], np.nan)
    mfm = mfm.fillna(0.0)
    
    # Calculate the Money Flow Volume
    mfv = mfm * df['volume']
    
    # Calculate the Chaikin Money Flow
    cmf = mfv.rolling(n, min_periods=0).sum() / df['volume'].rolling(n, min_periods=0).sum()
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')

def add_divergences_ichimoku(dataframe: pd.DataFrame, lookback_period: int=52) -> pd.DataFrame:
    # First, calculate the Ichimoku Cloud components if they don't exist
    if not {'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span'}.issubset(dataframe.columns):
        dataframe = calculate_ichimoku_cloud(dataframe)
        
    # Then, find divergences using the Ichimoku Cloud components
    dataframe = ichimoku_cloud_divergence_finder(dataframe, lookback_period)
    return dataframe

def calculate_ichimoku_cloud(dataframe: pd.DataFrame, tenkan_period: int=9, kijun_period: int=26, senkou_span_b_period: int=52, displacement: int=26) -> pd.DataFrame:
    # Tenkan-sen (Conversion Line)
    tenkan_sen_high = dataframe['high'].rolling(window=tenkan_period).max()
    tenkan_sen_low = dataframe['low'].rolling(window=tenkan_period).min()
    dataframe['tenkan_sen'] = (tenkan_sen_high + tenkan_sen_low) / 2
    
    # Kijun-sen (Base Line)
    kijun_sen_high = dataframe['high'].rolling(window=kijun_period).max()
    kijun_sen_low = dataframe['low'].rolling(window=kijun_period).min()
    dataframe['kijun_sen'] = (kijun_sen_high + kijun_sen_low) / 2
    
    # Senkou Span A (Leading Span A)
    dataframe['senkou_span_a'] = ((dataframe['tenkan_sen'] + dataframe['kijun_sen']) / 2).shift(displacement)
    
    # Senkou Span B (Leading Span B)
    senkou_span_b_high = dataframe['high'].rolling(window=senkou_span_b_period).max()
    senkou_span_b_low = dataframe['low'].rolling(window=senkou_span_b_period).min()
    dataframe['senkou_span_b'] = ((senkou_span_b_high + senkou_span_b_low) / 2).shift(displacement)
    
    # Chikou Span (Lagging Span)
    dataframe['chikou_span'] = dataframe['close'].shift(-displacement)
    
    # Kumo (Cloud)
    dataframe['kumo_up'] = dataframe['senkou_span_a'].where(dataframe['senkou_span_a'] > dataframe['senkou_span_b'], dataframe['senkou_span_b'])
    dataframe['kumo_down'] = dataframe['senkou_span_a'].where(dataframe['senkou_span_a'] <= dataframe['senkou_span_b'], dataframe['senkou_span_b'])
    return dataframe

def ichimoku_cloud_divergence_finder(dataframe: pd.DataFrame, lookback_period: int=26) -> pd.DataFrame:
    # Ensure Ichimoku Cloud components are present
    if not {'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span'}.issubset(dataframe.columns):
        raise ValueError('DataFrame must include Ichimoku Cloud components')
    
    # Initialize divergence columns for each component
    dataframe['kijun_bullish_divergence'] = False
    dataframe['kijun_bearish_divergence'] = False
    dataframe['senkou_span_a_bullish_divergence'] = False
    dataframe['senkou_span_a_bearish_divergence'] = False
    dataframe['senkou_span_b_bullish_divergence'] = False
    dataframe['senkou_span_b_bearish_divergence'] = False
    dataframe['chikou_bullish_divergence'] = False
    dataframe['chikou_bearish_divergence'] = False
    
    # Loop through the DataFrame to find divergences
    for i in range(lookback_period, len(dataframe)):
        # Current and past price for Chikou Span comparison
        current_price = dataframe['close'][i]
        chikou_span_price = dataframe['close'][i - lookback_period]
        
        # Check for Kijun-Sen bullish divergence (price makes a lower low, Kijun-Sen does not)
        if dataframe['low'][i] < dataframe['low'].iloc[i - lookback_period:i].min() and dataframe['kijun_sen'][i] >= dataframe['kijun_sen'].iloc[i - lookback_period:i].min():
            dataframe.at[i, 'kijun_bullish_divergence'] = True
        
        # Check for Kijun-Sen bearish divergence (price makes a higher high, Kijun-Sen does not)
        if dataframe['high'][i] > dataframe['high'].iloc[i - lookback_period:i].max() and dataframe['kijun_sen'][i] <= dataframe['kijun_sen'].iloc[i - lookback_period:i].max():
            dataframe.at[i, 'kijun_bearish_divergence'] = True
        
        # Check for Senkou Span A bullish divergence (price breaks above Senkou Span A)
        if current_price > dataframe['senkou_span_a'][i] and chikou_span_price <= dataframe['senkou_span_a'][i - lookback_period]:
            dataframe.at[i, 'senkou_span_a_bullish_divergence'] = True
        
        # Check for Senkou Span A bearish divergence (price breaks below Senkou Span A)
        if current_price < dataframe['senkou_span_a'][i] and chikou_span_price >= dataframe['senkou_span_a'][i - lookback_period]:
            dataframe.at[i, 'senkou_span_a_bearish_divergence'] = True
        
        # Check for Senkou Span B bullish divergence (price breaks above Senkou Span B)
        if current_price > dataframe['senkou_span_b'][i] and chikou_span_price <= dataframe['senkou_span_b'][i - lookback_period]:
            dataframe.at[i, 'senkou_span_b_bullish_divergence'] = True
        
        # Check for Senkou Span B bearish divergence (price breaks below Senkou Span B)
        if current_price < dataframe['senkou_span_b'][i] and chikou_span_price >= dataframe['senkou_span_b'][i - lookback_period]:
            dataframe.at[i, 'senkou_span_b_bearish_divergence'] = True
        
        # Check for Chikou Span bullish divergence (Chikou Span moves above price from 26 periods ago)
        if dataframe['chikou_span'][i] > chikou_span_price and dataframe['chikou_span'][i - lookback_period] <= chikou_span_price:
            dataframe.at[i, 'chikou_bullish_divergence'] = True
        
        # Check for Chikou Span bearish divergence (Chikou Span moves below price from 26 periods ago)
        if dataframe['chikou_span'][i] < chikou_span_price and dataframe['chikou_span'][i - lookback_period] >= chikou_span_price:
            dataframe.at[i, 'chikou_bearish_divergence'] = True
    return dataframe

# RECALCULATE AND SETUP CORRECTLY THE PRICES (LOW AND HIGH ARE FROM THE WHOLE DATAFRAME) 
# & THE VALUES OF THE FIBS SHOULD BE INTERPRETED AS PERCENTAGE TO CORRECTLY ENTER OR EXIT
# USE CURRENT CANDLE

def add_fibonacci_retracement_levels(dataframe: pd.DataFrame) -> pd.DataFrame:
    
    # Calculate the high price of the current candle
    high_price = dataframe['high'].iloc[-1]
    
    # Calculate the low price of the current candle
    low_price = dataframe['low'].iloc[-1]
    
    # Calculate the difference between high and low prices
    price_difference = high_price - low_price
    
    # Calculate Fibonacci levels
    dataframe['fib_23.6%'] = high_price - price_difference * 0.236
    dataframe['fib_38.2%'] = high_price - price_difference * 0.382
    dataframe['fib_50.0%'] = high_price - price_difference * 0.5
    dataframe['fib_61.8%'] = high_price - price_difference * 0.618
    dataframe['fib_78.6%'] = high_price - price_difference * 0.786
    
    # Optionally, you can also add the 0% and 100% levels which are the high and low prices
    dataframe['fib_0%'] = high_price
    dataframe['fib_100%'] = low_price
    return dataframe

def add_confirmation_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    
    # Define criteria for confirmation signals
    bullish_confirmation_criteria = (
        (dataframe['rsi'] > 50) &                           # RSI above 50
        (dataframe['stoch'] > 50) &                         # Stochastic above 50
        (dataframe['roc'] > 0) &                            # Rate of Change positive
        (dataframe['uo'] > 50) &                            # Ultimate Oscillator above 50
        (dataframe['ao'] > 0) &                             # Awesome Oscillator positive
        # (dataframe['macd'] > informative['macd_4h']) &           # MACD bullish crossover
        (dataframe['cci'] > 0) &                            # Commodity Channel Index positive
        (dataframe['cmf'] > 0) &                            # Chaikin Money Flow positive
        (dataframe['obv'] > dataframe['obv'].shift(1)) &           # On Balance Volume increasing
        (dataframe['mfi'] > 50) &                           # Money Flow Index above 50
        (dataframe['adx'] > 25) &                           # Average Directional Index above 25
        (dataframe['close'] > dataframe['ema9']) &                 # Price above 9-period EMA
        (dataframe['close'] > dataframe['ema20']) &                # Price above 20-period EMA
        (dataframe['close'] > dataframe['ema50']) &                # Price above 50-period EMA
        (dataframe['close'] > dataframe['ema200']) &               # Price above 200-period EMA
        (dataframe['close'] > dataframe['tema']) &                 # Price above Triple Exponential Moving Average
        (dataframe['close'] > dataframe['sar'])                    # Price above Parabolic SAR
    )

    bearish_confirmation_criteria = (
        (dataframe['rsi'] < 50) &                           # RSI below 50
        (dataframe['stoch'] < 50) &                         # Stochastic below 50
        (dataframe['roc'] < 0) &                            # Rate of Change negative
        (dataframe['uo'] < 50) &                            # Ultimate Oscillator below 50
        (dataframe['ao'] < 0) &                             # Awesome Oscillator negative
        # (dataframe['macd'] < informative['macd_4h']) &           # MACD bearish crossover
        (dataframe['cci'] < 0) &                            # Commodity Channel Index negative
        (dataframe['cmf'] < 0) &                            # Chaikin Money Flow negative
        (dataframe['obv'] < dataframe['obv'].shift(1)) &           # On Balance Volume decreasing
        (dataframe['mfi'] < 50) &                           # Money Flow Index below 50
        (dataframe['adx'] < 25) &                           # Average Directional Index below 25
        (dataframe['close'] < dataframe['ema9']) &                 # Price below 9-period EMA
        (dataframe['close'] < dataframe['ema20']) &                # Price below 20-period EMA
        (dataframe['close'] < dataframe['ema50']) &                # Price below 50-period EMA
        (dataframe['close'] < dataframe['ema200']) &               # Price below 200-period EMA
        (dataframe['close'] < dataframe['tema']) &                 # Price below Triple Exponential Moving Average
        (dataframe['close'] < dataframe['sar'])                    # Price below Parabolic SAR
    )

    # Add confirmation signals columns
    dataframe['bullish_confirmation_signal'] = bullish_confirmation_criteria.astype(int)
    dataframe['bearish_confirmation_signal'] = bearish_confirmation_criteria.astype(int)

    return dataframe

# # Define entry conditions for long positions based on Fibonacci levels for both timeframes
# dataframe.loc[
#     (
#         (dataframe['low'] <= dataframe['fib_61.8%']) |  # Standard timeframe retracement to or below the 61.8% Fibonacci level
#         (dataframe['low'] <= dataframe['fib_38.2%'])    # OR Standard timeframe retracement to or below the 38.2% Fibonacci level
#     ) &
#     (
#         (dataframe['low_4h'] <= dataframe['fib_61.8%_4h']) |  # 4-hour timeframe retracement to or below the 61.8% Fibonacci level
#         (dataframe['low_4h'] <= dataframe['fib_38.2%_4h'])    # OR 4-hour timeframe retracement to or below the 38.2% Fibonacci level
#     ) &
#     (dataframe['volume'] > 0),  # Sufficient trading volume
#     ['enter_long', 'enter_tag']
# ] = (1, 'Fibonacci Retracement Level Long Position Both Timeframes')
# # Define entry conditions for short positions based on Fibonacci levels for both timeframes
# dataframe.loc[
#     (
#         (dataframe['high'] >= dataframe['fib_61.8%']) |  # Standard timeframe retracement to or above the 61.8% Fibonacci level
#         (dataframe['high'] >= dataframe['fib_38.2%'])    # OR Standard timeframe retracement to or above the 38.2% Fibonacci level
#     ) &
#     (
#         (dataframe['high_4h'] >= dataframe['fib_61.8%_4h']) |  # 4-hour timeframe retracement to or above the 61.8% Fibonacci level
#         (dataframe['high_4h'] >= dataframe['fib_38.2%_4h'])    # OR 4-hour timeframe retracement to or above the 38.2% Fibonacci level
#     ) &
#     (dataframe['volume'] > 0),  # Sufficient trading volume
#     ['enter_short', 'enter_tag']
# ] = (1, 'Fibonacci Retracement Level Short Position Both Timeframes')
# # Detect bullish trend entries
# dataframe.loc[(dataframe['high'] > dataframe['high'].shift()) & (dataframe['total_bull_d'].shift() > 0) 
# & two_bands_check(dataframe) & (dataframe['volume'] > 0), ['enter_long', 'enter_tag']] = (1, 'Bullish Divergence')
# # Detect bearish trend entries
# dataframe.loc[(dataframe['low'] < dataframe['low'].shift()) & (dataframe['total_bear_d'].shift() > 0) 
# & two_bands_check(dataframe) & (dataframe['volume'] > 0), ['enter_short', 'enter_tag']] = (1, 'Bearish Divergence')
# dataframe = super().populate_indicators(dataframe, metadata)
# # Base conditions for entering a long position (Bullish)
# bull_base_conditions = [
#     (dataframe['high'] > dataframe['high'].shift()),
#     (dataframe['total_bull_d'].shift() > 0),
#     (dataframe['total_bull_d_4h'].shift() > 0),
#     two_bands_check(dataframe),
#     (dataframe['volume'] > 0),
#     (dataframe['total_bull_d_count'] > 1)  # More than two bullish divergences
# ]
# # Check for the presence of each indicator's divergence and apply the corresponding self.value parameter
# bull_divergence_conditions = [
#     dataframe['total_bull_d_names'].str.contains('RSI') & (dataframe['rsi'] < self.buy_rsi.value),
#     dataframe['total_bull_d_names'].str.contains('STOCH') & (dataframe['stoch'] < self.buy_stoch.value),
#     dataframe['total_bull_d_names'].str.contains('ROC') & (dataframe['roc'] < self.buy_roc_signal.value),
#     dataframe['total_bull_d_names'].str.contains('UO') & (dataframe['uo'] < self.buy_uo_any_period_signal.value),
#     dataframe['total_bull_d_names'].str.contains('AO') & (dataframe['ao'] < self.buy_ao_signal.value),
#     dataframe['total_bull_d_names'].str.contains('MACD') & (dataframe['macd'] < self.buy_macd_signal.value),
#     dataframe['total_bull_d_names'].str.contains('CCI') & (dataframe['cci'] < self.buy_cci_signal.value),
#     dataframe['total_bull_d_names'].str.contains('CMF') & (dataframe['cmf'] < self.buy_cmf_signal.value),
#     dataframe['total_bull_d_names'].str.contains('MFI') & (dataframe['mfi'] < self.buy_mfi_signal.value),
#     dataframe['total_bull_d_names'].str.contains('ADX') & (dataframe['adx'] > self.buy_adx_signal.value)
# ]
# # Combine base conditions with a logical AND
# combined_bull_base_condition = reduce(lambda x, y: x & y, bull_base_conditions)
# # Check if any of the divergence conditions are met
# # This is a logical OR across all divergence conditions
# any_bull_divergence_condition = reduce(lambda x, y: x | y, bull_divergence_conditions)
# # Final combined condition
# combined_bull_condition = combined_bull_base_condition & any_bull_divergence_condition
# # Apply the combined condition to the dataframe
# dataframe.loc[
#     combined_bull_condition,
#     ['enter_long', 'enter_tag']
# ] = (1, 'Harmonic Bullish Divergence Conditions Met')
# # Base conditions for entering a short position (Bearish)
# bear_base_conditions = [
#     (dataframe['low'] < dataframe['low'].shift()),
#     (dataframe['total_bear_d'].shift() > 0),
#     (dataframe['total_bear_d_4h'].shift() > 0),
#     two_bands_check(dataframe),
#     (dataframe['volume'] > 0),
#     (dataframe['total_bear_d_count'] > 1)  # More than two bearish divergences
# ]
# # Check for the presence of each indicator's divergence and apply the corresponding self.value parameter
# bear_divergence_conditions = [
#     dataframe['total_bear_d_names'].str.contains('RSI') & (dataframe['rsi'] > self.sell_rsi.value),
#     dataframe['total_bear_d_names'].str.contains('STOCH') & (dataframe['stoch'] > self.sell_stoch.value),
#     dataframe['total_bear_d_names'].str.contains('ROC') & (dataframe['roc'] > self.sell_roc_signal.value),
#     dataframe['total_bear_d_names'].str.contains('UO') & (dataframe['uo'] > self.sell_uo_any_period_signal.value),
#     dataframe['total_bear_d_names'].str.contains('AO') & (dataframe['ao'] > self.sell_ao_signal.value),
#     dataframe['total_bear_d_names'].str.contains('MACD') & (dataframe['macd'] > self.sell_macd_signal.value),
#     dataframe['total_bear_d_names'].str.contains('CCI') & (dataframe['cci'] > self.sell_cci_signal.value),
#     dataframe['total_bear_d_names'].str.contains('CMF') & (dataframe['cmf'] > self.sell_cmf_signal.value),
#     dataframe['total_bear_d_names'].str.contains('MFI') & (dataframe['mfi'] > self.sell_mfi_signal.value),
#     dataframe['total_bear_d_names'].str.contains('ADX') & (dataframe['adx'] < self.sell_adx_signal.value)
# ]
# # Combine bear base conditions with a logical AND
# combined_bear_base_condition = reduce(lambda x, y: x & y, bear_base_conditions)
# # Check if any of the bear divergence conditions are met
# # This is a logical OR across all bear divergence conditions
# any_bear_divergence_condition = reduce(lambda x, y: x | y, bear_divergence_conditions)
# # Final combined bear condition
# combined_bear_condition = combined_bear_base_condition & any_bear_divergence_condition
# # Apply the combined bear condition to the dataframe
# dataframe.loc[
#     combined_bear_condition,
#     ['enter_short', 'enter_tag']
# ] = (-1, 'Harmonic Bearish Divergence Conditions Met')