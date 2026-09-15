# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt



def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.SMA(df, timeperiod=ema_length)
    ema2 = ta.SMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif



def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

    dataframe['adx'] = ta.ADX(dataframe)
    dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
    dataframe['plus_di'] = ta.PLUS_DI(dataframe)
    dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
    dataframe['minus_di'] = ta.MINUS_DI(dataframe)

    aroon = ta.AROON(dataframe)
    dataframe['aroonup'] = aroon['aroonup']
    dataframe['aroondown'] = aroon['aroondown']
    dataframe['aroonosc'] = ta.AROONOSC(dataframe)
    dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

    keltner = qtpylib.keltner_channel(dataframe)
    dataframe['kc_upperband'] = keltner['upper']
    dataframe['kc_lowerband'] = keltner['lower']
    dataframe['kc_middleband'] = keltner['mid']
    dataframe['kc_percent'] = (dataframe['close'] - dataframe['kc_lowerband']) / (dataframe['kc_upperband'] - dataframe['kc_lowerband'])
    dataframe['kc_width'] = (dataframe['kc_upperband'] - dataframe['kc_lowerband']) / dataframe['kc_middleband']
    dataframe['uo'] = ta.ULTOSC(dataframe)
    dataframe['cci'] = ta.CCI(dataframe)
    dataframe['rsi'] = ta.RSI(dataframe)

    rsi = 0.1 * (dataframe['rsi'] - 50)
    dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
    dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

    stoch = ta.STOCH(dataframe)
    dataframe['slowd'] = stoch['slowd']
    dataframe['slowk'] = stoch['slowk']


    stoch_fast = ta.STOCHF(dataframe)
    dataframe['fastd'] = stoch_fast['fastd']
    dataframe['fastk'] = stoch_fast['fastk']


    stoch_rsi = ta.STOCHRSI(dataframe)
    dataframe['fastd_rsi'] = stoch_rsi['fastd']
    dataframe['fastk_rsi'] = stoch_rsi['fastk']


    macd = ta.MACD(dataframe)
    dataframe['macd'] = macd['macd']
    dataframe['macdsignal'] = macd['macdsignal']
    dataframe['macdhist'] = macd['macdhist']
    dataframe['mfi'] = ta.MFI(dataframe)
    dataframe['roc'] = ta.ROC(dataframe)



    bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
    dataframe['bb_lowerband'] = bollinger['lower']
    dataframe['bb_middleband'] = bollinger['mid']
    dataframe['bb_upperband'] = bollinger['upper']
    dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lowerband']) / (dataframe['bb_upperband'] - dataframe['bb_lowerband'])
    dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']

    dataframe['sar'] = ta.SAR(dataframe)
    dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)


    hilbert = ta.HT_SINE(dataframe)
    dataframe['htsine'] = hilbert['sine']
    dataframe['htleadsine'] = hilbert['leadsine']

    dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
    dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
    dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
    dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe)  # values [0, 100]
    dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe)  # values [0, 100]
    dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe)  # values [0, 100]

    dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
    dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
    dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
    dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
    dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
    dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)

    dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
    dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe)  # values [0, -100, 100]
    dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe)  # values [0, -100, 100]
    dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe)  # values [0, -100, 100]
    dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe)  # values [0, -100, 100]
    dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe)  # values [0, -100, 100]


    heikinashi = qtpylib.heikinashi(dataframe)
    dataframe['ha_open'] = heikinashi['open']
    dataframe['ha_close'] = heikinashi['close']
    dataframe['ha_high'] = heikinashi['high']
    dataframe['ha_low'] = heikinashi['low']
    return dataframe



class el(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True

    # Buy hyperspace params:
    entry_params = {
        "base_nb_candles_entry": 12,
        "ewo_high": 4.428,
        "ewo_low": -12.383,
        "low_offset": 0.915,
        "rsi_entry": 44,
    }

    # Sell hyperspace params:
    exit_params = {
        "base_nb_candles_exit": 72,
        "high_offset": 1.008,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.219,
        "24": 0.087,
        "67": 0.024,
        "164": 0
    }

    # Stoploss:
    stoploss = -0.242

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.103
    trailing_stop_positive_offset = 0.2
    trailing_only_offset_is_reached = True


    # Max Open Trades:
    max_open_trades = 9
    
    # SMAOffset
    base_nb_candles_entry = IntParameter(5, 80, default=entry_params['base_nb_candles_entry'], space='entry', optimize=True)
    base_nb_candles_exit = IntParameter(5, 80, default=exit_params['base_nb_candles_exit'], space='exit', optimize=True)
    low_offset = DecimalParameter(0.9, 0.99, default=entry_params['low_offset'], space='entry', optimize=True)
    high_offset = DecimalParameter(0.99, 1.1, default=exit_params['high_offset'], space='exit', optimize=True)
    
    
    # Protection
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0, default=entry_params['ewo_low'], space='entry', optimize=True)
    ewo_high = DecimalParameter(2.0, 12.0, default=entry_params['ewo_high'], space='entry', optimize=True)
    rsi_entry = IntParameter(30, 70, default=entry_params['rsi_entry'], space='entry', optimize=True)
    
    
    
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    
    
    # Sell signal
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = True
    
    
    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe = '1h'
    process_only_new_candles = True
    #startup_candle_count = 2000
    plot_config = {'main_plot': {'ma_entry': {'color': 'orange'}, 'ma_exit': {'color': 'orange'}}}
    use_custom_stoploss = False



    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs



    def get_informative_indicators(self, metadata: dict):
        dataframe = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
        return dataframe



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        for val in self.base_nb_candles_entry.range:
            dataframe[f'ma_entry_{val}'] = ta.EMA(dataframe, timeperiod=val)
        for val in self.base_nb_candles_exit.range:
            dataframe[f'ma_exit_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe



    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        entry_conditions = []
        entry_conditions.append((dataframe['close'] < dataframe[f'ma_entry_{self.base_nb_candles_entry.value}'] * self.low_offset.value) & (dataframe['EWO'] > self.ewo_high.value) & (dataframe['rsi'] < self.rsi_entry.value) & (dataframe['volume'] > 0))
        entry_conditions.append((dataframe['close'] < dataframe[f'ma_entry_{self.base_nb_candles_entry.value}'] * self.low_offset.value) & (dataframe['EWO'] < self.ewo_low.value) & (dataframe['volume'] > 0))
        if entry_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, entry_conditions), 'enter_long'] = 1


        exit_conditions = []
        exit_conditions.append((dataframe['close'] > dataframe[f'ma_exit_{self.base_nb_candles_exit.value}'] * self.high_offset.value) & (dataframe['volume'] > 0))
        if exit_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, exit_conditions), 'enter_short'] = 1


        return dataframe





    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = []
        exit_long_conditions.append((dataframe['close'] > dataframe[f'ma_exit_{self.base_nb_candles_exit.value}'] * self.high_offset.value) & (dataframe['volume'] > 0))
        if exit_long_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, exit_long_conditions), 'exit_long'] = 1

        exit_short_conditions = []
        exit_short_conditions.append((dataframe['close'] < dataframe[f'ma_entry_{self.base_nb_candles_entry.value}'] * self.low_offset.value) & (dataframe['EWO'] > self.ewo_high.value) & (dataframe['rsi'] < self.rsi_entry.value) & (dataframe['volume'] > 0))
        exit_short_conditions.append((dataframe['close'] < dataframe[f'ma_entry_{self.base_nb_candles_entry.value}'] * self.low_offset.value) & (dataframe['EWO'] < self.ewo_low.value) & (dataframe['volume'] > 0))
        if exit_short_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, exit_short_conditions), 'exit_short'] = 1


            return dataframe


    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag:str, side: str, **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (entry_tag) if provided with the entry signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 10.0