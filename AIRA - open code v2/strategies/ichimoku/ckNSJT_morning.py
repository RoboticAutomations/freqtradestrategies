# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.vendor.qtpylib import indicators
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open
from datetime import datetime, timedelta
from typing import Optional, Union
import pandas_ta as pta
from pandas import DataFrame, Series
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
import technical.indicators as ftt

from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce
import warnings

warnings.simplefilter(action="ignore", category=RuntimeWarning)

low_offset = 0.958 # something lower than 1
high_offset = 1.012 # something higher than 1
class ckNSJT_morning(IStrategy):
    minimal_roi = {
        "0": 1
    }
    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 120
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': "market",
        'stoploss': 'market',
        'stoploss_on_exchange': True,
        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_market_ratio': 0.99
    }
    stoploss = -0.99
    use_custom_stoploss = True

    #DCA
    last_dca_timeframe = {}
    max_entry_position_adjustment = 11
    max_dca_multiplier = 3.3
    open_trade_limit = 6
    position_adjustment_enable = False
    dca_threshold_pct = DecimalParameter(0.01, 0.50, default=0.03, decimals=2, space='buy',
                                         optimize=position_adjustment_enable)
    candles_before_dca = IntParameter(1, 10, default=7, space='buy', optimize=True)

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.006 #povodne 0.001
    trailing_stop_positive_offset = 0.08 #povodne 0.012
    trailing_only_offset_is_reached = True
    # Make sure these match or are not overridden in config
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    is_optimize_32 = True
    buy_rsi_fast_32 = IntParameter(20, 70, default=45, space='buy', optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=35, space='buy', optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.961, decimals=3, space='buy', optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 0, default=-0.58, decimals=2, space='buy', optimize=is_optimize_32)
    sell_fastx = IntParameter(50, 100, default=70, space='sell', optimize=True)

    sell_loss_cci = IntParameter(low=0, high=600, default=148, space='sell', optimize=False)
    sell_loss_cci_profit = DecimalParameter(-0.15, 0, default=-0.04, decimals=2, space='sell', optimize=False)
    sell_cci = IntParameter(low=0, high=200, default=90, space='sell', optimize=False)


    buy_params = {
        # Hyperopt
        # Multi Offset
        "base_nb_candles_buy": 36,
        "buy_chop_min_19": 58.2,
        "buy_rsi_1h_min_19": 65.3,
        "ewo_high": 4.3,
        "ewo_low": -8.5,
        "low_offset_ema": 0.929,
        "low_offset_kama": 0.972,
        "low_offset_sma": 0.955,
        "low_offset_t3": 0.975,
        "low_offset_trima": 0.949,
    }

    sell_params = {
        #############
        # Hyperopt
        # Multi Offset
        "base_nb_candles_sell": 34,
        "high_offset_ema": 1.047,
        "high_offset_kama": 1.07,
        "high_offset_sma": 1.051,
        "high_offset_t3": 0.999,
        "high_offset_trima": 1.096,
        "pHSL": -0.397,  # value loaded from strategy
        "pPF_1": 0.012,  # value loaded from strategy
        "pPF_2": 0.07,  # value loaded from strategy
        "pSL_1": 0.015,  # value loaded from strategy
        "pSL_2": 0.068,  # value loaded from strategy
    }
    # MA list
    ma_types = ['sma', 'ema', 'trima', 't3', 'kama']
    ma_map = {
        'sma': {
            'low_offset': buy_params['low_offset_sma'],
            'high_offset': sell_params['high_offset_sma'],
            'calculate': ta.SMA
        },
        'ema': {
            'low_offset': buy_params['low_offset_ema'],
            'high_offset': sell_params['high_offset_ema'],
            'calculate': ta.EMA
        },
        'trima': {
            'low_offset': buy_params['low_offset_trima'],
            'high_offset': sell_params['high_offset_trima'],
            'calculate': ta.TRIMA
        },
        't3': {
            'low_offset': buy_params['low_offset_t3'],
            'high_offset': sell_params['high_offset_t3'],
            'calculate': ta.T3
        },
        'kama': {
            'low_offset': buy_params['low_offset_kama'],
            'high_offset': sell_params['high_offset_kama'],
            'calculate': ta.KAMA
        }
    }
    # hard stoploss profit
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe['tcp_percent_4'] = top_percent_change(dataframe , 4)

        # RSI
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)

        # buy_1 indicators
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)


        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)

        rsi = ta.RSI(dataframe)
        dataframe["rsi"] = rsi
        rsi = 0.1 * (rsi - 50)
        dataframe["fisher"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        dataframe['ema_3'] = ta.EMA(dataframe, timeperiod=3)
        dataframe['ema_5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # Donchian Channels
        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=19)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=19)
        dataframe['dc_mid'] = ta.TEMA(((dataframe['dc_upper'] + dataframe['dc_lower']) / 2),
                                    timeperiod=19)
        # Fibonacci Levels (of Donchian Channel)
        dataframe['dc_dist'] = (dataframe['dc_upper']  - dataframe['dc_lower'])
        dataframe['dc_width'] = ((dataframe['dc_upper'] - dataframe['dc_lower']) / dataframe['dc_mid'])
        dataframe['dc_hf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.236 # Highest Fib
        dataframe['dc_chf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.382 # Centre High Fib
        dataframe['dc_clf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.618 # Centre Low Fib
        dataframe['dc_lf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.764 # Low Fib

        dataframe['dc_percent'] = \
                    (dataframe['close'] - dataframe['dc_lf']) / (dataframe['dc_hf'] - dataframe['dc_lf'])

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)
        dataframe['vwma'] = pta.vwma(dataframe["ha_close"], dataframe["volume"], 28*0.98)

        # Bollinger bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=40, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        # is DIP
        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']
        dataframe['bb_delta'] = ((dataframe['bb_lowerband2'] - dataframe['bb_lowerband3']) / dataframe['bb_lowerband2'])


        dataframe['close_delta'] = (dataframe['close'] - 0.990 * dataframe['bb_lowerband2']) / dataframe['close']
        dataframe['close_delta2'] = (dataframe['close'] - 0.990 * dataframe['dc_lf']) / dataframe['close']
        dataframe['bbdelta'] = (dataframe['bb_middleband'] - dataframe['bb_lowerband']).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()

        #pump stregth
        dataframe['zema_30'] = ftt.zema(dataframe, period=30)
        dataframe['zema_200'] = ftt.zema(dataframe, period=200)
        dataframe['pump_strength'] = (dataframe['zema_30'] - dataframe['zema_200']) / dataframe['zema_30']

        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['vwap_low'] = vwap_low
        dataframe['vwap'] = vwap
        dataframe['vwap_upperband'] = vwap_high
        dataframe['vwap_middleband'] = vwap
        dataframe['vwap_lowerband'] = vwap_low
        dataframe['vwap_width'] = ( (dataframe['vwap_upperband'] - dataframe['vwap_lowerband']) / dataframe['vwap_middleband'] ) * 100


        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)
        dataframe['volume_mean_4'] = dataframe['volume'].rolling(4).mean().shift(1)




        dataframe['trend_close_5m'] = dataframe['close']
        dataframe['trend_close_15m'] = ta.EMA(dataframe['close'], timeperiod=3)
        dataframe['trend_close_30m'] = ta.EMA(dataframe['close'], timeperiod=6)
        dataframe['trend_close_1h'] = ta.EMA(dataframe['close'], timeperiod=12)
        dataframe['trend_close_2h'] = ta.EMA(dataframe['close'], timeperiod=24)
        dataframe['trend_close_4h'] = ta.EMA(dataframe['close'], timeperiod=48)
        dataframe['trend_close_6h'] = ta.EMA(dataframe['close'], timeperiod=72)
        dataframe['trend_close_8h'] = ta.EMA(dataframe['close'], timeperiod=96)

        dataframe['trend_open_5m'] = dataframe['open']
        dataframe['trend_open_15m'] = ta.EMA(dataframe['open'], timeperiod=3)
        dataframe['trend_open_30m'] = ta.EMA(dataframe['open'], timeperiod=6)
        dataframe['trend_open_1h'] = ta.EMA(dataframe['open'], timeperiod=12)
        dataframe['trend_open_2h'] = ta.EMA(dataframe['open'], timeperiod=24)
        dataframe['trend_open_4h'] = ta.EMA(dataframe['open'], timeperiod=48)
        dataframe['trend_open_6h'] = ta.EMA(dataframe['open'], timeperiod=72)
        dataframe['trend_open_8h'] = ta.EMA(dataframe['open'], timeperiod=96)

        dataframe['fan_magnitude'] = (dataframe['trend_close_1h'] / dataframe['trend_close_8h'])
        dataframe['fan_magnitude_gain'] = dataframe['fan_magnitude'] / dataframe['fan_magnitude'].shift(1)

        ichimoku = ftt.ichimoku(dataframe, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=30)
        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']
        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        dataframe['cloud_green'] = ichimoku['cloud_green']
        dataframe['cloud_red'] = ichimoku['cloud_red']

        dataframe['zvwap']= calc_zvwap(dataframe, pds=14, source1=dataframe['close'])

        #lambo2
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)
        # EWO
        dataframe['ewo'] = EWO(dataframe, 50, 200)

        # Offset
        for i in self.ma_types:
            dataframe[f'{i}_offset_buy'] = self.ma_map[f'{i}']['calculate'](
                dataframe, 20) * \
                self.ma_map[f'{i}']['low_offset']
            dataframe[f'{i}_offset_sell'] = self.ma_map[f'{i}']['calculate'](
                dataframe, 20) * \
                self.ma_map[f'{i}']['high_offset']

        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe,5,3,3)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # # Stochastic RSI
        stoch_rsi = ta.STOCHRSI(dataframe)
        dataframe['fastd_rsi'] = stoch_rsi['fastd']
        dataframe['fastk_rsi'] = stoch_rsi['fastk']



        dataframe['goodloss'] = (
              #(dataframe['ema_200'] > dataframe['ema_200'].shift(1)) &
              (dataframe['ema_5'] < dataframe['ema_5'].shift(1)) &
              (dataframe['dc_hf'].shift(1) < dataframe['close'].shift(1)) &
              (dataframe['vwap_upperband'].shift(1) < dataframe['close'].shift(1)) &
              (dataframe['fisher'].shift(1) > 0.99)).astype(int)

        dataframe['dca_buy_signal'] = (
              (dataframe['rsi_fast'] > dataframe['rsi_fast'].shift(1)) &
              (dataframe['ema_200'] < dataframe['ema_200'].shift(1)) & #Solo al ribasso
              (dataframe['pump_strength'] < 0) &
              (dataframe['ha_close'].shift(1) > dataframe['close'].shift(1)) & #Solo al ribasso
              (dataframe['rsi_fast'].shift(1) < 10)&
              (dataframe['fisher'].shift(1) < -0.98) &
              (dataframe['volume'] > 0)).astype(int)

        #dataframe['dca_buy_signal2'] = (
        #      (dataframe['rocr'] < 0.95) &
        #      (dataframe['ema_3'] <= 0.988*dataframe['bb_lowerband']) &
        #      (dataframe['volume'] > 0)).astype(int)

        dataframe['dca_buy_signal2'] = (
              (dataframe['cci'] < -100) &
              (dataframe['dc_lf'] > dataframe['close']) &
              (dataframe['bb_lowerband2'] > dataframe['close']) &
              (dataframe['dc_percent'] < -0.30) &
              (dataframe['rsi_fast'] < 15) &
              (dataframe['fisher'] < -0.95) &
              (dataframe['volume'] > 0)).astype(int)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''


        for i in self.ma_types:
            conditions.append(
                (dataframe['cti'] < -0.8) &
                #(dataframe['tcp_percent_4'] > 0.04) & #Testing
                (dataframe['pump_strength'] < 0) &
                (dataframe['close'] < dataframe[f'{i}_offset_buy']) &
                (dataframe['mfi'] < 27) &
                (dataframe['close'] < dataframe['ema_20']) &
                (
                    (dataframe['ewo'] < -8.5) |
                    (dataframe['ewo'] > 4.3)
                ) &
                (dataframe['volume'] > 0)
            )

        # Lambo2 condition
        lambo2 = (
            #(dataframe['bb_width'] > 0.1) & #Testing
            #(dataframe['vwap_width'] > 2) &
            #(dataframe['tcp_percent_4'] > 0.04) &
            #(dataframe['vwap_low'].shift(1) > dataframe['close'].shift(1)) & #Testing
            #(dataframe['vwma'] > dataframe['ha_open']) & #Testing
            ##(dataframe['cti'] < -0.8) &
            (dataframe['pump_strength'] < 0) &
            (dataframe['vwap_low'].shift(1) > dataframe['close'].shift(1)) & #Testing
            (dataframe['bb_lowerband2'].shift(1) < dataframe['close'].shift(1)) &
            (dataframe['close'] < (dataframe['ema_14'] * 0.978)) &
            (dataframe['rsi_4'] < 46) &
            (dataframe['rsi_14'] < 53)
        )
        dataframe.loc[lambo2, 'enter_tag'] += 'lambo2_'
        conditions.append(lambo2)

        buy_1 = (
                #(dataframe['pump_strength'] < 0) &
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < self.buy_rsi_fast_32.value) &
                (dataframe['rsi'] > self.buy_rsi_32.value) &
                (dataframe['close'] < dataframe['sma_15'] * self.buy_sma15_32.value) &
                (dataframe['cti'] < self.buy_cti_32.value)
        )
        #conditions.append(buy_1)
        #dataframe.loc[buy_1, 'enter_tag'] += 'E0'
        buy_2 = (
                (dataframe['close'] < (dataframe['sma_15'] * low_offset)) &
                (dataframe['close'] > dataframe['close'].shift(4)) &
                (dataframe['close'].shift(8) > dataframe['close'].shift(4)) &
                (dataframe['close'].shift(12) > dataframe['close'].shift(8)) &
                (dataframe['volume'] > 0)
        )
        conditions.append(buy_2)
        dataframe.loc[buy_2, 'enter_tag'] += 'BT'


        buy_5 = (
                (dataframe['cci'] < -100) &
                (dataframe['fastk'] >= 20) & (dataframe['fastk'] <= 80) &
                (dataframe['fastd'] >= 20) & (dataframe['fastd'] <= 80) &
                (dataframe['trend_close_5m'] >= dataframe['trend_open_5m']) &
                (dataframe['close'] < dataframe['dc_lf']) &
                (dataframe['rocr'] < 0.965) &   # 0.985
                (dataframe['ha_high'].shift(1) > dataframe['dc_lf'].shift(1)) &
                (dataframe['ha_high'] < dataframe['dc_lf']) &
                (dataframe['fisher'] > -0.95) &
                (dataframe['vwap_low'] > dataframe['close'])
        )
        conditions.append(buy_5)
        dataframe.loc[buy_5, 'enter_tag'] += 'ScalpROI'

        buy_6 = (
                #(dataframe['mfi'] < 27) &
                #(dataframe['cti'] < -0.8) &
                #(dataframe['vwap_low'].shift(1) > dataframe['close'].shift(1)) & #Testing
                #(dataframe['close'] < dataframe['vwap_low']) &
                #(dataframe['tcp_percent_4'] > 0.01) &
                (dataframe['bb_width'] > 0.1) &
                (dataframe['close'] < dataframe['dc_lf']) & #Testing
                (dataframe['rocr'] < 0.95) &
                (dataframe['ema_3'] <= 0.988*dataframe['bb_lowerband']) &
                (dataframe['volume'] > 0)
        )
        conditions.append(buy_6)
        dataframe.loc[buy_6, 'enter_tag'] += 'JT_1'

        buy_7 = (
                (dataframe['bb_delta'] > 0.025) &
                (dataframe['bb_width'] > 0.095) &
                (dataframe['closedelta'] > dataframe['close'] * 15.0 / 1000 ) &    # from BinH
                (dataframe['close'] < dataframe['bb_lowerband3'] * 0.995)
        )
        conditions.append(buy_7)
        dataframe.loc[buy_7, 'enter_tag'] += 'Break Signal'




        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        buy_tag = 'empty'
        if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag
        buy_tags = buy_tag.split()

        if "ScalpROI" in buy_tags and current_candle is not None:
            if current_profit >= 0.0065:
                return 'scalp_min_profit'
            if current_time - timedelta(minutes=5) > trade.open_date_utc and (current_profit < 0) and (current_candle['pump_strength'] > 0) and (current_candle['ewo'] < previous_candle['ewo']) and (current_candle['ema_5'] < previous_candle['ema_5']) and (current_candle['ewo'] > 0):
                return 'safe_scalp'
        if current_time - timedelta(hours=6) > trade.open_date_utc:
            if (current_profit > 0) and (current_candle['ema_5'] < previous_candle['ema_5']) and (current_profit < 0.01):
                return "protection_no_profit"
        #if (current_profit < 0) and (current_candle['goodloss'] == 1):
        #    return "good_loss"

        # stoploss - deadfish
        #if ((current_profit < -0.05)
        #        and (current_candle['bb_width'] < 0.05)
        #        and (current_candle['close'] > current_candle['bb_middleband2'] * 1.0)
        #        and (current_candle['volume_mean_12'] < current_candle['volume_mean_24'] * 1.0)):
        #    return "sell_stoploss_deadfish"
        #return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        #Segnale di uscita BigTrade - Grande profitto
        dataframe.loc[
            (
                (dataframe['open'] > (dataframe['sma_15'] * high_offset))
                &
                (dataframe['open'] < dataframe['close'].shift(4))
                &
                (dataframe['close'].shift(8) < dataframe['close'].shift(4))
                &
                (dataframe['close'].shift(12) < dataframe['close'].shift(8))
                &
                (dataframe['volume'] > 0)
                ), ['exit_long', 'exit_tag']] = (1, 'BigTrade')

        #Segnale di uscita mio - Grande profitto in discesa
        dataframe.loc[
            (dataframe['ema_5'] < dataframe['ema_5'].shift(1)) &
            #(dataframe['bb_upperband2'].shift(1) < dataframe['close'].shift(1)) &
            #(dataframe['vwap_upperband'].shift(1) < dataframe['close'].shift(1)) & # Testing
            (dataframe['dc_hf'].shift(1) < dataframe['close'].shift(1)) &
            (dataframe['fisher'].shift(1) > 0.90) &
            (dataframe['rsi_fast'] < dataframe['rsi_fast'].shift(1)) &
            (dataframe['volume'] > 0), ['exit_long', 'exit_tag']] = (1, 'BigProfitEMALow')
        #Protezione chiusura veloce basso profitto - LowFish
        dataframe.loc[
            #(dataframe['bb_lowerband2'] > dataframe['bb_lowerband2'].shift(1)) &
            (dataframe['volume_mean_12'] < dataframe['volume_mean_24']) &
            (dataframe['ema_200'] < dataframe['ema_200'].shift(1)) & #Segnale facile solo caduta.
            #(dataframe['vwap_low'] < dataframe['vwap_low'].shift(1)) & #Testing old
            #(dataframe['pump_strength'] > 0) &
            #(dataframe['rsi_fast'] > 50) &
            #(dataframe['fisher'] > 0.80) &
            (dataframe['dc_hf'] < dataframe['close']) &
            (dataframe['vwma'] < dataframe['vwma'].shift(1)) & #Testing
            (dataframe['volume'] > 0), ['exit_long', 'exit_tag']] = (1, 'FastVolumeClose')

        dataframe.loc[
            (dataframe['rsi_fast'] < dataframe['rsi_fast'].shift(1)) &
            (dataframe['fisher'] > 0.77) & #0.38414
            (dataframe['ha_high'].le(dataframe['ha_high'].shift(1))) &
            (dataframe['ha_high'].shift(1).le(dataframe['ha_high'].shift(2))) &
            (dataframe['ha_close'].le(dataframe['ha_close'].shift(1))) &
            (dataframe['ema_3'] > dataframe['ha_close']) &
            ((dataframe['ha_close'] * 1.07634) > dataframe['bb_middleband2']) &
            (dataframe['volume'] > 0)
            , ['exit_long', 'exit_tag']] = (1, 'altrastrMODME')

        dataframe.loc[
            (dataframe['pump_strength'] > 0) &
            (dataframe['fisher'] > 0) &
            (dataframe['rsi_fast'].shift(1) > 70) &
            (dataframe['vwma'] < dataframe['close']) &
            (dataframe['trend_close_15m'] < dataframe['vwap_upperband']) &
            (dataframe['trend_close_15m'] < dataframe['trend_open_15m']) &
            #(dataframe['ema_200'] < dataframe['ema_200'].shift(1)) & #USARE ICHIMOKU ANZICHE' EMA, vendere solo quando il trend non è piu confermato
            #(dataframe['dc_hf'].shift(1) < dataframe['close'].shift(1)) &
            #(dataframe['vwap_upperband'].shift(1) < dataframe['close'].shift(1)) &
            #(dataframe['bb_upperband2'].shift(1) < dataframe['close'].shift(1)) &
            #(dataframe['dc_hf'] > dataframe['close']) &
            #(dataframe['vwap_upperband'] > dataframe['close']) &
            #(dataframe['bb_upperband2'] > dataframe['close']) &
            (dataframe['volume'] > 0)
            , ['exit_long', 'exit_tag']] = (0, 'LowTrend')
        dataframe.loc[
            #(dataframe['rsi_fast'].shift(1) > 70) &
            #(dataframe['vwma'] < dataframe['close']) &
            #(dataframe['tenkan_sen'] < dataframe['tenkan_sen'].shift(1)) &
            #(dataframe['tenkan_sen'] > dataframe['close']) &
            (dataframe['pump_strength'] > 0) &
            (dataframe['fisher'] > 0) &
            (dataframe['tenkan_sen'] < dataframe['kijun_sen']) &
            (dataframe['trend_close_15m'] < dataframe['vwap_upperband']) &
            (dataframe['trend_close_15m'] < dataframe['trend_open_15m']) &
            (dataframe['volume'] > 0)
            , ['exit_long', 'exit_tag']] = (0, 'LowTrendIchi')


        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                                  current_rate: float, current_profit: float,
                                  min_stake: Optional[float], max_stake: float,
                                  current_entry_rate: float, current_exit_rate: float,
                                  current_entry_profit: float, current_exit_profit: float,
                                  **kwargs) -> Optional[float]:

            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            last_candle = dataframe.iloc[-1].squeeze()
            previous_candle = dataframe.iloc[-2].squeeze()

            if ((current_profit > (self.dca_threshold_pct.value * 3)) and (trade.nr_of_successful_exits == 0)):
                return -(trade.stake_amount / 2)

            if trade.id in self.last_dca_timeframe.keys():
                td = current_time - self.last_dca_timeframe[trade.id]
                if ((td.total_seconds() / 60)
                        < (self.timeframe_to_minutes(self.timeframe) * self.candles_before_dca.value)):
                    return None

            filled_entries = trade.select_filled_orders(trade.entry_side)
            count_of_entries = trade.nr_of_successful_entries - trade.nr_of_successful_exits
            if ((count_of_entries >= self.max_entry_position_adjustment) or
                    (last_candle['close'] < previous_candle['close']) or
                    (current_profit > -self.dca_threshold_pct.value)):
                return None

            if ((current_profit < -self.dca_threshold_pct.value * count_of_entries) and (
                    (last_candle['dca_buy_signal'] == 1) or (last_candle['dca_buy_signal2'] == 1) or (previous_candle['dca_buy_signal'] == 1) or (previous_candle['dca_buy_signal2'] == 1))):
                try:
                    averaged_stake = filled_entries[0].stake_amount
                except Exception as exception:
                    averaged_stake = 5
                    pass
                if (count_of_entries > 0):
                    averaged_stake = (sum([entry.stake_amount
                                           for entry in filled_entries if entry.stake_amount > 0]) / count_of_entries)
                stake_amount = averaged_stake * (1 + (count_of_entries * 0.25))
                self.last_dca_timeframe[trade.id] = current_time
                return stake_amount

    def timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert a timeframe string to minutes."""
        if 'm' in timeframe:
            return int(timeframe.replace('m', ''))
        elif 'h' in timeframe:
            return int(timeframe.replace('h', '')) * 60
        elif 'd' in timeframe:
            return int(timeframe.replace('d', '')) * 1440
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
# VWAP bands
def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df,window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']


def top_percent_change(dataframe: DataFrame, length: int) -> float:
        """
        Percentage change of the current close from the range maximum Open price

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        if length == 0:
            return (dataframe['open'] - dataframe['close']) / dataframe['close']
        else:
            return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']

def calc_zvwap(dataframe, pds, source1):
    volume = dataframe['volume'].rolling(pds).mean()
    close = dataframe['close'].rolling(pds).mean()

    mean = indicators.sma(volume * source1, pds) / indicators.sma(volume, pds)
    vwapsd = indicators.sma(pow(source1 - mean, 2), pds).apply(lambda x: pow(x, 0.5))
    zvwap = (close - mean) / vwapsd

    return zvwap

# Elliot Wave Oscillator
def EWO(dataframe, sma1_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.EMA(df, timeperiod=sma1_length)
    sma2 = ta.EMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif