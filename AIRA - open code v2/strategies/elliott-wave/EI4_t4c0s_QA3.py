import logging
from functools import reduce
from datetime import datetime
from typing import Optional
from typing import Dict, List
import pandas as pd
import numpy as np
import math
import technical.indicators as ftt
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
from freqtrade.persistence import Trade
from datetime import datetime, timedelta

from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy, merge_informative_pair


logger = logging.getLogger(__name__)

def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


class EI4_t4c0s_QA3(IStrategy):

    """
          ______   __          __              __    __   ______   __    __        __     __    __             ______            
     /      \ /  |       _/  |            /  |  /  | /      \ /  \  /  |      /  |   /  |  /  |           /      \           
    /$$$$$$  |$$ |____  / $$ |    _______ $$ | /$$/ /$$$$$$  |$$  \ $$ |     _$$ |_  $$ |  $$ |  _______ /$$$$$$  |  _______ 
    $$ |  $$/ $$      \ $$$$ |   /       |$$ |/$$/  $$ ___$$ |$$$  \$$ |    / $$   | $$ |__$$ | /       |$$$  \$$ | /       |
    $$ |      $$$$$$$  |  $$ |  /$$$$$$$/ $$  $$<     /   $$< $$$$  $$ |    $$$$$$/  $$    $$ |/$$$$$$$/ $$$$  $$ |/$$$$$$$/ 
    $$ |   __ $$ |  $$ |  $$ |  $$ |      $$$$$  \   _$$$$$  |$$ $$ $$ |      $$ | __$$$$$$$$ |$$ |      $$ $$ $$ |$$      \ 
    $$ \__/  |$$ |  $$ | _$$ |_ $$ \_____ $$ |$$  \ /  \__$$ |$$ |$$$$ |      $$ |/  |     $$ |$$ \_____ $$ \$$$$ | $$$$$$  |
    $$    $$/ $$ |  $$ |/ $$   |$$       |$$ | $$  |$$    $$/ $$ | $$$ |______$$  $$/      $$ |$$       |$$   $$$/ /     $$/ 
     $$$$$$/  $$/   $$/ $$$$$$/  $$$$$$$/ $$/   $$/  $$$$$$/  $$/   $$//      |$$$$/       $$/  $$$$$$$/  $$$$$$/  $$$$$$$/  
                                                                       $$$$$$/                                               
Here be stonks
1. freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --strategy RL_kdog_spot --freqaimodel ReinforcementLearnerSpot --spaces roi stoploss --timerange "$(date --date='-1 week' '+%Y%m%d')"-"$(date '+%Y%m%d')" -e 1000
2. freqtrade trade --logfile ./logs --freqaimodel ReinforcementLearnerSpot --strategy RL_kdog_spot
    """

    minimal_roi = {"0": 0.1, "2400": -1}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "prediction": {"prediction": {"color": "blue"}},
            "target_roi": {
                "target_roi": {"color": "brown"},
            },
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
            "&-action": {
                "&-action": {"color": "green"},
            },
        },
    }
    timeframe = '5m'
    # position_adjustment_enable = True
    # max_entry_position_adjustment = 2
    # max_dca_multiplier = 4
    process_only_new_candles = True
    stoploss = -0.3
    use_exit_signal = True
    startup_candle_count: int = 300
    can_short = False
    # Specific variables
    linear_roi_offset = DecimalParameter(0.00, 0.02, default=0.005, space="sell", optimize=False, load=True)
    max_roi_time_long = IntParameter(0, 800, default=400, space="sell", optimize=False, load=True)

    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy": 12,
        "rsi_buy": 58,
        "low_offset": 0.987,
        "lambo2_ema_14_factor": 0.981,
        "lambo2_enabled": True,
        "lambo2_rsi_14_limit": 39,
        "lambo2_rsi_4_limit": 44,
        "buy_adx": 20,
        "buy_fastd": 20,
        "buy_fastk": 22,
        "buy_ema_cofi": 0.98,
        "buy_ewo_high": 4.179
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 22,
        "high_offset": 1.014,
        "high_offset_2": 1.01
    }

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]

    # ROI table:
    minimal_roi = {
        "0": 0.99,
        
    }

    # Stoploss:
    stoploss = -0.99

    # SMAOffset
    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)
    low_offset = DecimalParameter(0.985, 0.995, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(1.005, 1.015, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.010, 1.020, default=sell_params['high_offset_2'], space='sell', optimize=True)

    # lambo2
    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3,  default=buy_params['lambo2_ema_14_factor'], space='buy', optimize=True)
    lambo2_rsi_4_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_4_limit'], space='buy', optimize=True)
    lambo2_rsi_14_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_14_limit'], space='buy', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=False)

    #cofi
    is_optimize_cofi = False
    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=0.97 , optimize = is_optimize_cofi)
    buy_fastk = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    buy_fastd = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    buy_adx = IntParameter(20, 30, default=30, optimize = is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize = is_optimize_cofi)

    atr_length = IntParameter(10, 30, default=14, space='buy', optimize=True)
    increment = DecimalParameter(low=1.0005, high=1.001, default=1.0007, decimals=4 ,space='buy', optimize=True, load=True)

    use_custom_stoploss = True
    process_only_new_candles = True
    # Custom Entry
    last_entry_price = None
    #     # This is called when placing the initial order (opening trade)
    # def informative_pairs(self):
    #     whitelist_pairs = self.dp.current_whitelist()
    #     corr_pairs = self.config["freqai"]["feature_parameters"]["include_corr_pairlist"]
    #     informative_pairs = []
    #     for tf in self.config["freqai"]["feature_parameters"]["include_timeframes"]:
    #         for pair in whitelist_pairs:
    #             informative_pairs.append((pair, tf))
    #         for pair in corr_pairs:
    #             if pair in whitelist_pairs:
    #                 continue  # avoid duplication
    #             informative_pairs.append((pair, tf))
    #     return informative_pairs


    # def feature_engineering_expand_all(self, dataframe, period, **kwargs):
    #     dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
    #     dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
    #     dataframe["%-rocr-period"] = ta.ROCR(dataframe, timeperiod=period)
    #     dataframe["%-chop-period"] = qtpylib.chopiness(dataframe, period)
    #     dataframe["%-linear-period"] = ta.LINEARREG_ANGLE(
    #         dataframe['close'], timeperiod=period)
    #     dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
    #     dataframe["%-atr-periodp"] = dataframe["%-atr-period"] / \
    #         dataframe['close'] * 1000

        # return dataframe

    def feature_engineering_expand_basic(self, dataframe, metadata, **kwargs):
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-obv"] = ta.OBV(dataframe)

        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        # The following features are necessary for RL models

        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['%-ma_lo'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (self.low_offset.value)
        dataframe['%-hi'] = dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * (self.high_offset.value)
        dataframe['%-ma_hi_2'] = dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * (self.high_offset_2.value)

        dataframe['%-hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        # HMA-BUY SQUEEZE
        dataframe['%-HMA_SQZ'] = (((dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] - dataframe['%-hma_50']) 
            / dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']) * 100)


        dataframe['zero'] = 0
        # Elliot
        dataframe['%-EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        dataframe.loc[dataframe['%-EWO'] > 0, "EWO_UP"] = dataframe['%-EWO']
        dataframe.loc[dataframe['%-EWO'] < 0, "EWO_DN"] = dataframe['%-EWO']
        dataframe['EWO_UP'].ffill()
        dataframe['EWO_DN'].ffill()
        dataframe['%-EWO_MEAN_UP'] = dataframe['EWO_UP'].mean()
        dataframe['%-EWO_MEAN_DN'] = dataframe['EWO_DN'].mean()
        dataframe['%-EWO_UP_FIB'] = dataframe['%-EWO_MEAN_UP'] * 1.618
        dataframe['%-EWO_DN_FIB'] = dataframe['%-EWO_MEAN_DN'] * 1.618

        # RSI
        dataframe['%-rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['%-rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['%-rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        #lambo2
        dataframe['%-ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['%-rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['%-rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['%-fastd'] = stoch_fast['fastd']
        dataframe['%-fastk'] = stoch_fast['fastk']
        dataframe['%-adx'] = ta.ADX(dataframe)
        dataframe['%-ema_8'] = ta.EMA(dataframe, timeperiod=8)

        dataframe['%-OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4

        # Check how far we are from min and max 
        dataframe['%-max'] = dataframe['%-OHLC4'].rolling(4).max() / dataframe['%-OHLC4'] - 1
        dataframe['%-min'] = abs(dataframe['%-OHLC4'].rolling(4).min() / dataframe['%-OHLC4'] - 1)

        dataframe['%-max_l'] = dataframe['%-OHLC4'].rolling(120).max() / dataframe['%-OHLC4'] - 1
        dataframe['%-min_l'] = abs(dataframe['%-OHLC4'].rolling(120).min() / dataframe['%-OHLC4'] - 1)

        # Apply rolling window operation to the 'OHLC4'column
        rolling_window = dataframe['%-OHLC4'].rolling(16) 
        rolling_max = rolling_window.max()
        rolling_min = rolling_window.min()

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_value = rolling_window.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['%-move'] = ptp_value / dataframe['%-OHLC4']
        dataframe['%-move_mean'] = dataframe['%-move'].mean()
        dataframe['%-move_mean_x'] = dataframe['%-move'].mean() * 1.6
        dataframe['%-exit_mean'] = rolling_min * (1 + dataframe['%-move_mean'])
        dataframe['%-exit_mean_x'] = rolling_min * (1 + dataframe['%-move_mean_x'])
        dataframe['%-enter_mean'] = rolling_max * (1 - dataframe['%-move_mean'])
        dataframe['%-enter_mean_x'] = rolling_max * (1 - dataframe['%-move_mean_x'])
        dataframe['%-atr_pcnt'] = (ta.ATR(dataframe, timeperiod=5) / dataframe['%-OHLC4'])
        dataframe[f"%-raw_close"] = dataframe["close"]
        dataframe[f"%-raw_open"] = dataframe["open"]
        dataframe[f"%-raw_high"] = dataframe["high"]
        dataframe[f"%-raw_low"] = dataframe["low"]

        return dataframe


    def set_freqai_targets(self, dataframe, **kwargs):
        dataframe['OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4

        # Apply rolling window operation to the 'OHLC4'column
        rolling_window = dataframe['OHLC4'].rolling(16) 
        rolling_max = rolling_window.max()
        rolling_min = rolling_window.min()

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_value = rolling_window.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['&s-move'] = ptp_value / dataframe['OHLC4']
      
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self.freqai.start(dataframe, metadata, self)
       # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['ma_lo'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (self.low_offset.value)
        dataframe['ma_hi'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (self.high_offset.value)
        dataframe['ma_hi_2'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (self.high_offset_2.value)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        # HMA-BUY SQUEEZE
        dataframe['HMA_SQZ'] = (((dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] - dataframe['hma_50']) 
            / dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']) * 100)


        dataframe['zero'] = 0
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        dataframe.loc[dataframe['EWO'] > 0, "EWO_UP"] = dataframe['EWO']
        dataframe.loc[dataframe['EWO'] < 0, "EWO_DN"] = dataframe['EWO']
        dataframe['EWO_UP'].ffill()
        dataframe['EWO_DN'].ffill()
        dataframe['EWO_MEAN_UP'] = dataframe['EWO_UP'].mean()
        dataframe['EWO_MEAN_DN'] = dataframe['EWO_DN'].mean()
        dataframe['EWO_UP_FIB'] = dataframe['EWO_MEAN_UP'] * 1.618
        dataframe['EWO_DN_FIB'] = dataframe['EWO_MEAN_DN'] * 1.618

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        #lambo2
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)

        dataframe['OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4

        # Check how far we are from min and max 
        dataframe['max'] = dataframe['OHLC4'].rolling(4).max() / dataframe['OHLC4'] - 1
        dataframe['min'] = abs(dataframe['OHLC4'].rolling(4).min() / dataframe['OHLC4'] - 1)

        dataframe['max_l'] = dataframe['OHLC4'].rolling(120).max() / dataframe['OHLC4'] - 1
        dataframe['min_l'] = abs(dataframe['OHLC4'].rolling(120).min() / dataframe['OHLC4'] - 1)

        # Apply rolling window operation to the 'OHLC4'column
        rolling_window = dataframe['OHLC4'].rolling(16) 
        rolling_max = rolling_window.max()
        rolling_min = rolling_window.min()

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_value = rolling_window.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['move'] = ptp_value / dataframe['OHLC4']
        dataframe['move_mean'] = dataframe['move'].mean()
        dataframe['move_mean_x'] = dataframe['move'].mean() * 1.6
        dataframe['exit_mean'] = rolling_min * (1 + dataframe['move_mean'])
        dataframe['exit_mean_x'] = rolling_min * (1 + dataframe['move_mean_x'])
        dataframe['enter_mean'] = rolling_max * (1 - dataframe['move_mean'])
        dataframe['enter_mean_x'] = rolling_max * (1 - dataframe['move_mean_x'])
        dataframe['atr_pcnt'] = (ta.ATR(dataframe, timeperiod=5) / dataframe['OHLC4'])


        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        lambo2 = (
            (df['close'] < (df['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (df['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (df['rsi_14'] < int(self.lambo2_rsi_14_limit.value)) &
            (df['atr_pcnt'] > df['min_l']) &
            (df['do_predict'] == 1) &
            # (df["&-action"] == 1) &
            (df['volume'] > 0) 
        )
        df.loc[lambo2, 'enter_long'] = 1
        df.loc[lambo2, 'enter_tag'] = 'lambo '

        buy1ewo = (
            (df['rsi_fast'] < 35 ) &
            (df['close'] < df['ma_lo']) &
            (df['EWO'] > df['EWO_MEAN_UP']) &
            (df['close'] < df['enter_mean_x']) &
            (df['close'].shift() < df['enter_mean_x'].shift()) &
            (df['rsi'] < self.rsi_buy.value) &
            (df['atr_pcnt'] > df['min']) &
            (df['do_predict'] == 1) &
            # (df["&-action"] == 1) &
            (df['volume'] > 0) 
        )
        df.loc[buy1ewo, 'enter_long'] = 1
        df.loc[buy1ewo, 'enter_tag'] = 'buy1ewo'

        buy2ewo = (
            (df['rsi_fast'] < 35) &
            (df['close'] < df['ma_lo']) &
            (df['EWO'] < df['EWO_DN_FIB']) &
            (df['atr_pcnt'] > df['min']) &
            (df['do_predict'] == 1) &
            # (df["&-action"] == 1) &
            (df['volume'] > 0) 
        )
        df.loc[buy2ewo, 'enter_long'] = 1
        df.loc[buy2ewo, 'enter_tag'] = 'buy2ewo'

        is_cofi = (
            (df['open'] < df['ema_8'] * self.buy_ema_cofi.value) &
            (qtpylib.crossed_above(df['fastk'], df['fastd'])) &
            (df['fastk'] < self.buy_fastk.value) &
            (df['fastd'] < self.buy_fastd.value) &
            (df['adx'] > self.buy_adx.value) &
            (df['EWO'] > df['EWO_MEAN_UP']) &
            (df['atr_pcnt'] > df['min']) &
            (df['do_predict'] == 1) &
            # (df["&-action"] == 1) &
            (df['volume'] > 0) 
            )
        df.loc[is_cofi, 'enter_long'] = 1
        df.loc[is_cofi, 'enter_tag'] = 'cofi'

        # is_rl = (
        #     (df["&-action"].shift() != 1) &
        #     (df['do_predict'] == 1) &
        #     (df["&-action"] == 1) &
        #     (df['volume'] > 0) 
        #     )
        # df.loc[is_cofi, 'enter_long'] = 1
        # df.loc[is_cofi, 'enter_tag'] = 'RL Entry'

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        condition5 = (
                (df['close'] > df['hma_50']) &
                (df['close'] > df['ma_hi_2']) &
                (df['close'] > df['exit_mean_x']) &
                (df['rsi'] > 50 ) &
                (df['do_predict'] == 1) &
                # (df["&-action"] == 2) &
                (df['volume'] > 0 ) &
                (df['rsi_fast']>df['rsi_slow'])

            )
        df.loc[condition5, 'exit_long'] = 1
        df.loc[condition5, 'exit_tag'] = 'Close > Offset Hi 2'


        
        condition6 = (
                (df['close'] < df['hma_50']) &
                (df['close'] > df['ma_hi']) &
                (df['volume'] > 0) &
                (df['do_predict'] == 1) &
                # (df["&-action"] == 2) &
                (df['rsi_fast']>df['rsi_slow'])

            )
        df.loc[condition6, 'exit_long'] = 1
        df.loc[condition6, 'exit_tag'] = 'Close > Offset Hi 1'

        # condition7 = (

        #         (df['volume'] > 0) &
        #         (df['do_predict'] == 1) &
        #         (df["&-action"] == 2) &
        #         (df["&-action"].shift() != 2) 

        #     )
        # df.loc[condition7, 'exit_long'] = 1
        # df.loc[condition7, 'exit_tag'] = 'RL Exit'

        return df

    # def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
    #                         proposed_stake: float, min_stake: float, max_stake: float,
    #                         entry_tag: Optional[str], **kwargs) -> float:

    #     # We need to leave most of the funds for possible further DCA orders
    #     # This also applies to fixed stakes
    #     return proposed_stake / self.max_dca_multiplier

    # def adjust_trade_position(self, trade: Trade, current_time: datetime,
    #                           current_rate: float, current_profit: float, min_stake: float,
    #                           max_stake: float, **kwargs):
    #     """
    #     Custom trade adjustment logic, returning the stake amount that a trade should be increased.
    #     This means extra buy orders with additional fees.

    #     :param trade: trade object.
    #     :param current_time: datetime object, containing the current datetime
    #     :param current_rate: Current buy rate.
    #     :param current_profit: Current profit (as ratio), calculated based on current_rate.
    #     :param min_stake: Minimal stake size allowed by exchange.
    #     :param max_stake: Balance available for trading.
    #     :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
    #     :return float: Stake amount to adjust your trade
    #     """

    #     #if current_profit > -0.01:
    #         #return None
    #     # Don't rebuy for trades on hold
    #     is_backtest = self.dp.runmode.value == 'backtest'
    #     if (trade.open_date_utc.replace(tzinfo=None) < datetime(2022, 4, 6) and not is_backtest):
    #         return None
    #     dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
    #     if(len(dataframe) < 2):
    #         return None
    #     last_candle = dataframe.iloc[-1].squeeze()
    #     previous_candle = dataframe.iloc[-2].squeeze()

    #     filled_buys = trade.select_filled_orders()
    #     count_of_buys = trade.nr_of_successful_entries
    #     try:
    #         stake_amount = filled_buys[0].cost
    #         stake_amount = stake_amount * (1 + (count_of_buys * 0.25))
    #         if (
    #                 ((last_candle['do_predict'] == 1)
    #                 and (last_candle['&-action'] == 2)
    #                 and count_of_buys == 1)
    #                 or ((previous_candle['do_predict'] == 1)
    #                 and (previous_candle['&-action'] == 2)
    #                 and count_of_buys == 1)
    #         ):
    #             return stake_amount
    #         if (
    #                 ((last_candle['do_predict'] == 1)
    #                 and (last_candle['&-action'] == 3)
    #                 and count_of_buys == 2)
    #                 or ((previous_candle['do_predict'] == 1)
    #                 and (previous_candle['&-action'] == 3)
    #                 and count_of_buys == 2)
    #         ):
    #             return stake_amount
    #     except Exception as exception:
    #         logger.warning(f"{exception}")

    #     return None


    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:


        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        SLT1 = current_candle['move_mean']
        SL1 = 0.013
        SLT2 = current_candle['move_mean_x']
        SL2 = current_candle['move_mean_x'] - current_candle['move_mean']
        display_profit = current_profit * 100
        slt1 = SLT1 * 100
        sl1 = SL1 * 100
        slt2 = SLT2 * 100
        sl2 = SL2 * 100


        if current_candle['max_l'] > .003: #ignore stoploss if setting new highs
            if SLT2 is not None and current_profit > SLT2:
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.2f}% - {slt2:.2f}%/{sl2:.2f}% activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.2f}% - {slt2:.2f}%/{sl2:.2f}% activated')
                return SL2
            if SLT1 is not None and current_profit > SLT1:
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.2f}% - {SLT1:.2f}%/{SL1:.2f}% activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.2f}% - {slt1:.2f}%/{sl1:.2f}% activated')
                return SL1

        else:
            if SLT1 is not None and current_profit > SL1:
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.2f}% SWINGING FOR THE MOON!!!')
                logger.info(f'*** {pair} *** Profit {display_profit:.2f}% SWINGING FOR THE MOON!!!')
                return 0.99

        return self.stoploss


    def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)

        entry_price = (dataframe['close'].iat[-1] + dataframe['open'].iat[-1] + proposed_rate + proposed_rate) / 4
        logger.info(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}") 

        # Check if there is a stored last entry price and if it matches the proposed entry price
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.0001:  # Tolerance for floating-point comparison
            entry_price *= self.increment.value # Increment by 0.2%
            logger.info(f"{pair} Incremented entry price: {entry_price} based on previous entry price : {self.last_entry_price}.")

        # Update the last entry price
        self.last_entry_price = entry_price

        return entry_price


    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if exit_reason == 'roi' and (last_candle['max_l'] < 0.003):
            return False

        # Handle freak events

        if exit_reason == 'Down Trend Soon' and trade.calc_profit_ratio(rate) < 0.003:
            logger.info(f"{trade.pair} Waiting for Profit")
            self.dp.send_msg(f'{trade.pair} Waiting for Profit')
            return False

        if exit_reason == 'roi' and trade.calc_profit_ratio(rate) < 0.003:
            logger.info(f"{trade.pair} ROI is below 0")
            self.dp.send_msg(f'{trade.pair} ROI is below 0')
            return False

        if exit_reason == 'partial_exit' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} partial exit is below 0")
            self.dp.send_msg(f'{trade.pair} partial exit is below 0')
            return False

        if exit_reason == 'trailing_stop_loss' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} trailing stop price is below 0")
            self.dp.send_msg(f'{trade.pair} trailing stop price is below 0')
            return False

        return True

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        # Sell any positions at a loss if they are held for more than 7 days.
        if current_profit < -0.04 and (current_time - trade.open_date_utc).days >= 4:
            return 'unclog'
    
    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    def pct_change(a, b):
        return (b - a) / a