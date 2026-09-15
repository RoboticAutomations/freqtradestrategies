# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from functools import reduce
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union


from freqtrade.persistence import Trade

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, stoploss_from_open, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib

# custom indicators
# #############################################################################################################################################################################################


#  Coral Trend Indicator
'''
The "Coral Trend Indicator" is a trend-following indicator that uses a combination of exponential moving averages (EMAs) and the Commodity Channel Index (CCI) to identify trends and potential reversal points

This code imports the necessary libraries (pandas and numpy) and defines a function called coral_trend() that takes a DataFrame of cryptocurrency data (with a "close" column) and the EMA period, CCI period, and CCI threshold as inputs. The function first calculates the fast and slow exponential moving averages (EMAs) of the cryptocurrency and stores them in new columns called "ema_fast" and "ema_slow", respectively. It then calculates the Commodity Channel Index (CCI) of the cryptocurrency and stores it in a new column called "cci".

The function then determines the trend based on the EMAs and CCI, and stores it in a new column called "coral_trend".

The coral_trend() function determines the trend by using the np.where() function to set the value of the "coral_trend" column based on the following conditions:

    If the fast EMA is greater than the slow EMA and the CCI is less than the negative CCI threshold, then the trend is set to 1 (indicating an uptrend).
    If the fast EMA is less than the slow EMA and the CCI is greater than the positive CCI threshold, then the trend is set to -1 (indicating a downtrend).
    Otherwise, the trend is set to 0 (indicating no trend or a neutral market).

The coral_trend() function then returns the modified DataFrame with the "coral_trend" column added.
'''
def coral_trend(df, ema_period=10, cci_period=20, cci_threshold=100):
    # Calculate the exponential moving averages (EMAs)
    df['ema_fast'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_period*2, adjust=False).mean()

    # Calculate the Commodity Channel Index (CCI)
    df['cci'] = ((df['close'] - df['close'].rolling(cci_period).mean()) / df['close'].rolling(cci_period).std()) * np.sqrt(cci_period)

    # Determine the trend based on the EMAs and CCI
    df['coral_trend'] = np.where((df['ema_fast'] > df['ema_slow']) & (df['cci'] < -cci_threshold), 1, 0)
    df['coral_trend'] = np.where((df['ema_fast'] < df['ema_slow']) & (df['cci'] > cci_threshold), -1, df['coral_trend'])

    return df

# Chandalier Exit
'''
The "Chandelier Exit" technical indicator, also known as the "ATR Trailing Stop", is a volatility-based stop loss indicator that adjusts the stop loss level based on the average true range (ATR) of a cryptocurrency.

This code imports the necessary libraries (pandas and numpy) and defines a function called chandelier_exit() that takes a DataFrame of cryptocurrency data (with columns for "high", "low", and "close") and the ATR period and ATR multiplier as inputs. The function first calculates the ATR of the cryptocurrency and stores it in a new column called "atr". It then calculates the Chandelier Exit (ATR Trailing Stop) by subtracting the ATR multiplied by the ATR multiplier from the high price and stores it in a new column called "chandelier_exit".
'''
def chandelier_exit(df, atr_period=2, rolling_value=10, atr_mult=2.3):
    # Calculate the average true range (ATR)
    df['TR'] = df[['high', 'low', 'close']].apply(lambda x: max(x) - min(x), axis=1)
    df['atr'] = df['TR'].rolling(atr_period).mean()

    # Calculate the Chandelier Exit (ATR Trailing Stop)
    df['chandelier_exit_sl'] = df['close'] - df['atr'] * atr_mult
    df['chandelier_exit_tp'] = df['close'] + df['atr'] * atr_mult
    df['chandelier_exit_tp_short'] = df['close'] - df['atr'] * atr_mult
    df['chandelier_exit_sl_short'] = df['close'] + df['atr'] * atr_mult
    df['chandelier_exit_enter_long'] = df['high'].rolling(rolling_value).max() - (df['atr'] * atr_mult)
    df['chandelier_exit_enter_short'] = df['low'].rolling(rolling_value).min() + (df['atr'] * atr_mult)
    return df


# STC Indicator - A better MACD
'''
The stc_indicator() function takes a DataFrame of cryptocurrency data (with a "close" column) and the MACD fast period, MACD slow period, MACD signal period, Stochastic Oscillator K period, and Stochastic Oscillator D period as inputs. It then calculates the MACD and the Stochastic Oscillator of the cryptocurrency and stores them in new columns called "macd" and "stoch_k", respectively. It also calculates the MACD signal line and the Stochastic Oscillator D line

The stc_indicator() function then determines the trend based on the MACD and Stochastic Oscillator using the np.where() function. The trend is stored in a new column called "stc_trend". If the MACD is greater than the MACD signal line and the Stochastic Oscillator K is greater than the Stochastic Oscillator D, then the trend is set to 1 (indicating an uptrend). If the MACD is less than the MACD signal line and the Stochastic Oscillator K is less than the Stochastic Oscillator D, then the trend is set to -1 (indicating a downtrend). Otherwise, the trend is set to 0 (indicating no trend or a neutral market).
'''
def stc_indicator(df, macd_fast=12, macd_slow=26, macd_signal=9, stoch_k=14, stoch_d=3):
    # Calculate the MACD
    df['macd'] = df['close'].ewm(span=macd_fast, adjust=False).mean() - df['close'].ewm(span=macd_slow, adjust=False).mean()
    df['macd_signal'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()

    # Calculate the Stochastic Oscillator
    df['stoch_k'] = (df['close'] - df['low'].rolling(stoch_k).min()) / (df['high'].rolling(stoch_k).max() - df['low'].rolling(stoch_k).min()) * 100
    df['stoch_d'] = df['stoch_k'].rolling(stoch_d).mean()

    # Determine the trend based on the MACD and Stochastic Oscillator
    df['stc_trend'] = np.where((df['macd'] > df['macd_signal']) & (df['stoch_k'] > df['stoch_d']), 1, 0)
    df['stc_trend'] = np.where((df['macd'] < df['macd_signal']) & (df['stoch_k'] < df['stoch_d']), -1, df['stc_trend'])

    return df

# ############################################################################################################################################################################################

class picasso_CE_CTI_STC_EMA_1h_V5_4x_3mt_Jan16_np_Jan20(IStrategy):


    # class CE_CTI_STC_EMA_1h_V4_4x_4mt_Jan6(IStrategy):

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Can this strategy go short?
    can_short: bool = True


    '''
        docker-compose run --rm freqtrade backtesting --strategy CE_CTI_STC_EMA_1h_V5_3x_3mt_Jan16 -i 1h --export trades --breakdown month --timerange 20210101-20230115
        ((((3mt 320 pt))))
            ============================================================== ENTER TAG STATS ===============================================================
        |          TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
        |--------------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
        |     L_ema_ct |      3778 |           1.34 |        5067.87 |         16162.305 |        1616.23 |        2:11:00 |  2550     3  1225  67.5 |
        |     S_ema_ct |      3355 |           1.32 |        4416.73 |         14097.203 |        1409.72 |        2:29:00 |  2298     3  1054  68.5 |
        | L_adx_stc_ct |      3786 |           0.94 |        3540.03 |         11288.171 |        1128.82 |        2:15:00 |  2428     5  1353  64.1 |
        | S_adx_stc_ct |      2677 |           1.18 |        3158.16 |         10089.188 |        1008.92 |        2:26:00 |  1740     3   934  65.0 |
        |     L_stc_ct |      1445 |           1.26 |        1827.23 |          5831.890 |         583.19 |        2:41:00 |   977     2   466  67.6 |
        |     S_ct_stc |       690 |           1.30 |         897.04 |          2862.876 |         286.29 |        2:59:00 |   477     0   213  69.1 |
        |        TOTAL |     15731 |           1.20 |       18907.05 |         60331.634 |        6033.16 |        2:24:00 | 10470    16  5245  66.6 |
        ========================================================= EXIT REASON STATS =========================================================
        |           Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
        |-----------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
        |    trailing_stop_loss |    8919 |   7737     0  1182  86.7 |           4.11 |       36656.3  |        117007     |       12218.8  |
        |             stop_loss |    3990 |      0     0  3990     0 |         -14.06 |      -56100.3  |       -179080     |      -18700.1  |
        |                   roi |    2774 |   2721    16    37  98.1 |          13.97 |       38764.5  |        123724     |       12921.5  |
        |            ce_tp_long |      17 |      1     0    16   5.9 |          -5.65 |         -95.98 |          -305.655 |         -31.99 |
        |            ce_sl_long |      14 |      7     0     7  50.0 |          -0.58 |          -8.11 |           -25.896 |          -2.7  |
        | ce_sl_shortce_tp_long |      11 |      2     0     9  18.2 |          -2.83 |         -31.18 |           -99.677 |         -10.39 |
        | ce_tp_shortce_sl_long |       3 |      2     0     1  66.7 |          -1.65 |          -4.94 |           -15.819 |          -1.65 |
        |           liquidation |       3 |      0     0     3     0 |         -91.08 |        -273.23 |          -872.56  |         -91.08 |
        ======================================================= LEFT OPEN TRADES REPORT ========================================================
        |   Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
        |--------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
        |  TOTAL |         0 |           0.00 |           0.00 |             0.000 |           0.00 |           0:00 |     0     0     0     0 |
        ======================= MONTH BREAKDOWN ========================
        |      Month |   Tot Profit USDT |   Wins |   Draws |   Losses |
        |------------+-------------------+--------+---------+----------|
        | 31/01/2021 |          5209.44  |    617 |       0 |      385 |
        | 28/02/2021 |          5522.81  |    605 |       0 |      355 |
        | 31/03/2021 |          1578.98  |    465 |       0 |      252 |
        | 30/04/2021 |          2668.98  |    554 |       1 |      325 |
        | 31/05/2021 |          5253.01  |    685 |       1 |      444 |
        | 30/06/2021 |          2915.04  |    525 |       1 |      269 |
        | 31/07/2021 |          1560.41  |    406 |       0 |      200 |
        | 31/08/2021 |          2815.52  |    476 |       1 |      211 |
        | 30/09/2021 |          2631.12  |    477 |       2 |      252 |
        | 31/10/2021 |          1425.61  |    372 |       2 |      164 |
        | 30/11/2021 |          1902.5   |    382 |       5 |      184 |
        | 31/12/2021 |          4420.69  |    484 |       3 |      202 |
        | 31/01/2022 |          4656.94  |    478 |       0 |      190 |
        | 28/02/2022 |          2717.86  |    397 |       0 |      160 |
        | 31/03/2022 |           683.159 |    363 |       0 |      163 |
        | 30/04/2022 |          1474.65  |    322 |       0 |      135 |
        | 31/05/2022 |          2081.67  |    504 |       0 |      312 |
        | 30/06/2022 |          3667.7   |    508 |       0 |      255 |
        | 31/07/2022 |          3145.01  |    430 |       0 |      158 |
        | 31/08/2022 |           310.353 |    293 |       0 |      140 |
        | 30/09/2022 |          -450.899 |    240 |       0 |      124 |
        | 31/10/2022 |          1198.57  |    217 |       0 |       83 |
        | 30/11/2022 |          1682.39  |    350 |       0 |      163 |
        | 31/12/2022 |           531.086 |    195 |       0 |       70 |
        | 31/01/2023 |           729.029 |    125 |       0 |       49 |
        ================== SUMMARY METRICS ==================
        | Metric                      | Value               |
        |-----------------------------+---------------------|
        | Backtesting from            | 2021-01-02 06:00:00 |
        | Backtesting to              | 2023-01-15 00:00:00 |
        | Max open trades             | 3                   |
        |                             |                     |
        | Total/Daily Avg Trades      | 15731 / 21.2        |
        | Starting balance            | 1000 USDT           |
        | Final balance               | 61331.634 USDT      |
        | Absolute profit             | 60331.634 USDT      |
        | Total profit %              | 6033.16%            |
        | CAGR %                      | 657.51%             |
        | Profit factor               | 1.31                |
        | Trades per day              | 21.2                |
        | Avg. daily profit %         | 8.13%               |
        | Avg. stake amount           | 319.192 USDT        |
        | Total trade volume          | 5021215.864 USDT    |
        |                             |                     |
        | Long / Short                | 9009 / 6722         |
        | Total profit Long %         | 3328.24%            |
        | Total profit Short %        | 2704.93%            |
        | Absolute profit Long        | 33282.366 USDT      |
        | Absolute profit Short       | 27049.267 USDT      |
        |                             |                     |
        | Best Pair                   | SUSHI/USDT 4120.14% |
        | Worst Pair                  | DGB/USDT -141.99%   |
        | Best trade                  | OCEAN/USDT 17.56%   |
        | Worst trade                 | ONE/USDT -92.24%    |
        | Best day                    | 958.229 USDT        |
        | Worst day                   | -525.578 USDT       |
        | Days win/draw/lose          | 484 / 0 / 260       |
        | Avg. Duration Winners       | 2:02:00             |
        | Avg. Duration Loser         | 3:02:00             |
        | Rejected Entry signals      | 1885820             |
        | Entry/Exit Timeouts         | 0 / 5786            |
        |                             |                     |
        | Min balance                 | 1022.22 USDT        |
        | Max balance                 | 61360.507 USDT      |
        | Max % of account underwater | 17.25%              |
        | Absolute Drawdown (Account) | 6.49%               |
        | Absolute Drawdown           | 1267.67 USDT        |
        | Drawdown high               | 18535.762 USDT      |
        | Drawdown low                | 17268.092 USDT      |
        | Drawdown Start              | 2021-05-19 12:00:00 |
        | Drawdown End                | 2021-05-20 19:00:00 |
        | Market change               | 70.88%              |
        =====================================================


        Best result:

            91/155:   2301 trades. 1664/0/637 Wins/Draws/Losses. Avg profit   1.32%. Median profit   3.65%. Total profit 9676.98562735 USDT ( 967.70%). Avg duration 4:17:00 min. Objective: -16.84544


            # Buy hyperspace params:
            buy_params = {
                "adx_long": 20,
                "adx_short": 18,
                "ce_l_atr_mult": 4.3,
                "ce_l_atr_period": 38,
                "ce_l_rolling_value": 21,
                "ce_s_atr_mult": 3.1,
                "ce_s_atr_period": 23,
                "ce_s_rolling_value": 21,
                "leverage_num": 4,
                "stc_macd_fast": 11,
                "stc_macd_signal": 14,
                "stc_macd_slow": 24,
                "stc_stoch_d": 5,
                "stc_stoch_k": 15,
            }

            # Protection hyperspace params:
            protection_params = {
                "cooldown_lookback": 16,
                "max_allowed_drawdown": 0.17,
                "max_drawdown_lookback": 1,
                "max_drawdown_stop_duration": 11,
                "max_drawdown_trade_limit": 1,
                "stoploss_guard_lookback": 22,
                "stoploss_guard_stop_duration": 12,
                "stoploss_guard_trade_limit": 1,
            }

            # ROI table:
            minimal_roi = {
                "0": 0.149,
                "283": 0.103,
                "969": 0.047,
                "1357": 0
            }

            # Stoploss:
            stoploss = -0.139

            # Trailing stop:
            trailing_stop = True
            trailing_stop_positive = 0.014
            trailing_stop_positive_offset = 0.04
            trailing_only_offset_is_reached = True

    '''

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.149,
        "283": 0.103,
        "969": 0.047,
        "1357": 0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.139

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.014
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    #leverage here
    leverage_optimize = True
    leverage_num = IntParameter(low=1, high=4, default=4, space='buy', optimize=leverage_optimize)

    # Strategy parameters
    adx_long = IntParameter(10, 50, default=20, space="buy", optimize = True)
    adx_short = IntParameter(10, 50, default=18, space="buy", optimize = True)

    # ct_cci_period = IntParameter(5, 75, default=10, space="buy")
    # ct_cci_threshold = IntParameter(1, 200, default=88, space="buy")
    # ct_ema_period = IntParameter(5, 100, default=28, space="buy")


    '''

    '''

    stc_macd_fast = IntParameter(5, 25, default=11, space="buy")
    stc_macd_signal = IntParameter(3, 40, default=14, space="buy")
    stc_macd_slow = IntParameter(10, 50, default=24, space="buy")
    stc_stoch_d = IntParameter(1, 10, default=5, space="buy")
    stc_stoch_k = IntParameter(10, 30, default=15, space="buy")

    ce_l_atr_mult = DecimalParameter(1.0, 6.0, default=4.3, decimals = 1, space="buy")
    ce_l_atr_period = IntParameter(5, 40, default=38, space="buy")
    ce_s_atr_mult = DecimalParameter(1.0, 6.0, default=3.1, decimals = 1, space="buy")
    ce_s_atr_period = IntParameter(5, 40, default=23, space="buy")
    ce_l_rolling_value = IntParameter(5, 40, default=21, space="buy")
    ce_s_rolling_value = IntParameter(5, 40, default=21, space="buy")

    protect_optimize = True
    cooldown_lookback = IntParameter(1, 40, default=16, space="protection", optimize=protect_optimize)
    max_drawdown_lookback = IntParameter(1, 50, default=1, space="protection", optimize=protect_optimize)
    max_drawdown_trade_limit = IntParameter(1, 3, default=1, space="protection", optimize=protect_optimize)
    max_drawdown_stop_duration = IntParameter(1, 50, default=11, space="protection", optimize=protect_optimize)
    max_allowed_drawdown = DecimalParameter(0.05, 0.30, default=0.17, decimals=2, space="protection",
                                            optimize=protect_optimize)
    stoploss_guard_lookback = IntParameter(1, 50, default=22, space="protection", optimize=protect_optimize)
    stoploss_guard_trade_limit = IntParameter(1, 3, default=1, space="protection", optimize=protect_optimize)
    stoploss_guard_stop_duration = IntParameter(1, 50, default=12, space="protection", optimize=protect_optimize)

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": self.cooldown_lookback.value
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": self.max_drawdown_lookback.value,
                "trade_limit": self.max_drawdown_trade_limit.value,
                "stop_duration_candles": self.max_drawdown_stop_duration.value,
                "max_allowed_drawdown": self.max_allowed_drawdown.value
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": self.stoploss_guard_lookback.value,
                "trade_limit": self.stoploss_guard_trade_limit.value,
                "stop_duration_candles": self.stoploss_guard_stop_duration.value,
                "only_per_pair": False
            }
        ]




    # ema_long = IntParameter(50, 250, default=100, space="buy")
    # ema_short = IntParameter(50, 250, default=100, space="buy")

    # sell_rsi = IntParameter(60, 90, default=70, space="sell")

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            'main_plot': {
                'tema': {},
                'sar': {'color': 'white'},
            },
            'subplots': {
                # Subplots - each dict defines one additional plot
                "MACD": {
                    'macd': {'color': 'blue'},
                    'macdsignal': {'color': 'orange'},
                },
                "RSI": {
                    'rsi': {'color': 'red'},
                }
            }
        }

    def informative_pairs(self):

        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        # long_coral_trend = coral_trend(df=dataframe, ema_period = self.ct_ema_period.value, cci_period=self.ct_cci_period.value, cci_threshold = self.ct_cci_threshold.value)

        # dataframe['ct_ema_fast']= long_coral_trend['ema_fast']
        # dataframe['ct_ema_slow'] = long_coral_trend['ema_slow']
        # dataframe['ct_cci'] = long_coral_trend['cci']
        # dataframe['ct_coral_trend'] = long_coral_trend['coral_trend']


        long_chandelier_exit = chandelier_exit(df=dataframe, atr_period = self.ce_l_atr_period.value, atr_mult = self.ce_l_atr_mult.value, rolling_value = self.ce_l_rolling_value.value)

        dataframe['ce_atr'] = long_chandelier_exit['atr']
        dataframe['ce_chandelier_exit_sl'] = long_chandelier_exit['chandelier_exit_sl']
        dataframe['ce_chandelier_exit_tp'] = long_chandelier_exit['chandelier_exit_tp']
        dataframe['ce_enter_long'] = long_chandelier_exit['chandelier_exit_enter_long']

        short_chandelier_exit = chandelier_exit(df=dataframe, atr_period = self.ce_s_atr_period.value, atr_mult = self.ce_s_atr_mult.value, rolling_value = self.ce_s_rolling_value.value)

        dataframe['ce_chandelier_exit_tp_short'] = short_chandelier_exit['chandelier_exit_tp_short']
        dataframe['ce_chandelier_exit_sl_short'] = short_chandelier_exit['chandelier_exit_sl_short']
        dataframe['ce_enter_short'] = long_chandelier_exit['chandelier_exit_enter_short']




        long_stc_indicator = stc_indicator(df=dataframe, macd_fast = self.stc_macd_fast.value, macd_slow = self.stc_macd_slow.value, macd_signal = self.stc_macd_signal.value,stoch_k = self.stc_stoch_k.value, stoch_d = self.stc_stoch_d.value)

        dataframe['stc_macd'] = long_stc_indicator['macd']
        dataframe['stc_macd_signal'] = long_stc_indicator['macd_signal']
        dataframe['stc_stoch_k'] = long_stc_indicator['stoch_k']
        dataframe['stc_stoch_d'] = long_stc_indicator['stoch_d']
        dataframe['stc_stc_trend'] = long_stc_indicator['stc_trend']


        # ADX
        dataframe['adx'] = ta.ADX(dataframe)


        # # EMA - Exponential Moving Average
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['volume_mean_22'] = dataframe['volume'].rolling(22).mean().shift(1)



        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions_long = []
        conditions_short = []
        dataframe.loc[:, 'enter_tag'] = ''

        # buy_1 = (
        #         (dataframe['close'] < dataframe['ema_short']) &
        #         (dataframe['stc_stc_trend'] < 0) &  # Guard: tema is falling
        #         (dataframe['ct_coral_trend'] < 0) &  # Guard: tema is raising
        #         (dataframe['close'] <= dataframe['ce_enter_short']) &
        #         (dataframe['volume'] > dataframe['volume_mean_22'])
        # )

        buy_2 = (
                (dataframe['close'] < dataframe['ema_short']) &
                (dataframe['stc_stc_trend'] < 0) &  # Guard: tema is falling
                # (dataframe['ct_coral_trend'] < 0) &  # Guard: tema is raising
                (dataframe['close'] <= dataframe['ce_enter_short']) &
                (dataframe['volume'] > dataframe['volume_mean_22'])
        )

        buy_2_3 = (
                (dataframe['close'] < dataframe['ema_short']) &
                # (dataframe['stc_stc_trend'] < 0) &  # Guard: tema is falling
                # (dataframe['ct_coral_trend'] < 0) &  # Guard: tema is raising
                (dataframe['close'] <= dataframe['ce_enter_short']) &
                (dataframe['volume'] > dataframe['volume_mean_22'])
        )

        buy_2_4 = (
                # (dataframe['close'] < dataframe['ema_short']) &
                (dataframe['stc_stc_trend'] < 0) &  # Guard: tema is falling
                # (dataframe['ct_coral_trend'] < 0) &  # Guard: tema is raising
                (dataframe['close'] <= dataframe['ce_enter_short']) &
                (dataframe['volume'] > dataframe['volume_mean_22'])
        )

        buy_2_5 = (
                (dataframe['adx'] > self.adx_short.value) &
                (dataframe['stc_stc_trend'] < 0) &  # Guard: tema is falling
                # (dataframe['ct_coral_trend'] < 0) &  # Guard: tema is raising
                (dataframe['close'] <= dataframe['ce_enter_short']) &
                (dataframe['volume'] > dataframe['volume_mean_22'])
        )

        # long below

        # buy_3 = (
        #         (dataframe['close'] > dataframe['ema_long']) &
        #         (dataframe['stc_stc_trend'] > 0) &  # Guard: tema is falling
        #         (dataframe['ct_coral_trend'] > 0) &  # Guard: tema is raising
        #         (dataframe['close'] >= dataframe['ce_enter_long']) &
        #         (dataframe['volume'] > dataframe['volume_mean_22'])
        # )

        buy_4 = (
                (dataframe['close'] > dataframe['ema_long']) &
                (dataframe['stc_stc_trend'] > 0) &  # Guard: tema is falling
                # (dataframe['ct_coral_trend'] > 0) &  # Guard: tema is raising
                (dataframe['close'] >= dataframe['ce_enter_long']) &
                (dataframe['volume'] > dataframe['volume_mean_22'])
        )

        buy_5 = (
                (dataframe['close'] > dataframe['ema_long']) &
                # (dataframe['stc_stc_trend'] > 0) &  # Guard: tema is falling
                # (dataframe['ct_coral_trend'] > 0) &  # Guard: tema is raising
                (dataframe['close'] >= dataframe['ce_enter_long']) &
                (dataframe['volume'] > dataframe['volume_mean_22'])
        )

        buy_6 = (
                # (dataframe['close'] > dataframe['ema_long']) &
                (dataframe['stc_stc_trend'] > 0) &  # Guard: tema is falling
                # (dataframe['ct_coral_trend'] > 0) &  # Guard: tema is raising
                (dataframe['close'] >= dataframe['ce_enter_long']) &
                (dataframe['volume'] > dataframe['volume_mean_22'])
        )

        buy_7 = (
                (dataframe['adx'] > self.adx_long.value) &
                (dataframe['stc_stc_trend'] > 0) &  # Guard: tema is falling
                # (dataframe['ct_coral_trend'] > 0) &  # Guard: tema is raising
                (dataframe['close'] >= dataframe['ce_enter_long']) &
                (dataframe['volume'] > dataframe['volume_mean_22'])
        )

        # conditions_long.append(buy_3)
        # dataframe.loc[buy_3, 'enter_tag'] = 'L_ema_stc_ct'

        conditions_long.append(buy_4)
        dataframe.loc[buy_4, 'enter_tag'] = 'L_ema_stc'

        conditions_long.append(buy_5)
        dataframe.loc[buy_5, 'enter_tag'] = 'L_ema_ct'

        conditions_long.append(buy_6)
        dataframe.loc[buy_6, 'enter_tag'] = 'L_stc_ct'

        conditions_long.append(buy_7)
        dataframe.loc[buy_7, 'enter_tag'] = 'L_adx_stc_ct'


        # short below

        # conditions_short.append(buy_1)
        # dataframe.loc[buy_1, 'enter_tag'] = 'S_ema_stc_ct'

        conditions_short.append(buy_2)
        dataframe.loc[buy_2, 'enter_tag'] = 'S_ema_stc'

        conditions_short.append(buy_2_3)
        dataframe.loc[buy_2_3, 'enter_tag'] = 'S_ema_ct'

        conditions_short.append(buy_2_4)
        dataframe.loc[buy_2_4, 'enter_tag'] = 'S_ct_stc'

        conditions_short.append(buy_2_5)
        dataframe.loc[buy_2_5, 'enter_tag'] = 'S_adx_stc_ct'

        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_long),
                'enter_long'] = 1

        if conditions_short:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_short),
                'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions_long = []
        conditions_short = []
        dataframe.loc[:, 'exit_tag'] = ''

        sell_1 = (
                (dataframe['close'].shift(-1) <= dataframe['ce_chandelier_exit_tp_short']) &
                (dataframe['volume'] > 0)
        )

        sell_2 = (
                (dataframe['close'].shift(-1) >= dataframe['ce_chandelier_exit_sl_short']) &

                (dataframe['volume'] > 0)
        )

        sell_3 = (
                (dataframe['close'].shift(-1) >= dataframe['ce_chandelier_exit_tp']) &

                (dataframe['volume'] > 0)
        )

        sell_4 = (
                (dataframe['close'].shift(-1) <= dataframe['ce_chandelier_exit_sl']) &

                (dataframe['volume'] > 0)
        )

        conditions_short.append(sell_3)
        dataframe.loc[sell_3, 'exit_tag'] += 'ce_tp_short'

        conditions_short.append(sell_4)
        dataframe.loc[sell_4, 'exit_tag'] += 'ce_sl_short'

        conditions_long.append(sell_1)
        dataframe.loc[sell_1, 'exit_tag'] += 'ce_tp_long'

        conditions_long.append(sell_2)
        dataframe.loc[sell_2, 'exit_tag'] += 'ce_sl_long'

        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_long),
                'exit_long'] = 1

        if conditions_short:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_short),
                'exit_short'] = 1
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_num.value