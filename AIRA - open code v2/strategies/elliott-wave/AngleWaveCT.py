# --- Do not remove these libs ---
import math

from freqtrade.strategy.interface import IStrategy

from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
import pandas_ta as pta
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, \
    CategoricalParameter
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_minutes

# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 8,
    "ewof": 8,
    "ewos": 89,
    "ewo_high_mult": 1.3618,
    "ewo_bear_mult": 1.65,
    "ewo_bull_mult": 1.01618,
    "low_offset": 0.95,  # 34,  # 0.942 for BTC and   0.934 for USDT
    "rsi_buy": 61  # with regular ewo 45
}

# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 8,
    "high_offset": 1.01918  # 1.019
}


def EWO(dataframe, ema_length=8, ema2_length=89):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length.value)
    ema2 = ta.EMA(df, timeperiod=ema2_length.value)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


class AngleWaveCT(IStrategy):
    INTERFACE_VERSION = 3

    # ROI table:
    minimal_roi = {
        "0": 0.215,
        "40": 0.132,
        "87": 0.086,
        "360": 0.03
    }

    # Stoploss:
    stoploss = -0.318  # -0.042 # -0.318
    use_custom_stoploss = True

    # SMAOffset
    base_nb_candles_buy = IntParameter(5, 80, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(0.9, 0.99, default=buy_params['low_offset'], space='buy', decimals=2, optimize=True)
    high_offset = DecimalParameter(0.99, 1.1, default=sell_params['high_offset'], decimals=2, space='sell', optimize=True)

    fast_ewo = IntParameter(5, 8, default=buy_params['ewof'], space='buy', optimize=True)
    slow_ewo = IntParameter(50, 100, default=buy_params['ewos'], space='buy', optimize=True)
    ewo_lo_limit = DecimalParameter(-5, 0, default=-2.5, decimals=1, space='buy', optimize=True)

    ewo_bear_x = DecimalParameter(1.62, 1.80, default=buy_params['ewo_bear_mult'], decimals=2, space='buy', optimize=True)
    ewo_high_x = DecimalParameter(1.30, 1.40, default=buy_params['ewo_high_mult'], decimals=2, space='buy', optimize=True)
    ewo_bull_x = DecimalParameter(1.01, 1.3, default=buy_params['ewo_bull_mult'], decimals=3, space='buy', optimize=True)

    rsi_buy_low = IntParameter(30, 50, default=35, space='buy', optimize=True)
    rsi_buy_high = IntParameter(50, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)

    ### trailing stop loss optimiziation ###
    tsl_target5 = DecimalParameter(low=0.2, high=0.4, decimals=1, default=0.3, space='sell', optimize=True, load=True)
    ts5 = DecimalParameter(low=0.04, high=0.06, default=0.05, decimals=2,space='sell', optimize=True, load=True)
    tsl_target4 = DecimalParameter(low=0.15, high=0.2, default=0.2, decimals=2, space='sell', optimize=True, load=True)
    ts4 = DecimalParameter(low=0.03, high=0.05, default=0.045, decimals=2,  space='sell', optimize=True, load=True)
    tsl_target3 = DecimalParameter(low=0.10, high=0.15, default=0.15, decimals=2,  space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, decimals=3,  space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.06, high=0.10, default=0.1, decimals=3, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, decimals=3, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.04, high=0.06, default=0.06, decimals=3, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, decimals=3, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.02, high=0.04, default=0.03, decimals=3, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.008, high=0.015, default=0.013, decimals=3, space='sell', optimize=True, load=True)

    # # Trailing stop:
    # trailing_stop = True
    # trailing_stop_positive = 0.00618
    # trailing_stop_positive_offset = 0.042
    # trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = True

    ## Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'ioc'
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe = '1h'

    process_only_new_candles = True
    startup_candle_count = 89

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 1
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 30,
                "trade_limit": 6,
                "stop_duration_candles": 16,
                "max_allowed_drawdown": 0.1
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 10,
                "trade_limit": 3,
                "stop_duration_candles": 6,
                "required_profit": 0.0,
                "only_per_pair": False,
                "only_per_side": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02
            }
        ]

    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:


        for stop5 in self.tsl_target5.range:
            if (current_profit > stop5):
                for stop5a in self.ts5.range:
                    self.dp.send_msg(f'*** {pair} *** Profit: {current_profit} - lvl5 {stop5}/{stop5a} activated')
                    return stop5a 
        for stop4 in self.tsl_target4.range:
            if (current_profit > stop4):
                for stop4a in self.ts4.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl4 {stop4}/{stop4a} activated')
                    return stop4a 
        for stop3 in self.tsl_target3.range:
            if (current_profit > stop3):
                for stop3a in self.ts3.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl3 {stop3}/{stop3a} activated')
                    return stop3a 
        for stop2 in self.tsl_target2.range:
            if (current_profit > stop2):
                for stop2a in self.ts2.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl2 {stop2}/{stop2a} activated')
                    return stop2a 
        for stop1 in self.tsl_target1.range:
            if (current_profit > stop1):
                for stop1a in self.ts1.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl1 {stop1}/{stop1a} activated')
                    return stop1a 
        for stop0 in self.tsl_target0.range:
            if (current_profit > stop0):
                for stop0a in self.ts0.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl0 {stop0}/{stop0a} activated')
                    return stop0a 

        return self.stoploss

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]

        return informative_pairs

    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.informative_timeframe)

        return dataframe

    def do_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # EMAs for Bear/Bull
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.fast_ewo.value)
        dataframe['ema2'] = ta.EMA(dataframe, timeperiod=self.slow_ewo.value)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)  # 14)
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'], timeperiod=8)  # 5)

        # absolute ewo
        dataframe["ABSEWO"] = dataframe["EWO"]
        dataframe.loc[dataframe["EWO"] < 0, "ABSEWO"] = dataframe["EWO"] * - 1
        # seperate pos and neg ewo
        dataframe.loc[dataframe["EWO"] > 0, "POSEWO"] = dataframe["EWO"]
        dataframe.loc[dataframe["EWO"] < 0, "NEGEWO"] = dataframe["EWO"]
        dataframe["POSEWO"].fillna(method="ffill", inplace=True)
        dataframe["NEGEWO"].fillna(method="ffill", inplace=True) 
        

        # angle of the ema
        candlesBack = 3
        backQuote = dataframe['ema'].shift(candlesBack)
        deltaX = candlesBack
        deltaY = dataframe['ema'] - backQuote
        dataframe['angleRad'] = (deltaY / deltaX).apply(lambda x: math.atan(x))
        dataframe['angle_perc'] = (dataframe['angleRad'] / dataframe['close']) * 100
        threshold = 0  # .0000001
        dataframe["ema_angle_up"] = dataframe['angleRad'] > 0 + threshold
        dataframe["ema_angle_down"] = dataframe['angleRad'] < 0 - threshold

        # ewo normalized
        dataframe['min'] = dataframe["EWO"].rolling(89).min()
        dataframe['max'] = dataframe["EWO"].rolling(89).max()
        dataframe['dif'] = dataframe['max'].sub(dataframe['min'])
        dataframe["ewoNorm"] = dataframe["EWO"] / dataframe['dif']
        dataframe['ewoNormMa'] =  ta.SMA(dataframe["ewoNorm"], timeperiod=self.fast_ewo.value)
        dataframe['ewoNormMaHigh'] = 0.0618

        # angle of the ewoNorm
        candlesBack2 = 1
        backQuote2 = dataframe['ewoNorm'].shift(candlesBack2)
        deltaX2 = candlesBack2
        deltaY2 = dataframe['ewoNorm'] - backQuote2
        dataframe['ewoAngleRad'] = (deltaY2 / deltaX2).apply(lambda x: math.atan(x))
        dataframe["ewo_angle_up"] = dataframe['ewoAngleRad'] > -0.0001
        dataframe["ewo_angle_down"] = dataframe['ewoAngleRad'] < 0.0001

        emaPercent = dataframe['ema'] / dataframe['ema2']

        avg_positive_ewo = dataframe["POSEWO"].rolling(self.base_nb_candles_buy.value).mean()
        avg_negative_ewo = dataframe["NEGEWO"].rolling(self.base_nb_candles_buy.value).mean()
        avg_abs_ewo = dataframe["ABSEWO"].rolling(self.base_nb_candles_buy.value).mean()

        # scale the buys. if we bear then push back the buys if we bull make more buys
        dataframe["ewo_low_mult"] = self.ewo_bull_x.value  # / emaPercent # 0.618
        dataframe.loc[dataframe["ema"] < dataframe["ema2"], "ewo_low_mult"] = self.ewo_bear_x.value

        # for buy below rolling ewo low
        ewo_low = avg_negative_ewo * dataframe["ewo_low_mult"] 
        dataframe['ewo_low'] = ewo_low
        # for rsi buy
        ewo_high = avg_positive_ewo * self.ewo_high_x.value
        dataframe['ewo_high'] = ewo_high

        dataframe['ewo_limit'] = self.ewo_lo_limit.value

        dataframe['atr_pcnt'] = (qtpylib.atr(dataframe) / dataframe['close'])

        dataframe['ma_lo'] = (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (self.low_offset.value - (dataframe['atr_pcnt']/3)))
        dataframe['ma_hi'] = (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (self.high_offset.value + dataframe['atr_pcnt']))


        return dataframe

    def do_informative_indicators(self, inf: DataFrame, metadata: dict) -> DataFrame:
        # informative = dataframe = self.dp.get_pair_dataframe(
        # pair=metadata['pair'], timeframe=self.informative_timeframe)

        inf['inf_ema'] = ta.SMA(inf["close"], timeperiod=self.fast_ewo.value)
        inf['inf_ema2'] = ta.SMA(inf["close"], timeperiod=self.slow_ewo.value)
        # print(inf['inf_ema'])
        return inf

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # if self.config['runmode'].value in ('backtest', 'hyperopt'):
        #    assert (timeframe_to_minutes(self.timeframe) <= 5), "Backtest this strategy in 5m or 1m timeframe."

        if self.timeframe == self.informative_timeframe:
            dataframe = self.do_indicators(dataframe, metadata)
        else:
            if not self.dp:
                return dataframe

            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

            # todo this dont work why when i run informative indicators does this not work...
            informative = self.do_informative_indicators(informative.copy(), metadata)
            # informative = self.do_indicators(informative.copy(), metadata)
            dataframe = self.do_indicators(dataframe.copy(), metadata)

            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe,
                                               ffill=True)

            skip_columns = [(s + "_" + self.informative_timeframe) for s in
                            ['date', 'open', 'high', 'low', 'close', 'volume']]
            dataframe.rename(columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "") if (
                not s in skip_columns) else s, inplace=True)

        return dataframe

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:


        df.loc[
            (
                (df['EWO'] > df['ewo_high']) &
                # (df['close'] < (df[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (df['close'] < df['ma_lo']) &
                # (df["ewo_angle_up"]) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO > HIGH')

        df.loc[
            (
                (df['EWO'] < df['ewo_high']) &
                (df['close'] < df['ma_lo']) &
                (df["ewo_angle_up"]) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO < HIGH')

        df.loc[
            (
                (qtpylib.crossed_above(df['EWO'], df['ewo_high'])) &
                (df['close'] < df['ma_lo']) &
                (df["ewo_angle_up"]) &
                (df['rsi'] < self.rsi_buy_high.value) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO XO HIGH')

        df.loc[
            (
                (qtpylib.crossed_above(df['EWO'], df['ewo_low'])) &
                (df['close'] < df['ma_lo']) &
                (df['EWO'] < self.ewo_lo_limit.value) &
                (df['rsi'] < self.rsi_buy_low.value) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO XO LOW')

        df.loc[
            (
                (df['EWO'] < df['ewo_low']) &
                (df['close'] < df['ma_lo']) &
                (df['EWO'] < self.ewo_lo_limit.value) &
                (df['rsi'] < self.rsi_buy_low.value) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO < LOW')

        # df.loc[
        #     (
        #         ((df['angle_perc'] - df['angle_perc'].shift(1)) > 0.07 ) &
        #         ((df['angle_perc'].shift(1) - df['angle_perc'].shift(2)) < 0 ) &
        #         (df['angle_perc'] < 0.17) &
        #         (df['angle_perc'] > 0) &
        #         (df['close'] > (df[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'Angle Change Pos')

        df.loc[
            (
                ((df['angle_perc'] - df['angle_perc'].shift(1)) > 0.07 ) &
                ((df['angle_perc'].shift(1) - df['angle_perc'].shift(2)) < 0 ) &
                (df['angle_perc'] < 0) &
                (df['open'] < (df[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'Angle Change Neg')

        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['close'] > df['ma_hi']) &
                (df['EWO'] > df['ewo_high']) &
                # (df['angle_perc'] < 0.10) &
                (df['EWO'] > 0) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'EWO High')

        df.loc[
            (
                (df['close'] > df['ma_hi']) &
                (df['rsi'] >= 89) &
                (df['EWO'] > 0) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'EWO rsi limit')

        df.loc[
            (
                (df['close'] > df['ma_hi']) &
                (df['rsi'] >= 89) &
                (df['EWO'] < 0) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'EWO low rsi limit')


        return df

      # "0": 0.202,
      # "141": 0.094,
      # "380": 0.017,
      # "1459": 0

# ============================================================ BACKTESTING REPORT ============================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
# |  ORAI/USDT |       383 |           1.00 |         383.32 |        519743.915 |       51974.39 |        3:08:00 |   295     0    88  77.0 |
# |   CTI/USDT |       426 |           0.98 |         415.54 |        518202.754 |       51820.28 |        4:10:00 |   319     1   106  74.9 |
# |   EWT/USDT |       380 |           1.31 |         498.67 |        271591.642 |       27159.16 |        3:40:00 |   325     0    55  85.5 |
# |  EOSC/USDT |      1003 |           1.34 |        1348.19 |        268456.489 |       26845.65 |        3:44:00 |   720     8   275  71.8 |
# |  OPUL/USDT |       133 |           0.72 |          95.39 |        234658.280 |       23465.83 |        4:49:00 |   106     0    27  79.7 |
# | YFDAI/USDT |       178 |           1.06 |         189.22 |        234636.196 |       23463.62 |        3:59:00 |   141     0    37  79.2 |
# |  RNDR/USDT |       269 |           0.53 |         142.58 |        228253.623 |       22825.36 |        4:44:00 |   210     0    59  78.1 |
# |   QNT/USDT |       251 |           1.05 |         264.73 |        214297.194 |       21429.72 |        3:45:00 |   207     1    43  82.5 |
# | JASMY/USDT |        95 |           0.49 |          46.49 |        151439.029 |       15143.90 |        3:54:00 |    76     0    19  80.0 |
# | GALAX/USDT |       111 |           0.53 |          58.78 |        143250.873 |       14325.09 |        4:55:00 |    90     0    21  81.1 |
# |  AGIX/USDT |       127 |           0.55 |          69.52 |        134443.919 |       13444.39 |        5:26:00 |    96     0    31  75.6 |
# |  ROSE/USDT |       295 |           0.40 |         118.29 |        104638.442 |       10463.84 |        4:37:00 |   223     0    72  75.6 |
# |  COMP/USDT |       248 |           0.61 |         151.96 |         62535.598 |        6253.56 |        5:48:00 |   179     0    69  72.2 |
# |  AVAX/USDT |       188 |           0.62 |         117.26 |         57317.005 |        5731.70 |        4:48:00 |   145     0    43  77.1 |
# |  DYDX/USDT |        93 |           0.18 |          16.67 |         48686.734 |        4868.67 |        5:01:00 |    70     0    23  75.3 |
# |  DOGE/USDT |       105 |           0.43 |          45.23 |         42279.791 |        4227.98 |        3:32:00 |    74     0    31  70.5 |
# |   GRT/USDT |       246 |           0.42 |         104.23 |         39110.810 |        3911.08 |        4:38:00 |   186     0    60  75.6 |
# |   UNI/USDT |       168 |          -0.10 |         -17.03 |         33827.663 |        3382.77 |        5:46:00 |   124     0    44  73.8 |
# |  IOTA/USDT |         7 |           1.55 |          10.86 |         32847.307 |        3284.73 |        2:57:00 |     6     0     1  85.7 |
# |   VET/USDT |       441 |           0.92 |         406.32 |         28803.019 |        2880.30 |        6:07:00 |   344     0    97  78.0 |
# |   ZEC/USDT |       229 |           0.84 |         191.68 |         23252.881 |        2325.29 |        6:48:00 |   173     1    55  75.5 |
# |   INJ/USDT |        25 |           0.29 |           7.33 |         22088.809 |        2208.88 |        5:26:00 |    17     0     8  68.0 |
# | OCEAN/USDT |         4 |           1.33 |           5.30 |         16089.918 |        1608.99 |        5:36:00 |     4     0     0   100 |
# |   ENJ/USDT |       116 |           0.22 |          25.00 |         15146.552 |        1514.66 |        5:42:00 |    78     0    38  67.2 |
# |  ALGO/USDT |       520 |           0.26 |         137.67 |         11437.256 |        1143.73 |        6:29:00 |   374     1   145  71.9 |
# |   RLY/USDT |        90 |           0.37 |          33.65 |         10712.860 |        1071.29 |        5:49:00 |    63     0    27  70.0 |
# |   BAT/USDT |        66 |          -0.11 |          -7.07 |          7533.069 |         753.31 |        5:58:00 |    47     0    19  71.2 |
# | SUSHI/USDT |       149 |           0.06 |           8.21 |          6935.974 |         693.60 |        6:20:00 |    99     0    50  66.4 |
# |   XDC/USDT |       100 |           0.36 |          35.84 |          4308.025 |         430.80 |        8:49:00 |    67     1    32  67.0 |
# |   FIL/USDT |       100 |          -0.45 |         -45.41 |          3294.538 |         329.45 |        6:33:00 |    64     0    36  64.0 |
# |   FTM/USDT |       171 |          -0.31 |         -52.17 |         -4205.788 |        -420.58 |        5:37:00 |   118     0    53  69.0 |
# |  KAVA/USDT |        13 |          -0.23 |          -2.95 |        -13975.465 |       -1397.55 |        7:03:00 |     9     0     4  69.2 |
# |   XLM/USDT |       259 |           0.25 |          63.71 |        -15020.157 |       -1502.02 |        7:40:00 |   186     2    71  71.8 |
# |   XTZ/USDT |       354 |           0.51 |         181.46 |        -19408.109 |       -1940.81 |        6:04:00 |   269     1    84  76.0 |
# |   DOT/USDT |       155 |           0.34 |          52.52 |        -21405.279 |       -2140.53 |        5:22:00 |   125     0    30  80.6 |
# |  KLAY/USDT |         2 |          -3.85 |          -7.69 |        -22372.845 |       -2237.28 |       15:42:00 |     0     0     2     0 |
# |   SOL/USDT |        50 |          -0.19 |          -9.48 |        -27906.484 |       -2790.65 |        6:07:00 |    34     0    16  68.0 |
# |   BTC/USDT |       203 |           0.24 |          48.51 |        -28814.674 |       -2881.47 |       12:30:00 |   141    11    51  69.5 |
# |   EOS/USDT |       199 |          -0.07 |         -13.47 |        -32819.270 |       -3281.93 |        8:45:00 |   147     1    51  73.9 |
# |  ANKR/USDT |        72 |           0.44 |          31.36 |        -34286.282 |       -3428.63 |        5:27:00 |    57     0    15  79.2 |
# |   TRX/USDT |       245 |           0.46 |         112.00 |        -34358.719 |       -3435.87 |        8:22:00 |   169     2    74  69.0 |
# |   ETC/USDT |       289 |           0.11 |          31.07 |        -35982.250 |       -3598.22 |        7:05:00 |   209     3    77  72.3 |
# |   IMX/USDT |        41 |          -0.21 |          -8.78 |        -37435.415 |       -3743.54 |        6:07:00 |    30     0    11  73.2 |
# | THETA/USDT |       134 |           0.43 |          57.78 |        -44210.763 |       -4421.08 |        5:33:00 |    98     0    36  73.1 |
# |  LINK/USDT |       149 |          -0.17 |         -25.23 |        -44337.942 |       -4433.79 |        6:50:00 |   103     0    46  69.1 |
# |  HBAR/USDT |        40 |          -0.37 |         -14.96 |        -50657.739 |       -5065.77 |        7:56:00 |    27     0    13  67.5 |
# |   TRU/USDT |        20 |          -1.13 |         -22.54 |        -80748.034 |       -8074.80 |        6:40:00 |    11     0     9  55.0 |
# |  AGLD/USDT |        44 |          -0.58 |         -25.51 |        -89057.213 |       -8905.72 |        7:43:00 |    29     0    15  65.9 |
# |  EGLD/USDT |        29 |          -0.84 |         -24.34 |        -89991.975 |       -8999.20 |        8:14:00 |    16     0    13  55.2 |
# |   ADA/USDT |       238 |          -0.23 |         -54.71 |        -91312.660 |       -9131.27 |        7:32:00 |   162     0    76  68.1 |
# | MATIC/USDT |        91 |          -0.78 |         -70.84 |        -93385.185 |       -9338.52 |        6:42:00 |    61     0    30  67.0 |
# |  NEAR/USDT |       107 |          -0.23 |         -24.27 |        -97835.185 |       -9783.52 |        7:33:00 |    73     0    34  68.2 |
# |   YFI/USDT |       196 |           0.19 |          37.86 |       -145312.915 |      -14531.29 |        5:43:00 |   138     0    58  70.4 |
# |  ATOM/USDT |       597 |           0.09 |          51.52 |       -172430.204 |      -17243.02 |        6:50:00 |   424     0   173  71.0 |
# |   XRP/USDT |       443 |          -0.20 |         -87.01 |       -222127.681 |      -22212.77 |        9:22:00 |   301     5   137  67.9 |
# |  VELO/USDT |       342 |          -0.45 |        -155.22 |       -262351.608 |      -26235.16 |        6:34:00 |   232     2   108  67.8 |
# |   ETH/USDT |       311 |          -0.20 |         -63.46 |       -320341.045 |      -32034.10 |       11:08:00 |   206     5   100  66.2 |
# |      TOTAL |     11340 |           0.43 |        4863.54 |       1357729.279 |      135772.93 |        6:01:00 |  8297    45  2998  73.2 |
# ============================================================= ENTER TAG STATS ==============================================================
# |        TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
# |  EWO < LOW |     10364 |           0.40 |        4149.99 |        938103.124 |       93810.31 |        6:15:00 |  7572    44  2748  73.1 |
# | EWO > HIGH |       966 |           0.73 |         707.13 |        418948.553 |       41894.86 |        3:29:00 |   717     1   248  74.2 |
# | EWO XO LOW |        10 |           0.64 |           6.42 |           677.602 |          67.76 |        4:42:00 |     8     0     2  80.0 |
# |      TOTAL |     11340 |           0.43 |        4863.54 |       1357729.279 |      135772.93 |        6:01:00 |  8297    45  2998  73.2 |
# ======================================================= EXIT REASON STATS ========================================================
# |        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# | trailing_stop_loss |    6011 |   5441     0   570  90.5 |           1.51 |        9103.56 |       7.36194e+06 |        1517.26 |
# |           EWO High |    4425 |   2087     0  2338  47.2 |          -1.22 |       -5409.28 |      -6.85188e+06 |        -901.55 |
# |                roi |     753 |    708    45     0   100 |           2.44 |        1835.52 |       1.16115e+06 |         305.92 |
# |      EWO rsi limit |      87 |     37     0    50  42.5 |          -1.53 |        -133.1  |  -33764.1         |         -22.18 |
# |  EWO low rsi limit |      46 |     24     0    22  52.2 |          -0.54 |         -24.82 |   -3339.25        |          -4.14 |
# |          stop_loss |      18 |      0     0    18     0 |         -28.24 |        -508.33 | -276373           |         -84.72 |
# ======================================================= LEFT OPEN TRADES REPORT ========================================================
# |   Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
# |--------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
# |  TOTAL |         0 |           0.00 |           0.00 |             0.000 |           0.00 |           0:00 |     0     0     0     0 |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2019-01-01 00:00:00 |
# | Backtesting to              | 2023-04-30 00:00:00 |
# | Max open trades             | 6                   |
# |                             |                     |
# | Total/Daily Avg Trades      | 11340 / 7.18        |
# | Starting balance            | 1000 USDT           |
# | Final balance               | 1358729.279 USDT    |
# | Absolute profit             | 1357729.279 USDT    |
# | Total profit %              | 135772.93%          |
# | CAGR %                      | 429.41%             |
# | Profit factor               | 1.11                |
# | Trades per day              | 7.18                |
# | Avg. daily profit %         | 85.93%              |
# | Avg. stake amount           | 94657.76 USDT       |
# | Total trade volume          | 1073419000.083 USDT |
# |                             |                     |
# | Best Pair                   | EOSC/USDT 1348.19%  |
# | Worst Pair                  | VELO/USDT -155.22%  |
# | Best trade                  | TRX/USDT 182.12%    |
# | Worst trade                 | VELO/USDT -28.24%   |
# | Best day                    | 222555.567 USDT     |
# | Worst day                   | -709320.388 USDT    |
# | Days win/draw/lose          | 886 / 338 / 352     |
# | Avg. Duration Winners       | 3:27:00             |
# | Avg. Duration Loser         | 12:45:00            |
# | Rejected Entry signals      | 1579329             |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 968.649 USDT        |
# | Max balance                 | 2634960.342 USDT    |
# | Max % of account underwater | 68.52%              |
# | Absolute Drawdown (Account) | 49.81%              |
# | Absolute Drawdown           | 1312533.509 USDT    |
# | Drawdown high               | 2633960.342 USDT    |
# | Drawdown low                | 1321426.832 USDT    |
# | Drawdown Start              | 2021-12-03 10:20:00 |
# | Drawdown End                | 2022-11-10 03:25:00 |
# | Market change               | 149.10%             |
# =====================================================

        # "pair_whitelist": [


        #     "ATOM/USDT",
        #     "XRP/USDT",
        #     "QNT/USDT",
        #     "AKT/USDT",
        #     "IOTA/USDT",
        #     "ADA/USDT",
        #     "OSMO/USDT",
        #     "BTC/USDT",
        #     "ETH/USDT",
        #     "CSPR/USDT",
        #     "ETC/USDT",
        #     "XDC/USDT",
        #     "LINK/USDT",
        #     "MATIC/USDT",
        #     "AVAX/USDT",
        #     "HBAR/USDT",
        #     "SCRT/USDT",
        #     "KAVA/USDT",
        #     "INJ/USDT",
        #     "NEAR/USDT",
        #     "DOT/USDT",
        #     "ALGO/USDT",
        #     "XLM/USDT",
        #     "RNDR/USDT",
        #     "ORAI/USDT",
        #     "AGIX/USDT",
        #     "GRT/USDT",
        #     "DOGE/USDT",
        #     "SOL/USDT",
        #     "TRX/USDT",
        #     "UNI/USDT",
        #     "LDO/USDT",
        #     "FIL/USDT",
        #     "VET/USDT",
        #     "EGLD/USDT",
        #     "THETA/USDT",
        #     "XTZ/USDT",
        #     "IMX/USDT",
        #     "ZEC/USDT",
        #     "GMX/USDT",
        #     "KLAY/USDT",
        #     "OP/USDT",
        #     "ROSE/USDT",
        #     "ENJ/USDT",
        #     "DYDX/USDT",
        #     "OCEAN/USDT",
        #     "FLR/USDT",
        #     "EWT/USDT",
        #     "JASMY/USDT",
        #     "APT/USDT",
        #     "APE/USDT",
        #     "EOS/USDT",
        #     "LUNC/USDT",
        #     "ANKR/USDT",
        #     "GALAX/USDT",
        #     "OPUL/USDT",
        #     "FET/USDT",
        #     "FTM/USDT",
        #     "EOSC/USDT",
        #     "VELO/USDT",
        #     "YFDAI/USDT",
        #     "BAT/USDT",
        #     "TRU/USDT",
        #     "YFI/USDT",
        #     "COMP/USDT",
        #     "AGLD/USDT",
        #     "SUSHI/USDT",
        #     "RLY/USDT",
        #     "CTI/USDT",

