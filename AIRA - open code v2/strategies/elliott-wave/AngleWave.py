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


class AngleWave(IStrategy):
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

        dataframe['ma_lo'] = (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)
        dataframe['ma_hi'] = (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.high_offset.value)


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
                (df['EWO'] < df['ewo_high']) &
                (df['close'] < (df[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (df["ewo_angle_up"]) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO > HIGH')

        df.loc[
            (
                (qtpylib.crossed_above(df['EWO'], df['ewo_high'])) &
                (df['close'] < (df[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (df["ewo_angle_up"]) &
                (df['rsi'] < self.rsi_buy_high.value) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO XO HIGH')

        df.loc[
            (
                (qtpylib.crossed_above(df['EWO'], df['ewo_low'])) &
                (df['close'] < (df[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (df['EWO'] < self.ewo_lo_limit.value) &
                (df['rsi'] < self.rsi_buy_low.value) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO XO LOW')

        df.loc[
            (
                (df['EWO'] < df['ewo_low']) &
                (df['close'] < (df[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (df['EWO'] < self.ewo_lo_limit.value) &
                (df['rsi'] < self.rsi_buy_low.value) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO < LOW')

        # df.loc[
        #     (
        #         (df['EWO'] < df['ewo_low']) &
        #         (df['EWO'] < self.ewo_lo_limit.value) &
        #         (df['close'] < (df[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
        #         (qtpylib.crossed_above(df['ewoNorm'], df['ewo_low'])) &
        #         (df['rsi'] < self.rsi_buy_low.value) &
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'ewoNorm XO Low')

        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['close'] > (df[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (df['EWO'] > df['ewo_high']) &
                (df['EWO'] > 0) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'EWO High')

        df.loc[
            (
                (df['close'] > (df[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (df['rsi'] >= 89) &
                (df['EWO'] > 0) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'EWO rsi limit')

        df.loc[
            (
                (df['close'] > (df[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (df['rsi'] >= 89) &
                (df['EWO'] < 0) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'EWO low rsi limit')


        return df
