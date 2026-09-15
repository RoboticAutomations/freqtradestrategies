
import logging
import numpy as np
import pandas as pd
from technical import qtpylib
from pandas import DataFrame
from datetime import datetime, timezone
from typing import Optional
from functools import reduce
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from technical import qtpylib, pivots_points

class CTIBS2(IStrategy):
    stoploss = -0.28
    timeframe ='15m'
    use_custom_stoploss = True
    trailing_stop = True
    ignore_roi_if_entry_signal = True
    use_exit_signal = True
    minimal_roi = {
        "0":  0.10
    }
    # DCA settings
    exit_profit_only = True
    position_adjustment_enable = True
    max_entry_position_adjustment = 0
    max_dca_multiplier = 1

    ## Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'ioc'
    }


    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    ### ------------------ HYPER-OPT PARAMETERS ----------------------- ###

    ### protections ####
    # CooldownPeriod 
    cooldown_lookback = IntParameter(0, 48, default=5, space="protection", optimize=True)
    
    # StoplossGuard    
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    stop_protection_only_per_pair = BooleanParameter(default=False, space="protection", optimize=True)
    stop_protection_only_per_side = BooleanParameter(default=False, space="protection", optimize=True)
    stop_protection_trade_limit = IntParameter(1, 10, default=4, space="protection", optimize=True)
    stop_protection_required_profit = DecimalParameter(-1.0, 3.0, default=0.0, space="protection", optimize=True)

    # LowProfitPairs    
    use_lowprofit_protection = BooleanParameter(default=True, space="protection", optimize=True)
    lowprofit_protection_lookback = IntParameter(1, 10, default=6, space="protection", optimize=True)
    lowprofit_trade_limit = IntParameter(1, 10, default=4, space="protection", optimize=True)
    lowprofit_stop_duration = IntParameter(1, 100, default=60, space="protection", optimize=True)
    lowprofit_required_profit = DecimalParameter(-1.0, 3.0, default=0.0, space="protection", optimize=True)
    lowprofit_only_per_pair = BooleanParameter(default=False, space="protection", optimize=True)

    # MaxDrawdown    
    use_maxdrawdown_protection = BooleanParameter(default=True, space="protection", optimize=True)
    maxdrawdown_protection_lookback = IntParameter(1, 10, default=6, space="protection", optimize=True)
    maxdrawdown_trade_limit = IntParameter(1, 20, default=10, space="protection", optimize=True)
    maxdrawdown_stop_duration = IntParameter(1, 100, default=6, space="protection", optimize=True)
    maxdrawdown_allowed_drawdown = DecimalParameter(0.01, 0.10, default=0.0, decimals=2, space="protection", optimize=True)

  
    ### DCA ###
    # max_epa = CategoricalParameter([0, 1, 2, 3], default=0, space="buy", optimize=True)

  
    ### trailing stop loss optimiziation ###
    tsl_target5 = DecimalParameter(low=0.2, high=0.4, decimals=1, default=0.3, space='sell', optimize=True, load=True)
    ts5 = DecimalParameter(low=0.04, high=0.06, default=0.05, decimals=2,space='sell', optimize=True, load=True)
    tsl_target4 = DecimalParameter(low=0.15, high=0.2, default=0.2, decimals=2, space='sell', optimize=True, load=True)
    ts4 = DecimalParameter(low=0.03, high=0.05, default=0.045, decimals=2,  space='sell', optimize=True, load=True)
    tsl_target3 = DecimalParameter(low=0.10, high=0.15, default=0.15, decimals=2,  space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, decimals=3,  space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.08, high=0.10, default=0.1, decimals=3, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, decimals=3, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.05, high=0.08, default=0.06, decimals=3, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, decimals=3, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.02, high=0.045, default=0.03, decimals=3, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.008, high=0.015, default=0.013, decimals=3, space='sell', optimize=True, load=True)


    ### indicators ###
    #buy
    reference_ma_length = IntParameter(185, 195, default=200, space="buy" ,optimize=False)
    smoothing_length = IntParameter(40, 60, default=48, space="buy")#IntParameter(15, 100, default=30, space="buy", optimize=True)
    buy_offset1 = DecimalParameter(low=0.98, high=0.99, decimals=2, default=0.99, space='buy', optimize=False, load=True)
    buy_offset2 = DecimalParameter(low=0.90, high=0.95, decimals=2, default=0.94, space='buy', optimize=False, load=True)
    buy_change = DecimalParameter(1.2, 2.0, default=1.5, decimals=1, space="buy", optimize=False)
    sell_change = DecimalParameter(1.2, 2.0, default=1.5, decimals=1, space="sell", optimize=True)
    max_length = CategoricalParameter([192, 240, 288, 336], default=240, space="buy", optimize=False)

    # selling 
    filterlength = IntParameter(low=30, high=40, default=37, space='sell', optimize=True)
    sell_offset1 = DecimalParameter(low=1.01, high=1.08, decimals=2, default=1.05, space='sell', optimize=True, load=True)
    sell_change = DecimalParameter(1.2, 2.0, default=1.5, decimals=1, space="sell", optimize=True)

    ### buying values ###
    buy_change = DecimalParameter(1.5, 3.5, default=2.0, decimals=1, space="buy", optimize=False)
    rsi01 = IntParameter(20, 40, default=32, space="buy", optimize=True)
    rsi02 = IntParameter(30, 60, default=44, space="buy", optimize=True)
    rsi03 = IntParameter(20, 35, default=29, space="buy", optimize=True)
    rsi04 = IntParameter(45, 65, default=46, space="buy", optimize=True)
    rsi05 = IntParameter(45, 65, default=49, space="buy", optimize=True)
    rsi06 = IntParameter(45, 65, default=53, space="buy", optimize=True)
    rsi07 = IntParameter(50, 70, default=70, space="buy", optimize=True)
    rsi08 = IntParameter(40, 60, default=48, space="buy", optimize=True)
    rsi09 = IntParameter(55, 70, default=60, space="buy", optimize=True)
    rsi10 = IntParameter(20, 35, default=26, space="buy", optimize=True)
    rsi11 = IntParameter(20, 35, default=30, space="buy", optimize=True)
    rsi12 = IntParameter(35, 55, default=38, space="buy", optimize=True)
    rsi13 = IntParameter(50, 70, default=56, space="buy", optimize=True)
    rsi14 = IntParameter(55, 70, default=67, space="buy", optimize=True)
    rsi15 = IntParameter(25, 40, default=26, space="buy", optimize=True)
    rsi16 = IntParameter(30, 50, default=39, space="buy", optimize=True)
    rsi17 = IntParameter(40, 55, default=47, space="buy", optimize=True)
    rsi18 = IntParameter(20, 35, default=22, space="buy", optimize=True)
    rsi19 = IntParameter(20, 35, default=21, space="buy", optimize=True)
    rsi20 = IntParameter(20, 70, default=41, space="buy", optimize=True)
    rsi21 = IntParameter(20, 40, default=28, space="buy", optimize=True)
    rsi22 = IntParameter(40, 60, default=54, space="buy", optimize=True)
    rsi23 = IntParameter(40, 60, default=53, space="buy", optimize=True)
    rsi24 = IntParameter(20, 35, default=31, space="buy", optimize=True)
    rsi25 = IntParameter(20, 35, default=24, space="buy", optimize=True)
    rsi26 = IntParameter(20, 35, default=21, space="buy", optimize=True)
    rsi27 = IntParameter(20, 70, default=61, space="buy", optimize=True)
    rsi28 = IntParameter(20, 35, default=26, space="buy", optimize=True)
    rsi29 = IntParameter(50, 70, default=67, space="buy", optimize=True)
    rsi30 = IntParameter(20, 35, default=22, space="buy", optimize=True)

    ### selling values ###
    sell_slope = DecimalParameter(-0.05, 0.10, default=-0.02, decimals=2, space="sell", optimize=True)

    ### I.B.S. ###
    # Reference Ma - Long Term Direction
    ref_bull = DecimalParameter(0.05, 0.125, default=0.088, decimals=3, space="buy", optimize=False)
    ref_up = DecimalParameter(0.01, 0.05, default=0.027, decimals=3, space="buy", optimize=False)
    ref_down = DecimalParameter(-0.05, -0.01, default=-0.048, decimals=3, space="buy", optimize=False)
    ref_bear = DecimalParameter(-0.125, -0.05, default=-0.1, decimals=3, space="buy", optimize=False)

    # Smooth Change Ma - Short Term Direction
    smooth_bull = DecimalParameter(0.05, 0.125, default=0.085, decimals=3, space="buy", optimize=False)
    smooth_up = DecimalParameter(0.01, 0.05, default=0.035, decimals=3, space="buy", optimize=False)
    smooth_down = DecimalParameter(-0.05, -0.01, default=-0.042, decimals=3, space="buy", optimize=False)
    smooth_bear = DecimalParameter(-0.125, -0.05, default=-0.066, decimals=3, space="buy", optimize=False)

    # Distance from Long Term High
    from_bull = DecimalParameter(-5, -1, default=-4.2, decimals=1, space="buy", optimize=False)
    from_up = DecimalParameter(-10, -2, default=-5.3, decimals=1, space="buy", optimize=False)
    from_ranging = DecimalParameter(-10, -2, default=-6.9, decimals=1, space="buy", optimize=False)
    from_down = DecimalParameter(-10, -5.0, default=-8.9, decimals=1, space="buy", optimize=False)
    from_bear = DecimalParameter(-15, -10, default=-13.4, decimals=1, space="buy", optimize=False)

    # Selecting what works
    buy01 = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
    buy02 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    buy03 = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
    buy04 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    buy05 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    buy06 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    buy07 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    buy08 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    buy09 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    buy10 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    buy11 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    buy12 = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
    buy13 = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
    buy14 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    buy15 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    buy16 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    buy17 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    buy18 = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
    buy19 = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
    buy20 = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
    buy21 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    buy22 = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
    buy23 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    buy24 = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
    buy25 = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
    buy26 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    buy27 = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
    buy28 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    buy29 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    buy30 = CategoricalParameter([True, False], default=True, space="buy", optimize=False)

    sell01 = CategoricalParameter([True, False], default=True, space="sell")
    sell02 = CategoricalParameter([True, False], default=True, space="sell")
    sell03 = CategoricalParameter([True, False], default=True, space="sell")
    sell04 = CategoricalParameter([True, False], default=True, space="sell")
    sell05 = CategoricalParameter([True, False], default=True, space="sell")
    sell06 = CategoricalParameter([True, False], default=True, space="sell")
    sell07 = CategoricalParameter([True, False], default=True, space="sell")
    sell08 = CategoricalParameter([True, False], default=True, space="sell")
    sell09 = CategoricalParameter([True, False], default=True, space="sell")
    sell10 = CategoricalParameter([True, False], default=True, space="sell")



    #----------------------- END OF HYPER-OPT PARAMETERS -------------------------#

    @property
    def protections(self):
    
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,
                "trade_limit": self.stop_protection_trade_limit.value,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": self.stop_protection_only_per_pair.value,
                "required_profit": self.stop_protection_required_profit.value,
            "only_per_side": self.stop_protection_only_per_side.value
        })

        if self.use_lowprofit_protection.value:
            prot.append({
                    "method": "LowProfitPairs",
                    "lookback_period_candles": self.lowprofit_protection_lookback.value,
                    "trade_limit": self.lowprofit_trade_limit.value,
                    "stop_duration_candles": self.lowprofit_stop_duration.value,
                    "required_profit": self.lowprofit_required_profit.value,
                    "only_per_pair": self.lowprofit_only_per_pair.value
        })

        if self.use_maxdrawdown_protection.value:
            prot.append({
                    "method": "MaxDrawdown",
                    "lookback_period_candles": self.maxdrawdown_protection_lookback.value,
                    "trade_limit": self.maxdrawdown_trade_limit.value,
                    "stop_duration_candles": self.maxdrawdown_stop_duration.value,
                    "max_allowed_drawdown": self.maxdrawdown_allowed_drawdown.value
        })

        return prot    


    # @property
    # def max_entry_position_adjustment(self):
    #     return self.max_epa.value

    ### Dollar Cost Averaging ### This can be Turned on ###
    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        # We need to leave most of the funds for possible further DCA orders
        # This also applies to fixed stakes
        return proposed_stake / self.max_dca_multiplier

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        if current_profit > 0.1 and current_profit < 0.15 and trade.nr_of_successful_exits == 0:
            # Take 50% of the profit at +10%
            return -(trade.stake_amount / 2)

        if current_profit > -0.07 and trade.nr_of_successful_entries == 1:
            return None

        if current_profit > -0.1 and trade.nr_of_successful_entries == 2:
            return None

        if current_profit > -0.16 and trade.nr_of_successful_entries == 3:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        # Allow up to 3 additional increasingly larger buys (4 in total)
        # Initial buy is 1x
        # If that falls to -5% profit, we buy 1.25x more, average profit should increase to roughly -2.2%
        # If that falls down to -5% again, we buy 1.5x more
        # If that falls once again down to -5%, we buy 1.75x more
        # Total stake for this trade would be 1 + 1.25 + 1.5 + 1.75 = 5.5x of the initial allowed stake.
        # Total stake for this trade would be 1 + 1.5 + 2 + 2.5 = 5.5x of the initial allowed stake.
        # That is why max_dca_multiplier is 5.5
        # Hope you have a deep wallet!
        try:
            # This returns first order stake size
            stake_amount = filled_entries[0].cost
            # This then calculates current safety order size
            stake_amount = stake_amount * (1 + (count_of_entries * 0.5))
            return stake_amount
        except Exception as exception:
            return None

        return None


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


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate all indicators used by the strategy"""

        # Pivot Points
        pivots = pivots_points.pivots_points(dataframe, timeperiod=32, levels=5) #50
        dataframe['pivot'] = pivots['pivot']
        dataframe['s5'] = pivots['s5']
        dataframe['r5'] = pivots['r5']
        dataframe['s3'] = pivots['s3']
        dataframe['r3'] = pivots['r3']
        dataframe['s2'] = pivots['s2']
        dataframe['r2'] = pivots['r2']      
        dataframe['r3-dif'] = (dataframe['r3'] - dataframe['r2']) / 4 
        dataframe['r2.25'] = dataframe['r2'] + dataframe['r3-dif'] 
        dataframe['r2.50'] = dataframe['r2'] + (dataframe['r3-dif'] * 2) 
        dataframe['r2.75'] = dataframe['r2'] + (dataframe['r3-dif'] * 3)

        # Filter ZEMA for selling
        for length in self.filterlength.range:
            dataframe[f'ema_1{length}'] = ta.EMA(dataframe['close'], timeperiod=length)
            dataframe[f'ema_2{length}'] = ta.EMA(dataframe[f'ema_1{length}'], timeperiod=length)
            dataframe[f'ema_dif{length}'] = dataframe[f'ema_1{length}'] - dataframe[f'ema_2{length}']
            dataframe[f'zema_{length}'] = dataframe[f'ema_1{length}'] + dataframe[f'ema_dif{length}']

        # Reference MA and offsets
        for valma in self.reference_ma_length.range:
            dataframe[f'reference_ma_{valma}'] = ta.SMA(dataframe['close'], timeperiod=valma)

        dataframe['buy_offset1'] = dataframe[f'reference_ma_{valma}'] * self.buy_offset1.value
        dataframe['buy_offset2'] = dataframe[f'reference_ma_{valma}'] * self.buy_offset2.value
        dataframe['sell_offset1'] = dataframe[f'reference_ma_{valma}'] * self.sell_offset1.value
        dataframe['ref_slope'] =((dataframe[f'reference_ma_{valma}'] - dataframe[f'reference_ma_{valma}'].shift(1)) / dataframe[f'reference_ma_{valma}'].shift(1)) 
        dataframe['ref_slope_sma'] = ta.SMA((dataframe['ref_slope'] * 100), timeperiod=5)
        
        # distance from reference ma to current close
        dataframe['change'] = ((dataframe['close'] - dataframe[f'reference_ma_{valma}']) / dataframe['close']) * 100

        # smoothing the change and offsets
        for valsma in self.smoothing_length.range:
            dataframe[f'smooth_change_{valsma}'] = ta.SMA(dataframe['change'], timeperiod=valsma)


        dataframe['buy_offset3'] = dataframe[f'smooth_change_{valsma}'] - self.buy_change.value
        dataframe['sell_offset2'] = dataframe[f'smooth_change_{valsma}'] + self.sell_change.value
        dataframe['smooth_ma_slope'] = pta.momentum.slope(dataframe[f'smooth_change_{valsma}'])
        dataframe['smooth_slope_sma'] = ta.SMA(dataframe['smooth_ma_slope'], timeperiod=5)

        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)

        ### I.ntelligent B.uying S.ystem ###
        # 300 Candle Rolling Min-Max
        for l in self.max_length.range:
            dataframe['min'] = dataframe['open'].rolling(l).min()
            dataframe['max'] = dataframe['close'].rolling(l).max()

        # distance from the rolling max in percent
        dataframe['from_max'] = ((dataframe['close'] - dataframe['max']) / dataframe['close']) * 100
        # distance from the rolling min in percent
        dataframe['from_min'] = ((dataframe['open'] - dataframe['min']) / dataframe['open']) * 100
        dataframe['min_zema'] = ((dataframe[f'zema_{length}'] - dataframe['min']) / dataframe[f'zema_{length}']) * 100
        dataframe['from dif'] = dataframe['from_min'] + dataframe['from_max']

        return dataframe


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bull.value)) &
                (df['ref_slope_sma'] > self.ref_bull.value) &
                (df['from_max'] > self.from_bull.value) &
                (self.buy01.value == True) &
                (self.rsi01.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '1 Smooth Bull - Ref Bull')


        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_up.value)) &
                (df['smooth_slope_sma'] < self.smooth_bull.value) &
                (df['from_max'] < self.from_bull.value) &
                (df['ref_slope_sma'] > self.ref_bull.value) &
                (self.buy02.value == True) &
                (self.rsi02.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '2 Smooth Up - Ref Bull')


        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  0)) &
                (df['smooth_slope_sma'] < self.smooth_up.value) &
                (df['smooth_slope_sma'] > self.smooth_down.value) &
                (df['from_max'] < self.from_ranging.value) &
                (df['ref_slope_sma'] > self.ref_bull.value) &
                (self.buy03.value == True) &
                (self.rsi03.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '3 Smooth Range - Ref Bull')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_down.value)) &
                (df['smooth_slope_sma'] > self.smooth_bear.value) &
                (df['ref_slope_sma'] > self.ref_bull.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy04.value == True) &
                (self.rsi04.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '4Smooth Down - Ref Bull')

#not using
        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bear.value)) &
                (df['close'] < df['buy_offset2']) & # changed from 1 - 2
                (df['smooth_slope_sma'] < self.smooth_bear.value) &
                (df['ref_slope_sma'] > self.ref_bull.value) &
                (df['from_max'] < self.from_bear.value) &
                (self.buy05.value == True) &
                (self.rsi05.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '5 Smooth Bear - Ref Bull')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bull.value)) &
                (df['ref_slope_sma'] < self.ref_bull.value) &
                (df['ref_slope_sma'] > self.ref_up.value) &
                (df['from_max'] < self.from_up.value) &
                (self.buy06.value == True) &
                (self.rsi06.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '6 Smooth Bull - Ref Up')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_up.value)) &
                (df['smooth_slope_sma'] < self.smooth_bull.value) &
                (df['ref_slope_sma'] < self.ref_bull.value) &
                (df['ref_slope_sma'] > self.ref_up.value) &
                (df['from_max'] < self.from_up.value) &
                (self.buy07.value == True) &
                (self.rsi07.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '7 Smooth Up - Ref Up')

# not using
        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  0)) &
                (df['smooth_slope_sma'] < self.smooth_up.value) &
                (df['smooth_slope_sma'] > self.smooth_down.value) &
                (df['ref_slope_sma'] < self.ref_bull.value) &
                (df['ref_slope_sma'] > self.ref_up.value) &
                (df['from_max'] < self.from_ranging.value) &
                (self.buy08.value == True) &
                (self.rsi08.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '8 Smooth Range - Ref Up')
# not using
        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_down.value)) &
                (df['smooth_slope_sma'] > self.smooth_bear.value) &
                (df['ref_slope_sma'] < self.ref_bull.value) &
                (df['ref_slope_sma'] > self.ref_up.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy09.value == True) &
                (self.rsi09.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '9 Smooth Down - Ref Up')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bear.value)) &
                (df['close'] < df['buy_offset1']) &
                (df['smooth_slope_sma'] < self.smooth_bear.value) &
                (df['ref_slope_sma'] < self.ref_bull.value) &
                (df['ref_slope_sma'] > self.ref_up.value) &
                (df['from_max'] < self.from_bear.value) &
                (self.buy10.value == True) &
                (self.rsi10.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '10 Smooth Bear - Ref Up')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bull.value)) &
                (df['ref_slope_sma'] < self.ref_up.value) &
                (df['ref_slope_sma'] > self.ref_down.value) &
                (df['from_max'] < self.from_ranging.value) &
                (self.buy11.value == True) &
                (self.rsi11.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '11 Smooth Bull - Ref Range')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_up.value)) &
                (df['smooth_slope_sma'] < self.smooth_bull.value) &
                (df['ref_slope_sma'] < self.ref_up.value) &
                (df['ref_slope_sma'] > self.ref_down.value) &
                (df['from_max'] < self.from_ranging.value) &
                (self.buy12.value == True) &
                (self.rsi12.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '12 Smooth Up - Ref Range')


        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'], 0)) &
                (df['smooth_slope_sma'] < self.smooth_up.value) &
                (df['smooth_slope_sma'] > self.smooth_down.value) &
                (df['ref_slope_sma'] < self.ref_up.value) &
                (df['ref_slope_sma'] > self.ref_down.value) &
                (df['from_max'] < self.from_ranging.value) &
                (self.buy13.value == True) &
                (self.rsi13.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '13 Smooth Range - Ref Range')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_down.value)) &
                (df['smooth_slope_sma'] > self.smooth_bear.value) &
                (df['ref_slope_sma'] < self.ref_up.value) &
                (df['ref_slope_sma'] > self.ref_down.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy14.value == True) &
                (self.rsi14.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '14 Smooth Down - Ref Range')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bear.value)) &
                (df['close'] < df['buy_offset1']) &
                (df['smooth_slope_sma'] < self.smooth_bear.value) &
                (df['ref_slope_sma'] < self.ref_up.value) &
                (df['ref_slope_sma'] > self.ref_down.value) &
                (df['from_max'] < self.from_bear.value) &
                (self.buy15.value == True) &
                (self.rsi15.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '15 Smooth Bear - Ref Range')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bull.value)) &
                (df['ref_slope_sma'] < self.ref_down.value) &
                (df['ref_slope_sma'] > self.ref_bear.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy16.value == True) &
                (self.rsi16.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '16 Smooth Bull - Ref Down')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_up.value)) &
                (df['smooth_slope_sma'] > self.smooth_bull.value) &
                (df['ref_slope_sma'] < self.ref_down.value) &
                (df['ref_slope_sma'] > self.ref_bear.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy17.value == True) &
                (self.rsi17.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '17 Smooth Up - Ref Down')


        df.loc[
            (
                (qtpylib.crossed_above(df['change'], df[f'smooth_change_{self.smoothing_length.value}'])) &
                (df['close'] < df['buy_offset1']) &
                (df['smooth_slope_sma'] < self.smooth_up.value) &
                (df['smooth_slope_sma'] > self.smooth_down.value) &
                (df['ref_slope_sma'] < self.ref_down.value) &
                (df['ref_slope_sma'] > self.ref_bear.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy18.value == True) &
                (self.rsi18.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '18 Smooth Range - Ref Down')

        df.loc[
            (
                (qtpylib.crossed_above(df['change'], df[f'smooth_change_{self.smoothing_length.value}'])) &
                (df['smooth_slope_sma'] > self.smooth_bear.value) &
                (df['ref_slope_sma'] < self.ref_down.value) &
                (df['ref_slope_sma'] > self.ref_bear.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy19.value == True) &
                (self.rsi19.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '19 Smooth Down - Ref Down - % XO')

        df.loc[
            (
                (df['min'] > df['close']) &
                (df['smooth_slope_sma'] > self.smooth_bear.value) &
                (df['ref_slope_sma'] < self.ref_down.value) &
                (df['ref_slope_sma'] > self.ref_bear.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy20.value == True) &
                (self.rsi20.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '20 Smooth Down - Ref Down - Open < Min')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bear.value)) &
                (df['smooth_slope_sma'] < self.smooth_bear.value) &
                (df['ref_slope_sma'] < self.ref_down.value) &
                (df['ref_slope_sma'] > self.ref_bear.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy21.value == True) &
                (self.rsi21.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '21 Smooth Bear - Ref Down')


        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bull.value)) &
                (df['ref_slope_sma'] < self.ref_bear.value) &
                (df['from_max'] < self.from_bear.value) &
                (self.buy22.value == True) &
                (self.rsi22.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '22 Smooth Bull - Ref Bear')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_up.value)) &
                (df['smooth_slope_sma'] < self.smooth_bull.value) &
                (df['ref_slope_sma'] < self.ref_bear.value) &
                (df['from_max'] < self.from_bear.value) &
                (self.buy23.value == True) &
                (self.rsi23.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '23 Smooth Up - Ref Bear')


        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  0)) &
                (df['smooth_slope_sma'] < self.smooth_up.value) &
                (df['smooth_slope_sma'] > self.smooth_down.value) &
                (df['ref_slope_sma'] < self.ref_bear.value) &
                (df['from_max'] < self.from_bear.value) &
                (self.buy24.value == True) &
                (self.rsi24.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '24 Smooth Range - Ref Bear')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_down.value)) &
                (df['smooth_slope_sma'] > self.smooth_bear.value) &
                (df['ref_slope_sma'] < self.ref_bear.value) &
                (df['from_max'] < self.from_bear.value) &
                (self.buy25.value == True) &
                (self.rsi25.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '25 Smooth Down - Ref Bear')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bear.value)) &
                # (df['close'] < df['buy_offset2']) &
                (df['smooth_slope_sma'] < self.smooth_bear.value) &
                (df['ref_slope_sma'] < self.ref_bear.value) &
                (df['from_max'] < self.from_bear.value) &
                (self.buy26.value == True) &
                (self.rsi26.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '26 Smooth Bear - Ref Bear')


        df.loc[
            (
                (qtpylib.crossed_above(df['change'], df['buy_offset3'])) &
                (df['pivot'] < df['sell_offset1']) &
                (df['close'] < df['buy_offset1']) &
                (df['smooth_slope_sma'] > self.smooth_down.value) &
                (df['from_max'] < self.from_ranging.value) &
                (self.buy27.value == True) &
                (self.rsi27.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '27 XO above buy_offset3 | Range - Bull')

        df.loc[
            (
                (df['min'] < df['buy_offset1']) &
                ((df['from_max'] - df['from_max'].shift(8)) < self.from_down.value) &
                (df['close'] < df['buy_offset1']) &
                (self.buy28.value == True) &
                (self.rsi28.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '28 Low XB Min < buy_offset1')


        df.loc[
            (
                (qtpylib.crossed_above(df['change'], df['buy_offset3'])) &
                (df['pivot'] < df['sell_offset1']) &
                (df['close'] < df['buy_offset2']) &
                (df['smooth_slope_sma'] < self.smooth_down.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy29.value == True) &
                (self.rsi29.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '29 XO above buy_offset3 | Down - Bear')

        df.loc[
            (
                (df['min'] < df['buy_offset2']) &
                ((df['from_max'] - df['from_max'].shift(8)) < self.from_bear.value) &
                (df['close'] < df['buy_offset2']) &
                (self.buy30.value == True) &
                (self.rsi30.value > df['rsi']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '30 Low XB Min < buy_offset2')

        # df.loc[
        #     (
        #         (qtpylib.crossed_above(df['pivot'], df[f'reference_ma_{self.reference_ma_length.value}'])) &
        #         # (df['close'] < df[f'reference_ma_{self.reference_ma_length.value}']) &
        #         # (df[f'reference_ma_{self.reference_ma_length.value}'] > df[f'reference_ma_{self.reference_ma_length.value}'].shift(1)) &
        #         (df['smooth_ma_slope'] > df['ref_slope_sma']) &
        #         (df['ref_slope_sma'] > 0) &
        #         (self.buy30.value == True) &
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, '30 Golden XO')

        df.loc[
            (
                (df['from dif'] > df['from_max']) &
                (df['from dif'].shift(1) == df['from_max'].shift(1)) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'DIF = FROM MAX')

        # df.loc[
        #     (
        #         (qtpylib.crossed_above(df['change'], df['buy_offset3'])) &
        #         (df['smooth_ma_slope'] > -0.005) &
        #         (df['close'] < df[f'reference_ma_{self.reference_ma_length.value}']) &
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'XO above buy_offset|slope up')

        # df.loc[
        #     (
        #         (df['min'] < df['buy_offset1']) &
        #         ((df['from_max'] - df['from_max'].shift(8)) < self.bear_from_max.value) &
        #         (df['close'] < df['buy_offset1']) &
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'Low XB Min < buy_offset1')

        # df.loc[
        #     (
        #         (qtpylib.crossed_above(df['close'], df['buy_offset2'])) &
        #         (df['change'] < df['buy_offset3']) &
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'XO above buy_offset2')

        # df.loc[
        #     (
        #         (qtpylib.crossed_above(df['change'], df[f'smooth_change_{self.smoothing_length.value}'])) &
        #         (df['close'] < df[f'reference_ma_{self.reference_ma_length.value}']) &
        #         (df[f'reference_ma_{self.reference_ma_length.value}'] > df[f'reference_ma_{self.reference_ma_length.value}'].shift(1)) &
        #         (df['close'] > df['buy_offset1']) &
        #         (df['smooth_ma_slope'] < 0) &
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'XO above Ref')

        return df


    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r5'])) &
                (df['close'] > (df[f'reference_ma_{self.reference_ma_length.value}'] * self.sell_offset1.value)) &
                (self.sell01.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R5 - XO')
      
        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r3'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (self.sell02.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R3 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2.75'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (self.sell03.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2.75 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2.50'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (self.sell04.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2.5 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2.25'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (self.sell05.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2.25 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (self.sell06.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['pivot'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (self.sell07.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'Pivot - XO')

        return df


        # Best result:

        #    585/1000:    279 trades. 259/0/20 Wins/Draws/Losses. Avg profit   2.54%. Median profit   3.90%. Total profit 2061.56810074 USDT ( 206.16%). Avg duration 1 day, 23:57:00 min. Objective: -2061.56810


        #     # Buy hyperspace params:
        #     buy_params = {
        #         "buy1": True,  # value loaded from strategy
        #         "buy10": True,  # value loaded from strategy
        #         "buy11": False,  # value loaded from strategy
        #         "buy12": False,  # value loaded from strategy
        #         "buy13": False,  # value loaded from strategy
        #         "buy14": False,  # value loaded from strategy
        #         "buy15": False,  # value loaded from strategy
        #         "buy16": True,  # value loaded from strategy
        #         "buy17": True,  # value loaded from strategy
        #         "buy18": True,  # value loaded from strategy
        #         "buy19": False,  # value loaded from strategy
        #         "buy2": False,  # value loaded from strategy
        #         "buy20": True,  # value loaded from strategy
        #         "buy21": True,  # value loaded from strategy
        #         "buy22": True,  # value loaded from strategy
        #         "buy23": True,  # value loaded from strategy
        #         "buy24": False,  # value loaded from strategy
        #         "buy25": True,  # value loaded from strategy
        #         "buy26": True,  # value loaded from strategy
        #         "buy27": True,  # value loaded from strategy
        #         "buy28": False,  # value loaded from strategy
        #         "buy29": True,  # value loaded from strategy
        #         "buy3": True,  # value loaded from strategy
        #         "buy4": True,  # value loaded from strategy
        #         "buy5": False,  # value loaded from strategy
        #         "buy6": True,  # value loaded from strategy
        #         "buy7": True,  # value loaded from strategy
        #         "buy8": False,  # value loaded from strategy
        #         "buy9": False,  # value loaded from strategy
        #         "buy_change": 1.6,  # value loaded from strategy
        #         "buy_offset1": 0.99,  # value loaded from strategy
        #         "buy_offset2": 0.92,  # value loaded from strategy
        #         "from_bear": -11.7,  # value loaded from strategy
        #         "from_bull": -3.1,  # value loaded from strategy
        #         "from_down": -5.4,  # value loaded from strategy
        #         "from_ranging": -3.5,  # value loaded from strategy
        #         "from_up": -4.7,  # value loaded from strategy
        #         "ref_bear": -0.113,  # value loaded from strategy
        #         "ref_bull": 0.098,  # value loaded from strategy
        #         "ref_down": -0.067,  # value loaded from strategy
        #         "ref_up": 0.027,  # value loaded from strategy
        #         "reference_ma_length": 192,  # value loaded from strategy
        #         "smooth_bear": -0.117,  # value loaded from strategy
        #         "smooth_bull": 0.086,  # value loaded from strategy
        #         "smooth_down": -0.013,  # value loaded from strategy
        #         "smooth_up": 0.073,  # value loaded from strategy
        #         "smoothing_length": 23,  # value loaded from strategy
        #     }

        #     # Sell hyperspace params:
        #     sell_params = {
        #         "filterlength": 35,
        #         "sell_change": 1.8,
        #         "sell_offset1": 1.03,
        #         "sell_slope": -0.04,
        #         "ts0": 0.008,
        #         "ts1": 0.011,
        #         "ts2": 0.021,
        #         "ts3": 0.031,
        #         "ts4": 0.03,
        #         "ts5": 0.05,
        #         "tsl_target0": 0.044,
        #         "tsl_target1": 0.055,
        #         "tsl_target2": 0.083,
        #         "tsl_target3": 0.12,
        #         "tsl_target4": 0.2,
        #         "tsl_target5": 0.4,
        #     }

        #     # Protection hyperspace params:
        #     protection_params = {
        #         "cooldown_lookback": 1,
        #         "lowprofit_only_per_pair": True,
        #         "lowprofit_protection_lookback": 1,
        #         "lowprofit_required_profit": -0.829,
        #         "lowprofit_stop_duration": 83,
        #         "lowprofit_trade_limit": 1,
        #         "maxdrawdown_allowed_drawdown": 0.1,
        #         "maxdrawdown_protection_lookback": 3,
        #         "maxdrawdown_stop_duration": 74,
        #         "maxdrawdown_trade_limit": 1,
        #         "stop_duration": 102,
        #         "stop_protection_only_per_pair": True,
        #         "stop_protection_only_per_side": True,
        #         "stop_protection_required_profit": -0.924,
        #         "stop_protection_trade_limit": 9,
        #         "use_lowprofit_protection": False,
        #         "use_maxdrawdown_protection": True,
        #         "use_stop_protection": False,
        #     }

        #     # ROI table:  # value loaded from strategy
        #     minimal_roi = {
        #         "0": 0.1
        #     }

        #     # Stoploss:
        #     stoploss = -0.2  # value loaded from strategy

        #     # Trailing stop:
        #     trailing_stop = True  # value loaded from strategy
        #     trailing_stop_positive = None  # value loaded from strategy
        #     trailing_stop_positive_offset = 0.0  # value loaded from strategy
        #     trailing_only_offset_is_reached = False  # value loaded from strategy


# {
#   "strategy_name": "CTIBS",
#   "params": {
#     "roi": {
#       "0": 0.14300000000000002,
#       "736": 0.099,
#       "2369": 0.024,
#       "5731": 0
#     },
#     "stoploss": {
#       "stoploss": -0.28
#     },
#     "trailing": {
#       "trailing_stop": true,
#       "trailing_stop_positive": null,
#       "trailing_stop_positive_offset": 0.0,
#       "trailing_only_offset_is_reached": false
#     },
#     "max_open_trades": {
#       "max_open_trades": 6
#     },
#     "buy": {
#       "buy1": true,
#       "buy10": false,
#       "buy11": true,
#       "buy12": false,
#       "buy13": false,
#       "buy14": true,
#       "buy15": false,
#       "buy16": true,
#       "buy17": false,
#       "buy18": false,
#       "buy19": false,
#       "buy2": false,
#       "buy20": false,
#       "buy21": false,
#       "buy22": false,
#       "buy23": true,
#       "buy24": false,
#       "buy25": true,
#       "buy26": false,
#       "buy27": true,
#       "buy28": true,
#       "buy29": true,
#       "buy3": true,
#       "buy4": false,
#       "buy5": false,
#       "buy6": true,
#       "buy7": false,
#       "buy8": false,
#       "buy9": true,
#       "buy_change": 1.5,
#       "buy_offset1": 0.98,
#       "buy_offset2": 0.94,
#       "from_bear": -12.4,
#       "from_bull": -3.1,
#       "from_down": -7.1,
#       "from_ranging": -2.7,
#       "from_up": -2.0,
#       "max_length": 48,
#       "ref_bear": -0.088,
#       "ref_bull": 0.087,
#       "ref_down": -0.075,
#       "ref_up": 0.062,
#       "reference_ma_length": 187,
#       "smooth_bear": -0.098,
#       "smooth_bull": 0.085,
#       "smooth_down": -0.058,
#       "smooth_up": 0.052,
#       "smoothing_length": 15
#     },
#     "sell": {
#       "filterlength": 40,
#       "sell_change": 1.5,
#       "sell_offset1": 1.04,
#       "sell_slope": 0.01,
#       "ts0": 0.008,
#       "ts1": 0.01,
#       "ts2": 0.018,
#       "ts3": 0.031,
#       "ts4": 0.05,
#       "ts5": 0.04,
#       "tsl_target0": 0.041,
#       "tsl_target1": 0.079,
#       "tsl_target2": 0.099,
#       "tsl_target3": 0.14,
#       "tsl_target4": 0.17,
#       "tsl_target5": 0.3
#     },
#     "protection": {
#       "cooldown_lookback": 48,
#       "lowprofit_only_per_pair": true,
#       "lowprofit_protection_lookback": 5,
#       "lowprofit_required_profit": -0.704,
#       "lowprofit_stop_duration": 100,
#       "lowprofit_trade_limit": 2,
#       "maxdrawdown_allowed_drawdown": 0.045,
#       "maxdrawdown_protection_lookback": 3,
#       "maxdrawdown_stop_duration": 15,
#       "maxdrawdown_trade_limit": 5,
#       "stop_duration": 19,
#       "stop_protection_only_per_pair": true,
#       "stop_protection_only_per_side": true,
#       "stop_protection_required_profit": -0.126,
#       "stop_protection_trade_limit": 9,
#       "use_lowprofit_protection": true,
#       "use_maxdrawdown_protection": false,
#       "use_stop_protection": false
#     }
#   },
#   "ft_stratparam_v": 1,
#   "export_time": "2023-05-02 22:49:05.899725+00:00"
# }

# Best result:

#   1425/1500:    365 trades. 318/20/27 Wins/Draws/Losses. Avg profit   2.60%. Median profit   3.93%. Total profit 3495.92254012 USDT ( 349.59%). Avg duration 1 day, 18:44:00 min. Objective: -3495.92254


#     # Buy hyperspace params:
#     buy_params = {
#         "buy01": True,
#         "buy02": False,
#         "buy03": True,
#         "buy04": True,
#         "buy05": True,
#         "buy06": False,
#         "buy07": True,
#         "buy08": True,
#         "buy09": False,
#         "buy10": False,
#         "buy11": False,
#         "buy12": False,
#         "buy13": False,
#         "buy14": False,
#         "buy15": False,
#         "buy16": False,
#         "buy17": False,
#         "buy18": False,
#         "buy19": False,
#         "buy20": False,
#         "buy21": True,
#         "buy22": True,
#         "buy23": True,
#         "buy24": True,
#         "buy25": True,
#         "buy26": False,
#         "buy27": False,
#         "buy28": False,
#         "buy29": True,
#         "buy30": False,
#         "buy_change": 1.2,
#         "buy_offset1": 0.99,
#         "buy_offset2": 0.92,
#         "from_bear": -10.5,
#         "from_bull": -3.1,
#         "from_down": -6.3,
#         "from_ranging": -2.5,
#         "from_up": -2.2,
#         "max_length": 192,
#         "ref_bear": -0.097,
#         "ref_bull": 0.095,
#         "ref_down": -0.07,
#         "ref_up": 0.043,
#         "reference_ma_length": 191,
#         "smooth_bear": -0.095,
#         "smooth_bull": 0.076,
#         "smooth_down": -0.056,
#         "smooth_up": 0.04,
#         "smoothing_length": 23,
#     }

#     # Sell hyperspace params:
#     sell_params = {
#         "filterlength": 30,
#         "sell01": False,
#         "sell02": True,
#         "sell03": True,
#         "sell04": True,
#         "sell05": False,
#         "sell06": False,
#         "sell07": True,
#         "sell08": True,
#         "sell09": False,
#         "sell10": True,
#         "sell_change": 1.7,
#         "sell_offset1": 1.04,
#         "sell_slope": 0.0,
#         "ts0": 0.008,
#         "ts1": 0.01,
#         "ts2": 0.022,
#         "ts3": 0.038,
#         "ts4": 0.03,
#         "ts5": 0.04,
#         "tsl_target0": 0.044,
#         "tsl_target1": 0.065,
#         "tsl_target2": 0.097,
#         "tsl_target3": 0.12,
#         "tsl_target4": 0.16,
#         "tsl_target5": 0.3,
#     }

#     # Protection hyperspace params:
#     protection_params = {
#         "cooldown_lookback": 48,  # value loaded from strategy
#         "lowprofit_only_per_pair": True,  # value loaded from strategy
#         "lowprofit_protection_lookback": 5,  # value loaded from strategy
#         "lowprofit_required_profit": -0.704,  # value loaded from strategy
#         "lowprofit_stop_duration": 100,  # value loaded from strategy
#         "lowprofit_trade_limit": 2,  # value loaded from strategy
#         "maxdrawdown_allowed_drawdown": 0.045,  # value loaded from strategy
#         "maxdrawdown_protection_lookback": 3,  # value loaded from strategy
#         "maxdrawdown_stop_duration": 15,  # value loaded from strategy
#         "maxdrawdown_trade_limit": 5,  # value loaded from strategy
#         "stop_duration": 19,  # value loaded from strategy
#         "stop_protection_only_per_pair": True,  # value loaded from strategy
#         "stop_protection_only_per_side": True,  # value loaded from strategy
#         "stop_protection_required_profit": -0.126,  # value loaded from strategy
#         "stop_protection_trade_limit": 9,  # value loaded from strategy
#         "use_lowprofit_protection": True,  # value loaded from strategy
#         "use_maxdrawdown_protection": False,  # value loaded from strategy
#         "use_stop_protection": False,  # value loaded from strategy
#     }

#     # ROI table:  # value loaded from strategy
#     minimal_roi = {
#         "0": 0.143,
#         "736": 0.099,
#         "2369": 0.024,
#         "5731": 0
#     }

#     # Stoploss:
#     stoploss = -0.28  # value loaded from strategy

#     # Trailing stop:
#     trailing_stop = True  # value loaded from strategy
#     trailing_stop_positive = None  # value loaded from strategy
#     trailing_stop_positive_offset = 0.0  # value loaded from strategy
#     trailing_only_offset_is_reached = False  # value loaded from strategy
    

#     # Max Open Trades:
#     max_open_trades = 6  # value loaded from strategy

# Result for strategy CTIBS
# ============================================================= BACKTESTING REPORT =============================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |  OPUL/USDT |        30 |           4.64 |         139.20 |           708.395 |          70.84 |         14:03:00 |    28     1     1  93.3 |
# |  AGIX/USDT |        37 |           3.02 |         111.78 |           342.690 |          34.27 |         14:09:00 |    32     0     5  86.5 |
# |   INJ/USDT |        21 |           2.79 |          58.60 |           332.616 |          33.26 |  1 day, 16:47:00 |    18     1     2  85.7 |
# |   LDO/USDT |        20 |           5.12 |         102.36 |           302.659 |          30.27 |         23:04:00 |    19     1     0   100 |
# |  RNDR/USDT |        23 |           2.46 |          56.48 |           292.762 |          29.28 |   1 day, 1:40:00 |    22     0     1  95.7 |
# |    OP/USDT |        11 |           4.43 |          48.75 |           225.127 |          22.51 |         10:14:00 |    11     0     0   100 |
# |   IMX/USDT |        18 |           2.33 |          41.89 |           218.038 |          21.80 |  1 day, 13:00:00 |    17     0     1  94.4 |
# |  ORAI/USDT |        19 |           2.63 |          49.93 |           156.959 |          15.70 |         14:02:00 |    18     0     1  94.7 |
# |   APT/USDT |        10 |           4.28 |          42.77 |           144.811 |          14.48 |   1 day, 1:15:00 |    10     0     0   100 |
# |  KLAY/USDT |         6 |           3.28 |          19.71 |           132.670 |          13.27 |  1 day, 12:40:00 |     4     1     1  66.7 |
# |  DYDX/USDT |        16 |           2.16 |          34.52 |           125.692 |          12.57 | 2 days, 13:57:00 |    14     1     1  87.5 |
# |   FIL/USDT |         6 |           4.08 |          24.47 |           125.056 |          12.51 |         20:08:00 |     6     0     0   100 |
# |   ARB/USDT |         5 |           3.15 |          15.75 |           112.833 |          11.28 | 3 days, 16:30:00 |     4     1     0   100 |
# |  CSPR/USDT |         7 |           2.80 |          19.58 |           108.527 |          10.85 |   1 day, 7:24:00 |     7     0     0   100 |
# |  ROSE/USDT |         4 |           4.21 |          16.85 |           106.125 |          10.61 |         18:41:00 |     4     0     0   100 |
# |   FLR/USDT |         4 |           5.50 |          21.98 |            99.699 |           9.97 |         15:41:00 |     4     0     0   100 |
# |   GRT/USDT |         5 |           3.21 |          16.05 |            96.898 |           9.69 |   1 day, 0:21:00 |     5     0     0   100 |
# | GALAX/USDT |         8 |           5.04 |          40.31 |            84.824 |           8.48 |          8:26:00 |     7     0     1  87.5 |
# |  HBAR/USDT |         6 |           2.17 |          13.02 |            80.588 |           8.06 | 9 days, 16:55:00 |     3     3     0   100 |
# | JASMY/USDT |        11 |           2.85 |          31.36 |            78.855 |           7.89 |  2 days, 1:59:00 |     9     0     2  81.8 |
# |   CRO/USDT |         1 |           8.57 |           8.57 |            67.589 |           6.76 |         16:30:00 |     1     0     0   100 |
# |   EWT/USDT |         5 |           2.53 |          12.67 |            64.415 |           6.44 |   1 day, 3:06:00 |     5     0     0   100 |
# |   FET/USDT |         2 |           3.41 |           6.83 |            49.115 |           4.91 |   1 day, 2:38:00 |     2     0     0   100 |
# |   XRP/USDT |         2 |           3.26 |           6.53 |            41.612 |           4.16 |         17:22:00 |     2     0     0   100 |
# |  SAND/USDT |         3 |           4.62 |          13.86 |            38.537 |           3.85 |         12:20:00 |     3     0     0   100 |
# |  NEAR/USDT |         4 |           2.51 |          10.03 |            36.898 |           3.69 |   1 day, 3:08:00 |     4     0     0   100 |
# |   XLM/USDT |         2 |           3.03 |           6.07 |            28.982 |           2.90 |         16:08:00 |     2     0     0   100 |
# |   TRX/USDT |         2 |           4.11 |           8.23 |            27.729 |           2.77 |         12:08:00 |     2     0     0   100 |
# |  KAVA/USDT |         2 |           4.18 |           8.36 |            27.547 |           2.75 |   1 day, 1:38:00 |     2     0     0   100 |
# |   SOL/USDT |         7 |           1.90 |          13.32 |            27.384 |           2.74 |  2 days, 6:04:00 |     4     3     0   100 |
# |   QNT/USDT |         2 |           2.24 |           4.48 |            25.599 |           2.56 |  2 days, 8:00:00 |     2     0     0   100 |
# |   APE/USDT |         3 |           3.18 |           9.54 |            24.531 |           2.45 |         13:15:00 |     3     0     0   100 |
# |   ETC/USDT |         2 |           4.58 |           9.16 |            16.738 |           1.67 |         22:45:00 |     2     0     0   100 |
# |   ENJ/USDT |         2 |           2.21 |           4.42 |            14.311 |           1.43 |  2 days, 2:22:00 |     1     1     0   100 |
# |   BTC/USDT |         1 |           2.40 |           2.40 |            14.117 |           1.41 |  2 days, 6:45:00 |     1     0     0   100 |
# |   ZEC/USDT |         1 |           2.40 |           2.40 |            11.422 |           1.14 |  2 days, 8:00:00 |     1     0     0   100 |
# |   CRV/USDT |         3 |           1.03 |           3.09 |            10.368 |           1.04 | 2 days, 20:15:00 |     2     1     0   100 |
# |  ATOM/USDT |         3 |           0.65 |           1.94 |             8.839 |           0.88 | 3 days, 12:30:00 |     2     1     0   100 |
# |   XTZ/USDT |         1 |           1.79 |           1.79 |             8.540 |           0.85 | 2 days, 22:30:00 |     1     0     0   100 |
# |   KSM/USDT |         1 |           2.40 |           2.40 |             8.217 |           0.82 | 2 days, 14:15:00 |     1     0     0   100 |
# |  AVAX/USDT |         3 |           0.90 |           2.71 |             8.148 |           0.81 | 6 days, 15:15:00 |     2     1     0   100 |
# |  SCRT/USDT |         1 |           2.40 |           2.40 |             5.970 |           0.60 |  1 day, 21:45:00 |     1     0     0   100 |
# | THETA/USDT |         2 |           0.65 |           1.29 |             4.826 |           0.48 | 9 days, 15:52:00 |     1     1     0   100 |
# |  IOTA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |  OSMO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |   ETH/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |   XDC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |  LINK/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |   DOT/USDT |         1 |           0.00 |           0.00 |             0.000 |           0.00 |  6 days, 8:00:00 |     0     1     0     0 |
# |   UNI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |   VET/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |   EOS/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |  DASH/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# | OCEAN/USDT |        12 |           1.49 |          17.92 |            -8.904 |          -0.89 | 2 days, 18:38:00 |    10     1     1  83.3 |
# |  DOGE/USDT |         2 |          -1.34 |          -2.68 |           -21.435 |          -2.14 | 12 days, 9:15:00 |     1     0     1  50.0 |
# |   ADA/USDT |         2 |          -1.69 |          -3.39 |           -28.639 |          -2.86 | 5 days, 17:38:00 |     1     0     1  50.0 |
# |  EGLD/USDT |         2 |          -2.38 |          -4.76 |           -47.439 |          -4.74 |   1 day, 2:45:00 |     1     0     1  50.0 |
# |   AKT/USDT |         7 |          -1.86 |         -13.02 |          -113.644 |         -11.36 |  2 days, 7:26:00 |     6     0     1  85.7 |
# |  LUNC/USDT |         2 |         -12.83 |         -25.66 |          -140.852 |         -14.09 | 17 days, 5:38:00 |     0     1     1     0 |
# | MATIC/USDT |         3 |          -6.91 |         -20.73 |          -158.044 |         -15.80 | 5 days, 14:05:00 |     2     0     1  66.7 |
# |  ALGO/USDT |         2 |         -12.97 |         -25.95 |          -182.961 |         -18.30 |  8 days, 4:00:00 |     1     0     1  50.0 |
# |  ANKR/USDT |        10 |          -2.20 |         -21.96 |          -249.415 |         -24.94 |  1 day, 19:15:00 |     7     0     3  70.0 |
# |      TOTAL |       365 |           2.60 |         947.62 |          3495.923 |         349.59 |  1 day, 18:44:00 |   318    20    27  87.1 |
# =========================================================== LEFT OPEN TRADES REPORT ===========================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |      Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+-------------------+-------------------------|
# |   INJ/USDT |         1 |          -1.17 |          -1.17 |            -9.476 |          -0.95 |           4:30:00 |     0     0     1     0 |
# |   ADA/USDT |         1 |          -4.28 |          -4.28 |           -32.818 |          -3.28 | 10 days, 12:15:00 |     0     0     1     0 |
# | JASMY/USDT |         1 |          -5.83 |          -5.83 |           -42.769 |          -4.28 | 13 days, 20:45:00 |     0     0     1     0 |
# |  EGLD/USDT |         1 |          -9.01 |          -9.01 |           -73.187 |          -7.32 |   1 day, 19:15:00 |     0     0     1     0 |
# |  DOGE/USDT |         1 |         -16.97 |         -16.97 |          -116.795 |         -11.68 | 24 days, 18:15:00 |     0     0     1     0 |
# |   AKT/USDT |         1 |         -19.49 |         -19.49 |          -148.353 |         -14.84 |  11 days, 4:30:00 |     0     0     1     0 |
# |      TOTAL |         6 |          -9.46 |         -56.73 |          -423.397 |         -42.34 |  10 days, 9:15:00 |     0     0     6     0 |
# ============================================================================ ENTER TAG STATS ============================================================================
# |                                   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |---------------------------------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |               8 Smooth Range - Ref Up |        71 |           3.87 |         275.08 |          1176.933 |         117.69 |  1 day, 10:28:00 |    66     4     1  93.0 |
# |              1 Smooth Bull - Ref Bull |        62 |           3.90 |         241.71 |          1018.094 |         101.81 |   1 day, 3:00:00 |    57     2     3  91.9 |
# |             3 Smooth Range - Ref Bull |       126 |           2.14 |         269.55 |           773.706 |          77.37 |  1 day, 13:27:00 |   107     6    13  84.9 |
# | 29 XO above buy_offset3 | Down - Bear |        77 |           1.05 |          80.76 |           252.870 |          25.29 | 2 days, 19:45:00 |    64     6     7  83.1 |
# |                  7 Smooth Up - Ref Up |        27 |           2.81 |          75.83 |           246.711 |          24.67 |  2 days, 8:07:00 |    22     2     3  81.5 |
# |             22 Smooth Bull - Ref Bear |         1 |           3.81 |           3.81 |            23.052 |           2.31 |         21:30:00 |     1     0     0   100 |
# |               4Smooth Down - Ref Bull |         1 |           0.87 |           0.87 |             4.556 |           0.46 |          0:15:00 |     1     0     0   100 |
# |                                 TOTAL |       365 |           2.60 |         947.62 |          3495.923 |         349.59 |  1 day, 18:44:00 |   318    20    27  87.1 |
# ======================================================= EXIT REASON STATS ========================================================
# |        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# | trailing_stop_loss |     249 |    228     0    21  91.6 |           3.09 |         770.24 |          2813.35  |         128.37 |
# |                roi |      74 |     54    20     0   100 |           2.71 |         200.51 |           922.846 |          33.42 |
# |         Pivot - XO |      21 |     21     0     0   100 |           0.8  |          16.75 |            99.357 |           2.79 |
# |          R2.5 - XO |       8 |      8     0     0   100 |           1.3  |          10.37 |            54.745 |           1.73 |
# |         force_exit |       6 |      0     0     6     0 |          -9.46 |         -56.73 |          -423.397 |          -9.46 |
# |         R2.75 - XO |       4 |      4     0     0   100 |           1    |           4.01 |            17.628 |           0.67 |
# |            R3 - XO |       3 |      3     0     0   100 |           0.82 |           2.47 |            11.393 |           0.41 |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2023-01-01 00:00:00 |
# | Backtesting to              | 2023-04-30 00:00:00 |
# | Max open trades             | 6                   |
# |                             |                     |
# | Total/Daily Avg Trades      | 365 / 3.07          |
# | Starting balance            | 1000 USDT           |
# | Final balance               | 4495.923 USDT       |
# | Absolute profit             | 3495.923 USDT       |
# | Total profit %              | 349.59%             |
# | CAGR %                      | 9954.11%            |
# | Sortino                     | 7.32                |
# | Sharpe                      | 14.53               |
# | Calmar                      | 185.66              |
# | Profit factor               | 2.33                |
# | Expectancy                  | 0.04                |
# | Trades per day              | 3.07                |
# | Avg. daily profit %         | 2.94%               |
# | Avg. stake amount           | 486.735 USDT        |
# | Total trade volume          | 177658.245 USDT     |
# |                             |                     |
# | Best Pair                   | OPUL/USDT 139.20%   |
# | Worst Pair                  | ALGO/USDT -25.95%   |
# | Best trade                  | LDO/USDT 14.29%     |
# | Worst trade                 | MATIC/USDT -27.97%  |
# | Best day                    | 316.573 USDT        |
# | Worst day                   | -423.397 USDT       |
# | Days win/draw/lose          | 87 / 23 / 9         |
# | Avg. Duration Winners       | 18:38:00            |
# | Avg. Duration Loser         | 6 days, 21:24:00    |
# | Rejected Entry signals      | 455976              |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 1001.773 USDT       |
# | Max balance                 | 4919.32 USDT        |
# | Max % of account underwater | 30.23%              |
# | Absolute Drawdown (Account) | 30.23%              |
# | Absolute Drawdown           | 1316.127 USDT       |
# | Drawdown high               | 3353.592 USDT       |
# | Drawdown low                | 2037.465 USDT       |
# | Drawdown Start              | 2023-02-23 01:45:00 |
# | Drawdown End                | 2023-03-10 10:00:00 |
# | Market change               | 84.63%              |
# =====================================================

# Result for strategy CTIBS
# ============================================================== BACKTESTING REPORT =============================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |      Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+-------------------+-------------------------|
# |  OPUL/USDT |        27 |           4.61 |         124.52 |           728.300 |          72.83 |          19:35:00 |    24     1     2  88.9 |
# |   INJ/USDT |        32 |           3.46 |         110.67 |           671.137 |          67.11 |          21:04:00 |    30     0     2  93.8 |
# |   LDO/USDT |        24 |           6.10 |         146.30 |           490.133 |          49.01 |          17:21:00 |    23     1     0   100 |
# |  RNDR/USDT |        32 |           1.84 |          59.03 |           279.977 |          28.00 |          21:03:00 |    28     1     3  87.5 |
# |   GRT/USDT |         9 |           3.62 |          32.54 |           219.261 |          21.93 |          17:40:00 |     8     1     0   100 |
# |  ORAI/USDT |        21 |           3.07 |          64.40 |           207.313 |          20.73 |    1 day, 5:08:00 |    19     1     1  90.5 |
# | JASMY/USDT |         8 |           4.72 |          37.79 |           174.735 |          17.47 |          16:45:00 |     7     1     0   100 |
# |   ARB/USDT |         5 |           4.40 |          21.99 |           162.643 |          16.26 |   1 day, 15:00:00 |     5     0     0   100 |
# | GALAX/USDT |        11 |           5.15 |          56.66 |           131.737 |          13.17 |           8:15:00 |    11     0     0   100 |
# |   IMX/USDT |        12 |           1.94 |          23.27 |           129.586 |          12.96 |          23:56:00 |    11     0     1  91.7 |
# |    OP/USDT |        17 |           2.04 |          34.71 |           120.534 |          12.05 |    1 day, 9:02:00 |    14     2     1  82.4 |
# |   FLR/USDT |         7 |           3.27 |          22.89 |           117.094 |          11.71 |  5 days, 11:32:00 |     5     2     0   100 |
# | OCEAN/USDT |         8 |           4.12 |          32.97 |           108.094 |          10.81 |          22:15:00 |     7     1     0   100 |
# |  ROSE/USDT |         4 |           3.56 |          14.24 |            99.164 |           9.92 |    1 day, 9:22:00 |     4     0     0   100 |
# |   APT/USDT |         6 |           4.43 |          26.59 |            91.821 |           9.18 |    1 day, 0:55:00 |     5     1     0   100 |
# |   SOL/USDT |        10 |           3.42 |          34.16 |            89.145 |           8.91 |    1 day, 2:54:00 |     8     2     0   100 |
# |  AGIX/USDT |        43 |           1.94 |          83.31 |            81.060 |           8.11 |    1 day, 1:54:00 |    37     1     5  86.0 |
# |  CSPR/USDT |         6 |           2.56 |          15.38 |            63.607 |           6.36 |   2 days, 6:20:00 |     3     2     1  50.0 |
# |  DYDX/USDT |        11 |           1.36 |          14.92 |            59.651 |           5.97 |    1 day, 8:29:00 |    10     0     1  90.9 |
# |  ATOM/USDT |         2 |           3.88 |           7.75 |            57.938 |           5.79 |    1 day, 2:30:00 |     2     0     0   100 |
# |  SCRT/USDT |         2 |           4.90 |           9.79 |            46.472 |           4.65 |   2 days, 6:30:00 |     2     0     0   100 |
# |   ETC/USDT |         2 |           4.68 |           9.36 |            38.937 |           3.89 |   1 day, 13:08:00 |     2     0     0   100 |
# |   FET/USDT |         1 |           4.23 |           4.23 |            32.966 |           3.30 |           6:15:00 |     1     0     0   100 |
# |   EWT/USDT |         2 |           2.07 |           4.14 |            30.594 |           3.06 |          22:22:00 |     2     0     0   100 |
# |   CRV/USDT |         5 |           1.95 |           9.75 |            30.347 |           3.03 |          20:54:00 |     4     1     0   100 |
# | MATIC/USDT |         3 |           1.87 |           5.61 |            29.883 |           2.99 |    1 day, 9:25:00 |     3     0     0   100 |
# |   ADA/USDT |         2 |           2.24 |           4.48 |            28.407 |           2.84 |  3 days, 20:45:00 |     1     1     0   100 |
# |   FIL/USDT |         2 |           1.82 |           3.63 |            27.909 |           2.79 |  2 days, 12:52:00 |     1     1     0   100 |
# |   ENJ/USDT |         3 |           1.97 |           5.91 |            25.345 |           2.53 |   1 day, 16:55:00 |     2     1     0   100 |
# |  DASH/USDT |         1 |           4.40 |           4.40 |            23.775 |           2.38 |          12:15:00 |     1     0     0   100 |
# |   VET/USDT |         1 |           3.69 |           3.69 |            23.667 |           2.37 |  2 days, 10:00:00 |     1     0     0   100 |
# |  LINK/USDT |         1 |           1.88 |           1.88 |            15.497 |           1.55 |    1 day, 5:45:00 |     1     0     0   100 |
# |  IOTA/USDT |         1 |           3.92 |           3.92 |            14.363 |           1.44 |           2:15:00 |     1     0     0   100 |
# |  AVAX/USDT |         2 |           2.45 |           4.90 |            13.495 |           1.35 |   2 days, 6:08:00 |     1     1     0   100 |
# |   APE/USDT |         2 |           2.45 |           4.90 |            12.568 |           1.26 |  2 days, 13:45:00 |     1     1     0   100 |
# |  SAND/USDT |         2 |           1.79 |           3.57 |            12.316 |           1.23 |  16 days, 5:30:00 |     1     1     0   100 |
# |   TRX/USDT |         1 |           5.50 |           5.50 |            10.079 |           1.01 |          15:30:00 |     1     0     0   100 |
# |   BTC/USDT |         1 |           0.53 |           0.53 |             3.598 |           0.36 |           7:45:00 |     1     0     0   100 |
# |   KSM/USDT |         1 |           0.28 |           0.28 |             2.161 |           0.22 |          15:45:00 |     1     0     0   100 |
# |   XTZ/USDT |         1 |           0.20 |           0.20 |             0.755 |           0.08 |   1 day, 22:30:00 |     1     0     0   100 |
# |   QNT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |  OSMO/USDT |         1 |           0.00 |           0.00 |             0.000 |           0.00 |  3 days, 13:15:00 |     0     1     0     0 |
# |   ETH/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |   XDC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |   DOT/USDT |         1 |           0.00 |           0.00 |             0.000 |           0.00 |   3 days, 1:00:00 |     0     1     0     0 |
# |  ALGO/USDT |         2 |           0.00 |           0.00 |             0.000 |           0.00 |  10 days, 5:08:00 |     0     2     0     0 |
# |   XLM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |   UNI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# | THETA/USDT |         2 |           0.00 |           0.00 |             0.000 |           0.00 |   7 days, 4:45:00 |     0     2     0     0 |
# |   ZEC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |   EOS/USDT |         1 |           0.00 |           0.00 |             0.000 |           0.00 | 11 days, 19:45:00 |     0     1     0     0 |
# |   CRO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |  EGLD/USDT |         2 |          -1.69 |          -3.38 |           -28.966 |          -2.90 |          23:15:00 |     1     0     1  50.0 |
# |   XRP/USDT |         4 |          -1.44 |          -5.75 |           -41.284 |          -4.13 |  8 days, 20:52:00 |     2     1     1  50.0 |
# |  NEAR/USDT |        10 |           0.94 |           9.45 |           -42.698 |          -4.27 |   1 day, 18:36:00 |     8     1     1  80.0 |
# |   AKT/USDT |         6 |          -0.22 |          -1.30 |           -54.806 |          -5.48 |   2 days, 0:22:00 |     5     0     1  83.3 |
# |  KAVA/USDT |         6 |          -1.14 |          -6.82 |           -86.156 |          -8.62 |   1 day, 23:58:00 |     5     0     1  83.3 |
# |  ANKR/USDT |         7 |          -0.53 |          -3.69 |          -105.833 |         -10.58 |          19:43:00 |     4     1     2  57.1 |
# |  DOGE/USDT |         1 |         -15.93 |         -15.93 |          -119.122 |         -11.91 | 24 days, 21:30:00 |     0     0     1     0 |
# |  HBAR/USDT |         4 |          -4.69 |         -18.77 |          -122.504 |         -12.25 |  7 days, 22:38:00 |     2     1     1  50.0 |
# |  KLAY/USDT |         5 |          -2.85 |         -14.25 |          -129.767 |         -12.98 |   3 days, 9:09:00 |     3     1     1  60.0 |
# |  LUNC/USDT |         1 |         -26.94 |         -26.94 |          -172.469 |         -17.25 | 26 days, 11:45:00 |     0     0     1     0 |
# |      TOTAL |       384 |           2.52 |         967.41 |          3602.456 |         360.25 |   1 day, 16:41:00 |   320    36    28  83.3 |
# ========================================================== LEFT OPEN TRADES REPORT ===========================================================
# |      Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |      Avg Duration |   Win  Draw  Loss  Win% |
# |-----------+-----------+----------------+----------------+-------------------+----------------+-------------------+-------------------------|
# | EGLD/USDT |         1 |          -7.15 |          -7.15 |           -60.213 |          -6.02 |   1 day, 19:30:00 |     0     0     1     0 |
# | RNDR/USDT |         1 |          -7.21 |          -7.21 |           -61.238 |          -6.12 |   1 day, 17:45:00 |     0     0     1     0 |
# |  INJ/USDT |         1 |          -9.94 |          -9.94 |           -81.340 |          -8.13 |  3 days, 10:30:00 |     0     0     1     0 |
# |  XRP/USDT |         1 |         -14.13 |         -14.13 |           -99.555 |          -9.96 | 31 days, 11:30:00 |     0     0     1     0 |
# | NEAR/USDT |         1 |         -14.20 |         -14.20 |          -117.479 |         -11.75 | 10 days, 15:45:00 |     0     0     1     0 |
# | DOGE/USDT |         1 |         -15.93 |         -15.93 |          -119.122 |         -11.91 | 24 days, 21:30:00 |     0     0     1     0 |
# |     TOTAL |         6 |         -11.43 |         -68.56 |          -538.947 |         -53.89 |  12 days, 8:05:00 |     0     0     6     0 |
# ============================================================================ ENTER TAG STATS ============================================================================
# |                                    TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
# |----------------------------------------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
# |               1 Smooth Bull - Ref Bull |        73 |           4.10 |         299.58 |          1265.486 |         126.55 |  1 day, 3:23:00 |    62     8     3  84.9 |
# |              3 Smooth Range - Ref Bull |       197 |           1.94 |         381.62 |           940.448 |          94.04 | 2 days, 0:34:00 |   159    19    19  80.7 |
# |            28 Low XB Min < buy_offset1 |        58 |           2.72 |         157.60 |           752.815 |          75.28 |  1 day, 5:25:00 |    54     1     3  93.1 |
# |                8 Smooth Range - Ref Up |        22 |           2.32 |          51.15 |           201.082 |          20.11 | 2 days, 3:48:00 |    18     2     2  81.8 |
# |              16 Smooth Bull - Ref Down |        10 |           2.61 |          26.12 |           151.362 |          15.14 | 2 days, 2:44:00 |     7     3     0   100 |
# |                   7 Smooth Up - Ref Up |         5 |           3.89 |          19.43 |           108.707 |          10.87 |         3:24:00 |     5     0     0   100 |
# | 20 Smooth Down - Ref Down - Open < Min |        16 |           0.72 |          11.46 |           108.544 |          10.85 | 1 day, 17:03:00 |    12     3     1  75.0 |
# |                4Smooth Down - Ref Bull |         2 |           5.40 |          10.80 |            55.878 |           5.59 |        10:38:00 |     2     0     0   100 |
# |                 6 Smooth Bull - Ref Up |         1 |           9.65 |           9.65 |            18.134 |           1.81 |         5:45:00 |     1     0     0   100 |
# |                                  TOTAL |       384 |           2.52 |         967.41 |          3602.456 |         360.25 | 1 day, 16:41:00 |   320    36    28  83.3 |
# ======================================================= EXIT REASON STATS ========================================================
# |        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# | trailing_stop_loss |     282 |    260     0    22  92.2 |           2.96 |         834.71 |          3190.72  |         139.12 |
# |                roi |      66 |     30    36     0   100 |           2.68 |         177.01 |           795.239 |          29.5  |
# |         Pivot - XO |      18 |     18     0     0   100 |           0.5  |           9    |            59.991 |           1.5  |
# |          R2.5 - XO |       8 |      8     0     0   100 |           1.22 |           9.8  |            61.384 |           1.63 |
# |         force_exit |       6 |      0     0     6     0 |         -11.43 |         -68.56 |          -538.947 |         -11.43 |
# |            R3 - XO |       2 |      2     0     0   100 |           0.65 |           1.3  |             2.226 |           0.22 |
# |         R2.75 - XO |       2 |      2     0     0   100 |           2.08 |           4.15 |            31.84  |           0.69 |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2023-01-01 00:00:00 |
# | Backtesting to              | 2023-04-30 00:00:00 |
# | Max open trades             | 6                   |
# |                             |                     |
# | Total/Daily Avg Trades      | 384 / 3.23          |
# | Starting balance            | 1000 USDT           |
# | Final balance               | 4602.456 USDT       |
# | Absolute profit             | 3602.456 USDT       |
# | Total profit %              | 360.25%             |
# | CAGR %                      | 10702.89%           |
# | Sortino                     | 7.31                |
# | Sharpe                      | 11.86               |
# | Calmar                      | 238.31              |
# | Profit factor               | 1.92                |
# | Expectancy                  | -0.03               |
# | Trades per day              | 3.23                |
# | Avg. daily profit %         | 3.03%               |
# | Avg. stake amount           | 545.828 USDT        |
# | Total trade volume          | 209597.94 USDT      |
# |                             |                     |
# | Best Pair                   | LDO/USDT 146.30%    |
# | Worst Pair                  | LUNC/USDT -26.94%   |
# | Best trade                  | LDO/USDT 32.47%     |
# | Worst trade                 | DYDX/USDT -27.75%   |
# | Best day                    | 301.831 USDT        |
# | Worst day                   | -538.947 USDT       |
# | Days win/draw/lose          | 83 / 22 / 14        |
# | Avg. Duration Winners       | 13:49:00            |
# | Avg. Duration Loser         | 7 days, 3:32:00     |
# | Rejected Entry signals      | 463175              |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 1006.101 USDT       |
# | Max balance                 | 5141.404 USDT       |
# | Max % of account underwater | 24.27%              |
# | Absolute Drawdown (Account) | 24.27%              |
# | Absolute Drawdown           | 1229.443 USDT       |
# | Drawdown high               | 4065.839 USDT       |
# | Drawdown low                | 2836.396 USDT       |
# | Drawdown Start              | 2023-02-24 06:30:00 |
# | Drawdown End                | 2023-03-10 11:00:00 |
# | Market change               | 84.63%              |
# =====================================================

# 2023-05-04 16:18:58,554 - freqtrade.resolvers.iresolver - WARNING - Could not import /home/core-rho/freqtrade/user_data/strategies/NASOSv5.py due to 'name 'TrailingBuySellStrat' is not defined'
# 2023-05-04 16:18:58,556 - freqtrade.resolvers.iresolver - WARNING - Could not import /home/core-rho/freqtrade/user_data/strategies/EWOGPT.py due to 'name 'DataFrame' is not defined'
# 2023-05-04 16:18:58,560 - NFIX - INFO - pandas_ta successfully imported
# 2023-05-04 16:18:58,582 - freqtrade.optimize.hyperopt_tools - INFO - Dumping parameters to /home/core-rho/freqtrade/user_data/strategies/CTIBS.json

# Epoch details:

#   1156/1500:    384 trades. 320/36/28 Wins/Draws/Losses. Avg profit   2.52%. Median profit   4.17%. Total profit 3602.45640151 USDT ( 360.25%). Avg duration 1 day, 16:41:00 min. Objective: -3602.45640


#     # Buy hyperspace params:
#     buy_params = {
#         "buy01": True,
#         "buy02": False,
#         "buy03": True,
#         "buy04": True,
#         "buy05": False,
#         "buy06": True,
#         "buy07": True,
#         "buy08": True,
#         "buy09": False,
#         "buy10": True,
#         "buy11": False,
#         "buy12": False,
#         "buy13": False,
#         "buy14": False,
#         "buy15": False,
#         "buy16": True,
#         "buy17": True,
#         "buy18": False,
#         "buy19": False,
#         "buy20": True,
#         "buy21": True,
#         "buy22": False,
#         "buy23": False,
#         "buy24": False,
#         "buy25": False,
#         "buy26": False,
#         "buy27": False,
#         "buy28": True,
#         "buy29": False,
#         "buy30": False,
#         "buy_change": 1.9,
#         "buy_offset1": 0.99,
#         "buy_offset2": 0.94,
#         "from_bear": -11.1,
#         "from_bull": -3.4,
#         "from_down": -5.5,
#         "from_ranging": -2.0,
#         "from_up": -2.8,
#         "max_length": 48,
#         "ref_bear": -0.092,
#         "ref_bull": 0.076,
#         "ref_down": -0.057,
#         "ref_up": 0.06,
#         "reference_ma_length": 189,
#         "smooth_bear": -0.093,
#         "smooth_bull": 0.09,
#         "smooth_down": -0.075,
#         "smooth_up": 0.069,
#         "smoothing_length": 17,
#     }

#     # Sell hyperspace params:
#     sell_params = {
#         "filterlength": 30,  # value loaded from strategy
#         "sell01": False,  # value loaded from strategy
#         "sell02": True,  # value loaded from strategy
#         "sell03": True,  # value loaded from strategy
#         "sell04": True,  # value loaded from strategy
#         "sell05": False,  # value loaded from strategy
#         "sell06": False,  # value loaded from strategy
#         "sell07": True,  # value loaded from strategy
#         "sell08": True,  # value loaded from strategy
#         "sell09": False,  # value loaded from strategy
#         "sell10": True,  # value loaded from strategy
#         "sell_change": 1.7,  # value loaded from strategy
#         "sell_offset1": 1.04,  # value loaded from strategy
#         "sell_slope": 0.0,  # value loaded from strategy
#         "ts0": 0.008,  # value loaded from strategy
#         "ts1": 0.01,  # value loaded from strategy
#         "ts2": 0.022,  # value loaded from strategy
#         "ts3": 0.038,  # value loaded from strategy
#         "ts4": 0.03,  # value loaded from strategy
#         "ts5": 0.04,  # value loaded from strategy
#         "tsl_target0": 0.044,  # value loaded from strategy
#         "tsl_target1": 0.065,  # value loaded from strategy
#         "tsl_target2": 0.097,  # value loaded from strategy
#         "tsl_target3": 0.12,  # value loaded from strategy
#         "tsl_target4": 0.16,  # value loaded from strategy
#         "tsl_target5": 0.3,  # value loaded from strategy
#     }

#     # Protection hyperspace params:
#     protection_params = {
#         "cooldown_lookback": 48,  # value loaded from strategy
#         "lowprofit_only_per_pair": True,  # value loaded from strategy
#         "lowprofit_protection_lookback": 5,  # value loaded from strategy
#         "lowprofit_required_profit": -0.704,  # value loaded from strategy
#         "lowprofit_stop_duration": 100,  # value loaded from strategy
#         "lowprofit_trade_limit": 2,  # value loaded from strategy
#         "maxdrawdown_allowed_drawdown": 0.045,  # value loaded from strategy
#         "maxdrawdown_protection_lookback": 3,  # value loaded from strategy
#         "maxdrawdown_stop_duration": 15,  # value loaded from strategy
#         "maxdrawdown_trade_limit": 5,  # value loaded from strategy
#         "stop_duration": 19,  # value loaded from strategy
#         "stop_protection_only_per_pair": True,  # value loaded from strategy
#         "stop_protection_only_per_side": True,  # value loaded from strategy
#         "stop_protection_required_profit": -0.126,  # value loaded from strategy
#         "stop_protection_trade_limit": 9,  # value loaded from strategy
#         "use_lowprofit_protection": True,  # value loaded from strategy
#         "use_maxdrawdown_protection": False,  # value loaded from strategy
#         "use_stop_protection": False,  # value loaded from strategy
#     }

#     # ROI table:  # value loaded from strategy
#     minimal_roi = {
#         "0": 0.325,
#         "899": 0.119,
#         "1993": 0.049,
#         "4089": 0
#     }

#     # Stoploss:
#     stoploss = -0.28  # value loaded from strategy

#     # Trailing stop:
#     trailing_stop = True  # value loaded from strategy
#     trailing_stop_positive = None  # value loaded from strategy
#     trailing_stop_positive_offset = 0.0  # value loaded from strategy
#     trailing_only_offset_is_reached = False  # value loaded from strategy
    

#     # Max Open Trades:
#     max_open_trades = 6  # value loaded from strategy

# ============================================================= BACKTESTING REPORT =============================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |  OPUL/USDT |        74 |           2.44 |         180.60 |          5715.068 |         571.51 |   1 day, 6:30:00 |    66     0     8  89.2 |
# |  ORAI/USDT |       135 |           1.71 |         231.32 |          5555.068 |         555.51 |   1 day, 8:03:00 |   118     0    17  87.4 |
# |  ROSE/USDT |        74 |           3.29 |         243.33 |          5523.756 |         552.38 |  1 day, 17:36:00 |    69     0     5  93.2 |
# |  AGIX/USDT |        71 |           2.62 |         185.67 |          4052.021 |         405.20 |  1 day, 21:16:00 |    65     0     6  91.5 |
# |   EWT/USDT |       105 |           1.59 |         166.67 |          4021.753 |         402.18 |  2 days, 1:28:00 |    93     0    12  88.6 |
# |   QNT/USDT |        85 |           1.57 |         133.39 |          3675.504 |         367.55 |  1 day, 13:31:00 |    70     0    15  82.4 |
# |  SAND/USDT |        41 |           2.92 |         119.65 |          3119.027 |         311.90 |   1 day, 7:53:00 |    39     0     2  95.1 |
# | THETA/USDT |        29 |           3.99 |         115.65 |          2759.294 |         275.93 |  1 day, 16:17:00 |    29     0     0   100 |
# |  DYDX/USDT |        23 |           3.51 |          80.66 |          2399.571 |         239.96 |  2 days, 7:53:00 |    22     0     1  95.7 |
# | OCEAN/USDT |        21 |           3.95 |          82.96 |          2130.780 |         213.08 |  3 days, 8:42:00 |    21     0     0   100 |
# |  SCRT/USDT |        17 |           5.46 |          92.80 |          2090.395 |         209.04 |  2 days, 2:13:00 |    17     0     0   100 |
# | GALAX/USDT |        24 |           2.16 |          51.80 |          2053.474 |         205.35 |  2 days, 5:01:00 |    22     0     2  91.7 |
# |   LDO/USDT |        36 |           2.79 |         100.43 |          1980.960 |         198.10 |  3 days, 5:13:00 |    34     0     2  94.4 |
# |   INJ/USDT |        27 |           2.40 |          64.83 |          1869.676 |         186.97 |  2 days, 4:46:00 |    24     0     3  88.9 |
# |  AVAX/USDT |        61 |           1.43 |          87.11 |          1806.670 |         180.67 |  1 day, 20:42:00 |    52     0     9  85.2 |
# |   SOL/USDT |        18 |           3.77 |          67.94 |          1698.354 |         169.84 |         14:32:00 |    17     0     1  94.4 |
# |  RNDR/USDT |        88 |           0.53 |          46.33 |          1659.739 |         165.97 |  2 days, 0:27:00 |    73     0    15  83.0 |
# |   VET/USDT |       124 |           2.46 |         305.00 |          1486.684 |         148.67 |  1 day, 11:37:00 |   114     0    10  91.9 |
# |   AKT/USDT |        31 |           2.97 |          92.08 |          1422.603 |         142.26 |   1 day, 3:25:00 |    26     0     5  83.9 |
# |   KSM/USDT |       132 |           2.63 |         347.10 |          1419.068 |         141.91 |  1 day, 10:54:00 |   118     0    14  89.4 |
# |   ETH/USDT |        68 |           1.51 |         102.72 |          1322.055 |         132.21 |  2 days, 1:44:00 |    56     0    12  82.4 |
# |   ENJ/USDT |        20 |           1.66 |          33.23 |          1245.821 |         124.58 | 2 days, 23:34:00 |    18     0     2  90.0 |
# |   CRV/USDT |        45 |           0.97 |          43.43 |           924.294 |          92.43 |  1 day, 21:46:00 |    40     0     5  88.9 |
# |   APE/USDT |        16 |           1.93 |          30.84 |           861.966 |          86.20 | 2 days, 12:36:00 |    15     0     1  93.8 |
# |   FLR/USDT |         4 |           6.08 |          24.32 |           855.297 |          85.53 | 5 days, 11:22:00 |     4     0     0   100 |
# |   FET/USDT |         2 |           6.82 |          13.64 |           582.818 |          58.28 |   1 day, 2:38:00 |     2     0     0   100 |
# |  EGLD/USDT |        15 |           0.72 |          10.81 |           539.496 |          53.95 | 2 days, 18:02:00 |    13     0     2  86.7 |
# |  OSMO/USDT |         4 |           3.39 |          13.55 |           516.823 |          51.68 |   1 day, 2:08:00 |     4     0     0   100 |
# |   TRX/USDT |        78 |           0.55 |          42.80 |           396.875 |          39.69 |  3 days, 6:04:00 |    70     0     8  89.7 |
# |  CSPR/USDT |        18 |           0.10 |           1.74 |           361.325 |          36.13 |  1 day, 13:02:00 |    14     0     4  77.8 |
# |   XDC/USDT |        37 |           0.35 |          13.00 |           352.431 |          35.24 | 2 days, 12:54:00 |    31     0     6  83.8 |
# |   BTC/USDT |        41 |           1.38 |          56.40 |           320.135 |          32.01 | 3 days, 18:18:00 |    35     0     6  85.4 |
# |  NEAR/USDT |        30 |          -0.73 |         -21.98 |           164.149 |          16.41 |  4 days, 7:22:00 |    25     0     5  83.3 |
# |   ADA/USDT |       112 |           1.42 |         158.88 |            92.352 |           9.24 |  1 day, 19:16:00 |    89     0    23  79.5 |
# |  KAVA/USDT |        27 |           0.24 |           6.46 |            56.103 |           5.61 | 2 days, 23:17:00 |    24     0     3  88.9 |
# |  ALGO/USDT |       141 |           0.52 |          73.09 |             6.494 |           0.65 | 2 days, 13:07:00 |   123     0    18  87.2 |
# |   ARB/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |   UNI/USDT |        49 |          -1.27 |         -62.04 |           -63.883 |          -6.39 | 2 days, 10:39:00 |    41     0     8  83.7 |
# |  DASH/USDT |        26 |          -0.62 |         -16.16 |          -145.830 |         -14.58 |  1 day, 18:00:00 |    22     0     4  84.6 |
# |   ZEC/USDT |        73 |           1.81 |         132.17 |          -175.020 |         -17.50 |  2 days, 6:35:00 |    66     0     7  90.4 |
# |   EOS/USDT |        49 |           0.80 |          39.23 |          -213.580 |         -21.36 |  3 days, 9:42:00 |    45     0     4  91.8 |
# |  IOTA/USDT |         7 |          -0.66 |          -4.65 |          -523.362 |         -52.34 |  5 days, 4:47:00 |     5     0     2  71.4 |
# |   XLM/USDT |        88 |          -0.26 |         -23.15 |          -581.949 |         -58.19 |  1 day, 22:03:00 |    77     0    11  87.5 |
# |    OP/USDT |        27 |          -0.42 |         -11.33 |          -631.289 |         -63.13 | 2 days, 15:55:00 |    23     0     4  85.2 |
# |   DOT/USDT |        35 |          -0.12 |          -4.12 |          -715.886 |         -71.59 |  4 days, 4:14:00 |    29     0     6  82.9 |
# |   FIL/USDT |        47 |           0.39 |          18.56 |          -750.934 |         -75.09 |  1 day, 18:17:00 |    40     0     7  85.1 |
# |   GRT/USDT |        63 |           0.06 |           3.55 |          -766.297 |         -76.63 |  1 day, 18:40:00 |    54     0     9  85.7 |
# |   IMX/USDT |        30 |           0.30 |           9.12 |          -834.521 |         -83.45 |  1 day, 23:48:00 |    26     0     4  86.7 |
# |   APT/USDT |        10 |          -2.37 |         -23.74 |          -936.892 |         -93.69 |  4 days, 0:57:00 |     8     0     2  80.0 |
# |  LINK/USDT |        44 |          -0.81 |         -35.75 |          -994.326 |         -99.43 |   1 day, 3:44:00 |    38     0     6  86.4 |
# |  ANKR/USDT |        26 |           0.08 |           1.99 |         -1364.234 |        -136.42 |  1 day, 20:42:00 |    22     0     4  84.6 |
# |   ETC/USDT |        94 |           0.54 |          50.45 |         -1442.073 |        -144.21 |  2 days, 5:46:00 |    79     0    15  84.0 |
# |  KLAY/USDT |        11 |          -5.62 |         -61.86 |         -1478.647 |        -147.86 | 7 days, 14:29:00 |     7     0     4  63.6 |
# |  ATOM/USDT |       190 |          -0.19 |         -35.40 |         -1519.221 |        -151.92 |  1 day, 19:55:00 |   147     0    43  77.4 |
# |  HBAR/USDT |        18 |          -2.26 |         -40.71 |         -1666.849 |        -166.68 | 3 days, 21:18:00 |    14     0     4  77.8 |
# |  LUNC/USDT |        28 |          -3.02 |         -84.63 |         -1847.354 |        -184.74 | 2 days, 21:45:00 |    20     0     8  71.4 |
# |   XTZ/USDT |        88 |           0.95 |          83.80 |         -2607.238 |        -260.72 |  3 days, 6:01:00 |    78     0    10  88.6 |
# |   XRP/USDT |       141 |           0.16 |          21.89 |         -2686.901 |        -268.69 |  2 days, 5:05:00 |   109     0    32  77.3 |
# |   CRO/USDT |        26 |           0.48 |          12.40 |         -2936.951 |        -293.70 |  2 days, 4:55:00 |    21     0     5  80.8 |
# |  DOGE/USDT |        44 |          -3.16 |        -138.84 |         -4832.220 |        -483.22 |  3 days, 1:19:00 |    34     0    10  77.3 |
# | MATIC/USDT |        31 |          -5.21 |        -161.60 |         -5139.417 |        -513.94 |  3 days, 1:10:00 |    21     0    10  67.7 |
# | JASMY/USDT |        28 |          -6.85 |        -191.81 |         -5237.158 |        -523.72 | 4 days, 16:00:00 |    19     0     9  67.9 |
# |      TOTAL |      3147 |           0.91 |        2875.60 |         24945.863 |        2494.59 |  2 days, 4:42:00 |  2697     0   450  85.7 |
# ========================================================== LEFT OPEN TRADES REPORT ===========================================================
# |      Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |      Avg Duration |   Win  Draw  Loss  Win% |
# |-----------+-----------+----------------+----------------+-------------------+----------------+-------------------+-------------------------|
# | EGLD/USDT |         1 |          -4.18 |          -4.18 |          -199.943 |         -19.99 |    1 day, 9:45:00 |     0     0     1     0 |
# | AVAX/USDT |         1 |          -8.86 |          -8.86 |          -434.655 |         -43.47 | 10 days, 15:30:00 |     0     0     1     0 |
# | KLAY/USDT |         1 |         -11.53 |         -11.53 |          -546.770 |         -54.68 | 12 days, 21:15:00 |     0     0     1     0 |
# | DOGE/USDT |         1 |         -12.81 |         -12.81 |          -585.414 |         -58.54 |  24 days, 3:15:00 |     0     0     1     0 |
# | AGIX/USDT |         1 |         -12.38 |         -12.38 |          -607.465 |         -60.75 | 10 days, 15:30:00 |     0     0     1     0 |
# |  AKT/USDT |         1 |         -16.16 |         -16.16 |          -800.262 |         -80.03 |  10 days, 6:30:00 |     0     0     1     0 |
# |     TOTAL |         6 |         -10.99 |         -65.92 |         -3174.508 |        -317.45 | 11 days, 15:58:00 |     0     0     6     0 |
# ============================================================================ ENTER TAG STATS ============================================================================
# |                                   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |---------------------------------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# | 29 XO above buy_offset3 | Down - Bear |      1277 |           0.89 |        1139.61 |         14459.914 |        1445.99 |  2 days, 9:43:00 |  1136     0   141  89.0 |
# |            14 Smooth Down - Ref Range |       279 |           0.95 |         265.62 |          5490.037 |         549.00 | 2 days, 20:43:00 |   251     0    28  90.0 |
# |                  7 Smooth Up - Ref Up |        66 |           3.26 |         215.35 |          5485.316 |         548.53 |  1 day, 16:28:00 |    63     0     3  95.5 |
# |                9 Smooth Down - Ref Up |       121 |           1.82 |         220.61 |          4641.163 |         464.12 |  2 days, 4:00:00 |   113     0     8  93.4 |
# |                6 Smooth Bull - Ref Up |       132 |           1.99 |         262.63 |          3943.323 |         394.33 |  2 days, 1:45:00 |   119     0    13  90.2 |
# |           30 Low XB Min < buy_offset2 |       243 |           1.77 |         429.96 |          3331.014 |         333.10 |          6:34:00 |   151     0    92  62.1 |
# |               4Smooth Down - Ref Bull |        79 |          -0.07 |          -5.90 |          -643.499 |         -64.35 |  1 day, 21:31:00 |    67     0    12  84.8 |
# |           28 Low XB Min < buy_offset1 |       404 |           0.95 |         383.66 |         -2746.185 |        -274.62 |  1 day, 14:15:00 |   332     0    72  82.2 |
# |            11 Smooth Bull - Ref Range |       473 |           0.23 |         108.81 |         -3871.968 |        -387.20 | 2 days, 21:59:00 |   407     0    66  86.0 |
# |             16 Smooth Bull - Ref Down |        73 |          -1.98 |        -144.73 |         -5143.251 |        -514.33 |  2 days, 2:37:00 |    58     0    15  79.5 |
# |                                 TOTAL |      3147 |           0.91 |        2875.60 |         24945.863 |        2494.59 |  2 days, 4:42:00 |  2697     0   450  85.7 |
# ======================================================= EXIT REASON STATS ========================================================
# |        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# | trailing_stop_loss |    2465 |   2041     0   424  82.8 |           0.72 |        1767.39 |         13389.9   |         294.56 |
# |         Pivot - XO |     188 |    188     0     0   100 |           0.76 |         142.38 |          2307.61  |          23.73 |
# |            R2 - XO |     129 |    129     0     0   100 |           0.84 |         108.62 |          1908.76  |          18.1  |
# |                roi |     125 |    125     0     0   100 |          10.06 |        1257.98 |         20757.8   |         209.66 |
# |          R2.5 - XO |     111 |    111     0     0   100 |           0.97 |         107.15 |          1664.67  |          17.86 |
# |         R2.25 - XO |      87 |     87     0     0   100 |           1.06 |          92.43 |          1078.97  |          15.41 |
# |          stop_loss |      20 |      0     0    20     0 |         -28.14 |        -562.83 |        -13160.1   |         -93.81 |
# |            R5 - XO |      16 |     16     0     0   100 |           1.78 |          28.41 |           172.787 |           4.73 |
# |         force_exit |       6 |      0     0     6     0 |         -10.99 |         -65.92 |         -3174.51  |         -10.99 |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2019-03-30 01:00:00 |
# | Backtesting to              | 2023-04-30 00:00:00 |
# | Max open trades             | 6                   |
# |                             |                     |
# | Total/Daily Avg Trades      | 3147 / 2.11         |
# | Starting balance            | 1000 USDT           |
# | Final balance               | 25945.863 USDT      |
# | Absolute profit             | 24945.863 USDT      |
# | Total profit %              | 2494.59%            |
# | CAGR %                      | 121.90%             |
# | Sortino                     | 0.77                |
# | Sharpe                      | 1.32                |
# | Calmar                      | 39.52               |
# | Profit factor               | 1.13                |
# | Expectancy                  | 0.02                |
# | Trades per day              | 2.11                |
# | Avg. daily profit %         | 1.67%               |
# | Avg. stake amount           | 1901.128 USDT       |
# | Total trade volume          | 5982849.002 USDT    |
# |                             |                     |
# | Best Pair                   | KSM/USDT 347.10%    |
# | Worst Pair                  | JASMY/USDT -191.81% |
# | Best trade                  | ADA/USDT 26.20%     |
# | Worst trade                 | JASMY/USDT -68.18%  |
# | Best day                    | 1665.999 USDT       |
# | Worst day                   | -18236.68 USDT      |
# | Days win/draw/lose          | 901 / 430 / 158     |
# | Avg. Duration Winners       | 1 day, 18:11:00     |
# | Avg. Duration Loser         | 4 days, 19:44:00    |
# | Rejected Entry signals      | 2633571             |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 945.791 USDT        |
# | Max balance                 | 36573.418 USDT      |
# | Max % of account underwater | 80.89%              |
# | Absolute Drawdown (Account) | 80.89%              |
# | Absolute Drawdown           | 29583.112 USDT      |
# | Drawdown high               | 35573.418 USDT      |
# | Drawdown low                | 5990.306 USDT       |
# | Drawdown Start              | 2021-12-03 09:30:00 |
# | Drawdown End                | 2022-06-14 01:30:00 |
# | Market change               | 55.41%              |
# =====================================================

# Backtested 2019-03-30 01:00:00 -> 2023-04-30 00:00:00 | Max open trades : 6
# ========================================================================== STRATEGY SUMMARY ==========================================================================
# |   Strategy |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |               Drawdown |
# |------------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------+------------------------|
# |     CTIBS2 |      3147 |           0.91 |        2875.60 |         24945.863 |        2494.59 | 2 days, 4:42:00 |  2697     0   450  85.7 | 29583.112 USDT  80.89% |
# ======================================================================================================================================================================

# Result for strategy CTIBS2 5minute
# ============================================================== BACKTESTING REPORT =============================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |      Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+-------------------+-------------------------|
# |  OPUL/USDT |        49 |           4.40 |         215.48 |           878.489 |          87.85 |           7:56:00 |    48     0     1  98.0 |
# |   LDO/USDT |        20 |           3.18 |          63.56 |           211.134 |          21.11 |          20:54:00 |    20     0     0   100 |
# |   APT/USDT |        18 |           2.93 |          52.67 |           174.551 |          17.46 |          13:48:00 |    18     0     0   100 |
# |    OP/USDT |        16 |           2.64 |          42.29 |           157.208 |          15.72 |          10:22:00 |    16     0     0   100 |
# |   AKT/USDT |        13 |           3.37 |          43.84 |           142.432 |          14.24 |          13:02:00 |    13     0     0   100 |
# |  CSPR/USDT |        10 |           3.00 |          29.96 |           128.471 |          12.85 |  3 days, 23:36:00 |    10     0     0   100 |
# |   IMX/USDT |        10 |           2.89 |          28.87 |           120.659 |          12.07 |           6:14:00 |    10     0     0   100 |
# |   FLR/USDT |         9 |           3.97 |          35.71 |           114.495 |          11.45 |   1 day, 22:11:00 |     9     0     0   100 |
# | GALAX/USDT |        13 |           2.99 |          38.83 |           109.016 |          10.90 |          19:57:00 |    13     0     0   100 |
# | OCEAN/USDT |        10 |           3.02 |          30.23 |            96.229 |           9.62 |          10:47:00 |    10     0     0   100 |
# |  AGIX/USDT |        46 |           1.66 |          76.38 |            95.470 |           9.55 |          17:57:00 |    43     0     3  93.5 |
# |  RNDR/USDT |        27 |           1.17 |          31.72 |            87.993 |           8.80 |   1 day, 13:25:00 |    25     0     2  92.6 |
# |  KLAY/USDT |         5 |           3.07 |          15.37 |            72.180 |           7.22 |          11:13:00 |     5     0     0   100 |
# |   APE/USDT |         6 |           2.72 |          16.35 |            54.749 |           5.47 |           9:04:00 |     6     0     0   100 |
# |  ROSE/USDT |         2 |           4.35 |           8.70 |            38.186 |           3.82 |           8:38:00 |     2     0     0   100 |
# |  OSMO/USDT |         3 |           2.79 |           8.38 |            33.566 |           3.36 |    1 day, 1:08:00 |     3     0     0   100 |
# |  ORAI/USDT |        26 |           1.31 |          34.11 |            32.027 |           3.20 |          15:42:00 |    24     0     2  92.3 |
# |   FET/USDT |         3 |           2.48 |           7.43 |            30.945 |           3.09 |          10:25:00 |     3     0     0   100 |
# |   ENJ/USDT |         3 |           2.97 |           8.90 |            27.859 |           2.79 |           6:27:00 |     3     0     0   100 |
# |  SCRT/USDT |         9 |           0.80 |           7.16 |            27.483 |           2.75 |          12:44:00 |     8     0     1  88.9 |
# |  HBAR/USDT |         4 |           1.40 |           5.59 |            18.702 |           1.87 |          10:01:00 |     4     0     0   100 |
# |  NEAR/USDT |         4 |           1.61 |           6.46 |            17.723 |           1.77 |          15:08:00 |     4     0     0   100 |
# |   INJ/USDT |        14 |           0.66 |           9.22 |            17.198 |           1.72 |   1 day, 19:08:00 |    12     0     2  85.7 |
# | MATIC/USDT |         1 |           4.03 |           4.03 |            14.849 |           1.48 |           3:20:00 |     1     0     0   100 |
# | THETA/USDT |         1 |           3.95 |           3.95 |            12.926 |           1.29 |           3:15:00 |     1     0     0   100 |
# |  LUNC/USDT |         3 |           1.98 |           5.94 |            11.641 |           1.16 |          10:23:00 |     3     0     0   100 |
# |  AVAX/USDT |         2 |           2.14 |           4.29 |             9.763 |           0.98 |          13:00:00 |     2     0     0   100 |
# |   SOL/USDT |         2 |           2.63 |           5.27 |             9.035 |           0.90 |           4:55:00 |     2     0     0   100 |
# |   XRP/USDT |         1 |           1.58 |           1.58 |             7.522 |           0.75 |           6:45:00 |     1     0     0   100 |
# |   ETC/USDT |         1 |           1.39 |           1.39 |             4.932 |           0.49 |          10:45:00 |     1     0     0   100 |
# |  SAND/USDT |         2 |           0.67 |           1.34 |             3.028 |           0.30 |   3 days, 9:55:00 |     2     0     0   100 |
# |   KSM/USDT |         1 |           1.01 |           1.01 |             2.235 |           0.22 |           3:50:00 |     1     0     0   100 |
# |   QNT/USDT |         1 |           0.36 |           0.36 |             1.783 |           0.18 |           6:20:00 |     1     0     0   100 |
# |  DASH/USDT |         1 |           0.23 |           0.23 |             0.741 |           0.07 |           6:10:00 |     1     0     0   100 |
# |  ATOM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |  IOTA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |   ADA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |   BTC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |   ETH/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |   XDC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |  LINK/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |   DOT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |  ALGO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |   XLM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |   TRX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |   UNI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |   VET/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |   ZEC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |   EOS/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |   CRO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |              0:00 |     0     0     0     0 |
# |  DYDX/USDT |        11 |          -0.17 |          -1.83 |           -15.495 |          -1.55 |   1 day, 12:37:00 |     9     0     2  81.8 |
# |  KAVA/USDT |         5 |          -0.45 |          -2.25 |           -17.441 |          -1.74 |  2 days, 20:40:00 |     4     0     1  80.0 |
# |  DOGE/USDT |         2 |          -1.73 |          -3.47 |           -18.352 |          -1.84 |   5 days, 2:42:00 |     1     0     1  50.0 |
# |  EGLD/USDT |         1 |          -6.77 |          -6.77 |           -34.189 |          -3.42 |  2 days, 10:05:00 |     0     0     1     0 |
# | JASMY/USDT |         7 |          -1.68 |         -11.78 |           -53.467 |          -5.35 |  3 days, 16:04:00 |     6     0     1  85.7 |
# |  ANKR/USDT |         7 |          -1.08 |          -7.55 |           -86.253 |          -8.63 |    1 day, 6:46:00 |     6     0     1  85.7 |
# |   ARB/USDT |         3 |          -5.73 |         -17.20 |           -88.918 |          -8.89 |  7 days, 14:38:00 |     2     0     1  66.7 |
# |   CRV/USDT |         4 |          -4.31 |         -17.23 |           -97.983 |          -9.80 |   4 days, 6:26:00 |     3     0     1  75.0 |
# |   EWT/USDT |         6 |          -2.58 |         -15.51 |           -98.835 |          -9.88 |   6 days, 5:58:00 |     5     0     1  83.3 |
# |   FIL/USDT |         2 |         -11.72 |         -23.44 |          -108.050 |         -10.80 | 22 days, 13:55:00 |     1     0     1  50.0 |
# |   XTZ/USDT |         2 |         -13.80 |         -27.60 |          -137.650 |         -13.77 |   7 days, 2:25:00 |     1     0     1  50.0 |
# |   GRT/USDT |         3 |          -8.36 |         -25.08 |          -140.912 |         -14.09 |   1 day, 21:55:00 |     2     0     1  66.7 |
# |      TOTAL |       389 |           1.77 |         686.87 |          1867.674 |         186.77 |    1 day, 7:03:00 |   365     0    24  93.8 |
# ========================================================== LEFT OPEN TRADES REPORT ===========================================================
# |      Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |      Avg Duration |   Win  Draw  Loss  Win% |
# |-----------+-----------+----------------+----------------+-------------------+----------------+-------------------+-------------------------|
# | DOGE/USDT |         1 |          -4.43 |          -4.43 |           -22.791 |          -2.28 |   9 days, 7:20:00 |     0     0     1     0 |
# |  INJ/USDT |         1 |          -4.89 |          -4.89 |           -24.661 |          -2.47 |   1 day, 10:00:00 |     0     0     1     0 |
# | KAVA/USDT |         1 |          -5.65 |          -5.65 |           -29.046 |          -2.90 | 10 days, 15:35:00 |     0     0     1     0 |
# | EGLD/USDT |         1 |          -6.77 |          -6.77 |           -34.189 |          -3.42 |  2 days, 10:05:00 |     0     0     1     0 |
# |  ARB/USDT |         1 |         -18.76 |         -18.76 |           -96.457 |          -9.65 | 10 days, 22:10:00 |     0     0     1     0 |
# |     TOTAL |         5 |          -8.10 |         -40.50 |          -207.144 |         -20.71 |  6 days, 22:38:00 |     0     0     5     0 |
# ============================================================================ ENTER TAG STATS ============================================================================
# |                                   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |---------------------------------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# | 29 XO above buy_offset3 | Down - Bear |       203 |           1.99 |         404.39 |          1321.044 |         132.10 |   1 day, 4:56:00 |   193     0    10  95.1 |
# |           28 Low XB Min < buy_offset1 |        34 |           2.26 |          76.71 |           202.075 |          20.21 |         16:48:00 |    30     0     4  88.2 |
# |               4Smooth Down - Ref Bull |         8 |           5.40 |          43.23 |           130.645 |          13.06 |         15:26:00 |     8     0     0   100 |
# |            14 Smooth Down - Ref Range |        52 |           1.11 |          57.66 |            97.025 |           9.70 |  1 day, 22:55:00 |    49     0     3  94.2 |
# |                6 Smooth Bull - Ref Up |         7 |           3.72 |          26.07 |            71.843 |           7.18 |         14:49:00 |     7     0     0   100 |
# |           30 Low XB Min < buy_offset2 |         3 |           4.31 |          12.92 |            48.179 |           4.82 |          0:28:00 |     3     0     0   100 |
# |            11 Smooth Bull - Ref Range |        52 |           0.82 |          42.81 |            28.470 |           2.85 |   1 day, 4:12:00 |    47     0     5  90.4 |
# |                9 Smooth Down - Ref Up |        19 |           1.51 |          28.75 |            16.018 |           1.60 |         19:36:00 |    18     0     1  94.7 |
# |             16 Smooth Bull - Ref Down |         1 |           0.23 |           0.23 |             0.741 |           0.07 |          6:10:00 |     1     0     0   100 |
# |                  7 Smooth Up - Ref Up |        10 |          -0.59 |          -5.90 |           -48.365 |          -4.84 | 4 days, 16:18:00 |     9     0     1  90.0 |
# |                                 TOTAL |       389 |           1.77 |         686.87 |          1867.674 |         186.77 |   1 day, 7:03:00 |   365     0    24  93.8 |
# ======================================================= EXIT REASON STATS ========================================================
# |        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# | trailing_stop_loss |     232 |    213     0    19  91.8 |           2.2  |         511.19 |          1294     |          85.2  |
# |         Pivot - XO |      59 |     59     0     0   100 |           0.83 |          48.69 |           173.512 |           8.11 |
# |            R2 - XO |      34 |     34     0     0   100 |           1.15 |          39.11 |           147.361 |           6.52 |
# |         R2.25 - XO |      27 |     27     0     0   100 |           0.94 |          25.38 |            92.888 |           4.23 |
# |          R2.5 - XO |      23 |     23     0     0   100 |           1.18 |          27.1  |           103.307 |           4.52 |
# |                roi |       7 |      7     0     0   100 |           9.99 |          69.93 |           243.224 |          11.66 |
# |         force_exit |       5 |      0     0     5     0 |          -8.1  |         -40.5  |          -207.144 |          -6.75 |
# |            R5 - XO |       2 |      2     0     0   100 |           2.99 |           5.97 |            20.528 |           1    |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2023-01-01 00:00:00 |
# | Backtesting to              | 2023-04-30 00:00:00 |
# | Max open trades             | 6                   |
# |                             |                     |
# | Total/Daily Avg Trades      | 389 / 3.27          |
# | Starting balance            | 1000 USDT           |
# | Final balance               | 2867.674 USDT       |
# | Absolute profit             | 1867.674 USDT       |
# | Total profit %              | 186.77%             |
# | CAGR %                      | 2431.32%            |
# | Sortino                     | 5.62                |
# | Sharpe                      | 9.81                |
# | Calmar                      | 143.78              |
# | Profit factor               | 1.80                |
# | Expectancy                  | 0.05                |
# | Trades per day              | 3.27                |
# | Avg. daily profit %         | 1.57%               |
# | Avg. stake amount           | 370.242 USDT        |
# | Total trade volume          | 144024.147 USDT     |
# |                             |                     |
# | Best Pair                   | OPUL/USDT 215.48%   |
# | Worst Pair                  | XTZ/USDT -27.60%    |
# | Best trade                  | FLR/USDT 9.99%      |
# | Worst trade                 | GRT/USDT -30.21%    |
# | Best day                    | 131.958 USDT        |
# | Worst day                   | -290.598 USDT       |
# | Days win/draw/lose          | 84 / 21 / 15        |
# | Avg. Duration Winners       | 18:47:00            |
# | Avg. Duration Loser         | 9 days, 1:49:00     |
# | Rejected Entry signals      | 707473              |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 998.351 USDT        |
# | Max balance                 | 3163.647 USDT       |
# | Max % of account underwater | 20.85%              |
# | Absolute Drawdown (Account) | 20.85%              |
# | Absolute Drawdown           | 654.567 USDT        |
# | Drawdown high               | 2138.665 USDT       |
# | Drawdown low                | 1484.099 USDT       |
# | Drawdown Start              | 2023-02-08 00:45:00 |
# | Drawdown End                | 2023-03-10 10:25:00 |
# | Market change               | 81.51%              |
# =====================================================

# Backtested 2023-01-01 00:00:00 -> 2023-04-30 00:00:00 | Max open trades : 6
# ========================================================================= STRATEGY SUMMARY ========================================================================
# |   Strategy |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |             Drawdown |
# |------------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------+----------------------|
# |     CTIBS2 |       389 |           1.77 |         686.87 |          1867.674 |         186.77 | 1 day, 7:03:00 |   365     0    24  93.8 | 654.567 USDT  20.85% |
# ===================================================================================================================================================================
