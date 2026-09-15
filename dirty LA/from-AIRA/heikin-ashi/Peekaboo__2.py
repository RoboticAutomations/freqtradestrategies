
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from typing import Optional
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter
import technical.indicators as ftt
import math
import logging
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)


class Peekaboo(IStrategy):
    INTERFACE_VERSION = 2


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
        "0": 0.195,
        "39": 0.10600000000000001,
        "91": 0.04,
        "210": 0
    }

    # Stoploss:
    stoploss = -0.99

    # Trailing stop:
    use_custom_stoploss = True


    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    ## Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    startup_candle_count = 100
    process_only_new_candles = True
    
    # DCA
    position_adjustment_enable = True

    # Custom Entry
    last_entry_price = None

    # Hyper-opt parameters
    base_nb_candles_buy = IntParameter(60, 80, default=72, space='buy', optimize=True)
    bull = IntParameter(60, 80, default=72, space='buy', optimize=True)
    bear = IntParameter(490, 510, default=500, space='buy', optimize=True)
    blah = IntParameter(200, 225, default=221, space='buy', optimize=True)
    up = DecimalParameter(low=1.020, high=1.025, default=1.021, decimals=3 ,space='buy', optimize=True, load=True)
    dn = DecimalParameter(low=0.983, high=0.987, default=0.984, decimals=3 ,space='buy', optimize=True, load=True)
    enable1 = BooleanParameter(default=True, space="buy", optimize=False)
    enable2 = BooleanParameter(default=True, space="buy", optimize=False)
    enable3 = BooleanParameter(default=True, space="buy", optimize=False)
    enable4 = BooleanParameter(default=True, space="buy", optimize=False)
    enable5 = BooleanParameter(default=True, space="buy", optimize=False)
    enable6 = BooleanParameter(default=True, space="buy", optimize=False)
    increment = DecimalParameter(low=1.0005, high=1.002, default=1.001, decimals=4 ,space='buy', optimize=True, load=True)

    # DCA
    initial_safety_order_trigger = DecimalParameter(low=-0.02, high=-0.010, default=-0.018, decimals=3 ,space='buy', optimize=True, load=True)
    max_safety_orders = IntParameter(1, 6, default=2, space='buy', optimize=True)
    safety_order_step_scale = DecimalParameter(low=1.05, high=1.5, default=1.25, decimals=2 ,space='buy', optimize=True, load=True)
    safety_order_volume_scale = DecimalParameter(low=1.1, high=2, default=1.4, decimals=1 ,space='buy', optimize=True, load=True)

    # Unclog Function
    days = IntParameter(2, 7, default=4, space='sell', optimize=True)
    loss = DecimalParameter(-0.07, -0.01, default=-0.04, space='sell', optimize=True)

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


    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:


        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if current_candle['rsi'] < 80:

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
        else:
            for stop0 in self.tsl_target0.range:
                if (current_profit > stop0):
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} SWINGING FOR THE MOON!!!')
                    return 0.99

        return self.stoploss

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        if current_profit > self.initial_safety_order_trigger.value:
            logger.info(f"{trade.pair} - Current Profit: {current_profit} Trigger: {self.initial_safety_order_trigger.value}")
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        count_of_buys = 0
        for order in trade.orders:
            if order.ft_is_open or order.ft_order_side != 'buy':
                continue
            if order.status == "closed":
                count_of_buys += 1

        if 1 <= count_of_buys <= self.max_safety_orders.value:
            
            safety_order_trigger = abs(self.initial_safety_order_trigger.value) + (abs(self.initial_safety_order_trigger.value) * self.safety_order_step_scale.value * (math.pow(self.safety_order_step_scale.value,(count_of_buys - 1)) - 1) / (self.safety_order_step_scale.value - 1))

            if current_profit <= (-1 * abs(safety_order_trigger)):
                try:
                    stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
                    stake_amount = stake_amount * math.pow(self.safety_order_volume_scale.value,(count_of_buys - 1))
                    amount = stake_amount / current_rate
                    logger.info(f"Initiating safety order buy #{count_of_buys} for {trade.pair} with stake amount of {stake_amount} which equals {amount}")
                    return stake_amount
                except Exception as exception:
                    logger.info(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}') 
                    return None
            else:
                stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
                stake_amount = stake_amount * math.pow(self.safety_order_volume_scale.value,(count_of_buys - 1))
                logger.info(f"{trade.pair} Next Safety Order #{count_of_buys} @ Trigger -{safety_order_trigger} Current Profit: {current_profit}")    
                return None
        return None

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        # Sell any positions at a loss if they are held for more than 7 days.
        if current_profit < self.loss.value and (current_time - trade.open_date_utc).days >= self.days.value:
            return 'unclog'

    def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)

        entry_price = (dataframe['close'].iat[-1] + dataframe['open'].iat[-1] + proposed_rate) / 3
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

        if exit_reason == 'trailing_stop_loss' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} trailing stop price is below 0")
            self.dp.send_msg(f'{trade.pair} trailing stop price is below 0')
            return False

        return True


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        pair = metadata['pair']

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']
        dataframe['ha_closedelta'] = (heikinashi['close'] - heikinashi['close'].shift())

        dataframe["&s-extrema"] = 0

        dataframe['sma'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']
        dataframe['sma_up'] = dataframe['sma'] * self.up.value
        dataframe['sma_dn'] = dataframe['sma'] * self.dn.value
        


        #Bull Mode
        if dataframe['sma'].iloc[-1] > dataframe['sma'].iloc[-3]:
            order = self.bull.value
            logger.info(f"{pair} BULL MODE!!!")
            dataframe.loc[:, 'MODE'] = 1
            

        #Bear Mode
        elif (dataframe['sma'].iloc[-1] < dataframe['sma'].iloc[-3]) and (dataframe['close'].iloc[-1] > dataframe['sma'].iloc[-1]):
            order = self.bear.value
            logger.info(f"{pair} BEAR MODE!!!")
            dataframe.loc[:, 'MODE'] = -1
        #Blah Mode
        else:
            order = self.blah.value
            logger.info(f"{pair} SIDEWAYS MODE!!!")
            dataframe.loc[:, 'MODE'] = 0

        min_peaks = argrelextrema(
            dataframe["ha_open"].values, np.less,
            order=order
        )
        max_peaks = argrelextrema(
            dataframe["ha_close"].values, np.greater,
            order=order
        )
        for mp in min_peaks[0]:
            dataframe.at[mp, "&s-extrema"] = -1
        for mp in max_peaks[0]:
            dataframe.at[mp, "&s-extrema"] = 1
        dataframe["minima"] = np.where(dataframe["&s-extrema"] == -1, 1, 0)
        dataframe["maxima"] = np.where(dataframe["&s-extrema"] == 1, 1, 0)

        dataframe['maxima_check'] = dataframe['maxima'].rolling(3).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)
        dataframe['minima_check'] = dataframe['minima'].rolling(3).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)

        maxima_indices = dataframe[dataframe['maxima'] ==1].index
        minima_indices = dataframe[dataframe['minima'] ==1].index
        distances = []

        for max_index in maxima_indices:
        	for min_index in minima_indices:
        		distances.append(abs(max_index - min_index))
        dataframe['mean_distance'] = np.mean(distances) if distances else 0

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['minima'] == 1) &
                (dataframe['MODE'] == 1) &
                (self.enable1.value == True) &
                (dataframe['close'] < dataframe['sma_up'])

            ),
            ['enter_long', 'enter_tag']] = (1, 'minima BULL')

        dataframe.loc[
            (
                (dataframe['minima_check'] == 0) &
                (dataframe['MODE'] == 1) &
                (self.enable2.value == True) &
                (dataframe['close'] < dataframe['sma_up'])
            ),
           ['enter_long', 'enter_tag']] = (1, 'minima_check BULL')

        dataframe.loc[
            (
                (dataframe['minima'] == 1) &
                (dataframe['MODE'] == 0) &
                (self.enable3.value == True) &
                (dataframe['close'] < dataframe['sma'])

            ),
            ['enter_long', 'enter_tag']] = (1, 'minima SIDEWAYS')

        dataframe.loc[
            (
                (dataframe['minima_check'] == 0) &
                (dataframe['MODE'] == 0) &
                (self.enable4.value == True) &
                (dataframe['close'] < dataframe['sma'])
            ),
           ['enter_long', 'enter_tag']] = (1, 'minima_check SIDEWAYS')



        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['maxima'] == 1)

            ),
            ['exit_long', 'exit_tag']] = (1, 'maxima')

        dataframe.loc[
            (
                (dataframe['maxima_check'] == 0)

            ),
            ['exit_long', 'exit_tag']] = (1, 'maxima_check')

        return dataframe






