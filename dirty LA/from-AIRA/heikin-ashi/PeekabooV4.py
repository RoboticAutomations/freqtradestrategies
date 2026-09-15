
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from typing import Optional
from pandas import DataFrame
import warnings
import pandas as pd
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
from scipy.signal import find_peaks
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


logger = logging.getLogger(__name__)


class PeekabooV4(IStrategy):
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

#     # ROI table:
#     minimal_roi = {
#         "0": 0.195,
#         "39": 0.10600000000000001,
#         "91": 0.04,
#         "210": 0
#     }

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
    timeframe = '15m'
    startup_candle_count = 100
    process_only_new_candles = True
    
    # DCA
    position_adjustment_enable = True

    # Custom Entry
    last_entry_price = None

    # Hyper-opt parameters
    base_nb_candles_buy = IntParameter(150, 200, default=184, space='buy', optimize=True)
    up = DecimalParameter(low=1.020, high=1.025, default=1.02, decimals=3 ,space='buy', optimize=True, load=True)
    dn = DecimalParameter(low=0.983, high=0.987, default=0.984, decimals=3 ,space='buy', optimize=True, load=True)
    # enable1 = BooleanParameter(default=True, space="buy", optimize=False)
    enable2 = BooleanParameter(default=True, space="buy", optimize=False)
    # enable3 = BooleanParameter(default=True, space="buy", optimize=False)
    # enable4 = BooleanParameter(default=True, space="buy", optimize=False)
    # enable5 = BooleanParameter(default=True, space="buy", optimize=False)
    # enable6 = BooleanParameter(default=True, space="buy", optimize=False)
    increment = DecimalParameter(low=1.0005, high=1.001, default=1.0007, decimals=4 ,space='buy', optimize=True, load=True)
    # quick = DecimalParameter(low=0.75, high=0.9, default=0.85, decimals=2 ,space='buy', optimize=True, load=True)
    m00n = DecimalParameter(low=1.60, high=1.80, default=1.68, decimals=2 ,space='buy', optimize=True, load=True)

    # Modulus    
    peaks = IntParameter(70, 200, default=120, space='buy', optimize=True) ### initial smallest window
    # bull_bear = IntParameter(120, 160, default=155, space='buy', optimize=True)
    # trend = DecimalParameter(low=25, high=40, default=29.6, decimals=1 ,space='buy', optimize=True, load=True)
    # volatility = DecimalParameter(low=30, high=50, default=38.4, decimals=1 ,space='buy', optimize=True, load=True)
    # sensitivity = IntParameter(7, 15, default=11, space='buy', optimize=True, load=True)
    # lookback_candles = IntParameter(30, 60, default=55, space='buy', optimize=True, load=True)
    # atr = IntParameter(3, 7, default=5, space='buy', optimize=True, load=True)
    perc_target = DecimalParameter(low=1, high=5, default=1.5, decimals=1 ,space='buy', optimize=True, load=True)

    # DCA
    initial_safety_order_trigger = DecimalParameter(low=-0.02, high=-0.015, default=-0.016, decimals=3 ,space='buy', optimize=True, load=True)
    max_safety_orders = IntParameter(1, 6, default=2, space='buy', optimize=True)
    max_dca_multiplier = IntParameter(1, 6, default=2, space='buy', optimize=True)
    safety_order_step_scale = DecimalParameter(low=1.1, high=1.4, default=1.3, decimals=2 ,space='buy', optimize=True, load=True)
    safety_order_volume_scale = DecimalParameter(low=1.1, high=1.5, default=1.2, decimals=1 ,space='buy', optimize=True, load=True)

    # Unclog Function
    days = IntParameter(2, 7, default=3, space='sell', optimize=True)
    loss = DecimalParameter(-0.07, -0.04, default=-0.055, space='sell', optimize=True)

    ### trailing stop loss optimiziation ###
    tsl_target3 = DecimalParameter(low=0.10, high=0.15, default=0.15, decimals=2,  space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, decimals=3,  space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.06, high=0.10, default=0.1, decimals=3, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, decimals=3, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.04, high=0.06, default=0.06, decimals=3, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, decimals=3, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.04, high=0.05, default=0.04, decimals=3, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.005, high=0.012, default=0.01, decimals=3, space='sell', optimize=True, load=True)
    moon = IntParameter(80, 90, default=85, space='sell', optimize=True)


    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:


        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if current_candle['rsi'] < self.moon.value:

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
                    logger.debug(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}') 
                    return None
            else:
                stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
                stake_amount = stake_amount * math.pow(self.safety_order_volume_scale.value,(count_of_buys - 1))
                logger.info(f"{trade.pair} Next Safety Order #{count_of_buys} @ Trigger -{safety_order_trigger} Current Profit: {current_profit}")    
                return None

        logger.info(f"{trade.pair} - Current Profit: {current_profit} All safety orders used")     
        return None

    ### Custom Functions ###
    # This is called when placing the initial order (opening trade)
    # Let unlimited stakes leave funds open for DCA orders
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        # We need to leave most of the funds for possible further DCA orders
        # This also applies to fixed stakes
        total_stake = self.max_dca_multiplier.value + self.max_safety_orders.value
        return proposed_stake / total_stake

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        # Sell any positions at a loss if they are held for more than 7 days.
        if current_profit < self.loss.value and (current_time - trade.open_date_utc).days >= self.days.value:
            return 'unclog'

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

        # Handle freak events

        if exit_reason == 'partial_exit' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} partial exit is below 0")
            self.dp.send_msg(f'{trade.pair} partial exit is below 0')
            return False

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

        # heikinashi = qtpylib.heikinashi(dataframe)
        # dataframe['ha_open'] = heikinashi['open']
        # dataframe['ha_close'] = heikinashi['close']
        # dataframe['ha_high'] = heikinashi['high']
        # dataframe['ha_low'] = heikinashi['low']
        # dataframe['ha_closedelta'] = (heikinashi['close'] - heikinashi['close'].shift())

        dataframe["&s-extrema"] = 0

        dataframe['sma'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']
        dataframe['sma_up'] = dataframe['sma'] * self.up.value
        dataframe['sma_dn'] = dataframe['sma'] * self.dn.value

        # dataframe['sma_pc'] = abs((dataframe['sma'] - dataframe['sma'].shift(1)) / dataframe['sma']) * 100
        # dataframe['atr_pcnt'] = (qtpylib.atr(dataframe, window = self.atr.value)) / dataframe['ha_close']
       
        # dataframe['modulation'] = 1 + (dataframe['sma_pc'] * self.trend.value) + (dataframe['atr_pcnt'] * self.volatility.value)

        # min_window = self.peaks.value 
        # max_window = self.bull_bear.value

        # # Set minimum and maximum window size
        # dataframe['order'] = (dataframe['modulation'] * self.peaks.value).round().fillna(self.bull_bear.value).astype(int)
        # dataframe['order'] = np.where(dataframe['order'] > max_window, max_window, dataframe['order'])
        # dataframe['order'] = np.where(dataframe['order'] < min_window, min_window, dataframe['order'])
        
        # if not dataframe['order'].empty:
        #     order = dataframe['order'].iloc[-1]
        # else:
        #     order = self.bear.value

        # peak prominence
        ptc_target= self.perc_target.value
        dataframe['peak_prominance'] = float(0)
        dataframe['peak'] = float(0)

        high_peaks, high_properties = find_peaks(dataframe['high'].values, prominence=dataframe['high'].values / 100 * ptc_target,
                                                 wlen=120)

        lower_peaks, low_properties = find_peaks(-dataframe['low'].values, prominence=dataframe['low'].values / 100 * ptc_target,
                                                 wlen=120)

        dataframe.iloc[lower_peaks, dataframe.columns.get_loc('peak_prominance')] = low_properties["prominences"].astype('float')
        dataframe.iloc[high_peaks, dataframe.columns.get_loc('peak_prominance')] = -high_properties["prominences"].astype('float')

        dataframe.iloc[lower_peaks, dataframe.columns.get_loc('peak')] = 1
        dataframe.iloc[high_peaks, dataframe.columns.get_loc('peak')] = -1

        positive_values = dataframe['peak_prominance'].where(dataframe['peak_prominance'] > 0)
        negative_values = dataframe['peak_prominance'].where(dataframe['peak_prominance'] < 0)

        # Calculate the expanding mean for positive values
        dataframe['positive_expanding_mean'] = positive_values.expanding().mean()

        # Calculate the expanding mean for negative values
        dataframe['negative_expanding_mean'] = negative_values.expanding().mean()

        # Fill NaN values with the previous non-null values
        dataframe['positive_expanding_mean'].ffill(inplace=True)
        dataframe['negative_expanding_mean'].ffill(inplace=True)
        dataframe.reset_index(drop=True, inplace=True)

        
        # prom = dataframe['peak_prominance'].iloc[-1]
        # if prom != 0:
        #     logger.info(f"*** {pair} *** Prominence 0: {prom}")
        # prom1 = dataframe['peak_prominance'].iloc[-2]
        # if prom1 != 0:
        #     logger.info(f"*** {pair} *** Prominence 1: {prom1}")
        # prom2 = dataframe['peak_prominance'].iloc[-3]
        # if prom2 != 0:
        #     logger.info(f"*** {pair} *** Prominence 2: {prom2}")
        # prom3 = dataframe['peak_prominance'].iloc[-4]
        # if prom3 != 0:
        #     logger.info(f"*** {pair} *** Prominence 3: {prom3}")
        # Debugging print statements
        # print("High Peaks Indices:", high_peaks)
        # print("Original High Values:", dataframe['high'].iloc[high_peaks])
        # print("Assigned Prominences:", -high_properties["prominences"].astype('float'))

   
        '''comment this out if you don't want signal messages to Telegram'''
        pair = metadata['pair']
        if dataframe['peak'].iloc[-1] < 0:
            self.dp.send_msg(f'*** {pair} *** Maxima Detected - Prominence: {prom}')
            logger.info(f"*** {pair} *** Maxima Detected - Prominence: {prom}")

        if dataframe['peak'].iloc[-1] > 0:
            self.dp.send_msg(f'*** {pair} *** Minima Detected - Prominence: {prom}')
            logger.info(f"*** {pair} *** Maxima Detected - Prominence: {prom}")

        # last_few_rows_column_A = dataframe['peak_prominance'].tail()
        # print(pair, prom, prom1, prom2, prom3 ,last_few_rows_column_A)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # condition1 = (
        #     (dataframe['peak_prominance'] >= dataframe['positive_expanding_mean']) &
        #     # (dataframe['order'] < (self.peaks.value * self.quick.value)) &
        #     (self.enable1.value == True) 
        #     # (dataframe['close'] < dataframe['sma_dn'])
        # )

        # dataframe.loc[condition1, 'enter_long'] = 1
        # dataframe.loc[condition1, 'enter_tag'] = 'minima_check_quick_dn'

        condition2 = (
                (dataframe['peak_prominance'].shift() >= dataframe['positive_expanding_mean'].shift()) &
        #         (dataframe['order'] < (self.peaks.value * self.quick.value)) &
                (self.enable2.value == True) 
        #         (dataframe['close'] < dataframe['sma'])
            )

        dataframe.loc[condition2, 'enter_long'] = 1
        dataframe.loc[condition2, 'enter_tag'] = 'minima_check_1_delay'

        # condition3 = (
        #         (dataframe['minima_check'] == 0) &
        #         (dataframe['order'] > (self.peaks.value * self.quick.value)) &
        #         (dataframe['order'] < (self.peaks.value * self.m00n.value)) &
        #         (self.enable3.value == True) &
        #         (dataframe['close'] < dataframe['sma_dn'])
        #     )

        # dataframe.loc[condition3, 'enter_long'] = 1
        # dataframe.loc[condition3, 'enter_tag'] = 'minima_check_med_dn'

        # condition4 = (
        #         (dataframe['minima_check'] == 0) &
        #         (dataframe['order'] > (self.peaks.value * self.quick.value)) &
        #         (dataframe['order'] < (self.peaks.value * self.m00n.value)) &
        #         (self.enable4.value == True) &
        #         (dataframe['close'] < dataframe['sma'])
        #     )

        # dataframe.loc[condition4, 'enter_long'] = 1
        # dataframe.loc[condition4, 'enter_tag'] = 'minima_check_med'

        # condition5 = (
        #         (dataframe['minima_check'] == 0) &
        #         (dataframe['order'] > (self.peaks.value * self.m00n.value)) &
        #         (self.enable5.value == True) &
        #         (dataframe['close'] < dataframe['sma_dn'])
        #     )

        # dataframe.loc[condition5, 'enter_long'] = 1
        # dataframe.loc[condition5, 'enter_tag'] = 'minima_check_m00n_dn'

        # condition6 = (
        #         (dataframe['minima_check'] == 0) &
        #         (dataframe['order'] > (self.peaks.value * self.m00n.value)) &
        #         (self.enable6.value == True) &
        #         (dataframe['close'] < dataframe['sma'])
        #     )

        # dataframe.loc[condition6, 'enter_long'] = 1
        # dataframe.loc[condition6, 'enter_tag'] = 'minima_check_moon'

        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        condition7 = (
                (dataframe['peak_prominance'] <= dataframe['negative_expanding_mean']) 
            )

        dataframe.loc[condition7, 'exit_long'] = 1
        dataframe.loc[condition7, 'exit_tag'] = 'maxima_check'

        condition8 = (
                (dataframe['peak_prominance'] < 0)&
                (dataframe['close'] < dataframe['sma'])
                
            )

        dataframe.loc[condition8, 'exit_long'] = 1
        dataframe.loc[condition8, 'exit_tag'] = 'maxima_check_below_sma'

        condition9 = (
                (dataframe['peak_prominance'].shift() <= dataframe['negative_expanding_mean'].shift()) 
            )

        dataframe.loc[condition9, 'exit_long'] = 1
        dataframe.loc[condition9, 'exit_tag'] = 'maxima_check'

        condition10 = (
                (dataframe['peak_prominance'].shift() < 0)&
                (dataframe['close'].shift() < dataframe['sma'].shift())
                
            )

        dataframe.loc[condition10, 'exit_long'] = 1
        dataframe.loc[condition10, 'exit_tag'] = 'maxima_check_below_sma'

        return dataframe

