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
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
import math
import logging

logger = logging.getLogger(__name__)

# @Rallipanos # changes by IcHiAT


def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif



class EI1_t4c0s(IStrategy):

    '''
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
                                                                                                                             
    '''                                                                        

    ### 3-13-2023 ###
    # WIP - V1
    # No hyper-opt using just defaults from orig EI3 strat, you need to try and hyper-opt this file -
    # Will suggest adding in an roi table.
    ###


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

    # Stoploss - Unclog
    stoploss = -0.99
    moon = DecimalParameter(0.9, 3.0, default=2.5,  decimals=1, space='sell', optimize=True)
    start = DecimalParameter(0.10, 0.35, default=0.21,  decimals=2, space='sell', optimize=True)
    unclog1 = DecimalParameter(0.04, 0.2, default=0.19,  decimals=2, space='sell', optimize=True)
    unclog2 = DecimalParameter(0.04, 0.15, default=0.11,  decimals=2, space='sell', optimize=True)
    unclog3 = DecimalParameter(0.04, 0.18, default=0.14,  decimals=2, space='sell', optimize=True)
    unclog4 = DecimalParameter(0.04, 0.18, default=0.18,  decimals=2, space='sell', optimize=True)
    day1 = IntParameter(2, 4, default=2, space='sell', optimize=True)
    day2 = IntParameter(3, 5, default=4, space='sell', optimize=True)
    day3 = IntParameter(4, 8, default=6, space='sell', optimize=True)
    day4 = IntParameter(7, 10, default=10, space='sell', optimize=True)


    # SMAOffset
    base_nb_candles_buy = IntParameter(15, 30, default=20, space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(15, 30, default=18, space='sell', optimize=True)
    low_offset = DecimalParameter(0.80, 0.90, default=0.98, space='buy', optimize=True)
    high_offset = DecimalParameter(1.04, 1.10, default=1.004, space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.10, 1.20, default=1.016, space='sell', optimize=True)

    # lambo2
    lambo2_ema_14_factor = DecimalParameter(0.8, 0.9, decimals=3,  default=0.88, space='buy', optimize=True)
    lambo2_rsi_4_limit = IntParameter(10, 30, default=12, space='buy', optimize=True)
    lambo2_rsi_14_limit = IntParameter(10, 37, default=33, space='buy', optimize=True)

    # Protection
    fast_ewo = IntParameter(30, 50, default=50, space='buy', optimize=True)
    slow_ewo = IntParameter(80, 200, default=200, space='buy', optimize=True)

    rsi_buy = IntParameter(50, 70, default=69, space='buy', optimize=True)
    window = IntParameter(12, 36, default=19, space='buy', optimize=True)

    #cofi
    is_optimize_cofi = True
    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=0.967 , optimize = is_optimize_cofi)
    buy_fastk = IntParameter(20, 30, default=25, optimize = is_optimize_cofi)
    buy_fastd = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    buy_adx = IntParameter(20, 30, default=29, optimize = is_optimize_cofi)
    
    atr_length = IntParameter(10, 15, default=12, space='buy', optimize=True)
    increment = DecimalParameter(low=1.0005, high=1.001, default=1.0007, decimals=4 ,space='buy', optimize=True, load=True)

    use_custom_stoploss = True
    process_only_new_candles = True
    # Custom Entry
    last_entry_price = None


    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:


        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        SLT1 = (current_candle['move_mean'] * self.moon.value)
        SL1 = (current_candle['move_mean'] * self.start.value) * self.moon.value
        SLT2 = (current_candle['move_mean_x'] * self.moon.value)
        SL2 = (current_candle['move_mean_x'] - current_candle['move_mean']) * self.moon.value
        display_profit = current_profit * 100
        slt1 = SLT1 * 100
        sl1 = SL1 * 100
        slt2 = SLT2 * 100
        sl2 = SL2 * 100


        if current_candle['max_l'] != 0: #ignore stoploss if setting new highs
            if SLT2 is not None and current_profit > SLT2:
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                return SL2
            if SLT1 is not None and current_profit > SLT1:
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                return SL1

            else:
                if SLT1 is not None and current_profit > SLT1:
                    self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% SWINGING FOR THE MOON!!!')
                    logger.info(f'*** {pair} *** Profit {display_profit:.3f}% SWINGING FOR THE MOON!!!')
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

        if exit_reason == 'roi' and (last_candle['max_l'] < 0.002):
            return False

        # Handle freak events

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
        # Sell any positions at a loss if they are held for more than X days.
        if current_profit < -self.unclog4.value and (current_time - trade.open_date_utc).days >= self.day4.value:
            return 'unclog 4'
        if current_profit < -self.unclog3.value and (current_time - trade.open_date_utc).days >= self.day3.value:
            return 'unclog 3'
        if current_profit < -self.unclog2.value and (current_time - trade.open_date_utc).days >= self.day2.value:
            return 'unclog 2'
        if current_profit < -self.unclog1.value and (current_time - trade.open_date_utc).days >= self.day1.value:
            return 'unclog 1'
    
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
    timeframe = '1h'

    position_adjustment_enable = False
    process_only_new_candles = True
    startup_candle_count = 200


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        # HMA-BUY SQUEEZE
        dataframe['HMA_SQZ'] = (((dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] - dataframe['hma_50']) 
            / dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']) * 100)


        dataframe['zero'] = 0
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo.value, self.slow_ewo.value)
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

        dataframe['max_l'] = dataframe['OHLC4'].rolling(48).max() / dataframe['OHLC4'] - 1
        dataframe['min_l'] = abs(dataframe['OHLC4'].rolling(48).min() / dataframe['OHLC4'] - 1)

        dataframe['max_x'] = dataframe['OHLC4'].rolling(336).max() / dataframe['OHLC4'] - 1
        dataframe['min_x'] = abs(dataframe['OHLC4'].rolling(336).min() / dataframe['OHLC4'] - 1)


        # Apply rolling window operation to the 'OHLC4'column
        rolling_window = dataframe['OHLC4'].rolling(self.window.value) 
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

        # Apply rolling window operation to the 'OHLC4'column
        rolling_window_x = dataframe['OHLC4'].rolling(200)
        rolling_max_x = rolling_window_x.max()
        rolling_min_x = rolling_window_x.min()

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_value_x = rolling_window_x.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['move_l'] = ptp_value_x / dataframe['OHLC4']
        dataframe['move_mean_l'] = dataframe['move_l'].mean()
        dataframe['move_mean_xl'] = dataframe['move_l'].mean() * 1.6
        dataframe['exit_mean_l'] = rolling_min * (1 + dataframe['move_mean_l'])
        dataframe['exit_mean_xl'] = rolling_min * (1 + dataframe['move_mean_xl'])
        dataframe['enter_mean_l'] = rolling_max * (1 - dataframe['move_mean_l'])
        dataframe['enter_mean_xL'] = rolling_max * (1 - dataframe['move_mean_xl'])

        dataframe['ma_lo'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (1 - dataframe['move_mean'])
        dataframe['ma_hi'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (1 + dataframe['move_mean'])
        dataframe['ma_hi_2'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (1 + dataframe['move_mean_x'])

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        lambo2 = (
            (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value)) &
            (dataframe['atr_pcnt'] > dataframe['min_l']) &
            (dataframe['volume'] > 0) 
        )
        dataframe.loc[lambo2, 'entry'] = 1
        # dataframe.loc[lambo2, 'enter_tag'] = 'lambo '

        buy1ewo = (
                (dataframe['rsi_fast'] < 35 ) &
                (dataframe['close'] < dataframe['ma_lo']) &
                (dataframe['EWO'] > dataframe['EWO_MEAN_UP']) &
                (dataframe['close'] < dataframe['enter_mean_x']) &
                (dataframe['close'].shift() < dataframe['enter_mean_x'].shift()) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['atr_pcnt'] > dataframe['min']) &
                (dataframe['volume'] > 0) 
        )
        dataframe.loc[buy1ewo, 'entry'] = 1
        # dataframe.loc[buy1ewo, 'enter_tag'] = 'buy1ewo'

        buy2ewo = (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < dataframe['ma_lo']) &
                (dataframe['EWO'] < dataframe['EWO_DN_FIB']) &
                (dataframe['atr_pcnt'] > dataframe['min']) &
                (dataframe['volume'] > 0) 
        )
        dataframe.loc[buy2ewo, 'entry'] = 1
        # dataframe.loc[buy2ewo, 'enter_tag'] = 'buy2ewo'

        is_cofi = (
                (dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value) &
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['fastk'] < self.buy_fastk.value) &
                (dataframe['fastd'] < self.buy_fastd.value) &
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['EWO'] > dataframe['EWO_MEAN_UP']) &
                (dataframe['atr_pcnt'] > dataframe['min']) &
                (dataframe['volume'] > 0) 
            )
        dataframe.loc[is_cofi, 'entry'] = 1
        # dataframe.loc[is_cofi, 'enter_tag'] = 'cofi'

        is_entry = (
                (dataframe['entry'].shift() == 1) &
                (dataframe['entry'] != 1 ) 
            )
        dataframe.loc[is_entry, 'enter_long'] = 1
        dataframe.loc[is_entry, 'enter_tag'] = 'entry'


        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        condition5 = (
                (dataframe['close'] > dataframe['hma_50']) &
                (dataframe['close'] > dataframe['ma_hi_2']) &
                (dataframe['max_l'] != 0) &
                (dataframe['close'] > dataframe['exit_mean_x']) &
                (dataframe['rsi'] > 50 ) &
                (dataframe['volume'] > 0 ) &
                (dataframe['rsi_fast']>dataframe['rsi_slow'])

            )
        dataframe.loc[condition5, 'exit'] = 1
        # dataframe.loc[condition5, 'exit_tag'] = 'Close > Offset Hi 2'


        
        condition6 = (
                (dataframe['close'] < dataframe['hma_50']) &
                (dataframe['close'] > dataframe['ma_hi']) &
                (dataframe['max_l'] != 0) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast']>dataframe['rsi_slow'])

            )
        dataframe.loc[condition6, 'exit'] = 1
        # dataframe.loc[condition6, 'exit_tag'] = 'Close > Offset Hi 1'
        exit = (
                (dataframe['exit'].shift() == 1) &
                (dataframe['exit'] != 1) 
            )
        dataframe.loc[exit, 'exit_long'] = 1

        return dataframe


def pct_change(a, b):
    return (b - a) / a




