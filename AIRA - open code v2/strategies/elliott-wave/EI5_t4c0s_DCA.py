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



class EI5_t4c0s_DCA(IStrategy):

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

    # Stoploss - Unclog
    stoploss = -0.99
    moon = DecimalParameter(0.001, 0.004, default=0.35,  decimals=3, space='sell', optimize=True)
    unclog = DecimalParameter(0.04, 0.2, default=0.04,  decimals=2, space='sell', optimize=True)
    day = IntParameter(2, 5, default=4, space='sell', optimize=True)

    # DCA
    max_dca_multiplier = DecimalParameter(1.0, 4.0, default=2.5, decimals=1, space='buy', optimize=True)

    # SMAOffset
    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)

    # lambo2
    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3,  default=buy_params['lambo2_ema_14_factor'], space='buy', optimize=True)
    lambo2_rsi_4_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_4_limit'], space='buy', optimize=True)
    lambo2_rsi_14_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_14_limit'], space='buy', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)
    window = IntParameter(30, 70, default=48, space='buy', optimize=True)

    #cofi
    is_optimize_cofi = True
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

    # DCA
    position_adjustment_enable = True
    max_entry_position_adjustment = 2 # initially set to 3 
    process_only_new_candles = True
    startup_candle_count = 400

    ### Custom Functions ###
    # This is called when placing the initial order (opening trade)
    # Let unlimited stakes leave funds open for DCA orders
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        # We need to leave most of the funds for possible further DCA orders
        proposed_stake = proposed_stake / self.max_dca_multiplier.value  #  Leaving some reserve incase the market dumps!!!

        # This also applies to fixed staked
        return proposed_stake 


    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        DCA1 = current_candle['move_mean']
        DCA2 = current_candle['move_mean_x']

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        if current_profit is not None:
            logger.info(f"{trade.pair} - Current Profit: {current_profit:.4f} # of Entries: {trade.nr_of_successful_entries} DCA1: -{DCA1:.4f} DCA2: -{DCA2:.4f}")
        # Take Profit if m00n
        if current_profit > (DCA1*2) and trade.nr_of_successful_exits == 0:
            # Take half of the profit at +5%
            return -(trade.stake_amount / 2)
        if current_profit > (DCA1*2.618) and trade.nr_of_successful_exits == 1:
            # Take half of the profit at +5%
            return -(trade.stake_amount / 1)

        # Profit Based DCA    
        if current_profit > -DCA1 and trade.nr_of_successful_entries == 1:
            return None

        if current_profit > -DCA2 and trade.nr_of_successful_entries == 2:
            return None

        try:
            # This returns first order stake size 
            # Modify the following parameters to enable more levels or different buy size:
            # max_entry_position_adjustment = 3 
            # max_dca_multiplier = 3.5 

            stake_amount = filled_entries[0].cost
            # This then calculates current safety order size
            if count_of_entries == 1: 
                stake_amount = stake_amount * 1
            elif count_of_entries == 2:
                stake_amount = stake_amount * 2
            else:
                stake_amount = stake_amount

            return stake_amount
        except Exception as exception:
            return None

        return None

    ### Pair Adaptive Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:


        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        SLT1 = current_candle['move_mean']
        SL1 = SLT1 * 0.5
        SLT2 = current_candle['move_mean_x']
        SL2 = current_candle['move_mean_x'] - current_candle['move_mean']
        SLT0 = SLT1 * 0.5
        SL0 = SLT0 * 0.382
        display_profit = current_profit * 100
        slt1 = SLT1 * 100
        sl1 = SL1 * 100
        slt2 = SLT2 * 100
        sl2 = SL2 * 100
        slt0 = SLT0 * 100
        sl0 = SL0 * 100


        if current_candle['max_l'] > self.moon.value: #ignore stoploss if setting new highs
            if SLT2 is not None and current_profit > SLT2:
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                return SL2
            if SLT1 is not None and current_profit > SLT1:
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                return SL1
            if SLT1 is not None and current_profit > SLT0:
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% - {slt0:.3f}%/{sl0:.3f}% activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.3f}% - {slt0:.3f}%/{sl0:.3f}% activated')
                return 

        else:
            if SLT1 is not None and current_profit > SL1:
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
        # Sell any positions at a loss if they are held for more than X days.
        if current_profit < -self.unclog.value and (current_time - trade.open_date_utc).days >= self.day.value:
            return 'unclog'
    


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

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
        dataframe['max'] = dataframe['OHLC4'].rolling(12).max() / dataframe['OHLC4'] - 1
        dataframe['min'] = abs(dataframe['OHLC4'].rolling(12).min() / dataframe['OHLC4'] - 1)

        dataframe['max_l'] = dataframe['OHLC4'].rolling(360).max() / dataframe['OHLC4'] - 1
        dataframe['min_l'] = abs(dataframe['OHLC4'].rolling(360).min() / dataframe['OHLC4'] - 1)

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

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['ma_lo'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (1 - (dataframe['move_mean'] * 0.5))
        dataframe['ma_hi'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (1 + (dataframe['move_mean'] * 0.5))
        dataframe['ma_hi_2'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (1 + (dataframe['move_mean'] * 0.618))

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        # HMA-BUY SQUEEZE
        dataframe['HMA_SQZ'] = (((dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] - dataframe['hma_50']) 
            / dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']) * 100)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        lambo2 = (
            (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value)) &
            (dataframe['atr_pcnt'] > dataframe['min_l']) &
            (dataframe['volume'] > 0) 
        )
        dataframe.loc[lambo2, 'enter_long'] = 1
        dataframe.loc[lambo2, 'enter_tag'] = 'lambo '

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
        dataframe.loc[buy1ewo, 'enter_long'] = 1
        dataframe.loc[buy1ewo, 'enter_tag'] = 'buy1ewo'

        buy2ewo = (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < dataframe['ma_lo']) &
                (dataframe['EWO'] < dataframe['EWO_DN_FIB']) &
                (dataframe['atr_pcnt'] > dataframe['min']) &
                (dataframe['volume'] > 0) 
        )
        dataframe.loc[buy2ewo, 'enter_long'] = 1
        dataframe.loc[buy2ewo, 'enter_tag'] = 'buy2ewo'

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
        dataframe.loc[is_cofi, 'enter_long'] = 1
        dataframe.loc[is_cofi, 'enter_tag'] = 'cofi'

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
        dataframe.loc[is_cofi, 'enter_long'] = 1
        dataframe.loc[is_cofi, 'enter_tag'] = 'cofi'


        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        condition5 = (
                (dataframe['close'] > dataframe['hma_50']) &
                (dataframe['close'] > dataframe['ma_hi_2']) &
                (dataframe['close'] > dataframe['exit_mean_x']) &
                (dataframe['rsi'] > 50 ) &
                (dataframe['volume'] > 0 ) &
                (dataframe['rsi_fast']>dataframe['rsi_slow'])

            )
        dataframe.loc[condition5, 'exit_long'] = 1
        dataframe.loc[condition5, 'exit_tag'] = 'Close > Offset Hi 2'


        
        condition6 = (
                (dataframe['close'] < dataframe['hma_50']) &
                (dataframe['close'] > dataframe['ma_hi']) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast']>dataframe['rsi_slow'])

            )
        dataframe.loc[condition6, 'exit_long'] = 1
        dataframe.loc[condition6, 'exit_tag'] = 'Close > Offset Hi 1'

        return dataframe


def pct_change(a, b):
    return (b - a) / a




