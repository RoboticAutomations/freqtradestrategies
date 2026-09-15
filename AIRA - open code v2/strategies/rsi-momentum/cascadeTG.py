# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame, Series
from typing import Optional, Union
import pandas_ta as pda

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, RealParameter,
                                IStrategy, IntParameter, stoploss_from_absolute, stoploss_from_open, informative, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtp
import technical.indicators as ti
import technical.bouncyhouse as bh
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from typing import *
from decimal import Decimal
# from smartmoneyconcepts.smc import smc
import math
import logging
logger = logging.getLogger(__name__)





'''

Cascading Grid Trading strategy by sx584

based on https://www.youtube.com/watch?v=YFzlBQCeynQ by EcoEngineering for Forex Markets


'''
def to_minutes(**timdelta_kwargs):
    # Convert numpy.int64 values to int
    timdelta_kwargs = {k: int(v) for k, v in timdelta_kwargs.items()}
    return int(timedelta(**timdelta_kwargs).total_seconds() / 60)

class cascadeTG(IStrategy):
    class HyperOpt:
        def max_open_trades_space() -> List[Dimension]:
            return [
                Integer(5, 20, name='max_open_trades'),
            ]
        
    INTERFACE_VERSION = 3
    can_short: bool = True
    minimal_roi = {
        "0": 1000
    }
    stoploss = -0.99

    position_adjustment_enable = True
    use_custom_stoploss = True

    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    # Leverage max
    lev = IntParameter(low=2, high=10, default=1, space="buy", optimize=True, load=True)

    # # sl parameters
    atr_period = IntParameter(4, 50, default=50, space='buy', optimize=True, load=True)
    timeframe_minutes = timeframe_to_minutes(timeframe)
    risk_per_trade = DecimalParameter(0.02, 0.10, default=0.03, decimals=2, space='buy', optimize=True, load=True)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 300

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    plot_config = {
        "main_plot": {
        },
        "subplots": {}
        }

    @property
    def protections(self):
        return [
            {
                # Don't enter a trade right after selling a trade.
                "method": "CooldownPeriod",
                "stop_duration": to_minutes(minutes=60),
            },
            {
                # Stop trading if max-drawdown is reached.
                "method": "MaxDrawdown",
                "lookback_period": to_minutes(hours=12),
                "trade_limit": 20,  # Considering all pairs that have a minimum of 20 trades
                "stop_duration": to_minutes(hours=1),
                "max_allowed_drawdown": 0.2,  # If max-drawdown is > 20% this will activate
            },
            {
                # Stop trading if a certain amount of stoploss occurred within a certain time window.
                "method": "StoplossGuard",
                "lookback_period": to_minutes(hours=6),
                "trade_limit": 4,  # Considering all pairs that have a minimum of 4 trades
                "stop_duration": to_minutes(minutes=30),
                "only_per_pair": False,  # Looks at all pairs
            },
            {
                # Lock pairs with low profits
                "method": "LowProfitPairs",
                "lookback_period": to_minutes(hours=1, minutes=30),
                "trade_limit": 2,  # Considering all pairs that have a minimum of 2 trades
                "stop_duration": to_minutes(hours=15),
                "required_profit": 0.02,  # If profit < 2% this will activate for a pair
            },
            {
                # Lock pairs with low profits
                "method": "LowProfitPairs",
                "lookback_period": to_minutes(hours=6),
                "trade_limit": 4,  # Considering all pairs that have a minimum of 4 trades
                "stop_duration": to_minutes(minutes=30),
                "required_profit": 0.01,  # If profit < 1% this will activate for a pair
            },
        ]
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:

        return self.lev.value
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:       
        
        dataframe["atr"] = calculate_atr(dataframe, atr_period=self.atr_period.value)
        dataframe["rs_volatility"] = rogers_satchell(dataframe)
        dataframe["ht_volatility"] = hodges_tompkins(dataframe)

        dataframe["x_atr"] = dataframe["atr"] *2.5 
        dataframe["x_atr_sl"] = dataframe["x_atr"] / 2
        
        for i in range(1, 11):
            dataframe[f"grid_buy{i}"] = dataframe["close"] + (dataframe['x_atr'] * i)
            dataframe[f"grid_buy{i}_sl"] = dataframe[f"grid_buy{i}"] - dataframe["x_atr_sl"]
            
            dataframe[f"grid_sell{i}"] = dataframe["close"] - (dataframe['x_atr'] * i)
            dataframe[f"grid_sell{i}_sl"] = dataframe[f"grid_sell{i}"] + dataframe["x_atr_sl"]
        
        
        dataframe["rsi4"] = ta.RSI(dataframe, timeperiod=4)
        dataframe["rsi14"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi4_classic"] = ta.RSI(dataframe, timeperiod=4)
        dataframe["rsi_overbought"] = 70
        dataframe["rsi_oversold"] = 30

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        long = (
            # Momentum is bullish
            (qtp.crossed_above(dataframe["rsi4"], 70)) &
            # volume is not 0
            (dataframe["volume"] > 0)
        )
        dataframe.loc[long, 'enter_long'] = 1
        dataframe.loc[long, 'enter_tag'] = 'RSI overbought - going long'
        
        short = (
            # Momentum is bearish
            (qtp.crossed_below(dataframe["rsi4"], 30)) &
            # volume is not 0
            (dataframe["volume"] > 0)
        )
        dataframe.loc[short, 'enter_short'] = 1
        dataframe.loc[short, 'enter_tag'] = 'RSI oversold - going short'     

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                # not used
            ),

            'exit_long'] = 1

        dataframe.loc[
            (
                # not used
            ),
            'exit_short'] = 1

        return dataframe
    
    

    
    def custom_stake_amount(self, pair: str, current_rate: float, proposed_stake: float, side: str, **kwargs) -> float:
        stake_currency = "USDT"

        # Equity and Capital
        total_equity = self.wallets.get_total(currency=stake_currency)
        max_open_trades = self.max_open_trades
        def_stake = total_equity / max_open_trades

        # Get analyzed dataframe and last candle/entry price
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        entry_price = last_candle["close"]
        capital_at_risk = total_equity * self.risk_per_trade.value

        # Calculate stop loss and max risk per trade
        if side == "long":
            stop_loss = entry_price - last_candle["x_atr_sl"]
            max_risk_per_trade = entry_price - stop_loss
        else:
            stop_loss = entry_price + last_candle["x_atr_sl"]
            max_risk_per_trade = stop_loss - entry_price

        if max_risk_per_trade == 0:
            return 0  #  No risk, no trade

        # calculate position size 
        position_size = capital_at_risk * self.lev.value / max_risk_per_trade
        custom_stake = position_size * entry_price

        # Check if custom stake is smaller than default stake
        custom_stake = min(custom_stake, def_stake * self.lev.value)

        return custom_stake

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        
        entry_time = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        cur_time = timeframe_to_prev_date(self.timeframe, current_time)
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        signal_time = entry_time - timedelta(minutes=int(self.timeframe_minutes))
        signal_candle = dataframe.loc[dataframe['date'] == signal_time]

        if not signal_candle.empty:
            signal_candle = signal_candle.iloc[-1].squeeze()
            if trade.trade_direction == "long":
                if current_rate >= signal_candle["grid_buy10"]:
                    return "Final Grid Hit"
            else:
                if current_rate <= signal_candle["grid_sell10"]:
                    return "Final Grid Hit Short"

        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        # # kill trade if its older than 1440 minutes
        # if trade_duration > 1440:
        #     return "trade expired"
        
        return None

    def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)
        new_entryprice = dataframe['close'].iat[-1]

        return new_entryprice

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs) -> float:
        # Fetch the dataframe for the pair
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        trade_open_candle = dataframe.loc[dataframe['date'] == trade_date]
        entry_price = trade.open_rate

        if trade_open_candle.empty:
            return None
        
        if trade.trade_direction == "long":
            if trade.nr_of_successful_entries == 1:
                init_sl = stoploss_from_absolute((trade.open_rate - trade_open_candle['x_atr_sl'].values[0]), current_rate, trade.is_short, self.lev.value)
                if init_sl == 0:
                    logger.error(f"Stoploss is 0 for {trade.pair}_{trade.trade_direction}")
                else:
                    logger.info(f"Initial Stoploss for {trade.pair}_{trade.trade_direction} is {init_sl}")
                    return init_sl
                
            elif trade.nr_of_successful_entries > 1:
                for i in range(1, 11):
                    if trade.nr_of_successful_entries == i:
                        new_sl = stoploss_from_absolute((trade_open_candle[f"grid_buy{i-1}"].values[0] - trade_open_candle['x_atr_sl'].values[0]/2), current_rate, False, self.lev.value)
                        logger.info(f"Set new Stoploss for {trade.pair}_{trade.trade_direction} is {new_sl}")
                        return new_sl
        
        elif trade.trade_direction == "short":
            if trade.nr_of_successful_entries == 1:
                init_sl = stoploss_from_absolute((trade.open_rate + trade_open_candle['x_atr_sl'].values[0]), current_rate, trade.is_short, self.lev.value)
                if init_sl == 0:
                    logger.error(f"Stoploss is 0 for {trade.pair}_{trade.trade_direction}")
                else:
                    logger.info(f"Initial Stoploss for {trade.pair}_{trade.trade_direction} is {init_sl}")
                    return init_sl        
        
            elif trade.nr_of_successful_entries > 1:
                for i in range(1, 11):
                    if trade.nr_of_successful_entries == i:
                        new_sl = stoploss_from_absolute((trade_open_candle[f"grid_sell{i-1}"].values[0] + trade_open_candle['x_atr_sl'].values[0]/2), current_rate, True, self.lev.value)
                        logger.info(f"Set new Stoploss for {trade.pair}_{trade.trade_direction} is {new_sl}")
                        return new_sl  
        else:
            return None                
            
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs
                              ) -> Union[Optional[float], Tuple[Optional[float], Optional[str]]]:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        trade_open_candle = dataframe.loc[dataframe['date'] == trade_date]

        if trade_open_candle.empty:
            # logger.error(f"No trade open candle found for {trade.pair} at {trade_date}.")
            return None
        
        if trade.trade_direction == "long":
            for i in range(1, 11):
                if current_rate >= trade_open_candle[f"grid_buy{i}"].values[0] and trade.nr_of_successful_entries == i:
                    logger.info(f"Adding more to the position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    self.dp.send_msg(f"Adding more to the position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    return trade.stake_amount
        else:
            for i in range(1, 11):
                if current_rate <= trade_open_candle[f"grid_sell{i}"].values[0] and trade.nr_of_successful_entries == i:
                    logger.info(f"Adding more to the position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    self.dp.send_msg(f"Adding more to the position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    return trade.stake_amount        
  
def calculate_atr(df, atr_period=28):
    atr = ta.ATR(df, timeperiod=atr_period)
    # round atr to 5 decimals
    atr = atr.apply(lambda x: round(x, 4))
    return atr

def determine_session(time):
    hour = time.hour
    minute = time.minute

    if 23 <= hour or hour < 4:
        return 'asia'
    elif 7 <= hour < 9 or (hour == 9 and minute == 0):
        return 'entry_asia'
    elif 11 <= hour < 13 or (hour == 13 and minute <= 30):
        return 'ny_am_analysis'
    elif 13 <= hour < 15 or (hour == 13 and minute > 30):
        return 'ny_am_entry'
    elif hour == 15 and minute <= 30:
        return "ny_pause"
    elif 15 <= hour < 17 or (hour == 15 and minute >= 30) or (hour == 17 and minute <= 30):
        return 'ny_pm_analysis'
    elif 17 <= hour < 19 or (hour == 17 and minute > 30):
        return 'ny_pm_entry'
    else:
        return 'no session'

def twap(df, period):
    tp = (df['low'] + df['close'] + df['high']).divide(3)
    df_with_twap = df.assign(twap=(tp.rolling(period).sum().divide(period)))
    return df_with_twap['twap']  # Return only the 'twap' Series

def hodges_tompkins(price_data, window=30, trading_periods=252, clean=True):

    log_return = (price_data["close"] / price_data["close"].shift(1)).apply(np.log)

    vol = log_return.rolling(window=window, center=False).std() * math.sqrt(
        trading_periods
    )

    h = window
    n = (log_return.count() - h) + 1

    adj_factor = 1.0 / (1.0 - (h / n) + ((h ** 2 - 1) / (3 * n ** 2)))

    result = vol * adj_factor

    if clean:
        return result.dropna()
    else:
        return result
    
def rogers_satchell(price_data, window=30, trading_periods=252, clean=True):

    log_ho = (price_data["high"] / price_data["open"]).apply(np.log)
    log_lo = (price_data["low"] / price_data["open"]).apply(np.log)
    log_co = (price_data["close"] / price_data["open"]).apply(np.log)

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    def f(v):
        return (trading_periods * v.mean()) ** 0.5

    result = rs.rolling(window=window, center=False).apply(func=f)

    if clean:
        return result.dropna()
    else:
        return result