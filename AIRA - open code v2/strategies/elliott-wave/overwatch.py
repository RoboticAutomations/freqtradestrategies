# --- Do not remove these libs ---
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
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_minutes

# Buy hyperspace params:
buy_params = {
      "base_nb_candles_buy": 46,
      "ewo_high": 2.239,
      "ewo_low": -8.029,
      "low_offset": 0.938,
      "rsi_buy": 64
    }

# Sell hyperspace params:
sell_params = {
      "base_nb_candles_sell": 49,
      "high_offset": 1.019
    }

def EWO(dataframe, ema_length=5, ema2_length=30):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

class overwatch(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        "0": 0.078,
        "13": 0.044,
        "60": 0.016,
        "162": 0
    }

    # Stoploss:
    stoploss = -0.329

    # SMAOffset
    base_nb_candles_buy = IntParameter(5, 80, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(0.99, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0, default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)
    ### BTC 
    fast_1h = IntParameter(3, 8, default=3, space='buy', optimize=True)
    slow_1h = IntParameter(25, 40, default=39, space='buy', optimize=True)


    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

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
    timeframe = '1h'
    informative_timeframe = '1d'

    process_only_new_candles = True
    startup_candle_count = 79

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    # Define the parameter spaces
    
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
    maxdrawdown_allowed_drawdown = DecimalParameter(0.01, 0.10, default=0.0, space="protection", optimize=True)


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

    
    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]

        return informative_pairs
    
    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.informative_timeframe)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Elliot
        #dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'], timeperiod=5)

        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            
        dataframe.loc[
            (
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO above high')


        dataframe.loc[
            (
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO below low')

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0)
            ),
            ['exit_long', 'exit_tag']] = (1, 'Close > MA-Sell')

        return dataframe

