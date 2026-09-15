# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List, Optional
from functools import reduce
from pandas import DataFrame
import pandas_ta as pta
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



class EI3v2_tag_cofi_green(IStrategy):
    INTERFACE_VERSION = 2
    """
    # ROI table:
    minimal_roi = {
        "0": 0.08,
        "20": 0.04,
        "40": 0.032,
        "87": 0.016,
        "201": 0,
        "202": -1
    }
    """
    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy": 12,
        "rsi_buy": 58,
        "ewo_high": 3.001,
        "ewo_low": -10.289,
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
        "0": 9.99,
        
    }

    # Stoploss:
    stoploss = -0.85

    # SMAOffset
    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)
    low_offset = DecimalParameter(0.985, 0.995, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(1.005, 1.015, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.010, 1.020, default=sell_params['high_offset_2'], space='sell', optimize=True)

    # lambo2
    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3,  default=buy_params['lambo2_ema_14_factor'], space='buy', optimize=True)
    lambo2_rsi_4_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_4_limit'], space='buy', optimize=True)
    lambo2_rsi_14_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_14_limit'], space='buy', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    ewo_low = DecimalParameter(-20.0, -8.0,default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(3.0, 3.4, default=buy_params['ewo_high'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=False)

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.09
    trailing_only_offset_is_reached = True

    #cofi
    is_optimize_cofi = False
    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=0.97 , optimize = is_optimize_cofi)
    buy_fastk = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    buy_fastd = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    buy_adx = IntParameter(20, 30, default=30, optimize = is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize = is_optimize_cofi)

    entry_falling_x = DecimalParameter(0.95, 1.0, default= 0.98, decimals = 2, optimize=True, space='buy')
    entry_rising_x = DecimalParameter(0.95, 1.0, default= 0.98, decimals = 2, optimize=True, space='buy')
    

    # Sell signal
    use_exit_signal = False
    exit_profit_only = True
    exit_profit_offset = 0.001
    ignore_roi_if_entry_signal = False

    ## Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    # Optimal timeframe for the strategy
    timeframe = '15m'
    inf_1h = '1h'

    process_only_new_candles = True
    startup_candle_count = 400

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }
    
    #entry_falling_x = DecimalParameter(0.95, 1.0, default= 0.98, decimals = 2, optimize=True)
    #entry_rising_x = DecimalParameter(0.95, 1.0, default= 0.98, decimals = 2, optimize=True)
   
    #def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float,
    #                       entry_tag: Optional[str], side: str, **kwargs) -> float:

    #    dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
    #                                                            timeframe=self.timeframe)
    #    if dataframe['open'].iat[-1] > dataframe['close'].iat[-1]:
    #        new_entryprice = dataframe['close'].iat[-1] * self.entry_falling_x.value
    #    else:
    #       new_entryprice = dataframe['open'].iat[-1] * self.entry_rising_x.value
    #
    #    return new_entryprice
    

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        # Sell any positions at a loss if they are held for more than 4 hours.
        if current_profit < -0.40 and (current_time - trade.open_date_utc).total_seconds() >= 48 * 60 * 60:
            return 'unclog'



    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]

        if self.config['stake_currency'] in ['USDT:USDT','BUSD','USDC','DAI','TUSD','PAX','USD','EUR','GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT:USDT"

        informative_pairs.append((btc_info_pair, self.timeframe))
        informative_pairs.append((btc_info_pair, self.inf_1h))

        return informative_pairs

    def pump_dump_protection(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        df36h = dataframe.copy().shift( 432 ) # TODO FIXME: This assumes 5m timeframe
        df24h = dataframe.copy().shift( 288 ) # TODO FIXME: This assumes 5m timeframe

        dataframe['volume_mean_short'] = dataframe['volume'].rolling(4).mean()
        dataframe['volume_mean_long'] = df24h['volume'].rolling(48).mean()
        dataframe['volume_mean_base'] = df36h['volume'].rolling(288).mean()

        dataframe['volume_change_percentage'] = (dataframe['volume_mean_long'] / dataframe['volume_mean_base'])

        dataframe['rsi_mean'] = dataframe['rsi'].rolling(48).mean()

        dataframe['pnd_volume_warn'] = np.where((dataframe['volume_mean_short'] / dataframe['volume_mean_long'] > 5.0), -1, 0)

        return dataframe


    def base_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['price_trend_long'] = (dataframe['close'].rolling(8).mean() / dataframe['close'].shift(8).rolling(144).mean())

        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

        return dataframe

    def info_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['rsi_8'] = ta.RSI(dataframe, timeperiod=8)

        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.config['stake_currency'] in ['USDT:USDT','BUSD']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT:USDT"

        btc_info_tf = self.dp.get_pair_dataframe(btc_info_pair, self.inf_1h)
        btc_info_tf = self.info_tf_btc_indicators(btc_info_tf, metadata)
        dataframe = merge_informative_pair(dataframe, btc_info_tf, self.timeframe, self.inf_1h, ffill=True)
        drop_columns = [f"{s}_{self.inf_1h}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        btc_base_tf = self.dp.get_pair_dataframe(btc_info_pair, self.timeframe)
        btc_base_tf = self.base_tf_btc_indicators(btc_base_tf, metadata)
        dataframe = merge_informative_pair(dataframe, btc_base_tf, self.timeframe, self.timeframe, ffill=True)
        drop_columns = [f"{s}_{self.timeframe}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        # Keltner Channel
        #keltner = qtpylib.keltner_channel(dataframe, window=8)
        #dataframe['keltner_lowerband'] = keltner['lower']
        #dataframe['keltner_middleband'] = keltner['mid']
        #dataframe['keltner_upperband'] = keltner['upper']

        #dataframe["keltner_percent"] = (
        #    (dataframe["close"] - dataframe["keltner_lowerband"]) /
        #    (dataframe["keltner_upperband"] - dataframe["keltner_lowerband"])
        #)
        #dataframe["keltner_width"] = (
        #    (dataframe["keltner_upperband"] - dataframe["keltner_lowerband"]) / dataframe["keltner_middleband"]
        #)

        #bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2.2)
        #dataframe["bb_lowerband"] = bollinger["lower"]
        #dataframe["bb_middleband"] = bollinger["mid"]
        #dataframe["bb_upperband"] = bollinger["upper"]
        #dataframe["bb_width"] = (dataframe["bb_upperband"] -
        #                           dataframe["bb_lowerband"]) / dataframe["bb_middleband"]

        #dataframe['dpo'] = pta.dpo(dataframe['close'], 7, centered=False) 
        #dataframe['cci'] = ta.CCI(dataframe, timeperiod=28)
        #dataframe['cmo'] = pta.momentum.cmo(dataframe['close'], length=16)

        dataframe['ema_4'] = ta.EMA(dataframe, timeperiod=3)



        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)


        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        #lambo2
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)


        # Pump strength
        dataframe['zema_30'] = ftt.zema(dataframe, period=30)
        dataframe['zema_200'] = ftt.zema(dataframe, period=200)
        dataframe['pump_strength'] = (dataframe['zema_30'] - dataframe['zema_200']) / dataframe['zema_30']

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)

        #dataframe['sma3'] = ta.SMA(dataframe['close'], timeperiod=3)
        dataframe['sma2'] = ta.SMA(dataframe['close'], timeperiod=2)

        # Heikin Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        #dataframe['ha_sma2'] = ta.SMA(dataframe['ha_close'], timeperiod=4)

        dataframe = self.pump_dump_protection(dataframe, metadata)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''


        lambo2condition = (
            #bool(self.lambo2_enabled.value) &
            #(dataframe['pump_warning'] == 0) &
            (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value)) 
        )

        buy_window = 6


        laboactive = (lambo2condition.rolling(window=buy_window).max() > 0)

        #keltneractive = (qtpylib.crossed_above(dataframe['keltner_percent'], dataframe['keltner_width']))

        #keltnerrolling = (keltneractive.rolling(window=1).max() > 0)

        #periods_under_condition = ((dataframe['keltner_percent'] < dataframe['keltner_width'])).rolling(window=30).sum()

        # Trigger the call if 'keltner_percent' has been under 'keltner_width' for at least 4 periods
        #keltner_condition = (periods_under_condition >= 3)

        lambo2 = (
            laboactive &
            (dataframe['ha_open'] < dataframe['ha_close']) &
            #(dataframe['ha_close'] < dataframe['close']) &
            #(dataframe['ha_open'].shift(1) < dataframe['ha_close'].shift(1)) &
            #(dataframe['ha_open'].shift(2) > dataframe['ha_close'].shift(2)) &
            #(dataframe['keltner_width'] > dataframe['keltner_percent'])

            #(dataframe['ha_sma2'] < dataframe['close']) &
            #(qtpylib.crossed_above(dataframe['close'], dataframe['sma2'])) 
            #(dataframe['keltner_width'] > dataframe['keltner_percent']) &
            #(dataframe['keltner_width'] > dataframe['keltner_percent'].shift(1)) &
            #(dataframe['keltner_width'].shift(1) > dataframe['keltner_percent'].shift(2)) &
            #(dataframe['keltner_width'].shift(2) > dataframe['keltner_percent'].shift(3)) &
            #(dataframe["bb_lowerband"] < dataframe['close']) 
            #keltner_condition
            (dataframe['ema_4'] < dataframe['ha_close'])
            
            #(qtpylib.crossed_above(dataframe['keltner_percent2'], dataframe['keltner_width2'])) 
            #cmo_increasing 
            #(dataframe['cmo'] < -30)
            #(dataframe['keltner_percent'] > dataframe['keltner_width']) &
            #(dataframe['dpo'] < 0) 
            #(dataframe['cci'] < -150)
            #keltnerrolling
            #(dataframe['keltner_percent'].shift(1) < dataframe['keltner_width'].shift(1)) &
            #(dataframe['keltner_percent'].shift(2) < dataframe['keltner_width'].shift(2))
           
        )

        dataframe.loc[lambo2, 'buy_tag'] += 'lambo2_'
        conditions.append(lambo2)

        buy1ewocondition = (
                (dataframe['rsi_fast'] <35)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                #(qtpylib.crossed_above(dataframe['close'], blabla2)) & 
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) 
                #(dataframe['dpo'] < 0) &
                #(dataframe['cci'] < -180) 
                #(dataframe['[6/8]P'] > dataframe['close'])
        )

        buy1ewoactive = (buy1ewocondition.rolling(window=buy_window).max() > 0)

        buy1ewo = (
            buy1ewoactive &
            (dataframe['ha_open'] < dataframe['ha_close']) &
            #(dataframe['ha_close'] < dataframe['close']) &
            (dataframe['ema_4'] < dataframe['ha_close'])

            #(qtpylib.crossed_above(dataframe['close'], dataframe['sma2'])) &
            #(dataframe['keltner_width'] > dataframe['keltner_percent']) &
            #(dataframe['keltner_width'] > dataframe['keltner_percent'].shift(1)) &
            #(dataframe['keltner_width'].shift(1) > dataframe['keltner_percent'].shift(2)) &
            #(dataframe['keltner_width'].shift(2) > dataframe['keltner_percent'].shift(3)) &
            #(dataframe["bb_lowerband"] < dataframe['close'])
            #(qtpylib.crossed_above(dataframe['keltner_percent2'], dataframe['keltner_width2'])) 
            #(dataframe['keltner_percent'] > dataframe['keltner_width']) 
            #(dataframe['dpo'] < 0) &
            #(dataframe['cci'] < -150)
            #keltnerrolling
            #(dataframe['keltner_percent'].shift(1) < dataframe['keltner_width'].shift(1)) &
            #(dataframe['keltner_percent'].shift(2) < dataframe['keltner_width'].shift(2))
           
        )

        dataframe.loc[buy1ewo, 'buy_tag'] += 'buy1eworsi_'
        conditions.append(buy1ewo)

        buy2ewocondition = (
                (dataframe['rsi_fast'] < 35)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0)&
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) 
                #(dataframe['dpo'] < 0) &
                #(dataframe['cci'] < -180) 
                #(dataframe['[6/8]P'] > dataframe['close'])
        )

        buy2ewoactive = (buy2ewocondition.rolling(window=buy_window).max() > 0)

        buy2ewo = (
            buy2ewoactive &
            (dataframe['ha_open'] < dataframe['ha_close']) &
            #(dataframe['ha_close'] < dataframe['close']) &
            (dataframe['ema_4'] < dataframe['ha_close'])

            #(qtpylib.crossed_above(dataframe['close'], dataframe['sma2'])) &
            #(dataframe['keltner_width'] > dataframe['keltner_percent']) &
            #(dataframe['keltner_width'] > dataframe['keltner_percent'].shift(1)) &
            #(dataframe['keltner_width'].shift(1) > dataframe['keltner_percent'].shift(2)) &
            #(dataframe['keltner_width'].shift(2) > dataframe['keltner_percent'].shift(3)) &
            #(dataframe["bb_lowerband"] < dataframe['close'])
            #(qtpylib.crossed_above(dataframe['keltner_percent2'], dataframe['keltner_width2'])) 
            #(dataframe['keltner_percent'] > dataframe['keltner_width']) &
            #(dataframe['dpo'] < 0) &
            #(dataframe['cci'] < -150)
            #keltnerrolling
            #(dataframe['keltner_percent'].shift(1) < dataframe['keltner_width'].shift(1)) &
            #(dataframe['keltner_percent'].shift(2) < dataframe['keltner_width'].shift(2))
           
        )

        dataframe.loc[buy2ewo, 'buy_tag'] += 'buy2ewo_'
        conditions.append(buy2ewo)

        is_coficondition = (
                (dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value) &
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['fastk'] < self.buy_fastk.value) &
                (dataframe['fastd'] < self.buy_fastd.value) &
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['EWO'] > self.buy_ewo_high.value) 
                #(dataframe['dpo'] < 0) &
                #(dataframe['cci'] < -150) 
                #(dataframe['[6/8]P'] > dataframe['close'])
            )

        is_cofiactive = (is_coficondition.rolling(window=buy_window).max() > 0)

        is_cofi = (
            is_cofiactive &
            (dataframe['ha_open'] < dataframe['ha_close']) &
            #(dataframe['ha_close'] < dataframe['close']) &
            (dataframe['ema_4'] < dataframe['ha_close'])
            #(qtpylib.crossed_above(dataframe['close'], dataframe['sma2'])) &
            #(dataframe['keltner_width'] > dataframe['keltner_percent']) &
            #(dataframe['keltner_width'] > dataframe['keltner_percent'].shift(1)) &
            #(dataframe['keltner_width'].shift(1) > dataframe['keltner_percent'].shift(2)) &
            #(dataframe['keltner_width'].shift(2) > dataframe['keltner_percent'].shift(3)) &
            #(dataframe["bb_lowerband"] < dataframe['close'])
            #(qtpylib.crossed_above(dataframe['keltner_percent2'], dataframe['keltner_width2'])) 
            #(dataframe['keltner_percent'] > dataframe['keltner_width']) &
            #(dataframe['dpo'] < 0) &
            #(dataframe['cci'] < -150)

            #keltnerrolling
            #(dataframe['keltner_percent'].shift(1) < dataframe['keltner_width'].shift(1)) &
            #(dataframe['keltner_percent'].shift(2) < dataframe['keltner_width'].shift(2))
           
        )

        dataframe.loc[is_cofi, 'buy_tag'] += 'cofi_'
        conditions.append(is_cofi)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ]=1


        dont_buy_conditions = []

        # don't buy if there seems to be a Pump and Dump event.
        dont_buy_conditions.append((dataframe['pnd_volume_warn'] < 0.0))

        # BTC price protection
        dont_buy_conditions.append((dataframe['btc_rsi_8_1h'] < 35.0))

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'buy'] = 0

        return dataframe



    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            #(   (dataframe['close']>dataframe['hma_50'])&
            #    (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
            #    (dataframe['rsi']>50)&
            #    (dataframe['volume'] > 0)&
            #    (dataframe['rsi_fast']>dataframe['rsi_slow'])
#
            #)
            #|
            #(
            #    (dataframe['close']<dataframe['hma_50'])&
            #    (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
            #    (dataframe['volume'] > 0)&
            #    (dataframe['rsi_fast']>dataframe['rsi_slow'])
            #)
            #|
            #(
            #    (qtpylib.crossed_below(dataframe['keltner_percent'], dataframe['keltner_width'])) 
            #)
            #|
            (
                (dataframe['ha_open'] > dataframe['ha_close']) 
            )

        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1


        return dataframe

    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        trade.exit_reason = exit_reason + "_" + trade.buy_tag

        return True

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 13.0

def pct_change(a, b):
    return (b - a) / a

