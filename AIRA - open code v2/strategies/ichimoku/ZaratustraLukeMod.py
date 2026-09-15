# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
from freqtrade.constants import Config
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative, IntParameter
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from datetime import datetime, timedelta
from pandas import DataFrame
from typing import Dict, List, Optional, Union, Tuple
import talib.abstract as ta
from technical import qtpylib
from freqtrade.strategy import IntParameter, DecimalParameter
import requests

    
class ZaratustraLukeMod(IStrategy):
    # Parameters
    INTERFACE_VERSION = 3
    timeframe = '5m'
    
    can_short = True
    use_exit_signal = False
    exit_profit_only = False
        

        # Parametri Ichimoku
    ichimoku_tenkan_period = 9
    ichimoku_kijun_period = 26
    ichimoku_senkou_period = 52
    ichimoku_chikou_shift = -26




    # ROI table:
    minimal_roi = {
        "1": 0.25,  # Prendi profitto al 5%
        "3": 0.3,  # Dopo 30 min abbassa il take profit al 3%
        "5": 0.5,   # Dopo 60 min abbassa il take profit al 2%
        "15": 0.6,   # Dopo 60 min abbassa il take profit al 2%
        
    }
    # Stoploss:
    stoploss = -0.15


    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.15
    trailing_only_offset_is_reached = True



    def leverage(self, pair: str, current_time: "datetime", current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return 10
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['dx']  = ta.DX(dataframe)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['pdi'] = ta.PLUS_DI(dataframe)
        dataframe['mdi'] = ta.MINUS_DI(dataframe)
        dataframe[['bbl', 'bbm', 'bbu']] = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)[['lower', 'mid', 'upper']]


               # WMAs ( Weighted Moving Averages ltf)
        dataframe['wma25'] = ta.WMA(dataframe, timeperiod=25)
        dataframe['wma100'] = ta.WMA(dataframe, timeperiod=100)

               # SMAs (Simple Moving Averages)
        dataframe['sma150'] = ta.SMA(dataframe, timeperiod=150)
        dataframe['sma3'] = ta.SMA(dataframe, timeperiod=3)
        dataframe['sma25'] = ta.SMA(dataframe, timeperiod=25)
               

               # WMAs ( Weighted Moving Averages htf)
        dataframe['wma140'] = ta.WMA(dataframe, timeperiod=140)
        dataframe['wma200'] = ta.WMA(dataframe, timeperiod=200)

               
                # RSI (Relative Strength Index)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
    
                    # Calcola i componenti dell'Ichimoku Cloud
        dataframe['tenkan_sen'] = ta.SMA(dataframe, timeperiod=self.ichimoku_tenkan_period)
        dataframe['kijun_sen'] = ta.SMA(dataframe, timeperiod=self.ichimoku_kijun_period)
        dataframe['senkou_span_a'] = ((dataframe['tenkan_sen'] + dataframe['kijun_sen']) / 2).shift(self.ichimoku_kijun_period)
        dataframe['senkou_span_b'] = ta.SMA(dataframe, timeperiod=self.ichimoku_senkou_period).shift(self.ichimoku_kijun_period)
        dataframe['chikou_span'] = dataframe['close'].shift(self.ichimoku_chikou_shift)

        # Definisce la nuvola (Cloud)
        dataframe['cloud_green'] = dataframe['senkou_span_a'] > dataframe['senkou_span_b']
        dataframe['cloud_red'] = dataframe['senkou_span_a'] < dataframe['senkou_span_b']

                # KAMAs (Kaufman Moving Average Adaptive)
        dataframe['kama50'] = ta.KAMA(dataframe, timeperiod=50)
        dataframe['kama25'] = ta.KAMA(dataframe, timeperiod=25)
        dataframe['kama200'] = ta.KAMA(dataframe, timeperiod=200)

        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        dataframe['prev_low'] = dataframe['low'].shift(1)
        dataframe['prev_high'] = dataframe['high'].shift(1)

    # Higher Low: il minimo attuale è maggiore del minimo precedente
        dataframe['higher_low'] = dataframe['low'] > dataframe['prev_low']
    # Higher High: il massimo attuale è maggiore del massimo precedente
        dataframe['higher_High'] = dataframe['high'] > dataframe['prev_high']
    # Lower Low: il minimo attuale è minore del minimo precedente
        dataframe['lower_Low'] = dataframe['low'] < dataframe['prev_low'] 
    # Lower High: il massimo attuale è minore del massimo precedente
        dataframe['lower_high'] = dataframe['high'] < dataframe['prev_high']  

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        dataframe['drop'] = (dataframe['close'] - dataframe['open']) / dataframe['open']

         

        




        
        return dataframe
    

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

                (dataframe['close'] > dataframe['kama50']) &
                (dataframe['wma25'] > dataframe['wma100']) &  # wma25 sopra wma100
                (dataframe['sma3'] > dataframe['wma140']) &
                #(dataframe['sma3'] > dataframe['sma150']) &
                (dataframe['close'] > dataframe[['senkou_span_a', 'senkou_span_b']].max(axis=1)) &  # Prezzo sopra la nuvola
                #(dataframe['tenkan_sen'] > dataframe['kijun_sen'])&   # Tenkan-sen > Kijun-sen
                (dataframe['close'] > dataframe['hma_50']) &
                (dataframe['low'] > dataframe['prev_low'])|
                (dataframe['drop'] > 0.004) 

            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Long DI enter')


        dataframe.loc[
            (

                
                (dataframe['close'] < dataframe['kama50']) &
                (dataframe['wma25'] < dataframe['wma100']) &  # wma25 sopra wma100
                (dataframe['sma3'] < dataframe['wma140']) &
                #(dataframe['sma3'] < dataframe['sma150']) &
                (dataframe['close'] < dataframe[['senkou_span_a', 'senkou_span_b']].min(axis=1)) &  # Prezzo sotto la nuvola
                #(dataframe['tenkan_sen'] < dataframe['kijun_sen'])&  # Tenkan-sen < Kijun-sen
                (dataframe['close'] < dataframe['hma_50']) &
                (dataframe['high'] < dataframe['prev_high'])|
                (dataframe['drop'] < -0.004) 


            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Short DI enter')

        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 75)
                #(dataframe['drop'] < -0.018) 
                

            ),
            ['exit_long', 'enter_tag']
        ] = (1, 'Long DI exit')


        dataframe.loc[
            (

                (dataframe['rsi'] < 20)
                #(dataframe['drop'] > 0.018) 
               
            ),
            ['exit_short', 'enter_tag']
        ] = (1, 'Short DI exit')
        

        
        return dataframe