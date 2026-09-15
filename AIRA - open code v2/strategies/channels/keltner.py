import logging
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
from pandas import DataFrame
import pandas_ta as ta

class keltner(IStrategy):
    
    # Define the timeframe for your strategy
    timeframe = '15m'

    keltner_multiplier = 2.0
    overbought = 1
    oversold = 0
    custom_timeframe = False
    high_timeframe = '60m'
    startup_candle_count = 720
    process_only_new_candles = True
    can_short = True
    leverage_value = 20  # Set leverage for the strategy

    # Configuration parameters
    moving_average_length = IntParameter(low=10, high=50, default=47, space='buy', optimize=True, load=True)
    atr_multiplier_min = DecimalParameter(low=1.1, high=1.19, default=1.86, space='buy', optimize=True, load=True)
    atr_multiplier_max = DecimalParameter(low=2.1, high=4.5, default=3.813, space='buy', optimize=True, load=True)
    atr_length = IntParameter(low=60, high=95, default=84, space='buy', optimize=True, load=True)
    keltner_length = IntParameter(low=18, high=40, default=24, space='buy', optimize=True, load=True)
   
    # Configuration parameters for short positions
    short_moving_average_length = IntParameter(low=10, high=50, default=47, space='sell', optimize=True, load=True)
    short_atr_multiplier_min = DecimalParameter(low=1.1, high=1.19, default=1.86, space='sell', optimize=True, load=True)
    short_atr_multiplier_max = DecimalParameter(low=2.1, high=4.5, default=3.813, space='sell', optimize=True, load=True)
    short_atr_length = IntParameter(low=60, high=95, default=84, space='sell', optimize=True, load=True)
    short_keltner_length = IntParameter(low=18, high=40, default=24, space='sell', optimize=True, load=True)

    # Buy hyperspace params:
    buy_params = {
        "atr_length": 61,
        "atr_multiplier_max": 2.825,
        "atr_multiplier_min": 1.135,
        "keltner_length": 26,
        "moving_average_length": 23,
    }

    # Sell hyperspace params:
    sell_params = {
        "short_atr_length": 92,
        "short_atr_multiplier_max": 3.779,
        "short_atr_multiplier_min": 1.147,
        "short_keltner_length": 40,
        "short_moving_average_length": 30,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.276,
        "93": 0.06,
        "158": 0.042,
        "255": 0
    }

    # Stoploss:
    stoploss = -0.24

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate the moving average (EMA)
        dataframe['ema'] = ta.ema(dataframe['close'], self.moving_average_length.value)

        # Calculate ATR
        dataframe['atr'] = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], self.atr_length.value)

        # Calculate Keltner Channels
        dataframe['kc_upper_min'] = dataframe['ema'] + dataframe['atr'] * self.atr_multiplier_min.value
        dataframe['kc_lower_min'] = dataframe['ema'] - dataframe['atr'] * self.atr_multiplier_min.value
        dataframe['kc_upper_max'] = dataframe['ema'] + dataframe['atr'] * self.atr_multiplier_max.value
        dataframe['kc_lower_max'] = dataframe['ema'] - dataframe['atr'] * self.atr_multiplier_max.value

        # Standard deviation and Keltner basis
        dataframe['stddev'] = dataframe['close'].rolling(window=self.keltner_length.value).std()
        dataframe['kc_basis'] = dataframe['ema']

        # Calculate upper and lower bands for Keltner Channel
        dataframe['kc_upper'] = dataframe['kc_basis'] + (dataframe['stddev'] * self.keltner_multiplier)
        dataframe['kc_lower'] = dataframe['kc_basis'] - (dataframe['stddev'] * self.keltner_multiplier)

        # Bollinger Bands ratio
        dataframe['bbr'] = (dataframe['close'] - dataframe['kc_lower']) / (dataframe['kc_upper'] - dataframe['kc_lower'])
        
        # Short indicators
        dataframe['short_ema'] = ta.ema(dataframe['close'], self.short_moving_average_length.value)
        dataframe['short_atr'] = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], self.short_atr_length.value)
        dataframe['short_kc_upper_min'] = dataframe['short_ema'] + dataframe['short_atr'] * self.short_atr_multiplier_min.value
        dataframe['short_kc_lower_min'] = dataframe['short_ema'] - dataframe['short_atr'] * self.short_atr_multiplier_min.value
        dataframe['short_kc_upper_max'] = dataframe['short_ema'] + dataframe['short_atr'] * self.short_atr_multiplier_max.value
        dataframe['short_kc_lower_max'] = dataframe['short_ema'] - dataframe['short_atr'] * self.short_atr_multiplier_max.value
        dataframe['short_stddev'] = dataframe['close'].rolling(window=self.short_keltner_length.value).std()
        dataframe['short_kc_basis'] = dataframe['short_ema']
        dataframe['short_kc_upper'] = dataframe['short_kc_basis'] + (dataframe['short_stddev'] * self.keltner_multiplier)
        dataframe['short_kc_lower'] = dataframe['short_kc_basis'] - (dataframe['short_stddev'] * self.keltner_multiplier)
        dataframe['short_bbr'] = (dataframe['close'] - dataframe['short_kc_lower']) / (dataframe['short_kc_upper'] - dataframe['short_kc_lower'])

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long entry
        dataframe.loc[
            (
                (dataframe['bbr'].shift(1) < self.oversold) & 
                (dataframe['bbr'] > self.oversold) & 
                (dataframe['close'] < dataframe['kc_basis'])
            ),
            'enter_long'] = 1

        # Short entry
        dataframe.loc[
            (
                (dataframe['short_bbr'].shift(1) > self.overbought) & 
                (dataframe['short_bbr'] < self.overbought) & 
                (dataframe['close'] > dataframe['short_kc_basis'])
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['short_bbr'].shift(1) > self.overbought) & 
                (dataframe['short_bbr'] < self.overbought) & 
                (dataframe['close'] > dataframe['short_kc_basis'])
            ),
            'exit_long'] = 1
        dataframe.loc[
            (
                (dataframe['bbr'].shift(1) < self.oversold) & 
                (dataframe['bbr'] > self.oversold) & 
                (dataframe['close'] < dataframe['kc_basis'])
            ),
            'exit_short'] = 1

        return dataframe

    def customize_trade(self, pair: str, trade_type: str, current_time, current_rate: float,
                        proposed_leverage: float, max_leverage: float, *args, **kwargs) -> float:
        leverage = min(self.leverage_value, max_leverage)
        logging.info(f"Applying leverage: {leverage}")
        return leverage
