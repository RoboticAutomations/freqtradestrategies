from freqtrade.strategy import IStrategy
from freqtrade.strategy.hyper import IntParameter, NumericParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from datetime import datetime

class SuperTrade(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = True
    use_exit_signal = True
    exit_profit_only = True

    # ROI table:
    minimal_roi = {}

    # Stoploss:
    stoploss = -0.296

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = True

    # Max Open Trades:
    max_open_trades = -1

    # Hyperoptable parameters
    wt_channel_length = IntParameter(6, 20, default=9, space='buy')
    wt_average_length = IntParameter(10, 30, default=21, space='buy')
    wt_ob_level1 = NumericParameter(50, 80, default=60, space='sell')
    wt_os_level1 = NumericParameter(-80, -50, default=-60, space='buy')
    fast_money_flow_length = IntParameter(5, 15, default=9, space='buy')
    slow_money_flow_length = IntParameter(5, 15, default=10, space='sell')
    rsi_length = IntParameter(10, 20, default=14, space='buy')
    rsi_overbought = NumericParameter(65, 85, default=70, space='sell')
    rsi_oversold = NumericParameter(15, 35, default=30, space='buy')

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        return 10.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Ensure all required columns are available
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        missing_columns = required_columns - set(dataframe.columns)
        if missing_columns:
            self.log(f"Dataframe missing columns: {missing_columns}", level="error")
            return dataframe

        # Calculate WaveTrend
        esa = ta.EMA(dataframe['close'], timeperiod=self.wt_channel_length.value)
        d = ta.EMA(np.abs(dataframe['close'] - esa), timeperiod=self.wt_channel_length.value)
        ci = (dataframe['close'] - esa) / (0.015 * d)
        tci = ta.EMA(ci, timeperiod=self.wt_average_length.value)
        dataframe['wt1'] = tci
        dataframe['wt2'] = ta.SMA(dataframe['wt1'], timeperiod=2)

        # Money Flow
        dataframe['money_flow'] = (2 * ta.SMA(dataframe['close'], self.fast_money_flow_length.value)) / \
                                  ta.SMA(dataframe['high'] - dataframe['low'], self.fast_money_flow_length.value)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.rsi_length.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['wt1'] < self.wt_os_level1.value)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'WaveTrend Oversold')

        dataframe.loc[
            (
                (dataframe['wt1'] > self.wt_ob_level1.value)
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'WaveTrend Overbought')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['wt1'] > 0),
            ['exit_long', 'exit_tag']
        ] = (1, 'WaveTrend Neutral')

        dataframe.loc[
            (dataframe['wt1'] < 0),
            ['exit_short', 'exit_tag']
        ] = (1, 'WaveTrend Neutral')

        return dataframe