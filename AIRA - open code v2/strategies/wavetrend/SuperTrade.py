from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from datetime import datetime

class SuperTrade(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'  # 使用5分钟时间框架
    can_short = True
    use_exit_signal = True  # 使用退出信号
    exit_profit_only = True

    # ROI table:
    minimal_roi = {}

    # Stoploss:
    stoploss = -0.296  # This is the stop loss percentage

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = True

    # Max Open Trades:
    max_open_trades = -1

    # Parameters
    # WaveTrend Parameters
    wt_channel_length = 9
    wt_average_length = 21
    wt_ob_level1 = 60
    wt_os_level1 = -60

    # Money Flow Parameters
    fast_money_flow_length = 9
    slow_money_flow_length = 10

    # RSI Parameters
    rsi_length = 14
    rsi_overbought = 70
    rsi_oversold = 30

    # Stochastic RSI Parameters
    stochastic_rsi_length = 14
    stochastic_k_length = 3
    stochastic_d_length = 3

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        return 10.0  # 设置杠杆为10倍

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 验证数据框是否包含所需的列
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        missing_columns = required_columns - set(dataframe.columns)
        if missing_columns:
            self.log(
                f"Dataframe missing required columns: {missing_columns} for pair: {metadata['pair']}",
                level="error",
            )
            return dataframe

        # 如果数据框为空，跳过指标计算
        if dataframe.empty:
            self.log(f"Empty dataframe for pair: {metadata['pair']} at {self.timeframe}", level="warning")
            return dataframe

        # WaveTrend Calculation
        esa = ta.EMA(dataframe['close'], timeperiod=self.wt_channel_length)
        d = ta.EMA(np.abs(dataframe['close'] - esa), timeperiod=self.wt_channel_length)
        ci = (dataframe['close'] - esa) / (0.015 * d)
        tci = ta.EMA(ci, timeperiod=self.wt_average_length)
        dataframe['wt1'] = tci
        dataframe['wt2'] = ta.SMA(dataframe['wt1'], timeperiod=2)

        # Money Flow Calculation
        raw_money_flow = (2 * ta.SMA(dataframe['close'] - ta.SMA(dataframe['close'], self.fast_money_flow_length), self.fast_money_flow_length)) / ta.SMA(dataframe['high'] - dataframe['low'], self.fast_money_flow_length)
        dataframe['money_flow'] = raw_money_flow * 5  # Multiplier for Y position

        raw_money_flow_slow = (2 * ta.SMA(dataframe['close'] - ta.SMA(dataframe['close'], self.slow_money_flow_length), self.slow_money_flow_length)) / ta.SMA(dataframe['high'] - dataframe['low'], self.slow_money_flow_length)
        dataframe['money_flow_slow'] = raw_money_flow_slow * 5  # Multiplier for Y position

        # RSI Calculation
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.rsi_length)

        # Stochastic RSI Calculation
        stoch_rsi = ta.STOCHF(dataframe, fastk_period=self.stochastic_rsi_length, fastd_period=self.stochastic_k_length)
        dataframe['stochastic_d'] = stoch_rsi['fastd']

        # WaveTrend Divergence Signals
        dataframe['bullish_divergence'] = (dataframe['wt1'] < self.wt_os_level1) & (dataframe['close'] < dataframe['close'].shift(1))
        dataframe['bearish_divergence'] = (dataframe['wt1'] > self.wt_ob_level1) & (dataframe['close'] > dataframe['close'].shift(1))

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe

        dataframe.loc[
            (
                (dataframe['bullish_divergence']) |
                (dataframe['wt1'] < self.wt_os_level1)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Bullish Divergence or WaveTrend Oversold')

        dataframe.loc[
            (
                (dataframe['bearish_divergence']) |
                (dataframe['wt1'] > self.wt_ob_level1)
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Bearish Divergence or WaveTrend Overbought')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe

        dataframe.loc[
            (dataframe['wt1'] > 0),
            ['exit_long', 'exit_tag']
        ] = (1, 'WaveTrend Neutral')

        dataframe.loc[
            (dataframe['wt1'] < 0),
            ['exit_short', 'exit_tag']
        ] = (1, 'WaveTrend Neutral')

        return dataframe
    