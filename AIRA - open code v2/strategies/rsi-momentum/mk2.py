import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from datetime import datetime
from typing import Optional
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (
    IStrategy, IntParameter
)


class mk2(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = '5m'
    can_short = True
    process_only_new_candles = True
    use_custom_exit = True

    minimal_roi = {
        "0": 0.128,
        "16": 0.102,
        "31": 0.078,
        "46": 0.046,
        "61": 0.038,
    }

    stoploss = -0.213
    trailing_stop = True
    trailing_stop_positive = 0.035
    trailing_stop_positive_offset = 0.128
    trailing_only_offset_is_reached = True

    max_stake_per_trade = 100
    max_portfolio_percentage_per_trade = 0.05
    max_entry_position_adjustment = 3
    process_only_new_candles = True
    max_dca_orders = 3
    max_total_stake_per_pair = 250
    max_single_dca_amount = 50

    buy_params = {
        "rsi_entry_long": 41,
        "rsi_entry_short": 59,
        "window": 24,
    }

    sell_params = {
        "rsi_exit_long": 17,
        "rsi_exit_short": 83,
    }

    max_open_trades = 20

    rsi_entry_long = IntParameter(0, 100, default=buy_params.get('rsi_entry_long'), space='buy', optimize=True)
    rsi_exit_long = IntParameter(0, 100, default=sell_params.get('rsi_exit_long'), space='sell', optimize=True)
    rsi_entry_short = IntParameter(0, 100, default=buy_params.get('rsi_entry_short'), space='buy', optimize=True)
    rsi_exit_short = IntParameter(0, 100, default=sell_params.get('rsi_exit_short'), space='sell', optimize=True)
    window = IntParameter(5, 100, default=buy_params.get('window'), space='buy', optimize=False)

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        return 5

    @property
    def protections(self):
        return [{
            "method": "CooldownPeriod",
            "stop_duration_candles": 12,
            "protection_per_coin": True
        }]

    @property
    def plot_config(self):
        return {
            'main_plot': {
                'ema_fast': {'color': 'orange'},
                'ema_slow': {'color': 'pink'},
                'ema_long': {'color': 'blue'},
                'rsi_ema': {},
            },
            'subplots': {
                'MACD': {
                    'macd': {'color': 'orange'},
                    'macdsignal': {'color': 'pink'},
                },
                'Misc': {
                    'rsi': {},
                    'rsi_gra': {},
                    'mom': {},
                    'plus_di': {},
                    'minus_di': {},
                },
            }
        }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_ema'] = dataframe['rsi'].ewm(span=self.window.value).mean()
        dataframe['rsi_gra'] = np.gradient(dataframe['rsi_ema'])

        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=100)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        dataframe['macd_golden_cross'] = qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']).astype(int)
        dataframe['macd_dead_cross'] = qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal']).astype(int)

        # 保留DI和动量指标计算（出场信号仍可能用）
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=10)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['macd_slope'] = np.gradient(dataframe['macd'])

        # 多头开仓：保留RSI和梯度判断，增加MACD斜率判断和macd<macdsignal（金叉前），去掉EMA和DI判断，保留mom判断
        cond_long = (
            (dataframe['rsi'] < self.rsi_entry_long.value) &
            qtpylib.crossed_above(dataframe['rsi_gra'], 0) &
            (dataframe['macd_slope'] > 0) &
            (dataframe['macd'] < dataframe['macdsignal'])
        )
        dataframe.loc[cond_long, 'enter_long'] = 1

        # 空头开仓：保留RSI和梯度判断，增加MACD斜率判断和macd>macdsignal（死叉前），去掉EMA和DI判断，保留mom判断
        cond_short = (
            (dataframe['rsi'] > self.rsi_entry_short.value) &
            qtpylib.crossed_below(dataframe['rsi_gra'], 0) &
            (dataframe['macd_slope'] < 0) &
            (dataframe['macd'] > dataframe['macdsignal']) 
        )
        dataframe.loc[cond_short, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        dataframe.loc[
            (
                (dataframe['rsi'] > self.rsi_exit_long.value) &
                qtpylib.crossed_below(dataframe['rsi_gra'], 0) &
                (dataframe['low'] < dataframe['low'].rolling(window=20).min())
            ),
            'exit_long'
        ] = 1

        dataframe.loc[
            (
                (dataframe['rsi'] < self.rsi_exit_short.value) &
                qtpylib.crossed_above(dataframe['rsi_gra'], 0) &
                (dataframe['high'] > dataframe['high'].rolling(window=20).max())
            ),
            'exit_short'
        ] = 1

        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[str]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) < 1:
            return None

        last_candle = dataframe.iloc[-1]
        opened_at = trade.open_date_utc
        df_since_entry = dataframe[dataframe['date'] >= opened_at]

        if trade.trade_direction == 'long':
            had_golden_cross = df_since_entry['macd_golden_cross'].sum() > 0
            if (
                had_golden_cross and
                last_candle['macdsignal'] > last_candle['macd'] and
                last_candle['ema_slow'] > last_candle['ema_fast'] and
                last_candle['rsi'] < 36
            ):
                return "exit_long_custom"

        if trade.trade_direction == 'short':
            had_dead_cross = df_since_entry['macd_dead_cross'].sum() > 0
            if (
                had_dead_cross and
                last_candle['macd'] > last_candle['macdsignal'] and
                last_candle['ema_fast'] > last_candle['ema_slow'] and
                last_candle['rsi'] > 64
            ):
                return "exit_short_custom"

        return None
