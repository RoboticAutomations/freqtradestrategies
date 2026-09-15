"""
###########################
freqtrade hyperopt --hyperopt-loss SharpeHyperOptLossDaily --strategy mk4 --spaces buy sell roi stoploss trailing \
    --config user_data/config_binance_futures_backtest_informative.json --epochs 1000 --timerange 20250616-20250715 --timeframe-detail 5m  --max-open-trades 3

# 838/1000: 179 trades. 102/9/68 Wins/Draws/Losses. Avg profit   1.27%. Median profit   0.68%. Total profit 992.39404689 USDC (  99.24%). Avg duration 3:12:00 min. Objective: -6.57654

    # Buy hyperspace params:
    buy_params = {
        "rsi_entry_long": 57,
        "rsi_entry_short": 24,
        "window": 48,
    }

    # Sell hyperspace params:
    sell_params = {
        "rsi_exit_long": 11,
        "rsi_exit_short": 66,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.121,
        "108": 0.096,
        "246": 0.033,
        "276": 0
    }

    # Stoploss:
    stoploss = -0.189

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.149
    trailing_stop_positive_offset = 0.171
    trailing_only_offset_is_reached = False
    

    # Max Open Trades:
    max_open_trades = 3  # value loaded from strategy

###########################
freqtrade hyperopt --hyperopt-loss SharpeHyperOptLossDaily --strategy mk4 --spaces buy sell roi stoploss trailing \
    --config user_data/config_binance_futures_backtest_informative.json --epochs 1000 --timerange 20250616-20250715 --timeframe-detail 5m  --max-open-trades 3 --timeframe 1h

# 292/1000: 12 trades. 9/0/3 Wins/Draws/Losses. Avg profit   7.03%. Median profit   9.28%. Total profit 314.77176421 USDC (  31.48%). Avg duration 9:55:00 min. Objective: -7.96761

    # Buy hyperspace params:
    buy_params = {
        "rsi_entry_long": 55,
        "rsi_entry_short": 90,
        "window": 93,
    }

    # Sell hyperspace params:
    sell_params = {
        "rsi_exit_long": 44,
        "rsi_exit_short": 59,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.181,
        "447": 0.12,
        "1106": 0.06,
        "1557": 0
    }

    # Stoploss:
    stoploss = -0.122

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.217
    trailing_stop_positive_offset = 0.292
    trailing_only_offset_is_reached = True
    
    # Max Open Trades:
    max_open_trades = 3  # value loaded from strategy

###########################
####### BEST parameters 
###########################

freqtrade hyperopt --hyperopt-loss SharpeHyperOptLossDaily --strategy mk4 --spaces buy sell roi stoploss trailing \
    --config user_data/config_binance_futures_backtest_informative.json --epochs 1000 --timerange 20250616-20250715 --timeframe-detail 5m  --max-open-trades 3 --timeframe 4h

# 180/1000: 24 trades. 20/0/4 Wins/Draws/Losses. Avg profit   9.63%. Median profit   8.04%. Total profit 1039.63690468 USDC ( 103.96%). Avg duration 1 day, 2:49:00 min. Objective: -10.51142

    # Buy hyperspace params:
    buy_params = {
        "rsi_entry_long": 71,
        "rsi_entry_short": 17,
        "window": 59,
    }

    # Sell hyperspace params:
    sell_params = {
        "rsi_exit_long": 92,
        "rsi_exit_short": 48,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.495,
        "1172": 0.206,
        "3708": 0.064,
        "8031": 0
    }

    # Stoploss:
    stoploss = -0.19

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.112
    trailing_stop_positive_offset = 0.169
    trailing_only_offset_is_reached = True
    

    # Max Open Trades:
    max_open_trades = 3  # value loaded from strategy

###########################
"""
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


class mk4(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = '15m'
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
    trailing_stop_positive = 0.028
    trailing_stop_positive_offset = 0.128
    trailing_only_offset_is_reached = True

    max_stake_per_trade = 100
    max_portfolio_percentage_per_trade = 0.05
    max_entry_position_adjustment = 3
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
    window = IntParameter(0, 100, default=buy_params.get('window'), space='buy', optimize=True)

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

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        return 5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_ema'] = dataframe['rsi'].ewm(span=self.window.value).mean()
        dataframe['rsi_gra'] = dataframe['rsi_ema'].diff().ewm(span=3).mean()

        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=100)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        dataframe['macd_golden_cross'] = qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']).astype(int)
        dataframe['macd_dead_cross'] = qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal']).astype(int)

        # 计算KDJ指标
        low_min = dataframe['low'].rolling(window=9).min()
        high_max = dataframe['high'].rolling(window=9).max()
        rsv = (dataframe['close'] - low_min) / (high_max - low_min) * 100
        dataframe['kdj_k'] = rsv.ewm(com=2).mean()
        dataframe['kdj_d'] = dataframe['kdj_k'].ewm(com=2).mean()
        dataframe['kdj_j'] = 3 * dataframe['kdj_k'] - 2 * dataframe['kdj_d']

        # 保留DI和动量指标计算（出场信号仍可能用）
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=10)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        #dataframe['macd_slope'] = np.gradient(dataframe['macd'])

        # 新增：MACD 背离计算
        dataframe['macd_bullish_div'] = (
            (dataframe['low'] < dataframe['low'].shift(1)) &
            (dataframe['macd'] > dataframe['macd'].shift(1))
        ).astype(int)

        dataframe['macd_bearish_div'] = (
            (dataframe['high'] > dataframe['high'].shift(1)) &
            (dataframe['macd'] < dataframe['macd'].shift(1))
        ).astype(int)

        # 反转做多
        cond_reversal_long = (
            (dataframe['rsi'] < self.rsi_entry_long.value) &
            qtpylib.crossed_above(dataframe['rsi_gra'], 0) &
            (dataframe['macd'] < dataframe['macdsignal']) &
            (dataframe['macd_bullish_div'] == 1)
        )
        dataframe.loc[cond_reversal_long, 
            ["enter_long", "enter_tag"]
            ] = [1, "cond_reversal_long"]

        # 反转做空
        cond_reversal_short = (
            (dataframe['rsi'] > self.rsi_entry_short.value) &
            qtpylib.crossed_below(dataframe['rsi_gra'], 0) &
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['macd_bearish_div'] == 1)
        )
        dataframe.loc[cond_reversal_short,
            ["enter_short", "enter_tag"]
            ] = [1, "cond_reversal_short"]

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
        if dataframe is None or len(dataframe) < 20:
            return None

        last_candle = dataframe.iloc[-1]
        opened_at = trade.open_date_utc
        df_since_entry = dataframe[dataframe['date'] >= opened_at]

        kdj_golden_cross = qtpylib.crossed_above(dataframe['kdj_k'], dataframe['kdj_d']).astype(int)
        kdj_dead_cross = qtpylib.crossed_below(dataframe['kdj_k'], dataframe['kdj_d']).astype(int)

        if trade.trade_direction == 'long':
            had_golden_cross = df_since_entry['macd_golden_cross'].sum() > 0
            current_macd_goldencross = last_candle['macd'] > last_candle['macdsignal']
            if had_golden_cross and current_macd_goldencross and kdj_dead_cross.iloc[-1] == 1:
                return "exit_long_custom"

        if trade.trade_direction == 'short':
            had_dead_cross = df_since_entry['macd_dead_cross'].sum() > 0
            current_macd_deadcross = last_candle['macd'] < last_candle['macdsignal']
            if had_dead_cross and current_macd_deadcross and kdj_golden_cross.iloc[-1] == 1:
                return "exit_short_custom"

        return None
