import numpy as np
import logging
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
import pandas_ta as pta
from datetime import timedelta, datetime, timezone
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IStrategy, IntParameter, informative)
logger = logging.getLogger(__name__)

class MKV3(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = '15m'
    can_short = True
    process_only_new_candles = True
    use_custom_exit = True
    use_custom_stoploss = True

    minimal_roi = {
        "0": 0.128,
        "16": 0.058,
        "31": 0.042,
        "61": 0.038,
        "90": 0.016,
        "120": 0.008,
    }

    stoploss = -0.168
    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.056
    trailing_only_offset_is_reached = True

    position_adjustment_enable = True
    max_portfolio_percentage_per_trade = 0.5
    max_entry_position_adjustment = 2
    max_dca_orders = 2
    startup_candle_count = 52

    buy_params = {
        "rsi_entry_long": 47,
        "rsi_entry_long_min": 29,
        "rsi_entry_short": 49,
        "rsi_entry_short_max": 78,
        "window": 24,
    }

    sell_params = {
        "rsi_exit_long": 96,
        "rsi_exit_short": 65,
    }

    max_open_trades = 2

    rsi_entry_long = IntParameter(0, 100, default=buy_params.get('rsi_entry_long'), space='buy', optimize=True)
    rsi_entry_long_min = IntParameter(0, 100, default=buy_params.get('rsi_entry_long_min'), space='buy', optimize=True)
    rsi_entry_short = IntParameter(0, 100, default=buy_params.get('rsi_entry_short'), space='buy', optimize=True)
    rsi_entry_short_max = IntParameter(0, 100, default=buy_params.get('rsi_entry_short_max'), space='buy', optimize=True)
    rsi_exit_long = IntParameter(0, 100, default=sell_params.get('rsi_exit_long'), space='sell', optimize=True)
    rsi_exit_short = IntParameter(0, 100, default=sell_params.get('rsi_exit_short'), space='sell', optimize=True)
    window = IntParameter(5, 100, default=buy_params.get('window'), space='buy', optimize=False)

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

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        try:
            total_portfolio = self.wallets.get_total_stake_amount()
            max_stake_per_trade = (total_portfolio / self.max_open_trades) * 0.7
            max_stake_from_portfolio = total_portfolio * self.max_portfolio_percentage_per_trade
        except:
            total_portfolio = 1000.0
            max_stake_per_trade = (total_portfolio / self.max_open_trades) * 0.7
            max_stake_from_portfolio = total_portfolio * self.max_portfolio_percentage_per_trade
        stake_amount = max_stake_per_trade
        final_stake = min(stake_amount, max_stake_per_trade, max_stake_from_portfolio, max_stake)
        limit_reason = "首次开仓金额"
        if final_stake == max_stake_per_trade:
            limit_reason = "单笔交易最大值"
        elif final_stake == max_stake_from_portfolio:
            limit_reason = "投资组合百分比"
        elif final_stake == max_stake:
            limit_reason = "Freqtrade最大值"
        logger.info(f"{pair} 初始仓位计算：{final_stake:.8f} (建议仓位: {proposed_stake:.8f}, 受限于: {limit_reason}, 占投资组合: {(final_stake/total_portfolio)*100:.1f}%)")
        if min_stake is not None and final_stake < min_stake:
            logger.info(f"{pair} 初始仓位 {final_stake:.8f} 低于最小交易金额 {min_stake:.8f}。调整为最小交易金额。")
            final_stake = min_stake
        return final_stake

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                              current_profit: float, min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float, **kwargs) -> Optional[float]:
        count_of_entries = trade.nr_of_successful_entries
        if not self.position_adjustment_enable:
            return None
        if not hasattr(self, 'max_dca_orders'):
            logger.error(f"{trade.pair} 缺少max_dca_orders参数")
            return None
        max_dca_for_pair = self.max_dca_orders
        try:
            total_portfolio = self.wallets.get_total_stake_amount()
            max_total_stake = total_portfolio / self.max_open_trades
        except:
            total_portfolio = 1000.0
            max_total_stake = total_portfolio / self.max_open_trades
        if count_of_entries > max_dca_for_pair:
            logger.info(f"{trade.pair} 🛑 已达到最大DCA次数: {count_of_entries}/{max_dca_for_pair + 1}")
            return None
        if trade.stake_amount >= max_total_stake:
            logger.info(f"{trade.pair} 🛑 已达到最大仓位: {trade.stake_amount:.2f}/{max_total_stake:.2f} USDT")
            return None
        if count_of_entries == 1 and current_profit > -0.018:
            logger.info(f"{trade.pair} 未触发DCA。当前盈利 {current_profit:.2%} 阈值范围 -1.8%")
            return None
        if count_of_entries == 2 and current_profit > -0.036:
            logger.info(f"{trade.pair} 未触发DCA。当前盈利 {current_profit:.2%} 阈值范围 -3.6%")
            return None
        try:
            filled_entry_orders = trade.select_filled_orders(trade.entry_side)
            if not filled_entry_orders:
                return None
            dca_stake_amount = (total_portfolio / self.max_open_trades) * (0.15 if count_of_entries == 1 else 0.15)
            remaining_budget = max_total_stake - trade.stake_amount
            if dca_stake_amount > remaining_budget:
                if remaining_budget > 1:
                    dca_stake_amount = remaining_budget
                else:
                    return None
            if min_stake and dca_stake_amount < min_stake:
                dca_stake_amount = min_stake
            logger.info(f"{trade.pair} 🛑 第{count_of_entries}次DCA: +{dca_stake_amount:.2f} USDT")
            return dca_stake_amount
        except Exception as e:
            logger.error(f"{trade.pair} DCA计算错误: {e}")
            return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        return 5

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
        low_min = dataframe['low'].rolling(window=9).min()
        high_max = dataframe['high'].rolling(window=9).max()
        rsv = (dataframe['close'] - low_min) / (high_max - low_min) * 100
        dataframe['kdj_k'] = rsv.ewm(com=2).mean()
        dataframe['kdj_d'] = dataframe['kdj_k'].ewm(com=2).mean()
        dataframe['kdj_j'] = 3 * dataframe['kdj_k'] - 2 * dataframe['kdj_d']
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=10)
        # Add MACD divergence calculations
        dataframe['macd_bullish_div'] = (
            (dataframe['low'] < dataframe['low'].shift(1)) &
            (dataframe['macd'] > dataframe['macd'].shift(1))
        ).astype(int)
        dataframe['macd_bearish_div'] = (
            (dataframe['high'] > dataframe['high'].shift(1)) &
            (dataframe['macd'] < dataframe['macd'].shift(1))
        ).astype(int)
        # Add ATR for custom stoploss and exit
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['macd_slope'] = np.gradient(dataframe['macd'])
        # Calculate RSI bullish divergence: lower price low, higher RSI low
        dataframe['rsi_bullish_div'] = (
            (dataframe['low'] < dataframe['low'].shift(1)) &
            (dataframe['rsi'] > dataframe['rsi'].shift(1))
        ).astype(int)
        # Calculate RSI bearish divergence: higher price high, lower RSI high
        dataframe['rsi_bearish_div'] = (
            (dataframe['high'] > dataframe['high'].shift(1)) &
            (dataframe['rsi'] < dataframe['rsi'].shift(1))
        ).astype(int)
        # Long entry condition with RSI and MACD bullish divergence
        cond_reversal_long = (
            (dataframe['rsi'] < self.rsi_entry_long.value) &
            (dataframe['rsi'] > self.rsi_entry_long_min.value) &
            qtpylib.crossed_above(dataframe['rsi_gra'], 0) &
            (dataframe['macd'] < dataframe['macdsignal']) &
            (dataframe['macd_bullish_div'] == 1) &
            (dataframe['rsi_bullish_div'] == 1)
        )
        dataframe.loc[cond_reversal_long, 'enter_long'] = 1
        # Short entry condition with RSI and MACD bearish divergence
        cond_reversal_short = (
            (dataframe['rsi'] > self.rsi_entry_short.value) &
            (dataframe['rsi'] < self.rsi_entry_short_max.value) &
            qtpylib.crossed_below(dataframe['rsi_gra'], 0) &
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['macd_bearish_div'] == 1) &
            (dataframe['rsi_bearish_div'] == 1)
        )
        dataframe.loc[cond_reversal_short, 'enter_short'] = 1
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
        for i in range(1, len(dataframe['close'])):
            if dataframe.iloc[-i]['date'].to_pydatetime().replace(tzinfo=datetime.timezone.utc) == trade.open_date_utc:
                entry_candle = dataframe.iloc[-i-1].squeeze()
                atr = entry_candle.get('atr', 0)
                if trade.is_short:
                    entry_low = entry_candle.get('low', 0)
                    takeprofit = entry_low - atr
                    if current_rate <= takeprofit:
                        return 'takeprofit_atr_reached'
                else:
                    entry_high = entry_candle.get('high', 0)
                    takeprofit = entry_high + atr
                    if current_rate >= takeprofit:
                        return 'takeprofit_atr_reached'
                break
        return None

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        stoploss = 999999
        for i in range(1,len(dataframe['close'])):
            if dataframe.iloc[-i]['date'].to_pydatetime().replace(tzinfo=datetime.timezone.utc) == trade.open_date_utc:
                buy_candle = dataframe.iloc[-i-1].squeeze()
                if trade.is_short:
                    stoploss = buy_candle['high'] + buy_candle['atr']
                else:
                    stoploss = buy_candle['low'] - buy_candle['atr']
                break
        if trade.is_short:
            if stoploss > current_rate:
                return (stoploss / current_rate) - 1
        else:
            if stoploss < current_rate:
                return (stoploss / current_rate) - 1
        return 1