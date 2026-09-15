from datetime import datetime
from typing import Dict, Any

import numpy as np  # noqa
import pandas as pd  # noqa

from freqtrade.strategy import IStrategy
from pandas import DataFrame


class RSI_Futures9(IStrategy):
    """
    周期：5m
    指标：RSI(14)
    做多：RSI < 25 开多，RSI > 75 且 ROE > 0 平多；
    做空：RSI > 75 开空，RSI < 25 且 ROE > 0 平空；
    止损：由 config.json 中的 stoploss 参数控制；
    杠杆：5x（期货/合约）
    """

    timeframe = "5m"

    # 允许做多和做空
    can_short: bool = True

    # 期货模式（如使用现货可关闭）
    is_futures_trading = True

    # 启用加仓（DCA）功能
    position_adjustment_enable = True

    # 最多加仓次数（不含首单）
    MAX_DCA_ENTRIES = 3

    # 触发加仓的 ROE 阈值（亏损 65%）
    DCA_TRIGGER_ROE = -0.65

    # 默认杠杆倍数（在 leverage() 方法中使用）
    DEFAULT_LEVERAGE = 5.0
    trailing_stop = False

    # ROI 表：设为极大值，让 ROI 不主动触发止盈（主要依赖全局止损）
    minimal_roi = {
        "0": 100.0,
    }

    process_only_new_candles = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算 RSI14 指标
        """
        import talib.abstract as ta

        dataframe["rsi"] = ta.RSI(dataframe["close"], timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        开仓条件：
        - 做多：RSI < 25 开多；
        - 做空：RSI > 75 开空。
        """
        dataframe.loc[:, "enter_long"] = 0
        dataframe.loc[:, "enter_short"] = 0

        rsi = dataframe["rsi"]

        # 做多：RSI < 25
        long_entry = rsi < 25
        # 做空：RSI > 75
        short_entry = rsi > 75

        dataframe.loc[long_entry, "enter_long"] = 1
        dataframe.loc[short_entry, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        这里不再使用固定的 RSI 平仓信号，
        实际平仓逻辑放在 custom_exit 中根据 RSI + ROE 决定。
        """
        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Dict[str, Any] | None:
        """
        平仓逻辑（结合 RSI + ROE）：
        - 做多：RSI > 75 且当前 ROE > 0 时平多；
        - 做空：RSI < 25 且当前 ROE > 0 时平空；
        - 若 RSI 已达阈值但 ROE < 0，则暂不平仓，等待之后再次满足 RSI 阈值且 ROE 转正再平仓。
        """
        # 无数据提供器则不做自定义平仓
        if self.dp is None:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1].squeeze()
        rsi = last_candle.get("rsi", None)
        if rsi is None or np.isnan(rsi):
            return None

        # 做多：RSI > 75 且 ROE > 0 平仓
        if not trade.is_short:
            if rsi > 75 and current_profit > 0:
                return {"exit_tag": "rsi_long_take_profit"}
            return None

        # 做空：RSI < 25 且 ROE > 0 平仓
        if trade.is_short:
            if rsi < 25 and current_profit > 0:
                return {"exit_tag": "rsi_short_take_profit"}
            return None

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        自定义杠杆：固定使用 5 倍（不超过交易所允许的最大杠杆）。
        """
        return min(self.DEFAULT_LEVERAGE, float(max_leverage))

    def adjust_trade_position(
        self,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float,
        max_stake: float,
        **kwargs,
    ) -> float | None:
        """
        DCA 加仓逻辑：
        - 当整笔仓位 ROE（current_profit）<= -65% 时触发加仓；
        - 每次加仓额度 = 当前持仓总额度（按当前持仓额度加仓，相当于仓位翻倍）；
        - 最多加仓 3 次（不含首单）；
        - 加仓后按均价（加权平均价）计算整笔仓位 ROE，若之后再次整体亏损到 -65% 再加下一次。
        """
        # 达到触发阈值才考虑加仓
        if current_profit > self.DCA_TRIGGER_ROE:
            return None

        entry_side = "sell" if trade.is_short else "buy"
        filled_entries = trade.select_filled_orders(entry_side)
        entries_count = len(filled_entries)
        if entries_count <= 0:
            return None

        # 已加仓次数（不含首单）
        dca_count = entries_count - 1
        if dca_count >= self.MAX_DCA_ENTRIES:
            return None

        # 当前持仓总额度（首单 + 之前所有加仓）
        total_cost = sum(o.cost for o in filled_entries)
        if total_cost <= 0:
            return None

        # 每次加仓额度：按当前持仓额度加仓（翻倍）
        additional_stake = total_cost

        # 遵守 min_stake / max_stake 限制
        additional_stake = max(min_stake, min(additional_stake, max_stake))
        if additional_stake < min_stake:
            return None

        return additional_stake

