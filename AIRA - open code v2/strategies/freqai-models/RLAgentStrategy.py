import datetime
import logging
from functools import reduce
from typing import Any, Final, Literal, Optional

import numpy as np
import pandas as pd

# import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy
from pandas import DataFrame

TradingMode = Literal["margin", "futures", "spot"]
TradeDirection = Literal["long", "short"]

logger = logging.getLogger(__name__)

ACTION_COLUMN: Final = "&-action"


_EPOCH_MS_MIN = 1_262_304_000_000  # 2010-01-01T00:00:00Z
_EPOCH_MS_MAX = 2_051_222_400_000  # 2035-01-01T00:00:00Z


def _ensure_datetime_series(series: pd.Series | None) -> pd.Series:
    """Ensure a date series is datetime64[ms, UTC], following freqtrade's data handler pattern."""
    if series is None:
        raise ValueError(
            "Expected a date Series but received None. "
            "The 'date' column is missing from the dataframe."
        )
    if pd.api.types.is_integer_dtype(series):
        sample = series.dropna()
        if sample.empty:
            return pd.to_datetime(series, unit="ms", utc=True).dt.as_unit("ms")
        probe = int(sample.iat[0])
        if not (_EPOCH_MS_MIN <= probe <= _EPOCH_MS_MAX):
            raise ValueError(
                f"Integer date column value {probe} is outside the expected epoch-ms "
                f"range [{_EPOCH_MS_MIN}, {_EPOCH_MS_MAX}]. "
                "Data is likely corrupted or uses a different unit."
            )
        return pd.to_datetime(series, unit="ms", utc=True).dt.as_unit("ms")
    return series.dt.as_unit("ms")


class RLAgentStrategy(IStrategy):

    INTERFACE_VERSION = 3
    stoploss = -0.60
    _ACTION_ENTER_LONG: Final = 1
    _ACTION_EXIT_LONG: Final = 2
    _ACTION_ENTER_SHORT: Final = 3
    _ACTION_EXIT_SHORT: Final = 4
    _TRADE_DIRECTIONS: Final = ("long", "short")
 
    @property
    def can_short(self) -> bool:
        return self.is_short_allowed()

    def is_short_allowed(self) -> bool:
        trading_mode = self.config.get("trading_mode", "spot")
        trading_mode_value = getattr(trading_mode, "value", trading_mode)
        return trading_mode_value in {"margin", "futures"}

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict[str, Any], **kwargs
    ) -> DataFrame:
        dataframe["%-close_pct_change"] = np.log(dataframe.get("close")).diff()
        dataframe["%-raw_volume"] = dataframe.get("volume")

        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict[str, Any], **kwargs
    ) -> DataFrame:
        dates = _ensure_datetime_series(dataframe.get("date"))
        dataframe["%-day_of_week"] = (dates.dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dates.dt.hour + 1) / 25

        dataframe["%-raw_close"] = dataframe.get("close")
        dataframe["%-raw_open"] = dataframe.get("open")
        dataframe["%-raw_high"] = dataframe.get("high")
        dataframe["%-raw_low"] = dataframe.get("low")

        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict[str, Any], **kwargs
    ) -> DataFrame:
        dataframe[ACTION_COLUMN] = 0

        return dataframe

    def populate_indicators(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        enter_long_conditions = [
            dataframe.get("do_predict") == 1,
            dataframe.get(ACTION_COLUMN) == RLAgentStrategy._ACTION_ENTER_LONG,  # 1,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, enter_long_conditions),
            ["enter_long", "enter_tag"],
        ] = (1, RLAgentStrategy._TRADE_DIRECTIONS[0])  # "long"

        enter_short_conditions = [
            dataframe.get("do_predict") == 1,
            dataframe.get(ACTION_COLUMN) == RLAgentStrategy._ACTION_ENTER_SHORT,  # 3,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, enter_short_conditions),
            ["enter_short", "enter_tag"],
        ] = (1, RLAgentStrategy._TRADE_DIRECTIONS[1])  # "short"

        return dataframe

    def populate_exit_trend(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        exit_long_conditions = [
            dataframe.get("do_predict") == 1,
            dataframe.get(ACTION_COLUMN) == RLAgentStrategy._ACTION_EXIT_LONG,  # 2,
        ]
        dataframe.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [
            dataframe.get("do_predict") == 1,
            dataframe.get(ACTION_COLUMN) == RLAgentStrategy._ACTION_EXIT_SHORT,  # 4,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"
        ] = 1

        last_candle = dataframe.iloc[-1]
        if last_candle.get("do_predict") == 2:
            trades = Trade.get_trades_proxy(pair=metadata.get("pair"), is_open=True)
            for trade in trades:
                last_index = dataframe.index[-1]
                if trade.is_short:
                    dataframe.at[last_index, "exit_short"] = 1
                else:
                    dataframe.at[last_index, "exit_long"] = 1

        return dataframe
