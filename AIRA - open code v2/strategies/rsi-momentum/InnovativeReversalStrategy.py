import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
from typing import Optional
from datetime import datetime
import logging


class InnovativeReversalStrategy(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 0.5,
        "60": 0.45,
        "120": 0.4,
        "240": 0.3,
        "360": 0.25,
        "720": 0.2,
        "1440": 0.15,
        "2880": 0.1,
        "3600": 0.05,
        "7200": 0.02,
    }
    can_short = True
    stoploss = -0.99
    use_exit_signal = False
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    irm_threshold_high = DecimalParameter(1.0, 5.0, default=2.0, space="buy")
    irm_threshold_low = DecimalParameter(-5.0, -1.0, default=-1.5, space="sell")

    timeframe = "1h"
    leverage_input_buy = DecimalParameter(1.0, 10.0, default=1.0, space="buy")
    leverage_input_sell = DecimalParameter(1.0, 10.0, default=1.0, space="sell")

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
        "force_entry": "limit",
        "force_exit": "limit",
    }

    order_time_in_force = {
        "entry": "gtc",
        "exit": "gtc",
    }

    plot_config = {
        "main_plot": {
            "irm": {
                "color": "blue",
                "title": "Índice de Reversão de Mercado (IRM)",
            },
            "momentum_change": {
                "color": "red",
                "title": "Mudança de Momento",
            },
        },
        "subplots": {
            "Candlestick Patterns": {
                "hammer": {
                    "color": "green",
                    "title": "Hammer",
                    "plot": "bar",
                },
                "shooting_star": {
                    "color": "orange",
                    "title": "Shooting Star",
                    "plot": "bar",
                },
            },
            "Volume Deviation": {
                "volume_deviation": {
                    "color": "purple",
                    "title": "Desvio de Volume",
                },
            },
        },
    }

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        if side == "long":
            return min(self.leverage_input_buy.value, max_leverage)
        else:
            return min(self.leverage_input_sell.value, max_leverage)

    def calculate_irm(self, dataframe: DataFrame) -> DataFrame:
        dataframe["price_diff"] = dataframe["close"].diff()
        dataframe["momentum_change"] = dataframe["price_diff"].diff()

        dataframe["hammer"] = (
            (dataframe["close"] > dataframe["open"])
            & ((dataframe["high"] - dataframe["low"]) > 2 * (dataframe["open"] - dataframe["low"]))
            & (
                (dataframe["close"] - dataframe["open"])
                / (0.001 + dataframe["high"] - dataframe["low"])
                > 0.6
            )
        ).astype(int)

        dataframe["shooting_star"] = (
            (dataframe["close"] < dataframe["open"])
            & (
                (dataframe["high"] - dataframe["low"])
                > 2 * (dataframe["high"] - dataframe["close"])
            )
            & (
                (dataframe["open"] - dataframe["close"])
                / (0.001 + dataframe["high"] - dataframe["low"])
                > 0.6
            )
        ).astype(int)

        dataframe["volume_deviation"] = (
            dataframe["volume"] - dataframe["volume"].rolling(window=20).mean()
        ) / dataframe["volume"].rolling(window=20).std()

        dataframe["irm"] = (dataframe["momentum_change"] * dataframe["hammer"]) - (
            dataframe["momentum_change"] * dataframe["shooting_star"]
        )
        dataframe["irm"] += dataframe["volume_deviation"]

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.calculate_irm(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["irm"] > self.irm_threshold_high.value),
            ["enter_long", "enter_tag"],
        ] = (1, "reversal_up")
        dataframe.loc[
            (dataframe["irm"] < self.irm_threshold_low.value),
            ["enter_short", "enter_tag"],
        ] = (1, "reversal_down")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["irm"] < self.irm_threshold_low.value),
            ["exit_long", "exit_tag"],
        ] = (1, "exit_reversal_down")
        dataframe.loc[
            (dataframe["irm"] > self.irm_threshold_high.value),
            ["exit_short", "exit_tag"],
        ] = (1, "exit_reversal_up")
        return dataframe

    def confirm_trade_entry(
        self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs
    ) -> bool:
        logging.info(
            f"Confirmando entrada de trade para o par {pair} com tipo de ordem {order_type} e quantidade {amount} a taxa {rate}."
        )
        return True

    def confirm_trade_exit(
        self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs
    ) -> bool:
        logging.info(
            f"Confirmando saída de trade para o par {pair} com tipo de ordem {order_type} e quantidade {amount} a taxa {rate}."
        )
        return True

    def execute_strategy(self, dataframe: DataFrame, metadata: dict):
        try:
            dataframe = self.populate_indicators(dataframe, metadata)
            dataframe = self.populate_entry_trend(dataframe, metadata)
            dataframe = self.populate_exit_trend(dataframe, metadata)
        except Exception as e:
            logging.error(f"Erro ao executar a estratégia: {str(e)}")
        return dataframe
