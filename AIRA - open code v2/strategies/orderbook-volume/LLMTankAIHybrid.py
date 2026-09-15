# pragma pylint: disable=missing-docstring, invalid-name

import json
import time
import logging
import requests
import numpy as np
import talib.abstract as ta

from pandas import DataFrame
from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class LLMTankAIHybrid(IStrategy):

    # =============================
    # ===== OLLAMA SETTINGS =======
    # =============================
    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "phi4-reasoning:14b"
    OLLAMA_TIMEOUT = 25

    # =============================
    # ===== STRATEGY CONFIG =======
    # =============================
    timeframe = "15m"
    startup_candle_count = 200
    process_only_new_candles = True

    can_short = False

    stoploss = -0.15  # hard safety

    exit_profit_only = True
    ignore_roi_if_entry_signal = True

    trailing_stop = False

    decision_confidence_threshold = 0.65
    min_expected_profit = 0.012

    ATR_PROFIT_MULTIPLIER = 1.5
    MAX_DRAWDOWN_EXIT = -0.10  # 🔥 emergency exit

    IMPULSE_LOOKBACK = 3

    # ===== DCA CONFIG =====
    position_adjustment_enable = True
    max_entry_position_adjustment = 2
    DCA_ATR_MULTIPLIER = 1.2
    DCA_MIN_CONFIDENCE = 0.55

    # ===== PARTIAL TP CONFIG =====
    minimal_roi = {
        "0": 0.06,
        "30": 0.04,
        "90": 0.02,
    }

    # =============================
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.last_llm_decision = {}

    # =============================
    # ===== INDICATORS ============
    # =============================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["rsi"] = ta.RSI(dataframe["close"], 14)
        dataframe["atr"] = ta.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], 14
        )

        tp = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        dataframe["vwap"] = (tp * dataframe["volume"]).cumsum() / dataframe["volume"].cumsum()
        dataframe["price_vs_vwap"] = (dataframe["close"] - dataframe["vwap"]) / dataframe["vwap"]

        body = dataframe["close"] - dataframe["open"]
        dataframe["body_atr"] = body.abs() / dataframe["atr"]

        dataframe["bull_impulse"] = np.where(body > 0, dataframe["body_atr"], 0)
        dataframe["bear_impulse"] = np.where(body < 0, dataframe["body_atr"], 0)

        dataframe["bull_impulse_cluster"] = dataframe["bull_impulse"].rolling(
            self.IMPULSE_LOOKBACK
        ).sum()

        dataframe["bear_impulse_cluster"] = dataframe["bear_impulse"].rolling(
            self.IMPULSE_LOOKBACK
        ).sum()

        dataframe["llm_decision"] = None
        dataframe["llm_confidence"] = np.nan
        dataframe["llm_projected_profit"] = np.nan

        if len(dataframe) > self.startup_candle_count:
            self._run_llm(metadata["pair"], dataframe)

        return dataframe

    # =============================
    # ===== LLM ===================
    # =============================
    def _run_llm(self, pair: str, dataframe: DataFrame):
        last = dataframe.iloc[-1]

        trade = Trade.get_trades(
            [Trade.pair == pair, Trade.is_open.is_(True)]
        ).first()

        position = {
            "in_position": bool(trade),
            "price": last["close"],
            "unrealized": trade.calc_profit_ratio(last["close"]) if trade else 0
        }

        prompt = f"""
Evaluate trade continuation quality.

RSI: {last["rsi"]:.2f}
ATR: {last["atr"]:.6f}
Price vs VWAP: {last["price_vs_vwap"]:.4f}

Bull impulse cluster: {last["bull_impulse_cluster"]:.3f}
Bear impulse cluster: {last["bear_impulse_cluster"]:.3f}

Position:
{json.dumps(position)}

Respond ONLY in JSON:
{{
  "decision": "BUY" | "HOLD" | "SELL",
  "confidence": 0-1,
  "projected_profit": 0-0.05
}}
"""

        try:
            r = requests.post(
                self.OLLAMA_URL,
                json={
                    "model": self.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
                timeout=self.OLLAMA_TIMEOUT,
            )

            raw = r.json().get("response", "")
            logger.info(f"[LLM RAW] {pair}: {raw}")

            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end == -1:
                return

            parsed = json.loads(raw[start:end + 1])

            dataframe.loc[dataframe.index[-1], "llm_decision"] = parsed.get("decision")
            dataframe.loc[dataframe.index[-1], "llm_confidence"] = float(parsed.get("confidence", 0))
            dataframe.loc[dataframe.index[-1], "llm_projected_profit"] = float(
                parsed.get("projected_profit", np.nan)
            )

        except Exception:
            pass

    # =============================
    # ===== ENTRY =================
    # =============================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0

        last = dataframe.iloc[-1]

        if (
            last["llm_decision"] == "BUY"
            and last["llm_confidence"] >= self.decision_confidence_threshold
            and last["bull_impulse_cluster"] > last["bear_impulse_cluster"]
            and last["price_vs_vwap"] < 0.004
        ):
            dataframe.loc[dataframe.index[-1], "enter_long"] = 1
            logger.info(f"[ENTRY] {metadata['pair']}")

        return dataframe

    # =============================
    # ===== SMART DCA =============
    # =============================
    def adjust_trade_position(
        self,
        trade: Trade,
        current_time,
        current_rate,
        current_profit,
        **kwargs
    ):
        if trade.nr_of_successful_entries >= self.max_entry_position_adjustment:
            return None

        if current_profit > -0.01:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last = dataframe.iloc[-1]

        atr_trigger = -self.DCA_ATR_MULTIPLIER * (last["atr"] / current_rate)

        if (
            current_profit < atr_trigger
            and last["bull_impulse_cluster"] > last["bear_impulse_cluster"]
            and last["llm_confidence"] >= self.DCA_MIN_CONFIDENCE
        ):
            logger.info(f"[DCA] {trade.pair} | profit={current_profit:.3f}")
            return trade.stake_amount

        return None

    # =============================
    # ===== EXIT ==================
    # =============================
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0

        pair = metadata["pair"]
        last = dataframe.iloc[-1]

        trade = Trade.get_trades(
            [Trade.pair == pair, Trade.is_open.is_(True)]
        ).first()

        if not trade:
            return dataframe

        profit = trade.calc_profit_ratio(last["close"])

        # 🔥 HARD SAFETY
        if profit <= self.MAX_DRAWDOWN_EXIT:
            dataframe.loc[dataframe.index[-1], "exit_long"] = 1
            logger.warning(f"[EMERGENCY EXIT] {pair} drawdown={profit:.3f}")
            return dataframe

        impulse_break = last["bull_impulse_cluster"] < last["bear_impulse_cluster"]

        # Partial TP logic via ROI table handles scaling
        if (
            impulse_break
            and profit > 0
            and last["llm_decision"] == "SELL"
        ):
            dataframe.loc[dataframe.index[-1], "exit_long"] = 1
            logger.info(f"[LLM EXIT] {pair} profit={profit:.3f}")

        return dataframe
