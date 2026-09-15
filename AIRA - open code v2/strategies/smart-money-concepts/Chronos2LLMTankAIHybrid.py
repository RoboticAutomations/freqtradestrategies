# pragma pylint: disable=missing-module-docstring, invalid-name

import json
import logging
import time
import requests
import numpy as np
import pandas as pd
import talib.abstract as ta

from pandas import DataFrame
from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

logger = logging.getLogger(__name__)

# =========================
# USER CONFIG
# =========================
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi4-reasoning:14b"
OLLAMA_TIMEOUT = 30

ENSEMBLE_THRESHOLD = 0.65
LLM_CONF_MIN = 0.60
CHRONOS_CONF_MIN = 0.55

CHRONOS_CONTEXT = 120
CHRONOS_PRED_STEPS = 3
CHRONOS_REFIT_SECONDS = 1800
MIN_CHRONOS_EDGE = 0.006


class Chronos2LLMTankAIHybrid(IStrategy):
    """
    Chronos-2 + LLM Ensemble Strategy (FINAL FIXED VERSION)
    """

    timeframe = "15m"
    startup_candle_count = 200
    process_only_new_candles = True

    stoploss = -0.12
    exit_profit_only = True
    ignore_roi_if_entry_signal = True

    minimal_roi = {"0": 0.08}

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.chronos_models = {}
        self.chronos_last_fit = {}

    # =========================
    # INDICATORS
    # =========================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["atr"] = ta.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14
        )
        dataframe["rsi"] = ta.RSI(dataframe["close"], timeperiod=14)

        dataframe["ob_imbalance"] = 0.0
        if self.dp and self.dp.runmode.value in ("live", "dry_run"):
            try:
                ob = self.dp.orderbook(metadata["pair"], 5)
                bids = np.array([b[0] * b[1] for b in ob["bids"]])
                asks = np.array([a[0] * a[1] for a in ob["asks"]])
                dataframe.loc[dataframe.index[-1], "ob_imbalance"] = (
                    (bids.sum() - asks.sum()) / max(bids.sum() + asks.sum(), 1)
                )
            except Exception:
                pass

        return dataframe

    # =========================
    # CHRONOS-2 (CORRECT PRESET)
    # =========================
    def _chronos_predict(self, pair: str, df: DataFrame):

        try:
            hist = df[["date", "close"]].tail(CHRONOS_CONTEXT).copy()

            # 🚨 REQUIRED: strip timezone
            hist["timestamp"] = (
                pd.to_datetime(hist["date"], utc=True)
                .dt.tz_localize(None)
            )

            hist["item_id"] = pair
            hist = hist[["item_id", "timestamp", "close"]]

            ts_df = TimeSeriesDataFrame.from_data_frame(
                hist,
                id_column="item_id",
                timestamp_column="timestamp"
            )

            predictor = self.chronos_models.get(pair)
            if predictor is None or (time.time() - self.chronos_last_fit.get(pair, 0)) > CHRONOS_REFIT_SECONDS:
                logger.info(f"[CHRONOS INIT] {pair}")
                predictor = TimeSeriesPredictor(
                    prediction_length=CHRONOS_PRED_STEPS,
                    target="close",
                    verbosity=0
                )
                predictor.fit(
                    ts_df,
                    presets="chronos2"   # ✅ FIXED
                )
                self.chronos_models[pair] = predictor
                self.chronos_last_fit[pair] = time.time()

            forecast = predictor.predict(ts_df)

            # robust extraction
            if "0.5" in forecast.columns:
                future = forecast["0.5"].loc[pair].mean()
            else:
                future = forecast.mean(axis=1).loc[pair].mean()

            last_price = df["close"].iloc[-1]
            edge = (future - last_price) / last_price

            if edge > MIN_CHRONOS_EDGE:
                direction = "UP"
            elif edge < -MIN_CHRONOS_EDGE:
                direction = "DOWN"
            else:
                direction = "FLAT"

            confidence = min(abs(edge) * 10, 1.0)

            logger.info(
                f"[CHRONOS] {pair} dir={direction} edge={edge:.2%} conf={confidence:.2f}"
            )

            return direction, confidence

        except Exception as e:
            logger.warning(f"[CHRONOS FAIL] {pair}: {e}")
            return "FLAT", 0.0

    # =========================
    # LLM
    # =========================
    def _llm(self, row: pd.Series, trade: Trade | None, pair: str):

        prompt = f"""
Market snapshot:
Price: {row.close}
ATR: {row.atr}
RSI: {row.rsi}
Orderbook imbalance: {row.ob_imbalance}

Position:
{ "NONE" if trade is None else f"entry={trade.open_rate}, pnl={trade.calc_profit_ratio(row.close)}" }

Return JSON only:
{{"decision":"BUY|SELL|HOLD","confidence":0-1,"reason":"short"}}
"""

        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=OLLAMA_TIMEOUT
            )
            data = json.loads(r.json()["response"])

            logger.info(
                f"[LLM] {pair} → {data['decision']} conf={data['confidence']} | {data.get('reason','')}"
            )

            return data["decision"], float(data["confidence"])

        except Exception as e:
            logger.warning(f"[LLM FAIL] {pair}: {e}")
            return "HOLD", 0.0

    # =========================
    # ENTRY
    # =========================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        pair = metadata["pair"]
        row = dataframe.iloc[-1]

        chronos_dir, chronos_conf = self._chronos_predict(pair, dataframe)
        llm_dec, llm_conf = self._llm(row, None, pair)

        ensemble = 0.55 * llm_conf + 0.45 * chronos_conf

        logger.info(
            f"[ENSEMBLE ENTRY] {pair} LLM={llm_dec}({llm_conf:.2f}) "
            f"Chronos={chronos_dir}({chronos_conf:.2f}) → {ensemble:.2f}"
        )

        if (
            chronos_dir == "UP"
            and llm_dec == "BUY"
            and llm_conf >= LLM_CONF_MIN
            and chronos_conf >= CHRONOS_CONF_MIN
            and ensemble >= ENSEMBLE_THRESHOLD
        ):
            dataframe.loc[dataframe.index[-1], "enter_long"] = 1
            logger.info(f"[ENTRY OK] {pair}")
        else:
            logger.info(f"[ENTRY BLOCKED] {pair}")

        return dataframe

    # =========================
    # EXIT
    # =========================
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        pair = metadata["pair"]
        row = dataframe.iloc[-1]

        trade = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True)]).first()
        if not trade:
            return dataframe

        profit = trade.calc_profit_ratio(row.close)
        if profit <= -0.10:
            dataframe.loc[dataframe.index[-1], "exit_long"] = 1
            logger.warning(f"[EMERGENCY EXIT] {pair} drawdown={profit:.2%}")
            return dataframe

        chronos_dir, chronos_conf = self._chronos_predict(pair, dataframe)
        llm_dec, llm_conf = self._llm(row, trade, pair)

        ensemble = 0.55 * llm_conf + 0.45 * chronos_conf

        logger.info(
            f"[ENSEMBLE EXIT] {pair} LLM={llm_dec}({llm_conf:.2f}) "
            f"Chronos={chronos_dir}({chronos_conf:.2f}) → {ensemble:.2f}"
        )

        if profit > 0 and chronos_dir == "DOWN" and llm_dec == "SELL" and ensemble >= ENSEMBLE_THRESHOLD:
            dataframe.loc[dataframe.index[-1], "exit_long"] = 1
            logger.info(f"[EXIT OK] {pair}")

        return dataframe
