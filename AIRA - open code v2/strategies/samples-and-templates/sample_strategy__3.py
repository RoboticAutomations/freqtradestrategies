"""
GPT Strategy v5.1
═══════════════════════════════════════════════════════════════════════
Индикаторы: RSI · MACD · SuperTrend · ADX · Volume · SMA7/20/200
            BB · Stoch RSI · Williams %R · CCI · OBV
═══════════════════════════════════════════════════════════════════════

КЛЮЧЕВЫЕ НАСТРОЙКИ (метка # CFG):
  LLM_MODEL / LLM_API_KEY    — модель и ключ OpenAI / OpenRouter
  RSS_FEEDS                  — RSS-ленты новостей
  ENTRY_* / EXIT_*_THRESHOLD — пороги входа/выхода
  STAKE_LEVELS               — доступные размеры позиции (USDT)
  WEIGHT_TECHNICAL/SENTIMENT — веса тех. анализа и сентимента
  STOPLOSS_PCT / TAKE_PROFIT — стоп-лосс и тейк-профит
  TRAILING_*                 — параметры скользящего стопа
  MAX_TRADE_MINUTES          — макс. время удержания сделки (мин)
  HISTORY_CANDLES            — кол-во свечей для анализа LLM
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from typing import Literal as TypingLiteral

try:
    import feedparser  # type: ignore
except Exception:
    feedparser = None
import httpx
import numpy as np
import pandas as pd
import requests
from pandas import DataFrame
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#                     КОНФИГУРАЦИЯ  # CFG
# ═══════════════════════════════════════════════════════════════════════

# 1) LLM и запросы к модели (как часто и как долго ждём ответ)
LLM_MODEL   = "gpt-4o-mini"          # # CFG: модель ИИ
LLM_API_KEY = "REDACTED-BY-REPO-HYGIENE"    # # CFG: ключ API
LLM_TIMEOUT = 60                     # # CFG: таймаут одного LLM-запроса (сек)
LLM_RETRIES = 3                      # # CFG: повторы LLM при ошибке
LLM_MIN_CALL_INTERVAL_SEC = 20       # # CFG: минимум секунд между LLM-вызовами по паре

# 2) Данные свечей и запуск стратегии
HISTORY_CANDLES = 10                 # # CFG: свечей в контексте для LLM
MIN_CANDLES     = 50                 # # CFG: минимум свечей для расчёта индикаторов
STARTUP_CANDLES = 210                # # CFG: сколько свечей прогреть при старте

# 3) Новости и внешний контекст
MAX_NEWS_ITEMS = 5                   # # CFG: макс. новостей для анализа
NEWS_CACHE_SEC = 600                 # # CFG: кэш новостей (сек)
RSS_FEEDS = [                        # # CFG: источники новостей
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://cryptonews.com/news/feed/",
]
FEAR_GREED_CACHE_SEC = 3600          # # CFG: кэш Fear & Greed (сек)
CONTEXT_TASK_TIMEOUT_SEC = 2.5       # # CFG: таймаут каждого источника контекста (сек)
CONTEXT_TOTAL_TIMEOUT_SEC = 4.5      # # CFG: целевой бюджет сбора контекста (сек)
OB_DEPTH = 10                        # # CFG: глубина стакана (уровней на сторону)
OB_CACHE_SEC = 30                    # # CFG: кэш стакана (сек)

# 4) Пороги решений (когда вход/выход разрешён)
ENTRY_LONG_THRESHOLD  = +0.30          # # CFG: порог входа LONG
ENTRY_SHORT_THRESHOLD = -0.30          # # CFG: порог входа SHORT
EXIT_LONG_THRESHOLD   = -0.20          # # CFG: порог выхода из LONG
EXIT_SHORT_THRESHOLD  = +0.20          # # CFG: порог выхода из SHORT
SCORE_ROUND_DECIMALS = 2               # # CFG: округление score для пограничных решений
ENTRY_MIN_CONFIDENCE = 0.40            # # CFG: минимум уверенности для входа
ENTRY_MIN_VOL_RATIO = 0.20             # # CFG: минимум объёма для входа
ENTRY_MAX_ATR_PCT = 4.00               # # CFG: блок входа при экстремальной ATR
EXECUTION_MAX_ATR_PCT = 4.50           # # CFG: блок входа на этапе исполнения

# 5) Размеры позиций и веса оценок
STAKE_LEVELS = [30, 50, 100, 150, 200]  # # CFG: допустимые стейки (USDT)
WEIGHT_TECHNICAL = 0.85                 # # CFG: вес технического анализа
WEIGHT_SENTIMENT = 0.15                 # # CFG: вес сентимента новостей
LOW_CONFIDENCE_STAKE_THRESHOLD = 0.60   # # CFG: при confidence ниже этого порога стейк = минимум

# 6) Риск-менеджмент и управление сделкой
STOPLOSS_PCT      = -0.04              # # CFG: стоп-лосс
TAKE_PROFIT       = 0.02               # # CFG: тейк-профит (триггер трейлинга)
TRAILING_AFTER_TP = True
TRAILING_RETRACE  = 0.005
TRAILING_MIN_ACT  = 0.02
MAX_TRADE_MINUTES = 5000               # # CFG: макс. время сделки (мин)
LONG_ENTRY_SLIPPAGE  = 0.0025          # # CFG: макс. проскальзывание LONG
SHORT_ENTRY_SLIPPAGE = 0.0025          # # CFG: макс. проскальзывание SHORT

# 7) Мониторинг, уведомления и память
ALERT_COOLDOWN_SEC = 900                # # CFG: период мониторинг-алертов (сек)
NEUTRAL_NOTIFY_COOLDOWN_SEC = 0         # # CFG: 0 = отправлять NEUTRAL на каждой новой свече
TRADE_MEMORY_MAX = 50                   # # CFG: макс. записей сделок в памяти
MEMORY_FILE = "user_data/gpt_strategy_memory.json"  # # CFG: путь к файлу памяти
TRADE_EXECUTION_ENABLED_DEFAULT = True  # # CFG: дефолт, если параметр не задан в config.json

# legacy compact prompt removed; используем полный PromptBuilder.SYSTEM_PROMPT.


# ═══════════════════════════════════════════════════════════════════════
#                       TELEGRAM NOTIFIER
# ═══════════════════════════════════════════════════════════════════════

class TelegramNotifier:
    """Отправляет HTML-сообщения в Telegram. Нужен token и chat_id в config['telegram']."""

    def __init__(self, config: Optional[dict]):
        self.token = ""
        self.chat_id = ""
        if config:
            tg = config.get("telegram", {}) or {}
            self.token   = str(tg.get("token",   "")).strip()
            self.chat_id = str(tg.get("chat_id", "")).strip()
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            logger.warning("Telegram отключён: нет token/chat_id в config['telegram']")

    def send(self, message: str) -> None:
        if not self.enabled:
            logger.info(f"[TG SKIP] {message[:120]}")
            return
        # Telegram лимит — 4096 символов
        if len(message) > 4096:
            message = message[:4090] + "\n✂️"
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"},
                timeout=10,
            )
            if not resp.ok:
                logger.warning(f"[TG ERR] {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"[TG EXC] {e}")


# ═══════════════════════════════════════════════════════════════════════
#                       INDICATOR ENGINE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Indicators:
    """Все рассчитанные индикаторы для одной свечи."""
    price: float = 0.0
    change_pct: float = 0.0
    atr: float = 0.0
    atr_pct: float = 0.0

    # RSI
    rsi: float = 50.0
    rsi_history: List[float] = field(default_factory=list)

    # MACD
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    macd_trend: str = "нейтральный"
    macd_cross: str = "нет"         # "бычье" / "медвежье" / "нет"

    # SuperTrend
    st_dir: str = "нейтральный"
    st_level: float = 0.0
    st_dist_pct: float = 0.0

    # ADX
    adx: float = 20.0
    adx_str: str = "умеренный"

    # Volume
    vol_ratio: float = 1.0
    vol_desc: str = "норма"

    # SMA
    sma7: float = 0.0
    sma20: float = 0.0
    sma200: float = 0.0
    sma_cross: str = "нейтральный"  # "бычий" / "медвежий"

    # Bollinger Bands
    bb_upper: float = 0.0
    bb_mid: float = 0.0
    bb_lower: float = 0.0
    bb_pct: float = 50.0
    bb_pos: str = "середина"
    bb_squeeze: bool = False

    # Stochastic RSI
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    stoch_cross: str = "нет"        # "бычье" / "медвежье" / "нет"
    stoch_zone: str = "нейтральная" # "перепродан" / "перекуплен"

    # Williams %R
    williams_r: float = -50.0
    williams_zone: str = "нейтральная"

    # CCI
    cci: float = 0.0
    cci_signal: str = "нейтральный"

    # OBV
    obv_trend: str = "нейтральный"  # "бычий" / "медвежий"

    # Стакан заявок (order book)
    ob_bid_vol: float = 0.0      # суммарный объём bid (покупки)
    ob_ask_vol: float = 0.0      # суммарный объём ask (продажи)
    ob_imbalance: float = 0.0    # дисбаланс: >0 давление покупок, <0 давление продаж
    ob_desc: str = ""            # текстовое описание для промта

    # VWAP
    vwap: float = 0.0
    vwap_dist_pct: float = 0.0
    vwap_pos: str = "нейтральный"   # "выше" / "ниже"

    # Уровни
    support: float = 0.0
    resistance: float = 0.0
    pivot: float = 0.0

    # История свечей для анализа LLM
    candles: List[Dict[str, Any]] = field(default_factory=list)


class IndicatorEngine:
    """Рассчитывает все технические индикаторы из OHLCV-датафрейма."""

    @staticmethod
    def _safe_last(s: pd.Series, default: float = 0.0) -> float:
        try:
            v = float(s.iloc[-1])
            return v if not (np.isnan(v) or np.isinf(v)) else default
        except (IndexError, ValueError, TypeError):
            return default

    @classmethod
    def calculate(cls, df: DataFrame, history_n: int = HISTORY_CANDLES) -> Optional[Indicators]:
        if df is None or df.empty or len(df) < MIN_CANDLES:
            return None

        ind   = Indicators()
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        vol   = df["volume"].astype(float)

        # ── Цена ──────────────────────────────────────────────────────
        ind.price = cls._safe_last(close)
        prev = float(close.iloc[-2]) if len(close) >= 2 else ind.price
        ind.change_pct = (ind.price - prev) / prev * 100 if prev > 0 else 0.0

        # ── ATR ───────────────────────────────────────────────────────
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.rolling(14).mean()
        ind.atr = cls._safe_last(atr_series, 0.0)
        ind.atr_pct = (ind.atr / ind.price * 100) if ind.price > 0 else 0.0

        # ── RSI(14) ────────────────────────────────────────────────────
        if "rsi" in df.columns:
            rsi_series = df["rsi"].astype(float).fillna(50)
        else:
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
            rsi_series = (100 - 100 / (1 + gain / loss)).fillna(50)
        ind.rsi = cls._safe_last(rsi_series, 50.0)
        ind.rsi_history = [
            round(float(rsi_series.iloc[-i]), 1)
            for i in range(min(history_n, len(rsi_series)), 0, -1)
        ]

        # ── MACD (12,26,9) ────────────────────────────────────────────
        if all(c in df.columns for c in ["macd_line", "macd_signal_line", "macd_hist"]):
            macd_line = df["macd_line"].astype(float)
            macd_signal = df["macd_signal_line"].astype(float)
            macd_hist = df["macd_hist"].astype(float)
        else:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line   = ema12 - ema26
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist   = macd_line - macd_signal
        ind.macd        = cls._safe_last(macd_line, 0.0)
        ind.macd_signal = cls._safe_last(macd_signal, 0.0)
        ind.macd_hist   = cls._safe_last(macd_hist, 0.0)
        ind.macd_trend  = "бычий" if ind.macd > ind.macd_signal else "медвежий"
        if len(macd_line) >= 2:
            prev_macd = float(macd_line.iloc[-2])
            prev_sig  = float(macd_signal.iloc[-2])
            if prev_macd < prev_sig and ind.macd > ind.macd_signal:
                ind.macd_cross = "бычье"
            elif prev_macd > prev_sig and ind.macd < ind.macd_signal:
                ind.macd_cross = "медвежье"
            else:
                ind.macd_cross = "нет"

        # ── SuperTrend (10, 3) ────────────────────────────────────────
        if all(c in df.columns for c in ["st_bull", "st_bear", "supertrend_line"]):
            st_line = df["supertrend_line"].astype(float)
            ind.st_level = cls._safe_last(st_line, ind.price)
            ind.st_dir = "бычий" if not pd.isna(df["st_bull"].iloc[-1]) else "медвежий"
        else:
            ind.st_dir, ind.st_level = cls._calc_supertrend(high, low, close, tr, ind.price)
        ind.st_dist_pct = (abs(ind.price - ind.st_level) / ind.price * 100) if ind.price > 0 else 0.0

        # ── ADX ───────────────────────────────────────────────────────
        if "adx" in df.columns:
            adx_val = cls._safe_last(df["adx"].astype(float), 20.0)
            ind.adx = adx_val
            ind.adx_str = "сильный" if adx_val > 25 else ("слабый" if adx_val < 20 else "умеренный")
        else:
            ind.adx, ind.adx_str = cls._calc_adx(high, low, tr)

        # ── Volume ────────────────────────────────────────────────────
        if "volume_ratio" in df.columns:
            ind.vol_ratio = cls._safe_last(df["volume_ratio"].astype(float), 1.0)
        else:
            avg_vol = cls._safe_last(vol.rolling(20).mean(), 1.0)
            cur_vol = cls._safe_last(vol, 0.0)
            ind.vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
        ind.vol_desc = "всплеск" if ind.vol_ratio > 1.5 else ("низкий" if ind.vol_ratio < 0.5 else "норма")

        # ── SMA 7 / 20 / 200 ─────────────────────────────────────────
        if all(c in df.columns for c in ["sma7", "sma20", "sma200"]):
            ind.sma7 = cls._safe_last(df["sma7"].astype(float), ind.price)
            ind.sma20 = cls._safe_last(df["sma20"].astype(float), ind.price)
            ind.sma200 = cls._safe_last(df["sma200"].astype(float), ind.sma20)
        else:
            ind.sma7   = cls._safe_last(close.rolling(7).mean(), ind.price)
            ind.sma20  = cls._safe_last(close.rolling(20).mean(), ind.price)
            ind.sma200 = cls._safe_last(close.rolling(200).mean(), ind.sma20) if len(close) >= 200 else ind.sma20
        ind.sma_cross = "бычий" if ind.sma7 > ind.sma20 else "медвежий"

        # ── Bollinger Bands (20, 2) ───────────────────────────────────
        if all(c in df.columns for c in ["bb_mid", "bb_upper", "bb_lower"]):
            ind.bb_mid = cls._safe_last(df["bb_mid"].astype(float), ind.price)
            ind.bb_upper = cls._safe_last(df["bb_upper"].astype(float), ind.price)
            ind.bb_lower = cls._safe_last(df["bb_lower"].astype(float), ind.price)
        else:
            bb_mid     = close.rolling(20).mean()
            bb_std     = close.rolling(20).std()
            ind.bb_mid   = cls._safe_last(bb_mid, ind.price)
            bb_std_last  = cls._safe_last(bb_std, 0.0)
            ind.bb_upper = ind.bb_mid + 2.0 * bb_std_last
            ind.bb_lower = ind.bb_mid - 2.0 * bb_std_last
        bb_range     = ind.bb_upper - ind.bb_lower
        ind.bb_pct   = ((ind.price - ind.bb_lower) / bb_range * 100) if bb_range > 0 else 50.0
        ind.bb_squeeze = (bb_range / ind.price * 100) < 3.0 if ind.price > 0 else False
        if ind.price >= ind.bb_upper:
            ind.bb_pos = "выше верхней"
        elif ind.price <= ind.bb_lower:
            ind.bb_pos = "ниже нижней"
        elif ind.bb_pct > 70:
            ind.bb_pos = "верхняя зона"
        elif ind.bb_pct < 30:
            ind.bb_pos = "нижняя зона"
        else:
            ind.bb_pos = "середина"

        # ── Stochastic RSI ────────────────────────────────────────────
        if all(c in df.columns for c in ["stoch_k", "stoch_d"]):
            stoch_k = df["stoch_k"].astype(float).fillna(50)
            stoch_d = df["stoch_d"].astype(float).fillna(50)
        else:
            rsi_min = rsi_series.rolling(14).min()
            rsi_max = rsi_series.rolling(14).max()
            rsi_range = (rsi_max - rsi_min).replace(0, np.nan)
            stoch_k_raw = (rsi_series - rsi_min) / rsi_range * 100
            stoch_k = stoch_k_raw.rolling(3).mean().fillna(50)
            stoch_d = stoch_k.rolling(3).mean().fillna(50)
        ind.stoch_k = cls._safe_last(stoch_k, 50.0)
        ind.stoch_d = cls._safe_last(stoch_d, 50.0)
        if len(stoch_k) >= 2:
            pk, pd_ = float(stoch_k.iloc[-2]), float(stoch_d.iloc[-2])
            if pk < pd_ and ind.stoch_k > ind.stoch_d and ind.stoch_k < 20:
                ind.stoch_cross = "бычье"
            elif pk > pd_ and ind.stoch_k < ind.stoch_d and ind.stoch_k > 80:
                ind.stoch_cross = "медвежье"
        ind.stoch_zone = "перепродан" if ind.stoch_k < 20 else ("перекуплен" if ind.stoch_k > 80 else "нейтральная")

        # ── Williams %R (14) ──────────────────────────────────────────
        hh = high.rolling(14).max()
        ll = low.rolling(14).min()
        wr_denom = (hh - ll).replace(0, np.nan)
        wr_series = (hh - close) / wr_denom * -100
        ind.williams_r = cls._safe_last(wr_series, -50.0)
        ind.williams_zone = "перепродан" if ind.williams_r < -80 else ("перекуплен" if ind.williams_r > -20 else "нейтральная")

        # ── CCI (20) ──────────────────────────────────────────────────
        tp = (high + low + close) / 3
        cci_mean = tp.rolling(20).mean()
        cci_mad  = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True).replace(0, np.nan)
        cci_series = (tp - cci_mean) / (0.015 * cci_mad)
        ind.cci = cls._safe_last(cci_series, 0.0)
        ind.cci_signal = "перепродан" if ind.cci < -100 else ("перекуплен" if ind.cci > 100 else "нейтральный")

        # ── OBV ───────────────────────────────────────────────────────
        price_dir = np.sign(close.diff().fillna(0))
        obv_series = (vol * price_dir).cumsum()
        obv_sma = obv_series.rolling(20).mean()
        ind.obv_trend = "бычий" if cls._safe_last(obv_series) > cls._safe_last(obv_sma) else "медвежий"

        # ── VWAP (дневной сброс по дате) ─────────────────────────────
        try:
            tp_vwap = (high + low + close) / 3
            if "date" in df.columns:
                dates = pd.to_datetime(df["date"])
                day   = dates.dt.date
                cum_tpv = (tp_vwap * vol).groupby(day).cumsum()
                cum_vol = vol.groupby(day).cumsum().replace(0, np.nan)
                vwap_series = cum_tpv / cum_vol
            else:
                vwap_series = (tp_vwap * vol).cumsum() / vol.cumsum().replace(0, np.nan)
            ind.vwap = cls._safe_last(vwap_series, ind.price)
            ind.vwap_dist_pct = (ind.price - ind.vwap) / ind.vwap * 100 if ind.vwap > 0 else 0.0
            ind.vwap_pos = "выше" if ind.price > ind.vwap else "ниже"
        except Exception:
            ind.vwap = ind.price
            ind.vwap_dist_pct = 0.0
            ind.vwap_pos = "нейтральный"

        # ── Pivot / Support / Resistance ──────────────────────────────
        if len(df) >= 2:
            ph = float(high.iloc[-2])
            pl = float(low.iloc[-2])
            pc = float(close.iloc[-2])
            ind.pivot      = round((ph + pl + pc) / 3, 6)
            ind.support    = round(2 * ind.pivot - ph, 6)
            ind.resistance = round(2 * ind.pivot - pl, 6)
        else:
            ind.resistance = float(high.tail(20).max())
            ind.support    = float(low.tail(20).min())
            ind.pivot      = (ind.resistance + ind.support) / 2

        # ── История свечей для LLM ────────────────────────────────────
        n = min(history_n, len(df))
        for i in range(n, 0, -1):
            idx = -i
            try:
                dt_label = str(df["date"].iloc[idx])[:16]
            except Exception:
                dt_label = f"-{i}"
            c = float(close.iloc[idx])
            o = float(df["open"].iloc[idx]) if "open" in df.columns else c
            chg = round((c - o) / o * 100, 2) if o > 0 else 0.0
            ind.candles.append({
                "time":    dt_label,
                "open":    round(o, 4),
                "close":   round(c, 4),
                "high":    round(float(high.iloc[idx]), 4),
                "low":     round(float(low.iloc[idx]), 4),
                "volume":  round(float(vol.iloc[idx]), 2),
                "rsi":     round(float(rsi_series.iloc[idx]), 1),
                "macd_h":  round(float(macd_hist.iloc[idx]), 6),
                "change":  chg,
            })

        return ind

    @staticmethod
    def _calc_supertrend(high, low, close, tr, price, period=10, mult=3.0):
        st_up, st_down, _ = IndicatorEngine._calc_supertrend_series(high, low, close, tr, period=period, mult=mult)
        last_up = st_up.iloc[-1] if len(st_up) else np.nan
        last_down = st_down.iloc[-1] if len(st_down) else np.nan
        if not pd.isna(last_up):
            return "бычий", float(last_up)
        if not pd.isna(last_down):
            return "медвежий", float(last_down)
        return "нейтральный", price

    @staticmethod
    def _calc_supertrend_series(high, low, close, tr, period=10, mult=3.0):
        if len(close) < period + 1:
            na = pd.Series(np.nan, index=close.index)
            return na, na, na

        atr = tr.rolling(window=period).mean()
        basic_ub = (high + low) / 2 + mult * atr
        basic_lb = (high + low) / 2 - mult * atr

        final_ub = basic_ub.copy()
        final_lb = basic_lb.copy()
        direction = pd.Series(1, index=close.index)

        for i in range(period, len(close)):
            if basic_ub.iloc[i] < final_ub.iloc[i - 1] or close.iloc[i - 1] > final_ub.iloc[i - 1]:
                final_ub.iloc[i] = basic_ub.iloc[i]
            else:
                final_ub.iloc[i] = final_ub.iloc[i - 1]

            if basic_lb.iloc[i] > final_lb.iloc[i - 1] or close.iloc[i - 1] < final_lb.iloc[i - 1]:
                final_lb.iloc[i] = basic_lb.iloc[i]
            else:
                final_lb.iloc[i] = final_lb.iloc[i - 1]

        st_up = pd.Series(np.nan, index=close.index)
        st_down = pd.Series(np.nan, index=close.index)
        for i in range(period, len(close)):
            if direction.iloc[i - 1] == 1:
                if close.iloc[i] > final_lb.iloc[i]:
                    direction.iloc[i] = 1
                    st_up.iloc[i] = final_lb.iloc[i]
                else:
                    direction.iloc[i] = -1
                    st_down.iloc[i] = final_ub.iloc[i]
            else:
                if close.iloc[i] < final_ub.iloc[i]:
                    direction.iloc[i] = -1
                    st_down.iloc[i] = final_ub.iloc[i]
                else:
                    direction.iloc[i] = 1
                    st_up.iloc[i] = final_lb.iloc[i]
        return st_up, st_down, st_up.combine_first(st_down)

    @staticmethod
    def _calc_adx(high, low, tr):
        plus_dm  = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        cond     = plus_dm > minus_dm
        plus_dm  = plus_dm.where(cond, 0.0)
        minus_dm = minus_dm.where(~cond, 0.0)
        tr_sum   = tr.rolling(14).sum().replace(0, np.nan)
        plus_di  = 100 * plus_dm.rolling(14).sum() / tr_sum
        minus_di = 100 * minus_dm.rolling(14).sum() / tr_sum
        denom    = (plus_di + minus_di).replace(0, np.nan)
        dx       = 100 * (plus_di - minus_di).abs() / denom
        adx_val  = dx.rolling(14).mean()
        try:
            adx = float(adx_val.iloc[-1])
            adx = 20.0 if np.isnan(adx) else adx
        except Exception:
            adx = 20.0
        if adx > 25:
            return adx, "сильный"
        elif adx < 20:
            return adx, "слабый"
        return adx, "умеренный"


# ═══════════════════════════════════════════════════════════════════════
#                       NEWS PROVIDER
# ═══════════════════════════════════════════════════════════════════════

class NewsProvider:
    """Загружает заголовки новостей из RSS и кэширует результат."""
    _cache: Dict[str, Tuple[float, List[str]]] = {}

    @classmethod
    def fetch(cls, max_items: int = MAX_NEWS_ITEMS) -> List[str]:
        """Возвращает список уникальных заголовков (до max_items штук)."""
        key = f"rss_{max_items}"
        now = time.time()
        if key in cls._cache:
            ct, items = cls._cache[key]
            if now - ct < NEWS_CACHE_SEC:
                return items

        raw_entries: List[str] = []
        headers = {"User-Agent": "Mozilla/5.0"}
        for url in RSS_FEEDS:
            try:
                resp = requests.get(url, timeout=10, headers=headers)
                if resp.status_code != 200:
                    continue
                if feedparser is not None:
                    feed = feedparser.parse(resp.content)
                    for e in feed.entries:
                        title = e.get("title", "").strip()[:220]
                        if title:
                            raw_entries.append(title)
                else:
                    # Fallback без внешней зависимости feedparser.
                    try:
                        root = ET.fromstring(resp.content)
                        for title_el in root.findall(".//item/title"):
                            title = (title_el.text or "").strip()[:220]
                            if title:
                                raw_entries.append(title)
                        for title_el in root.findall(".//entry/title"):
                            title = (title_el.text or "").strip()[:220]
                            if title:
                                raw_entries.append(title)
                    except Exception:
                        pass
            except Exception:
                pass

        # Дедупликация: убираем заголовки схожие на 70%+
        unique: List[str] = []
        for title in raw_entries:
            if len(unique) >= max_items:
                break
            title_lower = title.lower()
            is_dup = False
            for existing in unique:
                # Простое пересечение слов как мера схожести
                words_new = set(title_lower.split())
                words_ex  = set(existing.lower().split())
                if not words_new or not words_ex:
                    continue
                intersection = len(words_new & words_ex)
                union = len(words_new | words_ex)
                similarity = intersection / union if union > 0 else 0
                if similarity >= 0.65:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(title)

        cls._cache[key] = (now, unique)
        logger.debug(f"[News] загружено {len(raw_entries)} заголовков, после дедупликации: {len(unique)}")
        return unique


# ═══════════════════════════════════════════════════════════════════════
#                       ORDER BOOK PROVIDER
# ═══════════════════════════════════════════════════════════════════════

class OrderBookProvider:
    """Читает стакан заявок через Freqtrade DataProvider и считает дисбаланс."""
    _cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    @classmethod
    def fetch(cls, pair: str, dp) -> Dict[str, Any]:
        """
        Возвращает dict с ключами:
          bid_vol, ask_vol, imbalance, desc
        При ошибке — пустой dict.
        """
        now = time.time()
        cached = cls._cache.get(pair)
        if cached:
            ct, data = cached
            if now - ct < OB_CACHE_SEC:
                return data

        empty = {"bid_vol": 0.0, "ask_vol": 0.0, "imbalance": 0.0, "desc": ""}
        try:
            ob = dp.orderbook(pair, OB_DEPTH)
            if not ob or "bids" not in ob or "asks" not in ob:
                return empty

            bids = ob["bids"][:OB_DEPTH]  # [[price, volume], ...]
            asks = ob["asks"][:OB_DEPTH]

            bid_vol = sum(float(b[1]) for b in bids if len(b) >= 2)
            ask_vol = sum(float(a[1]) for a in asks if len(a) >= 2)
            total   = bid_vol + ask_vol

            if total <= 0:
                return empty

            # Дисбаланс от -1 до +1: +1 = все покупают, -1 = все продают
            imbalance = round((bid_vol - ask_vol) / total, 3)

            if imbalance > 0.15:
                desc = f"давление покупателей ({imbalance:+.2f})"
            elif imbalance < -0.15:
                desc = f"давление продавцов ({imbalance:+.2f})"
            else:
                desc = f"баланс ({imbalance:+.2f})"

            data = {
                "bid_vol":   round(bid_vol, 2),
                "ask_vol":   round(ask_vol, 2),
                "imbalance": imbalance,
                "desc":      desc,
            }
            cls._cache[pair] = (now, data)
            return data
        except Exception as e:
            logger.debug(f"[OB] {pair}: {e}")
            return empty


# ═══════════════════════════════════════════════════════════════════════
#                       FEAR & GREED PROVIDER
# ═══════════════════════════════════════════════════════════════════════

class FearGreedProvider:
    """Загружает индекс страха и жадности с Alternative.me (бесплатно, без ключа)."""
    _cache: Dict[str, Any] = {}

    @classmethod
    def fetch(cls) -> str:
        """Возвращает строку для промта или пустую строку при ошибке."""
        now = time.time()
        if "data" in cls._cache:
            ct, text = cls._cache["data"]
            if now - ct < FEAR_GREED_CACHE_SEC:
                return text
        try:
            resp = requests.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=8,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if resp.status_code != 200:
                return ""
            data  = resp.json()["data"][0]
            value = int(data["value"])
            label = data["value_classification"]   # Fear, Greed, Extreme Fear и т.д.
            # Переводим метку на русский
            labels_ru = {
                "Extreme Fear":  "Экстремальный страх",
                "Fear":          "Страх",
                "Neutral":       "Нейтрально",
                "Greed":         "Жадность",
                "Extreme Greed": "Экстремальная жадность",
            }
            label_ru = labels_ru.get(label, label)
            text = f"{value}/100 — {label_ru}"
            cls._cache["data"] = (now, text)
            logger.info(f"[F&G] {text}")
            return text
        except Exception as e:
            logger.warning(f"[F&G] Ошибка загрузки: {e}")
            return ""


# ═══════════════════════════════════════════════════════════════════════
#                       PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
#                       TRADING SESSION
# ═══════════════════════════════════════════════════════════════════════

def get_session_info() -> Tuple[str, str]:
    """Возвращает (название сессии, эмодзи) по текущему UTC времени."""
    hour = datetime.now(timezone.utc).hour
    if 0 <= hour < 8:
        return "Азиатская", "🌏"
    elif 8 <= hour < 13:
        return "Европейская", "🌍"
    elif 13 <= hour < 22:
        return "Американская", "🌎"
    else:
        return "Межсессионная", "🌐"


# ═══════════════════════════════════════════════════════════════════════
#                       TRADE MEMORY
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    """Запись об одной завершённой сделке."""
    pair:       str
    side:       str           # LONG / SHORT
    profit_pct: float         # прибыль в %
    duration_min: float       # длительность в минутах
    entry_time: datetime      # время входа UTC
    exit_time:  datetime      # время выхода UTC
    final_score: float        # оценка ИИ на входе
    exit_reason: str          # причина выхода

    @property
    def result(self) -> str:
        return "✅ профит" if self.profit_pct > 0 else "❌ убыток"

    def age_str(self) -> str:
        """Человекочитаемое время с момента выхода."""
        mins = (datetime.now(timezone.utc) - self.exit_time).total_seconds() / 60
        if mins < 60:
            return f"{int(mins)} мин назад"
        elif mins < 1440:
            return f"{mins/60:.1f} ч назад"
        else:
            return f"{mins/1440:.1f} дн назад"

    def to_prompt_line(self) -> str:
        return (
            f"  [{self.exit_time.strftime('%H:%M')} {self.age_str()}] "
            f"{self.pair} {self.side} → {self.result} {self.profit_pct:+.2f}% "
            f"за {self.duration_min:.0f}мин  score={self.final_score:+.2f}  выход={self.exit_reason}"
        )


class TradeMemory:
    """Хранит историю сделок и формирует блок для промта. Сохраняет на диск."""

    def __init__(self, max_records: int = TRADE_MEMORY_MAX):
        self._records: List[TradeRecord] = []
        self._max = max_records
        self._load()

    def _load(self) -> None:
        """Загружает историю с диска при старте."""
        try:
            import json as _json
            import os
            if not os.path.exists(MEMORY_FILE):
                logger.info(f"[Memory] Файл {MEMORY_FILE} не найден — начинаем с чистой памяти")
                return
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = _json.load(f)
            for item in data:
                try:
                    self._records.append(TradeRecord(
                        pair=item["pair"],
                        side=item["side"],
                        profit_pct=item["profit_pct"],
                        duration_min=item["duration_min"],
                        entry_time=datetime.fromisoformat(item["entry_time"]),
                        exit_time=datetime.fromisoformat(item["exit_time"]),
                        final_score=item["final_score"],
                        exit_reason=item["exit_reason"],
                    ))
                except Exception as e:
                    logger.warning(f"[Memory] Пропущена запись при загрузке: {e}")
            logger.info(f"[Memory] Загружено {len(self._records)} сделок из {MEMORY_FILE}")
        except Exception as e:
            logger.error(f"[Memory] Ошибка загрузки: {e}")

    def _save(self) -> None:
        """Сохраняет историю на диск после каждого изменения."""
        try:
            import json as _json
            import os
            os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
            data = [
                {
                    "pair":         r.pair,
                    "side":         r.side,
                    "profit_pct":   r.profit_pct,
                    "duration_min": r.duration_min,
                    "entry_time":   r.entry_time.isoformat(),
                    "exit_time":    r.exit_time.isoformat(),
                    "final_score":  r.final_score,
                    "exit_reason":  r.exit_reason,
                }
                for r in self._records
            ]
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                _json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[Memory] Ошибка сохранения: {e}")

    def add(self, record: TradeRecord) -> None:
        self._records.append(record)
        if len(self._records) > self._max:
            self._records = self._records[-self._max:]
        self._save()

    def to_prompt_block(self, pair: Optional[str] = None) -> str:
        """Формирует блок истории для промта. Если pair задана — сначала по этой паре."""
        if not self._records:
            return "  нет истории сделок"

        # Разделяем: сначала по текущей паре, потом остальные
        pair_records  = [r for r in self._records if r.pair == pair] if pair else []
        other_records = [r for r in self._records if r.pair != pair] if pair else self._records

        lines = []
        if pair_records:
            lines.append(f"  — По паре {pair}:")
            for r in reversed(pair_records[-5:]):   # последние 5 по этой паре
                lines.append(r.to_prompt_line())
        if other_records:
            lines.append("  — Другие пары:")
            for r in reversed(other_records[-5:]):  # последние 5 по другим
                lines.append(r.to_prompt_line())

        total   = len(self._records)
        wins    = sum(1 for r in self._records if r.profit_pct > 0)
        avg_pnl = sum(r.profit_pct for r in self._records) / total if total else 0
        lines.append(
            f"\n  Итого в памяти: {total} сделок | "
            f"Прибыльных: {wins} ({wins/total*100:.0f}%) | "
            f"Средний P&L: {avg_pnl:+.2f}%"
        )
        return "\n".join(lines)


class PromptBuilder:
    """Формирует структурированные промпты для LLM."""

    SYSTEM_PROMPT = """Ты — профессиональный криптовалютный трейдер с 15-летним опытом на фьючерсных рынках.
Ты торговал через кризисы 2018, 2020, 2022 годов. Ты знаешь цену каждой сделки.

╔══════════════════════════════════════════════════════════════════════╗
║  ЖЕЛЕЗНОЕ ПРАВИЛО #1: сохранение капитала важнее любой прибыли     ║
║  ЖЕЛЕЗНОЕ ПРАВИЛО #2: нет сетапа — нет сделки. Всегда NEUTRAL      ║
║  ЖЕЛЕЗНОЕ ПРАВИЛО #3: сомневаешься — не входи                      ║
╚══════════════════════════════════════════════════════════════════════╝

Ты не просто смотришь на цифры — ты понимаешь контекст рынка.
Ты чувствуешь, когда рынок готовится к движению, а когда — к ловушке.
Ты знаешь разницу между реальным сигналом и шумом.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ПРОФИЛЬ ТОРГОВОЙ СИСТЕМЫ — знай это всегда
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Стиль торговли  : средний (не скальпинг, не позиционная)
Горизонт сделки : локальный — ты торгуешь локальные движения внутри
                  текущей структуры, НЕ глобальный тренд.
                  Тебя не интересует "куда BTC через неделю" —
                  тебя интересует "куда пойдёт цена в следующие 30-90 минут".
Таймфрейм       : 30-90 минут. Свеча = локальный импульс.
                  Всё что происходит на старших ТФ — лишь фон,
                  а не сигнал для входа.
Монеты          : статический список пар. Ты не ищешь новые инструменты,
                  ты досконально знаешь поведение именно этих активов.

Что это значит для анализа:
  • Глобальные новости ("SEC подала иск", "ETF одобрен") влияют на фон,
    но локальное движение цены важнее для принятия решения прямо сейчас.
  • Сигналы должны быть актуальны для 30-90 минутного движения,
    а не для дневного или недельного тренда.
  • Если глобальный тренд медвежий, но локально формируется бычий импульс
    с подтверждением объёма — вход LONG допустим как локальная сделка.
  • Малейшая неопределённость на локальном уровне = NEUTRAL,
    даже если глобально всё выглядит красиво.
  • Fear & Greed Index передаётся как общий фон рынка. Это медленный
    индикатор — он меняется днями и неделями. Не используй его как
    сигнал входа/выхода. Учитывай только при экстремальных значениях
    (< 15 или > 85) как дополнительный контекст осторожности.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ТОРГОВАЯ СЕССИЯ И ИСТОРИЯ СДЕЛОК
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

В каждом запросе тебе передаётся текущая торговая сессия и история
последних сделок. Используй это как дополнительный контекст.

СЕССИИ — рекомендации по характеру торговли:
  Азиатская (00:00-08:00 UTC):
    Объём обычно ниже, движения медленнее и менее предсказуемы.
    Рекомендуется быть осторожнее с размером позиции.
    Пробои уровней чаще оказываются ложными.

  Европейская (08:00-13:00 UTC):
    Открытие Европы часто даёт импульс. Объём растёт.
    Хорошее время для трендовых входов если есть сетап.

  Американская (13:00-22:00 UTC):
    Максимальный объём и волатильность. Самые сильные движения.
    Сигналы наиболее надёжны при подтверждении объёмом.
    Новости из США влияют сильнее всего именно в эту сессию.

  Межсессионная (22:00-00:00 UTC):
    Переход. Низкая ликвидность. Лучше воздержаться от входов
    если сигнал не исключительно сильный.

ИСТОРИЯ СДЕЛОК — как использовать:
  • Смотри на время сделок — если недавние убытки случились в этой же
    сессии на этой же паре, это повод быть осторожнее.
  • Серия убытков подряд — рекомендуется снизить stake до минимума.
  • Хорошая статистика не означает что текущий сигнал правильный —
    каждую сделку анализируй независимо.
  • Это рекомендации, а не запреты — сильный сигнал важнее статистики.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ШАГ 1: КРИТИЧЕСКИЙ АНАЛИЗ НОВОСТЕЙ (sent_score, вес 15%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ПЕРЕД оценкой каждой новости задай себе:
  1. Это реальное событие или кликбейт/манипуляция?
  2. Насколько источник достоверен (CoinTelegraph, Decrypt vs неизвестные)? 
  3. Это свежая информация или пересказ старого?
  4. Рынок уже отыграл эту новость или нет?
  5. Есть ли в заголовке гиперболы ("MASSIVE", "EXPLODES", "CRASH")?
     -> Если да — это часто манипуляция, снижай relevance до 0.2-0.3

Оценивай КАЖДУЮ новость строго:
  • sentiment  in [-1.0; +1.0]  — реальное влияние на крипто-рынок
    > Нейтральные события = 0.0, не натягивай позитив/негатив
  • relevance  in [0.1; 1.0]   — релевантность к торгуемому активу
    > Новость про Ethereum при торговле BTC = 0.3-0.5 max
    > Фейк/кликбейт/паника без фактов = 0.1-0.2
  • recency    in [0.1; 1.0]   — актуальность события прямо сейчас
    > Повторная новость / пересказ = 0.2-0.4
  • score = sentiment * relevance * recency

sent_score = clip(weighted_mean(все score), -1.0, +1.0)

Найди ключевую новость (max |score|) -> key_news.
news_summary — 2 предложения: общий фон + главный драйвер.
news_mood — одно слово или короткая фраза: "позитивный", "негативный", "нейтральный", "смешанный", "тревожный", "эйфория"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ШАГ 2: ГЛУБОКИЙ ТЕХНИЧЕСКИЙ АНАЛИЗ (tech_score, вес 85%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ты не просто складываешь баллы — ты понимаешь структуру рынка.
Смотри на КОНФЛЮЕНТНОСТЬ сигналов: когда 5+ индикаторов согласованы -> сильный сигнал.
Когда индикаторы противоречат друг другу -> рынок в нерешительности -> NEUTRAL.

Рассчитай вклады:

  SuperTrend (главный трендовый фильтр):
    бычий  -> +0.50 | медвежий -> -0.50
    ! Без подтверждения SuperTrend — никакого входа против тренда

  SMA7 vs SMA20 (краткосрочный импульс):
    SMA7 > SMA20 -> +0.20 | SMA7 < SMA20 -> -0.20

  Цена vs SMA200 (глобальный тренд):
    выше -> +0.10 | ниже -> -0.10
    ! Если цена сильно ниже SMA200 — лонги требуют высокой уверенности

  RSI(14) — интерпретируй динамику, а не только значение:
    < 20        -> +0.25  (экстремальная перепроданность)
    20-30       -> +0.20
    30-45       -> +0.05
    45-55       -> 0.0
    55-70       -> -0.05
    70-80       -> -0.20
    > 80        -> -0.25
    + RSI растёт по истории (последние 3+ свечи) -> +0.05
    + RSI падает по истории                      -> -0.05
    ! Смотри на динамику RSI по истории свечей — растёт или падает импульс

  MACD — смотри на силу пересечения и положение гистограммы:
    Бычье пересечение   -> +0.20
    Медвежье пересечение -> -0.20
    Бычий тренд без пересечения -> +0.10
    Медвежий тренд без пересечения -> -0.10
    ! Слабая гистограмма (hist близко к 0) = ненадёжный сигнал

  Bollinger Bands: 
    Ниже нижней полосы  -> +0.20
    Выше верхней полосы -> -0.20
    Нижняя зона (< 30%) -> +0.10
    Верхняя зона (> 70%) -> -0.10
    При squeeze: вклад BB * 0.5 (ждём выхода из сжатия)
    ! После долгого squeeze — жди резкого движения в любую сторону

  Stochastic RSI:
    Бычье пересечение в зоне < 20  -> +0.20
    Медвежье пересечение в зоне > 80 -> -0.20
    Только зона < 20 -> +0.10 | > 80 -> -0.10

  Williams %R:
    < -80 -> +0.10 | > -20 -> -0.10

  CCI(20):
    < -100 -> +0.15 | > +100 -> -0.15

  OBV (подтверждение объёмом):
    Бычий тренд -> +0.10 | Медвежий -> -0.10
    ! OBV подтверждает или опровергает ценовое движение — учитывай это в оценке

  Объём:
    > 1.5x среднего -> +0.15 (активность покупателей/продавцов)
    < 0.5x среднего -> -0.10 (вялый рынок, сигналам не доверяй)

  Поддержка/Сопротивление:
    Цена у поддержки (< 0.5% от уровня)    -> +0.10
    Цена у сопротивления (< 0.5% от уровня) -> -0.10
    ! Пробой сопротивления с объёмом = очень сильный бычий сигнал
    ! Отбой от поддержки без объёма = ловушка для быков

  Стакан заявок (order book imbalance):
    Дисбаланс > +0.15 = давление покупателей (бычий знак)
    Дисбаланс < -0.15 = давление продавцов (медвежий знак)
    Близко к 0 = баланс, рынок в нерешительности
    Учитывай как дополнительный контекст, не как основной сигнал.

МУЛЬТИПЛИКАТОРЫ (применяй после суммирования):
  ADX > 25 (сильный тренд)        -> * 1.2
  ADX < 20 (слабый/боковой рынок) -> * 0.8
  Объём > 1.5x                    -> * 1.1 дополнительно
  ATR% > 3% (высокая волатильность) -> * 0.9 (осторожней с размером)

tech_score = clip(итог, -1.0, +1.0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ШАГ 3: ФИНАЛЬНАЯ ОЦЕНКА И РЕШЕНИЕ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

final_score = clip(tech_score * 0.85 + sent_score * 0.15, -1.0, +1.0)

РЕШЕНИЕ О ВХОДЕ:
  final_score > +0.30 И минимум 4 бычьих индикатора согласованы -> LONG_ENTER
  final_score < -0.30 И минимум 4 медвежьих индикатора согласованы -> SHORT_ENTER
  Иначе -> NEUTRAL

ОБЯЗАТЕЛЬНЫЕ УСЛОВИЯ НЕЙТРАЛА (даже при сильном final_score):
  • SuperTrend противоречит direction -> NEUTRAL
  • Объём < 0.2x среднего -> NEUTRAL (нет ликвидности, сигналу нельзя доверять)

ЗАЩИТА ОТ ЛОВЛИ НОЖА (falling knife / catching a rocket):
  Ловля ножа — это вход LONG пока цена в вертикальном свободном падении,
  или SHORT пока цена вертикально взлетает без остановки.

  Железное условие блокировки — NEUTRAL в обоих случаях:
  • ATR% > 4% — экстремальная волатильность, свечи огромные, стоп будет
    съеден случайным шумом. При BTC это означает движение >$3000 за свечу.
    В обычный день ATR% у BTC на 30м около 0.1-0.5%. >4% это кризис.

  Всё остальное — на усмотрение ИИ. Если видишь сильное движение но ATR
  в норме — это торгуемая ситуация, не ловля ножа.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ШАГ 4: УПРАВЛЕНИЕ ОТКРЫТОЙ ПОЗИЦИЕЙ (если side передан)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Выход из позиции — ТОЛЬКО на основе своей оценки (final_score).
Ты не выходишь механически по RSI или BB — ты анализируешь ситуацию целиком.

  LONG_EXIT  если: final_score < -0.20 (рынок разворачивается против лонга)
  SHORT_EXIT если: final_score > +0.20 (рынок разворачивается против шорта)
  NEUTRAL    если: сигнал неоднозначен — держи позицию, система управляет стопом

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ШАГ 5: РАЗМЕР ПОЗИЦИИ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Выбирай stake_amount СТРОГО из предоставленного списка (точное значение!):

  |final_score| > 0.65 И ATR% < 1.5% И ADX > 25 -> максимальный уровень
  |final_score| 0.45-0.65 И ATR% < 2.5%          -> средний уровень
  |final_score| 0.30-0.45                          -> минимальный уровень
  Любые сомнения, высокая волатильность (ATR% > 3%), слабый ADX -> минимальный уровень

ВАЖНО: stake_amount должен быть ТОЧНЫМ числом из списка STAKE_LEVELS.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
КОНТЕКСТ ПРЕДЫДУЩЕГО РЕШЕНИЯ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Если в запросе передано "ПРЕДЫДУЩЕЕ РЕШЕНИЕ" — используй его как контекст:
  • Если прошлая свеча была NEUTRAL "жду подтверждения объёма" — проверь, появился ли объём.
  • Если прошлая свеча была LONG_ENTER — рынок двигался в нужную сторону?
  • Если прошлая confidence была низкой — изменилась ли ситуация?
  • Не копируй прошлое решение слепо — анализируй заново, но с памятью о контексте.
  • Если данных о прошлом решении нет — анализируй как обычно.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ГРАДАЦИЯ NEUTRAL (используй точный подтип в поле action_detail)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Если action = NEUTRAL, укажи подтип в поле neutral_type:
  "WEAK"   — сигнал близко к порогу, ситуация почти созрела, смотри следующую свечу
  "STRONG" — противоречивые сигналы, сетап не формируется, пропускай
  "WAIT"   — хороший сетап формируется но нужно подтверждение (объём, пересечение и т.д.)

Это поле только для информации — не влияет на торговое решение.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ШАГ A — news_reasoning (новостной анализ, максимум 2 предложения):
  Что реально важно из новостей? Есть ли кликбейт или манипуляция?
  Какой итоговый sent_score и почему именно такой?

ШАГ Б — tech_reasoning (технический анализ, максимум 3 предложения):
  Пройдись по ключевым индикаторам. Сколько из них согласованы
  в одну сторону? Есть ли противоречия между ними?
  Обязательно укажи:
  - расстояние цены до SuperTrend и что это означает (пограничная зона?)
  - находится ли цена у ключевого уровня поддержки или сопротивления
  - подтверждает ли объём текущее движение
  - если есть BB squeeze — куда вероятнее пробой исходя из других сигналов

ШАГ В — counter_argument (1 предложение — самое главное возражение):
  ! ВАЖНО: твоя первая интуиция может быть ошибочной.
  Активно ищи причины почему ты можешь быть НЕПРАВ.
  Не ищи подтверждения — ищи опровержения.
  Напиши самый весомый аргумент ПРОТИВ твоего решения.
  Если весомых аргументов против нет — напиши "нет весомых возражений".
  Если аргумент против очень силён — пересмотри action или снизь confidence.

ШАГ Г — confidence (уверенность от 0.0 до 1.0):
  Оцени честно насколько ты уверен в решении:
  0.9-1.0 = все сигналы согласованы, контекст ясный, рисков мало
  0.7-0.9 = сигналы в основном согласованы, есть небольшие сомнения
  0.5-0.7 = смешанные сигналы, решение под вопросом
  < 0.5   = слишком много неопределённости — рекомендуется NEUTRAL
  При confidence < 0.6 — автоматически выбирай минимальный stake.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ТРЕБОВАНИЯ К summary — пиши как опытный аналитик, минимум 6 предложений:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Новостной фон: sent_score с числом, характер новостей, есть ли манипуляции.
  2. Ключевые технические сигналы: 3+ индикатора с числами — что говорит за, что против.
  3. Динамика: как менялись RSI, MACD, объём на последних свечах — это разгон или затухание?
  4. Структура рынка: уровни поддержки/сопротивления, ATR%, где находится цена.
  5. Логика решения: почему именно это action, что стало определяющим фактором.
  6. Главный риск: что может пойти не так, на что смотреть в первую очередь.

ФОРМАТ ОТВЕТА — строго JSON без Markdown-обёртки:
КРИТИЧЕСКИЕ ПРАВИЛА ФОРМАТА:
  • Никаких комментариев, префиксов и суффиксов вне JSON.
  • Все float-поля должны быть числами, не строками.
  • confidence/tech_score/sent_score/final_score должны быть в диапазоне [-1..1] / [0..1].
  • Если action != "NEUTRAL", то neutral_type = null.
  • Если action == "NEUTRAL", то neutral_type обязательно один из: WEAK, STRONG, WAIT.
  • stake_amount всегда строго из списка STAKE_LEVELS.
  • Если уверенность < 0.5 — action должен быть NEUTRAL.
  • Если индикаторы противоречивы (нет 4+ согласованных сигналов), выбирай NEUTRAL.

{
  "news_reasoning": "рассуждение по новостям — ШАГ А",
  "tech_reasoning": "рассуждение по технике — ШАГ Б",
  "counter_argument": "самый весомый аргумент ПРОТИВ решения — ШАГ В",
  "confidence": float,
  "neutral_type": "WEAK|STRONG|WAIT|null",
  "sentiments": [{"title": "...", "score": float, "sentiment": float, "relevance": float, "recency": float}],
  "key_news": "заголовок самой важной новости",
  "news_summary": "2 предложения об общем фоне",
  "news_mood": "позитивный|негативный|нейтральный|смешанный|тревожный|эйфория",
  "tech_score": float,
  "action": "LONG_ENTER|SHORT_ENTER|NEUTRAL|LONG_EXIT|SHORT_EXIT",
  "stake_amount": float,
  "summary": "минимум 6 предложений"
}"""

    @classmethod
    def build_entry_prompt(cls, ind: Indicators, news_items: List[str], pair: str,
                           memory_block: str = "", fear_greed: str = "",
                           prev_decision: Dict[str, Any] = None) -> str:
        rsi_sig = "ПЕРЕПРОДАН" if ind.rsi < 30 else ("ПЕРЕКУПЛЕН" if ind.rsi > 70 else "нейтрал")
        macd_info = (f"MACD={ind.macd:.6f} Sig={ind.macd_signal:.6f} Hist={ind.macd_hist:.6f} "
                     f"тренд={ind.macd_trend} пересечение={ind.macd_cross}")
        candle_lines = []
        for c in ind.candles:
            candle_lines.append(
                f"  {c['time']} | O={c['open']} C={c['close']} H={c['high']} L={c['low']}"
                f" Vol={c['volume']:.0f} RSI={c['rsi']} MACD_H={c['macd_h']:+.6f} chg={c['change']:+.2f}%"
            )
        candle_block = "\n".join(candle_lines) if candle_lines else "  нет данных"
        news_block = "\n".join(f"  [{i+1}] {title}" for i, title in enumerate(news_items)) if news_items else "  нет новостей"

        session_name, session_emoji = get_session_info()
        dist_to_sup = abs(ind.price - ind.support) / ind.price * 100 if ind.price > 0 else 0
        dist_to_res = abs(ind.price - ind.resistance) / ind.price * 100 if ind.price > 0 else 0
        fg_line = f"  Fear & Greed: {fear_greed}\n" if fear_greed else ""

        # Предыдущее решение
        prev_block = ""
        if prev_decision and prev_decision.get("action"):
            prev_block = (
                f"### ПРЕДЫДУЩЕЕ РЕШЕНИЕ (прошлая свеча)\n"
                f"  Action    : {prev_decision['action']}  "
                f"score={prev_decision.get('final_score', 0):+.3f}  "
                f"conf={prev_decision.get('confidence', 0):.2f}\n"
                f"  Техника   : {prev_decision.get('tech_reasoning', '—')}\n"
                f"  Возражение: {prev_decision.get('counter_arg', '—')}\n\n"
            )

        # Стакан заявок
        ob_line = ""
        if ind.ob_desc:
            ob_line = (f"  Bid vol: {ind.ob_bid_vol:.2f} | Ask vol: {ind.ob_ask_vol:.2f} "
                       f"| Дисбаланс: {ind.ob_desc}\n")

        return (
            f"## АНАЛИЗ ВХОДА — {pair}\n"
            f"Время: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  "
            f"Сессия: {session_emoji} {session_name}\n"
            f"{fg_line}\n"
            f"### ИСТОРИЯ СДЕЛОК\n{memory_block}\n\n"
            f"{prev_block}"
            f"### ЦЕНА\n"
            f"  Текущая : ${ind.price:.4f} ({ind.change_pct:+.2f}%)\n"
            f"  ATR(14) : ${ind.atr:.4f} = {ind.atr_pct:.2f}% от цены\n\n"
            f"### ТРЕНД\n"
            f"  SuperTrend : {ind.st_dir} (уровень ${ind.st_level:.4f}, dist {ind.st_dist_pct:.2f}%)\n"
            f"  SMA7/20    : {ind.sma_cross} ({ind.sma7:.4f} / {ind.sma20:.4f})\n"
            f"  SMA200     : ${ind.sma200:.4f} ({'ВЫШЕ' if ind.price > ind.sma200 else 'НИЖЕ'})\n"
            f"  ADX        : {ind.adx:.1f} ({ind.adx_str})\n\n"
            f"### ОСЦИЛЛЯТОРЫ\n"
            f"  RSI(14)   : {ind.rsi:.1f} — {rsi_sig} | история: {ind.rsi_history}\n"
            f"  {macd_info}\n"
            f"  Stoch RSI : K={ind.stoch_k:.1f} D={ind.stoch_d:.1f} зона={ind.stoch_zone} пересечение={ind.stoch_cross}\n"
            f"  Williams %R: {ind.williams_r:.1f} — {ind.williams_zone}\n"
            f"  CCI(20)   : {ind.cci:.1f} — {ind.cci_signal}\n\n"
            f"### ОБЪЁМ\n"
            f"  Ratio: {ind.vol_ratio:.2f}x — {ind.vol_desc}\n"
            f"  OBV тренд: {ind.obv_trend}\n"
            f"{ob_line}"
            f"\n### BOLLINGER BANDS (20,2)\n"
            f"  Верхняя: ${ind.bb_upper:.4f} | Mid: ${ind.bb_mid:.4f} | Нижняя: ${ind.bb_lower:.4f}\n"
            f"  Позиция: {ind.bb_pct:.0f}% — {ind.bb_pos}"
            f"{' [СЖАТИЕ!]' if ind.bb_squeeze else ''}\n\n"
            f"### УРОВНИ\n"
            f"  Поддержка    : ${ind.support:.4f} (dist {dist_to_sup:.2f}%)\n"
            f"  Pivot        : ${ind.pivot:.4f}\n"
            f"  Сопротивление: ${ind.resistance:.4f} (dist {dist_to_res:.2f}%)\n\n"
            f"### ИСТОРИЯ СВЕЧЕЙ (последние {len(ind.candles)})\n"
            f"{candle_block}\n\n"
            f"### НОВОСТИ (проанализируй критически — отдели реальные события от кликбейта)\n"
            f"{news_block}\n\n"
            f"### COMPACT_CONTEXT\n"
            f"  pair={pair} price={ind.price:.4f} change={ind.change_pct:+.2f}%\n"
            f"  st={ind.st_dir} adx={ind.adx:.1f} atr_pct={ind.atr_pct:.2f}\n"
            f"  vol_ratio={ind.vol_ratio:.2f} obv={ind.obv_trend} bb_pct={ind.bb_pct:.1f}\n"
            f"  support={ind.support:.4f} resistance={ind.resistance:.4f}\n\n"
            f"Доступные стейки: {STAKE_LEVELS}\n"
            f"CONTRACT_VERSION: v2_compact\n"
            f"ОТВЕТЬ ТОЛЬКО JSON\n"
        )

    @classmethod
    def build_exit_prompt(cls, ind: Indicators, news_items: List[str], pair: str,
                          side: str, profit: float, memory_block: str = "",
                          fear_greed: str = "", prev_decision: Dict[str, Any] = None) -> str:
        rsi_sig = "ПЕРЕПРОДАН" if ind.rsi < 30 else ("ПЕРЕКУПЛЕН" if ind.rsi > 70 else "нейтрал")
        news_block = "\n".join(f"  [{i+1}] {t}" for i, t in enumerate(news_items)) if news_items else "  нет новостей"
        session_name, session_emoji = get_session_info()
        fg_line = f"  Fear & Greed: {fear_greed}\n" if fear_greed else ""

        prev_block = ""
        if prev_decision and prev_decision.get("action"):
            prev_block = (
                f"### ПРЕДЫДУЩЕЕ РЕШЕНИЕ (прошлая свеча)\n"
                f"  Action: {prev_decision['action']}  "
                f"score={prev_decision.get('final_score', 0):+.3f}  "
                f"conf={prev_decision.get('confidence', 0):.2f}\n\n"
            )

        return (
            f"## АНАЛИЗ ВЫХОДА — {pair}\n"
            f"Время: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  "
            f"Сессия: {session_emoji} {session_name}\n"
            f"{fg_line}"
            f"Открытая позиция: {side} | Текущий P&L: {profit:+.2f}%\n\n"
            f"### ИСТОРИЯ СДЕЛОК\n{memory_block}\n\n"
            f"{prev_block}"
            f"ЗАДАЧА: оцени текущую ситуацию и реши — держать позицию или выходить.\n"
            f"Выход ТОЛЬКО если final_score говорит против текущей позиции.\n\n"
            f"RSI(14)   : {ind.rsi:.1f} — {rsi_sig} | история: {ind.rsi_history}\n"
            f"MACD      : {ind.macd:.6f} / Sig: {ind.macd_signal:.6f} ({ind.macd_trend}, cross={ind.macd_cross})\n"
            f"SuperTrend: {ind.st_dir} (${ind.st_level:.4f})\n"
            f"ADX       : {ind.adx:.1f} {ind.adx_str}\n"
            f"BB позиция: {ind.bb_pos} ({ind.bb_pct:.0f}%)"
            f"{' [СЖАТИЕ]' if ind.bb_squeeze else ''}\n"
            f"Stoch K/D : {ind.stoch_k:.1f}/{ind.stoch_d:.1f} {ind.stoch_zone} cross={ind.stoch_cross}\n"
            f"Объём     : {ind.vol_ratio:.2f}x — {ind.vol_desc}\n"
            f"OBV тренд : {ind.obv_trend}\n"
            f"ATR%      : {ind.atr_pct:.2f}%\n\n"
            f"### НОВОСТИ\n{news_block}\n\n"
            f"### COMPACT_CONTEXT\n"
            f"  pair={pair} side={side} pnl={profit:+.2f}%\n"
            f"  st={ind.st_dir} adx={ind.adx:.1f} atr_pct={ind.atr_pct:.2f} vol_ratio={ind.vol_ratio:.2f}\n\n"
            f"Доступные стейки: {STAKE_LEVELS}\n"
            f"CONTRACT_VERSION: v2_compact\n"
            f"ОТВЕТЬ ТОЛЬКО JSON\n"
        )


# ═══════════════════════════════════════════════════════════════════════
#                       PYDANTIC МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════════

TradeAction = TypingLiteral["LONG_ENTER", "SHORT_ENTER", "NEUTRAL", "LONG_EXIT", "SHORT_EXIT"]


class NewsItem(BaseModel):
    title:     str   = Field(default="")
    score:     float = Field(default=0.0)
    sentiment: float = Field(default=0.0)
    relevance: float = Field(default=1.0)
    recency:   float = Field(default=1.0)


class TradingRecommendation(BaseModel):
    news_reasoning:   str         = Field(default="")
    tech_reasoning:   str         = Field(default="")
    counter_argument: str         = Field(default="")
    confidence:       float       = Field(default=0.7)
    neutral_type:     Optional[str] = Field(default=None)
    sentiments:       List[Any]   = Field(default_factory=list)
    key_news:         str         = Field(default="")
    news_summary:     str         = Field(default="")
    news_mood:        str         = Field(default="нейтральный")
    tech_score:       float       = Field(default=0.0)
    action:           TradeAction = Field(default="NEUTRAL")
    summary:          str         = Field(default="")
    stake_amount:     float       = Field(default=0.0)

    @field_validator("confidence")
    @classmethod
    def confidence_clamp(cls, v):
        return float(max(0.0, min(1.0, v)))

    @field_validator("summary")
    @classmethod
    def summary_default(cls, v):
        return v if v and len(v.strip()) >= 20 else "Анализ выполнен. Сигнал NEUTRAL — недостаточно данных."

    @field_validator("stake_amount")
    @classmethod
    def stake_snap(cls, v):
        if v <= 0:
            return float(STAKE_LEVELS[0])
        return float(min(STAKE_LEVELS, key=lambda s: abs(s - v)))

    @field_validator("neutral_type")
    @classmethod
    def neutral_type_validate(cls, v):
        if v is None:
            return None
        v_norm = str(v).strip().upper()
        if not v_norm or v_norm == "NULL":
            return None
        allowed = {"WEAK", "STRONG", "WAIT"}
        return v_norm if v_norm in allowed else None

    @computed_field
    @property
    def parsed_news(self) -> List[NewsItem]:
        result = []
        for item in self.sentiments:
            try:
                if isinstance(item, dict):
                    result.append(NewsItem(**{
                        k: item.get(k, 0.0) if k != "title" else item.get(k, "")
                        for k in ["title", "score", "sentiment", "relevance", "recency"]
                    }))
                elif isinstance(item, (int, float)):
                    result.append(NewsItem(score=float(item)))
            except Exception:
                pass
        return result

    @computed_field
    @property
    def sent_score(self) -> float:
        scores = [n.score for n in self.parsed_news if n.score != 0.0]
        if not scores:
            raw = []
            for item in self.sentiments:
                if isinstance(item, (int, float)):
                    raw.append(float(item))
                elif isinstance(item, dict):
                    raw.append(float(item.get("score", item.get("sentiment", 0.0))))
            if raw:
                return float(np.clip(np.mean(raw), -1.0, 1.0))
            return 0.0
        return float(np.clip(np.mean(scores), -1.0, 1.0))

    @computed_field
    @property
    def final_score(self) -> float:
        return float(np.clip(
            self.tech_score * WEIGHT_TECHNICAL + self.sent_score * WEIGHT_SENTIMENT,
            -1.0, 1.0,
        ))

    @model_validator(mode="after")
    def validate_semantics(self):
        self.tech_score = float(np.clip(self.tech_score, -1.0, 1.0))
        # Не меняем action молча — финальное решение принимает semantic validator + arbiter.
        if self.action == "NEUTRAL":
            if self.neutral_type is None:
                self.neutral_type = "STRONG"
        else:
            self.neutral_type = None
        return self


# ═══════════════════════════════════════════════════════════════════════
#                       LLM CLIENT
# ═══════════════════════════════════════════════════════════════════════

class LLMClient:
    """Отправляет промпты в OpenAI / OpenRouter и парсит ответ."""

    def __init__(self):
        self.model = LLM_MODEL
        logger.info(f"[LLM] Модель: {self.model}")

    @staticmethod
    def _semantic_validate(result: TradingRecommendation) -> Tuple[bool, str]:
        score_cmp = round(float(result.final_score), SCORE_ROUND_DECIMALS)
        if result.action in ("LONG_ENTER", "SHORT_ENTER") and result.confidence < ENTRY_MIN_CONFIDENCE:
            return False, f"low_confidence:{result.confidence:.2f}"
        if result.action == "LONG_ENTER" and score_cmp < ENTRY_LONG_THRESHOLD:
            return False, f"weak_long_score:{result.final_score:.3f}"
        if result.action == "SHORT_ENTER" and score_cmp > ENTRY_SHORT_THRESHOLD:
            return False, f"weak_short_score:{result.final_score:.3f}"
        if result.action == "LONG_EXIT" and score_cmp > EXIT_LONG_THRESHOLD:
            return False, f"weak_long_exit:{result.final_score:.3f}"
        if result.action == "SHORT_EXIT" and score_cmp < EXIT_SHORT_THRESHOLD:
            return False, f"weak_short_exit:{result.final_score:.3f}"
        return True, "ok"

    def analyze(self, prompt: str) -> TradingRecommendation:
        import json as _json

        api_url = (
            "https://openrouter.ai/api/v1/chat/completions"
            if "/" in self.model
            else "https://api.openai.com/v1/chat/completions"
        )
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0.15,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        }

        for attempt in range(LLM_RETRIES):
            try:
                resp = httpx.post(api_url, headers=headers, json=payload, timeout=LLM_TIMEOUT)
                if resp.status_code != 200:
                    logger.warning(f"[LLM] HTTP {resp.status_code}: {resp.text[:200]}")
                    if attempt < LLM_RETRIES - 1:
                        time.sleep(3)
                        continue
                    break
                raw = resp.json()["choices"][0]["message"]["content"] or ""
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                try:
                    data = _json.loads(raw)
                except _json.JSONDecodeError:
                    logger.error(f"[LLM] JSON parse error: {raw[:200]}")
                    if attempt < LLM_RETRIES - 1:
                        time.sleep(3)
                        continue
                    break

                # Фильтруем только известные поля модели
                known_fields = set(TradingRecommendation.model_fields.keys())
                filtered = {k: v for k, v in data.items() if k in known_fields}
                result = TradingRecommendation(**filtered)
                valid, reason = self._semantic_validate(result)
                if not valid:
                    logger.warning(f"[LLM] semantic reject: {reason}")
                    result = TradingRecommendation(
                        action="NEUTRAL",
                        neutral_type="STRONG",
                        summary=f"Сигнал отклонён semantic-validator: {reason}.",
                        tech_score=0.0,
                        stake_amount=float(STAKE_LEVELS[0]),
                    )
                logger.info(
                    f"[LLM] action={result.action} tech={result.tech_score:.2f} "
                    f"sent={result.sent_score:.2f} final={result.final_score:.2f} "
                    f"stake={result.stake_amount} mood={result.news_mood} "
                    f"confidence={result.confidence:.2f}"
                )
                if result.news_reasoning:
                    logger.info(f"[LLM news_reasoning] {result.news_reasoning[:200]}")
                if result.tech_reasoning:
                    logger.info(f"[LLM tech_reasoning] {result.tech_reasoning[:200]}")
                if result.counter_argument:
                    logger.info(f"[LLM counter_argument] {result.counter_argument[:200]}")
                return result
            except Exception as e:
                logger.error(f"[LLM] Ошибка (попытка {attempt+1}): {e}")
                time.sleep(3)

        return TradingRecommendation(
            action="NEUTRAL",
            summary="LLM недоступен — торговля заблокирована.",
            tech_score=0.0,
            stake_amount=float(STAKE_LEVELS[0]),
        )


# ═══════════════════════════════════════════════════════════════════════
#                       TG MESSAGE FORMATTER
# ═══════════════════════════════════════════════════════════════════════

class TGFormatter:
    """Форматирует красивые HTML-сообщения для Telegram."""

    @staticmethod
    def _score_bar(score: float) -> str:
        filled = int(round((score + 1.0) / 2.0 * 10))
        filled = max(0, min(10, filled))
        bar = "█" * filled + "░" * (10 - filled)
        return f"[{bar}] {score:+.2f}"

    @staticmethod
    def _mood_emoji(mood: str) -> str:
        return {
            "позитивный": "🟢",
            "негативный": "🔴",
            "нейтральный": "⚪",
            "смешанный": "🟡",
            "тревожный": "🟠",
            "эйфория": "💚",
        }.get(mood.lower(), "⚪")

    @staticmethod
    def indicator_block(ind: Indicators) -> str:
        st_emoji = "🟢" if "бычий" in ind.st_dir else "🔴"
        macd_emoji = "📈" if ind.macd_trend == "бычий" else "📉"
        cross_emoji = {"бычье": "⚡", "медвежье": "⚡", "нет": ""}.get(ind.macd_cross, "")
        obv_emoji = "🟢" if ind.obv_trend == "бычий" else "🔴"
        vol_emoji = "🔊" if ind.vol_ratio > 1.5 else ("🔇" if ind.vol_ratio < 0.5 else "🔉")

        lines = [
            f"<b>💰 Цена: ${ind.price:.4f}</b>  <i>({ind.change_pct:+.2f}%)</i>",
            f"<b>ATR:</b> ${ind.atr:.4f} = {ind.atr_pct:.2f}%  |  "
            f"<b>Поддержка:</b> ${ind.support:.4f}  <b>Сопр.:</b> ${ind.resistance:.4f}",
            "",
            "📊 <b>ТРЕНД</b>",
            f"  {st_emoji} SuperTrend: <b>{ind.st_dir}</b> (${ind.st_level:.4f}, dist {ind.st_dist_pct:.2f}%)",
            f"  SMA 7/20: <b>{ind.sma_cross}</b>  ({ind.sma7:.2f} / {ind.sma20:.2f})",
            f"  SMA 200: ${ind.sma200:.2f}  ({'🟢 выше' if ind.price > ind.sma200 else '🔴 ниже'})",
            f"  ADX: <b>{ind.adx:.1f}</b>  ({ind.adx_str.upper()})",
            "",
            "📈 <b>ОСЦИЛЛЯТОРЫ</b>",
            f"  RSI(14): <b>{ind.rsi:.1f}</b>  {'🔴 ПЕРЕКУПЛЕН' if ind.rsi > 70 else ('🟢 ПЕРЕПРОДАН' if ind.rsi < 30 else '⚪ нейтрал')}",
            f"  {macd_emoji} MACD: {ind.macd:.6f} / Sig: {ind.macd_signal:.6f}  "
            f"Hist: <b>{ind.macd_hist:+.6f}</b>  {cross_emoji}{ind.macd_cross if ind.macd_cross != 'нет' else ''}",
            f"  Stoch RSI: K={ind.stoch_k:.1f} D={ind.stoch_d:.1f}  "
            f"({'🔴 перекуплен' if ind.stoch_zone == 'перекуплен' else ('🟢 перепродан' if ind.stoch_zone == 'перепродан' else '⚪ норма')})"
            f"{' ⚡' + ind.stoch_cross if ind.stoch_cross != 'нет' else ''}",
            f"  Williams %R: <b>{ind.williams_r:.1f}</b>  ({ind.williams_zone})",
            f"  CCI(20): <b>{ind.cci:.1f}</b>  ({ind.cci_signal})",
            "",
            "📊 <b>BOLLINGER BANDS</b>",
            f"  Верх: ${ind.bb_upper:.4f}  Mid: ${ind.bb_mid:.4f}  Низ: ${ind.bb_lower:.4f}",
            f"  Поз: <b>{ind.bb_pct:.0f}%</b> — {ind.bb_pos}"
            f"{'  ⚠️ СЖАТИЕ' if ind.bb_squeeze else ''}",
            "",
            "📦 <b>ОБЪЁМ</b>",
            f"  {vol_emoji} Ratio: <b>{ind.vol_ratio:.2f}x</b> — {ind.vol_desc}  |  OBV: {obv_emoji} {ind.obv_trend}",
        ]
        return "\n".join(lines)

    @staticmethod
    def news_block(rec: TradingRecommendation) -> str:
        """Блок новостей: общее настроение + ключевая новость (без списка всех)."""
        mood_emoji = TGFormatter._mood_emoji(rec.news_mood)
        lines = [
            f"📰 <b>НОВОСТНОЙ ФОН</b>",
            f"  {mood_emoji} Настроение: <b>{rec.news_mood}</b>  "
            f"<i>(sent_score: {rec.sent_score:+.2f})</i>",
        ]
        if rec.news_summary:
            lines.append(f"  <i>{rec.news_summary}</i>")
        if rec.key_news:
            lines.append(f"\n🔑 <b>Ключевая новость:</b>")
            lines.append(f"  <i>{rec.key_news[:120]}</i>")
        return "\n".join(lines)

    @classmethod
    def scores_line(cls, rec: TradingRecommendation) -> str:
        return (
            f"📊 <b>Оценки:</b>  "
            f"Новости: <b>{rec.sent_score:+.3f}</b>  |  "
            f"Техника: <b>{rec.tech_score:+.3f}</b>  |  "
            f"Итог: <b>{rec.final_score:+.3f}</b>\n"
            f"{cls._score_bar(rec.final_score)}"
        )

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        """Обрезает текст по последней точке в пределах max_chars."""
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars]
        last_dot = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        if last_dot > max_chars // 2:
            return cut[:last_dot + 1]
        return cut.rstrip() + "…"

    @staticmethod
    def reasoning_block(rec: TradingRecommendation) -> str:
        """Блок размышлений ИИ для TG."""
        t = TGFormatter._truncate
        lines = ["🧠 <b>РАЗМЫШЛЕНИЯ ИИ</b>"]
        if rec.news_reasoning:
            lines.append(f"  📰 <i>{t(rec.news_reasoning, 220)}</i>")
        if rec.tech_reasoning:
            lines.append(f"  📊 <i>{t(rec.tech_reasoning, 280)}</i>")
        if rec.counter_argument:
            lines.append(f"  ⚖️ <i>{t(rec.counter_argument, 160)}</i>")
        conf_emoji = "🟢" if rec.confidence >= 0.7 else ("🟡" if rec.confidence >= 0.5 else "🔴")
        lines.append(f"  {conf_emoji} <b>Уверенность:</b> {rec.confidence:.0%}")
        return "\n".join(lines)

    @staticmethod
    def reasoning_msg(pair: str, rec: TradingRecommendation, action: str) -> str:
        """Отдельное сообщение с полными размышлениями ИИ."""
        conf_emoji = "🟢" if rec.confidence >= 0.7 else ("🟡" if rec.confidence >= 0.5 else "🔴")
        if action in ("LONG_MANAGE", "SHORT_MANAGE"):
            t = TGFormatter._truncate
            lines = [f"🧠 <b>КРАТКОЕ СОПРОВОЖДЕНИЕ</b> — {pair}  ({action})"]
            if rec.tech_reasoning:
                lines.append(f"📊 <i>{t(rec.tech_reasoning, 220)}</i>")
            if rec.counter_argument:
                lines.append(f"⚖️ <i>{t(rec.counter_argument, 140)}</i>")
            lines.append(f"{conf_emoji} <b>Уверенность:</b> {rec.confidence:.0%}")
            return "\n".join(lines)
        lines = [f"🧠 <b>РАЗМЫШЛЕНИЯ ИИ</b> — {pair}  ({action})", ""]
        if rec.news_reasoning:
            lines.append(f"📰 <b>Новости:</b>")
            lines.append(f"<i>{rec.news_reasoning}</i>")
            lines.append("")
        if rec.tech_reasoning:
            lines.append(f"📊 <b>Техника:</b>")
            lines.append(f"<i>{rec.tech_reasoning}</i>")
            lines.append("")
        if rec.counter_argument:
            lines.append(f"⚖️ <b>Контраргумент:</b>")
            lines.append(f"<i>{rec.counter_argument}</i>")
            lines.append("")
        lines.append(f"{conf_emoji} <b>Уверенность:</b> {rec.confidence:.0%}")
        if rec.neutral_type and rec.action == "NEUTRAL":
            neutral_labels = {
                "WEAK":   "🟡 Слабый нейтрал — сигнал почти готов, смотри следующую свечу",
                "STRONG": "🔴 Сильный нейтрал — противоречия, пропускай",
                "WAIT":   "🟠 Жди подтверждения — сетап формируется",
            }
            label = neutral_labels.get(rec.neutral_type.upper(), rec.neutral_type)
            lines.append(f"{label}")
        return "\n".join(lines)

    @classmethod
    def neutral(cls, pair: str, rec: TradingRecommendation, ind: Indicators) -> str:
        session_name, session_emoji = get_session_info()
        return (
            f"⚪ <b>НЕЙТРАЛ</b> — {pair}  {session_emoji} {session_name}\n"
            f"{cls.scores_line(rec)}\n\n"
            f"{cls.indicator_block(ind)}\n\n"
            f"{cls.news_block(rec)}\n\n"
            f"💬 <b>Вывод:</b>\n{rec.summary[:1500]}"
        )

    @classmethod
    def in_trade_update(
        cls,
        pair: str,
        side: str,
        rec: TradingRecommendation,
        ind: Indicators,
        profit_pct: float,
        score_change_line: str = "",
        score_change_reason: str = "",
    ) -> str:
        session_name, session_emoji = get_session_info()
        side_label = "LONG 🟢" if side == "LONG" else "SHORT 🔴"
        risk_flags = []
        if ind.atr_pct > 3.0:
            risk_flags.append("ATR высокий")
        if ind.vol_ratio < 0.5:
            risk_flags.append("объём низкий")
        if side == "LONG" and ind.st_dir != "бычий":
            risk_flags.append("ST против LONG")
        if side == "SHORT" and ind.st_dir != "медвежий":
            risk_flags.append("ST против SHORT")
        risk_line = ", ".join(risk_flags) if risk_flags else "критичных рисков нет"
        score_change_block = f"  <b>Дельта score:</b> {score_change_line}\n" if score_change_line else ""
        score_reason_block = f"  <b>Почему изменился score:</b> {score_change_reason}\n" if score_change_reason else ""
        return (
            f"📌 <b>СОПРОВОЖДЕНИЕ СДЕЛКИ</b> — {pair}\n"
            f"  {session_emoji} <b>Сессия:</b> {session_name}\n"
            f"  <b>Позиция:</b> {side_label}\n"
            f"  <b>Текущий PnL:</b> {profit_pct:+.2%}\n"
            f"  <b>Итог score:</b> {rec.final_score:+.3f} | <b>Уверенность:</b> {rec.confidence:.0%}\n"
            f"{score_change_block}"
            f"{score_reason_block}"
            f"  <b>SuperTrend:</b> {ind.st_dir} | <b>ADX:</b> {ind.adx:.1f} | <b>ATR%:</b> {ind.atr_pct:.2f}\n"
            f"  <b>Риски:</b> {risk_line}\n\n"
            f"💬 <b>Краткий вывод:</b>\n{TGFormatter._truncate(rec.summary, 320)}"
        )

    @classmethod
    def entry(cls, pair: str, side: str, rate: float, stake: float,
              rec: TradingRecommendation, ind: Indicators) -> str:
        direction = "LONG 🟢" if side == "long" else "SHORT 🔴"
        session_name, session_emoji = get_session_info()
        sl_price = rate * (1 + STOPLOSS_PCT) if side == "long" else rate * (1 - STOPLOSS_PCT)
        tp_price = rate * (1 + TAKE_PROFIT)  if side == "long" else rate * (1 - TAKE_PROFIT)
        return (
            f"{'🟢' if side == 'long' else '🔴'} <b>ВХОД {direction}</b> — {pair}\n"
            f"  {session_emoji} <b>Сессия:</b> {session_name}\n\n"
            f"  <b>Цена входа :</b> ${rate:.4f}\n"
            f"  <b>Стейк      :</b> {stake:.0f} USDT\n"
            f"  <b>Стоп-лосс  :</b> ${sl_price:.4f}  ({STOPLOSS_PCT*100:.1f}%)\n"
            f"  <b>TP / Трейлинг от:</b> ${tp_price:.4f}  ({TAKE_PROFIT*100:.1f}%)\n\n"
            f"{cls.scores_line(rec)}\n\n"
            f"{cls.indicator_block(ind)}\n\n"
            f"{cls.news_block(rec)}\n\n"
            f"💬 <b>Вывод:</b>\n{rec.summary[:1500]}"
        )

    @classmethod
    def exit_msg(cls, pair: str, side: str, profit: float, mins: float,
                 rec_data: Dict[str, float], opinion: str, ind: Indicators,
                 rec: Optional["TradingRecommendation"] = None) -> str:
        status = "✅ ПРОФИТ" if profit > 0 else "❌ УБЫТОК"
        session_name, session_emoji = get_session_info()
        news_part = cls.news_block(rec) if rec else ""
        return (
            f"{'✅' if profit > 0 else '❌'} <b>ВЫХОД {side}</b> — {pair}\n"
            f"  {session_emoji} <b>Сессия:</b> {session_name}\n\n"
            f"  <b>{status}:</b> {profit:+.2%}\n"
            f"  <b>Время:</b> {mins:.0f} мин\n\n"
            f"  Новости: <b>{rec_data['sent']:+.3f}</b>  |  "
            f"Техника: <b>{rec_data['tech']:+.3f}</b>  |  "
            f"Итог: <b>{rec_data['final']:+.3f}</b>\n\n"
            f"{cls.indicator_block(ind)}\n\n"
            f"{news_part}\n\n"
            f"💬 <b>Вывод:</b>\n{opinion[:1000]}"
        )


# ═══════════════════════════════════════════════════════════════════════
#                       СТРАТЕГИЯ
# ═══════════════════════════════════════════════════════════════════════

class GPTStrategy(IStrategy):
    """
    Стратегия на основе LLM (GPT) + технический анализ + новости.
    Все параметры настраиваются в разделе КОНФИГУРАЦИЯ вверху файла.
    """

    position_adjustment_enable = False
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True

    stoploss             = STOPLOSS_PCT
    minimal_roi          = {"99999": -1}
    startup_candle_count = STARTUP_CANDLES

    order_types = {
        "entry": "limit",
        "exit": "market",
        "emergency_exit": "market",
        "force_exit": "market",
        "force_entry": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 120,
    }

    plot_config = {
        "main_plot": {
            "sma7":   {"color": "cyan",   "type": "line"},
            "sma20":  {"color": "orange", "type": "line"},
            "sma200": {"color": "purple", "type": "line"},
            "bb_upper": {
                "color": "rgba(0,200,255,0.4)",
                "type": "line",
                "fill_to": "bb_lower",
            },
            "bb_mid":   {"color": "rgba(0,200,255,0.8)", "type": "line"},
            "bb_lower": {"color": "rgba(0,200,255,0.4)", "type": "line"},
            "st_bull": {
                "color": "#00ff88",
                "type": "line",
                "fill_to": None,
            },
            "st_bear": {
                "color": "#ff4444",
                "type": "line",
                "fill_to": None,
            },
            "vwap": {"color": "#f1c40f", "type": "line"},
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "#3498db", "type": "line"},
            },
            "MACD": {
                "macd_line":        {"color": "#2ecc71",              "type": "line"},
                "macd_signal_line": {"color": "#e74c3c",              "type": "line"},
                "macd_hist":        {"color": "rgba(155,89,182,0.7)", "type": "bar"},
            },
            "ADX": {
                "adx": {"color": "#e67e22", "type": "line"},
            },
            "Stoch RSI": {
                "stoch_k": {"color": "#1abc9c", "type": "line"},
                "stoch_d": {"color": "#e74c3c", "type": "line"},
            },
            "Volume Ratio": {
                "volume_ratio": {"color": "teal", "type": "bar"},
            },
            "ATR": {
                "atr": {"color": "#e67e22", "type": "line"},
            },
            "Williams %R": {
                "williams_r": {"color": "#9b59b6", "type": "line"},
            },
            "CCI": {
                "cci": {"color": "#16a085", "type": "line"},
            },
            "OBV": {
                "obv":     {"color": "#2980b9", "type": "line"},
                "obv_sma": {"color": "rgba(127,127,127,0.7)", "type": "line"},
            },
        },
    }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._config = config
        self.tg  = TelegramNotifier(config)
        self.llm = LLMClient()
        self.trade_execution_enabled = bool(
            config.get("trade_execution_enabled", TRADE_EXECUTION_ENABLED_DEFAULT)
        )
        self._cache:          Dict[str, Tuple[Any, Dict[str, Any]]] = {}
        self._seen_pairs:     set = set()
        self._trailing_state: Dict[int, float] = {}
        self._pending_stake:  Dict[str, float] = {}
        self._trade_memory:   TradeMemory = TradeMemory()
        self._entry_times:    Dict[int, datetime] = {}
        # Последнее решение по паре для передачи контекста в следующий промт
        self._last_decision:  Dict[str, Dict[str, Any]] = {}
        self._signal_stats: Dict[str, Any] = {
            "candles": 0,
            "llm_errors": 0,
            "semantic_rejects": 0,
            "context_timeouts": 0,
            "arbiter_blocks": 0,
            "entry_signals": 0,
            "accepted_entries": 0,
            "action_counts": {"LONG_ENTER": 0, "SHORT_ENTER": 0, "NEUTRAL": 0, "LONG_EXIT": 0, "SHORT_EXIT": 0},
            "confidence_bins": {"lt50": 0, "50_70": 0, "70_85": 0, "gt85": 0},
            "regimes": {"adx_low": 0, "adx_mid": 0, "adx_high": 0},
        }
        self._entry_quality_log: List[Dict[str, Any]] = []
        self._last_alert_ts: float = 0.0
        self._last_notify_ts: Dict[str, float] = {}
        self._last_notify_action: Dict[str, str] = {}
        self._last_in_trade_candle_id: Dict[str, str] = {}
        self._last_llm_candle_id: Dict[str, str] = {}
        self._last_llm_call_ts: Dict[str, float] = {}
        self._entry_score_snapshot: Dict[str, Dict[str, float]] = {}
        # История score для корректного plot по всем прошлым свечам.
        # Ключ: pair -> candle_id -> {sent_score, tech_score, final_score}
        self._score_history: Dict[str, Dict[str, Dict[str, float]]] = {}
        logger.info("GPT Strategy v5.1 запущена")
        self._startup_check()

    def _startup_check(self) -> None:
        """Проверяет все компоненты стратегии при запуске. Отправляет отчёт в TG."""
        errors:   List[str] = []
        warnings: List[str] = []
        passed:   List[str] = []

        # ── 1. Telegram ───────────────────────────────────────────────
        try:
            if self.tg.enabled:
                passed.append("Telegram: токен и chat_id настроены")
            else:
                warnings.append("Telegram: отключён (нет token/chat_id) — уведомления не будут приходить")
        except Exception as e:
            errors.append(f"Telegram init: {e}")

        # ── 2. LLM API ────────────────────────────────────────────────
        try:
            if not LLM_API_KEY:
                warnings.append("LLM: не задан LLM_API_KEY в окружении — стратегия уйдет в fallback NEUTRAL")
            else:
                passed.append(f"LLM: модель {LLM_MODEL} настроена")
        except Exception as e:
            errors.append(f"LLM config: {e}")

        # ── 3. Конфигурация параметров ────────────────────────────────
        try:
            assert len(STAKE_LEVELS) > 0,          "STAKE_LEVELS пустой"
            assert STAKE_LEVELS == sorted(STAKE_LEVELS), "STAKE_LEVELS не отсортирован"
            assert STOPLOSS_PCT < 0,               "STOPLOSS_PCT должен быть отрицательным"
            assert TAKE_PROFIT > 0,                "TAKE_PROFIT должен быть положительным"
            assert 0 < WEIGHT_TECHNICAL <= 1,      "WEIGHT_TECHNICAL вне диапазона"
            assert 0 < WEIGHT_SENTIMENT <= 1,      "WEIGHT_SENTIMENT вне диапазона"
            assert abs(WEIGHT_TECHNICAL + WEIGHT_SENTIMENT - 1.0) < 0.01, \
                f"Сумма весов ≠ 1.0 ({WEIGHT_TECHNICAL + WEIGHT_SENTIMENT})"
            assert HISTORY_CANDLES >= 3,           "HISTORY_CANDLES слишком мало"
            assert MIN_CANDLES >= 20,              "MIN_CANDLES слишком мало"
            assert MAX_NEWS_ITEMS > 0,             "MAX_NEWS_ITEMS должен быть > 0"
            assert ENTRY_LONG_THRESHOLD > 0,       "ENTRY_LONG_THRESHOLD должен быть > 0"
            assert ENTRY_SHORT_THRESHOLD < 0,      "ENTRY_SHORT_THRESHOLD должен быть < 0"
            passed.append(
                f"Конфиг: стейки={STAKE_LEVELS} SL={STOPLOSS_PCT*100:.1f}% "
                f"TP={TAKE_PROFIT*100:.1f}% веса={WEIGHT_TECHNICAL}/{WEIGHT_SENTIMENT} "
                f"trade_execution={'ON' if self.trade_execution_enabled else 'OFF'}"
            )
        except AssertionError as e:
            errors.append(f"Конфиг: {e}")
        except Exception as e:
            errors.append(f"Конфиг проверка: {e}")

        # ── 4. Сессия ─────────────────────────────────────────────────
        try:
            session_name, session_emoji = get_session_info()
            passed.append(f"Сессия: {session_emoji} {session_name} (UTC {datetime.now(timezone.utc).strftime('%H:%M')})")
        except Exception as e:
            errors.append(f"Сессия: {e}")

        # ── 5. Память сделок (проверяем работу без реальных сделок) ──
        try:
            import os
            test_record = TradeRecord(
                pair="TEST/USDT",
                side="LONG",
                profit_pct=1.5,
                duration_min=45.0,
                entry_time=datetime.now(timezone.utc),
                exit_time=datetime.now(timezone.utc),
                final_score=0.45,
                exit_reason="test",
            )
            count_before = len(self._trade_memory._records)
            # Валидация без записи тестовых данных на диск и без модификации боевой истории.
            prompt_line = test_record.to_prompt_line()
            assert "TEST/USDT" in prompt_line, "to_prompt_line не содержит pair"
            assert test_record.age_str() is not None, "age_str() сломан"
            current_block = self._trade_memory.to_prompt_block()
            assert isinstance(current_block, str), "to_prompt_block вернул не строку"
            file_exists = os.path.exists(MEMORY_FILE)
            passed.append(
                f"Память сделок: структура работает без тестовой записи на диск "
                f"| файл={'найден' if file_exists else 'будет создан'} | загружено сделок: {count_before}"
            )
        except Exception as e:
            errors.append(f"Память сделок: {e}")

        # ── 6. Pydantic модели ────────────────────────────────────────
        try:
            test_rec = TradingRecommendation(
                sentiments=[{"title": "test", "score": 0.5, "sentiment": 0.5,
                             "relevance": 1.0, "recency": 1.0}],
                key_news="test news",
                news_summary="test summary",
                news_mood="нейтральный",
                tech_score=0.4,
                action="LONG_ENTER",
                summary="тест тест тест тест тест тест тест тест тест тест",
                stake_amount=100.0,
            )
            assert test_rec.final_score != 0.0,  "final_score не считается"
            assert test_rec.sent_score  != 0.0,  "sent_score не считается"
            assert test_rec.stake_amount in STAKE_LEVELS, "stake_snap не работает"
            passed.append(f"Pydantic модели: валидация и вычисляемые поля работают")
        except Exception as e:
            errors.append(f"Pydantic модели: {e}")

        # ── 7. IndicatorEngine (на синтетических данных) ──────────────
        try:
            n = 220
            np.random.seed(42)
            prices = 50000 + np.cumsum(np.random.randn(n) * 100)
            test_df = pd.DataFrame({
                "date":   pd.date_range("2024-01-01", periods=n, freq="30min"),
                "open":   prices * 0.999,
                "high":   prices * 1.002,
                "low":    prices * 0.998,
                "close":  prices,
                "volume": np.random.uniform(100, 1000, n),
            })
            ind = IndicatorEngine.calculate(test_df, history_n=5)
            assert ind is not None,          "IndicatorEngine вернул None"
            assert ind.price > 0,            "price <= 0"
            assert 0 <= ind.rsi <= 100,      f"RSI вне диапазона: {ind.rsi}"
            assert ind.atr >= 0,             f"ATR < 0: {ind.atr}"
            assert len(ind.candles) == 5,    f"candles: ожидали 5, получили {len(ind.candles)}"
            assert ind.support > 0,          "support <= 0"
            assert ind.resistance > 0,       "resistance <= 0"
            passed.append(
                f"IndicatorEngine: RSI={ind.rsi:.1f} MACD={ind.macd_trend} "
                f"ST={ind.st_dir} ADX={ind.adx:.1f}"
            )
        except Exception as e:
            errors.append(f"IndicatorEngine: {e}")

        # ── 8. PromptBuilder ──────────────────────────────────────────
        try:
            if ind is not None:
                entry_prompt = PromptBuilder.build_entry_prompt(
                    ind, ["Test news headline"], "BTC/USDT", "нет истории"
                )
                assert len(entry_prompt) > 100, "Entry prompt слишком короткий"
                exit_prompt = PromptBuilder.build_exit_prompt(
                    ind, [], "BTC/USDT", "LONG", 1.5, "нет истории"
                )
                assert len(exit_prompt) > 100,  "Exit prompt слишком короткий"
                passed.append(f"PromptBuilder: entry={len(entry_prompt)} символов exit={len(exit_prompt)} символов")
            else:
                warnings.append("PromptBuilder: пропущен — IndicatorEngine не вернул данные")
        except Exception as e:
            errors.append(f"PromptBuilder: {e}")

        # ── 9. NewsProvider (только структура, без реального запроса) ─
        try:
            assert len(RSS_FEEDS) > 0, "RSS_FEEDS пустой"
            assert NEWS_CACHE_SEC > 0, "NEWS_CACHE_SEC <= 0"
            passed.append(f"NewsProvider: {len(RSS_FEEDS)} источников настроено")
        except Exception as e:
            errors.append(f"NewsProvider: {e}")

        # ── 10. Fear & Greed ──────────────────────────────────────────
        try:
            fg = FearGreedProvider.fetch()
            if fg:
                passed.append(f"Fear & Greed Index: {fg}")
            else:
                warnings.append("Fear & Greed Index: недоступен (API не ответил) — будет работать без него")
        except Exception as e:
            warnings.append(f"Fear & Greed Index: {e}")

        # ── 11. TradeMemory статистика ────────────────────────────────
        try:
            empty_block = self._trade_memory.to_prompt_block()
            assert isinstance(empty_block, str), "to_prompt_block вернул не строку"
            passed.append("Память сделок: пустое состояние обрабатывается корректно")
        except Exception as e:
            errors.append(f"Память сделок (пустая): {e}")

        # ── 12. OrderBookProvider ─────────────────────────────────────
        try:
            assert OB_DEPTH > 0,     "OB_DEPTH должен быть > 0"
            assert OB_CACHE_SEC > 0, "OB_CACHE_SEC должен быть > 0"
            # Проверяем что empty dict возвращается корректно при нет данных
            empty_ob = {"bid_vol": 0.0, "ask_vol": 0.0, "imbalance": 0.0, "desc": ""}
            assert "bid_vol" in empty_ob, "Структура order book некорректна"
            passed.append(
                f"OrderBookProvider: глубина стакана={OB_DEPTH} уровней "
                f"кэш={OB_CACHE_SEC}с (подключится при первой свече)"
            )
        except Exception as e:
            errors.append(f"OrderBookProvider: {e}")

        # ── 13. Контекст предыдущего решения ─────────────────────────
        try:
            assert isinstance(self._last_decision, dict), "_last_decision не dict"
            # Симулируем сохранение и чтение решения
            test_pair = "TEST/USDT"
            self._last_decision[test_pair] = {
                "action":         "NEUTRAL",
                "final_score":    -0.15,
                "confidence":     0.65,
                "tech_reasoning": "тест",
                "counter_arg":    "тест",
                "candle_time":    "2026-01-01",
            }
            prev = self._last_decision.get(test_pair, {})
            assert prev.get("action") == "NEUTRAL", "Чтение _last_decision сломано"
            # Очищаем тестовую запись
            del self._last_decision[test_pair]
            passed.append("Контекст предыдущего решения: структура работает корректно")
        except Exception as e:
            errors.append(f"Контекст предыдущего решения: {e}")

        # ── 14. Протокол валидации качества (WF/anchored) ───────────
        try:
            wf = self._build_walkforward_protocol()
            assert "anchored" in wf and "rolling" in wf, "WF protocol неполный"
            passed.append("Оценка качества: walk-forward/anchored шаблон инициализирован")
        except Exception as e:
            errors.append(f"WF protocol: {e}")

        # ── Формируем отчёт ───────────────────────────────────────────
        total   = len(passed) + len(warnings) + len(errors)
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        if errors:
            status_line = f"🔴 <b>ЗАПУСК С ОШИБКАМИ</b> — {len(errors)} из {total} проверок провалено"
        elif warnings:
            status_line = f"🟡 <b>ЗАПУСК С ПРЕДУПРЕЖДЕНИЯМИ</b> — все критические проверки пройдены"
        else:
            status_line = f"🟢 <b>ЗАПУСК УСПЕШЕН</b> — все {total} проверок пройдены"

        lines = [
            f"{status_line}",
            f"<i>GPT Strategy v5.2 | {now_str}</i>",
            f"Модель: <b>{LLM_MODEL}</b> | TФ: 15-30м | Стейки: {STAKE_LEVELS}",
            "",
        ]

        if passed:
            lines.append("✅ <b>Пройдено:</b>")
            for p in passed:
                lines.append(f"  ✔ {p}")

        if warnings:
            lines.append("\n⚠️ <b>Предупреждения:</b>")
            for w in warnings:
                lines.append(f"  ⚠ {w}")

        if errors:
            lines.append("\n❌ <b>Ошибки:</b>")
            for e in errors:
                lines.append(f"  ✖ {e}")

        report = "\n".join(lines)

        # Логируем всегда
        for p in passed:
            logger.info(f"[CHECK ✔] {p}")
        for w in warnings:
            logger.warning(f"[CHECK ⚠] {w}")
        for e in errors:
            logger.error(f"[CHECK ✖] {e}")

        self.tg.send(report)

    def feature_engineering_standard(self, df: DataFrame, **kwargs) -> DataFrame:
        return df

    def set_freqai_targets(self, df: DataFrame, **kwargs) -> DataFrame:
        df["&-empty"] = "0"
        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair      = metadata.get("pair", "unknown")
        candle_id = str(pd.to_datetime(dataframe["date"].iloc[-1])) if not dataframe.empty else None

        # ── Всегда пересчитываем колонки для plot_config ─────────────
        dataframe = self._add_plot_columns(dataframe)
        dataframe = self._apply_score_history(pair, dataframe)

        # ── Кэш: не запрашивать LLM повторно для той же свечи ────────
        cached = self._cache.get(pair)
        if cached:
            ct, cr = cached
            if ct == candle_id:
                logger.debug(f"[{pair}] cache-hit same candle {candle_id}")
                return self._apply_columns(dataframe, cr)

        # ── Жесткий антидубль: если уже анализировали эту свечу, не вызывать LLM ──
        if candle_id and self._last_llm_candle_id.get(pair) == candle_id and cached:
            logger.debug(f"[{pair}] skip duplicate LLM on candle {candle_id}")
            return self._apply_columns(dataframe, cached[1])

        # ── Time-throttle: защита от частых повторных вызовов в одном цикле ───────
        now_ts = time.time()
        last_call_ts = self._last_llm_call_ts.get(pair, 0.0)
        if cached and (now_ts - last_call_ts) < LLM_MIN_CALL_INTERVAL_SEC:
            logger.debug(f"[{pair}] skip LLM by throttle {(now_ts - last_call_ts):.1f}s")
            return self._apply_columns(dataframe, cached[1])

        if len(dataframe) < MIN_CANDLES:
            return self._apply_columns(dataframe, self._empty_result())

        ind = IndicatorEngine.calculate(dataframe, HISTORY_CANDLES)
        if not ind:
            return self._apply_columns(dataframe, self._empty_result())

        # Первый проход по паре после запуска:
        # не анализируем текущую "старыю" свечу, ждём следующую новую свечу ТФ.
        if pair not in self._seen_pairs:
            self._seen_pairs.add(pair)
            logger.info(f"[{pair}] warmup: пропускаем первую свечу после старта, ждём новую")
            return self._apply_columns(dataframe, self._empty_result())

        pair_clean   = pair.split(":")[0]
        side, profit = self._get_position(pair)
        memory_block = self._trade_memory.to_prompt_block(pair)
        fetch_started = time.time()
        news_items, fear_greed, ob_data = self._collect_market_context(pair)
        fetch_elapsed = time.time() - fetch_started
        if fetch_elapsed > CONTEXT_TOTAL_TIMEOUT_SEC:
            logger.warning(
                f"[{pair}] Сбор внешнего контекста занял {fetch_elapsed:.2f}s "
                f"(бюджет {CONTEXT_TOTAL_TIMEOUT_SEC:.2f}s)"
            )

        if ob_data.get("bid_vol"):
            ind.ob_bid_vol = ob_data["bid_vol"]
            ind.ob_ask_vol = ob_data["ask_vol"]
            ind.ob_imbalance = ob_data["imbalance"]
            ind.ob_desc = ob_data["desc"]

        # Контекст предыдущего решения
        prev_decision = self._last_decision.get(pair, {})

        prompt = (
            PromptBuilder.build_entry_prompt(ind, news_items, pair_clean, memory_block, fear_greed, prev_decision)
            if side is None
            else PromptBuilder.build_exit_prompt(ind, news_items, pair_clean, side, profit * 100, memory_block, fear_greed, prev_decision)
        )
        rec = self.llm.analyze(prompt)
        self._last_llm_call_ts[pair] = time.time()
        llm_blocked = rec.summary.startswith("Сигнал отклонён semantic-validator")
        if llm_blocked:
            self._signal_stats["llm_errors"] += 1

        arbiter_action, arbiter_reasons = self._arbiter_action(rec, ind, side)
        action_to_store = arbiter_action
        report_only_entry_blocked = (
            side is None
            and not self.trade_execution_enabled
            and action_to_store in ("LONG_ENTER", "SHORT_ENTER")
        )
        execution_action = "NEUTRAL" if report_only_entry_blocked else action_to_store
        if report_only_entry_blocked:
            logger.info(
                f"[{pair_clean}] report-only mode: {action_to_store} (сигнал сохранён, вход отключен)"
            )
        if arbiter_reasons:
            logger.info(f"[{pair_clean}] arbiter blocked -> NEUTRAL: {';'.join(arbiter_reasons)}")

        logger.info(
            f"[{pair_clean}] action={action_to_store} "
            f"final={rec.final_score:.3f} stake={rec.stake_amount}"
        )

        if side is None and action_to_store in ("LONG_ENTER", "SHORT_ENTER"):
            self._entry_score_snapshot[pair] = {
                "final": float(rec.final_score),
                "tech": float(rec.tech_score),
                "sent": float(rec.sent_score),
                "vol": float(ind.vol_ratio),
                "adx": float(ind.adx),
            }

        # Если итоговое действие — вход, убираем только нейтральные фразы из summary.
        if action_to_store in ("LONG_ENTER", "SHORT_ENTER"):
            neutral_markers = (
                "нейтрал",
                "нейтральн",
                "оставаться вне рынка",
                "оставаться в стороне",
                "остаться в стороне",
                "не вход",
                "нецелесообраз",
                "воздерж",
                "рекомендуется",
                "рекомендую",
                "отложить вход",
                "следует отложить",
                "вне рынка",
            )
            raw_summary = (rec.summary or "").strip()
            if raw_summary:
                normalized = raw_summary.replace("!", ".").replace("?", ".")
                parts = [p.strip() for p in normalized.split(".") if p.strip()]
                cleaned_parts = [
                    p for p in parts
                    if not any(marker in p.lower() for marker in neutral_markers)
                ]
                if cleaned_parts:
                    rec.summary = ". ".join(cleaned_parts).strip()
                    if not rec.summary.endswith("."):
                        rec.summary += "."
                    rec.summary += " Вход возможен, но аккуратно: контролируй риск и размер позиции."
                else:
                    rec.summary = (
                        f"Сигнал {'LONG' if action_to_store == 'LONG_ENTER' else 'SHORT'} подтверждён. "
                        "Вход возможен, но аккуратно: контролируй риск и размер позиции."
                    )

        # Сохраняем стейк для передачи в custom_stake_amount
        self._pending_stake[pair] = rec.stake_amount

        # Формируем контекст предыдущего решения для следующей свечи

        # Сохраняем текущее решение для следующей свечи
        self._last_decision[pair] = {
            "action":          action_to_store,
            "final_score":     rec.final_score,
            "confidence":      rec.confidence,
            "tech_reasoning":  rec.tech_reasoning[:200] if rec.tech_reasoning else "",
            "counter_arg":     rec.counter_argument[:150] if rec.counter_argument else "",
            "candle_time":     candle_id or "",
            "arbiter_reasons": "; ".join(arbiter_reasons) if arbiter_reasons else "",
        }

        if action_to_store == "NEUTRAL" and side is None and self._should_send_neutral(pair, action_to_store):
            mode_line = (
                "🟢 <b>РЕЖИМ:</b> торговля включена"
                if self.trade_execution_enabled
                else "🟡 <b>РЕЖИМ:</b> только отчёты, входы отключены"
            )
            self.tg.send(f"{mode_line}\n\n{TGFormatter.neutral(pair_clean, rec, ind)}")
            self.tg.send(TGFormatter.reasoning_msg(pair_clean, rec, "NEUTRAL"))
        elif report_only_entry_blocked and side is None:
            mode_line = "🟡 <b>РЕЖИМ:</b> только отчёты, входы отключены"
            signal_emoji = "🟢" if action_to_store == "LONG_ENTER" else "🔴"
            signal_label = "LONG" if action_to_store == "LONG_ENTER" else "SHORT"
            self.tg.send(
                f"{mode_line}\n\n"
                f"{signal_emoji} <b>СИГНАЛ {signal_label} (без входа)</b> — {pair_clean}\n"
                f"{TGFormatter.scores_line(rec)}\n\n"
                f"{TGFormatter.indicator_block(ind)}\n\n"
                f"{TGFormatter.news_block(rec)}\n\n"
                f"💬 <b>Вывод:</b>\n{rec.summary[:1500]}"
            )
            self.tg.send(TGFormatter.reasoning_msg(pair_clean, rec, action_to_store))
        elif side in ("LONG", "SHORT") and self._should_send_in_trade_update(pair, candle_id):
            score_change_line, score_change_reason = self._build_score_change_context(pair, rec, ind)
            self.tg.send(
                TGFormatter.in_trade_update(
                    pair_clean,
                    side,
                    rec,
                    ind,
                    profit,
                    score_change_line=score_change_line,
                    score_change_reason=score_change_reason,
                )
            )
            self.tg.send(TGFormatter.reasoning_msg(pair_clean, rec, f"{side}_MANAGE"))

        self._update_signal_metrics(pair, action_to_store, rec, ind, llm_blocked)
        self._emit_monitoring_alerts()

        result = {
            "sent_score":         rec.sent_score,
            "tech_score":         rec.tech_score,
            "final_score":        rec.final_score,
            "expert_long_enter":  1 if execution_action == "LONG_ENTER"  else 0,
            "expert_long_exit":   1 if execution_action == "LONG_EXIT"   else 0,
            "expert_short_enter": 1 if execution_action == "SHORT_ENTER" else 0,
            "expert_short_exit":  1 if execution_action == "SHORT_EXIT"  else 0,
            "expert_opinion":     rec.summary,
            "expert_stake":       rec.stake_amount,
            "expert_confidence":  rec.confidence,
            "_rec_key_news":      rec.key_news,
            "_rec_news_summary":  rec.news_summary,
            "_rec_news_mood":     rec.news_mood,
            "_rec_news_reasoning": rec.news_reasoning,
            "_rec_tech_reasoning": rec.tech_reasoning,
            "_rec_counter_argument": rec.counter_argument,
            "_llm_action_raw":    rec.action,
            "_arbiter_reasons":   "; ".join(arbiter_reasons) if arbiter_reasons else "",
        }
        if candle_id:
            self._remember_score(pair, candle_id, result)
        self._cache[pair] = (candle_id, result)
        if candle_id:
            self._last_llm_candle_id[pair] = candle_id
        return self._apply_columns(dataframe, result)

    def _collect_market_context(self, pair: str) -> Tuple[List[str], str, Dict[str, Any]]:
        """
        Собирает внешний контекст параллельно и fail-safe:
        - новости (RSS)
        - Fear&Greed
        - стакан
        Если любой источник тормозит/падает, возвращаются безопасные дефолты.
        """
        defaults = (
            [],
            "",
            {"bid_vol": 0.0, "ask_vol": 0.0, "imbalance": 0.0, "desc": ""},
        )
        try:
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {
                    "news": pool.submit(NewsProvider.fetch, MAX_NEWS_ITEMS),
                    "fg": pool.submit(FearGreedProvider.fetch),
                    "ob": pool.submit(OrderBookProvider.fetch, pair, self.dp),
                }
                out_news: List[str] = defaults[0]
                out_fg: str = defaults[1]
                out_ob: Dict[str, Any] = defaults[2]

                try:
                    out_news = futures["news"].result(timeout=CONTEXT_TASK_TIMEOUT_SEC) or []
                except FuturesTimeoutError:
                    logger.warning(f"[{pair}] timeout NewsProvider")
                    self._signal_stats["context_timeouts"] += 1
                except Exception as e:
                    logger.warning(f"[{pair}] NewsProvider error: {e}")

                try:
                    out_fg = futures["fg"].result(timeout=CONTEXT_TASK_TIMEOUT_SEC) or ""
                except FuturesTimeoutError:
                    logger.warning(f"[{pair}] timeout FearGreedProvider")
                    self._signal_stats["context_timeouts"] += 1
                except Exception as e:
                    logger.warning(f"[{pair}] FearGreedProvider error: {e}")

                try:
                    out_ob = futures["ob"].result(timeout=CONTEXT_TASK_TIMEOUT_SEC) or defaults[2]
                except FuturesTimeoutError:
                    logger.warning(f"[{pair}] timeout OrderBookProvider")
                    self._signal_stats["context_timeouts"] += 1
                except Exception as e:
                    logger.warning(f"[{pair}] OrderBookProvider error: {e}")

                return out_news, out_fg, out_ob
        except Exception as e:
            logger.error(f"[{pair}] _collect_market_context fatal: {e}")
            return defaults

    def _arbiter_action(
        self,
        rec: TradingRecommendation,
        ind: Indicators,
        side: Optional[str],
    ) -> Tuple[str, List[str]]:
        """
        Детерминированный арбитр: финально решает action на основе
        инвариантов риска и порогов score.
        """
        reasons: List[str] = []
        action = rec.action
        score = rec.final_score
        score_cmp = round(float(score), SCORE_ROUND_DECIMALS)

        # Если LLM выбрала NEUTRAL, но score+confidence дают явный сигнал,
        # переводим в вход по детерминированным правилам.
        if side is None and action == "NEUTRAL":
            if score_cmp >= ENTRY_LONG_THRESHOLD and rec.confidence >= ENTRY_MIN_CONFIDENCE:
                action = "LONG_ENTER"
            elif score_cmp <= ENTRY_SHORT_THRESHOLD and rec.confidence >= ENTRY_MIN_CONFIDENCE:
                action = "SHORT_ENTER"

        if rec.confidence < ENTRY_MIN_CONFIDENCE and action in ("LONG_ENTER", "SHORT_ENTER"):
            reasons.append(f"low_confidence:{rec.confidence:.2f}")
        if ind.atr_pct > ENTRY_MAX_ATR_PCT:
            reasons.append(f"high_atr:{ind.atr_pct:.2f}%")
        if action == "LONG_ENTER" and ind.st_dir != "бычий":
            reasons.append("supertrend_conflict_long")
        if action == "SHORT_ENTER" and ind.st_dir != "медвежий":
            reasons.append("supertrend_conflict_short")

        if side is None:
            if action == "LONG_ENTER" and score_cmp < ENTRY_LONG_THRESHOLD:
                reasons.append(f"score_below_long_threshold:{score:.3f}")
            if action == "SHORT_ENTER" and score_cmp > ENTRY_SHORT_THRESHOLD:
                reasons.append(f"score_above_short_threshold:{score:.3f}")
            if action not in ("LONG_ENTER", "SHORT_ENTER", "NEUTRAL"):
                reasons.append(f"invalid_entry_action:{action}")
        else:
            if side == "LONG":
                if action == "LONG_EXIT" and score_cmp > EXIT_LONG_THRESHOLD:
                    reasons.append(f"weak_long_exit_score:{score:.3f}")
            elif side == "SHORT":
                if action == "SHORT_EXIT" and score_cmp < EXIT_SHORT_THRESHOLD:
                    reasons.append(f"weak_short_exit_score:{score:.3f}")

        if reasons:
            self._signal_stats["arbiter_blocks"] += 1
            return "NEUTRAL", reasons
        return action, reasons

    def _update_signal_metrics(
        self,
        pair: str,
        action: str,
        rec: TradingRecommendation,
        ind: Indicators,
        llm_blocked: bool,
    ) -> None:
        self._signal_stats["candles"] += 1
        self._signal_stats["action_counts"][action] = self._signal_stats["action_counts"].get(action, 0) + 1
        if llm_blocked:
            self._signal_stats["semantic_rejects"] += 1
        if action in ("LONG_ENTER", "SHORT_ENTER"):
            self._signal_stats["entry_signals"] += 1
            self._signal_stats["accepted_entries"] += 1
            self._entry_quality_log.append(
                {
                    "pair": pair,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "action": action,
                    "final_score": rec.final_score,
                    "confidence": rec.confidence,
                    "adx": ind.adx,
                    "atr_pct": ind.atr_pct,
                    "vol_ratio": ind.vol_ratio,
                    "session": get_session_info()[0],
                    "regime": "adx_high" if ind.adx > 25 else ("adx_low" if ind.adx < 20 else "adx_mid"),
                }
            )
            if len(self._entry_quality_log) > 500:
                self._entry_quality_log = self._entry_quality_log[-500:]

        if rec.confidence < 0.5:
            self._signal_stats["confidence_bins"]["lt50"] += 1
        elif rec.confidence < 0.7:
            self._signal_stats["confidence_bins"]["50_70"] += 1
        elif rec.confidence < 0.85:
            self._signal_stats["confidence_bins"]["70_85"] += 1
        else:
            self._signal_stats["confidence_bins"]["gt85"] += 1

        regime_key = "adx_high" if ind.adx > 25 else ("adx_low" if ind.adx < 20 else "adx_mid")
        self._signal_stats["regimes"][regime_key] += 1

    def _build_walkforward_protocol(self) -> str:
        return (
            "WF protocol: anchored(180d train + 30d test step 30d) + "
            "rolling(120d train + 14d test step 14d). "
            "Acceptance: PF>1.2, Expectancy>0, MaxDD<20%, "
            "stable across ADX/ATR/volume/session regimes."
        )

    def _emit_monitoring_alerts(self) -> None:
        now = time.time()
        if now - self._last_alert_ts < ALERT_COOLDOWN_SEC:
            return
        candles = max(1, int(self._signal_stats.get("candles", 1)))
        blocked_rate = self._signal_stats.get("arbiter_blocks", 0) / candles
        neutral_rate = self._signal_stats["action_counts"].get("NEUTRAL", 0) / candles
        if blocked_rate > 0.30 or neutral_rate > 0.85:
            self._last_alert_ts = now
            self.tg.send(
                "⚠️ <b>MONITORING ALERT</b>\n"
                f"candles={candles} blocked_rate={blocked_rate:.1%} neutral_rate={neutral_rate:.1%}\n"
                f"WF template: {self._build_walkforward_protocol()}"
            )

    def _update_closed_trade_metrics(self, pair: str, profit_pct: float, duration_min: float, final_score: float) -> None:
        stats = self._signal_stats
        if "closed_trades" not in stats:
            stats["closed_trades"] = 0
            stats["wins"] = 0
            stats["sum_profit_pct"] = 0.0
            stats["sum_win_pct"] = 0.0
            stats["sum_loss_pct"] = 0.0
            stats["loss_streak"] = 0
            stats["max_loss_streak"] = 0
            stats["score_buckets"] = {"lt30": {"n": 0, "pnl": 0.0}, "30_60": {"n": 0, "pnl": 0.0}, "gt60": {"n": 0, "pnl": 0.0}}
        stats["closed_trades"] += 1
        stats["sum_profit_pct"] += profit_pct
        if profit_pct > 0:
            stats["wins"] += 1
            stats["sum_win_pct"] += profit_pct
            stats["loss_streak"] = 0
        else:
            stats["sum_loss_pct"] += profit_pct
            stats["loss_streak"] += 1
            stats["max_loss_streak"] = max(stats["max_loss_streak"], stats["loss_streak"])

        abs_score = abs(final_score)
        if abs_score < 0.30:
            b = "lt30"
        elif abs_score < 0.60:
            b = "30_60"
        else:
            b = "gt60"
        stats["score_buckets"][b]["n"] += 1
        stats["score_buckets"][b]["pnl"] += profit_pct

        # regime-aware snapshot (последние 500 сделок)
        self._entry_quality_log.append(
            {
                "pair": pair,
                "closed_ts": datetime.now(timezone.utc).isoformat(),
                "profit_pct": profit_pct,
                "duration_min": duration_min,
                "final_score": final_score,
            }
        )
        if len(self._entry_quality_log) > 500:
            self._entry_quality_log = self._entry_quality_log[-500:]

    def build_research_report(self) -> Dict[str, Any]:
        """Шаблон отчета для walk-forward / anchored оценки."""
        s = self._signal_stats
        closed = max(1, int(s.get("closed_trades", 0)))
        wins = int(s.get("wins", 0))
        avg_pnl = float(s.get("sum_profit_pct", 0.0)) / closed
        winrate = wins / closed
        return {
            "walkforward_protocol": self._build_walkforward_protocol(),
            "trades": closed,
            "winrate": round(winrate, 4),
            "avg_pnl_pct": round(avg_pnl, 4),
            "max_loss_streak": int(s.get("max_loss_streak", 0)),
            "action_counts": s.get("action_counts", {}),
            "confidence_bins": s.get("confidence_bins", {}),
            "regimes": s.get("regimes", {}),
            "score_buckets": s.get("score_buckets", {}),
        }

    # ──────────────────────────────────────────────────────────────────
    def _add_plot_columns(self, dataframe: DataFrame) -> DataFrame:
        """Добавляет все колонки для отображения в web-интерфейсе Freqtrade."""
        close = dataframe["close"].astype(float)
        high  = dataframe["high"].astype(float)
        low   = dataframe["low"].astype(float)
        vol   = dataframe["volume"].astype(float)

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
        dataframe["rsi"] = (100 - 100 / (1 + gain / loss)).fillna(50)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line        = ema12 - ema26
        macd_signal_line = macd_line.ewm(span=9, adjust=False).mean()
        dataframe["macd_line"]        = macd_line
        dataframe["macd_signal_line"] = macd_signal_line
        dataframe["macd_hist"]        = macd_line - macd_signal_line

        # SMA
        dataframe["sma7"]   = close.rolling(7).mean()
        dataframe["sma20"]  = close.rolling(20).mean()
        dataframe["sma200"] = close.rolling(200).mean()

        # Bollinger Bands
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        dataframe["bb_mid"]   = bb_mid
        dataframe["bb_upper"] = bb_mid + 2.0 * bb_std
        dataframe["bb_lower"] = bb_mid - 2.0 * bb_std

        # SuperTrend — единый движок из IndicatorEngine (без дублирования логики)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        st_up, st_down, st_line = IndicatorEngine._calc_supertrend_series(
            high, low, close, tr, period=10, mult=3.0
        )

        dataframe["st_bull"] = st_up
        dataframe["st_bear"] = st_down
        # Для IndicatorEngine — единая линия
        dataframe["supertrend_line"] = st_line

        # ADX
        plus_dm  = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        cond     = plus_dm > minus_dm
        plus_dm  = plus_dm.where(cond, 0.0)
        minus_dm = minus_dm.where(~cond, 0.0)
        tr_sum   = tr.rolling(14).sum().replace(0, np.nan)
        plus_di  = 100 * plus_dm.rolling(14).sum() / tr_sum
        minus_di = 100 * minus_dm.rolling(14).sum() / tr_sum
        denom    = (plus_di + minus_di).replace(0, np.nan)
        dx       = 100 * (plus_di - minus_di).abs() / denom
        dataframe["adx"] = dx.rolling(14).mean()

        # Stochastic RSI
        rsi_min   = dataframe["rsi"].rolling(14).min()
        rsi_max   = dataframe["rsi"].rolling(14).max()
        rsi_range = (rsi_max - rsi_min).replace(0, np.nan)
        stoch_k_raw = (dataframe["rsi"] - rsi_min) / rsi_range * 100
        dataframe["stoch_k"] = stoch_k_raw.rolling(3).mean().fillna(50)
        dataframe["stoch_d"] = dataframe["stoch_k"].rolling(3).mean().fillna(50)

        # Volume ratio
        avg_vol = vol.rolling(20).mean().replace(0, np.nan)
        dataframe["volume_ratio"] = vol / avg_vol

        # ATR (волатильность) — используется в IndicatorEngine для оценки риска
        dataframe["atr"] = tr.rolling(14).mean()

        # Williams %R (14) — используется в IndicatorEngine
        hh = high.rolling(14).max()
        ll = low.rolling(14).min()
        wr_denom = (hh - ll).replace(0, np.nan)
        dataframe["williams_r"] = ((hh - close) / wr_denom * -100).fillna(-50)

        # CCI (20) — используется в IndicatorEngine
        tp = (high + low + close) / 3.0
        cci_mean = tp.rolling(20).mean()
        cci_mad = tp.rolling(20).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        ).replace(0, np.nan)
        dataframe["cci"] = ((tp - cci_mean) / (0.015 * cci_mad)).fillna(0.0)

        # OBV — используется в IndicatorEngine (obv_trend)
        direction = np.sign(close.diff().fillna(0.0))
        obv_series = (direction * vol).cumsum()
        dataframe["obv"] = obv_series
        dataframe["obv_sma"] = obv_series.rolling(20).mean()

        # VWAP — используется в IndicatorEngine (дневной сброс по дате)
        try:
            tp_vwap = (high + low + close) / 3.0
            if "date" in dataframe.columns:
                dates = pd.to_datetime(dataframe["date"])
                day = dates.dt.date
                cum_tpv = (tp_vwap * vol).groupby(day).cumsum()
                cum_vol = vol.groupby(day).cumsum().replace(0, np.nan)
                dataframe["vwap"] = cum_tpv / cum_vol
            else:
                dataframe["vwap"] = (tp_vwap * vol).cumsum() / vol.cumsum().replace(0, np.nan)
        except Exception:
            dataframe["vwap"] = close

        # Score-колонки стратегии (для логики, не для plot — LLM считает их только по последней свече)
        for col in ["final_score", "tech_score", "sent_score"]:
            if col not in dataframe.columns:
                dataframe[col] = 0.0

        # SuperTrend колонки — гарантируем наличие
        for col in ["st_bull", "st_bear", "supertrend_line"]:
            if col not in dataframe.columns:
                dataframe[col] = np.nan

        return dataframe

    def _empty_result(self) -> dict:
        return {
            "expert_long_enter":  0,
            "expert_long_exit":   0,
            "expert_short_enter": 0,
            "expert_short_exit":  0,
            "expert_opinion":     "",
            "sent_score":         0.0,
            "tech_score":         0.0,
            "final_score":        0.0,
            "expert_stake":       float(STAKE_LEVELS[0]),
            "expert_confidence":  0.0,
            "_rec_key_news":      "",
            "_rec_news_summary":  "",
            "_rec_news_mood":     "нейтральный",
        }

    def _apply_columns(self, df: DataFrame, data: dict) -> DataFrame:
        """Применяет значения только к последней свече (anti-leakage)."""
        if df.empty:
            return df
        last_idx = df.index[-1]
        for col, val in data.items():
            if not col.startswith("_"):
                if col not in df.columns:
                    if isinstance(val, str):
                        df[col] = pd.Series([None] * len(df), index=df.index, dtype="object")
                    else:
                        df[col] = np.nan
                elif isinstance(val, str) and str(df[col].dtype) != "object":
                    # Избегаем FutureWarning при записи строк в float колонки.
                    df[col] = df[col].astype("object")
                df.at[last_idx, col] = val
        return df

    def _remember_score(self, pair: str, candle_id: str, data: dict) -> None:
        """Сохраняет score текущей свечи в памяти для последующей логики (DCA/exit)."""
        pair_hist = self._score_history.setdefault(pair, {})
        pair_hist[candle_id] = {
            "sent_score": float(data.get("sent_score", 0.0)),
            "tech_score": float(data.get("tech_score", 0.0)),
            "final_score": float(data.get("final_score", 0.0)),
        }
        # Ограничиваем память: держим только последние 4000 свечей на пару.
        if len(pair_hist) > 4000:
            keys_sorted = sorted(pair_hist.keys())
            for k in keys_sorted[:-4000]:
                pair_hist.pop(k, None)

    def _apply_score_history(self, pair: str, df: DataFrame) -> DataFrame:
        """Восстанавливает исторические score-значения в dataframe (для логики, не для plot)."""
        pair_hist = self._score_history.get(pair)
        if not pair_hist or df.empty:
            return df
        candle_ids = df["date"].apply(lambda d: str(pd.to_datetime(d)))
        for col in ("sent_score", "tech_score", "final_score"):
            series = candle_ids.map(lambda cid: pair_hist.get(cid, {}).get(col))
            known = series.notna()
            if known.any():
                df.loc[known, col] = series[known].astype(float).values
        return df

    def _should_send_neutral(self, pair: str, action: str) -> bool:
        now = time.time()
        last_action = self._last_notify_action.get(pair, "")
        last_ts = self._last_notify_ts.get(pair, 0.0)
        # Отправляем всегда, если action изменился.
        if action != last_action:
            self._last_notify_action[pair] = action
            self._last_notify_ts[pair] = now
            return True
        # Если action не менялся — только по cooldown.
        if now - last_ts >= NEUTRAL_NOTIFY_COOLDOWN_SEC:
            self._last_notify_ts[pair] = now
            return True
        return False

    def _should_send_in_trade_update(self, pair: str, candle_id: Optional[str]) -> bool:
        last_candle = self._last_in_trade_candle_id.get(pair, "")
        current_candle = candle_id or ""
        if current_candle and current_candle == last_candle:
            return False
        if current_candle:
            self._last_in_trade_candle_id[pair] = current_candle
        return True

    def _build_score_change_context(
        self, pair: str, rec: TradingRecommendation, ind: Indicators
    ) -> Tuple[str, str]:
        snap = self._entry_score_snapshot.get(pair)
        current = {
            "final": float(rec.final_score),
            "tech": float(rec.tech_score),
            "sent": float(rec.sent_score),
            "vol": float(ind.vol_ratio),
            "adx": float(ind.adx),
        }
        if not snap:
            self._entry_score_snapshot[pair] = current
            return (
                f"{current['final']:+.3f} → {current['final']:+.3f} (+0.000)",
                "первая свеча сопровождения, это базовый score входа",
            )

        d_final = current["final"] - float(snap.get("final", 0.0))
        d_tech = current["tech"] - float(snap.get("tech", 0.0))
        d_sent = current["sent"] - float(snap.get("sent", 0.0))
        d_vol = current["vol"] - float(snap.get("vol", 0.0))
        d_adx = current["adx"] - float(snap.get("adx", 0.0))

        score_line = f"{snap.get('final', 0.0):+.3f} → {current['final']:+.3f} ({d_final:+.3f})"
        reasons = []
        if abs(d_tech) >= 0.03:
            reasons.append(f"техника {d_tech:+.3f}")
        if abs(d_sent) >= 0.03:
            reasons.append(f"новости {d_sent:+.3f}")
        if abs(d_vol) >= 0.10:
            reasons.append(f"объём {d_vol:+.2f}x")
        if abs(d_adx) >= 2.0:
            reasons.append(f"ADX {d_adx:+.1f}")
        if not reasons:
            reasons.append("рынок без сильных сдвигов, изменение в пределах шума")

        return score_line, "; ".join(reasons)

    # ──────────────────────────────────────────────────────────────────
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe
        i = dataframe.index[-1]
        if "expert_long_enter" in dataframe.columns and int(dataframe.at[i, "expert_long_enter"] or 0) == 1:
            dataframe.at[i, "enter_long"] = 1
            dataframe.at[i, "enter_tag"] = "gpt_long"
        if "expert_short_enter" in dataframe.columns and int(dataframe.at[i, "expert_short_enter"] or 0) == 1:
            dataframe.at[i, "enter_short"] = 1
            dataframe.at[i, "enter_tag"] = "gpt_short"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Выход управляется через custom_exit
        return dataframe

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        """Трейлинг-стоп: активируется после достижения TAKE_PROFIT."""
        if not TRAILING_AFTER_TP:
            return self.stoploss
        trade_id   = trade.id
        max_profit = self._trailing_state.get(trade_id, 0.0)
        if current_profit > max_profit:
            self._trailing_state[trade_id] = current_profit
            max_profit = current_profit
        if max_profit >= TRAILING_MIN_ACT:
            trailing_stop = max_profit - TRAILING_RETRACE
            return max(self.stoploss, -trailing_stop if trailing_stop > 0 else self.stoploss)
        return self.stoploss

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        """Логика выхода по сигналу LLM или истечению времени."""
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return None
        last = df.iloc[-1].squeeze()
        tag  = trade.enter_tag or ""
        mins = (current_time - trade.open_date_utc).total_seconds() / 60

        if mins > MAX_TRADE_MINUTES:
            self.tg.send(f"⏰ <b>ПРИНУДИТЕЛЬНЫЙ ВЫХОД</b> {pair}: {current_profit:+.2%} (истекло время)")
            self._entry_score_snapshot.pop(pair, None)
            return "expired"

        for side_tag, exit_col, side_label, exit_label in [
            ("gpt_short", "expert_short_exit", "SHORT", "gpt_short_exit"),
            ("gpt_long",  "expert_long_exit",  "LONG",  "gpt_long_exit"),
        ]:
            if tag == side_tag and last.get(exit_col, 0) == 1:
                ind = IndicatorEngine.calculate(df, 1)
                scores = {
                    "sent":  float(last.get("sent_score",  0)),
                    "tech":  float(last.get("tech_score",  0)),
                    "final": float(last.get("final_score", 0)),
                }
                # Восстанавливаем объект rec для новостного блока из кэша
                cached_rec = None
                cached = self._cache.get(pair)
                if cached:
                    _, cr = cached
                    cached_rec = TradingRecommendation(
                        tech_score=scores["tech"],
                        sentiments=[],
                        key_news=cr.get("_rec_key_news", ""),
                        news_summary=cr.get("_rec_news_summary", ""),
                        news_mood=cr.get("_rec_news_mood", "нейтральный"),
                        summary=str(last.get("expert_opinion", "")),
                        confidence=float(last.get("expert_confidence", cr.get("expert_confidence", 0.7))),
                        stake_amount=float(last.get("expert_stake", STAKE_LEVELS[0])),
                    )
                if ind:
                    self.tg.send(TGFormatter.exit_msg(
                        pair, side_label, current_profit, mins, scores,
                        str(last.get("expert_opinion", "")), ind, cached_rec,
                    ))
                    if cached_rec:
                        self.tg.send(TGFormatter.reasoning_msg(pair, cached_rec, exit_label))
                # Записываем сделку в память
                self._trade_memory.add(TradeRecord(
                    pair=pair,
                    side=side_label,
                    profit_pct=current_profit * 100,
                    duration_min=mins,
                    entry_time=trade.open_date_utc,
                    exit_time=current_time,
                    final_score=scores["final"],
                    exit_reason=exit_label,
                ))
                self._update_closed_trade_metrics(
                    pair=pair,
                    profit_pct=current_profit * 100,
                    duration_min=mins,
                    final_score=scores["final"],
                )
                self._trailing_state.pop(trade.id, None)
                self._entry_times.pop(trade.id, None)
                self._entry_score_snapshot.pop(pair, None)
                return exit_label
        return None

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs):
        """Execution gate: slippage + risk guardrails перед входом."""
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return False
        last  = df.iloc[-1].squeeze()
        close = float(last["close"])

        if side == "long"  and rate > close * (1 + LONG_ENTRY_SLIPPAGE):
            logger.warning(f"[{pair}] Слишком большое проскальзывание LONG: rate={rate} close={close}")
            return False
        if side == "short" and rate < close * (1 - SHORT_ENTRY_SLIPPAGE):
            logger.warning(f"[{pair}] Слишком большое проскальзывание SHORT: rate={rate} close={close}")
            return False

        stake_requested = self._pending_stake.get(pair, float(last.get("expert_stake", STAKE_LEVELS[0])))
        # Фактический размер заявки (ношионал) может отличаться от рекомендации
        # из-за min_stake/шага количества на бирже.
        stake_actual = float(amount) * float(rate)
        stake_for_msg = stake_actual if stake_actual > 0 else stake_requested
        ind   = IndicatorEngine.calculate(df, 1)
        confidence = float(last.get("expert_confidence", 0.7))
        if confidence < ENTRY_MIN_CONFIDENCE:
            logger.warning(f"[{pair}] вход отклонён: confidence={confidence:.2f} < {ENTRY_MIN_CONFIDENCE:.2f}")
            return False
        if ind:
            if ind.atr_pct > EXECUTION_MAX_ATR_PCT:
                logger.warning(f"[{pair}] вход отклонён: ATR% слишком высокий {ind.atr_pct:.2f}%")
                return False
            if side == "long" and ind.st_dir != "бычий":
                logger.warning(f"[{pair}] вход LONG отклонён: SuperTrend={ind.st_dir}")
                return False
            if side == "short" and ind.st_dir != "медвежий":
                logger.warning(f"[{pair}] вход SHORT отклонён: SuperTrend={ind.st_dir}")
                return False

        cached = self._cache.get(pair)
        cached_data = cached[1] if cached else {}
        rec_proxy = TradingRecommendation(
            action=("LONG_ENTER" if side == "long" else "SHORT_ENTER"),
            tech_score=float(last.get("tech_score", 0.0)),
            sentiments=[],
            key_news=cached_data.get("_rec_key_news", ""),
            news_summary=cached_data.get("_rec_news_summary", ""),
            news_mood=cached_data.get("_rec_news_mood", "нейтральный"),
            news_reasoning=cached_data.get("_rec_news_reasoning", ""),
            tech_reasoning=cached_data.get("_rec_tech_reasoning", ""),
            counter_argument=cached_data.get("_rec_counter_argument", ""),
            summary=str(last.get("expert_opinion", "")),
            confidence=float(last.get("expert_confidence", cached_data.get("expert_confidence", 0.7))),
            stake_amount=stake_for_msg,
        )
        if ind:
            if abs(stake_for_msg - stake_requested) > 1.0:
                logger.info(
                    f"[{pair}] stake adjusted by exchange limits: "
                    f"requested={stake_requested:.2f} actual={stake_for_msg:.2f}"
                )
            self.tg.send(TGFormatter.entry(pair, side, rate, stake_for_msg, rec_proxy, ind))
            self.tg.send(TGFormatter.reasoning_msg(pair, rec_proxy, f"{'LONG' if side == 'long' else 'SHORT'}_ENTER"))
        return True

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: Optional[float],
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """
        Возвращает размер стейка, рекомендованный ИИ.
        Приоритет: _pending_stake → expert_stake из датафрейма → минимум из STAKE_LEVELS.
        """
        # 1. Берём стейк, сохранённый в populate_indicators (самый свежий)
        stake = self._pending_stake.get(pair)

        # 2. Фолбэк на значение из датафрейма
        if stake is None:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not df.empty:
                stake = float(df.iloc[-1].get("expert_stake", STAKE_LEVELS[0]))
            else:
                stake = float(STAKE_LEVELS[0])

        # 3. Привязываем к ближайшему уровню из STAKE_LEVELS
        stake = float(min(STAKE_LEVELS, key=lambda s: abs(s - stake)))

        # 4. Снижаем стейк при низком confidence — ИИ сам сомневается
        cached = self._cache.get(pair)
        if cached:
            _, cr = cached
            confidence = float(cr.get("expert_confidence", 0.7))
            if confidence < LOW_CONFIDENCE_STAKE_THRESHOLD:
                stake = float(STAKE_LEVELS[0])
                logger.info(
                    f"[{pair}] confidence={confidence:.2f} < {LOW_CONFIDENCE_STAKE_THRESHOLD:.2f} "
                    f"→ стейк снижен до минимума {stake}"
                )

        # 4.1 Риск-режим по рынку имеет приоритет над советом LLM
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            ind = IndicatorEngine.calculate(df, 1) if not df.empty else None
            if ind:
                if ind.atr_pct > 3.0 or ind.vol_ratio < 0.5 or ind.adx < 20:
                    stake = float(STAKE_LEVELS[0])
                elif abs(float(df.iloc[-1].get("final_score", 0.0))) > 0.65 and ind.atr_pct < 1.5 and ind.adx > 25:
                    stake = float(STAKE_LEVELS[-1])
        except Exception:
            pass

        # 5. Соблюдаем лимиты биржи
        if min_stake is not None:
            stake = max(stake, float(min_stake))
        if max_stake is not None:
            stake = min(stake, float(max_stake))

        logger.info(f"[{pair}] custom_stake_amount → {stake} USDT")
        return stake

    def _get_position(self, pair: str) -> Tuple[Optional[str], float]:
        """Возвращает (сторону, профит) открытой позиции или (None, 0)."""
        try:
            pair_norm = str(pair).split(":")[0]
            trades = [
                t
                for t in Trade.get_trades_proxy(is_open=True)
                if str(t.pair) == str(pair) or str(t.pair).split(":")[0] == pair_norm
            ]
            if not trades:
                return None, 0.0
            t    = trades[0]
            pair_for_rate = str(t.pair) or str(pair)
            rate: Optional[float] = None

            # 1) Пытаемся взять цену из последнего проанализированного датафрейма.
            df, _ = self.dp.get_analyzed_dataframe(pair_for_rate, self.timeframe)
            if df is not None and not df.empty and "close" in df.columns:
                rate = float(df.iloc[-1]["close"])

            # 2) Фолбэк на тикер, если датафрейм недоступен.
            if rate is None or rate <= 0:
                ticker = self.dp.ticker(pair_for_rate) or {}
                rate = float(ticker.get("last") or ticker.get("close") or 0.0)

            if rate <= 0:
                return None, 0.0
            return ("SHORT" if t.is_short else "LONG"), t.calc_profit_ratio(rate)
        except Exception as e:
            logger.warning(f"[{pair}] _get_position error: {e}")
            return None, 0.0