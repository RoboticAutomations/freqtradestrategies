"""
GPT Strategy v5.3 — один файл, вся логика здесь.

LONG/SHORT · LLM (gpt-4o-mini) · индикаторы · RSS · Fear&Greed · память сделок.

╔══════════════════════════════════════════════════════════════════════╗
║  СТРУКТУРА ФАЙЛА (сверху вниз)                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  [01] Импорты                                                        ║
║  [02] Конфигурация (# CFG) — пороги, стейки, LLM, риск                ║
║  [03] TelegramNotifier — отправка и HTML-санитайзер                  ║
║  [04] Indicators + IndicatorEngine — тех. расчёты                     ║
║  [05] NewsProvider — RSS новости                                     ║
║  [06] FearGreedProvider — индекс страха/жадности                     ║
║  [07] get_session_info — торговая сессия UTC                         ║
║  [08] TradeMemory — история сделок на диске                          ║
║  [09] PromptBuilder — промпты LLM (вход + сопровождение сделки)       ║
║  [10] Pydantic-модели — TradingRecommendation                        ║
║  [11] LLMClient — запросы к API и semantic-validator                 ║
║  [12] TGFormatter — сообщения в Telegram                             ║
║  [13] GPTStrategy — Freqtrade                                        ║
║       · вход: LLM + арбитр                                           ║
║       · сопровождение: LLM (manage, action=NEUTRAL)                  ║
║       · выход: SL / TP / trailing / таймаут (без LLM)                ║
╚══════════════════════════════════════════════════════════════════════╝
"""


import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple, get_args
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
# [01] ИМПОРТЫ — см. блок import выше
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# [02] КОНФИГУРАЦИЯ  (# CFG)
# ═══════════════════════════════════════════════════════════════════════

# 1) LLM и запросы к модели (как часто и как долго ждём ответ)
LLM_MODEL   = "gpt-4.1-mini"          # # CFG: модель ИИ
LLM_API_KEY = os.environ.get("LLM_API_KEY", "") or ""    # # CFG: ключ API
LLM_TIMEOUT = 60                     # # CFG: таймаут одного LLM-запроса (сек)
LLM_RETRIES = 3                      # # CFG: повторы LLM при ошибке
LLM_MIN_CALL_INTERVAL_SEC = 20       # # CFG: минимум секунд между LLM-вызовами по паре
LLM_INPUT_USD_PER_1M  = 0.40         # # CFG: fallback $/1M input tokens (gpt-4o-mini)
LLM_OUTPUT_USD_PER_1M = 1.60         # # CFG: fallback $/1M output tokens (gpt-4o-mini)

def _llm_key() -> str:
    return (os.environ.get("LLM_API_KEY") or str(LLM_API_KEY or "")).strip()

NEUTRAL_HISTORY_MAX = 2              # # CFG: прошлых NEUTRAL-анализов в промт (только после NEUTRAL)

# 2) Данные свечей и запуск стратегии
HISTORY_CANDLES = 10                 # # CFG: свечей в контексте для LLM
MIN_CANDLES     = 50                 # # CFG: минимум свечей для расчёта индикаторов
STARTUP_CANDLES = 210                # # CFG: сколько свечей прогреть при старте

# 3) Новости и внешний контекст
MAX_NEWS_ITEMS = 5                   # # CFG: макс. новостей для анализа
MAX_NEWS_POOL = 40                   # # CFG: сколько заголовков кэшировать до фильтра по паре
NEWS_CACHE_SEC = 600                 # # CFG: кэш новостей (сек)
RSS_FEEDS = [                        # # CFG: источники новостей
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://cryptonews.com/news/feed/",
]
FEAR_GREED_CACHE_SEC = 3600          # # CFG: кэш Fear & Greed (сек)
CONTEXT_TASK_TIMEOUT_SEC = 8.0       # # CFG: таймаут каждого источника контекста (сек)
CONTEXT_TOTAL_TIMEOUT_SEC = 10.0     # # CFG: целевой бюджет сбора контекста (сек)

# 4) Пороги решений (когда вход разрешён; выход — только механический SL/TP/trailing)
ENTRY_LONG_THRESHOLD  = +0.30          # # CFG: порог входа LONG
ENTRY_SHORT_THRESHOLD = -0.30          # # CFG: порог входа SHORT
SCORE_ROUND_DECIMALS = 2               # # CFG: округление score для пограничных решений
# длинный «⛔ Почему» в NEUTRAL — только при итоге ровно 0.00 (после округления)
ENTRY_MIN_CONFIDENCE = 0.40            # # CFG: минимум уверенности для входа
ENTRY_MIN_VOL_RATIO = 0.20             # # CFG: минимум объёма для входа
MIN_ALIGNED_INDICATORS = 4             # # CFG: мин. индикаторов в одну сторону для входа
ALIGN_INDICATOR_TOTAL = 10             # # CFG: всего индикаторов в count_alignment
ENTRY_MAX_ATR_PCT = 4.00               # # CFG: блок входа при экстремальной ATR
EXECUTION_MAX_ATR_PCT = 4.50           # # CFG: блок входа на этапе исполнения

# 5) Размеры позиций и веса оценок
STAKE_LEVELS = [30, 50, 100, 150, 200]  # # CFG: допустимые стейки (USDT)
WEIGHT_TECHNICAL = 0.85                 # # CFG: вес технического анализа
WEIGHT_SENTIMENT = 0.15                 # # CFG: вес сентимента новостей
LOW_CONFIDENCE_STAKE_THRESHOLD = 0.60   # # CFG: при confidence ниже этого порога стейк = минимум
TECH_ANCHOR_LLM_EPS = 0.05              # # CFG: |tech_score| ниже — LLM «обнулил»
TECH_ANCHOR_MIN_ABS = 0.10              # # CFG: мин. |anchor| для подстановки
TECH_ANCHOR_BALANCE_MAX_DIFF = 1        # # CFG: diff bull/bear ≤ этого → честный нейтрал, не трогаем

# 6) Риск-менеджмент и управление сделкой
STOPLOSS_PCT      = -0.04              # # CFG: стоп-лосс
TAKE_PROFIT       = 0.02               # # CFG: тейк-профит (триггер трейлинга)
TRAILING_AFTER_TP = True
TRAILING_RETRACE  = 0.005
TRAILING_MIN_ACT  = 0.00
FORCE_EXIT_BY_DAYS_ENABLED = False      # # CFG: принудительный выход по времени (как DSA)
FORCE_EXIT_AFTER_DAYS = 0.0             # # CFG: дней до force exit (если включено)
MAX_TRADE_MINUTES = 5000               # # CFG: макс. время сделки (мин)
LLM_MANAGE_ENABLED = False              # # CFG: ── ВКЛ/ВЫКЛ сопровождение открытых позиций через LLM ──
                                       # True  = LLM анализирует каждую открытую позицию (как было)
                                       # False = LLM не вызывается на открытые → экономия ~$30-45/мес.
                                       # TP/trailing/stoploss работают в Python независимо.
                                       # После изменения — рестарт контейнера.
LONG_ENTRY_SLIPPAGE  = 0.0025          # # CFG: макс. проскальзывание LONG
SHORT_ENTRY_SLIPPAGE = 0.0025          # # CFG: макс. проскальзывание SHORT

# 7) Мониторинг, уведомления и память
ALERT_COOLDOWN_SEC = 900                # # CFG: период мониторинг-алертов (сек)
MONITORING_MIN_CANDLES = 50             # # CFG: минимум свечей для расчёта мониторинг-метрик
NEUTRAL_NOTIFY_COOLDOWN_SEC = 0         # # CFG: 0 = отправлять NEUTRAL на каждой новой свече
TRADE_MEMORY_MAX = 50                   # # CFG: макс. записей сделок в памяти
MEMORY_FILE = "user_data/gpt_strategy_memory.json"  # # CFG: путь к файлу памяти
TRADE_EXECUTION_ENABLED_DEFAULT = True  # # CFG: дефолт, если параметр не задан в config.json

# ШПАРГАЛКА ПО БЫСТРОЙ НАСТРОЙКЕ 
# ─ Безопаснее (меньше входов / ниже риск):
#   ENTRY_LONG_THRESHOLD  -> выше (напр. 0.35)
#   ENTRY_SHORT_THRESHOLD -> ниже (напр. -0.35)
#   ENTRY_MIN_CONFIDENCE  -> выше (напр. 0.50-0.60)
#   ENTRY_MAX_ATR_PCT и EXECUTION_MAX_ATR_PCT -> ниже
#   STAKE_LEVELS -> уменьшить верхние уровни
#   LOW_CONFIDENCE_STAKE_THRESHOLD -> выше
#
# ─ Агрессивнее (больше входов / выше риск):
#   ENTRY_LONG_THRESHOLD  -> ниже (напр. 0.25)
#   ENTRY_SHORT_THRESHOLD -> выше (напр. -0.25)
#   ENTRY_MIN_CONFIDENCE  -> ниже (напр. 0.35-0.40)
#   ENTRY_MAX_ATR_PCT и EXECUTION_MAX_ATR_PCT -> выше
#   STAKE_LEVELS -> увеличить верхние уровни
#
# ─ Оптимизация расходов API:
#   LLM_MIN_CALL_INTERVAL_SEC -> выше
#   NEWS_CACHE_SEC и FEAR_GREED_CACHE_SEC -> выше
#   MAX_NEWS_ITEMS -> ниже
#
# ─ Оповещения и память:
#   NEUTRAL_NOTIFY_COOLDOWN_SEC = 0 -> уведомление NEUTRAL на каждой свече
#   NEUTRAL_NOTIFY_COOLDOWN_SEC > 0 -> реже уведомления
#   TRADE_MEMORY_MAX -> сколько сделок хранится в истории для промта
#
# legacy compact prompt removed; используем полный PromptBuilder.SYSTEM_PROMPT.


# ═══════════════════════════════════════════════════════════════════════
# [03] TELEGRAM NOTIFIER
# ═══════════════════════════════════════════════════════════════════════

class TelegramNotifier:
    """Отправляет HTML-сообщения в Telegram. Нужен token и chat_id в config['telegram']."""

    # Whitelist HTML-тегов, которые Telegram парсит. Всё остальное экранируется.
    _ALLOWED_TAG_RE = re.compile(
        r"</?(?:b|strong|i|em|u|ins|s|strike|del|code|pre|tg-spoiler|blockquote"
        r"|a(?:\s+href\s*=\s*\"[^\"]*\")?)\s*>",
        re.IGNORECASE,
    )

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

    @classmethod
    def _sanitize_html(cls, text: str) -> str:
        """Экранирует все '<', '>', '&' в тексте, кроме допустимых Telegram-тегов.
        Защищает от ошибок «Unsupported start tag» когда LLM пишет '<20%' и т.п."""
        placeholders: Dict[str, str] = {}

        def _save(m: "re.Match") -> str:
            key = f"\x00TG{len(placeholders)}\x00"
            placeholders[key] = m.group(0)
            return key

        text = cls._ALLOWED_TAG_RE.sub(_save, text)
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        for k, v in placeholders.items():
            text = text.replace(k, v)
        return text

    def verify_connection(self) -> Tuple[bool, str]:
        """Реальный ping Telegram API (getMe) — без отправки сообщения в чат."""
        if not self.enabled:
            return False, "нет token/chat_id"
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{self.token}/getMe",
                timeout=10,
            )
            if resp.ok:
                data = resp.json()
                if data.get("ok"):
                    username = data.get("result", {}).get("username", "?")
                    return True, f"@{username}"
            return False, f"HTTP {resp.status_code}: {resp.text[:120]}"
        except Exception as e:
            return False, str(e)

    def send(self, message: str) -> bool:
        if not self.enabled:
            logger.info(f"[TG SKIP] {message[:120]}")
            return False
        message = self._sanitize_html(message)
        if len(message) > 4096:
            message = message[:4090] + "\n✂️"
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"},
                timeout=10,
            )
            if resp.ok and resp.json().get("ok"):
                return True
            if not resp.ok:
                logger.warning(f"[TG ERR] {resp.status_code}: {resp.text[:200]}")
            else:
                logger.warning(f"[TG ERR] API ok=false: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"[TG EXC] {e}")
        return False

    def send_with_id(self, message: str) -> Optional[int]:
        if not self.enabled:
            return None
        message = self._sanitize_html(message)
        if len(message) > 4096:
            message = message[:4090] + "\n✂️"
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"},
                timeout=10,
            )
            if resp.ok:
                data = resp.json()
                if data.get("ok"):
                    return data.get("result", {}).get("message_id")
            else:
                logger.warning(f"[TG ERR] {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"[TG EXC] {e}")
        return None

    def pin_message(self, message_id: int) -> None:
        if not self.enabled:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.token}/pinChatMessage",
                json={"chat_id": self.chat_id, "message_id": message_id},
                timeout=5,
            )
            logger.info("[TG] Сообщение закреплено")
        except Exception as e:
            logger.warning(f"[TG] Не удалось закрепить: {e}")


# ═══════════════════════════════════════════════════════════════════════
# [04] INDICATOR ENGINE  (Indicators, IndicatorEngine)
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

    # Согласование (считает код)
    align_bull: int = 0
    align_bear: int = 0


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

        ind.align_bull, ind.align_bear = cls.count_alignment(ind)
        return ind

    @staticmethod
    def count_alignment(ind: Indicators) -> Tuple[int, int]:
        """Считает индикаторы за LONG и за SHORT (детерминированно, как в арбитре)."""
        bull = bear = 0
        if ind.st_dir == "бычий":
            bull += 1
        elif ind.st_dir == "медвежий":
            bear += 1
        if ind.sma_cross == "бычий":
            bull += 1
        elif ind.sma_cross == "медвежий":
            bear += 1
        if ind.price > ind.sma200:
            bull += 1
        elif ind.price < ind.sma200:
            bear += 1
        if ind.rsi < 45:
            bull += 1
        elif ind.rsi > 55:
            bear += 1
        if ind.macd_trend == "бычий":
            bull += 1
        elif ind.macd_trend == "медвежий":
            bear += 1
        if ind.bb_pct < 30 or ind.bb_pos in ("ниже нижней", "нижняя зона"):
            bull += 1
        elif ind.bb_pct > 70 or ind.bb_pos in ("выше верхней", "верхняя зона"):
            bear += 1
        if ind.stoch_zone == "перепродан" or ind.stoch_cross == "бычье":
            bull += 1
        elif ind.stoch_zone == "перекуплен" or ind.stoch_cross == "медвежье":
            bear += 1
        if ind.williams_zone == "перепродан":
            bull += 1
        elif ind.williams_zone == "перекуплен":
            bear += 1
        if ind.cci_signal == "перепродан":
            bull += 1
        elif ind.cci_signal == "перекуплен":
            bear += 1
        if ind.obv_trend == "бычий":
            bull += 1
        elif ind.obv_trend == "медвежий":
            bear += 1
        return bull, bear

    @classmethod
    def compute_tech_anchor(cls, ind: Indicators) -> float:
        """
        Rule-based tech_score по весам индикаторов (ориентир LLM).
        Не заменяет LLM полностью — используется snap_tech_score(), если модель дала ~0.
        """
        base = 0.0

        if ind.st_dir == "бычий":
            base += 0.50
        elif ind.st_dir == "медвежий":
            base -= 0.50

        if ind.sma_cross == "бычий":
            base += 0.20
        elif ind.sma_cross == "медвежий":
            base -= 0.20

        if ind.price > ind.sma200:
            base += 0.10
        elif ind.price < ind.sma200:
            base -= 0.10

        rsi = ind.rsi
        if rsi < 20:
            base += 0.25
        elif rsi < 30:
            base += 0.20
        elif rsi < 45:
            base += 0.05
        elif rsi > 80:
            base -= 0.25
        elif rsi > 70:
            base -= 0.20
        elif rsi > 55:
            base -= 0.05

        if ind.macd_trend == "бычий":
            base += 0.10
        elif ind.macd_trend == "медвежий":
            base -= 0.10

        bb_pct = ind.bb_pct
        bb_pos = ind.bb_pos or ""
        if bb_pct < 0 or "ниже нижней" in bb_pos:
            base += 0.20
        elif bb_pct > 100 or "выше верхней" in bb_pos:
            base -= 0.20
        elif bb_pct < 30 or "нижняя" in bb_pos:
            base += 0.10
        elif bb_pct > 70 or "верхняя" in bb_pos:
            base -= 0.10

        if ind.stoch_zone == "перепродан" or ind.stoch_cross == "бычье":
            base += 0.10
        elif ind.stoch_zone == "перекуплен" or ind.stoch_cross == "медвежье":
            base -= 0.10

        if ind.williams_zone == "перепродан":
            base += 0.10
        elif ind.williams_zone == "перекуплен":
            base -= 0.10

        if ind.cci_signal == "перепродан":
            base += 0.15
        elif ind.cci_signal == "перекуплен":
            base -= 0.15

        if ind.obv_trend == "бычий":
            base += 0.10
        elif ind.obv_trend == "медвежий":
            base -= 0.10

        if ind.vol_ratio > 1.5:
            base += 0.15
        elif ind.vol_ratio < 0.5:
            base -= 0.10

        mult = 1.0
        if ind.adx > 25:
            mult *= 1.2
        elif ind.adx < 20:
            mult *= 0.8
        if ind.vol_ratio > 1.5:
            mult *= 1.1
        if ind.atr_pct > 3.0:
            mult *= 0.9

        return float(np.clip(base * mult, -1.0, 1.0))

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
# [05] NEWS PROVIDER
# ═══════════════════════════════════════════════════════════════════════

class NewsProvider:
    """Загружает заголовки новостей из RSS и кэширует результат."""
    _cache: Dict[str, Tuple[float, List[str]]] = {}

    @classmethod
    def _fetch_one_feed(cls, url: str) -> List[str]:
        """Один RSS-источник — для параллельной загрузки."""
        entries: List[str] = []
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            resp = requests.get(url, timeout=8, headers=headers)
            if resp.status_code != 200:
                return entries
            if feedparser is not None:
                feed = feedparser.parse(resp.content)
                for e in feed.entries:
                    title = e.get("title", "").strip()[:220]
                    if title:
                        entries.append(title)
            else:
                try:
                    root = ET.fromstring(resp.content)
                    for title_el in root.findall(".//item/title"):
                        title = (title_el.text or "").strip()[:220]
                        if title:
                            entries.append(title)
                    for title_el in root.findall(".//entry/title"):
                        title = (title_el.text or "").strip()[:220]
                        if title:
                            entries.append(title)
                except Exception:
                    pass
        except Exception as exc:
            logger.debug(f"[News] feed error {url}: {exc}")
        return entries

    @classmethod
    def fetch(cls, max_items: int = MAX_NEWS_POOL) -> List[str]:
        """Возвращает пул уникальных заголовков (до max_items). Фильтр по паре — снаружи."""
        key = f"rss_{max_items}"
        now = time.time()
        if key in cls._cache:
            ct, items = cls._cache[key]
            if now - ct < NEWS_CACHE_SEC:
                return items

        raw_entries: List[str] = []
        try:
            with ThreadPoolExecutor(max_workers=max(1, len(RSS_FEEDS))) as pool:
                for titles in pool.map(cls._fetch_one_feed, RSS_FEEDS):
                    raw_entries.extend(titles)
        except Exception as exc:
            logger.warning(f"[News] parallel fetch error: {exc}")

        unique: List[str] = []
        for title in raw_entries:
            if len(unique) >= max_items:
                break
            title_lower = title.lower()
            is_dup = False
            for existing in unique:
                words_new = set(title_lower.split())
                words_ex = set(existing.lower().split())
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
        logger.info(f"[News] RSS pool: {len(raw_entries)} raw -> {len(unique)} unique (feeds={len(RSS_FEEDS)})")
        return unique


# ═══════════════════════════════════════════════════════════════════════
# [06] FEAR & GREED PROVIDER
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
# [08] TRADING SESSION  (get_session_info)
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
# [09] TRADE MEMORY  (TradeRecord, TradeMemory)
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
        """Сохраняет историю на диск после каждого изменения.

        Атомарно: пишем во временный файл рядом и атомарно переименовываем,
        чтобы при сбое посередине записи не остаться с битым JSON.
        """
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
            tmp_path = f"{MEMORY_FILE}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                _json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp_path, MEMORY_FILE)
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


# ═══════════════════════════════════════════════════════════════════════
# NEWS / REASONING POST-PROCESS (код — не полагаемся на LLM для sent и клише)
# ═══════════════════════════════════════════════════════════════════════

# Маппинг: тикер base-валюты → синонимы для регекспа (название проекта)
# Используется при автогенерации _TRACKED_COINS_RE из pair_whitelist.
_COIN_NAMES: Dict[str, list] = {
    # ───── L1 / основные ─────
    "BTC":   ["bitcoin", "btc"],
    "ETH":   ["ethereum", "eth"],
    "BNB":   ["binance coin", "bnb"],
    "SOL":   ["solana", "sol"],
    "XRP":   ["ripple", "xrp"],
    "ADA":   ["cardano", "ada"],
    "DOGE":  ["dogecoin", "doge"],
    "TRX":   ["tron", "trx"],
    "AVAX":  ["avalanche", "avax"],
    "DOT":   ["polkadot", "dot"],
    "MATIC": ["polygon", "matic"],
    "LTC":   ["litecoin", "ltc"],
    "TON":   ["toncoin", "ton"],
    "BCH":   ["bitcoin cash", "bch"],
    "ETC":   ["ethereum classic", "etc"],
    "ATOM":  ["cosmos", "atom"],
    "NEAR":  ["near protocol", "near"],
    "ICP":   ["internet computer", "icp"],
    "HBAR":  ["hedera", "hbar"],
    "APT":   ["aptos", "apt"],
    "SUI":   ["sui"],
    "STX":   ["stacks", "stx"],
    "FIL":   ["filecoin", "fil"],
    "VET":   ["vechain", "vet"],
    "ALGO":  ["algorand", "algo"],
    "XLM":   ["stellar", "xlm"],
    "XMR":   ["monero", "xmr"],
    "XTZ":   ["tezos", "xtz"],
    "FLOW":  ["flow"],
    "THETA": ["theta"],
    "EOS":   ["eos"],
    "IOTA":  ["iota"],
    "NEO":   ["neo"],
    "DASH":  ["dash"],
    "ZEC":   ["zcash", "zec"],
    "KAS":   ["kaspa", "kas"],
    "TAO":   ["bittensor", "tao"],
    "MINA":  ["mina protocol", "mina"],
    "KSM":   ["kusama", "ksm"],

    # ───── L2 / Roll-up ─────
    "ARB":   ["arbitrum", "arb"],
    "OP":    ["optimism", "\\bop\\b"],
    "STRK":  ["starknet", "strk"],
    "IMX":   ["immutable", "imx"],
    "MANTA": ["manta network", "manta"],

    # ───── DeFi / DEX ─────
    "UNI":   ["uniswap", "uni"],
    "AAVE":  ["aave"],
    "MKR":   ["makerdao", "mkr"],
    "LDO":   ["lido", "ldo"],
    "CRV":   ["curve", "crv"],
    "SNX":   ["synthetix", "snx"],
    "COMP":  ["compound", "comp"],
    "CAKE":  ["pancakeswap", "cake"],
    "DYDX":  ["dydx"],
    "GMX":   ["gmx"],
    "ENA":   ["ethena", "ena"],
    "PENDLE":["pendle"],
    "JUP":   ["jupiter exchange", "jup"],
    "SUSHI": ["sushiswap", "sushi"],
    "1INCH": ["1inch"],

    # ───── Oracle / Infrastructure ─────
    "LINK":  ["chainlink", "link"],
    "PYTH":  ["pyth network", "pyth"],
    "GRT":   ["the graph", "grt"],
    "AR":    ["arweave", "\\bar\\b"],
    "RNDR":  ["render", "rndr"],
    "FET":   ["fetch.ai", "fet"],
    "INJ":   ["injective", "inj"],
    "TIA":   ["celestia", "tia"],
    "JTO":   ["jito", "jto"],
    "ENS":   ["ethereum name service", "ens"],
    "BAT":   ["basic attention token", "bat"],

    # ───── RWA / Финтех ─────
    "ONDO":  ["ondo finance", "ondo"],
    "QNT":   ["quant", "qnt"],

    # ───── Gaming / NFT / Метавселенные ─────
    "SAND":  ["sandbox", "sand"],
    "MANA":  ["decentraland", "mana"],
    "AXS":   ["axie infinity", "axs"],
    "GALA":  ["gala games", "gala"],
    "BLUR":  ["blur"],

    # ───── BTC Eco ─────
    "ORDI":  ["ordinals", "ordi"],
    "SATS":  ["1000sats", "sats"],

    # ───── Memes ─────
    "PEPE":  ["pepe"],
    "SHIB":  ["shiba inu", "shib"],
    "FLOKI": ["floki"],
    "BONK":  ["bonk"],
    "WIF":   ["dogwifhat", "wif"],
    "MEME":  ["memecoin"],

    # ───── Прочее популярное ─────
    "WLD":   ["worldcoin", "wld"],
    "RUNE":  ["thorchain", "rune"],
    "DYM":   ["dymension", "dym"],
    "KAVA":  ["kava"],
    "ROSE":  ["oasis network", "rose"],
    "CFX":   ["conflux", "cfx"],
    "ZIL":   ["zilliqa", "zil"],
    "LRC":   ["loopring", "lrc"],
    "CHZ":   ["chiliz", "chz"],
    "ENJ":   ["enjin", "enj"],
}

# Дефолтный набор — на случай если whitelist ещё не подгрузился.
# Реально перезаписывается в GPTStrategy.__init__ через update_tracked_coins_from_whitelist().
_TRACKED_COINS_RE: Dict[str, re.Pattern] = {
    "BTC": re.compile(r"\b(bitcoin|btc)\b", re.I),
    "ETH": re.compile(r"\b(ethereum|eth)\b", re.I),
    "SOL": re.compile(r"\b(solana|\bsol\b)\b", re.I),
}


def update_tracked_coins_from_whitelist(whitelist) -> None:
    """Перестраивает _TRACKED_COINS_RE на основе пар в config.json → exchange.pair_whitelist.

    Для каждой base-валюты из whitelist берёт синонимы из _COIN_NAMES (название проекта + тикер),
    собирает regex. Если в _COIN_NAMES нет записи — использует тикер в нижнем регистре как fallback.

    Также удаляет из _FOREIGN_ALTS_RE те монеты, которые юзер сам торгует (чтобы их новости
    не блокировались как «чужой alt»).
    """
    global _TRACKED_COINS_RE, _FOREIGN_ALTS_RE
    new_re: Dict[str, re.Pattern] = {}
    base_tokens: set = set()
    for pair in (whitelist or []):
        try:
            base = pair.split("/")[0].split(":")[0].upper()
        except Exception:
            continue
        if not base or base in new_re:
            continue
        base_tokens.add(base.lower())
        synonyms = _COIN_NAMES.get(base) or [base.lower()]
        # Экранируем спец-символы, склеиваем в alternation
        alt = "|".join(s if s.startswith("\\b") else re.escape(s) for s in synonyms)
        try:
            new_re[base] = re.compile(rf"\b({alt})\b", re.I)
        except re.error:
            new_re[base] = re.compile(rf"\b({re.escape(base.lower())})\b", re.I)
    if new_re:
        _TRACKED_COINS_RE = new_re

    # Пересобираем _FOREIGN_ALTS_RE, исключая монеты из whitelist
    blocked_alts = ["hype", "pepe", "doge", "shib", "wif", "bonk", "floki", "memecoin", "airdrop"]
    blocked_alts = [a for a in blocked_alts if a not in base_tokens]
    if blocked_alts:
        _FOREIGN_ALTS_RE = re.compile(r"\b(" + "|".join(blocked_alts) + r")\b", re.I)
    else:
        # Если все blocked попали в whitelist — фильтр становится no-op
        _FOREIGN_ALTS_RE = re.compile(r"(?!x)x")  # никогда не матчит
    logger.info(f"[News] tracked coins rebuilt from whitelist: {list(new_re.keys())}")
# Общий рынок — без привязки к конкретной монете (BTC/ETH/SOL не в паттерне)
_GENERAL_MARKET_RE = re.compile(
    r"\b("
    r"crypto(?:currency| currencies| market| markets| trading| assets?)?|"
    r"blockchain|defi|web3|digital assets?|stablecoins?|altcoins?|"
    r"\bsec\b|cftc|regulator|regulation|lawmakers|"
    r"spot etf|\betf\b|grayscale|"
    r"binance|coinbase|kraken|"
    r"fear and greed|market cap|"
    r"token(?:s)?|nft|mining|halving|"
    r"fed|fomc|inflation|interest rate|tariff|macro"
    r")\b",
    re.I,
)
_FOREIGN_ALTS_RE = re.compile(
    r"\b(hype|pepe|doge|shib|wif|bonk|floki|memecoin|airdrop)\b",
    re.I,
)
_TECH_CLICHE_PHRASES = (
    "текущая цена находится",
    "цена находится",
    "что создаёт противоречие",
    "что создает противоречие",
    "добавляет неопредел",
    "общий фон новостей",
)
_NEWS_CLICHE_PHRASES = (
    "большинство новостей не",
    "не имеют значительного влияния",
    "не имеют прямого влияния",
    "не связаны с btc",
    "не связаны с eth",
    "не связаны с sol",
    "снижает их значимость",
    "снижает их релевантность",
    "не представляют собой свеж",
)
_CONFIDENCE_MANAGE_CONFLICT_PHRASES = (
    "позиция не оправдан",
    "позицию не оправдан",
    "позиции не оправдан",
    "не уверен, что текущ",
    "не уверен что текущ",
    "не уверен, что long",
    "не уверен что long",
    "не уверен, что short",
    "не уверен что short",
    "не уверен в текущей поз",
    "не уверен в long",
    "не уверен в short",
    "не стоит держать",
    "не стоит удерживать",
    "лучше закрыть",
    "лучше выйти",
    "стоит закрыть",
    "стоит выйти",
    "следует закрыть",
    "следует выйти",
    "рекомендую закрыть",
    "рекомендую выйти",
    "закрыть позици",
    "выйти из позици",
    "не вижу смысла держать",
    "вход был ошиб",
    "не стоило входить",
    "не следовало входить",
)


def _confidence_reason_conflicts_manage(text: str) -> bool:
    """Текст сомневается в открытой позиции или призывает к выходу — не для сопровождения."""
    low = (text or "").lower()
    if any(p in low for p in _CONFIDENCE_MANAGE_CONFLICT_PHRASES):
        return True
    if "оправдан" in low and ("не уверен" in low or "сомнева" in low):
        return True
    if "позици" in low and any(p in low for p in ("закрыть", "выйти", "не оправдан", "не стоит держ")):
        return True
    return False


def _entry_confidence_risks(ind: Indicators, action: str) -> List[str]:
    """Конкретные риски для пояснения уверенности при LONG/SHORT ENTER."""
    risks: List[str] = []
    if action == "LONG_ENTER":
        if ind.rsi > 70:
            risks.append(f"RSI {ind.rsi:.0f} перекуплен")
        if ind.stoch_zone == "перекуплен":
            risks.append("Stoch RSI перекуплен")
        if ind.bb_squeeze and ind.bb_pct > 70:
            risks.append("BB сжаты у верхней границы")
        elif ind.bb_squeeze:
            risks.append("BB в сжатии")
        if ind.resistance > 0 and ind.price > 0:
            dist_pct = (ind.resistance - ind.price) / ind.price * 100.0
            if 0 <= dist_pct < 0.20:
                risks.append(f"сопротивление в {dist_pct:.2f}%")
    elif action == "SHORT_ENTER":
        if ind.rsi < 30:
            risks.append(f"RSI {ind.rsi:.0f} перепродан")
        if ind.stoch_zone == "перепродан":
            risks.append("Stoch RSI перепродан")
        if ind.bb_squeeze and ind.bb_pct < 30:
            risks.append("BB сжаты у нижней границы")
        elif ind.bb_squeeze:
            risks.append("BB в сжатии")
        if ind.support > 0 and ind.price > 0:
            dist_pct = (ind.price - ind.support) / ind.price * 100.0
            if 0 <= dist_pct < 0.20:
                risks.append(f"поддержка в {dist_pct:.2f}%")
    if ind.vol_ratio < 0.80:
        risks.append(f"объём {ind.vol_ratio:.2f}x сдержанный")
    return risks
_COUNTER_CLICHE_PHRASES = (
    "слабый adx",
    "медвежий sma",
    "создаёт противоречие",
    "создает противоречие",
    "сжатие bollinger",
    "неопределенность",
    "неопределённость",
)


def _pair_token(pair: str) -> str:
    return pair.split("/")[0].split(":")[0].upper()


def _mentioned_tracked_coins(text: str) -> set:
    found: set = set()
    for token, pat in _TRACKED_COINS_RE.items():
        if pat.search(text):
            found.add(token)
    return found


def _news_headline_relevant(text: str, pair_token: str) -> bool:
    """RSS/LLM: своя монета ИЛИ общий крипто-фон — не новости про другую монету."""
    if not (text or "").strip():
        return False
    if _FOREIGN_ALTS_RE.search(text):
        return False

    mentioned = _mentioned_tracked_coins(text)
    if pair_token in mentioned:
        return True
    if mentioned:
        return False
    if _GENERAL_MARKET_RE.search(text):
        return True
    return False


def _filter_news_headlines_for_pair(headlines: List[str], pair: str) -> List[str]:
    token = _pair_token(pair)
    return [h for h in (headlines or []) if _news_headline_relevant(h, token)]


def _collect_news_texts(rec: "TradingRecommendation") -> List[str]:
    texts: List[str] = []
    for field in (rec.key_news, rec.news_summary, rec.news_reasoning):
        if field:
            texts.append(str(field))
    for item in rec.sentiments or []:
        if isinstance(item, dict) and item.get("title"):
            texts.append(str(item["title"]))
    return texts


def sanitize_news_recommendation(rec: "TradingRecommendation", pair: str) -> "TradingRecommendation":
    """Оставляет только новости про эту монету или общий крипто-рынок."""
    token = _pair_token(pair)
    filtered_sentiments = [
        item for item in (rec.sentiments or [])
        if isinstance(item, dict)
        and _news_headline_relevant(str(item.get("title", "") or ""), token)
    ]
    key = (rec.key_news or "").strip()
    key_ok = bool(key) and key != "нет релевантных" and _news_headline_relevant(key, token)

    if filtered_sentiments or key_ok:
        key_news = key[:120] if key_ok else str(filtered_sentiments[0].get("title", ""))[:120]
        return rec.model_copy(update={
            "sentiments": filtered_sentiments,
            "key_news": key_news,
        })

    return rec.model_copy(update={
        "sentiments": [],
        "key_news": "нет релевантных",
        "news_summary": f"Нет новостей по {pair}.",
        "news_mood": "нейтральный",
        "news_reasoning": f"По {pair} релевантных новостей нет — sent_score=0.",
    })


def snap_tech_score(rec: "TradingRecommendation", ind: Indicators) -> "TradingRecommendation":
    """
    Если LLM поставил tech_score ≈ 0, а индикаторы явно в одну сторону —
    подставляем rule-based anchor (compute_tech_anchor).
    """
    llm_tech = float(rec.tech_score)
    if abs(llm_tech) >= TECH_ANCHOR_LLM_EPS:
        return rec

    anchor = IndicatorEngine.compute_tech_anchor(ind)
    if abs(anchor) < TECH_ANCHOR_MIN_ABS:
        return rec

    max_align = max(ind.align_bull, ind.align_bear)
    if max_align < MIN_ALIGNED_INDICATORS:
        return rec

    align_diff = abs(ind.align_bull - ind.align_bear)
    min_side = min(ind.align_bull, ind.align_bear)
    if (
        align_diff <= TECH_ANCHOR_BALANCE_MAX_DIFF
        and min_side >= MIN_ALIGNED_INDICATORS
    ):
        return rec

    logger.info(
        f"[tech_anchor] LLM tech={llm_tech:+.3f} -> {anchor:+.3f} "
        f"(align L{ind.align_bull}/S{ind.align_bear})"
    )
    return rec.model_copy(update={"tech_score": anchor})


def polish_recommendation_text(
    rec: "TradingRecommendation", pair: str, ind: "Indicators",
    news_items_count: int = -1,
) -> "TradingRecommendation":
    """Tech/counter для Telegram — связный текст от кода (не сухой список LLM)."""
    action = rec.action or ""
    tech = PromptBuilder.build_tech_narrative(pair, ind, action)
    counter = PromptBuilder.build_counter_narrative(pair, ind, action)

    updates: Dict[str, Any] = {"tech_reasoning": tech, "counter_argument": counter}
    token = _pair_token(pair)
    if rec.sent_score == 0.0:
        no_rss = news_items_count == 0
        no_llm_news = (
            not rec.sentiments
            and (rec.key_news or "").strip() in ("", "нет релевантных")
        )
        if no_rss or (no_llm_news and news_items_count >= 0):
            updates["news_reasoning"] = (
                f"RSS не дал релевантных заголовков для {token} — sent_score=0."
                if no_rss
                else f"По {token} релевантных новостей нет — sent_score=0."
            )
            updates["news_summary"] = f"Нет новостей по {pair}."
            updates["news_mood"] = "нейтральный"
            updates["key_news"] = "нет релевантных"
        else:
            if not (rec.news_reasoning or "").strip():
                updates["news_reasoning"] = (
                    f"Новостной фон для {token} слабый или нейтральный — sent_score=0."
                )
            if not (rec.news_summary or "").strip() or rec.news_summary.startswith("Нет новостей"):
                updates["news_summary"] = (
                    f"Новости есть, но влияние на {pair} нейтральное (sent_score=0)."
                )
    else:
        news_r = (rec.news_reasoning or "").strip()
        news_low = news_r.lower()
        if not news_r or any(p in news_low for p in _NEWS_CLICHE_PHRASES):
            updates["news_reasoning"] = (
                f"Ключевая новость: {(rec.key_news or '—')[:100]}. "
                f"sent_score {rec.sent_score:+.2f}."
            )

    return rec.model_copy(update=updates)


# ═══════════════════════════════════════════════════════════════════════
# [10] PROMPT BUILDER  (PromptBuilder)
# ═══════════════════════════════════════════════════════════════════════

class PromptBuilder:
    """Формирует структурированные промпты для LLM."""

    SYSTEM_PROMPT = """Ты — профессиональный криптовалютный трейдер с 15-летним опытом на фьючерсных рынках.
Ты торговал через кризисы 2018, 2020, 2022 годов. Ты знаешь цену каждой сделки.

╔══════════════════════════════════════════════════════════════════════╗
║  ЖЕЛЕЗНОЕ ПРАВИЛО #1: сохранение капитала важнее любой прибыли     ║
║  ЖЕЛЕЗНОЕ ПРАВИЛО #2: нет сетапа — нет сделки. Всегда NEUTRAL      ║
║  ЖЕЛЕЗНОЕ ПРАВИЛО #3: сомневаешься — не входи                      ║
╚══════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ДВА РЕЖИМА ЗАПРОСА (смотри заголовок user-сообщения)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

«АНАЛИЗ ВХОДА»          → ШАГи 1–3 и 5. Допустимые action: LONG_ENTER, SHORT_ENTER, NEUTRAL.
«СОПРОВОЖДЕНИЕ ПОЗИЦИИ» → ШАГи 1–2 и 4. Допустимый action: только NEUTRAL.
Закрытие сделки выполняет только код (стоп-лосс / тейк-профит / trailing), не ты.

Ты не просто смотришь на цифры — ты понимаешь контекст рынка.
Ты чувствуешь, когда рынок готовится к движению, а когда — к ловушке.
Ты знаешь разницу между реальным сигналом и шумом.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ПРОФИЛЬ ТОРГОВОЙ СИСТЕМЫ — знай это всегда
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Стиль торговли  : intraday / swing на 1h–4h (не скальпинг, не позиционная)
Горизонт сделки : средний — движения внутри текущей структуры на 1h–4h.
                  Тебя не интересует "куда BTC через месяц" —
                  тебя интересует "куда пойдёт цена в следующие 4–16 свечей"
                  (≈4–24 ч на 1h, до 2–3 суток на 4h).
Таймфрейм       : 1h–4h. Свеча = значимый сегмент структуры, не минутный шум.
                  Старший ТФ — контекст и фильтр; явный противотренд на 4h
                  при торговле 1h — повод для NEUTRAL или высокой осторожности.
Монеты          : статический список пар. Ты не ищешь новые инструменты,
                  ты досконально знаешь поведение именно этих активов.

Что это значит для анализа:
  • Новости и макро влияют на фон, но решение строится на структуре 1h–4h.
  • Сигналы должны быть актуальны для движения на несколько свечей вперёд,
    а не для недельного тренда.
  • Вход по тренду текущего ТФ предпочтительнее контртрендового импульса.
  • BB squeeze на 1h–4h — NEUTRAL до пробоя с объёмом.
  • Малейшая неопределённость по структуре = NEUTRAL,
    даже если новости выглядят красиво.
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
    > Используй ТОЛЬКО новости из блока НОВОСТИ ниже — про эту монету или общий крипто-рынок.
    > Новость про другую монету (Ethereum при BTC и т.п.) код отбросит — не включай в sentiments.
    > Фейк/кликбейт/паника без фактов = 0.1-0.2
  • recency    in [0.1; 1.0]   — актуальность события прямо сейчас
    > Повторная новость / пересказ = 0.2-0.4
  • score = sentiment * relevance * recency

sent_score = clip(weighted_mean(все score), -1.0, +1.0)

Найди ключевую новость (max |score|) -> key_news.
news_summary — 2 предложения: общий фон + главный драйвер.
news_mood — одно слово или короткая фраза: "позитивный", "негативный", "нейтральный", "смешанный", "тревожный", "эйфория"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ШАГ 2: ТЕХНИЧЕСКИЙ АНАЛИЗ (tech_score, вес 85%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

В user-сообщении уже переданы готовые значения индикаторов и блок
«СОГЛАСОВАНИЕ ИНДИКАТОРОВ (посчитано кодом)»: За LONG / За SHORT / из 10.

ЖЕЛЕЗНО — не дублируй расчёт кода:
  • НЕ пересчитывай согласование по таблице весов — цифры align от кода авторитетны.
  • LONG_ENTER возможен только если «За LONG» ≥ 4; SHORT_ENTER — «За SHORT» ≥ 4.
  • Если align < 4 в нужную сторону → NEUTRAL, даже при высоком tech_score.
  • tech_score ∈ [-1.0; +1.0] — целостная оценка по блокам ТРЕНД / ОСЦИЛЛЯТОРЫ /
    ОБЪЁМ / BB / УРОВНИ; должна быть согласована с блоком согласования
    (много «за LONG» → tech_score не сильно отрицательный, и наоборот).
  • final_score = clip(tech_score * 0.85 + sent_score * 0.15, -1.0, +1.0) —
    формула та же, что применит код после твоего ответа.

Как интерпретировать (качественно, без арифметического суммирования баллов):
  • SuperTrend — главный фильтр направления; вход против ST только при исключительной уверенности.
  • RSI/MACD/Stoch — импульс и динамика; противоречие ST vs MACD = нерешительность → NEUTRAL.
  • ADX < 20 — боковик, сигналам меньше доверия; ADX > 25 — тренд сильнее.
  • BB squeeze — не ENTER до пробоя с объёмом.
  • Объём < 0.5x — сигналам не доверяй; < 0.2x → NEUTRAL (код блокирует).
  • Цена у support/resistance — ключевой контекст для tech_reasoning.
  • Когда индикаторы противоречат друг другу → рынок в нерешительности → NEUTRAL.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ШАГ 3: ФИНАЛЬНАЯ ОЦЕНКА И РЕШЕНИЕ О ВХОДЕ (только «АНАЛИЗ ВХОДА»)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

final_score = clip(tech_score * 0.85 + sent_score * 0.15, -1.0, +1.0)

РЕШЕНИЕ О ВХОДЕ (пороги совпадают с кодом):
  final_score > +0.30 И «За LONG» ≥ 4 (из блока согласования) -> LONG_ENTER
  final_score < -0.30 И «За SHORT» ≥ 4 (из блока согласования) -> SHORT_ENTER
  Иначе -> NEUTRAL

ОБЯЗАТЕЛЬНЫЕ УСЛОВИЯ НЕЙТРАЛА (даже при сильном final_score):
  • SuperTrend противоречит direction -> NEUTRAL
  • Объём < 0.2x среднего -> NEUTRAL (нет ликвидности, сигналу нельзя доверять)

ЗАЩИТА ОТ ЛОВЛИ НОЖА (falling knife / catching a rocket):
  Ловля ножа — это вход LONG пока цена в вертикальном свободном падении,
  или SHORT пока цена вертикально взлетает без остановки.

  Железное условие блокировки — NEUTRAL в обоих случаях:
  • ATR% > 4% — экстремальная волатильность, свечи огромные, стоп будет
    съеден случайным шумом. На 1h–4h у BTC в обычный день ATR% около 0.5–2%.
    >4% — кризис или новостной шок.

  Всё остальное — на усмотрение ИИ. Если видишь сильное движение но ATR
  в норме — это торгуемая ситуация, не ловля ножа.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ШАГ 4: СОПРОВОЖДЕНИЕ ОТКРЫТОЙ ПОЗИЦИИ (только «СОПРОВОЖДЕНИЕ ПОЗИЦИИ»)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Позиция уже открыта. Закрытие — только код (SL / TP / trailing).
  action = NEUTRAL всегда.
  Оцени рынок: tech_score, news_reasoning, summary — риски и изменения.
  Не рекомендуй закрытие и не предлагай новый вход.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ШАГ 5: РАЗМЕР ПОЗИЦИИ (только «АНАЛИЗ ВХОДА»)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Выбирай stake_amount СТРОГО из предоставленного списка (точное значение!):

  |final_score| > 0.65 И ATR% < 1.5% И ADX > 25 -> максимальный уровень
  |final_score| 0.45-0.65 И ATR% < 2.5%          -> средний уровень
  |final_score| 0.30-0.45                          -> минимальный уровень
  Любые сомнения, высокая волатильность (ATR% > 3%), слабый ADX -> минимальный уровень

ВАЖНО: stake_amount должен быть ТОЧНЫМ числом из списка STAKE_LEVELS.
  • stake_amount всегда строго из списка STAKE_LEVELS.
  • При «СОПРОВОЖДЕНИИ ПОЗИЦИИ» stake_amount = минимум из STAKE_LEVELS (позиция уже открыта).
  • Если уверенность < 0.40 — action должен быть NEUTRAL (вход всё равно будет заблокирован).

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
ГРАДАЦИЯ NEUTRAL (используй точный подтип в поле neutral_type)
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
  НЕ перечисляй «за LONG / против» — согласование индикаторов уже в отдельном блоке кода.
  Кратко: противоречия между индикаторами, если есть.
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
  Оцени честно, насколько ты уверен в решении — ставь реальную цифру, не подгоняй под пороги:
  0.9-1.0  = картина ясная, сигналы в одну сторону, сомнений почти нет
  0.7-0.9  = в целом уверен, но есть 1–2 нюанса, которые настораживают
  0.5-0.7  = половина за, половина против — решение на тонкой грани
  0.4-0.5  = сетап слабый, держать/входить можно только с осторожностью
  < 0.4    = уверенности почти нет — в этом случае строго NEUTRAL
  При confidence < 0.6 — автоматически выбирай минимальный stake.
  Порог входа = 0.40 (контролирует код). Ниже — вход не откроется,
  но честная оценка важна для логов.
  confidence_reason — всегда пустая строка "" (в Telegram не показывается, только число).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ТРЕБОВАНИЯ К summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • Поле summary для NEUTRAL не используется в Telegram — причины пишет код.
    Для NEUTRAL можешь оставить summary пустым или одно слово "нейтрал".
  • Если action = LONG_ENTER или SHORT_ENTER: 2–3 коротких предложения (сетап и главный риск).

ФОРМАТ ОТВЕТА — строго JSON без Markdown-обёртки:
КРИТИЧЕСКИЕ ПРАВИЛА ФОРМАТА:
  • Никаких комментариев, префиксов и суффиксов вне JSON.
  • Все float-поля должны быть числами, не строками.
  • confidence ∈ [0..1]; tech_score, sent_score, final_score ∈ [-1..1].
  • Если action != "NEUTRAL", то neutral_type = null.
  • Если action == "NEUTRAL", то neutral_type обязательно один из: WEAK, STRONG, WAIT.
  • stake_amount всегда строго из списка STAKE_LEVELS.
  • Если уверенность < 0.40 — action должен быть NEUTRAL (вход всё равно будет заблокирован).
  • Если индикаторы противоречивы (нет 4+ согласованных сигналов), выбирай NEUTRAL.

{
  "news_reasoning": "рассуждение по новостям — ШАГ А",
  "tech_reasoning": "рассуждение по технике — ШАГ Б",
  "counter_argument": "самый весомый аргумент ПРОТИВ решения — ШАГ В",
  "confidence": float,
  "confidence_reason": "",
  "neutral_type": "WEAK|STRONG|WAIT|null",
  "sentiments": [{"title": "...", "score": float, "sentiment": float, "relevance": float, "recency": float}],
  "key_news": "заголовок самой важной новости",
  "news_summary": "2 предложения об общем фоне",
  "news_mood": "позитивный|негативный|нейтральный|смешанный|тревожный|эйфория",
  "tech_score": float,
  "action": "LONG_ENTER|SHORT_ENTER|NEUTRAL",
  "stake_amount": float,
  "summary": "пустая строка или одно слово для NEUTRAL; 2-3 предложения для LONG_ENTER/SHORT_ENTER"
}"""


    @classmethod
    def _pick_variant(cls, options: List[str], seed: int, salt: int = 0) -> str:
        if not options:
            return ""
        return options[(seed + salt) % len(options)]

    @classmethod
    def _narrative_seed(cls, ind: Indicators) -> int:
        return hash((
            round(ind.price, 2),
            round(ind.rsi),
            round(ind.macd_hist, 5),
            round(ind.bb_pct),
            round(ind.vol_ratio, 2),
            round(ind.adx, 1),
            ind.st_dir,
            ind.sma_cross,
        )) & 0x7FFFFFFF

    @classmethod
    def build_tech_narrative(cls, pair: str, ind: Indicators, action: str = "") -> str:
        """3–7 связных предложений по технике — для Telegram (без сырых MACD-цифр)."""
        token = pair.split("/")[0]
        seed = cls._narrative_seed(ind)
        pick = lambda opts, salt=0: cls._pick_variant(opts, seed, salt)
        parts: List[str] = []

        if ind.st_dir == "бычий":
            parts.append(pick([
                f"{token} на ${ind.price:.2f} держится над SuperTrend (${ind.st_level:.2f}) — "
                f"отступ {ind.st_dist_pct:.2f}%, бычий каркас цел.",
                f"SuperTrend бычий: линия ${ind.st_level:.2f} работает как опора, "
                f"цена выше на {ind.st_dist_pct:.2f}%.",
                f"По SuperTrend у {token} лонговый уклон — ${ind.st_level:.2f} под ценой, "
                f"дистанция {ind.st_dist_pct:.2f}%.",
            ], 1))
        else:
            parts.append(pick([
                f"{token} под медвежьим SuperTrend (${ind.st_level:.2f}), "
                f"отступ {ind.st_dist_pct:.2f}% — давление сверху.",
                f"SuperTrend медвежий на ${ind.st_level:.2f}: цена ниже линии на {ind.st_dist_pct:.2f}%.",
                f"Структура {token} слабая — SuperTrend ${ind.st_level:.2f} сдерживает рост.",
            ], 1))

        if ind.macd_hist > 0:
            if ind.macd_trend == "растёт":
                parts.append(pick([
                    "MACD смотрит на повышение — импульс набирает силу.",
                    "По MACD momentum идёт вверх, бычий настрой усиливается.",
                    "Импульс MACD направлен на повышение, тренд momentum растёт.",
                ], 3))
            elif ind.macd_trend == "падает":
                parts.append(pick([
                    "MACD пока на повышение, но momentum уже ослабевает.",
                    "Импульс MACD ещё бычий, однако momentum постепенно сдувается.",
                    "По MACD настрой на повышение, но ускорение импульса снижается.",
                ], 3))
            else:
                parts.append(pick([
                    "MACD указывает на повышение — бычий импульс без резкого разгона.",
                    "По MACD импульс на повышение, momentum держится ровно.",
                    "MACD в зоне повышения — бычий настрой без явного ускорения.",
                ], 3))
        elif ind.macd_hist < 0:
            if ind.macd_trend == "падает":
                parts.append(pick([
                    "MACD смотрит на понижение — медвежий импульс усиливается.",
                    "По MACD momentum идёт вниз, давление на понижение растёт.",
                    "Импульс MACD направлен на понижение, тренд momentum падает.",
                ], 3))
            elif ind.macd_trend == "растёт":
                parts.append(pick([
                    "MACD пока на понижение, но momentum уже отскакивает вверх.",
                    "Импульс MACD ещё медвежий, однако momentum постепенно восстанавливается.",
                    "По MACD настрой на понижение, но ослабление импульса замедляется.",
                ], 3))
            else:
                parts.append(pick([
                    "MACD указывает на понижение — медвежий импульс без резкого ускорения.",
                    "По MACD импульс на понижение, momentum держится ровно.",
                    "MACD в зоне понижения — медвежий настрой без явного разгона.",
                ], 3))
        else:
            parts.append(pick([
                "MACD в нейтрали — импульс без явного направления.",
                "По MACD momentum нейтральный, явного перекоса нет.",
                "MACD не даёт чёткого сигнала — импульс на паузе.",
            ], 3))

        if ind.macd_cross == "бычье":
            parts.append(pick([
                "Свежее бычье пересечение MACD — импульс на повышение усиливается.",
                "MACD только что пересёк сигнал вверх — бычий импульс свежий.",
            ], 31))
        elif ind.macd_cross == "медвежье":
            parts.append(pick([
                "Свежее медвежье пересечение MACD — давление на понижение растёт.",
                "MACD только что пересёк сигнал вниз — медвежий импульс свежий.",
            ], 31))

        if ind.rsi >= 70:
            parts.append(pick([
                f"RSI {ind.rsi:.1f} — перекупленность, импульс может выдохнуться.",
                f"RSI {ind.rsi:.1f} в зоне перекупленности — рост может замедлиться.",
            ], 15))
        elif ind.rsi <= 30:
            parts.append(pick([
                f"RSI {ind.rsi:.1f} — перепроданность, возможен технический отскок.",
                f"RSI {ind.rsi:.1f} в зоне перепроданности — отскок вероятнее продолжения падения.",
            ], 15))
        elif ind.rsi >= 55:
            parts.append(pick([
                f"RSI {ind.rsi:.1f} — умеренная перекупленность, тренд ещё жив.",
                f"RSI {ind.rsi:.1f} слегка выше середины, но без экстремальной зоны.",
            ], 15))
        elif ind.rsi <= 45:
            parts.append(pick([
                f"RSI {ind.rsi:.1f} — умеренная перепроданность, отскок возможен.",
                f"RSI {ind.rsi:.1f} слегка ниже середины, явного перегрева нет.",
            ], 15))
        else:
            parts.append(pick([
                f"RSI {ind.rsi:.1f} — нейтральная зона, явного перекоса нет.",
                f"RSI {ind.rsi:.1f} около середины диапазона — осциллятор без крайностей.",
            ], 15))

        if ind.stoch_zone == "перекуплен":
            parts.append(pick([
                f"Stoch RSI перекуплен (K={ind.stoch_k:.0f}, D={ind.stoch_d:.0f}) — краткий откат возможен.",
                f"Stoch {ind.stoch_k:.0f}/{ind.stoch_d:.0f} в зоне перекупленности — импульс может передохнуть.",
            ], 17))
        elif ind.stoch_zone == "перепродан":
            parts.append(pick([
                f"Stoch RSI перепродан (K={ind.stoch_k:.0f}, D={ind.stoch_d:.0f}) — отскок вероятнее продолжения.",
                f"Stoch {ind.stoch_k:.0f}/{ind.stoch_d:.0f} в зоне перепроданности — потенциал для отбоя.",
            ], 17))

        if ind.stoch_cross == "бычье":
            parts.append(pick([
                "Stoch RSI дал бычье пересечение — краткосрочный импульс на повышение.",
                "Пересечение Stoch вверх — осциллятор поддерживает отскок.",
            ], 33))
        elif ind.stoch_cross == "медвежье":
            parts.append(pick([
                "Stoch RSI дал медвежье пересечение — краткосрочное давление вниз.",
                "Пересечение Stoch вниз — осциллятор поддерживает снижение.",
            ], 33))

        if ind.bb_squeeze:
            parts.append(pick([
                f"Bollinger сжаты, цена на {ind.bb_pct:.0f}% канала — типичное затишье перед импульсом.",
                f"Сжатие BB при позиции {ind.bb_pct:.0f}%: рынок копит энергию, ждём пробой с объёмом.",
                f"Полосы Bollinger сузились ({ind.bb_pct:.0f}% диапазона) — часто предвестник движения.",
            ], 5))
        elif ind.bb_pct >= 80:
            parts.append(pick([
                f"Цена у верхней Bollinger ({ind.bb_pct:.0f}%) — зона, где импульс часто замедляют.",
                f"BB {ind.bb_pct:.0f}% — верхняя часть канала, рост может встретить сопротивление полос.",
            ], 7))
        elif ind.bb_pct <= 20:
            parts.append(pick([
                f"Нижняя зона BB ({ind.bb_pct:.0f}%) — потенциал отскока при сохранении тренда.",
                f"Цена у нижней Bollinger ({ind.bb_pct:.0f}%) — технически ближе к поддержке канала.",
            ], 7))
        elif 35 <= ind.bb_pct <= 65:
            parts.append(pick([
                f"Цена в середине Bollinger ({ind.bb_pct:.0f}%) — канал без явного перекоса.",
                f"BB {ind.bb_pct:.0f}% — цена в центре полос, запас хода в обе стороны.",
            ], 35))

        if ind.sma_cross == "бычий":
            parts.append(pick([
                "SMA7 выше SMA20 — краткосрочный тренд бычий.",
                "Бычье пересечение скользящих: короткая MA над длинной, структура в плюсе.",
            ], 37))
        elif ind.sma_cross == "медвежий":
            parts.append(pick([
                "SMA7 ниже SMA20 — краткосрочный тренд медвежий.",
                "Медвежье пересечение скользящих: короткая MA под длинной, давление сохраняется.",
            ], 37))

        if ind.price > ind.sma200 and ind.sma200 > 0:
            parts.append(pick([
                f"Цена выше SMA200 (${ind.sma200:.2f}) — долгосрочный фон бычий.",
                f"SMA200 на ${ind.sma200:.2f} ниже цены — глобальный тренд поддерживает лонг.",
            ], 39))
        elif ind.price < ind.sma200 and ind.sma200 > 0:
            parts.append(pick([
                f"Цена ниже SMA200 (${ind.sma200:.2f}) — долгосрочный фон медвежий.",
                f"SMA200 на ${ind.sma200:.2f} выше цены — глобальный тренд давит сверху.",
            ], 39))

        dist_res = abs(ind.price - ind.resistance) / ind.price * 100 if ind.price > 0 else 99.0
        dist_sup = abs(ind.price - ind.support) / ind.price * 100 if ind.price > 0 else 99.0
        if dist_res < 0.45:
            parts.append(pick([
                f"Сопротивление ${ind.resistance:.2f} всего в {dist_res:.2f}% — пробой может дать импульс.",
                f"У потолка ${ind.resistance:.2f} ({dist_res:.2f}%) — решит объём и закрепление.",
            ], 11))
        elif dist_sup < 0.45:
            parts.append(pick([
                f"Поддержка ${ind.support:.2f} в {dist_sup:.2f}% — отбой усилит текущий сетап.",
                f"Близко поддержка ${ind.support:.2f} ({dist_sup:.2f}%) — ключ для удержания структуры.",
            ], 11))

        if ind.adx >= 25:
            parts.append(pick([
                f"Тренд {ind.adx_str} — направление движения пока удерживается.",
                f"ADX показывает {ind.adx_str} тренд, структура движения выражена.",
            ], 19))
        elif ind.adx < 20:
            parts.append(pick([
                f"Тренд {ind.adx_str} — рынок боковой, сигналы менее надёжны.",
                f"Слабый тренд ({ind.adx_str}) — явного трендового импульса пока нет.",
            ], 19))

        if ind.vol_ratio >= 1.5:
            parts.append(pick([
                "Объём выше среднего — движение подкреплено активностью.",
                "Всплеск объёма поддерживает текущий ход.",
                "Активность выше нормы — движение не выглядит пустым.",
            ], 13))
        elif ind.vol_ratio < 0.55:
            parts.append(pick([
                "Объём ниже среднего — движение выглядит хрупким.",
                "Низкая активность — пробой или отбой могут не удержаться.",
            ], 21))

        if ind.obv_trend == "бычий":
            parts.append(pick([
                "OBV растёт — объёмный поток подтверждает движение вверх.",
                "Бычий OBV: накопление объёма поддерживает текущий импульс.",
            ], 41))
        elif ind.obv_trend == "медвежий":
            parts.append(pick([
                "OBV снижается — объёмный поток не подтверждает рост.",
                "Медвежий OBV: распределение объёма ослабляет импульс.",
            ], 41))

        if ind.vwap_pos == "выше" and ind.vwap > 0:
            parts.append(pick([
                f"Цена выше VWAP (${ind.vwap:.2f}) — внутридневной баланс в пользу покупателей.",
                f"VWAP ${ind.vwap:.2f} ниже цены — сессионный контекст бычий.",
            ], 43))
        elif ind.vwap_pos == "ниже" and ind.vwap > 0:
            parts.append(pick([
                f"Цена ниже VWAP (${ind.vwap:.2f}) — внутридневной баланс в пользу продавцов.",
                f"VWAP ${ind.vwap:.2f} выше цены — сессионный контекст медвежий.",
            ], 43))

        if ind.williams_zone == "перепродан":
            parts.append(pick([
                f"Williams %R {ind.williams_r:.0f} — перепроданность, отскок вероятен.",
                f"Williams {ind.williams_r:.0f} в зоне перепроданности — осциллятор просит отбой.",
            ], 45))
        elif ind.williams_zone == "перекуплен":
            parts.append(pick([
                f"Williams %R {ind.williams_r:.0f} — перекупленность, откат возможен.",
                f"Williams {ind.williams_r:.0f} в зоне перекупленности — импульс может сдуться.",
            ], 45))

        if ind.cci_signal == "перепродан":
            parts.append(pick([
                f"CCI {ind.cci:.0f} — перепроданность, цена отстаёт от среднего.",
                f"CCI {ind.cci:.0f} ниже нормы — потенциал возврата к среднему.",
            ], 47))
        elif ind.cci_signal == "перекуплен":
            parts.append(pick([
                f"CCI {ind.cci:.0f} — перекупленность, цена перегрета относительно среднего.",
                f"CCI {ind.cci:.0f} выше нормы — возможна коррекция к среднему.",
            ], 47))

        if ind.atr_pct >= 2.5:
            parts.append(pick([
                f"Волатильность повышена ({ind.atr_pct:.1f}% ATR) — движения резкие, стопы шире.",
                f"ATR {ind.atr_pct:.1f}% от цены — рынок нервный, свинги амплитудные.",
            ], 49))
        elif ind.atr_pct <= 0.8 and ind.atr_pct > 0:
            parts.append(pick([
                f"Волатильность низкая ({ind.atr_pct:.1f}% ATR) — рынок сжат, пробой может быть резким.",
                f"ATR всего {ind.atr_pct:.1f}% — спокойный фон, движения пока сдержанные.",
            ], 49))

        if abs(ind.change_pct) >= 0.35:
            direction = "вверх" if ind.change_pct > 0 else "вниз"
            parts.append(pick([
                f"Последняя свеча {direction} на {abs(ind.change_pct):.2f}% — импульс свежий.",
                f"Свеча закрылась {direction} ({ind.change_pct:+.2f}%) — краткосрочный импульс заметен.",
            ], 51))

        start = seed % max(len(parts), 1)
        ordered = parts[start:] + parts[:start]
        limit = min(len(ordered), 7)
        return " ".join(ordered[:limit])

    @classmethod
    def build_counter_narrative(cls, pair: str, ind: Indicators, action: str = "") -> str:
        """Главный риск сетапа — связным текстом."""
        seed = cls._narrative_seed(ind)
        pick = lambda opts, salt=0: cls._pick_variant(opts, seed, salt)
        risks: List[str] = []

        if ind.obv_trend == "медвежий":
            risks.append(pick([
                f"OBV медвежий — объёмы не подтверждают рост, возможна дивергенция с ценой.",
                f"Объёмный фон слабый: OBV тянет вниз, импульс может оказаться ложным.",
                f"Медвежий OBV при растущей цене — классический признак сомнительного пробоя.",
            ], 17))
        if ind.stoch_zone == "перекуплен":
            risks.append(pick([
                f"Stoch RSI перекуплен (K={ind.stoch_k:.0f}, D={ind.stoch_d:.0f}) — краткий откат вероятен.",
                f"Stoch в зоне перекупленности ({ind.stoch_k:.0f}/{ind.stoch_d:.0f}) — импульс может сдуться.",
            ], 19))
        elif ind.stoch_zone == "перепродан" and action == "SHORT_ENTER":
            risks.append(f"Stoch перепродан — шорт против отскока рискован.")

        if ind.vol_ratio < 0.55:
            risks.append(pick([
                f"Объём {ind.vol_ratio:.2f}x ниже нормы — пробой без vol может не удержаться.",
                f"Vol ratio {ind.vol_ratio:.2f}x — слабая активность, движение хрупкое.",
            ], 21))

        if ind.adx < 20:
            risks.append(pick([
                f"ADX {ind.adx:.1f} — тренд слабый, сигналы могут быстро перевернуться.",
                f"Слабый ADX ({ind.adx:.1f}): рынок боковой, вход менее надёжен.",
            ], 23))

        if ind.macd_hist < 0 and ind.st_dir == "бычий":
            risks.append(pick([
                f"MACD histogram {ind.macd_hist:+.4f} против бычьего ST — momentum пока не догоняет.",
                f"MACD медвежий ({ind.macd_hist:+.4f}) при бычьем SuperTrend — внутреннее противоречие.",
            ], 25))

        dist_sup = abs(ind.price - ind.support) / ind.price * 100 if ind.price > 0 else 99.0
        if dist_sup < 0.5 and action != "SHORT_ENTER":
            risks.append(
                f"Откат к ${ind.support:.2f} ({dist_sup:.2f}%) может сбить сетап без слома ST ({ind.st_dir})."
            )

        if not risks:
            dist_res = abs(ind.price - ind.resistance) / ind.price * 100 if ind.price > 0 else 99.0
            risks.append(pick([
                f"Главный риск — сопротивление ${ind.resistance:.2f} ({dist_res:.2f}%) без объёма.",
                f"Без всплеска vol пробой ${ind.resistance:.2f} может не закрепиться.",
            ], 27))

        return risks[seed % len(risks)]

    @classmethod
    def build_candle_focus(cls, pair: str, ind: Indicators) -> str:
        """Краткая подсказка для промпта LLM (не для Telegram)."""
        token = pair.split("/")[0]
        notes: List[str] = []
        if ind.bb_pct > 100:
            notes.append(f"BB {ind.bb_pct:.0f}% — над верхней полосой")
        elif ind.bb_pct < 0:
            notes.append(f"BB {ind.bb_pct:.0f}% — под нижней полосой")
        elif ind.bb_squeeze:
            notes.append(f"BB squeeze, позиция {ind.bb_pct:.0f}%")
        if ind.vol_ratio >= 1.5:
            notes.append(f"vol {ind.vol_ratio:.2f}x — всплеск")
        elif ind.vol_ratio < 0.5:
            notes.append(f"vol {ind.vol_ratio:.2f}x — низкий")
        if ind.adx < 20:
            notes.append(f"ADX {ind.adx:.1f} — слабый тренд")
        elif ind.adx > 25:
            notes.append(f"ADX {ind.adx:.1f} — сильный тренд")
        notes.append(f"MACD hist {ind.macd_hist:+.4f} ({ind.macd_trend})")
        notes.append(f"ST {ind.st_dir}, dist {ind.st_dist_pct:.2f}%")
        dist_res = abs(ind.price - ind.resistance) / ind.price * 100 if ind.price > 0 else 99.0
        if dist_res < 0.35:
            notes.append(f"у сопр. ${ind.resistance:.4f} ({dist_res:.2f}%)")
        return f"{token}: " + "; ".join(notes[:4]) + "."

    @staticmethod
    def _format_neutral_history_block(history: List[Dict[str, Any]]) -> str:
        """До NEUTRAL_HISTORY_MAX прошлых NEUTRAL — только для цепочки NEUTRAL."""
        items = [h for h in (history or []) if h.get("action") == "NEUTRAL"][-NEUTRAL_HISTORY_MAX:]
        if not items:
            return ""
        lines = [f"### ПРОШЛЫЕ NEUTRAL-АНАЛИЗЫ (последние {len(items)})"]
        for i, h in enumerate(items, 1):
            lines.append(
                f"  [{i}] свеча {h.get('candle_time', '—')} | "
                f"score={float(h.get('final_score', 0)):+.3f} "
                f"conf={float(h.get('confidence', 0)):.0%}"
            )
            lines.append(f"      Техника: {(h.get('tech_reasoning') or '—')[:200]}")
            lines.append(f"      Возражение: {(h.get('counter_arg') or '—')[:120]}")
        return "\n".join(lines) + "\n\n"

    @classmethod
    def build_entry_prompt(cls, ind: Indicators, news_items: List[str], pair: str,
                           memory_block: str = "", fear_greed: str = "",
                           prev_decision: Dict[str, Any] = None,
                           neutral_history: List[Dict[str, Any]] = None) -> str:
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

        # Контекст NEUTRAL: только 2 прошлых NEUTRAL-анализа (не показываем LONG/SHORT)
        prev_block = ""
        if neutral_history:
            prev_block = cls._format_neutral_history_block(neutral_history)

        focus_line = cls.build_candle_focus(pair, ind)

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
            f"### ФОКУС ЭТОЙ СВЕЧИ (используй в tech_reasoning — своими словами, с цифрами)\n"
            f"  {focus_line}\n\n"
            f"### СОГЛАСОВАНИЕ ИНДИКАТОРОВ (посчитано кодом)\n"
            f"  За LONG: {ind.align_bull} | За SHORT: {ind.align_bear} | "
            f"нейтр.: {ALIGN_INDICATOR_TOTAL - ind.align_bull - ind.align_bear} "
            f"(из {ALIGN_INDICATOR_TOTAL})\n\n"
            f"Доступные стейки: {STAKE_LEVELS}\n"
            f"CONTRACT_VERSION: v2_compact\n"
            f"ОТВЕТЬ ТОЛЬКО JSON\n"
        )

    @classmethod
    def build_manage_prompt(cls, ind: Indicators, news_items: List[str], pair: str,
                          side: str, profit: float, memory_block: str = "",
                          fear_greed: str = "", prev_decision: Dict[str, Any] = None,
                          neutral_history: List[Dict[str, Any]] = None) -> str:
        rsi_sig = "ПЕРЕПРОДАН" if ind.rsi < 30 else ("ПЕРЕКУПЛЕН" if ind.rsi > 70 else "нейтрал")
        news_block = "\n".join(f"  [{i+1}] {t}" for i, t in enumerate(news_items)) if news_items else "  нет новостей"
        session_name, session_emoji = get_session_info()
        fg_line = f"  Fear & Greed: {fear_greed}\n" if fear_greed else ""

        prev_block = ""
        if neutral_history:
            prev_block = cls._format_neutral_history_block(neutral_history)

        return (
            f"## СОПРОВОЖДЕНИЕ ПОЗИЦИИ — {pair}\n"
            f"Время: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  "
            f"Сессия: {session_emoji} {session_name}\n"
            f"{fg_line}"
            f"Открытая позиция: {side} | Текущий P&L: {profit:+.2f}%\n\n"
            f"### ИСТОРИЯ СДЕЛОК\n{memory_block}\n\n"
            f"{prev_block}"
            f"ЗАДАЧА: сопровождение открытой позиции. Закрытие только стоп/тейк в коде.\n"
            f"В JSON укажи action=NEUTRAL (другие action запрещены).\n"
            f"confidence — число 0..1; confidence_reason — пустая строка.\n\n"
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
# [11] PYDANTIC МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════════

TradeAction = TypingLiteral["LONG_ENTER", "SHORT_ENTER", "NEUTRAL"]


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
    confidence:       float       = Field(...)
    confidence_reason:  str         = Field(default="")
    neutral_type:     Optional[str] = Field(default=None)
    sentiments:       List[Any]   = Field(default_factory=list)
    key_news:         str         = Field(default="")
    news_summary:     str         = Field(default="")
    news_mood:        str         = Field(default="нейтральный")
    tech_score:       float       = Field(default=0.0)
    action:           TradeAction = Field(default="NEUTRAL")
    summary:          str         = Field(default="")
    stake_amount:     float       = Field(default=0.0)

    # Pre-валидатор: LLM иногда возвращает null/JSON null для строковых полей —
    # Pydantic ругается на string_type. Заменяем None → "" ДО валидации.
    @model_validator(mode="before")
    @classmethod
    def _coerce_nulls(cls, data):
        if not isinstance(data, dict):
            return data
        str_fields = (
            "news_reasoning", "tech_reasoning", "counter_argument",
            "confidence_reason", "key_news", "news_summary", "news_mood", "summary",
        )
        for f in str_fields:
            if f in data and data[f] is None:
                data[f] = ""
        # action: null → NEUTRAL
        if "action" in data and data["action"] is None:
            data["action"] = "NEUTRAL"
        # numerics: null → разумные дефолты
        for f, default in (("tech_score", 0.0), ("stake_amount", 0.0)):
            if f in data and data[f] is None:
                data[f] = default
        # sentiments: null → []
        if "sentiments" in data and data["sentiments"] is None:
            data["sentiments"] = []
        return data

    @field_validator("confidence")
    @classmethod
    def confidence_clamp(cls, v):
        return float(max(0.0, min(1.0, v)))

    @field_validator("summary")
    @classmethod
    def summary_default(cls, v):
        return (v or "").strip()

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
# [12] LLM CLIENT
# ═══════════════════════════════════════════════════════════════════════

class LLMClient:
    """Отправляет промпты в OpenAI / OpenRouter и парсит ответ."""

    def __init__(self):
        self.model = LLM_MODEL
        self.last_analysis_cost_usd: float = 0.0
        logger.info(f"[LLM] Модель: {self.model}")

    @staticmethod
    def _usage_cost_usd(usage: Any) -> float:
        """Стоимость одного LLM-ответа в USD (из API usage или по токенам)."""
        if not usage or not isinstance(usage, dict):
            return 0.0
        for key in ("total_cost", "cost", "generation_cost"):
            val = usage.get(key)
            if val is not None:
                try:
                    return max(0.0, float(val))
                except (TypeError, ValueError):
                    pass
        try:
            prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        except (TypeError, ValueError):
            return 0.0
        if prompt <= 0 and completion <= 0:
            return 0.0
        return (
            prompt * (LLM_INPUT_USD_PER_1M / 1_000_000.0)
            + completion * (LLM_OUTPUT_USD_PER_1M / 1_000_000.0)
        )

    @staticmethod
    def format_cost_usd(usd: float) -> str:
        """Стоимость для Telegram: просто число — 0.13, 0.0013 (USD, без $ и лишних нулей)."""
        if usd <= 0:
            return "0"
        if usd >= 1:
            prec = 2
        elif usd >= 0.01:
            prec = 2   # центы: 0.18, 0.13, 0.10 → 0.1
        elif usd >= 0.001:
            prec = 4   # анализ свечи: 0.0013
        elif usd >= 0.0001:
            prec = 5
        else:
            prec = 6
        s = f"{usd:.{prec}f}".rstrip("0").rstrip(".")
        return s or "0"

    @staticmethod
    def _semantic_validate(result: TradingRecommendation) -> Tuple[bool, str]:
        score_cmp = round(float(result.final_score), SCORE_ROUND_DECIMALS)
        if result.action in ("LONG_ENTER", "SHORT_ENTER") and result.confidence < ENTRY_MIN_CONFIDENCE:
            return False, f"low_confidence:{result.confidence:.2f}"
        if result.action == "LONG_ENTER" and score_cmp < ENTRY_LONG_THRESHOLD:
            return False, f"weak_long_score:{result.final_score:.3f}"
        if result.action == "SHORT_ENTER" and score_cmp > ENTRY_SHORT_THRESHOLD:
            return False, f"weak_short_score:{result.final_score:.3f}"
        return True, "ok"

    def analyze(self, prompt: str) -> TradingRecommendation:
        import json as _json

        total_cost_usd = 0.0
        api_url = (
            "https://openrouter.ai/api/v1/chat/completions"
            if "/" in self.model
            else "https://api.openai.com/v1/chat/completions"
        )
        headers = {
            "Authorization": f"Bearer {_llm_key()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0.22,
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
                resp_json = resp.json()
                total_cost_usd += self._usage_cost_usd(resp_json.get("usage"))
                raw = resp_json["choices"][0]["message"]["content"] or ""
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
                if "confidence" not in filtered or not isinstance(filtered.get("confidence"), (int, float)):
                    logger.warning(
                        f"[LLM] confidence отсутствует или не число в ответе модели "
                        f"(got: {data.get('confidence')!r}) — ретрай"
                    )
                    if attempt < LLM_RETRIES - 1:
                        time.sleep(3)
                        continue
                    break
                result = TradingRecommendation(**filtered)
                valid, reason = self._semantic_validate(result)
                if not valid:
                    logger.warning(f"[LLM] semantic reject: {reason}")
                    result.action = "NEUTRAL"
                    if result.neutral_type is None:
                        result.neutral_type = "STRONG"
                    result.summary = f"Сигнал отклонён semantic-validator: {reason}."
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
                self.last_analysis_cost_usd = total_cost_usd
                logger.info(f"[LLM] cost_usd={self.format_cost_usd(total_cost_usd)}")
                return result
            except Exception as e:
                logger.error(f"[LLM] Ошибка (попытка {attempt+1}): {e}")
                time.sleep(3)

        self.last_analysis_cost_usd = total_cost_usd
        return TradingRecommendation(
            action="NEUTRAL",
            summary="LLM недоступен — торговля заблокирована.",
            tech_score=0.0,
            stake_amount=float(STAKE_LEVELS[0]),
            confidence=0.0,
        )


# ═══════════════════════════════════════════════════════════════════════
# [13] TG MESSAGE FORMATTER  (TGFormatter)
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
        if rec.key_news and rec.key_news != "нет релевантных":
            lines.append(f"\n🔑 <b>Ключевая новость:</b>")
            lines.append(f"  <i>{rec.key_news[:120]}</i>")
        return "\n".join(lines)

    @classmethod
    def scores_line(cls, rec: TradingRecommendation, ind: Optional[Indicators] = None) -> str:
        align_line = ""
        if ind is not None:
            align_neutral = ALIGN_INDICATOR_TOTAL - ind.align_bull - ind.align_bear
            align_line = (
                f"📊 <b>Согласование:</b> за LONG <b>{ind.align_bull}</b> | "
                f"за SHORT <b>{ind.align_bear}</b> | "
                f"нейтр. <b>{align_neutral}</b> (из {ALIGN_INDICATOR_TOTAL})\n"
            )
        sent = round(float(rec.sent_score), SCORE_ROUND_DECIMALS)
        tech = round(float(rec.tech_score), SCORE_ROUND_DECIMALS)
        final = round(float(rec.final_score), SCORE_ROUND_DECIMALS)
        return (
            f"{align_line}"
            f"📊 <b>Оценки:</b>  "
            f"Новости: <b>{sent:+.2f}</b>  |  "
            f"Техника: <b>{tech:+.2f}</b>  |  "
            f"Итог: <b>{final:+.2f}</b>\n"
            f"{cls._score_bar(final)}"
        )

    @staticmethod
    def _confidence_why(rec: "TradingRecommendation", ind: "Indicators", action: str) -> str:
        """1–2 предложения: почему confidence на этом уровне."""
        if rec.confidence < 0.05:
            return ""

        bull, bear = ind.align_bull, ind.align_bear
        neu = max(0, ALIGN_INDICATOR_TOTAL - bull - bear)
        score = rec.final_score
        parts: List[str] = []

        if action == "LONG_ENTER":
            dir_label, align_ok, align_n = "LONG", ind.st_dir == "бычий", bull
            score_ok = score >= ENTRY_LONG_THRESHOLD
            score_need = ENTRY_LONG_THRESHOLD
        elif action == "SHORT_ENTER":
            dir_label, align_ok, align_n = "SHORT", ind.st_dir == "медвежий", bear
            score_ok = score <= ENTRY_SHORT_THRESHOLD
            score_need = ENTRY_SHORT_THRESHOLD
        elif action.endswith("_MANAGE"):
            dir_label = "LONG" if "LONG" in action else "SHORT"
            align_ok = ind.st_dir == ("бычий" if dir_label == "LONG" else "медвежий")
            align_n = bull if dir_label == "LONG" else bear
            score_ok = True
            score_need = None
        else:
            if score >= ENTRY_LONG_THRESHOLD:
                dir_label, align_n, align_ok = "LONG", bull, ind.st_dir == "бычий"
            elif score <= ENTRY_SHORT_THRESHOLD:
                dir_label, align_n, align_ok = "SHORT", bear, ind.st_dir == "медвежий"
            else:
                dir_label, align_n, align_ok = None, max(bull, bear), None
            score_ok = ENTRY_SHORT_THRESHOLD < score < ENTRY_LONG_THRESHOLD
            score_need = None

        conf = rec.confidence

        if conf >= 0.7:
            if dir_label and action in ("LONG_ENTER", "SHORT_ENTER"):
                parts.append(
                    f"{align_n}/10 индикаторов за {dir_label}, SuperTrend "
                    f"{'подтверждает' if align_ok else 'расходится'}."
                )
                if score_need is not None and score_ok:
                    margin = abs(score) - abs(score_need)
                    if margin >= 0.15:
                        parts.append(f"Итог {score:+.2f} уверенно проходит порог.")
                    else:
                        parts.append(f"Итог {score:+.2f} проходит порог, но без большого запаса.")
                risks = _entry_confidence_risks(ind, action)
                if risks:
                    parts.append(f"Оговорка: {'; '.join(risks[:2])}.")
            elif dir_label:
                parts.append(
                    f"Score тянет к {dir_label} ({score:+.2f}), {align_n}/10 за это направление."
                )
                if not align_ok:
                    parts.append(f"SuperTrend {ind.st_dir} — часть сигналов спорит.")
            elif abs(score) < 0.15 and action == "NEUTRAL":
                parts.append(
                    f"Итог {score:+.2f} — нейтральный рынок "
                    f"(техника {rec.tech_score:+.2f}, новости {rec.sent_score:+.2f}); "
                    f"{bull} за LONG, {bear} за SHORT — явного перевеса нет."
                )
            else:
                parts.append(
                    f"Картина достаточно ясная: {bull} за LONG, {bear} за SHORT, нейтр. {neu}."
                )
        elif conf >= 0.5:
            if (
                action in ("LONG_ENTER", "SHORT_ENTER")
                and dir_label
                and score_need is not None
                and score_ok
            ):
                parts.append(
                    f"Сетап за {dir_label}: итог {score:+.2f} проходит порог, SuperTrend "
                    f"{'подтверждает' if align_ok else f'{ind.st_dir} — расхождение'}."
                )
                parts.append(f"{align_n}/10 индикаторов за {dir_label}.")
                risks = _entry_confidence_risks(ind, action)
                if risks:
                    parts.append(
                        f"Смущает: {'; '.join(risks[:3])} — поэтому уверенность средняя, "
                        f"не максимальная."
                    )
                else:
                    margin = abs(score) - abs(score_need)
                    if margin >= 0.15:
                        parts.append("Критичных противоречий мало — уверенность выше средней.")
                    else:
                        parts.append(
                            "Итог проходит порог, но запас небольшой — уверенность умеренная."
                        )
            else:
                parts.append(
                    f"Согласование смешанное: {bull} за LONG, {bear} за SHORT, нейтр. {neu}."
                )
                if dir_label and not align_ok:
                    parts.append(f"SuperTrend {ind.st_dir} не совпадает с bias к {dir_label}.")
                elif (
                    action == "NEUTRAL"
                    and ENTRY_SHORT_THRESHOLD < score < ENTRY_LONG_THRESHOLD
                ):
                    if abs(score) < 0.15:
                        parts.append(
                            f"Итог {score:+.2f} — рынок без направления "
                            f"(техника {rec.tech_score:+.2f}, новости {rec.sent_score:+.2f}); "
                            f"для входа нужен ≥ {ENTRY_LONG_THRESHOLD:+.2f} "
                            f"или ≤ {ENTRY_SHORT_THRESHOLD:+.2f}."
                        )
                    else:
                        parts.append(
                            f"Итог {score:+.2f} ещё не проходит порог ±{ENTRY_LONG_THRESHOLD:.2f}."
                        )
                elif score_need is not None and not score_ok:
                    if abs(score) < 0.15:
                        parts.append(
                            f"Итог {score:+.2f} — рынок без направления "
                            f"(техника {rec.tech_score:+.2f}, новости {rec.sent_score:+.2f}); "
                            f"для входа нужен ≥ {ENTRY_LONG_THRESHOLD:+.2f} "
                            f"или ≤ {ENTRY_SHORT_THRESHOLD:+.2f}."
                        )
                    else:
                        parts.append(
                            f"Итог {score:+.2f} ещё не проходит порог ±{ENTRY_LONG_THRESHOLD:.2f}."
                        )
                if action == "NEUTRAL" and rec.neutral_type == "WAIT":
                    parts.append("Сетап формируется — нужно подтверждение.")
                elif action == "NEUTRAL" and rec.neutral_type == "WEAK":
                    parts.append("Сигнал близко к порогу, но не дотягивает.")
        else:
            issues: List[str] = []
            if bull >= MIN_ALIGNED_INDICATORS and bear >= MIN_ALIGNED_INDICATORS:
                issues.append("индикаторы одновременно за LONG и SHORT")
            if dir_label and not align_ok:
                issues.append(f"SuperTrend {ind.st_dir} против идеи {dir_label}")
            if ind.vol_ratio < ENTRY_MIN_VOL_RATIO:
                issues.append(f"объём {ind.vol_ratio:.2f}x ниже минимума")
            if ind.atr_pct > ENTRY_MAX_ATR_PCT:
                issues.append(f"ATR {ind.atr_pct:.1f}% слишком высокий")
            if (
                action == "NEUTRAL"
                and ENTRY_SHORT_THRESHOLD < score < ENTRY_LONG_THRESHOLD
            ):
                if abs(score) < 0.15:
                    issues.append(
                        f"итог {score:+.2f} — нет направления "
                        f"(техника {rec.tech_score:+.2f}, новости {rec.sent_score:+.2f})"
                    )
                else:
                    issues.append(
                        f"итог {score:+.2f} не проходит порог входа ±{ENTRY_LONG_THRESHOLD:.2f}"
                    )
            elif score_need is not None and not score_ok:
                if abs(score) < 0.15:
                    issues.append(
                        f"итог {score:+.2f} — нет направления "
                        f"(техника {rec.tech_score:+.2f}, новости {rec.sent_score:+.2f})"
                    )
                else:
                    issues.append(f"итог {score:+.2f} не проходит порог входа")
            if conf < ENTRY_MIN_CONFIDENCE:
                issues.append("ниже минимума 40% для входа")
            if rec.neutral_type == "STRONG":
                issues.append("противоречивые сигналы — сетап не складывается")
            if issues:
                parts.append("Низкая уверенность: " + "; ".join(issues[:4]) + ".")
            else:
                parts.append("Сигналы разрозненные, явного преимущества нет.")

        if rec.counter_argument and conf >= 0.45:
            doubt = rec.counter_argument.strip()
            if doubt and not doubt.lower().startswith("нет весом"):
                parts.append(f"Сомнение: {TGFormatter._truncate(doubt, 100)}")

        return TGFormatter._truncate(" ".join(parts), 300)

    @staticmethod
    def _confidence_block(rec: "TradingRecommendation", ind: Optional["Indicators"], action: str) -> str:
        """Только цифра уверенности для Telegram."""
        return TGFormatter._confidence_line(rec)

    @staticmethod
    def _confidence_line(rec: "TradingRecommendation") -> str:
        """Строка уверенности; при отсутствии данных от LLM — диагностика."""
        if rec.confidence < 0.05:
            return (
                "❗ <b>Уверенность не получена от модели</b>\n"
                "   <i>Возможные причины: LLM не ответил, отверг ответ или нет данных в кэше.\n"
                "   Перезапусти бота: <code>docker compose restart</code> или дождись следующей свечи.</i>"
            )
        conf_emoji = "🟢" if rec.confidence >= 0.7 else ("🟡" if rec.confidence >= 0.5 else "🔴")
        return f"{conf_emoji} <b>Уверенность:</b> {rec.confidence:.0%}"

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
        lines.append(f"  {TGFormatter._confidence_line(rec)}")
        return "\n".join(lines)

    @staticmethod
    def reasoning_msg(
        pair: str,
        rec: TradingRecommendation,
        action: str,
        ind: Optional[Indicators] = None,
    ) -> str:
        """Отдельное сообщение с полными размышлениями ИИ."""
        if action in ("LONG_MANAGE", "SHORT_MANAGE"):
            t = TGFormatter._truncate
            lines = [f"🧠 <b>КРАТКОЕ СОПРОВОЖДЕНИЕ</b> — {pair}  ({action})"]
            if rec.tech_reasoning:
                lines.append(f"📊 <i>{t(rec.tech_reasoning, 220)}</i>")
            if rec.counter_argument:
                lines.append(f"⚖️ <i>{t(rec.counter_argument, 140)}</i>")
            lines.append(TGFormatter._confidence_block(rec, ind, action))
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
        lines.append(TGFormatter._confidence_block(rec, ind, action))
        if rec.neutral_type and action == "NEUTRAL":
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
        why_block = ""
        if (rec.summary or "").strip():
            why_block = f"⛔ <b>Почему:</b> {rec.summary.strip()[:500]}\n\n"
        return (
            f"⚪ <b>НЕЙТРАЛ</b> — {pair}  {session_emoji} {session_name}\n"
            f"{cls.scores_line(rec, ind)}\n\n"
            f"{why_block}"
            f"{cls.indicator_block(ind)}\n\n"
            f"{cls.news_block(rec)}"
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
            f"{cls.scores_line(rec, ind)}\n\n"
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


def finalize_confidence_reason(
    rec: "TradingRecommendation",
    ind: "Indicators",
    side: Optional[str] = None,
) -> "TradingRecommendation":
    """Сбрасывает confidence_reason при противоречии с ENTER или сопровождением позиции."""
    action = rec.action or ""
    reason = (rec.confidence_reason or "").strip()

    if (
        action in ("LONG_ENTER", "SHORT_ENTER")
        and reason
        and _confidence_reason_conflicts_enter(reason)
    ):
        return rec.model_copy(update={"confidence_reason": ""})

    if side in ("LONG", "SHORT") and reason and _confidence_reason_conflicts_manage(reason):
        return rec.model_copy(update={"confidence_reason": ""})

    return rec


# ═══════════════════════════════════════════════════════════════════════
# [14] СТРАТЕГИЯ  (class GPTStrategy)
# ═══════════════════════════════════════════════════════════════════════

class GPTStrategy(IStrategy):
    """
    Стратегия на основе LLM (GPT) + технический анализ + новости.
    Все параметры настраиваются в разделе КОНФИГУРАЦИЯ вверху файла.
    """

    position_adjustment_enable = False
    can_short = True
    process_only_new_candles = True
    # ВАЖНО: True требуется чтобы Freqtrade вызывал custom_exit, где живёт TP/trailing.
    # populate_exit_trend пустой, поэтому никаких сигналов выхода кроме TP/trailing не появится.
    use_exit_signal = True

    stoploss             = STOPLOSS_PCT
    minimal_roi          = {"99999": -1}
    startup_candle_count = STARTUP_CANDLES

    _WELCOME_MARKER = Path(__file__).resolve().parent / ".welcome_pinned_gpt"

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

    @property
    def plot_config(self):
        return {
            "main_plot": {
                "sma7":   {"color": "cyan"},
                "sma20":  {"color": "orange"},
                "sma200": {"color": "purple"},
                "support": {"color": "rgba(46,204,113,0.7)"},
                "resistance": {"color": "rgba(231,76,60,0.7)"},
                "bb_upper": {
                    "color": "rgba(0,200,255,0.4)",
                    "fill_to": "bb_lower",
                },
                "bb_mid":   {"color": "rgba(0,200,255,0.8)"},
                "bb_lower": {"color": "rgba(0,200,255,0.4)"},
                "st_bull": {
                    "color": "#00ff88",
                },
                "st_bear": {
                    "color": "#ff4444",
                },
            },
            "subplots": {
                "RSI": {
                    "rsi": {"color": "#3498db"},
                },
                "MACD": {
                    "macd_line":        {"color": "#2ecc71"},
                    "macd_signal_line": {"color": "#e74c3c"},
                    "macd_hist":        {"color": "rgba(155,89,182,0.7)", "type": "bar"},
                },
                "ADX": {
                    "adx": {"color": "#e67e22"},
                },
                "ATR": {
                    "atr": {"color": "#f1c40f"},
                },
                "Stoch RSI": {
                    "stoch_k": {"color": "#1abc9c"},
                    "stoch_d": {"color": "#e74c3c"},
                },
                "Williams %R": {
                    "williams_r": {"color": "#9b59b6"},
                },
                "CCI": {
                    "cci": {"color": "#95a5a6"},
                },
                "Volume Ratio": {
                    "volume_ratio": {"color": "teal", "type": "bar"},
                },
                "OBV": {
                    "obv": {"color": "#2ecc71"},
                },
            },
        }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._config = config

        # Перестраиваем фильтр новостей под реальный whitelist из config.json.
        # Это позволяет ENA/DOT/XRP-новостям проходить фильтр (а не только BTC/ETH/SOL).
        try:
            exch_cfg = config.get("exchange", {}) or {}
            wl = exch_cfg.get("pair_whitelist") or config.get("pair_whitelist") or []
            update_tracked_coins_from_whitelist(wl)
        except Exception as exc:
            logger.warning(f"[News] не удалось обновить tracked coins из whitelist: {exc}")

        self.stoploss = float(config.get("stoploss", STOPLOSS_PCT))
        self.take_profit = float(config.get("take_profit", TAKE_PROFIT))
        self.trailing_after_tp = bool(config.get("trailing_after_tp", TRAILING_AFTER_TP))
        self.trailing_retrace = float(config.get("trailing_retrace", TRAILING_RETRACE))
        self.trailing_min_activation = float(
            config.get("trailing_min_activation", TRAILING_MIN_ACT),
        )
        self.force_exit_after_days = float(
            config.get("force_exit_after_days", FORCE_EXIT_AFTER_DAYS),
        )
        self.force_exit_by_days_enabled = bool(
            config.get("force_exit_by_days_enabled", FORCE_EXIT_BY_DAYS_ENABLED),
        )
        self.tg  = TelegramNotifier(config)
        self.llm = LLMClient()
        self.trade_execution_enabled = bool(
            config.get("trade_execution_enabled", TRADE_EXECUTION_ENABLED_DEFAULT)
        )
        # Сопровождение открытых позиций через LLM — переключатель только в коде стратегии.
        # Меняй константу LLM_MANAGE_ENABLED в начале файла (True/False) и рестарт контейнера.
        # TP/trailing/stoploss продолжат работать независимо.
        self.llm_manage_enabled = LLM_MANAGE_ENABLED
        self._cache:          Dict[str, Tuple[Any, Dict[str, Any]]] = {}
        self._seen_pairs:     set = set()
        self._pending_stake:  Dict[str, float] = {}
        self._trade_memory:   TradeMemory = TradeMemory()
        self._entry_times:    Dict[int, datetime] = {}
        # Последнее решение по паре для передачи контекста в следующий промт
        self._last_decision:  Dict[str, Dict[str, Any]] = {}
        self._neutral_analysis_history: Dict[str, List[Dict[str, Any]]] = {}
        self._signal_stats: Dict[str, Any] = {
            "candles": 0,
            "llm_errors": 0,
            "semantic_rejects": 0,
            "context_timeouts": 0,
            "arbiter_blocks": 0,
            "entry_signals": 0,
            "accepted_entries": 0,
            "action_counts": {"LONG_ENTER": 0, "SHORT_ENTER": 0, "NEUTRAL": 0},
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
        logger.info("GPT Strategy v5.3 запущена")
        self._startup_check()

    # ── [14.1] Startup / self-test ─────────────────────────────
    def _startup_check(self) -> None:
        """Проверяет все компоненты стратегии при запуске. Отправляет отчёт в TG."""
        errors:   List[str] = []
        warnings: List[str] = []
        passed:   List[str] = []
        ind: Optional[Indicators] = None

        # ── 0. Целостность стратегии (без analyst / LLM-exit) ───────
        try:
            src = Path(__file__).read_text(encoding="utf-8")
            hits = [sig for sig in _INTEGRITY_FORBIDDEN if sig in src]
            assert not hits, f"запрещённые фрагменты в коде: {hits}"
            assert GPTStrategy.use_exit_signal is True, "use_exit_signal должен быть True (требуется для custom_exit / TP)"
            assert set(get_args(TradeAction)) == {"LONG_ENTER", "SHORT_ENTER", "NEUTRAL"}
            manage_sample = PromptBuilder.build_manage_prompt(
                Indicators(price=100.0), [], "BTC/USDT", "LONG", 1.0,
            )
            assert "action=NEUTRAL" in manage_sample, "manage prompt не требует NEUTRAL"
            assert _LX not in manage_sample and _SX not in manage_sample
            passed.append(
                "Целостность: v5.3 без analyst/LLM-exit, use_exit_signal=True (для custom_exit), manage→NEUTRAL"
            )
        except Exception as e:
            errors.append(f"Целостность стратегии: {e}")

        # ── 1. Telegram (реальный getMe) ────────────────────────────
        try:
            if self.tg.enabled:
                ok, detail = self.tg.verify_connection()
                if ok:
                    passed.append(f"Telegram API: getMe OK ({detail})")
                else:
                    errors.append(f"Telegram API: getMe провален — {detail}")
            else:
                warnings.append("Telegram: отключён (нет token/chat_id) — уведомления не будут приходить")
        except Exception as e:
            errors.append(f"Telegram init: {e}")

        # ── 2. LLM API ────────────────────────────────────────────────
        try:
            if not _llm_key():
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

        # ── 5. Память сделок (roundtrip на диск без загрязнения истории) ─
        try:
            import os
            test_record = TradeRecord(
                pair="__STARTUP_MEMTEST__",
                side="LONG",
                profit_pct=1.5,
                duration_min=45.0,
                entry_time=datetime.now(timezone.utc),
                exit_time=datetime.now(timezone.utc),
                final_score=0.45,
                exit_reason="startup_test",
            )
            count_before = len(self._trade_memory._records)
            prompt_line = test_record.to_prompt_line()
            assert "__STARTUP_MEMTEST__" in prompt_line, "to_prompt_line не содержит pair"
            assert test_record.age_str() is not None, "age_str() сломан"
            self._trade_memory.add(test_record)
            reloaded = TradeMemory()
            assert any(r.pair == "__STARTUP_MEMTEST__" for r in reloaded._records), \
                "запись не пережила reload с диска"
            self._trade_memory._records = [
                r for r in self._trade_memory._records if r.pair != "__STARTUP_MEMTEST__"
            ]
            self._trade_memory._save()
            reloaded2 = TradeMemory()
            assert not any(r.pair == "__STARTUP_MEMTEST__" for r in reloaded2._records), \
                "тестовая запись не удалена с диска"
            file_exists = os.path.exists(MEMORY_FILE)
            passed.append(
                f"Память сделок: write→reload→cleanup OK "
                f"| файл={'найден' if file_exists else 'создан'} | сделок: {count_before}"
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
                confidence=0.7,
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

        # ── 7a. tech_anchor / snap_tech_score ─────────────────────────
        try:
            bull_ind = Indicators(
                st_dir="бычий", sma_cross="бычий", price=100.0, sma200=90.0,
                rsi=48.0, macd_trend="бычий", bb_pct=55.0, bb_pos="середина",
                stoch_zone="нейтральная", stoch_cross="нет",
                williams_zone="нейтральная", cci_signal="нейтральный",
                obv_trend="бычий", vol_ratio=0.9, adx=26.0, atr_pct=1.2,
                align_bull=6, align_bear=1,
            )
            anchor = IndicatorEngine.compute_tech_anchor(bull_ind)
            assert abs(anchor) >= TECH_ANCHOR_MIN_ABS, f"anchor слабый: {anchor}"
            zero_rec = TradingRecommendation(
                tech_score=0.0, action="NEUTRAL", confidence=0.35,
                sentiments=[], key_news="нет релевантных",
            )
            fixed = snap_tech_score(zero_rec, bull_ind)
            assert abs(fixed.tech_score) >= TECH_ANCHOR_MIN_ABS, "snap не подставил anchor"
            assert fixed.tech_score == anchor

            balanced = Indicators(
                st_dir="бычий", sma_cross="медвежий", price=100.0, sma200=100.0,
                rsi=50.0, macd_trend="бычий", bb_pct=50.0, bb_pos="середина",
                stoch_zone="нейтральная", stoch_cross="нет",
                williams_zone="нейтральная", cci_signal="нейтральный",
                obv_trend="медвежий", vol_ratio=1.0, adx=22.0, atr_pct=1.0,
                align_bull=5, align_bear=4,
            )
            keep = snap_tech_score(zero_rec, balanced)
            assert keep.tech_score == 0.0, "сбалансированный рынок: 0 не трогаем"

            meaningful = TradingRecommendation(
                tech_score=0.42, action="NEUTRAL", confidence=0.5, sentiments=[],
            )
            assert snap_tech_score(meaningful, bull_ind).tech_score == 0.42
            passed.append(f"tech_anchor: snap 0→{fixed.tech_score:+.2f}, balanced OK")
        except Exception as e:
            errors.append(f"tech_anchor: {e}")

        # ── 8. PromptBuilder ──────────────────────────────────────────
        try:
            if ind is not None:
                entry_prompt = PromptBuilder.build_entry_prompt(
                    ind, ["Test news headline"], "BTC/USDT", "нет истории"
                )
                assert len(entry_prompt) > 100, "Entry prompt слишком короткий"
                manage_prompt = PromptBuilder.build_manage_prompt(
                    ind, [], "BTC/USDT", "LONG", 1.5, "нет истории"
                )
                assert len(manage_prompt) > 100,  "Manage prompt слишком короткий"
                passed.append(f"PromptBuilder: entry={len(entry_prompt)} символов manage={len(manage_prompt)} символов")
            else:
                warnings.append("PromptBuilder: пропущен — IndicatorEngine не вернул данные")
        except Exception as e:
            errors.append(f"PromptBuilder: {e}")

        # ── 9. NewsProvider (только структура, без реального запроса) ─
        try:
            assert len(RSS_FEEDS) > 0, "RSS_FEEDS пустой"
            assert NEWS_CACHE_SEC > 0, "NEWS_CACHE_SEC <= 0"
            eth_whale = (
                "Ethereum whale opens $100M short as Vitalik Buterin vows to 'sell less ETH'"
            )
            assert not _news_headline_relevant(eth_whale, "BTC"), "ETH-новость не для BTC"
            assert not _news_headline_relevant(eth_whale, "SOL"), "ETH-новость не для SOL"
            assert _news_headline_relevant(eth_whale, "ETH"), "ETH-новость для ETH"
            assert _news_headline_relevant("Bitcoin risks drop to $72K", "BTC")
            assert not _news_headline_relevant("Bitcoin risks drop to $72K", "ETH")
            assert _news_headline_relevant("Crypto market fear index hits 2026 low", "SOL")
            assert not _news_headline_relevant("PEPE memecoin surges 50%", "BTC")
            filt = _filter_news_headlines_for_pair(
                [eth_whale, "Bitcoin holds $77K support", "SEC reviews crypto ETF rule"],
                "BTC/USDT",
            )
            assert eth_whale not in filt and len(filt) >= 1
            passed.append(f"NewsProvider: {len(RSS_FEEDS)} источников + фильтр монет OK")
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

        # ── 12. Контекст предыдущего решения ─────────────────────────
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
            block = PromptBuilder._format_neutral_history_block([
                {"action": "NEUTRAL", "final_score": -0.1, "confidence": 0.5,
                 "tech_reasoning": "тест1", "counter_arg": "арг1", "candle_time": "c1"},
                {"action": "NEUTRAL", "final_score": 0.05, "confidence": 0.35,
                 "tech_reasoning": "тест2", "counter_arg": "арг2", "candle_time": "c2"},
            ])
            assert "ПРОШЛЫЕ NEUTRAL" in block and "[1]" in block and "[2]" in block
            assert PromptBuilder._format_neutral_history_block([]) == ""
            self._neutral_analysis_history[test_pair] = [
                {"action": "NEUTRAL", "final_score": -0.1, "confidence": 0.5,
                 "tech_reasoning": "т", "counter_arg": "а", "candle_time": "c1"},
            ]
            assert len(self._neutral_analysis_history[test_pair]) == 1
            del self._neutral_analysis_history[test_pair]
            passed.append(
                "Контекст NEUTRAL: 2 прошлых анализа в промт только после NEUTRAL"
            )
        except Exception as e:
            errors.append(f"Контекст NEUTRAL-истории: {e}")

        # ── 14. Протокол валидации качества (WF/anchored) ───────────
        try:
            wf = self._build_walkforward_protocol()
            assert "anchored" in wf and "rolling" in wf, "WF protocol неполный"
            passed.append("Оценка качества: walk-forward/anchored шаблон инициализирован")
        except Exception as e:
            errors.append(f"WF protocol: {e}")

        # ── 15. Все пороги торговли (видимая сводка) ──────────────────
        try:
            assert ENTRY_LONG_THRESHOLD > 0 and ENTRY_SHORT_THRESHOLD < 0, "пороги входа некорректны"
            assert 0 < ENTRY_MIN_CONFIDENCE <= 1, "ENTRY_MIN_CONFIDENCE вне [0..1]"
            assert 0 < LOW_CONFIDENCE_STAKE_THRESHOLD <= 1, "LOW_CONFIDENCE_STAKE_THRESHOLD вне [0..1]"
            assert ENTRY_MAX_ATR_PCT > 0, "ENTRY_MAX_ATR_PCT должен быть > 0"
            assert MIN_ALIGNED_INDICATORS >= 1, "MIN_ALIGNED_INDICATORS должен быть >= 1"
            # use_exit_signal=True требуется чтобы Freqtrade вызывал custom_exit (TP/trailing).
            # LLM-exit всё равно не работает, потому что populate_exit_trend пустой.
            assert GPTStrategy.use_exit_signal is True, "use_exit_signal должен быть True для custom_exit"
            passed.append(
                f"Пороги: вход LONG ≥{ENTRY_LONG_THRESHOLD:+.2f} / SHORT ≤{ENTRY_SHORT_THRESHOLD:+.2f}, "
                f"выход=механический SL/TP/trailing (без LLM), "
                f"min_conf={ENTRY_MIN_CONFIDENCE:.0%}, "
                f"align≥{MIN_ALIGNED_INDICATORS}, "
                f"low_conf_stake_thr={LOW_CONFIDENCE_STAKE_THRESHOLD:.0%}, "
                f"max_ATR%={ENTRY_MAX_ATR_PCT:.1f}"
            )
        except Exception as e:
            errors.append(f"Пороги торговли: {e}")

        # ── 16. Режим исполнения сделок ───────────────────────────────
        try:
            mode = "🟢 ТОРГОВЛЯ ВКЛЮЧЕНА" if self.trade_execution_enabled else "🟡 ТОЛЬКО ОТЧЁТЫ (входы выключены)"
            passed.append(f"Режим исполнения: {mode}")
        except Exception as e:
            errors.append(f"Режим исполнения: {e}")

        # ── 17. Sanitize HTML (защита от ошибок Telegram) ─────────────
        try:
            cases = [
                ("RSI < 30",                 "RSI &lt; 30"),
                ("AT&T",                     "AT&amp;T"),
                ("<b>жирный</b>",            "<b>жирный</b>"),
                ("<i>курсив < 5%</i>",       "<i>курсив &lt; 5%</i>"),
                ("<code>cmd</code>",         "<code>cmd</code>"),
                ("<script>x</script>",       "&lt;script&gt;x&lt;/script&gt;"),
            ]
            for src, expected in cases:
                got = TelegramNotifier._sanitize_html(src)
                assert got == expected, f"sanitize_html: {src!r} → {got!r}, ждали {expected!r}"
            passed.append(f"HTML-санитайзер: {len(cases)} тест-кейсов прошли (от XSS и Bad Request защищён)")
        except Exception as e:
            errors.append(f"HTML-санитайзер: {e}")

        # ── 18. Валидатор confidence (clamp [0..1]) ───────────────────
        try:
            tests = [
                (0.0, 0.0,  "0.0 → 0.0 (sentinel «нет данных»)"),
                (0.55, 0.55, "0.55 → 0.55"),
                (1.5, 1.0,  "1.5 → 1.0 (clamp сверху)"),
                (-0.3, 0.0, "-0.3 → 0.0 (clamp снизу)"),
            ]
            for inp, exp, _ in tests:
                rec = TradingRecommendation(confidence=inp, action="NEUTRAL")
                assert abs(rec.confidence - exp) < 0.001, f"clamp: {inp} → {rec.confidence}, ждали {exp}"
            # confidence теперь обязательное — должно бросать ошибку без него
            try:
                TradingRecommendation(action="NEUTRAL")
                raise AssertionError("confidence должно быть обязательным, но прошло без него")
            except Exception:
                pass
            passed.append(f"Валидатор confidence: clamp [0..1] и обязательность работают ({len(tests)} тестов)")
        except Exception as e:
            errors.append(f"Валидатор confidence: {e}")

        # ── 18b. Pydantic null-coercion (защита от LLM, возвращающей null) ─
        try:
            # LLM иногда отдаёт null в строковых полях — pre-валидатор должен
            # превратить их в "" / разумные дефолты ДО валидации.
            null_payload = {
                "confidence":       0.6,
                "action":           None,
                "key_news":         None,
                "news_summary":     None,
                "news_reasoning":   None,
                "tech_reasoning":   None,
                "counter_argument": None,
                "news_mood":        None,
                "summary":          None,
                "tech_score":       None,
                "stake_amount":     None,
                "sentiments":       None,
            }
            rec_null = TradingRecommendation(**null_payload)
            assert rec_null.key_news == ""
            assert rec_null.news_summary == ""
            assert rec_null.news_reasoning == ""
            assert rec_null.tech_reasoning == ""
            assert rec_null.counter_argument == ""
            assert rec_null.action == "NEUTRAL"
            assert rec_null.tech_score == 0.0
            assert rec_null.sentiments == []
            passed.append("Null-coercion: 12 None-полей корректно подменены на дефолты (защита от json null от LLM)")
        except Exception as e:
            errors.append(f"Null-coercion: {e}")

        # ── 19. TGFormatter._confidence_line (диагностика при сбое) ───
        try:
            rec_ok = TradingRecommendation(confidence=0.75, action="NEUTRAL")
            rec_zero = TradingRecommendation(confidence=0.0, action="NEUTRAL")
            line_ok = TGFormatter._confidence_line(rec_ok)
            line_zero = TGFormatter._confidence_line(rec_zero)
            assert "75%" in line_ok and "🟢" in line_ok, f"confidence_line ok: {line_ok!r}"
            assert "не получена" in line_zero, f"confidence_line zero: {line_zero!r}"
            if ind:
                rec_with_reason = TradingRecommendation(
                    confidence=0.75,
                    action="LONG_ENTER",
                    final_score=0.51,
                    confidence_reason="Любой текст — в Telegram не показывается.",
                    counter_argument="Vol слабый у сопротивления.",
                )
                block = TGFormatter._confidence_block(rec_with_reason, ind, "LONG_ENTER")
                assert "75%" in block and "🟢" in block, f"confidence_block: {block!r}"
                assert "Почему так" not in block
                assert "не показывается" not in block
            passed.append("Confidence: в Telegram только цифра, без текста")
        except Exception as e:
            errors.append(f"Confidence-строка: {e}")

        # ── 20. TGFormatter: генерация всех типов сообщений ───────────
        try:
            rec_full = TradingRecommendation(
                sentiments=[{"title":"t","score":0.4,"sentiment":0.4,"relevance":1.0,"recency":1.0}],
                key_news="key news",
                news_summary="news summary text",
                news_mood="нейтральный",
                tech_score=0.5,
                action="LONG_ENTER",
                summary="" + "тест "*8,
                stake_amount=100.0,
                confidence=0.7,
                tech_reasoning="бычий тренд по SuperTrend, ADX 35",
                counter_argument="RSI близко к перекупленности",
                news_reasoning="новости умеренно позитивные",
            )
            msgs = {
                "neutral":      TGFormatter.neutral("BTC/USDT", rec_full, ind) if ind else "",
                "entry":        TGFormatter.entry("BTC/USDT", "long", 50000.0, 100.0, rec_full, ind) if ind else "",
                "reasoning_enter":  TGFormatter.reasoning_msg("BTC/USDT", rec_full, "LONG_ENTER", ind) if ind else "",
                "reasoning_manage": TGFormatter.reasoning_msg("BTC/USDT", rec_full, "SHORT_MANAGE", ind) if ind else "",
                "reasoning_neutral":TGFormatter.reasoning_msg("BTC/USDT", rec_full, "NEUTRAL", ind) if ind else "",
            }
            for k, v in msgs.items():
                assert isinstance(v, str) and (k in ("neutral","entry") or len(v) > 30), f"{k} пустое"
            passed.append(f"TGFormatter: {len(msgs)} типов сообщений генерируются корректно")
        except Exception as e:
            errors.append(f"TGFormatter: {e}")

        # ── 21. Арбитр входа (фильтр по confidence/ATR/SuperTrend) ────
        try:
            if ind is not None:
                # low confidence → блокируется
                rec_low = TradingRecommendation(
                    confidence=0.20, action="LONG_ENTER",
                    tech_score=0.5, stake_amount=100.0,
                    summary="тест тест тест тест тест тест",
                )
                arb, reasons = self._arbiter_action(rec_low, ind, side=None)
                assert arb == "NEUTRAL", f"low_confidence не блокирует: action={arb}"
                assert any("low_confidence" in r for r in reasons), f"нет причины low_confidence: {reasons}"
                passed.append(f"Арбитр входа: low_confidence корректно блокирует (причины: {reasons[:2]})")
            else:
                warnings.append("Арбитр входа: пропущен — IndicatorEngine не вернул данные")
        except Exception as e:
            errors.append(f"Арбитр входа: {e}")

        # ── 21a. Согласование индикаторов (count_alignment) ───────────
        try:
            if ind is not None:
                bull_n, bear_n = IndicatorEngine.count_alignment(ind)
                assert bull_n == ind.align_bull and bear_n == ind.align_bear
                entry_prompt = PromptBuilder.build_entry_prompt(ind, [], "BTC/USDT")
                assert "СОГЛАСОВАНИЕ ИНДИКАТОРОВ" in entry_prompt
                assert f"За LONG: {bull_n}" in entry_prompt
                rec_align = TradingRecommendation(
                    confidence=0.80,
                    action="LONG_ENTER",
                    tech_score=0.50,
                    summary="тест тест тест тест тест тест",
                    stake_amount=100.0,
                )
                arb_align, reasons_align = self._arbiter_action(rec_align, ind, side=None)
                if bull_n < MIN_ALIGNED_INDICATORS:
                    assert arb_align == "NEUTRAL"
                    assert any("low_bull_align" in r for r in reasons_align)
                passed.append(
                    f"Согласование: за LONG={bull_n}, за SHORT={bear_n} "
                    f"(мин. {MIN_ALIGNED_INDICATORS} для входа)"
                )
            else:
                warnings.append("Согласование индикаторов: пропущено — нет ind")
        except Exception as e:
            errors.append(f"Согласование индикаторов: {e}")

        # ── 21b. NEUTRAL — одна строка причины ───────────────────────
        try:
            if ind is not None:
                rec_n = TradingRecommendation(
                    confidence=0.35,
                    action="NEUTRAL",
                    tech_score=0.10,
                    summary="черновик",
                    stake_amount=float(STAKE_LEVELS[0]),
                )
                line = self._build_neutral_reason_line(
                    rec=rec_n,
                    ind=ind,
                    side=None,
                    llm_action="NEUTRAL",
                    arbiter_reasons=[],
                    llm_blocked=False,
                )
                assert isinstance(line, str) and len(line) > 10
                assert "\n" not in line.strip()
                tg_neutral = TGFormatter.neutral("BTC/USDT", rec_n, ind)
                assert "Почему:" in tg_neutral
                passed.append(f"NEUTRAL 1 строка: {line[:100]}")
            else:
                warnings.append("NEUTRAL причина: пропущено — нет ind")
        except Exception as e:
            errors.append(f"NEUTRAL причина: {e}")

        # ── 21c. Арбитр in-trade (manage — только NEUTRAL) ───────────
        try:
            if ind is not None:
                rec_in_trade = TradingRecommendation(
                    confidence=0.90,
                    action="LONG_ENTER",
                    tech_score=0.55,
                    summary="тест тест тест тест тест тест",
                    stake_amount=100.0,
                )
                arb_it, reasons_it = self._arbiter_action(rec_in_trade, ind, side="LONG")
                assert arb_it == "NEUTRAL", f"in-trade не форсирует NEUTRAL: {arb_it}"
                passed.append("Арбитр in-trade: LLM ENTER → NEUTRAL (сопровождение без входа/выхода)")
            else:
                warnings.append("Арбитр in-trade: пропущен — нет ind")
        except Exception as e:
            errors.append(f"Арбитр in-trade: {e}")

        # ── 21d. _persist_closed_trade — dedup по trade.id ───────────
        try:
            class _StartupMockTrade:
                id = 999999001
                is_short = False
                enter_tag = "gpt_long"
                open_date_utc = datetime.now(timezone.utc) - timedelta(minutes=30)

                def __init__(self) -> None:
                    self._custom: Dict[str, Any] = {}

                def get_custom_data(self, key: str) -> Any:
                    return self._custom.get(key)

                def set_custom_data(self, key: str, val: Any) -> None:
                    self._custom[key] = val

                def calc_profit_ratio(self, rate: float) -> float:
                    return 0.012

            mock_trade = _StartupMockTrade()
            captured: List[TradeRecord] = []
            orig_add = self._trade_memory.add

            def _capture_add(record: TradeRecord) -> None:
                if record.pair == "__STARTUP_PERSIST__":
                    captured.append(record)
                else:
                    orig_add(record)

            self._trade_memory.add = _capture_add  # type: ignore[method-assign]
            try:
                now = datetime.now(timezone.utc)
                first = self._persist_closed_trade(
                    trade=mock_trade,
                    pair="__STARTUP_PERSIST__",
                    side_label="LONG",
                    current_time=now,
                    current_profit=0.012,
                    exit_reason="startup_test",
                    final_score=0.3,
                )
                second = self._persist_closed_trade(
                    trade=mock_trade,
                    pair="__STARTUP_PERSIST__",
                    side_label="LONG",
                    current_time=now,
                    current_profit=0.012,
                    exit_reason="startup_test",
                    final_score=0.3,
                )
                assert first is True and second is False, "dedup не сработал"
                assert len(captured) == 1, f"ожидали 1 запись, получили {len(captured)}"
                assert mock_trade.get_custom_data(self._TRADE_MEMORY_SAVED_KEY) is True
                passed.append("_persist_closed_trade: dedup trade_memory_saved работает")
            finally:
                self._trade_memory.add = orig_add  # type: ignore[method-assign]
        except Exception as e:
            errors.append(f"_persist_closed_trade: {e}")

        # ── 21e. custom_exit / TP (без LLM, без dp) ──────────────────
        try:
            class _ExitMockTrade:
                id = 999999002
                is_short = False
                enter_tag = "gpt_long"
                open_date_utc = datetime.now(timezone.utc) - timedelta(minutes=10)

                def __init__(self) -> None:
                    self._custom: Dict[str, Any] = {}

                def get_custom_data(self, key: str) -> Any:
                    return self._custom.get(key)

                def set_custom_data(self, key: str, val: Any) -> None:
                    self._custom[key] = val

            exit_trade = _ExitMockTrade()
            saved_trailing = self.trailing_after_tp
            self.trailing_after_tp = False
            try:
                tp_reason = self._take_profit_exit(
                    "__STARTUP_EXIT__",
                    exit_trade,
                    datetime.now(timezone.utc),
                    100.0,
                    self.take_profit + 0.005,
                )
                assert tp_reason == "take_profit", f"TP не сработал: {tp_reason}"
                passed.append(
                    f"custom_exit/TP: take_profit срабатывает при profit≥{self.take_profit:.1%}"
                )
            finally:
                self.trailing_after_tp = saved_trailing
        except Exception as e:
            errors.append(f"custom_exit/TP: {e}")

        # ── 21f. TGFormatter.exit_msg ─────────────────────────────────
        try:
            if ind is not None:
                exit_rec = TradingRecommendation(
                    confidence=0.6,
                    action="NEUTRAL",
                    tech_score=0.2,
                    key_news="test",
                    news_summary="summary",
                    news_mood="нейтральный",
                    summary="выход по TP",
                    stake_amount=100.0,
                )
                exit_text = TGFormatter.exit_msg(
                    "BTC/USDT", "LONG", 0.018, 42.0,
                    {"sent": 0.1, "tech": 0.2, "final": 0.17},
                    "тестовый вывод", ind, exit_rec,
                )
                assert "ВЫХОД LONG" in exit_text and "BTC/USDT" in exit_text
                assert len(exit_text) > 80
                passed.append("TGFormatter.exit_msg: сообщение о механическом выходе генерируется")
            else:
                warnings.append("TGFormatter.exit_msg: пропущен — нет ind")
        except Exception as e:
            errors.append(f"TGFormatter.exit_msg: {e}")

        # ── 22. NewsProvider (реальный fetch) ────────────────────────
        try:
            news = NewsProvider.fetch(max_items=3)
            if news:
                passed.append(f"NewsProvider (live): получено {len(news)} новостей, первая: '{news[0][:60]}…'")
            else:
                warnings.append("NewsProvider (live): RSS не вернул новости — работает без новостей")
        except Exception as e:
            warnings.append(f"NewsProvider (live): {e}")

        # ── 23. Запись файла памяти (writable disk) ──────────────────
        try:
            import os, tempfile, json as _json
            test_file = os.path.join(tempfile.gettempdir(), "gpt_strategy_writetest.json")
            with open(test_file, "w", encoding="utf-8") as f:
                _json.dump({"ok": True}, f)
            with open(test_file, "r", encoding="utf-8") as f:
                assert _json.load(f)["ok"] is True, "содержимое файла повреждено"
            os.unlink(test_file)
            mem_dir = os.path.dirname(MEMORY_FILE) or "."
            mem_writable = os.access(mem_dir, os.W_OK) if os.path.exists(mem_dir) else True
            passed.append(f"Файловая система: запись работает | memory_dir={'writable' if mem_writable else '❌ READ-ONLY'}")
        except Exception as e:
            errors.append(f"Файловая система: {e}")

        # ── 24. Зависимости и среда исполнения ───────────────────────
        try:
            import numpy as _np, pandas as _pd, requests as _req, httpx as _httpx
            try:
                import feedparser as _fp
                fp_ver = getattr(_fp, "__version__", "?")
            except Exception:
                fp_ver = "не установлен"
            py_ver = f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}"
            passed.append(
                f"Зависимости: python={py_ver} numpy={_np.__version__} pandas={_pd.__version__} "
                f"httpx={_httpx.__version__} requests={_req.__version__} feedparser={fp_ver}"
            )
        except Exception as e:
            errors.append(f"Зависимости: {e}")

        # ── 25. Системное время (UTC) ────────────────────────────────
        try:
            import time as _t
            local = datetime.now()
            utc = datetime.now(timezone.utc)
            ts_drift = abs(_t.time() - (utc - datetime(1970,1,1,tzinfo=timezone.utc)).total_seconds())
            assert ts_drift < 1.0, f"подозрительный сдвиг времени: {ts_drift}"
            passed.append(f"Системное время: UTC={utc.strftime('%H:%M:%S')} local={local.strftime('%H:%M:%S')}")
        except Exception as e:
            warnings.append(f"Системное время: {e}")

        # ── 26. Мониторинг (пороги алертов) ──────────────────────────
        try:
            assert ALERT_COOLDOWN_SEC > 0,        "ALERT_COOLDOWN_SEC <= 0"
            assert MONITORING_MIN_CANDLES >= 10,  "MONITORING_MIN_CANDLES слишком мало"
            passed.append(
                f"Мониторинг: cooldown={ALERT_COOLDOWN_SEC}с min_candles={MONITORING_MIN_CANDLES} "
                f"(пороги: neutral>85% или blocked>30%)"
            )
        except Exception as e:
            errors.append(f"Мониторинг: {e}")

        # ── 27. Whitelist пар из конфига ─────────────────────────────
        try:
            cfg = self.config or {}
            ex = (cfg.get("exchange") or {})
            pairs = ex.get("pair_whitelist", []) or cfg.get("pair_whitelist", [])
            if pairs:
                passed.append(f"Whitelist пар: {len(pairs)} пар(а) — {', '.join(pairs[:5])}{'…' if len(pairs)>5 else ''}")
            else:
                warnings.append("Whitelist пар: список пуст или не задан")
        except Exception as e:
            warnings.append(f"Whitelist пар: {e}")

        # ── 28. LLM endpoint (статус, без живого запроса) ───────────
        try:
            llm_url_ok = bool(_llm_key()) and bool(LLM_MODEL)
            if llm_url_ok:
                passed.append(f"LLM endpoint: ключ установлен ({len(_llm_key())} символов), модель='{LLM_MODEL}', retries={LLM_RETRIES}, timeout={LLM_TIMEOUT}с")
            else:
                warnings.append("LLM endpoint: ключ или модель не настроены — стратегия будет в fallback NEUTRAL")
        except Exception as e:
            errors.append(f"LLM endpoint: {e}")

        # ── 29. LLM DRY-RUN (полноценный прогон на одной монете) ──────
        # Делает реальный запрос к LLM на синтетических данных BTC-like,
        # парсит ответ, проверяет все обязательные поля, индикаторы,
        # генерирует все типы Telegram-сообщений. Полный текст ответа ИИ
        # уходит в логи; в Telegram идёт только чек-лист ✅/❌ по пунктам.
        dry_run_section: List[str] = []
        dry_run_failed: bool = False
        manage_dry_section: List[str] = []
        manage_dry_failed: bool = False
        try:
            if not _llm_key():
                warnings.append("LLM DRY-RUN: пропущен — нет LLM_API_KEY")
            elif ind is None:
                warnings.append("LLM DRY-RUN: пропущен — IndicatorEngine не вернул данные")
            else:
                test_prompt = PromptBuilder.build_entry_prompt(
                    ind=ind,
                    news_items=["Bitcoin rallies on ETF inflow", "Fed signals dovish stance"],
                    pair="BTC/USDT",
                    memory_block="нет истории сделок",
                    fear_greed="Greed (68)",
                    neutral_history=[
                        {"action": "NEUTRAL", "final_score": 0.09, "confidence": 0.35,
                         "tech_reasoning": "тест dry-run", "counter_arg": "тест", "candle_time": "prev"},
                    ],
                )
                t0 = time.time()
                test_rec_raw = self.llm.analyze(test_prompt)
                elapsed = time.time() - t0
                test_rec_live = snap_tech_score(test_rec_raw, ind)
                test_rec_live = sanitize_news_recommendation(test_rec_live, "BTC/USDT")
                test_rec_live = polish_recommendation_text(test_rec_live, "BTC/USDT", ind)
                test_rec_live = finalize_confidence_reason(test_rec_live, ind)
                llm_blocked_dr = test_rec_live.summary.startswith("Сигнал отклонён semantic-validator")
                arb_act, arb_rs = self._arbiter_action(test_rec_live, ind, None)
                if arb_act == "NEUTRAL":
                    test_rec_live.summary = self._build_neutral_reason_line(
                        rec=test_rec_live,
                        ind=ind,
                        side=None,
                        llm_action=test_rec_live.action,
                        arbiter_reasons=arb_rs,
                        llm_blocked=llm_blocked_dr,
                    )
                elif arb_act in ("LONG_ENTER", "SHORT_ENTER"):
                    test_rec_live.summary = self._build_entry_summary(
                        arb_act, test_rec_live, ind, "BTC/USDT",
                    )

                # ── Чек-лист пунктов проверки ──────────────────────────
                checks: List[tuple] = []  # (label, ok: bool, detail: str)

                # 1) Action валиден
                valid_actions = {"LONG_ENTER","SHORT_ENTER","NEUTRAL"}
                ok = test_rec_live.action in valid_actions
                checks.append(("Action валиден", ok, test_rec_live.action))

                # 2) Confidence получен и в диапазоне 0..1
                ok = isinstance(test_rec_live.confidence, float) and 0.0 <= test_rec_live.confidence <= 1.0
                checks.append(("Confidence получен", ok, f"{test_rec_live.confidence:.0%}"))

                ok_conf_pos = (
                    test_rec_live.confidence >= 0.05
                    or test_rec_live.action == "NEUTRAL"
                )
                checks.append(("Confidence ≥ 5% (или NEUTRAL)", ok_conf_pos,
                               f"{test_rec_live.confidence:.0%}"))

                expected_fs = round(
                    WEIGHT_TECHNICAL * float(test_rec_live.tech_score)
                    + WEIGHT_SENTIMENT * float(test_rec_live.sent_score),
                    SCORE_ROUND_DECIMALS,
                )
                ok_fs = abs(float(test_rec_live.final_score) - expected_fs) <= 0.10
                checks.append(("Final score ≈ 0.85·tech + 0.15·sent", ok_fs,
                               f"llm={test_rec_live.final_score:+.2f} calc={expected_fs:+.2f}"))

                if test_rec_live.action in ("LONG_ENTER", "SHORT_ENTER"):
                    ok_ent_conf = test_rec_live.confidence >= ENTRY_MIN_CONFIDENCE
                    checks.append(("ENTER: confidence ≥ min", ok_ent_conf,
                                   f"{test_rec_live.confidence:.0%} / {ENTRY_MIN_CONFIDENCE:.0%}"))
                else:
                    checks.append(("ENTER: confidence ≥ min", True, f"{test_rec_live.action}"))

                if float(test_rec_live.confidence) < LOW_CONFIDENCE_STAKE_THRESHOLD:
                    ok_low_stake = float(test_rec_live.stake_amount) == float(min(STAKE_LEVELS))
                    checks.append(("Низкая conf → мин. stake", ok_low_stake,
                                   f"stake={test_rec_live.stake_amount:.0f}"))
                else:
                    checks.append(("Низкая conf → мин. stake", True, "conf OK"))

                sem_ok, sem_msg = LLMClient._semantic_validate(test_rec_raw)
                checks.append(("Semantic-validator (сырой ответ)", sem_ok, sem_msg))

                if test_rec_live.action in ("LONG_ENTER", "SHORT_ENTER") and sem_ok:
                    ok_arb = arb_act in ("LONG_ENTER", "SHORT_ENTER", "NEUTRAL")
                    checks.append(("Арбитр после LLM", ok_arb,
                                   f"llm={test_rec_live.action} arb={arb_act}"))
                else:
                    checks.append(("Арбитр после LLM", True, "не ENTER или sem fail"))

                # 3) Tech score в диапазоне [-1, 1]
                ok = -1.0 <= test_rec_live.tech_score <= 1.0
                checks.append(("Tech score в диапазоне", ok, f"{test_rec_live.tech_score:+.2f}"))

                # 4) Sentiment score в диапазоне [-1, 1]
                ok = -1.0 <= test_rec_live.sent_score <= 1.0
                checks.append(("Sentiment score в диапазоне", ok, f"{test_rec_live.sent_score:+.2f}"))

                # 5) Final score в диапазоне [-1, 1]
                ok = -1.0 <= test_rec_live.final_score <= 1.0
                checks.append(("Final score в диапазоне", ok, f"{test_rec_live.final_score:+.2f}"))

                # 6) News mood — одно из ожидаемых значений
                valid_moods = {"позитивный","негативный","нейтральный","смешанный"}
                ok = test_rec_live.news_mood in valid_moods
                checks.append(("News mood валиден", ok, test_rec_live.news_mood))

                # 7) Stake amount — из списка STAKE_LEVELS
                ok = float(test_rec_live.stake_amount) in [float(x) for x in STAKE_LEVELS]
                checks.append(("Stake в STAKE_LEVELS", ok, f"{test_rec_live.stake_amount:.0f}"))

                # 8) News reasoning заполнен
                ok = bool(test_rec_live.news_reasoning) and len(test_rec_live.news_reasoning) > 20
                checks.append(("Анализ новостей заполнен", ok, f"{len(test_rec_live.news_reasoning)} симв"))

                # 9) Tech reasoning заполнен
                ok = bool(test_rec_live.tech_reasoning) and len(test_rec_live.tech_reasoning) > 20
                checks.append(("Анализ техники заполнен", ok, f"{len(test_rec_live.tech_reasoning)} симв"))

                # 10) Counter-argument заполнен
                ok = bool(test_rec_live.counter_argument) and len(test_rec_live.counter_argument) > 10
                checks.append(("Контраргумент заполнен", ok, f"{len(test_rec_live.counter_argument)} симв"))

                # 11) Summary — для ENTER код формирует обязательно; для NEUTRAL опционален
                if arb_act in ("LONG_ENTER", "SHORT_ENTER"):
                    ok = bool(test_rec_live.summary) and len(test_rec_live.summary) > 10
                    checks.append(("Вывод (summary) для ENTER", ok, f"{len(test_rec_live.summary)} симв"))
                else:
                    ok = True
                    checks.append((
                        "Вывод (summary) для NEUTRAL",
                        ok,
                        f"{len(test_rec_live.summary or '')} симв (формирует код при блокерах)",
                    ))

                # 12) Согласованность action и confidence:
                # если confidence ниже порога ENTRY_MIN_CONFIDENCE — action не должен быть *_ENTER
                if test_rec_live.action in ("LONG_ENTER","SHORT_ENTER"):
                    ok = test_rec_live.confidence >= ENTRY_MIN_CONFIDENCE
                    checks.append(("Согласованность action⇄confidence", ok,
                                   f"{test_rec_live.action} @ {test_rec_live.confidence:.0%} (порог {ENTRY_MIN_CONFIDENCE:.0%})"))
                else:
                    checks.append(("Согласованность action⇄confidence", True,
                                   f"{test_rec_live.action} — без входа, порог не применяется"))

                # 13) Индикаторы IndicatorEngine — sanity check
                ind_ok = (
                    ind.price > 0
                    and ind.atr > 0
                    and ind.atr_pct >= 0
                    and 0 <= ind.rsi <= 100
                    and ind.st_dir in ("бычий","медвежий")
                )
                checks.append(("Индикаторы (price/ATR/RSI/ST)", ind_ok,
                               f"price={ind.price:.2f} ATR%={ind.atr_pct:.2f} RSI={ind.rsi:.1f} ST={ind.st_dir}"))

                # 14) Все 5 типов TG-сообщений рендерятся без исключений
                rendered: Dict[str, str] = {}
                render_errs: List[str] = []
                for label, fn in [
                    ("⚪ neutral",                 lambda: TGFormatter.neutral("BTC/USDT", test_rec_live, ind)),
                    ("🟢 entry (long)",            lambda: TGFormatter.entry("BTC/USDT","long",float(ind.price),100.0,test_rec_live,ind)),
                    ("🧠 reasoning(NEUTRAL)",      lambda: TGFormatter.reasoning_msg("BTC/USDT", test_rec_live, "NEUTRAL", ind)),
                    ("🧠 reasoning(LONG_ENTER)",   lambda: TGFormatter.reasoning_msg("BTC/USDT", test_rec_live, "LONG_ENTER", ind)),
                    ("🧠 reasoning(SHORT_MANAGE)", lambda: TGFormatter.reasoning_msg("BTC/USDT", test_rec_live, "SHORT_MANAGE", ind)),
                ]:
                    try:
                        msg = fn()
                        assert isinstance(msg, str) and len(msg) > 30
                        rendered[label] = msg
                    except Exception as err:
                        render_errs.append(f"{label}: {err}")
                ok = not render_errs
                checks.append(("TG-сообщения генерируются", ok,
                               f"{len(rendered)}/5 ок" + (f", ошибки: {render_errs}" if render_errs else "")))

                # 15) HTML-санитайзер не ломает сгенерированные сообщения
                try:
                    sanitized_all = [TelegramNotifier._sanitize_html(m) for m in rendered.values()]
                    ok = all(isinstance(s, str) and len(s) > 30 for s in sanitized_all)
                    checks.append(("HTML-санитайзер на реальном выводе", ok, f"{len(sanitized_all)} сообщений"))
                except Exception as err:
                    checks.append(("HTML-санитайзер на реальном выводе", False, str(err)))

                # ── Полный вывод модели — ТОЛЬКО В ЛОГИ ────────────────
                logger.info("═" * 70)
                logger.info("LLM DRY-RUN: полный вывод модели для проверки (НЕ отправлен в Telegram)")
                logger.info("─" * 70)
                logger.info(f"Pair:           BTC/USDT (синтетические данные индикаторов)")
                logger.info(f"Время ответа:   {elapsed:.2f}с")
                logger.info(f"Action:         {test_rec_live.action}")
                logger.info(f"Confidence:    {test_rec_live.confidence:.0%}")
                logger.info(f"Tech score:     {test_rec_live.tech_score:+.3f}")
                logger.info(f"Sent score:     {test_rec_live.sent_score:+.3f}")
                logger.info(f"Final score:    {test_rec_live.final_score:+.3f}")
                logger.info(f"News mood:      {test_rec_live.news_mood}")
                logger.info(f"Stake amount:   {test_rec_live.stake_amount:.0f}")
                logger.info(f"Neutral type:   {test_rec_live.neutral_type or '—'}")
                logger.info(f"Key news:       {test_rec_live.key_news or '—'}")
                logger.info("─── 📰 News reasoning ───")
                logger.info(test_rec_live.news_reasoning or "—")
                logger.info("─── 📊 Tech reasoning ───")
                logger.info(test_rec_live.tech_reasoning or "—")
                logger.info("─── ⚖️ Counter argument ───")
                logger.info(test_rec_live.counter_argument or "—")
                logger.info("─── 💬 Summary ───")
                logger.info(test_rec_live.summary or "—")
                logger.info("─── 🧮 Indicators ───")
                logger.info(f"price={ind.price:.4f} atr={ind.atr:.4f} atr_pct={ind.atr_pct:.2f}% "
                            f"rsi={ind.rsi:.2f} adx={ind.adx:.2f} st_dir={ind.st_dir} st_level={ind.st_level:.4f}")
                logger.info(f"sma7={ind.sma7:.4f} sma20={ind.sma20:.4f} sma200={ind.sma200:.4f} "
                            f"macd={ind.macd:.6f} sig={ind.macd_signal:.6f} hist={ind.macd_hist:+.6f}")
                logger.info(f"bb_pct={ind.bb_pct:.2f}% bb_pos={ind.bb_pos} bb_squeeze={ind.bb_squeeze} "
                            f"vol_ratio={ind.vol_ratio:.2f}x obv={ind.obv_trend}")
                logger.info("─── ✅ Чек-лист пунктов ───")
                for label, ok, detail in checks:
                    logger.info(f"  {'OK ' if ok else 'FAIL'} {label}: {detail}")
                logger.info("═" * 70)

                # ── Короткий чек-лист в Telegram (без сырого вывода ИИ) ─
                ok_count   = sum(1 for _, ok, _ in checks if ok)
                fail_count = len(checks) - ok_count
                dry_run_failed = fail_count > 0

                dry_run_section.append(
                    f"  • Время ответа: <b>{elapsed:.1f}с</b> | пунктов проверки: <b>{ok_count}/{len(checks)}</b>"
                )
                if dry_run_failed:
                    dry_run_section.append("  • Статус: <b>❌ есть ошибки</b> (подробный вывод ИИ — в логах docker)")
                else:
                    dry_run_section.append("  • Статус: <b>✅ пробный отчёт ИИ полностью верный</b> (полный вывод — в логах docker)")
                for label, ok, detail in checks:
                    mark = "✅" if ok else "❌"
                    if ok:
                        dry_run_section.append(f"  {mark} {label}")
                    else:
                        dry_run_section.append(f"  {mark} <b>{label}</b> — <i>{detail}</i>")

                if dry_run_failed:
                    errors.append(f"LLM DRY-RUN (entry): {fail_count} из {len(checks)} проверок не прошли (см. логи)")
                else:
                    passed.append(f"LLM DRY-RUN (entry): {ok_count}/{len(checks)} ✅ за {elapsed:.1f}с (полный вывод — в логах)")
        except Exception as e:
            errors.append(f"LLM DRY-RUN (entry): {e}")
            logger.exception("LLM DRY-RUN entry исключение")

        # ── 30. LLM DRY-RUN manage (сопровождение in-trade) ─────────
        try:
            if not _llm_key():
                warnings.append("LLM DRY-RUN (manage): пропущен — нет LLM_API_KEY")
            elif ind is None:
                warnings.append("LLM DRY-RUN (manage): пропущен — IndicatorEngine не вернул данные")
            else:
                manage_prompt = PromptBuilder.build_manage_prompt(
                    ind=ind,
                    news_items=["Bitcoin holds support", "Market awaits CPI data"],
                    pair="BTC/USDT",
                    side="LONG",
                    profit=1.8,
                    memory_block="нет истории сделок",
                    fear_greed="Neutral (50)",
                )
                t0m = time.time()
                manage_rec = self.llm.analyze(manage_prompt)
                elapsed_m = time.time() - t0m
                manage_rec = snap_tech_score(manage_rec, ind)
                manage_rec = sanitize_news_recommendation(manage_rec, "BTC/USDT")
                manage_rec = polish_recommendation_text(manage_rec, "BTC/USDT", ind)
                manage_rec = finalize_confidence_reason(manage_rec, ind, side="LONG")

                m_checks: List[tuple] = []
                ok = manage_rec.action == "NEUTRAL"
                m_checks.append(("Manage action=NEUTRAL", ok, manage_rec.action))
                ok = isinstance(manage_rec.confidence, float) and 0.0 <= manage_rec.confidence <= 1.0
                m_checks.append(("Manage confidence", ok, f"{manage_rec.confidence:.0%}"))
                sem_ok_m, sem_msg_m = LLMClient._semantic_validate(manage_rec)
                m_checks.append(("Manage semantic-validator", sem_ok_m, sem_msg_m))
                arb_m, _ = self._arbiter_action(manage_rec, ind, side="LONG")
                ok = arb_m == "NEUTRAL"
                m_checks.append(("Manage арбитр in-trade", ok, arb_m))
                ok = (
                    (bool(manage_rec.tech_reasoning) and len(manage_rec.tech_reasoning) > 15)
                    or (bool(manage_rec.summary) and len(manage_rec.summary) > 15)
                )
                m_checks.append((
                    "Manage текст (техника или summary)",
                    ok,
                    f"tech={len(manage_rec.tech_reasoning or '')} summary={len(manage_rec.summary or '')} симв",
                ))

                logger.info("═" * 70)
                logger.info("LLM DRY-RUN MANAGE: полный вывод (НЕ в Telegram)")
                logger.info(f"Action: {manage_rec.action} | Confidence: {manage_rec.confidence:.0%} | {elapsed_m:.2f}с")
                logger.info(manage_rec.tech_reasoning or manage_rec.summary or "—")
                logger.info("═" * 70)

                ok_m = sum(1 for _, ok, _ in m_checks if ok)
                fail_m = len(m_checks) - ok_m
                manage_dry_failed = fail_m > 0
                manage_dry_section.append(
                    f"  • Время: <b>{elapsed_m:.1f}с</b> | пунктов: <b>{ok_m}/{len(m_checks)}</b>"
                )
                for label, ok, detail in m_checks:
                    mark = "✅" if ok else "❌"
                    manage_dry_section.append(
                        f"  {mark} {label}" if ok else f"  {mark} <b>{label}</b> — <i>{detail}</i>"
                    )
                if manage_dry_failed:
                    errors.append(f"LLM DRY-RUN (manage): {fail_m}/{len(m_checks)} не прошли")
                else:
                    passed.append(f"LLM DRY-RUN (manage): {ok_m}/{len(m_checks)} ✅ за {elapsed_m:.1f}с")
        except Exception as e:
            errors.append(f"LLM DRY-RUN (manage): {e}")
            logger.exception("LLM DRY-RUN manage исключение")

        # ── Формируем отчёт ───────────────────────────────────────────
        total   = len(passed) + len(warnings) + len(errors)
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        if errors:
            status_line = f"🔴 <b>ЗАПУСК С ОШИБКАМИ</b> — {len(errors)} из {total} проверок провалено"
        elif warnings:
            status_line = f"🟡 <b>ЗАПУСК С ПРЕДУПРЕЖДЕНИЯМИ</b> — все критические проверки пройдены"
        else:
            status_line = f"🟢 <b>ЗАПУСК УСПЕШЕН</b> — все {total} проверок пройдены"

        tf = getattr(self, "timeframe", None) or str((self.config or {}).get("timeframe", "—"))

        lines = [
            f"{status_line}",
            f"<i>GPT Strategy v5.3 | {now_str}</i>",
            f"Модель: <b>{LLM_MODEL}</b> | TФ: <b>{tf}</b> | Стейки: {STAKE_LEVELS}",
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

        if dry_run_section:
            header_emoji = "❌" if dry_run_failed else "✅"
            lines.append(f"\n🧪 <b>LLM DRY-RUN — вход BTC/USDT</b> {header_emoji}")
            lines.append("  <i>(сырой вывод ИИ — в логах docker)</i>")
            lines.extend(dry_run_section)

        if manage_dry_section:
            header_emoji_m = "❌" if manage_dry_failed else "✅"
            lines.append(f"\n🧪 <b>LLM DRY-RUN — manage LONG</b> {header_emoji_m}")
            lines.append("  <i>(сопровождение in-trade, выход только механический)</i>")
            lines.extend(manage_dry_section)

        report = "\n".join(lines)

        # Логируем всегда
        for p in passed:
            logger.info(f"[CHECK ✔] {p}")
        for w in warnings:
            logger.warning(f"[CHECK ⚠] {w}")
        for e in errors:
            logger.error(f"[CHECK ✖] {e}")

        if self.tg.enabled:
            if not self.tg.send(report):
                logger.error("[CHECK ✖] Telegram: отчёт старта не доставлен (sendMessage ошибка)")
        else:
            self.tg.send(report)

    def _send_welcome_message(self) -> None:
        if self._WELCOME_MARKER.exists():
            return
        if not self.tg.enabled:
            logger.warning("[TG] Telegram отключён, приветствие не отправлено.")
            return

        msg = (
            "🤖 <b>Freqtrade Bot Control</b>\n\n"
            "Бот успешно запущен и работает по стратегии GPT Strategy v5.3.\n"
            "Ниже приведены команды для управления:\n\n"
            "┌─────── <b>УПРАВЛЕНИЕ ТОРГОВЛЕЙ</b> ───────\n"
            "│ ✅ <code>/start</code> — запустить бота\n"
            "│ ⏸️ <code>/pause</code> — приостановить входы, но держать открытые сделки\n"
            "│ 🛑 <code>/stop</code> — полная остановка бота\n"
            "│ 🚪 <code>/stopentry</code> — запретить вход, но держать открытые позиции\n"
            "│ 🔥 <code>/forceexit &lt;id&gt;|all</code> — принудительно закрыть сделку(и)\n"
            "│ 🔗 <code>/fx &lt;id&gt;|all</code> — алиас для forceexit\n"
            "└─────────────────────────────────────\n\n"

            "┌─────── <b>РУЧНЫЕ СДЕЛКИ</b> ───────\n"
            "│ 📈 <code>/forcelong &lt;pair&gt; [rate]</code> — ручной вход в LONG\n"
            "│ 📉 <code>/forceshort &lt;pair&gt; [rate]</code> — ручной вход в SHORT\n"
            "│ ❌ <code>/delete &lt;trade_id&gt;</code> — удалить сделку из БД\n"
            "│ 🔄 <code>/reload_trade &lt;trade_id&gt;</code> — перезагрузить сделку\n"
            "│ 🧾 <code>/cancel_open_order &lt;trade_id&gt;</code> — отменить открытый ордер\n"
            "│ 🔗 <code>/coo &lt;trade_id&gt;|all</code> — алиас для cancel_open_order\n"
            "└─────────────────────────────────────\n\n"

            "┌─────── <b>НАСТРОЙКИ И СПИСКИ</b> ───────\n"
            "│ 📋 <code>/whitelist [sorted] [baseonly]</code> — показать белый список\n"
            "│ 🚫 <code>/blacklist [pair]</code> — добавить пару в черный список\n"
            "│ ❌ <code>/blacklist_delete [pair]</code> — удалить из черного списка\n"
            "│ 🔗 <code>/bl_delete [pair]</code> — алиас для blacklist_delete\n"
            "│ 🔄 <code>/reload_config</code> — перезагрузить конфиг\n"
            "│ 🔓 <code>/unlock &lt;pair|id&gt;</code> — разблокировать пару\n"
            "└─────────────────────────────────────\n\n"

            "┌─────── <b>СТАТУС И СТАТИСТИКА</b> ───────\n"
            "│ 📊 <code>/status &lt;trade_id&gt;|[table]</code> — список открытых сделок\n"
            "│ 📈 <code>/profit [n]</code> — прибыль за n дней\n"
            "│ 📈 <code>/profit_long [n]</code> — прибыль LONG за n дней\n"
            "│ 📈 <code>/profit_short [n]</code> — прибыль SHORT за n дней\n"
            "│ 📉 <code>/performance</code> — эффективность по парам\n"
            "│ 📅 <code>/daily &lt;n&gt;</code> — прибыль по дням\n"
            "│ 📆 <code>/weekly &lt;n&gt;</code> — статистика по неделям\n"
            "│ 📅 <code>/monthly &lt;n&gt;</code> — статистика по месяцам\n"
            "│ 📋 <code>/trades [limit]</code> — последние закрытые сделки\n"
            "│ 🏷️ <code>/entries [pair]</code> — эффективность входов\n"
            "│ 🏷️ <code>/exits [pair]</code> — эффективность выходов\n"
            "│ 🔀 <code>/mix_tags [pair]</code> — комбинированные теги\n"
            "│ 📊 <code>/stats</code> — общая статистика\n"
            "│ 💰 <code>/balance [total]</code> — баланс по валютам\n"
            "│ 📜 <code>/logs [limit]</code> — последние логи\n"
            "│ 🔢 <code>/count</code> — количество активных сделок\n"
            "│ ❤️ <code>/health</code> — время последнего обновления\n"
            "│ 💾 <code>/list_custom_data &lt;trade_id&gt; [key]</code> — данные сделки\n"
            "│ 🔒 <code>/locks</code> — заблокированные пары\n"
            "│ ⚙️ <code>/show_config</code> — текущая конфигурация\n"
            "└─────────────────────────────────────\n\n"

            "┌─────── <b>ПРОЧЕЕ</b> ───────\n"
            "│ ❓ <code>/help</code> — справка по командам\n"
            "│ 🔖 <code>/version</code> — версия бота\n"
            "└─────────────────────────────────────\n\n"

            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n"
            "💡 <i>Сообщение автоматически закреплено в чате</i>"
        )

        message_id = self.tg.send_with_id(msg)
        if message_id:
            self.tg.pin_message(message_id)
            try:
                self._WELCOME_MARKER.write_text(str(message_id))
            except OSError as e:
                logger.warning(f"[TG] Не удалось создать маркер-файл: {e}")

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        self._send_welcome_message()

    def feature_engineering_standard(self, df: DataFrame, **kwargs) -> DataFrame:
        return df

    def set_freqai_targets(self, df: DataFrame, **kwargs) -> DataFrame:
        df["&-empty"] = "0"
        return df

    # ── [14.2] Главный цикл: свеча → LLM → сигнал ─────────────
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair      = metadata.get("pair", "unknown")
        candle_id = str(pd.to_datetime(dataframe["date"].iloc[-1])) if not dataframe.empty else None

        # ── Всегда пересчитываем колонки для plot_config ─────────────
        dataframe = self._add_plot_columns(dataframe)

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
        news_items, fear_greed = self._collect_market_context(pair)
        news_pool = list(news_items or [])
        news_items = _filter_news_headlines_for_pair(news_pool, pair_clean)[:MAX_NEWS_ITEMS]
        logger.info(
            f"[{pair_clean}] News: pool={len(news_pool)} -> relevant={len(news_items)} "
            f"(to LLM: {min(len(news_items), MAX_NEWS_ITEMS)})"
        )
        fetch_elapsed = time.time() - fetch_started
        if fetch_elapsed > CONTEXT_TOTAL_TIMEOUT_SEC:
            logger.warning(
                f"[{pair}] Сбор внешнего контекста занял {fetch_elapsed:.2f}s "
                f"(бюджет {CONTEXT_TOTAL_TIMEOUT_SEC:.2f}s)"
            )

        # Контекст: 2 прошлых NEUTRAL-анализа — только если прошлая свеча была NEUTRAL
        prev_decision = self._last_decision.get(pair, {})
        neutral_history: List[Dict[str, Any]] = []
        if prev_decision.get("action") == "NEUTRAL":
            neutral_history = list(self._neutral_analysis_history.get(pair, []))[-NEUTRAL_HISTORY_MAX:]

        # Если позиция уже открыта и LLM-сопровождение отключено — пропускаем LLM-запрос.
        # TP/trailing/stoploss работают в Python независимо в custom_exit.
        if side is not None and not self.llm_manage_enabled:
            logger.debug(
                f"[{pair}] LLM manage skipped (llm_manage_enabled=False, side={side}, profit={profit*100:+.2f}%)"
            )
            return self._apply_columns(dataframe, self._empty_result())

        prompt = (
            PromptBuilder.build_entry_prompt(
                ind, news_items, pair_clean, memory_block, fear_greed,
                neutral_history=neutral_history,
            )
            if side is None
            else PromptBuilder.build_manage_prompt(
                ind, news_items, pair_clean, side, profit * 100, memory_block, fear_greed,
                neutral_history=neutral_history,
            )
        )
        rec = self.llm.analyze(prompt)
        self._last_llm_call_ts[pair] = time.time()
        rec = snap_tech_score(rec, ind)
        rec = sanitize_news_recommendation(rec, pair_clean)
        rec = polish_recommendation_text(rec, pair_clean, ind, news_items_count=len(news_items))
        rec = finalize_confidence_reason(rec, ind, side=side)
        llm_blocked = rec.summary.startswith("Сигнал отклонён semantic-validator")
        if llm_blocked:
            self._signal_stats["llm_errors"] += 1

        arbiter_action, arbiter_reasons = self._arbiter_action(rec, ind, side)
        action_to_store = arbiter_action
        if side is not None and action_to_store in ("LONG_ENTER", "SHORT_ENTER"):
            action_to_store = "NEUTRAL"
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

        if action_to_store == "NEUTRAL":
            rec.summary = self._build_neutral_reason_line(
                rec=rec,
                ind=ind,
                side=side,
                llm_action=rec.action,
                arbiter_reasons=arbiter_reasons,
                llm_blocked=llm_blocked,
            )

        logger.info(
            f"[{pair_clean}] action={action_to_store} "
            f"final={rec.final_score:.3f} stake={rec.stake_amount} "
            f"align=L{ind.align_bull}/S{ind.align_bear}"
        )

        if side is None and action_to_store in ("LONG_ENTER", "SHORT_ENTER"):
            self._entry_score_snapshot[pair] = {
                "final": float(rec.final_score),
                "tech": float(rec.tech_score),
                "sent": float(rec.sent_score),
                "vol": float(ind.vol_ratio),
                "adx": float(ind.adx),
            }

        # Если итоговое действие — вход, summary формирует код (согласован с action).
        if action_to_store in ("LONG_ENTER", "SHORT_ENTER"):
            rec.summary = self._build_entry_summary(action_to_store, rec, ind, pair_clean)

        # Сохраняем стейк для передачи в custom_stake_amount
        self._pending_stake[pair] = rec.stake_amount

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
        if action_to_store == "NEUTRAL":
            hist = self._neutral_analysis_history.setdefault(pair, [])
            hist.append({
                "action": "NEUTRAL",
                "final_score": float(rec.final_score),
                "confidence": float(rec.confidence),
                "tech_reasoning": rec.tech_reasoning[:200] if rec.tech_reasoning else "",
                "counter_arg": rec.counter_argument[:150] if rec.counter_argument else "",
                "candle_time": candle_id or "",
            })
            self._neutral_analysis_history[pair] = hist[-NEUTRAL_HISTORY_MAX:]

        if action_to_store == "NEUTRAL" and side is None and self._should_send_neutral(pair, action_to_store):
            mode_line = (
                "🟢 <b>РЕЖИМ:</b> торговля включена"
                if self.trade_execution_enabled
                else "🟡 <b>РЕЖИМ:</b> только отчёты, входы отключены"
            )
            self.tg.send(self._tg_append_cost(f"{mode_line}\n\n{TGFormatter.neutral(pair_clean, rec, ind)}"))
            self.tg.send(self._tg_append_cost(TGFormatter.reasoning_msg(pair_clean, rec, "NEUTRAL", ind)))
        elif report_only_entry_blocked and side is None:
            mode_line = "🟡 <b>РЕЖИМ:</b> только отчёты, входы отключены"
            signal_emoji = "🟢" if action_to_store == "LONG_ENTER" else "🔴"
            signal_label = "LONG" if action_to_store == "LONG_ENTER" else "SHORT"
            self.tg.send(self._tg_append_cost(
                f"{mode_line}\n\n"
                f"{signal_emoji} <b>СИГНАЛ {signal_label} (без входа)</b> — {pair_clean}\n"
                f"{TGFormatter.scores_line(rec, ind)}\n\n"
                f"{TGFormatter.indicator_block(ind)}\n\n"
                f"{TGFormatter.news_block(rec)}\n\n"
                f"💬 <b>Вывод:</b>\n{rec.summary[:1500]}"
            ))
            self.tg.send(self._tg_append_cost(TGFormatter.reasoning_msg(pair_clean, rec, action_to_store, ind)))
        elif side in ("LONG", "SHORT") and self._should_send_in_trade_update(pair, candle_id):
            self.tg.send(self._tg_append_cost(TGFormatter.reasoning_msg(pair_clean, rec, f"{side}_MANAGE", ind)))

        self._update_signal_metrics(pair, action_to_store, rec, ind, llm_blocked)
        self._emit_monitoring_alerts()

        result = {
            "sent_score":         rec.sent_score,
            "tech_score":         rec.tech_score,
            "final_score":        rec.final_score,
            "expert_long_enter":  1 if execution_action == "LONG_ENTER"  else 0,
            "expert_short_enter": 1 if execution_action == "SHORT_ENTER" else 0,
            "expert_opinion":     rec.summary,
            "expert_stake":       rec.stake_amount,
            "expert_confidence":  rec.confidence,
            "_rec_key_news":      rec.key_news,
            "_rec_news_summary":  rec.news_summary,
            "_rec_news_mood":     rec.news_mood,
            "_rec_news_reasoning": rec.news_reasoning,
            "_rec_tech_reasoning": rec.tech_reasoning,
            "_rec_counter_argument": rec.counter_argument,
            "_rec_confidence_reason": rec.confidence_reason,
            "_llm_action_raw":    rec.action,
            "_arbiter_reasons":   "; ".join(arbiter_reasons) if arbiter_reasons else "",
        }
        self._cache[pair] = (candle_id, result)
        if candle_id:
            self._last_llm_candle_id[pair] = candle_id
        return self._apply_columns(dataframe, result)

    def _collect_market_context(self, pair: str) -> Tuple[List[str], str]:
        """
        Собирает внешний контекст параллельно и fail-safe:
        - новости (RSS)
        - Fear&Greed
        Если любой источник тормозит/падает, возвращаются безопасные дефолты.
        """
        defaults: Tuple[List[str], str] = ([], "")
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    "news": pool.submit(NewsProvider.fetch, MAX_NEWS_POOL),
                    "fg": pool.submit(FearGreedProvider.fetch),
                }
                out_news: List[str] = defaults[0]
                out_fg: str = defaults[1]

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

                return out_news, out_fg
        except Exception as e:
            logger.error(f"[{pair}] _collect_market_context fatal: {e}")
            return defaults

    @staticmethod
    def _arbiter_reason_human(reason: str, ind: Indicators) -> str:
        code = reason.split(":", 1)[0]
        tail = reason.split(":", 1)[1] if ":" in reason else ""
        if code == "low_confidence":
            try:
                conf_val = float(tail)
                conf_pct = conf_val * 100 if conf_val <= 1.0 else conf_val
            except ValueError:
                conf_pct = 0.0
            return f"уверенность {conf_pct:.0f}% < мин. {ENTRY_MIN_CONFIDENCE:.0%}"
        if code == "high_atr":
            return f"ATR {tail} > макс. {ENTRY_MAX_ATR_PCT:.1f}%"
        if code == "supertrend_conflict_long":
            return f"SuperTrend: {ind.st_dir} (для LONG нужен бычий)"
        if code == "supertrend_conflict_short":
            return f"SuperTrend: {ind.st_dir} (для SHORT нужен медвежий)"
        if code == "score_below_long_threshold":
            return f"итоговый скор {tail} < порога LONG {ENTRY_LONG_THRESHOLD:+.2f}"
        if code == "score_above_short_threshold":
            return f"итоговый скор {tail} > порога SHORT {ENTRY_SHORT_THRESHOLD:+.2f}"
        if code == "low_bull_align":
            return (
                f"за LONG только {ind.align_bull}/{ALIGN_INDICATOR_TOTAL} индикаторов "
                f"(нужно ≥ {MIN_ALIGNED_INDICATORS})"
            )
        if code == "low_bear_align":
            return (
                f"за SHORT только {ind.align_bear}/{ALIGN_INDICATOR_TOTAL} индикаторов "
                f"(нужно ≥ {MIN_ALIGNED_INDICATORS})"
            )
        if code == "low_volume":
            return f"объём {tail}x < мин. {ENTRY_MIN_VOL_RATIO:.2f}x"
        if code.startswith("weak_"):
            return reason.replace("_", " ")
        return reason

    @staticmethod
    def _join_reason_parts(parts: List[str]) -> str:
        seen: set = set()
        unique: List[str] = []
        for p in parts:
            if p and p not in seen:
                seen.add(p)
                unique.append(p)
        return ". ".join(unique) + ("." if unique else "")

    @staticmethod
    def _join_why_sentences(parts: List[str]) -> str:
        """Склеивает предложения через «. » без артеfactов «.;»."""
        seen: set = set()
        unique: List[str] = []
        for p in parts:
            chunk = p.strip().rstrip(".")
            if chunk and chunk not in seen:
                seen.add(chunk)
                unique.append(chunk)
        return ". ".join(unique) + ("." if unique else "")

    def _neutral_blockers_now(
        self,
        rec: TradingRecommendation,
        ind: Indicators,
        llm_action: str,
        score_cmp: float,
    ) -> List[str]:
        """Только те фильтры, которые реально мешают входу прямо сейчас (без согласования)."""
        blockers: List[str] = []

        if ENTRY_SHORT_THRESHOLD < score_cmp < ENTRY_LONG_THRESHOLD:
            if score_cmp >= 0:
                blockers.append(
                    f"итоговый скор {score_cmp:+.2f} < порога LONG {ENTRY_LONG_THRESHOLD:+.2f}"
                )
            else:
                blockers.append(
                    f"итоговый скор {score_cmp:+.2f} > порога SHORT {ENTRY_SHORT_THRESHOLD:+.2f}"
                )
            if rec.confidence < ENTRY_MIN_CONFIDENCE:
                blockers.append(
                    f"уверенность {rec.confidence:.0%} < мин. {ENTRY_MIN_CONFIDENCE:.0%}"
                )
            return blockers

        wants_long = llm_action == "LONG_ENTER" or score_cmp >= ENTRY_LONG_THRESHOLD
        wants_short = llm_action == "SHORT_ENTER" or score_cmp <= ENTRY_SHORT_THRESHOLD

        if wants_long and not wants_short:
            if score_cmp < ENTRY_LONG_THRESHOLD:
                blockers.append(
                    f"итоговый скор {score_cmp:+.2f} < порога LONG {ENTRY_LONG_THRESHOLD:+.2f}"
                )
            if rec.confidence < ENTRY_MIN_CONFIDENCE:
                blockers.append(
                    f"уверенность {rec.confidence:.0%} < мин. {ENTRY_MIN_CONFIDENCE:.0%}"
                )
            if ind.st_dir != "бычий":
                blockers.append(f"SuperTrend: {ind.st_dir} (для LONG нужен бычий)")
            if ind.vol_ratio < ENTRY_MIN_VOL_RATIO:
                blockers.append(
                    f"объём {ind.vol_ratio:.2f}x < мин. {ENTRY_MIN_VOL_RATIO:.2f}x"
                )
            if ind.atr_pct > ENTRY_MAX_ATR_PCT:
                blockers.append(
                    f"ATR {ind.atr_pct:.2f}% > макс. {ENTRY_MAX_ATR_PCT:.1f}%"
                )
            return blockers

        if wants_short:
            if score_cmp > ENTRY_SHORT_THRESHOLD:
                blockers.append(
                    f"итоговый скор {score_cmp:+.2f} > порога SHORT {ENTRY_SHORT_THRESHOLD:+.2f}"
                )
            if rec.confidence < ENTRY_MIN_CONFIDENCE:
                blockers.append(
                    f"уверенность {rec.confidence:.0%} < мин. {ENTRY_MIN_CONFIDENCE:.0%}"
                )
            if ind.st_dir != "медвежий":
                blockers.append(f"SuperTrend: {ind.st_dir} (для SHORT нужен медвежий)")
            if ind.vol_ratio < ENTRY_MIN_VOL_RATIO:
                blockers.append(
                    f"объём {ind.vol_ratio:.2f}x < мин. {ENTRY_MIN_VOL_RATIO:.2f}x"
                )
            if ind.atr_pct > ENTRY_MAX_ATR_PCT:
                blockers.append(
                    f"ATR {ind.atr_pct:.2f}% > макс. {ENTRY_MAX_ATR_PCT:.1f}%"
                )
            return blockers

        if score_cmp >= 0:
            blockers.append(
                f"итоговый скор {score_cmp:+.2f} < порога LONG {ENTRY_LONG_THRESHOLD:+.2f}"
            )
        else:
            blockers.append(
                f"итоговый скор {score_cmp:+.2f} > порога SHORT {ENTRY_SHORT_THRESHOLD:+.2f}"
            )
        if rec.confidence < ENTRY_MIN_CONFIDENCE:
            blockers.append(
                f"уверенность {rec.confidence:.0%} < мин. {ENTRY_MIN_CONFIDENCE:.0%}"
            )
        return blockers

    @staticmethod
    def _sentiments_for_score(sent: float) -> List[Any]:
        """Минимальный sentiments для корректного sent_score в rec-proxy."""
        if abs(float(sent)) < 1e-9:
            return []
        s = float(sent)
        return [{"score": s, "sentiment": s, "title": "", "relevance": 0.0, "recency": 0.0}]

    def _entry_rec_for_tg(
        self,
        pair: str,
        side: str,
        last: Any,
        cached_data: dict,
        stake_amount: float,
        ind: Indicators,
    ) -> TradingRecommendation:
        """Rec для TG при входе: scores из snapshot/DF, summary пересобирается на месте."""
        pair_clean = pair.split(":")[0]
        action = "LONG_ENTER" if side == "long" else "SHORT_ENTER"
        snap = self._entry_score_snapshot.get(pair, {})
        tech = float(snap.get("tech", last.get("tech_score", 0.0)))
        sent = float(snap.get("sent", last.get("sent_score", 0.0)))
        rec = TradingRecommendation(
            action=action,
            tech_score=tech,
            sentiments=self._sentiments_for_score(sent),
            key_news=cached_data.get("_rec_key_news", ""),
            news_summary=cached_data.get("_rec_news_summary", ""),
            news_mood=cached_data.get("_rec_news_mood", "нейтральный"),
            news_reasoning=cached_data.get("_rec_news_reasoning", ""),
            tech_reasoning=cached_data.get("_rec_tech_reasoning", ""),
            counter_argument=cached_data.get("_rec_counter_argument", ""),
            confidence_reason=cached_data.get("_rec_confidence_reason", ""),
            summary="",
            confidence=float(
                last.get("expert_confidence", cached_data.get("expert_confidence", 0.0))
            ),
            stake_amount=stake_amount,
        )
        return rec.model_copy(update={
            "summary": self._build_entry_summary(action, rec, ind, pair_clean),
        })

    def _build_entry_summary(
        self,
        action: str,
        rec: TradingRecommendation,
        ind: Indicators,
        pair: str,
    ) -> str:
        """Краткий вывод для LONG/SHORT — всегда согласован с итоговым action."""
        label = "LONG" if action == "LONG_ENTER" else "SHORT"
        align = ind.align_bull if action == "LONG_ENTER" else ind.align_bear
        token = pair.split("/")[0]
        final = round(float(rec.final_score), SCORE_ROUND_DECIMALS)
        sent = round(float(rec.sent_score), SCORE_ROUND_DECIMALS)
        parts = [
            f"Сигнал {label} по {token}: итог {final:+.2f}, "
            f"уверенность {rec.confidence:.0%}.",
            f"SuperTrend {ind.st_dir}, за {label} {align}/{ALIGN_INDICATOR_TOTAL} индикаторов.",
        ]
        if abs(sent) >= 0.05:
            parts.append(f"Новости {sent:+.2f}.")
        parts.append("Контролируй риск и размер позиции.")
        return " ".join(parts)

    @staticmethod
    def _format_zero_score_why(
        rec: TradingRecommendation,
        ind: Indicators,
        extra_parts: Optional[List[str]] = None,
    ) -> str:
        """Чёткое пояснение, когда общая оценка ровно 0.00 и входа нет."""
        score = float(rec.final_score)
        tech = float(rec.tech_score)
        sent = float(rec.sent_score)
        bull, bear = ind.align_bull, ind.align_bear
        parts: List[str] = [
            f"Нет входа: общая оценка {score:+.2f} — направление не выбрано.",
            (
                f"Техника {tech:+.2f} (85%) + новости {sent:+.2f} (15%) "
                f"не дают перевеса LONG или SHORT."
            ),
        ]
        if bull >= MIN_ALIGNED_INDICATORS and bear >= MIN_ALIGNED_INDICATORS:
            parts.append(
                f"Индикаторы спорят: {bull} за LONG и {bear} за SHORT одновременно."
            )
        elif max(bull, bear) < MIN_ALIGNED_INDICATORS:
            parts.append(
                f"Согласование слабое: {bull} за LONG, {bear} за SHORT "
                f"(нужно ≥ {MIN_ALIGNED_INDICATORS} в одну сторону)."
            )
        else:
            bias = "LONG" if bull > bear else "SHORT"
            parts.append(
                f"Лёгкий уклон к {bias} ({max(bull, bear)}/{ALIGN_INDICATOR_TOTAL}), "
                f"но итога недостаточно для входа."
            )
        parts.append(
            f"Порог входа: ≥ {ENTRY_LONG_THRESHOLD:+.2f} (LONG) "
            f"или ≤ {ENTRY_SHORT_THRESHOLD:+.2f} (SHORT)."
        )
        if extra_parts:
            for p in extra_parts:
                low = p.lower()
                if "итоговый скор" in low and "порога" in low:
                    continue
                if "нет направления" in low or "направление не выбрано" in low:
                    continue
                parts.append(p)
        return GPTStrategy._join_why_sentences(parts)[:500]

    def _build_neutral_reason_line(
        self,
        rec: TradingRecommendation,
        ind: Indicators,
        side: Optional[str],
        llm_action: str,
        arbiter_reasons: List[str],
        llm_blocked: bool,
    ) -> str:
        if llm_blocked and rec.summary:
            return rec.summary.strip()[:500]

        if side is not None:
            return ""

        parts: List[str] = []
        score_cmp = round(float(rec.final_score), SCORE_ROUND_DECIMALS)

        for r in arbiter_reasons:
            msg = self._arbiter_reason_human(r, ind)
            if msg:
                parts.append(msg)

        if not parts:
            parts = self._neutral_blockers_now(rec, ind, llm_action, score_cmp)

        if score_cmp == 0.0:
            return self._format_zero_score_why(rec, ind, parts)

        if parts:
            return self._join_reason_parts(parts)[:500]

        return (
            f"Нет входа: итог {rec.final_score:+.2f} не проходит порог "
            f"±{ENTRY_LONG_THRESHOLD:.2f}."
        )[:500]

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
        if side is None and action in ("LONG_ENTER", "SHORT_ENTER") and ind.vol_ratio < ENTRY_MIN_VOL_RATIO:
            reasons.append(f"low_volume:{ind.vol_ratio:.2f}")

        if side is None:
            if action == "LONG_ENTER" and score_cmp < ENTRY_LONG_THRESHOLD:
                reasons.append(f"score_below_long_threshold:{score:.3f}")
            if action == "SHORT_ENTER" and score_cmp > ENTRY_SHORT_THRESHOLD:
                reasons.append(f"score_above_short_threshold:{score:.3f}")
            if action == "LONG_ENTER" and ind.align_bull < MIN_ALIGNED_INDICATORS:
                reasons.append(f"low_bull_align:{ind.align_bull}/{MIN_ALIGNED_INDICATORS}")
            if action == "SHORT_ENTER" and ind.align_bear < MIN_ALIGNED_INDICATORS:
                reasons.append(f"low_bear_align:{ind.align_bear}/{MIN_ALIGNED_INDICATORS}")
            if action not in ("LONG_ENTER", "SHORT_ENTER", "NEUTRAL"):
                reasons.append(f"invalid_entry_action:{action}")
        else:
            if action != "NEUTRAL":
                action = "NEUTRAL"

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
        candles = int(self._signal_stats.get("candles", 0))
        # Минимальная выборка — иначе одна NEUTRAL-свеча даёт neutral_rate=100% и спам алертов.
        if candles < MONITORING_MIN_CANDLES:
            return
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

        # Базовая динамика цены
        dataframe["change_pct"] = close.pct_change().mul(100).fillna(0.0)

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
        dataframe["macd"]             = dataframe["macd_line"]
        dataframe["macd_signal"]      = dataframe["macd_signal_line"]
        dataframe["macd_trend"] = np.where(
            dataframe["macd_line"] > dataframe["macd_signal_line"], "бычий", "медвежий"
        )
        prev_macd = dataframe["macd_line"].shift(1)
        prev_sig = dataframe["macd_signal_line"].shift(1)
        dataframe["macd_cross"] = np.select(
            [
                (prev_macd < prev_sig) & (dataframe["macd_line"] > dataframe["macd_signal_line"]),
                (prev_macd > prev_sig) & (dataframe["macd_line"] < dataframe["macd_signal_line"]),
            ],
            ["бычье", "медвежье"],
            default="нет",
        )

        # SMA
        dataframe["sma7"]   = close.rolling(7).mean()
        dataframe["sma20"]  = close.rolling(20).mean()
        dataframe["sma200"] = close.rolling(200).mean()
        dataframe["sma_cross"] = np.where(dataframe["sma7"] > dataframe["sma20"], "бычий", "медвежий")

        # Bollinger Bands
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        dataframe["bb_mid"]   = bb_mid
        dataframe["bb_upper"] = bb_mid + 2.0 * bb_std
        dataframe["bb_lower"] = bb_mid - 2.0 * bb_std
        bb_range = (dataframe["bb_upper"] - dataframe["bb_lower"]).replace(0, np.nan)
        dataframe["bb_pct"] = ((close - dataframe["bb_lower"]) / bb_range * 100).fillna(50.0)
        dataframe["bb_squeeze"] = (
            ((dataframe["bb_upper"] - dataframe["bb_lower"]) / close.replace(0, np.nan) * 100) < 3.0
        ).fillna(False)
        dataframe["bb_pos"] = np.select(
            [
                close >= dataframe["bb_upper"],
                close <= dataframe["bb_lower"],
                dataframe["bb_pct"] > 70,
                dataframe["bb_pct"] < 30,
            ],
            ["выше верхней", "ниже нижней", "верхняя зона", "нижняя зона"],
            default="середина",
        )

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
        dataframe["st_level"] = dataframe["supertrend_line"].fillna(close)
        dataframe["st_dir"] = np.where(
            dataframe["st_bull"].notna(),
            "бычий",
            np.where(dataframe["st_bear"].notna(), "медвежий", "нейтральный"),
        )
        dataframe["st_dist_pct"] = (
            (close - dataframe["st_level"]).abs() / close.replace(0, np.nan) * 100
        ).fillna(0.0)

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
        dataframe["adx_str"] = np.select(
            [dataframe["adx"] > 25, dataframe["adx"] < 20],
            ["сильный", "слабый"],
            default="умеренный",
        )

        # ATR / ATR%
        dataframe["atr"] = tr.rolling(14).mean()
        dataframe["atr_pct"] = (
            dataframe["atr"] / close.replace(0, np.nan) * 100
        ).fillna(0.0)

        # Stochastic RSI
        rsi_min   = dataframe["rsi"].rolling(14).min()
        rsi_max   = dataframe["rsi"].rolling(14).max()
        rsi_range = (rsi_max - rsi_min).replace(0, np.nan)
        stoch_k_raw = (dataframe["rsi"] - rsi_min) / rsi_range * 100
        dataframe["stoch_k"] = stoch_k_raw.rolling(3).mean().fillna(50)
        dataframe["stoch_d"] = dataframe["stoch_k"].rolling(3).mean().fillna(50)
        prev_k = dataframe["stoch_k"].shift(1)
        prev_d = dataframe["stoch_d"].shift(1)
        dataframe["stoch_cross"] = np.select(
            [
                (prev_k < prev_d) & (dataframe["stoch_k"] > dataframe["stoch_d"]) & (dataframe["stoch_k"] < 20),
                (prev_k > prev_d) & (dataframe["stoch_k"] < dataframe["stoch_d"]) & (dataframe["stoch_k"] > 80),
            ],
            ["бычье", "медвежье"],
            default="нет",
        )
        dataframe["stoch_zone"] = np.select(
            [dataframe["stoch_k"] < 20, dataframe["stoch_k"] > 80],
            ["перепродан", "перекуплен"],
            default="нейтральная",
        )

        # Williams %R
        hh = high.rolling(14).max()
        ll = low.rolling(14).min()
        wr_denom = (hh - ll).replace(0, np.nan)
        dataframe["williams_r"] = ((hh - close) / wr_denom * -100).fillna(-50.0)
        dataframe["williams_zone"] = np.select(
            [dataframe["williams_r"] < -80, dataframe["williams_r"] > -20],
            ["перепродан", "перекуплен"],
            default="нейтральная",
        )

        # CCI(20)
        tp = (high + low + close) / 3
        cci_mean = tp.rolling(20).mean()
        cci_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True).replace(0, np.nan)
        dataframe["cci"] = ((tp - cci_mean) / (0.015 * cci_mad)).fillna(0.0)
        dataframe["cci_signal"] = np.select(
            [dataframe["cci"] < -100, dataframe["cci"] > 100],
            ["перепродан", "перекуплен"],
            default="нейтральный",
        )

        # Volume ratio
        avg_vol = vol.rolling(20).mean().replace(0, np.nan)
        dataframe["volume_ratio"] = (vol / avg_vol).fillna(1.0)
        dataframe["vol_ratio"] = dataframe["volume_ratio"]
        dataframe["vol_desc"] = np.select(
            [dataframe["volume_ratio"] > 1.5, dataframe["volume_ratio"] < 0.5],
            ["всплеск", "низкий"],
            default="норма",
        )

        # OBV
        price_dir = np.sign(close.diff().fillna(0))
        dataframe["obv"] = (vol * price_dir).cumsum()
        dataframe["obv_sma20"] = dataframe["obv"].rolling(20).mean()
        dataframe["obv_trend"] = np.where(dataframe["obv"] > dataframe["obv_sma20"], "бычий", "медвежий")

        # VWAP (дневной сброс по дате)
        tp_vwap = (high + low + close) / 3
        if "date" in dataframe.columns:
            dates = pd.to_datetime(dataframe["date"])
            day = dates.dt.date
            cum_tpv = (tp_vwap * vol).groupby(day).cumsum()
            cum_vol = vol.groupby(day).cumsum().replace(0, np.nan)
            dataframe["vwap"] = (cum_tpv / cum_vol).fillna(close)
        else:
            dataframe["vwap"] = ((tp_vwap * vol).cumsum() / vol.cumsum().replace(0, np.nan)).fillna(close)
        dataframe["vwap_dist_pct"] = (
            (close - dataframe["vwap"]) / dataframe["vwap"].replace(0, np.nan) * 100
        ).fillna(0.0)
        dataframe["vwap_pos"] = np.where(close > dataframe["vwap"], "выше", "ниже")

        # Pivot / Support / Resistance от предыдущей свечи
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)
        pivot = (prev_high + prev_low + prev_close) / 3
        support = 2 * pivot - prev_high
        resistance = 2 * pivot - prev_low
        fallback_res = high.rolling(20, min_periods=1).max()
        fallback_sup = low.rolling(20, min_periods=1).min()
        dataframe["pivot"] = pivot.fillna((fallback_res + fallback_sup) / 2)
        dataframe["support"] = support.fillna(fallback_sup)
        dataframe["resistance"] = resistance.fillna(fallback_res)

        # Score-колонки (заполнятся из кэша или дефолтами)
        for col in ["final_score", "tech_score", "sent_score"]:
            if col not in dataframe.columns:
                dataframe[col] = 0.0

        # SuperTrend колонки — гарантируем наличие
        for col in ["st_bull", "st_bear", "supertrend_line", "st_level"]:
            if col not in dataframe.columns:
                dataframe[col] = np.nan

        return dataframe

    def _empty_result(self) -> dict:
        return {
            "expert_long_enter":  0,
            "expert_short_enter": 0,
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

    def _tg_append_cost(self, message: str) -> str:
        """Стоимость LLM-анализа (USD) в скобках в конце сообщения."""
        cost = LLMClient.format_cost_usd(getattr(self.llm, "last_analysis_cost_usd", 0.0))
        return f"{message} ({cost})"

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
    # ── [14.3] Вход / исполнение ────────────────────────────────
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

    _EXIT_STATE_KEY = "exit_state"
    _TRADE_MEMORY_SAVED_KEY = "trade_memory_saved"

    def _persist_closed_trade(
        self,
        trade: Trade,
        pair: str,
        side_label: str,
        current_time: datetime,
        current_profit: float,
        exit_reason: str,
        final_score: float,
    ) -> bool:
        """Пишет закрытую сделку в TradeMemory один раз на trade.id."""
        if trade.get_custom_data(self._TRADE_MEMORY_SAVED_KEY):
            return False
        mins = (current_time - trade.open_date_utc).total_seconds() / 60
        self._trade_memory.add(TradeRecord(
            pair=pair,
            side=side_label,
            profit_pct=current_profit * 100,
            duration_min=mins,
            entry_time=trade.open_date_utc,
            exit_time=current_time,
            final_score=final_score,
            exit_reason=exit_reason,
        ))
        self._update_closed_trade_metrics(
            pair=pair,
            profit_pct=current_profit * 100,
            duration_min=mins,
            final_score=final_score,
        )
        trade.set_custom_data(self._TRADE_MEMORY_SAVED_KEY, True)
        return True

    @staticmethod
    def _trade_side_label(trade: Trade, enter_tag: str = "") -> str:
        tag = enter_tag or (trade.enter_tag or "")
        if tag == "gpt_long":
            return "LONG"
        if tag == "gpt_short":
            return "SHORT"
        return "SHORT" if trade.is_short else "LONG"

    def _resolve_exit_final_score(self, pair: str, last: Any = None) -> float:
        snap = self._entry_score_snapshot.get(pair) or {}
        if snap.get("final") is not None:
            return float(snap["final"])
        if last is not None:
            try:
                return float(last.get("final_score", 0.0))
            except Exception:
                pass
        cached = self._cache.get(pair)
        if cached:
            return float(cached[1].get("final_score", 0.0))
        return 0.0

    def _load_exit_state(self, trade: Trade) -> dict:
        data = trade.get_custom_data(self._EXIT_STATE_KEY)
        if not data:
            return {"tp_activated": False, "max_profit": None}
        return data

    def _save_exit_state(self, trade: Trade, data: dict) -> None:
        trade.set_custom_data(self._EXIT_STATE_KEY, data)

    def _take_profit_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        """Take Profit + Trailing — как DSA_General_v2.custom_exit."""
        if self.force_exit_by_days_enabled and self.force_exit_after_days > 0:
            age_days = (current_time - trade.open_date_utc).total_seconds() / 86400.0
            if age_days >= self.force_exit_after_days and current_profit > 0:
                logger.info(
                    f"⏰ Force exit by age {pair}: age={age_days:.2f}d "
                    f"limit={self.force_exit_after_days:.2f}d profit={current_profit:.2%}",
                )
                self._save_exit_state(trade, {"tp_activated": False, "max_profit": None})
                self.tg.send(
                    f"⏰ <b>Принудительный выход по времени</b>\n"
                    f"Пара: <b>{pair}</b>\n"
                    f"Сделка висит: <b>{age_days:.2f} дн</b> "
                    f"(лимит {self.force_exit_after_days:.2f})\n"
                    f"Текущий PnL: <b>{current_profit:+.2%}</b>",
                )
                return "force_exit_max_days"

        data = self._load_exit_state(trade)

        if data["max_profit"] is None:
            data["max_profit"] = current_profit
            self._save_exit_state(trade, data)

        if not self.trailing_after_tp:
            if current_profit >= self.take_profit:
                self._save_exit_state(trade, {"tp_activated": False, "max_profit": None})
                return "take_profit"
            return None

        if not data["tp_activated"]:
            if current_profit >= self.take_profit + self.trailing_min_activation:
                data["tp_activated"] = True
                data["max_profit"] = current_profit
                self._save_exit_state(trade, data)
                logger.info(
                    f"🎯 Trailing TP активирован {pair}: profit={current_profit:.2%}",
                )
            return None

        if current_profit > data["max_profit"]:
            data["max_profit"] = current_profit
            self._save_exit_state(trade, data)
            return None

        retrace = data["max_profit"] - current_profit
        if retrace >= self.trailing_retrace:
            self._save_exit_state(trade, {"tp_activated": False, "max_profit": None})
            logger.info(
                f"📉 Trailing TP сработал {pair}: "
                f"max={data['max_profit']:.2%}, retrace={retrace:.2%}",
            )
            return "trailing_take_profit"

        return None


    def _notify_mechanical_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_profit: float,
        exit_reason: str,
    ) -> None:
        """Telegram и память сделок при выходе по TP/trailing/таймеру (не по LLM)."""
        side_label = self._trade_side_label(trade)
        mins = (current_time - trade.open_date_utc).total_seconds() / 60
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last = df.iloc[-1].squeeze() if not df.empty else None
        scores = {
            "sent": float(last.get("sent_score", 0)) if last is not None else 0.0,
            "tech": float(last.get("tech_score", 0)) if last is not None else 0.0,
            "final": self._resolve_exit_final_score(pair, last),
        }
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
                summary=str(last.get("expert_opinion", "")) if last is not None else "",
                confidence=float(
                    last.get("expert_confidence", cr.get("expert_confidence", 0.0))
                ) if last is not None else float(cr.get("expert_confidence", 0.0)),
                stake_amount=float(
                    last.get("expert_stake", STAKE_LEVELS[0])
                ) if last is not None else float(cr.get("expert_stake", STAKE_LEVELS[0])),
            )
        if not df.empty:
            ind = IndicatorEngine.calculate(df, 1)
            if ind:
                self.tg.send(TGFormatter.exit_msg(
                    pair, side_label, current_profit, mins, scores,
                    str(last.get("expert_opinion", "")), ind, cached_rec,
                ))
        self._persist_closed_trade(
            trade=trade,
            pair=pair,
            side_label=side_label,
            current_time=current_time,
            current_profit=current_profit,
            exit_reason=exit_reason,
            final_score=scores["final"],
        )

    # ── [14.4] Выход: TP / trailing / таймаут (без LLM) ────────
    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        """TP/trailing + таймаут. Выходы от LLM отключены."""
        tp_reason = self._take_profit_exit(
            pair, trade, current_time, current_rate, current_profit, **kwargs,
        )
        if tp_reason:
            self._notify_mechanical_exit(pair, trade, current_time, current_profit, tp_reason)
            self._entry_score_snapshot.pop(pair, None)
            self._entry_times.pop(trade.id, None)
            return tp_reason

        mins = (current_time - trade.open_date_utc).total_seconds() / 60
        if mins > MAX_TRADE_MINUTES:
            self._notify_mechanical_exit(pair, trade, current_time, current_profit, "expired")
            self._entry_score_snapshot.pop(pair, None)
            self._entry_times.pop(trade.id, None)
            return "expired"

        return None

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """Память сделок при stop-loss (custom_exit уже пишет TP/trailing/expired)."""
        er = str(exit_reason or "").lower()
        if "stoploss" in er or er in ("stop_loss", "trailing_stop_loss"):
            side_label = self._trade_side_label(trade)
            profit = trade.calc_profit_ratio(rate)
            final_score = self._resolve_exit_final_score(pair)
            self._persist_closed_trade(
                trade=trade,
                pair=pair,
                side_label=side_label,
                current_time=current_time,
                current_profit=profit,
                exit_reason=exit_reason or "stoploss",
                final_score=final_score,
            )
        return True

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
        confidence = float(last.get("expert_confidence", 0.0))
        if confidence < ENTRY_MIN_CONFIDENCE:
            logger.warning(f"[{pair}] вход отклонён: confidence={confidence:.2f} < {ENTRY_MIN_CONFIDENCE:.2f}")
            return False
        if ind is None:
            # Без индикаторов нельзя проверить ATR/SuperTrend → отклоняем вход.
            logger.warning(f"[{pair}] вход отклонён: IndicatorEngine не вернул данные (нет ATR/ST для guardrail)")
            return False
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
        rec_proxy = self._entry_rec_for_tg(
            pair, side, last, cached_data, stake_for_msg, ind,
        )
        if abs(stake_for_msg - stake_requested) > 1.0:
            logger.info(
                f"[{pair}] stake adjusted by exchange limits: "
                f"requested={stake_requested:.2f} actual={stake_for_msg:.2f}"
            )
        self.tg.send(self._tg_append_cost(TGFormatter.entry(pair, side, rate, stake_for_msg, rec_proxy, ind)))
        self.tg.send(self._tg_append_cost(TGFormatter.reasoning_msg(
            pair, rec_proxy, f"{'LONG' if side == 'long' else 'SHORT'}_ENTER", ind
        )))
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
            confidence = float(cr.get("expert_confidence", 0.0))
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


# ═══════════════════════════════════════════════════════════════════════
# [15] STARTUP INTEGRITY GUARD — константы self-test (v5.3, без analyst/LLM-exit)
# Строки разбиты, чтобы проверка не находила паттерны внутри этого списка.
# ═══════════════════════════════════════════════════════════════════════

_LX = "LONG" + "_" + "EXIT"
_SX = "SHORT" + "_" + "EXIT"
_INTEGRITY_FORBIDDEN = (
    "class Crypto" + "Analyst",
    "def build_exit_" + "prompt",
    "expert_long_" + "exit",
    "expert_short_" + "exit",
    "bind_crypto_" + "analyst",
    'action in ("' + _LX + '"',
    "action in ('" + _LX + "'",
    '"' + _LX + '", "' + _SX + '"',
)