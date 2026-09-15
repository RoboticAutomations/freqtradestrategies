# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file

"""
DSA General v2 — универсальная стратегия LONG + SHORT с DCA (Dollar Cost Averaging).

Направление     : LONG и SHORT (can_short = True)
Таймфрейм       : 1m (переопределяется через config.json)
Индикаторы      : RSI · MACD · SuperTrend · SMA (5/20/50/100/200) · Bollinger Bands · ADX · ATR · MFI · KST
Вход LONG       : RSI < 35 при volume > 0 (тег RSI_long)
Вход SHORT      : RSI > 65 при volume > 0 (тег RSI_short)
Направление     : управляется динамически через /marketdir (long|short|even|none)
                  none — разрешены оба направления
Выход           : Take Profit + Trailing Stop (custom_exit по trade.id)
DCA             : до 7 доливок для обоих направлений:
                  LONG  — триггер при падении от open_rate (-2%, -4%, -6%...)
                  SHORT — триггер при росте от open_rate (+2%, +4%, +6%...)
                  объём доливки = first_cost × ratio × scale^dca_count
Риск            : stoploss = -99% (фактически отключён, управляется через TP/trailing)
Принудительный  : выход по времени (force_exit_after_days) если включён и PnL > 0

Ключевые настройки (все переопределяются из config.json):
  force_exit_by_days_enabled   = False  — выход по времени в плюсе
  force_exit_after_days          = 5     — через N дней, только если PnL > 0
  safety_order_ratio         = 1.2    — множитель объёма доливки
  safety_order_max_count     = 7      — макс. число доливок
  safety_order_volume_scale  = 1.2    — масштабирование объёма между доливками
  price_deviation_initial    = 0.02   — начальное отклонение для первой доливки (2%)
  take_profit                = 0.02   — порог тейк-профита (2%)
  trailing_after_tp          = True   — трейлинг после достижения TP
  trailing_retrace           = 0.005  — откат для срабатывания трейлинга (0.5%)
  trailing_min_activation    = 0.02   — мин. профит для активации трейлинга

Telegram: приветственное сообщение с командами при первом запуске (закрепляется один раз).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from pandas import DataFrame
from typing import Dict, Optional, Tuple, Any
import logging
import math
import requests
import json
import re
import threading
import time

from freqtrade.strategy import IStrategy, Trade
from freqtrade.enums import ExitType, RunMode
import talib.abstract as ta

try:
    from freqtrade_client.ft_rest_client import FtRestClient
    FT_CLIENT_AVAILABLE = True
except Exception:
    FT_CLIENT_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None  # type: ignore[assignment]
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# ⚙️  НАСТРОЙКИ АЛЕРТОВ TELEGRAM — ВСЁ В ОДНОМ МЕСТЕ
# -----------------------------------------------------------------------------
# Все пороги/флаги ниже относятся ТОЛЬКО к доп. уведомлениям в Telegram и НЕ
# влияют на торговую логику (когда открывать/закрывать сделки). Меняются
# правкой этого файла — сознательно НЕ читаются из config.json, чтобы не
# зависеть от деплоя конфига на сервере. Собраны здесь специально, чтобы не
# искать их по всему файлу — класс стратегии просто ссылается на эти константы.
# =============================================================================

# ── Общее ────────────────────────────────────────────────────────────────
# Слать ли алерты (пп. 1-3, 12-17) в backtest/hyperopt. В live/dry-run эти
# алерты включены всегда (если задан TG токен/chat_id в config.json).
ALERT_SEND_IN_BACKTEST: bool = True

# ── Алерт 3: глубокий минус по позиции (эскалация, % от входа) ────────────
ALERT_DEEP_DRAWDOWN_THRESHOLDS: list = [-15.0, -20.0, -25.0, -30.0]

# ── Алерты 12-13: просадка общего баланса от пикового значения за всё время
# работы процесса (эскалация) / восстановление выше самого мягкого порога ──
ALERT_BALANCE_DRAWDOWN_THRESHOLDS: list = [-10.0, -15.0, -20.0, -25.0]

# ── Алерт 14: рост общей прибыли от стартового баланса процесса (эскалация)
ALERT_BALANCE_GROWTH_THRESHOLDS: list = [10.0, 20.0, 30.0, 50.0]

# ── Алерты 8-11 (строго live/dry-run, см. _live_alerts_enabled_now) ───────
ALERT_ORDER_LATENCY_THRESHOLD_SECONDS: float = 30.0

# ── Алерт 15: концентрация риска — доля позиции по ОДНОЙ паре от баланса ──
ALERT_PAIR_CONCENTRATION_THRESHOLD_PCT: float = 30.0

# ── Алерт 16: несколько пар одновременно в тяжёлом DCA ────────────────────
ALERT_MULTI_DCA_PAIR_PCT: float = 70.0        # с какого % доборов пара считается "в зоне риска"
ALERT_MULTI_DCA_MIN_PAIRS: int = 3            # сколько таких пар нужно, чтобы сработал алерт
ALERT_MULTI_DCA_COOLDOWN_MINUTES: float = 60.0  # защита от дребезга на границе порога

# ── Алерт 17: выход из тяжёлого DCA (восстановилась без новых доборов) ────
ALERT_HEAVY_DCA_PCT: float = 50.0             # с какого % доборов позиция считается "тяжёлой"
ALERT_HEAVY_DCA_RECOVERY_PNL_PCT: float = -2.0  # PnL (%), начиная с которого считаем "восстановилась"
ALERT_HEAVY_DCA_REARM_DROP_PCT: float = 2.0   # на сколько п.п. PnL должен уйти ОБРАТНО НИЖЕ
                                               # recovery_pnl, чтобы алерт "перезарядился" и мог
                                               # сработать снова при следующем восстановлении
                                               # (гистерезис — иначе PnL, дребезжащий у recovery_pnl,
                                               # спамил бы алертом туда-обратно)

# ── Алерт 18: DCA "застрял" — давно нет ни новых доборов, ни движения цены ─
ALERT_DCA_STUCK_HOURS: float = 6.0            # часов без движения после последнего добора
ALERT_DCA_STUCK_PRICE_PCT: float = 0.3        # цена должна остаться в этом диапазоне (%)
ALERT_DCA_STUCK_COOLDOWN_HOURS: float = 12.0  # не повторять по той же сделке чаще этого, пока
                                               # реально "застряло" — периодическое напоминание
ALERT_DCA_STUCK_HYSTERESIS_MULT: float = 2.0  # выйти из "застоя" можно только уйдя дальше, чем
                                               # ALERT_DCA_STUCK_PRICE_PCT * этот множитель — иначе
                                               # мелкий шум цены у самой границы 0.3% не считается
                                               # "разморозкой" и не даёт дребезжать туда-сюда

# ── Алерт 19: суммарная загрузка капитала во ВСЕХ открытых позициях ───────
ALERT_CAPITAL_LOAD_PCT: float = 70.0          # % от баланса, занятый всеми позициями сразу
ALERT_CAPITAL_LOAD_COOLDOWN_MINUTES: float = 60.0  # защита от дребезга на границе порога

# ── Алерт 20: резкий разворот PnL позиции (был в плюсе → резко в минус) ───
ALERT_PNL_REVERSAL_DROP_PCT: float = 2.0            # мин. падение PnL в п.п. для срабатывания
ALERT_PNL_REVERSAL_SAMPLE_MINUTES: float = 15.0     # окно сравнения "было / стало"
ALERT_PNL_REVERSAL_COOLDOWN_MINUTES: float = 60.0   # не спамить чаще, чем раз в это время

# ── Алерт 21: резкий памп/дамп по паре с открытой позицией ────────────────
ALERT_PAIR_PUMP_DUMP_PCT: float = 6.0               # % изменения цены за lookback-окно
ALERT_PAIR_PUMP_DUMP_LOOKBACK_MINUTES: float = 60.0  # окно сравнения в минутах, НЕ зависит от ТФ
ALERT_PAIR_PUMP_DUMP_COOLDOWN_MINUTES: float = 60.0

# ── Алерт 22: резкое движение BTC — индикатор "рынок штормит" целиком ─────
ALERT_BTC_MOVE_PCT: float = 3.5
ALERT_BTC_MOVE_LOOKBACK_MINUTES: float = 120.0  # окно сравнения в минутах, НЕ зависит от ТФ
ALERT_BTC_PAIR: str = "BTC/USDT:USDT"
ALERT_BTC_MOVE_COOLDOWN_MINUTES: float = 60.0  # защита от дребезга на границе порога


# Префикс для имён потоков AI-листенера — используется для глобальной проверки
# дублей через threading.enumerate(), которая переживает переимпорт модуля
# при reload_config в Freqtrade.
_AI_THREAD_PREFIX = "FreqtradeAIListener-"

# Реестр живых listener'ов по токену — чтобы при reload_config мы могли найти
# существующий instance и обновить его strategy_params свежими значениями из
# нового config.json (без пересоздания самого listener'а, иначе TG long-poll
# дублируется).
_AI_LISTENER_REGISTRY: Dict[str, "TelegramAIListener"] = {}
_AI_REGISTRY_LOCK = threading.Lock()


def _ai_listener_already_running(tg_token: str) -> bool:
    """Глобальная проверка: уже есть живой поток листенера с этим токеном?
    Работает даже после reload_config когда модуль перезагружается."""
    name = _AI_THREAD_PREFIX + tg_token[:10]
    for t in threading.enumerate():
        if t.name == name and t.is_alive():
            return True
    return False


def _get_running_ai_listener(tg_token: str) -> Optional["TelegramAIListener"]:
    """Достаёт существующий listener из реестра, если он жив."""
    with _AI_REGISTRY_LOCK:
        inst = _AI_LISTENER_REGISTRY.get(tg_token)
    if inst is None:
        return None
    # Проверяем что поток ещё жив (на всякий — если упал)
    if not _ai_listener_already_running(tg_token):
        return None
    return inst


def _ensure_openai_installed() -> bool:
    """
    Если пакет openai отсутствует — установить его в текущий Python (pip install),
    затем импортировать и обновить глобальные OPENAI_AVAILABLE / OpenAI.
    Возвращает True если в итоге пакет доступен.
    """
    global OpenAI, OPENAI_AVAILABLE
    if OPENAI_AVAILABLE:
        return True
    try:
        import subprocess
        import sys
        import importlib
        logger.warning("[AI] openai пакет не найден — устанавливаю автоматически (pip install openai)...")
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "openai"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0:
            logger.error(
                "[AI] Автоустановка openai НЕ удалась (rc=%s): %s",
                proc.returncode,
                (proc.stderr or proc.stdout or "")[-400:],
            )
            return False
        # Свежий импорт
        importlib.invalidate_caches()
        import openai as _openai_mod  # noqa: F401
        importlib.reload(_openai_mod)
        from openai import OpenAI as _OpenAI
        OpenAI = _OpenAI
        OPENAI_AVAILABLE = True
        logger.info("[AI] openai установлен автоматически: версия %s", getattr(_openai_mod, "__version__", "?"))
        return True
    except Exception as e:
        logger.error("[AI] Исключение при автоустановке openai: %s", e, exc_info=True)
        return False


# ===========================================================================
# Telegram AI Listener — слушает свободный текст в TG и выполняет команды
# через OpenAI function-calling + Freqtrade REST API.
# ===========================================================================

class TelegramAIListener:
    """
    Слушает входящие сообщения в Telegram через long-polling и отвечает
    через OpenAI (function calling) + Freqtrade REST API.
    Запускается в отдельном демон-потоке. Игнорирует команды, начинающиеся с '/'
    (их обрабатывает встроенный telegram-bot Freqtrade).
    """

    HISTORY_LIMIT = 10           # история — больше для mini (контекст важнее)
    PENDING_TTL_SEC = 300        # сколько живёт ожидающее подтверждения действие
    MAX_TOOL_ITERATIONS = 8      # защита от циклов tool calls
    TG_MSG_LIMIT = 4096          # лимит Telegram на одно сообщение

    # Tool'ы которые ОБЯЗАТЕЛЬНО требуют подтверждения
    CONFIRMATION_REQUIRED_TOOLS = {
        "place_trade", "close_trade", "close_all_trades", "close_by_pair",
        "stop_bot",       # останавливает торговлю — критично
        "delete_trade",   # удаляет запись из БД без закрытия позиции на бирже
    }

    def __init__(self, tg_token: str, chat_id: str, openai_key: str,
                 ft_client: "FtRestClient", model: str = "gpt-4o",
                 daily_summary_time: str = "", summary_tz: str = "",
                 strategy_params: Optional[dict] = None,
                 show_cost: bool = True) -> None:
        self.token = tg_token
        self.chat_id = str(chat_id)
        # Показывать ли стоимость каждого ответа в конце сообщения (только для gpt-4.1-mini)
        self.show_cost = bool(show_cost)
        # Strategy-specific params (force_exit_after_days, safety_order_*,
        # price_deviation_initial и т.п.) — show_config их НЕ возвращает.
        # Два источника: (1) snapshot переданный стратегией при старте,
        # (2) config_files на диске — читаем их КАЖДЫЙ раз в _build_my_config_text,
        # чтобы юзер видел актуальные значения сразу после правки JSON +
        # reload_config (без перезапуска контейнера).
        self.strategy_params: dict = dict(strategy_params or {})
        self._strategy_params_lock = threading.Lock()
        self._config_files: list = []
        # Колбэки на market_direction стратегии (REST API для /marketdir нет)
        self._market_dir_get = None
        self._market_dir_set = None
        # Регистрируемся в глобальном реестре, чтобы стратегия могла нас найти
        # после reload_config и обновить strategy_params свежими значениями.
        with _AI_REGISTRY_LOCK:
            _AI_LISTENER_REGISTRY[tg_token] = self
        # max_retries=0 чтобы SDK НЕ ретраил при 429 — иначе плодит дубли ответов в чат
        self.openai = OpenAI(api_key=openai_key, max_retries=0)
        self.model = model
        self.ft = ft_client
        self.last_update_id = 0
        self._running = False
        self._history: list = []
        self._history_lock = threading.Lock()
        # Дедупликация: (text, ts) последнего обработанного сообщения — отсекаем повторы внутри 10 сек
        self._last_processed_text: str = ""
        self._last_processed_ts: float = 0.0
        # Защита от случайной самообработки: id и username бота (для фильтрации echo)
        self._self_bot_id: Optional[int] = None
        self._self_bot_username: Optional[str] = None
        # Расписание ежедневной сводки, формат "HH:MM" (24h). Пусто = выкл 
        self.daily_summary_time = (daily_summary_time or "").strip() 
        # Часовой пояс для расписания, например "Europe/Moscow". Пусто = время контейнера (обычно UTC)
        self.summary_tz = (summary_tz or "").strip()

        # Pending action — одно ожидающее подтверждения действие на чат
        self._pending_action: Optional[dict] = None
        self._pending_lock = threading.Lock()

        # Контекст бота — узнаём при инициализации, чтобы GPT понимал формат пар
        self._bot_ctx = self._probe_bot_context()

        self.tools = [
            # ── READ-ONLY ───────────────────────────────────────
            {"type": "function", "function": {
                "name": "fetch_bot_status",
                "description": "Get list of currently open trades with IDs, pair, side, profit, etc.",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "fetch_trade",
                "description": "Get full details of a single trade by its numeric ID.",
                "parameters": {"type": "object", "properties": {
                    "trade_id": {"type": "integer"},
                }, "required": ["trade_id"]}
            }},
            {"type": "function", "function": {
                "name": "fetch_count",
                "description": "Get count of currently open trades and the max allowed.",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "fetch_profit",
                "description": "Get overall profit summary.",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "fetch_balance",
                "description": "Get wallet/account balance.",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "fetch_performance",
                "description": "Get profit/performance grouped by trading pair.",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "fetch_stats",
                "description": "Get aggregated bot statistics (winning/losing trades, durations etc.).",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "fetch_daily",
                "description": "Get profit grouped by day for the last N days (default 7).",
                "parameters": {"type": "object", "properties": {
                    "days": {"type": "integer", "description": "Number of days, default 7"},
                }}
            }},
            {"type": "function", "function": {
                "name": "fetch_weekly",
                "description": "Get profit grouped by week for the last N weeks (default 4).",
                "parameters": {"type": "object", "properties": {
                    "weeks": {"type": "integer"},
                }}
            }},
            {"type": "function", "function": {
                "name": "fetch_monthly",
                "description": "Get profit grouped by month for the last N months (default 3).",
                "parameters": {"type": "object", "properties": {
                    "months": {"type": "integer"},
                }}
            }},
            {"type": "function", "function": {
                "name": "fetch_entries",
                "description": "Get statistics about entry signals (optionally filtered by pair).",
                "parameters": {"type": "object", "properties": {
                    "pair": {"type": "string"},
                }}
            }},
            {"type": "function", "function": {
                "name": "fetch_exits",
                "description": "Get statistics about exit signals (optionally filtered by pair).",
                "parameters": {"type": "object", "properties": {
                    "pair": {"type": "string"},
                }}
            }},
            {"type": "function", "function": {
                "name": "fetch_mix_tags",
                "description": "Get performance grouped by entry/exit tag mix (optionally filtered by pair).",
                "parameters": {"type": "object", "properties": {
                    "pair": {"type": "string"},
                }}
            }},
            {"type": "function", "function": {
                "name": "fetch_whitelist",
                "description": "Get whitelist of trading pairs.",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "fetch_blacklist",
                "description": "Get blacklist of trading pairs.",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "fetch_locks",
                "description": "Get list of currently active trade locks.",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "fetch_trades",
                "description": "Fetch FRESH history of closed trades from DB (most recent first). MUST be called every time user asks about closed trades / last trade / history — never reuse previous results.",
                "parameters": {"type": "object", "properties": {
                    "limit": {"type": "integer", "description": "Max number of trades, default 50 (covers full recent history)"},
                }}
            }},
            {"type": "function", "function": {
                "name": "fetch_config",
                "description": "Get current bot configuration (sanitized).",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "fetch_my_config",
                "description": (
                    "Return user-friendly Russian explanation of the user's DCA/TP "
                    "strategy config (force_exit_after_days, safety_order_ratio, "
                    "safety_order_max_count, safety_order_volume_scale, "
                    "price_deviation_initial, take_profit, trailing_*, stoploss). "
                    "Call this when user asks 'мой конфиг', 'покажи конфиг', "
                    "'параметры стратегии', 'параметры DCA', 'покажи force_exit/safety_order'. "
                    "DO NOT trigger on 'мои настройки' — that phrase is intentionally excluded. "
                    "Relay the result verbatim — it's already formatted for Telegram."
                ),
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "fetch_strategies",
                "description": "List strategies available to the bot.",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "fetch_pairlists_available",
                "description": "List available pairlist handlers/plugins.",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "fetch_market_data",
                "description": "Fetch latest OHLCV candlestick data for a trading pair and timeframe.",
                "parameters": {"type": "object", "properties": {
                    "pair": {"type": "string", "description": "e.g. BTC/USDT"},
                    "timeframe": {"type": "string", "description": "e.g. 1h, 5m, 1d"},
                    "limit": {"type": "integer"},
                }, "required": ["pair", "timeframe"]}
            }},
            {"type": "function", "function": {
                "name": "fetch_logs",
                "description": "Get recent log lines from the bot (default last 50).",
                "parameters": {"type": "object", "properties": {
                    "limit": {"type": "integer"},
                }}
            }},
            {"type": "function", "function": {
                "name": "fetch_health",
                "description": "Get bot health/last process status.",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "fetch_version",
                "description": "Get Freqtrade version.",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "fetch_sysinfo",
                "description": "Get system info (CPU, memory) of the host running the bot.",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "ping",
                "description": "Ping the bot's REST API to verify it's responsive.",
                "parameters": {"type": "object", "properties": {}}
            }},

            # ── BOT CONTROL ─────────────────────────────────────
            {"type": "function", "function": {
                "name": "start_bot",
                "description": "Start trading (resume after /stop).",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "stop_bot",
                "description": "Stop the bot (does NOT close open trades).",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "stopentry",
                "description": "Stop opening new trades but keep current ones running.",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "reload_config",
                "description": "Reload the bot's config.json without restarting the container.",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "fetch_market_direction",
                "description": (
                    "Get current market direction filter (long|short|even|none). "
                    "Call when user asks 'какое направление', 'только лонги?', "
                    "'что сейчас по marketdir'."
                ),
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "set_market_direction",
                "description": (
                    "Change which entry directions the strategy allows. "
                    "long=only longs, short=only shorts, even=alternating, none=both allowed. "
                    "Use for 'только лонги', 'переключи на шорт', 'разреши оба направления'."
                ),
                "parameters": {"type": "object", "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["long", "short", "even", "none"],
                        "description": "New market direction",
                    },
                }, "required": ["direction"]}
            }},

            # ── TRADE ACTIONS ───────────────────────────────────
            {"type": "function", "function": {
                "name": "place_trade",
                "description": (
                    "Open a NEW position via force-entry. Use for 'buy X', 'long X', 'short X'. "
                    "Do NOT use to close existing trades — use close_trade for that."
                ),
                "parameters": {"type": "object", "properties": {
                    "pair": {"type": "string", "description": "e.g. BTC/USDT"},
                    "side": {"type": "string", "enum": ["long", "short"]},
                    "stake_amount": {"type": "number", "description": "Stake in quote currency (optional)"},
                    "price": {"type": "number", "description": "Limit price. If provided — places a LIMIT order at this exact price (will wait in orderbook if price is worse than market). If omitted — uses default entry pricing (limit-with-spread-cross, executes immediately on liquid pairs)."},
                    "enter_tag": {"type": "string"},
                }, "required": ["pair", "side"]}
            }},
            {"type": "function", "function": {
                "name": "close_trade",
                "description": (
                    "Force-close (exit) a SPECIFIC open trade by its numeric trade_id. "
                    "If the user names a pair, FIRST call fetch_bot_status to find the trade_id. "
                    "Pass 'amount' for PARTIAL exit: 'закрой половину', 'зафиксируй 30 USDT по сделке 7'. "
                    "If user says 'половину' — compute half of trade amount from fetch_trade/fetch_bot_status."
                ),
                "parameters": {"type": "object", "properties": {
                    "trade_id": {"type": "integer"},
                    "amount": {"type": "number", "description": "Partial-exit amount in base currency (optional)"},
                }, "required": ["trade_id"]}
            }},
            {"type": "function", "function": {
                "name": "close_all_trades",
                "description": "Force-close ALL currently open trades.",
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "close_by_pair",
                "description": (
                    "Force-close all open trades on a specific pair (one or multiple). "
                    "Use this for 'close X', 'exit X', 'выйди из X', 'закрой X' — instead of "
                    "fetch_bot_status + multiple close_trade calls. The system handles "
                    "single/multiple cases and shows confirmation buttons."
                ),
                "parameters": {"type": "object", "properties": {
                    "pair": {"type": "string", "description": "e.g. BTC/USDT:USDT"},
                }, "required": ["pair"]}
            }},
            {"type": "function", "function": {
                "name": "cancel_open_order",
                "description": "Cancel a pending open order on a trade by trade_id.",
                "parameters": {"type": "object", "properties": {
                    "trade_id": {"type": "integer"},
                }, "required": ["trade_id"]}
            }},
            {"type": "function", "function": {
                "name": "reload_trade",
                "description": (
                    "Reload/sync an open trade from exchange orders (like /reload_trade). "
                    "Use when trade looks stuck or out of sync: 'перезагрузи сделку 12'."
                ),
                "parameters": {"type": "object", "properties": {
                    "trade_id": {"type": "integer"},
                }, "required": ["trade_id"]}
            }},
            {"type": "function", "function": {
                "name": "delete_trade",
                "description": (
                    "Remove a trade from the database entirely by trade_id. "
                    "DOES NOT close the exchange position — use close_trade for that."
                ),
                "parameters": {"type": "object", "properties": {
                    "trade_id": {"type": "integer"},
                }, "required": ["trade_id"]}
            }},

            # ── HELP / CAPABILITIES ─────────────────────────────
            {"type": "function", "function": {
                "name": "fetch_capabilities",
                "description": (
                    "Return the FULL list of what this AI assistant can do (commands, examples, "
                    "limits). Call when the user asks 'что ты умеешь', 'возможности', 'помощь', "
                    "'команды', 'help', 'capabilities', 'что можешь', 'список команд'. "
                    "Relay the returned text verbatim without summarizing or shortening."
                ),
                "parameters": {"type": "object", "properties": {}}
            }},

            # ── CONFIRMATION FLOW (для входа/выхода) ──────────
            {"type": "function", "function": {
                "name": "confirm_action",
                "description": (
                    "Execute the pending action that is currently awaiting confirmation. "
                    "Call this ONLY after the user has explicitly confirmed "
                    "(replies like 'да', 'yes', 'ок', 'подтверждаю', 'давай')."
                ),
                "parameters": {"type": "object", "properties": {}}
            }},
            {"type": "function", "function": {
                "name": "cancel_pending_action",
                "description": (
                    "Cancel the pending action awaiting confirmation. "
                    "Call when user says 'нет', 'отмена', 'не надо', 'cancel'."
                ),
                "parameters": {"type": "object", "properties": {}}
            }},

            # ── LISTS / LOCKS ───────────────────────────────────
            {"type": "function", "function": {
                "name": "add_blacklist",
                "description": "Add a pair to the blacklist (blocks future trades on it).",
                "parameters": {"type": "object", "properties": {
                    "pair": {"type": "string"},
                }, "required": ["pair"]}
            }},
            {"type": "function", "function": {
                "name": "delete_blacklist",
                "description": "Remove a pair from the blacklist.",
                "parameters": {"type": "object", "properties": {
                    "pair": {"type": "string"},
                }, "required": ["pair"]}
            }},
            {"type": "function", "function": {
                "name": "add_lock",
                "description": (
                    "Manually lock a pair from being traded until a given UTC datetime. "
                    "'until' must be ISO format like '2026-06-01T12:00:00'. "
                    "'side' is 'long', 'short' or '*' (both)."
                ),
                "parameters": {"type": "object", "properties": {
                    "pair": {"type": "string"},
                    "until": {"type": "string"},
                    "side": {"type": "string", "enum": ["long", "short", "*"]},
                    "reason": {"type": "string"},
                }, "required": ["pair", "until"]}
            }},
            {"type": "function", "function": {
                "name": "delete_lock",
                "description": "Delete a trade lock by its numeric lock_id.",
                "parameters": {"type": "object", "properties": {
                    "lock_id": {"type": "integer"},
                }, "required": ["lock_id"]}
            }},
        ]

    def _probe_bot_context(self) -> dict:
        """Узнаём trading_mode / stake_currency / whitelist / timeframe для подсказок GPT."""
        ctx = {"trading_mode": "spot", "stake_currency": "USDT", "whitelist": [], "timeframe": ""}
        try:
            cfg = self.ft.show_config() or {}
            ctx["trading_mode"] = cfg.get("trading_mode") or "spot"
            ctx["stake_currency"] = cfg.get("stake_currency") or "USDT"
            ctx["timeframe"] = cfg.get("timeframe") or ""
        except Exception as e:
            logger.warning(f"[AI] probe show_config error: {e}")
        try:
            wl = self.ft.whitelist() or {}
            pairs = wl.get("whitelist") if isinstance(wl, dict) else wl
            if isinstance(pairs, list):
                ctx["whitelist"] = pairs[:60]  # обрезаем чтобы не раздувать промпт
        except Exception as e:
            logger.warning(f"[AI] probe whitelist error: {e}")
        return ctx

    def _fix_pair(self, pair: Optional[str]) -> Optional[str]:
        """
        Приводит пару к формату, который ожидает биржа.
        Для futures: 'SOL/USDT' → 'SOL/USDT:USDT' если в whitelist есть с суффиксом.
        Сравнение нечувствительно к регистру.
        """
        if not pair:
            return pair
        p = pair.strip().upper()
        wl = self._bot_ctx.get("whitelist") or []
        wl_upper = {w.upper(): w for w in wl}
        if p in wl_upper:
            return wl_upper[p]
        # Пробуем добавить ':QUOTE' для futures
        if self._bot_ctx.get("trading_mode") == "futures" and ":" not in p:
            quote = self._bot_ctx.get("stake_currency", "USDT")
            candidate = f"{p}:{quote}".upper()
            if candidate in wl_upper:
                return wl_upper[candidate]
            # Возвращаем кандидата даже если его нет в whitelist (может быть динамический)
            return candidate
        return pair

    def _system_prompt(self) -> str:
        tm = self._bot_ctx.get("trading_mode", "spot")
        sc = self._bot_ctx.get("stake_currency", "USDT")
        wl = self._bot_ctx.get("whitelist", [])
        wl_str = ", ".join(wl[:30]) if wl else "(empty)"
        is_futures = tm == "futures"

        return f"""You are an assistant controlling a Freqtrade trading bot via tools.

═══ LANGUAGE ═══
The user writes in Russian. ALWAYS reply in Russian unless they switch to another language for ≥2 turns in a row. Never reply in Spanish/English/etc. unexpectedly.

═══ TIMEZONE — ВСЕГДА ПО МОСКВЕ (МСК, UTC+3) ═══
ВАЖНО: Freqtrade REST API возвращает все даты/timestamp'ы в UTC (open_date, close_date в "YYYY-MM-DD HH:MM:SS").
Юзер находится в Москве — ВСЕ времена в твоих ответах ОБЯЗАТЕЛЬНО конвертируй в МСК (UTC+3) и добавляй пометку «МСК».

Примеры:
- API close_date: "2026-06-05 06:21:31" (UTC) → ответ юзеру: "09:21:31 МСК"
- API open_date:  "2026-06-04 22:00:02" (UTC) → ответ юзеру: "01:00:02 МСК (5 июня)"

Если час+3 переваливает за 24 — переноси дату вперёд. Если ниже 0 — назад.
Никогда не пиши «по времени сервера» или «UTC» в ответе — это путает юзера.
Если показываешь только время (без даты), формат: HH:MM МСК. С датой — DD.MM HH:MM МСК.

═══ DATA FRESHNESS — НИКОГДА НЕ ОТВЕЧАЙ ИЗ ПАМЯТИ ═══
КРИТИЧЕСКОЕ ПРАВИЛО. Состояние бота МЕНЯЕТСЯ КАЖДУЮ СЕКУНДУ: открываются/закрываются сделки,
двигается баланс, прилетают DCA, срабатывают TP/SL. Любые данные из ПРЕДЫДУЩИХ tool-вызовов в
истории — УСТАРЕЛИ к моменту следующего сообщения юзера. Даже если ты только что отвечал на
похожий вопрос — данные уже могли поменяться.

ПРАВИЛА (обязательны):
1. На КАЖДЫЙ вопрос про сделки / баланс / профит / статус / историю / последнюю сделку →
   ВСЕГДА вызывай нужный fetch_* tool ЗАНОВО. НЕ копируй цифры из предыдущих сообщений в истории.
2. «Последняя закрытая сделка», «история закрытых», «всё что закрыто» → ВСЕГДА fetch_trades(limit=50).

2а. «ПОСЛЕДНИЙ ВХОД», «последняя открытая», «последняя сделка» (без слова «закрытая»),
    «крайний вход», «что последнее открылось» → ОБЯЗАТЕЛЬНО вызвать ОБА tool'а:
      a) fetch_bot_status → даёт ОТКРЫТЫЕ сейчас сделки с open_date
      b) fetch_trades(limit=20) → даёт ЗАКРЫТЫЕ с open_date
    Объединить ОБА списка и выбрать запись с МАКСИМАЛЬНОЙ open_date — это и есть последний вход.
    Самый свежий вход почти всегда в ОТКРЫТЫХ (если бот недавно зашёл в позицию).
    НЕ полагайся на одну fetch_trades — там нет открытых позиций!

2б. «Последний выход» / «последнее закрытие» → fetch_trades(limit=20), index 0 (отсортировано по close_timestamp DESC).
3. «Открытые сделки», «что в позиции», «что сейчас торгуется» → ВСЕГДА fetch_bot_status.
4. «Баланс», «сколько на счету», «бабло» → ВСЕГДА fetch_balance.
5. «Прибыль», «навар», «сколько заработал» → ВСЕГДА fetch_profit.
6. Если юзер повторяет вопрос или говорит «ты ошибаешься» / «нет, не так» / «проверь ещё раз» →
   ОБЯЗАТЕЛЬНО заново вызови соответствующий tool. Возможно с увеличенным limit (например 100).
   Не оправдывайся и не повторяй старый ответ — ВЫЗОВИ TOOL ЗАНОВО.
7. fetch_trades возвращает закрытые сделки СВЕЖИМИ из БД, сортировка most-recent-first.
   Первая сделка в результате = самая последняя закрытая. Доверяй этому, а не своей памяти.
8. Если результат tool'а ПРОТИВОРЕЧИТ тому что ты говорил раньше — НОВЫЙ результат правильный.
   Извинись коротко («Уточнил — последняя сделка ...») и дай актуальные данные.

═══ UNDERSTAND INFORMAL / SLOPPY INPUT ═══
Юзер — трейдер, пишет неформально, с опечатками, сленгом, может мешать RU/EN.
ВСЕГДА трактуй намерение мягко и щедро.

Общие принципы (НЕ жёсткие словари — мысли по смыслу):
- Любое сокращение/сленг для **монеты** → ищи лучшее совпадение в whitelist:
  начало названия base, фонетическое сходство, известный сленг (битос/бэтэшка → BTC,
  соль/солана → SOL, эфир/эфирка → ETH, доги/дож → DOGE, рип/рипл → XRP, etc.).
  Если непонятно — спроси в одну строку, какую именно из whitelist.
- Запросы про деньги/баланс: «бабло», «бабла», «балас», «скока на счету», «капитал» → fetch_balance.
- Запросы про заработок: «навар», «в плюсе», «зашибаю», «слив», «потери» → fetch_profit или fetch_performance.
- Открытие сделки: «купи», «бай», «лонг», «возьми» → side=long. «шорт», «селл», «зашорти» → side=short.
- Закрытие: «закрой», «выйди из», «фиксуй», «продай» (для лонга) → close_by_pair(pair) если указана пара,
  иначе close_trade(trade_id) если указан ID.
- Управление ботом: «стоп»/«стопари»/«выруби» → stop_bot. «запуск»/«погнали»/«врубай» → start_bot.
  «релоад»/«реложни»/«обнови конфиг» → reload_config.
- Чёрный список: «забань»/«бан»/«блок»/«в блек» → add_blacklist. «разбань»/«разблок» → delete_blacklist.
- Сводка/чек: «как дела», «че как», «че по боту», «отчёт», «дашборд» → несколько read-tools и сводка.
- Опечатки нормализуй мысленно («сделак»=«сделка», «закроой»=«закрой», «балас»=«баланс»).

Если СОВСЕМ непонятно — одна короткая строка уточнения, не пересjpравшивай на каждое сообщение.

═══ OUTPUT FORMAT ═══
Plain text for Telegram. NO Markdown bold/italic, NO HTML tags, NO code fences.
Use line breaks between sections. Format numbers like "12.34 USDT", percents like "+3.4%".
Be concise: 1–6 lines for simple questions, structured list for reports.

═══ BOT CONTEXT ═══
trading_mode: {tm}
stake_currency: {sc}
strategy_timeframe: {self._bot_ctx.get("timeframe", "?")} (это основной TF, live-данные есть только для него)
pair format: {"FUTURES — pairs MUST have ':QUOTE' suffix, e.g. SOL/USDT:USDT, BTC/USDT:USDT, DOT/USDT:USDT" if is_futures else "SPOT — pairs like BTC/USDT"}
whitelist (sample): {wl_str}

═══ INDICATORS — VALUES MATCH DASHBOARD 1:1 ═══
Indicator values come from the strategy's populate_indicators (live cache) via
fetch_market_data → pair_candles. NEVER recalculate yourself — read from the dataframe.
The dashboard reads the same source, so values are guaranteed identical.

Available columns in the returned data:
  - rsi (period 14)
  - macd, macdsignal, macdhist
  - sma_5, sma_20, sma_50, sma_100, sma_200
  - st_up, st_down (SuperTrend, period=10, multiplier=3.0)
      • Only ONE of (st_up, st_down) is non-NaN per candle. NaN means N/A for that side.
      • st_up has value → trend is UP (price above SuperTrend); current candle's level = st_up
      • st_down has value → trend is DOWN; level = st_down
      • DO NOT confuse SuperTrend with enter_long/enter_short — those are STRATEGY ENTRY SIGNALS,
        not the indicator's direction. SuperTrend's direction comes from which column is non-NaN.
  - bb_lower, bb_middle, bb_upper (Bollinger Bands, 20 / 2.0σ)
  - adx, atr, mfi (period 14)
  - kst, kst_sig (KST momentum + 9-period signal)

When user asks for an indicator on a pair:
  1) Call fetch_market_data with the strategy's main timeframe (default if user didn't specify)
  2) Read the LAST row's value(s) for the requested column(s)
  3) Reply with the single number or 2-3 numbers, no JSON dump
  4) For SuperTrend "куда направлен" → check which of st_up/st_down is non-NaN on the last row

═══ MARKET DATA RULES ═══
fetch_market_data работает надёжно ТОЛЬКО для:
  - пар из whitelist
  - таймфрейма стратегии ({self._bot_ctx.get("timeframe", "?")})
Для других таймфреймов есть fallback на исторические feather файлы, но их может не быть.
Если получаешь EMPTY_DATA — сообщи пользователю причину из текста ошибки и предложи
взять данные на основном таймфрейме или другую пару из whitelist.
Не выдумывай свечи — если данных нет, прямо так и скажи.

═══ PAIR NORMALIZATION ═══
{"For futures: if user writes 'SOL' → use 'SOL/USDT:USDT'. If user writes 'SOL/USDT' → use 'SOL/USDT:USDT'. The helper auto-fixes this, but you should still pass the full form." if is_futures else "Use BASE/QUOTE form."}
If a tool returns 'symbol does not exist' or 'market not active' — call fetch_whitelist and retry with a correct pair.

═══ CANCEL LIMIT ORDER — единственный правильный путь ═══
Если юзер хочет ОТМЕНИТЬ лимитный ордер (фразы: «отмени лимит», «убери ордер», «отмени ордер по X»,
«cancel limit on X», «отмени этот лимитник»):
→ Использовать `close_by_pair(pair=...)` — это ЕДИНСТВЕННЫЙ корректный путь.
  Tool автоматически различает «закрытие позиции» vs «отмена лимита» по факту филла
  и отображает правильный текст в подтверждении.
НЕ используй `cancel_open_order` для лимитных ордеров — оно требует знать trade_id и
часто возвращает мутные ответы. cancel_open_order применяй ТОЛЬКО если юзер прямо
говорит «отмени ордер у сделки N» с конкретным номером.
НЕ говори «ордер не найден» или «уже отменён» если close_by_pair прошёл успешно —
система покажет нормальный ответ через кнопки.

═══ LIMIT vs MARKET ORDERS ═══
ЛИМИТНЫЕ ОРДЕРА ПОДДЕРЖИВАЮТСЯ. У `place_trade` есть параметр `price`.
- Юзер говорит «лимит по цене 60», «лимитник 70000», «на 100 баков по 50» → ПЕРЕДАЙ price=60 (число).
- Без явной цены → НЕ передавай price, будет дефолтная entry-логика (limit-with-spread-cross,
  функционально как market на ликвидных парах).
НИКОГДА не отвечай «лимитные ордера не поддерживаются» — это неправда.
Если юзер ставит цену сильно «лучше рынка» (например лимит-лонг по цене НА 1%+ ВЫШЕ текущей),
Freqtrade автоматически сконвертирует в market — это его защита, не баг. Можешь упомянуть это юзеру.

═══ CRITICAL: TWO-STEP CONFIRMATION FOR TRADES ═══
The tools `place_trade`, `close_trade`, `close_all_trades` DO NOT execute immediately.
When you call them, they return "PENDING_CONFIRMATION (NOT executed yet). Action: ...".
This is the EXPECTED flow, NOT an error.

After receiving PENDING_CONFIRMATION:
  1) Ask the user to confirm in Russian, e.g.:
     "Подтверди: открыть LONG SOL/USDT:USDT на 50 USDT? Ответь 'да' или 'нет'."
  2) Wait for the user's next message.
  3) If user replies 'да' / 'yes' / 'ок' / 'давай' / 'подтверждаю' / 'го' → call confirm_action() with no args.
  4) If user replies 'нет' / 'отмена' / 'не надо' / 'cancel' → call cancel_pending_action() with no args.
  5) After confirm_action returns 'EXECUTED ...' → tell the user the result briefly.

Do NOT call confirm_action() without an explicit user confirmation in the previous message.
Do NOT re-call place_trade/close_trade with the same args after asking — use confirm_action().

═══ TRADE BY PAIR ═══
If user says 'закрой DOT', 'выйди из BTC', 'exit ETH' (no specific trade_id):
  → Call close_by_pair(pair="DOT/USDT:USDT") DIRECTLY. ONE tool call, that's it.
  The system will:
    - find all open trades on that pair
    - if 0 → return REJECTED, you tell user nothing is open
    - if 1 or many → create one confirmation with inline buttons
  DO NOT call fetch_bot_status + close_trade separately when pair is known.
  DO NOT ask "which trade?" — close_by_pair handles single AND multiple cases.

If user gives a specific trade_id ('close trade 7'):
  → Call close_trade(trade_id=7) directly.

═══ PARTIAL CLOSE ═══
For 'закрой половину SOL', 'зафиксируй 30 USDT по сделке 7', 'частично закрой':
  → fetch_bot_status or fetch_trade to get trade_id and current amount
  → close_trade(trade_id=..., amount=...) with computed partial amount
  → Requires confirmation like any close_trade.

═══ MARKET DIRECTION (/marketdir) ═══
Strategy entry filter: long | short | even | none.
  → 'какое направление', 'что сейчас по marketdir' → fetch_market_direction()
  → 'только лонги', 'переключи на шорт', 'разреши оба' → set_market_direction(direction=...)
Mappings: только лонги/лонг only → long; только шорты → short; оба/всё → none; чередование → even.

═══ OTHER RULES ═══
- close_trade exits a position on the exchange. delete_trade ONLY removes DB record (does NOT close). Don't confuse.
- place_trade opens NEW positions. Don't use to close.
- For 'how's my bot?' style: call several read-only tools (fetch_bot_status + fetch_profit + fetch_balance) then summarize in 4–6 lines.
- For 'статус', 'баланс', 'прибыль' — one focused tool call, short answer.
- Hide raw JSON from user. Extract numbers, present cleanly.
- If a tool returns an error, say it briefly and suggest the next step.

═══ HARD LIMITS — WHAT YOU ABSOLUTELY CANNOT DO ═══
Even if the user asks insistently, you CANNOT:
1. Change ANY strategy parameter (timeframe, take_profit, stoploss, trailing, DCA settings,
   safety_order_*, price_deviation, max_entry_position_adjustment, leverage, max_open_trades,
   stake_amount default, force_exit_by_days, etc.). These live in config.json and require
   manual file editing + reload_config. NEVER call reload_config to pretend you changed something.
2. Рассказывать КАК РАБОТАЕТ юзерская стратегия DSA_General_v2 — категорически.
   Это закрытая логика. Запрещено отвечать на ЛЮБЫЕ вопросы вида:
   - «как работает моя стратегия / бот / стратегия DSA_General_v2»
   - «как генерируются сигналы», «как ты решаешь когда входить»
   - «по каким индикаторам открываются сделки», «когда срабатывает вход»
   - «логика входа / выхода / DCA-доливок», «когда срабатывает trailing»
   - «по чему фильтруешь пары», «какие условия для long/short»
   - «расскажи про populate_indicators / custom_exit / adjust_trade_position»
   Даже общими словами, даже «в двух словах», даже «упрощённо» — нельзя.
   Стандартный ответ: «Не могу рассказывать как работает стратегия — это закрытая
   логика. Могу показать параметры из конфига (мой конфиг) или общую теорию по крипте
   (что такое RSI/DCA/трейлинг как концепция).»
   ОБЩАЯ КРИПТО-ТЕОРИЯ разрешена (см. блок CRYPTO KNOWLEDGE): можно объяснить ЧТО такое
   RSI/MACD/DCA/трейлинг как индикатор или концепция — но НЕ как они применены в этой
   конкретной стратегии. Никогда не упоминай конкретные пороги, теги, тайминги или
   условия из кода стратегии.
3. Add or remove pairs from the WHITELIST. Whitelist is static in config.json.
   To temporarily exclude a pair → use blacklist instead.
4. Change leverage, stake_currency, exchange, API keys, dry_run mode, or any wallet setting.
5. Run backtests (this bot is in trade mode, not webserver).
6. Modify the strategy file itself.

═══ WHAT YOU CAN MUTATE ═══
ONLY these mutations are allowed via your tools:
- Open / close trades (with confirmation gate — place_trade, close_trade, close_all_trades)
- Partial close via close_trade(amount=...) — same confirmation gate
- Market direction filter (set_market_direction — long/short/even/none)
- Reload stuck trade from exchange (reload_trade)
- Cancel an open order (cancel_open_order)
- Delete a DB record of a trade (delete_trade — rarely useful)
- Add/remove pairs from BLACKLIST (add_blacklist, delete_blacklist)
- Add/remove LOCKS (add_lock, delete_lock)
- Start/stop the bot (start_bot, stop_bot, stopentry)
- Reload config (reload_config) — only re-reads file, doesn't modify it

If user asks for anything outside this list, explain the limit and refuse politely.

═══ ADDING TO EXISTING POSITION (DCA) ═══
The user cannot manually "влить деньги" / "добавить" / "усреднить" / "DCA ордер" into an existing
trade via REST API — Freqtrade rejects forceenter when the pair is already in a position.
DCA happens automatically when the strategy's deviation triggers fire.
If user asks this, suggest: 1) close current trade + open new larger one, 2) wait for auto-DCA,
3) different pair. Don't try to forceenter on an occupied pair.

═══ POSITION CHECK BEFORE PLACE_TRADE ═══
The system auto-checks before forceenter. If the pair is occupied, you'll get a
"REJECTED: ..." result instead of PENDING_CONFIRMATION. In that case do NOT call confirm_action —
relay the situation to the user and ask what to do.

═══ CRYPTO KNOWLEDGE — РАЗРЕШЕНО ОТВЕЧАТЬ ═══
Юзер может задавать ЛЮБОЙ вопрос по крипте — не только про его бота. Отвечай чётко, по делу, на русском.
Можно объяснять:
- Что такое DCA, спот vs фьючерсы, кредитное плечо, маржин-колл, ликвидация
- Как работают индикаторы (RSI, MACD, Bollinger, ADX, ATR, SuperTrend, SMA/EMA, MFI, KST, Williams %R, OBV, CCI)
- Что такое стейкинг, ставка финансирования (funding rate), open interest, ордербук
- Принципы свечного / технического анализа, паттерны (двойное дно, голова-плечи и т.д.)
- Что такое токеномика, halving, газ, gwei, газовая комиссия, mempool
- Стейблкоины, DEX vs CEX, AMM, ликвидность, slippage
- Историю/природу крупных монет (BTC, ETH, SOL, BNB, XRP и т.д.)
- Риски, психология трейдинга, базовые стратегии, money management
- Что такое DeFi / NFT / Layer-1 / Layer-2 / роллапы

Принципы ответа:
- Если знаешь — отвечай чётко, в 2-8 строк, без воды.
- НЕ давай конкретных торговых сигналов («покупай BTC сейчас», «жди пампа»). Можно объяснять что показывают индикаторы, но не «вот это сейчас купи».
- НЕ давай финансовых советов («вложи 10000 в X»). Можно объяснять механики риска.
- НЕ выдумывай актуальные цены/курсы — у тебя нет live-данных для пар вне whitelist бота.
- Если вопрос про КОНКРЕТНУЮ ПАРУ ИЗ WHITELIST на тему индикаторов — используй fetch_market_data.
- Если вопрос про погоду / политику / личные темы / не-крипту — мягко откажись:
  "Я по крипте и твоему боту. Это не моя тема."

═══ MY CONFIG REQUESTS ═══
Если юзер спрашивает «мой конфиг», «покажи конфиг», «параметры DCA», «параметры стратегии»,
«покажи force_exit / safety_order / take_profit / stoploss», «что у меня настроено»:
→ ВЫЗОВИ fetch_my_config() РОВНО ОДИН РАЗ.
→ Перешли результат ВЕРБАТИМ (один-в-один), без сокращений и без перефразировки.
→ НЕ вызывай fetch_config ДОПОЛНИТЕЛЬНО — fetch_my_config уже включает базовые
  параметры (версия, биржа, режим, стейк, стоплосс, max_open_trades, таймфрейм,
  dry_run, состояние) + кастомные (force_exit, DCA). Двойной вызов = дубликат.
→ Ничего НЕ добавляй своими словами — только то что вернул tool.
ВАЖНО: «мои настройки» / «настройки бота» — это НЕ триггер. Не отвечай на них через
fetch_my_config. Просто скажи: «Скажи "мой конфиг" — покажу параметры. На вопросы как
работает стратегия не отвечаю — закрытая логика.»

═══ HELP / CAPABILITIES REQUESTS ═══
If the user asks any of these (Russian or English):
- "что ты умеешь", "что можешь", "возможности", "помощь", "команды", "список команд",
  "что я могу спрашивать", "help", "capabilities", "что доступно", "функции"
You MUST:
  1) Call fetch_capabilities()
  2) Return the tool's text VERBATIM (one for one, do not abbreviate, do not summarize,
     do not omit sections, do not paraphrase, do not translate)
  3) You MAY add ONE short intro line above it (max 1 line), but the bulk must be the full text
Never answer this kind of question from memory — always use the tool.

═══ TOOL ARGUMENT HYGIENE ═══
NEVER pass placeholder text like "[pair]", "<pair>", "PAIR", "X/USDT" as a real value to a tool.
If you don't know the exact pair string, call fetch_whitelist or fetch_blacklist first to read
real values, then use them. Same for trade_id — get it from fetch_bot_status, never invent numbers.

═══ EXAMPLE 1 — entry with confirmation ═══
User: "открой соль на 50 в лонг"
You: call place_trade(pair="SOL/USDT:USDT", side="long", stake_amount=50)
Tool: "PENDING_CONFIRMATION (NOT executed yet). Action: Открыть LONG SOL/USDT:USDT на 50 USDT (market). ..."
You (reply): "Подтверди: открыть LONG SOL/USDT:USDT на 50 USDT (по рынку)? Ответь 'да' или 'нет'."
User: "да"
You: call confirm_action()
Tool: "EXECUTED (Открыть LONG SOL/USDT:USDT на 50 USDT (market)): {{...}}"
You (reply): "✅ Открыл LONG SOL/USDT:USDT на 50 USDT."

═══ EXAMPLE 2 — close by pair ═══
User: "закрой DOT"
You: call fetch_bot_status
Tool: [{{'trade_id': 4, 'pair': 'DOT/USDT:USDT', 'profit_pct': 1.14, ...}}, ...]
You: call close_trade(trade_id=4)
Tool: "PENDING_CONFIRMATION ... Action: Закрыть сделку #4 ..."
You (reply): "Подтверди: закрыть сделку #4 DOT/USDT:USDT (текущий PnL +1.14%)? 'да' или 'нет'."
User: "да"
You: call confirm_action()
You (reply): "✅ Закрыл сделку #4 DOT/USDT:USDT."

═══ EXAMPLE 3 — cancel ═══
User: "купи 100 BTC"
You: call place_trade(pair="BTC/USDT:USDT", side="long", stake_amount=100)
You (reply): "Подтверди: открыть LONG BTC/USDT:USDT на 100 USDT? 'да' или 'нет'."
User: "нет, передумал"
You: call cancel_pending_action()
You (reply): "Отменил."
"""

    def _fresh_whitelist_pairs(self) -> list:
        """Актуальный whitelist с REST API (не кэш). Только реальные пары бота."""
        try:
            raw = self.ft.whitelist() or {}
            pairs = raw.get("whitelist") if isinstance(raw, dict) else raw
            if isinstance(pairs, list):
                fresh = [p.strip() for p in pairs if isinstance(p, str) and p.strip()]
                if fresh:
                    self._bot_ctx["whitelist"] = fresh
                    return fresh
        except Exception as e:
            logger.warning(f"[AI] fresh whitelist error: {e}")
        return list(self._bot_ctx.get("whitelist") or [])

    def _capabilities_text(self) -> str:
        """Полный список возможностей. Отдаётся как есть в ответ на 'что ты умеешь'.
        Примеры пар — только из актуального whitelist (каждый раз с REST API)."""
        wl = self._fresh_whitelist_pairs()

        def _wl_pick(i: int) -> tuple:
            """i-я пара из whitelist; если нет — первая. Никогда не выдумываем монеты."""
            if not wl:
                p = "BTC/USDT:USDT"
                return p, p.split("/")[0]
            idx = i if i < len(wl) else 0
            p = wl[idx]
            return p, p.split("/")[0]

        ex_pair, ex_base = _wl_pick(0)
        ex_pair2, ex_base2 = _wl_pick(1)
        ex_pair3, ex_base3 = _wl_pick(2)
        # Команды в <code> — тап-копия. Лимит TG 4096 — насыщенно, но без переполнения.
        return (
            "🤖 <b>ВОЗМОЖНОСТИ ПОМОЩНИКА</b>\n"
            "Текст и голосовые, без слешей. Тап по команде — копируется.\n"
            "Пиши как удобно — сленг и опечатки понимаю.\n"
            "Примеры: бабло, навар, закрой соль, че как.\n"
            "⚠️ <i>Может ошибаться — важное проверяй в основном боте.</i>\n"
            "\n"
            "📊 <b>СТАТУС / СДЕЛКИ</b>\n"
            "<code>открытые сделки</code>\n"
            "<code>что в позициях</code>\n"
            "<code>детали сделки 7</code>\n"
            "<code>история сделок</code>\n"
            "<code>последние 10 закрытых</code>\n"
            "<code>последний вход</code>\n"
            "<code>последнее закрытие</code>\n"
            "<code>последняя закрытая сделка</code>\n"
            "<code>сколько сделок из максимума</code>\n"
            "\n"
            "💰 <b>ДЕНЬГИ</b>\n"
            "<code>баланс</code>\n"
            "<code>сколько заработал</code>\n"
            "<code>общая прибыль</code>\n"
            "<code>прибыль за день</code>\n"
            "<code>прибыль за неделю</code>\n"
            "<code>прибыль за месяц</code>\n"
            "<code>статистика бота</code>\n"
            "\n"
            "📈 <b>АНАЛИЗ</b>\n"
            "<code>эффективность по парам</code>\n"
            "<code>лучшие пары</code>\n"
            "<code>худшие пары</code>\n"
            "<code>теги входов</code>\n"
            "<code>теги выходов</code>\n"
            f"<code>свечи {ex_base} за час</code>\n"
            f"<code>5m {ex_base2}</code>\n"
            f"<code>1h {ex_base3}</code>\n"
            "\n"
            "📊 <b>ИНДИКАТОРЫ</b> (как на дашборде, без пересчёта)\n"
            f"<code>RSI на {ex_base} 15m</code>\n"
            f"<code>MACD по {ex_base2}</code>\n"
            f"<code>ADX на {ex_base3}</code>\n"
            f"<code>SuperTrend {ex_base} — куда?</code>\n"
            f"<code>Bollinger {ex_base2}</code>\n"
            f"<code>SMA 50/100/200 {ex_base3}</code>\n"
            f"<code>ATR {ex_base}</code>\n"
            f"<code>MFI {ex_base2}</code>\n"
            f"<code>KST {ex_base3}</code>\n"
            "Все значения 1:1 с дашбордом стратегии.\n"
            "\n"
            "🚫 <b>СПИСКИ И ЛОКИ</b>\n"
            "<code>whitelist</code>\n"
            "<code>blacklist</code>\n"
            f"<code>забань {ex_pair}</code>\n"
            f"<code>убери {ex_pair} из blacklist</code>\n"
            "<code>активные локи</code>\n"
            f"<code>залочь {ex_pair2} до 2026-06-01T12:00 на long</code>\n"
            f"<code>залочь {ex_pair3} до 2026-06-01T18:00 на short</code>\n"
            "<code>сними лок 5</code>\n"
            "\n"
            "🎛 <b>УПРАВЛЕНИЕ БОТОМ</b>\n"
            "<code>запусти бота</code>\n"
            "<code>останови бота</code>\n"
            "<code>запрети новые входы</code>\n"
            "<code>перезагрузи конфиг</code>\n"
            "<code>ping</code>\n"
            "\n"
            "🧭 <b>НАПРАВЛЕНИЕ РЫНКА</b> (как /marketdir)\n"
            "<code>какое направление</code>\n"
            "<code>только лонги</code>\n"
            "<code>переключи на шорт</code>\n"
            "<code>разреши оба направления</code>\n"
            "long — только лонги, short — только шорты,\n"
            "even — чередование, none — оба направления.\n"
            "\n"
            "💵 <b>ТОРГОВЛЯ</b>\n"
            "Открытие/закрытие/частичное — ×2 подтверждение кнопками.\n"
            f"<code>купи 50 USDT {ex_base} в лонг</code>\n"
            f"<code>зашорти {ex_base2} на 30 USDT</code>\n"
            f"<code>купи 50 USDT {ex_base} по цене 60000</code>\n"
            f"<code>лимит лонг {ex_base2} 100 USDT по цене 50</code>\n"
            f"<code>зашорти {ex_base3} на 20 USDT</code>\n"
            "<code>закрой сделку 7</code>\n"
            f"<code>закрой {ex_base}</code>\n"
            f"<code>закрой половину {ex_base}</code>\n"
            "<code>зафиксируй 30 USDT по сделке 7</code>\n"
            "<code>закрой всё</code>\n"
            "<code>отмени ордер по сделке 7</code>\n"
            f"<code>отмени лимит по {ex_base}</code>\n"
            "<code>перезагрузи сделку 12</code> — сразу, без подтверждения\n"
            "\n"
            "🎙 <b>ГОЛОСОВЫЕ</b>\n"
            "Запиши голосовое — Whisper распознает и выполнит как текст.\n"
            "\n"
            "🧠 <b>КОМПЛЕКСНЫЕ</b>\n"
            "<code>как там бот?</code>\n"
            "<code>сделай полную сводку</code>\n"
            "<code>по какой паре больше всего теряю?</code>\n"
            "<code>посоветуй 3 худшие пары для блэклиста</code>\n"
            "\n"
            "⚙️ <b>МОЙ КОНФИГ</b>\n"
            "<code>мой конфиг</code>\n"
            "<code>параметры DCA</code>\n"
            "Показывает: force_exit_by_days, safety_order_max_count,\n"
            "safety_order_ratio, price_deviation_initial, take_profit.\n"
            "Менять нельзя — только config.json + reload.\n"
            "\n"
            "🪙 <b>ВОПРОСЫ ПО КРИПТЕ</b> (не только про бота)\n"
            "<code>что такое DCA?</code>\n"
            "<code>как работает RSI?</code>\n"
            "<code>чем спот отличается от фьючей?</code>\n"
            "<code>что такое funding rate?</code>\n"
            "<code>объясни Bollinger Bands</code>\n"
            "<code>что такое ликвидация на плече?</code>\n"
            "<code>как читать ордербук</code>\n"
            "Без торговых сигналов и финсоветов.\n"
            "\n"
            "🔧 <b>СИСТЕМА</b>\n"
            "<code>здоровье бота</code>\n"
            "<code>версия freqtrade</code>\n"
            "<code>системная инфа CPU RAM</code>\n"
            "<code>логи бота за 30 строк</code>\n"
            "\n"
            "❌ <b>ЧТО НЕ МОГУ</b>\n"
            "• Менять TP/SL/DCA и whitelist — только config.json\n"
            "• Долить в позицию вручную — DCA делает стратегия\n"
            "• Не-крипто темы (погода, новости и т.п.)\n"
            "\n"
            "🔧 <b>СЛУЖЕБНЫЕ</b>\n"
            "<code>сброс</code> — очистить память диалога\n"
            "<code>да</code> / <code>ок</code> / <code>го</code> — подтвердить\n"
            "<code>нет</code> / <code>отмена</code> — отменить\n"
            "<code>что умеешь</code> — этот список\n"
            "Память до рестарта контейнера.\n"
            "Подтверждение сделок ждёт до 5 минут.\n"
        )

    def _split_tg_chunks(self, text: str, max_chunk: int = 3800) -> list:
        """Режет длинный текст на куски ≤ max_chunk, не ломая HTML по середине строки."""
        if len(text) <= max_chunk:
            return [text]
        chunks: list = []
        current = ""
        for sec in text.split("\n\n"):
            candidate = f"{current}\n\n{sec}" if current else sec
            if len(candidate) <= max_chunk:
                current = candidate
                continue
            if current:
                chunks.append(current)
                current = ""
            if len(sec) <= max_chunk:
                current = sec
                continue
            for line in sec.split("\n"):
                cand = f"{current}\n{line}" if current else line
                if len(cand) > max_chunk:
                    if current:
                        chunks.append(current)
                    current = line
                else:
                    current = cand
        if current:
            chunks.append(current)
        return chunks or [text[:max_chunk]]

    def _clip_tg_text(self, text: str) -> str:
        """Обрезка только если реально больше лимита TG (4096), не раньше."""
        if len(text) <= self.TG_MSG_LIMIT:
            return text
        suffix = "\n…(обрезано)"
        return text[: self.TG_MSG_LIMIT - len(suffix)] + suffix

    # gpt-4.1-mini цены (USD за 1M токенов)
    _PRICE_IN_PER_M    = 0.40
    _PRICE_CACHED_PER_M = 0.10
    _PRICE_OUT_PER_M   = 1.60

    def _format_cost_line(self, in_tok: int, cached_tok: int, out_tok: int) -> str:
        """Просто число доллара (например '$0.0042') или пустая строка если токенов нет."""
        if (in_tok or out_tok or cached_tok) == 0:
            return ""
        fresh_in = max(0, in_tok - cached_tok)
        cost = (
            fresh_in   * self._PRICE_IN_PER_M     / 1_000_000
            + cached_tok * self._PRICE_CACHED_PER_M / 1_000_000
            + out_tok    * self._PRICE_OUT_PER_M    / 1_000_000
        )
        return f"${cost:.4f}"

    def _send_capabilities(self) -> None:
        """Отправить help-список в TG (одним сообщением; fallback на split если раздут)."""
        text = self._capabilities_text()
        if len(text) <= self.TG_MSG_LIMIT:
            self._send(text, parse_mode="HTML")
            return
        chunks = self._split_tg_chunks(text)
        for i, chunk in enumerate(chunks, start=1):
            self._send(f"{chunk}\n\n<i>— {i}/{len(chunks)} —</i>", parse_mode="HTML")

    def set_market_dir_handler(self, getter, setter) -> None:
        """Привязка к market_direction стратегии (нет REST endpoint для /marketdir)."""
        self._market_dir_get = getter
        self._market_dir_set = setter

    def update_strategy_params(self, params: dict) -> None:
        """Обновить кастомные параметры стратегии в живом listener'е.
        Вызывается стратегией при каждой её повторной инициализации
        (в т.ч. после reload_config) — иначе мы навсегда отдаём цифры
        с момента первого запуска контейнера."""
        if not params:
            return
        with self._strategy_params_lock:
            self.strategy_params.update({k: v for k, v in params.items() if v is not None})

    def set_config_files(self, paths: list) -> None:
        """Сохранить список путей к config.json для re-read с диска при каждом
        запросе 'мой конфиг'. Это самый надёжный путь: live values без кэша."""
        self._config_files = [str(p) for p in (paths or []) if p]

    def _read_strategy_params_from_disk(self) -> dict:
        """Прочитать актуальные значения strategy-кастомных параметров напрямую
        из config_files (JSON) на диске. Это обходит ВСЁ кэширование — даже
        если listener был создан давно со старым snapshot. Если файлов нет
        или ошибка чтения — возвращает пустой dict (фолбэк на snapshot)."""
        if not self._config_files:
            return {}
        keys = (
            "force_exit_by_days_enabled", "force_exit_after_days",
            "price_deviation_initial", "safety_order_max_count",
            "safety_order_ratio", "safety_order_volume_scale",
        )
        out: dict = {}
        for path in self._config_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    continue
                for k in keys:
                    if k in data:
                        out[k] = data[k]
            except Exception as e:
                logger.debug(f"[AI] _read_strategy_params_from_disk: {path}: {e}")
                continue
        return out

    def _build_my_config_text(self) -> str:
        """Полный конфиг: базовые параметры бота + кастомные параметры стратегии
        (DCA + принудительный выход). Используется одной точкой входа на любые
        фразы 'мой конфиг' / 'мои настройки' / 'параметры DCA' — без дубликата.
        Источник: show_config() для базовых полей, strategy_params для кастомных
        (show_config их не возвращает)."""
        try:
            cfg = self.ft.show_config() or {}
        except Exception:
            cfg = {}

        # ПРИОРИТЕТ: 1) свежее чтение с диска, 2) snapshot из памяти.
        # Это даёт юзеру актуальные значения сразу после правки config.json,
        # без необходимости перезапускать контейнер.
        disk_params = self._read_strategy_params_from_disk()
        with self._strategy_params_lock:
            sp = dict(self.strategy_params or {})
        sp.update({k: v for k, v in disk_params.items() if v is not None})

        def _g(key, default=None):
            if key in sp and sp[key] is not None:
                return sp[key]
            return cfg.get(key, default)

        def _fmt(v):
            if v is None:
                return "не задано"
            if isinstance(v, bool):
                return "вкл" if v else "выкл"
            return f"{v}"

        def _pct(v):
            if v is None:
                return "—"
            try:
                return f"{float(v) * 100:.2f}%"
            except Exception:
                return str(v)

        # ── BASIC BOT INFO (из show_config) ────────────────────
        version          = cfg.get("version") or cfg.get("api_version") or "?"
        strategy_name    = cfg.get("strategy", "?")
        timeframe        = cfg.get("timeframe", "?")
        trading_mode     = cfg.get("trading_mode", "?")
        margin_mode      = cfg.get("margin_mode", "")
        stake_currency   = cfg.get("stake_currency", "?")
        stake_amount     = cfg.get("stake_amount", "?")
        max_open_trades  = cfg.get("max_open_trades", "?")
        stoploss         = cfg.get("stoploss", None)
        trailing_stop    = cfg.get("trailing_stop", None)
        exchange         = (cfg.get("exchange") or "?") if isinstance(cfg.get("exchange"), str) else (
            (cfg.get("exchange") or {}).get("name", "?") if isinstance(cfg.get("exchange"), dict) else "?"
        )
        dry_run          = cfg.get("dry_run", None)
        bot_state        = cfg.get("state", "?")

        tm_label = trading_mode
        if trading_mode == "futures":
            tm_label = "фьючерсы"
            if margin_mode:
                tm_label += f" ({margin_mode})"
        elif trading_mode == "spot":
            tm_label = "спот"

        run_mode_label = "dry_run (тест, без реальных сделок)" if dry_run else "live (реальные сделки)"

        # ── STRATEGY CUSTOM PARAMS ─────────────────────────────
        force_exit_after_days = _g("force_exit_after_days", None)
        force_exit_enabled    = _g("force_exit_by_days_enabled", None)
        so_ratio              = _g("safety_order_ratio", None)
        so_max_count          = _g("safety_order_max_count", None)
        so_vol_scale          = _g("safety_order_volume_scale", None)
        price_dev_initial     = _g("price_deviation_initial", None)

        lines = []
        lines.append("⚙️ <b>МОЙ КОНФИГ</b>")
        lines.append("<i>Менять эти значения я не могу — только через config.json.</i>")
        lines.append("")
        lines.append("🤖 <b>Бот / общее</b>")
        lines.append(f"• Версия: {version}")
        lines.append(f"• Стратегия: {strategy_name}")
        lines.append(f"• Биржа: {exchange}")
        lines.append(f"• Режим торговли: {tm_label}")
        lines.append(f"• Режим работы: {run_mode_label}")
        lines.append(f"• Состояние: {bot_state}")
        lines.append(f"• Таймфрейм: {timeframe}")
        lines.append("")
        lines.append("💰 <b>Деньги / лимиты</b>")
        lines.append(f"• Валюта стейка: {stake_currency}")
        lines.append(f"• Размер стейка: {stake_amount} {stake_currency if stake_currency != '?' else ''}".rstrip())
        lines.append(f"• Максимум открытых сделок: {max_open_trades}")
        lines.append(f"• Стоплосс: {_pct(stoploss)} {'(фактически отключён, выход через TP/трейлинг)' if stoploss is not None and float(stoploss) <= -0.5 else ''}".rstrip())
        lines.append(f"• Трейлинг-стоп (freqtrade core): {_fmt(trailing_stop)}")
        lines.append("")
        lines.append("⏰ <b>Принудительный выход по времени</b>")
        lines.append(f"• <code>force_exit_by_days_enabled</code> = {_fmt(force_exit_enabled)}")
        lines.append(f"• <code>force_exit_after_days</code> = {_fmt(force_exit_after_days)} дн.")
        lines.append("")
        lines.append("🛡 <b>DCA (усреднение)</b>")
        lines.append(f"• <code>price_deviation_initial</code> = {_fmt(price_dev_initial)} ({_pct(price_dev_initial)})")
        lines.append("   На сколько цена должна уйти от входа, чтобы сработала ПЕРВАЯ доливка.")
        lines.append(f"• <code>safety_order_max_count</code> = {_fmt(so_max_count)}")
        lines.append("   Максимум доливок (safety orders) в одну сделку.")
        lines.append(f"• <code>safety_order_ratio</code> = {_fmt(so_ratio)}")
        lines.append("   Множитель объёма ПЕРВОЙ доливки относительно базового стейка.")
        lines.append(f"• <code>safety_order_volume_scale</code> = {_fmt(so_vol_scale)}")
        lines.append("   Во сколько раз каждая следующая доливка больше предыдущей.")
        return "\n".join(lines)

    def _build_dca_only_text(self) -> str:
        """Только блок DCA + force_exit (без раздела «Бот/Деньги»).
        Используется когда юзер просит «параметры DCA», «настройки DCA» и т.п."""
        # Свежее чтение с диска, fallback на snapshot
        disk_params = self._read_strategy_params_from_disk()
        with self._strategy_params_lock:
            sp = dict(self.strategy_params or {})
        sp.update({k: v for k, v in disk_params.items() if v is not None})

        def _g(key, default=None):
            return sp[key] if (key in sp and sp[key] is not None) else default

        def _fmt(v):
            if v is None:
                return "не задано"
            if isinstance(v, bool):
                return "вкл" if v else "выкл"
            return f"{v}"

        def _pct(v):
            if v is None:
                return "—"
            try:
                return f"{float(v) * 100:.2f}%"
            except Exception:
                return str(v)

        force_exit_after_days = _g("force_exit_after_days", None)
        force_exit_enabled    = _g("force_exit_by_days_enabled", None)
        so_ratio              = _g("safety_order_ratio", None)
        so_max_count          = _g("safety_order_max_count", None)
        so_vol_scale          = _g("safety_order_volume_scale", None)
        price_dev_initial     = _g("price_deviation_initial", None)

        lines = []
        lines.append("🛡 <b>DCA + ПРИНУДИТЕЛЬНЫЙ ВЫХОД</b>")
        lines.append("<i>Менять — только через config.json + reload_config.</i>")
        lines.append("")
        lines.append("⏰ <b>Принудительный выход по времени</b>")
        lines.append(f"• <code>force_exit_by_days_enabled</code> = {_fmt(force_exit_enabled)}")
        lines.append(f"• <code>force_exit_after_days</code> = {_fmt(force_exit_after_days)} дн.")
        lines.append("")
        lines.append("🛡 <b>DCA (усреднение)</b>")
        lines.append(f"• <code>price_deviation_initial</code> = {_fmt(price_dev_initial)} ({_pct(price_dev_initial)})")
        lines.append("   На сколько цена должна уйти от входа, чтобы сработала ПЕРВАЯ доливка.")
        lines.append(f"• <code>safety_order_max_count</code> = {_fmt(so_max_count)}")
        lines.append("   Максимум доливок (safety orders) в одну сделку.")
        lines.append(f"• <code>safety_order_ratio</code> = {_fmt(so_ratio)}")
        lines.append("   Множитель объёма ПЕРВОЙ доливки относительно базового стейка.")
        lines.append(f"• <code>safety_order_volume_scale</code> = {_fmt(so_vol_scale)}")
        lines.append("   Во сколько раз каждая следующая доливка больше предыдущей.")
        return "\n".join(lines)

    def _summarize_pending(self, tool: str, args: dict) -> str:
        """Человекочитаемое описание ожидающего действия для подтверждения."""
        if tool == "place_trade":
            side = "LONG" if args.get("side", "long").lower() == "long" else "SHORT"
            stake = args.get("stake_amount")
            stake_txt = f"{stake} {self._bot_ctx.get('stake_currency', 'USDT')}" if stake else "(default stake)"
            price_txt = f" @ {args['price']}" if args.get("price") else " (market)"
            return f"Открыть {side} {args.get('pair')} на {stake_txt}{price_txt}"
        if tool == "close_trade":
            amt = f", частично {args['amount']}" if args.get("amount") else ""
            return f"Закрыть сделку #{args.get('trade_id')}{amt}"
        if tool == "close_all_trades":
            return "ЗАКРЫТЬ ВСЕ открытые сделки"
        if tool == "close_by_pair":
            pair = args.get("pair", "?")
            ids = args.get("_resolved_trade_ids", [])
            # Различаем закрытие позиции vs отмену лимитного ордера
            states = args.get("_resolved_trade_states", {})  # tid -> "filled" | "unfilled"
            unfilled = [i for i in ids if states.get(i) == "unfilled"]
            filled   = [i for i in ids if states.get(i) != "unfilled"]
            if not ids:
                return f"Действий по {pair} нет (открытых сделок не найдено)"
            parts = []
            if filled:
                if len(filled) == 1:
                    parts.append(f"Закрыть сделку #{filled[0]} ({pair})")
                else:
                    parts.append(f"Закрыть {len(filled)} сделок по {pair}: #" + ", #".join(str(i) for i in filled))
            if unfilled:
                if len(unfilled) == 1:
                    parts.append(f"Отменить лимитный ордер #{unfilled[0]} ({pair})")
                else:
                    parts.append(f"Отменить {len(unfilled)} лимитных ордеров по {pair}: #" + ", #".join(str(i) for i in unfilled))
            return " + ".join(parts)
        if tool == "stop_bot":
            return "🛑 ОСТАНОВИТЬ торгового бота (открытые сделки не закроются, но новые входы будут запрещены)"
        if tool == "delete_trade":
            return f"⚠️ УДАЛИТЬ запись сделки #{args.get('trade_id')} из БД (позицию на бирже не закроет — деньги могут «потеряться»)"
        return f"{tool}({args})"

    def _extract_forceenter_reason(self, pair: str) -> Optional[str]:
        """Парсит последние логи Freqtrade чтобы найти ПОЧЕМУ forceenter упал."""
        try:
            logs = self.ft.logs(limit=80) or {}
            entries = logs.get("logs", []) if isinstance(logs, dict) else []
            # Идём с конца и ищем релевантные строки для этой пары
            for entry in reversed(entries[-40:]):
                # entry — список вида [timestamp_str, ts_ms, name, level, message]
                msg = entry[-1] if isinstance(entry, (list, tuple)) and entry else str(entry)
                if not isinstance(msg, str) or pair not in msg:
                    continue
                low = msg.lower()
                # Минимальный стейк
                if "too small" in low and "adjusted stake amount" not in low:
                    import re
                    m = re.search(r"\(([\d.]+)\s*<\s*([\d.]+)\)", msg)
                    if m:
                        return (
                            f"Минимальный размер позиции для {pair} = "
                            f"{float(m.group(2)):.1f} USDT (биржевой лот). "
                            f"Попробуй открыть на сумму больше."
                        )
                if "ignoring trade" in low and "more than" in low and "bigger" in low:
                    import re
                    m = re.search(r"<\s*([\d.]+)\)", msg)
                    if m:
                        return (
                            f"Минимальный лот для {pair} ≈ {float(m.group(1)):.1f} USDT, "
                            f"а ты просишь меньше. Freqtrade не разрешает превышать "
                            f"запрошенный стейк больше чем на 30%. Попробуй открыть на сумму больше."
                        )
                if "blacklist" in low:
                    return f"{pair} в blacklist."
                if "max_open_trades" in low or "max number of trades" in low:
                    return "Достигнут лимит max_open_trades. Закрой что-то или увеличь лимит в config."
                if "force_entry" in low and "disabl" in low:
                    return "force_entry_enable=false в config."
                if "insufficient" in low or "not enough" in low:
                    return "Недостаточно средств на балансе."
            return None
        except Exception as e:
            logger.debug(f"[AI] extract reason failed: {e}")
            return None

    def _execute_real(self, name: str, args: dict) -> str:
        """Реальное выполнение мутирующих tool'ов (после подтверждения).
        Возвращает либо чистый user-friendly результат, либо строку начинающуюся с 'ERROR: ...'."""
        if name == "place_trade":
            side = str(args.get("side", "long")).lower()
            if side not in ("long", "short"):
                side = "long" if side == "buy" else "short"
            pair = args["pair"]

            # Проверяем не занята ли пара — Freqtrade не даст открыть второй forceenter
            try:
                status = self.ft.status() or []
                for t in status:
                    if t.get("pair") == pair:
                        existing_side = "SHORT" if t.get("is_short") else "LONG"
                        pnl = t.get("profit_pct")
                        pnl_txt = f", PnL {pnl:+.2f}%" if isinstance(pnl, (int, float)) else ""
                        return (
                            f"ERROR: На {pair} уже открыта сделка #{t.get('trade_id')} "
                            f"{existing_side}{pnl_txt}. Freqtrade не позволяет открыть ещё одну "
                            f"на ту же пару. Сначала закрой существующую."
                        )
            except Exception as e:
                logger.warning(f"[AI] pre-check status error: {e}")

            result = self.ft.forceenter(
                pair=pair,
                side=side,
                price=args.get("price"),
                stake_amount=args.get("stake_amount"),
                enter_tag=args.get("enter_tag"),
            )
            if isinstance(result, dict) and "error" in result:
                err = str(result["error"])
                if "Failed to enter position" in err:
                    # Пытаемся выудить реальную причину из логов Freqtrade
                    reason = self._extract_forceenter_reason(pair)
                    if reason:
                        return f"ERROR: Не удалось открыть {pair}. {reason}"
                    return (
                        f"ERROR: Не удалось открыть позицию {pair}. "
                        f"Возможные причины: стейк слишком мал для минимального лота биржи, "
                        f"max_open_trades исчерпан, force_entry_enable выключен, "
                        f"или стратегия отклонила вход."
                    )
                return f"ERROR: {err}"
            return str(result)
        if name == "close_trade":
            return str(self.ft.forceexit(
                tradeid=int(args["trade_id"]),
                amount=args.get("amount"),
            ))
        if name == "close_all_trades":
            return str(self.ft.forceexit(tradeid="all"))
        if name == "close_by_pair":
            # Re-resolve на момент исполнения — между PENDING и confirm сделки могли закрыться сами
            pair = args.get("pair", "")
            try:
                fresh = self.ft.status() or []
                ids = [t["trade_id"] for t in fresh if t.get("pair") == pair]
            except Exception as e:
                return f"ERROR: Не удалось получить статус сделок: {e}"
            if not ids:
                return f"ERROR: На {pair} больше нет открытых сделок (возможно, закрылись сами пока ты подтверждал)."
            results = []
            for tid in ids:
                try:
                    r = self.ft.forceexit(tradeid=int(tid))
                    results.append(f"#{tid}: {r}")
                except Exception as e:
                    results.append(f"#{tid}: error {e}")
            return " | ".join(results)
        if name == "stop_bot":
            return str(self.ft.stop())
        if name == "delete_trade":
            try:
                tid = int(args["trade_id"])
            except (KeyError, ValueError, TypeError):
                return "ERROR: Не указан корректный trade_id для удаления."
            return str(self.ft.delete_trade(trade_id=tid))
        return f"_execute_real: unknown tool {name}"

    def _call_tool(self, name: str, args: dict) -> str:
        try:
            # Normalize pair arg if present
            if isinstance(args, dict) and "pair" in args and args["pair"]:
                args["pair"] = self._fix_pair(args["pair"])

            # ── HELP ───────────────────────────────────────────
            if name == "fetch_capabilities":
                return self._capabilities_text()

            # ── CONFIRMATION FLOW ──────────────────────────────
            # Гейт для мутирующих сделок: не выполняем, а кладём в pending.
            # confirmation_step: 0=ждём первый клик/да, 1=ждём второй (final)
            if name in self.CONFIRMATION_REQUIRED_TOOLS:
                # Для close_by_pair заранее резолвим список trade_id и их состояние
                if name == "close_by_pair":
                    pair = args.get("pair", "")
                    try:
                        status = self.ft.status() or []
                        matches = [t for t in status if t.get("pair") == pair]
                    except Exception as e:
                        return f"Error fetching status: {e}"
                    if not matches:
                        return (
                            f"REJECTED: на {pair} нет открытых сделок или лимитных ордеров. "
                            f"Сообщи пользователю что отменять/закрывать нечего."
                        )
                    args["_resolved_trade_ids"] = [t["trade_id"] for t in matches]
                    args["_resolved_trade_states"] = {
                        t["trade_id"]: ("unfilled" if not t.get("amount") else "filled")
                        for t in matches
                    }
                summary = self._summarize_pending(name, args)
                msg_id = self._send_with_inline_keyboard(
                    text=f"⚠️ Подтверди действие:\n\n<b>{summary}</b>",
                    buttons=[[
                        {"text": "✅ Подтвердить", "callback_data": "ai_confirm:1"},
                        {"text": "❌ Отмена",      "callback_data": "ai_cancel"},
                    ]],
                )
                with self._pending_lock:
                    self._pending_action = {
                        "tool": name,
                        "args": dict(args),
                        "summary": summary,
                        "timestamp": time.time(),
                        "confirmation_step": 0,
                        "message_id": msg_id,
                    }
                logger.info(f"[AI] PENDING (step 0): {summary}")
                return (
                    f"PENDING_CONFIRMATION (NOT executed yet). "
                    f"Action: {summary}. "
                    f"System already sent inline buttons to the user. "
                    f"DO NOT send your own confirmation message — STAY SILENT (return empty content)."
                )

            if name == "confirm_action":
                with self._pending_lock:
                    pa = self._pending_action
                    if not pa:
                        return "Нет действия, ожидающего подтверждения. Запроси действие заново."
                    if time.time() - pa["timestamp"] > self.PENDING_TTL_SEC:
                        self._pending_action = None
                        return f"Подтверждение истекло (>{self.PENDING_TTL_SEC // 60} мин). Запроси действие заново."
                    step = pa.get("confirmation_step", 0)
                    if step == 0:
                        # Первый «да» из чата → переходим на шаг 2 (двойное подтверждение)
                        pa["confirmation_step"] = 1
                        pa["timestamp"] = time.time()  # сбрасываем TTL
                        # Редактируем старое сообщение если оно есть; иначе шлём новое
                        new_msg_id = pa.get("message_id")
                        if new_msg_id:
                            self._edit_message(
                                new_msg_id,
                                text=f"🔴 <b>Финальное подтверждение</b>\n\n{pa['summary']}",
                                buttons=[[
                                    {"text": "🔴 ДА, ВЫПОЛНИТЬ", "callback_data": "ai_confirm:2"},
                                    {"text": "❌ Отмена",        "callback_data": "ai_cancel"},
                                ]],
                            )
                        else:
                            new_msg_id = self._send_with_inline_keyboard(
                                text=f"🔴 <b>Финальное подтверждение</b>\n\n{pa['summary']}",
                                buttons=[[
                                    {"text": "🔴 ДА, ВЫПОЛНИТЬ", "callback_data": "ai_confirm:2"},
                                    {"text": "❌ Отмена",        "callback_data": "ai_cancel"},
                                ]],
                            )
                            pa["message_id"] = new_msg_id
                        return "FIRST_CONFIRM_OK: sent second confirmation. STAY SILENT."
                    # step == 1 → реально выполняем
                    msg_id = pa.get("message_id")
                    self._pending_action = None
                logger.info(f"[AI] FINAL CONFIRMED → executing {pa['tool']} {pa['args']}")
                result_str = str(self._execute_real(pa['tool'], pa['args']))
                is_error = result_str.startswith("ERROR:")
                if is_error:
                    err_text = result_str[len("ERROR:"):].strip()
                    final_text = f"❌ Не выполнено: {pa['summary']}\n\n{err_text}"
                else:
                    final_text = f"✅ Выполнено: {pa['summary']}"
                if msg_id:
                    self._edit_message(msg_id, text=final_text, buttons=[])
                else:
                    self._send(final_text)
                return f"EXECUTED_AND_REPORTED: {final_text[:200]}. STAY SILENT, user already sees the result."

            if name == "cancel_pending_action":
                with self._pending_lock:
                    pa = self._pending_action
                    self._pending_action = None
                if not pa:
                    return "Нечего отменять — нет ожидающего действия."
                return f"Действие отменено: {pa['summary']}"

            # ── READ-ONLY ──────────────────────────────────────
            if name == "fetch_bot_status":
                status = self.ft.status() or []
                # СЛИМ — открытые сделки. Полная нагрузка ~3k токенов/трейд × 7 = 21k.
                keep = ("trade_id", "pair", "is_short", "open_date", "open_rate",
                        "current_rate", "amount", "stake_amount", "leverage",
                        "profit_pct", "profit_abs", "current_profit_abs",
                        "stop_loss_abs", "stop_loss_pct", "enter_tag",
                        "nr_of_successful_entries", "max_rate", "min_rate")
                if isinstance(status, list):
                    status = [{k: t.get(k) for k in keep if isinstance(t, dict) and k in t} for t in status]
                return str(status)
            if name == "fetch_trade":
                return str(self.ft.trade(int(args["trade_id"])))
            if name == "fetch_count":
                return str(self.ft.count())
            if name == "fetch_profit":
                return str(self.ft.profit())
            if name == "fetch_balance":
                bal = self.ft.balance() or {}
                # Оставляем только ненулевые валюты + сводные поля
                if isinstance(bal, dict):
                    cur = bal.get("currencies", [])
                    if isinstance(cur, list):
                        cur = [c for c in cur
                               if isinstance(c, dict)
                               and (float(c.get("free") or 0) + float(c.get("used") or 0) + float(c.get("balance") or 0)) > 1e-9]
                        bal = {k: v for k, v in bal.items() if k != "currencies"}
                        bal["currencies"] = cur
                return str(bal)
            if name == "fetch_performance":
                return str(self.ft.performance())
            if name == "fetch_stats":
                return str(self.ft.stats())
            if name == "fetch_daily":
                return str(self.ft.daily(days=args.get("days")))
            if name == "fetch_weekly":
                return str(self.ft.weekly(weeks=args.get("weeks")))
            if name == "fetch_monthly":
                return str(self.ft.monthly(months=args.get("months")))
            if name == "fetch_entries":
                return str(self.ft.entries(pair=args.get("pair")))
            if name == "fetch_exits":
                return str(self.ft.exits(pair=args.get("pair")))
            if name == "fetch_mix_tags":
                return str(self.ft.mix_tags(pair=args.get("pair")))
            if name == "fetch_whitelist":
                return str(self.ft.whitelist())
            if name == "fetch_blacklist":
                return str(self.ft.blacklist())
            if name == "fetch_locks":
                return str(self.ft.locks())
            if name == "fetch_trades":
                limit = args.get("limit", 30)
                # Freqtrade REST API возвращает от СТАРОЙ к НОВОЙ при offset=0.
                # Берём с конца через offset чтобы получить ПОСЛЕДНИЕ закрытые.
                try:
                    first = self.ft.trades(limit=1) or {}
                    total = int(first.get("total_trades") or 0)
                except Exception:
                    total = 0
                if total > limit:
                    raw = self.ft.trades(limit=limit, offset=max(0, total - limit)) or {}
                else:
                    raw = self.ft.trades(limit=limit) or {}
                trades_list = raw.get("trades", []) if isinstance(raw, dict) else []
                # Сортируем по close_timestamp DESC (epoch ms — точно до миллисекунды).
                # close_date в API без миллисекунд — две сделки в одной секунде дадут
                # одинаковую строку, и стабильная сортировка вернёт неправильный порядок.
                try:
                    trades_list = sorted(
                        trades_list,
                        key=lambda t: (int(t.get("close_timestamp") or 0), t.get("close_date") or ""),
                        reverse=True,
                    )
                except Exception:
                    trades_list = list(reversed(trades_list))
                # СЛИМ — оставляем только нужные поля, иначе 50 трейдов = 80k токенов → 429
                keep = ("trade_id", "pair", "is_short", "open_date", "close_date",
                        "open_rate", "close_rate", "amount", "stake_amount",
                        "close_profit", "close_profit_abs", "exit_reason", "enter_tag")
                slim = []
                for t in trades_list:
                    if isinstance(t, dict):
                        slim.append({k: t.get(k) for k in keep if k in t})
                return str({
                    "trades": slim,
                    "count": len(slim),
                    "total_closed": total,
                    "_note": "Slim view. Sorted by close_timestamp (epoch ms, precise) DESC — index 0 = MOST RECENT closed trade. Profit in close_profit (decimal, 0.05 = +5%), absolute in close_profit_abs (USDT).",
                })
            if name == "fetch_config":
                return str(self.ft.show_config())
            if name == "fetch_my_config":
                return self._build_my_config_text()
            if name == "fetch_strategies":
                return str(self.ft.strategies())
            if name == "fetch_pairlists_available":
                return str(self.ft.pairlists_available())
            if name == "fetch_market_data":
                pair = args["pair"]
                timeframe = args["timeframe"]
                limit = args.get("limit")
                # Сначала пробуем pair_candles (live кэш стратегии)
                try:
                    res = self.ft.pair_candles(pair=pair, timeframe=timeframe, limit=limit)
                except Exception as e:
                    res = {"error": str(e), "data": []}
                # Если пусто — fallback через pair_history (исторические feather файлы)
                empty = (
                    isinstance(res, dict)
                    and not res.get("data")
                    and res.get("length", 0) == 0
                )
                if empty:
                    wl = self._bot_ctx.get("whitelist") or []
                    strat_tf = self._bot_ctx.get("timeframe", "")
                    note = ""
                    if pair not in wl:
                        note = f" Пара {pair} НЕ в whitelist стратегии, поэтому live-кэш пуст."
                    elif timeframe != strat_tf and strat_tf:
                        note = f" Таймфрейм {timeframe} не основной (стратегия работает на {strat_tf})."
                    try:
                        hist = self.ft.pair_history(
                            pair=pair,
                            timeframe=timeframe,
                            strategy="DSA_General_v2",
                        )
                        if isinstance(hist, dict) and hist.get("length", 0) > 0:
                            return str(hist) + f"\n[note: live-кэш был пуст, использован pair_history. {note}]"
                        return (
                            f"EMPTY_DATA: для {pair} на {timeframe} нет ни live-кэша, ни истории."
                            f"{note} "
                            f"Скажи пользователю причину и предложи: "
                            f"1) другой таймфрейм (например {strat_tf or '15m'}); "
                            f"2) пару из whitelist ({', '.join(wl[:6]) if wl else 'whitelist пуст'}); "
                            f"3) скачать данные через freqtrade download-data."
                        )
                    except Exception as e:
                        return (
                            f"EMPTY_DATA: pair_candles пустой, pair_history ошибся: {e}.{note} "
                            f"Скажи пользователю что данных нет — предложи другой таймфрейм/пару."
                        )
                return str(res)
            if name == "fetch_logs":
                return str(self.ft.logs(limit=args.get("limit", 50)))
            if name == "fetch_health":
                return str(self.ft.health())
            if name == "fetch_version":
                return str(self.ft.version())
            if name == "fetch_sysinfo":
                return str(self.ft.sysinfo())
            if name == "ping":
                return str(self.ft.ping())

            # ── BOT CONTROL ─────────────────────────────────────
            if name == "start_bot":
                return str(self.ft.start())
            # stop_bot теперь через CONFIRMATION_REQUIRED_TOOLS — попадает в pending
            if name == "stopentry":
                return str(self.ft.stopbuy())
            if name == "reload_config":
                return str(self.ft.reload_config())
            if name == "fetch_market_direction":
                if not self._market_dir_get:
                    return "ERROR: market_direction недоступен (listener не привязан к стратегии)."
                return str(self._market_dir_get())
            if name == "set_market_direction":
                if not self._market_dir_set:
                    return "ERROR: market_direction недоступен (listener не привязан к стратегии)."
                direction = str(args.get("direction", "")).lower().strip()
                if direction not in ("long", "short", "even", "none"):
                    return f"ERROR: неверное направление '{direction}'. Допустимо: long, short, even, none."
                return str(self._market_dir_set(direction))

            # ── TRADE ACTIONS (place/close — через confirmation выше) ──
            if name == "reload_trade":
                try:
                    tid = int(args["trade_id"])
                except (KeyError, ValueError, TypeError):
                    return "ERROR: Не указан корректный trade_id."
                return str(self.ft._post(f"trades/{tid}/reload"))
            if name == "cancel_open_order":
                try:
                    return str(self.ft.cancel_open_order(trade_id=int(args["trade_id"])))
                except (KeyError, ValueError, TypeError):
                    return "ERROR: Не указан trade_id."
            # delete_trade теперь через CONFIRMATION_REQUIRED_TOOLS

            # ── LISTS / LOCKS ───────────────────────────────────
            if name == "add_blacklist":
                return str(self.ft.blacklist(args["pair"]))
            if name == "delete_blacklist":
                pair = args.get("pair", "").strip()
                if not pair or pair.startswith("[") or pair.startswith("<"):
                    return (
                        f"REJECTED: invalid pair '{pair}'. "
                        f"You must pass a real pair string like 'DOGE/USDT:USDT', "
                        f"never a placeholder. Call fetch_blacklist first if unsure."
                    )
                try:
                    return str(self.ft._delete("blacklist", params={"pairs_to_delete": pair}))
                except Exception as e:
                    return f"delete_blacklist error: {e}"
            if name == "add_lock":
                return str(self.ft.lock_add(
                    pair=args["pair"],
                    until=args["until"],
                    side=args.get("side", "*"),
                    reason=args.get("reason", ""),
                ))
            if name == "delete_lock":
                return str(self.ft.delete_lock(int(args["lock_id"])))

            return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error calling {name}: {e}"

    def _send_typing(self) -> None:
        """Показать в TG статус «печатает…» (живёт ~5 сек, поэтому повторяем в фоне)."""
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendChatAction",
                json={"chat_id": self.chat_id, "action": "typing"},
                timeout=5,
            )
        except Exception as e:
            logger.debug(f"[AI] TG typing error: {e}")

    def _send(self, text: str, parse_mode: Optional[str] = None) -> None:
        try:
            text = self._clip_tg_text(text)
            payload = {"chat_id": self.chat_id, "text": text}
            if parse_mode:
                payload["parse_mode"] = parse_mode
            r = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json=payload,
                timeout=10,
            )
            if not r.ok:
                logger.warning(
                    f"[AI] TG send failed ({r.status_code}): "
                    f"{(r.text or '')[:300]}"
                )
        except Exception as e:
            logger.warning(f"[AI] TG send error: {e}")

    def _send_with_inline_keyboard(self, text: str, buttons: list, parse_mode: str = "HTML") -> Optional[int]:
        """Отправить сообщение с inline-кнопками в TG. Возвращает message_id или None.
        buttons = [[{"text": "...", "callback_data": "..."}], ...]"""
        try:
            text = self._clip_tg_text(text)
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "reply_markup": {"inline_keyboard": buttons},
            }
            r = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json=payload,
                timeout=10,
            )
            data = r.json() if r.status_code == 200 else {}
            return data.get("result", {}).get("message_id")
        except Exception as e:
            logger.warning(f"[AI] TG send (inline) error: {e}")
            return None

    def _edit_message(self, message_id: int, text: str, buttons: Optional[list] = None,
                      parse_mode: str = "HTML") -> None:
        """Редактирует ранее отправленное сообщение: меняет текст и убирает (или меняет) кнопки."""
        try:
            text = self._clip_tg_text(text)
            payload = {
                "chat_id": self.chat_id,
                "message_id": message_id,
                "text": text,
                "parse_mode": parse_mode,
            }
            if buttons is not None:
                payload["reply_markup"] = {"inline_keyboard": buttons}
            else:
                # Убираем клавиатуру полностью
                payload["reply_markup"] = {"inline_keyboard": []}
            requests.post(
                f"https://api.telegram.org/bot{self.token}/editMessageText",
                json=payload,
                timeout=10,
            )
        except Exception as e:
            logger.warning(f"[AI] TG edit error: {e}")

    def _answer_callback(self, callback_query_id: str, text: str = "") -> None:
        """Подтверждаем TG что клик по кнопке получен (убирает спиннер на кнопке)."""
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.token}/answerCallbackQuery",
                json={"callback_query_id": callback_query_id, "text": text},
                timeout=10,
            )
        except Exception:
            pass

    def _get_updates(self):
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{self.token}/getUpdates",
                params={"offset": self.last_update_id + 1, "timeout": 20},
                timeout=25,
            )
            if resp.status_code == 200:
                return resp.json().get("result", [])
        except Exception:
            pass
        return []

    def _transcribe_voice(self, file_id: str) -> Optional[str]:
        """Скачивает голосовое из TG и распознаёт через Whisper. None если ошибка."""
        try:
            # 1) Получаем путь к файлу
            r = requests.get(
                f"https://api.telegram.org/bot{self.token}/getFile",
                params={"file_id": file_id},
                timeout=10,
            )
            if r.status_code != 200 or not r.json().get("ok"):
                return None
            file_path = r.json()["result"]["file_path"]

            # 2) Скачиваем сам OGG
            audio = requests.get(
                f"https://api.telegram.org/file/bot{self.token}/{file_path}",
                timeout=30,
            )
            if audio.status_code != 200:
                return None

            # 3) Сохраняем во временный файл с правильным расширением
            import tempfile, os
            ext = os.path.splitext(file_path)[1] or ".ogg"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio.content)
                tmp_path = tmp.name

            # 4) Whisper transcribe
            try:
                with open(tmp_path, "rb") as f:
                    tr = self.openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        language="ru",
                    )
                text = (tr.text or "").strip()
                logger.info(f"[AI] Voice transcribed: {text!r}")
                return text or None
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[AI] voice transcribe error: {e}")
            return None

    # Ключевые подстроки — если найдены в коротком сообщении, шлём help-список
    HELP_KEYWORDS = (
        # «умеешь / умешь / умет / умеет» — все варианты через короткий корень
        "умее", "умеш", "умет", "умею",
        # «можешь / можеш / могёшь»
        "можеш", "могеш",
        # «делаешь / делашь»
        "делае", "делаш",
        # «способен / способн / способна»
        "способ",
        # классика
        "что ты можешь", "что можешь", "возможност",
        "помощ", "хелп", "хэлп", "команды", "команд",
        "список команд", "функции", "функционал",
        "что доступно", "доступно",
        "для чего ты", "зачем ты", "кто ты", "ты кто",
        "что я могу", "что мне можно", "что спросить",
        "расскажи о себе", "о тебе", "бот что", "что за бот",
        "что за помощник", "помощник что",
        "меню", "опции",
        "help", "capabilities", "commands", "what can you", "what do you",
    )

    # Команды управления самим листенером (не AI) — точные слова после нормализации
    LISTENER_RESET_WORDS = (
        "сброс", "забудь", "забыть", "reset", "clear",
        "new chat", "новый чат", "очисти", "очистить",
    )

    # Триггеры «мой конфиг» — обходят GPT, шлют форматированный HTML напрямую
    MY_CONFIG_KEYWORDS = (
        "мой конфиг", "моего конфига", "мой config",
        "покажи конфиг", "покажи config", "покажи параметры",
        "параметры стратегии",
        "что у меня настроено", "что настроено",
        # короткие формы / опечатки «мой всё / мой все / мой всф»
        "мой все", "мой всё", "мой всф",
        "мое все", "моё всё", "моё все", "мое всё",
        "что у меня",
        "show config", "my config",
    )

    # Отдельный триггер: только блок DCA + force_exit (без раздела «Бот/Деньги»)
    DCA_ONLY_KEYWORDS = (
        # явные «параметры DCA / настройки DCA»
        "параметры dca", "настройки dca", "dca настройки",
        "покажи dca", "конфиг dca", "dca конфиг",
        # короткие формы
        "мой dca", "моя dca", "моё dca", "мое dca",
        "что по dca", "по dca",
        # русский синоним
        "усреднение", "усреднения", "усреднени",
        # force_exit алиасы
        "force exit", "force_exit", "принудительный выход",
    )

    # Точное слово «dca» как самостоятельное сообщение (1-2 слова) — тоже DCA-only
    DCA_BARE_WORDS = ("dca", "дса", "дска")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Приводит текст к каноничной форме: lower, без пунктуации,
        повторяющиеся буквы (умееешь / приветтт) схлопываются до 2."""
        s = text.lower().strip()
        s = re.sub(r"[!?.,;:'\"()\[\]«»—–\-]+", " ", s)
        # схлопываем 3+ одинаковых подряд до 2 ("умееешь" → "умеешь")
        s = re.sub(r"(.)\1{2,}", r"\1\1", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _is_help_query(self, text: str) -> bool:
        """Триггер help-списка: ключевое слово в коротком (<=10 слов) сообщении."""
        norm = self._normalize_text(text)
        if not norm:
            return False
        if len(norm.split()) > 10:
            return False
        for kw in self.HELP_KEYWORDS:
            if kw in norm:
                return True
        return False

    def _is_my_config_query(self, text: str) -> bool:
        """Триггер «мой конфиг»: ключевая фраза в коротком (<=10 слов) сообщении."""
        norm = self._normalize_text(text)
        if not norm:
            return False
        if len(norm.split()) > 10:
            return False
        # «мои настройки» — намеренно НЕ триггер (см. system prompt)
        if "мои настройки" in norm or "настройки бота" in norm:
            return False
        for kw in self.MY_CONFIG_KEYWORDS:
            if kw in norm:
                return True
        return False

    def _is_dca_only_query(self, text: str) -> bool:
        """Триггер для показа ТОЛЬКО блока DCA + force_exit (без общего конфига)."""
        norm = self._normalize_text(text)
        if not norm:
            return False
        words = norm.split()
        if len(words) > 10:
            return False
        # Сообщение целиком из 1-2 слов и одно из них «dca» (точное совпадение) — ловим
        if len(words) <= 2 and any(w in self.DCA_BARE_WORDS for w in words):
            return True
        for kw in self.DCA_ONLY_KEYWORDS:
            if kw in norm:
                return True
        return False

    def _send_my_config(self) -> None:
        """Отправить «мой конфиг» с HTML-разметкой напрямую в TG (минуя GPT)."""
        text = self._build_my_config_text()
        if len(text) <= self.TG_MSG_LIMIT:
            self._send(text, parse_mode="HTML")
            return
        for chunk in self._split_tg_chunks(text):
            self._send(chunk, parse_mode="HTML")

    def _send_dca_only(self) -> None:
        """Отправить только блок DCA + force_exit с HTML-разметкой (минуя GPT)."""
        text = self._build_dca_only_text()
        self._send(text, parse_mode="HTML")

    def _handle_voice(self, file_id: str) -> None:
        """Распознать голосовое и обработать как обычное сообщение."""
        text = self._transcribe_voice(file_id)
        if not text:
            self._send("🎙 Не смог распознать голосовое сообщение. Напиши текстом.")
            return
        # Показываем что распознали — пользователь видит и может скорректировать
        self._send(f"🎙 → {text}")
        # Единая нормализация
        norm = self._normalize_text(text)
        if norm in self.LISTENER_RESET_WORDS:
            with self._history_lock:
                self._history.clear()
            self._send("🧹 История диалога очищена.")
            return
        if self._is_help_query(text):
            logger.info(f"[AI] Help query bypassed (voice): {text!r}")
            self._send_capabilities()
            return
        # DCA-only ПЕРЕД my-config (более специфичный триггер)
        if self._is_dca_only_query(text):
            logger.info(f"[AI] DCA-only bypassed (voice): {text!r}")
            self._send_dca_only()
            return
        if self._is_my_config_query(text):
            logger.info(f"[AI] My-config bypassed (voice): {text!r}")
            self._send_my_config()
            return
        logger.info(f"[AI] Получено (voice): {text}")
        self._handle_message(text)

    @staticmethod
    def _msg_role(m) -> str:
        if isinstance(m, dict):
            return str(m.get("role") or "")
        return str(getattr(m, "role", "") or "")

    @staticmethod
    def _has_tool_calls(m) -> bool:
        if isinstance(m, dict):
            return bool(m.get("tool_calls"))
        return bool(getattr(m, "tool_calls", None))

    def _trim_history(self) -> None:
        """
        Обрезает историю до HISTORY_LIMIT, но только по БЕЗОПАСНОЙ границе:
        - первое сообщение после обрезки должно быть 'user'
        - запрещено резать между assistant.tool_calls и его tool-ответами
        """
        if len(self._history) <= self.HISTORY_LIMIT:
            return
        target_drop = len(self._history) - self.HISTORY_LIMIT
        # Ищем первый user message начиная с target_drop
        safe_start = None
        for i in range(target_drop, len(self._history)):
            if self._msg_role(self._history[i]) == "user":
                safe_start = i
                break
        if safe_start is None:
            # Нет user в хвосте — оставляем всё (редкий кейс)
            return
        self._history = self._history[safe_start:]

    def _sanitize_history(self, history: list) -> list:
        """
        Защита перед отправкой в OpenAI: гарантирует что каждое 'tool' имеет
        предшествующее assistant.tool_calls с подходящим tool_call_id.
        Сиротские tool/assistant.tool_calls вычищаются.
        """
        result: list = []
        pending_tool_call_ids: set = set()  # id'шники tool_calls в последнем assistant
        for m in history:
            role = self._msg_role(m)
            if role == "tool":
                tcid = m.get("tool_call_id") if isinstance(m, dict) else getattr(m, "tool_call_id", None)
                if tcid and tcid in pending_tool_call_ids:
                    result.append(m)
                    pending_tool_call_ids.discard(tcid)
                # иначе — сирота, выкидываем
                continue
            if role == "assistant" and self._has_tool_calls(m):
                # сбрасываем ожидание — новый блок tool_calls
                tcs = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", [])
                pending_tool_call_ids = {tc["id"] if isinstance(tc, dict) else tc.id for tc in (tcs or [])}
                result.append(m)
                continue
            # обычное user / assistant без tool_calls
            pending_tool_call_ids = set()
            result.append(m)
        return result

    def _handle_message(self, text: str) -> None:
        # «Печатает…» в TG пока идёт OpenAI + tools (статус живёт ~5 сек → повтор каждые 4)
        stop_typing = threading.Event()

        def _typing_loop() -> None:
            while True:
                self._send_typing()
                if stop_typing.wait(4.0):
                    break

        typing_thread = threading.Thread(target=_typing_loop, daemon=True, name="AI-Typing")
        typing_thread.start()
        try:
            self._handle_message_inner(text)
        finally:
            stop_typing.set()
            typing_thread.join(timeout=1.0)

    def _handle_message_inner(self, text: str) -> None:
        # Обновляем контекст бота на каждом запросе (whitelist может меняться)
        # Делаем это редко: только если контекст ещё не получен
        if not self._bot_ctx.get("whitelist"):
            self._bot_ctx = self._probe_bot_context()

        with self._history_lock:
            self._history.append({"role": "user", "content": text})
            self._trim_history()
            # Снимаем копию + очищаем потенциальных сирот
            history_snapshot = self._sanitize_history(list(self._history))

        messages = [{"role": "system", "content": self._system_prompt()}] + history_snapshot

        # Накопители токенов для подсчёта стоимости ответа (gpt-4.1-mini)
        tot_in = 0
        tot_cached = 0
        tot_out = 0

        def _accumulate(resp):
            nonlocal tot_in, tot_cached, tot_out
            try:
                u = getattr(resp, "usage", None)
                if not u:
                    return
                tot_in += int(getattr(u, "prompt_tokens", 0) or 0)
                tot_out += int(getattr(u, "completion_tokens", 0) or 0)
                ptd = getattr(u, "prompt_tokens_details", None)
                if ptd:
                    tot_cached += int(getattr(ptd, "cached_tokens", 0) or 0)
            except Exception:
                pass

        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                temperature=0.3,
            )
            _accumulate(response)
            msg = response.choices[0].message

            # Локальный список для tool-цикла — не пишем tool-call'ы в долгую историю
            local = list(messages)
            new_history_additions: list = []
            iterations = 0

            iter_limit_hit = False
            ui_already_sent = False  # выставляется если tool уже отправил сообщение/кнопки в TG
            while msg.tool_calls:
                iterations += 1
                if iterations > self.MAX_TOOL_ITERATIONS:
                    logger.warning(f"[AI] Tool iteration limit ({self.MAX_TOOL_ITERATIONS}) reached")
                    iter_limit_hit = True
                    break
                # сериализуем сообщение ассистента с tool_calls в dict
                assistant_msg = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
                local.append(assistant_msg)
                new_history_additions.append(assistant_msg)

                for tool_call in msg.tool_calls:
                    try:
                        args = json.loads(tool_call.function.arguments or "{}")
                    except Exception:
                        args = {}
                    result = self._call_tool(tool_call.function.name, args)
                    logger.info(
                        f"[AI] tool={tool_call.function.name} args={args} "
                        f"result={result[:500]}{'…' if len(result) > 500 else ''}"
                    )
                    # Если тула уже сама отправила в TG — глушим финальный ответ AI
                    if any(result.startswith(prefix) for prefix in (
                        "PENDING_CONFIRMATION",
                        "FIRST_CONFIRM_OK",
                        "EXECUTED_AND_REPORTED",
                    )):
                        ui_already_sent = True
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                    local.append(tool_msg)
                    new_history_additions.append(tool_msg)

                response = self.openai.chat.completions.create(
                    model=self.model,
                    messages=local,
                    tools=self.tools,
                    temperature=0.3,
                )
                _accumulate(response)
                msg = response.choices[0].message

            # Финальный ответ ассистента
            final_text = msg.content or ""
            if iter_limit_hit and not final_text.strip():
                final_text = "Слишком много шагов подряд, остановился. Уточни запрос."
            final_assistant = {"role": "assistant", "content": final_text}
            new_history_additions.append(final_assistant)

            # Сохраняем в долгую историю всё, что появилось за этот ход
            with self._history_lock:
                self._history.extend(new_history_additions)
                self._trim_history()

            # Логируем токены всегда (для диагностики), даже если стоимость в чат не шлём
            cost_line = self._format_cost_line(tot_in, tot_cached, tot_out)
            logger.info(f"[AI cost] chat={self.chat_id} {cost_line.strip()}")

            # Глушим текстовый ответ AI если система уже отправила сообщение с кнопками/результатом
            if ui_already_sent:
                logger.info("[AI] AI text response suppressed (UI already sent)")
            elif final_text:
                if self.show_cost and cost_line:
                    final_text = f"{final_text}\n\n{cost_line}"
                self._send(final_text)

        except Exception as e:
            err_str = str(e)
            # Восстановление при битой структуре истории (сиротский tool / tool_calls)
            if ("must be a response to a preceeding message with 'tool_calls'" in err_str
                or "an assistant message with 'tool_calls'" in err_str
                or ("tool_calls" in err_str and "400" in err_str)):
                logger.warning(f"[AI] History corruption detected, resetting and retrying: {err_str}")
                with self._history_lock:
                    self._history = [{"role": "user", "content": text}]
                try:
                    fresh_messages = [
                        {"role": "system", "content": self._system_prompt()},
                        {"role": "user", "content": text},
                    ]
                    response = self.openai.chat.completions.create(
                        model=self.model,
                        messages=fresh_messages,
                        tools=self.tools,
                        temperature=0.3,
                    )
                    retry_msg = response.choices[0].message
                    if retry_msg.content:
                        self._send(retry_msg.content)
                        with self._history_lock:
                            self._history.append({"role": "assistant", "content": retry_msg.content})
                    else:
                        self._send("⚠️ Сбросил историю диалога из-за внутренней ошибки. Попробуй ещё раз.")
                    return
                except Exception as e2:
                    self._send(f"❌ Не получилось восстановиться: {e2}")
                    logger.error(f"[AI] Retry after history reset failed: {e2}", exc_info=True)
                    return
            # Короткое сообщение для юзера: rate-limit отдельно, остальные — без жирного JSON-дампа
            err_text = str(e)
            if "rate_limit" in err_text.lower() or "429" in err_text:
                # Вытаскиваем сколько секунд ждать
                import re as _re
                wait_m = _re.search(r"try again in ([\d.]+)\s*s", err_text)
                wait = wait_m.group(1) if wait_m else "~30"
                self._send(f"⏳ OpenAI лимит токенов. Попробуй через {wait} сек.")
            elif len(err_text) > 200:
                self._send(f"❌ Ошибка AI: {err_text[:200]}…")
            else:
                self._send(f"❌ Ошибка AI: {err_text}")
            logger.error(f"[AI] OpenAI error: {e}", exc_info=True)

    def _handle_callback_query(self, cb: dict) -> None:
        """Обработка клика по inline-кнопке. Редактируем оригинальное сообщение, не шлём новые."""
        cb_id = cb.get("id")
        data = cb.get("data", "")
        from_chat = str(cb.get("message", {}).get("chat", {}).get("id", ""))
        msg_id = cb.get("message", {}).get("message_id")
        if from_chat != self.chat_id:
            self._answer_callback(cb_id, "Чужой чат")
            return

        if data == "ai_cancel":
            with self._pending_lock:
                pa = self._pending_action
                self._pending_action = None
            self._answer_callback(cb_id, "Отменено")
            text = f"❌ Отменено: {pa['summary']}" if pa else "❌ Нечего отменять"
            if msg_id:
                self._edit_message(msg_id, text=text, buttons=[])
            else:
                self._send(text)
            return

        if data == "ai_confirm:1":
            # Первый клик — поднимаем на шаг 2 (редактируем то же сообщение)
            with self._pending_lock:
                pa = self._pending_action
                if not pa:
                    self._answer_callback(cb_id, "Действие истекло")
                    if msg_id:
                        self._edit_message(msg_id, text="⏱ Действие уже не ожидает подтверждения", buttons=[])
                    return
                if time.time() - pa["timestamp"] > self.PENDING_TTL_SEC:
                    self._pending_action = None
                    self._answer_callback(cb_id, "Истекло (>5 мин)")
                    if msg_id:
                        self._edit_message(msg_id, text="⏱ Подтверждение истекло. Запроси заново.", buttons=[])
                    return
                pa["confirmation_step"] = 1
                pa["timestamp"] = time.time()
                pa["message_id"] = msg_id
                summary = pa["summary"]
            self._answer_callback(cb_id)
            self._edit_message(
                msg_id,
                text=f"🔴 <b>Финальное подтверждение</b>\n\n{summary}",
                buttons=[[
                    {"text": "🔴 ДА, ВЫПОЛНИТЬ", "callback_data": "ai_confirm:2"},
                    {"text": "❌ Отмена",        "callback_data": "ai_cancel"},
                ]],
            )
            return

        if data == "ai_confirm:2":
            with self._pending_lock:
                pa = self._pending_action
                self._pending_action = None
            if not pa:
                self._answer_callback(cb_id, "Нет действия")
                if msg_id:
                    self._edit_message(msg_id, text="⏱ Действие уже завершено.", buttons=[])
                return
            if time.time() - pa["timestamp"] > self.PENDING_TTL_SEC:
                self._answer_callback(cb_id, "Истекло")
                if msg_id:
                    self._edit_message(msg_id, text="⏱ Подтверждение истекло.", buttons=[])
                return
            self._answer_callback(cb_id, "Выполняю...")
            logger.info(f"[AI] FINAL CONFIRMED (button) → executing {pa['tool']} {pa['args']}")
            result_str = str(self._execute_real(pa["tool"], pa["args"]))
            is_error = result_str.startswith("ERROR:")
            if is_error:
                err_text = result_str[len("ERROR:"):].strip()
                final_text = f"❌ Не выполнено: {pa['summary']}\n\n{err_text}"
            else:
                final_text = f"✅ Выполнено: {pa['summary']}"
            if msg_id:
                self._edit_message(msg_id, text=final_text, buttons=[])
            else:
                self._send(final_text)
            return

        # Неизвестный callback
        self._answer_callback(cb_id, "Неизвестно")

    def _loop(self) -> None:
        logger.info("[AI] Telegram AI listener started")
        while self._running:
            updates = self._get_updates()
            for update in updates:
                self.last_update_id = update["update_id"]
                # Inline-кнопки приходят как callback_query, а не как message
                if "callback_query" in update:
                    try:
                        self._handle_callback_query(update["callback_query"])
                    except Exception as e:
                        logger.error(f"[AI] callback error: {e}", exc_info=True)
                    continue
                msg = update.get("message", {})
                if str(msg.get("chat", {}).get("id", "")) != self.chat_id:
                    continue
                # Игнорируем сообщения от ботов (включая возможный echo от самого себя)
                sender = msg.get("from", {}) or {}
                if sender.get("is_bot"):
                    continue
                if self._self_bot_id and sender.get("id") == self._self_bot_id:
                    continue
                text = (msg.get("text") or "").strip()
                # Дедупликация: тот же текст в пределах 10 сек — пропускаем
                now_ts = time.time()
                if text and text == self._last_processed_text and (now_ts - self._last_processed_ts) < 10.0:
                    logger.info(f"[AI] Дубль пропущен (<10с): {text[:50]!r}")
                    continue
                if text:
                    self._last_processed_text = text
                    self._last_processed_ts = now_ts

                # ── Voice / audio messages ───────────────────────
                if not text:
                    voice = msg.get("voice") or msg.get("audio") or msg.get("video_note")
                    if voice and voice.get("file_id"):
                        file_id = voice["file_id"]
                        logger.info(f"[AI] Voice received, transcribing...")
                        # Транскрипция в отдельном потоке (не блокируем polling)
                        threading.Thread(
                            target=self._handle_voice,
                            args=(file_id,),
                            daemon=True,
                        ).start()
                        continue

                if not text or text.startswith("/"):
                    continue
                # Команды управления самим AI listener (нормализация — единая)
                norm = self._normalize_text(text)
                if norm in self.LISTENER_RESET_WORDS:
                    with self._history_lock:
                        self._history.clear()
                    self._send("🧹 История диалога очищена. Начинаем с чистого листа.")
                    continue
                # Help/capabilities — обходим GPT, гарантируем полный текст
                if self._is_help_query(text):
                    logger.info(f"[AI] Help query bypassed to direct response: {text!r}")
                    self._send_capabilities()
                    continue
                # DCA-only ПЕРЕД my-config (более специфичный триггер)
                if self._is_dca_only_query(text):
                    logger.info(f"[AI] DCA-only bypassed to direct response: {text!r}")
                    self._send_dca_only()
                    continue
                # «Мой конфиг» — обходим GPT, шлём с HTML-разметкой
                if self._is_my_config_query(text):
                    logger.info(f"[AI] My-config bypassed to direct response: {text!r}")
                    self._send_my_config()
                    continue
                logger.info(f"[AI] Получено: {text}")
                threading.Thread(target=self._handle_message, args=(text,), daemon=True).start()
            time.sleep(1)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        # Узнаём свой bot_id чтобы фильтровать эхо-сообщения от самого себя
        try:
            r = requests.get(f"https://api.telegram.org/bot{self.token}/getMe", timeout=5)
            if r.ok and r.json().get("ok"):
                me = r.json().get("result", {})
                self._self_bot_id = me.get("id")
                self._self_bot_username = me.get("username")
                logger.info(f"[AI] Self bot identified: @{self._self_bot_username} (id={self._self_bot_id})")
        except Exception as e:
            logger.warning(f"[AI] Не удалось получить self bot info: {e}")
        thread_name = _AI_THREAD_PREFIX + (self.token or "")[:10]
        threading.Thread(target=self._loop, daemon=True, name=thread_name).start()
        # Scheduler для авто-сводок (если настроен)
        if self.daily_summary_time:
            threading.Thread(target=self._scheduler_loop, daemon=True,
                             name=thread_name + "-sched").start()
            logger.info(f"[AI] Scheduler запущен (daily summary @ {self.daily_summary_time})")

    def stop(self) -> None:
        self._running = False

    # ── Авто-сводка ──────────────────────────────────────────

    def _scheduler_loop(self) -> None:
        """Фоновый цикл, проверяет каждую минуту нужно ли слать сводку.
        Использует часовой пояс из self.summary_tz (например Europe/Moscow),
        иначе локальное время контейнера (обычно UTC)."""
        from datetime import datetime as _dt
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(self.summary_tz) if self.summary_tz else None
        except Exception as e:
            logger.warning(f"[AI] неизвестный timezone '{self.summary_tz}': {e}. Используется локальное время.")
            tz = None
        sent_for_date = None
        while self._running:
            try:
                now = _dt.now(tz) if tz else _dt.now()
                hhmm = now.strftime("%H:%M")
                today_key = now.strftime("%Y-%m-%d")
                if hhmm == self.daily_summary_time and sent_for_date != today_key:
                    logger.info(f"[AI] Отправка ежедневной сводки за {today_key} (tz={tz or 'local'})")
                    try:
                        text = self._build_daily_summary()
                        self._send(text)
                        sent_for_date = today_key
                    except Exception as e:
                        logger.error(f"[AI] Ошибка построения сводки: {e}", exc_info=True)
            except Exception as e:
                logger.warning(f"[AI] scheduler loop error: {e}")
            time.sleep(30)  # чаще минуты чтобы точно не пропустить

    def _build_daily_summary(self) -> str:
        """Собирает богатую сводку из множества tools."""
        from datetime import datetime as _dt
        # Часовой пояс для отображения времени в сводке (берём тот же что у scheduler).
        # По дефолту — Europe/Moscow, чтобы все часы были по МСК.
        _tz = None
        try:
            from zoneinfo import ZoneInfo
            tz_name = self.summary_tz or "Europe/Moscow"
            _tz = ZoneInfo(tz_name)
        except Exception:
            _tz = None

        def safe(fn, default=None):
            try:
                return fn()
            except Exception as e:
                logger.debug(f"[AI] summary tool error: {e}")
                return default

        status   = safe(lambda: self.ft.status() or [], [])
        profit   = safe(lambda: self.ft.profit() or {}, {})
        balance  = safe(lambda: self.ft.balance() or {}, {})
        daily    = safe(lambda: self.ft.daily(days=2) or {}, {})
        weekly   = safe(lambda: self.ft.weekly(weeks=2) or {}, {})
        perf     = safe(lambda: self.ft.performance() or [], [])
        # ИСПРАВЛЕНО: API возвращает от СТАРОЙ к НОВОЙ при offset=0.
        # Берём ПОСЛЕДНИЕ 50 закрытых через offset, чтобы корректно увидеть «за 12 часов».
        try:
            first = self.ft.trades(limit=1) or {}
            total = int(first.get("total_trades") or 0)
        except Exception:
            total = 0
        if total > 50:
            trades = safe(lambda: self.ft.trades(limit=50, offset=max(0, total - 50)) or {}, {})
        else:
            trades = safe(lambda: self.ft.trades(limit=50) or {}, {})
        stats    = safe(lambda: self.ft.stats() or {}, {})

        now_str = _dt.now(_tz).strftime("%d.%m %H:%M") if _tz else _dt.now().strftime("%d.%m %H:%M")
        tz_label = " МСК" if _tz else ""
        lines = [f"🌅 Сводка — {now_str}{tz_label}", ""]

        # ── Позиции ────────────
        n = len(status)
        if n:
            longs  = sum(1 for t in status if not t.get("is_short"))
            shorts = n - longs
            total_stake = sum(float(t.get("stake_amount") or 0) for t in status)
            total_pnl   = sum(float(t.get("profit_abs") or 0) for t in status)
            pnl_pct     = (total_pnl / total_stake * 100) if total_stake else 0
            lines += [
                f"📊 ПОЗИЦИИ (открыто: {n})",
                f"Total stake: {total_stake:.1f} USDT",
                f"Unrealized PnL: {total_pnl:+.2f} USDT ({pnl_pct:+.2f}%)",
                f"LONG: {longs} · SHORT: {shorts}",
                "",
            ]
        else:
            lines += ["📊 ПОЗИЦИИ: открытых сделок нет", ""]

        # ── За ночь ─────────────
        closed_recent = []
        opened_recent = []
        cutoff_hours = 12
        cutoff_ts = time.time() - cutoff_hours * 3600
        for t in (trades.get("trades") or []):
            close_ts = t.get("close_timestamp")
            if close_ts and close_ts / 1000 > cutoff_ts:
                closed_recent.append(t)
        for t in status:
            open_ts = t.get("open_timestamp")
            if open_ts and open_ts / 1000 > cutoff_ts:
                opened_recent.append(t)

        lines.append(f"📈 ЗА ПОСЛЕДНИЕ {cutoff_hours}ч")
        if closed_recent:
            lines.append(f"Закрыто: {len(closed_recent)}")
            for t in closed_recent[:5]:
                side = "SHORT" if t.get("is_short") else "LONG"
                pp   = float(t.get("profit_ratio") or 0) * 100
                tag  = t.get("exit_reason", "—")
                lines.append(f"  • {t.get('pair','?')} {side} {pp:+.2f}% ({tag})")
        else:
            lines.append("Закрытий нет")
        if opened_recent:
            lines.append(f"Открыто: {len(opened_recent)}")
            for t in opened_recent[:5]:
                side = "SHORT" if t.get("is_short") else "LONG"
                lines.append(f"  • {t.get('pair','?')} {side} @ {t.get('open_rate')}")
        lines.append("")

        # ── Баланс ─────────────
        free = balance.get("starting_capital") or balance.get("total")
        for cur in (balance.get("currencies") or []):
            if cur.get("currency") == self._bot_ctx.get("stake_currency", "USDT"):
                free = cur.get("free")
                break
        total_wallet = balance.get("total")
        change_24h = profit.get("profit_closed_coin")
        lines += ["💰 БАЛАНС"]
        if free is not None:
            lines.append(f"USDT свободно: {float(free):.2f}")
        if total_wallet is not None:
            lines.append(f"Всего: {float(total_wallet):.2f} USDT")
        if change_24h is not None:
            lines.append(f"Закрытая прибыль всего: {float(change_24h):+.2f} USDT")
        lines.append("")

        # ── Топ пар за неделю ────
        if perf:
            top = sorted(perf, key=lambda x: x.get("profit_abs", 0), reverse=True)[:3]
            lines.append("🏆 ТОП ПАР")
            medals = ["🥇", "🥈", "🥉"]
            for i, p in enumerate(top):
                pair = p.get("pair", "?")
                prof = p.get("profit_abs", 0)
                cnt  = p.get("count", 0)
                lines.append(f"{medals[i]} {pair}: {prof:+.2f} USDT ({cnt} сделок)")
            lines.append("")

        # ── Внимание ────────────
        warnings = []
        for t in status:
            try:
                age_h = (time.time() - t["open_timestamp"] / 1000) / 3600
                pnl_pct = float(t.get("profit_ratio") or 0) * 100
                if age_h > 72:  # > 3 дней
                    warnings.append(f"• {t['pair']} #{t['trade_id']} висит {age_h/24:.1f} дн, PnL {pnl_pct:+.2f}%")
                elif pnl_pct < -5:
                    warnings.append(f"• {t['pair']} #{t['trade_id']} в минусе {pnl_pct:+.2f}%")
            except Exception:
                pass
        if warnings:
            lines.append("⚠️ ТРЕБУЮТ ВНИМАНИЯ")
            lines += warnings[:5]
            lines.append("")

        # ── Общая статистика ───────
        def _fmt_dur(sec):
            """Форматирует секунды/строку в человекочитаемое 'Xд Yч' или 'Xч Yм'."""
            try:
                if isinstance(sec, str):
                    # Иногда API отдаёт '1 day, 12:34:56' — пробуем распарсить
                    if "day" in sec:
                        parts = sec.split(",")
                        days = int(parts[0].split()[0])
                        hh = parts[1].strip().split(":")[0] if len(parts) > 1 else "0"
                        return f"{days}д {int(hh)}ч"
                    if ":" in sec:
                        h, m, _ = sec.split(":")
                        return f"{int(h)}ч {int(m)}м"
                    sec = float(sec)
                s = float(sec)
                if s >= 86400:
                    return f"{s/86400:.1f}д"
                if s >= 3600:
                    return f"{s/3600:.1f}ч"
                if s >= 60:
                    return f"{s/60:.0f}м"
                return f"{s:.0f}с"
            except Exception:
                return str(sec)

        lines.append("📈 ОБЩАЯ СТАТИСТИКА")
        # Winrate + win/loss/draw counts
        wr = stats.get("winrate")
        wins_total = profit.get("winning_trades") or 0
        losses_total = profit.get("losing_trades") or 0
        n_total = profit.get("closed_trade_count") or profit.get("trade_count") or (wins_total + losses_total)
        try:
            if wr is not None:
                lines.append(f"Winrate: {float(wr) * 100:.1f}% ({int(wins_total)}W / {int(losses_total)}L из {int(n_total)})")
            elif n_total:
                wr2 = (wins_total / n_total * 100) if n_total else 0
                lines.append(f"Winrate: {wr2:.1f}% ({int(wins_total)}W / {int(losses_total)}L из {int(n_total)})")
        except Exception:
            pass
        # Средняя прибыль/убыток на сделку
        avg_profit_pct = profit.get("profit_closed_percent_mean") or profit.get("profit_all_percent_mean")
        if avg_profit_pct is not None:
            try:
                lines.append(f"Средн. прибыль/сделку: {float(avg_profit_pct) * 100:+.2f}%")
            except Exception:
                pass
        # Лучшая / худшая закрытая сделка
        best = profit.get("best_trade")
        worst = profit.get("worst_trade")
        try:
            if best is not None:
                bp = best.get("profit_ratio", 0) if isinstance(best, dict) else 0
                bpair = best.get("pair", "?") if isinstance(best, dict) else "?"
                lines.append(f"Лучшая: {bpair} {float(bp)*100:+.2f}%")
        except Exception:
            pass
        try:
            if worst is not None:
                wp = worst.get("profit_ratio", 0) if isinstance(worst, dict) else 0
                wpair = worst.get("pair", "?") if isinstance(worst, dict) else "?"
                lines.append(f"Худшая: {wpair} {float(wp)*100:+.2f}%")
        except Exception:
            pass
        # Время в сделках — раздельно по выигрышным/проигрышным/общему
        durs = stats.get("durations") or {}
        avg_w = durs.get("wins")
        avg_l = durs.get("losses")
        avg_all = durs.get("total") or avg_w or avg_l
        if avg_all is not None:
            lines.append(f"Среднее время сделки: {_fmt_dur(avg_all)}")
        if avg_w is not None and avg_w != avg_all:
            lines.append(f"  • прибыльные: {_fmt_dur(avg_w)}")
        if avg_l is not None and avg_l != avg_all:
            lines.append(f"  • убыточные: {_fmt_dur(avg_l)}")
        # Сделок в день (за последние 7 дней)
        try:
            if daily and isinstance(daily, dict):
                d_data = daily.get("data") or []
                if d_data:
                    trades_per_day = sum(int(d.get("trade_count", 0) or 0) for d in d_data) / max(1, len(d_data))
                    lines.append(f"Сделок/день: {trades_per_day:.1f}")
        except Exception:
            pass
        # Самый частый exit_reason
        try:
            exits = stats.get("exit_reasons") or {}
            if exits:
                top_exit = max(exits.items(), key=lambda x: int(x[1].get("count", 0)) if isinstance(x[1], dict) else int(x[1] or 0))
                cnt = top_exit[1].get("count") if isinstance(top_exit[1], dict) else top_exit[1]
                lines.append(f"Чаще всего выходит по: {top_exit[0]} ({cnt}×)")
        except Exception:
            pass

        return "\n".join(lines).strip()


# ===========================================================================
# Helper: SuperTrend (FutureWarning fix — .iloc= заменён на .at[])
# ===========================================================================

def supertrend(dataframe: DataFrame, period: int = 10, multiplier: float = 3.0):
    df = dataframe.copy()
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    basic_ub = (high + low) / 2 + multiplier * atr
    basic_lb = (high + low) / 2 - multiplier * atr

    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    direction = pd.Series(1, index=df.index)

    idx = df.index
    for i in range(period, len(df)):
        prev_ub = final_ub.at[idx[i - 1]]
        prev_lb = final_lb.at[idx[i - 1]]
        prev_close = close.at[idx[i - 1]]
        cur_bub = basic_ub.at[idx[i]]
        cur_blb = basic_lb.at[idx[i]]

        final_ub.at[idx[i]] = cur_bub if (cur_bub < prev_ub or prev_close > prev_ub) else prev_ub
        final_lb.at[idx[i]] = cur_blb if (cur_blb > prev_lb or prev_close < prev_lb) else prev_lb

    supertrend_up = pd.Series(np.nan, index=df.index)
    supertrend_down = pd.Series(np.nan, index=df.index)

    for i in range(period, len(df)):
        prev_dir = direction.at[idx[i - 1]]
        cur_close = close.at[idx[i]]
        cur_ub = final_ub.at[idx[i]]
        cur_lb = final_lb.at[idx[i]]

        if prev_dir == 1:
            if cur_close > cur_lb:
                direction.at[idx[i]] = 1
                supertrend_up.at[idx[i]] = cur_lb
            else:
                direction.at[idx[i]] = -1
                supertrend_down.at[idx[i]] = cur_ub
        else:
            if cur_close < cur_ub:
                direction.at[idx[i]] = -1
                supertrend_down.at[idx[i]] = cur_ub
            else:
                direction.at[idx[i]] = 1
                supertrend_up.at[idx[i]] = cur_lb

    return supertrend_up, supertrend_down


# ===========================================================================
# Strategy
# ===========================================================================

# =============================================================================
# СПРАВКА: ВСЕ TELEGRAM-АЛЕРТЫ ЭТОЙ СТРАТЕГИИ (примеры реального текста)
# =============================================================================
# Ничего из этого не влияет на торговую логику — только уведомления в тот же
# TG-чат, что и штатные сообщения бота (self._tg_token/self._tg_chat_id из
# config.json → "telegram"). Актуальный текст всегда смотри в коде метода,
# это just quick-reference, чтобы не листать весь файл.
#
# ── Обычные (не критические) — order_filled() ──────────────────────────────
#
#   🟢 ВХОД
#   Бот: DSA General v2 · DRY-RUN
#   Пара: BTC/USDT:USDT · LONG
#   Тег: entry
#   Вложено: 50.00 USDT
#   В позиции: 50.00 USDT
#   Баланс: 1000.00 USDT
#
#   🔵 DCA #1
#   Бот: DSA General v2 · DRY-RUN
#   Пара: BTC/USDT:USDT · LONG
#   Доливка: 60.00 USDT
#   Всего в позиции: 110.00 USDT
#   Баланс: 940.00 USDT
#
#   🔴 ВЫХОД
#   Бот: DSA General v2 · DRY-RUN
#   Пара: BTC/USDT:USDT · LONG
#   Причина: take_profit
#   PnL: +2.34% (+12.50 USDT)
#   Было в позиции: 110.00 USDT
#   Баланс: 1012.50 USDT
#
# ── Алерт 1: половина DCA использована — _check_dca_usage_alerts() ─────────
#
#   🟡 ПОЛОВИНА DCA ИСПОЛЬЗОВАНА
#
#   Пара: BTC/USDT:USDT
#   Ордеров DCA: 4 из 7 (57%)
#   Текущий PnL позиции: -8.20%
#   Средняя цена входа: 61234.5
#   Баланс позиции: 310.00 USDT
#
#   ⚠️ Ещё 3 доборов в запасе, следи внимательнее
#
# ── Алерт 2: все DCA использованы (шлётся ДВАЖДЫ подряд, НЕ закрепляется —
# закреплено может быть только приветственное сообщение со списком команд) —
# _check_dca_usage_alerts() ─────────────────────────────────────────────────
#
#   🚨🆘🚨🆘🚨 ВСЕ DCA ИСПОЛЬЗОВАНЫ 🚨🆘🚨🆘🚨
#
#   Пара: BTC/USDT:USDT
#   Ордеров DCA: 7 из 7 (100%)
#   PnL позиции: -18.40%
#   Подушки больше НЕТ
#
#   ❗️❗️❗️ БУДЬТЕ ОЧЕНЬ ВНИМАТЕЛЬНЫ ❗️❗️❗️
#   Следующий сценарий — только SL/ликвидация
#   или ручное решение
#
# ── Алерт 3: глубокий минус по позиции — _check_deep_drawdown_alert() ───────
# (эскалация по порогам alert_deep_drawdown_thresholds, по одному разу на
# каждый пробитый порог: -15/-20/-25/-30% по умолчанию)
#
#   🔻 ПОЗИЦИЯ В ГЛУБОКОМ МИНУСЕ
#
#   Пара: BTC/USDT:USDT
#   PnL: -20.15%
#   Время в позиции: 2д 6ч
#   DCA использовано: 5/7
#
#   Порог -20% пробит, следим за развитием
#
# ── Алерт 4: СТОП-ЛОСС / ЛИКВИДАЦИЯ — _check_stoploss_liquidation_alert() ───
# (шлётся ДВАЖДЫ подряд, НЕ закрепляется — отдельно от обычного 🔴 ВЫХОД)
#
#   🚨🆘🔴🆘🔴🆘🚨 СТОП-ЛОСС СРАБОТАЛ 🚨🆘🔴🆘🔴🆘🚨
#
#   ПАРА: BTC/USDT:USDT
#   УБЫТОК: -45.30 USDT (-29.80%)
#   БАЛАНС ПОСЛЕ: 954.70 USDT
#
#   ❗️❗️❗️ КРИТИЧЕСКОЕ СОБЫТИЕ ❗️❗️❗️
#   Требуется анализ причины
#
#   (для ликвидации заголовок меняется на "ЛИКВИДАЦИЯ", остальной текст тот же)
#
# ── Алерт 5: тейк-профит активирован — custom_exit() ────────────────────────
#
#   🎯 ТЕЙК-ПРОФИТ АКТИВИРОВАН
#
#   Пара: BTC/USDT:USDT
#   Текущий профит: +2.15%
#
# ── Алерт 6: ордер не исполнился — _check_rejected_cancelled_orders_alert() ─
# (строго live/dry-run, см. _live_alerts_enabled_now — в бэктесте не шлётся)
#
#   ⛔ ОРДЕР НЕ ИСПОЛНИЛСЯ
#
#   Пара: BTC/USDT:USDT
#   Тип: limit (entry)
#   Причина: insufficient funds
#   Объём: 0.0012
#
#   Сделка не была открыта/изменена как ожидалось
#
# ── Алерт 7: аномальная задержка исполнения — _check_order_latency_alert() ──
# (строго live/dry-run, порог alert_order_latency_threshold_seconds, 30 сек.
# ТОЛЬКО для MARKET-ордеров — для LIMIT время created→filled это не
# латентность, а нормальное ожидание цены, поэтому limit-ордера в эту
# проверку сознательно не попадают, см. код метода)
#
#   🐢 АНОМАЛЬНАЯ ЗАДЕРЖКА ИСПОЛНЕНИЯ
#
#   Пара: BTC/USDT:USDT
#   Тип ордера: market (exit)
#   Задержка: 47 сек (порог 30 сек)
#
#   ⚠️ Возможны проблемы с биржей или сетью
#
# ── Алерт 12: просадка ОБЩЕГО баланса — _check_balance_drawdown_alert() ─────
# (эскалация по порогам alert_balance_drawdown_thresholds от пикового
# баланса за всё время работы процесса: -10/-15/-20/-25% по умолчанию,
# жёстко зашиты в коде стратегии, НЕ в config.json)
#
#   📉 ПРОСАДКА ОБЩЕГО БАЛАНСА
#
#   Текущий баланс: 850.00 USDT
#   Пиковый баланс: 1000.00 USDT
#   Просадка: -15.00%
#
#   Порог -15% пробит, следим за развитием
#
# ── Алерт 13: просадка ЗАВЕРШЕНА — _check_balance_drawdown_alert() ──────────
# (шлётся один раз, когда баланс отрастает обратно выше самого мягкого
# порога просадки, после того как до этого была зафиксирована просадка)
#
#   ✅ ПРОСАДКА ЗАВЕРШЕНА
#
#   Текущий баланс: 920.00 USDT
#   Пиковый баланс: 1000.00 USDT
#   Текущее отклонение от пика: -8.00%
#
#   Баланс вернулся выше порога -10%
#
# ── Алерт 14: рост ОБЩЕЙ прибыли — _check_balance_growth_alert() ───────────
# (эскалация по порогам alert_balance_growth_thresholds от стартового
# баланса — первый замер после запуска бота: +10/+20/+30/+50% по умолчанию,
# жёстко зашиты в коде стратегии, НЕ в config.json)
#
#   🚀 РОСТ ОБЩЕЙ ПРИБЫЛИ
#
#   Текущий баланс: 1300.00 USDT
#   Стартовый баланс: 1000.00 USDT
#   Рост: +30.00%
#
#   Порог +30% пробит 🎉
#
# ── Алерт 15: концентрация риска по одной паре — _check_pair_concentration_alert() ─
# (привязан к сделке, проверяется каждую свечу в custom_exit; порог
# alert_pair_concentration_threshold_pct, дефолт 30%)
#
#   ⚠️ КОНЦЕНТРАЦИЯ РИСКА ПО ПАРЕ
#
#   Пара: ENA/USDT:USDT
#   В позиции: 320.00 USDT
#   Баланс: 1000.00 USDT
#   Доля от баланса: 32.0%
#
#   Порог 30% превышен — большая часть капитала в одной паре
#
# ── Алерт 16: несколько пар одновременно в тяжёлом DCA —
# _check_multi_dca_portfolio_alert() (портфель целиком, проверяется каждую
# свечу в bot_loop_start; пороги alert_multi_dca_pair_pct=70%,
# alert_multi_dca_min_pairs=3)
#
#   🔴 НЕСКОЛЬКО ПАР В ТЯЖЁЛОМ DCA
#
#   Пар в зоне риска (≥70% доборов): 3
#
#   • ENA/USDT:USDT — 100% (6/6)
#   • DOGE/USDT:USDT — 86% (6/7)
#   • ADA/USDT:USDT — 71% (5/7)
#
#   ⚠️ Общий риск портфеля повышен, не только по одной паре
#
# ── Алерт 17: выход из тяжёлого DCA — _check_heavy_dca_recovery_alert()
# (привязан к сделке, проверяется каждую свечу в custom_exit; пороги
# alert_heavy_dca_pct=50%, alert_heavy_dca_recovery_pnl_pct=-2%; повторяемый —
# alert_heavy_dca_rearm_drop_pct=2 п.п.: если PnL после восстановления снова
# уйдёт ниже -4% (-2% минус 2 п.п.), алерт "перезарядится" и при следующем
# восстановлении до -2% сработает заново)
#
#   🟢 ВЫХОД ИЗ ТЯЖЁЛОГО DCA
#
#   Пара: ENA/USDT:USDT
#   Доборов было использовано: 4 из 6 (67%)
#   Текущий PnL: +0.85%
#   Средняя цена входа: 0.4521
#
#   Усреднение сработало — позиция восстановилась без новых доборов
#
# ── Алерт 18: DCA "застрял" — _check_dca_stuck_alert() ──────────────────────
# (привязан к сделке, проверяется каждую свечу в custom_exit; пороги
# alert_dca_stuck_hours=6ч, alert_dca_stuck_price_pct=0.3% (порог ВХОДА в
# застой), alert_dca_stuck_hysteresis_mult=2.0 → порог ВЫХОДА из застоя —
# 0.6%, чтобы мелкий шум цены у самой границы 0.3% не считался разморозкой;
# cooldown alert_dca_stuck_cooldown_hours=12ч — периодическое напоминание,
# пока сделка реально остаётся застрявшей, без спама на каждом дребезге)
#
#   🐌 DCA ЗАСТРЯЛ
#
#   Пара: ENA/USDT:USDT
#   Доборов: 3 из 6 (50%)
#   Последний добор: 6ч 40м назад
#   Цена почти не двигалась: ±0.30%
#   PnL позиции: -1.20%
#
#   Капитал заморожен без движения — не забудь про пару
#
# ── Алерт 19: суммарная загрузка капитала — _check_capital_load_alert() ─────
# (портфель целиком, проверяется каждую свечу в bot_loop_start; порог
# alert_capital_load_pct=70%)
#
#   🧮 ВЫСОКАЯ ЗАГРУЗКА КАПИТАЛА
#
#   В позициях суммарно: 720.00 USDT
#   Баланс: 1000.00 USDT
#   Занято капитала: 72.0%
#   Открытых сделок: 6
#
#   ⚠️ Мало свободной подушки для новых входов/доборов
#
# ── Алерт 20: резкий разворот PnL позиции — _check_pnl_reversal_alert() ─────
# (привязан к сделке, проверяется каждую свечу в custom_exit; пороги
# alert_pnl_reversal_drop_pct=2 п.п., alert_pnl_reversal_sample_minutes=15,
# cooldown alert_pnl_reversal_cooldown_minutes=60)
#
#   🔀 РЕЗКИЙ РАЗВОРОТ ПОЗИЦИИ
#
#   Пара: SOL/USDT:USDT
#   Было: +0.80% (12 мин назад)
#   Стало: -1.40%
#   Изменение: -2.20% за 12 минут
#
#   Резкое движение против позиции, обрати внимание
#
# ── Алерт 21: резкий памп/дамп по паре — _check_pair_pump_dump_alert() ──────
# (привязан к сделке с открытой позицией, проверяется каждую свечу в
# custom_exit; пороги alert_pair_pump_dump_pct=6%,
# alert_pair_pump_dump_lookback_minutes=60 (окно в минутах, НЕ зависит от
# таймфрейма стратегии), cooldown alert_pair_pump_dump_cooldown_minutes=60)
#
#   🚀 РЕЗКИЙ ПАМП ПО ПАРЕ
#
#   Пара: DOGE/USDT:USDT
#   Цена 1ч назад: 0.0850
#   Цена сейчас: 0.0918
#   Изменение: +8.00%
#
#   Аномальное движение — есть открытая позиция (SHORT, PnL -3.20%)
#
#   (для дампа — 📉 вместо 🚀 и "ДАМП" вместо "ПАМП" в заголовке)
#
# ── Алерт 22: резкое движение BTC — _check_btc_market_move_alert() ──────────
# (портфель целиком, индикатор "весь рынок штормит"; проверяется каждую
# свечу в bot_loop_start; пороги alert_btc_move_pct=3.5%,
# alert_btc_move_lookback_minutes=120 (окно в минутах, НЕ зависит от
# таймфрейма стратегии), пара alert_btc_pair="BTC/USDT:USDT")
#
#   📰 📉 РЕЗКОЕ ДВИЖЕНИЕ BTC — РЫНОК ШТОРМИТ
#
#   BTC 2ч назад: 68,400 USDT
#   BTC сейчас: 65,900 USDT
#   Изменение: -3.70%
#
#   ⚠️ Весь рынок может двигаться вслед за BTC — проверь открытые позиции (5 шт.)
# =============================================================================


class DSA_General_v2(IStrategy):

    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short: bool = True

    minimal_roi = {}
    stoploss = -0.99

    trailing_stop = False
    process_only_new_candles = True

    use_exit_signal = True
    startup_candle_count: int = 200

    position_adjustment_enable = True
    max_entry_position_adjustment = 7

    # ==== Дефолтные значения (переопределяются из config.json) ====
    safety_order_ratio: float = 1.2
    safety_order_max_count: int = 7
    safety_order_volume_scale: float = 1.2
    price_deviation_initial: float = 0.02
    take_profit: float = 0.02
    trailing_after_tp: bool = True
    trailing_retrace: float = 0.005
    trailing_min_activation: float = 0.02
    force_exit_by_days_enabled: bool = False
    force_exit_after_days: float = 0.0

    # ==== Алерты Telegram (доп. уведомления, не влияют на торговую логику).
    # Значения берутся из блока "НАСТРОЙКИ АЛЕРТОВ" в самом начале файла
    # (сразу после импортов) — правь их там, а не здесь. ====
    alert_deep_drawdown_thresholds: list = ALERT_DEEP_DRAWDOWN_THRESHOLDS
    send_alerts_in_backtest: bool = ALERT_SEND_IN_BACKTEST

    # ==== Алерты 12-14: просадка/рост ОБЩЕГО баланса ====
    alert_balance_drawdown_thresholds: list = ALERT_BALANCE_DRAWDOWN_THRESHOLDS
    alert_balance_growth_thresholds: list = ALERT_BALANCE_GROWTH_THRESHOLDS

    # ==== Алерты (строго live/dry-run, см. _live_alerts_enabled_now) ====
    alert_order_latency_threshold_seconds: float = ALERT_ORDER_LATENCY_THRESHOLD_SECONDS

    # ==== Алерт 15: концентрация риска по одной паре ====
    alert_pair_concentration_threshold_pct: float = ALERT_PAIR_CONCENTRATION_THRESHOLD_PCT

    # ==== Алерт 16: несколько пар одновременно в тяжёлом DCA ====
    alert_multi_dca_pair_pct: float = ALERT_MULTI_DCA_PAIR_PCT
    alert_multi_dca_min_pairs: int = ALERT_MULTI_DCA_MIN_PAIRS
    alert_multi_dca_cooldown_minutes: float = ALERT_MULTI_DCA_COOLDOWN_MINUTES

    # ==== Алерт 17: выход из тяжёлого DCA ====
    alert_heavy_dca_pct: float = ALERT_HEAVY_DCA_PCT
    alert_heavy_dca_recovery_pnl_pct: float = ALERT_HEAVY_DCA_RECOVERY_PNL_PCT
    alert_heavy_dca_rearm_drop_pct: float = ALERT_HEAVY_DCA_REARM_DROP_PCT

    # ==== Алерт 18: DCA "застрял" ====
    alert_dca_stuck_hours: float = ALERT_DCA_STUCK_HOURS
    alert_dca_stuck_price_pct: float = ALERT_DCA_STUCK_PRICE_PCT
    alert_dca_stuck_cooldown_hours: float = ALERT_DCA_STUCK_COOLDOWN_HOURS
    alert_dca_stuck_hysteresis_mult: float = ALERT_DCA_STUCK_HYSTERESIS_MULT

    # ==== Алерт 19: суммарная загрузка капитала ====
    alert_capital_load_pct: float = ALERT_CAPITAL_LOAD_PCT
    alert_capital_load_cooldown_minutes: float = ALERT_CAPITAL_LOAD_COOLDOWN_MINUTES

    # ==== Алерт 20: резкий разворот PnL позиции ====
    alert_pnl_reversal_drop_pct: float = ALERT_PNL_REVERSAL_DROP_PCT
    alert_pnl_reversal_sample_minutes: float = ALERT_PNL_REVERSAL_SAMPLE_MINUTES
    alert_pnl_reversal_cooldown_minutes: float = ALERT_PNL_REVERSAL_COOLDOWN_MINUTES

    # ==== Алерт 21: резкий памп/дамп по паре с открытой позицией ====
    alert_pair_pump_dump_pct: float = ALERT_PAIR_PUMP_DUMP_PCT
    alert_pair_pump_dump_lookback_minutes: float = ALERT_PAIR_PUMP_DUMP_LOOKBACK_MINUTES
    alert_pair_pump_dump_cooldown_minutes: float = ALERT_PAIR_PUMP_DUMP_COOLDOWN_MINUTES

    # ==== Алерт 22: резкое движение BTC ====
    alert_btc_move_pct: float = ALERT_BTC_MOVE_PCT
    alert_btc_move_lookback_minutes: float = ALERT_BTC_MOVE_LOOKBACK_MINUTES
    alert_btc_pair: str = ALERT_BTC_PAIR
    alert_btc_move_cooldown_minutes: float = ALERT_BTC_MOVE_COOLDOWN_MINUTES

    _WELCOME_MARKER = Path(__file__).resolve().parent / ".welcome_pinned_general"

    # Журнал сработавших пер-парных алертов — используется, чтобы рисовать
    # маркеры алертов на графике пары (populate_indicators / plot_config).
    _ALERT_HISTORY_FILE = Path(__file__).resolve().parent / ".alert_history_general.jsonl"

    _EXIT_STATE_KEY = "exit_state"

    def _load_exit_state(self, trade) -> dict:
        data = trade.get_custom_data(self._EXIT_STATE_KEY)
        if not data:
            return {"tp_activated": False, "max_profit": None}
        return data

    def _save_exit_state(self, trade, data: dict) -> None:
        trade.set_custom_data(self._EXIT_STATE_KEY, data)

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # ── Таймфрейм из конфига ──────────────────────────────
        if "timeframe" in config:
            self.timeframe = config["timeframe"]

        # ── Стоп-лосс из конфига ─────────────────────────────
        self.stoploss = float(config.get("stoploss", self.stoploss))

        # ── DCA параметры из конфига ──────────────────────────
        self.safety_order_ratio = float(
            config.get("safety_order_ratio", self.safety_order_ratio)
        )
        self.safety_order_max_count = int(
            config.get("safety_order_max_count", self.safety_order_max_count)
        )
        self.safety_order_volume_scale = float(
            config.get("safety_order_volume_scale", self.safety_order_volume_scale)
        )
        self.price_deviation_initial = float(
            config.get("price_deviation_initial", self.price_deviation_initial)
        )

        # ── Take Profit параметры из конфига ──────────────────
        self.take_profit = float(
            config.get("take_profit", self.take_profit)
        )
        self.trailing_after_tp = bool(
            config.get("trailing_after_tp", self.trailing_after_tp)
        )
        self.trailing_retrace = float(
            config.get("trailing_retrace", self.trailing_retrace)
        )
        self.trailing_min_activation = float(
            config.get("trailing_min_activation", self.trailing_min_activation)
        )
        self.force_exit_by_days_enabled = bool(
            config.get("force_exit_by_days_enabled", self.force_exit_by_days_enabled)
        )
        self.force_exit_after_days = float(
            config.get("force_exit_after_days", self.force_exit_after_days)
        )

        # ── max_entry_position_adjustment синхронизируем с max_count ─
        self.max_entry_position_adjustment = self.safety_order_max_count

        # ── Телеграм настройки ────────────────────────────────
        tg_config = config.get("telegram", {})
        self._tg_token = tg_config.get("token", "").strip()
        self._tg_chat_id = tg_config.get("chat_id", "").strip()
        self._alert_bot_name = "DSA General v2"

        # ── Алерты 12-14: просадка/рост ОБЩЕГО баланса — процесс-уровневое
        # состояние (не переживает рестарт контейнера — после рестарта
        # peak/baseline просто пересчитаются от текущего баланса на первом
        # же тике bot_loop_start). Пороги — см. class-level
        # alert_balance_drawdown_thresholds / alert_balance_growth_thresholds
        # чуть выше; сознательно НЕ читаются из config.json.
        self._alert_balance_peak: Optional[float] = None
        self._alert_balance_baseline: Optional[float] = None
        self._alert_balance_dd_threshold: float = 0.0
        self._alert_balance_growth_threshold: float = 0.0

        # ── Алерт 16: несколько пар одновременно в тяжёлом DCA — процесс-
        # уровневый флаг, чтобы не слать алерт повторно каждую свечу, пока
        # ситуация не изменится (кол-во "тяжёлых" пар упадёт ниже порога).
        # last_alert — доп. cooldown-таймстемп поверх флага: без него метрика,
        # "дрожащая" ровно у порога (флаг то true, то false каждую свечу),
        # могла бы слать алерт заново на каждом таком дребезге.
        self._alert_multi_dca_active: bool = False
        self._alert_multi_dca_last_alert: Optional[datetime] = None

        # ── Алерт 19: суммарная загрузка капитала — процесс-уровневый флаг +
        # cooldown, как и алерт 16 выше.
        self._alert_capital_load_active: bool = False
        self._alert_capital_load_last_alert: Optional[datetime] = None

        # ── Алерт 22: резкое движение BTC — процесс-уровневый флаг + cooldown.
        self._alert_btc_move_active: bool = False
        self._alert_btc_move_last_alert: Optional[datetime] = None

        # ── Алерты: пороги/расписание — жёстко зашиты в коде стратегии как
        # class-level константы (alert_deep_drawdown_thresholds чуть выше и
        # т.д.), НЕ читаются из config.json. Менять — только правкой этого
        # файла.
        # Внутрипроцессный fallback-дедуп (используется только если по каким-то
        # причинам недоступен trade.get_custom_data/set_custom_data — в этой версии
        # freqtrade он доступен и используется как основной механизм, см. ниже).
        self._alert_memory_fallback: Dict[int, dict] = {}

        # ── Индикатор алертов на графике: кэш прочитанной истории по паре.
        # Формат значения: (mtime_файла, {pair: [события, ...]}).
        self._alert_history_cache: Optional[tuple] = None

        # ── OpenAI AI Listener настройки ──────────────────────
        self._openai_key = str(config.get("openai_api_key", "")).strip()
        self._openai_model = str(config.get("openai_model", "gpt-4o")).strip()
        # ВАЖНО: для AI listener нужен ОТДЕЛЬНЫЙ telegram-бот (новый токен от @BotFather),
        # иначе он конфликтует за getUpdates со встроенным TG-ботом Freqtrade.
        # Если поле пустое — listener запущен не будет.
        self._ai_tg_token = str(config.get("openai_telegram_token", "")).strip()
        self._ai_tg_chat_id = str(config.get("openai_telegram_chat_id", self._tg_chat_id)).strip()
        # Время ежедневной сводки в TG (24h, "HH:MM"). Пусто = выкл
        self._ai_daily_summary_time = str(config.get("openai_daily_summary_time", "")).strip()
        # Часовой пояс для расписания, например "Europe/Moscow". Пусто = время контейнера (UTC)
        self._ai_summary_tz = str(config.get("openai_summary_tz", "")).strip()
        # Показывать стоимость каждого ответа в конце сообщения.
        # В config.json: "openai_show_cost": true|false (по умолчанию true)
        self._ai_show_cost = bool(config.get("openai_show_cost", True))
        self._ai_listener: Optional["TelegramAIListener"] = None
        self._ai_init_done = False  # чтобы попытка инициализации не повторялась

        logger.info("✅ DSA_General_v2 initialized")
        logger.info(f"   timeframe             = {self.timeframe}")
        logger.info(f"   market_direction      = dynamic (via /marketdir TG command)")
        logger.info(f"   take_profit           = {self.take_profit:.1%}")
        logger.info(f"   trailing_after_tp     = {self.trailing_after_tp}")
        logger.info(f"   trailing_retrace      = {self.trailing_retrace:.1%}")
        logger.info(f"   trailing_min_activ.   = {self.trailing_min_activation:.1%}")
        logger.info(f"   force_exit_by_days_en = {self.force_exit_by_days_enabled}")
        logger.info(f"   force_exit_after_days = {self.force_exit_after_days}")
        logger.info(f"   safety_order_ratio    = {self.safety_order_ratio}")
        logger.info(f"   safety_order_max_count= {self.safety_order_max_count}")
        logger.info(f"   safety_order_vol_scale= {self.safety_order_volume_scale}")
        logger.info(f"   price_deviation_init  = {self.price_deviation_initial:.1%}")
        logger.info(f"   alert_deep_dd_thresh  = {self.alert_deep_drawdown_thresholds}")
        logger.info(f"   send_alerts_in_bt     = {self.send_alerts_in_backtest}")
        logger.info(f"   alert_balance_dd_thresh     = {self.alert_balance_drawdown_thresholds}")
        logger.info(f"   alert_balance_growth_thresh = {self.alert_balance_growth_thresholds}")

    # ── AI listener: market_direction (нет REST API для /marketdir) ──

    def _ai_get_market_direction(self) -> str:
        direction = str(getattr(self, "market_direction", "none")).lower()
        labels = {
            "long": "только LONG (новые шорты запрещены)",
            "short": "только SHORT (новые лонги запрещены)",
            "even": "чередование long/short (режим even)",
            "none": "оба направления разрешены",
        }
        return f"market_direction={direction} — {labels.get(direction, direction)}"

    def _ai_set_market_direction(self, direction: str) -> str:
        try:
            from freqtrade.enums import MarketDirection
        except Exception:
            return "ERROR: не удалось импортировать MarketDirection."
        mapping = {
            "long": MarketDirection.LONG,
            "short": MarketDirection.SHORT,
            "even": MarketDirection.EVEN,
            "none": MarketDirection.NONE,
        }
        d = str(direction).lower().strip()
        if d not in mapping:
            return f"ERROR: неверное направление '{direction}'. Допустимо: long, short, even, none."
        old = str(getattr(self, "market_direction", MarketDirection.NONE)).lower()
        self.market_direction = mapping[d]
        return f"OK: направление изменено {old} → {d}"

    def _wire_ai_market_dir_handler(self, listener: "TelegramAIListener") -> None:
        setter = getattr(listener, "set_market_dir_handler", None)
        if callable(setter):
            setter(self._ai_get_market_direction, self._ai_set_market_direction)

    # ── Telegram ──────────────────────────────────────────────

    def _tg(self, text: str) -> Optional[int]:
        """
        Отправляет HTML-сообщение в Telegram.
        Возвращает message_id отправленного сообщения или None при ошибке.
        """
        if not self._tg_token or not self._tg_chat_id:
            return None
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{self._tg_token}/sendMessage",
                json={"chat_id": self._tg_chat_id, "text": text, "parse_mode": "HTML"},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    return data.get("result", {}).get("message_id")
        except Exception:
            pass
        return None

    def _pin_message(self, message_id: int) -> None:
        """Закрепляет сообщение по его ID в чате."""
        if not self._tg_token or not self._tg_chat_id:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self._tg_token}/pinChatMessage",
                json={"chat_id": self._tg_chat_id, "message_id": message_id},
                timeout=5,
            )
            logger.info("[TG] Сообщение закреплено")
        except Exception as e:
            logger.warning(f"[TG] Не удалось закрепить сообщение: {e}")

    def _unpin_message(self, message_id: int) -> None:
        """Открепляет конкретное сообщение по его ID (unpinChatMessage)."""
        if not self._tg_token or not self._tg_chat_id:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self._tg_token}/unpinChatMessage",
                json={"chat_id": self._tg_chat_id, "message_id": message_id},
                timeout=5,
            )
            logger.info("[TG] Сообщение откреплено")
        except Exception as e:
            logger.warning(f"[TG] Не удалось открепить сообщение: {e}")

    # Закрепление зарезервировано ТОЛЬКО за приветственным сообщением со
    # списком команд (см. _send_welcome_message/_pin_message) — ни один
    # алерт (включая "ВСЕ DCA ИСПОЛЬЗОВАНЫ" и "СТОП-ЛОСС/ЛИКВИДАЦИЯ")
    # больше не закрепляется, чтобы не перебивать этот постоянный пин.
    # Вместо пина такие алерты усилены визуально (двойная отправка,
    # сирена 🚨 в шапке/подвале сообщения) — см. соответствующие методы.

    # ── Индикатор алертов на графике (журнал + чтение с кэшем) ──────────────

    def _record_pair_alert(self, pair: str, current_time: datetime, code: str) -> None:
        """
        Пишет одну строку JSONL в журнал сработавших пер-парных алертов.
        Файл читается обратно в populate_indicators(), чтобы отрисовать
        маркеры алертов на графике этой пары в FreqUI.
        Ошибки записи не должны влиять на торговую логику.
        """
        try:
            event = {"pair": pair, "date": current_time.isoformat(), "code": code}
            with open(self._ALERT_HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[ALERT-HISTORY] Не удалось записать событие: {e}")

    def _load_alert_history(self, pair: str) -> list:
        """
        Возвращает список событий алертов для данной пары. Кэширует все
        события в памяти процесса и перечитывает файл только если его
        mtime изменился — чтобы не читать файл заново на каждую свечу.
        """
        try:
            mtime = self._ALERT_HISTORY_FILE.stat().st_mtime
        except OSError:
            return []

        cached_mtime = self._alert_history_cache[0] if self._alert_history_cache else None
        if cached_mtime != mtime:
            by_pair: Dict[str, list] = {}
            try:
                with open(self._ALERT_HISTORY_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except Exception:
                            continue
                        by_pair.setdefault(event.get("pair", ""), []).append(event)
            except Exception as e:
                logger.warning(f"[ALERT-HISTORY] Не удалось прочитать журнал: {e}")
                by_pair = {}
            self._alert_history_cache = (mtime, by_pair)

        return self._alert_history_cache[1].get(pair, [])

    def _fmt_usdt(self, value: Optional[float]) -> str:
        if value is None:
            return "—"
        return f"{value:.2f} USDT"

    def _alert_mode_label(self) -> str:
        return "🧪 TEST" if self.config.get("dry_run", True) else "🔴 LIVE"

    def _wallet_total_usdt(self) -> Optional[float]:
        try:
            cur = self.config.get("stake_currency", "USDT")
            return float(self.wallets.get_total(cur))
        except Exception:
            return None

    def _position_stake_usdt(self, trade: Trade) -> float:
        orders = trade.select_filled_orders(trade.entry_side)
        return sum(float(o.cost or 0) for o in orders)

    # =========================================================================
    # АЛЕРТЫ (доп. уведомления в TG) — половина/все DCA, глубокий минус,
    # СТОП-ЛОСС/ЛИКВИДАЦИЯ, дневная/недельная/месячная сводка.
    # Всё аддитивно: НЕ меняет какие сделки открываются/закрываются, только
    # шлёт уведомления вокруг существующей торговой логики.
    # =========================================================================

    def _is_backtest_like_run(self) -> bool:
        return self.config.get("runmode") in (RunMode.BACKTEST, RunMode.HYPEROPT)

    def _alerts_enabled_now(self) -> bool:
        """Общий гейт для НОВЫХ алертов (пп. 1-7 из ТЗ).
        В live/dry-run — всегда включены (если задан TG токен/chat_id).
        В backtest/hyperopt — по флагу send_alerts_in_backtest (дефолт True),
        т.к. hyperopt гоняет backtest сотни/тысячи раз за прогон и может
        заспамить чат, если явно не выключить."""
        if not self._tg_token or not self._tg_chat_id:
            return False
        if self._is_backtest_like_run():
            return self.send_alerts_in_backtest
        return True

    def _live_alerts_enabled_now(self) -> bool:
        """Гейт СТРОГО для алертов 8-11 (бот не отвечает / связь с биржей /
        отклонённый ордер / аномальная задержка исполнения).

        В отличие от _alerts_enabled_now() (алерты 1-7, где backtest/hyperopt
        МОЖЕТ быть включён флагом send_alerts_in_backtest, т.к. это симуляция
        реальных торговых событий) — эти 4 события технически бессмысленны
        или некорректны в backtest/hyperopt: там нет "зависшего" процесса,
        нет реального соединения с биржей, ордера не могут быть "отклонены"
        биржей, а "задержка исполнения" — это просто скорость симуляции.
        Поэтому здесь backtest/hyperopt ВСЕГДА отключены, без каких-либо
        флагов конфига (send_alerts_in_backtest сюда не относится)."""
        if not self._tg_token or not self._tg_chat_id:
            return False
        if self._is_backtest_like_run():
            return False
        return True

    def _trade_custom_get(self, trade: Trade, key: str, default=None):
        """Дедуп по конкретной сделке через trade.get_custom_data — пишется в БД
        (переживает рестарт бота) и работает в backtest тоже (in-memory реализация
        в самом freqtrade). Fallback на self._alert_memory_fallback — НЕ переживает
        рестарт бота — только если API недоступен в установленной версии freqtrade."""
        try:
            return trade.get_custom_data(key, default)
        except Exception:
            return self._alert_memory_fallback.get(trade.id, {}).get(key, default)

    def _trade_custom_set(self, trade: Trade, key: str, value) -> None:
        try:
            trade.set_custom_data(key, value)
        except Exception:
            self._alert_memory_fallback.setdefault(trade.id, {})[key] = value

    def _fmt_duration(self, delta: timedelta) -> str:
        total_min = int(delta.total_seconds() // 60)
        days, rem = divmod(total_min, 1440)
        hours, minutes = divmod(rem, 60)
        parts = []
        if days:
            parts.append(f"{days}д")
        if hours:
            parts.append(f"{hours}ч")
        if not days:
            parts.append(f"{minutes}м")
        return " ".join(parts) if parts else "0м"

    # ── Алерты 1-2: половина / все DCA использованы ────────────────────────

    def _check_dca_usage_alerts(self, trade: Trade, current_time: datetime, order) -> None:
        if not self._alerts_enabled_now():
            return
        max_dca = int(self.safety_order_max_count)
        if max_dca <= 0:
            return
        used = max(0, trade.nr_of_successful_entries - 1)
        half_threshold = math.ceil(max_dca / 2)

        avg_price = float(trade.open_rate or 0)
        # Реальная текущая рыночная цена — цена только что исполненного филла
        # (тот же приём, что и в order_filled() для расчёта profit_pct алерта
        # "ВЫХОД"). Раньше здесь передавался avg_price сам в себя, что всегда
        # давало PnL ≈ 0% вместо фактической просадки/прибыли позиции.
        rate = float(getattr(order, "average", None) or getattr(order, "price", None) or avg_price or 0)
        pnl_pct = float(trade.calc_profit_ratio(rate)) if rate > 0 else 0.0
        stake = self._position_stake_usdt(trade)

        if used >= max_dca:
            if self._trade_custom_get(trade, "alert_all_dca_sent", False):
                return
            self._trade_custom_set(trade, "alert_all_dca_sent", True)
            # Не закрепляется (закреплено может быть только приветственное
            # сообщение со списком команд, см. _send_welcome_message) —
            # вместо пина усилена визуально (шапка/подвал сиреной) и
            # отправляется дважды подряд, как СТОП-ЛОСС/ЛИКВИДАЦИЯ.
            text = (
                f"🚨🆘🚨🆘🚨 <b>ВСЕ DCA ИСПОЛЬЗОВАНЫ</b> 🚨🆘🚨🆘🚨\n\n"
                f"Пара: <b>{trade.pair}</b>\n"
                f"Ордеров DCA: <b>{max_dca} из {max_dca}</b> (100%)\n"
                f"PnL позиции: <b>{pnl_pct:+.2%}</b>\n"
                f"Подушки больше <b>НЕТ</b>\n\n"
                f"❗️❗️❗️ БУДЬТЕ ОЧЕНЬ ВНИМАТЕЛЬНЫ ❗️❗️❗️\n"
                f"Следующий сценарий — только SL/ликвидация\n"
                f"или ручное решение"
            )
            self._tg(text)
            self._tg(text)
            self._record_pair_alert(trade.pair, current_time, "dca_full")
            return

        if used >= half_threshold:
            if self._trade_custom_get(trade, "alert_half_dca_sent", False):
                return
            self._trade_custom_set(trade, "alert_half_dca_sent", True)
            remaining = max_dca - used
            pct_of_max = used / max_dca * 100
            self._tg(
                f"🟡 <b>ПОЛОВИНА DCA ИСПОЛЬЗОВАНА</b>\n\n"
                f"Пара: <b>{trade.pair}</b>\n"
                f"Ордеров DCA: <b>{used} из {max_dca}</b> ({pct_of_max:.0f}%)\n"
                f"Текущий PnL позиции: <b>{pnl_pct:+.2%}</b>\n"
                f"Средняя цена входа: <b>{avg_price:g}</b>\n"
                f"Баланс позиции: <b>{stake:.2f} USDT</b>\n\n"
                f"⚠️ Ещё {remaining} доборов в запасе, следи внимательнее"
            )
            self._record_pair_alert(trade.pair, current_time, "dca_half")

    # ── Алерт 3: глубокий минус по позиции (эскалация по порогам) ──────────

    def _check_deep_drawdown_alert(
        self, trade: Trade, current_time: datetime, current_profit: float
    ) -> None:
        if not self._alerts_enabled_now():
            return
        # По убыванию строгости: -15, -20, -25, -30 (наименее глубокий первый)
        thresholds = sorted(set(self.alert_deep_drawdown_thresholds), reverse=True)
        if not thresholds:
            return
        pnl_pct = float(current_profit) * 100
        last_alerted = float(self._trade_custom_get(trade, "alert_deep_dd_threshold", 0.0) or 0.0)

        for th in thresholds:
            if th < last_alerted and pnl_pct <= th:
                used = max(0, trade.nr_of_successful_entries - 1)
                max_dca = int(self.safety_order_max_count)
                duration = self._fmt_duration(current_time - trade.open_date_utc)
                self._tg(
                    f"🔻 <b>ПОЗИЦИЯ В ГЛУБОКОМ МИНУСЕ</b>\n\n"
                    f"Пара: <b>{trade.pair}</b>\n"
                    f"PnL: <b>{pnl_pct:+.2f}%</b>\n"
                    f"Время в позиции: <b>{duration}</b>\n"
                    f"DCA использовано: <b>{used}/{max_dca}</b>\n\n"
                    f"Порог {th:.0f}% пробит, следим за развитием"
                )
                self._trade_custom_set(trade, "alert_deep_dd_threshold", th)
                last_alerted = th
                self._record_pair_alert(trade.pair, current_time, "deep_dd")

    # ── Алерты 12-14: просадка/рост ОБЩЕГО баланса (портфель целиком,
    # не привязаны к конкретной сделке) ─────────────────────────────────────

    def _check_portfolio_balance_alerts(self, current_time: datetime) -> None:
        if not self._alerts_enabled_now():
            return
        balance = self._wallet_total_usdt()
        if balance is None:
            return
        if self._alert_balance_baseline is None:
            # Первый замер за процесс — фиксируем базу/пик, алертов ещё не шлём
            # (не с чем сравнивать).
            self._alert_balance_baseline = balance
            self._alert_balance_peak = balance
            return
        if balance > self._alert_balance_peak:
            self._alert_balance_peak = balance

        self._check_balance_drawdown_alert(balance)
        self._check_balance_growth_alert(balance)

    def _check_balance_drawdown_alert(self, balance: float) -> None:
        """Алерт 12 (просадка) + Алерт 13 (просадка завершена), эскалация по
        порогам alert_balance_drawdown_thresholds от пикового баланса за всё
        время работы процесса."""
        thresholds = sorted(set(self.alert_balance_drawdown_thresholds), reverse=True)
        if not thresholds or not self._alert_balance_peak:
            return
        dd_pct = (balance - self._alert_balance_peak) / self._alert_balance_peak * 100

        for th in thresholds:
            if th < self._alert_balance_dd_threshold and dd_pct <= th:
                self._tg(
                    f"📉 <b>ПРОСАДКА ОБЩЕГО БАЛАНСА</b>\n\n"
                    f"Текущий баланс: <b>{balance:.2f} USDT</b>\n"
                    f"Пиковый баланс: <b>{self._alert_balance_peak:.2f} USDT</b>\n"
                    f"Просадка: <b>{dd_pct:+.2f}%</b>\n\n"
                    f"Порог {th:.0f}% пробит, следим за развитием"
                )
                self._alert_balance_dd_threshold = th

        # Восстановление — баланс отрос выше (мягче) самого мягкого порога.
        if self._alert_balance_dd_threshold < 0 and dd_pct > thresholds[0]:
            self._tg(
                f"✅ <b>ПРОСАДКА ЗАВЕРШЕНА</b>\n\n"
                f"Текущий баланс: <b>{balance:.2f} USDT</b>\n"
                f"Пиковый баланс: <b>{self._alert_balance_peak:.2f} USDT</b>\n"
                f"Текущее отклонение от пика: <b>{dd_pct:+.2f}%</b>\n\n"
                f"Баланс вернулся выше порога {thresholds[0]:.0f}%"
            )
            self._alert_balance_dd_threshold = 0.0

    def _check_balance_growth_alert(self, balance: float) -> None:
        """Алерт 14: рост общей прибыли, эскалация по порогам
        alert_balance_growth_thresholds от стартового баланса (первый замер
        после запуска бота)."""
        thresholds = sorted(set(self.alert_balance_growth_thresholds))
        if not thresholds or not self._alert_balance_baseline:
            return
        growth_pct = (balance - self._alert_balance_baseline) / self._alert_balance_baseline * 100

        for th in thresholds:
            if th > self._alert_balance_growth_threshold and growth_pct >= th:
                self._tg(
                    f"🚀 <b>РОСТ ОБЩЕЙ ПРИБЫЛИ</b>\n\n"
                    f"Текущий баланс: <b>{balance:.2f} USDT</b>\n"
                    f"Стартовый баланс: <b>{self._alert_balance_baseline:.2f} USDT</b>\n"
                    f"Рост: <b>{growth_pct:+.2f}%</b>\n\n"
                    f"Порог +{th:.0f}% пробит 🎉"
                )
                self._alert_balance_growth_threshold = th

        # Тихий сброс — баланс откатился ниже самого мягкого порога роста.
        if self._alert_balance_growth_threshold > 0 and growth_pct < thresholds[0]:
            self._alert_balance_growth_threshold = 0.0

    # ── Алерт 15: концентрация риска — доля позиции по ОДНОЙ паре от баланса
    # (привязан к конкретной сделке, проверяется каждую свечу в custom_exit) ─

    def _check_pair_concentration_alert(self, trade: Trade, current_time: datetime) -> None:
        if not self._alerts_enabled_now():
            return
        threshold = float(self.alert_pair_concentration_threshold_pct)
        if threshold <= 0:
            return
        balance = self._wallet_total_usdt()
        if not balance:
            return
        stake = self._position_stake_usdt(trade)
        pct = stake / balance * 100

        if pct >= threshold:
            if self._trade_custom_get(trade, "alert_concentration_sent", False):
                return
            self._trade_custom_set(trade, "alert_concentration_sent", True)
            self._tg(
                f"⚠️ <b>КОНЦЕНТРАЦИЯ РИСКА ПО ПАРЕ</b>\n\n"
                f"Пара: <b>{trade.pair}</b>\n"
                f"В позиции: <b>{stake:.2f} USDT</b>\n"
                f"Баланс: <b>{balance:.2f} USDT</b>\n"
                f"Доля от баланса: <b>{pct:.1f}%</b>\n\n"
                f"Порог {threshold:.0f}% превышен — большая часть капитала в одной паре"
            )
            self._record_pair_alert(trade.pair, current_time, "concentration")
        else:
            # Сбрасываем флаг, если доля снова упала ниже порога — чтобы
            # алерт мог сработать заново, если позиция опять "разрастётся"
            # (например, из-за новых доборов DCA в будущем).
            if self._trade_custom_get(trade, "alert_concentration_sent", False):
                self._trade_custom_set(trade, "alert_concentration_sent", False)

    # ── Алерт 16: несколько пар одновременно в тяжёлом DCA (портфель целиком,
    # проверяется каждую свечу в bot_loop_start, как алерты 12-14) ──────────

    def _check_multi_dca_portfolio_alert(self, current_time: datetime) -> None:
        if not self._alerts_enabled_now():
            return
        pct_threshold = float(self.alert_multi_dca_pair_pct)
        min_pairs = int(self.alert_multi_dca_min_pairs)
        if pct_threshold <= 0 or min_pairs <= 0:
            return
        max_dca = int(self.safety_order_max_count)
        if max_dca <= 0:
            return

        try:
            open_trades = Trade.get_open_trades()
        except Exception as e:
            logger.warning(f"[Alerts] multi-DCA portfolio check failed: {e}")
            return

        heavy = []
        for trade in open_trades:
            used = max(0, trade.nr_of_successful_entries - 1)
            pct = used / max_dca * 100
            if pct >= pct_threshold:
                heavy.append((trade.pair, used, pct))

        if len(heavy) >= min_pairs:
            if self._alert_multi_dca_active:
                return
            cooldown_minutes = float(self.alert_multi_dca_cooldown_minutes)
            if self._alert_multi_dca_last_alert is not None:
                elapsed_min = (current_time - self._alert_multi_dca_last_alert).total_seconds() / 60.0
                if elapsed_min < cooldown_minutes:
                    # Метрика "дрожит" вокруг порога (флаг успел сброситься и
                    # взвестись заново) — не спамим раньше cooldown.
                    return
            self._alert_multi_dca_active = True
            self._alert_multi_dca_last_alert = current_time
            heavy.sort(key=lambda x: x[2], reverse=True)
            lines = "\n".join(
                f"• <b>{pair}</b> — {pct:.0f}% ({used}/{max_dca})"
                for pair, used, pct in heavy
            )
            self._tg(
                f"🔴 <b>НЕСКОЛЬКО ПАР В ТЯЖЁЛОМ DCA</b>\n\n"
                f"Пар в зоне риска (≥{pct_threshold:.0f}% доборов): <b>{len(heavy)}</b>\n\n"
                f"{lines}\n\n"
                f"⚠️ Общий риск портфеля повышен, не только по одной паре"
            )
        else:
            self._alert_multi_dca_active = False

    # ── Алерт 17: выход из тяжёлого DCA — позиция восстановилась без новых
    # доборов (привязан к сделке, проверяется каждую свечу в custom_exit) ───

    def _check_heavy_dca_recovery_alert(
        self, trade: Trade, current_time: datetime, current_profit: float
    ) -> None:
        if not self._alerts_enabled_now():
            return
        max_dca = int(self.safety_order_max_count)
        if max_dca <= 0:
            return
        heavy_pct = float(self.alert_heavy_dca_pct)
        recovery_pnl = float(self.alert_heavy_dca_recovery_pnl_pct) / 100.0
        used = max(0, trade.nr_of_successful_entries - 1)
        pct_used = used / max_dca * 100

        was_heavy = bool(self._trade_custom_get(trade, "alert_heavy_dca_seen", False))
        if pct_used >= heavy_pct and not was_heavy:
            self._trade_custom_set(trade, "alert_heavy_dca_seen", True)
            was_heavy = True

        if not was_heavy:
            return

        # Гистерезис: сработать заново можно, только если PnL перед этим
        # реально ушёл обратно ниже recovery_pnl на rearm_drop_pct п.п. — иначе
        # PnL, дребезжащий вокруг recovery_pnl, слал бы алерт туда-обратно.
        rearm_pnl = recovery_pnl - float(self.alert_heavy_dca_rearm_drop_pct) / 100.0
        recovered_sent = bool(self._trade_custom_get(trade, "alert_heavy_dca_recovered_sent", False))

        if recovered_sent:
            if current_profit <= rearm_pnl:
                self._trade_custom_set(trade, "alert_heavy_dca_recovered_sent", False)
            return

        if current_profit >= recovery_pnl:
            self._trade_custom_set(trade, "alert_heavy_dca_recovered_sent", True)
            avg_price = float(trade.open_rate or 0)
            self._tg(
                f"🟢 <b>ВЫХОД ИЗ ТЯЖЁЛОГО DCA</b>\n\n"
                f"Пара: <b>{trade.pair}</b>\n"
                f"Доборов было использовано: <b>{used} из {max_dca}</b> ({pct_used:.0f}%)\n"
                f"Текущий PnL: <b>{current_profit * 100:+.2f}%</b>\n"
                f"Средняя цена входа: <b>{avg_price:g}</b>\n\n"
                f"Усреднение сработало — позиция восстановилась без новых доборов"
            )
            self._record_pair_alert(trade.pair, current_time, "heavy_dca_recovery")

    # ── Алерт 18: DCA "застрял" — давно нет ни новых доборов, ни движения
    # цены (привязан к сделке, проверяется каждую свечу в custom_exit) ──────

    def _check_dca_stuck_alert(
        self, trade: Trade, current_time: datetime, current_rate: float
    ) -> None:
        if not self._alerts_enabled_now():
            return
        max_dca = int(self.safety_order_max_count)
        used = max(0, trade.nr_of_successful_entries - 1)
        if max_dca <= 0 or used <= 0:
            return
        stuck_hours = float(self.alert_dca_stuck_hours)
        price_pct = float(self.alert_dca_stuck_price_pct)
        if stuck_hours <= 0:
            return

        orders = trade.select_filled_orders(trade.entry_side)
        dca_orders = [o for o in orders if (getattr(o, "ft_order_tag", "") or "").upper().startswith("DCA_")]
        if not dca_orders:
            return
        last_order = dca_orders[-1]
        last_time = getattr(last_order, "order_filled_utc", None) or getattr(last_order, "order_date_utc", None)
        last_price = float(getattr(last_order, "average", None) or getattr(last_order, "price", None) or 0)
        if not last_time or last_price <= 0 or current_rate <= 0:
            return

        hours_since = (current_time - last_time).total_seconds() / 3600.0
        price_move_pct = abs(current_rate - last_price) / last_price * 100

        # Гистерезис: "войти" в застой можно только строго внутри price_pct
        # (0.3%), но "выйти" — только уйдя дальше exit_pct (0.6%). Мелкий шум
        # цены между 0.3% и 0.6% не считается ни новым застоем, ни разморозкой
        # — так граница перестаёт дребезжать при каждом микро-колебании.
        was_stuck = bool(self._trade_custom_get(trade, "alert_dca_stuck_active", False))
        exit_pct = price_pct * float(self.alert_dca_stuck_hysteresis_mult)
        if was_stuck:
            is_stuck = price_move_pct < exit_pct
        else:
            is_stuck = hours_since >= stuck_hours and price_move_pct < price_pct

        self._trade_custom_set(trade, "alert_dca_stuck_active", is_stuck)

        if is_stuck:
            cooldown_hours = float(self.alert_dca_stuck_cooldown_hours)
            last_alert_iso = self._trade_custom_get(trade, "alert_dca_stuck_last_alert", None)
            if last_alert_iso:
                last_alert_time = datetime.fromisoformat(last_alert_iso)
                if (current_time - last_alert_time).total_seconds() / 3600.0 < cooldown_hours:
                    # Уже сообщали недавно по этой сделке — просто напоминание
                    # раз в cooldown, пока реально всё ещё застряло.
                    return
            self._trade_custom_set(trade, "alert_dca_stuck_last_alert", current_time.isoformat())
            pnl_pct = float(trade.calc_profit_ratio(current_rate)) * 100
            self._tg(
                f"🐌 <b>DCA ЗАСТРЯЛ</b>\n\n"
                f"Пара: <b>{trade.pair}</b>\n"
                f"Доборов: <b>{used} из {max_dca}</b> ({used / max_dca * 100:.0f}%)\n"
                f"Последний добор: <b>{self._fmt_duration(current_time - last_time)} назад</b>\n"
                f"Цена почти не двигалась: <b>±{price_move_pct:.2f}%</b>\n"
                f"PnL позиции: <b>{pnl_pct:+.2f}%</b>\n\n"
                f"Капитал заморожен без движения — не забудь про пару"
            )
            self._record_pair_alert(trade.pair, current_time, "dca_stuck")

    # ── Алерт 19: суммарная загрузка капитала во ВСЕХ открытых позициях
    # (портфель целиком, проверяется каждую свечу в bot_loop_start) ─────────

    def _check_capital_load_alert(self, current_time: datetime) -> None:
        if not self._alerts_enabled_now():
            return
        threshold = float(self.alert_capital_load_pct)
        if threshold <= 0:
            return
        balance = self._wallet_total_usdt()
        if not balance:
            return

        try:
            open_trades = Trade.get_open_trades()
        except Exception as e:
            logger.warning(f"[Alerts] capital load check failed: {e}")
            return

        total_stake = sum(self._position_stake_usdt(t) for t in open_trades)
        pct = total_stake / balance * 100

        if pct >= threshold:
            if self._alert_capital_load_active:
                return
            cooldown_minutes = float(self.alert_capital_load_cooldown_minutes)
            if self._alert_capital_load_last_alert is not None:
                elapsed_min = (current_time - self._alert_capital_load_last_alert).total_seconds() / 60.0
                if elapsed_min < cooldown_minutes:
                    return
            self._alert_capital_load_active = True
            self._alert_capital_load_last_alert = current_time
            self._tg(
                f"🧮 <b>ВЫСОКАЯ ЗАГРУЗКА КАПИТАЛА</b>\n\n"
                f"В позициях суммарно: <b>{total_stake:.2f} USDT</b>\n"
                f"Баланс: <b>{balance:.2f} USDT</b>\n"
                f"Занято капитала: <b>{pct:.1f}%</b>\n"
                f"Открытых сделок: <b>{len(open_trades)}</b>\n\n"
                f"⚠️ Мало свободной подушки для новых входов/доборов"
            )
        else:
            self._alert_capital_load_active = False

    # ── Алерт 20: резкий разворот PnL позиции — был в плюсе, резко ушёл в
    # минус (привязан к сделке, проверяется каждую свечу в custom_exit) ─────

    def _check_pnl_reversal_alert(
        self, trade: Trade, current_time: datetime, current_profit: float
    ) -> None:
        if not self._alerts_enabled_now():
            return
        reversal_pct = float(self.alert_pnl_reversal_drop_pct)
        sample_minutes = float(self.alert_pnl_reversal_sample_minutes)
        cooldown_minutes = float(self.alert_pnl_reversal_cooldown_minutes)
        if reversal_pct <= 0 or sample_minutes <= 0:
            return
        current_pct = float(current_profit) * 100

        snapshot_pct = self._trade_custom_get(trade, "pnl_reversal_snapshot_pct", None)
        snapshot_time_iso = self._trade_custom_get(trade, "pnl_reversal_snapshot_time", None)

        if snapshot_pct is None or not snapshot_time_iso:
            self._trade_custom_set(trade, "pnl_reversal_snapshot_pct", current_pct)
            self._trade_custom_set(trade, "pnl_reversal_snapshot_time", current_time.isoformat())
            return

        snapshot_time = datetime.fromisoformat(snapshot_time_iso)
        elapsed_min = (current_time - snapshot_time).total_seconds() / 60.0
        if elapsed_min < sample_minutes:
            return

        drop = float(snapshot_pct) - current_pct
        if snapshot_pct > 0 and current_pct < 0 and drop >= reversal_pct:
            last_alert_iso = self._trade_custom_get(trade, "pnl_reversal_alert_time", None)
            can_fire = True
            if last_alert_iso:
                last_alert_time = datetime.fromisoformat(last_alert_iso)
                if (current_time - last_alert_time).total_seconds() / 60.0 < cooldown_minutes:
                    can_fire = False
            if can_fire:
                self._trade_custom_set(trade, "pnl_reversal_alert_time", current_time.isoformat())
                self._tg(
                    f"🔀 <b>РЕЗКИЙ РАЗВОРОТ ПОЗИЦИИ</b>\n\n"
                    f"Пара: <b>{trade.pair}</b>\n"
                    f"Было: <b>{snapshot_pct:+.2f}%</b> ({int(elapsed_min)} мин назад)\n"
                    f"Стало: <b>{current_pct:+.2f}%</b>\n"
                    f"Изменение: <b>{-drop:+.2f}%</b> за {int(elapsed_min)} минут\n\n"
                    f"Резкое движение против позиции, обрати внимание"
                )
                self._record_pair_alert(trade.pair, current_time, "pnl_reversal")

        # Обновляем снапшот для следующего окна сравнения.
        self._trade_custom_set(trade, "pnl_reversal_snapshot_pct", current_pct)
        self._trade_custom_set(trade, "pnl_reversal_snapshot_time", current_time.isoformat())

    # ── Алерт 21: резкий памп/дамп по паре с открытой позицией (привязан к
    # сделке, проверяется каждую свечу в custom_exit) ───────────────────────

    def _check_pair_pump_dump_alert(
        self, trade: Trade, current_time: datetime, current_rate: float
    ) -> None:
        if not self._alerts_enabled_now():
            return
        threshold_pct = float(self.alert_pair_pump_dump_pct)
        lookback_minutes = float(self.alert_pair_pump_dump_lookback_minutes)
        cooldown_minutes = float(self.alert_pair_pump_dump_cooldown_minutes)
        if threshold_pct <= 0 or lookback_minutes <= 0 or not self.dp:
            return

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        except Exception:
            return
        if dataframe is None or dataframe.empty:
            return

        # Окно фиксировано в минутах, а не в свечах — не зависит от таймфрейма
        # стратегии. Берём последнюю свечу на момент времени "current_time
        # минус lookback_minutes".
        target_time = current_time - timedelta(minutes=lookback_minutes)
        past = dataframe.loc[dataframe["date"] <= target_time]
        if past.empty:
            return

        price_then = float(past["close"].iloc[-1])
        if price_then <= 0 or current_rate <= 0:
            return
        change_pct = (current_rate - price_then) / price_then * 100

        if abs(change_pct) < threshold_pct:
            return

        last_alert_iso = self._trade_custom_get(trade, "pump_dump_alert_time", None)
        if last_alert_iso:
            last_alert_time = datetime.fromisoformat(last_alert_iso)
            if (current_time - last_alert_time).total_seconds() / 60.0 < cooldown_minutes:
                return

        self._trade_custom_set(trade, "pump_dump_alert_time", current_time.isoformat())
        pnl_pct = float(trade.calc_profit_ratio(current_rate)) * 100
        side = "SHORT" if trade.is_short else "LONG"
        is_pump = change_pct > 0
        emoji = "🚀" if is_pump else "📉"
        label = "ПАМП" if is_pump else "ДАМП"
        self._tg(
            f"{emoji} <b>РЕЗКИЙ {label} ПО ПАРЕ</b>\n\n"
            f"Пара: <b>{trade.pair}</b>\n"
            f"Цена {self._fmt_duration(timedelta(minutes=lookback_minutes))} назад: <b>{price_then:g}</b>\n"
            f"Цена сейчас: <b>{current_rate:g}</b>\n"
            f"Изменение: <b>{change_pct:+.2f}%</b>\n\n"
            f"Аномальное движение — есть открытая позиция ({side}, PnL {pnl_pct:+.2f}%)"
        )
        self._record_pair_alert(trade.pair, current_time, "pump_dump")

    # ── Алерт 22: резкое движение BTC — индикатор "весь рынок штормит"
    # (портфель целиком, проверяется каждую свечу в bot_loop_start) ─────────

    def _check_btc_market_move_alert(self, current_time: datetime) -> None:
        if not self._alerts_enabled_now():
            return
        threshold_pct = float(self.alert_btc_move_pct)
        lookback_minutes = float(self.alert_btc_move_lookback_minutes)
        if threshold_pct <= 0 or lookback_minutes <= 0 or not self.dp:
            return

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(self.alert_btc_pair, self.timeframe)
        except Exception:
            return
        if dataframe is None or dataframe.empty:
            return

        # Окно фиксировано в минутах, а не в свечах — не зависит от таймфрейма
        # стратегии.
        target_time = current_time - timedelta(minutes=lookback_minutes)
        past = dataframe.loc[dataframe["date"] <= target_time]
        if past.empty:
            return

        price_then = float(past["close"].iloc[-1])
        price_now = float(dataframe["close"].iloc[-1])
        if price_then <= 0:
            return
        change_pct = (price_now - price_then) / price_then * 100

        if abs(change_pct) >= threshold_pct:
            if self._alert_btc_move_active:
                return
            cooldown_minutes = float(self.alert_btc_move_cooldown_minutes)
            if self._alert_btc_move_last_alert is not None:
                elapsed_min = (current_time - self._alert_btc_move_last_alert).total_seconds() / 60.0
                if elapsed_min < cooldown_minutes:
                    return
            self._alert_btc_move_active = True
            self._alert_btc_move_last_alert = current_time
            try:
                open_count = len(Trade.get_open_trades())
            except Exception:
                open_count = 0
            emoji = "📈" if change_pct > 0 else "📉"
            self._tg(
                f"📰 {emoji} <b>РЕЗКОЕ ДВИЖЕНИЕ BTC — РЫНОК ШТОРМИТ</b>\n\n"
                f"BTC {self._fmt_duration(timedelta(minutes=lookback_minutes))} назад: <b>{price_then:,.0f} USDT</b>\n"
                f"BTC сейчас: <b>{price_now:,.0f} USDT</b>\n"
                f"Изменение: <b>{change_pct:+.2f}%</b>\n\n"
                f"⚠️ Весь рынок может двигаться вслед за BTC — проверь открытые позиции ({open_count} шт.)"
            )
        else:
            self._alert_btc_move_active = False

    # ── Алерт 4: СТОП-ЛОСС / ЛИКВИДАЦИЯ (критично, дублируется + пин) ───────

    _SL_EXIT_REASONS = {
        ExitType.STOP_LOSS.value,
        ExitType.STOPLOSS_ON_EXCHANGE.value,
        ExitType.TRAILING_STOP_LOSS.value,
    }
    _LIQUIDATION_EXIT_REASONS = {ExitType.LIQUIDATION.value}

    def _check_stoploss_liquidation_alert(
        self,
        trade: Trade,
        current_time: datetime,
        reason: str,
        profit_pct: float,
        profit_abs: float,
        balance: Optional[float],
    ) -> None:
        if not self._alerts_enabled_now():
            return
        if reason in self._LIQUIDATION_EXIT_REASONS:
            title = "ЛИКВИДАЦИЯ"
        elif reason in self._SL_EXIT_REASONS:
            title = "СТОП-ЛОСС СРАБОТАЛ"
        else:
            return

        text = (
            f"🚨🆘🔴🆘🔴🆘🚨 <b>{title}</b> 🚨🆘🔴🆘🔴🆘🚨\n\n"
            f"ПАРА: <b>{trade.pair}</b>\n"
            f"УБЫТОК: <b>{profit_abs:+.2f} USDT</b> ({profit_pct:+.2%})\n"
            f"БАЛАНС ПОСЛЕ: <b>{self._fmt_usdt(balance)}</b>\n\n"
            f"❗️❗️❗️ <b>КРИТИЧЕСКОЕ СОБЫТИЕ</b> ❗️❗️❗️\n"
            f"Требуется анализ причины"
        )
        # Шлём дважды подряд — максимально заметно, отдельно от обычного 🔴 ВЫХОД.
        # Не закрепляется (закреплено может быть только приветственное сообщение
        # со списком команд, см. _send_welcome_message) — усилена визуально
        # (сирена в шапке/подвале) вместо пина.
        self._tg(text)
        self._tg(text)
        self._record_pair_alert(trade.pair, current_time, "sl_liq")

    # =========================================================================
    # АЛЕРТЫ (строго live/dry-run, см. _live_alerts_enabled_now()).
    # Ноль сообщений по этим пунктам в backtest/hyperopt.
    #   • Ордер отклонён/отменён       → _check_rejected_cancelled_orders_alert()
    #   • Аномальная задержка ордера   → _check_order_latency_alert()
    # =========================================================================

    # ── Ордер отклонён/отменён биржей ───────────────────────────────────────
    #
    # Order.status приходит от ccxt/биржи и приводится freqtrade к одному из
    # constants.CANCELED_EXCHANGE_STATES = ("cancelled", "canceled", "expired",
    # "rejected") либо "closed"/"open" (freqtrade/constants.py). Проверялись
    # также check_entry_timeout/check_exit_timeout (freqtrade/strategy/
    # interface.py) — они реально существуют, НО вызываются из ft_check_timed_
    # out ТОЛЬКО если конфиг unfilledtimeout ещё не признал ордер просроченным
    # (freqtrade/strategy/interface.py: ft_check_timed_out) — а в этом проекте
    # unfilledtimeout.entry/exit заданы (10 минут), т.е. штатный тайм-аут почти
    # всегда сработает РАНЬШЕ и хук стратегии просто не будет вызван. Поэтому
    # они ненадёжны как основной триггер и сознательно не используются —
    # вместо этого сканируем итоговые статусы ордеров (trade.orders) на каждой
    # итерации bot_loop_start, что ловит ЛЮБую причину (тайм-аут, insufficient
    # funds, ручная отмена, cancel при replace и т.д.), а не только тайм-аут.
    _REJECTED_ORDER_STATUSES = {"canceled", "cancelled", "expired", "rejected"}

    def _check_rejected_cancelled_orders_alert(self, current_time: datetime) -> None:
        if not self._live_alerts_enabled_now():
            return
        try:
            open_trades = Trade.get_open_trades()
        except Exception as e:
            logger.warning(f"[Alerts] не удалось получить открытые сделки: {e}")
            return

        for trade in open_trades:
            try:
                orders = list(trade.orders or [])
            except Exception:
                continue

            for order in orders:
                status = str(getattr(order, "status", "") or "").lower()
                if status not in self._REJECTED_ORDER_STATUSES:
                    continue

                order_key = str(getattr(order, "order_id", None) or id(order))
                alerted = self._trade_custom_get(trade, "alerted_rejected_orders", []) or []
                if order_key in alerted:
                    continue

                side = "entry" if order.ft_order_side == trade.entry_side else "exit"
                reason = (getattr(order, "ft_cancel_reason", "") or "").strip() or status
                amount = getattr(order, "safe_amount", None)
                if amount is None:
                    amount = getattr(order, "amount", 0) or 0

                self._tg(
                    f"⛔ <b>ОРДЕР НЕ ИСПОЛНИЛСЯ</b>\n\n"
                    f"Пара: <b>{trade.pair}</b>\n"
                    f"Тип: <b>{order.order_type}</b> ({side})\n"
                    f"Причина: <b>{reason}</b>\n"
                    f"Объём: <b>{amount:g}</b>\n\n"
                    f"Сделка не была открыта/изменена как ожидалось"
                )

                alerted.append(order_key)
                self._trade_custom_set(trade, "alerted_rejected_orders", alerted)

    # ── Аномальная задержка исполнения ордера ───────────────────────────────
    #
    # Order.order_date_utc — момент создания ордера, Order.order_filled_utc —
    # момент фактического наполнения (оба свойства в freqtrade/persistence/
    # trade_model.py, класс Order). Вызывается из order_filled() — там order
    # уже пришёл заполненным, оба поля точно доступны.
    def _check_order_latency_alert(self, trade: Trade, order, pair: str) -> None:
        if not self._live_alerts_enabled_now():
            return
        # ВАЖНО: считаем "задержкой исполнения" (признак проблем с биржей/
        # сетью) только для MARKET-ордеров — они должны исполняться почти
        # мгновенно, если с биржей всё ок. Для LIMIT-ордеров время created→
        # filled — это НЕ латентность, а просто "сколько шла цена до лимитной
        # отметки" (совершенно нормально ждать минуты/часы), поэтому такие
        # ордера сюда сознательно не пускаем — иначе это гарантированный
        # ложный алерт на каждом обычном лимитном входе/выходе.
        order_type = str(getattr(order, "order_type", "") or "").lower()
        if order_type != "market":
            return
        try:
            created = order.order_date_utc
            filled = order.order_filled_utc
        except Exception:
            return
        if not created or not filled:
            return

        latency_seconds = (filled - created).total_seconds()
        if latency_seconds < self.alert_order_latency_threshold_seconds:
            return

        order_key = str(getattr(order, "order_id", None) or id(order))
        alerted = self._trade_custom_get(trade, "alerted_latency_orders", []) or []
        if order_key in alerted:
            return

        side = "entry" if order.ft_order_side == trade.entry_side else "exit"
        self._tg(
            f"🐢 <b>АНОМАЛЬНАЯ ЗАДЕРЖКА ИСПОЛНЕНИЯ</b>\n\n"
            f"Пара: <b>{pair}</b>\n"
            f"Тип ордера: <b>{order.order_type}</b> ({side})\n"
            f"Задержка: <b>{latency_seconds:.0f} сек</b> "
            f"(порог {self.alert_order_latency_threshold_seconds:.0f} сек)\n\n"
            f"⚠️ Возможны проблемы с биржей или сетью"
        )

        alerted.append(order_key)
        self._trade_custom_set(trade, "alerted_latency_orders", alerted)

    def order_filled(
        self,
        pair: str,
        trade: Trade,
        order,
        current_time: datetime,
        **kwargs: Any,
    ) -> None:
        if not self._tg_token or not self._tg_chat_id:
            return

        mode = self._alert_mode_label()
        balance = self._wallet_total_usdt()
        fill_usdt = float(order.cost or 0)
        tag = (getattr(order, "ft_order_tag", None) or "").strip()
        is_entry = order.ft_order_side == trade.entry_side
        side = "SHORT" if trade.is_short else "LONG"

        # ── Аномальная задержка исполнения (строго live/dry-run) ────────────
        try:
            self._check_order_latency_alert(trade, order, pair)
        except Exception as e:
            logger.warning(f"[Alerts] order latency check failed: {e}")

        if is_entry:
            position_total = self._position_stake_usdt(trade)
            if tag.upper().startswith("DCA_"):
                dca_num = tag.split("_", 1)[1] if "_" in tag else "?"
                self._tg(
                    f"🔵 <b>DCA #{dca_num}</b>\n"
                    f"Бот: <b>{self._alert_bot_name}</b> · {mode}\n"
                    f"Пара: <b>{pair}</b> · {side}\n"
                    f"Доливка: <b>{self._fmt_usdt(fill_usdt)}</b>\n"
                    f"Всего в позиции: <b>{self._fmt_usdt(position_total)}</b>\n"
                    f"Баланс: <b>{self._fmt_usdt(balance)}</b>"
                )
            else:
                self._tg(
                    f"🟢 <b>ВХОД</b>\n"
                    f"Бот: <b>{self._alert_bot_name}</b> · {mode}\n"
                    f"Пара: <b>{pair}</b> · {side}\n"
                    f"Тег: <code>{tag or 'entry'}</code>\n"
                    f"Вложено: <b>{self._fmt_usdt(fill_usdt)}</b>\n"
                    f"В позиции: <b>{self._fmt_usdt(position_total)}</b>\n"
                    f"Баланс: <b>{self._fmt_usdt(balance)}</b>"
                )

            # ── Алерты 1-2: половина/все DCA ──
            try:
                if self._alerts_enabled_now():
                    self._check_dca_usage_alerts(trade, current_time, order)
            except Exception as e:
                logger.warning(f"[Alerts] DCA usage check failed: {e}")
            return

        reason = trade.exit_reason or tag or "exit"
        rate = float(order.average or order.price or trade.close_rate or 0)
        if trade.close_profit is not None:
            profit_pct = float(trade.close_profit)
        elif rate > 0:
            profit_pct = float(trade.calc_profit_ratio(rate))
        else:
            profit_pct = 0.0

        if trade.close_profit_abs is not None:
            profit_abs = float(trade.close_profit_abs)
        elif rate > 0:
            profit_abs = float(trade.calc_profit(rate))
        else:
            profit_abs = 0.0

        stake_before = self._position_stake_usdt(trade)
        self._tg(
            f"🔴 <b>ВЫХОД</b>\n"
            f"Бот: <b>{self._alert_bot_name}</b> · {mode}\n"
            f"Пара: <b>{pair}</b> · {side}\n"
            f"Причина: <code>{reason}</code>\n"
            f"PnL: <b>{profit_pct:+.2%}</b> ({profit_abs:+.2f} USDT)\n"
            f"Было в позиции: <b>{self._fmt_usdt(stake_before)}</b>\n"
            f"Баланс: <b>{self._fmt_usdt(balance)}</b>"
        )

        # ── Алерт 4: СТОП-ЛОСС/ЛИКВИДАЦИЯ — В ДОПОЛНЕНИЕ к 🔴 ВЫХОД выше,
        # не вместо него.
        try:
            self._check_stoploss_liquidation_alert(
                trade, current_time, reason, profit_pct, profit_abs, balance
            )
        except Exception as e:
            logger.warning(f"[Alerts] SL/liquidation check failed: {e}")

    def _send_welcome_message(self) -> None:
        """Отправляет приветственное сообщение с инструкцией по управлению ботом (один раз навсегда)."""
        if self._WELCOME_MARKER.exists():
            return
        if not self._tg_token or not self._tg_chat_id:
            logger.warning("[TG] Токен или Chat ID не заданы, приветствие не отправлено.")
            return

        msg = (
            "🤖 <b>Freqtrade Bot Control</b>\n\n"
            "Бот успешно запущен и работает по стратегии DSA General v2.\n"
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
            "│ 🏷 <code>/entries [pair]</code> — эффективность входов\n"
            "│ 🏷 <code>/exits [pair]</code> — эффективность выходов\n"
            "│ 🔀 <code>/mix_tags [pair]</code> — комбинированные теги\n"
            "│ 📊 <code>/stats</code> — общая статистика\n"
            "│ 💰 <code>/balance [total]</code> — баланс по валютам\n"
            "│ 📜 <code>/logs [limit]</code> — последние логи\n"
            "│ 🔢 <code>/count</code> — количество активных сделок\n"
            "│ ❤️ <code>/health</code> — время последнего обновления\n"
            "│ 🧭 <code>/marketdir [long|short|even|none]</code> — направление рынка\n"
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

        message_id = self._tg(msg)
        if message_id:
            self._pin_message(message_id)
            try:
                self._WELCOME_MARKER.write_text(str(message_id))
            except OSError as e:
                logger.warning(f"[TG] Не удалось создать маркер-файл: {e}")

    def informative_pairs(self):
        # Алерт 22 (резкое движение BTC) читает датафрейм alert_btc_pair через
        # self.dp.get_analyzed_dataframe — без этого объявления пара не
        # подгружается и не анализируется, если бот сам её не торгует.
        return [(self.alert_btc_pair, self.timeframe)]

    # =========================================================================
    # INDICATORS
    # =========================================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['sma_5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['sma_20'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma_100'] = ta.SMA(dataframe, timeperiod=100)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)

        dataframe['st_up'], dataframe['st_down'] = supertrend(dataframe, period=10, multiplier=3.0)

        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_upper'] = bollinger['upperband']

        for ind, func in [('adx', ta.ADX), ('atr', ta.ATR), ('mfi', ta.MFI)]:
            try:
                dataframe[ind] = func(dataframe, timeperiod=14)
            except Exception:
                dataframe[ind] = np.nan

        # ── Паттерны свечей (TA-Lib CDL*) — только "тяжеловесные" ───────────
        # Оставлены исключительно многосвечные подтверждённые паттерны —
        # статистически самые надёжные, с наибольшим historical win-rate
        # разворота (Engulfing, Morning/Evening Star, 3 Soldiers/Crows).
        # Однослойные паттерны (Hammer, ShootingStar, HangingMan, Harami)
        # убраны — они срабатывают на одной свече без подтверждения и дают
        # намного больше ложных сигналов.
        # Каждая колонка: 100 = бычий паттерн, -100 = медвежий, 0 = нет сигнала.
        for col, func in [
            ('cdl_engulfing', ta.CDLENGULFING),            # поглощение
            ('cdl_morningstar', ta.CDLMORNINGSTAR),        # утренняя звезда
            ('cdl_eveningstar', ta.CDLEVENINGSTAR),        # вечерняя звезда
            ('cdl_3whitesoldiers', ta.CDL3WHITESOLDIERS),  # три белых солдата
            ('cdl_3blackcrows', ta.CDL3BLACKCROWS),        # три чёрные вороны
        ]:
            try:
                dataframe[col] = func(dataframe)
            except Exception:
                dataframe[col] = 0

        # ── Объединённый сигнал паттернов для графика ───────────────────────
        # Несколько тонких отдельных полосок в одном subplot нечитаемы —
        # вместо этого суммируем сработавшие паттерны в 2 понятные колонки:
        # чем больше паттернов совпало на одной свече, тем выше/ниже бар.
        cdl_cols = [
            'cdl_engulfing', 'cdl_morningstar', 'cdl_eveningstar',
            'cdl_3whitesoldiers', 'cdl_3blackcrows',
        ]
        cdl_sum = dataframe[cdl_cols].sum(axis=1)
        dataframe['cdl_bullish'] = cdl_sum.where(cdl_sum > 0, 0)
        dataframe['cdl_bearish'] = cdl_sum.where(cdl_sum < 0, 0)

        try:
            close = dataframe['close']
            dataframe['kst'] = (
                close.pct_change(10).rolling(10).mean() * 100
                + close.pct_change(15).rolling(10).mean() * 200
                + close.pct_change(20).rolling(10).mean() * 300
                + close.pct_change(30).rolling(15).mean() * 400
            )
            dataframe['kst_sig'] = dataframe['kst'].rolling(9).mean()
        except Exception:
            dataframe['kst'] = np.nan
            dataframe['kst_sig'] = np.nan

        self._populate_alert_markers(dataframe, metadata['pair'])

        return dataframe

    # Код алерта → колонка-маркер на графике (см. _record_pair_alert).
    # Значения — имена колонок дataframe. Они же ключи в plot_config и, как
    # следствие, подписи в легенде/тултипе графика FreqUI — поэтому на русском.
    _ALERT_MARKER_COLUMNS = {
        "dca_half": "DCA_половина",
        "dca_full": "DCA_полный",
        "deep_dd": "Просадка_глубокая",
        "sl_liq": "Стоп_ликвидация",
        "tp_activated": "TP_активирован",
        "concentration": "Концентрация",
        "heavy_dca_recovery": "DCA_восстановление",
        "dca_stuck": "DCA_застрял",
        "pnl_reversal": "PnL_разворот",
        "pump_dump": "Памп_дамп",
    }

    def _populate_alert_markers(self, dataframe: DataFrame, pair: str) -> None:
        """
        Добавляет по одной колонке на каждый тип пер-парного алерта: NaN везде,
        кроме свечи, ближайшей ко времени срабатывания алерта из журнала
        (_ALERT_HISTORY_FILE), где ставится close этой свечи — так маркер
        рисуется точкой прямо на линии цены в FreqUI (см. plot_config).
        """
        for column in self._ALERT_MARKER_COLUMNS.values():
            dataframe[column] = np.nan

        events = self._load_alert_history(pair)
        if not events or dataframe.empty:
            return

        dates = dataframe['date']
        for event in events:
            column = self._ALERT_MARKER_COLUMNS.get(event.get("code", ""))
            if column is None:
                continue
            try:
                event_time = pd.Timestamp(event["date"])
            except Exception:
                continue
            idx = dates.searchsorted(event_time, side="right") - 1
            if 0 <= idx < len(dataframe):
                dataframe.iloc[idx, dataframe.columns.get_loc(column)] = dataframe['close'].iloc[idx]

    # =========================================================================
    # PLOT CONFIG
    # =========================================================================

    @property
    def plot_config(self):
        """
        Возвращаем конфиг динамически.
        FreqUI ожидает базовые plot-поля; нестандартные ключи могут ломать синхронизацию
        нижних панелей с основной временной осью.
        """
        return {
            "main_plot": {
                "sma_5": {"color": "white"},
                "sma_20": {"color": "yellow"},
                "sma_50": {"color": "green"},
                "sma_100": {"color": "red"},
                "sma_200": {"color": "purple"},
                "st_up": {"color": "lime"},
                "st_down": {"color": "red"},
                "bb_upper": {"color": "rgba(0,150,255,0.7)"},
                "bb_middle": {"color": "rgba(255,165,0,0.8)"},
                "bb_lower": {"color": "rgba(0,150,255,0.7)"},
                # ── Маркеры сработавших алертов по этой паре (см. _record_pair_alert).
                # FreqUI (веб) рендерит графики через ECharts, а не Plotly — единственное
                # поле, которое реально влияет на размер scatter-точки, это
                # scatterSymbolSize (дефолт движка — 3px). Разных форм маркеров (звезда,
                # крестик и т.п.) FreqUI не поддерживает — различаем алерты только цветом
                # и размером. Палитра — Tableau 10 (приглушённые, "дашбордные" тона вместо
                # кислотных неоновых), цвет каждого алерта подобран по смыслу события.
                "Стоп_ликвидация": {"color": "#FF0000", "type": "scatter", "scatterSymbolSize": 20},     # красный — критично
                "Просадка_глубокая": {"color": "#FF6600", "type": "scatter", "scatterSymbolSize": 18},   # оранжевый — опасность
                "DCA_полный": {"color": "#8B0000", "type": "scatter", "scatterSymbolSize": 20},          # сливовый — весь DCA использован
                "DCA_половина": {"color": "#FFCC00", "type": "scatter", "scatterSymbolSize": 18},        # золотой — предупреждение
                "TP_активирован": {"color": "#00FE00", "type": "scatter", "scatterSymbolSize": 18},      # зелёный — профит
                "DCA_восстановление": {"color": "#038903", "type": "scatter", "scatterSymbolSize": 18},  # бирюзовый — восстановление
                "Концентрация": {"color": "#0000FF", "type": "scatter", "scatterSymbolSize": 18},        # синий — инфо
                "PnL_разворот": {"color": "#00FFFF", "type": "scatter", "scatterSymbolSize": 18},        # розовый — разворот тренда
                "Памп_дамп": {"color": "#FF00FF", "type": "scatter", "scatterSymbolSize": 18},           # коричневый — аномалия рынка
                "DCA_застрял": {"color": "#808080", "type": "scatter", "scatterSymbolSize": 18},         # серый — стагнация
            },
            "subplots": {
                "Свечные паттерны": {
                    # Бар вверх = бычий паттерн, вниз = медвежий. Чем выше/ниже
                    # бар — тем больше разных паттернов совпало на этой свече
                    # (сумма по всем cdl_* колонкам из populate_indicators).
                    "cdl_bullish": {"color": "#00FE00", "type": "bar"},   # зелёный — бычьи паттерны
                    "cdl_bearish": {"color": "#FF0000", "type": "bar"},   # красный — медвежьи паттерны
                },
                "RSI": {"rsi": {"color": "purple"}},
                "MACD": {
                    "macd": {"color": "blue"},
                    "macdsignal": {"color": "orange"},
                    "macdhist": {"color": "green"},
                },
                "ADX": {"adx": {"color": "magenta"}},
                "ATR": {"atr": {"color": "brown"}},
                "MFI": {"mfi": {"color": "cyan"}},
                "KST": {"kst": {"color": "blue"}, "kst_sig": {"color": "red"}},
            },
        }

    # =========================================================================
    # ENTRY — лонг и шорт по RSI
    # =========================================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, "enter_long"] = 0
        dataframe.loc[:, "enter_short"] = 0

        # Направление берём динамически из атрибута стратегии.
        # Обновляется через Telegram-команду /marketdir long|short|even|none
        # при каждом обновлении свечи (process_only_new_candles = True).
        direction = str(getattr(self, "market_direction", None) or "none").lower()

        allow_long = direction in ("long", "none")
        allow_short = direction in ("short", "none")

        if allow_long:
            dataframe.loc[
                (dataframe['rsi'] < 35) & (dataframe['volume'] > 0),
                ["enter_long", "enter_tag"],
            ] = (1, "direct_long_entry")

        if allow_short:
            dataframe.loc[
                (dataframe['rsi'] > 65) & (dataframe['volume'] > 0),
                ["enter_short", "enter_tag"],
            ] = (1, "direct_short_entry")

        return dataframe

    # =========================================================================
    # EXIT — через custom_exit
    # =========================================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0
        return dataframe

    # =========================================================================
    # DCA — лонг и шорт
    # =========================================================================

    # ИДЕЯ (не реализовано, по решению пользователя от 2026-07-03):
    # Уменьшение позиции при затягивании: если сделка висит долго и в минусе —
    # частично сокращать объём (partial exit), не дожидаясь полного разворота.

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        **kwargs: Any,
    ) -> Optional[Tuple[float, str]]:

        # Количество выполненных доливок (первый вход не считается)
        dca_count = trade.nr_of_successful_entries - 1

        if dca_count >= self.safety_order_max_count:
            return None

        if trade.open_orders:
            return None

        # Триггер растёт с каждой доливкой
        # LONG:  -2%, -4%, -6% ...
        # SHORT: +2%, +4%, +6% ...
        deviation = self.price_deviation_initial * (dca_count + 1)

        if not trade.is_short:
            trigger_price = trade.open_rate * (1 - deviation)
            if current_rate > trigger_price:
                return None
        else:
            trigger_price = trade.open_rate * (1 + deviation)
            if current_rate < trigger_price:
                return None

        filled_orders = trade.select_filled_orders(trade.entry_side)
        if not filled_orders:
            return None

        first_order_cost = filled_orders[0].cost

        # Объём: ratio*1, ratio*scale, ratio*scale^2 ...
        so_amount = first_order_cost * self.safety_order_ratio * (
            self.safety_order_volume_scale ** dca_count
        )

        if min_stake is not None and so_amount < min_stake:
            so_amount = min_stake
        so_amount = min(so_amount, max_stake)

        tag = f"DCA_{dca_count + 1}"
        logger.info(
            f"🔄 {tag} {trade.pair} ({'SHORT' if trade.is_short else 'LONG'}): "
            f"amount={so_amount:.2f} USDT, "
            f"deviation={deviation:.1%}, trigger={trigger_price:.6f}"
        )
        return so_amount, tag

    # =========================================================================
    # TAKE PROFIT + TRAILING
    # =========================================================================

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> Optional[str]:
        # ── Алерт 3: глубокий минус по позиции ──────────────────────────
        # custom_exit дёргается КАЖДУЮ свечу на КАЖДУЮ открытую сделку и в
        # live, и в backtest (freqtrade/strategy/interface.py: should_exit),
        # current_time/current_profit тут — историческое время/PnL симуляции
        # в backtest, а не datetime.now(). Именно поэтому все проверки,
        # которым нужно "триггериться каждую свечу по открытой позиции",
        # размещены здесь, а не в bot_loop_start.
        try:
            self._check_deep_drawdown_alert(trade, current_time, current_profit)
        except Exception as e:
            logger.warning(f"[Alerts] deep drawdown check failed: {e}")

        # ── Алерт 15: концентрация риска по одной паре ──────────────────
        try:
            self._check_pair_concentration_alert(trade, current_time)
        except Exception as e:
            logger.warning(f"[Alerts] pair concentration check failed: {e}")

        # ── Алерт 17: выход из тяжёлого DCA ──────────────────────────────
        try:
            self._check_heavy_dca_recovery_alert(trade, current_time, current_profit)
        except Exception as e:
            logger.warning(f"[Alerts] heavy DCA recovery check failed: {e}")

        # ── Алерт 18: DCA "застрял" ───────────────────────────────────────
        try:
            self._check_dca_stuck_alert(trade, current_time, current_rate)
        except Exception as e:
            logger.warning(f"[Alerts] DCA stuck check failed: {e}")

        # ── Алерт 20: резкий разворот PnL позиции ────────────────────────
        try:
            self._check_pnl_reversal_alert(trade, current_time, current_profit)
        except Exception as e:
            logger.warning(f"[Alerts] PnL reversal check failed: {e}")

        # ── Алерт 21: резкий памп/дамп по паре ───────────────────────────
        try:
            self._check_pair_pump_dump_alert(trade, current_time, current_rate)
        except Exception as e:
            logger.warning(f"[Alerts] pair pump/dump check failed: {e}")

        if self.force_exit_by_days_enabled and self.force_exit_after_days > 0:
            age_days = (current_time - trade.open_date_utc).total_seconds() / 86400.0
            if age_days >= self.force_exit_after_days and current_profit > 0:
                logger.info(
                    f"⏰ Force exit by age {pair}: age={age_days:.2f}d "
                    f"limit={self.force_exit_after_days:.2f}d profit={current_profit:.2%}"
                )
                self._save_exit_state(trade, {"tp_activated": False, "max_profit": None})
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
                logger.info(f"🎯 Trailing TP активирован {pair}: profit={current_profit:.2%}")
                if self._alerts_enabled_now():
                    try:
                        self._tg(
                            f"🎯 <b>ТЕЙК-ПРОФИТ АКТИВИРОВАН</b>\n\n"
                            f"Пара: <b>{pair}</b>\n"
                            f"Текущий профит: <b>{current_profit * 100:+.2f}%</b>"
                        )
                        self._record_pair_alert(pair, current_time, "tp_activated")
                    except Exception as e:
                        logger.warning(f"[Alerts] TP activated notify failed: {e}")
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
                f"max={data['max_profit']:.2%}, retrace={retrace:.2%}"
            )
            return "trailing_take_profit"

        return None

    # =========================================================================
    # BOT LOOP START — отправка приветствия при запуске
    # =========================================================================

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Отправляет приветственное сообщение при первом запуске бота и
        запускает Telegram AI listener (один раз).
        """
        self._send_welcome_message()

        # ── Технические алерты (строго live/dry-run, см. _live_alerts_enabled_now) ──
        # Эти НЕ должны выполняться в backtest/hyperopt вообще (гейт внутри
        # каждого метода), поэтому вызываем их безусловно здесь — сами методы
        # решат, слать что-то или нет.
        try:
            self._check_rejected_cancelled_orders_alert(current_time)
        except Exception as e:
            logger.warning(f"[Alerts] rejected/cancelled orders check failed: {e}")

        # ── Алерты 12-14: просадка/рост ОБЩЕГО баланса (см. _alerts_enabled_now,
        # работают в live/dry-run всегда, в backtest/hyperopt — по флагу
        # send_alerts_in_backtest, как и алерты 1-3) ──
        try:
            self._check_portfolio_balance_alerts(current_time)
        except Exception as e:
            logger.warning(f"[Alerts] portfolio balance check failed: {e}")

        # ── Алерт 16: несколько пар одновременно в тяжёлом DCA ──
        try:
            self._check_multi_dca_portfolio_alert(current_time)
        except Exception as e:
            logger.warning(f"[Alerts] multi-DCA portfolio check failed: {e}")

        # ── Алерт 19: суммарная загрузка капитала ──
        try:
            self._check_capital_load_alert(current_time)
        except Exception as e:
            logger.warning(f"[Alerts] capital load check failed: {e}")

        # ── Алерт 22: резкое движение BTC ──
        try:
            self._check_btc_market_move_alert(current_time)
        except Exception as e:
            logger.warning(f"[Alerts] BTC market move check failed: {e}")

        # ── Запуск AI listener (один раз за процесс) ──────────
        if not self._ai_init_done and self._openai_key:
            self._ai_init_done = True  # ставим флаг до проверок, чтобы не спамить
            # Авто-установка openai если пакета нет в контейнере
            if not OPENAI_AVAILABLE:
                _ensure_openai_installed()
            if not OPENAI_AVAILABLE:
                logger.warning(
                    "[AI] openai не установился автоматически. "
                    "Попробуй вручную: docker exec freqtrade3 pip install openai (и docker restart freqtrade3)"
                )
            elif not FT_CLIENT_AVAILABLE:
                logger.warning("[AI] freqtrade_client недоступен в окружении.")
            elif not self._ai_tg_token or not self._ai_tg_chat_id:
                logger.warning(
                    "[AI] openai_telegram_token не задан — listener не запущен. "
                    "Создай отдельного бота через @BotFather и пропиши его токен в "
                    "config.json → openai_telegram_token (нельзя переиспользовать основной "
                    "TG-токен Freqtrade — будет конфликт getUpdates)."
                )
            elif self._ai_tg_token == self._tg_token:
                logger.warning(
                    "[AI] openai_telegram_token совпадает с основным TG-токеном Freqtrade — "
                    "это вызовет конфликт getUpdates. Создай отдельного бота через @BotFather."
                )
            else:
                try:
                    api_cfg = self.config.get("api_server", {})
                    host = api_cfg.get("listen_ip_address", "127.0.0.1") or "127.0.0.1"
                    if host == "0.0.0.0":
                        host = "127.0.0.1"
                    port = int(api_cfg.get("listen_port", 8080))
                    api_url = f"http://{host}:{port}"
                    username = api_cfg.get("username", "")
                    password = api_cfg.get("password", "")

                    ft_client = FtRestClient(api_url, username, password)
                    # Свежие значения strategy-кастомных параметров — snapshot
                    # на момент init. Listener дополнительно читает их с диска
                    # на КАЖДЫЙ запрос «мой конфиг», поэтому даже устаревший
                    # snapshot не страшен — disk имеет приоритет.
                    fresh_strategy_params = {
                        "force_exit_by_days_enabled": self.force_exit_by_days_enabled,
                        "force_exit_after_days": self.force_exit_after_days,
                        "price_deviation_initial": self.price_deviation_initial,
                        "safety_order_max_count": self.safety_order_max_count,
                        "safety_order_ratio": self.safety_order_ratio,
                        "safety_order_volume_scale": self.safety_order_volume_scale,
                    }
                    config_file_paths = list(
                        self.config.get("config_files")
                        or self.config.get("original_config_files")
                        or []
                    )
                    # Singleton через threading.enumerate(): защищает от дублей даже после
                    # того как Freqtrade переимпортирует модуль стратегии при reload_config
                    if _ai_listener_already_running(self._ai_tg_token):
                        existing = _get_running_ai_listener(self._ai_tg_token)
                        if existing is not None:
                            # Старый listener (созданный из предыдущей версии класса)
                            # может не знать новых методов — защищаемся getattr-ом.
                            updater = getattr(existing, "update_strategy_params", None)
                            if callable(updater):
                                updater(fresh_strategy_params)
                            setter = getattr(existing, "set_config_files", None)
                            if callable(setter):
                                setter(config_file_paths)
                            # Свежее значение тумблера показа стоимости (можно гонять reload_config)
                            try:
                                existing.show_cost = self._ai_show_cost
                            except Exception:
                                pass
                            self._wire_ai_market_dir_handler(existing)
                            self._ai_listener = existing
                            logger.info(
                                f"[AI] Listener для токена {self._ai_tg_token[:10]}... "
                                f"уже работает — обновил strategy_params и config_files "
                                f"свежими значениями (force_exit/DCA)."
                            )
                        else:
                            logger.info(
                                f"[AI] Listener для токена {self._ai_tg_token[:10]}... "
                                f"УЖЕ работает в этом процессе — пропускаю создание дубля."
                            )
                    else:
                        self._ai_listener = TelegramAIListener(
                            tg_token=self._ai_tg_token,
                            chat_id=self._ai_tg_chat_id,
                            openai_key=self._openai_key,
                            ft_client=ft_client,
                            model=self._openai_model,
                            daily_summary_time=self._ai_daily_summary_time,
                            summary_tz=self._ai_summary_tz,
                            strategy_params=fresh_strategy_params,
                            show_cost=self._ai_show_cost,
                        )
                        self._ai_listener.set_config_files(config_file_paths)
                        self._wire_ai_market_dir_handler(self._ai_listener)
                        self._ai_listener.start()
                        logger.info(
                            f"[AI] TelegramAIListener запущен (model={self._openai_model}, "
                            f"api={api_url}, chat_id={self._ai_tg_chat_id})"
                        )
                except Exception as e:
                    logger.error(f"[AI] Не удалось запустить listener: {e}")