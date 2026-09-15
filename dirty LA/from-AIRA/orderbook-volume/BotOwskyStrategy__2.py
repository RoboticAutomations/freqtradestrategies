"""
# BOT-OWSKY Strategy - Advanced Multi-Phase ML Trading Strategy
#
# VERSION: 1.0.0 - Initial Implementation
#
# ARCHITECTURE:
# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Mathematical Tools & Pattern Loading
#   - Local Max/Min + Adaptive VWAP (fair value determination)
#   - Fibonacci Retracements (support/resistance levels)
#   - ATR for Volatility (trend and entry/exit zones)
#   - FFT for Cycle Detection (filter false signals)
#   - Kelly Criterion for Position Sizing (optimal position size)
#   - Markov Chains for State Transition Probabilities
#   - Candlestick Patterns (Encyclopedia of Candlestick Charts - Bulkowski)
#   - Clustering (pattern-strategy matching)
#   - Gaussian Processes (probabilistic predictions with uncertainty)
#   - Bayesian Models (probabilistic market sentiment)
#
# Phase 2: LSTM Neural Network
#   - Multivariate LSTM with dual input channels:
#     1. Mathematical/Statistical indicators
#     2. Clustered pattern recognition
#   - Temporal sequence capture
#   - Pattern recognition and risk management
#
# Phase 3: Decision System
#   - Random Forest scoring/ranking system
#   - Final buy/sell decision making
#   - Market regime detection (bullish/bearish/sideways)
#
# ═══════════════════════════════════════════════════════════════════════════════
#
# Author: Based on BOT-OWSKY concept
# Created: 2025-10-05
# Freqtrade Version: 2025.9+
# Python Version: 3.11+

# QUICK TEST CHECKLIST:
#   1. Backtest: `freqtrade backtesting --strategy BotOwskyStrategy --timerange 20240101-20240201`.
#   2. Enable alerts: set `notify_enabled=True` and configure Telegram RPC to receive strategy notifications.
#   3. Review logs: confirm REPAIR_* events, guardrail `PAIR_REGIME` notices, and leverage governor output.
"""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import talib.abstract as ta
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    optim = None
    _TORCH_AVAILABLE = False
import warnings
import logging
import pickle
import json
import math
from pathlib import Path
from typing import Any
from datetime import datetime, timedelta, date

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    CategoricalParameter,
    BooleanParameter,
    informative,
    merge_informative_pair
)
from freqtrade.persistence import Trade

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class BotOwskyStrategy(IStrategy):
    """
    BOT-OWSKY: Advanced Multi-Phase Machine Learning Strategy
    
    This strategy implements a three-phase approach:
    1. Mathematical analysis and pattern recognition
    2. LSTM-based temporal modeling
    3. Random Forest decision system
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY METADATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    INTERFACE_VERSION = 3
    
    # Strategy configuration
    timeframe = '1h'
    startup_candle_count: int = 200

    # --- Debug / Alerts ---
    debug_alerts: bool = False
    _debug_alerts_min_secs: int = 60  # min seconds between alerts per pair
    notify_enabled: bool = True

    # --- Auto-retrain toggles ---
    auto_retrain_enabled: bool = False   # OFF by default
    auto_retrain_hours: int = 24

    # --- Repair / risk controls ---
    enable_repair: bool = True

    # Strategy inputs (tunable risk controls)
    adx_min: float = 18.0
    flip_cooldown_candles: int = 2
    reduce_dd: float = -0.015
    reduce_conf: float = 0.55
    flip_conf: float = 0.6
    time_exit_candles: int = 30
    atr_k: float = 1.5
    trail_k: float = 1.0
    tp_mult: float = 2.0

    # Internal runtime state
    _last_alert_ts: dict[str, float]
    _auto_last_retrain_ts: float = 0.0

    # Trading mode
    can_short = True
    
    # ROI - Take profit configuration
    minimal_roi = {
        "0": 0.05,   # 5%
        "30": 0.035, # 3.5%
        "60": 0.02,  # 2%
        "120": 0.01  # 1%
    }

    
    # Stoploss
    stoploss = -0.08  # 8% stoploss
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True
    
    # Order types
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }
    
    # Order time in force
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }
    
    # Position adjustment
    position_adjustment_enable = True
    max_entry_position_adjustment = 2

        # --- DCA (averaging) controls ---
    dca_enabled: bool = True
    dca_triggers: list[float] = [-0.02, -0.05]   # add at ~-2% and ~-5% drawdown
    dca_multipliers: list[float] = [1.0, 1.5]    # size relative to initial stake
    dca_min_conf: float = 0.52                   # require model confidence in same direction
    dca_cooldown_candles: int = 2                # min bars between DCA adds

    
    # ===== LSTM configuration =====
    enable_lstm: bool = True                   # Turn LSTM signal on/off
    lstm_seq_len: int = 64                     # sequence length in timesteps
    lstm_hidden: int = 64                      # LSTM hidden size
    lstm_layers: int = 2                       # number of LSTM layers
    lstm_dropout: float = 0.1                  # dropout between LSTM layers
    lstm_lr: float = 1e-3                      # learning rate
    lstm_weight_decay: float = 1e-5            # weight decay
    lstm_epochs: int = 50                       # small fit for backtest only
    lstm_batch_size: int = 128                 # training batch size
    lstm_min_fit_bars: int = 400               # minimum candles to allow training
    lstm_label_horizon: int = 3                # future return horizon (bars)
    lstm_ret_threshold: float = 0.004          # 0.4% threshold for up/down labels
    lstm_train_on_backtest: bool = True        # allow training in backtests
    lstm_retrain_interval_candles: int = 10**9 # effectively "train once"
    lstm_feature_cols: list[str] | None = None # will be set in __init__

    # ═══════════════════════════════════════════════════════════════════════════
    # HYPEROPTABLE PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Phase 1: Mathematical Indicators
    atr_period = IntParameter(10, 30, default=14, space='buy', optimize=True)
    atr_multiplier = DecimalParameter(1.0, 3.0, default=2.0, space='buy', optimize=True)
    
    # Fibonacci levels
    use_fibonacci = BooleanParameter(default=True, space='buy', optimize=True)
    fib_lookback = IntParameter(20, 100, default=50, space='buy', optimize=True)
    
    # VWAP
    vwap_period = IntParameter(10, 50, default=20, space='buy', optimize=True)
    vwap_deviation_threshold = DecimalParameter(0.01, 0.05, default=0.02, space='buy', optimize=True)
    
    # FFT Cycle Detection
    use_fft_filter = BooleanParameter(default=True, space='buy', optimize=True)
    fft_threshold = DecimalParameter(0.3, 0.8, default=0.5, space='buy', optimize=True)
    
    # Kelly Criterion
    use_kelly = BooleanParameter(default=True, space='buy', optimize=True)
    kelly_fraction = DecimalParameter(0.1, 0.5, default=0.25, space='buy', optimize=True)
    
    # Machine Learning
    ml_confidence_threshold = DecimalParameter(0.5, 0.9, default=0.7, space='buy', optimize=True)
    use_gaussian_process = BooleanParameter(default=True, space='buy', optimize=True)
    use_clustering = BooleanParameter(default=True, space='buy', optimize=True)
    
    # Random Forest Decision
    rf_signal_threshold = DecimalParameter(0.5, 0.9, default=0.65, space='buy', optimize=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        # Model storage paths
        self.model_dir = Path(config.get('user_data_dir', 'user_data')) / 'models' / 'bot_owsky'
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._last_alert_ts = {}
        self._load_meta()

        self._pending_entry: dict[str, dict[str, Any]] = {}
        self._pair_history: dict[str, list[dict[str, float]]] = {}
        self._pair_guardrails: dict[str, dict[str, Any]] = {}
        self._pair_adjustments: dict[str, dict[str, float]] = {}
        self._pair_guardrail_date: dict[str, date] = {}
        self._pair_flip_requests: dict[str, dict[str, Any]] = {}

        self.lstm_model_path = self.model_dir / 'lstm_model.pt'
        self.lstm_scaler_path = self.model_dir / 'lstm_scaler.pkl'  # reuse sklearn scaler, separate instance
        self.lstm_device = torch.device('cpu') if _TORCH_AVAILABLE else None
        self.lstm_model = None
        self.lstm_scaler = StandardScaler()
        self._lstm_last_train_index = -1
        self._lstm_trained_global = False


        if self.lstm_feature_cols is None:
            self.lstm_feature_cols = [
                'rsi', 'rsi_slow', 'atr_percent', 'bb_width', 'volume_ratio',
                'vwap_distance', 'macd', 'macdhist', 'stoch_k', 'stoch_d',
                'distance_to_max', 'distance_to_min', 'bayesian_sentiment'
            ]

        if not _TORCH_AVAILABLE and self.enable_lstm:
            logger.warning('PyTorch not available - disabling LSTM signals.')
            self.enable_lstm = False

        if self.enable_lstm:
            self.load_lstm()

        # Initialize models
        self.random_forest_long = None
        self.random_forest_short = None
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.gp_model = None

        self.models_loaded = False
        self.load_models()
        
        # Pattern library (simplified - expandable with Bulkowski patterns)
        self.candlestick_patterns = [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
            'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS',
            'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBELTHOLD',
            'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL',
            'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
            'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING',
            'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE',
            'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN',
            'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE',
            'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON',
            'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER',
            'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM',
            'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU',
            'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR',
            'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING',
            'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES',
            'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP',
            'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI',
            'CDLTASUKIGAP', 'CDLTHRUSTING', 'CDLTRISTAR',
            'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'
        ]
        
        # Markov chain transition matrix (will be learned from data)
        self.markov_states = ['bullish', 'bearish', 'sideways']
        self.transition_matrix = None
        
        # Model loaded flags handled via load_models()

        logger.info("🚀 BOT-OWSKY Strategy initialized")
        # --- Compatibility helpers for Freqtrade versions (custom_info vs custom_data) ---

    
    # === Daily audit logging (NDJSON per day) ===
    def _audit_dir(self) -> Path:
        return self.model_dir / "audit"   # under user_data/models/bot_owsky/audit

    def _audit_path_for(self, dt: datetime | None = None) -> Path:
        dt = dt or datetime.utcnow()
        return self._audit_dir() / f"{dt.strftime('%Y-%m-%d')}.log"

    def _audit_write(self, event: str, payload: dict | None = None) -> None:
        try:
            p = dict(payload or {})
            # Normalize a few common fields
            rec = {
                "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "event": event,                       # e.g. ENTRY_ACCEPT, REJECT_low_conf, DCA_ADD, DCA_BLOCKED, ROI, etc.
                "pair": p.pop("pair", None),
                "side": p.pop("side", None),
                **p,
            }
            d = self._audit_dir()
            d.mkdir(parents=True, exist_ok=True)
            with open(self._audit_path_for(), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            # don't disrupt trading if disk is full or path invalid
            pass


    def _to_utc(self, ts):
        from datetime import datetime, timezone
        import pandas as pd
        """Return timezone-aware UTC datetime for pandas/py datetime or None."""
        if ts is None:
            return None
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is None:
                return ts.tz_localize("UTC").to_pydatetime()
            return ts.tz_convert("UTC").to_pydatetime()
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            return ts.astimezone(timezone.utc)
        # Fallback: try pandas to coerce
        ts = pd.Timestamp(ts)
        return self._to_utc(ts)
    
    # ===== Safe per-trade scratchpad (no ORM access) =====
    _ti_store: dict[int, dict] = {}

    def _ti_get(self, trade) -> dict:
        tid = getattr(trade, 'id', None)
        if tid in self._ti_store:
            return self._ti_store[tid]
        # try non-loading read (won't trigger lazy loader)
        dct = getattr(trade, '__dict__', {})
        info = dct.get('custom_info') or dct.get('custom_data') or {}
        if isinstance(info, dict) and tid is not None:
            self._ti_store[tid] = info
        elif tid is not None:
            self._ti_store[tid] = {}
        return self._ti_store.get(tid, {})

    def _ti_set(self, trade, info: dict) -> None:
        tid = getattr(trade, 'id', None)
        if tid is not None:
            self._ti_store[tid] = info

    def _ti_ensure(self, trade, pair: str) -> None:
        info = self._ti_get(trade)
        if not info:
            info = {}
        info.setdefault('pair', pair)
        self._ti_set(trade, info)

    def _ti_clear(self, trade) -> None:
        tid = getattr(trade, 'id', None)
        if tid in self._ti_store:
            self._ti_store.pop(tid, None)


    def _send_tg(self, text: str) -> None:
        try:
            from freqtrade.rpc import RPCMessageType
            # freqtrade Strategy exposes rpc_send_msg in live mode
            fn = getattr(self, "rpc_send_msg", None) or getattr(self, "_rpc_send_msg", None)
            if fn:
                fn({"type": RPCMessageType.INFO, "status": text})
            else:
                logger.info(f"[TG] {text}")
        except Exception:
            logger.info(f"[TG] {text}")

    def _can_alert(self, pair: str) -> bool:
        import time
        now = time.time()
        last = self._last_alert_ts.get(pair, 0.0)
        if now - last >= self._debug_alerts_min_secs:
            self._last_alert_ts[pair] = now
            return True
        return False

    def _meta_path(self):
        return self.model_dir / "train_meta.json"

    def _load_meta(self) -> None:
        try:
            if self._meta_path().exists():
                with open(self._meta_path(), "r") as f:
                    m = json.load(f)
                    self._auto_last_retrain_ts = float(m.get("last_ts", 0.0))
            else:
                self._auto_last_retrain_ts = 0.0
        except Exception:
            self._auto_last_retrain_ts = 0.0

    def _save_meta(self, ts: float) -> None:
        try:
            with open(self._meta_path(), "w") as f:
                json.dump({"last_ts": float(ts)}, f)
        except Exception:
            pass


    def notify(self, reason: str, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        pair = payload.get('pair', 'N/A')
        side = payload.get('side', '-')
        action = payload.get('action', reason)
        parts = [str(part) for part in (pair, side, action) if part is not None]
        for key in ('pnl', 'conf', 'adx', 'vol', 'r'):
            if key in payload:
                parts.append(f"{key}={payload[key]}")
        for key, value in payload.items():
            if key in {'pair', 'side', 'action', 'pnl', 'conf', 'adx', 'vol', 'r'}:
                continue
            parts.append(f"{key}={value}")
        parts.append(f"reason={reason}")
        message = ' '.join(parts).strip()
        # >>> NEW: write to daily audit log
        try:
            self._audit_write(reason, dict(payload, action=action))
        except Exception:
            pass
        if self.notify_enabled:
            try:
                self._send_tg(message)
            except Exception:
                logger.info(f"[NOTIFY] {message}")
        else:
            logger.info(f"[NOTIFY] {message}")
    def _dca_dbg(self, trade, reason: str, pnl=None, conf=None, opp=None):
        self.notify('DCA_BLOCKED', {
            'pair': getattr(trade, 'pair', 'UNKNOWN'),
            'side': 'short' if trade.is_short else 'long',
            'pnl': f"{(pnl or 0)*100:.2f}%",
            'conf': f"{(conf or 0):.2f}",
            'opp': f"{(opp or 0):.2f}",
            'reason': reason,
        })
        return 0

    def _timeframe_minutes(self) -> int:
        tf = self.timeframe
        if isinstance(tf, int):
            return max(1, int(tf))
        unit = tf[-1]
        value = int(tf[:-1]) if tf[:-1].isdigit() else 1
        multipliers = {'m': 1, 'h': 60, 'd': 1440}
        return max(1, value * multipliers.get(unit, 1))

    def _get_analyzed_dataframe(self, pair: str) -> DataFrame | None:
        """Return analyzed dataframe for pair, unwrapping (df, meta) tuples across FT versions."""
        try:
            res = self.dp.get_analyzed_dataframe(pair, self.timeframe)  # type: ignore[attr-defined]
            # Newer FT may return (df, last_analyzed) – normalize to df
            if isinstance(res, tuple):
                df = res[0]
            else:
                df = res
            return df
        except Exception:
            return None

    def _get_latest_row(self, pair: str) -> tuple[pd.Timestamp | None, pd.Series | None]:
        """Return (timestamp, last_row) for the analyzed dataframe, or (None, None)."""
        df = self._get_analyzed_dataframe(pair)
        if df is None or getattr(df, "empty", True):
            return None, None
        last = df.iloc[-1]
        ts = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None
        return ts, last


    def _ensure_trade_info(self, trade: Trade, pair: str) -> None:
        info = self._ti_get(trade) or {}
        if info.get('initialized'):
            return
        pending = self._pending_entry.pop(pair, {}) if pair in self._pending_entry else {}
        if isinstance(pending, dict):
            info.update(pending)
        info.setdefault('initialized', True)
        info.setdefault('reduced', False)
        info.setdefault('trail_active', False)
        info.setdefault('move_to_be', False)
        info.setdefault('flip_armed', False)
        info.setdefault('history_updated', False)
        adjustments = self._pair_adjustments.get(pair, {'atr_factor': 1.0, 'tp_factor': 1.0, 'min_conf': 0.5})
        info.setdefault('pair_adjustments', adjustments)
        info.setdefault('atr_mult', self.atr_k * adjustments.get('atr_factor', 1.0))
        info.setdefault('trail_mult', self.trail_k)
        info.setdefault('tp_target', self.tp_mult * adjustments.get('tp_factor', 1.0))
        info.setdefault('min_conf', max(0.4, adjustments.get('min_conf', 0.4)))
        info.setdefault('max_leverage', pending.get('max_leverage') if isinstance(pending, dict) else None)
        info.setdefault('dca_count', 0)
        info.setdefault('dca_last_bar', -999_999)

        self._ti_set(trade, info)

    def _bars_in_trade(self, trade, current_time) -> int:
        from datetime import datetime, timezone
        """
        How many bars the trade has been open for.
        Makes timestamps timezone-safe (UTC) to avoid naive/aware subtraction errors.
        """
        # Prefer the UTC field from Freqtrade’s Trade object
        open_time = getattr(trade, "open_date_utc", None) or getattr(trade, "open_date", None)
        open_time = self._to_utc(open_time)

        # current_time may be naive; normalize
        now_utc = self._to_utc(current_time) or datetime.now(timezone.utc)

        if open_time is None:
            return 0

        delta = now_utc - open_time
        # Convert to bars for this strategy timeframe
        tf_sec = self.timeframe_to_minutes(self.timeframe) * 60 if hasattr(self, "timeframe_to_minutes") else (
            int(pd.Timedelta(self.timeframe).total_seconds())
        )
        return max(0, int(delta.total_seconds() // tf_sec))


    def _refresh_pair_guardrails(self, pair: str, current_time: datetime) -> None:
        today = current_time.date()
        if self._pair_guardrail_date.get(pair) == today:
            return
        history = self._pair_history.get(pair, [])
        if not history:
            self._pair_adjustments[pair] = {'atr_factor': 1.0, 'tp_factor': 1.0, 'min_conf': 0.4}
            self._pair_guardrail_date[pair] = today
            return
        sample = history[-20:]
        wins = sum(1 for h in sample if h.get('win'))
        winrate = wins / len(sample)
        avg_mae = sum(h.get('mae', 0.0) for h in sample) / len(sample)
        baseline = self._pair_guardrails.get(pair, {}).get('baseline_mae', avg_mae)
        if baseline == 0:
            baseline = avg_mae or 1e-6
        adjustments = {'atr_factor': 1.0, 'tp_factor': 1.0, 'min_conf': 0.5}
        triggered = False
        if winrate < 0.4 or avg_mae > 1.2 * baseline:
            adjustments['atr_factor'] = 1.2
            adjustments['tp_factor'] = 0.8
            adjustments['min_conf'] = max(0.4, adjustments['min_conf'])
            triggered = True
        self._pair_adjustments[pair] = adjustments
        self._pair_guardrails.setdefault(pair, {})['baseline_mae'] = baseline
        self._pair_guardrail_date[pair] = today
        if triggered:
            self.notify('PAIR_REGIME', {
                'pair': pair,
                'side': '-',
                'action': 'adjust',
                'winrate': f"{winrate:.2f}",
                'mae': f"{avg_mae:.4f}"
            })

    def _update_pair_history(self, pair: str, trade: Trade, close_rate: float) -> None:
        history = self._pair_history.setdefault(pair, [])
        is_short = trade.is_short
        open_rate = trade.open_rate
        max_rate = trade.max_rate or open_rate
        min_rate = trade.min_rate or open_rate
        if is_short:
            mae = abs((max_rate - open_rate) / open_rate)
            mfe = abs((open_rate - min_rate) / open_rate)
        else:
            mae = abs((open_rate - min_rate) / open_rate)
            mfe = abs((max_rate - open_rate) / open_rate)
        profit = trade.calc_profit_ratio(close_rate)
        history.append({'win': profit > 0, 'mae': mae, 'mfe': mfe, 'profit': profit})
        if len(history) > 20:
            del history[:-20]

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                return default
            return val
        except Exception:
            return default

    if _TORCH_AVAILABLE:
        class _OWSLSTM(nn.Module):
            def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=in_dim,
                    hidden_size=hidden,
                    num_layers=layers,
                    batch_first=True,
                    dropout=dropout if layers > 1 else 0.0,
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 2)
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                logits = self.head(last)
                return logits
    else:
        class _OWSLSTM:  # pragma: no cover - torch unavailable fallback
            def __init__(self, *args, **kwargs):
                raise RuntimeError('PyTorch is required for LSTM functionality.')

            def forward(self, *args, **kwargs):
                raise RuntimeError('PyTorch is required for LSTM functionality.')

    def load_lstm(self):
        if not self.enable_lstm or not _TORCH_AVAILABLE:
            return
        try:
            if self.lstm_model_path.exists() and self.lstm_scaler_path.exists():
                model = self._OWSLSTM(
                    in_dim=len(self.lstm_feature_cols),
                    hidden=self.lstm_hidden,
                    layers=self.lstm_layers,
                    dropout=self.lstm_dropout
                ).to(self.lstm_device)
                state = torch.load(self.lstm_model_path, map_location=self.lstm_device or 'cpu')
                model.load_state_dict(state)
                model.eval()
                with open(self.lstm_scaler_path, 'rb') as f:
                    self.lstm_scaler = pickle.load(f)
                self.lstm_model = model
                logger.info("Loaded LSTM model and scaler")
            else:
                logger.info("LSTM model/scaler not found, will use neutral outputs.")
        except Exception as e:
            logger.warning(f"LSTM load failed: {e}")
            self.lstm_model = None

    def save_lstm(self):
        if not self.enable_lstm or not _TORCH_AVAILABLE:
            return
        try:
            if self.lstm_model is not None:
                torch.save(self.lstm_model.state_dict(), self.lstm_model_path)
                with open(self.lstm_scaler_path, 'wb') as f:
                    pickle.dump(self.lstm_scaler, f)
                logger.info("Saved LSTM model and scaler")
        except Exception as e:
            logger.error(f"LSTM save failed: {e}")

    def _lstm_build_dataset(self, df: DataFrame, feature_cols: list[str], seq_len: int, horizon: int, thr: float):
        tmp = df.copy()
        for c in feature_cols:
            if c not in tmp.columns:
                tmp[c] = 0.0

        fut = tmp['close'].pct_change(horizon).shift(-horizon)
        y = np.full(len(tmp), -1, dtype=np.int64)
        y[(fut > thr).fillna(False)] = 0
        y[(fut < -thr).fillna(False)] = 1

        X = tmp[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.lstm_scaler.fit_transform(X)

        seqs, labels = [], []
        for i in range(seq_len, len(tmp) - horizon):
            if y[i] == -1:
                continue
            seq = X_scaled[i - seq_len:i, :]
            seqs.append(seq)
            labels.append(y[i])

        if not seqs:
            return None, None

        X_tensor = torch.tensor(np.stack(seqs, axis=0), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(labels, dtype=np.int64))
        return X_tensor, y_tensor

    def train_lstm(self, df: DataFrame):
        if not self.enable_lstm or not self.lstm_train_on_backtest or not _TORCH_AVAILABLE:
            return False
        if len(df) < self.lstm_min_fit_bars:
            return False

        try:
            self.lstm_scaler = StandardScaler()
            X, y = self._lstm_build_dataset(
                df, self.lstm_feature_cols, self.lstm_seq_len,
                self.lstm_label_horizon, self.lstm_ret_threshold
            )
            if X is None:
                logger.info("LSTM: Not enough labeled samples to train.")
                return False

            model = self._OWSLSTM(
                in_dim=X.shape[-1],
                hidden=self.lstm_hidden,
                layers=self.lstm_layers,
                dropout=self.lstm_dropout
            ).to(self.lstm_device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.lstm_lr, weight_decay=self.lstm_weight_decay)

            model.train()
            dataset = torch.utils.data.TensorDataset(X, y)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.lstm_batch_size, shuffle=True, drop_last=False)

            for epoch in range(self.lstm_epochs):
                total = 0.0
                for xb, yb in loader:
                    xb = xb.to(self.lstm_device)
                    yb = yb.to(self.lstm_device)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    total += float(loss.item()) * len(yb)
                logger.info(f"LSTM epoch {epoch + 1}/{self.lstm_epochs} loss={total/len(dataset):.6f}")

            self.lstm_model = model.eval()
            self.save_lstm()
            return True

        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            self.lstm_model = None
            return False

    def lstm_predict_series(self, df: DataFrame) -> tuple[Series, Series, Series]:
        idx = df.index
        long_s = pd.Series(0.5, index=idx)
        short_s = pd.Series(0.5, index=idx)
        conf_s = pd.Series(0.0, index=idx)

        if not self.enable_lstm or self.lstm_model is None or not _TORCH_AVAILABLE:
            return long_s, short_s, conf_s

        try:
            need = self.lstm_seq_len
            if len(df) < need + 1:
                return long_s, short_s, conf_s

            feats = df.copy()
            for c in (self.lstm_feature_cols or []):
                if c not in feats.columns:
                    feats[c] = 0.0
            X = feats[self.lstm_feature_cols].values.astype(np.float32) if self.lstm_feature_cols else feats.values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            if not hasattr(self.lstm_scaler, 'scale_'):
                return long_s, short_s, conf_s
            X_scaled = self.lstm_scaler.transform(X)

            seq = torch.tensor(X_scaled[-need:, :], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = self.lstm_model(seq.to(self.lstm_device))
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            p_long, p_short = float(probs[0]), float(probs[1])
            confidence = float(abs(p_long - p_short))

            long_s.iloc[-1] = p_long
            short_s.iloc[-1] = p_short
            conf_s.iloc[-1] = confidence
            return long_s, short_s, conf_s

        except Exception as e:
            logger.warning(f"LSTM inference failed: {e}")
            return long_s, short_s, conf_s

    def load_models(self):
        """Load pre-trained models if available"""
        try:
            rf_long_path = self.model_dir / 'random_forest_long.pkl'
            rf_short_path = self.model_dir / 'random_forest_short.pkl'
            scaler_path = self.model_dir / 'scaler.pkl'
            cluster_path = self.model_dir / 'cluster_model.pkl'
            
            if rf_long_path.exists():
                with open(rf_long_path, 'rb') as f:
                    self.random_forest_long = pickle.load(f)
                logger.info("✅ Loaded Random Forest LONG model")
            
            if rf_short_path.exists():
                with open(rf_short_path, 'rb') as f:
                    self.random_forest_short = pickle.load(f)
                logger.info("✅ Loaded Random Forest SHORT model")
            
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("✅ Loaded scaler")
            
            if cluster_path.exists():
                with open(cluster_path, 'rb') as f:
                    self.cluster_model = pickle.load(f)
                logger.info("✅ Loaded clustering model")
            have = (self.random_forest_long is not None) and (self.random_forest_short is not None)
            self.models_loaded = have
            # self.models_loaded = True
            
        except Exception as e:
            logger.warning(f"⚠️  Could not load models: {e}")
            self.models_loaded = False
    
    def save_models(self):
        """Save trained models"""
        try:
            if self.random_forest_long:
                with open(self.model_dir / 'random_forest_long.pkl', 'wb') as f:
                    pickle.dump(self.random_forest_long, f)
            
            if self.random_forest_short:
                with open(self.model_dir / 'random_forest_short.pkl', 'wb') as f:
                    pickle.dump(self.random_forest_short, f)
            
            if self.scaler:
                with open(self.model_dir / 'scaler.pkl', 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            if self.cluster_model:
                with open(self.model_dir / 'cluster_model.pkl', 'wb') as f:
                    pickle.dump(self.cluster_model, f)
            
            logger.info("💾 Models saved successfully")
        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1: MATHEMATICAL TOOLS & INDICATORS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_local_extrema(self, dataframe: DataFrame, column: str = 'close', order: int = 5) -> tuple[Series, Series]:
        """
        Calculate local maxima and minima using scipy
        """
        data = dataframe[column].values
        
        # Find local maxima
        max_indices = signal.argrelextrema(data, np.greater, order=order)[0]
        local_max = pd.Series(np.nan, index=dataframe.index)
        if len(max_indices) > 0:
            local_max.iloc[max_indices] = data[max_indices]
        
        # Find local minima
        min_indices = signal.argrelextrema(data, np.less, order=order)[0]
        local_min = pd.Series(np.nan, index=dataframe.index)
        if len(min_indices) > 0:
            local_min.iloc[min_indices] = data[min_indices]
        
        # Forward fill for easier reference
        local_max_filled = local_max.ffill()
        local_min_filled = local_min.ffill()
        
        return local_max_filled, local_min_filled
    
    def calculate_adaptive_vwap(self, dataframe: DataFrame, period: int = 20) -> Series:
        """
        Calculate adaptive VWAP (Volume Weighted Average Price)
        """
        typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        vwap = (typical_price * dataframe['volume']).rolling(window=period).sum() / dataframe['volume'].rolling(window=period).sum()
        return vwap
    
    def calculate_fibonacci_levels(self, dataframe: DataFrame, lookback: int = 50) -> dict[str, Series]:
        """
        Calculate Fibonacci retracement levels
        """
        rolling_max = dataframe['high'].rolling(window=lookback).max()
        rolling_min = dataframe['low'].rolling(window=lookback).min()
        diff = rolling_max - rolling_min
        
        fib_levels = {
            'fib_0': rolling_min,
            'fib_236': rolling_min + 0.236 * diff,
            'fib_382': rolling_min + 0.382 * diff,
            'fib_500': rolling_min + 0.500 * diff,
            'fib_618': rolling_min + 0.618 * diff,
            'fib_786': rolling_min + 0.786 * diff,
            'fib_100': rolling_max,
        }
        
        return fib_levels
    
    def calculate_fft_cycles(self, dataframe: DataFrame, column: str = 'close', threshold: float = 0.5) -> Series:
        """
        Fast Fourier Transform for cycle detection
        Filters out high-frequency noise to identify underlying trends
        """
        if len(dataframe) < 50:
            return pd.Series(0, index=dataframe.index)
        
        # Get price data
        prices = dataframe[column].values
        
        # Apply FFT
        fft_values = fft(prices)
        frequencies = fftfreq(len(prices))
        
        # Filter out high frequencies (noise)
        fft_filtered = fft_values.copy()
        fft_filtered[np.abs(frequencies) > threshold] = 0
        
        # Inverse FFT to get filtered signal
        filtered_signal = np.fft.ifft(fft_filtered).real
        
        # Calculate cycle strength (difference between original and filtered)
        cycle_strength = np.abs(prices - filtered_signal)
        
        return pd.Series(cycle_strength, index=dataframe.index)
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing
        
        Formula: f* = (p * b - q) / b
        where:
        - p = win_rate (probability of winning)
        - q = 1 - p (probability of losing)
        - b = avg_win / avg_loss (win/loss ratio)
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        b = avg_win / avg_loss
        q = 1 - win_rate
        
        kelly = (win_rate * b - q) / b
        
        # Return conservative Kelly (usually use 1/4 to 1/2 Kelly)
        return max(0, min(kelly * self.kelly_fraction.value, 1.0))
    
    def calculate_markov_probabilities(self, dataframe: DataFrame) -> dict[str, float]:
        """
        Calculate Markov chain state transition probabilities
        States: bullish, bearish, sideways
        """
        # Simplified state detection based on trend
        returns = dataframe['close'].pct_change()
        
        # Define states
        bullish_threshold = 0.005  # 0.5% per candle
        bearish_threshold = -0.005
        
        states = pd.Series('sideways', index=dataframe.index)
        states[returns > bullish_threshold] = 'bullish'
        states[returns < bearish_threshold] = 'bearish'
        
        # Count transitions
        if len(states) < 2:
            return {'bullish': 0.33, 'bearish': 0.33, 'sideways': 0.34}
        
        current_state = states.iloc[-1]
        
        # Look at last N candles for transition probabilities
        recent_states = states.iloc[-50:] if len(states) >= 50 else states
        
        # Count state occurrences
        state_counts = recent_states.value_counts()
        total = len(recent_states)
        
        probabilities = {
            'bullish': state_counts.get('bullish', 0) / total,
            'bearish': state_counts.get('bearish', 0) / total,
            'sideways': state_counts.get('sideways', 0) / total
        }
        
        return probabilities
    
    def detect_candlestick_patterns(self, dataframe: DataFrame) -> DataFrame:
        """
        Detect candlestick patterns using TA-Lib
        Based on Encyclopedia of Candlestick Charts (Thomas N. Bulkowski)
        """
        df = dataframe.copy()
        
        # Detect all patterns
        for pattern in self.candlestick_patterns:
            try:
                df[f'pattern_{pattern}'] = getattr(ta, pattern)(df)
            except Exception:
                df[f'pattern_{pattern}'] = 0
        
        # Create aggregated pattern signals
        pattern_cols = [col for col in df.columns if col.startswith('pattern_')]
        df['bullish_pattern_count'] = df[pattern_cols].apply(lambda x: (x > 0).sum(), axis=1)
        df['bearish_pattern_count'] = df[pattern_cols].apply(lambda x: (x < 0).sum(), axis=1)
        df['pattern_strength'] = df['bullish_pattern_count'] - df['bearish_pattern_count']
        
        return df
    
    def calculate_gaussian_process_prediction(self, dataframe: DataFrame, feature_cols: list[str]) -> tuple[Series, Series]:
        """
        Gaussian Process for probabilistic predictions with uncertainty
        Returns: (mean_prediction, uncertainty)
        """
        if len(dataframe) < 50:
            return pd.Series(0, index=dataframe.index), pd.Series(1, index=dataframe.index)
        
        try:
            # Prepare training data (use last 100 candles)
            train_size = min(100, len(dataframe) - 1)
            X_train = dataframe[feature_cols].iloc[-train_size:-1].values
            y_train = dataframe['close'].pct_change().iloc[-train_size:].values[1:]
            
            # Handle NaN values
            X_train = np.nan_to_num(X_train, nan=0.0)
            y_train = np.nan_to_num(y_train, nan=0.0)
            
            # Define kernel
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            
            # Create and fit GP model
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-6)
            gp.fit(X_train, y_train)
            
            # Predict on current data
            X_current = dataframe[feature_cols].iloc[-1:].values
            X_current = np.nan_to_num(X_current, nan=0.0)
            
            mean_pred, std_pred = gp.predict(X_current, return_std=True)
            
            # Create series for full dataframe
            mean_series = pd.Series(0, index=dataframe.index)
            mean_series.iloc[-1] = mean_pred[0]
            
            std_series = pd.Series(1, index=dataframe.index)
            std_series.iloc[-1] = std_pred[0]
            
            return mean_series, std_series
            
        except Exception as e:
            logger.warning(f"GP prediction failed: {e}")
            return pd.Series(0, index=dataframe.index), pd.Series(1, index=dataframe.index)
    
    def calculate_bayesian_sentiment(self, dataframe: DataFrame) -> Series:
        """
        Bayesian model for market sentiment combining multiple indicators
        Returns posterior probability of upward movement
        """
        # Prior: neutral 50/50
        prior_up = 0.5
        
        # Likelihood from RSI
        rsi = dataframe.get('rsi', pd.Series(50, index=dataframe.index))
        p_rsi_up = (100 - rsi) / 100  # Probability of up given oversold
        
        # Likelihood from volume
        volume_ratio = dataframe['volume'] / dataframe['volume'].rolling(20).mean()
        p_volume_up = (volume_ratio - 1).clip(0, 1)  # Higher volume suggests momentum
        
        # Likelihood from trend (EMA)
        if 'ema_20' in dataframe.columns and 'ema_50' in dataframe.columns:
            trend_up = (dataframe['ema_20'] > dataframe['ema_50']).astype(float)
            p_trend_up = trend_up * 0.7 + 0.15  # 70% when uptrend, 15% when downtrend
        else:
            p_trend_up = pd.Series(0.5, index=dataframe.index)
        
        # Bayesian update (simplified)
        # P(up|evidence) = P(evidence|up) * P(up) / P(evidence)
        # Using log odds for numerical stability
        
        log_odds = np.log(prior_up / (1 - prior_up))
        log_odds += np.log(p_rsi_up / (1 - p_rsi_up + 1e-10))
        log_odds += np.log(p_trend_up / (1 - p_trend_up + 1e-10))
        
        # Convert back to probability
        posterior_up = 1 / (1 + np.exp(-log_odds))
        
        return pd.Series(posterior_up, index=dataframe.index)
    
    def perform_clustering(self, dataframe: DataFrame, feature_cols: list[str], n_clusters: int = 5) -> Series:
        """
        Cluster market conditions to identify similar patterns
        """
        if len(dataframe) < n_clusters * 2:
            return pd.Series(0, index=dataframe.index)
        
        try:
            # Prepare data
            X = dataframe[feature_cols].values
            X = np.nan_to_num(X, nan=0.0)
            
            # Fit or use existing cluster model
            if self.cluster_model is None:
                self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                self.cluster_model.fit(X)
            
            # Predict clusters
            clusters = self.cluster_model.predict(X)
            
            return pd.Series(clusters, index=dataframe.index)
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return pd.Series(0, index=dataframe.index)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: LSTM NEURAL NETWORK
    # ═══════════════════════════════════════════════════════════════════════════
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3: RANDOM FOREST DECISION SYSTEM
    # ═══════════════════════════════════════════════════════════════════════════
    
    def train_random_forest(self, dataframe: DataFrame, feature_cols: list[str]):
        """
        Train Random Forest models for buy/sell decisions
        """
        if len(dataframe) < 200:
            logger.warning("Not enough data to train Random Forest")
            return
        
        try:
            # Prepare features
            X = dataframe[feature_cols].values
            X = np.nan_to_num(X, nan=0.0)
            
            # Create labels (future returns)
            future_returns = dataframe['close'].pct_change(3).shift(-3)  # 3-candle forward return
            
            # Binary classification: up vs down
            y_long = (future_returns > 0.01).astype(int).values  # 1% threshold for long
            y_short = (future_returns < -0.01).astype(int).values  # -1% threshold for short
            
            # Remove NaN values
            valid_mask = ~np.isnan(future_returns.values)
            X = X[valid_mask]
            y_long = y_long[valid_mask]
            y_short = y_short[valid_mask]
            
            if len(X) < 50:
                return
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train LONG model
            self.random_forest_long = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            self.random_forest_long.fit(X_scaled, y_long)
            
            # Train SHORT model
            self.random_forest_short = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            self.random_forest_short.fit(X_scaled, y_short)
            
            logger.info("✅ Random Forest models trained successfully")
            
        except Exception as e:
            logger.error(f"❌ Random Forest training failed: {e}")
    
    def random_forest_predict(self, dataframe: DataFrame, feature_cols: list[str]) -> tuple[Series, Series]:
        """
        Get Random Forest predictions for current data
        Returns: (long_probability, short_probability)
        """
        if self.random_forest_long is None or self.random_forest_short is None:
            return pd.Series(0.5, index=dataframe.index), pd.Series(0.5, index=dataframe.index)
        
        try:
            # Prepare features
            X = dataframe[feature_cols].values
            X = np.nan_to_num(X, nan=0.0)
            X_scaled = self.scaler.transform(X)
            
            # Get predictions
            long_proba = self.random_forest_long.predict_proba(X_scaled)[:, 1]
            short_proba = self.random_forest_short.predict_proba(X_scaled)[:, 1]
            
            return pd.Series(long_proba, index=dataframe.index), pd.Series(short_proba, index=dataframe.index)
            
        except Exception as e:
            logger.warning(f"RF prediction failed: {e}")
            return pd.Series(0.5, index=dataframe.index), pd.Series(0.5, index=dataframe.index)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FREQTRADE INTERFACE METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add all indicators to the dataframe
        This is where all three phases are computed
        """
        
        # ═══ PHASE 1: MATHEMATICAL TOOLS ═══
        
        # Basic indicators
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=21)
        
        # Moving averages
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        
        # ATR for volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
        dataframe['atr_percent'] = (dataframe['atr'] / dataframe['close']) * 100
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Stochastic
        stoch = ta.STOCH(dataframe)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']
        
        # Directional movement and volatility regime signals
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        returns = np.log(dataframe['close'] / dataframe['close'].shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        tf_minutes = max(1, self._timeframe_minutes())
        window_1h = max(1, int(round(60 / tf_minutes)))
        window_24h = max(window_1h, int(round(1440 / tf_minutes)))
        dataframe['rv_1h'] = returns.rolling(window_1h).std(ddof=0).fillna(0.0)
        dataframe['rv_24h'] = returns.rolling(window_24h).std(ddof=0).fillna(0.0)
        dataframe['vol_ratio'] = dataframe['rv_1h'] / (dataframe['rv_24h'] + 1e-9)
        dataframe['regime_chop'] = (dataframe['adx'].fillna(0.0) < self.adx_min)

        # Volume indicators
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # Adaptive VWAP
        dataframe['vwap'] = self.calculate_adaptive_vwap(dataframe, period=self.vwap_period.value)
        dataframe['vwap_distance'] = (dataframe['close'] - dataframe['vwap']) / dataframe['vwap']
        
        # Local extrema
        dataframe['local_max'], dataframe['local_min'] = self.calculate_local_extrema(dataframe)
        dataframe['distance_to_max'] = (dataframe['close'] - dataframe['local_max']) / dataframe['local_max']
        dataframe['distance_to_min'] = (dataframe['close'] - dataframe['local_min']) / dataframe['local_min']
        
        # Fibonacci levels
        if self.use_fibonacci.value:
            fib_levels = self.calculate_fibonacci_levels(dataframe, lookback=self.fib_lookback.value)
            for key, series in fib_levels.items():
                dataframe[key] = series
            
            # Distance to key Fibonacci levels
            dataframe['distance_to_fib_618'] = (dataframe['close'] - dataframe['fib_618']) / dataframe['fib_618']
            dataframe['distance_to_fib_382'] = (dataframe['close'] - dataframe['fib_382']) / dataframe['fib_382']
        
        # FFT cycle detection
        if self.use_fft_filter.value:
            dataframe['fft_cycle_strength'] = self.calculate_fft_cycles(dataframe, threshold=self.fft_threshold.value)
        
        # Markov probabilities
        markov_probs = self.calculate_markov_probabilities(dataframe)
        dataframe['markov_bullish'] = markov_probs['bullish']
        dataframe['markov_bearish'] = markov_probs['bearish']
        dataframe['markov_sideways'] = markov_probs['sideways']
        
        # Candlestick patterns
        dataframe = self.detect_candlestick_patterns(dataframe)
        
        # Bayesian sentiment
        dataframe['bayesian_sentiment'] = self.calculate_bayesian_sentiment(dataframe)
        
        # ═══ PHASE 1: CLUSTERING ═══
        
        if self.use_clustering.value:
            cluster_features = [
                'rsi', 'atr_percent', 'bb_width', 'volume_ratio',
                'vwap_distance', 'macd', 'stoch_k'
            ]
            dataframe['market_cluster'] = self.perform_clustering(dataframe, cluster_features, n_clusters=5)
        else:
            dataframe['market_cluster'] = 0
        
        # ═══ PHASE 1: GAUSSIAN PROCESS ═══
        
        if self.use_gaussian_process.value:
            gp_features = ['rsi', 'atr_percent', 'volume_ratio', 'macd', 'bb_width']
            dataframe['gp_prediction'], dataframe['gp_uncertainty'] = self.calculate_gaussian_process_prediction(
                dataframe, gp_features
            )
            
            # High confidence signal: positive prediction with low uncertainty
            dataframe['gp_confidence'] = dataframe['gp_prediction'] / (dataframe['gp_uncertainty'] + 1e-6)
        else:
            dataframe['gp_prediction'] = 0
            dataframe['gp_uncertainty'] = 1
            dataframe['gp_confidence'] = 0
        
        # ===== PHASE 2: LSTM (full) =====
        # Train once in backtesting if requested and not trained yet.
        # ===== PHASE 2: LSTM (global one-time training) =====
        if self.enable_lstm and not self._lstm_trained_global and self.lstm_train_on_backtest:
            if len(dataframe) >= self.lstm_min_fit_bars:
                logger.info("🚀 Training LSTM globally (once per run)...")
                if self.train_lstm(dataframe.copy()):
                    self._lstm_trained_global = True
                    logger.info("✅ LSTM model trained and saved globally.")
                else:
                    logger.warning("⚠️ LSTM training failed or not enough data.")


        # Always attempt to load if still None (e.g., live mode, pre-trained externally)
        if self.enable_lstm and self.lstm_model is None:
            self.load_lstm()

        # Predict last bar probabilities (neutral if model missing)
        dataframe['lstm_long'], dataframe['lstm_short'], dataframe['lstm_conf'] = self.lstm_predict_series(dataframe)
        
        # ═══ PHASE 3: RANDOM FOREST DECISION ═══
        
        # Define feature set for RF
        rf_features = [
            'rsi', 'rsi_slow', 'atr_percent', 'bb_width', 'volume_ratio',
            'vwap_distance', 'macd', 'macdhist', 'stoch_k', 'stoch_d',
            'distance_to_max', 'distance_to_min',
            'pattern_strength', 'bullish_pattern_count', 'bearish_pattern_count',
            'markov_bullish', 'markov_bearish', 'markov_sideways',
            'bayesian_sentiment', 'gp_confidence', 'market_cluster'
        ]
        
        # Add Fibonacci features if enabled
        if self.use_fibonacci.value:
            rf_features.extend(['distance_to_fib_618', 'distance_to_fib_382'])
        
        # Add FFT feature if enabled
        if self.use_fft_filter.value:
            rf_features.append('fft_cycle_strength')
        
        # Train RF models (in backtesting mode)
        if not self.models_loaded and len(dataframe) > 200:
            self.train_random_forest(dataframe, rf_features)
            self.save_models()
            self.models_loaded = True
        
        # Get RF predictions
        dataframe['rf_long_signal'], dataframe['rf_short_signal'] = self.random_forest_predict(dataframe, rf_features)
        
        # ═══ FINAL COMPOSITE SIGNALS ═══
        
        # Combine all signals into composite score
        dataframe['composite_long_score'] = (
            dataframe['rf_long_signal'] * 0.5 +  # 50% RF
            dataframe['bayesian_sentiment'] * 0.2 +  # 20% Bayesian
            dataframe['gp_confidence'].clip(-1, 1) * 0.15 +  # 15% GP
            (dataframe['pattern_strength'] / 10).clip(-1, 1) * 0.15  # 15% Patterns
        )
        
        dataframe['composite_short_score'] = (
            dataframe['rf_short_signal'] * 0.5 +  # 50% RF
            (1 - dataframe['bayesian_sentiment']) * 0.2 +  # 20% Bayesian (inverted)
            (-dataframe['gp_confidence']).clip(-1, 1) * 0.15 +  # 15% GP (inverted)
            (-dataframe['pattern_strength'] / 10).clip(-1, 1) * 0.15  # 15% Patterns (inverted)
        )
        
        lstm_w = 0.20

        base_long = dataframe['composite_long_score'].copy()
        base_short = dataframe['composite_short_score'].copy()

        dataframe['composite_long_score'] = (
            base_long * (1 - lstm_w)
            + (dataframe['lstm_long'] * dataframe['lstm_conf']).clip(0, 1) * lstm_w
        )

        dataframe['composite_short_score'] = (
            base_short * (1 - lstm_w)
            + (dataframe['lstm_short'] * dataframe['lstm_conf']).clip(0, 1) * lstm_w
        )

        # === Optional: auto-retrain once per N hours ===
        if self.auto_retrain_enabled and len(dataframe) > self.lstm_min_fit_bars:
            try:
                now_ts = float(pd.Timestamp(dataframe.index[-1]).timestamp())
                need_secs = float(self.auto_retrain_hours * 3600)
                last_ts = getattr(self, "_auto_last_retrain_ts", 0.0)
                if (now_ts - last_ts) >= need_secs:
                    # retrain LSTM (independent of lstm_train_on_backtest flag)
                    lstm_ok = False
                    if self.enable_lstm:
                        prev_flag = self.lstm_train_on_backtest
                        try:
                            self.lstm_train_on_backtest = True
                            lstm_ok = self.train_lstm(dataframe.copy())
                        finally:
                            self.lstm_train_on_backtest = prev_flag
                    # retrain RF using existing rf_features from this method
                    self.train_random_forest(dataframe, rf_features)
                    self.save_models()
                    self._auto_last_retrain_ts = now_ts
                    self._save_meta(now_ts)
                    self._send_tg(f"🧠 Auto-retrained models at {pd.to_datetime(now_ts, unit='s').isoformat()}  LSTM={'OK' if lstm_ok else 'skip'}  RF=OK")
            except Exception as e:
                logger.warning(f"[AUTO] retrain error: {e}")

        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry signals based on Phase 3 Random Forest decisions
        """
        
        # Initialize entry columns
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        # Confidence plumbing (0..1)
        rf_long = dataframe['rf_long_signal'].clip(0.0, 1.0).fillna(0.5)
        rf_short = dataframe['rf_short_signal'].clip(0.0, 1.0).fillna(0.5)
        comp_long = dataframe['composite_long_score'].clip(0.0, 1.0).fillna(0.0)
        comp_short = dataframe['composite_short_score'].clip(0.0, 1.0).fillna(0.0)
        lstm_long = (dataframe['lstm_long'] * dataframe['lstm_conf']).fillna(0.0)
        lstm_short = (dataframe['lstm_short'] * dataframe['lstm_conf']).fillna(0.0)
        dataframe['conf_long'] = (0.5 * rf_long + 0.3 * comp_long + 0.2 * lstm_long).clip(0.0, 1.0)
        dataframe['conf_short'] = (0.5 * rf_short + 0.3 * comp_short + 0.2 * lstm_short).clip(0.0, 1.0)

        # ═══ LONG ENTRY CONDITIONS - RF DECISION BASED ═══
        
        # Phase 3: Random Forest makes the decision
        # Entry when RF and composite models agree
        dataframe.loc[
            (
                # Primary: Moderate composite score from all models (relaxed)
                (dataframe['composite_long_score'] > 0.25)
                # Secondary: Either strong RF signal OR moderate signal with RSI confirmation
                & (
                    (dataframe['rf_long_signal'] > 0.5)  # Strong RF signal alone
                    | (
                        (dataframe['rf_long_signal'] > 0.25)  # Moderate RF signal (relaxed)
                        & (dataframe['rsi'] < 45)  # with RSI support
                    )
                )
                # Tertiary: Decent volume
                & (dataframe['volume_ratio'] > 0.6)
            ),
            'enter_long'
        ] = 1
        
        # ═══ SHORT ENTRY CONDITIONS - RF DECISION BASED ═══
        
        # Phase 3: Random Forest makes the decision
        # Entry when RF and composite models agree
        dataframe.loc[
            (
                # Primary: Moderate composite score from all models (relaxed)
                (dataframe['composite_short_score'] > 0.25)
                # Secondary: Either strong RF signal OR moderate signal with RSI confirmation
                & (
                    (dataframe['rf_short_signal'] > 0.5)  # Strong RF signal alone
                    | (
                        (dataframe['rf_short_signal'] > 0.25)  # Moderate RF signal (relaxed)
                        & (dataframe['rsi'] > 55)  # with RSI support
                    )
                )
                # Tertiary: Decent volume
                & (dataframe['volume_ratio'] > 0.6)
            ),
            'enter_short'
        ] = 1
        
        pair = metadata.get('pair', 'UNKNOWN')
        if self.debug_alerts and len(dataframe) > 0:
            last = dataframe.iloc[-1]

            comp_thr  = 0.25
            rf_strong = 0.50
            rf_mid    = 0.25
            vol_thr   = 0.60
            rsi_long_max  = 45
            rsi_short_min = 55

            # recompute booleans on the last row
            cond_comp_long = last['composite_long_score'] > comp_thr
            cond_rf_long_strong = last['rf_long_signal'] > rf_strong
            cond_rf_long_mid    = (last['rf_long_signal'] > rf_mid) and (last['rsi'] < rsi_long_max)
            cond_vol_ok         = last['volume_ratio'] > vol_thr

            cond_comp_short = last['composite_short_score'] > comp_thr
            cond_rf_short_strong = last['rf_short_signal'] > rf_strong
            cond_rf_short_mid    = (last['rf_short_signal'] > rf_mid) and (last['rsi'] > rsi_short_min)
            cond_vol_ok_s        = last['volume_ratio'] > vol_thr

            long_ok  = cond_comp_long and (cond_rf_long_strong or cond_rf_long_mid) and cond_vol_ok
            short_ok = cond_comp_short and (cond_rf_short_strong or cond_rf_short_mid) and cond_vol_ok_s

            if self._can_alert(pair):
                if long_ok:
                    self._send_tg(f"✅ RAW LONG accepted [{pair}] comp={last['composite_long_score']:.3f} rf={last['rf_long_signal']:.3f} rsi={last['rsi']:.1f} volr={last['volume_ratio']:.2f} lstm={last.get('lstm_long',0.5):.2f}/{last.get('lstm_conf',0.0):.2f}")
                else:
                    reasons = []
                    if not cond_comp_long:
                        reasons.append(f"comp≤{comp_thr:.2f}({last['composite_long_score']:.3f})")
                    if not (cond_rf_long_strong or cond_rf_long_mid):
                        reasons.append(f"rf weak({last['rf_long_signal']:.3f}, rsi {last['rsi']:.1f})")
                    if not cond_vol_ok:
                        reasons.append(f"volr {last['volume_ratio']:.2f}")
                    self._send_tg(f"❌ RAW LONG rejected [{pair}] {', '.join(reasons) or '—'}")

                if short_ok:
                    self._send_tg(f"✅ RAW SHORT accepted [{pair}] comp={last['composite_short_score']:.3f} rf={last['rf_short_signal']:.3f} rsi={last['rsi']:.1f} volr={last['volume_ratio']:.2f} lstm={last.get('lstm_short',0.5):.2f}/{last.get('lstm_conf',0.0):.2f}")
                else:
                    reasons = []
                    if not cond_comp_short:
                        reasons.append(f"comp≤{comp_thr:.2f}({last['composite_short_score']:.3f})")
                    if not (cond_rf_short_strong or cond_rf_short_mid):
                        reasons.append(f"rf weak({last['rf_short_signal']:.3f}, rsi {last['rsi']:.1f})")
                    if not cond_vol_ok_s:
                        reasons.append(f"volr {last['volume_ratio']:.2f}")
                    self._send_tg(f"❌ RAW SHORT rejected [{pair}] {', '.join(reasons) or '—'}")

        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit signals
        """
        
        # Initialize exit columns
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        # ═══ LONG EXIT CONDITIONS - STRICT ═══
        
        # Exit only when multiple conditions confirm trend reversal
        dataframe.loc[
            (
                # Exit when composite score turns clearly negative (strict)
                (dataframe['composite_long_score'] < 0.2)
                # AND price is overbought (strict)
                & (dataframe['rsi'] > 75)
                # AND RF confirms exit
                & (dataframe['rf_long_signal'] < 0.4)
                # Confirm with volume
                & (dataframe['volume_ratio'] > 0.7)
            ),
            'exit_long'
        ] = 1
        
        # ═══ SHORT EXIT CONDITIONS - STRICT ═══
        
        # Exit only when multiple conditions confirm trend reversal
        dataframe.loc[
            (
                # Exit when composite score turns clearly negative (strict)
                (dataframe['composite_short_score'] < 0.2)
                # AND price is oversold (strict)
                & (dataframe['rsi'] < 25)
                # AND RF confirms exit
                & (dataframe['rf_short_signal'] < 0.4)
                # Confirm with volume
                & (dataframe['volume_ratio'] > 0.7)
            ),
            'exit_short'
        ] = 1
        
        return dataframe
    

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str,
                            side: str, **kwargs) -> bool:
        side_key = (side or 'long').lower()
        self._refresh_pair_guardrails(pair, current_time)
        _, row = self._get_latest_row(pair)
        if row is None:
            return True
        adjustments = self._pair_adjustments.get(pair, {'atr_factor': 1.0, 'tp_factor': 1.0, 'min_conf': 0.4})
        conf_key = 'conf_long' if side_key == 'long' else 'conf_short'
        conf = self._safe_float(row.get(conf_key), 0.5)
        adx_val = self._safe_float(row.get('adx'), 0.0)
        vol_ratio = self._safe_float(row.get('volume_ratio'), 0.0)
        min_conf = max(0.4, adjustments.get('min_conf', 0.5))

        payload_base = {
            'pair': pair,
            'side': side_key,
            'pnl': '0.00%',
            'conf': f"{conf:.2f}",
            'adx': f"{adx_val:.1f}",
            'vol': f"{vol_ratio:.2f}"
        }
        flip_req = self._pair_flip_requests.pop(pair, None)
        if isinstance(flip_req, dict):
            # Allow the first opposite entry unconditionally (or with relaxed min_conf)
            if flip_req.get('side') == side_key:
                self.notify('ENTRY_ACCEPT', dict(payload_base, action='flip_entry'))
                self._pending_entry[pair] = {
                    'entry_conf': conf, 'side': side_key, 'adx': adx_val, 'vol_ratio': vol_ratio,
                    'atr_mult': self.atr_k, 'trail_mult': self.trail_k,
                    'tp_target': self.tp_mult, 'min_conf': 0.5, 'timestamp': current_time.isoformat(),
                    'max_leverage': None,
                }
                return True
        if conf < min_conf:
            reject_payload = dict(payload_base, action='entry_reject')
            self.notify('REJECT_low_conf', reject_payload)
            logger.info("Entry reject %s %s reason=low_conf conf=%.2f adx=%.1f vol=%.2f", pair, side_key, conf, adx_val, vol_ratio)
            return False

        if adx_val < self.adx_min:
            reject_payload = dict(payload_base, action='entry_reject')
            self.notify('REJECT_adx_low', reject_payload)
            logger.info("Entry reject %s %s reason=adx_low conf=%.2f adx=%.1f vol=%.2f", pair, side_key, conf, adx_val, vol_ratio)
            return False

        entry_state = {
            'entry_conf': conf,
            'side': side_key,
            'adx': adx_val,
            'vol_ratio': vol_ratio,
            'atr_mult': self.atr_k * adjustments.get('atr_factor', 1.0),
            'trail_mult': self.trail_k,
            'tp_target': self.tp_mult * adjustments.get('tp_factor', 1.0),
            'min_conf': min_conf,
            'timestamp': current_time.isoformat()
        }

        if vol_ratio > 1.8:
            entry_state['atr_mult'] *= 0.75
            entry_state['max_leverage'] = 1.0
            entry_state['vol_spike'] = True
            spike_payload = dict(payload_base, action='vol_cap')
            self.notify('VOL_SPIKE', spike_payload)
        else:
            entry_state['max_leverage'] = None

        self._pending_entry[pair] = entry_state
        accept_payload = dict(payload_base, action='entry_accept')
        self.notify('ENTRY_ACCEPT', accept_payload)
        logger.info("Entry accept %s %s conf=%.2f adx=%.1f vol=%.2f", pair, side_key, conf, adx_val, vol_ratio)
        return True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float | None:
        self._refresh_pair_guardrails(pair, current_time)
        self._ti_ensure(trade, pair)
        info = self._ti_get(trade)
        _, row = self._get_latest_row(pair)
        conf = 0.5
        opp_conf = 0.5
        atr = trade.open_rate * 0.01
        vol_ratio = 0.0
        if row is not None:
            conf = self._safe_float(row.get('conf_long' if not trade.is_short else 'conf_short'), 0.5)
            opp_conf = self._safe_float(row.get('conf_short' if not trade.is_short else 'conf_long'), 0.5)
            atr = max(self._safe_float(row.get('atr'), atr), 1e-9)
            vol_ratio = self._safe_float(row.get('volume_ratio'), 0.0)
        atr_mult = info.get('atr_mult', self.atr_k)
        trail_mult = info.get('trail_mult', self.trail_k)
        open_rate = trade.open_rate
        atr_r = (atr_mult * atr) / open_rate if open_rate else 0.0
        if atr_r <= 0:
            atr_r = 1e-6
        is_short = trade.is_short
        stop_price = info.get('stop_price')
        if stop_price is None:
            stop_price = open_rate + atr_mult * atr if is_short else open_rate - atr_mult * atr
            info['stop_price'] = stop_price
            info['initial_sl_set'] = True

        if current_profit >= atr_r and not info.get('move_to_be'):
            info['stop_price'] = open_rate
            info['move_to_be'] = True
            self.notify('MOVE_BE', {
                'pair': pair,
                'side': 'short' if is_short else 'long',
                'action': 'move_be',
                'pnl': f"{current_profit*100:.2f}%",
                'conf': f"{conf:.2f}",
                'vol': f"{vol_ratio:.2f}"
            })

        if current_profit >= 1.5 * atr_r:
            desired = (current_rate + trail_mult * atr) if is_short else (current_rate - trail_mult * atr)
            current_stop = info.get('stop_price', desired)
            updated = False
            if is_short and desired < current_stop:
                info['stop_price'] = desired
                updated = True
            elif not is_short and desired > current_stop:
                info['stop_price'] = desired
                updated = True
            if updated:
                info['trail_active'] = True
                self.notify('MOVE_TRAIL', {
                    'pair': pair,
                    'side': 'short' if is_short else 'long',
                    'action': 'trail',
                    'pnl': f"{current_profit*100:.2f}%",
                    'conf': f"{conf:.2f}",
                    'vol': f"{vol_ratio:.2f}"
                })

        if self.enable_repair and current_profit <= self.reduce_dd and conf < self.reduce_conf and not info.get('reduced'):
            reduced = False
            try:
                partial = getattr(trade, 'partial_exit', None)
                if callable(partial):
                    partial(amount=trade.amount / 2, rate=current_rate, reason='REPAIR_reduce')
                    reduced = True
            except Exception as exc:
                info['force_exit_reason'] = 'repair_reduce'
                info['force_exit_note'] = str(exc)
            if reduced:
                info['reduced'] = True
                self.notify('REPAIR_reduce', {
                    'pair': pair,
                    'side': 'short' if is_short else 'long',
                    'action': 'reduce50',
                    'pnl': f"{current_profit*100:.2f}%",
                    'conf': f"{conf:.2f}",
                    'vol': f"{vol_ratio:.2f}"
                })
            else:
                info['force_exit_reason'] = info.get('force_exit_reason', 'repair_reduce')
                self.notify('REPAIR_reduce', {
                    'pair': pair,
                    'side': 'short' if is_short else 'long',
                    'action': 'reduce_fail',
                    'pnl': f"{current_profit*100:.2f}%",
                    'conf': f"{conf:.2f}",
                    'vol': f"{vol_ratio:.2f}"
                })

        stop_price = info.get('stop_price', open_rate)
        self._ti_set(trade, info)
        if is_short:
            return (open_rate - stop_price) / open_rate
        return (stop_price - open_rate) / open_rate

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        self._refresh_pair_guardrails(pair, current_time)
        self._ti_ensure(trade, pair)
        info = self._ti_get(trade)
        _, row = self._get_latest_row(pair)
        conf = 0.5
        opp_conf = 0.5
        atr = trade.open_rate * 0.01
        if row is not None:
            conf = self._safe_float(row.get('conf_long' if not trade.is_short else 'conf_short'), 0.5)
            opp_conf = self._safe_float(row.get('conf_short' if not trade.is_short else 'conf_long'), 0.5)
            atr = max(self._safe_float(row.get('atr'), atr), 1e-9)
        atr_mult = info.get('atr_mult', self.atr_k)
        atr_r = (atr_mult * atr) / trade.open_rate if trade.open_rate else 0.0
        reason = info.pop('force_exit_reason', None)
        side_key = 'short' if trade.is_short else 'long'
        if reason:
            self.notify(reason.upper(), {
                'pair': pair,
                'side': side_key,
                'action': 'exit_force',
                'pnl': f"{current_profit*100:.2f}%",
                'conf': f"{conf:.2f}"
            })
            if not info.get('history_updated'):
                self._update_pair_history(pair, trade, current_rate)
                info['history_updated'] = True
            self._ti_set(trade, info)
            self._ti_clear(trade)
            return reason
        tp_target = info.get('tp_target')
        if tp_target and atr > 0 and current_profit >= tp_target * (atr / trade.open_rate):
            self.notify('REPAIR_tp', {
                'pair': pair,
                'side': side_key,
                'action': 'exit_tp',
                'pnl': f"{current_profit*100:.2f}%",
                'conf': f"{conf:.2f}"
            })
            if not info.get('history_updated'):
                self._update_pair_history(pair, trade, current_rate)
                info['history_updated'] = True
            self._ti_set(trade, info)
            self._ti_clear(trade)
            return 'repair_tp'
        bars = self._bars_in_trade(trade, current_time)
        if self.enable_repair and bars >= self.time_exit_candles and 0.45 <= conf <= 0.55:
            self.notify('REPAIR_time_exit', {
                'pair': pair,
                'side': side_key,
                'action': 'exit_time',
                'pnl': f"{current_profit*100:.2f}%",
                'conf': f"{conf:.2f}",
                'r': bars
            })
            if not info.get('history_updated'):
                self._update_pair_history(pair, trade, current_rate)
                info['history_updated'] = True
            self._ti_set(trade, info)
            self._ti_clear(trade)
            return 'repair_time_exit'
        if self.enable_repair and bars >= self.flip_cooldown_candles and opp_conf > self.flip_conf:
            flip_side = 'long' if trade.is_short else 'short'
            self.notify('REPAIR_flip', {
                'pair': pair,
                'side': flip_side,
                'action': 'flip',
                'pnl': f"{current_profit*100:.2f}%",
                'conf': f"{opp_conf:.2f}",
                'r': bars
            })
            self._pair_flip_requests[pair] = {'side': flip_side, 'timestamp': current_time.isoformat()}
            if not info.get('history_updated'):
                self._update_pair_history(pair, trade, current_rate)
                info['history_updated'] = True
            self._ti_set(trade, info)
            self._ti_clear(trade)
            return 'repair_flip'
        self._ti_set(trade, info)
        # self._ti_clear(trade)
        return None
    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ):
        """
        DCA logic: When a position is in drawdown and the model still supports the same direction,
        add to the position at predefined triggers. Returns additional stake (quote) or 0/None.
        """
        # Disabled / no levels
        if not getattr(self, 'dca_enabled', False):
            return 0

        # Safety: respect global max DCAs from config + our own triggers
        max_adj_cfg = int(getattr(self, 'max_entry_position_adjustment', 0) or 0)
        max_levels = min(max_adj_cfg, len(getattr(self, 'dca_triggers', []) or []))
        if max_levels <= 0:
            return self._dca_dbg(trade, 'low_conf', current_profit, same_conf, opp_conf)
            return 0

        # Ensure per-trade info
        pair = getattr(trade, 'pair', 'UNKNOWN')
        self._ti_ensure(trade, pair)
        info = self._ti_get(trade)

        # Current bar index for cooldown math
        # Uses analyzed dataframe index if available, otherwise fall back to minutes
        bars_open = self._bars_in_trade(trade, current_time)

        # Cooldown between DCAs
        last_bar = int(info.get('dca_last_bar', -999_999))
        if (bars_open - last_bar) < int(getattr(self, 'dca_cooldown_candles', 1)):
            return self._dca_dbg(trade, 'low_conf', current_profit, same_conf, opp_conf)

            return 0

        # How many DCAs already placed?
        dca_count = int(info.get('dca_count', 0))
        if dca_count >= max_levels:
            return self._dca_dbg(trade, 'low_conf', current_profit, same_conf, opp_conf)
            return 0

        # Model confidence must still support the SAME side
        _, row = self._get_latest_row(pair)
        if row is None:
            return 0

        # Read confidence streams
        same_conf = self._safe_float(row.get('conf_short' if trade.is_short else 'conf_long'), 0.5)
        opp_conf  = self._safe_float(row.get('conf_long' if trade.is_short else 'conf_short'), 0.5)

        # Require minimum confidence and not clearly flipping
        min_conf = float(getattr(self, 'dca_min_conf', 0.52))
        if same_conf < min_conf or opp_conf > max(0.60, self.flip_conf):
            return 0

        # Check if current drawdown reached next DCA trigger
        triggers = list(getattr(self, 'dca_triggers', [-0.02]))
        mults    = list(getattr(self, 'dca_multipliers', [1.0]))
        # Clamp to available slots
        triggers = triggers[:max_levels]
        mults    = (mults + [mults[-1]])[:max_levels]

        next_trigger = float(triggers[dca_count])  # e.g. -0.02, -0.05
        if current_profit > next_trigger:
            return 0

        # Compute additional stake:
        # - base it on the original stake (trade.stake_amount)
        # - scale by configured multiplier for this DCA level
        base_stake = float(getattr(trade, 'stake_amount', 0.0) or 0.0)
        if base_stake <= 0:
            return 0

        add_stake = base_stake * float(mults[dca_count])

        # Optional guardrails: cap by available wallet or exchange max stake if provided in kwargs
        max_stake = float(kwargs.get('max_stake', add_stake))
        add_stake = float(min(add_stake, max_stake))

        if add_stake <= 0:
            return 0

        # Record and notify
        info['dca_count'] = dca_count + 1
        info['dca_last_bar'] = bars_open
        self._ti_set(trade, info)

        self.notify('DCA_ADD', {
            'pair': pair,
            'side': 'short' if trade.is_short else 'long',
            'action': f"dca_{dca_count+1}",
            'pnl': f"{current_profit*100:.2f}%",
            'conf': f"{same_conf:.2f}",
            'rate': f"{current_rate:.6f}",
            'add_stake': f"{add_stake:.2f}"
        })

        return add_stake

    def leverage(self, pair: str, current_time, current_rate: float,
            proposed_leverage: float, max_leverage: float, entry_tag: str | None,
            side: str, **kwargs) -> float:
        """
        Prefer static leverage from config if present; otherwise Kelly-derived,
        rounded UP to an integer, capped to exchange limit and 5x.
        Telegram a compact line for traceability.
        """
        _, row = self._get_latest_row(pair)
        vol_ratio = self._safe_float(row.get('volume_ratio'), 0.0) if row is not None else 0.0
        caps: list[float] = []
        pending = self._pending_entry.get(pair)
        if isinstance(pending, dict) and pending.get('max_leverage') is not None:
            caps.append(self._safe_float(pending.get('max_leverage'), 1.0))
        if vol_ratio > 1.6:
            caps.append(1.0)

        static = int(self.config.get('futures_leverage', 0) or 0)
        if static >= 2:
            ret = float(min(static, int(max_leverage), 5))
            if caps:
                ret = min(ret, min(caps))
            try:
                self._send_tg(f"[LEV] {pair} static={static} max={max_leverage} vol={vol_ratio:.2f} -> {ret}x")
            except Exception:
                pass
            return ret

        if not self.use_kelly.value:
            ret = 1.0
        else:
            estimated_win_rate = 0.55
            estimated_avg_win = 0.03
            estimated_avg_loss = 0.02
            kelly_frac = self.calculate_kelly_criterion(estimated_win_rate, estimated_avg_win, estimated_avg_loss)
            dynamic = 1.0 + (kelly_frac * 8.0)
            import math
            ret = float(min(math.ceil(dynamic), int(max_leverage), 5))

        if caps:
            ret = min(ret, min(caps))
        try:
            self._send_tg(f"[LEV] {pair} final={ret}x vol={vol_ratio:.2f} max={max_leverage}")
        except Exception:
            pass
        return ret

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                        proposed_stake: float, min_stake: float | None, max_stake: float,
                        leverage: float, entry_tag: str | None, side: str,
                        **kwargs) -> float:
        import math
        price = max(float(current_rate or 0.0), 1e-9)
        lev   = max(float(leverage or 1.0), 1.0)

        # start from proposed stake
        stake = float(proposed_stake or 0.0)

        # optional Kelly (unchanged)
        if self.use_kelly.value:
            est_win_rate, est_avg_win, est_avg_loss = 0.55, 0.03, 0.02
            kelly_fraction = self.calculate_kelly_criterion(est_win_rate, est_avg_win, est_avg_loss)
            base = float(self.kelly_fraction.value or 0.0)
            if base > 0:
                stake *= (kelly_fraction / base)

        # --- read exchange limits / precision
        m = {}
        try:
            m = (self.dp.market(pair) or {})  # type: ignore[attr-defined]
        except Exception:
            m = {}

        limits   = m.get('limits', {}) if isinstance(m, dict) else {}
        amt_lim  = limits.get('amount', {}) if isinstance(limits, dict) else {}
        cost_lim = limits.get('cost',   {}) if isinstance(limits, dict) else {}
        prec     = (m.get('precision', {}) or {}).get('amount')  # decimals, e.g. 0,1,2,...

        min_qty   = float(amt_lim.get('min') or 0.0)     # e.g. CAKE often 1.0 on some contracts
        step_qty  = float(amt_lim.get('step') or 0.0)    # may be 0.0/missing on some feeds
        min_cost  = float(cost_lim.get('min') or 5.0)    # Binance notional rule (~5 USDT)

        # --- qty implied by current stake
        qty = (stake * lev) / price

        # --- qty required by rules
        req_qty_cost = (min_cost / price)         # to pass notional
        req_qty_min  = min_qty                    # to pass min lot
        req_qty_raw  = max(qty, req_qty_cost, req_qty_min)

        # --- determine EFFECTIVE step to round UP
        eff_step = 0.0
        if step_qty and step_qty > 0:
            eff_step = step_qty
        elif isinstance(prec, int) and prec >= 0:
            # exchange defines decimal precision -> step = 10^-precision
            eff_step = 10.0 ** (-prec)
        elif min_qty >= 1.0:
            # when precision missing and lots are whole integers
            eff_step = 1.0
        else:
            # last resort
            eff_step = 1e-6

        # round UP so final qty cannot be rounded down by exchange
        req_qty = math.ceil(req_qty_raw / eff_step) * eff_step

        # back-solve stake from required qty
        required_stake = (req_qty * price) / lev

        # apply caps
        if min_stake is not None:
            required_stake = max(required_stake, float(min_stake))
        stake = min(max(stake, required_stake), float(max_stake))

        logger.info(
            f"[AUTO-STAKE] {pair} price={price:.6f} lev={lev:.2f} "
            f"min_qty={min_qty} step={step_qty} prec={prec} min_cost={min_cost} "
            f"qty_raw={req_qty_raw:.6f} eff_step={eff_step} -> qty≥{req_qty:.6f} stake→{stake:.6f}"
        )
        try:
            self._send_tg(f"[AUTO-STAKE] {pair} qty≥{req_qty:.6f} (step {eff_step}, min_qty {min_qty}) stake→{stake:.3f}")
        except Exception:
            pass

        return float(stake)





