# pragma pylint: disable=missing-docstring, invalid-name, too-many-arguments, too-many-positional-arguments
"""
RSI_Futures — 震荡(range) + 趋势(trend)，独立平仓 + 状态锁（同 K 禁止第二笔不同模式）。

- 4h RSI(14) merge 到 15m；15m EMA20/50/100/200。
- ROE：以 current_profit 为收益率代理，custom_data 记峰值；开仓满 roe_trailing_grace_hours 后才启用移动止盈（此前仍累计峰值）。
- 防对冲：趋势多成立时禁震荡空，趋势空成立时禁震荡多。
- 实盘单币单仓限制仍在；回测叠仓需 position_stacking: true。
"""
from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy
from freqtrade.strategy.parameters import DecimalParameter, IntParameter
from freqtrade.strategy.strategy_helper import merge_informative_pair

logger = logging.getLogger(__name__)


class RSI_Futures(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short = True
    process_only_new_candles = True

    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # 允许：趋势盈利加仓 + 分批止盈（部分减仓）
    position_adjustment_enable = True
    max_entry_position_adjustment = 2
    max_dca_multiplier = 2.0

    minimal_roi = {"0": 10000}
    # 硬止损：避免长时间扛单导致爆仓/清算
    stoploss = -0.15

    startup_candle_count = 400

    cooldown_hours_after_exit = 4
    leverage_target = 5.0
    leverage_max = 3.0  # 杠杆硬上限（优先保证不爆仓）

    # =========================
    # 分类器优化模式（控制变量）
    # - True：仅优化市场分类逻辑；关闭/旁路其它动态系统，避免 hyperopt 掩盖分类错误
    # - False：正常完整策略（V3）
    # =========================
    classification_opt_only = True

    # --- 最大持仓时间（15m）---
    # 时间止损按 market_state 分层：trend > range > mixed
    max_hold_bars = 192  # fallback：≈ 2天
    max_hold_bars_trend = 192
    max_hold_bars_range = 96
    max_hold_bars_mixed = 72
    # 低波动不允许长期占用仓位（atr_ratio 很低时更快强制退出）
    low_vol_hold_atr_ratio = 0.004
    low_vol_max_hold_bars = 96  # ≈ 1天
    # 盈利保护时间退出：持仓超过一定时间但收益不达标 -> 强制退出
    time_profit_fail_bars = 96
    time_profit_fail_min_profit = 0.02
    time_profit_fail_trend_cutoff = 0.5

    # range 质量过滤：EMA20/EMA50 发生交叉后，震荡信号冷却（避免“弱趋势/刚启动趋势”段被吃掉）
    range_ema_cross_cooldown_bars = 8

    # 独立通道阈值（不再用 strength 融合体系的 0.6）
    trend_entry_min_score_base = 0.55
    range_entry_min_score_base = 0.35

    # --- V3.5：自适应市场分类器 / 路由权重 ---
    # 阈值由滚动分位数生成，不再依赖固定阈值 hyperopt
    adaptive_classifier_window = 200
    adaptive_trend_quantile = 0.7
    adaptive_range_quantile = 0.3
    adaptive_atr_quantile = 0.5
    adaptive_threshold_smooth_bars = 10
    adaptive_trend_th_clip = (0.4, 0.8)
    adaptive_range_th_clip = (0.2, 0.6)
    adaptive_persist_bars = 5
    adaptive_persist_min_hits = 3

    # 路由权重（先固定，避免同时优化干扰分类器评估）
    regime_mixed_weight = 0.60
    regime_low_weight = 0.20
    regime_mixed_entry_tighten = 0.05  # mixed 区额外收紧（降频）
    # --- 进化代理：结构权重自适应 ---
    evolve_enabled = True
    evolve_update_every_n_trades = 80
    evolve_min_samples = 30
    evolve_max_weight_step = 0.20
    evolve_weight_min = 0.30
    evolve_weight_max = 2.00
    evolve_disable_score = 0.0
    evolve_enable_weight_floor = 0.50
    heatmap_enabled = True
    heatmap_min_cell_samples = 20
    heatmap_disable_profit_threshold = 0.0

    # --- 实盘执行：滑点建模 / 挂单回退 / 成交质量日志 ---
    use_custom_entry_price = True
    use_custom_exit_price = True
    exec_use_orderbook = False
    exec_orderbook_levels = 5
    exec_max_spread_bps = 12.0          # 价差过大 -> 回退为更激进价格（更易成交）
    exec_min_top_qty = 0.0              # 顶层挂单量过滤（0=不启用；单位按交易所盘口数量）
    exec_maker_offset_bps = 1.0         # maker 贴近盘口的偏移（long: bid*(1+off), short: ask*(1-off)）
    exec_aggressive_slip_bps = 6.0      # 回退时的“吃单/滑点”基准（bps）
    exec_slip_vol_k = 800.0             # 波动率(atr_ratio) 转滑点 bps 的系数：slip += atr_ratio * k
    exec_log_every = True               # 是否每次定价都写日志（实盘建议 True，回测可关）

    # 开仓后未满此时长（小时）不执行 ROE 移动止盈判断；结构止损/RSI/趋势保本等仍照常
    roe_trailing_grace_hours = 1.0

    # 震荡：峰值 ROE >0.05 用回撤 0.20；>0.10 用回撤 0.30
    range_roe_trail_stages: List[Tuple[float, float]] = [(0.05, 0.20), (0.10, 0.30)]
    # 趋势：峰值 ROE >0.10 用回撤 0.20；>0.20 用回撤 0.30
    trend_roe_trail_stages: List[Tuple[float, float]] = [(0.10, 0.20), (0.20, 0.30)]

    # --- 融合权重/阈值 ---
    trend_score_smooth_bars = 4  # 3~5根平滑以减少抖动
    trend_score_range_block = 0.7  # 趋势过强时禁止震荡方向开仓
    entry_strength_min = DecimalParameter(
        0.50, 0.75, default=0.60, decimals=2, space="buy", optimize=False
    )  # 做多/做空强度阈值
    rsi_trend_base = 50.0  # trend_weight 对应的 RSI 基准阈值

    # --- 动态仓位上限 ---
    stake_max_mult = 1.5  # position_size = min(base*(0.5+trend_score), base*stake_max_mult)

    # --- 趋势加仓（顺势金字塔）---
    trend_add_min_score_1 = DecimalParameter(
        0.50, 0.90, default=0.60, decimals=2, space="buy", optimize=False
    )
    trend_add_profit_1 = DecimalParameter(
        0.02, 0.10, default=0.03, decimals=3, space="sell", optimize=False
    )
    trend_add_min_score_2 = DecimalParameter(
        0.50, 0.90, default=0.80, decimals=2, space="buy", optimize=False
    )
    trend_add_profit_2 = DecimalParameter(
        0.02, 0.10, default=0.08, decimals=3, space="sell", optimize=False
    )
    trend_add_size_mult = 0.3  # 每次加仓 0.3 * base_position
    trend_add_max_times = 2
    trend_total_max_mult = 2.0  # 总仓位 <= 2.0 * base_position

    # --- 分批止盈 ---
    tp1_ratio_default = DecimalParameter(
        0.20, 0.50, default=0.30, decimals=2, space="sell", optimize=False
    )
    tp2_ratio_default = DecimalParameter(
        0.20, 0.50, default=0.30, decimals=2, space="sell", optimize=False
    )
    tp1_ratio_trend_strong = 0.20  # trend_score>0.7 时 TP1 比例降到 20%
    tp1_ratio_range_strong = 0.50  # range_score>0.7 时 TP1 直接锁 50%

    # --- 连续亏损风控 ---
    loss_streak_step = 0.2         # 每亏1单，risk_factor 下降0.2
    loss_streak_floor = 0.3        # risk_factor 下限
    loss_entry_step = 0.1          # 每亏1单，入场强度阈值 +0.1
    loss_entry_cap = 0.3           # 入场强度最多上调 0.3
    loss_rsi_tighten_step = 2.0    # 每亏1单，RSI阈值收紧2点
    loss_cooldown_streak = 3       # 连亏达到3触发冷却
    loss_cooldown_candles = 12     # 15m下约3h（可改10~20）
    strict_risk_trend_cutoff = 0.5 # 仅 trend_score<0.5 启用严格风控冷却

    # --- 波动率(ATR)驱动过滤 ---
    atr_period = 14
    atr_ratio_min_open = DecimalParameter(
        0.002, 0.005, default=0.003, decimals=4, space="protection", optimize=False
    )  # 低波动阈值
    atr_ratio_high = DecimalParameter(
        0.008, 0.02, default=0.01, decimals=4, space="protection", optimize=False
    )  # 高波动阈值
    atr_entry_relax = 0.05          # 高波动时入场强度阈值下调
    atr_rsi_relax = 2.0             # 高波动时 RSI 条件放宽
    atr_rsi_tighten = 2.0           # 低波动时 RSI 条件收紧
    target_atr_ratio = DecimalParameter(
        0.004, 0.01, default=0.006, decimals=4, space="protection", optimize=False
    )  # 仓位波动目标
    atr_pos_min_mult = 0.5
    atr_pos_max_mult = 1.2          # 极端波动上限保护
    atr_trail_scale = 2.0           # 止盈参数随波动放大系数

    # --- 动态止盈（可选优化项，默认不参与 hyperopt）---
    dyn_start_range = DecimalParameter(
        0.03, 0.08, default=0.05, decimals=3, space="sell", optimize=False
    )
    dyn_start_trend = DecimalParameter(
        0.10, 0.20, default=0.15, decimals=3, space="sell", optimize=False
    )
    dyn_trail_range = DecimalParameter(
        0.15, 0.30, default=0.20, decimals=3, space="sell", optimize=False
    )
    dyn_trail_trend = DecimalParameter(
        0.25, 0.50, default=0.35, decimals=3, space="sell", optimize=False
    )

    # --- 市场级ATR(BTC)总仓位控制 ---
    market_vol_enabled = True
    market_vol_low = 0.003          # BTC atr_ratio 低于0.3% -> 缩仓
    market_vol_high = 0.01          # BTC atr_ratio 高于1.0% -> 放大
    market_exposure_low_mult = 0.5  # 低波动全局降至50%
    market_exposure_high_mult = 1.2 # 高波动全局升至120%
    market_bear_long_mult = 0.5     # BTC EMA50<EMA200 时，多头仓位降至50%
    market_bear_disable_trend_long = True  # 熊市可禁做 trend long

    # --- Hyperopt / 可调参数 ---
    # 阶段1：仅市场分类器（配合 classification_opt_only=True）
    div_threshold = DecimalParameter(
        0.15, 0.6, default=0.3, decimals=2, space="buy", optimize=True
    )
    trend_bias20_50_min = DecimalParameter(
        0.35, 0.8, default=0.5, decimals=2, space="buy", optimize=True
    )

    range_rsi_long = DecimalParameter(
        25.0, 40.0, default=35.0, decimals=1, space="buy", optimize=False
    )
    range_rsi_short = DecimalParameter(
        55.0, 75.0, default=65.0, decimals=1, space="buy", optimize=False
    )

    # 震荡判定核心：三组乖离率都需满足 |bias| < threshold（即 -th ~ +th）
    range_bias_threshold = DecimalParameter(
        0.15, 0.6, default=0.3, decimals=2, space="buy", optimize=True
    )

    range_rsi_exit_long = DecimalParameter(
        65.0, 78.0, default=70.0, decimals=1, space="sell", optimize=False
    )
    range_rsi_exit_short = DecimalParameter(
        22.0, 35.0, default=30.0, decimals=1, space="sell", optimize=False
    )

    trend_hold_hours = IntParameter(1, 12, default=2, space="sell", optimize=False)

    # --- Tag 中文说明（不改变原 tag 值，仅用于日志/自定义数据可读性）---
    ENTRY_TAG_CN: Dict[str, str] = {
        "range": "震荡模式入场",
        "trend": "趋势模式入场",
    }
    ENTRY_STATE_CN: Dict[str, str] = {
        "strong_trend_long": "强趋势多头",
        "strong_trend_short": "强趋势空头",
        "pullback_long": "多头回调",
        "pullback_short": "空头回调",
        "pure_range": "纯震荡",
        "choppy": "混沌",
        "unknown": "未知状态",
    }
    ENTRY_STRUCTURE_CN: Dict[str, str] = {
        "bos": "结构突破(BOS)",
        "choch": "结构反转(CHoCH)",
        "sweep": "流动性扫单",
        "none": "无结构",
    }
    ENTRY_TRIGGER_CN: Dict[str, str] = {
        "rsi_ema_trend": "趋势触发(RSI+EMA)",
        "rsi_ema_range": "震荡触发(RSI+EMA)",
        "core": "基础触发",
    }
    # 中文 enter_tag 解析回英文键（用于策略逻辑与统计键一致）
    ENTRY_STATE_CN_REV: Dict[str, str] = {v: k for k, v in ENTRY_STATE_CN.items()}
    ENTRY_STRUCTURE_CN_REV: Dict[str, str] = {v: k for k, v in ENTRY_STRUCTURE_CN.items()}
    ENTRY_TRIGGER_CN_REV: Dict[str, str] = {v: k for k, v in ENTRY_TRIGGER_CN.items()}
    EXIT_TAG_CN: Dict[str, str] = {
        # range exits
        "range_struct_exit_long": "震荡多：结构破坏出场（EMA20 < EMA50）",
        "range_struct_exit_short": "震荡空：结构破坏出场（EMA20 > EMA50）",
        "range_rsi_exit": "震荡：RSI 反转出场（4h RSI 达到阈值）",
        "range_dynamic_lock": "震荡：动态保本/锁利润出场",
        "range_dynamic_trail": "震荡：动态 ROE 回撤止盈出场",
        # trend exits
        "trend_ema_cross": "趋势：EMA20/EMA50 反向交叉出场",
        "trend_dynamic_lock": "趋势：动态保本/锁利润出场",
        "trend_dynamic_trail": "趋势：动态 ROE 回撤止盈出场",
        # time-based exits
        "time_max_hold_exit": "时间止损：超过最大持仓时长强制出场",
        "time_low_vol_exit": "时间止损：低波动环境超过限定时长强制出场",
        "time_profit_fail_exit": "时间止损：持仓过久且收益不达标强制出场",
        "structure_invalidation_exit": "结构失效退出：趋势状态出现反向 CHoCH",
    }

    @classmethod
    def _tag_cn(cls, tag: Optional[str]) -> Optional[str]:
        if tag is None:
            return None
        t = str(tag).strip().lower()
        if not t:
            return None
        if "|" in t:
            parts = t.split("|")
            if len(parts) == 3:
                st, structure, trigger = parts
                st_cn = cls.ENTRY_STATE_CN.get(st, st)
                structure_cn = cls.ENTRY_STRUCTURE_CN.get(structure, structure)
                trigger_cn = cls.ENTRY_TRIGGER_CN.get(trigger, trigger)
                return f"{st_cn} | {structure_cn} | {trigger_cn}"
        if t in cls.ENTRY_TAG_CN:
            return cls.ENTRY_TAG_CN[t]
        if t in cls.EXIT_TAG_CN:
            return cls.EXIT_TAG_CN[t]
        return t

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._exit_cooldown_until: Dict[str, datetime] = {}
        self._entry_mode_this_candle: Dict[str, Tuple[datetime, Optional[str]]] = {}
        self._pair_loss_streak: Dict[str, int] = {}
        self._pair_loss_cooldown_until: Dict[str, datetime] = {}
        self._structure_weights: Dict[str, float] = {"bos": 1.0, "choch": 1.0, "sweep": 1.0}
        self._tag_stats: Dict[str, Dict[str, float]] = {}
        self._structure_stats: Dict[str, Dict[str, float]] = {"bos": {"score_sum": 0.0, "count": 0.0}, "choch": {"score_sum": 0.0, "count": 0.0}, "sweep": {"score_sum": 0.0, "count": 0.0}}
        self._disabled_entry_tags: set[str] = set()
        self._heatmap_stats: Dict[str, Dict[str, float]] = {}
        self._disabled_market_structure: set[str] = set()
        self._market_structure_weight: Dict[str, float] = {}
        self._evolve_seen_closed_trades = 0

    @staticmethod
    def _safe_tag_part(v: Any, fallback: str) -> str:
        if v is None:
            return fallback
        s = str(v).strip()
        return s if s else fallback

    def _compose_entry_tag(self, final_state: Any, structure: Any, trigger: Any) -> str:
        fs = self._safe_tag_part(final_state, "choppy")
        st = self._safe_tag_part(structure, "none")
        tg = self._safe_tag_part(trigger, "core")
        raw = f"{fs}|{st}|{tg}"
        cn = self._tag_cn(raw)
        return cn if cn else raw

    @classmethod
    def _parse_entry_tag(cls, entry_tag: Optional[str]) -> Tuple[str, str, str]:
        t = "" if entry_tag is None else str(entry_tag).strip()
        if not t:
            return ("unknown", "none", "core")
        parts = [p.strip() for p in t.split("|")]
        if len(parts) != 3:
            return ("unknown", "none", "core")
        a, b, c = parts

        def _norm_state(x: str) -> str:
            if x in cls.ENTRY_STATE_CN:
                return x
            return cls.ENTRY_STATE_CN_REV.get(x, x)

        def _norm_struct(x: str) -> str:
            if x in cls.ENTRY_STRUCTURE_CN:
                return x
            return cls.ENTRY_STRUCTURE_CN_REV.get(x, x)

        def _norm_trig(x: str) -> str:
            if x in cls.ENTRY_TRIGGER_CN:
                return x
            return cls.ENTRY_TRIGGER_CN_REV.get(x, x)

        return (_norm_state(a), _norm_struct(b), _norm_trig(c))

    @staticmethod
    def _market_bucket(final_state: str) -> str:
        s = str(final_state).strip().lower()
        if s.startswith("strong_trend"):
            return "strong_trend"
        if s.startswith("pullback"):
            return "pullback"
        if s == "pure_range":
            return "range"
        if s == "choppy":
            return "choppy"
        return "unknown"

    def _score_trade(self, entry_tag: Optional[str], profit_ratio: float, trade_duration_min: float) -> int:
        final_state, structure, _ = self._parse_entry_tag(entry_tag)
        score = 0
        if ("strong_trend" in final_state) and structure == "bos":
            score += 2
        if ("pullback" in final_state) and structure == "choch":
            score += 2
        if ("range" in final_state) and structure == "sweep":
            score += 2
        score += 1 if float(profit_ratio) > 0.0 else -1
        score += 1 if float(trade_duration_min) < 150.0 else -1
        return int(score)

    def _record_trade_feedback(self, entry_tag: Optional[str], profit_ratio: float, trade_duration_min: float) -> None:
        score = self._score_trade(entry_tag, profit_ratio, trade_duration_min)
        key = str(entry_tag) if entry_tag else "unknown|none|core"
        stat = self._tag_stats.get(key)
        if stat is None:
            stat = {"score_sum": 0.0, "count": 0.0, "profit_sum": 0.0, "dur_sum": 0.0}
            self._tag_stats[key] = stat
        stat["score_sum"] += float(score)
        stat["count"] += 1.0
        stat["profit_sum"] += float(profit_ratio)
        stat["dur_sum"] += float(trade_duration_min)

        _, structure, _ = self._parse_entry_tag(entry_tag)
        if structure in self._structure_stats:
            self._structure_stats[structure]["score_sum"] += float(score)
            self._structure_stats[structure]["count"] += 1.0

        # 热力图统计：market x structure
        final_state, structure2, _ = self._parse_entry_tag(entry_tag)
        market = self._market_bucket(final_state)
        cell_key = f"{market}|{structure2}"
        h = self._heatmap_stats.get(cell_key)
        if h is None:
            h = {"profit_sum": 0.0, "score_sum": 0.0, "dur_sum": 0.0, "win_sum": 0.0, "count": 0.0}
            self._heatmap_stats[cell_key] = h
        h["profit_sum"] += float(profit_ratio)
        h["score_sum"] += float(score)
        h["dur_sum"] += float(trade_duration_min)
        h["win_sum"] += 1.0 if float(profit_ratio) > 0.0 else 0.0
        h["count"] += 1.0
        self._evolve_seen_closed_trades += 1

    def _refresh_heatmap_policy(self) -> None:
        if not bool(self.heatmap_enabled):
            return

        min_cnt = float(self.heatmap_min_cell_samples)
        disable_profit = float(self.heatmap_disable_profit_threshold)
        self._disabled_market_structure.clear()
        next_weights: Dict[str, float] = {}

        for cell_key, h in self._heatmap_stats.items():
            cnt = float(h.get("count", 0.0))
            if cnt < min_cnt:
                continue
            avg_profit = float(h.get("profit_sum", 0.0)) / max(1.0, cnt)
            avg_score = float(h.get("score_sum", 0.0)) / max(1.0, cnt)
            if avg_profit < disable_profit or avg_score < 0.0:
                self._disabled_market_structure.add(cell_key)

            # 单元格权重：根据平均收益做轻度缩放（0.8~1.2）
            p = max(-0.03, min(0.03, avg_profit))
            next_weights[cell_key] = 1.0 + (p / 0.03) * 0.2

        self._market_structure_weight = next_weights

        # 输出热力图摘要（profit / winrate / count / score）
        rows = ["strong_trend", "pullback", "range", "choppy"]
        cols = ["bos", "choch", "sweep"]
        for m in rows:
            parts: List[str] = []
            for s in cols:
                k = f"{m}|{s}"
                st = self._heatmap_stats.get(k, None)
                if st is None or float(st.get("count", 0.0)) <= 0:
                    parts.append(f"{s}:n/a")
                    continue
                cnt = float(st.get("count", 0.0))
                pft = float(st.get("profit_sum", 0.0)) / max(1.0, cnt)
                wr = float(st.get("win_sum", 0.0)) / max(1.0, cnt)
                sc = float(st.get("score_sum", 0.0)) / max(1.0, cnt)
                parts.append(f"{s}:p={pft:.4f},w={wr:.2f},n={int(cnt)},s={sc:.2f}")
            logger.info("[HEATMAP] %s | %s", m, " ; ".join(parts))

    def _evolve_structure_weights(self) -> None:
        if not bool(self.evolve_enabled):
            return
        if self._evolve_seen_closed_trades <= 0:
            return
        if self._evolve_seen_closed_trades % int(self.evolve_update_every_n_trades) != 0:
            return

        min_samples = float(self.evolve_min_samples)
        max_step = float(self.evolve_max_weight_step)
        w_min = float(self.evolve_weight_min)
        w_max = float(self.evolve_weight_max)

        for structure, stat in self._structure_stats.items():
            cnt = float(stat.get("count", 0.0))
            if cnt < min_samples:
                continue
            avg_score = float(stat.get("score_sum", 0.0)) / max(1.0, cnt)
            w_old = float(self._structure_weights.get(structure, 1.0))
            if avg_score < 1.0:
                ratio = max(0.0, 1.0 - max_step)
            elif avg_score > 3.0:
                ratio = 1.0 + max_step
            else:
                ratio = 1.0
            w_new = min(w_max, max(w_min, w_old * ratio))
            self._structure_weights[structure] = w_new

        # 自动禁用弱组合（样本足够且平均分小于阈值）
        self._disabled_entry_tags.clear()
        for tag, stat in self._tag_stats.items():
            cnt = float(stat.get("count", 0.0))
            if cnt < min_samples:
                continue
            avg_score = float(stat.get("score_sum", 0.0)) / max(1.0, cnt)
            if avg_score < float(self.evolve_disable_score):
                self._disabled_entry_tags.add(tag)

        self._refresh_heatmap_policy()

        logger.info(
            "[EVOLVE] bos_weight=%.3f choch_weight=%.3f sweep_weight=%.3f disabled=%d closed=%d",
            float(self._structure_weights.get("bos", 1.0)),
            float(self._structure_weights.get("choch", 1.0)),
            float(self._structure_weights.get("sweep", 1.0)),
            len(self._disabled_entry_tags),
            self._evolve_seen_closed_trades,
        )

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(p, "1h") for p in pairs] + [(p, "4h") for p in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        for period in (20, 50, 100, 200):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period)
        dataframe["atr_ratio"] = (dataframe["atr"] / dataframe["close"]).replace([np.inf, -np.inf], np.nan)

        if not self.dp:
            return dataframe

        informative = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="4h").copy()
        informative["rsi"] = ta.RSI(informative, timeperiod=14)
        informative["ema_50"] = ta.EMA(informative, timeperiod=50)
        informative["ema_200"] = ta.EMA(informative, timeperiod=200)
        dataframe = merge_informative_pair(
            dataframe, informative, self.timeframe, "4h", ffill=True
        )
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1h").copy()
        for period in (20, 50, 100, 200):
            informative_1h[f"ema_{period}"] = ta.EMA(informative_1h, timeperiod=period)
        dataframe = merge_informative_pair(
            dataframe, informative_1h, self.timeframe, "1h", ffill=True
        )

        rsi4 = dataframe["rsi_4h"]
        e20, e50 = dataframe["ema_20"], dataframe["ema_50"]
        e100, e200 = dataframe["ema_100"], dataframe["ema_200"]
        close = dataframe["close"]

        th = float(self.div_threshold.value)
        g1 = (e20 / e50 - 1.0) * 100.0
        g2 = (e50 / e100 - 1.0) * 100.0
        g3 = (e100 / e200 - 1.0) * 100.0

        div_long = (g1 > th) & (g2 > th) & (g3 > th)
        div_short = (g1 < -th) & (g2 < -th) & (g3 < -th)
        bmin = float(self.trend_bias20_50_min.value)
        strong_long = g1 > bmin
        strong_short = g1 < -bmin

        order_long = (e20 > e50) & (e50 > e100) & (e100 > e200)
        order_short = (e20 < e50) & (e50 < e100) & (e100 < e200)

        trend_long = order_long & div_long & strong_long
        trend_short = order_short & div_short & strong_short

        # --- 融合评分：趋势越强，range 权重越低 ---
        range_th = float(self.range_bias_threshold.value)  # 百分比（例如 0.3 表示 0.3%）
        bias_abs_1 = abs(g1)
        bias_abs_2 = abs(g2)
        bias_abs_3 = abs(g3)

        # trend_score_raw 越大表示 EMA 乖离越大（越像趋势）
        # 放缓趋势评分：避免 bias 轻微放大就过快冲到 1.0
        trend_score_raw = (
            bias_abs_1 / (range_th * 2.0) +
            bias_abs_2 / (range_th * 2.0) +
            bias_abs_3 / (range_th * 2.0)
        ) / 3.0
        trend_score_raw = np.tanh(trend_score_raw)
        trend_score_raw = trend_score_raw.clip(lower=0.0, upper=1.0)
        trend_score = trend_score_raw.rolling(
            self.trend_score_smooth_bars, min_periods=1
        ).mean()
        trend_score = trend_score.clip(lower=0.0, upper=1.0)
        # range_score 保底：防止 range 权重被“数学压死”
        range_score = np.clip(1.0 - trend_score, 0.2, 0.8)
        # 趋势越强，对 range 做软惩罚（不再绝对禁用）
        range_penalty = np.clip(trend_score * 0.5, 0.0, 0.5)

        # --- 动态 RSI 阈值：range 越强越用 35/65；trend 越强越向 50 靠拢 ---
        rsi_long_base = float(self.range_rsi_long.value)   # 默认约 35
        rsi_short_base = float(self.range_rsi_short.value)  # 默认约 65
        rsi_long_dyn = range_score * rsi_long_base + trend_score * float(self.rsi_trend_base)
        rsi_short_dyn = range_score * rsi_short_base + trend_score * float(self.rsi_trend_base)

        atr_ratio = dataframe["atr_ratio"].fillna(0.0)
        high_vol = atr_ratio > float(self.atr_ratio_high.value)
        low_vol = atr_ratio < float(self.atr_ratio_min_open.value)
        if not bool(self.classification_opt_only):
            # 连亏后收紧 RSI 条件（更难抄底/摸顶）
            ls = int(self._pair_loss_streak.get(pair, 0))
            tighten = float(ls) * float(self.loss_rsi_tighten_step)
            rsi_long_dyn = (rsi_long_dyn - tighten).clip(lower=5.0, upper=95.0)
            rsi_short_dyn = (rsi_short_dyn + tighten).clip(lower=5.0, upper=95.0)

            # 波动率联动 RSI：高波动放宽，低波动收紧
            rsi_long_dyn = np.where(high_vol, rsi_long_dyn + float(self.atr_rsi_relax), rsi_long_dyn)
            rsi_short_dyn = np.where(high_vol, rsi_short_dyn - float(self.atr_rsi_relax), rsi_short_dyn)
            rsi_long_dyn = np.where(low_vol, rsi_long_dyn - float(self.atr_rsi_tighten), rsi_long_dyn)
            rsi_short_dyn = np.where(low_vol, rsi_short_dyn + float(self.atr_rsi_tighten), rsi_short_dyn)
            rsi_long_dyn = np.clip(rsi_long_dyn, 5.0, 95.0)
            rsi_short_dyn = np.clip(rsi_short_dyn, 5.0, 95.0)

        # 震荡区仍需要“乖离率收敛”作为基本前提（避免 trend 段抄底）
        range_bias_valid = (
            (bias_abs_1 < range_th) & (bias_abs_2 < range_th) & (bias_abs_3 < range_th)
        )

        # ===== V3.5：自适应市场分类器（trend / range / mixed）=====
        # 阈值来自滚动分位数，自动适配不同市场阶段
        trend_strength = (bias_abs_1 + bias_abs_2 + bias_abs_3) / 3.0
        trend_strength_norm = np.tanh(trend_strength / max(1e-6, (range_th * 2.0))).clip(lower=0.0, upper=1.0)
        ema_order_long3 = (e20 > e50) & (e50 > e100)
        ema_order_short3 = (e20 < e50) & (e50 < e100)

        # trend_score：用归一化强度表示（0~1），ATR 作为过滤
        trend_score_v3 = trend_strength_norm.clip(lower=0.0, upper=1.0)
        win = max(50, int(self.adaptive_classifier_window))
        trend_th = trend_score_v3.rolling(win, min_periods=win).quantile(float(self.adaptive_trend_quantile))
        range_th2 = trend_score_v3.rolling(win, min_periods=win).quantile(float(self.adaptive_range_quantile))
        atr_th = atr_ratio.rolling(win, min_periods=win).quantile(float(self.adaptive_atr_quantile))

        smooth_n = max(1, int(self.adaptive_threshold_smooth_bars))
        trend_th = trend_th.rolling(smooth_n, min_periods=1).mean()
        range_th2 = range_th2.rolling(smooth_n, min_periods=1).mean()

        trend_clip_lo, trend_clip_hi = self.adaptive_trend_th_clip
        range_clip_lo, range_clip_hi = self.adaptive_range_th_clip
        trend_th = trend_th.clip(lower=float(trend_clip_lo), upper=float(trend_clip_hi))
        range_th2 = range_th2.clip(lower=float(range_clip_lo), upper=float(range_clip_hi))

        # 冷启动阶段会有 NaN，反向填充后再前向填充，避免早期分类缺失。
        trend_th = trend_th.bfill().ffill()
        range_th2 = range_th2.bfill().ffill()
        atr_th = atr_th.bfill().ffill()

        trend_persist = (
            (trend_score_v3 > trend_th)
            .rolling(int(self.adaptive_persist_bars), min_periods=1)
            .sum() >= int(self.adaptive_persist_min_hits)
        )
        vol_regime = np.where(atr_ratio > atr_th, "high_vol", "low_vol")

        is_trend_regime = (
            (trend_score_v3 > trend_th)
            & (atr_ratio > atr_th)
            & trend_persist
            & (ema_order_long3 | ema_order_short3)
        )
        is_range_regime = (
            (trend_score_v3 < range_th2)
            & (atr_ratio < atr_th)
            & range_bias_valid
        )
        market_state = np.select([is_trend_regime, is_range_regime], ["trend", "range"], default="mixed")

        trend_weight = np.where(
            market_state == "trend",
            1.0,
            np.where(market_state == "mixed", float(self.regime_mixed_weight), float(self.regime_low_weight)),
        )
        range_weight = np.where(
            market_state == "range",
            1.0,
            np.where(market_state == "mixed", float(self.regime_mixed_weight), float(self.regime_low_weight)),
        )

        # EMA20/EMA50 交叉冷却：交叉发生后 N 根内不做 range（提高 range 胜率）
        ema_up = (e20 > e50)
        ema_cross = (ema_up != ema_up.shift(1)).fillna(False)
        cooldown_n = max(1, int(self.range_ema_cross_cooldown_bars))
        ema_cross_recent = ema_cross.rolling(cooldown_n, min_periods=1).max().astype(bool)
        range_long_candidate = (
            (rsi4 < rsi_long_dyn) & range_bias_valid & ~ema_cross_recent
        )
        range_short_candidate = (
            (rsi4 > rsi_short_dyn) & range_bias_valid & ~ema_cross_recent
        )

        # 防对冲：趋势多成立时禁震荡空；趋势空成立时禁震荡多
        range_long = range_long_candidate & ~trend_short
        range_short = range_short_candidate & ~trend_long

        # --- 独立通道：趋势与震荡分别判断，不再做强度竞争融合 ---
        # 震荡信号做软惩罚（趋势越强，range_strength 越低，但不归零）
        range_strength_long = (1.0 - range_penalty) * range_long.astype(int)
        range_strength_short = (1.0 - range_penalty) * range_short.astype(int)
        trend_strength_long = trend_score * trend_long.astype(int)
        trend_strength_short = trend_score * trend_short.astype(int)

        dataframe["range_long_sig"] = range_long
        dataframe["range_short_sig"] = range_short
        dataframe["trend_long_sig"] = trend_long
        dataframe["trend_short_sig"] = trend_short

        dataframe["trend_score"] = trend_score
        dataframe["range_score"] = range_score
        dataframe["range_penalty"] = range_penalty
        dataframe["range_ema_cross_recent"] = ema_cross_recent
        dataframe["trend_strength_norm"] = trend_strength_norm
        dataframe["trend_score_v3"] = trend_score_v3
        dataframe["trend_th_adaptive"] = trend_th
        dataframe["range_th_adaptive"] = range_th2
        dataframe["atr_th_adaptive"] = atr_th
        dataframe["trend_persist"] = trend_persist
        dataframe["vol_regime"] = vol_regime
        dataframe["market_state"] = market_state
        dataframe["trend_weight"] = trend_weight
        dataframe["range_weight"] = range_weight
        dataframe["dyn_rsi_long_th"] = rsi_long_dyn
        dataframe["dyn_rsi_short_th"] = rsi_short_dyn

        dataframe["range_strength_long"] = range_strength_long
        dataframe["range_strength_short"] = range_strength_short
        dataframe["trend_strength_long"] = trend_strength_long
        dataframe["trend_strength_short"] = trend_strength_short
        dataframe["high_vol"] = high_vol
        dataframe["low_vol"] = low_vol

        # ===== V4：多周期分类融合（4h方向 + 1h状态）=====
        ema50_4h = dataframe["ema_50_4h"]
        ema200_4h = dataframe["ema_200_4h"]
        htf_bias = np.where(ema50_4h > ema200_4h, "long", "short")

        e20_1h = dataframe["ema_20_1h"]
        e50_1h = dataframe["ema_50_1h"]
        e100_1h = dataframe["ema_100_1h"]
        e200_1h = dataframe["ema_200_1h"]
        g1_1h = (e20_1h / e50_1h - 1.0) * 100.0
        g2_1h = (e50_1h / e100_1h - 1.0) * 100.0
        g3_1h = (e100_1h / e200_1h - 1.0) * 100.0
        trend_strength_1h = (abs(g1_1h) + abs(g2_1h) + abs(g3_1h)) / 3.0
        trend_score_1h = np.tanh(trend_strength_1h / max(1e-6, (range_th * 2.0))).clip(lower=0.0, upper=1.0)
        trend_th_1h = trend_score_1h.rolling(win, min_periods=win).quantile(float(self.adaptive_trend_quantile)).rolling(smooth_n, min_periods=1).mean()
        range_th_1h = trend_score_1h.rolling(win, min_periods=win).quantile(float(self.adaptive_range_quantile)).rolling(smooth_n, min_periods=1).mean()
        trend_th_1h = trend_th_1h.clip(lower=float(trend_clip_lo), upper=float(trend_clip_hi)).bfill().ffill()
        range_th_1h = range_th_1h.clip(lower=float(range_clip_lo), upper=float(range_clip_hi)).bfill().ffill()
        mtf_state = np.select(
            [trend_score_1h > trend_th_1h, trend_score_1h < range_th_1h],
            ["trend", "range"],
            default="mixed",
        )
        final_state = np.select(
            [
                (htf_bias == "long") & (mtf_state == "trend"),
                (htf_bias == "short") & (mtf_state == "trend"),
                (htf_bias == "long") & (mtf_state == "range"),
                (htf_bias == "short") & (mtf_state == "range"),
                (mtf_state == "range"),
            ],
            ["strong_trend_long", "strong_trend_short", "pullback_long", "pullback_short", "pure_range"],
            default="choppy",
        )

        # ===== V5：15m 微观结构（BOS / CHoCH / Liquidity Sweep）=====
        bos_lookback = 20
        liq_lookback = 10
        prev_high = dataframe["high"].shift(1)
        prev_low = dataframe["low"].shift(1)
        bos_long = dataframe["close"] > dataframe["high"].rolling(bos_lookback, min_periods=1).max().shift(1)
        bos_short = dataframe["close"] < dataframe["low"].rolling(bos_lookback, min_periods=1).min().shift(1)
        choch_long = (dataframe["low"] > prev_low) & (dataframe["close"] > prev_high)
        choch_short = (dataframe["high"] < prev_high) & (dataframe["close"] < prev_low)
        liq_sweep_long = (dataframe["low"] < dataframe["low"].rolling(liq_lookback, min_periods=1).min().shift(1)) & (dataframe["close"] > dataframe["open"])
        liq_sweep_short = (dataframe["high"] > dataframe["high"].rolling(liq_lookback, min_periods=1).max().shift(1)) & (dataframe["close"] < dataframe["open"])

        dataframe["htf_bias"] = htf_bias
        dataframe["trend_score_1h"] = trend_score_1h
        dataframe["trend_th_1h"] = trend_th_1h
        dataframe["range_th_1h"] = range_th_1h
        dataframe["mtf_state"] = mtf_state
        dataframe["final_state"] = final_state
        dataframe["bos_long"] = bos_long.fillna(False)
        dataframe["bos_short"] = bos_short.fillna(False)
        dataframe["choch_long"] = choch_long.fillna(False)
        dataframe["choch_short"] = choch_short.fillna(False)
        dataframe["liq_sweep_long"] = liq_sweep_long.fillna(False)
        dataframe["liq_sweep_short"] = liq_sweep_short.fillna(False)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        dataframe.loc[:, "enter_long"] = 0
        dataframe.loc[:, "enter_short"] = 0
        dataframe.loc[:, "enter_tag"] = None

        trend_score = dataframe["trend_score"].fillna(0.0)
        range_score = dataframe["range_score"].fillna(0.0)
        market_state = dataframe["market_state"].fillna("mixed")
        tw = dataframe["trend_weight"].fillna(float(self.regime_mixed_weight))
        rw = dataframe["range_weight"].fillna(float(self.regime_mixed_weight))
        high_vol = dataframe["high_vol"].fillna(False)
        low_vol = dataframe["low_vol"].fillna(False)

        range_sig_long = dataframe["range_long_sig"].fillna(False)
        range_sig_short = dataframe["range_short_sig"].fillna(False)
        trend_sig_long = dataframe["trend_long_sig"].fillna(False)
        trend_sig_short = dataframe["trend_short_sig"].fillna(False)
        final_state = dataframe["final_state"].fillna("choppy")
        bos_long = dataframe["bos_long"].fillna(False)
        bos_short = dataframe["bos_short"].fillna(False)
        choch_long = dataframe["choch_long"].fillna(False)
        choch_short = dataframe["choch_short"].fillna(False)
        liq_sweep_long = dataframe["liq_sweep_long"].fillna(False)
        liq_sweep_short = dataframe["liq_sweep_short"].fillna(False)

        if bool(self.classification_opt_only):
            # 分类优化阶段：固定入场阈值，避免其它动态系统干扰分类器评估
            trend_min = float(self.trend_entry_min_score_base)
            range_min = float(self.range_entry_min_score_base)
        else:
            ls = int(self._pair_loss_streak.get(pair, 0))
            # 独立通道阈值：更低的基线 + 连亏动态收紧
            extra = min(float(ls) * float(self.loss_entry_step), float(self.loss_entry_cap))
            trend_min = float(self.trend_entry_min_score_base) + extra
            trend_min = np.where(high_vol, trend_min - float(self.atr_entry_relax), trend_min)
            trend_min = np.clip(trend_min, 0.40, 0.95)

            range_min = float(self.range_entry_min_score_base) + min(float(ls) * 0.05, 0.2)
            range_min = np.clip(range_min, 0.25, 0.80)

        trend_entry_long = trend_sig_long & (trend_score > trend_min)
        trend_entry_short = trend_sig_short & (trend_score > trend_min)
        range_entry_long = range_sig_long & (range_score > range_min)
        range_entry_short = range_sig_short & (range_score > range_min)

        # V4+V5：状态路由 + 结构触发（再叠加原有 RSI/EMA 触发）
        bos_w = float(self._structure_weights.get("bos", 1.0))
        choch_w = float(self._structure_weights.get("choch", 1.0))
        sweep_w = float(self._structure_weights.get("sweep", 1.0))
        market_bucket = final_state.map(self._market_bucket)
        bos_cell_w = np.array(
            [float(self._market_structure_weight.get(f"{market_bucket.iloc[i]}|bos", 1.0)) for i in range(len(dataframe))]
        )
        choch_cell_w = np.array(
            [float(self._market_structure_weight.get(f"{market_bucket.iloc[i]}|choch", 1.0)) for i in range(len(dataframe))]
        )
        sweep_cell_w = np.array(
            [float(self._market_structure_weight.get(f"{market_bucket.iloc[i]}|sweep", 1.0)) for i in range(len(dataframe))]
        )
        w_floor = float(self.evolve_enable_weight_floor)
        bos_long_on = bos_long & ((bos_w * bos_cell_w) >= w_floor)
        bos_short_on = bos_short & ((bos_w * bos_cell_w) >= w_floor)
        choch_long_on = choch_long & ((choch_w * choch_cell_w) >= w_floor)
        choch_short_on = choch_short & ((choch_w * choch_cell_w) >= w_floor)
        sweep_long_on = liq_sweep_long & ((sweep_w * sweep_cell_w) >= w_floor)
        sweep_short_on = liq_sweep_short & ((sweep_w * sweep_cell_w) >= w_floor)

        structure_long = np.select(
            [
                final_state == "strong_trend_long",
                final_state == "pullback_long",
                final_state == "pure_range",
            ],
            [
                bos_long_on,
                (sweep_long_on | choch_long_on),
                sweep_long_on,
            ],
            default=False,
        ).astype(bool)
        structure_short = np.select(
            [
                final_state == "strong_trend_short",
                final_state == "pullback_short",
                final_state == "pure_range",
            ],
            [
                bos_short_on,
                (sweep_short_on | choch_short_on),
                sweep_short_on,
            ],
            default=False,
        ).astype(bool)

        # mixed 区降频：额外收紧（避免“混沌乱打”）
        tighten = np.where(market_state == "mixed", float(self.regime_mixed_entry_tighten), 0.0)
        trend_entry_long = trend_entry_long & (trend_score > (trend_min + tighten))
        trend_entry_short = trend_entry_short & (trend_score > (trend_min + tighten))
        range_entry_long = range_entry_long & (range_score > (range_min + tighten))
        range_entry_short = range_entry_short & (range_score > (range_min + tighten))

        # V3 路由：只有对应权重足够才启用该引擎；低波动仍只做 range
        allow_trend = tw > 0.5
        allow_range = rw > 0.5
        m_long = (
            (trend_entry_long & allow_trend)
            | (range_entry_long & allow_range)
        )
        m_short = (
            (trend_entry_short & allow_trend)
            | (range_entry_short & allow_range)
        )
        m_long = np.where(low_vol, (range_entry_long & allow_range), m_long)
        m_short = np.where(low_vol, (range_entry_short & allow_range), m_short)
        m_long = m_long & structure_long
        m_short = m_short & structure_short

        structure_type_long = np.select(
            [bos_long_on, choch_long_on, sweep_long_on],
            ["bos", "choch", "sweep"],
            default="none",
        )
        structure_type_short = np.select(
            [bos_short_on, choch_short_on, sweep_short_on],
            ["bos", "choch", "sweep"],
            default="none",
        )
        trigger_long = np.where(trend_entry_long, "rsi_ema_trend", np.where(range_entry_long, "rsi_ema_range", "core"))
        trigger_short = np.where(trend_entry_short, "rsi_ema_trend", np.where(range_entry_short, "rsi_ema_range", "core"))

        # BTC 方向过滤（分类优化阶段关闭，避免干扰分类器评估）
        if not bool(self.classification_opt_only):
            market_is_bear = self._is_market_bear()
            if market_is_bear and bool(self.market_bear_disable_trend_long):
                m_long = m_long & ~trend_entry_long

        long_tags = np.array(
            [
                self._compose_entry_tag(final_state.iloc[i], structure_type_long[i], trigger_long[i])
                for i in range(len(dataframe))
            ],
            dtype=object,
        )
        short_tags = np.array(
            [
                self._compose_entry_tag(final_state.iloc[i], structure_type_short[i], trigger_short[i])
                for i in range(len(dataframe))
            ],
            dtype=object,
        )
        if self._disabled_entry_tags:
            m_long = m_long & ~np.isin(long_tags, list(self._disabled_entry_tags))
            m_short = m_short & ~np.isin(short_tags, list(self._disabled_entry_tags))
        if self._disabled_market_structure:
            long_cell_keys = np.array([f"{market_bucket.iloc[i]}|{structure_type_long[i]}" for i in range(len(dataframe))], dtype=object)
            short_cell_keys = np.array([f"{market_bucket.iloc[i]}|{structure_type_short[i]}" for i in range(len(dataframe))], dtype=object)
            bad_cells = list(self._disabled_market_structure)
            m_long = m_long & ~np.isin(long_cell_keys, bad_cells)
            m_short = m_short & ~np.isin(short_cell_keys, bad_cells)

        m_long_np = np.asarray(m_long, dtype=bool)
        m_short_np = np.asarray(m_short, dtype=bool)
        dataframe.loc[m_long, "enter_long"] = 1
        dataframe.loc[m_long, "enter_tag"] = long_tags[m_long_np]

        dataframe.loc[m_short, "enter_short"] = 1
        dataframe.loc[m_short, "enter_tag"] = short_tags[m_short_np]

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def _update_max_roe(self, trade: Trade, key: str, current_profit: float) -> float:
        prev = trade.get_custom_data(key, None)
        if prev is None:
            prev = 0.0
        mx = max(float(prev), float(current_profit))
        if trade.id is not None:
            trade.set_custom_data(key, mx)
        return mx

    def _roe_trailing_exit(
        self,
        trade: Trade,
        current_profit: float,
        current_time: datetime,
        max_key: str,
        stages: List[Tuple[float, float]],
    ) -> bool:
        if trade.id is None:
            return False
        open_u = trade.open_date_utc
        if open_u is None:
            return False
        hours = (current_time - open_u).total_seconds() / 3600.0
        max_p = self._update_max_roe(trade, max_key, current_profit)
        if hours < self.roe_trailing_grace_hours:
            return False
        dd = 0.0
        armed = False
        for arm, pull in sorted(stages, key=lambda x: x[0]):
            if max_p >= arm:
                dd = float(pull)
                armed = True
        if not armed or dd <= 0:
            return False
        floor = max_p * (1.0 - dd)
        return current_profit < floor

    def _dynamic_lock_profit_exit(
        self,
        trade: Trade,
        current_profit: float,
        current_time: datetime,
        trend_score: float,
        range_score: float,
    ) -> bool:
        """
        保本/锁利润：
        0 < roe < 15% 且 holding_time > 2h，当 roe 回落到 lock_profit 以下则平仓。
        lock_profit 由 trend_score 动态决定。
        """
        if trade.id is None:
            return False

        open_u = trade.open_date_utc
        if open_u is None:
            return False

        hours = (current_time - open_u).total_seconds() / 3600.0
        hold_h = int(self.trend_hold_hours.value)
        if hours <= hold_h:
            return False

        if not (0 < current_profit < 0.15):
            return False

        lock_profit = range_score * 0.02 + trend_score * 0.06
        return current_profit < lock_profit

    def _dynamic_roe_trailing_exit(
        self,
        trade: Trade,
        current_profit: float,
        current_time: datetime,
        max_key: str,
        trend_score: float,
        range_score: float,
        atr_ratio: float,
    ) -> bool:
        """
        动态 ROE 回撤止盈（跟随 max_roe）：
        - 开仓未满 1h：不启动 trailing（roe_trailing_grace_hours）
        - max_roe > start_profit 才允许回撤卖出
        - floor = max_roe * (1 - trail_ratio)
        """
        if trade.id is None:
            return False

        open_u = trade.open_date_utc
        if open_u is None:
            return False

        hours = (current_time - open_u).total_seconds() / 3600.0
        if hours < self.roe_trailing_grace_hours:
            return False

        max_p = self._update_max_roe(trade, max_key, current_profit)

        vol_scale = 1.0 + max(0.0, float(atr_ratio) - float(self.atr_ratio_min_open.value)) * float(self.atr_trail_scale)
        start_profit = (range_score * float(self.dyn_start_range.value) + trend_score * float(self.dyn_start_trend.value)) * vol_scale
        if max_p < start_profit:
            return False

        trail_ratio = (range_score * float(self.dyn_trail_range.value) + trend_score * float(self.dyn_trail_trend.value)) * vol_scale
        trail_ratio = min(trail_ratio, 0.85)
        floor = max_p * (1.0 - trail_ratio)
        return current_profit < floor

    def _get_market_vol_factor(self) -> float:
        """
        用 BTC 的 atr_ratio 估计市场整体波动，返回全局仓位系数。
        """
        if not self.market_vol_enabled or not self.dp:
            return 1.0

        candidates = [
            "BTC/USDT:USDT",
            "BTC/USDT",
            "BTC/USDC:USDC",
            "BTC/USDC",
        ]

        atr_ratio = None
        for m_pair in candidates:
            try:
                mdf = self.dp.get_pair_dataframe(pair=m_pair, timeframe=self.timeframe)
                if mdf is None or len(mdf) < self.atr_period:
                    continue
                tmp = mdf.copy()
                tmp["atr"] = ta.ATR(tmp, timeperiod=self.atr_period)
                val = tmp.iloc[-1]["atr"] / tmp.iloc[-1]["close"]
                if val is not None and not np.isnan(val):
                    atr_ratio = float(val)
                    break
            except Exception:
                continue

        if atr_ratio is None:
            return 1.0

        if atr_ratio < float(self.market_vol_low):
            return float(self.market_exposure_low_mult)
        if atr_ratio > float(self.market_vol_high):
            return float(self.market_exposure_high_mult)
        return 1.0

    def _get_orderbook_top(self, pair: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        返回 (best_bid, best_ask, spread, spread_bps)。
        无法获取则返回 (None, None, None, None)。
        """
        if not self.exec_use_orderbook or not self.dp:
            return (None, None, None, None)
        try:
            ob = self.dp.orderbook(pair, self.exec_orderbook_levels)
            bids = (ob or {}).get("bids") or []
            asks = (ob or {}).get("asks") or []
            if not bids or not asks:
                return (None, None, None, None)
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            if best_bid <= 0 or best_ask <= 0:
                return (None, None, None, None)
            spread = best_ask - best_bid
            mid = (best_ask + best_bid) / 2.0
            spread_bps = (spread / mid) * 10000.0 if mid > 0 else None
            return (best_bid, best_ask, spread, float(spread_bps) if spread_bps is not None else None)
        except Exception:
            return (None, None, None, None)

    def _estimate_slippage_bps(self, pair: str, atr_ratio: float, spread_bps: Optional[float]) -> float:
        """
        简易滑点模型（bps）：结合价差与波动率。
        """
        base = float(self.exec_aggressive_slip_bps)
        vol_add = max(0.0, float(atr_ratio)) * float(self.exec_slip_vol_k)
        half_spread = max(0.0, float(spread_bps) / 2.0) if spread_bps is not None else 0.0
        return max(base + vol_add, half_spread)

    def _log_pricing(self, msg: str, payload: Dict[str, Any]) -> None:
        if not bool(self.exec_log_every):
            return
        try:
            logger.info("%s | %s", msg, payload)
        except Exception:
            return

    def custom_entry_price(
        self,
        pair: str,
        trade: Trade | None,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        """
        挂单优先：价差可控时贴近 bid/ask 做 maker；
        回退：价差过大/盘口不可用时，给更激进的价格模拟吃单与滑点。
        """
        if proposed_rate is None or proposed_rate <= 0:
            return proposed_rate

        df = None
        atr_ratio = 0.0
        try:
            if self.dp:
                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is not None and len(df):
                ar = df.iloc[-1].get("atr_ratio", 0.0)
                atr_ratio = 0.0 if ar is None or np.isnan(ar) else float(ar)
        except Exception:
            atr_ratio = 0.0

        bid, ask, _, spread_bps = self._get_orderbook_top(pair)
        slip_bps = self._estimate_slippage_bps(pair, atr_ratio=atr_ratio, spread_bps=spread_bps)

        maker_off = float(self.exec_maker_offset_bps) / 10000.0
        slip = float(slip_bps) / 10000.0
        max_spread = float(self.exec_max_spread_bps)

        mode = "fallback"
        reason = "no_orderbook"
        price = float(proposed_rate)

        if bid is not None and ask is not None and spread_bps is not None:
            if float(spread_bps) <= max_spread:
                mode = "maker"
                reason = "spread_ok"
                if side == "long":
                    price = float(bid) * (1.0 + maker_off)
                else:
                    price = float(ask) * (1.0 - maker_off)
            else:
                mode = "aggressive"
                reason = "spread_wide"
                if side == "long":
                    price = float(ask) * (1.0 + slip)
                else:
                    price = float(bid) * (1.0 - slip)

        # 兜底：避免给出明显劣于 proposed_rate 的价格（过度追价）
        if side == "long":
            price = min(price, float(proposed_rate) * (1.0 + slip))
        else:
            price = max(price, float(proposed_rate) * (1.0 - slip))

        if trade is not None and trade.id is not None:
            try:
                trade.set_custom_data(
                    "last_entry_pricing",
                    {
                        "t": current_time.isoformat(),
                        "tag": entry_tag,
                        "tag_cn": self._tag_cn(entry_tag),
                        "side": side,
                        "mode": mode,
                        "reason": reason,
                        "proposed": float(proposed_rate),
                        "price": float(price),
                        "bid": bid,
                        "ask": ask,
                        "spread_bps": spread_bps,
                        "atr_ratio": atr_ratio,
                        "slip_bps": float(slip_bps),
                    },
                )
            except Exception:
                pass

        self._log_pricing(
            "entry_pricing",
            {
                "pair": pair,
                "side": side,
                "tag": entry_tag,
                "tag_cn": self._tag_cn(entry_tag),
                "mode": mode,
                "reason": reason,
                "proposed": float(proposed_rate),
                "price": float(price),
                "bid": bid,
                "ask": ask,
                "spread_bps": spread_bps,
                "atr_ratio": atr_ratio,
                "slip_bps": float(slip_bps),
            },
        )
        return float(price)

    def custom_exit_price(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        proposed_rate: float,
        current_profit: float,
        exit_tag: str | None,
        **kwargs: Any,
    ) -> float:
        """
        出场同样做 maker 优先 + 回退（更易成交），并记录成交质量。
        """
        if proposed_rate is None or proposed_rate <= 0:
            return proposed_rate

        side = "short" if trade.is_short else "long"

        df = None
        atr_ratio = 0.0
        try:
            if self.dp:
                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is not None and len(df):
                ar = df.iloc[-1].get("atr_ratio", 0.0)
                atr_ratio = 0.0 if ar is None or np.isnan(ar) else float(ar)
        except Exception:
            atr_ratio = 0.0

        bid, ask, _, spread_bps = self._get_orderbook_top(pair)
        slip_bps = self._estimate_slippage_bps(pair, atr_ratio=atr_ratio, spread_bps=spread_bps)

        maker_off = float(self.exec_maker_offset_bps) / 10000.0
        slip = float(slip_bps) / 10000.0
        max_spread = float(self.exec_max_spread_bps)

        mode = "fallback"
        reason = "no_orderbook"
        price = float(proposed_rate)

        # 平多 = 卖出；平空 = 买入
        closing_side = "sell" if not trade.is_short else "buy"
        if bid is not None and ask is not None and spread_bps is not None:
            if float(spread_bps) <= max_spread:
                mode = "maker"
                reason = "spread_ok"
                if closing_side == "sell":
                    price = float(ask) * (1.0 - maker_off)
                else:
                    price = float(bid) * (1.0 + maker_off)
            else:
                mode = "aggressive"
                reason = "spread_wide"
                if closing_side == "sell":
                    price = float(bid) * (1.0 - slip)
                else:
                    price = float(ask) * (1.0 + slip)

        if closing_side == "sell":
            price = max(price, float(proposed_rate) * (1.0 - slip))
        else:
            price = min(price, float(proposed_rate) * (1.0 + slip))

        if trade is not None and trade.id is not None:
            try:
                trade.set_custom_data(
                    "last_exit_pricing",
                    {
                        "t": current_time.isoformat(),
                        "tag": exit_tag,
                        "tag_cn": self._tag_cn(exit_tag),
                        "side": side,
                        "closing_side": closing_side,
                        "mode": mode,
                        "reason": reason,
                        "proposed": float(proposed_rate),
                        "price": float(price),
                        "bid": bid,
                        "ask": ask,
                        "spread_bps": spread_bps,
                        "atr_ratio": atr_ratio,
                        "slip_bps": float(slip_bps),
                        "profit": float(current_profit),
                    },
                )
            except Exception:
                pass

        self._log_pricing(
            "exit_pricing",
            {
                "pair": pair,
                "side": side,
                "closing_side": closing_side,
                "tag": exit_tag,
                "tag_cn": self._tag_cn(exit_tag),
                "mode": mode,
                "reason": reason,
                "proposed": float(proposed_rate),
                "price": float(price),
                "bid": bid,
                "ask": ask,
                "spread_bps": spread_bps,
                "atr_ratio": atr_ratio,
                "slip_bps": float(slip_bps),
                "profit": float(current_profit),
            },
        )
        return float(price)

    def _is_market_bear(self) -> bool:
        """
        使用 BTC EMA50/EMA200 判断大盘方向。
        """
        if not self.dp:
            return False

        candidates = [
            "BTC/USDT:USDT",
            "BTC/USDT",
            "BTC/USDC:USDC",
            "BTC/USDC",
        ]
        for m_pair in candidates:
            try:
                mdf = self.dp.get_pair_dataframe(pair=m_pair, timeframe=self.timeframe)
                if mdf is None or len(mdf) < 220:
                    continue
                tmp = mdf.copy()
                tmp["ema_50"] = ta.EMA(tmp, timeperiod=50)
                tmp["ema_200"] = ta.EMA(tmp, timeperiod=200)
                e50 = tmp.iloc[-1].get("ema_50", np.nan)
                e200 = tmp.iloc[-1].get("ema_200", np.nan)
                if e50 is None or e200 is None or np.isnan(e50) or np.isnan(e200):
                    continue
                return float(e50) < float(e200)
            except Exception:
                continue
        return False

    @staticmethod
    def _trade_open_datetime(trade: Trade) -> Optional[datetime]:
        u = getattr(trade, "open_date_utc", None)
        if u is None:
            u = getattr(trade, "open_date", None)
        if u is None:
            return None
        if u.tzinfo is not None:
            return u.astimezone(timezone.utc).replace(tzinfo=None)
        return u

    @staticmethod
    def _datetime_diff_minutes_safe(a: datetime, b: datetime) -> float:
        """(a - b) 分钟；统一为 naive UTC，避免 naive/aware 相减抛错导致时间止损整块失效。"""
        aa = a.astimezone(timezone.utc).replace(tzinfo=None) if a.tzinfo is not None else a
        bb = b.astimezone(timezone.utc).replace(tzinfo=None) if b.tzinfo is not None else b
        return (aa - bb).total_seconds() / 60.0

    def _time_based_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_profit: float,
    ) -> Optional[str]:
        """最大持仓/时间类出场：不依赖 dataframe 行数，优先于其它 custom_exit。"""
        open_u = self._trade_open_datetime(trade)
        if open_u is None:
            return None
        try:
            tf_min = float(timeframe_to_minutes(self.timeframe))
            hold_minutes = self._datetime_diff_minutes_safe(current_time, open_u)
            hold_bars = int(max(0.0, hold_minutes / max(1.0, tf_min)))
        except Exception:
            return None

        market_state_now = "mixed"
        atr_ratio_now = 0.0
        trend_score_now = 0.0
        try:
            if self.dp:
                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if df is not None and len(df) >= 1:
                    row_now = df.iloc[-1]
                    ar = row_now.get("atr_ratio", 0.0)
                    atr_ratio_now = 0.0 if ar is None or np.isnan(ar) else float(ar)
                    ts = row_now.get("trend_score", 0.0)
                    try:
                        trend_score_now = 0.0 if ts is None or np.isnan(ts) else float(ts)
                    except Exception:
                        trend_score_now = 0.0
                    ms = row_now.get("market_state", "mixed")
                    market_state_now = "mixed" if ms is None else str(ms).strip().lower()
        except Exception:
            pass

        # 分类优化阶段：只保留 max_hold 硬限制，其它时间类退出关闭（控制变量）
        if not bool(self.classification_opt_only):
            if hold_bars > int(self.time_profit_fail_bars):
                if market_state_now == "range":
                    should_exit = float(current_profit) < float(self.time_profit_fail_min_profit)
                elif market_state_now == "mixed":
                    should_exit = float(current_profit) < 0.01
                else:
                    should_exit = (
                        trend_score_now < float(self.time_profit_fail_trend_cutoff)
                        and float(current_profit) < float(self.time_profit_fail_min_profit)
                    )
                if should_exit:
                    tag_out = "time_profit_fail_exit"
                    try:
                        trade.set_custom_data("last_exit_tag", tag_out)
                        trade.set_custom_data("last_exit_tag_cn", self._tag_cn(tag_out))
                    except Exception:
                        pass
                    return tag_out

            if atr_ratio_now < float(self.low_vol_hold_atr_ratio) and hold_bars >= int(self.low_vol_max_hold_bars):
                tag_out = "time_low_vol_exit"
                try:
                    trade.set_custom_data("last_exit_tag", tag_out)
                    trade.set_custom_data("last_exit_tag_cn", self._tag_cn(tag_out))
                except Exception:
                    pass
                return tag_out

        if market_state_now == "trend":
            max_hold_now = int(self.max_hold_bars_trend)
        elif market_state_now == "range":
            max_hold_now = int(self.max_hold_bars_range)
        else:
            max_hold_now = int(self.max_hold_bars_mixed)

        if hold_bars >= max_hold_now:
            tag_out = "time_max_hold_exit"
            try:
                trade.set_custom_data("last_exit_tag", tag_out)
                trade.set_custom_data("last_exit_tag_cn", self._tag_cn(tag_out))
            except Exception:
                pass
            return tag_out
        return None

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> Optional[str]:
        te = self._time_based_exit(pair, trade, current_time, current_profit)
        if te is not None:
            return te

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) < 2:
            return None

        # 分类优化阶段：只评估分类器对交易行为的影响，不让 exit 逻辑“救”分类
        if bool(self.classification_opt_only):
            return None

        row = dataframe.iloc[-1]
        prev = dataframe.iloc[-2]
        rsi4 = row.get("rsi_4h")
        e20, e50 = row.get("ema_20"), row.get("ema_50")
        p20, p50 = prev.get("ema_20"), prev.get("ema_50")
        if e20 is None or e50 is None or p20 is None or p50 is None:
            return None

        fs_tag, structure_kind, _ = self._parse_entry_tag(trade.enter_tag)
        fs_l = fs_tag.lower()
        is_range = ("range" in fs_l) or (structure_kind == "sweep")
        is_trend = ("strong_trend" in fs_l) or ("pullback" in fs_l) or (structure_kind in ("bos", "choch"))
        final_state_now = str(row.get("final_state", "choppy"))
        choch_long_now = bool(row.get("choch_long", False))
        choch_short_now = bool(row.get("choch_short", False))

        # V5 结构失效退出：强趋势状态出现反向 CHoCH 立即退出
        if "strong_trend" in final_state_now:
            is_short_trade = bool(getattr(trade, "is_short", False))
            invalid = (is_short_trade and choch_long_now) or ((not is_short_trade) and choch_short_now)
            if invalid:
                tag_out = "structure_invalidation_exit"
                try:
                    trade.set_custom_data("last_exit_tag", tag_out)
                    trade.set_custom_data("last_exit_tag_cn", self._tag_cn(tag_out))
                except Exception:
                    pass
                return tag_out

        # 来自 populate_indicators 的融合评分（0~1）
        trend_score = row.get("trend_score", 0.0)
        range_score = row.get("range_score", None)
        atr_ratio = row.get("atr_ratio", 0.0)
        try:
            trend_score = 0.0 if trend_score is None or np.isnan(trend_score) else float(trend_score)
        except Exception:
            trend_score = 0.0
        try:
            range_score = 1.0 - trend_score if (range_score is None or np.isnan(range_score)) else float(range_score)
        except Exception:
            range_score = 1.0 - trend_score
        try:
            atr_ratio = 0.0 if atr_ratio is None or np.isnan(atr_ratio) else float(atr_ratio)
        except Exception:
            atr_ratio = 0.0

        if is_range:
            if not trade.is_short:
                if e20 < e50:
                    tag_out = "range_struct_exit_long"
                    try:
                        trade.set_custom_data("last_exit_tag", tag_out)
                        trade.set_custom_data("last_exit_tag_cn", self._tag_cn(tag_out))
                    except Exception:
                        pass
                    return tag_out
            else:
                if e20 > e50:
                    tag_out = "range_struct_exit_short"
                    try:
                        trade.set_custom_data("last_exit_tag", tag_out)
                        trade.set_custom_data("last_exit_tag_cn", self._tag_cn(tag_out))
                    except Exception:
                        pass
                    return tag_out

            ex_l = float(self.range_rsi_exit_long.value)
            ex_s = float(self.range_rsi_exit_short.value)
            if trade.is_short:
                if rsi4 is not None and not np.isnan(rsi4) and rsi4 < ex_s:
                    tag_out = "range_rsi_exit"
                    try:
                        trade.set_custom_data("last_exit_tag", tag_out)
                        trade.set_custom_data("last_exit_tag_cn", self._tag_cn(tag_out))
                    except Exception:
                        pass
                    return tag_out
            else:
                if rsi4 is not None and not np.isnan(rsi4) and rsi4 > ex_l:
                    tag_out = "range_rsi_exit"
                    try:
                        trade.set_custom_data("last_exit_tag", tag_out)
                        trade.set_custom_data("last_exit_tag_cn", self._tag_cn(tag_out))
                    except Exception:
                        pass
                    return tag_out

            # 条件2：保本/锁利润（动态 lock_profit）
            if self._dynamic_lock_profit_exit(
                trade,
                current_profit,
                current_time,
                trend_score=trend_score,
                range_score=range_score,
            ):
                tag_out = "range_dynamic_lock"
                try:
                    trade.set_custom_data("last_exit_tag", tag_out)
                    trade.set_custom_data("last_exit_tag_cn", self._tag_cn(tag_out))
                except Exception:
                    pass
                return tag_out

            # 条件3：动态 ROE 回撤 trailing
            if self._dynamic_roe_trailing_exit(
                trade,
                current_profit,
                current_time,
                "range_max_roe",
                trend_score=trend_score,
                range_score=range_score,
                atr_ratio=atr_ratio,
            ):
                tag_out = "range_dynamic_trail"
                try:
                    trade.set_custom_data("last_exit_tag", tag_out)
                    trade.set_custom_data("last_exit_tag_cn", self._tag_cn(tag_out))
                except Exception:
                    pass
                return tag_out

            return None

        if is_trend:
            if not trade.is_short:
                if p20 >= p50 and e20 < e50:
                    tag_out = "trend_ema_cross"
                    try:
                        trade.set_custom_data("last_exit_tag", tag_out)
                        trade.set_custom_data("last_exit_tag_cn", self._tag_cn(tag_out))
                    except Exception:
                        pass
                    return tag_out
            else:
                if p20 <= p50 and e20 > e50:
                    tag_out = "trend_ema_cross"
                    try:
                        trade.set_custom_data("last_exit_tag", tag_out)
                        trade.set_custom_data("last_exit_tag_cn", self._tag_cn(tag_out))
                    except Exception:
                        pass
                    return tag_out

            # 条件2：保本/锁利润（动态 lock_profit）
            if self._dynamic_lock_profit_exit(
                trade,
                current_profit,
                current_time,
                trend_score=trend_score,
                range_score=range_score,
            ):
                tag_out = "trend_dynamic_lock"
                try:
                    trade.set_custom_data("last_exit_tag", tag_out)
                    trade.set_custom_data("last_exit_tag_cn", self._tag_cn(tag_out))
                except Exception:
                    pass
                return tag_out

            # 条件3：动态 ROE 回撤 trailing
            if self._dynamic_roe_trailing_exit(
                trade,
                current_profit,
                current_time,
                "trend_max_roe",
                trend_score=trend_score,
                range_score=range_score,
                atr_ratio=atr_ratio,
            ):
                tag_out = "trend_dynamic_trail"
                try:
                    trade.set_custom_data("last_exit_tag", tag_out)
                    trade.set_custom_data("last_exit_tag_cn", self._tag_cn(tag_out))
                except Exception:
                    pass
                return tag_out

        return None

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs: Any,
    ) -> Optional[float]:
        """
        分批止盈 + 趋势顺势加仓
        - TP1 / TP2：部分减仓（负数）
        - 趋势盈利加仓：最多2次，每次0.3*base_position
        """
        if bool(self.classification_opt_only):
            return None
        if trade.id is None or not self.dp:
            return None

        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if df is None or len(df) == 0:
            return None
        row = df.iloc[-1]

        ts = row.get("trend_score", 0.0)
        rs = row.get("range_score", None)
        try:
            trend_score = 0.0 if ts is None or np.isnan(ts) else float(ts)
        except Exception:
            trend_score = 0.0
        try:
            range_score = 1.0 - trend_score if (rs is None or np.isnan(rs)) else float(rs)
        except Exception:
            range_score = 1.0 - trend_score

        # 初始化基础仓位（首次开仓的 stake）
        base_pos = trade.get_custom_data("base_position", None)
        if base_pos is None:
            base_pos = float(trade.stake_amount)
            trade.set_custom_data("base_position", base_pos)
        else:
            base_pos = float(base_pos)

        # --- 分批止盈 ---
        tp1 = range_score * 0.04 + trend_score * 0.08
        tp2 = range_score * 0.08 + trend_score * 0.15
        tp1_done = bool(trade.get_custom_data("tp1_done", False))
        tp2_done = bool(trade.get_custom_data("tp2_done", False))

        # 趋势强时延后 TP1，避免过早卖飞趋势利润
        if trend_score > 0.7:
            tp1 = tp1 * (1.0 + trend_score)

        if current_profit > tp1 and not tp1_done:
            if range_score > 0.7:
                ratio = float(self.tp1_ratio_range_strong)
            elif trend_score > 0.7:
                ratio = float(self.tp1_ratio_trend_strong)
            else:
                ratio = float(self.tp1_ratio_default.value)
            trade.set_custom_data("tp1_done", True)
            return -float(trade.stake_amount) * ratio

        if current_profit > tp2 and not tp2_done:
            ratio = float(self.tp2_ratio_default.value)
            trade.set_custom_data("tp2_done", True)
            return -float(trade.stake_amount) * ratio

        # --- 趋势盈利加仓（仅趋势仓）；用解析后的英文键判断，兼容中文 enter_tag ---
        fs_a, st_a, tg_a = self._parse_entry_tag(trade.enter_tag)
        if "trend" not in f"{fs_a}|{st_a}|{tg_a}".lower():
            return None

        if trend_score < float(self.trend_add_min_score_1.value):
            return None

        add_times = int(trade.get_custom_data("trend_add_times", 0))
        if add_times >= int(self.trend_add_max_times):
            return None

        total_cap = base_pos * float(self.trend_total_max_mult)
        remaining_cap = total_cap - float(trade.stake_amount)
        if remaining_cap <= 0:
            return None

        trigger = None
        if add_times == 0 and current_profit > float(self.trend_add_profit_1.value) and trend_score > float(self.trend_add_min_score_1.value):
            trigger = 1
        elif add_times == 1 and current_profit > float(self.trend_add_profit_2.value) and trend_score > float(self.trend_add_min_score_2.value):
            trigger = 2
        if trigger is None:
            return None

        add_amt = base_pos * float(self.trend_add_size_mult)
        add_amt = min(add_amt, remaining_cap, max_stake)
        if min_stake is not None and add_amt < min_stake:
            return None
        if add_amt <= 0:
            return None

        trade.set_custom_data("trend_add_times", add_times + 1)
        return add_amt

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs: Any,
    ) -> float:
        lev = min(float(self.leverage_target), float(max_leverage), float(self.leverage_max))

        # 高波动主动降杠杆（减少清算风险）
        try:
            if self.dp:
                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if df is not None and len(df):
                    ar = df.iloc[-1].get("atr_ratio", 0.0)
                    atr_ratio = 0.0 if ar is None or np.isnan(ar) else float(ar)

                    high = float(self.atr_ratio_high.value)
                    low = float(self.atr_ratio_min_open.value)

                    # atr_ratio 高于 high 开始线性降到 1.5x；再高则降到 1.0x
                    if atr_ratio > high and high > 0:
                        # 在 [high, 2*high] 区间把系数从 1 -> 0.5
                        x = min(1.0, (atr_ratio - high) / high)
                        lev = min(lev, 3.0 - 1.5 * x)  # 3.0 -> 1.5
                    if atr_ratio > 2.0 * high and high > 0:
                        lev = min(lev, 1.0)

                    # 低波动时也不必给太高杠杆（防止“磨损+长持仓”）
                    if atr_ratio < low:
                        lev = min(lev, 2.0)
        except Exception:
            pass

        return max(1.0, float(lev))

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs: Any,
    ) -> float:
        if bool(self.classification_opt_only):
            # 分类优化阶段：固定仓位（不启用 trend/range/mixed 分层、ATR仓位、连亏降仓、市场级缩放）
            currency = self.config.get("stake_currency", "USDT")
            total = float(self.wallets.get_total(currency))
            raw = (int(total) // 1000) * 10
            stake = proposed_stake if raw <= 0 else float(raw)
            lo = min_stake if min_stake is not None else 0.0
            stake = max(stake, lo)
            return min(float(stake), float(max_stake))

        pair_loss_streak = int(self._pair_loss_streak.get(pair, 0))
        currency = self.config.get("stake_currency", "USDT")
        total = float(self.wallets.get_total(currency))
        raw = (int(total) // 1000) * 10
        if raw <= 0:
            stake = proposed_stake
        else:
            stake = float(raw)

        # 趋势越强，给的仓位越大：仓位系数 = 0.5 + trend_score
        trend_w = 0.5
        try:
            if self.dp:
                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if df is not None and len(df):
                    ts = df.iloc[-1].get("trend_score", None)
                    if ts is not None and not np.isnan(ts):
                        trend_w = 0.5 + float(ts)
        except Exception:
            trend_w = 0.5

        base = stake
        stake = base * trend_w
        stake = min(stake, base * float(self.stake_max_mult))

        # V3：按 market_state 分层仓位（trend / range / mixed）
        market_state = "mixed"
        try:
            if self.dp:
                df2, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if df2 is not None and len(df2):
                    ms = df2.iloc[-1].get("market_state", "mixed")
                    market_state = "mixed" if ms is None else str(ms)
        except Exception:
            market_state = "mixed"

        final_state = "choppy"
        try:
            if self.dp:
                df3, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if df3 is not None and len(df3):
                    fs = df3.iloc[-1].get("final_state", "choppy")
                    final_state = "choppy" if fs is None else str(fs)
        except Exception:
            final_state = "choppy"

        if final_state in ("strong_trend_long", "strong_trend_short"):
            stake = stake * 1.2
        elif final_state in ("pullback_long", "pullback_short"):
            stake = stake * 1.0
        elif final_state == "pure_range":
            stake = stake * 0.6
        else:
            stake = 0.0

        # 波动率联动仓位：vol_factor = clamp(atr_ratio / target_vol, 0.5, 1.2)
        try:
            atr_ratio = 0.0
            if self.dp:
                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if df is not None and len(df):
                    ar = df.iloc[-1].get("atr_ratio", None)
                    if ar is not None and not np.isnan(ar):
                        atr_ratio = float(ar)
            tar = float(self.target_atr_ratio.value)
            vol_factor = atr_ratio / tar if tar > 0 else 1.0
            vol_factor = max(float(self.atr_pos_min_mult), min(float(self.atr_pos_max_mult), vol_factor))
            stake = stake * vol_factor
        except Exception:
            pass

        # 连续亏损降仓：risk_factor = max(0.3, 1 - loss_streak*0.2)
        risk_factor = max(
            float(self.loss_streak_floor),
            1.0 - float(pair_loss_streak) * float(self.loss_streak_step),
        )
        stake = stake * risk_factor

        # 市场级仓位调节（BTC atr_ratio）
        stake = stake * self._get_market_vol_factor()
        # BTC 方向过滤：熊市时仅缩减多单仓位，空单不缩
        if side == "long" and self._is_market_bear():
            stake = stake * float(self.market_bear_long_mult)

        # 硬上限保护：避免多重系数叠加后仓位过度放大
        stake = min(stake, base * 2.0)

        lo = min_stake if min_stake is not None else 0.0
        stake = max(stake, lo)
        return min(stake, max_stake)

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs: Any,
    ) -> bool:
        if bool(self.classification_opt_only):
            # 分类优化阶段：关闭冷却/连亏等额外限制，只保留“同K不同模式禁止第二笔”的状态锁
            prev = self._entry_mode_this_candle.get(pair)
            t_new = (entry_tag or "").lower()
            if prev is not None:
                pt, ptag = prev
                t_old = (ptag or "").lower()
                if pt == current_time and t_new and t_old and t_new != t_old:
                    return False
            self._entry_mode_this_candle[pair] = (current_time, entry_tag)
            return True

        loss_cd = self._pair_loss_cooldown_until.get(pair)
        if loss_cd is not None and current_time < loss_cd:
            return False

        until = self._exit_cooldown_until.get(pair)
        if until is not None and current_time < until:
            return False

        prev = self._entry_mode_this_candle.get(pair)
        t_new = (entry_tag or "").lower()
        if prev is not None:
            pt, ptag = prev
            t_old = (ptag or "").lower()
            if pt == current_time and t_new and t_old and t_new != t_old:
                return False

        self._entry_mode_this_candle[pair] = (current_time, entry_tag)
        return True

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
        **kwargs: Any,
    ) -> bool:
        if bool(self.classification_opt_only):
            return True
        if amount >= trade.amount * 0.999:
            self._exit_cooldown_until[pair] = current_time + timedelta(
                hours=self.cooldown_hours_after_exit
            )

            # 连续亏损计数：亏损+1，盈利则-1（逐步恢复，不直接归0）
            pair_key = trade.pair
            cur = int(self._pair_loss_streak.get(pair_key, 0))
            profit = float(trade.calc_profit_ratio(rate))
            if profit < 0:
                cur += 1
            else:
                cur = max(0, cur - 1)
            self._pair_loss_streak[pair_key] = cur

            # 严格风控：仅在偏震荡环境下，连亏>=阈值触发额外冷却
            tscore = 0.0
            try:
                if self.dp:
                    df, _ = self.dp.get_analyzed_dataframe(pair_key, self.timeframe)
                    if df is not None and len(df):
                        val = df.iloc[-1].get("trend_score", 0.0)
                        tscore = 0.0 if val is None or np.isnan(val) else float(val)
            except Exception:
                tscore = 0.0
            if cur >= int(self.loss_cooldown_streak) and tscore < float(self.strict_risk_trend_cutoff):
                mins = timeframe_to_minutes(self.timeframe) * int(self.loss_cooldown_candles)
                self._pair_loss_cooldown_until[pair_key] = current_time + timedelta(minutes=mins)

            # 交易诊断 + 评分 + 自进化更新
            try:
                open_u = trade.open_date_utc
                dur_min = 0.0 if open_u is None else max(0.0, (current_time - open_u).total_seconds() / 60.0)
                self._record_trade_feedback(
                    entry_tag=trade.enter_tag,
                    profit_ratio=profit,
                    trade_duration_min=dur_min,
                )
                self._evolve_structure_weights()
            except Exception:
                pass
        return True
