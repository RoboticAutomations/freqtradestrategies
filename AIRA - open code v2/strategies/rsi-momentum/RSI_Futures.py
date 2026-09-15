# pragma pylint: disable=missing-docstring, invalid-name
# RSI 多周期波动率自适应策略（期货 / 杠杆）
# 依据《策略逻辑.txt》与 Freqtrade 官方文档实现
# https://www.freqtrade.io/en/stable/
#版本4
# 开发规则 1.6：
# - 数据计算要向量化：指标与入场/出场信号均用整列或滚动向量运算，ART 用 sliding_window_view + numpy。
# - 标签要中文：enter_tag、custom_exit 返回值等对外展示的标签一律使用中文。
# - populate_* 在回测时每对只调用一次；custom_stoploss/custom_exit 为逐 K 线回调，无法向量化。
#
# TA-Lib 使用约定（避免 TypeError / 返回值误用）：
# - 多输出指标（如 BBANDS）返回元组，需解包：upper, middle, lower = ta.BBANDS(...)，不可用 ["upperband"]。
# - 数值型可选参数（如 nbdevup/nbdevdn）传 float：nbdevup=2.0, nbdevdn=2.0；timeperiod 保持 int。
# - ATR 需要 high/low/close，传入完整 dataframe 或含三列的 DataFrame，不可只传 close。

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy
from freqtrade.strategy.parameters import BooleanParameter, DecimalParameter
try:
    # 可选加速：若环境中安装了 numba，则使用 njit 加速；否则退回普通 Python 实现
    from numba import njit
except ImportError:
    def njit(*dargs, **dkwargs):
        """
        回退占位：环境无 numba 时直接返回原函数。
        兼容两种使用方式：
        - @njit
        - @njit(...)
        """
        # 直接 @njit 装饰：njit(func)
        if dargs and callable(dargs[0]) and len(dargs) == 1 and not dkwargs:
            return dargs[0]

        # 带参数形式 @njit(...)
        def decorator(func):
            return func

        return decorator

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 常量：与策略逻辑清单一致
# -----------------------------------------------------------------------------
LEVERAGE = 5.0  # 固定 5 倍杠杆
# 下单保证金 = (账户总额 // 500 + 1) × 10
STAKE_DIVISOR = 500
STAKE_BASE = 10
# 止损为 ROE（收益率）= 币价涨跌幅度 × 杠杆；Freqtrade 期货下 stoploss/current_profit 即按 ROE 比较
TRAILING_ACTIVATION_ROE = 0.2  # 震荡/趋势 RSI 移动止盈：历史最高 ROE ≥ 20% 时启用
TREND_ROE_TRAIL_ACTIVATION_ROE = 0.3  # 趋势盈利回撤表：历史最高 ROE ≥ 30% 时启用（策略逻辑 5.2）
BASE_ENTRY_PCT = 0.02  # 追踪建仓基础阈值 2%
BASE_RANGE_EXTREME_PCT = 0.02  # 震荡极值止盈基础阈值 2%
BASE_ROE_TRAIL_PCT = 0.20  # 加仓后 ROE 盈利回撤基础阈值 20%
TREND_TRAIL_PCT = 0.05  # 趋势 RSI 极值后盈利回撤 5% 平仓
RSI_1D_TREND_HIGH = 60
RSI_1D_TREND_LOW = 40
RSI_1D_CONFIRM_LONG = 62
RSI_1D_CONFIRM_SHORT = 38
RSI_1D_EXTREME_LONG = 25
RSI_1D_EXTREME_SHORT = 75
RSI_1D_EXIT_LONG = 70
RSI_1D_EXIT_SHORT = 30
RSI_1D_TREND_ENTRY_BLOCK_LOW = 45  # 趋势订单禁止开仓区间下限
RSI_1D_TREND_ENTRY_BLOCK_HIGH = 55  # 趋势订单禁止开仓区间上限
# 趋势订单极限值开仓
RSI_1D_TREND_EXTREME_LONG = 20  # 趋势多极限值：RSI1D ≤ 20 直接做多
RSI_1D_TREND_EXTREME_SHORT = 80  # 趋势空极限值：RSI1D ≥ 80 直接做空
# 趋势多新增条件
RSI_1D_TREND_LONG_TRACK_LOW = 30  # 趋势多追踪建仓：RSI1D < 30时开始追踪
RSI_1D_TREND_LONG_TRACK_ENTRY = 32  # 趋势多追踪建仓：从最低反弹到32时开仓
RSI_1D_TREND_LONG_CROSS_40_LOW = 40  # 趋势多上穿40下限
RSI_1D_TREND_LONG_CROSS_40_HIGH = 42  # 趋势多上穿40上限
RSI_1D_TREND_LONG_CROSS_60_LOW = 60  # 趋势多上穿60下限
RSI_1D_TREND_LONG_CROSS_60_HIGH = 62  # 趋势多上穿60上限
# 趋势空新增条件
RSI_1D_TREND_SHORT_TRACK_HIGH = 70  # 趋势空追踪建仓：RSI1D > 70时开始追踪
RSI_1D_TREND_SHORT_TRACK_ENTRY = 68  # 趋势空追踪建仓：从最高回落到68时开仓
RSI_1D_TREND_SHORT_CROSS_60_LOW = 58  # 趋势空下穿60下限
RSI_1D_TREND_SHORT_CROSS_60_HIGH = 60  # 趋势空下穿60上限
RSI_1D_TREND_SHORT_CROSS_40_LOW = 38  # 趋势空下穿40下限
RSI_1D_TREND_SHORT_CROSS_40_HIGH = 40  # 趋势空下穿40上限
RSI_1H_OVERBOUGHT = 75
RSI_1H_OVERSOLD = 25
RSI_1H_EXTREME_LONG = 10
RSI_1H_EXTREME_SHORT = 90
RSI_1H_RANGE_EXIT_LONG = 85
RSI_1H_RANGE_EXIT_SHORT = 15
ART_MIN = 0.3
ART_MAX = 1.2
ART_WINDOW = 100  # 100 根 1h K 线
# 1.2 风控与过滤
BIAS_IGNORE_PCT = 0.02  # 乖离率过滤：价格远离 EMA20 超过该比例×ART 则不开仓，防止追高杀低
MAX_SAME_SIDE_LOSING = 2  # 同向订单亏损保护：同向有 ≥2 个持仓 ROE < LOSING_ROE_THRESHOLD 时停止新开该向仓位
LOSING_ROE_THRESHOLD = -0.10  # 同向亏损判定阈值 -10%
ART_EXTREME_VOL_PERCENTILE = 90.0  # 波动率分位 > 此值时启用极波动动态止损
RSI_1D_PERIOD = 14
RSI_12H_CANDLES = 12  # 1h 下 12h = 12 根

# 趋势单 ROE 回撤平仓表（策略逻辑 5.2：历史最高 ROE ≥ 30% 激活，平仓线 = 历史最高 ROE × (1 − 回撤比例 × ART)）
ROE_TRAIL_TABLE = [
    (0.30, 0.50, 0.15),
    (0.50, 1.00, 0.20),
    (1.00, 1.50, 0.25),
    (1.50, 2.00, 0.30),
    (2.00, 2.50, 0.35),
    (2.50, 3.00, 0.40),
    (3.00, 3.50, 0.45),
    (3.50, 4.00, 0.50),
    (4.00, 4.50, 0.55),
    (4.50, 999.0, 0.60),
]

# 震荡模式盈利回撤表（策略逻辑 第六部分 6.5）
RANGE_ROE_TRAIL_TABLE = [
    (0.20, 0.30, 0.05),
    (0.30, 0.40, 0.07),
    (0.40, 0.50, 0.09),
    (0.50, 0.60, 0.11),
    (0.60, 0.70, 0.13),
    (0.70, 0.80, 0.15),
    (0.80, 0.90, 0.17),
    (0.90, 1.00, 0.19),
    (1.00, 999.0, 0.21),
]


def _rsi_sma(close: pd.Series, period: int = 14) -> pd.Series:
    """标准 SMA 平滑 RSI（策略逻辑 第二部分）。"""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    avg_up = up.rolling(period, min_periods=1).mean()
    avg_down = down.rolling(period, min_periods=1).mean()
    rs = avg_up / avg_down.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def _art_multiplier_from_quantile(quantile_pct: float) -> float:
    """
    波动率分位数 -> ART 倍数 [0.3, 1.2]。
    调整为：波动率越高，ART 倍数越大，从而提高入场确认幅度，过滤极端行情噪音。
    """
    q = max(0.0, min(100.0, quantile_pct))
    if q <= 30.0:
        # 低波动：0.3 ~ 0.5
        mult = 0.3 + (q / 30.0) * 0.2
    elif q <= 70.0:
        # 中等波动：0.5 ~ 0.9
        mult = 0.5 + ((q - 30.0) / 40.0) * 0.4
    else:
        # 高波动：0.9 ~ 1.2
        mult = 0.9 + ((q - 70.0) / 30.0) * 0.3
    return max(ART_MIN, min(ART_MAX, mult))


def _art_multiplier_vectorized(quantile_pct: pd.Series) -> pd.Series:
    """向量化：波动率分位数 -> ART 倍数 [0.3, 1.2]，高波动 → 更高倍数。"""
    q = quantile_pct.clip(0.0, 100.0)
    mult = np.where(
        q <= 30.0,
        0.3 + (q / 30.0) * 0.2,  # 0.3 ~ 0.5
        np.where(
            q <= 70.0,
            0.5 + ((q - 30.0) / 40.0) * 0.4,  # 0.5 ~ 0.9
            0.9 + ((q - 70.0) / 30.0) * 0.3,  # 0.9 ~ 1.2
        ),
    )
    return pd.Series(np.clip(mult, ART_MIN, ART_MAX), index=quantile_pct.index)


def _roe_trail_drawdown(max_roe: float, table: list) -> float:
    """根据历史最高 ROE 查表得到允许回撤比例。"""
    for low, high, draw in table:
        if low <= max_roe < high:
            return draw
    return table[-1][2]


@njit
def _rsi_1d_numba(
    daily_last_close: np.ndarray,
    day_codes: np.ndarray,
    closes_1h: np.ndarray,
    period: int,
) -> np.ndarray:
    """
    使用 Numba 加速的 RSI(1D) 计算，逻辑与 populate_indicators 中一致：
    每根 1h K 线用「过去 period-1 日完整收盘 + 当前 1h close 作为当日」计算日线 RSI。
    """
    n = len(closes_1h)
    rsi_1d_arr = np.full(n, 50.0)
    for i in range(n):
        d_code = int(day_codes[i])
        if d_code + 1 < period:
            rsi_1d_arr[i] = 50.0
            continue
        start_d = d_code - period + 1
        sum_up = 0.0
        sum_down = 0.0
        prev = daily_last_close[start_d]
        for j in range(start_d + 1, d_code):
            curr = daily_last_close[j]
            d = curr - prev
            if d > 0:
                sum_up += d
            else:
                sum_down -= d
            prev = curr
        curr = closes_1h[i]
        d = curr - prev
        if d > 0:
            sum_up += d
        else:
            sum_down -= d
        num_deltas = period - 1
        avg_up = sum_up / num_deltas
        avg_down = sum_down / num_deltas
        rs = avg_up / (avg_down if avg_down != 0 else 1e-10)
        rsi_1d_arr[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi_1d_arr


@njit
def _calculate_range_tracking_numba(
    closes: np.ndarray,
    rsi_1h_arr: np.ndarray,
    rsi_low_arr: np.ndarray,
    rsi_high_arr: np.ndarray,
    entry_pct: float,
) -> tuple:
    """
    使用 numba 加速的震荡追踪建仓状态机。
    逻辑保持与原 for 循环一致：
    - RSI(1h) 进入超卖/超买后，追踪观察期以来的最低/最高收盘价；
    - 当前收盘价相对基准反弹/回撤达到 entry_pct 即视为“触发”，并重置观察；
    - 若 RSI 回到中性区且未触发，则放弃本次观察，重置。
    """
    n = len(closes)
    track_low_close = np.empty(n, dtype=np.float64)
    track_high_close = np.empty(n, dtype=np.float64)

    for i in range(n):
        track_low_close[i] = np.nan
        track_high_close[i] = np.nan

    long_active = False
    short_active = False
    min_close = np.nan
    max_close = np.nan

    for i in range(n):
        c = closes[i]
        rsi1h = rsi_1h_arr[i]

        # 做多观察期启动/更新：RSI(1h) < oversold
        if rsi_low_arr[i]:
            if not long_active:
                long_active = True
                min_close = c
            else:
                if c < min_close:
                    min_close = c

        # 做空观察期启动/更新：RSI(1h) > overbought
        if rsi_high_arr[i]:
            if not short_active:
                short_active = True
                max_close = c
            else:
                if c > max_close:
                    max_close = c

        # 将基准写入列
        if long_active and not np.isnan(min_close):
            track_low_close[i] = min_close
        if short_active and not np.isnan(max_close):
            track_high_close[i] = max_close

        # 重置：RSI 回到中性区且尚未触发
        if (
            long_active
            and not np.isnan(rsi1h)
            and rsi1h >= 50.0
            and (np.isnan(min_close) or c < min_close * (1.0 + entry_pct))
        ):
            long_active = False
            min_close = np.nan
        if (
            short_active
            and not np.isnan(rsi1h)
            and rsi1h <= 50.0
            and (np.isnan(max_close) or c > max_close * (1.0 - entry_pct))
        ):
            short_active = False
            max_close = np.nan

        # 触发后重置（使下一次追踪独立）
        if long_active and not np.isnan(min_close) and c >= min_close * (1.0 + entry_pct):
            long_active = False
            min_close = np.nan
        if short_active and not np.isnan(max_close) and c <= max_close * (1.0 - entry_pct):
            short_active = False
            max_close = np.nan

    return track_low_close, track_high_close


@njit
def _trend_rsi1d_tracking_numba(
    rsi_1d_arr: np.ndarray,
    track_low_threshold: float,
    track_entry_threshold: float,
    track_high_threshold: float,
    track_entry_high_threshold: float,
) -> tuple:
    """
    使用 Numba 加速的趋势订单 RSI1D 追踪建仓状态机。
    - 趋势多：RSI1D < track_low_threshold(30) 时开始追踪最低值，从最低反弹到 track_entry_threshold(32) 时触发
    - 趋势空：RSI1D > track_high_threshold(70) 时开始追踪最高值，从最高回落到 track_entry_high_threshold(68) 时触发
    """
    n = len(rsi_1d_arr)
    track_long_ready = np.zeros(n, dtype=np.bool_)
    track_short_ready = np.zeros(n, dtype=np.bool_)
    
    long_active = False
    short_active = False
    min_rsi = np.nan
    max_rsi = np.nan
    
    for i in range(n):
        rsi = rsi_1d_arr[i]
        if np.isnan(rsi):
            continue
            
        # 趋势多追踪：RSI1D < 30 时开始追踪最低值
        if rsi < track_low_threshold:
            if not long_active:
                long_active = True
                min_rsi = rsi
            else:
                if rsi < min_rsi:
                    min_rsi = rsi
        elif long_active:
            # 如果已经追踪到最低值，且当前RSI反弹到32，触发开仓
            if not np.isnan(min_rsi) and rsi >= track_entry_threshold:
                track_long_ready[i] = True
                long_active = False
                min_rsi = np.nan
            elif rsi >= 50.0:
                # 如果RSI回到50以上且未触发，重置追踪
                long_active = False
                min_rsi = np.nan
            
        # 趋势空追踪：RSI1D > 70 时开始追踪最高值
        if rsi > track_high_threshold:
            if not short_active:
                short_active = True
                max_rsi = rsi
            else:
                if rsi > max_rsi:
                    max_rsi = rsi
        elif short_active:
            # 如果已经追踪到最高值，且当前RSI回落到68，触发开仓
            if not np.isnan(max_rsi) and rsi <= track_entry_high_threshold:
                track_short_ready[i] = True
                short_active = False
                max_rsi = np.nan
            elif rsi <= 50.0:
                # 如果RSI回到50以下且未触发，重置追踪
                short_active = False
                max_rsi = np.nan
    
    return track_long_ready, track_short_ready


@njit
def _trend_cooling_allowed_numba(
    cross_d: np.ndarray,
    cross_u: np.ndarray,
    align_long: np.ndarray,
    align_short: np.ndarray,
) -> tuple:
    """
    使用 Numba 加速的趋势冷却状态机：先 EMA50/100 相交再并列才允许开仓。
    返回 (allowed_long, allowed_short) 布尔数组。
    """
    n = len(cross_d)
    allowed_long = np.ones(n, dtype=np.bool_)
    allowed_short = np.ones(n, dtype=np.bool_)
    state_long = 0  # 0=allow, 1=need_realign
    state_short = 0
    for i in range(n):
        if state_long == 1 and align_long[i]:
            state_long = 0
        if cross_d[i]:
            state_long = 1
        if state_long != 0:
            allowed_long[i] = False

        if state_short == 1 and align_short[i]:
            state_short = 0
        if cross_u[i]:
            state_short = 1
        if state_short != 0:
            allowed_short[i] = False
    return allowed_long, allowed_short


class RSI_Futures(IStrategy):
    """
    RSI 多周期 + ART 动态波动率策略（期货/杠杆）。
    主周期 1h，辅助 1D（RSI(1D) 由 1h 按日聚合）；固定 5 倍杠杆；趋势/震荡双模式。

    需追踪的信号与持久化（Freqtrade 框架保证）：
    - enter_tag：开仓时由 populate_entry_trend 写入 dataframe，框架存入 Trade.enter_tag 并持久化到库；custom_stoploss / custom_exit 依赖此字段区分趋势/震荡。
    - exit_reason：custom_exit 返回值或 populate_exit_trend 的 exit_tag 经 custom_exit 返回后，由框架写入 Trade.exit_reason，平仓时持久化。
    - max_roe_reached：custom_exit 内通过 Trade.set_custom_data("max_roe_reached", roe) 写入，框架以 JSON 持久化到库，重启后 get_custom_data 可恢复，用于 ROE 回撤表与趋势移动止盈。
    """

    INTERFACE_VERSION = 3
    timeframe = "1h"
    max_open_trades = 8  # 风控：限制最大同时持仓数量，避免全市场同向极端风险
    can_short = True
    use_custom_stoploss = False  # 关闭后框架按静态 stoploss=-0.30 触发，避免 trailing 导致提前平仓；custom_exit 仍作 -30% 判断备份
    # Dry Run / Live 若希望在单根 K 线内更高频率追踪 ROE 回撤，
    # 需配合 config.json: process_only_new_candles=false + 更小的 process_throttle_secs
    process_only_new_candles = False
    # 官方文档：需满足最大指标预热，ema_200=200、ART_WINDOW=100、RSI(1D) 约 14 日；回测/实盘会据此裁剪或拉取数据
    startup_candle_count = 400

    # 回测阶段将 ROI 兜底目标设为极大值，尽量让盈利平仓完全交由 custom_exit 控制
    minimal_roi = {"0": 100.0}
    stoploss = -0.30  # 全局固定止损 ROE -30%（5 倍杠杆下 ≈ 6% 价格）；真实触发出 custom_exit 判断
    trailing_stop = False

    # 趋势平仓后冷却：同方向须先 1h EMA50/100 相交再并列才允许再开仓；(pair, "long"|"short") -> "need_cross"|"need_realign"|None
    _trend_exit_wait_state: Dict[Tuple[str, str], Optional[str]] = {}
    # 震荡极限平仓后的“同根K线反手”信号：
    # (pair, candle_open_time) -> "long" | "short"
    # - 由 custom_exit 在返回 "震荡RSI极值平多/平空" 时写入
    # - 由 populate_entry_trend 在同一根K线的最后一行读出并发出反向入场信号
    # - 由 confirm_trade_entry 在反手下单被允许后清除，防止重复开仓
    _range_flip_signal: Dict[Tuple[str, datetime], str] = {}
    # 趋势极限平仓后的“同根K线反手”信号：
    # (pair, candle_open_time) -> "long" | "short"
    # - 由 custom_exit 在触发趋势 RSI 极值平仓时写入
    # - 由 populate_entry_trend 在同一根K线的最后一行读出并发出反向入场信号
    # - 由 confirm_trade_entry 在反手下单被允许后清除，防止重复开仓
    _trend_flip_signal: Dict[Tuple[str, datetime], str] = {}

    def _is_backtest(self) -> bool:
        """自动区分运行模式：仅当明确为回测时返回 True，dry_run / live 返回 False，不影响实盘与模拟。"""
        try:
            runmode = getattr(self.dp, "runmode", None)
            if runmode is None:
                return False
            return getattr(runmode, "name", "") == "BACKTEST"
        except Exception:
            return False

    # 可调参数（便于超参优化）
    base_entry_pct = DecimalParameter(0.005, 0.02, default=BASE_ENTRY_PCT, space="buy", optimize=True)
    base_range_extreme_pct = DecimalParameter(0.01, 0.04, default=BASE_RANGE_EXTREME_PCT, space="sell", optimize=True)
    rsi_1h_oversold = DecimalParameter(20, 30, default=RSI_1H_OVERSOLD, space="buy", optimize=True)
    rsi_1h_overbought = DecimalParameter(70, 80, default=RSI_1H_OVERBOUGHT, space="buy", optimize=True)

    # -------------------------------------------------------------------------
    # 杠杆与仓位（固定 5 倍杠杆 / 策略逻辑 1.5）
    # -------------------------------------------------------------------------
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return min(LEVERAGE, max_leverage)

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                             proposed_stake: float, min_stake: Optional[float], max_stake: float,
                             leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        自定义下单保证金：下单保证金 = (账户总额 // 500 + 1) × 10。
        结果会被 min_stake / max_stake 约束，保证不超出交易所限制。
        """
        total = float(self.wallets.get_total_stake_amount() or 0)
        stake = (int(total) // STAKE_DIVISOR + 1) * STAKE_BASE

        # 统一应用交易所最小/最大下单限制
        stake_capped = min(float(stake), float(max_stake))
        if min_stake is not None:
            stake_capped = max(float(min_stake), stake_capped)
        return stake_capped

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """主周期 1h：EMA、RSI、ATR、BB、ART、RSI(1D)（策略逻辑 第一～三部分）；所有 EMA 均为 1h 周期。"""
        # ----- 1h 基础指标（策略逻辑 1.4）-----
        dataframe["ema_20"] = ta.EMA(dataframe["close"], timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe["close"], timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe["close"], timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe["close"], timeperiod=200)
        # EMA20 斜率：趋势 EMA 排列时仅在近 6 根 1h 平均斜率方向一致时允许开仓
        dataframe["ema20_slope_1h"] = dataframe["ema_20"].diff()
        dataframe["ema20_slope_6h"] = dataframe["ema20_slope_1h"].rolling(6, min_periods=6).mean()
        dataframe["rsi_1h"] = ta.RSI(dataframe["close"], timeperiod=14)
        macd, macdsignal, macdhist = ta.MACD(
            dataframe["close"], fastperiod=12, slowperiod=26, signalperiod=9
        )
        dataframe["macd_hist"] = macdhist
        ohlc = {"high": dataframe["high"], "low": dataframe["low"], "close": dataframe["close"]}
        dataframe["atr_1h"] = ta.ATR(ohlc, timeperiod=14)
        upper, middle, lower = ta.BBANDS(
            dataframe["close"], timeperiod=20, nbdevup=2.0, nbdevdn=2.0
        )
        dataframe["bb_upper"] = upper
        dataframe["bb_middle"] = middle
        dataframe["bb_lower"] = lower
        # 1.2 震荡入场成交量过滤：Volume > MA(Volume, 20)
        dataframe["volume_ma_20"] = dataframe["volume"].rolling(20, min_periods=1).mean()

        # ----- ART 动态波动率（策略逻辑 第三部分，100 根 1h K 线，向量化）-----
        atr_pct = (dataframe["atr_1h"] / dataframe["close"].replace(0, 1e-10)) * 100
        bb_width_pct = (
            (dataframe["bb_upper"] - dataframe["bb_lower"])
            / dataframe["bb_middle"].replace(0, 1e-10)
        ) * 100
        window = ART_WINDOW
        def _rolling_percentile_rank(series: pd.Series, w: int) -> pd.Series:
            n = len(series)
            if n < w:
                return pd.Series(50.0, index=series.index)
            arr = np.asarray(series, dtype=float)
            windows = np.lib.stride_tricks.sliding_window_view(arr, w)
            current = windows[:, -1]
            rank_pct = (windows <= current[:, np.newaxis]).sum(axis=1) / w * 100
            out = np.full(n, 50.0, dtype=float)
            out[w - 1 :] = rank_pct
            return pd.Series(out, index=series.index)

        atr_rank = _rolling_percentile_rank(atr_pct, window)
        bb_rank = _rolling_percentile_rank(bb_width_pct, window)
        vol_quantile = atr_rank.fillna(50).combine(bb_rank.fillna(50), max)
        dataframe["art_multiplier"] = _art_multiplier_vectorized(vol_quantile)
        dataframe["vol_quantile_pct"] = vol_quantile  # 1.2 极波动止损用

        # ----- RSI(1D) 实时合成版：避免未来函数 -----
        # 目标：
        # - 仍然以“自然日”为单位，使用最近 RSI_1D_PERIOD 个自然日的收盘价计算日线 RSI；
        # - 但对“当日”不再使用该日最后一根 1h 的收盘价，而是使用“当前这根 1h 的收盘价”作为当日临时收盘；
        # - 这样在回测中任意时刻只使用“当前时刻以及之前”的价格信息，不再预知当日 23:00 收盘价，消除未来函数。
        if not dataframe.empty:
            dates_1h = pd.to_datetime(
                dataframe["date"] if "date" in dataframe.columns else dataframe.index,
                utc=True,
            )
            day_1h = pd.Series(dates_1h, index=dataframe.index).dt.normalize()

            # 自然日序列及其“完整日收盘价”（用于历史已结束的自然日）
            # 注意：这里 daily_last_close 只在“当前日之前的日线”上使用；当前日的收盘价使用当前 K 线的 close。
            daily_last = dataframe.assign(_day=day_1h).groupby("_day").agg({"close": "last"})
            unique_days = daily_last.index.to_numpy()
            daily_last_close = daily_last["close"].to_numpy(dtype=float)

            # 为每根 1h K 线映射其所在自然日的索引（0, 1, 2, ...）
            # np.searchsorted 保证与 unique_days 顺序一致
            day_values = day_1h.to_numpy()
            day_codes = np.searchsorted(unique_days, day_values)

            n = len(dataframe)
            closes_1h = dataframe["close"].astype(float).to_numpy(copy=False)
            day_codes_arr = day_codes.astype(np.int64)
            rsi_1d_arr = _rsi_1d_numba(
                daily_last_close, day_codes_arr, closes_1h, RSI_1D_PERIOD
            )
            dataframe["rsi_1d"] = rsi_1d_arr
        else:
            dataframe["rsi_1d"] = 50.0

        # ----- 趋势确认与 12h RSI 波动（策略逻辑 第五部分 5.1）-----
        # 趋势开仓条件：仅看 1h EMA 均线排列。趋势多 EMA20>EMA50>EMA100>EMA200，趋势空相反。
        dataframe["trend_long_align"] = (
            (dataframe["ema_20"] > dataframe["ema_50"])
            & (dataframe["ema_50"] > dataframe["ema_100"])
            & (dataframe["ema_100"] > dataframe["ema_200"])
        )
        dataframe["trend_short_align"] = (
            (dataframe["ema_20"] < dataframe["ema_50"])
            & (dataframe["ema_50"] < dataframe["ema_100"])
            & (dataframe["ema_100"] < dataframe["ema_200"])
        )
        # 1h EMA50 与 EMA100 相交：平仓后需先相交再并列才允许同向再开仓（策略逻辑 5.2 冷却）
        dataframe["ema50_4h_cross_down"] = (
            (dataframe["ema_50"] < dataframe["ema_100"])
            & (dataframe["ema_50"].shift(1) >= dataframe["ema_100"].shift(1))
        )
        dataframe["ema50_4h_cross_up"] = (
            (dataframe["ema_50"] > dataframe["ema_100"])
            & (dataframe["ema_50"].shift(1) <= dataframe["ema_100"].shift(1))
        )
        rsi_1d = dataframe["rsi_1d"]
        dataframe["rsi_1d_range_12h"] = (
            rsi_1d.rolling(RSI_12H_CANDLES, min_periods=RSI_12H_CANDLES).max()
            - rsi_1d.rolling(RSI_12H_CANDLES, min_periods=RSI_12H_CANDLES).min()
        )
        # 锚点：12h 窗口起点 RSI(1D)，用于趋势多锚点动态触发（v2.2_Plus）
        dataframe["rsi_anchor_active"] = dataframe["rsi_1d"].shift(RSI_12H_CANDLES)

        # ----- 震荡追踪建仓：极值追踪基于“收盘价”而非 high/low（用户新规则）-----
        # 规则要点：
        # - 当 RSI(1h) 进入超卖/超买观察区后，实时更新观察期以来的最低/最高“收盘价”作为基准
        # - 当前收盘价相对该基准反弹/回撤达到 base_entry_pct（默认 2%）即确认开仓
        # - 非“窗口移动”追踪：追踪的极值在观察期内持续更新；触发开仓后重置
        rsi_low = dataframe["rsi_1h"] < self.rsi_1h_oversold.value
        rsi_high = dataframe["rsi_1h"] > self.rsi_1h_overbought.value

        closes = dataframe["close"].astype(float).to_numpy(copy=False)
        rsi_low_arr = rsi_low.astype(bool).to_numpy(copy=False)
        rsi_high_arr = rsi_high.astype(bool).to_numpy(copy=False)
        rsi_1h_arr = dataframe["rsi_1h"].astype(float).to_numpy(copy=False)
        entry_pct = float(self.base_entry_pct.value)  # 默认 0.02（2%）

        # 使用 numba 加速的状态机计算追踪极值
        track_low_close, track_high_close = _calculate_range_tracking_numba(
            closes, rsi_1h_arr, rsi_low_arr, rsi_high_arr, entry_pct
        )

        dataframe["range_track_low_close"] = track_low_close
        dataframe["range_track_high_close"] = track_high_close

        # 确认条件：以“收盘价基准”的 2%×ART 反弹/回撤确认
        base_low = dataframe["range_track_low_close"].replace(0, 1e-10)
        base_high = dataframe["range_track_high_close"].replace(0, 1e-10)
        entry_threshold_long = (dataframe["close"] - base_low) / base_low
        entry_threshold_short = (base_high - dataframe["close"]) / base_high
        # 这里的阈值 = base_entry_pct（默认 2%）× ART，使波动率越大，要求的确认幅度越深
        dataframe["range_confirm_long"] = entry_threshold_long >= (self.base_entry_pct.value * dataframe["art_multiplier"])
        dataframe["range_confirm_short"] = entry_threshold_short >= (self.base_entry_pct.value * dataframe["art_multiplier"])

        # ----- 震荡极限追踪建仓：RSI(1h)≤10 / ≥90 启动追踪，等待固定 2% 反弹/回撤确认 -----
        # 多侧：RSI(1h) ≤ 10 时启动/重置多侧观察期，从当根 K 线收盘价开始追踪最低收盘价，
        #       当 (close - min_close) / min_close ≥ 2% 时确认“震荡多_极限追踪建仓”。
        # 空侧：RSI(1h) ≥ 90 时启动/重置空侧观察期，从当根 K 线收盘价开始追踪最高收盘价，
        #       当 (max_close - close) / max_close ≥ 2% 时确认“震荡空_极限追踪建仓”。
        # 观察期在 RSI 回到中性区（多侧：RSI(1h)≥50；空侧：RSI(1h)≤50）但未触发时失效，重置。
        rsi_extreme_low = dataframe["rsi_1h"] <= RSI_1H_EXTREME_LONG
        rsi_extreme_high = dataframe["rsi_1h"] >= RSI_1H_EXTREME_SHORT
        rsi_extreme_low_arr = rsi_extreme_low.astype(bool).to_numpy(copy=False)
        rsi_extreme_high_arr = rsi_extreme_high.astype(bool).to_numpy(copy=False)
        extreme_entry_pct = 0.02  # 固定 2% 反弹/回撤
        extreme_low_close, extreme_high_close = _calculate_range_tracking_numba(
            closes, rsi_1h_arr, rsi_extreme_low_arr, rsi_extreme_high_arr, extreme_entry_pct
        )
        dataframe["range_extreme_track_low_close"] = extreme_low_close
        dataframe["range_extreme_track_high_close"] = extreme_high_close
        base_ext_low = dataframe["range_extreme_track_low_close"].replace(0, 1e-10)
        base_ext_high = dataframe["range_extreme_track_high_close"].replace(0, 1e-10)
        extreme_entry_long = (dataframe["close"] - base_ext_low) / base_ext_low
        extreme_entry_short = (base_ext_high - dataframe["close"]) / base_ext_high
        dataframe["range_extreme_confirm_long"] = extreme_entry_long >= 0.02
        dataframe["range_extreme_confirm_short"] = extreme_entry_short >= 0.02

        # ----- 趋势订单 RSI1D 追踪建仓（新增）-----
        rsi_1d_arr = dataframe["rsi_1d"].astype(float).fillna(50.0).to_numpy(copy=False)
        track_long_ready, track_short_ready = _trend_rsi1d_tracking_numba(
            rsi_1d_arr,
            RSI_1D_TREND_LONG_TRACK_LOW,
            RSI_1D_TREND_LONG_TRACK_ENTRY,
            RSI_1D_TREND_SHORT_TRACK_HIGH,
            RSI_1D_TREND_SHORT_TRACK_ENTRY,
        )
        dataframe["trend_rsi1d_track_long"] = track_long_ready
        dataframe["trend_rsi1d_track_short"] = track_short_ready
        
        # RSI1D 上穿/下穿检测
        rsi_1d_prev = dataframe["rsi_1d"].shift(1).fillna(50.0)
        # 趋势多：RSI1D上穿40到42，或上穿60到62
        dataframe["trend_rsi1d_cross_40_long"] = (
            (rsi_1d_prev < RSI_1D_TREND_LONG_CROSS_40_LOW)
            & (dataframe["rsi_1d"] >= RSI_1D_TREND_LONG_CROSS_40_LOW)
            & (dataframe["rsi_1d"] <= RSI_1D_TREND_LONG_CROSS_40_HIGH)
        )
        dataframe["trend_rsi1d_cross_60_long"] = (
            (rsi_1d_prev < RSI_1D_TREND_LONG_CROSS_60_LOW)
            & (dataframe["rsi_1d"] >= RSI_1D_TREND_LONG_CROSS_60_LOW)
            & (dataframe["rsi_1d"] <= RSI_1D_TREND_LONG_CROSS_60_HIGH)
        )
        # 趋势空：RSI1D下穿60到58，或下穿40到38
        dataframe["trend_rsi1d_cross_60_short"] = (
            (rsi_1d_prev > RSI_1D_TREND_SHORT_CROSS_60_HIGH)
            & (dataframe["rsi_1d"] <= RSI_1D_TREND_SHORT_CROSS_60_HIGH)
            & (dataframe["rsi_1d"] >= RSI_1D_TREND_SHORT_CROSS_60_LOW)
        )
        dataframe["trend_rsi1d_cross_40_short"] = (
            (rsi_1d_prev > RSI_1D_TREND_SHORT_CROSS_40_HIGH)
            & (dataframe["rsi_1d"] <= RSI_1D_TREND_SHORT_CROSS_40_HIGH)
            & (dataframe["rsi_1d"] >= RSI_1D_TREND_SHORT_CROSS_40_LOW)
        )

        return dataframe

    # -------------------------------------------------------------------------
    # 模式判定（策略逻辑 第二部分 2.2）
    # -------------------------------------------------------------------------
    def _is_trend_mode(self, rsi_1d: float) -> bool:
        return rsi_1d >= RSI_1D_TREND_HIGH or rsi_1d <= RSI_1D_TREND_LOW

    def _is_range_mode(self, rsi_1d: float) -> bool:
        return RSI_1D_TREND_LOW < rsi_1d < RSI_1D_TREND_HIGH

    # -------------------------------------------------------------------------
    # 入场（策略逻辑 第五部分 趋势 / 第六部分 震荡）
    # -------------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        rsi_1d = dataframe["rsi_1d"]
        rsi_1h = dataframe["rsi_1h"]
        pair = metadata.get("pair", "")

        # ----- 趋势平仓后冷却：先 1h EMA50/100 相交再并列才允许同向再开仓（Numba 加速）-----
        if "ema50_4h_cross_down" in dataframe.columns:
            cross_d = dataframe["ema50_4h_cross_down"].fillna(False).astype(bool).to_numpy()
            cross_u = dataframe["ema50_4h_cross_up"].fillna(False).astype(bool).to_numpy()
            align_long = dataframe["trend_long_align"].fillna(False).astype(bool).to_numpy()
            align_short = dataframe["trend_short_align"].fillna(False).astype(bool).to_numpy()
            allowed_long, allowed_short = _trend_cooling_allowed_numba(
                cross_d, cross_u, align_long, align_short
            )
            trend_long_allowed = pd.Series(allowed_long, index=dataframe.index)
            trend_short_allowed = pd.Series(allowed_short, index=dataframe.index)
        else:
            trend_long_allowed = pd.Series(True, index=dataframe.index) if len(dataframe) else True
            trend_short_allowed = pd.Series(True, index=dataframe.index) if len(dataframe) else True
        # 实盘/模拟：推进 custom_exit 设置的冷却状态（ROE/RSI 平仓后也须先相交再并列），并覆盖允许位
        # 官方文档：回测时 populate_* 收到全量 df，iloc[-1] 为回测结束行（未来数据），故仅非回测时用 last 更新状态
        if not self._is_backtest() and pair and not dataframe.empty and "ema50_4h_cross_down" in dataframe.columns:
            last = dataframe.iloc[-1]
            for direction, cross_key, align_key, state_key in (
                ("long", "ema50_4h_cross_down", "trend_long_align", (pair, "long")),
                ("short", "ema50_4h_cross_up", "trend_short_align", (pair, "short")),
            ):
                st = self._trend_exit_wait_state.get(state_key)
                if st == "need_cross" and last.get(cross_key):
                    self._trend_exit_wait_state[state_key] = "need_realign"
                elif st == "need_realign" and last.get(align_key):
                    self._trend_exit_wait_state[state_key] = None
            if self._trend_exit_wait_state.get((pair, "long")) is not None:
                trend_long_allowed = pd.Series(False, index=dataframe.index)
            if self._trend_exit_wait_state.get((pair, "short")) is not None:
                trend_short_allowed = pd.Series(False, index=dataframe.index)

        # ----- 乖离率过滤（1.2）：价格远离 EMA20 超过 BIAS_IGNORE_PCT×ART 则不开仓 -----
        bias_long_pct = (dataframe["close"] - dataframe["ema_20"]) / dataframe["ema_20"].replace(0, 1e-10)
        bias_short_pct = (dataframe["ema_20"] - dataframe["close"]) / dataframe["ema_20"].replace(0, 1e-10)
        bias_ok_long = bias_long_pct <= (BIAS_IGNORE_PCT * dataframe["art_multiplier"])
        bias_ok_short = bias_short_pct <= (BIAS_IGNORE_PCT * dataframe["art_multiplier"])
        bias_filter = (
            (dataframe["close"] - dataframe["ema_20"]).abs()
            / dataframe["ema_20"].replace(0, 1e-10)
            <= (BIAS_IGNORE_PCT * dataframe["art_multiplier"])
        )

        # 模式判定：RSI(1D) 严格在 40-60 之间视为震荡区间
        range_mode = (rsi_1d > RSI_1D_TREND_LOW) & (rsi_1d < RSI_1D_TREND_HIGH)

        # 趋势订单过滤：RSI(1D) 在 45-55 区间时禁止开趋势订单
        rsi_1d_fill = rsi_1d.fillna(50.0)
        trend_block_zone = (rsi_1d_fill >= RSI_1D_TREND_ENTRY_BLOCK_LOW) & (rsi_1d_fill <= RSI_1D_TREND_ENTRY_BLOCK_HIGH)
        trend_rsi_entry_ok = ~trend_block_zone

        # 趋势 EMA 排列额外限制：仅在 EMA20 过去 6h 平均斜率与方向一致时允许通过 EMA 排列方式开仓
        ema20_slope_6h = dataframe["ema20_slope_6h"].fillna(0.0)
        trend_ema_slope_long_ok = ema20_slope_6h > 0.0
        trend_ema_slope_short_ok = ema20_slope_6h < 0.0

        # ----- 趋势多：5种开仓方式（满足一个即可，按优先级顺序设置标签）-----
        macd_hist = dataframe["macd_hist"]
        # 条件1：均线趋势多 - 1h EMA完全多头排列（20>50>100>200）+ 价格相对1h EMA20绝对乖离率≤2%×ART + 成交量>0 + 趋势冷却允许 + RSI(1D)不在45～55 + MACD 柱>0
        trend_long_original = (
            trend_long_allowed
            & (dataframe["volume"] > 0)
            & bias_filter
            & trend_rsi_entry_ok
            & dataframe["trend_long_align"]
            & trend_ema_slope_long_ok
            & (macd_hist > 0)
        )
        # 条件2：RSI1D追踪建仓多 - RSI(1D)<30时开始追踪最低值，从最低反弹到≥32时开多
        trend_long_rsi_track = (
            trend_long_allowed
            & dataframe["trend_rsi1d_track_long"]
            & trend_rsi_entry_ok
        )
        # 条件3：RSI1D上穿40做多 - RSI(1D)从<40上穿到40～42区间时开多
        trend_long_rsi_cross_40 = (
            trend_long_allowed
            & dataframe["trend_rsi1d_cross_40_long"]
            & trend_rsi_entry_ok
        )
        # 条件4：RSI1D上穿60做多 - RSI(1D)从<60上穿到60～62区间时开多
        trend_long_rsi_cross_60 = (
            trend_long_allowed
            & dataframe["trend_rsi1d_cross_60_long"]
            & trend_rsi_entry_ok
        )
        # 条件5：RSI1D极限值做多 - RSI(1D)≤20时直接做多（优先级最高，不受45～55禁止区、乖离率等限制，仅受冷却约束）
        trend_long_rsi_extreme = (
            trend_long_allowed
            & (rsi_1d_fill <= RSI_1D_TREND_EXTREME_LONG)
        )
        # 按优先级顺序设置标签（优先级高的会覆盖优先级低的）
        dataframe.loc[trend_long_original, "enter_long"] = 1
        dataframe.loc[trend_long_original, "enter_tag"] = "趋势多_均线排列"
        dataframe.loc[trend_long_rsi_track, "enter_long"] = 1
        dataframe.loc[trend_long_rsi_track, "enter_tag"] = "趋势多_RSI1D追踪"
        dataframe.loc[trend_long_rsi_cross_40, "enter_long"] = 1
        dataframe.loc[trend_long_rsi_cross_40, "enter_tag"] = "趋势多_RSI1D上穿40"
        dataframe.loc[trend_long_rsi_cross_60, "enter_long"] = 1
        dataframe.loc[trend_long_rsi_cross_60, "enter_tag"] = "趋势多_RSI1D上穿60"
        dataframe.loc[trend_long_rsi_extreme, "enter_long"] = 1
        dataframe.loc[trend_long_rsi_extreme, "enter_tag"] = "趋势多_RSI1D极限值"

        # ----- 趋势空：5种开仓方式（满足一个即可，按优先级顺序设置标签）-----
        # 条件1：均线趋势空 - 1h EMA完全空头排列（20<50<100<200）+ 做空侧乖离率(EMA20−Close)/EMA20≤2%×ART + 趋势冷却允许 + RSI(1D)不在45～55 + MACD 柱<0
        trend_short_original = (
            trend_short_allowed
            & dataframe["trend_short_align"]
            & bias_ok_short
            & trend_rsi_entry_ok
            & trend_ema_slope_short_ok
            & (macd_hist < 0)
        )
        # 条件2：RSI1D追踪建仓空 - RSI(1D)>70时开始追踪最高值，从最高回落到≤68时开空
        trend_short_rsi_track = (
            trend_short_allowed
            & dataframe["trend_rsi1d_track_short"]
            & trend_rsi_entry_ok
        )
        # 条件3：RSI1D下穿60做空 - RSI(1D)从>60下穿到58～60区间时开空
        trend_short_rsi_cross_60 = (
            trend_short_allowed
            & dataframe["trend_rsi1d_cross_60_short"]
            & trend_rsi_entry_ok
        )
        # 条件4：RSI1D下穿40做空 - RSI(1D)从>40下穿到38～40区间时开空
        trend_short_rsi_cross_40 = (
            trend_short_allowed
            & dataframe["trend_rsi1d_cross_40_short"]
            & trend_rsi_entry_ok
        )
        # 条件5：RSI1D极限值做空 - RSI(1D)≥80时直接做空（优先级最高，不受45～55禁止区、乖离率等限制，仅受冷却约束）
        trend_short_rsi_extreme = (
            trend_short_allowed
            & (rsi_1d_fill >= RSI_1D_TREND_EXTREME_SHORT)
        )
        # 按优先级顺序设置标签（优先级高的会覆盖优先级低的）
        dataframe.loc[trend_short_original, "enter_short"] = 1
        dataframe.loc[trend_short_original, "enter_tag"] = "趋势空_均线排列"
        dataframe.loc[trend_short_rsi_track, "enter_short"] = 1
        dataframe.loc[trend_short_rsi_track, "enter_tag"] = "趋势空_RSI1D追踪"
        dataframe.loc[trend_short_rsi_cross_60, "enter_short"] = 1
        dataframe.loc[trend_short_rsi_cross_60, "enter_tag"] = "趋势空_RSI1D下穿60"
        dataframe.loc[trend_short_rsi_cross_40, "enter_short"] = 1
        dataframe.loc[trend_short_rsi_cross_40, "enter_tag"] = "趋势空_RSI1D下穿40"
        dataframe.loc[trend_short_rsi_extreme, "enter_short"] = 1
        dataframe.loc[trend_short_rsi_extreme, "enter_tag"] = "趋势空_RSI1D极限值"

        # ----- 震荡多（策略逻辑 6.1 / 6.3；1.2：仅 Close > EMA200 且 成交量 > MA(Volume,20) 时允许）-----
        range_long_ok = range_mode & (dataframe["close"] > dataframe["ema_200"]) & (dataframe["volume"] > dataframe["volume_ma_20"])
        range_long_track = range_long_ok & (rsi_1h < self.rsi_1h_oversold.value) & dataframe["range_confirm_long"]
        range_long_extreme = range_long_ok & dataframe["range_extreme_confirm_long"]
        dataframe.loc[range_long_track, "enter_long"] = 1
        dataframe.loc[range_long_track, "enter_tag"] = "震荡多"
        dataframe.loc[range_long_extreme, "enter_long"] = 1
        dataframe.loc[range_long_extreme, "enter_tag"] = "震荡多_极限"

        # ----- 震荡空（1.2：成交量过滤）-----
        range_short_ok = range_mode & (dataframe["volume"] > dataframe["volume_ma_20"])
        range_short_track = range_short_ok & (rsi_1h > self.rsi_1h_overbought.value) & dataframe["range_confirm_short"]
        range_short_extreme = range_short_ok & dataframe["range_extreme_confirm_short"]
        dataframe.loc[range_short_track, "enter_short"] = 1
        dataframe.loc[range_short_track, "enter_tag"] = "震荡空"
        dataframe.loc[range_short_extreme, "enter_short"] = 1
        dataframe.loc[range_short_extreme, "enter_tag"] = "震荡空_极限"

        # ----- 趋势 / 震荡极限平仓后同根K线反手（仅实盘/模拟有意义；回测无法按秒级复现）-----
        # 触发：
        # - 震荡：custom_exit 返回 "震荡RSI极值平多/平空" 时写入 _range_flip_signal；
        # - 趋势：custom_exit 返回 "趋势RSI极值平多/平空" 时写入 _trend_flip_signal；
        # 在同一根 1h K 线内（process_only_new_candles=false）立刻按反方向强制开仓。
        if (not self._is_backtest()) and pair and (not dataframe.empty) and ("date" in dataframe.columns):
            last_row = dataframe.iloc[-1]
            candle_time = last_row.get("date")
            if candle_time is not None and not pd.isna(candle_time):
                # 清理过期信号（只保留当前K线）
                for (p, dt) in list(self._range_flip_signal.keys()):
                    if p == pair and dt != candle_time:
                        del self._range_flip_signal[(p, dt)]
                last_idx = dataframe.index[-1]
                for (p, dt) in list(self._trend_flip_signal.keys()):
                    if p == pair and dt != candle_time:
                        del self._trend_flip_signal[(p, dt)]

                # 震荡极限平仓反手：强制反手，不受 EMA200 / 成交量过滤
                flip_range = self._range_flip_signal.get((pair, candle_time))
                if flip_range == "short":
                    dataframe.loc[last_idx, "enter_long"] = 0
                    dataframe.loc[last_idx, "enter_short"] = 1
                    dataframe.loc[last_idx, "enter_tag"] = "震荡反手空"
                elif flip_range == "long":
                    dataframe.loc[last_idx, "enter_short"] = 0
                    dataframe.loc[last_idx, "enter_long"] = 1
                    dataframe.loc[last_idx, "enter_tag"] = "震荡反手多"

                # 趋势极限平仓反手：强制反手，不受趋势 EMA 排列 / 乖离率 / 冷却限制
                flip_trend = self._trend_flip_signal.get((pair, candle_time))
                if flip_trend == "short":
                    dataframe.loc[last_idx, "enter_long"] = 0
                    dataframe.loc[last_idx, "enter_short"] = 1
                    dataframe.loc[last_idx, "enter_tag"] = "趋势反手空"
                elif flip_trend == "long":
                    dataframe.loc[last_idx, "enter_short"] = 0
                    dataframe.loc[last_idx, "enter_long"] = 1
                    dataframe.loc[last_idx, "enter_tag"] = "趋势反手多"

        return dataframe

    # -------------------------------------------------------------------------
    # 出场信号（向量化部分）
    # -------------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""

        rsi_1d = dataframe["rsi_1d"]
        rsi_1h = dataframe["rsi_1h"]

        # 震荡极限值平仓（优先级 1，后写以覆盖同 K 线趋势信号）
        range_exit_long = (rsi_1h > RSI_1H_RANGE_EXIT_LONG)
        range_exit_short = (rsi_1h < RSI_1H_RANGE_EXIT_SHORT)
        dataframe.loc[range_exit_long, ["exit_long", "exit_tag"]] = (1, "震荡RSI极值平多")
        dataframe.loc[range_exit_short, ["exit_short", "exit_tag"]] = (1, "震荡RSI极值平空")

        return dataframe

    # -------------------------------------------------------------------------
    # 自定义出场：趋势平仓优先级见策略逻辑 5.2（1→2→3）；震荡见第六部分
    # -------------------------------------------------------------------------
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        entry_tag = (trade.enter_tag or "").strip()
        is_trend = "趋势" in entry_tag
        # 任何含“震荡”的入场标签（含反手单）均视为震荡模式持仓
        is_range = "震荡" in entry_tag
        # 含“反手”的标签视为反手单：不再由 RSI 极限 / 均线相交等信号平仓，只交给 ROE 回撤表 + 全局硬止损
        is_reverse = "反手" in entry_tag

        # 当前 ROE（Freqtrade 中 current_profit 为 ratio；期货下即近似 ROE）；使用 float 保证 custom_data 可序列化持久化
        roe = float(current_profit)
        max_roe = trade.get_custom_data("max_roe_reached", default=roe)
        max_roe = float(max_roe) if max_roe is not None else roe
        if roe > max_roe:
            trade.set_custom_data("max_roe_reached", roe)
            max_roe = roe

        # 固定 ROE 止损：使用策略 self.stoploss，以中文标签平仓（避免日志显示 trailing_stop_loss）
        if roe <= self.stoploss:
            return "策略硬止损触发"

        # 回调频率说明：
        # - Backtest：Freqtrade 仅在每根 K 线收盘时调用一次 custom_exit，ROE 阶梯回撤表按“每 K 线一次”生效。
        # - Dry Run / Live：若在 config.json 中设置 process_only_new_candles=false 且 process_throttle_secs 足够小，
        #   则 custom_exit 会在单根 K 线内部被多次调用，max_roe_reached 与 ROE 阶梯回撤表将接近“秒级”追踪当前 ROE。
        # 回测时同一 (pair, current_time) 会被多次调用，缓存避免重复 get_analyzed_dataframe；dry_run/live 保持小缓存
        dataframe = kwargs.get("dataframe") if kwargs else None
        if dataframe is None or (hasattr(dataframe, "empty") and dataframe.empty):
            cache = getattr(self, "_exit_df_cache", None)
            if cache is None:
                self._exit_df_cache = {}
                cache = self._exit_df_cache
            cache_key = (pair, current_time)
            if cache_key in cache:
                dataframe = cache[cache_key]
            else:
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                cache[cache_key] = dataframe
            # 回测：大缓存减少重复计算；dry_run/live：小缓存避免占用
            max_cache = 5000 if self._is_backtest() else 50
            while len(cache) > max_cache:
                first_key = next(iter(cache))
                del cache[first_key]
        has_df = dataframe is not None and not dataframe.empty
        # 防御性：只使用 current_time 及之前的 K 线，避免回测中 get_analyzed_dataframe 返回全量时 iloc[-1] 取到未来数据
        if has_df and "date" in dataframe.columns:
            df_slice = dataframe[dataframe["date"] <= current_time]
            last_row = df_slice.iloc[-1] if not df_slice.empty else None
        else:
            last_row = dataframe.iloc[-1] if has_df else None

        # ---------- 趋势平仓（满足一条即平，优先级从高到低）----------
        # 规则1：趋势单盈利>30% 时，仅允许 OE 回撤平仓，其他方式（RSI移动止盈、均线相交）一律不触发。
        # 方式2：盈利回撤平仓（历史最高 ROE≥30% 激活，ROE_TRAIL_TABLE，平仓线=最高ROE×(1−回撤比例×ART)≥0%）
        if is_trend and max_roe >= TREND_ROE_TRAIL_ACTIVATION_ROE:
            draw = _roe_trail_drawdown(max_roe, ROE_TRAIL_TABLE)
            # 回测折中：ART 固定 1.0，避免频繁读 dataframe，保留 ROE 回撤逻辑；实盘/模拟用实时 ART
            if self._is_backtest():
                art = 1.0
            elif last_row is not None and "art_multiplier" in getattr(dataframe, "columns", []):
                art_val = last_row.get("art_multiplier", 1.0)
                art = float(art_val) if not pd.isna(art_val) else 1.0
            else:
                art = 1.0
            threshold_roe = max(0.0, max_roe * (1 - draw * art))
            if roe <= threshold_roe:
                if is_trend:
                    self._trend_exit_wait_state[(pair, "short" if trade.is_short else "long")] = "need_cross"
                return "ROE回撤表平仓"
            # 盈利>30% 时仅 OE 回撤可平仓，不触发方式1/方式3
            return None

        # 方式1：RSI(1D) 极值 + 移动止盈（仅当趋势单盈利<30% 时有效；RSI≥70 或 ≤30 启动，历史最高 ROE≥20%，回撤≥5% 平仓）
        # 仅对“普通趋势单”生效：反手单不再走 RSI 移动止盈，由 ROE 回撤表 + 全局止损接管
        if is_trend and (not is_reverse) and max_roe < TREND_ROE_TRAIL_ACTIVATION_ROE and last_row is not None:
            rsi_1d = last_row.get("rsi_1d", 50)
            if pd.isna(rsi_1d):
                rsi_1d = 50.0
            if rsi_1d >= RSI_1D_EXIT_LONG or rsi_1d <= RSI_1D_EXIT_SHORT:
                if max_roe >= TRAILING_ACTIVATION_ROE and roe <= max_roe - TREND_TRAIL_PCT:
                    if is_trend:
                        self._trend_exit_wait_state[(pair, "short" if trade.is_short else "long")] = "need_cross"
                    return "趋势RSI移动止盈5%"

        # 优先级 3：均线相交（盈利<30% 时有效；populate_exit_trend 为 EMA50 与 EMA100 相交）
        # 多单 EMA50 下穿 EMA100 → "趋势EMA50与100死叉"；空单 EMA50 上穿 EMA100 → "趋势EMA50与100金叉"

        # ---------- 震荡：盈利回撤平仓（策略逻辑 第六部分 6.5）----------
        # 规则3：震荡单盈利>20% 时，仅允许 OE 回撤平仓，其他方式（震荡RSI极值）不触发。
        if is_range and max_roe >= TRAILING_ACTIVATION_ROE:
            table = RANGE_ROE_TRAIL_TABLE
            draw = _roe_trail_drawdown(max_roe, table)
            # 回测折中：ART 固定 1.0；实盘/模拟用实时 ART
            if self._is_backtest():
                art = 1.0
            elif last_row is not None and "art_multiplier" in getattr(dataframe, "columns", []):
                art_val = last_row.get("art_multiplier", 1.0)
                art = float(art_val) if not pd.isna(art_val) else 1.0
            else:
                art = 1.0
            threshold_roe = max(0.0, max_roe * (1 - draw * art))
            if roe <= threshold_roe:
                return "ROE回撤表平仓"
            # 震荡单盈利 ROE≥20% 时，平仓条件全部由盈利回撤止盈接管，其他条件停止平仓（不透传任何 exit_tag）
            return None

        # 由 populate_exit_trend 触发的平仓：用 dataframe 的 exit_tag 作为 exit_reason，避免显示为 exit_signal
        if (
            last_row is not None
            and "exit_tag" in getattr(dataframe, "columns", [])
            and "exit_long" in getattr(dataframe, "columns", [])
            and "exit_short" in getattr(dataframe, "columns", [])
            # 趋势单在 max_roe < 30% 才允许方式3（均线相交）触发；否则由方式2独占接管
            and ((not is_trend) or (max_roe < TREND_ROE_TRAIL_ACTIVATION_ROE))
        ):
            tag = last_row.get("exit_tag")
            if isinstance(tag, str) and tag.strip():
                # 反手单：不响应任何由 populate_exit_trend 产生的技术性平仓（震荡极限 / 均线相交等），
                # 仅由 ROE 回撤表与全局硬止损处理，避免在极限区被多次“技术信号”反复收割
                if is_reverse:
                    return None
                # 震荡单盈利>20% 时仅允许 OE 回撤平仓，不采用震荡 RSI 极值平仓
                if is_range and max_roe >= TRAILING_ACTIVATION_ROE and tag in ("震荡RSI极值平多", "震荡RSI极值平空"):
                    pass  # 不返回该 tag，相当于只能用 OE 回撤平仓
                # 震荡极限平仓：同根K线反手信号（仅在 ROE<20% 且允许极值平仓时）
                elif is_range and tag in ("震荡RSI极值平多", "震荡RSI极值平空") and ("date" in getattr(dataframe, "columns", [])):
                    candle_time = last_row.get("date")
                    if candle_time is not None and not pd.isna(candle_time):
                        # 平多 -> 反手空；平空 -> 反手多
                        # 同一根 K 线的 flip 只允许来自单一模式：
                        # 震荡极限触发时，清理可能存在的趋势 flip 信号，避免同根 K 线多套反手竞争
                        self._trend_flip_signal.pop((pair, candle_time), None)
                        self._range_flip_signal[(pair, candle_time)] = "short" if tag == "震荡RSI极值平多" else "long"
                elif trade.is_short and last_row.get("exit_short") == 1:
                    if is_trend:
                        self._trend_exit_wait_state[(pair, "short")] = "need_cross"
                    return tag.strip()
                elif not trade.is_short and last_row.get("exit_long") == 1:
                    if is_trend:
                        self._trend_exit_wait_state[(pair, "long")] = "need_cross"
                    return tag.strip()

        return None

    # -------------------------------------------------------------------------
    # 入场前确认（1.2：同向订单亏损保护，≥2 个同向持仓 ROE<-10% 则禁止新开该向仓位）
    # -------------------------------------------------------------------------
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        try:
            dp = getattr(self, "dp", None)
            # 回测 / hyperopt / 实盘 / 模拟统一执行同向亏损保护，保证回测结果更接近真实风险
            if dp is None:
                return True

            # 仅做多/做空与 strategy 中一致：side 为 'long' 或 'short'
            want_short = (side == "short" or (hasattr(self, "can_short") and side == "sell"))
            open_trades = Trade.get_trades_proxy(is_open=True)
            same_side = [t for t in open_trades if getattr(t, "is_short", False) == want_short]
            losing_count = 0
            # 按 pair 缓存 dataframe，同方向多仓同交易对只取一次，显著减少回测时 get_analyzed_dataframe 调用
            pair_df_cache: Dict[str, Optional[DataFrame]] = {}
            for t in same_side:
                if t.pair not in pair_df_cache:
                    df, _ = self.dp.get_analyzed_dataframe(t.pair, self.timeframe)
                    pair_df_cache[t.pair] = df
                df = pair_df_cache[t.pair]
                if df is None or df.empty:
                    continue
                # 仅用 current_time 及之前的 K 线取价，避免回测全量 df 时取到未来数据
                if "date" in df.columns:
                    df_use = df[df["date"] <= current_time]
                    row = df_use.iloc[-1] if not df_use.empty else df.iloc[-1]
                else:
                    row = df.iloc[-1]
                current_rate = float(row.get("close", t.open_rate) or t.open_rate)
                lev = getattr(t, "leverage", 1.0) or 1.0
                if t.is_short:
                    roe = (t.open_rate - current_rate) / t.open_rate * lev
                else:
                    roe = (current_rate - t.open_rate) / t.open_rate * lev
                if roe <= LOSING_ROE_THRESHOLD:
                    losing_count += 1
            if losing_count >= MAX_SAME_SIDE_LOSING:
                logger.info(
                    "同向订单亏损保护：%s 方向已有 %d 个持仓 ROE≤%s%%，拒绝新开仓",
                    "空" if want_short else "多", losing_count, int(LOSING_ROE_THRESHOLD * 100),
                )
                return False

            # 进一步风控：若最近已连续亏损 >= 5 笔同向订单，则暂停该方向 12 小时不开新仓
            closed_trades = Trade.get_trades_proxy(is_open=False)
            # 只看同方向已平仓订单，并按平仓时间倒序
            # 官方文档：优先使用 close_date_utc，close_date 已弃用
            same_side_closed = [
                t for t in closed_trades
                if getattr(t, "is_short", False) == want_short
                and (getattr(t, "close_date_utc", None) or getattr(t, "close_date", None)) is not None
            ]
            _close_date = lambda x: getattr(x, "close_date_utc", None) or getattr(x, "close_date", None)
            same_side_closed.sort(key=_close_date, reverse=True)

            consecutive_losing = 0
            latest_close_time: Optional[datetime] = None
            for t in same_side_closed:
                profit_ratio = float(getattr(t, "profit_ratio", 0.0))
                if profit_ratio < 0.0:
                    consecutive_losing += 1
                    if latest_close_time is None:
                        latest_close_time = getattr(t, "close_date_utc", None) or getattr(t, "close_date", None)
                else:
                    break

            if consecutive_losing >= 5 and latest_close_time is not None:
                # 若距离最近一笔连续亏损订单平仓时间不足 12 小时，则直接拒绝该方向新开仓
                cooldown_until = latest_close_time + timedelta(hours=12)
                if current_time < cooldown_until:
                    logger.info(
                        "连续亏损风控：%s 方向最近连续亏损 %d 笔，暂停该方向交易至 %s",
                        "空" if want_short else "多", consecutive_losing, cooldown_until.isoformat(),
                    )
                    return False

            # 震荡订单过滤：若同交易对已存在趋势订单持仓，则不开该交易对的震荡订单
            if entry_tag and ("震荡" in entry_tag):
                trend_trades = [
                    t for t in open_trades
                    if t.pair == pair and t.enter_tag and "趋势" in t.enter_tag
                ]
                if trend_trades:
                    logger.info(
                        "震荡订单过滤：交易对 %s 当前有 %d 个趋势订单持仓，拒绝开震荡订单 %s",
                        pair, len(trend_trades), entry_tag,
                    )
                    return False

            # 震荡“反手单”保护：仅允许在同根K线存在 flip 信号时开仓，且一旦允许则清除信号避免重复
            if entry_tag in ("震荡反手多", "震荡反手空"):
                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if df is None or df.empty or "date" not in df.columns:
                    return False
                df_use = df[df["date"] <= current_time]
                row = df_use.iloc[-1] if not df_use.empty else df.iloc[-1]
                candle_time = row.get("date")
                if candle_time is None or pd.isna(candle_time):
                    return False
                flip = self._range_flip_signal.get((pair, candle_time))
                want = "long" if entry_tag == "震荡反手多" else "short"
                if flip != want:
                    return False
                # 允许开仓则清除，确保同根K线只反手一次
                del self._range_flip_signal[(pair, candle_time)]

            # 趋势“反手单”保护：仅允许在同根K线存在趋势 flip 信号时开仓，且一旦允许则清除信号避免重复
            if entry_tag in ("趋势反手多", "趋势反手空"):
                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if df is None or df.empty or "date" not in df.columns:
                    return False
                df_use = df[df["date"] <= current_time]
                row = df_use.iloc[-1] if not df_use.empty else df.iloc[-1]
                candle_time = row.get("date")
                if candle_time is None or pd.isna(candle_time):
                    return False
                flip = self._trend_flip_signal.get((pair, candle_time))
                want = "long" if entry_tag == "趋势反手多" else "short"
                if flip != want:
                    return False
                # 允许开仓则清除，确保同根K线只反手一次
                del self._trend_flip_signal[(pair, candle_time)]
        except Exception as e:  # get_trades_proxy / runmode 在部分模式下可能不可用
            logger.debug("confirm_trade_entry 同向亏损检查跳过: %s", e)
        return True
