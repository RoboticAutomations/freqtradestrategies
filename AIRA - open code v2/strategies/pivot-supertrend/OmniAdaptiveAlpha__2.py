"""
OmniAdaptiveAlpha

Ambitious, all-in-one freqtrade strategy assembled from multiple discretionary and
systematic concepts: multi-timeframe regime detection, volatility-aware position
sizing, adaptive DCA, pyramiding, layered entries (reversion, breakout, momentum),
and protective exits. Designed as a comprehensive template rather than a promise
of profitability — validate thoroughly for your market/exchange.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pandas_ta as pta
import talib.abstract as ta
from datetime import datetime

from pandas import DataFrame, Series

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
)

import freqtrade.vendor.qtpylib.indicators as qtpylib


logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    RANGE = "range"


class VolatilityState(str, Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class LiquidityState(str, Enum):
    LIQUID = "liquid"
    THIN = "thin"


class MomentumState(str, Enum):
    STRONGLY_UP = "strong_up"
    UP = "up"
    SIDEWAYS = "sideways"
    DOWN = "down"


@dataclass
class TopLevelSnapshot:
    regime: Series
    volatility: Series
    liquidity: Series
    momentum: Series
    entry_gate: Series
    btc_health: Series
    daily_trend: Series


class OmniAdaptiveAlpha(IStrategy):
    # Core configuration
    timeframe = "15m"
    informative_timeframe = "1h"
    informative_timeframe_fast = "5m"
    informative_timeframe_slow = "4h"
    informative_timeframe_daily = "1d"
    btc_pair = "BTC/EUR"
    eth_pair = "ETH/EUR"
    startup_candle_count = 400

    process_only_new_candles = True
    can_short = False
    use_exit_signal = True
    exit_profit_only = False

    top_level_prefix = "omni_"

    # Trailing defaults (may be overridden by hyperopt runtime overrides)
    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    # ROI structure (signals expected to manage exits)
    minimal_roi = {
        "0": 0.14,
        "120": 0.06,
        "360": 0.02,
        "720": 0.0,
    }
    stoploss = -0.35

    # Protections
    cooldown_lookback = IntParameter(1, 24, default=2, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=72, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)
    max_drawdown_allowed = DecimalParameter(0.08, 0.35, default=0.18, decimals=2, space="protection", optimize=True)

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": int(self.cooldown_lookback.value)},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 72,
                "trade_limit": 2,
                "stop_duration_candles": int(self.stop_duration.value),
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 72,
                "trade_limit": 10,
                "stop_duration_candles": 24,
                "max_allowed_drawdown": float(self.max_drawdown_allowed.value),
            },
        ]

    # Hyperoptables / knobs
    vol_ma_factor = DecimalParameter(0.4, 1.4, default=0.8, decimals=2, space="buy", optimize=True)
    max_spread_pct = DecimalParameter(0.002, 0.02, default=0.009, decimals=3, space="buy", optimize=True)

    rsi_low = IntParameter(15, 40, default=28, space="buy", optimize=True)
    rsi_high = IntParameter(60, 85, default=72, space="sell", optimize=True)

    bb_length = IntParameter(16, 28, default=20, space="buy", optimize=True)
    bb_mult = DecimalParameter(1.8, 2.6, default=2.2, decimals=1, space="buy", optimize=True)
    kc_scaler = DecimalParameter(1.2, 2.5, default=1.8, decimals=1, space="buy", optimize=True)

    adx_threshold = IntParameter(18, 35, default=24, space="buy", optimize=True)
    supertrend_multiplier = DecimalParameter(2.5, 4.5, default=3.0, decimals=1, space="buy", optimize=True)
    supertrend_period = IntParameter(7, 21, default=14, space="buy", optimize=True)

    macd_fast = IntParameter(8, 18, default=12, space="buy", optimize=True)
    macd_slow = IntParameter(17, 31, default=26, space="buy", optimize=True)
    macd_signal = IntParameter(4, 12, default=9, space="buy", optimize=True)

    mfi_threshold = IntParameter(15, 40, default=25, space="buy", optimize=True)
    cci_threshold = IntParameter(90, 180, default=120, space="buy", optimize=True)

    daily_slope_threshold = DecimalParameter(0.0005, 0.0030, default=0.0012, decimals=4, space="buy", optimize=True)
    btc_adx_confirm = IntParameter(14, 35, default=22, space="buy", optimize=True)

    # DCA / scaling
    position_adjustment_enable = True
    max_entry_position_adjustment = 4
    max_dca_multiplier = 8
    initial_safety_order_trigger = DecimalParameter(-0.03, -0.01, default=-0.018, decimals=3, space="buy", optimize=True)
    max_safety_orders = IntParameter(0, 4, default=2, space="buy", optimize=True)
    safety_order_step_scale = DecimalParameter(1.0, 1.6, default=1.2, decimals=2, space="buy", optimize=True)
    safety_order_volume_scale = DecimalParameter(1.0, 2.4, default=1.4, decimals=2, space="buy", optimize=True)

    dca_cooldown_minutes = IntParameter(30, 240, default=90, space="buy", optimize=True)
    dca_atr_mult = DecimalParameter(1.5, 4.5, default=2.4, decimals=1, space="buy", optimize=True)
    dca_max_depth = DecimalParameter(-0.40, -0.10, default=-0.25, decimals=2, space="buy", optimize=True)
    dca_rsi_cap = IntParameter(55, 75, default=65, space="buy", optimize=True)

    enable_pyramiding = BooleanParameter(default=True, space="buy", optimize=True)
    pyramid_trigger = DecimalParameter(0.03, 0.12, default=0.05, decimals=2, space="buy", optimize=True)
    pyramid_scale = DecimalParameter(0.2, 0.8, default=0.35, decimals=2, space="buy", optimize=True)

    # Trailing overrides
    dyn_trail_enable = BooleanParameter(default=True, space="sell", optimize=True)
    dyn_trail_loosen = DecimalParameter(1.0, 1.8, default=1.3, decimals=2, space="sell", optimize=True)
    dyn_trail_tighten = DecimalParameter(0.4, 1.0, default=0.7, decimals=2, space="sell", optimize=True)

    # Scale-out controls
    scaleout_profit_1 = DecimalParameter(0.03, 0.12, default=0.06, decimals=3, space="sell", optimize=True)
    scaleout_size_1 = DecimalParameter(0.1, 0.4, default=0.2, decimals=2, space="sell", optimize=True)
    scaleout_profit_2 = DecimalParameter(0.08, 0.30, default=0.14, decimals=3, space="sell", optimize=True)
    scaleout_size_2 = DecimalParameter(0.15, 0.5, default=0.3, decimals=2, space="sell", optimize=True)

    # Risk budgets
    risk_target_atr = DecimalParameter(0.006, 0.03, default=0.015, decimals=3, space="buy", optimize=True)
    min_stake_scale = DecimalParameter(0.2, 1.0, default=0.6, decimals=2, space="buy", optimize=True)
    max_stake_scale = DecimalParameter(1.0, 2.0, default=1.4, decimals=2, space="buy", optimize=True)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._version = "2025-Omni-alpha"

    def informative_pairs(self):
        pairs = []
        try:
            wl = self.dp.current_whitelist()
            pairs.extend([(p, self.informative_timeframe) for p in wl])
            pairs.extend([(p, self.informative_timeframe_fast) for p in wl])
        except Exception:
            pass
        pairs.extend([
            (self.btc_pair, tf)
            for tf in [self.timeframe, self.informative_timeframe, self.informative_timeframe_slow, self.informative_timeframe_daily]
        ])
        pairs.append((self.eth_pair, self.informative_timeframe))
        return pairs

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
        **kwargs,
    ) -> float:
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            atr = float(df["atr14"].iat[-1])
            price = float(df["close"].iat[-1])
            atr_pct = atr / price if price > 0 else None
        except Exception:
            atr_pct = None

        if atr_pct and atr_pct > 0:
            target = float(self.risk_target_atr.value)
            atr_scale = target / max(atr_pct, 1e-6)
            atr_scale = max(min(atr_scale, float(self.max_stake_scale.value)), float(self.min_stake_scale.value))
        else:
            atr_scale = float(self.min_stake_scale.value)

        try:
            gate_flag = df[f"{self.top_level_prefix}entry_gate"].iat[-1]
            if gate_flag < 1:
                atr_scale *= 0.5
        except Exception:
            pass
        try:
            vol_state = df[f"{self.top_level_prefix}volatility"].iat[-1]
            if vol_state == VolatilityState.HIGH.value:
                atr_scale *= 0.7
        except Exception:
            pass

        atr_scale = max(min(atr_scale, float(self.max_stake_scale.value)), float(self.min_stake_scale.value))

        base = proposed_stake * atr_scale
        base = min(base, max_stake)
        if min_stake and base < float(min_stake):
            base = float(min_stake)
        return base

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        # Core indicators
        df["ema21"] = ta.EMA(df, timeperiod=21)
        df["ema50"] = ta.EMA(df, timeperiod=50)
        df["ema200"] = ta.EMA(df, timeperiod=200)
        df["sma200"] = ta.SMA(df, timeperiod=200)
        df["rsi"] = ta.RSI(df, timeperiod=14)
        df["mfi"] = ta.MFI(df)
        df["cci"] = ta.CCI(df)
        df["atr14"] = ta.ATR(df, timeperiod=14)
        df["adx14"] = ta.ADX(df, timeperiod=14)
        df["roc"] = ta.ROC(df, timeperiod=10)

        # Bollinger + Keltner based squeeze detection
        bb = ta.BBANDS(df, timeperiod=int(self.bb_length.value), nbdevup=float(self.bb_mult.value), nbdevdn=float(self.bb_mult.value))
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = bb["upperband"], bb["middleband"], bb["lowerband"]
        kc = pta.kc(high=df["high"], low=df["low"], close=df["close"], length=int(self.bb_length.value), scalar=float(self.kc_scaler.value))
        if kc is not None and kc.shape[1] >= 3:
            df["kc_lower"], df["kc_middle"], df["kc_upper"] = kc.iloc[:, 0], kc.iloc[:, 1], kc.iloc[:, 2]
        else:
            df["kc_lower"], df["kc_middle"], df["kc_upper"] = np.nan, np.nan, np.nan
        df["squeeze_on"] = ((df["bb_upper"] <= df["kc_upper"]) & (df["bb_lower"] >= df["kc_lower"])) * 1

        # MACD
        macd = ta.MACD(df, fastperiod=int(self.macd_fast.value), slowperiod=int(self.macd_slow.value), signalperiod=int(self.macd_signal.value))
        df["macd"] = macd["macd"]
        df["macd_signal"] = macd["macdsignal"]
        df["macd_hist"] = macd["macdhist"]

        # Supertrend
        try:
            st = pta.supertrend(df["high"], df["low"], df["close"], length=int(self.supertrend_period.value), multiplier=float(self.supertrend_multiplier.value))
            df["supertrend"] = st["SUPERTl_" + str(int(self.supertrend_period.value)) + "_" + str(float(self.supertrend_multiplier.value))]
            df["supertrend_dir"] = st["SUPERTd_" + str(int(self.supertrend_period.value)) + "_" + str(float(self.supertrend_multiplier.value))]
        except Exception:
            df["supertrend"], df["supertrend_dir"] = df["close"], 1

        # Heikin-Ashi closes
        try:
            ha = qtpylib.heikinashi(df)
            df["ha_close"] = ha["close"]
        except Exception:
            df["ha_close"] = df["close"]

        # Volume metrics
        vol_ma = df["volume"].rolling(20).mean()
        vol_std = df["volume"].rolling(20).std()
        df["vol_z"] = ((df["volume"] - vol_ma) / vol_std.replace(0, np.nan)).fillna(0.0)
        df["vol_ok"] = (df["volume"] > (vol_ma * float(self.vol_ma_factor.value))).astype(int)

        # Spread approximation
        df["spread_pct"] = ((df["high"] - df["low"]) / df["close"].replace(0, np.nan)).fillna(0.0)

        # Quantile RSI bands
        try:
            df["rsi_low_q"] = df["rsi"].rolling(200).quantile(0.2)
            df["rsi_high_q"] = df["rsi"].rolling(200).quantile(0.8)
        except Exception:
            df["rsi_low_q"], df["rsi_high_q"] = np.nan, np.nan

        # Informative dataframes merges
        df = self._merge_informatives(df, metadata)
        self._annotate_top_level(df, metadata)
        return df

    def _merge_informatives(self, df: DataFrame, metadata: Dict) -> DataFrame:
        pair = metadata.get("pair")
        try:
            inf1 = self.dp.get_pair_dataframe(pair=pair, timeframe=self.informative_timeframe)
            inf1["ema50"] = ta.EMA(inf1, timeperiod=50)
            inf1["ema200"] = ta.EMA(inf1, timeperiod=200)
            inf1["adx14"] = ta.ADX(inf1, timeperiod=14)
            inf1["roc"] = ta.ROC(inf1, timeperiod=10)
            df = merge_informative_pair(
                df,
                inf1,
                self.timeframe,
                self.informative_timeframe,
                ffill=True,
                append_timeframe=True,
            )
        except Exception:
            pass

        try:
            fast = self.dp.get_pair_dataframe(pair=pair, timeframe=self.informative_timeframe_fast)
            fast["ema21"] = ta.EMA(fast, timeperiod=21)
            df = merge_informative_pair(
                df,
                fast,
                self.timeframe,
                self.informative_timeframe_fast,
                ffill=True,
                append_timeframe=True,
            )
        except Exception:
            pass

        for tf in [self.informative_timeframe, self.informative_timeframe_slow, self.informative_timeframe_daily]:
            try:
                btc_df = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe=tf)
                btc_df["ema50"] = ta.EMA(btc_df, timeperiod=50)
                btc_df["ema200"] = ta.EMA(btc_df, timeperiod=200)
                btc_df["adx14"] = ta.ADX(btc_df, timeperiod=14)
                df = merge_informative_pair(
                    df,
                    btc_df,
                    self.timeframe,
                    tf,
                    ffill=True,
                    append_timeframe=True,
                    suffix="BTC",
                )
            except Exception:
                pass

        try:
            eth_df = self.dp.get_pair_dataframe(pair=self.eth_pair, timeframe=self.informative_timeframe)
            eth_df["ema50"] = ta.EMA(eth_df, timeperiod=50)
            eth_df["ema200"] = ta.EMA(eth_df, timeperiod=200)
            df = merge_informative_pair(
                df,
                eth_df,
                self.timeframe,
                self.informative_timeframe,
                ffill=True,
                append_timeframe=True,
                suffix="ETH",
            )
        except Exception:
            pass
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        snapshot = self._annotate_top_level(df, metadata)
        gating = snapshot.entry_gate.astype(bool)

        # Spread / liquidity guard
        base_conditions = (
            (df["vol_ok"] == 1)
            & (df["spread_pct"] <= float(self.max_spread_pct.value))
            & gating
        )

        # Mean reversion entry
        cond_mr = (
            base_conditions
            & (df["rsi"] <= int(self.rsi_low.value))
            & (df["close"] <= df["bb_lower"])
            & (df["ha_close"] <= df["bb_lower"])
            & (df["macd_hist"] > -0.01)
            & (snapshot.regime != MarketRegime.BEAR.value)
            & (snapshot.daily_trend != "down")
            & (df["mfi"] <= int(self.mfi_threshold.value))
            & (df["cci"] <= -int(self.cci_threshold.value))
        )
        df.loc[cond_mr, ["enter_long", "enter_tag"]] = (1, "MR")

        # Breakout / trend entry
        cond_bo = (
            base_conditions
            & (df["close"] > df["ema50"])
            & (df["ema50"] > df["ema200"])
            & (df["supertrend_dir"] == 1)
            & (df["macd"] > df["macd_signal"])
            & (df["adx14"] > int(self.adx_threshold.value))
            & (snapshot.momentum.isin([MomentumState.UP.value, MomentumState.STRONGLY_UP.value]))
            & (snapshot.daily_trend == "up")
            & (df["cci"] >= int(self.cci_threshold.value))
            & (df["mfi"] >= max(50, int(self.mfi_threshold.value)))
        )
        df.loc[cond_bo, ["enter_long", "enter_tag"]] = (1, "Breakout")

        # Squeeze release entry
        cond_sq = (
            base_conditions
            & (df["squeeze_on"].rolling(5).max() == 1)
            & (df["close"] > df["bb_upper"])
            & (df["vol_z"] > 0)
            & (df["macd_hist"] > 0)
            & (snapshot.regime == MarketRegime.BULL.value)
            & (snapshot.daily_trend == "up")
            & (df["cci"] >= int(self.cci_threshold.value) * 0.5)
        )
        df.loc[cond_sq, ["enter_long", "enter_tag"]] = (1, "Squeeze")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[(df["rsi"] >= int(self.rsi_high.value)) & (df["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "RSI High")
        df.loc[
            (df["macd_hist"] < 0)
            & (df["close"] < df["supertrend"])
            & (df["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "Supertrend Flip")
        df.loc[
            (df["close"] < df["ema50"])
            & (df["ema50"] < df["ema200"])
            & (df["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "Trend Fail")
        return df

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, exit_reason: str, current_time: datetime, **kwargs) -> bool:
        return True

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
        **kwargs,
    ) -> Optional[float]:
        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if df is None or len(df) < 10:
            return None

        if trade.open_orders:
            return None

        entries = trade.select_filled_orders(trade.entry_side)
        filled_entries = sum(1 for e in entries if getattr(e, "status", None) == "closed")

        snapshot = self._annotate_top_level(df, {"pair": trade.pair})

        # Scale out profits progressively
        if current_profit >= float(self.scaleout_profit_2.value) and trade.nr_of_successful_exits < 2:
            reduce_amt = trade.amount * float(self.scaleout_size_2.value)
            reduce_amt = min(reduce_amt, trade.amount)
            if reduce_amt > 0:
                return -reduce_amt
        if current_profit >= float(self.scaleout_profit_1.value) and trade.nr_of_successful_exits < 1:
            reduce_amt = trade.amount * float(self.scaleout_size_1.value)
            reduce_amt = min(reduce_amt, trade.amount)
            if reduce_amt > 0:
                return -reduce_amt

        # Pyramiding into strength
        if self.enable_pyramiding.value and current_profit >= float(self.pyramid_trigger.value):
            initial_value = entries[0].cost if entries else trade.stake_amount
            add_amount = min(initial_value * float(self.pyramid_scale.value), max_stake)
            if add_amount > 0:
                return add_amount

        if filled_entries - 1 >= int(self.max_entry_position_adjustment):
            return None

        price = float(df["close"].iat[-1])
        atr = float(df["atr14"].iat[-1]) if "atr14" in df else price * 0.02
        atr_pct = atr / max(price, 1e-6)
        trigger = -max(atr_pct * float(self.dca_atr_mult.value), abs(float(self.initial_safety_order_trigger.value)))

        if current_profit > trigger:
            return None
        if current_profit < float(self.dca_max_depth.value):
            return None
        if df["rsi"].iat[-1] > int(self.dca_rsi_cap.value):
            return None
        if not bool(snapshot.entry_gate.iloc[-1]):
            return None

        minutes_since_open = (current_time - trade.open_date_utc).total_seconds() / 60.0
        if minutes_since_open < float(self.dca_cooldown_minutes.value):
            return None

        desired = entries[0].cost if entries else trade.stake_amount
        so_index = max(filled_entries, 1)
        desired *= float(self.safety_order_volume_scale.value) ** (so_index - 1)
        desired = min(desired, max_stake)
        if min_stake and desired < float(min_stake):
            desired = float(min_stake)
        return desired

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or len(df) < 10:
            return self.stoploss
        atr = float(df["atr14"].iat[-1])
        price = float(df["close"].iat[-1])
        atr_pct = atr / max(price, 1e-6)
        base_sl = -min(max(atr_pct * 3.5, 0.02), 0.25)

        snapshot = self._annotate_top_level(df, {"pair": pair})
        if self.dyn_trail_enable.value:
            if snapshot.momentum.iloc[-1] == MomentumState.STRONGLY_UP.value and current_profit > 0.04:
                base_sl *= float(self.dyn_trail_loosen.value)
            elif snapshot.volatility.iloc[-1] == VolatilityState.HIGH.value:
                base_sl *= float(self.dyn_trail_tighten.value)
        try:
            if snapshot.daily_trend.iloc[-1] == "down" and current_profit > 0.01:
                base_sl = max(base_sl, -0.04)
        except Exception:
            pass
        try:
            if not bool(snapshot.entry_gate.iloc[-1]) and current_profit > 0:
                base_sl = max(base_sl, -0.02)
        except Exception:
            pass

        try:
            age_minutes = (current_time - trade.open_date_utc).total_seconds() / 60.0
        except Exception:
            age_minutes = None
        if age_minutes is not None:
            if age_minutes > 720 and current_profit > -0.01:
                base_sl = max(base_sl, -0.01)
            elif age_minutes > 360 and current_profit > 0.001:
                base_sl = max(base_sl, -0.005)

        if current_profit > 0.20:
            return max(base_sl, -0.05)
        if current_profit > 0.10:
            return max(base_sl, -0.08)
        if current_profit > 0.05:
            return max(base_sl, -0.12)
        return max(base_sl, self.stoploss)

    def _annotate_top_level(self, df: DataFrame, metadata: Dict) -> TopLevelSnapshot:
        prefix = self.top_level_prefix
        atr = df.get("atr14", pd.Series(0.0, index=df.index)).fillna(0.0)
        price = df.get("close", pd.Series(np.nan, index=df.index))
        atr_pct = atr / price.replace(0, np.nan)
        atr_z = (atr_pct - atr_pct.rolling(100).mean()) / atr_pct.rolling(100).std()
        atr_z = atr_z.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        volatility = pd.Series(VolatilityState.NORMAL.value, index=df.index)
        volatility[atr_z >= 1.0] = VolatilityState.HIGH.value
        volatility[atr_z <= -1.0] = VolatilityState.LOW.value

        liquidity = pd.Series(
            np.where(df.get("vol_ok", pd.Series(0, index=df.index)) == 1, LiquidityState.LIQUID.value, LiquidityState.THIN.value),
            index=df.index,
        )

        ema50 = df.get("ema50", pd.Series(np.nan, index=df.index))
        ema200 = df.get("ema200", pd.Series(np.nan, index=df.index))
        adx = df.get("adx14", pd.Series(20, index=df.index))
        roc = df.get("roc", pd.Series(0, index=df.index))

        regime = pd.Series(MarketRegime.RANGE.value, index=df.index)
        regime[(price > ema200) & (ema50 > ema200) & (adx > 18)] = MarketRegime.BULL.value
        regime[(price < ema200) & (ema50 < ema200) & (adx > 18)] = MarketRegime.BEAR.value

        momentum = pd.Series(MomentumState.SIDEWAYS.value, index=df.index)
        ema21_series = df.get("ema21", pd.Series(np.nan, index=df.index))
        momentum[ema21_series > ema50] = MomentumState.UP.value
        momentum[(roc > 2) & (adx > 22)] = MomentumState.STRONGLY_UP.value
        momentum[(roc < -2) & (adx > 22)] = MomentumState.DOWN.value

        base_gate = (regime != MarketRegime.BEAR.value) & (liquidity == LiquidityState.LIQUID.value)

        daily_trend = pd.Series("neutral", index=df.index, dtype="object")
        btc_health = pd.Series(True, index=df.index)
        try:
            btc_1h_up = df[f"{self.informative_timeframe}_BTC_ema50"] > df[f"{self.informative_timeframe}_BTC_ema200"]
            btc_4h_up = df[f"{self.informative_timeframe_slow}_BTC_ema50"] > df[f"{self.informative_timeframe_slow}_BTC_ema200"]
            btc_daily_ema50 = df[f"{self.informative_timeframe_daily}_BTC_ema50"]
            btc_daily_ema200 = df[f"{self.informative_timeframe_daily}_BTC_ema200"]
            slope = (btc_daily_ema50 - btc_daily_ema50.shift(3)) / btc_daily_ema50.shift(3)
            slope = slope.replace([np.inf, -np.inf], 0.0).fillna(0.0)
            daily_up_mask = (btc_daily_ema50 > btc_daily_ema200) & (slope > float(self.daily_slope_threshold.value))
            daily_down_mask = (btc_daily_ema50 < btc_daily_ema200) & (slope < -float(self.daily_slope_threshold.value))
            daily_trend.loc[daily_up_mask] = "up"
            daily_trend.loc[daily_down_mask] = "down"
            btc_adx_ok = df[f"{self.informative_timeframe}_BTC_adx14"] > int(self.btc_adx_confirm.value)
            btc_health = btc_1h_up & btc_4h_up & daily_up_mask & btc_adx_ok
        except Exception:
            btc_health = pd.Series(True, index=df.index)

        entry_gate = base_gate & btc_health

        df.loc[:, f"{prefix}regime"] = regime
        df.loc[:, f"{prefix}volatility"] = volatility
        df.loc[:, f"{prefix}liquidity"] = liquidity
        df.loc[:, f"{prefix}momentum"] = momentum
        df.loc[:, f"{prefix}entry_gate_raw"] = base_gate.astype(int)
        df.loc[:, f"{prefix}entry_gate"] = entry_gate.astype(int)
        df.loc[:, f"{prefix}btc_health"] = btc_health.astype(int)
        df.loc[:, f"{prefix}daily_trend"] = daily_trend

        return TopLevelSnapshot(
            regime=regime,
            volatility=volatility,
            liquidity=liquidity,
            momentum=momentum,
            entry_gate=entry_gate,
            btc_health=btc_health,
            daily_trend=daily_trend,
        )
