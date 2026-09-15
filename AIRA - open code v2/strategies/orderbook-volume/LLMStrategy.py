import logging
from functools import reduce
import datetime
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TAG_ENTER_LONG = "entering long"
TAG_ENTER_SHORT = "entering short"
TAG_EXIT_LONG = "exiting long"
TAG_EXIT_SHORT = "exiting short"


class LLMStrategy(IStrategy):

    """
    $$$$$$$  |  ______    ______  $$ |   __ $$$$$$$  |  ______    ______    ______    ______            ______  
    $$ |  $$ | /      \  /      \ $$ |  /  |$$ |__$$ | /      \  /      \  /      \  /      \    /$$$$     \ 
    $$ |  $$ | $$$$$$  |/$$$$$$  |$$ |_/$$/ $$    $$< /$$$$$$  | $$$$$$  |/$$$$$$  |/$$$$$$  |  |$$$$$$
    $$ |  $$ | /    $$ |$$ |  $$/ $$   $$<  $$$$$$$  |$$    $$ | /    $$ |$$ |  $$ |$$    $$ |   $$   $$
    $$ |__$$ |/$$$$$$$ |$$ |      $$$$$$  \ $$ |  $$ |$$$$$$$$/ /$$$$$$$ |$$ |__$$ |$$$$$$$$/    $$$$$$$ 
    $$    $$/ $$    $$ |$$ |      $$ | $$  |$$ |  $$ |$$       |$$    $$ |$$    $$/ $$       |   $$   $$$     
    $$$$$$$/   $$$$$$$/ $$/       $$/   $$/ $$/   $$/  $$$$$$$/  $$$$$$$/ $$$$$$$/   $$$$$$$/    $$     $$         
                                                                          $$$$$
                                                                          $$$$$
                                                                          $$$$$
    """

    timeframe = "15m"
    process_only_new_candles = True
    can_short = True
    use_exit_signal = True
    position_adjustment_enable = False
    stoploss = -0.28
    minimal_roi = {"0": 0.09, "5000": -1}

    order_types = {
        "entry": "limit",
        "exit": "market",
        "emergency_exit": "market",
        "force_exit": "market",
        "force_entry": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Pentru EMA200, VWAP96, VP_WINDOW=96 etc
    startup_candle_count: int = 500
    plot_config = {
        "main_plot": {
            # Price + EMAs (trend overlay)
            "ema_fast": {"color": "#2962FF", "linewidth": 1.5},
            "ema_slow": {"color": "#FF6D00", "linewidth": 1.5},
            "ema_mid": {"color": "#00E676", "linewidth": 2},
            "ema_long": {"color": "#D500F9", "linewidth": 2.5},

            # Bollinger Bands
            "bb_upper_20": {"color": "rgba(33, 150, 243, 0.3)", "linewidth": 1},
            "bb_mid_20": {"color": "rgba(33, 150, 243, 0.5)", "linewidth": 1},
            "bb_lower_20": {"color": "rgba(33, 150, 243, 0.3)", "linewidth": 1},
            "bb_upper_20_fill": {
                "color": "rgba(33, 150, 243, 0.1)",
                "fill_to": "bb_lower_20",
            },

            # Donchian Channels
            "donchian_high_32": {"color": "rgba(255, 152, 0, 0.4)", "linewidth": 1},
            "donchian_low_32": {"color": "rgba(255, 152, 0, 0.4)", "linewidth": 1},

            # SuperTrend
            "supertrend_14_3": {"color": "#E91E63", "linewidth": 2},

            # VWAP levels
            "vwap_48": {"color": "rgba(156, 39, 176, 0.5)", "linewidth": 1.5, "linestyle": "--"},
            "vwap_96": {"color": "rgba(103, 58, 183, 0.5)", "linewidth": 1.5, "linestyle": "--"},

            # Pivot points (support/resistance)
            "pivot_high": {"color": "rgba(244, 67, 54, 0.6)", "linewidth": 1, "linestyle": ":"},
            "pivot_low": {"color": "rgba(76, 175, 80, 0.6)", "linewidth": 1, "linestyle": ":"},

            # Volume Profile levels
            "poc_96": {"color": "rgba(255, 235, 59, 0.7)", "linewidth": 2, "linestyle": "-."},
            "lvn_96": {"color": "rgba(255, 193, 7, 0.5)", "linewidth": 1, "linestyle": "-."},
        },
        "subplots": {
            # Expert signals (FreqAI/LLM)
            "expert_signals": {
                "expert_long_enter": {"color": "#00C853", "type": "bar"},
                "expert_long_exit": {"color": "#D50000", "type": "bar"},
                "expert_short_enter": {"color": "#AA00FF", "type": "bar"},
                "expert_short_exit": {"color": "#FF6D00", "type": "bar"},
            },

            # RSI + divergences
            "rsi": {
                "rsi_14": {"color": "#2962FF", "linewidth": 1.5},
                "bearish_div_rsi_48": {"color": "#D50000", "type": "scatter"},
                "bullish_div_rsi_48": {"color": "#00C853", "type": "scatter"},
            },

            # CCI (mean reversion)
            "cci": {
                "cci_20": {"color": "#FF6D00", "linewidth": 1.5},
            },

            # MFI (money flow)
            "mfi": {
                "mfi_14": {"color": "#9C27B0", "linewidth": 1.5},
            },

            # ADX + Directional Indicators
            "adx_di": {
                "adx_20": {"color": "#000000", "linewidth": 2},
                "di_plus_20": {"color": "#00C853", "linewidth": 1.5},
                "di_minus_20": {"color": "#D50000", "linewidth": 1.5},
            },

            # ATR (volatility)
            "atr": {
                "atr_14_wilder": {"color": "#FF5722", "linewidth": 1.5},
                "atr_for_adx_20": {"color": "rgba(255, 87, 34, 0.5)", "linewidth": 1},
            },

            # Bollinger Band position indicators
            "bb_indicators": {
                "bb_width_20": {"color": "#2196F3", "linewidth": 1.5},
                "bb_percent_b_20": {"color": "#03A9F4", "linewidth": 1.5},
            },

            # Donchian position
            "donchian_pos": {
                "donchian_pos_32": {"color": "#FF9800", "linewidth": 1.5},
            },

            # Regime indicators
            "regime": {
                "chop_14": {"color": "#607D8B", "linewidth": 1.5},
                "eff_ratio_20": {"color": "#795548", "linewidth": 1.5},
            },

            # Volume z-scores
            "volume_z": {
                "volume_z_48": {"color": "#00BCD4", "linewidth": 1.5},
                "volume_z_96": {"color": "#0097A7", "linewidth": 1.5},
            },

            # OBV + z-score
            "obv": {
                "obv": {"color": "#4CAF50", "linewidth": 1.5},
                "obv_z_48": {"color": "#8BC34A", "linewidth": 1},
            },

            # CMF (Chaikin Money Flow)
            "cmf": {
                "cmf_20": {"color": "#673AB7", "linewidth": 1.5},
            },

            # VWAP distances (normalized by ATR)
            "vwap_dist": {
                "vwap_dist_48_atr": {"color": "#9C27B0", "linewidth": 1.5},
                "vwap_dist_96_atr": {"color": "#7B1FA2", "linewidth": 1.5},
            },

            # SuperTrend indicators
            "supertrend_indicators": {
                "supertrend_dir_14_3": {"color": "#E91E63", "type": "bar"},
                "supertrend_dist_atr_14_3": {"color": "#C2185B", "linewidth": 1.5},
            },

            # Pivot distances (ATR normalized)
            "pivot_dist": {
                "pivot_high_dist_atr": {"color": "#F44336", "linewidth": 1.5},
                "pivot_low_dist_atr": {"color": "#4CAF50", "linewidth": 1.5},
            },

            # Volume Profile distances
            "vp_dist": {
                "poc_dist_96_atr": {"color": "#FFEB3B", "linewidth": 1.5},
                "lvn_dist_96_atr": {"color": "#FFC107", "linewidth": 1.5},
            },

            # EMA derivatives
            "ema_derivatives": {
                "ema_diff": {"color": "#3F51B5", "linewidth": 1.5},
                "ema_mid_slope_10": {"color": "#00BCD4", "linewidth": 1.5},
            },

            # EMA alignment signals
            "ema_alignment": {
                "ema_trend_mid_long": {"color": "#9C27B0", "type": "bar"},
                "price_above_ema_mid": {"color": "#673AB7", "type": "bar"},
                "price_above_ema_long": {"color": "#512DA8", "type": "bar"},
            },

            # Returns and volatility
            "returns_vol": {
                "return_1": {"color": "#FF5722", "linewidth": 1},
                "volatility_20": {"color": "#FF9800", "linewidth": 1.5},
            },

            # Market Regime Detection (Advanced)
            "regime_scores": {
                "trend_strength": {"color": "#1976D2", "linewidth": 2},
                "range_score": {"color": "#F57C00", "linewidth": 2},
                "volatility_score": {"color": "#7B1FA2", "linewidth": 2},
            },

            # Regime Classification
            "regime_classification": {
                "regime_confidence": {"color": "#00897B", "linewidth": 1.5},
                "regime_transition": {"color": "#D32F2F", "type": "bar"},
                "trend_direction": {"color": "#455A64", "type": "bar"},
            },
        },
    }

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2,
            },
        ]


    # 15m tf optimization

    EMA_FAST = 12
    EMA_SLOW = 26
    EMA_MID = 50
    EMA_LONG = 200
    EMA_SLOPE_LEN = 10  # 10*15m = 2.5h, bun pentru „acceleration”

    RSI_LEN = 14        # standard; pe 15m e stabil și comparabil
    CCI_LEN = 20        # standard; pe 15m reduce zgomotul și prinde mean-reversion intraday
    MFI_LEN = 14        # standard; împreună cu CMF/OBV acoperă confirmarea pe volum
    CMF_LEN = 20        # standard; ~5h pe 15m

    ATR_LEN = 14        # Wilder ATR ca unitate naturală de normalizare
    ADX_LEN = 20        # pe 15m, 20 reduce whipsaw în chop (≈ 5h)

    BB_LEN = 20
    BB_STD = 2.0

    DONCHIAN_LEN = 32   # ~8h, mai robust decât 20 (~5h) pentru breakout/fakeout
    CHOP_LEN = 14
    ER_LEN = 20

    # Confirmare / anomalii intraday și daily
    ZVOL_INTRADAY = 48  # 12h
    ZVOL_DAILY = 96     # 24h

    # VWAP rolling intraday și daily
    VWAP_INTRADAY = 48
    VWAP_DAILY = 96

    # SuperTrend: stabil pe 15m cu ATR 14 și mult 3
    ST_ATR_LEN = 14
    ST_MULT = 3.0

    # Pivoturi fractale: (3,3) reduce zgomotul față de (2,2)
    PIVOT_LEFT = 3
    PIVOT_RIGHT = 3

    # Divergențe RSI: orizont intraday (12h)
    DIV_LOOKBACK = 48

    # Volume Profile proxy: daily-ish pe 15m (96 = 24h)
    VP_WINDOW = 96
    VP_BINS = 40  # compromis: rezoluție bună, cost moderat

    # Market Regime Detection thresholds (15m optimized)
    REGIME_ADX_TREND = 25      # ADX > 25 = strong trend
    REGIME_CHOP_RANGING = 61.8 # Chop > 61.8 = ranging
    REGIME_CHOP_TRENDING = 38.2 # Chop < 38.2 = trending
    REGIME_ER_HIGH = 0.6       # ER > 0.6 = high efficiency trend
    REGIME_ER_LOW = 0.3        # ER < 0.3 = choppy/noisy
    REGIME_VOL_Z_HIGH = 2.0    # Vol Z > 2.0 = high volatility spike
    REGIME_ATR_PERCENTILE = 70 # ATR above 70th percentile = high volatility

    # Market Regime Detection
    REGIME_LOOKBACK_SHORT = 48   # 12h pentru micro regime shifts
    REGIME_LOOKBACK_LONG = 288   # 72h pentru macro regime context

    @staticmethod
    def _rma(series: pd.Series, length: int) -> pd.Series:
        """Wilder RMA (EMA cu alpha=1/length)."""
        length = int(max(1, length))
        return series.ewm(alpha=1.0 / length, adjust=False).mean()

    @staticmethod
    def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1
        ).max(axis=1)
        return tr

    @staticmethod
    def _zscore(series: pd.Series, length: int) -> pd.Series:
        length = int(max(2, length))
        mean = series.rolling(length).mean()
        std = series.rolling(length).std(ddof=0)
        return (series - mean) / (std.replace(0, np.nan))

    @staticmethod
    def _rolling_vwap(tp: pd.Series, volume: pd.Series, length: int) -> pd.Series:
        length = int(max(2, length))
        pv = (tp * volume).rolling(length).sum()
        vv = volume.rolling(length).sum()
        return pv / vv.replace(0, np.nan)

    @staticmethod
    def _bollinger(close: pd.Series, length: int, n_std: float):
        length = int(max(2, length))
        mid = close.rolling(length).mean()
        std = close.rolling(length).std(ddof=0)
        upper = mid + (n_std * std)
        lower = mid - (n_std * std)
        width = (upper - lower) / mid.replace(0, np.nan)
        percent_b = (close - lower) / (upper - lower).replace(0, np.nan)
        return mid, upper, lower, width, percent_b

    @staticmethod
    def _donchian(high: pd.Series, low: pd.Series, close: pd.Series, length: int):
        length = int(max(2, length))
        dc_h = high.rolling(length).max()
        dc_l = low.rolling(length).min()
        pos = (close - dc_l) / (dc_h - dc_l).replace(0, np.nan)
        return dc_h, dc_l, pos.clip(lower=0, upper=1)

    @staticmethod
    def _efficiency_ratio(close: pd.Series, length: int) -> pd.Series:
        length = int(max(2, length))
        change = (close - close.shift(length)).abs()
        volatility = close.diff().abs().rolling(length).sum()
        return change / volatility.replace(0, np.nan)

    @staticmethod
    def _choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
        length = int(max(2, length))
        tr = LLMStrategy._true_range(high, low, close)
        sum_tr = tr.rolling(length).sum()
        hi = high.rolling(length).max()
        lo = low.rolling(length).min()
        denom = (hi - lo).replace(0, np.nan)
        chop = 100.0 * (np.log10(sum_tr / denom) / np.log10(length))
        return chop

    @staticmethod
    def _rsi_wilder(close: pd.Series, length: int) -> pd.Series:
        length = int(max(2, length))
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = LLMStrategy._rma(gain, length)
        avg_loss = LLMStrategy._rma(loss, length)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def _adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, length: int):
        length = int(max(2, length))

        up = high.diff()
        down = -low.diff()

        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)

        tr = LLMStrategy._true_range(high, low, close)
        atr = LLMStrategy._rma(tr, length)

        plus_di = 100.0 * LLMStrategy._rma(pd.Series(plus_dm, index=high.index), length) / atr.replace(0, np.nan)
        minus_di = 100.0 * LLMStrategy._rma(pd.Series(minus_dm, index=high.index), length) / atr.replace(0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = LLMStrategy._rma(dx, length)

        return plus_di, minus_di, adx, atr

    @staticmethod
    def _cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
        length = int(max(2, length))
        tp = (high + low + close) / 3.0
        sma = tp.rolling(length).mean()

        # MAD (Mean Absolute Deviation) – definția standard pentru CCI
        mad = tp.rolling(length).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        denom = (0.015 * mad).replace(0, np.nan)
        return (tp - sma) / denom

    @staticmethod
    def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
        length = int(max(2, length))
        tp = (high + low + close) / 3.0
        rmf = tp * volume
        dtp = tp.diff()

        pos_mf = rmf.where(dtp > 0, 0.0)
        neg_mf = rmf.where(dtp < 0, 0.0).abs()

        pos_sum = pos_mf.rolling(length).sum()
        neg_sum = neg_mf.rolling(length).sum().replace(0, np.nan)

        mfr = pos_sum / neg_sum
        mfi = 100.0 - (100.0 / (1.0 + mfr))
        return mfi

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff()).fillna(0.0)
        return (direction * volume).cumsum()

    @staticmethod
    def _cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
        length = int(max(2, length))
        hl_range = (high - low)
        hl_range_safe = hl_range.replace(0, np.nan)
        mfm = (((close - low) - (high - close)) / hl_range_safe).fillna(0.0)
        mfv = mfm * volume
        cmf = mfv.rolling(length).sum() / volume.rolling(length).sum().replace(0, np.nan)
        return cmf

    @staticmethod
    def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series, atr_len: int, mult: float):
        """
        SuperTrend clasic (leak-safe, iterativ).
        Returnează: supertrend_line, direction (+1 bull / -1 bear)
        """
        atr_len = int(max(2, atr_len))
        tr = LLMStrategy._true_range(high, low, close)
        atr = LLMStrategy._rma(tr, atr_len)

        hl2 = (high + low) / 2.0
        upperband = hl2 + (mult * atr)
        lowerband = hl2 - (mult * atr)

        fu = upperband.copy()
        fl = lowerband.copy()

        st = pd.Series(index=close.index, dtype="float64")
        direction = pd.Series(index=close.index, dtype="float64")

        # Start from first valid ATR
        first_valid = atr.first_valid_index()
        if first_valid is None:
            return st, direction

        idx = close.index.get_loc(first_valid)

        # Initialize bands up to idx
        for i in range(idx + 1, len(close)):
            prev_i = i - 1

            # final upper
            if (upperband.iat[i] < fu.iat[prev_i]) or (close.iat[prev_i] > fu.iat[prev_i]):
                fu.iat[i] = upperband.iat[i]
            else:
                fu.iat[i] = fu.iat[prev_i]

            # final lower
            if (lowerband.iat[i] > fl.iat[prev_i]) or (close.iat[prev_i] < fl.iat[prev_i]):
                fl.iat[i] = lowerband.iat[i]
            else:
                fl.iat[i] = fl.iat[prev_i]

        # Initialize SuperTrend at idx
        st.iat[idx] = fu.iat[idx] if close.iat[idx] <= fu.iat[idx] else fl.iat[idx]
        direction.iat[idx] = 1.0 if close.iat[idx] > st.iat[idx] else -1.0

        for i in range(idx + 1, len(close)):
            prev_i = i - 1
            prev_st = st.iat[prev_i]

            if np.isnan(prev_st):
                st.iat[i] = fu.iat[i] if close.iat[i] <= fu.iat[i] else fl.iat[i]
            else:
                # If previous ST used upper band
                if prev_st == fu.iat[prev_i]:
                    st.iat[i] = fu.iat[i] if close.iat[i] <= fu.iat[i] else fl.iat[i]
                else:
                    st.iat[i] = fl.iat[i] if close.iat[i] >= fl.iat[i] else fu.iat[i]

            direction.iat[i] = 1.0 if close.iat[i] > st.iat[i] else -1.0

        return st, direction

    @staticmethod
    def _percentile_rank(series: pd.Series, length: int) -> pd.Series:
        """Calculate percentile rank (0-100) of current value within rolling window.

        Returns where current value stands in the distribution of last N values.
        0 = lowest in period, 100 = highest in period.
        """
        length = int(max(2, length))
        def calc_pct(x):
            if len(x) < 2:
                return 50.0
            val = x.iloc[-1]
            if np.isnan(val):
                return 50.0
            rank = (x < val).sum()
            return 100.0 * rank / (len(x) - 1)

        return series.rolling(length).apply(calc_pct, raw=False)

    @staticmethod
    def _trend_strength_score(
        adx: pd.Series,
        di_plus: pd.Series,
        di_minus: pd.Series,
        st_dir: pd.Series,
        ema_trend: pd.Series,
        price_above_mid: pd.Series,
        ema_diff: pd.Series
    ) -> pd.Series:
        """Calculate multi-factor trend strength score (0-100).

        Combines:
        - ADX strength (0-40+ mapped to 0-100)
        - DI spread (how much DI+ dominates DI- or vice versa)
        - SuperTrend direction consistency
        - EMA alignment
        - EMA momentum (fast vs slow divergence)

        Returns: 0-100 where:
        - 0-20: No trend / ranging
        - 20-40: Weak trend
        - 40-60: Moderate trend
        - 60-80: Strong trend
        - 80-100: Very strong trend
        """
        # ADX component (0-40 -> 0-40 points)
        adx_score = np.clip(adx, 0, 40)

        # DI spread component (0-50 -> 0-30 points)
        di_spread = (di_plus - di_minus).abs()
        di_score = np.clip(di_spread * 0.6, 0, 30)

        # Direction consistency (0-15 points)
        # SuperTrend agrees with DI direction
        st_agree = np.where(
            ((st_dir > 0) & (di_plus > di_minus)) | ((st_dir < 0) & (di_minus > di_plus)),
            15.0,
            0.0
        )

        # EMA alignment (0-10 points)
        # Mid>Long and Price>Mid for bullish, opposite for bearish
        ema_align_score = np.where(
            ((ema_trend > 0) & (price_above_mid > 0)) | ((ema_trend < 0) & (price_above_mid < 0)),
            10.0,
            0.0
        )

        # EMA momentum (0-5 points)
        # Strong positive or negative diff
        ema_mom_score = np.clip(ema_diff.abs() * 500, 0, 5)  # scaled for typical values

        total = adx_score + di_score + st_agree + ema_align_score + ema_mom_score
        return pd.Series(np.clip(total, 0, 100), index=adx.index)

    @staticmethod
    def _range_score(
        chop: pd.Series,
        eff_ratio: pd.Series,
        bb_width: pd.Series,
        adx: pd.Series
    ) -> pd.Series:
        """Calculate range/choppiness score (0-100).

        Higher score = more ranging/choppy market.

        Combines:
        - Choppiness Index (>61.8 = choppy)
        - Efficiency Ratio (<0.3 = inefficient/sideways)
        - Bollinger Width (narrow = consolidation)
        - ADX (low = no trend)

        Returns: 0-100 where:
        - 0-30: Trending market
        - 30-50: Transitional
        - 50-70: Ranging
        - 70-100: Extremely choppy
        """
        # Choppiness component (0-100 mapped)
        # 38.2 -> 0, 61.8 -> 50, 80 -> 100
        chop_normalized = np.clip((chop - 38.2) / (80 - 38.2) * 100, 0, 100)

        # Efficiency Ratio component (inverted, 1.0->0, 0.0->100)
        # 0.6 -> 0, 0.3 -> 50, 0.0 -> 100
        er_normalized = np.clip((0.6 - eff_ratio) / 0.6 * 100, 0, 100)

        # BB Width component (percentile-based, low width = high range score)
        # We'll use raw width and invert
        bb_inv = np.where(bb_width > 0, 1.0 / (bb_width + 0.001), 0)
        bb_normalized = np.clip(bb_inv * 10, 0, 100)  # scale factor

        # ADX component (inverted: low ADX = high range score)
        # 40 -> 0, 20 -> 50, 0 -> 100
        adx_inv = np.clip((40 - adx) / 40 * 100, 0, 100)

        # Weighted average
        total = (chop_normalized * 0.35 + er_normalized * 0.25 + 
                bb_normalized * 0.20 + adx_inv * 0.20)

        return pd.Series(np.clip(total, 0, 100), index=chop.index)

    @staticmethod
    def _volatility_regime(
        atr: pd.Series,
        bb_width: pd.Series,
        vol_z_48: pd.Series,
        returns_vol: pd.Series,
        lookback: int
    ):
        """Classify volatility regime.

        Returns tuple: (volatility_score 0-100, regime_label)

        volatility_score:
        - 0-25: Very Low Volatility (compression)
        - 25-45: Low Volatility
        - 45-55: Normal Volatility
        - 55-75: High Volatility
        - 75-100: Extreme Volatility (expansion)

        regime_label: 'VERY_LOW', 'LOW', 'NORMAL', 'HIGH', 'EXTREME'
        """
        # ATR percentile in lookback window
        atr_pct = LLMStrategy._percentile_rank(atr, lookback)

        # BB Width percentile
        bb_pct = LLMStrategy._percentile_rank(bb_width, lookback)

        # Volume Z-score component
        # -2 to +2 Z -> 0 to 100
        vol_z_norm = np.clip((vol_z_48 + 2) / 4 * 100, 0, 100)

        # Returns volatility percentile
        returns_pct = LLMStrategy._percentile_rank(returns_vol, lookback)

        # Combined volatility score
        vol_score = (atr_pct * 0.35 + bb_pct * 0.30 + 
                    vol_z_norm * 0.20 + returns_pct * 0.15)

        # Classify regime
        def classify(score):
            if np.isnan(score):
                return 'UNKNOWN'
            if score < 25:
                return 'VERY_LOW'
            elif score < 45:
                return 'LOW'
            elif score < 55:
                return 'NORMAL'
            elif score < 75:
                return 'HIGH'
            else:
                return 'EXTREME'

        regime_labels = vol_score.apply(classify)

        return vol_score, regime_labels

    @staticmethod
    def _market_regime_classification(
        trend_strength: pd.Series,
        range_score: pd.Series,
        volatility_score: pd.Series
    ):
        """Classify overall market regime.

        Regimes:
        - STRONG_UPTREND / STRONG_DOWNTREND: High trend strength + directional
        - WEAK_TREND: Moderate trend strength
        - RANGING_CALM: High range score + low volatility
        - RANGING_VOLATILE: High range score + high volatility
        - BREAKOUT_SETUP: Low volatility transitioning (compression before expansion)
        - CHAOTIC: High volatility + high range
        """
        regime = []
        confidence = []

        for i in range(len(trend_strength)):
            ts = trend_strength.iloc[i]
            rs = range_score.iloc[i]
            vs = volatility_score.iloc[i]

            if np.isnan(ts) or np.isnan(rs) or np.isnan(vs):
                regime.append('UNKNOWN')
                confidence.append(0.0)
                continue

            # Decision tree for regime classification

            # Strong trend (trend strength > 60, range < 50)
            if ts > 60 and rs < 50:
                regime.append('STRONG_TREND')
                conf = min(ts, 100 - rs) * 0.8  # confidence based on clarity
                confidence.append(conf)

            # Weak/Moderate trend (40 < trend < 60)
            elif 40 < ts < 60 and rs < 60:
                regime.append('WEAK_TREND')
                conf = (ts - 40) / 20 * 60 + 20  # 20-80 confidence
                confidence.append(conf)

            # Ranging Calm (high range, low vol)
            elif rs > 60 and vs < 45:
                regime.append('RANGING_CALM')
                conf = min(rs, 100 - vs) * 0.7
                confidence.append(conf)

            # Ranging Volatile (high range, high vol)
            elif rs > 60 and vs > 55:
                regime.append('RANGING_VOLATILE')
                conf = min(rs, vs) * 0.75
                confidence.append(conf)

            # Breakout Setup (low vol, transitioning from range)
            elif vs < 30 and 40 < rs < 70:
                regime.append('BREAKOUT_SETUP')
                conf = (100 - vs) * 0.5 + 25  # moderate confidence
                confidence.append(conf)

            # Chaotic (high vol, high range, low trend)
            elif vs > 70 and rs > 60 and ts < 40:
                regime.append('CHAOTIC')
                conf = min(vs, rs) * 0.8
                confidence.append(conf)

            # Transitional (mixed signals)
            else:
                regime.append('TRANSITIONAL')
                # Low confidence for transitional states
                conf = 30.0
                confidence.append(conf)

        return pd.Series(regime, index=trend_strength.index), pd.Series(confidence, index=trend_strength.index)

    @staticmethod
    def _detect_market_regime(
        adx: float,
        chop: float,
        eff_ratio: float,
        vol_z: float,
        atr_percentile: float,
        supertrend_dir: float
    ) -> str:
        """
        Detect current market regime based on multiple indicators.
        Returns one of:
        - STRONG_TREND_UP: Strong bullish trend (high confidence)
        - STRONG_TREND_DOWN: Strong bearish trend (high confidence)
        - WEAK_TREND_UP: Weak bullish trend
        - WEAK_TREND_DOWN: Weak bearish trend
        - RANGING: Sideways choppy market
        - HIGH_VOLATILITY: High volatility
        - TRANSITION: Market in transition
        """
        # High volatility override
        if vol_z > 2.5 or atr_percentile > 85:
            return "HIGH_VOLATILITY"

        # Strong trending conditions
        is_strong_trend = (
            adx > 30 and
            chop < 40 and
            eff_ratio > 0.55
        )

        if is_strong_trend:
            return "STRONG_TREND_UP" if supertrend_dir > 0 else "STRONG_TREND_DOWN"

        # Weak trending conditions
        is_weak_trend = (
            (adx > 20 or eff_ratio > 0.4) and
            chop < 55
        )

        if is_weak_trend:
            return "WEAK_TREND_UP" if supertrend_dir > 0 else "WEAK_TREND_DOWN"

        # Ranging conditions
        is_ranging = (
            chop > 61.8 or
            (adx < 15 and eff_ratio < 0.3)
        )

        if is_ranging:
            return "RANGING"

        # Default: transition
        return "TRANSITION"

    @staticmethod
    def _volume_profile_levels(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        window: int,
        bins: int
    ):
        """
        Proxy Volume Profile (rolling):
          - POC = nivelul de preț (bin) cu volum maxim
          - LVN = nivelul de preț (bin) cu volum minim pozitiv (zone de „low acceptance”)
        """
        window = int(max(10, window))
        bins = int(max(10, bins))

        tp = (high + low + close) / 3.0
        n = len(tp)
        poc = np.full(n, np.nan, dtype="float64")
        lvn = np.full(n, np.nan, dtype="float64")

        tp_vals = tp.values.astype("float64", copy=False)
        vol_vals = volume.values.astype("float64", copy=False)

        for i in range(n):
            start = max(0, i - window + 1)
            w_tp = tp_vals[start:i + 1]
            w_vol = vol_vals[start:i + 1]

            if len(w_tp) < 10:
                continue

            pmin = np.nanmin(w_tp)
            pmax = np.nanmax(w_tp)
            if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
                continue

            edges = np.linspace(pmin, pmax, bins + 1)
            idxs = np.clip(np.digitize(w_tp, edges) - 1, 0, bins - 1)

            vol_by_bin = np.zeros(bins, dtype="float64")
            for b, v in zip(idxs, w_vol):
                if np.isfinite(v):
                    vol_by_bin[int(b)] += float(v)

            if np.all(vol_by_bin <= 0):
                continue

            poc_bin = int(np.argmax(vol_by_bin))
            centers = (edges[:-1] + edges[1:]) / 2.0
            poc[i] = centers[poc_bin]

            # LVN
            pos_mask = vol_by_bin > 0
            if np.any(pos_mask):
                lvn_bin = int(np.argmin(np.where(pos_mask, vol_by_bin, np.inf)))
                lvn[i] = centers[lvn_bin]

        return pd.Series(poc, index=tp.index), pd.Series(lvn, index=tp.index)

    def feature_engineering_standard(self, dataframe: DataFrame, **kwargs) -> DataFrame:

        df = dataframe.copy()

        # Ensure dtypes
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = df[col].astype("float64")

        # Time context calendar micro

        df["%-day_of_week"] = df["date"].dt.dayofweek
        df["%-hour_of_day"] = df["date"].dt.hour

        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]


        # Returns / volatility (micro)

        df["%-return_1"] = close.pct_change()
        df["%-volatility_20"] = df["%-return_1"].rolling(20).std(ddof=0)


        # Trend (EMA family) + slope

        df["%-ema_fast"] = close.ewm(span=self.EMA_FAST, adjust=False).mean()
        df["%-ema_slow"] = close.ewm(span=self.EMA_SLOW, adjust=False).mean()
        df["%-ema_mid"] = close.ewm(span=self.EMA_MID, adjust=False).mean()
        df["%-ema_long"] = close.ewm(span=self.EMA_LONG, adjust=False).mean()

        df["%-ema_diff"] = (df["%-ema_fast"] - df["%-ema_slow"]) / close.replace(0, np.nan)

        # Slope pe EMA50 (derivată discretă pe orizont ~2.5h)
        df["%-ema_mid_slope_10"] = (df["%-ema_mid"] - df["%-ema_mid"].shift(self.EMA_SLOPE_LEN)) / close.replace(0, np.nan)

        # Alignment features
        df["%-ema_trend_mid_long"] = (df["%-ema_mid"] > df["%-ema_long"]).astype(int)
        df["%-price_above_ema_mid"] = (close > df["%-ema_mid"]).astype(int)
        df["%-price_above_ema_long"] = (close > df["%-ema_long"]).astype(int)


        # ATR Wilder

        tr = self._true_range(high, low, close)
        df["%-atr_14_wilder"] = self._rma(tr, self.ATR_LEN)


        # RSI (Wilder), CCI, MFI

        df["%-rsi_14"] = self._rsi_wilder(close, self.RSI_LEN)
        df["%-cci_20"] = self._cci(high, low, close, self.CCI_LEN)
        df["%-mfi_14"] = self._mfi(high, low, close, volume, self.MFI_LEN)

        # ADX / DI
        di_plus, di_minus, adx, atr_for_adx = self._adx_wilder(high, low, close, self.ADX_LEN)
        df["%-di_plus_20"] = di_plus
        df["%-di_minus_20"] = di_minus
        df["%-adx_20"] = adx
        df["%-atr_for_adx_20"] = atr_for_adx

        bb_mid, bb_up, bb_low, bb_width, bb_pb = self._bollinger(close, self.BB_LEN, self.BB_STD)
        df["%-bb_mid_20"] = bb_mid
        df["%-bb_upper_20"] = bb_up
        df["%-bb_lower_20"] = bb_low
        df["%-bb_width_20"] = bb_width
        df["%-bb_percent_b_20"] = bb_pb

        dc_h, dc_l, dc_pos = self._donchian(high, low, close, self.DONCHIAN_LEN)
        df["%-donchian_high_32"] = dc_h
        df["%-donchian_low_32"] = dc_l
        df["%-donchian_pos_32"] = dc_pos


        df["%-chop_14"] = self._choppiness_index(high, low, close, self.CHOP_LEN)
        df["%-eff_ratio_20"] = self._efficiency_ratio(close, self.ER_LEN)

        df["%-volume_z_48"] = self._zscore(volume, self.ZVOL_INTRADAY)
        df["%-volume_z_96"] = self._zscore(volume, self.ZVOL_DAILY)

        df["%-obv"] = self._obv(close, volume)
        df["%-obv_z_48"] = self._zscore(df["%-obv"], self.ZVOL_INTRADAY)

        df["%-cmf_20"] = self._cmf(high, low, close, volume, self.CMF_LEN)


        # Rolling VWAP (intraday + daily) + distance in ATR units

        tp = (high + low + close) / 3.0
        df["%-vwap_48"] = self._rolling_vwap(tp, volume, self.VWAP_INTRADAY)
        df["%-vwap_96"] = self._rolling_vwap(tp, volume, self.VWAP_DAILY)

        atr = df["%-atr_14_wilder"].replace(0, np.nan)
        df["%-vwap_dist_48_atr"] = (close - df["%-vwap_48"]) / atr
        df["%-vwap_dist_96_atr"] = (close - df["%-vwap_96"]) / atr

        st_line, st_dir = self._supertrend(high, low, close, self.ST_ATR_LEN, self.ST_MULT)
        df["%-supertrend_14_3"] = st_line
        df["%-supertrend_dir_14_3"] = st_dir
        df["%-supertrend_dist_atr_14_3"] = (close - st_line) / atr


        # Market structure: fractal pivots (3,3) leak-safe

        win = self.PIVOT_LEFT + self.PIVOT_RIGHT + 1

        # Pivot high becomes known after RIGHT candles iplemented via shift
        roll_max = high.rolling(win).max()
        pivot_high = np.where(high.shift(self.PIVOT_RIGHT) == roll_max, high.shift(self.PIVOT_RIGHT), np.nan)
        df["%-pivot_high"] = pd.Series(pivot_high, index=df.index).ffill()

        roll_min = low.rolling(win).min()
        pivot_low = np.where(low.shift(self.PIVOT_RIGHT) == roll_min, low.shift(self.PIVOT_RIGHT), np.nan)
        df["%-pivot_low"] = pd.Series(pivot_low, index=df.index).ffill()

        df["%-pivot_high_dist_atr"] = (close - df["%-pivot_high"]) / atr
        df["%-pivot_low_dist_atr"] = (close - df["%-pivot_low"]) / atr


        df["%-swing_high_dist_atr"] = df["%-pivot_high_dist_atr"]
        df["%-swing_low_dist_atr"] = df["%-pivot_low_dist_atr"]


        # RSI divergences (simplified, leak-saf
        rsi = df["%-rsi_14"]
        df["%-bearish_div_rsi_48"] = ((close > close.shift(self.DIV_LOOKBACK)) & (rsi < rsi.shift(self.DIV_LOOKBACK))).astype(int)
        df["%-bullish_div_rsi_48"] = ((close < close.shift(self.DIV_LOOKBACK)) & (rsi > rsi.shift(self.DIV_LOOKBACK))).astype(int)


        # Volume Profile proxy POC/LVN daily window (96)

        poc, lvn = self._volume_profile_levels(high, low, close, volume, self.VP_WINDOW, self.VP_BINS)
        df["%-poc_96"] = poc
        df["%-lvn_96"] = lvn
        df["%-poc_dist_96_atr"] = (close - df["%-poc_96"]) / atr
        df["%-lvn_dist_96_atr"] = (close - df["%-lvn_96"]) / atr


        # Market Regime Detection

        # Trend Strength Score
        df["%-trend_strength"] = self._trend_strength_score(
            df["%-adx_20"],
            df["%-di_plus_20"],
            df["%-di_minus_20"],
            df["%-supertrend_dir_14_3"],
            df["%-ema_trend_mid_long"],
            df["%-price_above_ema_mid"],
            df["%-ema_diff"]
        )

        df["%-range_score"] = self._range_score(
            df["%-chop_14"],
            df["%-eff_ratio_20"],
            df["%-bb_width_20"],
            df["%-adx_20"]
        )

        vol_score, vol_regime = self._volatility_regime(
            df["%-atr_14_wilder"],
            df["%-bb_width_20"],
            df["%-volume_z_48"],
            df["%-volatility_20"],
            self.REGIME_LOOKBACK_LONG
        )
        df["%-volatility_score"] = vol_score
        df["%-volatility_regime"] = vol_regime

        # Overall Market Regime Classification
        regime_label, regime_confidence = self._market_regime_classification(
            df["%-trend_strength"],
            df["%-range_score"],
            df["%-volatility_score"]
        )
        df["%-market_regime"] = regime_label
        df["%-regime_confidence"] = regime_confidence

        # Regime Transition Detection (regime changed from previous candle)
        df["%-regime_transition"] = (df["%-market_regime"] != df["%-market_regime"].shift(1)).astype(int)

        df["%-trend_direction"] = np.where(
            df["%-supertrend_dir_14_3"] > 0,
            1,
            np.where(df["%-supertrend_dir_14_3"] < 0, -1, 0)
        )

        plain_cols = {}
        for col in list(df.columns):
            if col.startswith("%-"):
                plain = col[2:]
                if plain not in df.columns:
                    plain_cols[plain] = df[col]
        if plain_cols:
            df = df.assign(**plain_cols)


        # Final cleanup

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        return df


    # Freqtrade hooks

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        for col in dataframe.columns:
            if col.startswith('%-'):
                plot_name = col[2:]
                dataframe[plot_name] = dataframe[col]


        self._notify_llm_decision(dataframe, metadata)

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """Provide at least one numeric target column for FreqAI.

        This strategy uses LLM-driven signals and does not rely on classical ML labels,
        but FreqAI's return-value/plotting pipeline expects at least one `&`-prefixed label.
        """
        if "&-empty" not in dataframe.columns:
            dataframe["&-empty"] = 0.0
        else:
            # Ensure numeric dtype (avoid object dtype which skips mean/std handling)
            dataframe["&-empty"] = pd.to_numeric(dataframe["&-empty"], errors="coerce").fillna(0.0)
        return dataframe

    def _send_llm_telegram(self, message: str, *, always_send: bool = False) -> None:
        """Send a custom message via Freqtrade RPC (Telegram, webhook, etc).

        Uses DataProvider.send_msg which is safe in DRY_RUN/LIVE and no-op otherwise.
        """
        try:
            dp = getattr(self, "dp", None)
            if dp is None:
                return

            msg = (message or "").strip()
            if not msg:
                return

            # Telegram max message size is 4096 chars.
            if len(msg) > 3800:
                msg = msg[:3800] + "…"

            dp.send_msg(msg, always_send=always_send)
        except Exception:
            logger.exception("Failed to send LLM decision to RPC/Telegram")

    def _notify_llm_decision(self, dataframe: DataFrame, metadata: dict) -> None:
        """Send Telegram notification with LLM's latest decision (including NEUTRAL).

        Called from populate_indicators after FreqAI has run, to provide visibility
        into what the LLM is thinking on each candle.
        """
        try:
            if len(dataframe) == 0:
                return

            last_candle = dataframe.iloc[-1]
            pair = metadata.get("pair", "UNKNOWN")

            # Extract LLM decision from expert columns
            long_enter = last_candle.get("expert_long_enter", 0)
            short_enter = last_candle.get("expert_short_enter", 0)
            long_exit = last_candle.get("expert_long_exit", 0)
            short_exit = last_candle.get("expert_short_exit", 0)
            neutral = last_candle.get("expert_neutral", 0)
            opinion = str(last_candle.get("expert_opinion", "")).strip()

            # Determine action
            if long_enter == 1:
                action = "LONG_ENTER"
            elif short_enter == 1:
                action = "SHORT_ENTER"
            elif long_exit == 1:
                action = "LONG_EXIT"
            elif short_exit == 1:
                action = "SHORT_EXIT"
            elif neutral == 1:
                action = "NEUTRAL"
            else:
                action = "UNKNOWN"

            # Build message
            sentiment = last_candle.get("sentiment", 0.0)
            heat = last_candle.get("heat", 0.0)

            msg_lines = [
                f"LLM Analysis: {pair}",
                f"Action: {action}",
                f"Sentiment: {sentiment:.2f}",
                f"Heat: {heat:.2f}",
                f"Opinion: {opinion}",
            ]

            # Send with caching (won't spam same message multiple times per candle)
            self._send_llm_telegram("\n".join(msg_lines), always_send=False)

        except Exception:
            logger.exception("Failed to notify LLM decision")

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [df["expert_long_enter"] == 1]
        enter_short_conditions = [df["expert_short_enter"] == 1]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions),
                ["enter_long", "enter_tag"],
            ] = (1, TAG_ENTER_LONG)

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions),
                ["enter_short", "enter_tag"],
            ] = (1, TAG_ENTER_SHORT)

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        return df

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime.datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[str]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        entry_tag = trade.enter_tag if hasattr(trade, "enter_tag") else None

        if last_candle.get("expert_short_exit", 0) == 1 and entry_tag == TAG_ENTER_SHORT:
            opinion = str(last_candle.get("expert_opinion", "")).strip()
            self._send_llm_telegram(
                "\n".join(
                    [
                        "LLM decision: SHORT_EXIT",
                        f"Pair: {pair}",
                        f"Profit: {current_profit:.4f}",
                        f"Opinion: {opinion}",
                    ]
                )
            )
            return f"{last_candle.get('expert_opinion', '')}, {TAG_EXIT_SHORT}"

        if last_candle.get("expert_long_exit", 0) == 1 and entry_tag == TAG_ENTER_LONG:
            opinion = str(last_candle.get("expert_opinion", "")).strip()
            self._send_llm_telegram(
                "\n".join(
                    [
                        "LLM decision: LONG_EXIT",
                        f"Pair: {pair}",
                        f"Profit: {current_profit:.4f}",
                        f"Opinion: {opinion}",
                    ]
                )
            )
            return f"{last_candle.get('expert_opinion', '')}, {TAG_EXIT_LONG}"

        return None

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime.datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> bool:
        """
        Filtru simplu anti-slippage:
          - refuză intrarea dacă prețul a fugit >0.25% față de ultimul close (în direcția intrării).
        """
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        last_close = float(last_candle["close"])

        opinion = str(last_candle.get("expert_opinion", "")).strip()
        llm_action = "LONG_ENTER" if side == "long" else "SHORT_ENTER"
        rejected_reason = ""

        if side == "long":
            if rate > (last_close * (1.0 + 0.0025)):
                rejected_reason = "REJECTED (slippage filter)"
                self._send_llm_telegram(
                    "\n".join(
                        [
                            f"LLM decision: {llm_action} {rejected_reason}",
                            f"Pair: {pair}",
                            f"Rate: {rate}",
                            f"Last close: {last_close}",
                            f"Opinion: {opinion}",
                        ]
                    )
                )
                return False
        else:
            if rate < (last_close * (1.0 - 0.0025)):
                rejected_reason = "REJECTED (slippage filter)"
                self._send_llm_telegram(
                    "\n".join(
                        [
                            f"LLM decision: {llm_action} {rejected_reason}",
                            f"Pair: {pair}",
                            f"Rate: {rate}",
                            f"Last close: {last_close}",
                            f"Opinion: {opinion}",
                        ]
                    )
                )
                return False

        # Accepted entry
        self._send_llm_telegram(
            "\n".join(
                [
                    f"LLM decision: {llm_action} ACCEPTED",
                    f"Pair: {pair}",
                    f"Rate: {rate}",
                    f"Last close: {last_close}",
                    f"Opinion: {opinion}",
                ]
            )
        )

        logger.info(f"{last_candle.get('expert_opinion', '')}\n\nEntering {side} on {pair} at {rate}")
        return True
