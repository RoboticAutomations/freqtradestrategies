# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy.parameters import CategoricalParameter, DecimalParameter, IntParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair, Trade
import numpy as np
from freqtrade.strategy import stoploss_from_open
import numexpr as ne
pd.options.mode.chained_assignment = None  # default='warn'
ne.set_num_threads(32)
pd.options.mode.chained_assignment = None  # default='warn'

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=int(n), adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    n = int(n)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/float(n), adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/float(n), adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.replace([np.inf, -np.inf], np.nan).fillna(50.0).astype(float)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/float(n), adjust=False).mean().fillna(0).astype(float)


def _rolling_range(df: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series]:

    window = int(max(2, window))
    range_high = df["high"].rolling(window, min_periods=1).max()
    range_low = df["low"].rolling(window, min_periods=1).min()
    return range_high.astype(float), range_low.astype(float)


def _stepped_range(df: pd.DataFrame, window: int, step_bars: int) -> tuple[pd.Series, pd.Series]:
    window = int(max(2, window))
    step_bars = int(max(1, step_bars))

    # 1. Calculate a standard rolling max/min (trailing)
    # This represents the High/Low of the last 'window' bars at any given point
    rolling_high = df["high"].rolling(window, min_periods=1).max()
    rolling_low = df["low"].rolling(window, min_periods=1).min()

    # 2. Resample/Step this data
    # We want to hold the value constant for 'step_bars'
    # To avoid lookahead, we take the value at the *start* of the step (which is based on trailing data)
    # and hold it for the duration of the step.
    
    # Create a series identifying the step blocks
    # e.g., 0, 0, 0, 1, 1, 1...
    # We can simply sample the rolling data every 'step_bars'
    
    range_high = pd.Series(index=df.index, dtype=float)
    range_low = pd.Series(index=df.index, dtype=float)

    # We iterate by steps to set the values
    # For a block starting at 'i', we use the rolling value from 'i' (which looks back 'window' bars)
    # This is safe because rolling_high[i] only knows 0..i
    for i in range(0, len(df), step_bars):
        val_high = rolling_high.iloc[i]
        val_low = rolling_low.iloc[i]
        
        # Apply this value to the next 'step_bars' (forward fill basically)
        end_idx = min(i + step_bars, len(df))
        range_high.iloc[i:end_idx] = val_high
        range_low.iloc[i:end_idx] = val_low

    return range_high, range_low


def _vwap_with_value_area(
    df: pd.DataFrame,
    window: int,
    va_percent: float = 0.70,
) -> tuple[
    pd.Series, pd.Series, pd.Series, pd.Series
]:

    window = int(max(2, window))
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    tp = (high + low + close) / 3.0

    # VWAP
    pv = tp * volume
    sum_pv = pv.rolling(window, min_periods=1).sum()
    sum_v = volume.rolling(window, min_periods=1).sum().replace(0, np.nan)
    vwap = (sum_pv / sum_v).ffill().bfill()

    # Standard Deviation pentru Value Area
    sum_p2v = ((tp ** 2) * volume).rolling(window, min_periods=1).sum()
    var = (sum_p2v / sum_v - vwap ** 2).clip(lower=0)
    stdev = np.sqrt(var).fillna(0)

    # Value Area aproximată ca VWAP ± factor x stdev
    va_factor = 1.0  # Aproximativ 1 stdev pentru 68%

    vah = (vwap + va_factor * stdev).astype(float)  # Value Area High
    val = (vwap - va_factor * stdev).astype(float)  # Value Area Low

    # POC
    n = len(df)
    poc = np.zeros(n)
    num_bins = 30

    for i in range(n):
        start_idx = max(0, i - window + 1)

        window_tp = tp.iloc[start_idx:i+1].values
        window_vol = volume.iloc[start_idx:i+1].values
        window_high = high.iloc[start_idx:i+1].values
        window_low = low.iloc[start_idx:i+1].values

        if len(window_tp) < 2:
            poc[i] = close.iloc[i]
            continue

        price_min = np.min(window_low)
        price_max = np.max(window_high)

        if price_max <= price_min:
            poc[i] = close.iloc[i]
            continue

        # Bins de pret
        bins = np.linspace(price_min, price_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        bin_indices = np.digitize(window_tp, bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        bin_volumes = np.zeros(num_bins)
        for j, bin_idx in enumerate(bin_indices):
            bin_volumes[bin_idx] += window_vol[j]

        # POC = centrul bin-ului cu cel mai mare volum
        poc_bin_idx = np.argmax(bin_volumes)
        poc[i] = bin_centers[poc_bin_idx]

    poc_series = pd.Series(poc, index=df.index).astype(float)

    return vwap.astype(float), poc_series, vah, val


def _volume_oscillator_normalized(df: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.Series:
    """
    Volume Oscillator Normalized: (EMA_fast(vol) - EMA_slow(vol)) / EMA_slow(vol) x 100
    Normalizat între -100 si 100
    """
    volume = df["volume"].astype(float)
    ema_fast = _ema(volume, fast)
    ema_slow = _ema(volume, slow).replace(0, np.nan)

    vol_osc = ((ema_fast - ema_slow) / ema_slow * 100).replace([np.inf, -np.inf], np.nan).fillna(0)

    vol_osc_norm = vol_osc.clip(-100, 100).astype(float)

    return vol_osc_norm


def _cumulative_delta_normalized(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Cumulative Delta Normalized: aproximare bazată pe close position în range
    Delta = (close - low) / (high - low) - 0.5, scalat
    Pozitiv = buying pressure
    Negativ = selling pressure
    Normalizat pe fereastra rulanta.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    # Calculăm poziția close în range sunt calcule grele :)
    range_size = (high - low).replace(0, np.nan)
    close_position = (close - low) / range_size  # 0=low, 1=high

    # Delta per bară: pozitiv dacă close spre high, negativ dacă spre low
    delta_per_bar = (close_position - 0.5) * 2 * volume  # Scalat cu volum

    # Cumulative delta pe fereastră
    cum_delta = delta_per_bar.rolling(window, min_periods=1).sum()

    # Normalizare
    cum_delta_max = cum_delta.rolling(window * 2, min_periods=1).max().abs()
    cum_delta_min = cum_delta.rolling(window * 2, min_periods=1).min().abs()
    normalizer = pd.concat([cum_delta_max, cum_delta_min], axis=1).max(axis=1).replace(0, 1)

    cum_delta_norm = (cum_delta / normalizer * 100).clip(-100, 100).fillna(0).astype(float)

    return cum_delta_norm


class VVRPStrategyBETA0_1(IStrategy):
    """
    Volume Weighted Range

    Logica strategiei este simpla, acum am conceput_o pentru test :)
    - LONG când prețul cade sub range_low suport dinamic
    - SHORT când prețul urcă peste range_high rezistență dinamică
    - Confirmări: RSI, Volume Oscillator, Cumulative Delta
    - Exit la POC (VWAP) sau la range opus
    ~ Daca grupul dorește pot face și o versiune avansată cu multiple moduri de intrare/ieșire
      eventual
      sa posteze versiunea imbunatatită sa vad si eu unde am gresit

    ~ As vrea din suflet sa repostati strategia cu imbunatatirile voastre, chiar ma ajuta sa invat
      din propriile greseli :)

    Notă: Aceasta este o versiune beta inițială pentru testare și optimizare.
    Nu este recomandată pentru utilizare live fără teste și ajustări suplimentare.
    !!!!!!!!!!!!!!!  Salutarii de la DarkReaper :) !!!!!!!!!!!!!!!!!!!!!
    """

    timeframe = "15m" # merge si pe 1h si 30m 4h etc testeaza!!!
    can_short = True
    startup_candle_count = 200

    # Risk Management
    stoploss = -0.28
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    minimal_roi = {
        "0": 0.99
    }

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Range
    range_window = IntParameter(20, 100, default=48, space="buy", optimize=True)
    range_step = IntParameter(4, 24, default=8, space="buy", optimize=True) #trepte mai frecvente

    #  VWAP
    vwap_window = IntParameter(20, 200, default=48, space="buy", optimize=True) #POC mai reactiv


    rsi_length = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsi_oversold = IntParameter(20, 45, default=40, space="buy", optimize=True)  # Mai relaxat
    rsi_overbought = IntParameter(55, 80, default=60, space="buy", optimize=True)  # Mai relaxat


    vol_osc_fast = IntParameter(3, 10, default=5, space="buy", optimize=True)
    vol_osc_slow = IntParameter(15, 30, default=20, space="buy", optimize=True)
    vol_osc_threshold = DecimalParameter(-50, 50, default=0, decimals=0, space="buy", optimize=True)


    cum_delta_window = IntParameter(10, 50, default=20, space="buy", optimize=True)
    cum_delta_threshold = DecimalParameter(
        -30,
        30,
        default=0,
        decimals=0,
        space="buy",
        optimize=True,
    )

    # ===== Entry Mode =====
    entry_mode = CategoricalParameter(
        ["range_only", "range_rsi", "range_volume", "range_all"],
        default="range_only",  # Cel mai sensibil mod
        space="buy",
        optimize=True
    )


    exit_at_poc = CategoricalParameter(
        ["on", "off"],
        default="on",
        space="sell",
        optimize=True,
    )
    exit_at_opposite_range = CategoricalParameter(
        ["on", "off"],
        default="off",
        space="sell",
        optimize=True,
    )


    plot_config = {
        'main_plot': {
            'range_high': {'color': '#FF1744', 'linewidth': 2},
            'range_low': {'color': '#00E676', 'linewidth': 2},
            'vwap': {'color': '#2962FF', 'linewidth': 1.5},
            'poc': {'color': '#FF9800', 'linewidth': 1, 'linestyle': '--'},
            'vwap_value_area_high': {'color': '#00BCD4', 'linewidth': 1, 'linestyle': ':'},
            'vwap_value_area_low': {'color': '#00BCD4', 'linewidth': 1, 'linestyle': ':'},
        },
        'subplots': {
            "RSI": {
                'rsi': {'color': '#FF9800', 'linewidth': 1.5},
            },
            "Volume Oscillator": {
                'volume_osc_norm': {'color': '#00BCD4', 'linewidth': 1.5},
            },
            "Cumulative Delta": {
                'cum_delta_norm': {'color': '#2962FF', 'linewidth': 1.5},
            },
            "Volume": {
                'volume': {'color': '#78909C', 'type': 'bar'},
            },
        }
    }

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = dataframe.copy()


        df["range_high"], df["range_low"] = _stepped_range(
            df,
            self.range_window.value,
            self.range_step.value
        )

        (
            df["vwap"],
            df["poc"],
            df["vwap_value_area_high"],
            df["vwap_value_area_low"],
        ) = _vwap_with_value_area(
            df,
            self.vwap_window.value,
        )


        df["rsi"] = _rsi(df["close"], self.rsi_length.value)


        df["volume_osc_norm"] = _volume_oscillator_normalized(
            df,
            self.vol_osc_fast.value,
            self.vol_osc_slow.value
        )

        df["cum_delta_norm"] = _cumulative_delta_normalized(
            df,
            self.cum_delta_window.value
        )

        df["atr"] = _atr(df, 14)

        df["dist_from_range_high"] = (
            ((df["close"] - df["range_high"]) / df["range_high"]) * 100
        ).astype(float)
        df["dist_from_range_low"] = (
            ((df["close"] - df["range_low"]) / df["range_low"]) * 100
        ).astype(float)

        return df

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = dataframe.copy()
        df["enter_long"] = 0
        df["enter_short"] = 0

        range_size = df["range_high"] - df["range_low"]
        tolerance = range_size * 0.01

        range_long_touch = df["low"] <= (df["range_low"] + tolerance)
        range_long_break = df["close"] < df["range_low"]
        range_long = range_long_touch | range_long_break

        range_short_touch = df["high"] >= (df["range_high"] - tolerance)
        range_short_break = df["close"] > df["range_high"]
        range_short = range_short_touch | range_short_break

        rsi_oversold = df["rsi"] < float(self.rsi_oversold.value)
        rsi_overbought = df["rsi"] > float(self.rsi_overbought.value)
        rsi_neutral_low = df["rsi"] < 50  # RSI sub 50 = momentum descendent
        rsi_neutral_high = df["rsi"] > 50  # RSI peste 50 = momentum ascendent

        vol_positive = df["volume_osc_norm"] > float(self.vol_osc_threshold.value)
        vol_negative = df["volume_osc_norm"] < -float(self.vol_osc_threshold.value)

        delta_bullish = df["cum_delta_norm"] > float(self.cum_delta_threshold.value)
        delta_bearish = df["cum_delta_norm"] < -float(self.cum_delta_threshold.value)

        mode = self.entry_mode.value

        if mode == "range_only":
            # Doar range, fără alte filtre - CEL MAI SENSIBIL
            long_cond = range_long
            short_cond = range_short

        elif mode == "range_rsi":
            # Range + RSI relaxat (nu neapărat oversold/overbought)
            long_cond = range_long & (rsi_oversold | rsi_neutral_low)
            short_cond = range_short & (rsi_overbought | rsi_neutral_high)

        elif mode == "range_volume":
            # Range + Volume
            long_cond = range_long & vol_positive
            short_cond = range_short & vol_negative

        else:  # range_all
            # Toate condițiile - CEL MAI STRICT
            long_cond = range_long & rsi_oversold & delta_bullish
            short_cond = range_short & rsi_overbought & delta_bearish

        df.loc[long_cond, "enter_long"] = 1
        df.loc[short_cond, "enter_short"] = 1

        return df

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = dataframe.copy()
        df["exit_long"] = 0
        df["exit_short"] = 0

        # Exit la POC
        if self.exit_at_poc.value == "on":
            # Long exit când pret atinge POC/VWAP
            poc_long_exit = df["close"] >= df["poc"]
            # Short exit când pret atinge POC/VWAP
            poc_short_exit = df["close"] <= df["poc"]
        else:
            poc_long_exit = pd.Series(False, index=df.index)
            poc_short_exit = pd.Series(False, index=df.index)

        if self.exit_at_opposite_range.value == "on":

            range_long_exit = df["close"] >= df["range_high"]

            range_short_exit = df["close"] <= df["range_low"]
        else:
            range_long_exit = pd.Series(False, index=df.index)
            range_short_exit = pd.Series(False, index=df.index)

        rsi_long_exit = df["rsi"] > 70  # Exit long când RSI overbought
        rsi_short_exit = df["rsi"] < 30  # Exit short când RSI oversold

        exit_long = poc_long_exit | range_long_exit | rsi_long_exit
        exit_short = poc_short_exit | range_short_exit | rsi_short_exit

        df.loc[exit_long.fillna(False), "exit_long"] = 1
        df.loc[exit_short.fillna(False), "exit_short"] = 1

        return df

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Stoploss dinamic bazat pe ATR.
        """
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is None or df.empty:
                return self.stoploss

            atr = float(df.iloc[-1].get("atr", np.nan))
            if not np.isfinite(atr) or atr <= 0:
                return self.stoploss

            atr_pct = atr / float(current_rate)

            # Profit protection
            if current_profit > 0.04:
                return -0.01  # Lock in la 1%
            if current_profit > 0.02:
                return -0.015  # Lock in la 1.5%

            # ATR-based stoploss (2x ATR)
            sl = -float(2.0 * atr_pct)
            return float(np.clip(sl, -0.06, -0.015))  # Min -6%, Max -1.5%

        except Exception:
            return self.stoploss
