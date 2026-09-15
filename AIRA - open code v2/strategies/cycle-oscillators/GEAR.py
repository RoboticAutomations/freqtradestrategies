# Creador> Gregorio.exe (@DickMan16)
"""
GEAR — estrategia FIEL al 'ift_zlrsi_fast_clean' de Gregorio (baseline latch 0.85).
============================================================================================
Indicador: IFT( RSI( ZLEMA(close) ) ) -> oscilador [-1,1] (calco fiel, night_pv/ift_core.py).
LIMPIEZA = 'clean latch' con umbral fijo 0.85 (baseline de Gregorio,
mantiene el ÚLTIMO extremo (|ift|>=0.85) y ffill -> una ONDA CUADRADA (escalera). Su SIGNO es
el estado: +1 = último extremo de SOBRECOMPRA, -1 = último extremo de SOBREVENTA.

La señal ES el estado (no hay umbrales que cruzar en una onda cuadrada): reversión a la media,
como en ift_core (valle del oscilador -> long, pico -> short). Entra en la barra donde el estado
FLIPEA. Todo causal (latch = ffill del pasado), sin repintado (verificado por truncación).

⚠ Se prueba la calidad del indicador antes su comportamiento en todos los regimenes de mercado, continuar con el refinamiento de tanh(k·x³/(x²+δ²))
ya que aveces no define estados.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import talib
from pandas import DataFrame
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter


class GEAR(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True
    startup_candle_count = 300
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    # ── Riesgo: defaults CONSERVADORES — REFINAR con backtest ──────────────────
    stoploss = -0.06
    minimal_roi = {"0": 0.03, "40": 0.015, "120": 0.0}
    trailing_stop = False

    # ── Indicador ift_zlrsi + latch baseline (hyperopt-able) ───────────────────
    zlema_period = IntParameter(2, 6, default=3, space="buy", optimize=True)
    rsi_period   = IntParameter(5, 14, default=7, space="buy", optimize=True)
    latch_thresh = DecimalParameter(0.70, 0.95, default=0.85, decimals=2, space="buy", optimize=True)
    # dirección: True = reversión a la media (fade extremos: sobreventa->long). False = momentum.
    mean_reversion = BooleanParameter(default=True, space="buy", optimize=False)
    # REFINAMIENTO opt-in (fase 2): umbral por envolvente adaptativa (mata ranciedad). Default OFF = fiel al baseline.
    use_envelope = BooleanParameter(default=False, space="buy", optimize=False)
    env_frac  = DecimalParameter(0.45, 0.90, default=0.70, decimals=2, space="buy", optimize=True)
    env_floor = DecimalParameter(0.20, 0.55, default=0.35, decimals=2, space="buy", optimize=True)
    env_win   = IntParameter(60, 240, default=120, space="buy", optimize=True)
    # REFINAMIENTO opt-in (fase 2): squash ZONA-MUERTA tanh(k·x³/(x²+δ²)) — cosmético (nitidez/ruido), no crea edge.
    use_deadzone = BooleanParameter(default=False, space="buy", optimize=False)
    dz_k     = DecimalParameter(0.30, 1.20, default=0.60, decimals=2, space="buy", optimize=True)
    dz_delta = DecimalParameter(0.80, 2.50, default=1.50, decimals=2, space="buy", optimize=True)

    # Leverage fijo conservador. 40x = validar liquidación aparte.
    leverage_value = 3.0

    # =========================================================================
    #  ift_zlrsi — calco fiel de Gregorio (ZLEMA -> RSI -> Inverse Fisher). Causal.
    # =========================================================================
    @staticmethod
    def _ift_zlrsi(close: pd.Series, zlema_period: int, rsi_period: int,
                   dz_k: float = None, dz_delta: float = 1.5) -> pd.Series:
        lag = (int(zlema_period) - 1) // 2
        zlema_adj = 2.0 * close - close.shift(lag)                 # zero-lag EMA (adelanta)
        zlema = talib.EMA(zlema_adj.to_numpy(dtype=float), timeperiod=int(max(2, zlema_period)))
        rsi = talib.RSI(zlema, timeperiod=int(max(2, rsi_period)))
        x = np.clip((rsi / 100.0) * 10.0 - 5.0, -5.0, 5.0)          # RSI -> [-5, 5]
        if dz_k is not None:
            # ZONA-MUERTA (fisher-mod, opt-in): tanh(k·x³/(x²+δ²)) — chato cerca de 0 (mata ruido/jitter),
            # agudo en los extremos. Autocontenida (no necesita el UKF). COSMÉTICO: desacopla nitidez de
            # ruido pero NO crea edge (empata al IFT plano en 15m real, ver fisher-mod).
            arg = float(dz_k) * (x ** 3) / (x ** 2 + float(dz_delta) ** 2 + 1e-12)
            ift = np.tanh(arg)
        else:
            ift = np.tanh(x)                                       # IFT PLANO = (e^{2x}-1)/(e^{2x}+1). Baseline Gregorio.
        return pd.Series(ift, index=close.index).fillna(0.0)

    @staticmethod
    def _clean_latch(ift: pd.Series, thresh: float) -> pd.Series:
        """BASELINE Gregorio: mantiene el último extremo (|ift|>=thresh) y ffill. Onda cuadrada. Causal."""
        ext = ift.abs() >= float(thresh)
        return ift.where(ext).ffill().fillna(0.0)

    @staticmethod
    def _clean_latch_envelope(ift: pd.Series, frac: float, floor: float, win: int) -> pd.Series:
        """REFINAMIENTO opt-in: umbral = max(floor, frac·envolvente_trailing(|ift|)) — mata ranciedad. Causal."""
        env = ift.abs().rolling(int(win), min_periods=1).max()
        thr = np.maximum(float(floor), float(frac) * env.to_numpy())
        ext = ift.abs().to_numpy() >= thr
        return ift.where(pd.Series(ext, index=ift.index)).ffill().fillna(0.0)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        c = dataframe["close"].astype(float)
        _dzk = self.dz_k.value if self.use_deadzone.value else None
        ift = self._ift_zlrsi(c, self.zlema_period.value, self.rsi_period.value,
                              dz_k=_dzk, dz_delta=self.dz_delta.value)
        if self.use_envelope.value:
            latch = self._clean_latch_envelope(ift, self.env_frac.value, self.env_floor.value, self.env_win.value)
        else:
            latch = self._clean_latch(ift, self.latch_thresh.value)
        dataframe["ift_zlrsi"] = ift
        dataframe["ift_latch"] = latch                    # onda cuadrada (escalera)
        dataframe["ift_state"] = np.sign(latch.to_numpy())  # +1 sobrecompra-últ / -1 sobreventa-últ
        return dataframe

    def _flips(self, dataframe: DataFrame):
        """Barras donde el ESTADO flipea (la señal en una onda cuadrada). Causal (usa i e i-1)."""
        st = dataframe["ift_state"]
        prev = st.shift(1).fillna(0.0)
        to_os = (st < 0) & (prev >= 0)     # nuevo extremo de SOBREVENTA
        to_ob = (st > 0) & (prev <= 0)     # nuevo extremo de SOBRECOMPRA
        return to_os, to_ob

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        to_os, to_ob = self._flips(dataframe)
        vol = dataframe["volume"] > 0
        if self.mean_reversion.value:      # fade: sobreventa -> LONG, sobrecompra -> SHORT (como ift_core)
            long_sig, short_sig = to_os, to_ob
        else:                              # momentum: sobrecompra -> LONG, sobreventa -> SHORT
            long_sig, short_sig = to_ob, to_os
        dataframe.loc[long_sig  & vol, ["enter_long",  "enter_tag"]] = (1, "ift_state_long")
        dataframe.loc[short_sig & vol, ["enter_short", "enter_tag"]] = (1, "ift_state_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        to_os, to_ob = self._flips(dataframe)
        # Sale al flip OPUESTO (stop-and-reverse) — el mismo evento que abre la contraria.
        if self.mean_reversion.value:
            dataframe.loc[to_ob, ["exit_long",  "exit_tag"]] = (1, "ift_flip_ob")
            dataframe.loc[to_os, ["exit_short", "exit_tag"]] = (1, "ift_flip_os")
        else:
            dataframe.loc[to_os, ["exit_long",  "exit_tag"]] = (1, "ift_flip_os")
            dataframe.loc[to_ob, ["exit_short", "exit_tag"]] = (1, "ift_flip_ob")
        return dataframe

    def leverage(self, pair: str, current_time, current_rate: float, proposed_leverage: float,
                 max_leverage: float, entry_tag, side: str, **kwargs) -> float:
        return float(min(self.leverage_value, max_leverage))

    plot_config = {
        "main_plot": {},
        "subplots": {
            "IFT ZLRSI + latch 0.85 (baseline · onda cuadrada) — la señal es el ESTADO": {
                "ift_zlrsi": {"color": "#9e9e9e", "width": 2},
            },
            "latch 0.85 ": {
                "ift_latch": {"color": "#e91e63", "width": 2},
            },
            "IFT ZLRSI": {
                "ift_state": {"color": "#3949ab", "width": 2},
            },
        },
    }
