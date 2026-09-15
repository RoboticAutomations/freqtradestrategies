"""
ANF_v2 — Adaptive Nexus Forge with HMM Regime Detection (Freqtrade)
===================================================================
EN: ANF plus a Gaussian Hidden Markov Model that classifies the market regime
    (ranging / trending / volatile) from stationary observables (log-returns,
    realized volatility, volume z-score, return autocorrelation) on the BTC
    anchor. The posterior probabilities modulate ensemble weighting and dynamic
    leverage (cut up to 40% in a confident volatile regime, +15% in a confident
    trend). The HMM is a controller layered on ANF, never a standalone signal;
    if hmmlearn is missing or the model is unfit it degrades to ANF's behaviour.

ES: ANF más un Modelo Oculto de Markov Gaussiano que clasifica el régimen de
    mercado (lateral / tendencia / volátil) a partir de observables estacionarios
    (log-returns, volatilidad realizada, z-score de volumen, autocorrelación) en
    el ancla BTC. Las probabilidades posteriores modulan la ponderación del
    ensemble y el leverage dinámico (recorta hasta 40% en régimen volátil con
    confianza, +15% en tendencia con confianza). El HMM es un controlador sobre
    ANF, nunca una señal aislada; si falta hmmlearn o el modelo no está entrenado,
    degrada al comportamiento de ANF.

Timeframe: 1h  |  Shorts: yes  |  Freqtrade: v3+
Exchanges: Binance Futures, Hyperliquid (USDT-M perpetuals).
Optional dependency: hmmlearn (pip install hmmlearn). Without it, falls back to ANF.

EN: ML and HMM train live/dry-run only; both are disabled in backtest to avoid
    look-ahead, so backtests run on the technical tiers alone — a pessimistic floor.
ES: ML y HMM solo entrenan en live/dry-run; ambos se desactivan en backtest para
    evitar look-ahead, así que el backtest corre solo los tiers técnicos — un suelo
    pesimista.

EN: Dual-mode file. (1) As a Freqtrade strategy: `freqtrade trade --strategy
    ANF_v2`. (2) Standalone, WITHOUT Freqtrade installed, to train the HMM on a
    long history and produce a frozen model:
        pip install numpy pandas hmmlearn ccxt
        python ANF_v2.py --train-hmm --years 3 --anchor-pair BTC/USDT:USDT
    This downloads BTC history via ccxt, fits the HMM on the whole series, and
    writes a frozen user_data/ml_models/ANF_v2/hmm_regime.pkl. Copy that file to
    the VPS: ANF_v2 loads it on startup and never overwrites a frozen model.
    Delete it to return to the default rolling-window training. The .pkl stores
    the training library versions and warns on load if they differ from the
    running environment (pickled hmmlearn models are version-sensitive).
ES: Fichero de doble modo. (1) Como estrategia Freqtrade: `freqtrade trade
    --strategy ANF_v2`. (2) Standalone, SIN Freqtrade instalado, para entrenar el
    HMM con histórico largo y generar un modelo congelado:
        pip install numpy pandas hmmlearn ccxt
        python ANF_v2.py --train-hmm --years 3 --anchor-pair BTC/USDT:USDT
    Descarga el histórico de BTC vía ccxt, ajusta el HMM sobre toda la serie y
    escribe un user_data/ml_models/ANF_v2/hmm_regime.pkl congelado. Copia ese
    fichero al VPS: ANF_v2 lo carga al arrancar y nunca sobrescribe un modelo
    congelado. Bórralo para volver al entrenamiento por ventana móvil. El .pkl
    guarda las versiones de las librerías de entrenamiento y avisa al cargar si
    difieren del entorno de ejecución (los modelos hmmlearn serializados son
    sensibles a la versión).

EN: Based on AlexNexusForge V8 (class AlexNexusForgeV8AIV7) by Alex, shared in
    the Freqtrade community. ANF keeps that core architecture (ML ensemble +
    Wavelet/FFT + Murrey Math + multi-tier long/short entries) and ANF_v2 adds
    the HMM regime layer on top. Full credit to the original author. The base
    strategy carried no explicit license; following that, ANF_v2 is shared in
    the same community spirit, without a formal license. Use, study and adapt
    it freely, keep the attribution, and do your own testing before risking
    real funds.
ES: Basada en AlexNexusForge V8 (clase AlexNexusForgeV8AIV7) de Alex, compartida
    en la comunidad de Freqtrade. ANF conserva esa arquitectura base (ensemble
    ML + Wavelet/FFT + Murrey Math + entradas long/short multi-nivel) y ANF_v2
    añade encima la capa de régimen HMM. Todo el crédito al autor original. La
    estrategia base no incluía licencia explícita; siguiendo eso, ANF_v2 se
    comparte con el mismo espíritu comunitario, sin licencia formal. Úsala,
    estúdiala y adáptala libremente, mantén la atribución, y haz tus propias
    pruebas antes de arriesgar fondos reales.


⚠ Not financial advice. Run dry-run first. | No es asesoramiento financiero. Prueba en dry-run primero.

Author / Autor:   @bustillo
Community / Comunidad (ES):   https://t.me/EsFreqtrade
"""

import logging
import numpy as np
import pandas as pd
import pickle
import warnings
import sys
import os
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict
from importlib import metadata
from functools import lru_cache
# EN: talib and scipy are heavy/native dependencies needed only by the trading
#     strategy and its advanced indicators — NOT by the HMM trainer. We import
#     them lazily (via the _get_* helpers below) so `python ANF_v2.py
#     --train-hmm` runs on a machine that only has numpy/pandas/hmmlearn/ccxt.
# ES: talib y scipy son dependencias pesadas/nativas que solo necesita la
#     estrategia de trading y sus indicadores avanzados — NO el entrenador del
#     HMM. Las importamos de forma perezosa (con los helpers _get_* de abajo)
#     para que `python ANF_v2.py --train-hmm` funcione en una máquina que solo
#     tenga numpy/pandas/hmmlearn/ccxt.
_LAZY = {}


def _get_talib():
    if 'ta' not in _LAZY:
        import talib.abstract as _ta
        _LAZY['ta'] = _ta
    return _LAZY['ta']


def _get_scipy_fft():
    if 'fft' not in _LAZY:
        from scipy.fft import fft as _fft, fftfreq as _fftfreq
        _LAZY['fft'] = _fft
        _LAZY['fftfreq'] = _fftfreq
    return _LAZY['fft'], _LAZY['fftfreq']


def _get_scipy_stats():
    if 'skew' not in _LAZY:
        from scipy.stats import skew as _skew, kurtosis as _kurtosis
        _LAZY['skew'] = _skew
        _LAZY['kurtosis'] = _kurtosis
    return _LAZY['skew'], _LAZY['kurtosis']

# EN: Freqtrade is required to RUN the strategy (live/dry-run/backtest), but NOT
#     to train the HMM offline in standalone mode. We import it defensively so
#     `python ANF_v2.py --train-hmm ...` works on a machine without Freqtrade
#     installed. When absent, the ANF_v2 strategy class is simply not defined;
#     the HMMRegimeDetector and the CLI trainer below do not need it.
# ES: Freqtrade es necesario para EJECUTAR la estrategia (live/dry-run/backtest),
#     pero NO para entrenar el HMM offline en modo standalone. Lo importamos de
#     forma defensiva para que `python ANF_v2.py --train-hmm ...` funcione en una
#     máquina sin Freqtrade. Si no está, la clase ANF_v2 simplemente no se define;
#     el HMMRegimeDetector y el entrenador CLI de más abajo no lo necesitan.
try:
    import freqtrade.vendor.qtpylib.indicators as qtpylib
    from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter
    from freqtrade.persistence import Trade
    FREQTRADE_AVAILABLE = True
except ImportError:
    FREQTRADE_AVAILABLE = False

    # EN: Minimal stand-ins so the ANF_v2 class can still be *defined* (not run)
    #     on a machine without Freqtrade — e.g. when training the HMM offline via
    #     the CLI. They are never used to trade; live execution always has the
    #     real Freqtrade classes. Each parameter stub just remembers its default
    #     and exposes it via `.value`, mirroring Freqtrade's interface.
    # ES: Sustitutos mínimos para que la clase ANF_v2 se pueda *definir* (no
    #     ejecutar) en una máquina sin Freqtrade — p. ej. al entrenar el HMM
    #     offline por CLI. Nunca se usan para operar; la ejecución en vivo siempre
    #     tiene las clases reales de Freqtrade. Cada stub de parámetro recuerda su
    #     default y lo expone vía `.value`, imitando la interfaz de Freqtrade.
    qtpylib = None
    Trade = None

    class IStrategy:  # noqa: D401 - minimal stand-in
        pass

    class _ParamStub:
        def __init__(self, *args, **kwargs):
            # Freqtrade params are either Param(low, high, default=...) or
            # Param(default=...). Recover the default from kwargs or the 3rd arg.
            if 'default' in kwargs:
                self.value = kwargs['default']
            elif len(args) >= 3:
                self.value = args[2]
            elif len(args) == 1:
                self.value = args[0]
            else:
                self.value = None

    def DecimalParameter(*args, **kwargs):
        return _ParamStub(*args, **kwargs)

    def IntParameter(*args, **kwargs):
        return _ParamStub(*args, **kwargs)

    def BooleanParameter(*args, **kwargs):
        return _ParamStub(*args, **kwargs)

# Suppress deprecation warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

# Each cache is capped; oldest entries are evicted when full.
from collections import OrderedDict as _OD
_LOG_CACHE_MAX_GLOBAL = 256
_LOG_CACHE_MAX_PER_PAIR = 64
_LOG_CACHE_MAX_PAIRS = 128
_log_last_msg: "_OD[str, float]" = _OD()
_log_last_pair_msg: "_OD[str, _OD[str, float]]" = _OD()

def _log_info_throttled(msg: str, pair: Optional[str] = None, interval_sec: float = 300.0):
    """Log info only once per interval; bounded caches to prevent leaks."""
    import time
    now = time.time()
    if pair:
        pair_cache = _log_last_pair_msg.get(pair)
        if pair_cache is None:
            if len(_log_last_pair_msg) >= _LOG_CACHE_MAX_PAIRS:
                _log_last_pair_msg.popitem(last=False)  # drop oldest pair
            pair_cache = _OD()
            _log_last_pair_msg[pair] = pair_cache
        last = pair_cache.get(msg, 0)
        if now - last < interval_sec:
            return
        if len(pair_cache) >= _LOG_CACHE_MAX_PER_PAIR:
            pair_cache.popitem(last=False)
        pair_cache[msg] = now
    else:
        last = _log_last_msg.get(msg, 0)
        if now - last < interval_sec:
            return
        if len(_log_last_msg) >= _LOG_CACHE_MAX_GLOBAL:
            _log_last_msg.popitem(last=False)
        _log_last_msg[msg] = now
    logger.info(msg)

try:
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier, 
        ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
    from sklearn.feature_selection import SelectKBest, f_classif
    SKLEARN_AVAILABLE = True

    # Check sklearn version using modern approach
    try:
        sklearn_version = metadata.version("scikit-learn")
        logger.info(f"Using scikit-learn version: {sklearn_version}")
    except Exception as e:
        logger.debug(f"Could not get sklearn version: {e}")

except ImportError as e:
    logger.warning(f"scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False

# Modern PyWavelets import
try:
    import pywt
    WAVELETS_AVAILABLE = True
    try:
        pywt_version = metadata.version("PyWavelets")
        logger.info(f"Using PyWavelets version: {pywt_version}")
    except Exception as e:
        logger.debug(f"Could not get PyWavelets version: {e}")
except ImportError as e:
    logger.warning(f"PyWavelets not available: {e}")
    WAVELETS_AVAILABLE = False


# Define Murrey Math level names for consistency
MML_LEVEL_NAMES = [
    "[-3/8]P", "[-2/8]P", "[-1/8]P", "[0/8]P", "[1/8]P",
    "[2/8]P", "[3/8]P", "[4/8]P", "[5/8]P", "[6/8]P",
    "[7/8]P", "[8/8]P", "[+1/8]P", "[+2/8]P", "[+3/8]P"
]
def calculate_minima_maxima(df, window):
    """
    Vectorized version. The original used a per-row Python loop which
    cost O(n*window). This version uses rolling().min()/max() and checks
    uniqueness in a single pass. Semantics preserved:
      - minima[i] = -window  if ha_close[i] is the unique min in [i-window, i]
      - maxima[i] =  window  if ha_close[i] is the unique max in [i-window, i]
    """
    if df is None or df.empty or 'ha_close' not in df.columns:
        return np.zeros(len(df) if df is not None else 0), np.zeros(len(df) if df is not None else 0)

    n = len(df)
    if n == 0 or window <= 0:
        return np.zeros(n), np.zeros(n)

    ha = df['ha_close'].astype(float)
    w = int(window) + 1  # include current bar (matches original i-window..i inclusive)

    roll_min = ha.rolling(window=w, min_periods=w).min()
    roll_max = ha.rolling(window=w, min_periods=w).max()

    # Uniqueness: count how many values in the window equal the current value.
    # Implemented via a rolling-sum of (ha == ha[i]) is hard pointwise — but
    # we can approximate "unique extreme" by checking the second-best in the window.
    # Robust trick: compare to rolling min/max excluding current value.
    # We shift by 1 and compare the previous-window min/max to be strictly worse.
    prev_min = ha.shift(1).rolling(window=window, min_periods=window).min()
    prev_max = ha.shift(1).rolling(window=window, min_periods=window).max()

    is_min = (ha == roll_min) & (ha < prev_min)
    is_max = (ha == roll_max) & (ha > prev_max)

    minima = np.where(is_min.fillna(False).to_numpy(), -window, 0).astype(float)
    maxima = np.where(is_max.fillna(False).to_numpy(),  window, 0).astype(float)
    return minima, maxima


def calc_slope_advanced(series, period):
    """
    Enhanced linear regression slope calculation with Wavelet Transform and FFT analysis
    for superior trend detection and noise filtering
    """
    if len(series) < period:
        return 0

    # Use only the last 'period' values for consistency
    y = series.values[-period:]

    # Enhanced data validation
    if np.isnan(y).any() or np.isinf(y).any():
        return 0

    # Check for constant values (no trend)
    if np.all(y == y[0]):
        return 0

    try:
        # === 1. WAVELET DENOISING ===
        # decomposition can't hit UnboundLocalError in the edge case where
        # WAVELETS_AVAILABLE and len(y)>=8 hold but use_level falls to 0.
        coeffs = None
        if WAVELETS_AVAILABLE and len(y) >= 8:
            wavelet = 'db4'
            try:
                w = pywt.Wavelet(wavelet)
                max_level = pywt.dwt_max_level(len(y), w.dec_len)
                use_level = min(3, max_level)  # cap at 3 but adapt if shorter series
            except Exception:
                use_level = 1
            if use_level >= 1:
                # wavelet_trend_analysis below for consistent coefficients.
                coeffs = pywt.wavedec(y, wavelet, level=use_level, mode='symmetric')
                threshold = 0.1 * np.std(coeffs[-1]) if len(coeffs) > 1 else 0.0
                coeffs_thresh = list(coeffs)
                for i in range(1, len(coeffs_thresh)):
                    coeffs_thresh[i] = pywt.threshold(coeffs_thresh[i], threshold, mode='soft')
                y_denoised = pywt.waverec(coeffs_thresh, wavelet, mode='symmetric')
                if len(y_denoised) != len(y):
                    y_denoised = y_denoised[:len(y)]
            else:
                y_denoised = y
        else:
            y_denoised = y

        # === 2. FFT FREQUENCY ANALYSIS ===
        # Analyze dominant frequencies to identify trend components
        if len(y_denoised) >= 4:
            # Apply FFT
            fft, fftfreq = _get_scipy_fft()
            fft_values = fft(y_denoised)
            freqs = fftfreq(len(y_denoised))

            # Get magnitude spectrum
            magnitude = np.abs(fft_values)

            # Find dominant frequency (excluding DC component)
            non_dc_indices = np.where(freqs != 0)[0]
            if len(non_dc_indices) > 0:
                dominant_freq_idx = non_dc_indices[np.argmax(magnitude[non_dc_indices])]
                dominant_freq = freqs[dominant_freq_idx]

                # Calculate trend strength based on frequency content
                trend_frequency_weight = 1.0 / (1.0 + abs(dominant_freq) * 10)
            else:
                trend_frequency_weight = 1.0
        else:
            trend_frequency_weight = 1.0

        # === 3. MULTI-SCALE SLOPE CALCULATION ===
        x = np.linspace(0, period-1, period)

        # Original slope calculation
        slope_original = np.polyfit(x, y, 1)[0]

        # Wavelet-denoised slope calculation
        slope_denoised = np.polyfit(x, y_denoised, 1)[0]

        # === 4. WAVELET-BASED TREND DECOMPOSITION ===
        # above can leave coeffs as None when use_level=0 (pathological short
        # series). Without this guard we'd hit UnboundLocalError → exception
        # → silent fallback. The guard skips the trend term and uses denoised.
        if WAVELETS_AVAILABLE and len(y) >= 8 and coeffs is not None:
            # Extract trend component using wavelet approximation
            approx_coeffs = coeffs[0]  # Approximation coefficients (trend)

            # Reconstruct trend component
            trend_component = pywt.upcoef(
                'a', approx_coeffs, wavelet, level=use_level, take=len(y))
            if len(trend_component) > len(y):
                trend_component = trend_component[:len(y)]
            elif len(trend_component) < len(y):
                # Pad with last value if needed
                pad_length = len(y) - len(trend_component)
                trend_component = np.pad(trend_component, (0, pad_length), mode='edge')

            # Calculate slope of trend component
            slope_trend = np.polyfit(x, trend_component, 1)[0]
        else:
            slope_trend = slope_denoised

        # === 5. FREQUENCY-WEIGHTED SLOPE COMBINATION ===
        # Weight slopes based on signal characteristics
        weights = {
            'original': 0.3,
            'denoised': 0.4,
            'trend': 0.3
        }
        
        # Adjust weights based on noise level
        noise_level = np.std(y - y_denoised) / np.std(y) if np.std(y) > 0 else 0
        if noise_level > 0.1:  # High noise
            weights = {'original': 0.2, 'denoised': 0.5, 'trend': 0.3}
        elif noise_level < 0.05:  # Low noise
            weights = {'original': 0.4, 'denoised': 0.3, 'trend': 0.3}
        
        # Combined slope calculation
        slope_combined = (
            slope_original * weights['original'] +
            slope_denoised * weights['denoised'] +
            slope_trend * weights['trend']
        )
        
        # Apply frequency weighting
        final_slope = slope_combined * trend_frequency_weight
        
        # === 6. ENHANCED VALIDATION ===
        if np.isnan(final_slope) or np.isinf(final_slope):
            return (slope_original if not
                   (np.isnan(slope_original) or np.isinf(slope_original))
                   else 0)

        # Normalize extreme slopes
        max_reasonable_slope = np.std(y) / period
        if abs(final_slope) > max_reasonable_slope * 15:
            return np.sign(final_slope) * max_reasonable_slope * 15

        return final_slope

    except Exception:
        # Fallback to enhanced simple method if advanced processing fails
        try:
            # Apply simple moving average smoothing as fallback
            if len(y) >= 3:
                # center=False: a centred rolling window would peek at the
                # next candle, which is look-ahead in a live fallback path.
                y_smooth = (
                    pd.Series(y).rolling(window=3, center=False)
                    .mean().bfill().ffill().values
                )
                x = np.linspace(0, period-1, period)
                slope = np.polyfit(x, y_smooth, 1)[0]

                if not (np.isnan(slope) or np.isinf(slope)):
                    return slope

            # Ultimate fallback: simple difference
            simple_slope = (y[-1] - y[0]) / (period - 1)
            return (simple_slope if not
                   (np.isnan(simple_slope) or np.isinf(simple_slope))
                   else 0)

        except Exception:
            return 0


def calculate_advanced_trend_strength_with_wavelets(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced trend strength calculation using Wavelet Transform and FFT analysis
    """
    try:
        # === WAVELET-ENHANCED SLOPE CALCULATION ===
        dataframe['slope_5_advanced'] = dataframe['close'].rolling(5).apply(
            lambda x: calc_slope_advanced(x, 5), raw=False
        )
        dataframe['slope_10_advanced'] = dataframe['close'].rolling(10).apply(
            lambda x: calc_slope_advanced(x, 10), raw=False
        )
        dataframe['slope_20_advanced'] = dataframe['close'].rolling(20).apply(
            lambda x: calc_slope_advanced(x, 20), raw=False
        )
        
        # === WAVELET TREND DECOMPOSITION ===
        def wavelet_trend_analysis(series, window=20):
            """Analyze trend using adaptive wavelet (haar/db4), safe levels, symmetric mode, robust threshold."""
            if not WAVELETS_AVAILABLE or len(series) < window:
                return pd.Series([0.0] * len(series), index=series.index)
            results: list[float] = []
            for i in range(len(series)):
                if i < window:
                    results.append(0.0)
                    continue
                window_data = series.iloc[i-window+1:i+1].values
                n = len(window_data)
                if n < 12:
                    results.append(0.0)
                    continue
                wavelet_name = 'haar' if n < 24 else 'db4'
                try:
                    w = pywt.Wavelet(wavelet_name)
                    max_level = pywt.dwt_max_level(n, w.dec_len)
                except Exception:
                    max_level = 1
                if n < 48:
                    max_level = min(max_level, 2)
                use_level = max(1, min(3, max_level))
                try:
                    coeffs = pywt.wavedec(window_data, wavelet_name, level=use_level, mode='symmetric')
                    # Estimate sigma from finest detail
                    if len(coeffs) > 1 and len(coeffs[-1]):
                        detail = coeffs[-1]
                        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
                        thr = sigma * np.sqrt(2 * np.log(n)) if sigma > 0 else 0.0
                    else:
                        thr = 0.0
                    for j in range(1, len(coeffs)):
                        coeffs[j] = pywt.threshold(coeffs[j], thr, mode='soft')
                    approx = coeffs[0]
                    trend_strength = np.std(approx) / (np.std(window_data) + 1e-9)
                    direction = 0
                    if len(approx) >= 2:
                        direction = 1 if approx[-1] > approx[0] else -1
                    score = trend_strength * direction
                    if not np.isfinite(score):
                        score = 0.0
                    # Clamp extreme outliers
                    results.append(float(np.clip(score, -5, 5)))
                except Exception:
                    results.append(0.0)
            return pd.Series(results, index=series.index)
        
        # Apply wavelet trend analysis
        dataframe['wavelet_trend_strength'] = wavelet_trend_analysis(dataframe['close'])
        
        # === FFT-BASED CYCLE DETECTION ===
        def fft_cycle_analysis(series, window=50):
            """Detect market cycles using FFT"""
            if len(series) < window:
                return (pd.Series([0] * len(series), index=series.index),
                       pd.Series([0] * len(series), index=series.index))
            
            cycle_strength = []
            dominant_period = []
            
            for i in range(len(series)):
                if i < window:
                    cycle_strength.append(0)
                    dominant_period.append(0)
                    continue
                
                # Get window data
                window_data = series.iloc[i-window+1:i+1].values
                
                try:
                    # Remove linear trend
                    x = np.arange(len(window_data))
                    slope, intercept = np.polyfit(x, window_data, 1)
                    detrended = window_data - (slope * x + intercept)
                    
                    # Apply FFT
                    fft, fftfreq = _get_scipy_fft()
                    fft_values = fft(detrended)
                    freqs = fftfreq(len(detrended))
                    magnitude = np.abs(fft_values)
                    
                    # Find dominant cycle (excluding DC component)
                    positive_freqs = freqs[1:len(freqs)//2]
                    positive_magnitude = magnitude[1:len(magnitude)//2]
                    
                    if len(positive_magnitude) > 0:
                        max_idx = np.argmax(positive_magnitude)
                        dominant_freq = positive_freqs[max_idx]
                        dominant_per = 1.0 / (abs(dominant_freq) + 1e-8)
                        
                        # Cycle strength (normalized)
                        cycle_str = positive_magnitude[max_idx] / (
                            np.sum(positive_magnitude) + 1e-8)
                    else:
                        dominant_per = 0
                        cycle_str = 0
                    
                    cycle_strength.append(cycle_str)
                    dominant_period.append(dominant_per)
                    
                except Exception:
                    cycle_strength.append(0)
                    dominant_period.append(0)
            
            return (pd.Series(cycle_strength, index=series.index),
                   pd.Series(dominant_period, index=series.index))
        
        # Apply FFT cycle analysis
        (dataframe['cycle_strength'],
         dataframe['dominant_cycle_period']) = fft_cycle_analysis(dataframe['close'])
        
        # === ENHANCED TREND STRENGTH CALCULATION ===
        # Normalize advanced slopes by price
        dataframe['trend_strength_5_advanced'] = (
            dataframe['slope_5_advanced'] / dataframe['close'] * 100)
        dataframe['trend_strength_10_advanced'] = (
            dataframe['slope_10_advanced'] / dataframe['close'] * 100)
        dataframe['trend_strength_20_advanced'] = (
            dataframe['slope_20_advanced'] / dataframe['close'] * 100)
        
        # Wavelet-weighted combined trend strength
        dataframe['trend_strength_wavelet'] = (
            dataframe['trend_strength_5_advanced'] * 0.4 +
            dataframe['trend_strength_10_advanced'] * 0.35 +
            dataframe['trend_strength_20_advanced'] * 0.25
        )
        
        # Incorporate wavelet trend analysis
        dataframe['trend_strength_combined'] = (
            dataframe['trend_strength_wavelet'] * 0.7 +
            dataframe['wavelet_trend_strength'] * 0.3
        )
        
        # === CYCLE-ADJUSTED TREND STRENGTH ===
        # Adjust trend strength based on cycle analysis
        dataframe['trend_strength_cycle_adjusted'] = dataframe['trend_strength_combined'].copy()
        
        # Boost trend strength when aligned with dominant cycle
        strong_cycle_mask = dataframe['cycle_strength'] > 0.3
        dataframe.loc[strong_cycle_mask, 'trend_strength_cycle_adjusted'] *= (
            1 + dataframe.loc[strong_cycle_mask, 'cycle_strength'])
        
        # === FINAL TREND CLASSIFICATION WITH ADVANCED FEATURES ===
        strong_threshold = 0.02
        
        # Enhanced trend classification
        dataframe['strong_uptrend_advanced'] = (
            (dataframe['trend_strength_cycle_adjusted'] > strong_threshold) &
            (dataframe['wavelet_trend_strength'] > 0) &
            (dataframe['cycle_strength'] > 0.1)
        )
        
        dataframe['strong_downtrend_advanced'] = (
            (dataframe['trend_strength_cycle_adjusted'] < -strong_threshold) &
            (dataframe['wavelet_trend_strength'] < 0) &
            (dataframe['cycle_strength'] > 0.1)
        )
        
        dataframe['ranging_advanced'] = (
            (dataframe['trend_strength_cycle_adjusted'].abs() < strong_threshold * 0.5) |
            (dataframe['cycle_strength'] < 0.05)  # Very weak cycles indicate ranging
        )
        
        # === TREND CONFIDENCE SCORE ===
        # Calculate confidence based on agreement between methods
        methods_agreement = (
            (np.sign(dataframe['trend_strength_5_advanced']) ==
             np.sign(dataframe['trend_strength_10_advanced'])).astype(int) +
            (np.sign(dataframe['trend_strength_10_advanced']) ==
             np.sign(dataframe['trend_strength_20_advanced'])).astype(int) +
            (np.sign(dataframe['trend_strength_wavelet']) ==
             np.sign(dataframe['wavelet_trend_strength'])).astype(int)
        )
        
        dataframe['trend_confidence'] = methods_agreement / 3.0
        
        # High confidence trends
        dataframe['high_confidence_trend'] = (
            (dataframe['trend_confidence'] >= 0.67) &
            (dataframe['cycle_strength'] > 0.2) &
            (dataframe['trend_strength_cycle_adjusted'].abs() > strong_threshold * 0.8)
        )
        
        return dataframe
        
    except Exception as e:
        logger.warning(f"Advanced trend analysis failed: {e}. Using fallback method.")
        # Return dataframe with fallback values
        fallback_columns = [
            'slope_5_advanced', 'slope_10_advanced', 'slope_20_advanced',
            'wavelet_trend_strength', 'cycle_strength', 'dominant_cycle_period',
            'trend_strength_5_advanced', 'trend_strength_10_advanced',
            'trend_strength_20_advanced', 'trend_strength_wavelet',
            'trend_strength_combined', 'trend_strength_cycle_adjusted',
            'strong_uptrend_advanced', 'strong_downtrend_advanced',
            'ranging_advanced', 'trend_confidence', 'high_confidence_trend'
        ]
        
        boolean_fallback = {
            'strong_uptrend_advanced', 'strong_downtrend_advanced',
            'ranging_advanced', 'high_confidence_trend',
        }
        for col in fallback_columns:
            dataframe[col] = False if col in boolean_fallback else 0.0
        
        return dataframe


# =============================================================================
# ANF_v2: HIDDEN MARKOV MODEL REGIME DETECTION
# =============================================================================
#
# WHY THIS EXISTS
# ---------------
# The original ANF detected market regime ('trending'/'ranging'/'volatile')
# with three hard-coded if/else rules over the last 20 candles' volatility and
# price-change (see AdvancedPredictiveEngine._legacy_detect_market_condition,
# preserved below). That detector is the weakest link in an otherwise
# sophisticated pipeline: important decisions hang off its label —
#   - which ML model the ensemble weights more heavily,
#   - dynamic leverage,
#   - (optionally) which entry tier is allowed.
#
# A Gaussian Hidden Markov Model improves regime detection in four concrete,
# verifiable ways over the threshold rules:
#   1. It learns regimes from the joint distribution of multiple stationary
#      observables (log-returns, realized vol, volume z-score, return
#      autocorrelation) rather than a single volatility threshold. Two regimes
#      with identical nominal volatility but opposite return distributions
#      ('quiet bull' vs 'quiet bear') are separable; the threshold rule cannot.
#   2. It returns a POSTERIOR PROBABILITY per state, not a binary label. ANF_v2
#      uses this to make graded decisions (e.g. if P(regime) is weak, soften
#      leverage instead of assuming certainty).
#   3. It models PERSISTENCE via a transition matrix. The legacy detector
#      recomputes from scratch every candle and flips on noise; the HMM has
#      learned that ranging->trending transitions are rare and discounts a
#      single noisy candle.
#   4. It can be trained once on a long history and decoded cheaply per-candle.
#
# WHAT IT IS *NOT*
# ----------------
# - It is NOT a signal generator. As the regime-detection literature is clear
#   (e.g. Cube Exchange 2026, QuantStart), the correct use of an HMM is as a
#   *controller* on top of an existing strategy, not "buy in state 1, sell in
#   state 2". ANF_v2 uses the HMM state strictly to MODULATE the existing
#   ML ensemble, leverage, and risk gates — never as a standalone entry trigger.
# - It does NOT replace the ML ensemble. The ensemble still predicts entries.
# - It does NOT predict the future. It classifies the *current* latent state
#   with a probability; transitions are still discovered after the fact (though
#   sooner than a volatility threshold would).
#
# DESIGN DECISIONS (rationale documented inline):
# - 3 states by default: maps cleanly onto ANF's existing
#   'ranging'/'trending'/'volatile' vocabulary so downstream code (ensemble
#   weighting, leverage) needs no change.
# - covariance_type='full': lets states capture correlations between
#   observables (e.g. high vol + negative autocorr = mean-reverting toxic
#   state). 'diag' would miss this.
# - min_covar=1e-3: regularisation floor. Without it a Gaussian state can
#   collapse onto a tiny-variance region and fit noise (a known hmmlearn
#   failure mode).
# - States sorted by the mean level of realized_vol (within-state variance is
#   used only as a tie-break) so state identity is DETERMINISTIC across
#   retrains. Without this, hmmlearn's state 0 might be 'volatile' today and
#   'ranging' tomorrow, and the label mapping would drift.
# - Global model (trained on the BTC reference series) rather than per-pair:
#   regime is a market-wide property; per-pair HMMs would be 15x the overhead
#   for marginal benefit and far more overfit risk. One robust global model.
# - predict_proba (posterior) over Viterbi for per-candle inference: we want
#   the marginal probability of the *current* state, not the most-likely full
#   path (Viterbi can revise past states, which we don't want intra-candle).
#
# FALLBACK CONTRACT
# -----------------
# If hmmlearn is not installed, if the model is unfitted, or if anything throws
# during inference, the detector returns 'unknown' and the engine transparently
# falls back to the legacy threshold rules. The HMM can NEVER break trading.
# =============================================================================

# hmmlearn is an optional dependency. Import lazily and degrade gracefully.
try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    _GaussianHMM = None
    HMM_AVAILABLE = False
    logger.warning(
        "[HMM] hmmlearn not installed — ANF_v2 will fall back to the legacy "
        "threshold-based regime detector. To enable HMM regime detection: "
        "pip install hmmlearn"
    )


def _library_fingerprint() -> dict:
    """Capture versions of the libraries that affect HMM pickle compatibility.

    EN: A GaussianHMM pickled by one hmmlearn/numpy build may fail to unpickle,
        or unpickle subtly wrong, under a different build. We stamp the training
        environment into the .pkl so the live bot can compare and warn if the
        model was trained under different versions than it is running.
    ES: Un GaussianHMM serializado con un build de hmmlearn/numpy puede fallar al
        deserializarse, o hacerlo de forma sutilmente incorrecta, bajo otro build.
        Sellamos el entorno de entrenamiento en el .pkl para que el bot en vivo
        compare y avise si el modelo se entrenó con versiones distintas a las que
        está ejecutando.
    """
    fp = {}
    for name in ('hmmlearn', 'numpy', 'sklearn', 'scipy'):
        try:
            mod = __import__(name)
            fp[name] = getattr(mod, '__version__', 'unknown')
        except Exception:
            fp[name] = 'absent'
    return fp


class HMMRegimeDetector:
    """Gaussian Hidden Markov Model for market-regime classification.

    Public contract:
      - fit(df)                     -> train on the tail of df (idempotent)
      - detect(df)                  -> 'ranging'|'trending'|'volatile'|'unknown'
      - detect(df, with_probs=True) -> (label, {label: posterior, ...})
      - needs_retrain(now)          -> bool, time-based
      - save(path) / load(path)     -> pickle persistence
      - is_fitted                   -> bool

    All methods are exception-safe: any internal failure degrades to 'unknown'
    so the caller can fall back to the legacy detector. The HMM never raises
    into the trading loop.
    """

    # Default hyper-parameters (class-level so they're documented and tunable
    # in one place; NOT hyperopt-able — regime detection is infrastructure,
    # not a signal parameter to optimise).
    N_STATES = 3
    COV_TYPE = 'full'
    N_ITER = 200
    TOL = 1e-4
    MIN_COVAR = 1e-3
    RANDOM_STATE = 42

    # Training/inference windows
    # EN: FIT_LOOKBACK is the rolling window used to fit the live/dry-run HMM.
    #     4000 candles ~= 5.5 months at 1h: wide enough that the three regimes
    #     (ranging/trending/volatile) are usually all represented, yet light
    #     enough that the weekly fit stays a few seconds on a modest VPS. The
    #     HMM fit is cheap (3 states, 4 observables); the practical ceiling is
    #     how many candles the bot holds in memory (startup_candle_count grows
    #     this over runtime). Lower to 2000 on a very small VPS, or raise it if
    #     you also raise startup_candle_count. For a model trained on years of
    #     history, train it offline and load it as a frozen .pkl (see save()).
    # ES: FIT_LOOKBACK es la ventana móvil con la que se entrena el HMM en
    #     live/dry-run. 4000 velas ~= 5,5 meses en 1h: suficiente para que los
    #     tres regímenes (lateral/tendencia/volátil) suelan estar representados,
    #     y ligero para que el entrenamiento semanal dure segundos en un VPS
    #     modesto. El fit del HMM es barato (3 estados, 4 observables); el techo
    #     real es cuántas velas tiene el bot en memoria (startup_candle_count va
    #     creciéndolo en ejecución). Baja a 2000 en un VPS muy justo, o súbelo si
    #     también subes startup_candle_count. Para un modelo entrenado con años
    #     de histórico, entrénalo offline y cárgalo como un .pkl congelado (ver save()).
    FIT_LOOKBACK = 4000      # candles used to fit the model (~5.5 months at 1h)
    INFER_LOOKBACK = 100     # candles used to build observables for inference
    RETRAIN_EVERY_HOURS = 168  # weekly retrain (regimes drift slowly)

    # Minimum observables after dropna() before we trust a fit/inference
    MIN_OBS = 200

    # regime label. With 3 states, a uniform posterior is 0.33; anything below
    # this threshold means the HMM is genuinely uncertain, so we report
    # 'unknown' and let the caller use a safe default (no leverage modulation,
    # legacy detector for model weighting). Prevents acting on regime noise.
    CONFIDENCE_THRESHOLD = 0.50

    # predict_proba needs a few observations to produce a stable posterior.
    # We refuse to emit a label until at least this many observables exist in
    # the inference window.
    WARMUP_CANDLES = 20

    def __init__(self, n_states: int = None, fit_lookback: int = None,
                 retrain_every_hours: int = None):
        self.n_states = n_states or self.N_STATES
        self.fit_lookback = fit_lookback or self.FIT_LOOKBACK
        self.retrain_every_hours = retrain_every_hours or self.RETRAIN_EVERY_HOURS
        self.model = None
        self.is_fitted = False
        self.last_train_time: Optional[datetime] = None
        # EN: When True, this model was trained offline on a long history and
        #     must never be overwritten by the live bot's short rolling-window
        #     fit. needs_retrain() short-circuits to False while frozen. Set by
        #     load() from the persisted flag; live fits always leave it False.
        # ES: Si es True, el modelo se entrenó offline con histórico largo y el
        #     bot en vivo nunca debe sobrescribirlo con su fit de ventana corta.
        #     needs_retrain() devuelve False mientras está congelado. Lo fija
        #     load() desde el flag persistido; los fits en vivo lo dejan en False.
        self.frozen: bool = False
        self._last_fit_degenerate: bool = False
        # EN: Library-version mismatches found when load() restored a .pkl that
        #     was trained under a different environment. Populated by load();
        #     the strategy reads it after loading to notify (logs + Telegram).
        #     Empty list = no mismatch (or model trained in this environment).
        # ES: Desajustes de versión de librerías detectados cuando load() carga
        #     un .pkl entrenado en otro entorno. Lo rellena load(); la estrategia
        #     lo lee tras cargar para avisar (logs + Telegram). Lista vacía = sin
        #     desajuste (o modelo entrenado en este mismo entorno).
        self.version_mismatches: list[str] = []
        # time so detect() can refuse a mismatched input instead of crashing
        # inside predict_proba.
        self.n_features_: int = 0
        # Maps the HMM's internal (arbitrary) state index -> human label.
        # Populated deterministically in fit() by sorting states on volatility.
        self.state_labels: dict[int, str] = {}
        # Diagnostics from the last fit (logged, useful for the audit).
        self.fit_diagnostics: dict = {}

    # ------------------------------------------------------------------
    # Observable construction
    # ------------------------------------------------------------------
    # Canonical observable column order. The HMM's emission covariance is
    # indexed by this order, so _assign_state_labels can locate the
    # realized_vol column dynamically instead of hard-coding index 1. If you
    # add/remove/reorder observables, this constant is the single source of
    # truth — everything downstream derives the vol index from it.
    OBSERVABLE_COLUMNS = ('log_return', 'realized_vol', 'volume_z', 'return_autocorr')

    @classmethod
    def _vol_column_index(cls) -> int:
        """Index of the realized_vol observable in the emission covariance.
        Derived from OBSERVABLE_COLUMNS so it can never silently desync."""
        return cls.OBSERVABLE_COLUMNS.index('realized_vol')

    @classmethod
    def _build_observables(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Build the stationary observable matrix the HMM consumes.

        CRITICAL: every observable must be stationary. We never feed raw price
        (non-stationary) — only returns, rolling vol, normalised volume, and
        autocorrelation. Non-stationary inputs make the Gaussian emissions
        meaningless and the fit unstable.

        Observables (4 dimensions, order = OBSERVABLE_COLUMNS):
          - log_return        : log(close_t / close_{t-1})
          - realized_vol      : 10-candle rolling std of log returns
          - volume_z          : 50-candle z-score of volume (relative activity)
          - return_autocorr   : lag-1 autocorrelation over 20 candles
                                 (momentum vs mean-reversion signature)

        return_autocorr is computed with a
        VECTORISED rolling correlation against the lag-1 shifted series instead
        of rolling().apply(lambda x: x.autocorr()). Both are valid lag-1
        autocorrelation measures; the vectorised form is ~700x faster on a
        2000-candle fit window, with a one-element-per-window edge difference
        that is immaterial for regime classification.
        """
        feat = pd.DataFrame(index=df.index)
        close = df['close'].astype(float)
        log_ret = np.log(close / close.shift(1))
        feat['log_return'] = log_ret
        feat['realized_vol'] = log_ret.rolling(10).std()
        if 'volume' in df.columns:
            vol = df['volume'].astype(float)
            vol_mean = vol.rolling(50).mean()
            vol_std = vol.rolling(50).std()
            feat['volume_z'] = ((vol - vol_mean) / (vol_std + 1e-10)).clip(-5, 5)
        else:
            feat['volume_z'] = 0.0
        # Vectorised lag-1 rolling autocorrelation.
        feat['return_autocorr'] = log_ret.rolling(20).corr(log_ret.shift(1)).fillna(0.0)
        # Enforce canonical column order so the covariance index is stable.
        feat = feat[list(cls.OBSERVABLE_COLUMNS)]
        return feat.replace([np.inf, -np.inf], np.nan).dropna()

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, now: Optional[datetime] = None) -> bool:
        """Fit the HMM on the tail of `df`. Returns True on success.

        Exception-safe: on any failure logs and returns False (caller keeps
        using the legacy detector).
        """
        if not HMM_AVAILABLE:
            return False
        # EN: Defense-in-depth. needs_retrain() already returns False while
        #     frozen, so a frozen model normally never reaches fit(). This guard
        #     is the last line: if any future caller invokes fit() directly, an
        #     offline-trained model is still never overwritten by a live fit.
        # ES: Defensa en profundidad. needs_retrain() ya devuelve False si está
        #     congelado, así que un modelo frozen normalmente no llega a fit().
        #     Este guard es la última línea: si algún caller futuro llama a fit()
        #     directamente, un modelo entrenado offline nunca se sobrescribe.
        if getattr(self, 'frozen', False):
            logger.info(
                "[HMM] model is frozen (offline-trained); skipping live fit.")
            return False
        try:
            obs = self._build_observables(df.tail(self.fit_lookback))
            if len(obs) < self.MIN_OBS:
                logger.info(
                    f"[HMM] not enough observables to fit "
                    f"({len(obs)} < {self.MIN_OBS}); keeping legacy detector")
                return False
            X = obs.values
            self.n_features_ = X.shape[1]  # remember training dimensionality
            model = _GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.COV_TYPE,
                n_iter=self.N_ITER,
                tol=self.TOL,
                min_covar=self.MIN_COVAR,
                random_state=self.RANDOM_STATE,
            )
            model.fit(X)
            if not getattr(model, 'monitor_', None) or not model.monitor_.converged:
                logger.warning(
                    "[HMM] EM did not converge within n_iter; the fit may be "
                    "unreliable. Proceeding but flagging in diagnostics.")
            # Evaluate the labelling on the NEW model before committing it. If
            # its volatility states are near-degenerate, the regime labels would
            # be unstable; installing it is worse than keeping the previous
            # model (or the legacy detector), so we reject the fit. This does
            # not touch trading logic — it only guards which regime model is in
            # effect. The previous self.model (if any) stays untouched.
            candidate_labels = self._compute_state_labels(model)
            if self._last_fit_degenerate:
                logger.error(
                    "[HMM] fit produced near-degenerate volatility states; "
                    "rejecting it and keeping the previous model/detector.")
                return False
            self.model = model
            self.state_labels = candidate_labels
            self.is_fitted = True
            self.last_train_time = now or datetime.now(timezone.utc)
            self.fit_diagnostics = {
                'n_obs': int(len(obs)),
                'n_features': int(self.n_features_),
                'converged': bool(getattr(model.monitor_, 'converged', False)),
                'state_labels': dict(self.state_labels),
                'transition_matrix': model.transmat_.round(3).tolist(),
            }
            logger.info(
                f"[HMM] fitted on {len(obs)} obs, {self.n_states} states, "
                f"labels={self.state_labels}, "
                f"converged={self.fit_diagnostics['converged']}")
            return True
        except Exception as e:
            # If a previous model was already fitted, keep it rather than
            # discarding a working model because a retrain attempt failed.
            # Only fall back to the legacy detector if we never had a model.
            if self.model is not None and self.is_fitted:
                logger.warning(
                    f"[HMM] retrain failed ({e}); keeping the previously "
                    f"fitted model.")
                return False
            logger.warning(f"[HMM] fit failed ({e}); keeping legacy detector")
            self.is_fitted = False
            return False

    def _compute_state_labels(self, model) -> dict:
        """Deterministically map state index -> label by volatility ranking.

        Returns the label dict and sets self._last_fit_degenerate as a side
        effect (so fit() can reject a near-degenerate model before committing
        it). Does not mutate self.state_labels; the caller decides whether to
        adopt the returned mapping.

        We rank states by the mean level of the realized_vol observable in each
        state and assign: lowest vol -> 'ranging', highest -> 'volatile',
        everything between -> 'trending'. The within-state variance is used only
        as a tie-break.

        The realized_vol column index is derived dynamically from
        OBSERVABLE_COLUMNS via _vol_column_index() instead of being hard-coded.
        If observables are reordered, the labelling follows automatically.

        If two adjacent (in vol-rank) states have very similar mean volatility,
        their order can flip between retrains, silently swapping regime labels
        and inverting downstream leverage/weighting decisions. We flag this so
        the fit can be rejected rather than silently adopted.
        """
        vol_idx = self._vol_column_index()
        # Rank states by the MEAN level of realized_vol in each state (a
        # high-volatility regime is one whose volatility is high on average),
        # using the within-state variance only as a tie-break. Ranking by
        # variance alone would confuse a calm-but-jittery regime with a
        # genuinely high-volatility one.
        vol_means = []
        vol_vars = []
        for i in range(model.n_components):
            try:
                vol_means.append(float(np.asarray(model.means_)[i, vol_idx]))
            except Exception:
                vol_means.append(0.0)
            cov = model.covars_[i]
            # covars_ shape depends on covariance_type. For 'full' it is
            # (n_states, n_features, n_features); the realized_vol variance is
            # the [vol_idx, vol_idx] diagonal entry. For diag/spherical it is a
            # 1-D variance vector.
            try:
                if cov.ndim == 2:
                    vol_vars.append(float(cov[vol_idx, vol_idx]))
                else:
                    arr = np.atleast_1d(cov)
                    vol_vars.append(float(arr[min(vol_idx, len(arr) - 1)]))
            except Exception:
                vol_vars.append(float(np.mean(cov)))
        # Primary key = mean vol level, secondary key = within-state variance.
        order = np.lexsort((np.asarray(vol_vars), np.asarray(vol_means)))

        # Detect near-degenerate adjacent states by mean-vol separation. If two
        # adjacent states sit at almost the same volatility level their order
        # can flip between retrains, silently swapping regime labels.
        self._last_fit_degenerate = False
        for i in range(len(order) - 1):
            lo = vol_means[order[i]]
            hi = vol_means[order[i + 1]]
            ratio = (abs(hi) + 1e-12) / (abs(lo) + 1e-12)
            if ratio < 1.3:
                self._last_fit_degenerate = True
                logger.warning(
                    f"[HMM] states {order[i]} and {order[i+1]} have similar "
                    f"mean volatility ({lo:.6f} vs {hi:.6f}, ratio={ratio:.2f}). "
                    f"Regime labels may be unstable across retrains; consider "
                    f"reducing n_states from {self.n_states}.")

        labels = {}
        n = len(order)
        if n == 1:
            labels[int(order[0])] = 'ranging'
        elif n == 2:
            labels[int(order[0])] = 'ranging'
            labels[int(order[1])] = 'volatile'
        else:
            # lowest -> ranging, highest -> volatile, everything between -> trending
            labels[int(order[0])] = 'ranging'
            labels[int(order[-1])] = 'volatile'
            for mid in order[1:-1]:
                labels[int(mid)] = 'trending'
        return labels

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def detect(self, df: pd.DataFrame, with_probs: bool = False):
        """Classify the current candle's regime.

        Returns a label string, or (label, prob_dict) when with_probs=True.
        Falls back to 'unknown' on any problem so the caller can use the
        legacy detector.

        Inference guards (in order):
          warm-up: need >= WARMUP_CANDLES observables for a stable
                     forward-filter posterior.
          dimensionality: refuse if the observable count differs
                     from training (e.g. volume column vanished).
          confidence: if the top posterior < CONFIDENCE_THRESHOLD,
                     report 'unknown' rather than acting on a coin-flip. When
                     with_probs=True we still return the probability dict so a
                     caller can inspect it, but the label is 'unknown'.
        """
        if not (HMM_AVAILABLE and self.is_fitted and self.model is not None):
            return ('unknown', None) if with_probs else 'unknown'
        try:
            obs = self._build_observables(df.tail(self.INFER_LOOKBACK))
            if obs.empty or len(obs) < self.WARMUP_CANDLES:
                return ('unknown', None) if with_probs else 'unknown'
            X = obs.values
            # Inference dimensionality must match what we trained on.
            if self.n_features_ and X.shape[1] != self.n_features_:
                logger.warning(
                    f"[HMM] observable dim mismatch (trained={self.n_features_}, "
                    f"got={X.shape[1]}); falling back to legacy.")
                return ('unknown', None) if with_probs else 'unknown'
            posteriors = self.model.predict_proba(X)[-1]  # last candle
            state_idx = int(np.argmax(posteriors))
            max_prob = float(posteriors[state_idx])
            # Aggregate posteriors by label (multiple internal states can map
            # to the same label, e.g. two 'trending' states).
            agg: dict[str, float] = {}
            for i, p in enumerate(posteriors):
                lbl = self.state_labels.get(int(i), 'unknown')
                agg[lbl] = agg.get(lbl, 0.0) + float(p)
            # Confidence gate: weak posteriors degrade to 'unknown'.
            if max_prob < self.CONFIDENCE_THRESHOLD:
                label = 'unknown'
            else:
                label = self.state_labels.get(state_idx, 'unknown')
            if with_probs:
                return label, agg
            return label
        except Exception as e:
            logger.debug(f"[HMM] detect failed ({e}); falling back")
            return ('unknown', None) if with_probs else 'unknown'

    # ------------------------------------------------------------------
    # Retrain scheduling
    # ------------------------------------------------------------------
    def needs_retrain(self, now: Optional[datetime] = None) -> bool:
        """Time-based retrain check. Regimes drift slowly, so weekly is plenty.

        EN: A frozen model (trained offline on a long history) is never retrained
            in-place; the live bot must not overwrite it with its short window.
        ES: Un modelo congelado (entrenado offline con histórico largo) nunca se
            reentrena en vivo; el bot no debe sobrescribirlo con su ventana corta.
        """
        if getattr(self, 'frozen', False):
            return False
        if not self.is_fitted or self.last_train_time is None:
            return True
        now = now or datetime.now(timezone.utc)
        elapsed_h = (now - self.last_train_time).total_seconds() / 3600.0
        return elapsed_h >= self.retrain_every_hours

    # ------------------------------------------------------------------
    # Persistence (fork-aware: caller passes the path under
    # user_data/ml_models/<ClassName>/)
    # ------------------------------------------------------------------
    def save(self, path: Path, frozen: bool = False) -> bool:
        """Persist the fitted HMM to disk via pickle.

        EN: Pass frozen=True when saving a model trained offline on a long
            history so the live bot will load it and never retrain over it.
            Live/dry-run fits call save() with the default frozen=False.
        ES: Pasa frozen=True al guardar un modelo entrenado offline con
            histórico largo para que el bot lo cargue y nunca lo reentrene.
            Los fits en live/dry-run llaman a save() con frozen=False por defecto.
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'schema_version': 3,  # adds atomic write + sha256 sidecar
                'model': self.model,
                'state_labels': self.state_labels,
                'last_train_time': self.last_train_time,
                'n_states': self.n_states,
                'n_features': self.n_features_,
                'observable_columns': list(self.OBSERVABLE_COLUMNS),
                'fit_diagnostics': self.fit_diagnostics,
                'frozen': bool(frozen),
                'lib_fingerprint': _library_fingerprint(),
            }
            blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            # EN: Atomic write — serialise to a .tmp in the same directory, fsync,
            #     then os.replace() (atomic on POSIX/Windows). A crash or a partial
            #     scp can never leave a half-written .pkl that the bot would load.
            # ES: Escritura atómica — serializa a un .tmp en el mismo directorio,
            #     fsync, y os.replace() (atómico en POSIX/Windows). Un corte o un
            #     scp a medias nunca deja un .pkl a medio escribir que el bot cargue.
            tmp = path.with_suffix(path.suffix + '.tmp')
            with open(tmp, 'wb') as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            # Integrity sidecar: sha256 of the bytes actually written. load()
            # checks it to catch corruption (truncated/partial copy) before
            # unpickling. Written atomically too.
            digest = hashlib.sha256(blob).hexdigest()
            sig = path.with_suffix(path.suffix + '.sha256')
            sig_tmp = sig.with_suffix(sig.suffix + '.tmp')
            sig_tmp.write_text(digest)
            os.replace(sig_tmp, sig)
            logger.info(
                f"[HMM] saved regime model to {path}"
                + (" (frozen)" if frozen else "")
                + f" sha256={digest[:16]}…")
            return True
        except Exception as e:
            logger.warning(f"[HMM] save failed ({e})")
            return False

    def load(self, path: Path) -> bool:
        if not HMM_AVAILABLE:
            return False
        try:
            path = Path(path)
            if not path.exists():
                return False
            blob = path.read_bytes()
            # EN: If a .sha256 sidecar exists, verify it before unpickling. A
            #     mismatch means the file was corrupted or truncated (e.g. a
            #     partial scp); we refuse to load rather than feed a damaged
            #     pickle to the unpickler. Missing sidecar => older model, allow.
            # ES: Si existe un sidecar .sha256, se verifica antes de deserializar.
            #     Un desajuste significa que el fichero se corrompió o truncó (p.
            #     ej. un scp a medias); rehusamos cargar en vez de pasar un pickle
            #     dañado. Sin sidecar => modelo antiguo, se permite.
            sig = path.with_suffix(path.suffix + '.sha256')
            if sig.exists():
                expected = sig.read_text().strip()
                actual = hashlib.sha256(blob).hexdigest()
                if expected != actual:
                    logger.error(
                        f"[HMM] integrity check FAILED for {path} "
                        f"(sha256 mismatch). The file may be corrupted or a "
                        f"partial copy. Refusing to load; keeping legacy detector.")
                    self.is_fitted = False
                    return False
            data = pickle.loads(blob)
            saved_model = data.get('model')
            if saved_model is None:
                return False
            # Validate n_states consistency. If the persisted model was
            # trained with a different n_states than the current config, the
            # state_labels / posterior dimensions won't line up — discard and
            # force a retrain rather than risk an IndexError or silent
            # mislabelling.
            saved_n_states = data.get('n_states', self.N_STATES)
            if saved_n_states != self.n_states:
                logger.warning(
                    f"[HMM] persisted model has n_states={saved_n_states} but "
                    f"current config wants n_states={self.n_states}. Discarding "
                    f"persisted model; will retrain.")
                self.is_fitted = False
                return False
            # Validate observable schema. If the observable set changed
            # between versions, the covariance dimensions won't match — discard.
            saved_cols = data.get('observable_columns')
            if saved_cols is not None and list(saved_cols) != list(self.OBSERVABLE_COLUMNS):
                logger.warning(
                    f"[HMM] persisted observable schema {saved_cols} differs "
                    f"from current {list(self.OBSERVABLE_COLUMNS)}. Discarding "
                    f"persisted model; will retrain.")
                self.is_fitted = False
                return False
            self.model = saved_model
            self.state_labels = data['state_labels']
            self.last_train_time = data.get('last_train_time')
            self.n_states = data.get('n_states', self.N_STATES)
            self.n_features_ = data.get('n_features', 0)
            self.fit_diagnostics = data.get('fit_diagnostics', {})
            self.frozen = bool(data.get('frozen', False))
            self.is_fitted = self.model is not None
            # Compare the training-time library versions against the running
            # environment. A mismatch does not necessarily break the model, but
            # pickled hmmlearn/numpy objects are not guaranteed compatible across
            # versions, so we surface it loudly. We warn rather than discard:
            # the load succeeded, and the schema/n_states/observable checks above
            # already guard the structural contract.
            saved_fp = data.get('lib_fingerprint')
            self.version_mismatches = []
            if saved_fp:
                current_fp = _library_fingerprint()
                mismatches = [
                    f"{lib}: trained={saved_fp.get(lib)} running={current_fp.get(lib)}"
                    for lib in saved_fp
                    if lib in current_fp and saved_fp.get(lib) != current_fp.get(lib)
                ]
                if mismatches:
                    self.version_mismatches = mismatches
                    logger.warning(
                        "[HMM] model was trained under different library "
                        "versions than this environment. Pickle compatibility is "
                        "not guaranteed; if regime detection misbehaves, retrain "
                        "on this machine or align versions. Mismatches: "
                        + "; ".join(mismatches))
            if self.is_fitted:
                logger.info(
                    f"[HMM] loaded regime model from {path} "
                    f"(schema_v={data.get('schema_version', 1)}"
                    + (", frozen" if self.frozen else "")
                    + f"), labels={self.state_labels}")
            return self.is_fitted
        except Exception as e:
            logger.warning(f"[HMM] load failed ({e})")
            return False


# === ADVANCED PREDICTIVE ANALYSIS SYSTEM ===

class AdvancedPredictiveEngine:
    """
    Advanced machine learning engine for high-precision trade entry prediction
    """
    
    def __init__(self, config, strategy_name: Optional[str] = None,
                 models_dir_override: Optional[Path] = None):
        # Model containers
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.feature_columns: dict[str, list] = {}  # explicit feature list per pair
        self.prediction_history = {}
        self.is_trained = {}

        # Without this, two forks of the same strategy would clobber each
        # other's pickles. Falls back to "shared" with a warning if not given.
        self.strategy_name: str = strategy_name or "shared"
        if self.strategy_name == "shared" and models_dir_override is None:
            logger.warning(
                "[ML] AdvancedPredictiveEngine instantiated without strategy_name. "
                "Models will be stored under user_data/ml_models/shared/ which "
                "can be shared (and clobbered) between strategy forks. Pass "
                "strategy_name=self.__class__.__name__ from your bot_start()."
            )

        # Cached training dataframe per pair for incremental extension
        self.training_cache: dict[str, pd.DataFrame] = {}

        # Retraining control
        self.last_train_time: dict[str, datetime] = {}
        self.last_train_index: dict[str, int] = {}
        # Periodic retraining interval for the ML ensemble.
        self.retrain_interval_hours: int = 48
        self.initial_train_candles: int = 2000  # initial window size
        self.min_new_candles_for_retrain: int = 50  # skip tiny updates

        # engine would happily train with 350 samples on 30 features (~12
        # samples/feature, far below the 10-20 rule of thumb). This now refuses
        # to train until a meaningful dataset is available.
        self.min_train_candles: int = 1500

        # Strategy startup tracking for 48h retrain rule
        self.strategy_start_time: datetime = datetime.now(timezone.utc)
        self.retrain_after_startup_hours: int = 48

        # Enable periodic retrain after startup period
        self.enable_startup_retrain: bool = True

        # runmode is backtest/hyperopt/edge so ML training is bypassed entirely
        # (would otherwise cause look-ahead bias on the full backtest dataset).
        self.backtest_mode: bool = False

        # New default:  user_data/ml_models/<strategy_name>/
        # script) redirect ALL model I/O to a temp directory, keeping the real
        # production models untouched and preventing leftover files if the
        # tool is interrupted.
        if models_dir_override is not None:
            self.models_dir = Path(models_dir_override)
            self.models_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[ML] Model store (override): {self.models_dir.resolve()}")
        else:
            new_models_root = Path("user_data/ml_models")
            self.models_dir = new_models_root / self.strategy_name
            legacy_dir = Path("user_data/strategies/ml_models")
            if legacy_dir.exists() and not self.models_dir.exists():
                try:
                    legacy_pkls = list(legacy_dir.glob("*.pkl"))
                except Exception:
                    legacy_pkls = []
                if legacy_pkls:
                    logger.warning(
                        "[ML] Legacy models directory detected at %s with %d files. "
                        "New location is %s — these legacy models will NOT be loaded "
                        "automatically. To migrate: `mv %s/* %s/` (or delete them to "
                        "force retraining).",
                        legacy_dir, len(legacy_pkls), self.models_dir,
                        legacy_dir, self.models_dir,
                    )
            self.models_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[ML] Model store: {self.models_dir.resolve()}")

        # Load existing models if available
        self._load_models_from_disk()

        # ------------------------------------------------------------------
        # ANF_v2: HMM regime detector.
        # One global detector (regime is a market-wide property). It is trained
        # by the strategy's bot_start() on the BTC reference series and consulted
        # by _detect_market_condition(). Persisted alongside the ML models under
        # the same fork-aware models_dir, so ANF and ANF_v2 keep separate HMMs.
        # ------------------------------------------------------------------
        self.hmm_detector = HMMRegimeDetector()
        self._hmm_model_path = self.models_dir / "hmm_regime.pkl"
        # Attempt to load a previously-fitted HMM (cheap; no-op if absent).
        try:
            self.hmm_detector.load(self._hmm_model_path)
        except Exception as e:
            logger.debug(f"[HMM] no preloaded regime model ({e})")

        # Diagnostic: report HMM status once at startup (INFO, not debug, so it
        # shows up via /logs in production). Answers "is the HMM actually active,
        # and is it the frozen long-history model or the rolling-window one?".
        if getattr(self, 'diagnostic_logging', False):
            try:
                det = self.hmm_detector
                if not HMM_AVAILABLE:
                    logger.info("[DIAG][HMM] hmmlearn not installed; using legacy "
                                "threshold detector (behaves like ANF).")
                elif det.is_fitted:
                    src = "frozen long-history model" if getattr(det, 'frozen', False) \
                          else "rolling-window model"
                    logger.info(
                        f"[DIAG][HMM] active: loaded {src} from disk "
                        f"(labels={det.state_labels}, fit_lookback={det.fit_lookback}).")
                else:
                    logger.info(
                        "[DIAG][HMM] no model on disk yet; will train a rolling-window "
                        f"model in live/dry-run once >= {det.MIN_OBS} candles are available. "
                        "Drop a frozen hmm_regime.pkl in the models dir to skip this.")
            except Exception as e:
                logger.debug(f"[DIAG][HMM] status log skipped ({e})")

    # ----------------- ASSET EXISTENCE HELPERS -----------------
    def _required_asset_paths(self, pair: str) -> list[Path]:
        """Return list of required core asset file paths for a pair."""
        return [
            self._get_model_filepath(pair, "model_random_forest"),
            self._get_model_filepath(pair, "model_gradient_boosting"),
            self._get_model_filepath(pair, "scaler"),
            self._get_model_filepath(pair, "metadata"),
        ]

    def _assets_exist(self, pair: str) -> bool:
        """Check if all required asset files exist for pair."""
        return all(p.exists() for p in self._required_asset_paths(pair))

    def mark_trained_if_assets(self, pair: str):
        """Mark pair as trained if asset files exist (called at startup)."""
        if self._assets_exist(pair):
            self.is_trained[pair] = True

    def _get_model_filepath(self, pair: str, model_type: str) -> Path:
        """Get the filepath for saving/loading models"""
        safe_pair = pair.replace('/', '_').replace(':', '_')
        return self.models_dir / f"{safe_pair}_{model_type}.pkl"

    def _save_models_to_disk(self, pair: str):
        """Save trained models to disk for persistence."""
        try:
            if pair not in self.models:
                return

            # Save models
            for model_name, model in self.models[pair].items():
                filepath = self._get_model_filepath(pair, f"model_{model_name}")
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)

            # Save scaler
            if pair in self.scalers:
                scaler_filepath = self._get_model_filepath(pair, "scaler")
                with open(scaler_filepath, 'wb') as f:
                    pickle.dump(self.scalers[pair], f)

            # Save feature_columns + feature_importance + pair name in metadata.
            # uses the exact same feature set as training and loading is unambiguous.
            metadata_filepath = self._get_model_filepath(pair, "metadata")
            metadata = {
                'pair': pair,
                'feature_columns': self.feature_columns.get(pair, []),
                'feature_importance': self.feature_importance.get(pair, {}),
                'is_trained': self.is_trained.get(pair, False),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'schema_version': 2,
            }
            with open(metadata_filepath, 'wb') as f:
                pickle.dump(metadata, f)

            logger.info(f"Models saved to disk for {pair}")

        except Exception as e:
            logger.warning(f"Failed to save models for {pair}: {e}")

    def _load_models_from_disk(self):
        """Load existing models from disk.

        Pair name is read from each pair's metadata file rather than
        reconstructed from the filename. Reconstructing from the filename
        breaks on pairs like BTC/USDT (no `:`) or BTC/EUR (2 parts only).
        Backwards compatible:
        if metadata is missing or has no 'pair' field, we fall back to the
        old heuristic and log a warning.
        """
        try:
            if not self.models_dir.exists():
                return

            metadata_files = list(self.models_dir.glob("*_metadata.pkl"))
            pairs_found: set[str] = set()

            for meta_file in metadata_files:
                try:
                    with open(meta_file, 'rb') as f:
                        meta = pickle.load(f)
                    pair = meta.get('pair') if isinstance(meta, dict) else None
                    if not pair:
                        # Legacy file with no embedded pair: derive from filename.
                        pair = self._legacy_pair_from_filename(meta_file.stem)
                        logger.warning(
                            f"[ML] Legacy metadata without 'pair' field at {meta_file.name}; "
                            f"derived pair='{pair}' heuristically. Will be rewritten on next train.")
                    if pair:
                        pairs_found.add(pair)
                except Exception as e:
                    logger.warning(f"[ML] Failed to read metadata {meta_file.name}: {e}")

            # Backwards fallback: if no metadata files exist at all, scan model files.
            if not pairs_found:
                for model_file in self.models_dir.glob("*_model_*.pkl"):
                    pair = self._legacy_pair_from_filename(model_file.stem)
                    if pair:
                        pairs_found.add(pair)

            for pair in pairs_found:
                try:
                    self._load_pair_models(pair)
                except Exception as e:
                    logger.warning(f"Failed to load models for {pair}: {e}")

            if pairs_found:
                logger.info(f"Loaded ML models from disk for {len(pairs_found)} pairs: {sorted(pairs_found)}")

        except Exception as e:
            logger.warning(f"Failed to load models from disk: {e}")

    @staticmethod
    def _legacy_pair_from_filename(stem: str) -> Optional[str]:
        """Heuristic to recover pair name from a filename stem like
        'BTC_USDT_USDT_model_random_forest' or 'BTC_USDT_USDT_metadata'.
        Returns None if it can't make sense of it."""
        for suffix in ('_metadata', '_scaler'):
            if stem.endswith(suffix):
                base = stem[: -len(suffix)]
                break
        else:
            if '_model_' in stem:
                base = stem.split('_model_')[0]
            else:
                return None
        parts = base.split('_')
        if len(parts) >= 3:
            # Futures-style XXX/YYY:YYY
            return f"{parts[0]}/{parts[1]}:{parts[2]}"
        if len(parts) == 2:
            return f"{parts[0]}/{parts[1]}"
        return None

    def _load_pair_models(self, pair: str):
        """Load models for a specific pair"""
        # Load models
        models = {}
        for model_name in ['random_forest', 'gradient_boosting']:
            model_filepath = self._get_model_filepath(pair, f"model_{model_name}")
            if model_filepath.exists():
                with open(model_filepath, 'rb') as f:
                    models[model_name] = pickle.load(f)

        if models:
            self.models[pair] = models

        # Load scaler
        scaler_filepath = self._get_model_filepath(pair, "scaler")
        if scaler_filepath.exists():
            with open(scaler_filepath, 'rb') as f:
                self.scalers[pair] = pickle.load(f)

        metadata_filepath = self._get_model_filepath(pair, "metadata")
        if metadata_filepath.exists():
            with open(metadata_filepath, 'rb') as f:
                metadata = pickle.load(f)
                self.feature_importance[pair] = metadata.get('feature_importance', {})
                self.is_trained[pair] = metadata.get('is_trained', False)
                fc = metadata.get('feature_columns') or []
                if fc:
                    self.feature_columns[pair] = list(fc)

    def _cleanup_old_models(self, max_age_days: int = 7):
        """Remove models older than specified days"""
        try:
            cutoff_time = datetime.now(timezone.utc) - pd.Timedelta(days=max_age_days)

            for model_file in self.models_dir.glob("*.pkl"):
                if model_file.stat().st_mtime < cutoff_time.timestamp():
                    model_file.unlink()
                    logger.info(f"Removed old model file: {model_file.name}")

        except Exception as e:
            logger.warning(f"Failed to cleanup old models: {e}")

    def extract_advanced_features(self, dataframe: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
        """Extract sophisticated features for ML prediction with variance validation"""
        df = dataframe.copy()

        # === 1. ENHANCED PRICE ACTION FEATURES ===
        # Multi-period price patterns with variance
        for period in [1, 2, 3, 5]:
            df[f'price_velocity_{period}'] = df['close'].pct_change(period)
            df[f'price_acceleration_{period}'] = df[f'price_velocity_{period}'].diff(1)
            
            # Add rolling statistics for variance
            df[f'price_velocity_std_{period}'] = df[f'price_velocity_{period}'].rolling(20).std()
            df[f'price_velocity_skew_{period}'] = df[f'price_velocity_{period}'].rolling(20).skew()
            
        # Volatility-adjusted momentum
        returns = df['close'].pct_change(1)
        vol_20 = returns.rolling(20).std()
        df['vol_adjusted_momentum'] = returns / (vol_20 + 1e-10)
        
        # Price position within recent range
        for window in [10, 20, 50]:
            high_window = df['high'].rolling(window).max()
            low_window = df['low'].rolling(window).min()
            range_size = high_window - low_window
            df[f'price_position_{window}'] = (df['close'] - low_window) / (range_size + 1e-10)

        # Support/Resistance with dynamic thresholds
        if 'minima_sort_threshold' in df.columns:
            support_distance = abs(df['low'] - df['minima_sort_threshold']) / (df['close'] + 1e-10)
            df['support_strength'] = (support_distance < 0.02).astype(int).rolling(20).mean()
            df['support_distance_norm'] = support_distance
        else:
            # poisons the model with noise and breaks reproducibility).
            df['support_strength'] = 0.5
            df['support_distance_norm'] = 0.0

        if 'maxima_sort_threshold' in df.columns:
            resistance_distance = abs(df['high'] - df['maxima_sort_threshold']) / (df['close'] + 1e-10)
            df['resistance_strength'] = (resistance_distance < 0.02).astype(int).rolling(20).mean()
            df['resistance_distance_norm'] = resistance_distance
        else:
            df['resistance_strength'] = 0.5
            df['resistance_distance_norm'] = 0.0

        # === 2. VOLUME DYNAMICS ===
        # Volume profile analysis
        df['volume_profile_score'] = self._calculate_volume_profile_score(df)
        df['volume_imbalance'] = self._calculate_volume_imbalance(df)
        df['smart_money_index'] = self._calculate_smart_money_index(df)

        # Volume-price correlation
        df['volume_price_correlation'] = df['volume'].rolling(20).corr(df['close'])
        df['volume_breakout_strength'] = self._calculate_volume_breakout_strength(df)

        # === 3. VOLATILITY CLUSTERING ===
        df['volatility_regime'] = self._calculate_volatility_regime(df)
        df['volatility_persistence'] = self._calculate_volatility_persistence(df)
        df['volatility_mean_reversion'] = self._calculate_volatility_mean_reversion(df)

        # === 4. MOMENTUM DECOMPOSITION ===
        for period in [3, 5, 8, 13, 21]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'momentum_strength_{period}'] = abs(df[f'momentum_{period}'])
            df[f'momentum_consistency_{period}'] = (
                np.sign(df[f'momentum_{period}']).rolling(5).mean()
            )

        # Momentum regime detection
        df['momentum_regime'] = self._classify_momentum_regime(df)
        df['momentum_divergence_strength'] = self._calculate_momentum_divergence(df)

        # === 5. MICROSTRUCTURE FEATURES ===
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        df['market_impact'] = df['volume'] * df['spread_proxy']
        df['order_flow_imbalance'] = self._calculate_order_flow_imbalance(df)
        df['liquidity_index'] = self._calculate_liquidity_index(df)

        # === 6. STATISTICAL FEATURES ===
        skew, kurtosis = _get_scipy_stats()
        for window in [10, 20, 50]:
            returns = df['close'].pct_change(1)
            df[f'skewness_{window}'] = returns.rolling(window).apply(
                lambda x: skew(x.dropna()) if len(x.dropna()) > 3 else 0
            )
            df[f'kurtosis_{window}'] = returns.rolling(window).apply(
                lambda x: kurtosis(x.dropna()) if len(x.dropna()) > 3 else 0
            )
            df[f'entropy_{window}'] = self._calculate_entropy(df['close'], window)

        # === 7. REGIME DETECTION FEATURES ===
        df['market_regime'] = self._detect_market_regime(df)
        df['regime_stability'] = self._calculate_regime_stability(df)
        df['regime_transition_probability'] = self._calculate_regime_transition_prob(df)

        return df

    def _calculate_volume_profile_score(self, df: pd.DataFrame, window: int = 50) -> pd.Series:
        """Calculate volume profile score"""
        def volume_profile(data):
            if len(data) < 10:
                return 0.5

            prices = data['close'].values
            volumes = data['volume'].values

            # Create price bins
            price_min, price_max = prices.min(), prices.max()
            if price_max == price_min:
                return 0.5

            bins = np.linspace(price_min, price_max, 10)

            # Calculate volume at each price level
            volume_at_price = []
            for i in range(len(bins) - 1):
                mask = (prices >= bins[i]) & (prices < bins[i + 1])
                vol_sum = volumes[mask].sum()
                volume_at_price.append(vol_sum)

            # Point of Control (POC) - price level with highest volume
            if sum(volume_at_price) == 0:
                return 0.5

            poc_index = np.argmax(volume_at_price)
            current_price = prices[-1]
            poc_price = (bins[poc_index] + bins[poc_index + 1]) / 2

            # Score based on distance from POC
            distance_ratio = abs(current_price - poc_price) / (price_max - price_min + 1e-10)
            score = 1 - distance_ratio  # Closer to POC = higher score

            return max(0, min(1, score))

        # Apply rolling calculation
        result = []
        for i in range(len(df)):
            if i < window:
                result.append(0.5)
            else:
                window_data = df.iloc[i-window+1:i+1][['close', 'volume']]
                score = volume_profile(window_data)
                result.append(score)

        return pd.Series(result, index=df.index)

    def _calculate_volume_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume imbalance between buying and selling"""
        up_volume = df['volume'].where(df['close'] > df['open'], 0)
        down_volume = df['volume'].where(df['close'] < df['open'], 0)

        total_volume = up_volume + down_volume
        imbalance = (up_volume - down_volume) / (total_volume + 1e-10)

        return imbalance.rolling(10).mean()

    def _calculate_smart_money_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Smart Money Index (SMI)"""
        price_change = abs(df['close'].pct_change(1))
        volume_norm = df['volume'] / df['volume'].rolling(20).mean()

        smi = volume_norm / (price_change + 1e-10)
        return smi.rolling(10).mean()

    def _calculate_volume_breakout_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume breakout strength"""
        volume_ma = df['volume'].rolling(20).mean()
        volume_ratio = df['volume'] / (volume_ma + 1e-10)

        price_breakout = (
            (df['close'] > df['close'].rolling(20).max().shift(1)) |
            (df['close'] < df['close'].rolling(20).min().shift(1))
        ).astype(int)

        breakout_strength = volume_ratio * price_breakout
        return breakout_strength.rolling(5).mean()

    def _calculate_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect volatility regime"""
        returns = df['close'].pct_change(1)
        volatility = returns.rolling(20).std()
        vol_ma = volatility.rolling(50).mean()

        regime = pd.Series(1, index=df.index)  # Default normal
        regime[volatility < vol_ma * 0.7] = 0   # Low volatility
        regime[volatility > vol_ma * 1.5] = 2   # High volatility

        return regime

    def _calculate_volatility_persistence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility persistence"""
        returns = df['close'].pct_change(1)
        volatility = returns.rolling(5).std()

        persistence = volatility.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) > 10 else 0
        )

        return persistence

    def _calculate_volatility_mean_reversion(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility mean reversion tendency"""
        returns = df['close'].pct_change(1)
        volatility = returns.rolling(10).std()
        vol_ma = volatility.rolling(50).mean()

        vol_zscore = (volatility - vol_ma) / (volatility.rolling(50).std() + 1e-10)
        mean_reversion = -vol_zscore

        return mean_reversion

    def _classify_momentum_regime(self, df: pd.DataFrame) -> pd.Series:
        """Classify momentum regime"""
        mom_3 = df['close'].pct_change(3)
        mom_8 = df['close'].pct_change(8)
        mom_21 = df['close'].pct_change(21)
        
        regime = pd.Series(0, index=df.index)  # Neutral
        
        strong_up = (mom_3 > 0.02) & (mom_8 > 0.05) & (mom_21 > 0.1)
        regime[strong_up] = 2
        
        mod_up = (mom_3 > 0) & (mom_8 > 0) & (mom_21 > 0) & ~strong_up
        regime[mod_up] = 1
        
        mod_down = (mom_3 < 0) & (mom_8 < 0) & (mom_21 < 0) & (mom_21 > -0.1)
        regime[mod_down] = -1
        
        strong_down = (mom_3 < -0.02) & (mom_8 < -0.05) & (mom_21 < -0.1)
        regime[strong_down] = -2
        
        return regime
    
    def _calculate_momentum_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate momentum divergence strength"""
        price_momentum = df['close'].pct_change(10)
        
        if 'rsi' in df.columns:
            rsi_momentum = df['rsi'].diff(10)
        else:
            rsi_momentum = pd.Series(0, index=df.index)
            
        volume_momentum = df['volume'].pct_change(10)
        
        # Normalize momentums using rolling z-score
        price_norm = (price_momentum - price_momentum.rolling(50).mean()) / (
            price_momentum.rolling(50).std() + 1e-10)
        rsi_norm = (rsi_momentum - rsi_momentum.rolling(50).mean()) / (
            rsi_momentum.rolling(50).std() + 1e-10)
        volume_norm = (volume_momentum - volume_momentum.rolling(50).mean()) / (
            volume_momentum.rolling(50).std() + 1e-10)
        
        price_rsi_div = abs(price_norm - rsi_norm)
        price_volume_div = abs(price_norm - volume_norm)
        
        divergence_strength = (price_rsi_div + price_volume_div) / 2
        return divergence_strength.rolling(5).mean()
    
    def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate order flow imbalance"""
        price_impact = (df['close'] - df['open']) / df['open']
        volume_impact = df['volume'] / df['volume'].rolling(20).mean()
        
        flow_imbalance = price_impact * volume_impact
        return flow_imbalance.rolling(5).mean()
    
    def _calculate_liquidity_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market liquidity index"""
        spread = (df['high'] - df['low']) / df['close']
        volume_norm = df['volume'] / df['volume'].rolling(50).mean()
        
        liquidity = volume_norm / (spread + 1e-10)
        return liquidity.rolling(10).mean()

    def _calculate_entropy(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate information entropy"""
        def entropy(data):
            if len(data) < 5:
                return 0

            returns = np.diff(data) / (data[:-1] + 1e-10)

            bins = np.histogram_bin_edges(returns, bins=10)
            hist, _ = np.histogram(returns, bins=bins)

            probs = hist / (hist.sum() + 1e-10)
            probs = probs[probs > 0]

            ent = -np.sum(probs * np.log2(probs + 1e-10))
            return ent

        return series.rolling(window).apply(entropy, raw=False)

    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Continuous regime SCORE (pd.Series), used as an ML feature.

        ANF_v2 — DO NOT confuse this with _detect_market_condition():
          - _detect_market_regime()    -> pd.Series of a continuous score
            (trend/vol/momentum blend). Consumed as an input FEATURE by the ML
            ensemble (extract_advanced_features). It is computed per-candle over
            the whole dataframe and must stay a vectorised Series.
          - _detect_market_condition() -> a single str label
            ('ranging'/'trending'/'volatile') for the latest candle, now backed
            by the HMM. Consumed to WEIGHT ensemble models and modulate leverage.

        These two intentionally coexist: one is a continuous ML input, the other
        a discrete controller signal. We deliberately did NOT unify them into the
        HMM because (a) the ML feature needs a full per-candle Series the HMM
        posterior doesn't cheaply provide for history, and (b) changing a
        long-standing ML input feature would invalidate every persisted model and
        force a full retrain across all pairs. Unifying is a larger, separate
        decision to be made (if ever) with empirical evidence from the dry-run —
        not a quick refactor. The two similarly-named stores coexist by
        design; they are intentionally separate, not a bug.
        """
        if 'trend_strength' in df.columns:
            trend_regime = np.sign(df['trend_strength'])
        else:
            trend_regime = pd.Series(0, index=df.index)

        vol_regime = self._calculate_volatility_regime(df) - 1
        momentum_regime = self._classify_momentum_regime(df)

        market_regime = (
            trend_regime * 0.4 +
            vol_regime * 0.3 +
            momentum_regime * 0.3
        )

        return market_regime.rolling(5).mean()

    def _calculate_regime_stability(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime stability"""
        regime = self._detect_market_regime(df)
        regime_changes = abs(regime.diff(1))
        stability = 1 / (regime_changes.rolling(20).mean() + 1e-10)
        return stability

    def _calculate_regime_transition_prob(self, df: pd.DataFrame) -> pd.Series:
        """Calculate probability of regime transition"""
        regime = self._detect_market_regime(df)

        transitions = []
        for i in range(1, len(regime)):
            if not (pd.isna(regime.iloc[i]) or pd.isna(regime.iloc[i-1])):
                transition = abs(regime.iloc[i] - regime.iloc[i-1]) > 0.5
                transitions.append(transition)
            else:
                transitions.append(False)

        transition_prob = pd.Series([False] + transitions, index=regime.index)
        prob_smooth = transition_prob.astype(int).rolling(20).mean()

        return prob_smooth

    def create_target_variable(self, df: pd.DataFrame, forward_periods: int = 5,
                              profit_threshold: Optional[float] = None,
                              dynamic: bool = True,
                              quantile: float = 0.85,
                              k_atr: float = 1.2,
                              k_vol: float = 1.5,
                              min_abs: float = 0.003,
                              max_abs: float = 0.05) -> pd.Series:
        """Create target variable with optional dynamic profit threshold.

        Dynamic threshold logic (if dynamic=True and profit_threshold not provided):
          1. Compute ATR% (14) and rolling return volatility (20).
          2. base_series = k_atr * ATR% + k_vol * vola20
          3. base_scalar = median(base_series)
          4. q_thr = 85th percentile of forward_returns (future move distribution)
          5. blended = 0.5 * base_scalar + 0.5 * q_thr
          6. final_thr = clip(blended, min_abs, max_abs)
        This produces a stable scalar threshold per training batch (reproducible) rather than per-row noise.
        """
        # Forward returns used by several strategies
        forward_returns = df['close'].pct_change(forward_periods).shift(-forward_periods)

        # === DYNAMIC THRESHOLD CALCULATION ===
        if dynamic and (profit_threshold is None):
            try:
                # ATR% calculation
                high = df['high']
                low = df['low']
                close = df['close']
                prev_close = close.shift(1)
                tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
                atr = tr.rolling(14).mean()
                atr_pct = (atr / close).clip(lower=0)

                # Return volatility
                returns_1 = close.pct_change()
                vola20 = returns_1.rolling(20).std()

                base_series = k_atr * atr_pct + k_vol * vola20
                base_scalar = float(np.nanmedian(base_series.tail(300))) if len(base_series.dropna()) > 30 else float(np.nanmedian(base_series))

                # Distribution-based quantile of forward returns (future info okay for label construction stage)
                q_thr = float(forward_returns.quantile(quantile)) if forward_returns.notna().any() else min_abs
                if not np.isfinite(q_thr):
                    q_thr = min_abs
                blended = 0.5 * base_scalar + 0.5 * q_thr
                profit_threshold = float(np.clip(blended, min_abs, max_abs))
            except Exception as e:
                logger.warning(f"Dynamic threshold failed ({e}), falling back to default 0.015")
                profit_threshold = 0.015
        elif profit_threshold is None:
            profit_threshold = 0.015

        # === STRATEGY 1: SIMPLE FORWARD RETURNS ===
        simple_target = (forward_returns > profit_threshold).astype(int)

        # === STRATEGY 2: MAXIMUM PROFIT POTENTIAL ===
        forward_highs = df['high'].rolling(forward_periods).max().shift(-forward_periods)
        max_profit_potential = (forward_highs - df['close']) / df['close']
        profit_target = (max_profit_potential > profit_threshold).astype(int)

        # === STRATEGY 3: RISK-ADJUSTED RETURNS ===
        forward_lows = df['low'].rolling(forward_periods).min().shift(-forward_periods)
        max_loss_potential = (forward_lows - df['close']) / df['close']
        risk_adjusted_return = forward_returns / (abs(max_loss_potential) + 1e-10)
        risk_target = (
            (forward_returns > profit_threshold * 0.7) &
            (risk_adjusted_return > 0.5)
        ).astype(int)

        # === STRATEGY 4: VOLATILITY-ADJUSTED TARGET ===
        returns_std = df['close'].pct_change().rolling(20).std()
        volatility_adjusted_threshold = profit_threshold * (1 + returns_std)
        vol_target = (forward_returns > volatility_adjusted_threshold).astype(int)

        # === ENSEMBLE VOTE ===
        combined_target = simple_target + profit_target + risk_target + vol_target
        final_target = (combined_target >= 2).astype(int)

        # The last `forward_periods` rows have no future data (forward_returns
        # is NaN there). Leaving them as 0 would inject a repeated negative-class
        # bias at the tail of every retraining window. Mark them NaN so the
        # caller's valid_mask drops them from X/y.
        final_target = final_target.astype(float)
        final_target[forward_returns.isna()] = np.nan

        positive_ratio = final_target.mean()
        logger.info(
            f"Target created (forward={forward_periods}) dynamic_thr={profit_threshold:.4f} "
            f"positives={final_target.sum():.0f}/{len(final_target)} ratio={positive_ratio:.3f}")

        # Only log imbalance now; do not auto-alter labels (professional reproducibility)
        if positive_ratio < 0.05:
            logger.warning(f"Very low positive ratio ({positive_ratio:.3f}) at threshold {profit_threshold:.4f}")
        elif positive_ratio > 0.45:
            logger.warning(f"High positive ratio ({positive_ratio:.3f}) at threshold {profit_threshold:.4f}")

        return final_target

    def train_predictive_models(self, df: pd.DataFrame, pair: str) -> dict:
        """Train advanced ensemble of predictive models with hyperparameter optimization"""
        if not SKLEARN_AVAILABLE:
            return {'status': 'sklearn_not_available'}

        # the model would have too few samples per feature for stable learning.
        # Returns 'insufficient_data' so populate_indicators can skip and use
        # neutral 0.5 predictions until enough candles accumulate.
        if len(df) < self.min_train_candles:
            _log_info_throttled(
                f"[ML] {pair}: only {len(df)} candles available, "
                f"need >= {self.min_train_candles} before first training. "
                f"Will retry on next bar.",
                pair, 600,
            )
            return {'status': 'insufficient_data', 'have': len(df),
                    'need': self.min_train_candles}

        try:
            # Decide training slice: initial window or sliding window on retrain
            if pair not in self.is_trained or not self.is_trained[pair]:
                # First training: cut to last initial_train_candles
                if len(df) > self.initial_train_candles:
                    base_df = df.iloc[-self.initial_train_candles:].copy()
                else:
                    base_df = df.copy()
            else:
                # Incremental retrain: extend previous cached window with new rows since last_train_index
                prev_df = self.training_cache.get(pair)
                if prev_df is None:
                    prev_df = df.iloc[-self.initial_train_candles:].copy() if len(df) > self.initial_train_candles else df.copy()
                # New rows (simple index-based diff). If dataframe has 'date', we could filter last 24h.
                new_rows = df.iloc[self.last_train_index.get(pair, 0):].copy()
                if len(new_rows) == 0:
                    base_df = prev_df
                else:
                    combined = pd.concat([prev_df, new_rows], ignore_index=True)
                    # Keep only most recent window (rolling window behaviour)
                    if len(combined) > self.initial_train_candles:
                        base_df = combined.iloc[-self.initial_train_candles:].copy()
                    else:
                        base_df = combined

            feature_df = self.extract_advanced_features(base_df)
            target = self.create_target_variable(base_df)

            feature_columns = []
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'date',
                           'enter_long', 'enter_short', 'exit_long', 'exit_short']

            for col in feature_df.columns:
                if (col not in exclude_cols and 
                    feature_df[col].dtype in ['float64', 'int64'] and
                    not col.startswith('enter_') and
                    not col.startswith('exit_')):
                    feature_columns.append(col)

            X = feature_df[feature_columns].fillna(0)
            y = target  # NaN tail (no future) preserved; dropped by valid_mask below

            valid_mask = ~(pd.isna(y) | pd.isna(X).any(axis=1))
            X = X[valid_mask]
            y = y[valid_mask]

            if len(X) < 100:
                return {'status': 'insufficient_data'}

            # === FEATURE QUALITY VALIDATION ===
            # Remove constant features (zero variance)
            feature_variance = X.var()
            non_constant_features = feature_variance[feature_variance > 1e-10].index.tolist()

            logger.info(f"Removed {len(feature_columns) - len(non_constant_features)} "
                       f"constant features out of {len(feature_columns)}")

            if len(non_constant_features) < 5:
                logger.warning(f"Too few variable features ({len(non_constant_features)})")
                return {'status': 'insufficient_features'}

            X = X[non_constant_features]
            feature_columns = non_constant_features

            # === CLASS BALANCE VALIDATION ===
            positive_count = y.sum()
            negative_count = len(y) - positive_count
            positive_ratio = positive_count / len(y)

            logger.info(f"Class distribution: {positive_count} positive, "
                       f"{negative_count} negative ({positive_ratio:.3f} ratio)")

            # Check for severe class imbalance
            if positive_count < 10:
                logger.warning(f"Too few positive examples ({positive_count}), "
                             f"adjusting target variable")
                # Create more lenient target
                relaxed_target = self.create_target_variable(df, forward_periods=3,
                                                           profit_threshold=0.01)
                y = relaxed_target[valid_mask].fillna(0)
                positive_count = y.sum()
                positive_ratio = positive_count / len(y)
                logger.info(f"Adjusted class distribution: {positive_count} positive "
                           f"({positive_ratio:.3f} ratio)")

            if positive_count < 5:
                return {'status': 'insufficient_positive_examples'}

            # class ratio, useful training samples are ~1000. 30 features means
            # ~33 samples/feature, borderline. 15 features → ~66 samples/feature,
            # firmly inside the 10-20 rule of thumb for tree ensembles. Reduces
            # variance of the model across retrains.
            n_features_select = 15
            if len(feature_columns) > n_features_select:
                selector = SelectKBest(score_func=f_classif, k=min(n_features_select, len(feature_columns)))
                X_selected = selector.fit_transform(X, y)
                selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
                X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
                feature_columns = selected_features
                logger.info(f"Selected {len(selected_features)} best features for {pair}")

            split_idx = int(len(X) * 0.7)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Use RobustScaler for better handling of outliers
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = {}
            results = {}

            # === MODEL 1: OPTIMIZED RANDOM FOREST ===
            # Full parameter grid for comprehensive optimization
            rf_params_full = {
                'n_estimators': [150, 200, 250],
                'max_depth': [15, 20, 25],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 5, 8],
                'max_features': ['sqrt', 'log2', 0.8]
            }

            # Quick parameter grid for faster training (used by default)
            rf_params_quick = {
                'n_estimators': [150, 200],
                'max_depth': [15, 20],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 5],
                'max_features': ['sqrt', 0.8]
            }

            rf_base = RandomForestClassifier(
                random_state=42,
                n_jobs=1,
                class_weight='balanced'
            )

            # Use comprehensive grid search for datasets with sufficient data
            use_full_params = len(X_train) > 600  # More data = more thorough search
            selected_params = rf_params_full if use_full_params else rf_params_quick

            logger.info(f"Using {'full' if use_full_params else 'quick'} RF parameters "
                       f"for {len(X_train)} training samples")

            # randomly partitions samples for CV, which leaks future info into
            # training folds when the data is time-ordered. TimeSeriesSplit
            # only validates on samples strictly after the training fold.
            tscv = TimeSeriesSplit(n_splits=3)

            # Adaptive grid search based on dataset size
            rf_grid = GridSearchCV(
                rf_base,
                param_grid=selected_params,
                cv=tscv,
                scoring='f1',
                n_jobs=1
            )
            rf_grid.fit(X_train_scaled, y_train)
            models['random_forest'] = rf_grid.best_estimator_

            # === MODEL 2: OPTIMIZED GRADIENT BOOSTING ===
            gb_base = GradientBoostingClassifier(
                random_state=42,
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1e-4
            )

            gb_grid = GridSearchCV(
                gb_base,
                param_grid={
                    'n_estimators': [150, 200],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'min_samples_split': [10, 20],
                    'min_samples_leaf': [5, 10]
                },
                cv=tscv,
                scoring='f1',
                n_jobs=1
            )
            gb_grid.fit(X_train_scaled, y_train)
            models['gradient_boosting'] = gb_grid.best_estimator_

            # === MODEL 3: EXTRA TREES (EXTREMELY RANDOMIZED TREES) ===
            et = ExtraTreesClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=4,
                max_features='sqrt',
                random_state=42,
                n_jobs=1,
                class_weight='balanced'
            )
            et.fit(X_train_scaled, y_train)
            models['extra_trees'] = et

            # === MODEL 4: ADAPTIVE BOOSTING ===
            ada = AdaBoostClassifier(
                n_estimators=80,
                learning_rate=0.8,
                random_state=42
            )
            ada.fit(X_train_scaled, y_train)
            models['ada_boost'] = ada

            # === MODEL 5: SUPPORT VECTOR MACHINE (for small datasets) ===
            if len(X_train) < 2000:  # Only for smaller datasets due to computational cost
                svm = SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    probability=True,
                    random_state=42,
                    class_weight='balanced'
                )
                svm.fit(X_train_scaled, y_train)
                models['svm'] = svm

            # === MODEL 6: LOGISTIC REGRESSION (baseline) ===
            lr = LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
            lr.fit(X_train_scaled, y_train)
            models['logistic_regression'] = lr

            # === EVALUATE ALL MODELS ===
            for name, model in models.items():
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                # Calculate AUC metrics using probabilities when available
                try:
                    from sklearn.metrics import roc_auc_score, average_precision_score
                    if hasattr(model, 'predict_proba') and len(np.unique(y_test)) > 1:
                        auc_roc = roc_auc_score(y_test, y_pred_proba)
                        auc_pr = average_precision_score(y_test, y_pred_proba)
                    else:
                        # Fallback for models without predict_proba
                        auc_roc = roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0.5
                        auc_pr = average_precision_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0.5
                except ImportError:
                    auc_roc = 0.5
                    auc_pr = 0.5

                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train,
                    cv=TimeSeriesSplit(n_splits=3), scoring='f1'
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc_roc': auc_roc,
                    'auc_pr': auc_pr,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'probabilities': y_pred_proba,  # Store probabilities for ensemble
                    'feature_importance': self._get_feature_importance(model, feature_columns)
                }

                logger.info(f"{pair} {name}: Acc={accuracy:.3f}, F1={f1:.3f}, "
                           f"AUC={auc_roc:.3f}, CV={cv_mean:.3f}±{cv_std:.3f}")

            # === CREATE VOTING ENSEMBLE ===
            # Select top 3 models based on F1 score
            sorted_models = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
            top_models = [(name, results[name]['model']) for name, _ in sorted_models[:3]]

            if len(top_models) >= 2:
                voting_classifier = VotingClassifier(
                    estimators=top_models,
                    voting='soft'  # Use probability averaging
                )
                voting_classifier.fit(X_train_scaled, y_train)
                models['voting_ensemble'] = voting_classifier

                # Evaluate ensemble
                y_pred_ensemble = voting_classifier.predict(X_test_scaled)
                ensemble_f1 = f1_score(y_test, y_pred_ensemble, zero_division=0)
                ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

                results['voting_ensemble'] = {
                    'model': voting_classifier,
                    'accuracy': ensemble_accuracy,
                    'f1_score': ensemble_f1,
                    'feature_importance': {}  # Ensemble doesn't have direct feature importance
                }

                logger.info(f"{pair} Voting Ensemble: Acc={ensemble_accuracy:.3f}, F1={ensemble_f1:.3f}")

            self.models[pair] = models
            self.scalers[pair] = scaler
            self.feature_importance[pair] = results
            self.feature_columns[pair] = list(feature_columns)  # explicit list
            self.is_trained[pair] = True
            # Update retrain metadata and cache
            self.last_train_time[pair] = datetime.now(timezone.utc)
            self.last_train_index[pair] = len(df)
            self.training_cache[pair] = base_df.copy()
            
            # Save models to disk for persistence
            self._save_models_to_disk(pair)
            
            # Find best model
            best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
            best_f1 = results[best_model_name]['f1_score']
            
            return {
                'status': 'success',
                'results': results,
                'feature_columns': feature_columns,
                'n_samples': len(X),
                'best_model': best_model_name,
                'best_f1_score': best_f1,
                'n_models': len(models)
            }
            
        except Exception as e:
            logger.warning(f"Model training failed for {pair}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _get_feature_importance(self, model, feature_columns):
        """Extract feature importance from different model types.

        Removed `strict=True` from zip(). The strict
        argument requires Python 3.10+; with 3.9 it raises TypeError and the
        except clause silently swallows it, leaving feature_importance empty.
        We rely on the fact that both arrays come from the same model fit and
        therefore have the same length, which is the invariant strict was
        guarding against.
        """
        try:
            if hasattr(model, 'feature_importances_'):
                return dict(zip(feature_columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                # For linear models like LogisticRegression
                importance = abs(model.coef_[0])
                return dict(zip(feature_columns, importance))
            else:
                return {}
        except Exception:
            return {}
    
    def predict_entry_probability(self, df: pd.DataFrame, pair: str) -> pd.Series:
        """Predict probability of profitable entry using advanced ensemble models"""
        if not SKLEARN_AVAILABLE or pair not in self.is_trained or not self.is_trained[pair]:
            return pd.Series(0.5, index=df.index)

        try:
            feature_df = self.extract_advanced_features(df)

            # (legacy model), abort to neutral 0.5 rather than guessing — guessing
            # creates train/inference skew that silently breaks the scaler.
            feature_columns = self.feature_columns.get(pair)
            if not feature_columns:
                logger.warning(f"[ML] No persisted feature_columns for {pair}; "
                               f"returning neutral 0.5 until next retrain.")
                return pd.Series(0.5, index=df.index)

            # Some features may be missing at inference (e.g. minima_sort_threshold
            # not yet computed). Fill them with 0 to keep dimensions stable.
            missing = [c for c in feature_columns if c not in feature_df.columns]
            if missing:
                for c in missing:
                    feature_df[c] = 0.0

            X = feature_df[feature_columns].fillna(0)

            # Defensive: scaler must have been fit on same number of features.
            scaler = self.scalers.get(pair)
            if scaler is None:
                return pd.Series(0.5, index=df.index)
            try:
                X_scaled = scaler.transform(X)
            except Exception as e:
                logger.warning(f"[ML] Scaler transform failed for {pair} "
                               f"(feature mismatch?): {e}. Returning neutral 0.5.")
                return pd.Series(0.5, index=df.index)
            
            # Get predictions from all available models
            model_predictions = {}
            model_weights = {}
            
            for model_name, model in self.models[pair].items():
                try:
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_scaled)[:, 1]
                    else:
                        # For models without probability prediction
                        pred = model.predict(X_scaled)
                        prob = (pred + 1) / 2  # Convert -1,1 to 0,1 or similar normalization
                    
                    model_predictions[model_name] = prob
                    
                    # Weight based on model performance (F1 score or accuracy)
                    if pair in self.feature_importance and model_name in self.feature_importance[pair]:
                        performance_metrics = self.feature_importance[pair][model_name]
                        # Use F1 score if available, otherwise accuracy
                        weight = performance_metrics.get('f1_score', 
                                performance_metrics.get('accuracy', 0.5))
                    else:
                        weight = 0.5
                    
                    model_weights[model_name] = max(weight, 0.1)  # Minimum weight of 0.1
                    
                except Exception as e:
                    logger.warning(f"Failed to get predictions from {model_name} for {pair}: {e}")
                    continue
            
            if not model_predictions:
                return pd.Series(0.5, index=df.index)
            
            # === ADVANCED ENSEMBLE PREDICTION ===
            
            # Method 1: Weighted average by performance
            total_weight = sum(model_weights.values())
            if total_weight > 0:
                weighted_avg = np.zeros(len(X))
                for model_name, predictions in model_predictions.items():
                    weight = model_weights[model_name] / total_weight
                    weighted_avg += predictions * weight
            else:
                weighted_avg = np.mean(list(model_predictions.values()), axis=0)
            
            # Method 2: Voting ensemble (if available)
            if 'voting_ensemble' in model_predictions:
                ensemble_pred = model_predictions['voting_ensemble']
                # Combine weighted average with voting ensemble
                final_prediction = 0.6 * ensemble_pred + 0.4 * weighted_avg
            else:
                final_prediction = weighted_avg
            
            # Method 3: Dynamic model selection based on market conditions
            # exist on this class: _detect_market_regime() (defined at line
            # ~1308, returns a pd.Series of per-candle regime labels) and
            # _detect_market_condition() (defined at line ~1970, returns a
            # single string for the most recent candle). The hasattr() check
            # below tests that the class is well-formed; the call uses the
            # scalar-returning method because downstream code wants a single
            # 'trending'/'ranging'/'volatile' label. This works correctly —
            # do NOT "unify" the names without also updating the downstream
            # equality checks (lines 1921, 1925, etc.) which expect a string.
            if hasattr(self, '_detect_market_regime'):
                market_regime = self._detect_market_condition(df)
                
                # Adjust predictions based on market conditions
                if market_regime == 'trending':
                    # In trending markets, prefer gradient boosting
                    if 'gradient_boosting' in model_predictions:
                        final_prediction = (0.5 * final_prediction + 
                                          0.5 * model_predictions['gradient_boosting'])
                elif market_regime == 'volatile':
                    # In volatile markets, prefer random forest
                    if 'random_forest' in model_predictions:
                        final_prediction = (0.5 * final_prediction + 
                                          0.5 * model_predictions['random_forest'])
                elif market_regime == 'ranging':
                    # In ranging markets, prefer SVM or logistic regression
                    if 'svm' in model_predictions:
                        final_prediction = (0.6 * final_prediction + 
                                          0.4 * model_predictions['svm'])
                    elif 'logistic_regression' in model_predictions:
                        final_prediction = (0.6 * final_prediction + 
                                          0.4 * model_predictions['logistic_regression'])
            
            # === CONFIDENCE ADJUSTMENT ===
            
            # Calculate prediction confidence based on model agreement
            if len(model_predictions) > 1:
                predictions_array = np.array(list(model_predictions.values()))
                prediction_std = np.std(predictions_array, axis=0)
                
                # Higher standard deviation = lower confidence
                confidence_factor = 1 - np.clip(prediction_std * 2, 0, 0.3)
                
                # Adjust predictions toward neutral when confidence is low
                final_prediction = (final_prediction * confidence_factor + 
                                  0.5 * (1 - confidence_factor))
            
            # === OUTLIER DETECTION AND SMOOTHING ===
            
            # Apply rolling smoothing to reduce noise
            result_series = pd.Series(final_prediction, index=df.index)
            smoothed_result = result_series.rolling(window=3, center=False, min_periods=1).mean()
            smoothed_result = smoothed_result.fillna(result_series)
            
            # Ensure values are in valid range [0, 1]
            smoothed_result = smoothed_result.clip(0, 1)
            
            return smoothed_result
            
        except Exception as e:
            logger.warning(f"Advanced prediction failed for {pair}: {e}")
            return pd.Series(0.5, index=df.index)
    
    def _detect_market_condition(self, df: pd.DataFrame) -> str:
        """Detect current market regime for dynamic model selection.

        ANF_v2: HMM-first with transparent fallback. Tries the fitted Gaussian
        HMM; if it returns 'unknown' (not fitted, hmmlearn missing, inference
        error) we fall back to the legacy threshold rules below. The HMM can
        never break this method — _legacy_detect_market_condition always
        produces a usable answer.
        """
        try:
            if getattr(self, 'hmm_detector', None) is not None and self.hmm_detector.is_fitted:
                label = self.hmm_detector.detect(df)
                if label and label != 'unknown':
                    return label
        except Exception as e:
            logger.debug(f"[HMM] regime detection error, using legacy ({e})")
        return self._legacy_detect_market_condition(df)

    def detect_market_regime_with_confidence(self, df: pd.DataFrame):
        """ANF_v2: richer accessor returning (label, posterior_probs_dict).

        Used by the strategy to make GRADED decisions (e.g. soften leverage
        when the regime probability is weak). Falls back to (legacy_label, None)
        when the HMM is unavailable, so callers must handle a None prob dict.
        """
        try:
            if getattr(self, 'hmm_detector', None) is not None and self.hmm_detector.is_fitted:
                label, probs = self.hmm_detector.detect(df, with_probs=True)
                if label and label != 'unknown':
                    return label, probs
        except Exception as e:
            logger.debug(f"[HMM] confidence detection error, using legacy ({e})")
        return self._legacy_detect_market_condition(df), None

    def _legacy_detect_market_condition(self, df: pd.DataFrame) -> str:
        """Original threshold-based regime detector (pre-ANF_v2).

        Preserved verbatim as the fallback when the HMM is unavailable or
        unfitted. Three rules over the last 20 candles: high volatility ->
        'volatile', else strong move -> 'trending', else 'ranging'.
        """
        try:
            # Simple market regime detection
            recent_data = df.tail(20)
            
            if len(recent_data) < 10:
                return 'unknown'
            
            # Calculate volatility
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Calculate trend strength
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            if volatility > 0.03:  # High volatility threshold
                return 'volatile'
            elif abs(price_change) > 0.05:  # Strong trend threshold
                return 'trending'
            else:
                return 'ranging'
                
        except Exception:
            return 'unknown'


# Initialize the predictive engine globally
predictive_engine = None

def get_engine(config=None, strategy_name: Optional[str] = None):
    """Lazy global accessor. if called with a strategy_name and
    the existing engine has a different name, do NOT silently reuse it —
    create a fresh one. This protects against the case where two strategy
    classes share the same Python process (uncommon in Freqtrade, but possible
    in test harnesses) and would otherwise collide model stores."""
    global predictive_engine
    if predictive_engine is None:
        predictive_engine = AdvancedPredictiveEngine(config=config, strategy_name=strategy_name)
    elif strategy_name and getattr(predictive_engine, 'strategy_name', None) not in (None, "shared", strategy_name):
        logger.warning(
            "[ML] get_engine called with strategy_name=%r but existing engine "
            "is bound to %r. Replacing global engine to avoid model collision.",
            strategy_name, predictive_engine.strategy_name,
        )
        predictive_engine = AdvancedPredictiveEngine(config=config, strategy_name=strategy_name)
    return predictive_engine


def calculate_advanced_predictive_signals(dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Main function to calculate advanced predictive signals with enhanced models."""

    # === CRITICAL FIX: ENSURE PREDICTIVE ENGINE IS INITIALIZED ===
    global predictive_engine
    if predictive_engine is None:
        _log_info_throttled(f"[ML] Initializing predictive engine for {pair}", pair, 600)
        predictive_engine = AdvancedPredictiveEngine(config=None)

    # In backtest mode, populate_indicators is called once with the full dataset,
    # so any model trained inside it has access to "future" data relative to early
    # candles. We short-circuit to neutral predictions so T1 (ML Ultra) never
    # fires. T2/T3/T4 entries continue to work via the deterministic indicators.
    if getattr(predictive_engine, 'backtest_mode', False):
        dataframe['ml_entry_probability'] = pd.Series(0.5, index=dataframe.index)
        dataframe['ml_enhanced_score'] = dataframe.get('ultimate_score', pd.Series(0.5, index=dataframe.index))
        dataframe['ml_high_confidence'] = 0
        dataframe['ml_ultra_confidence'] = 0
        dataframe['ml_model_agreement'] = 0.5
        return dataframe

    try:
        need_training = False
        assets_exist = False
        
        # Safe check for assets existence
        try:
            assets_exist = predictive_engine._assets_exist(pair)
        except Exception as e:
            logger.warning(f"[ML] Could not check assets for {pair}: {e}")
            assets_exist = False

        # Check time since strategy startup
        now_utc = datetime.now(timezone.utc)
        
        # Safe access to strategy_start_time
        if hasattr(predictive_engine, 'strategy_start_time'):
            hours_since_startup = (now_utc - predictive_engine.strategy_start_time).total_seconds() / 3600.0
        else:
            # Fallback: assume we just started
            logger.info(f"[ML] No strategy_start_time found, assuming fresh start for {pair}")
            hours_since_startup = 0.0

        # Retrain decision logic:
        # - If no model assets exist for this pair → train immediately (cold start)
        # - Otherwise → only retrain when BOTH conditions hold:
        #     (a) at least `retrain_interval_hours` have passed since the last successful train
        #     (b) at least `min_new_candles_for_retrain` new candles have arrived since last train
        # This prevents the previous bug where retraining was triggered every candle once
        # `hours_since_startup >= retrain_after_startup_hours` became permanently true.
        if not assets_exist:
            # Missing assets -> train immediately
            if len(dataframe) >= 200:
                need_training = True
        elif predictive_engine.enable_startup_retrain:
            # Use per-pair last_train_time as the reference. If absent (e.g. models loaded
            # from disk after a process restart), fall back to strategy_start_time so the
            # retrain interval counts from when this process began running.
            last_train = predictive_engine.last_train_time.get(pair)
            if last_train is None:
                last_train = predictive_engine.strategy_start_time

            hours_since_train = (now_utc - last_train).total_seconds() / 3600.0
            new_candles = len(dataframe) - predictive_engine.last_train_index.get(pair, 0)

            if (hours_since_train >= predictive_engine.retrain_interval_hours and
                new_candles >= predictive_engine.min_new_candles_for_retrain and
                len(dataframe) >= 200):
                need_training = True
                _log_info_throttled(
                    f"[ML] {predictive_engine.retrain_interval_hours}h retrain triggered for {pair} "
                    f"(hours_since_train={hours_since_train:.1f}, new_candles={new_candles})",
                    pair, 600
                )

        if need_training:
            _log_info_throttled(f"[ML] Training models for {pair} (len={len(dataframe)})", pair, 600)
            training_result = predictive_engine.train_predictive_models(dataframe, pair)
            status = training_result.get('status')
            if status == 'success':
                best_model = training_result.get('best_model', 'unknown')
                best_f1 = training_result.get('best_f1_score', 0)
                n_models = training_result.get('n_models', 0)
                _log_info_throttled(f"🚀 {pair} ML Training Success: Best {best_model} F1={best_f1:.3f}", pair, 600)
            elif status == 'skipped_assets_exist':
                logger.debug(f"[ML] Skip training (assets present) {pair}")
            elif status == 'insufficient_data':
                # Log at debug-level via throttle. The detailed "only X candles
                # available, need >= Y" message was already emitted by
                # train_predictive_models() itself, so we don't repeat it here.
                logger.debug(f"[ML] {pair} waiting for more candles "
                             f"(have={training_result.get('have', '?')}, "
                             f"need={training_result.get('need', '?')})")
            else:
                logger.warning(f"❌ ML training status {pair}: {status}")
        
        # Enhanced ML probability prediction
        # Only use ML probability if model is actually trained — otherwise default to neutral 0.5
        # which will prevent any ML-based signal from triggering (tiers require >0.52 or <0.48)
        model_is_ready = predictive_engine.is_trained.get(pair, False)
        if model_is_ready:
            dataframe['ml_entry_probability'] = predictive_engine.predict_entry_probability(dataframe, pair)
        else:
            logger.debug(f"[ML] Models not yet trained for {pair} — using neutral 0.5")
            dataframe['ml_entry_probability'] = pd.Series(0.5, index=dataframe.index)
        
        # Get momentum and volatility regime safely
        momentum_regime = dataframe.get('momentum_regime')
        volatility_regime = dataframe.get('volatility_regime')
        quantum_coherence = dataframe.get('quantum_momentum_coherence')
        neural_pattern = dataframe.get('neural_pattern_score')
        
        # Advanced confidence scoring with safe comparisons
        ml_high_conf_conditions = (dataframe['ml_entry_probability'] > 0.8)
        
        if momentum_regime is not None:
            ml_high_conf_conditions &= (momentum_regime > 0)
        
        if volatility_regime is not None:
            ml_high_conf_conditions &= (volatility_regime < 2)
            
        dataframe['ml_high_confidence'] = ml_high_conf_conditions.astype(int)
        
        # Ultra-high confidence entries with safe checks
        ml_ultra_conf_conditions = (dataframe['ml_entry_probability'] > 0.9)
        
        if quantum_coherence is not None:
            ml_ultra_conf_conditions &= (quantum_coherence > 0.7)
        else:
            # Use fallback threshold if quantum analysis not available
            ml_ultra_conf_conditions &= (dataframe['ml_entry_probability'] > 0.92)
            
        if neural_pattern is not None:
            ml_ultra_conf_conditions &= (neural_pattern > 0.8)
        else:
            # Use fallback threshold if neural analysis not available
            ml_ultra_conf_conditions &= (dataframe['ml_entry_probability'] > 0.93)
            
        dataframe['ml_ultra_confidence'] = ml_ultra_conf_conditions.astype(int)
        
        # Enhanced score combination
        if 'ultimate_score' in dataframe.columns:
            # Dynamic weighting based on model performance - fix Series comparison
            ml_volatility = dataframe['ml_entry_probability'].rolling(20).std().fillna(0.3)
            ml_weight = ml_volatility.clip(upper=0.5)  # Safe way to apply min with Series
            traditional_weight = 1 - ml_weight
            
            dataframe['ml_enhanced_score'] = (
                dataframe['ultimate_score'] * traditional_weight +
                dataframe['ml_entry_probability'] * ml_weight
            )
        else:
            dataframe['ml_enhanced_score'] = dataframe['ml_entry_probability']
        
        # Model agreement indicator - fix Series std calculation
        if pair in predictive_engine.models and len(predictive_engine.models[pair]) > 1:
            # Calculate prediction variance as agreement measure
            prob_std = dataframe['ml_entry_probability'].rolling(5).std().fillna(0)
            dataframe['ml_model_agreement'] = (1 - prob_std).clip(lower=0, upper=1)
        else:
            dataframe['ml_model_agreement'] = 0.8  # Default agreement
        
        return dataframe
        
    except Exception as e:
        logger.warning(f"Advanced predictive analysis failed for {pair}: {e}")
        dataframe['ml_entry_probability'] = 0.5
        dataframe['ml_enhanced_score'] = dataframe.get('ultimate_score', 0.5)
        dataframe['ml_high_confidence'] = 0
        dataframe['ml_ultra_confidence'] = 0
        dataframe['ml_model_agreement'] = 0.5
        return dataframe


def calculate_quantum_momentum_analysis(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Quantum-inspired momentum analysis for ultra-precise predictions"""
    try:
        momentum_periods = [3, 5, 8, 13, 21, 34]
        momentum_matrix = pd.DataFrame()
        
        for period in momentum_periods:
            momentum_matrix[f'mom_{period}'] = dataframe['close'].pct_change(period)
        
        dataframe['quantum_momentum_coherence'] = (
            momentum_matrix.std(axis=1) / (momentum_matrix.mean(axis=1).abs() + 1e-10)
        )
        
        # Calculate momentum entanglement using correlation matrix
        def calculate_entanglement(window_data):
            if len(window_data) < 10:
                return 0
            try:
                corr_matrix = window_data.corr()
                if corr_matrix.empty or corr_matrix.isna().all().all():
                    return 0
                # Get upper triangular correlation values (excluding diagonal)
                upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
                correlations = corr_matrix.values[upper_tri_indices]
                # Remove NaN values and calculate mean
                valid_correlations = correlations[~np.isnan(correlations)]
                return valid_correlations.mean() if len(valid_correlations) > 0 else 0
            except Exception:
                return 0
        
        entanglement_values = []
        for i in range(len(momentum_matrix)):
            if i < 20:
                entanglement_values.append(0.5)
            else:
                window_data = momentum_matrix.iloc[i-19:i+1]
                entanglement = calculate_entanglement(window_data)
                entanglement_values.append(entanglement)
        
        dataframe['momentum_entanglement'] = pd.Series(entanglement_values, index=dataframe.index)
        
        price_uncertainty = dataframe['close'].rolling(20).std()
        momentum_uncertainty = momentum_matrix['mom_8'].rolling(20).std()
        dataframe['heisenberg_uncertainty'] = price_uncertainty * momentum_uncertainty
        
        if 'maxima_sort_threshold' in dataframe.columns:
            resistance_distance = (
                dataframe['maxima_sort_threshold'] - dataframe['close']) / dataframe['close']
            dataframe['quantum_tunnel_up_prob'] = np.exp(-resistance_distance.abs() * 10)
        else:
            dataframe['quantum_tunnel_up_prob'] = 0.5
        
        if 'minima_sort_threshold' in dataframe.columns:
            support_distance = (
                dataframe['close'] - dataframe['minima_sort_threshold']) / dataframe['close']
            dataframe['quantum_tunnel_down_prob'] = np.exp(-support_distance.abs() * 10)
        else:
            dataframe['quantum_tunnel_down_prob'] = 0.5
        
        return dataframe
        
    except Exception as e:
        logger.warning(f"Quantum momentum analysis failed: {e}")
        dataframe['quantum_momentum_coherence'] = 0.5
        dataframe['momentum_entanglement'] = 0.5
        dataframe['heisenberg_uncertainty'] = 1.0
        dataframe['quantum_tunnel_up_prob'] = 0.5
        dataframe['quantum_tunnel_down_prob'] = 0.5
        return dataframe


def calculate_neural_pattern_recognition(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Neural pattern recognition for complex market patterns"""
    try:
        dataframe['body_size'] = abs(dataframe['close'] - dataframe['open']) / dataframe['close']
        dataframe['upper_shadow'] = (
            dataframe['high'] - np.maximum(dataframe['open'], dataframe['close'])
        ) / dataframe['close']
        dataframe['lower_shadow'] = (
            np.minimum(dataframe['open'], dataframe['close']) - dataframe['low']
        ) / dataframe['close']
        dataframe['candle_range'] = (dataframe['high'] - dataframe['low']) / dataframe['close']

        pattern_memory = []
        for i in range(len(dataframe)):
            if i < 5:
                pattern_memory.append(0)
                continue

            recent_patterns = dataframe[['body_size', 'upper_shadow', 'lower_shadow']].iloc[i-4:i+1]
            pattern_signature = recent_patterns.values.flatten()
            pattern_norm = np.linalg.norm(pattern_signature)

            if pattern_norm > 0:
                pattern_score = min(1.0, pattern_norm / 0.1)
            else:
                pattern_score = 0

            pattern_memory.append(pattern_score)

        dataframe['neural_pattern_score'] = pd.Series(pattern_memory, index=dataframe.index)
        dataframe['pattern_prediction_confidence'] = dataframe['neural_pattern_score'].rolling(10).std()

        return dataframe

    except Exception as e:
        logger.warning(f"Neural pattern recognition failed: {e}")
        dataframe['neural_pattern_score'] = 0.5
        dataframe['pattern_prediction_confidence'] = 0.5
        dataframe['body_size'] = 0.01
        dataframe['candle_range'] = 0.02
        return dataframe
def calculate_exit_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced exit signals based on market deterioration"""
    # === MOMENTUM DETERIORATION ===
    dataframe['momentum_deteriorating'] = (
        (dataframe['momentum_quality'] < dataframe['momentum_quality'].shift(1)) &
        (dataframe['momentum_acceleration'] < 0) &
        (dataframe['price_momentum'] < dataframe['price_momentum'].shift(1))
    ).astype(int)

    # === VOLUME DETERIORATION ===
    dataframe['volume_deteriorating'] = (
        (dataframe['volume_strength'] < 0.8) &
        (dataframe['selling_pressure'] > dataframe['buying_pressure']) &
        (dataframe['volume_pressure'] < 0)
    ).astype(int)

    # === STRUCTURE DETERIORATION ===
    dataframe['structure_deteriorating'] = (
        (dataframe['structure_score'] < -1) &
        (dataframe['bearish_structure'] > dataframe['bullish_structure']) &
        (dataframe['structure_break_down'] == 1)
    ).astype(int)

    # === CONFLUENCE BREAKDOWN ===
    dataframe['confluence_breakdown'] = (
        (dataframe['confluence_score'] < 2) &
        (dataframe['near_resistance'] == 1) &
        (dataframe['volume_spike'] == 0)
    ).astype(int)

    # === TREND WEAKNESS ===
    dataframe['trend_weakening'] = (
        (dataframe['trend_strength'] < 0) &
        (dataframe['close'] < dataframe['ema50']) &
        (dataframe['strong_downtrend'] == 1)
    ).astype(int)

    # === ULTIMATE EXIT SCORE ===
    dataframe['exit_pressure'] = (
        dataframe['momentum_deteriorating'] * 2 +
        dataframe['volume_deteriorating'] * 2 +
        dataframe['structure_deteriorating'] * 2 +
        dataframe['confluence_breakdown'] * 1 +
        dataframe['trend_weakening'] * 1
    )

    # === RSI OVERBOUGHT WITH DIVERGENCE ===
    dataframe['rsi_exit_signal'] = (
        (dataframe['rsi'] > 75) &
        (
            (dataframe['rsi_divergence_bear'] == 1) |
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &
            (dataframe['close'] < dataframe['close'].shift(1))
        )
    ).astype(int)

    # === PROFIT TAKING LEVELS ===
    mml_resistance_levels = ['[6/8]P', '[8/8]P']
    dataframe['near_resistance_level'] = 0

    for level in mml_resistance_levels:
        if level in dataframe.columns:
            near_level = (
                (dataframe['close'] >= dataframe[level] * 0.99) &
                (dataframe['close'] <= dataframe[level] * 1.02)
            ).astype(int)
            dataframe['near_resistance_level'] += near_level

    # === VOLATILITY SPIKE EXIT ===
    dataframe['volatility_spike'] = (
        dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 1.5
    ).astype(int)

    # === EXHAUSTION SIGNALS ===
    dataframe['bullish_exhaustion'] = (
        (dataframe['consecutive_green'] >= 4) &
        (dataframe['rsi'] > 70) &
        (dataframe['volume'] < dataframe['avg_volume'] * 0.8) &
        (dataframe['momentum_acceleration'] < 0)
    ).astype(int)

    return dataframe


def calculate_dynamic_profit_targets(dataframe: pd.DataFrame, entry_type_col: str = 'entry_type') -> pd.DataFrame:
    """Calculate dynamic profit targets based on entry quality and market conditions"""

    # Base profit targets based on ATR
    dataframe['base_profit_target'] = dataframe['atr'] * 2

    # Adjust based on entry type
    dataframe['profit_multiplier'] = 1.0
    if entry_type_col in dataframe.columns:
        dataframe.loc[dataframe[entry_type_col] == 3, 'profit_multiplier'] = 2.0  # High quality
        dataframe.loc[dataframe[entry_type_col] == 2, 'profit_multiplier'] = 1.5  # Medium quality
        dataframe.loc[dataframe[entry_type_col] == 1, 'profit_multiplier'] = 1.2  # Backup
        dataframe.loc[dataframe[entry_type_col] == 4, 'profit_multiplier'] = 2.5  # Breakout
        dataframe.loc[dataframe[entry_type_col] == 5, 'profit_multiplier'] = 1.8  # Reversal

    # Final profit target
    dataframe['dynamic_profit_target'] = dataframe['base_profit_target'] * dataframe['profit_multiplier']

    return dataframe


def calculate_advanced_stop_loss(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe['base_stop_loss'] = dataframe['atr'] * 1.5
    if 'minima_sort_threshold' in dataframe.columns:
        dataframe['support_stop_loss'] = dataframe['close'] - dataframe['minima_sort_threshold']
        dataframe['support_stop_loss'] = dataframe['support_stop_loss'].clip(
            dataframe['base_stop_loss'] * 0.5,
            dataframe['base_stop_loss'] * 1.5  # Reduced from 2.0
        )
        dataframe['final_stop_loss'] = np.minimum(
            dataframe['base_stop_loss'],
            dataframe['support_stop_loss']
        ).clip(-0.15, -0.01)  # Hard cap at -15%
    else:
        dataframe['final_stop_loss'] = dataframe['base_stop_loss'].clip(-0.15, -0.01)
    return dataframe

def calculate_confluence_score(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Multi-factor confluence analysis - much better than BTC correlation"""

    # Support/Resistance Confluence
    dataframe['near_support'] = (
        (dataframe['close'] <= dataframe['minima_sort_threshold'] * 1.02) &
        (dataframe['close'] >= dataframe['minima_sort_threshold'] * 0.98)
    ).astype(int)

    dataframe['near_resistance'] = (
        (dataframe['close'] <= dataframe['maxima_sort_threshold'] * 1.02) &
        (dataframe['close'] >= dataframe['maxima_sort_threshold'] * 0.98)
    ).astype(int)

    # MML Level Confluence
    mml_levels = ['[0/8]P', '[2/8]P', '[4/8]P', '[6/8]P', '[8/8]P']
    dataframe['near_mml'] = 0

    for level in mml_levels:
        if level in dataframe.columns:
            near_level = (
                (dataframe['close'] <= dataframe[level] * 1.015) &
                (dataframe['close'] >= dataframe[level] * 0.985)
            ).astype(int)
            dataframe['near_mml'] += near_level

    # Volume Confluence
    dataframe['volume_spike'] = (
        dataframe['volume'] > dataframe['avg_volume'] * 1.5
    ).astype(int)

    # RSI Confluence Zones
    dataframe['rsi_oversold'] = (dataframe['rsi'] < 30).astype(int)
    dataframe['rsi_overbought'] = (dataframe['rsi'] > 70).astype(int)
    dataframe['rsi_neutral'] = (
        (dataframe['rsi'] >= 40) & (dataframe['rsi'] <= 60)
    ).astype(int)

    # EMA Confluence
    dataframe['above_ema'] = (dataframe['close'] > dataframe['ema50']).astype(int)

    # CONFLUENCE SCORE (0-6)
    dataframe['confluence_score'] = (
        dataframe['near_support'] +
        dataframe['near_mml'].clip(0, 2) +  # Max 2 points for MML
        dataframe['volume_spike'] +
        dataframe['rsi_oversold'] +
        dataframe['above_ema'] +
        (dataframe['trend_strength'] > 0.01).astype(int)  # Positive trend
    )

    return dataframe


def calculate_smart_volume(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Advanced volume analysis - beats any external correlation"""

    # Volume-Price Trend (VPT)
    price_change_pct = (dataframe['close'] - dataframe['close'].shift(1)) / dataframe['close'].shift(1)
    dataframe['vpt'] = (dataframe['volume'] * price_change_pct).fillna(0).cumsum()

    # Volume moving averages
    dataframe['volume_sma20'] = dataframe['volume'].rolling(20).mean()
    dataframe['volume_sma50'] = dataframe['volume'].rolling(50).mean()

    # Volume strength
    dataframe['volume_strength'] = dataframe['volume'] / dataframe['volume_sma20']

    # Smart money indicators
    dataframe['accumulation'] = (
        (dataframe['close'] > dataframe['open']) &  # Green candle
        (dataframe['volume'] > dataframe['volume_sma20'] * 1.2) &  # High volume
        (dataframe['close'] > (dataframe['high'] + dataframe['low']) / 2)  # Close in upper half
    ).astype(int)

    dataframe['distribution'] = (
        (dataframe['close'] < dataframe['open']) &  # Red candle
        (dataframe['volume'] > dataframe['volume_sma20'] * 1.2) &  # High volume
        (dataframe['close'] < (dataframe['high'] + dataframe['low']) / 2)  # Close in lower half
    ).astype(int)

    # Buying/Selling pressure
    dataframe['buying_pressure'] = dataframe['accumulation'].rolling(5).sum()
    dataframe['selling_pressure'] = dataframe['distribution'].rolling(5).sum()

    # Net volume pressure
    dataframe['volume_pressure'] = dataframe['buying_pressure'] - dataframe['selling_pressure']

    # Volume trend
    dataframe['volume_trend'] = (
        dataframe['volume_sma20'] > dataframe['volume_sma50']
    ).astype(int)

    # Money flow
    typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
    money_flow = typical_price * dataframe['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_flow_sum = positive_flow.rolling(14).sum()
    negative_flow_sum = negative_flow.rolling(14).sum()

    dataframe['money_flow_ratio'] = positive_flow_sum / (negative_flow_sum + 1e-10)
    dataframe['money_flow_index'] = 100 - (100 / (1 + dataframe['money_flow_ratio']))

    return dataframe


def calculate_advanced_momentum(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Multi-timeframe momentum system - superior to BTC correlation"""

    # Multi-timeframe momentum
    dataframe['momentum_3'] = dataframe['close'].pct_change(6)
    dataframe['momentum_7'] = dataframe['close'].pct_change(14)
    dataframe['momentum_14'] = dataframe['close'].pct_change(28)
    dataframe['momentum_21'] = dataframe['close'].pct_change(21)

    # Momentum acceleration
    dataframe['momentum_acceleration'] = (
        dataframe['momentum_3'] - dataframe['momentum_3'].shift(3)
    )

    # Momentum consistency
    dataframe['momentum_consistency'] = (
        (dataframe['momentum_3'] > 0).astype(int) +
        (dataframe['momentum_7'] > 0).astype(int) +
        (dataframe['momentum_14'] > 0).astype(int)
    )

    # Momentum divergence with volume
    dataframe['price_momentum_rank'] = dataframe['momentum_7'].rolling(20).rank(pct=True)
    dataframe['volume_momentum_rank'] = dataframe['volume_strength'].rolling(20).rank(pct=True)

    dataframe['momentum_divergence'] = (
        dataframe['price_momentum_rank'] - dataframe['volume_momentum_rank']
    ).abs()

    # Momentum strength
    dataframe['momentum_strength'] = (
        dataframe['momentum_3'].abs() +
        dataframe['momentum_7'].abs() +
        dataframe['momentum_14'].abs()
    ) / 3

    # Momentum quality score (0-5)
    dataframe['momentum_quality'] = (
        (dataframe['momentum_3'] > 0).astype(int) +
        (dataframe['momentum_7'] > 0).astype(int) +
        (dataframe['momentum_acceleration'] > 0).astype(int) +
        (dataframe['volume_strength'] > 1.1).astype(int) +
        (dataframe['momentum_divergence'] < 0.3).astype(int)
    )

    # Rate of Change
    dataframe['roc_5'] = dataframe['close'].pct_change(5) * 100
    dataframe['roc_10'] = dataframe['close'].pct_change(10) * 100
    dataframe['roc_20'] = dataframe['close'].pct_change(20) * 100

    # Momentum oscillator
    dataframe['momentum_oscillator'] = (
        dataframe['roc_5'] + dataframe['roc_10'] + dataframe['roc_20']
    ) / 3

    return dataframe


def calculate_market_structure(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Market structure analysis - intrinsic trend recognition"""

    # Higher highs, higher lows detection
    dataframe['higher_high'] = (
        (dataframe['high'] > dataframe['high'].shift(1)) &
        (dataframe['high'].shift(1) > dataframe['high'].shift(2))
    ).astype(int)

    dataframe['higher_low'] = (
        (dataframe['low'] > dataframe['low'].shift(1)) &
        (dataframe['low'].shift(1) > dataframe['low'].shift(2))
    ).astype(int)

    dataframe['lower_high'] = (
        (dataframe['high'] < dataframe['high'].shift(1)) &
        (dataframe['high'].shift(1) < dataframe['high'].shift(2))
    ).astype(int)

    dataframe['lower_low'] = (
        (dataframe['low'] < dataframe['low'].shift(1)) &
        (dataframe['low'].shift(1) < dataframe['low'].shift(2))
    ).astype(int)

    # Market structure scores
    dataframe['bullish_structure'] = (
        dataframe['higher_high'].rolling(5).sum() +
        dataframe['higher_low'].rolling(5).sum()
    )

    dataframe['bearish_structure'] = (
        dataframe['lower_high'].rolling(5).sum() +
        dataframe['lower_low'].rolling(5).sum()
    )

    dataframe['structure_score'] = (
        dataframe['bullish_structure'] - dataframe['bearish_structure']
    )

    # Swing highs and lows
    # Live-safe pivot detection without using future candles (no shift(-1))
    # Confirm swing at the PREVIOUS candle using only information up to current bar.
    # A swing high at t-1 is when high[t-1] > high[t-2] and high[t-1] > high[t].
    prev_high = dataframe['high'].shift(1)
    prev_low = dataframe['low'].shift(1)
    dataframe['swing_high'] = (
        (prev_high > dataframe['high'].shift(2)) &
        (prev_high > dataframe['high'])
    ).astype(int)

    # A swing low at t-1 is when low[t-1] < low[t-2] and low[t-1] < low[t].
    dataframe['swing_low'] = (
        (prev_low < dataframe['low'].shift(2)) &
        (prev_low < dataframe['low'])
    ).astype(int)

    # Market structure breaks
    # Use previous candle values where the swing was confirmed to avoid lookahead bias
    swing_highs = prev_high.where(dataframe['swing_high'] == 1)
    swing_lows = prev_low.where(dataframe['swing_low'] == 1)

    # Structure break detection
    dataframe['structure_break_up'] = (
        dataframe['close'] > swing_highs.ffill()
    ).astype(int)

    dataframe['structure_break_down'] = (
        dataframe['close'] < swing_lows.ffill()
    ).astype(int)

    # Trend strength based on structure
    dataframe['structure_trend_strength'] = (
        dataframe['structure_score'] / 10  # Normalize
    ).clip(-1, 1)

    # Support and resistance strength
    dataframe['support_strength'] = dataframe['swing_low'].rolling(20).sum()
    dataframe['resistance_strength'] = dataframe['swing_high'].rolling(20).sum()

    return dataframe


def calculate_advanced_entry_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Advanced entry signal generation"""

    # Multi-factor signal strength
    dataframe['signal_strength'] = 0

    # Confluence signals
    dataframe['confluence_signal'] = (dataframe['confluence_score'] >= 3).astype(int)
    dataframe['signal_strength'] += dataframe['confluence_signal'] * 2

    # Volume signals
    dataframe['volume_signal'] = (
        (dataframe['volume_pressure'] >= 2) &
        (dataframe['volume_strength'] > 1.2)
    ).astype(int)
    dataframe['signal_strength'] += dataframe['volume_signal'] * 2

    # Momentum signals
    dataframe['momentum_signal'] = (
        (dataframe['momentum_quality'] >= 3) &
        (dataframe['momentum_acceleration'] > 0)
    ).astype(int)
    dataframe['signal_strength'] += dataframe['momentum_signal'] * 2

    # Structure signals
    dataframe['structure_signal'] = (
        (dataframe['structure_score'] > 0) &
        (dataframe['structure_break_up'] == 1)
    ).astype(int)
    dataframe['signal_strength'] += dataframe['structure_signal'] * 1

    # RSI position signal
    dataframe['rsi_signal'] = (
        (dataframe['rsi'] > 30) & (dataframe['rsi'] < 70)
    ).astype(int)
    dataframe['signal_strength'] += dataframe['rsi_signal'] * 1

    # Trend alignment signal
    dataframe['trend_signal'] = (
        (dataframe['close'] > dataframe['ema50']) &
        (dataframe['trend_strength'] > 0)
    ).astype(int)
    dataframe['signal_strength'] += dataframe['trend_signal'] * 1

    # Money flow signal
    dataframe['money_flow_signal'] = (
        dataframe['money_flow_index'] > 50
    ).astype(int)
    dataframe['signal_strength'] += dataframe['money_flow_signal'] * 1

    return dataframe


# =============================================================================
#
# This class IS the audit tool. It is intentionally a sibling of the strategy
# (not a method on it) so it can be:
#
#   1. Imported and called from anywhere: `from ANF import ANFAudit`
#   2. Run standalone:   `python ANF_v2.py --audit`
#   3. Triggered from a running bot via a filesystem flag, with results
#      delivered to Telegram via self.dp.send_msg() in bot_loop_start.
#      for the trigger detection logic.
#
# Why not use Freqtrade's Telegram command system directly?
#   Freqtrade does NOT expose an API to register custom /commands from a
#   strategy. The command list in rpc/telegram.py is hard-coded. The cleanest
#   integration path is therefore a filesystem flag the user touches from
#   their shell or a Telegram bash one-liner. This keeps the strategy
#   forward-compatible with Freqtrade updates.
# =============================================================================

class ANFAudit:
    """Walk-forward audit for the ANF ML pipeline.

    Reports per-pair:
      - n_folds executed
      - in_sample_f1_mean      (average F1 across train splits)
      - out_of_sample_f1_mean  (average F1 on held-out test slices)
      - decay_pct              (1 - oos_f1/is_f1)
      - status                 (HEALTHY | DEGRADED | OVERFIT | NO_DATA)

    Workflow:
      1. Pre-compute the base indicators the engine reads (rsi, ha_close,
         minima_sort_threshold, maxima_sort_threshold). Skipping this would
         make extract_advanced_features fall to neutral fallbacks and the
         F1 numbers become meaningless.  
      2. Walk forward: train(2000) → embargo(48) → test(500) → step(500).
      3. Engine output is redirected to a tempdir via models_dir_override.
         Production models are never touched.  
    """

    TRAIN_WIN = 2000
    EMBARGO = 48
    TEST_WIN = 500
    STEP = 500

    DECAY_HEALTHY = 0.25
    DECAY_DEGRADED = 0.40

    def __init__(self, engine_cls=None, data_dir: Optional[Path] = None,
                 timeframe: str = '1h'):
        # Engine class to use; defaults to AdvancedPredictiveEngine in this module.
        self.engine_cls = engine_cls or AdvancedPredictiveEngine
        self.data_dir = Path(data_dir) if data_dir else Path("user_data/data/binance")
        self.timeframe = timeframe

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------
    def load_pair_dataframe(self, pair: str) -> Optional[pd.DataFrame]:
        """Load Freqtrade-format OHLCV. Tries feather first, then json."""
        safe = pair.replace('/', '_').replace(':', '_')
        candidates = [
            self.data_dir / 'futures' / f"{safe}-{self.timeframe}-futures.feather",
            self.data_dir / 'futures' / f"{safe}-{self.timeframe}-futures.json",
            self.data_dir / f"{safe}-{self.timeframe}.feather",
            self.data_dir / f"{safe}-{self.timeframe}.json",
        ]
        for p in candidates:
            if p.exists():
                try:
                    if p.suffix == '.feather':
                        return pd.read_feather(p)
                    return pd.read_json(p, orient='values')
                except Exception as e:
                    logger.debug(f"[audit] failed to read {p}: {e}")
        return None

    @staticmethod
    def normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if df.shape[1] == 6 and 0 in df.columns:
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        cols = {c.lower(): c for c in df.columns}
        df = df.rename(columns={cols[k]: k for k in ('open', 'high', 'low', 'close', 'volume')
                                if k in cols})
        if 'date' in cols:
            df = df.rename(columns={cols['date']: 'date'})
            df['date'] = pd.to_datetime(df['date'], unit='ms', errors='ignore')
        return df.reset_index(drop=True)

    @staticmethod
    def precompute_base_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Reproduce the small set of indicators the strategy adds in
        populate_indicators that AdvancedPredictiveEngine reads.

        Indicator pipeline overview:
          - rsi                       (TA-Lib RSI, default period 14)
          - ha_close                  ((O+H+L+C)/4)
          - minima_sort_threshold     (rolling min of close, window 10)
          - maxima_sort_threshold     (rolling max of close, window 10)
        """
        df = df.copy()
        try:
            import talib.abstract as ta
            df['rsi'] = ta.RSI(df['close'])
        except ImportError:
            delta = df['close'].diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / (loss + 1e-12)
            df['rsi'] = 100 - (100 / (1 + rs))
        df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['minima_sort_threshold'] = df['close'].rolling(window=10).min()
        df['maxima_sort_threshold'] = df['close'].rolling(window=10).max()
        return df

    # ------------------------------------------------------------------
    # Core walk-forward logic
    # ------------------------------------------------------------------
    def run_for_pair(self, pair: str, df: pd.DataFrame, tmpdir: Path) -> dict:
        """Run walk-forward for one pair. Returns a dict with keys:
        pair, n_folds, is_f1_mean, oos_f1_mean, decay_pct, status, errors."""
        try:
            from sklearn.metrics import f1_score
        except ImportError:
            return self._empty_result(pair, ["scikit-learn not available"])

        if df is None or df.empty:
            return self._empty_result(pair, ["no data"])
        if len(df) < self.TRAIN_WIN + self.EMBARGO + self.TEST_WIN:
            return self._empty_result(
                pair,
                [f"insufficient history ({len(df)} candles, "
                 f"need {self.TRAIN_WIN + self.EMBARGO + self.TEST_WIN})"],
            )

        df = self.precompute_base_indicators(df)

        pair_safe = pair.replace('/', '_').replace(':', '_')
        pair_tmp = tmpdir / pair_safe
        engine = self.engine_cls(
            config=None,
            strategy_name=f"_audit_{pair_safe}",
            models_dir_override=pair_tmp,
        )
        engine.backtest_mode = False
        engine.min_train_candles = self.TRAIN_WIN

        is_f1_list: list = []
        oos_f1_list: list = []
        errors: list = []
        n_folds = 0

        start = 0
        max_start = len(df) - self.TRAIN_WIN - self.EMBARGO - self.TEST_WIN
        while start <= max_start:
            train_df = df.iloc[start: start + self.TRAIN_WIN].copy().reset_index(drop=True)
            test_start = start + self.TRAIN_WIN + self.EMBARGO
            test_df = df.iloc[test_start: test_start + self.TEST_WIN].copy().reset_index(drop=True)

            try:
                tr = engine.train_predictive_models(train_df, pair)
                if tr.get('status') != 'success':
                    errors.append(f"train@{start}: {tr.get('status')}")
                    start += self.STEP
                    continue
                is_f1 = float(tr.get('best_f1_score', 0.0))

                preds = engine.predict_entry_probability(test_df, pair)
                y_test = engine.create_target_variable(test_df, forward_periods=5)
                n = min(len(preds), len(y_test))
                if n < 50:
                    errors.append(f"test@{start}: too few aligned samples ({n})")
                    start += self.STEP
                    continue
                y_true = y_test.iloc[:n].to_numpy()
                y_pred = (preds.iloc[:n].to_numpy() > 0.5).astype(int)
                mask = ~np.isnan(y_true) & ~np.isnan(y_pred.astype(float))
                if mask.sum() < 50:
                    errors.append(f"test@{start}: too few non-NaN samples")
                    start += self.STEP
                    continue
                oos_f1 = float(f1_score(y_true[mask].astype(int), y_pred[mask],
                                        zero_division=0))
                n_folds += 1
                is_f1_list.append(is_f1)
                oos_f1_list.append(oos_f1)
            except Exception as e:
                errors.append(f"fold@{start}: {e}")
            finally:
                start += self.STEP

        is_mean = float(np.mean(is_f1_list)) if is_f1_list else float('nan')
        oos_mean = float(np.mean(oos_f1_list)) if oos_f1_list else float('nan')
        if not np.isnan(is_mean) and is_mean > 0 and not np.isnan(oos_mean):
            decay = 1.0 - (oos_mean / is_mean)
        else:
            decay = float('nan')
        return {
            'pair': pair,
            'n_folds': n_folds,
            'is_f1_mean': is_mean,
            'oos_f1_mean': oos_mean,
            'decay_pct': decay,
            'status': self._status_for(decay),
            'errors': errors,
        }

    @staticmethod
    def _empty_result(pair: str, errors: list) -> dict:
        return {
            'pair': pair, 'n_folds': 0,
            'is_f1_mean': float('nan'), 'oos_f1_mean': float('nan'),
            'decay_pct': float('nan'), 'status': 'NO_DATA', 'errors': errors,
        }

    @classmethod
    def _status_for(cls, decay: float) -> str:
        if np.isnan(decay):
            return "NO_DATA"
        if decay < cls.DECAY_HEALTHY:
            return "HEALTHY"
        if decay < cls.DECAY_DEGRADED:
            return "DEGRADED"
        return "OVERFIT"

    # ------------------------------------------------------------------
    # Multi-pair driver
    # ------------------------------------------------------------------
    def run(self, pairs: list, progress_callback=None) -> list:
        """Audit a list of pairs. Returns a list of result dicts.

        progress_callback(i, total, pair, result_dict) is called after each
        pair finishes — useful for streaming progress to Telegram.
        """
        import tempfile
        results = []
        with tempfile.TemporaryDirectory(prefix='anf_audit_') as tmp:
            tmpdir = Path(tmp)
            for i, pair in enumerate(pairs, 1):
                raw = self.load_pair_dataframe(pair)
                df = self.normalise_ohlcv(raw) if raw is not None else None
                result = self.run_for_pair(pair, df, tmpdir)
                results.append(result)
                if progress_callback is not None:
                    try:
                        progress_callback(i, len(pairs), pair, result)
                    except Exception as e:
                        logger.debug(f"[audit] progress_callback error: {e}")
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @staticmethod
    def format_report(results: list) -> str:
        """Build a plain-text report. Returns a single string suitable for
        Telegram (under ~4000 chars for typical 15-pair whitelists)."""
        lines = []
        lines.append("```")
        lines.append("ANF walk-forward audit")
        lines.append("=" * 60)
        lines.append(f"{'Pair':<22}{'Folds':>6}{'IS_F1':>8}{'OOS_F1':>8}"
                     f"{'Decay':>8}  {'Status':<8}")
        lines.append("-" * 60)
        n_overfit = n_degraded = n_healthy = n_nodata = 0
        for r in sorted(results, key=lambda x: x['pair']):
            s = r['status']
            if s == "OVERFIT": n_overfit += 1
            elif s == "DEGRADED": n_degraded += 1
            elif s == "HEALTHY": n_healthy += 1
            else: n_nodata += 1
            decay_s = (f"{r['decay_pct']*100:>6.1f}%"
                       if not np.isnan(r['decay_pct']) else "  n/a ")
            is_s = (f"{r['is_f1_mean']:.3f}"
                    if not np.isnan(r['is_f1_mean']) else " n/a ")
            oos_s = (f"{r['oos_f1_mean']:.3f}"
                     if not np.isnan(r['oos_f1_mean']) else " n/a ")
            lines.append(f"{r['pair']:<22}{r['n_folds']:>6}{is_s:>8}"
                         f"{oos_s:>8}{decay_s:>8}  {s:<8}")
        lines.append("-" * 60)
        lines.append(f"Summary: {n_healthy} HEALTHY / {n_degraded} DEGRADED / "
                     f"{n_overfit} OVERFIT / {n_nodata} NO_DATA")
        lines.append("```")
        return "\n".join(lines)


# =============================================================================
# Telegram trigger integration (used by ANF.bot_loop_start).
#
# a single, strategy-name-based directory:
#
#     user_data/ml_models/<class_name>/
#         ├── <pair>_model_random_forest.pkl
#         ├── <pair>_model_gradient_boosting.pkl
#         ├── <pair>_scaler.pkl
#         ├── <pair>_metadata.pkl
#         ├── audit_trigger          (touch this to launch an audit)
#         ├── audit_last_result.txt  (formatted report of the last audit)
#         └── audit_last_result.json (machine-readable form of the last audit)
#
# Why this matters: the directory name is derived from `self.__class__.__name__`
# at bot_start time, so each strategy class gets its own directory automatically
# — no clobbering of another class's models, no shared audit state. ANF_v2 stores
# its ML models, its HMM regime model, and its audit artefacts under
# user_data/ml_models/ANF_v2/, completely isolated from an ANF instance running
# in parallel. The two can share a cloned trade database for dry-run comparison
# without touching each other's models.
#
# The two TRIGGER/RESULT paths below are module-level fallbacks for callers
# that don't have a strategy instance handy (e.g. the standalone CLI). The
# strategy itself overrides them with class-name-aware paths via
# self._audit_paths() at runtime.
# =============================================================================

ANF_DEFAULT_ROOT = Path("user_data/ml_models/ANF_v2")
ANF_AUDIT_TRIGGER_FILE = ANF_DEFAULT_ROOT / "audit_trigger"
ANF_AUDIT_RESULT_FILE = ANF_DEFAULT_ROOT / "audit_last_result.txt"


class ANF_v2(IStrategy):
    """
    EN: ANF's four-layer stack plus an HMM regime controller on top. The HMM
        classifies ranging/trending/volatile from the BTC anchor and modulates
        ensemble weighting and leverage; it never generates signals on its own.
    ES: La pila de cuatro capas de ANF más un controlador de régimen HMM encima.
        El HMM clasifica lateral/tendencia/volátil desde el ancla BTC y modula la
        ponderación del ensemble y el leverage; nunca genera señales por sí solo.

    EN: ML and HMM train only in live/dry-run; both are disabled in backtest, so
        only the technical tiers operate there. Without hmmlearn, behaves as ANF.

    EN: HMM training has two modes. (1) Default: the bot fits the HMM on a rolling
        window (FIT_LOOKBACK candles) and refits weekly. (2) Optional: train a
        model offline on a long history (years), save it with frozen=True, and
        drop the .pkl into user_data/ml_models/ANF_v2/. A frozen model is loaded
        on startup and never overwritten by the rolling-window fit. Delete the
        .pkl to return to the default rolling behaviour.
    ES: El entrenamiento del HMM tiene dos modos. (1) Por defecto: el bot ajusta
        el HMM sobre una ventana móvil (FIT_LOOKBACK velas) y reentrena cada
        semana. (2) Opcional: entrena un modelo offline con histórico largo
        (años), guárdalo con frozen=True y deja el .pkl en
        user_data/ml_models/ANF_v2/. Un modelo congelado se carga al arrancar y
        nunca se sobrescribe con el fit de ventana móvil. Borra el .pkl para
        volver al comportamiento por defecto.
    ES: ML y HMM solo entrenan en live/dry-run; ambos se desactivan en backtest,
        así que allí solo operan los tiers técnicos. Sin hmmlearn, se comporta como ANF.
    """

    # ==================== GENERAL CONFIG / CONFIGURACIÓN GENERAL ====================
    timeframe = "1h"

    # EN: startup_candle_count + 1 candles reach populate_indicators in live/dry-run.
    #     ML needs min_train_candles=1500, so 1600 = 1500 + 100 buffer (~67 days at 1h).
    # ES: startup_candle_count + 1 velas llegan a populate_indicators en live/dry-run.
    #     ML necesita min_train_candles=1500, así que 1600 = 1500 + 100 buffer (~67 días en 1h).
    startup_candle_count: int = 1600
    stoploss_on_exchange = False

    # Override in config via:
    #   "strategy_kwargs": {"btc_informative_pair": "BTC/USDT:USDT"}
    # or by editing this class attribute. Set to "" to disable BTC correlation.
    btc_informative_pair: str = "BTC/USDT:USDT"
    # Pairs included in informative_pairs() for cross-asset context.
    extra_informative_pairs: list = [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
    ]
    # Pairs that get verbose debug logging (kept small to avoid spam).
    debug_log_pairs: list = ["BTC/USDT:USDT", "ETH/USDT:USDT"]


    # Base stoploss (initial stop loss percentage)
    stoploss = -0.10  # 10% hard stop; custom_stoploss tightens this dynamically
    
    # Trailing stop configuration
    trailing_stop = True
    trailing_stop_positive = 0.005  # Start trailing at 0.5% profit (conservative)
    trailing_stop_positive_offset = 0.03  # Trigger trailing at 3% profit
    trailing_only_offset_is_reached = True

    use_custom_stoploss = True
    can_short = True
    use_exit_signal = True
    ignore_roi_if_entry_signal = False
    process_only_new_candles = True

    # EN: Diagnostic logging. When True, the strategy emits throttled INFO lines
    #     about HMM status, ML probability health, and why short tiers did/didn't
    #     fire. It NEVER changes any trading decision — purely visibility. Safe to
    #     run in live; throttling keeps log volume bounded. Set False to silence.
    # ES: Logging de diagnóstico. Con True, la estrategia emite líneas INFO
    #     (con throttle) sobre el estado del HMM, la salud de la probabilidad ML,
    #     y por qué los tiers short se activaron o no. NUNCA cambia una decisión
    #     de trading — solo visibilidad. Seguro en live; el throttle acota el
    #     volumen de log. Pon False para silenciarlo.
    diagnostic_logging = True
    use_custom_exits_advanced = True
    use_emergency_exits = True

    
    regime_change_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    regime_change_sensitivity = DecimalParameter(0.3, 0.8, default=0.5, decimals=2, space="sell", optimize=True, load=True)
    
    # Flash Move Detection
    flash_move_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    flash_move_threshold = DecimalParameter(0.03, 0.08, default=0.05, decimals=3, space="sell", optimize=True, load=True)
    flash_move_candles = IntParameter(3, 10, default=5, space="sell", optimize=True, load=True)
    
    # Volume Spike Detection
    volume_spike_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    volume_spike_multiplier = DecimalParameter(2.0, 5.0, default=3.0, decimals=1, space="sell", optimize=True, load=True)
    
    # Emergency Exit Protection
    emergency_exit_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    emergency_exit_profit_threshold = DecimalParameter(0.005, 0.03, default=0.015, decimals=3, space="sell", optimize=True, load=True)
    
    # Trailing Stop Exit Control (NEW: Fix for "Blocking trailing stop exit")
    trailing_exit_min_profit = DecimalParameter(-0.03, 0.02, default=-0.002, decimals=3,
                                               space="sell", optimize=True, load=True)
    allow_trailing_exit_when_negative = BooleanParameter(default=True, space="sell",
                                                         optimize=False, load=True)
    
    # Market Sentiment Protection
    sentiment_protection_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    sentiment_shift_threshold = DecimalParameter(0.2, 0.4, default=0.3, decimals=2, space="sell", optimize=True, load=True)

    # 🔧ATR STOPLOSS PARAMETERS (Anpassbar machen)
    atr_stoploss_multiplier = DecimalParameter(0.8, 2.0, default=1.0, decimals=1, space="sell", optimize=True, load=True)
    atr_stoploss_minimum = DecimalParameter(-0.25, -0.10, default=-0.12, decimals=2, space="sell", optimize=True, load=True)
    atr_stoploss_maximum = DecimalParameter(-0.30, -0.15, default=-0.18, decimals=2, space="sell", optimize=True, load=True)
    atr_stoploss_ceiling = DecimalParameter(-0.10, -0.06, default=-0.06, decimals=2, space="sell", optimize=True, load=True)
    # DCA parameters
    # has no `adjust_trade_position()` method, so position_adjustment_enable in
    # config is also moot. Kept here commented for documentation; if you ever
    # add DCA, uncomment AND implement adjust_trade_position(), AND set
    # `position_adjustment_enable: true` in config.
    #
    # initial_safety_order_trigger = DecimalParameter(
    #     low=-0.02, high=-0.01, default=-0.018, decimals=3, space="buy", optimize=True, load=True
    # )
    # max_safety_orders = IntParameter(1, 3, default=1, space="buy", optimize=True, load=True)
    # safety_order_step_scale = DecimalParameter(
    #     low=1.05, high=1.5, default=1.25, decimals=2, space="buy", optimize=True, load=True
    # )
    # safety_order_volume_scale = DecimalParameter(
    #     low=1.1, high=2.0, default=1.4, decimals=1, space="buy", optimize=True, load=True
    # )
    h2 = IntParameter(20, 60, default=40, space="buy", optimize=True, load=True)
    h1 = IntParameter(10, 40, default=20, space="buy", optimize=True, load=True)
    h0 = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)
    cp = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)

    # Entry parameters
    increment_for_unique_price = DecimalParameter(
        low=1.0005, high=1.002, default=1.001, decimals=4, space="buy", optimize=True, load=True
    )
    last_entry_price: Optional[float] = None

    # Protection parameters
    cooldown_lookback = IntParameter(2, 48, default=2, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=12, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # Murrey Math level parameters
    mml_const1 = DecimalParameter(1.0, 1.1, default=1.0699, decimals=4, space="buy", optimize=True, load=True)
    mml_const2 = DecimalParameter(0.99, 1.0, default=0.99875, decimals=5, space="buy", optimize=True, load=True)


    # Dynamic Stoploss parameters
    # Add these parameters
    stoploss_atr_multiplier = DecimalParameter(1.0, 3.0, default=1.5, space="sell", optimize=True)
    stoploss_max_reasonable = DecimalParameter(-0.30, -0.15, default=-0.20, space="sell", optimize=True)

    # === Hyperopt Parameters ===
    dominance_threshold = IntParameter(1, 10, default=3, space="buy", optimize=True)
    tightness_factor = DecimalParameter(0.5, 2.0, default=1.0, space="buy", optimize=True)
    long_rsi_threshold = IntParameter(50, 65, default=50, space="buy", optimize=True)
    short_rsi_threshold = IntParameter(30, 45, default=35, space="sell", optimize=True)

    # Dynamic Leverage parameters
    leverage_window_size = IntParameter(20, 100, default=70, space="buy", optimize=True, load=True)
    leverage_base = DecimalParameter(1.0, 5.0, default=2.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_rsi_low = DecimalParameter(20.0, 40.0, default=30.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_rsi_high = DecimalParameter(60.0, 80.0, default=70.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_long_increase_factor = DecimalParameter(1.1, 2.0, default=1.5, decimals=1, space="buy", optimize=True,
                                                     load=True)
    leverage_long_decrease_factor = DecimalParameter(0.3, 0.9, default=0.5, decimals=1, space="buy", optimize=True,
                                                     load=True)
    leverage_volatility_decrease_factor = DecimalParameter(0.5, 0.95, default=0.8, decimals=2, space="buy",
                                                           optimize=True, load=True)
    leverage_atr_threshold_pct = DecimalParameter(0.01, 0.05, default=0.03, decimals=3, space="buy", optimize=True,
                                                  load=True)

    # Indicator parameters
    indicator_extrema_order = IntParameter(3, 15, default=8, space="buy", optimize=True, load=True)  # War 5
    indicator_mml_window = IntParameter(50, 200, default=50, space="buy", optimize=True, load=True)  # War 50
    indicator_rolling_window_threshold = IntParameter(20, 100, default=50, space="buy", optimize=True, load=True)  # War 20
    indicator_rolling_check_window = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)  # War 5


    
    # Market breadth parameters
    market_breadth_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    market_breadth_threshold = DecimalParameter(0.3, 0.6, default=0.45, space="buy", optimize=True)
    
    # Total market cap parameters
    total_mcap_filter_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    total_mcap_ma_period = IntParameter(20, 100, default=50, space="buy", optimize=True)
    
    # Market regime parameters
    regime_filter_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    regime_lookback_period = IntParameter(24, 168, default=48, space="buy", optimize=True)  # hours
    
    # Fear & Greed parameters
    fear_greed_enabled = BooleanParameter(default=False, space="buy", optimize=True)  # Optional
    fear_greed_extreme_threshold = IntParameter(20, 30, default=25, space="buy", optimize=True)
    fear_greed_greed_threshold = IntParameter(70, 80, default=75, space="buy", optimize=True)
    # Momentum
    avoid_strong_trends = BooleanParameter(default=True, space="buy", optimize=True)
    trend_strength_threshold = DecimalParameter(0.01, 0.05, default=0.02, space="buy", optimize=True)
    momentum_confirmation_candles = IntParameter(1, 5, default=2, space="buy", optimize=True)

    # Dynamic exit based on entry quality
    dynamic_exit_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    exit_on_confluence_loss = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    exit_on_structure_break = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    
    # Profit target multipliers based on entry type
    high_quality_profit_multiplier = DecimalParameter(1.2, 3.0, default=2.0, space="sell", optimize=True, load=True)
    medium_quality_profit_multiplier = DecimalParameter(1.0, 2.5, default=1.5, space="sell", optimize=True, load=True)
    backup_profit_multiplier = DecimalParameter(0.8, 2.0, default=1.2, space="sell", optimize=True, load=True)
    
    # Advanced exit thresholds
    volume_decline_exit_threshold = DecimalParameter(0.3, 0.8, default=0.5, space="sell", optimize=True, load=True)
    momentum_decline_exit_threshold = IntParameter(1, 4, default=2, space="sell", optimize=True, load=True)
    structure_deterioration_threshold = DecimalParameter(-3.0, 0.0, default=-1.5, space="sell", optimize=True, load=True)
    
    # RSI exit levels
    rsi_overbought_exit = IntParameter(70, 85, default=75, space="sell", optimize=True, load=True)
    rsi_divergence_exit_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    
    # Trailing stop improvements
    use_advanced_trailing = BooleanParameter(default=False, space="sell", optimize=False, load=True)
    trailing_stop_positive_offset_high_quality = DecimalParameter(0.02, 0.08, default=0.04, space="sell", optimize=True, load=True)
    trailing_stop_positive_offset_medium_quality = DecimalParameter(0.015, 0.06, default=0.03, space="sell", optimize=True, load=True)
    
    # === NEUE ADVANCED PARAMETERS ===
    # Confluence Analysis
    confluence_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    confluence_threshold = DecimalParameter(2.0, 4.0, default=2.5, space="buy", optimize=True, load=True)  # War 3.0
    
    # Volume Analysis
    volume_analysis_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    volume_strength_threshold = DecimalParameter(1.1, 2.0, default=1.3, space="buy", optimize=True, load=True)
    volume_pressure_threshold = IntParameter(1, 3, default=1, space="buy", optimize=True, load=True)  # War 2

    
    # Momentum Analysis
    momentum_analysis_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    momentum_quality_threshold = IntParameter(2, 4, default=2, space="buy", optimize=True, load=True)  # War 3
    
    # Market Structure Analysis
    structure_analysis_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    structure_score_threshold = DecimalParameter(-2.0, 5.0, default=0.5, space="buy", optimize=True, load=True)
    
    # Ultimate Score
    ultimate_score_threshold = DecimalParameter(0.5, 3.0, default=1.5, space="buy", optimize=True, load=True)
    
    # Advanced Entry Filters
    require_volume_confirmation = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    require_momentum_confirmation = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    require_structure_confirmation = BooleanParameter(default=True, space="buy", optimize=False, load=True)

    # ✅ Replace your old ROI with this:
    minimal_roi = {
        "0": 0.10,      # Don't exit immediately at 6% — let the trend run
        "10": 0.07,
        "20": 0.05,
        "40": 0.035,
        "80": 0.025,
        "160": 0.015,
        "240": 0.01,
    }

    # Plot configuration for backtesting UI
    plot_config = {
        "main_plot": {
            # Trend indicators
            "ema50": {"color": "gray", "type": "line"},
            
            # Support/Resistance
            "minima_sort_threshold": {"color": "#4ae747", "type": "line"},
            "maxima_sort_threshold": {"color": "#5b5e4b", "type": "line"},
        },
        "subplots": {
            "extrema_analysis": {
                "s_extrema": {"color": "#f53580", "type": "line"},
                "maxima": {"color": "#a29db9", "type": "scatter"},
                "minima": {"color": "#aac7fc", "type": "scatter"},
            },
            "murrey_math_levels": {
                "[4/8]P": {"color": "blue", "type": "line"},        # 50% MML
                "[6/8]P": {"color": "green", "type": "line"},       # 75% MML
                "[2/8]P": {"color": "orange", "type": "line"},      # 25% MML
                "[8/8]P": {"color": "red", "type": "line"},         # 100% MML
                "[0/8]P": {"color": "red", "type": "line"},         # 0% MML
                "mmlextreme_oscillator": {"color": "purple", "type": "line"},
            },
            "rsi_analysis": {
                "rsi": {"color": "purple", "type": "line"},
                "rsi_divergence_bull": {"color": "green", "type": "scatter"},
                "rsi_divergence_bear": {"color": "red", "type": "scatter"},
            },
            "confluence_analysis": {
                "confluence_score": {"color": "gold", "type": "line"},
                "near_support": {"color": "green", "type": "scatter"},
                "near_resistance": {"color": "red", "type": "scatter"},
                "near_mml": {"color": "blue", "type": "line"},
                "volume_spike": {"color": "orange", "type": "scatter"},
            },
            "volume_analysis": {
                "volume_strength": {"color": "cyan", "type": "line"},
                "volume_pressure": {"color": "magenta", "type": "line"},
                "buying_pressure": {"color": "green", "type": "line"},
                "selling_pressure": {"color": "red", "type": "line"},
                "money_flow_index": {"color": "yellow", "type": "line"},
            },
            "momentum_analysis": {
                "momentum_quality": {"color": "brown", "type": "line"},
                "momentum_acceleration": {"color": "pink", "type": "line"},
                "momentum_consistency": {"color": "lime", "type": "line"},
                "momentum_oscillator": {"color": "navy", "type": "line"},
            },
            "structure_analysis": {
                "structure_score": {"color": "teal", "type": "line"},
                "bullish_structure": {"color": "green", "type": "line"},
                "bearish_structure": {"color": "red", "type": "line"},
                "structure_break_up": {"color": "lime", "type": "scatter"},
                "structure_break_down": {"color": "crimson", "type": "scatter"},
            },
            "trend_strength": {
                "trend_strength": {"color": "indigo", "type": "line"},
                "trend_strength_5": {"color": "lightblue", "type": "line"},
                "trend_strength_10": {"color": "mediumblue", "type": "line"},
                "trend_strength_20": {"color": "darkblue", "type": "line"},
            },
            "ultimate_signals": {
                "ultimate_score": {"color": "gold", "type": "line"},
                "signal_strength": {"color": "silver", "type": "line"},
                "high_quality_setup": {"color": "lime", "type": "scatter"},
                "entry_type": {"color": "white", "type": "line"},
            },
            "market_conditions": {
                "strong_uptrend": {"color": "green", "type": "scatter"},
                "strong_downtrend": {"color": "red", "type": "scatter"},
                "ranging": {"color": "yellow", "type": "scatter"},
                "strong_up_momentum": {"color": "lime", "type": "scatter"},
                "strong_down_momentum": {"color": "crimson", "type": "scatter"},
            },
            "di_analysis": {
                "DI_values": {"color": "orange", "type": "line"},
                "DI_catch": {"color": "red", "type": "scatter"},
                "plus_di": {"color": "green", "type": "line"},
                "minus_di": {"color": "red", "type": "line"},
            }
        },
    }

    # ==========================================================================
    # ==========================================================================
    def bot_start(self, **kwargs) -> None:
        """
        Initialise per-strategy state once when the bot starts.

        We bind the module-level engine to the instance so that both the
        module-level and instance access patterns resolve to the same engine,
        and the startup `mark_trained_if_assets()` optimization runs.

        Detect backtest/hyperopt runmode and DISABLE ML training
        in those modes. Rationale: in backtest, populate_indicators is called
        once per pair with the full dataset, so training-then-inference creates
        massive look-ahead bias (model knows the future of every early candle).
        With ML disabled, ml_entry_probability stays at 0.5 and Tier 1 ML Ultra
        never fires — only T2/T3/T4 (confluence, technical, MML mean-reversion)
        produce signals. Backtest results are then honest but more conservative.
        """
        try:
            # store under user_data/ml_models/<ClassName>/, preventing collisions
            # between forks of this strategy.
            engine = get_engine(
                config=self.config if hasattr(self, "config") else None,
                strategy_name=self.__class__.__name__,
            )
            self.predictive_engine = engine  # instance attribute, used by populate_indicators
            self._last_ai_state: dict = {}    # per-pair AI state cache (bounded use below)

            runmode_raw = ""
            try:
                rm = self.config.get('runmode')
                runmode_raw = str(rm.value if hasattr(rm, 'value') else rm).lower()
            except Exception:
                runmode_raw = ""
            is_backtest_or_hyperopt = any(
                tok in runmode_raw for tok in ('backtest', 'hyperopt', 'edge')
            )
            engine.backtest_mode = is_backtest_or_hyperopt
            if is_backtest_or_hyperopt:
                logger.warning(
                    "[ANF_v2] BACKTEST/HYPEROPT detected (runmode=%s). ML training "
                    "is DISABLED to prevent look-ahead bias. ml_entry_probability "
                    "will stay at 0.5 — Tier 1 ML Ultra entries will not fire. "
                    "T2/T3/T4 (confluence/technical/MML) work normally.",
                    runmode_raw,
                )
            else:
                logger.info(f"[ANF_v2] bot_start: engine attached, runmode={runmode_raw}, "
                           f"ML training enabled")
        except Exception as e:
            logger.warning(f"[ANF_v2] bot_start: failed to initialise engine ({e}); "
                           "falling back to lazy init inside populate_indicators")
            self.predictive_engine = None
            self._last_ai_state = {}

        # =====================================================================
        # user_data/ml_models/<ClassName>/. Audit trigger and result files
        # live here instead of user_data/. Forks of this strategy with a
        # different class name automatically get an isolated directory tree.
        # =====================================================================
        self._anf_root: Path = Path("user_data/ml_models") / self.__class__.__name__
        self._anf_root.mkdir(parents=True, exist_ok=True)
        self._audit_trigger_path: Path = self._anf_root / "audit_trigger"
        self._audit_result_txt: Path = self._anf_root / "audit_last_result.txt"
        self._audit_result_json: Path = self._anf_root / "audit_last_result.json"
        logger.info(f"[ANF_v2] ANF root dir: {self._anf_root.resolve()}")
        logger.info(f"[ANF_v2] Audit trigger watching: {self._audit_trigger_path}")

        # =====================================================================
        # Many users paste their Telegram token directly into the config and
        # commit it to git. Detect a likely hardcoded token and warn — without
        # disabling the bot. We do NOT log the token itself, only its prefix.
        # =====================================================================
        try:
            tg_cfg = (self.config or {}).get('telegram', {}) if hasattr(self, 'config') else {}
            tok = tg_cfg.get('token', '') or ''
            if tok and len(tok) > 30 and ':' in tok:
                # Likely a real BotFather token. Telegram tokens look like:
                # "5037236576:AAGYru70j..." — 8-10 digit prefix + colon + 35 chars.
                logger.warning(
                    "[SECURITY] Telegram token appears hardcoded in config "
                    "(prefix=%s...). Anyone with read access to this file or "
                    "the chat history where you shared it can control your bot. "
                    "Recommended: revoke via @BotFather /revoke, then use env vars: "
                    "export FREQTRADE__TELEGRAM__TOKEN='new_token' and leave "
                    "config.telegram.token empty.",
                    tok.split(':', 1)[0],
                )
        except Exception:
            pass  # security warning is best-effort; never fatal

        # =====================================================================
        # ANF_v2: HMM regime detector bootstrap.
        # We do NOT train here: in bot_start the dataprovider may not yet have
        # candles, and training needs ~2000 of them. Instead we set a flag and
        # train lazily on the first populate_indicators call for the BTC
        # reference pair (which always has the deepest history). The trained
        # model is then persisted under user_data/ml_models/<ClassName>/ and
        # reused across restarts. See _maybe_train_hmm().
        #
        # Rationale for BTC as the regime anchor: market regime is a
        # market-wide latent property. BTC is the dominant driver of crypto
        # beta, so a single HMM fitted on BTC returns/vol generalises to the
        # whole whitelist far better than 15 per-pair HMMs would (and with a
        # fraction of the overfit risk and compute).
        # =====================================================================
        self._hmm_regime_label: str = 'unknown'
        self._hmm_regime_probs: Optional[dict] = None
        self._hmm_anchor_pair: str = getattr(self, 'btc_informative_pair', '') \
            or 'BTC/USDT:USDT'
        if HMM_AVAILABLE:
            logger.info(
                f"[HMM] regime detection ENABLED (anchor={self._hmm_anchor_pair}).")
            # in the whitelist. If BTC is not traded, _maybe_train_hmm (which
            # only fires for the anchor pair in populate_indicators) would never
            # run and the HMM would stay unfitted forever, silently using the
            # legacy detector. We detect that here and still attempt a bot_start
            # pre-train from cached data so the HMM works regardless.
            try:
                whitelist = []
                if getattr(self, 'dp', None) is not None:
                    try:
                        whitelist = list(self.dp.current_whitelist())
                    except Exception:
                        whitelist = []
                if whitelist and self._hmm_anchor_pair not in whitelist:
                    logger.warning(
                        f"[HMM] anchor pair {self._hmm_anchor_pair} is NOT in the "
                        f"whitelist. The HMM will be pre-trained from cached data "
                        f"at startup and refreshed via informative data; it will "
                        f"NOT retrain through populate_indicators. Add the anchor "
                        f"to your whitelist (or informative_pairs) for periodic "
                        f"retrains.")
            except Exception:
                pass

            # Attempt to PRE-TRAIN the HMM here so all pairs use the HMM
            # from the very first candle (avoids the first-iteration
            # inconsistency where pairs processed before BTC use the legacy
            # detector while pairs after it use the HMM). Best-effort: in
            # backtest, when dp is unavailable, or when there isn't enough
            # cached history, we silently defer to lazy training.
            try:
                engine = getattr(self, 'predictive_engine', None)
                in_backtest = bool(engine and getattr(engine, 'backtest_mode', False))
                if engine is not None and getattr(engine, 'hmm_detector', None) is not None \
                        and not in_backtest and getattr(self, 'dp', None) is not None:
                    detector = engine.hmm_detector
                    if not detector.is_fitted:
                        tf = getattr(self, 'timeframe', '1h')
                        btc_df = None
                        try:
                            btc_df = self.dp.get_pair_dataframe(self._hmm_anchor_pair, tf)
                        except Exception:
                            btc_df = None
                        if btc_df is not None and len(btc_df) >= detector.MIN_OBS:
                            if detector.fit(btc_df):
                                detector.save(engine._hmm_model_path)
                                logger.info(
                                    "[HMM] pre-trained at bot_start on "
                                    f"{self._hmm_anchor_pair} ({len(btc_df)} candles); "
                                    "all pairs use HMM from first candle.")
                        else:
                            logger.info(
                                "[HMM] insufficient cached data at bot_start; "
                                "will train lazily on first anchor candle.")
            except Exception as e:
                logger.debug(f"[HMM] bot_start pre-train deferred ({e})")
        else:
            logger.warning(
                "[HMM] hmmlearn unavailable — ANF_v2 runs with the legacy "
                "threshold regime detector. Install hmmlearn to enable HMM.")

    def _maybe_train_hmm(self, dataframe: pd.DataFrame, pair: str) -> None:
        """ANF_v2: lazily fit / refresh the global HMM regime model.

        Called from populate_indicators. Only acts when:
          - hmmlearn is available, AND
          - this is the anchor pair (BTC by default), AND
          - the engine exists, AND
          - the model is unfitted OR a weekly retrain is due.

        Exception-safe: failures are logged and leave the legacy detector in
        charge. Never raises into populate_indicators.
        """
        if not HMM_AVAILABLE:
            return
        try:
            engine = getattr(self, 'predictive_engine', None)
            if engine is None or getattr(engine, 'hmm_detector', None) is None:
                return
            # Only the anchor pair drives the global regime model.
            if pair != self._hmm_anchor_pair:
                return
            # In backtest we skip HMM training for the same look-ahead reason
            # the ML engine is disabled: fitting on the full backtest series
            # then decoding per-candle would leak regime info backwards.
            if getattr(engine, 'backtest_mode', False):
                return
            detector = engine.hmm_detector
            now = datetime.now(timezone.utc)
            if detector.is_fitted and not detector.needs_retrain(now):
                return
            if len(dataframe) < detector.MIN_OBS:
                return  # wait for more candles
            ok = detector.fit(dataframe, now=now)
            if ok:
                detector.save(engine._hmm_model_path)
                logger.info(
                    f"[HMM] regime model trained on {pair} "
                    f"({len(dataframe)} candles) and persisted.")
        except Exception as e:
            logger.debug(f"[HMM] lazy training skipped ({e})")

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """Periodic housekeeping + decision-context persistence.

        Persist decision context to Trade.custom_data. Freqtrade does NOT
        auto-persist dataframe columns to the trades table, so the `decision_*`
        columns written in populate_entry_trend are copied here via
        `trade.set_custom_data()` to make post-mortem analysis possible.

        Strategy:
          1. For every open trade we haven't yet annotated, look up the entry
             candle in the analyzed dataframe.
          2. Read the decision_* columns we already wrote in populate_entry_trend.
          3. Call trade.set_custom_data() — Freqtrade serialises to BD.
          4. Track which trades are done via a 'decision_logged' flag inside
             custom_data itself, so we don't redo work on every bot loop.
        """
        # HMM version-mismatch notice (once per bot start).
        # EN: If the frozen HMM .pkl was trained under different library versions
        #     than this VPS, load() recorded the mismatches. We surface them once
        #     to Telegram here (bot_loop_start guarantees self.dp is ready),
        #     complementing the log warning emitted at load time. Pickle
        #     compatibility across versions is not guaranteed, so the operator
        #     should align versions or retrain on this machine if regime
        #     detection misbehaves.
        # ES: Si el .pkl congelado del HMM se entrenó con versiones de librerías
        #     distintas a las de este VPS, load() guardó los desajustes. Aquí los
        #     avisamos una vez por Telegram (en bot_loop_start self.dp ya está
        #     listo), además del warning de log al cargar. La compatibilidad de
        #     pickle entre versiones no está garantizada, así que conviene alinear
        #     versiones o reentrenar en esta máquina si el régimen se comporta mal.
        try:
            if not getattr(self, '_hmm_version_notified', False):
                detector = getattr(self, 'hmm_detector', None)
                mismatches = getattr(detector, 'version_mismatches', []) if detector else []
                if mismatches:
                    self._send_telegram_safe(
                        "⚠️ ANF_v2 HMM: el modelo congelado se entrenó con "
                        "versiones de librerías distintas a las de este entorno. "
                        "La compatibilidad de pickle no está garantizada; si la "
                        "detección de régimen falla, reentrena aquí o alinea "
                        "versiones.\nDesajustes: " + "; ".join(mismatches))
                self._hmm_version_notified = True
        except Exception as e:
            logger.debug(f"[HMM] version-notice skipped: {e}")

        # House-keeping: prune AI-state cache so it can't grow unbounded.
        try:
            if hasattr(self, "_last_ai_state") and len(self._last_ai_state) > 256:
                items = list(self._last_ai_state.items())[-128:]
                self._last_ai_state = dict(items)
        except Exception:
            pass

        # Decision-context persistence.
        try:
            open_trades = Trade.get_open_trades()
            for trade in open_trades:
                # Skip if already annotated.
                if trade.get_custom_data('decision_logged'):
                    continue

                df, _ = self.dp.get_analyzed_dataframe(
                    pair=trade.pair, timeframe=self.timeframe)
                if df is None or df.empty:
                    continue

                # Find the entry candle. trade.open_date_utc is the open
                # timestamp; the entry signal candle is typically the one
                # before (since Freqtrade enters on the next candle's open).
                # We use the candle whose `date` is <= open_date_utc, taking
                # the latest such candle.
                if 'date' not in df.columns:
                    continue
                # Be tolerant on timezone: trade.open_date_utc may be naive
                # or tz-aware depending on Freqtrade version.
                try:
                    open_ts = pd.Timestamp(trade.open_date_utc)
                    if open_ts.tzinfo is None:
                        open_ts = open_ts.tz_localize('UTC')
                    df_dates = pd.to_datetime(df['date'], utc=True, errors='coerce')
                    eligible = df[df_dates <= open_ts]
                    if eligible.empty:
                        continue
                    entry_row = eligible.iloc[-1]
                except Exception:
                    # Fallback: use most recent row.
                    entry_row = df.iloc[-1]

                # Collect available decision_* fields. Fail-soft on any missing.
                payload = {}
                for k in ('decision_ml_prob', 'decision_ml_enh',
                          'decision_ultimate', 'decision_signal_strength',
                          'decision_rsi', 'decision_dist_support_pct'):
                    v = entry_row.get(k)
                    if v is not None and not pd.isna(v):
                        try:
                            payload[k] = float(v)
                        except (ValueError, TypeError):
                            payload[k] = None
                if payload:
                    # Add entry_tag for traceability (already in BD but
                    # convenient to have alongside decision metrics).
                    payload['entry_tag'] = trade.enter_tag or ''
                    try:
                        trade.set_custom_data(key='decision_context', value=payload)
                        trade.set_custom_data(key='decision_logged', value=True)
                        logger.debug(
                            f"[ANF_v2] decision context logged for trade #{trade.id} "
                            f"{trade.pair}: {payload}")
                    except Exception as e:
                        logger.debug(
                            f"[ANF_v2] could not persist decision_context "
                            f"for trade #{trade.id}: {e}")
        except Exception as e:
            # Decision logging must never break the bot loop.
            logger.debug(f"[ANF_v2] decision-context persistence skipped: {e}")

        # ============================================================
        #
        #   - Trigger path is now self._audit_trigger_path (under
        #     user_data/ml_models/<ClassName>/) instead of a hardcoded
        #     user_data/anf_audit_trigger. A fork gets its own trigger.
        #   - Atomic unlink-FIRST: the file is removed BEFORE we even
        #     decide what to do with the request. This prevents the
        #     loop iterations re-firing the audit before it had a chance
        #     to start. The unlink failure is explicitly tolerated.
        #   - Audit now runs in a separate OS PROCESS via subprocess
        #     A thread would share the GIL and CPU with the
        #     trading loop. A subprocess is fully isolated: nothing the
        #     audit does can slow down trade execution.
        # ============================================================
        try:
            trigger = getattr(self, '_audit_trigger_path', None)
            if trigger is not None and trigger.exists():
                # Step 1: unlink FIRST and atomically. If this fails, abort —
                # we'd risk an infinite re-trigger loop otherwise.
                unlinked_ok = True
                try:
                    trigger.unlink()
                except FileNotFoundError:
                    unlinked_ok = False  # race with another worker, fine
                except Exception as e:
                    unlinked_ok = False
                    logger.warning(
                        f"[ANF_v2] could not delete audit trigger {trigger}: {e}; "
                        f"skipping this iteration to avoid runaway audits")
                if unlinked_ok:
                    # Step 2: spawn audit in a separate OS process.
                    self._spawn_audit_subprocess()
        except Exception as e:
            logger.debug(f"[ANF_v2] audit trigger check skipped: {e}")

    # ==========================================================================
    # ==========================================================================
    def _send_telegram_safe(self, text: str) -> None:
        """Send a message via Freqtrade's dataprovider. Truncates long messages
        to fit Telegram's ~4096 char limit. Never raises."""
        try:
            if not hasattr(self, 'dp') or self.dp is None:
                return
            # Telegram hard limit is 4096; leave headroom for prefix.
            if len(text) > 3900:
                text = text[:3800] + "\n... (truncated)"
            self.dp.send_msg(text)
        except Exception as e:
            logger.debug(f"[ANF_v2] send_msg skipped: {e}")

    def _spawn_audit_subprocess(self) -> None:
        """launch the audit as a fully isolated OS
        process via subprocess.Popen. Replaces an earlier daemon-thread
        approach which, although bounded by `n_jobs=1` and GIL-releasing
        numpy ops, could still steal cycles from the trading loop on
        single-core hosts.

        We DO NOT block waiting for the subprocess; we DO NOT capture
        stdout/stderr (those go to the strategy.log via Freqtrade if you
        configure logging accordingly). The audit subprocess writes its
        own results to disk under self._anf_root and we let Telegram
        notification happen from THIS process by spawning a tiny watcher
        thread that polls for the result file and sends it when ready.
        """
        import subprocess, sys, threading, time

        # Locate self file (the strategy module) — we re-launch it with
        # --audit. This works because of the __main__ block at the end.
        try:
            import inspect
            strategy_path = Path(inspect.getfile(self.__class__))
        except Exception as e:
            logger.warning(f"[ANF_v2] could not locate strategy file for audit: {e}")
            self._send_telegram_safe(
                "❌ ANF audit aborted: could not locate strategy file.")
            return

        # Build the config path (used by the standalone CLI to read whitelist).
        cfg_path = None
        try:
            cfg_path = self.config.get('config_files', [None])[0] if hasattr(self, 'config') else None
        except Exception:
            cfg_path = None
        if not cfg_path:
            cfg_path = "user_data/config_ANF_v2.json"

        cmd = [
            sys.executable, str(strategy_path),
            "--audit",
            "--config", str(cfg_path),
            "--timeframe", self.timeframe,
            "--result-dir", str(self._anf_root),
        ]
        logger.info(f"[ANF_v2] launching audit subprocess: {' '.join(cmd)}")
        self._send_telegram_safe(
            f"🔍 ANF audit launched as separate process (PID will follow). "
            f"Trading loop is NOT blocked.")

        try:
            # start_new_session=True detaches from parent's process group on
            # POSIX, so if Freqtrade is killed the audit keeps running (or
            # is also reaped, depending on your service manager). Either is
            # fine — the audit is idempotent.
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            logger.info(f"[ANF_v2] audit subprocess PID={proc.pid}")
            self._send_telegram_safe(f"🔍 ANF audit PID={proc.pid}. "
                                     f"Results will appear in this chat when ready.")
        except Exception as e:
            logger.warning(f"[ANF_v2] failed to launch audit subprocess: {e}")
            self._send_telegram_safe(f"❌ ANF audit launch failed: {e}")
            return

        # Spawn a tiny watcher thread (this IS fine for a thread — it's just
        # filesystem polling, no CPU work) that notifies Telegram when the
        # subprocess writes its result file.
        def _watch_result():
            target = self._audit_result_txt
            # Wait up to 2h for the audit to finish.
            deadline = time.time() + 7200
            # Snapshot mtime to detect a new write.
            mtime_initial = target.stat().st_mtime if target.exists() else 0
            while time.time() < deadline:
                if proc.poll() is not None:
                    # Subprocess ended; check result file
                    if target.exists() and target.stat().st_mtime > mtime_initial:
                        try:
                            report = target.read_text()
                            self._send_telegram_safe(report)
                        except Exception as e:
                            self._send_telegram_safe(
                                f"⚠️ ANF audit finished but result file unreadable: {e}")
                    else:
                        self._send_telegram_safe(
                            f"⚠️ ANF audit subprocess exited "
                            f"(code={proc.returncode}) but produced no result file. "
                            f"Check logs.")
                    return
                time.sleep(5)
            # Timed out
            self._send_telegram_safe(
                "⚠️ ANF audit timed out after 2h — subprocess still running?")

        threading.Thread(target=_watch_result, name="ANFAuditWatcher",
                         daemon=True).start()

    # Helper method to check if we have an active position in the opposite direction
    def has_active_trade(self, pair: str, side: str) -> bool:
        """
        Check if there's an active trade in the specified direction
        """
        try:
            trades = Trade.get_open_trades()
            for trade in trades:
                if trade.pair == pair:
                    if side == "long" and not trade.is_short:
                        return True
                    elif side == "short" and trade.is_short:
                        return True
        except Exception as e:
            logger.warning(f"Error checking active trades for {pair}: {e}")
        return False

    @staticmethod
    def _calculate_mml_core(mn: float, finalH: float, mx: float, finalL: float,
                            mml_c1: float, mml_c2: float) -> Dict[str, float]:
        dmml_calc = ((finalH - finalL) / 8.0) * mml_c1
        if dmml_calc == 0 or np.isinf(dmml_calc) or np.isnan(dmml_calc) or finalH == finalL:
            return {key: finalL for key in MML_LEVEL_NAMES}
        mml_val = (mx * mml_c2) + (dmml_calc * 3)
        if np.isinf(mml_val) or np.isnan(mml_val):
            return {key: finalL for key in MML_LEVEL_NAMES}
        ml = [mml_val - (dmml_calc * i) for i in range(16)]
        return {
            "[-3/8]P": ml[14], "[-2/8]P": ml[13], "[-1/8]P": ml[12],
            "[0/8]P": ml[11], "[1/8]P": ml[10], "[2/8]P": ml[9],
            "[3/8]P": ml[8], "[4/8]P": ml[7], "[5/8]P": ml[6],
            "[6/8]P": ml[5], "[7/8]P": ml[4], "[8/8]P": ml[3],
            "[+1/8]P": ml[2], "[+2/8]P": ml[1], "[+3/8]P": ml[0],
        }

    def calculate_rolling_murrey_math_levels_optimized(self, df: pd.DataFrame, window_size: int) -> Dict[str, pd.Series]:
        """
        OPTIMIZED Version - Calculate MML levels every 5 candles using only past data
        """
        murrey_levels_data: Dict[str, list] = {key: [np.nan] * len(df) for key in MML_LEVEL_NAMES}
        mml_c1 = self.mml_const1.value
        mml_c2 = self.mml_const2.value
        
        calculation_step = 5
        
        for i in range(0, len(df), calculation_step):
            if i < window_size:
                continue
                
            # Use data up to the previous candle for the rolling window
            window_end = i - 1
            window_start = window_end - window_size + 1
            if window_start < 0:
                window_start = 0
                
            window_data = df.iloc[window_start:window_end]
            mn_period = window_data["low"].min()
            mx_period = window_data["high"].max()
            current_close = df["close"].iloc[window_end] if window_end > 0 else df["close"].iloc[0]
            
            if pd.isna(mn_period) or pd.isna(mx_period) or mn_period == mx_period:
                for key in MML_LEVEL_NAMES:
                    murrey_levels_data[key][window_end] = current_close
                continue
                
            levels = self._calculate_mml_core(mn_period, mx_period, mx_period, mn_period, mml_c1, mml_c2)
            
            for key in MML_LEVEL_NAMES:
                murrey_levels_data[key][window_end] = levels.get(key, current_close)
        
        # Interpolate using only past data up to each point
        for key in MML_LEVEL_NAMES:
            series = pd.Series(murrey_levels_data[key], index=df.index)
            # Forward-fill only — never average levels, that produces wrong S/R values
            series = series.ffill().bfill()
            murrey_levels_data[key] = series.tolist()
        
        return {key: pd.Series(data, index=df.index) for key, data in murrey_levels_data.items()}

    def calculate_synthetic_market_breadth(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Calculate synthetic market breadth using technical indicators
        Simulates market sentiment based on multiple factors
        """
        try:
            # RSI component (30% weight)
            rsi_component = (dataframe['rsi'] - 50) / 50  # Normalize to -1 to 1
            
            # Volume component (25% weight)
            volume_ma = dataframe['volume'].rolling(20).mean()
            # when 20 consecutive candles have zero volume (rare but
            # possible on delisted pairs or testnet).
            volume_component = (dataframe['volume'] / (volume_ma + 1e-10) - 1).clip(-1, 1)
            
            # Momentum component (25% weight)
            momentum_3 = dataframe['close'].pct_change(3)
            momentum_component = np.tanh(momentum_3 * 100)  # Smooth normalization
            
            # Volatility component (20% weight) - inverted (lower vol = higher breadth)
            atr_normalized = dataframe['atr'] / dataframe['close']
            atr_ma = atr_normalized.rolling(20).mean()
            volatility_component = -(atr_normalized / atr_ma - 1).clip(-1, 1)
            
            # Combine components with weights
            synthetic_breadth = (
                rsi_component * 0.30 +
                volume_component * 0.25 +
                momentum_component * 0.25 +
                volatility_component * 0.20
            )
            
            # Normalize to 0-1 range (market breadth percentage)
            synthetic_breadth = (synthetic_breadth + 1) / 2
            
            # Smooth with rolling average
            synthetic_breadth = synthetic_breadth.rolling(3).mean()
            
            return synthetic_breadth.fillna(0.5)
            
        except Exception as e:
            logger.warning(f"Synthetic market breadth calculation failed: {e}")
            return pd.Series(0.5, index=dataframe.index)

    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend strength to avoid entering against strong trends
        """
        # Linear regression slope
        def calc_slope(series, period=10):
            """Calculate linear regression slope"""
            if len(series) < period:
                return 0
            x = np.arange(period)
            y = series.iloc[-period:].values
            if np.isnan(y).any() or np.isinf(y).any():
                return 0
            slope = np.polyfit(x, y, 1)[0]
            return slope
        
        # Calculate trend strength using multiple timeframes
        df['slope_5'] = df['close'].rolling(5).apply(
            lambda x: calc_slope(x, 5), raw=False
        )
        df['slope_10'] = df['close'].rolling(10).apply(
            lambda x: calc_slope(x, 10), raw=False
        )
        df['slope_20'] = df['close'].rolling(20).apply(
            lambda x: calc_slope(x, 20), raw=False
        )
        
        df['trend_strength_5'] = df['slope_5'] / df['close'] * 100
        df['trend_strength_10'] = df['slope_10'] / df['close'] * 100
        df['trend_strength_20'] = df['slope_20'] / df['close'] * 100
        
        # Combined trend strength
        df['trend_strength'] = (df['trend_strength_5'] + df['trend_strength_10'] + df['trend_strength_20']) / 3
        
        # Trend classification
        strong_threshold = 0.02
        df['strong_uptrend'] = df['trend_strength'] > strong_threshold
        df['strong_downtrend'] = df['trend_strength'] < -strong_threshold
        df['ranging'] = df['trend_strength'].abs() < (strong_threshold * 0.5)

        return df
    @property
    def protections(self):
        prot = [{"method": "CooldownPeriod", "stop_duration_candles": self.cooldown_lookback.value}]
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 72,
                "trade_limit": 2,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False,
            })
        return prot

    def informative_pairs(self):
        """
        Define additional pairs for correlation analysis.
        Pulls from self.extra_informative_pairs so users can override
        per market (USDT futures, USDC, spot, etc.). Pairs that aren't on the
        exchange will be silently ignored by Freqtrade.
        """
        pairs = [(p, self.timeframe) for p in (self.extra_informative_pairs or [])]
        return pairs


    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        IMPROVED: Dynamic stoploss that works for BOTH longs and shorts
        Returns the stop loss percentage (negative for longs, positive for shorts)
        """
        
        # Detect if this is a short position
        is_short = trade.is_short if hasattr(trade, 'is_short') else False
        
        # Get dataframe for ATR calculation
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        
        if dataframe.empty or 'atr' not in dataframe.columns:
            return self.stoploss if not is_short else -self.stoploss
        
        # Get latest ATR
        try:
            atr = dataframe["atr"].iat[-1]
        except (IndexError, KeyError):
            return self.stoploss if not is_short else -self.stoploss
        
        if pd.isna(atr) or atr <= 0:
            return self.stoploss if not is_short else -self.stoploss
        
        # Calculate ATR percentage
        atr_percent = atr / current_rate
        
        # ===== DYNAMIC STOP LOSS BASED ON PROFIT =====
        
        if current_profit > 0.15:  # 15%+ profit
            multiplier = 0.8
            min_stop = -0.08
            
        elif current_profit > 0.10:  # 10-15% profit
            multiplier = 1.0
            min_stop = -0.10
            
        elif current_profit > 0.05:  # 5-10% profit
            multiplier = 1.2
            min_stop = -0.12
            
        elif current_profit > 0.02:  # 2-5% profit
            multiplier = 1.4
            min_stop = -0.15
            
        elif current_profit > 0:  # 0-2% profit
            multiplier = 1.5
            min_stop = -0.18
            
        else:  # In loss
            multiplier = 1.8
            min_stop = -0.20
        
        # Calculate ATR-based stoploss
        atr_stoploss = -(atr_percent * multiplier)
        
        # ===== TRAILING STOP LOGIC =====
        
        trailing_offset = 0.0
        
        if current_profit > 0.025:
            if current_profit > 0.10:
                trailing_distance = 0.01
            elif current_profit > 0.05:
                trailing_distance = 0.015
            else:
                trailing_distance = 0.02
            
            trailing_offset = max(0, current_profit - trailing_distance)
            
            if trailing_offset > 0:
                atr_stoploss = min(atr_stoploss, -trailing_offset)
        
        # ===== APPLY LIMITS =====
        
        final_stoploss = max(atr_stoploss, min_stop)
        
        if current_profit > 0:
            final_stoploss = max(final_stoploss, self.stoploss)
        
        final_stoploss = max(final_stoploss, -0.06)
        
        # ===== INVERT FOR SHORT POSITIONS =====
        if is_short:
            final_stoploss = -final_stoploss  # Flip sign for shorts
        
        # ===== LOGGING =====
        
        if current_profit > 0.02 or trailing_offset > 0:
            logger.info(
                f"🛡️ {pair} {'SHORT' if is_short else 'LONG'} Stop Loss Update: "
                f"Profit={current_profit:.2%}, "
                f"ATR_SL={atr_stoploss:.2%}, "
                f"Final_SL={final_stoploss:.2%}, "
                f"Trailing={trailing_offset:.2%}"
            )
        
        return final_stoploss

    # ==========================================================================
    # Custom exits run per-tick per-trade, which is the idiomatic Freqtrade
    # pattern (rather than precomputing exit columns in the dataframe).
    #
    # Priority order (first match wins):
    #   1. Emergency AI exit on big adverse move + ML extreme
    #   2. Structure break against position with profit cushion
    #   3. MML target reached (long at 8/8, short at 0/8)
    #   4. AI degradation while profitable
    #   5. RSI exhaustion while profitable
    #   6. Time-based rotation when neutral ML and tiny move
    # ==========================================================================
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            if df is None or df.empty or len(df) < 10:
                return None

            is_short: bool = bool(getattr(trade, "is_short", False))
            duration_h: float = (current_time - trade.open_date_utc).total_seconds() / 3600.0

            # --- Pull last few rows defensively (columns may be missing if a
            # prior indicator block raised). Use .get + .iloc[-N:] for speed. ---
            last = df.iloc[-1]
            tail3 = df.iloc[-3:]
            tail10 = df.iloc[-10:]

            ml_prob = float(last.get('ml_entry_probability', 0.5))
            ml_enh  = float(last.get('ml_enhanced_score', 0.5))
            ml_prob_sma10 = float(tail10.get('ml_entry_probability', pd.Series([0.5])).mean()) \
                if 'ml_entry_probability' in df.columns else 0.5
            ml_prob_std10 = float(tail10.get('ml_entry_probability', pd.Series([0.0])).std() or 0.0) \
                if 'ml_entry_probability' in df.columns else 0.0

            rsi = float(last.get('rsi', 50.0))
            structure_break_down = int(last.get('structure_break_down', 0) or 0)
            structure_break_up   = int(last.get('structure_break_up', 0) or 0)
            bullish_struct = float(last.get('bullish_structure', 0) or 0)
            bearish_struct = float(last.get('bearish_structure', 0) or 0)
            volume = float(last.get('volume', 0) or 0)
            avg_volume = float(last.get('avg_volume', volume) or volume or 1.0)

            mml_8_8 = last.get('[8/8]P', np.nan)
            mml_0_8 = last.get('[0/8]P', np.nan)

            # ============================================================
            # 1. EMERGENCY AI EXIT (overrides everything except SL/ROI)
            # ============================================================
            if self.use_emergency_exits and current_profit < -0.05:
                if not is_short and ml_prob < 0.22 and ml_prob_sma10 < 0.30:
                    return "emergency_ai_long"
                if is_short and ml_prob > 0.78 and ml_prob_sma10 > 0.70:
                    return "emergency_ai_short"

            # ============================================================
            # 2. STRUCTURE BREAK AGAINST POSITION (require profit cushion)
            # ============================================================
            if self.exit_on_structure_break.value and current_profit > 0.005:
                if not is_short and structure_break_down == 1 and bearish_struct > bullish_struct:
                    if volume > avg_volume * 1.5:  # confirm with volume
                        return "structure_break_long"
                if is_short and structure_break_up == 1 and bullish_struct > bearish_struct:
                    if volume > avg_volume * 1.5:
                        return "structure_break_short"

            # ============================================================
            # 3. MML TARGET REACHED
            # ============================================================
            if current_profit > 0.02:
                if not is_short and pd.notna(mml_8_8):
                    try:
                        if current_rate >= float(mml_8_8) * 0.995:
                            return "mml_target_long"
                    except (TypeError, ValueError):
                        pass
                if is_short and pd.notna(mml_0_8):
                    try:
                        if current_rate <= float(mml_0_8) * 1.005:
                            return "mml_target_short"
                    except (TypeError, ValueError):
                        pass

            # ============================================================
            # 4. AI DEGRADATION WHILE PROFITABLE
            # Sustained adverse ML over 10 bars + still in profit.
            # We protect against single-candle noise via SMA + std checks.
            # ============================================================
            if self.dynamic_exit_enabled.value and current_profit > 0.01 and duration_h > 1.0:
                # Sustained drop in ml_prob (3 consecutive lower closes)
                if 'ml_entry_probability' in df.columns and len(tail3) >= 3:
                    mp = tail3['ml_entry_probability'].to_numpy()
                    sustained_down = (mp[0] > mp[1] > mp[2])
                    sustained_up   = (mp[0] < mp[1] < mp[2])
                else:
                    sustained_down = sustained_up = False

                if not is_short:
                    # Long: bad if ML probability collapsed
                    if (ml_prob < 0.35 and ml_prob_sma10 < 0.40
                        and (sustained_down or ml_enh < 0.30)):
                        return "ai_degradation_long"
                    # Long: uncertain + losing edge
                    if (ml_prob_std10 > 0.20 and ml_prob < 0.50
                        and ml_enh < 0.35 and current_profit > 0.02):
                        return "ai_uncertainty_long"
                else:
                    # Short: bad if ML probability rose strongly
                    if (ml_prob > 0.65 and ml_prob_sma10 > 0.60
                        and (sustained_up or ml_enh > 0.70)):
                        return "ai_degradation_short"
                    if (ml_prob_std10 > 0.20 and ml_prob > 0.50
                        and ml_enh > 0.65 and current_profit > 0.02):
                        return "ai_uncertainty_short"

            # ============================================================
            # 5. RSI EXHAUSTION WHILE PROFITABLE
            # ============================================================
            if current_profit > 0.015:
                if not is_short and rsi > self.rsi_overbought_exit.value:
                    return "rsi_exhaustion_long"
                if is_short and rsi < (100 - self.rsi_overbought_exit.value):
                    return "rsi_exhaustion_short"

            # ============================================================
            # 6. TIME-BASED ROTATION (drowning by attrition)
            # Position open >24h, ML neutral, tiny PnL: free up the slot.
            # ============================================================
            if duration_h > 24.0 and -0.01 < current_profit < 0.005:
                if 0.45 <= ml_prob <= 0.55 and ml_enh < 0.55:
                    return "time_rotation_neutral"

            return None

        except Exception as e:
            # custom_exit must never raise — log and let other exit logic decide.
            logger.warning(f"[ANF_v2] custom_exit error for {pair}: {e}")
            return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float,
                 max_leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        window_size = self.leverage_window_size.value
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if len(dataframe) < window_size:
            logger.warning(
                f"{pair} Not enough data ({len(dataframe)} candles) to calculate dynamic leverage (requires {window_size}). Using proposed: {proposed_leverage}")
            return proposed_leverage
        close_prices_series = dataframe["close"].tail(window_size)
        high_prices_series = dataframe["high"].tail(window_size)
        low_prices_series = dataframe["low"].tail(window_size)
        base_leverage = self.leverage_base.value
        ta = _get_talib()
        rsi_array = ta.RSI(close_prices_series, timeperiod=14)
        atr_array = ta.ATR(high_prices_series, low_prices_series, close_prices_series, timeperiod=14)
        sma_array = ta.SMA(close_prices_series, timeperiod=20)
        macd_output = ta.MACD(close_prices_series, fastperiod=12, slowperiod=26, signalperiod=9)

        current_rsi = rsi_array[-1] if rsi_array.size > 0 and not np.isnan(rsi_array[-1]) else 50.0
        current_atr = atr_array[-1] if atr_array.size > 0 and not np.isnan(atr_array[-1]) else 0.0
        current_sma = sma_array[-1] if sma_array.size > 0 and not np.isnan(sma_array[-1]) else current_rate
        current_macd_hist = 0.0

        if isinstance(macd_output, pd.DataFrame):
            if not macd_output.empty and 'macdhist' in macd_output.columns:
                valid_macdhist_series = macd_output['macdhist'].dropna()
                if not valid_macdhist_series.empty:
                    current_macd_hist = valid_macdhist_series.iloc[-1]

        # Apply rules based on indicators
        if side == "long":
            if current_rsi < self.leverage_rsi_low.value:
                base_leverage *= self.leverage_long_increase_factor.value
            elif current_rsi > self.leverage_rsi_high.value:
                base_leverage *= self.leverage_long_decrease_factor.value

            if current_atr > 0 and current_rate > 0:
                if (current_atr / current_rate) > self.leverage_atr_threshold_pct.value:
                    base_leverage *= self.leverage_volatility_decrease_factor.value

            if current_macd_hist > 0:
                base_leverage *= self.leverage_long_increase_factor.value

            if current_sma > 0 and current_rate < current_sma:
                base_leverage *= self.leverage_long_decrease_factor.value

        # === ANF_v2: HMM regime-based leverage modulation ===
        # Graded risk control. When the HMM identifies a 'volatile' regime with
        # meaningful confidence, scale leverage DOWN proportionally to the
        # posterior probability. This acts BEFORE the volatility shows up in
        # the ATR term above (which is reactive), giving an earlier brake.
        # No-op when the HMM is unavailable/unfitted (probs is None) so the
        # legacy behaviour is preserved exactly.
        try:
            engine = getattr(self, 'predictive_engine', None)
            if engine is not None and getattr(engine, 'hmm_detector', None) is not None \
                    and engine.hmm_detector.is_fitted:
                label, probs = engine.detect_market_regime_with_confidence(dataframe)
                if probs is not None:
                    p_volatile = probs.get('volatile', 0.0)
                    p_trending = probs.get('trending', 0.0)
                    # Volatile regime: cut leverage by up to 40% scaled by P(volatile).
                    if p_volatile > 0.5:
                        factor = 1.0 - 0.4 * p_volatile  # p=0.5->0.8x, p=1.0->0.6x
                        base_leverage *= factor
                        logger.info(
                            f"{pair} [HMM] volatile regime P={p_volatile:.2f} "
                            f"-> leverage x{factor:.2f}")
                    # Strong trending regime: allow a small boost (max +15%).
                    elif p_trending > 0.6:
                        factor = 1.0 + 0.15 * p_trending
                        base_leverage *= factor
                        logger.info(
                            f"{pair} [HMM] trending regime P={p_trending:.2f} "
                            f"-> leverage x{factor:.2f}")
        except Exception as e:
            logger.debug(f"[HMM] leverage modulation skipped ({e})")

        adjusted_leverage = round(max(1.0, min(base_leverage, max_leverage)), 2)
        logger.info(
            f"{pair} Dynamic Leverage: {adjusted_leverage:.2f} (Base: {base_leverage:.2f}, RSI: {current_rsi:.2f}, "
            f"ATR: {current_atr:.4f}, MACD Hist: {current_macd_hist:.4f}, SMA: {current_sma:.4f})")
        return adjusted_leverage

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        ULTIMATE indicator calculations with advanced market analysis
        """
            # === ENSURE ML ENGINE IS INITIALIZED ===
        # === INSTANCE STATE INITIALIZATION (safe guard) ===
        if not hasattr(self, '_last_ai_state'):
            self._last_ai_state = {}

        # === ENSURE ML ENGINE IS INITIALIZED ===
        global predictive_engine
        if predictive_engine is None:
            logger.info("Initializing ML predictive engine...")
            predictive_engine = AdvancedPredictiveEngine(config=None)
        
        # === ML ASSET STARTUP CHECK ===
        pair = metadata['pair']
        if hasattr(self, 'predictive_engine') and self.predictive_engine is not None:
            # Mark pair as trained if assets exist (startup optimization)
            self.predictive_engine.mark_trained_if_assets(pair)

        # === ANF_v2: HMM regime model lazy training ===
        # Trains/refreshes the global HMM on the anchor (BTC) series the first
        # time we see enough candles, then weekly. No-op for non-anchor pairs,
        # in backtest, or when hmmlearn is unavailable. Never raises.
        try:
            self._maybe_train_hmm(dataframe, pair)
        except Exception as e:
            logger.debug(f"[HMM] training hook error ({e})")

        # === EXTERNAL DATA INTEGRATION ===
        # string disables BTC correlation entirely.
        btc_pair = (self.btc_informative_pair or "").strip()
        try:
            if btc_pair and metadata['pair'] != btc_pair:
                btc_info = self.dp.get_pair_dataframe(btc_pair, self.timeframe)
                if not btc_info.empty and len(btc_info) >= len(dataframe):
                    btc_close_data = btc_info['close'].tail(len(dataframe)).reset_index(drop=True)
                    dataframe['btc_close'] = btc_close_data.values
                    _log_info_throttled(f"BTC correlation data added for {metadata['pair']} (ref={btc_pair})",
                                       metadata.get("pair"), 600)
                else:
                    # Fallback: use current pair data
                    dataframe['btc_close'] = dataframe['close']
                    logger.debug(f"{metadata['pair']} BTC data unavailable for {btc_pair}, using fallback")
            else:
                # For the BTC pair itself, or when BTC correlation is disabled, use own data
                dataframe['btc_close'] = dataframe['close']
        except Exception as e:
            logger.warning(f"{metadata['pair']} BTC data integration failed (ref={btc_pair}): {e}")
            dataframe['btc_close'] = dataframe['close']  # Safe fallback
        
        # === STANDARD INDICATORS ===
        ta = _get_talib()
        dataframe["ema50"] = ta.EMA(dataframe["close"], timeperiod=50)
        dataframe["ema100"] = ta.EMA(dataframe["close"], timeperiod=100) # Neu hinzufÃ¼gen
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe["close"])
        dataframe["atr"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=10)
        
        # === SYNTHETIC MARKET BREADTH CALCULATION ===
        try:
            # Calculate synthetic market breadth using multiple indicators
            # (after RSI and ATR are available)
            dataframe['market_breadth'] = self.calculate_synthetic_market_breadth(dataframe)
            _log_info_throttled(f"Synthetic market breadth calculated for {metadata['pair']}", metadata.get("pair"), 600)
        except Exception as e:
            logger.debug(f"{metadata['pair']} Market breadth calculation failed: {e}")
            dataframe['market_breadth'] = 0.5  # Neutral fallback
        dataframe["plus_di"] = ta.PLUS_DI(dataframe)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe)
        dataframe["DI_values"] = dataframe["plus_di"] - dataframe["minus_di"]
        dataframe["DI_cutoff"] = 0

        # === EXTREMA DETECTION ===
        extrema_order = self.indicator_extrema_order.value
        dataframe["maxima"] = (
            dataframe["close"] == dataframe["close"].shift(1).rolling(window=extrema_order).max()
        ).astype(int)
        dataframe["minima"] = (
            dataframe["close"] == dataframe["close"].shift(1).rolling(window=extrema_order).min()
        ).astype(int)

        dataframe["s_extrema"] = 0
        dataframe.loc[dataframe["minima"] == 1, "s_extrema"] = -1
        dataframe.loc[dataframe["maxima"] == 1, "s_extrema"] = 1

        # === HEIKIN-ASHI ===
        dataframe["ha_close"] = (dataframe["open"] + dataframe["high"] + dataframe["low"] + dataframe["close"]) / 4

        # === ROLLING EXTREMA ===
        dataframe["minh2"], dataframe["maxh2"] = calculate_minima_maxima(dataframe, self.h2.value)
        dataframe["minh1"], dataframe["maxh1"] = calculate_minima_maxima(dataframe, self.h1.value)
        dataframe["minh0"], dataframe["maxh0"] = calculate_minima_maxima(dataframe, self.h0.value)
        dataframe["mincp"], dataframe["maxcp"] = calculate_minima_maxima(dataframe, self.cp.value)

        # === MURREY MATH LEVELS ===
        mml_window = self.indicator_mml_window.value
        murrey_levels = self.calculate_rolling_murrey_math_levels_optimized(dataframe, window_size=mml_window)
        
        for level_name in MML_LEVEL_NAMES:
            if level_name in murrey_levels:
                dataframe[level_name] = murrey_levels[level_name]
            else:
                dataframe[level_name] = dataframe["close"]

        # === MML OSCILLATOR ===
        mml_4_8 = dataframe.get("[4/8]P")
        mml_plus_3_8 = dataframe.get("[+3/8]P")
        mml_minus_3_8 = dataframe.get("[-3/8]P")
        
        if mml_4_8 is not None and mml_plus_3_8 is not None and mml_minus_3_8 is not None:
            osc_denominator = (mml_plus_3_8 - mml_minus_3_8).replace(0, np.nan)
            dataframe["mmlextreme_oscillator"] = 100 * ((dataframe["close"] - mml_4_8) / osc_denominator)
        else:
            dataframe["mmlextreme_oscillator"] = np.nan

        # === DI CATCH ===
        dataframe["DI_catch"] = np.where(dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1)

        # === ROLLING THRESHOLDS ===
        rolling_window_threshold = self.indicator_rolling_window_threshold.value
        dataframe["minima_sort_threshold"] = dataframe["close"].rolling(
            window=rolling_window_threshold, min_periods=1
        ).min()
        dataframe["maxima_sort_threshold"] = dataframe["close"].rolling(
            window=rolling_window_threshold, min_periods=1
        ).max()

        # === EXTREMA CHECKS ===
        rolling_check_window = self.indicator_rolling_check_window.value
        dataframe["minima_check"] = (
            dataframe["minima"].rolling(window=rolling_check_window, min_periods=1).sum() == 0
        ).astype(int)
        dataframe["maxima_check"] = (
            dataframe["maxima"].rolling(window=rolling_check_window, min_periods=1).sum() == 0
        ).astype(int)

        # === VOLATILITY INDICATORS ===
        dataframe["volatility_range"] = dataframe["high"] - dataframe["low"]
        dataframe["avg_volatility"] = dataframe["volatility_range"].rolling(window=50).mean()
        dataframe["avg_volume"] = dataframe["volume"].rolling(window=50).mean()

        # === TREND STRENGTH INDICATORS ===
        # Use enhanced Wavelet+FFT method with fallback
        try:
            # Advanced wavelet & FFT method
            dataframe = calculate_advanced_trend_strength_with_wavelets(dataframe)
            
            # Use advanced trend strength as primary
            dataframe['trend_strength'] = dataframe['trend_strength_cycle_adjusted']
            dataframe['strong_uptrend'] = dataframe['strong_uptrend_advanced']
            dataframe['strong_downtrend'] = dataframe['strong_downtrend_advanced']
            dataframe['ranging'] = dataframe['ranging_advanced']
            
            _log_info_throttled(f"Using advanced Wavelet+FFT trend analysis for {metadata['pair']}", metadata.get("pair"), 600)
            
        except Exception as e:
            # Fallback to original enhanced method if advanced fails
            logger.warning(
                f"{metadata['pair']} Wavelet/FFT analysis failed: {e}. "
                "Using enhanced method."
            )
            
            def calc_slope(series, period):
                """Enhanced slope calculation as fallback"""
                if len(series) < period:
                    return 0
                y = series.values[-period:]
                if np.isnan(y).any() or np.isinf(y).any():
                    return 0
                if np.all(y == y[0]):
                    return 0
                x = np.linspace(0, period-1, period)
                try:
                    coefficients = np.polyfit(x, y, 1)
                    slope = coefficients[0]
                    if np.isnan(slope) or np.isinf(slope):
                        return 0
                    max_reasonable_slope = np.std(y) / period
                    if abs(slope) > max_reasonable_slope * 10:
                        return np.sign(slope) * max_reasonable_slope * 10
                    return slope
                except Exception:
                    try:
                        simple_slope = (y[-1] - y[0]) / (period - 1)
                        return (simple_slope if not
                               (np.isnan(simple_slope) or np.isinf(simple_slope))
                               else 0)
                    except Exception:
                        return 0
            
            # Original slope calculations
            dataframe['slope_5'] = dataframe['close'].rolling(5).apply(
                lambda x: calc_slope(x, 5), raw=False
            )
            dataframe['slope_10'] = dataframe['close'].rolling(10).apply(
                lambda x: calc_slope(x, 10), raw=False
            )
            dataframe['slope_20'] = dataframe['close'].rolling(20).apply(
                lambda x: calc_slope(x, 20), raw=False
            )
            
            dataframe['trend_strength_5'] = dataframe['slope_5'] / dataframe['close'] * 100
            dataframe['trend_strength_10'] = dataframe['slope_10'] / dataframe['close'] * 100
            dataframe['trend_strength_20'] = dataframe['slope_20'] / dataframe['close'] * 100
            
            dataframe['trend_strength'] = (
                dataframe['trend_strength_5'] +
                dataframe['trend_strength_10'] +
                dataframe['trend_strength_20']
            ) / 3
            
            strong_threshold = 0.02
            dataframe['strong_uptrend'] = dataframe['trend_strength'] > strong_threshold
            dataframe['strong_downtrend'] = dataframe['trend_strength'] < -strong_threshold
            dataframe['ranging'] = dataframe['trend_strength'].abs() < (strong_threshold * 0.5)

        # === MOMENTUM INDICATORS ===
        dataframe['price_momentum'] = dataframe['close'].pct_change(3)
        dataframe['momentum_increasing'] = (
            dataframe['price_momentum'] > dataframe['price_momentum'].shift(1)
        )
        dataframe['momentum_decreasing'] = (
            dataframe['price_momentum'] < dataframe['price_momentum'].shift(1)
        )

        dataframe['volume_momentum'] = (
            dataframe['volume'].rolling(3).mean() /
            dataframe['volume'].rolling(20).mean()
        )

        dataframe['rsi_divergence_bull'] = (
            (dataframe['close'] < dataframe['close'].shift(5)) &
            (dataframe['rsi'] > dataframe['rsi'].shift(5))
        )
        dataframe['rsi_divergence_bear'] = (
            (dataframe['close'] > dataframe['close'].shift(5)) &
            (dataframe['rsi'] < dataframe['rsi'].shift(5))
        )

        # === CANDLE PATTERNS ===
        dataframe['green_candle'] = dataframe['close'] > dataframe['open']
        dataframe['red_candle'] = dataframe['close'] < dataframe['open']
        dataframe['consecutive_green'] = dataframe['green_candle'].rolling(3).sum()
        dataframe['consecutive_red'] = dataframe['red_candle'].rolling(3).sum()

        # Define strong_threshold for momentum calculations
        strong_threshold = 0.02

        dataframe['strong_up_momentum'] = (
            (dataframe['consecutive_green'] >= 3) &
            (dataframe['volume'] > dataframe['avg_volume']) &
            (dataframe['trend_strength'] > strong_threshold)
        )
        dataframe['strong_down_momentum'] = (
            (dataframe['consecutive_red'] >= 3) &
            (dataframe['volume'] > dataframe['avg_volume']) &
            (dataframe['trend_strength'] < -strong_threshold)
        )

        # === ADVANCED ANALYSIS MODULES ===
        
        # 1. CONFLUENCE ANALYSIS
        if self.confluence_enabled.value:
            dataframe = calculate_confluence_score(dataframe)
        else:
            dataframe['confluence_score'] = 0
        
        # 2. SMART VOLUME ANALYSIS
        if self.volume_analysis_enabled.value:
            dataframe = calculate_smart_volume(dataframe)
        else:
            dataframe['volume_pressure'] = 0
            dataframe['volume_strength'] = 1.0
            dataframe['money_flow_index'] = 50
        
        # 3. ADVANCED MOMENTUM
        if self.momentum_analysis_enabled.value:
            dataframe = calculate_advanced_momentum(dataframe)
        else:
            dataframe['momentum_quality'] = 0
            dataframe['momentum_acceleration'] = 0
        
        # 4. MARKET STRUCTURE
        if self.structure_analysis_enabled.value:
            dataframe = calculate_market_structure(dataframe)
        else:
            dataframe['structure_score'] = 0
            dataframe['structure_break_up'] = 0
        
        # 5. ADVANCED ENTRY SIGNALS
        dataframe = calculate_advanced_entry_signals(dataframe)

        # === ULTIMATE MARKET SCORE ===
        dataframe['ultimate_score'] = (
            dataframe['confluence_score'] * 0.25 +           # 25% confluence
            dataframe['volume_pressure'] * 0.2 +             # 20% volume pressure
            dataframe['momentum_quality'] * 0.2 +            # 20% momentum quality
            (dataframe['structure_score'] / 5) * 0.15 +      # 15% structure (normalized)
            (dataframe['signal_strength'] / 10) * 0.2        # 20% signal strength
        )
        
        # Normalize ultimate score to 0-1 range
        dataframe['ultimate_score'] = dataframe['ultimate_score'].clip(0, 5) / 5

        # === FINAL QUALITY CHECKS ===
        dataframe['high_quality_setup'] = (
            (dataframe['ultimate_score'] > self.ultimate_score_threshold.value) &
            (dataframe['signal_strength'] >= 5) &
            (dataframe['volume_strength'] > 1.1) &
            (dataframe['rsi'] > 30) & (dataframe['rsi'] < 70)
        ).astype(int)

        # === DEBUG INFO ===
        if metadata['pair'] in (self.debug_log_pairs or []):
            latest_score = dataframe['ultimate_score'].iloc[-1]
            latest_signal = dataframe['signal_strength'].iloc[-1]
            logger.debug(f"{metadata['pair']} Ultimate Score: {latest_score:.3f}, Signal Strength: {latest_signal}")

        # ===========================================
        # REGIME CHANGE DETECTION
        # ===========================================
        
        if self.regime_change_enabled.value:
            
            # ===========================================
            # FLASH MOVE DETECTION
            # ===========================================
            
            flash_candles = self.flash_move_candles.value
            flash_threshold = self.flash_move_threshold.value
            
            # Schnelle Preisbewegungen
            dataframe['price_change_fast'] = dataframe['close'].pct_change(flash_candles)
            dataframe['flash_pump'] = dataframe['price_change_fast'] > flash_threshold
            dataframe['flash_dump'] = dataframe['price_change_fast'] < -flash_threshold
            dataframe['flash_move'] = dataframe['flash_pump'] | dataframe['flash_dump']
            
            # ===========================================
            # VOLUME SPIKE DETECTION
            # ===========================================
            
            volume_ma20 = dataframe['volume'].rolling(20).mean()
            volume_multiplier = self.volume_spike_multiplier.value
            dataframe['volume_spike'] = dataframe['volume'] > (volume_ma20 * volume_multiplier)
            
            # Volume + Bewegung kombiniert
            dataframe['volume_pump'] = dataframe['volume_spike'] & dataframe['flash_pump']
            dataframe['volume_dump'] = dataframe['volume_spike'] & dataframe['flash_dump']
            
            # ===========================================
            # MARKET SENTIMENT DETECTION
            # ===========================================
            
            # Market Breadth Change
            if 'market_breadth' in dataframe.columns:
                dataframe['market_breadth_change'] = dataframe['market_breadth'].diff(3)
                sentiment_threshold = self.sentiment_shift_threshold.value
                dataframe['sentiment_shift_bull'] = dataframe['market_breadth_change'] > sentiment_threshold
                dataframe['sentiment_shift_bear'] = dataframe['market_breadth_change'] < -sentiment_threshold
            else:
                dataframe['sentiment_shift_bull'] = False
                dataframe['sentiment_shift_bear'] = False
            
            # ===========================================
            # BTC CORRELATION MONITORING
            # ===========================================
            
            # BTC Flash Moves
            if 'btc_close' in dataframe.columns:
                dataframe['btc_change_fast'] = dataframe['btc_close'].pct_change(flash_candles)
                dataframe['btc_flash_pump'] = dataframe['btc_change_fast'] > flash_threshold
                dataframe['btc_flash_dump'] = dataframe['btc_change_fast'] < -flash_threshold
                
                # Correlation Break
                pair_movement = dataframe['price_change_fast'].abs()
                btc_movement = dataframe['btc_change_fast'].abs()
                dataframe['correlation_break'] = (btc_movement > flash_threshold) & (pair_movement < flash_threshold * 0.4)
            else:
                dataframe['btc_flash_pump'] = False
                dataframe['btc_flash_dump'] = False
                dataframe['correlation_break'] = False
            
            # ===========================================
            # REGIME CHANGE SCORE
            # ===========================================
            
           
            regime_signals = [
                'flash_move', 'volume_spike', 
                'sentiment_shift_bull', 'sentiment_shift_bear',
                'btc_flash_pump', 'btc_flash_dump', 'correlation_break'
            ]
            
            dataframe['regime_change_score'] = 0
            for signal in regime_signals:
                if signal in dataframe.columns:
                    dataframe['regime_change_score'] += dataframe[signal].astype(int)
            
            # Normalisiere auf 0-1
            max_signals = len(regime_signals)
            dataframe['regime_change_intensity'] = dataframe['regime_change_score'] / max_signals
            
            # Alert Level
            sensitivity = self.regime_change_sensitivity.value
            dataframe['regime_alert'] = dataframe['regime_change_intensity'] >= sensitivity
            
        else:
            # Falls Regime Detection deaktiviert
            dataframe['flash_pump'] = False
            dataframe['flash_dump'] = False
            dataframe['volume_pump'] = False
            dataframe['volume_dump'] = False
            dataframe['regime_alert'] = False
            dataframe['regime_change_intensity'] = 0.0
        
        # === ADVANCED PREDICTIVE ANALYSIS ===
        try:
            pair = metadata.get('pair', 'UNKNOWN')
            dataframe = calculate_advanced_predictive_signals(dataframe, pair)
            dataframe = calculate_quantum_momentum_analysis(dataframe)
            dataframe = calculate_neural_pattern_recognition(dataframe)
            
            _log_info_throttled(f"Advanced predictive analysis completed for {pair}", pair, 600)
        except Exception as e:
            logger.debug(f"Advanced predictive analysis failed: {e}")
            dataframe['ml_entry_probability'] = 0.5
            dataframe['ml_enhanced_score'] = dataframe.get('ultimate_score', 0.5)
            dataframe['ml_high_confidence'] = 0
            dataframe['ml_ultra_confidence'] = 0
            dataframe['quantum_momentum_coherence'] = 0.5
            dataframe['momentum_entanglement'] = 0.5
            dataframe['quantum_tunnel_up_prob'] = 0.5
            dataframe['neural_pattern_score'] = 0.5
        
        # === EXIT PREPARATION ===
        dataframe = calculate_exit_signals(dataframe)
        dataframe = calculate_dynamic_profit_targets(dataframe)
        dataframe = calculate_advanced_stop_loss(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        BALANCED ENTRY SYSTEM
        - 4 symmetric tiers per side (quality over quantity)
        - ML Ultra, Confluence High, Technical, MML Mean-Reversion
        - Short side uses same quality filters inverted + dedicated confluence
        """
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""
        dataframe['entry_type'] = 0

        ml_prob = dataframe.get('ml_entry_probability', pd.Series(0.5, index=dataframe.index))
        ml_enhanced = dataframe.get('ml_enhanced_score', pd.Series(0.5, index=dataframe.index))
        quantum_coherence = dataframe.get('quantum_momentum_coherence', pd.Series(0.5, index=dataframe.index))
        neural_pattern = dataframe.get('neural_pattern_score', pd.Series(0.5, index=dataframe.index))

        # === BASIC SAFETY ===
        basic_safety_long = (
            (dataframe['volume'] > 0) &
            (dataframe['rsi'] > 20) & (dataframe['rsi'] < 75) &
            (dataframe['close'] > dataframe['ema200'] * 0.95)
        )
        basic_safety_short = (
            (dataframe['volume'] > 0) &
            (dataframe['rsi'] > 30) & (dataframe['rsi'] < 85) &
            (dataframe['close'] < dataframe['ema200'] * 1.05)
        )

        # === TREND CONTEXT ===
        price_above_ema50 = dataframe['close'] > dataframe['ema50']
        price_below_ema50 = dataframe['close'] < dataframe['ema50']
        trend_positive = dataframe['trend_strength'] > 0.005
        trend_negative = dataframe['trend_strength'] < -0.005

        # === SHORT CONFLUENCE SCORE (inline for balance) ===
        short_confluence = (
            dataframe.get('near_resistance', pd.Series(0, index=dataframe.index)) +
            dataframe.get('rsi_overbought', pd.Series(0, index=dataframe.index)) +
            (price_below_ema50 & (dataframe['trend_strength'] < -0.01)).astype(int) +
            dataframe.get('volume_spike', pd.Series(0, index=dataframe.index)) +
            (dataframe.get('bearish_structure', pd.Series(0, index=dataframe.index)) > 
             dataframe.get('bullish_structure', pd.Series(0, index=dataframe.index))).astype(int)
        )

        # ============================================
        # LONG ENTRIES (4 Tiers)
        # ============================================

        # TIER 1: ML Ultra Long
        long_t1 = (
            (ml_prob > 0.70) &
            (ml_enhanced > 0.65) &
            (quantum_coherence > 0.55) &
            (neural_pattern > 0.55) &
            price_above_ema50 &
            trend_positive &
            (dataframe['volume_strength'] > 1.2) &
            (dataframe['rsi'] > 25) & (dataframe['rsi'] < 65) &
            basic_safety_long
        )

        # TIER 2: Confluence High Quality Long
        long_t2 = (
            (dataframe['confluence_score'] >= 4) &
            (dataframe['structure_score'] > 1) &
            (dataframe['volume_pressure'] >= 2) &
            (dataframe['momentum_quality'] >= 3) &
            price_above_ema50 &
            (dataframe['rsi'] > 25) & (dataframe['rsi'] < 70) &
            basic_safety_long &
            ~long_t1
        )

        # TIER 3: Technical Long (RSI bounce + support)
        long_t3 = (
            (dataframe['rsi'] < 32) &
            (dataframe.get('near_support', pd.Series(0, index=dataframe.index)) == 1) &
            (dataframe['volume_strength'] > 1.3) &
            (dataframe['close'] > dataframe['ema50']) &
            (dataframe['momentum_acceleration'] > 0) &
            basic_safety_long &
            ~(long_t1 | long_t2)
        )

        # TIER 4: MML Mean Reversion Long (oversold at MML bottom)
        mml_0_8 = dataframe.get('[0/8]P', pd.Series(np.nan, index=dataframe.index))
        long_t4 = (
            (dataframe['close'] <= mml_0_8 * 1.015) &
            (dataframe['rsi'] < 35) &
            (dataframe['volume'] > dataframe['avg_volume'] * 0.9) &
            (dataframe.get('bullish_structure', pd.Series(0, index=dataframe.index)) > 0) &
            basic_safety_long &
            ~(long_t1 | long_t2 | long_t3)
        )

        # ============================================
        # SHORT ENTRIES (4 Tiers - Symmetric)
        # ============================================

        # TIER 1: ML Ultra Short
        short_t1 = (
            (ml_prob < 0.30) &
            (ml_enhanced < 0.35) &
            (quantum_coherence < 0.45) &
            (neural_pattern < 0.45) &
            price_below_ema50 &
            trend_negative &
            (dataframe['volume_strength'] > 1.2) &
            (dataframe['rsi'] > 55) & (dataframe['rsi'] < 82) &
            basic_safety_short
        )

        # TIER 2: Confluence High Quality Short
        short_t2 = (
            (short_confluence >= 4) &
            (dataframe['structure_score'] < -1) &
            (dataframe['volume_pressure'] <= -2) &
            (dataframe['momentum_quality'] <= 1) &
            price_below_ema50 &
            (dataframe['rsi'] > 60) & (dataframe['rsi'] < 85) &
            basic_safety_short &
            ~short_t1
        )

        # TIER 3: Technical Short (RSI overbought + resistance)
        short_t3 = (
            (dataframe['rsi'] > 72) &
            (dataframe.get('near_resistance', pd.Series(0, index=dataframe.index)) == 1) &
            (dataframe['volume_strength'] > 1.3) &
            (dataframe['close'] < dataframe['ema50']) &
            (dataframe['momentum_acceleration'] < 0) &
            basic_safety_short &
            ~(short_t1 | short_t2)
        )

        # TIER 4: MML Mean Reversion Short (overbought at MML top)
        mml_8_8 = dataframe.get('[8/8]P', pd.Series(np.nan, index=dataframe.index))
        short_t4 = (
            (dataframe['close'] >= mml_8_8 * 0.985) &
            (dataframe['rsi'] > 70) &
            (dataframe['volume'] > dataframe['avg_volume'] * 0.9) &
            (dataframe.get('bearish_structure', pd.Series(0, index=dataframe.index)) > 0) &
            basic_safety_short &
            ~(short_t1 | short_t2 | short_t3)
        )

        # ============================================
        # APPLY LONGS (Priority order)
        # ============================================
        dataframe.loc[long_t1, ["enter_long", "entry_type", "enter_tag"]] = [1, 14, "ml_ultra_long"]
        dataframe.loc[long_t2 & (dataframe["enter_long"] == 0), ["enter_long", "entry_type", "enter_tag"]] = [1, 13, "confluence_high_long"]
        dataframe.loc[long_t3 & (dataframe["enter_long"] == 0), ["enter_long", "entry_type", "enter_tag"]] = [1, 12, "technical_long"]
        dataframe.loc[long_t4 & (dataframe["enter_long"] == 0), ["enter_long", "entry_type", "enter_tag"]] = [1, 11, "mml_meanrev_long"]

        # ============================================
        # APPLY SHORTS (Priority order)
        # ============================================
        dataframe.loc[short_t1, ["enter_short", "entry_type", "enter_tag"]] = [1, 24, "ml_ultra_short"]
        dataframe.loc[short_t2 & (dataframe["enter_short"] == 0), ["enter_short", "entry_type", "enter_tag"]] = [1, 23, "confluence_high_short"]
        dataframe.loc[short_t3 & (dataframe["enter_short"] == 0), ["enter_short", "entry_type", "enter_tag"]] = [1, 22, "technical_short"]
        dataframe.loc[short_t4 & (dataframe["enter_short"] == 0), ["enter_short", "entry_type", "enter_tag"]] = [1, 21, "mml_meanrev_short"]

        # ============================================
        # CONFLICT RESOLUTION
        # ============================================
        conflict_mask = (dataframe['enter_long'] == 1) & (dataframe['enter_short'] == 1)
        if conflict_mask.any():
            long_prio = dataframe['entry_type'].where(dataframe['enter_long'] == 1, 0)
            short_prio = dataframe['entry_type'].where(dataframe['enter_short'] == 1, 0)
            keep_long = long_prio >= short_prio
            dataframe.loc[conflict_mask & ~keep_long, 'enter_long'] = 0
            dataframe.loc[conflict_mask & keep_long, 'enter_short'] = 0

        # Light logging
        if metadata['pair'] in (self.debug_log_pairs or []):
            long_count = dataframe['enter_long'].sum()
            short_count = dataframe['enter_short'].sum()
            if long_count > 0 or short_count > 0:
                _log_info_throttled(f"ANF_v2 Entries — Longs: {int(long_count)}, Shorts: {int(short_count)}",
                                   metadata.get("pair"), 300)

        # =====================================================================
        # DIAGNOSTIC LOGGING (flag-gated, throttled, INFO). Never alters any
        # entry decision — it only reports on the current (last) candle so the
        # operator can answer, from /logs in live: (a) is the ML alive or stuck
        # at the 0.5 neutral value, and (b) when a short does not fire, which
        # specific gate blocked it. This is what turns "it never shorts" into
        # "it never shorts because ml_prob never drops below 0.30".
        # =====================================================================
        if getattr(self, 'diagnostic_logging', False):
            try:
                pair = metadata.get('pair', '?')
                last = dataframe.iloc[-1]
                ml_last = float(ml_prob.iloc[-1]) if len(ml_prob) else 0.5
                # (a) ML health: mean/std across the frame + last value. If mean
                #     ~0.5 and std ~0, the ML is not trained yet (neutral fill).
                ml_mean = float(ml_prob.mean()) if len(ml_prob) else 0.5
                ml_std = float(ml_prob.std()) if len(ml_prob) else 0.0
                ml_state = "NEUTRAL/untrained" if (abs(ml_mean - 0.5) < 0.01 and ml_std < 0.01) else "active"
                _log_info_throttled(
                    f"[DIAG][ML] {pair}: ml_prob last={ml_last:.3f} "
                    f"mean={ml_mean:.3f} std={ml_std:.3f} -> {ml_state}",
                    pair, 3600)  # once per hour per pair

                # (b) Short-gate diagnosis on the current candle: only when no
                #     short fired, report which primary gate failed. Cheap bool
                #     reads on the last row; no per-row loop.
                if int(last.get('enter_short', 0)) == 0:
                    reasons = []
                    if not bool(price_below_ema50.iloc[-1]):
                        reasons.append("price>=ema50")
                    if not bool(trend_negative.iloc[-1]):
                        reasons.append("trend_not_negative")
                    if ml_last >= 0.30:
                        reasons.append(f"ml_prob>=0.30({ml_last:.2f})")
                    if float(last.get('rsi', 50)) < 55:
                        reasons.append(f"rsi<55({float(last.get('rsi', 50)):.0f})")
                    if reasons:
                        _log_info_throttled(
                            f"[DIAG][SHORT] {pair}: no short — blocked by "
                            + ", ".join(reasons), pair, 3600)
            except Exception as e:
                logger.debug(f"[DIAG] entry diagnostics skipped ({e})")

        # =====================================================================
        # For every row that fires an entry, persist the feature values that
        # drove the decision. Stored as columns on the dataframe so Freqtrade
        # auto-persists them in custom_data on the Trade object. This is what
        # makes post-mortem analysis possible: "which tier was profitable?
        # What ml_prob/confluence/mml level looked like when winning trades
        # were taken vs losers?"
        #
        # We attach values only on signal rows to keep memory footprint small.
        # Use np.where to vectorise instead of looping per row.
        # =====================================================================
        signal_mask = (dataframe['enter_long'] == 1) | (dataframe['enter_short'] == 1)
        if signal_mask.any():
            # Snapshot of decision context at signal time. Use .get() with
            # fallbacks because some columns may not exist if optional
            # indicators were disabled.
            dataframe['decision_ml_prob'] = np.where(
                signal_mask,
                dataframe.get('ml_entry_probability', pd.Series(0.5, index=dataframe.index)),
                np.nan,
            )
            dataframe['decision_ml_enh'] = np.where(
                signal_mask,
                dataframe.get('ml_enhanced_score', pd.Series(0.5, index=dataframe.index)),
                np.nan,
            )
            dataframe['decision_ultimate'] = np.where(
                signal_mask,
                dataframe.get('ultimate_score', pd.Series(0.0, index=dataframe.index)),
                np.nan,
            )
            dataframe['decision_signal_strength'] = np.where(
                signal_mask,
                dataframe.get('signal_strength', pd.Series(0, index=dataframe.index)),
                np.nan,
            )
            dataframe['decision_rsi'] = np.where(
                signal_mask,
                dataframe.get('rsi', pd.Series(50.0, index=dataframe.index)),
                np.nan,
            )
            # MML position: distance from current close to nearest MML level,
            # normalised. Helps post-mortem: were entries near support or far?
            if 'minima_sort_threshold' in dataframe.columns:
                dist_support = ((dataframe['close'] - dataframe['minima_sort_threshold'])
                                / dataframe['close']).fillna(0)
                dataframe['decision_dist_support_pct'] = np.where(
                    signal_mask, dist_support, np.nan)

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        BACKUP EXIT SYSTEM
        Simple opposite-signal exits when custom_exit does not trigger.
        """
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""

        dataframe.loc[dataframe["enter_short"] == 1, "exit_long"] = 1
        dataframe.loc[dataframe["enter_short"] == 1, "exit_tag"] = "opposite_short_signal"

        dataframe.loc[dataframe["enter_long"] == 1, "exit_short"] = 1
        dataframe.loc[dataframe["enter_long"] == 1, "exit_tag"] = "opposite_long_signal"

        return dataframe

    # ==========================================================================
    #
    # Problem solved: with max_open_trades=3 and a whitelist dominated by
    # large-cap perpetuals, the engine routinely fires simultaneous long
    # signals on BTC/ETH/SOL (correlation typically > 0.80 on 1h). Without
    # this guard, the real portfolio risk is ~3x what the user thinks because
    # the three positions move as one. This is the most expensive silent bug
    # in any crypto multi-pair bot.
    #
    # Behaviour:
    #   - On every prospective entry, compute Pearson correlation between the
    #     new pair's last `corr_lookback` returns and each currently-open
    #     trade's pair returns.
    #   - If |correlation| > corr_threshold, reject the new entry regardless of
    #     direction. This is deliberately conservative: it blocks both amplified
    #     exposure (same-direction, positive corr) and the cases that look like
    #     hedges but are not (e.g. opposite-direction on negatively-correlated
    #     pairs, which is actually doubled exposure). Erring toward blocking
    #     keeps portfolio risk bounded; a signed-exposure refinement can be
    #     evaluated later with dry-run data.
    #
    # Tunable via class attributes (NOT hyperopt — these are risk parameters,
    # tuning them via hyperopt would defeat the purpose):
    #   correlation_filter_enabled  (default True)
    #   corr_threshold              (default 0.75)
    #   corr_lookback               (default 50 candles ≈ 2 days at 1h)
    # ==========================================================================
    correlation_filter_enabled: bool = True
    corr_threshold: float = 0.75
    corr_lookback: int = 50

    # Tunable but NOT hyperopt-able (risk control, not parameter to optimise).
    # Default of -0.05 means: if today's realised P/L sum is below -5% of
    # dry_run_wallet, block all new entries for the rest of the day. Set to
    # 0.0 or negative-large to effectively disable. The kill-switch resets
    # automatically at the start of each new UTC day.
    daily_loss_kill_switch_enabled: bool = True
    daily_loss_threshold_pct: float = -0.05  # -5% of wallet

    def _daily_loss_blocked(self, current_time: datetime) -> bool:
        """Return True if today's cumulative realised PnL is below the kill
        switch threshold. Sums close_profit_abs of all trades closed today
        (UTC) and compares it against daily_loss_threshold_pct * wallet.
        Robust: returns False (= no block) on any error, never raises.
        """
        if not self.daily_loss_kill_switch_enabled:
            return False
        try:
            today_utc = current_time.date()
            closed = Trade.get_trades_proxy(is_open=False)
            today_pnl_abs = 0.0
            for t in closed:
                close_date = getattr(t, 'close_date_utc', None) or getattr(t, 'close_date', None)
                if close_date is None:
                    continue
                # either a datetime (Freqtrade's standard) or a plain date
                # (hypothetical future API change). Normalise both cases.
                try:
                    close_date_only = close_date.date() if hasattr(close_date, 'date') else close_date
                    if close_date_only != today_utc:
                        continue
                except Exception:
                    continue
                pnl_abs = getattr(t, 'close_profit_abs', None)
                if pnl_abs is None:
                    pnl_abs = (getattr(t, 'close_profit', 0.0) or 0.0) * \
                              (getattr(t, 'stake_amount', 0.0) or 0.0)
                today_pnl_abs += float(pnl_abs or 0.0)

            # Wallet resolution order: the real wallet first (self.wallets),
            # then dry_run_wallet, then stake_amount * max_open_trades * 2 as a
            # last-resort heuristic so the switch still bites if mis-configured.
            # Reading only config['dry_run_wallet'] would be 0 in live mode,
            # making this kill switch a no-op there.
            wallet = 0.0
            try:
                if hasattr(self, 'wallets') and self.wallets is not None:
                    w = self.wallets.get_total_stake_amount()
                    if w and w > 0:
                        wallet = float(w)
            except Exception:
                pass
            if wallet <= 0:
                wallet = float(self.config.get('dry_run_wallet', 0.0) or 0.0)
            if wallet <= 0:
                # Heuristic: max_open_trades * stake_amount * 2 as a sane
                # floor so the switch is never completely disabled.
                try:
                    sa = self.config.get('stake_amount', 0)
                    mot = self.config.get('max_open_trades', 1)
                    if isinstance(sa, (int, float)) and sa > 0 and mot > 0:
                        wallet = float(sa) * float(mot) * 2.0
                except Exception:
                    pass
            if wallet <= 0:
                return False  # truly nothing to bound against, fail-open
            threshold_abs = self.daily_loss_threshold_pct * wallet
            if today_pnl_abs <= threshold_abs:
                # Throttled warning so we don't spam Telegram
                _log_info_throttled(
                    f"[ANF_v2][KILL] Daily loss threshold hit "
                    f"({today_pnl_abs:.2f} <= {threshold_abs:.2f}); "
                    f"blocking new entries until tomorrow UTC.",
                    pair="__daily_kill__", interval_sec=1800,
                )
                return True
            return False
        except Exception as e:
            logger.debug(f"[ANF_v2] daily-loss check error (failing open): {e}")
            return False

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                            rate: float, time_in_force: str, current_time: datetime,
                            entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """correlation gate + daily-loss kill switch."""
        # Backtest-friendly bypass: in backtest the correlation calc adds
        # significant cost and is mostly unnecessary (the look-ahead guard
        # already keeps ML off). Skip silently.
        try:
            rm = self.config.get('runmode')
            runmode_raw = str(rm.value if hasattr(rm, 'value') else rm).lower()
            if any(tok in runmode_raw for tok in ('backtest', 'hyperopt')):
                return True
        except Exception:
            pass

        # If today's realised losses exceed the threshold, no new entries at all.
        if self._daily_loss_blocked(current_time):
            return False

        if not self.correlation_filter_enabled:
            return True

        try:
            open_trades = Trade.get_open_trades()
            if not open_trades:
                return True  # no positions yet, nothing to correlate against

            # Pull current pair returns
            df_new, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            if df_new is None or df_new.empty or len(df_new) < self.corr_lookback + 1:
                return True  # not enough history, fail-open (don't block legitimate trades)
            new_returns = df_new['close'].pct_change().tail(self.corr_lookback).dropna()
            if len(new_returns) < self.corr_lookback // 2:
                return True

            new_is_short = (side == 'short')

            for trade in open_trades:
                if trade.pair == pair:
                    continue  # same pair: handled by Freqtrade's own dup guard
                # Direction match check: only block same-direction stacking
                trade_is_short = bool(getattr(trade, 'is_short', False))
                if trade_is_short != new_is_short:
                    continue  # opposite directions are fine (acts as hedge)

                df_other, _ = self.dp.get_analyzed_dataframe(
                    pair=trade.pair, timeframe=self.timeframe)
                if df_other is None or df_other.empty or len(df_other) < self.corr_lookback + 1:
                    continue
                other_returns = df_other['close'].pct_change().tail(self.corr_lookback).dropna()
                if len(other_returns) < self.corr_lookback // 2:
                    continue

                # Align lengths
                min_len = min(len(new_returns), len(other_returns))
                a = new_returns.tail(min_len).to_numpy()
                b = other_returns.tail(min_len).to_numpy()

                # strict zero to a small epsilon. Avoids the (rare) case
                # where np.corrcoef produces erratic values on series with
                # microscopic but non-zero variance (numerical noise).
                if np.std(a) < 1e-8 or np.std(b) < 1e-8:
                    continue
                corr = float(np.corrcoef(a, b)[0, 1])
                if np.isnan(corr):
                    continue

                if abs(corr) > self.corr_threshold:
                    direction = 'long' if not new_is_short else 'short'
                    _log_info_throttled(
                        f"[ANF_v2] BLOCKED {direction} {pair} — correlated {corr:.2f} "
                        f"with already-open {trade.pair} (threshold={self.corr_threshold})",
                        pair, 60,
                    )
                    return False

            return True
        except Exception as e:
            # Fail-open: an unexpected error in the filter must not block trading.
            logger.warning(f"[ANF_v2] correlation filter error for {pair}: {e}; "
                           f"allowing trade")
            return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        EXIT CONFIRMATION
        Smart gating: allow profitable exits, protect early stops.
        Recognises the tags emitted by the new custom_exit() implementation.
        """
        current_profit_ratio = trade.calc_profit_ratio(rate)
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600

        # acted on with the same urgency as a stoploss).
        if exit_reason in {"stoploss", "stop_loss", "custom_stoploss", "roi",
                           "trailing_stop_loss", "trailing_stop", "force_exit",
                           "emergency_exit"}:
            return True
        if "emergency_ai" in exit_reason:
            return True

        if any(tag in exit_reason for tag in ["mml_", "ai_degradation", "ai_uncertainty",
                                              "exhaustion", "structure_break", "time_rotation"]):
            if current_profit_ratio > -0.05:
                return True

        # Allow opposite-signal exits if not deeply underwater
        if "opposite" in exit_reason:
            if current_profit_ratio > -0.02:
                return True
            if trade_duration > 6:
                return True

        # Block panic exits in first hours while loss is small
        if current_profit_ratio < -0.02 and trade_duration < 2.0:
            return False

        # Default allow
        return True




# =============================================================================
# Running `python ANF_v2.py --audit` from the freqtrade root executes the
# walk-forward audit using ANFAudit and prints to stdout. This is the same
# code path the in-bot trigger uses.
# =============================================================================
def _cli_main(argv=None):
    """CLI entry point. Returns shell exit code (0 OK, 1 if anything OVERFIT)."""
    import argparse
    import json as _json
    import sys as _sys

    ap = argparse.ArgumentParser(
        prog='ANF_v2.py',
        description='ANF strategy — walk-forward ML audit (standalone mode).',
    )
    ap.add_argument('--audit', action='store_true',
                    help='Run walk-forward audit and exit.')
    ap.add_argument('--train-hmm', action='store_true',
                    help='Standalone: download BTC history with ccxt, train a '
                         'frozen HMM regime model, and save the .pkl. Works '
                         'without Freqtrade installed.')
    ap.add_argument('--years', type=float, default=3.0,
                    help='Years of history to download for --train-hmm (default 3).')
    ap.add_argument('--anchor-pair', default='BTC/USDT:USDT',
                    help='Anchor pair for the HMM (default BTC/USDT:USDT).')
    ap.add_argument('--exchange', default='binance',
                    help='ccxt exchange id for --train-hmm (default binance).')
    ap.add_argument('--out', default=None,
                    help='Output .pkl path for --train-hmm. Overrides '
                         '--strategy-name. Default: '
                         'user_data/ml_models/<strategy-name>/hmm_regime.pkl.')
    ap.add_argument('--strategy-name', default='ANF_v2',
                    help='Fork-aware model directory under user_data/ml_models/ '
                         'for --train-hmm output (default ANF_v2). Ignored if '
                         '--out is given.')
    ap.add_argument('--validate-artifact', default=None, metavar='PKL',
                    help='Inspect and verify an existing hmm_regime.pkl '
                         '(sha256, schema, labels, library fingerprint) without '
                         'running the bot. Use before copying to the VPS.')
    ap.add_argument('--no-freeze', action='store_true',
                    help='Save the trained model WITHOUT the frozen flag, so the '
                         'live bot may still refit it on its weekly schedule.')
    ap.add_argument('--config',
                    default='user_data/config_ANF_v2.json',
                    help='Freqtrade config (used to read whitelist + exchange).')
    ap.add_argument('--data-dir', default=None,
                    help='Override data directory. Default: '
                         'user_data/data/<exchange-from-config>.')
    ap.add_argument('--timeframe', default='1h')
    ap.add_argument('--pairs', nargs='*', default=None,
                    help='Override pair list (default: from config whitelist).')
    ap.add_argument('--max-pairs', type=int, default=0,
                    help='Cap number of pairs audited (0 = no cap).')
    ap.add_argument('--result-dir', default=None,
                    help='Where to write audit_last_result.{txt,json}. '
                         'Default: user_data/ml_models/ANF/.  When invoked '
                         'from within a running bot via subprocess, the bot '
                         'passes its own class-name-aware directory here.')
    args = ap.parse_args(argv)

    # Standalone HMM training path: independent of Freqtrade and of any config.
    if args.validate_artifact:
        return _validate_artifact(Path(args.validate_artifact))

    if args.train_hmm:
        return _train_hmm_standalone(
            years=args.years,
            pair=args.anchor_pair,
            timeframe=args.timeframe,
            exchange=args.exchange,
            out_path=args.out,
            freeze=not args.no_freeze,
            strategy_name=args.strategy_name,
        )

    if not args.audit:
        print("ANF strategy module. Use --audit to run the walk-forward audit.",
              file=_sys.stderr)
        print("Or use --train-hmm to train the HMM regime model standalone.",
              file=_sys.stderr)
        print("To run the strategy itself, use Freqtrade:", file=_sys.stderr)
        print("  freqtrade trade --strategy ANF_v2 --config <your-config>",
              file=_sys.stderr)
        return 2

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=_sys.stderr)
        return 2
    with open(cfg_path) as f:
        cfg = _json.load(f)

    pairs = args.pairs or cfg.get('exchange', {}).get('pair_whitelist', [])
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        print("ERROR: no pairs to audit.", file=_sys.stderr)
        return 2

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        exch = cfg.get('exchange', {}).get('name', 'binance')
        data_dir = Path(f"user_data/data/{exch}")
    if not data_dir.exists():
        print(f"WARN: data dir not found: {data_dir}", file=_sys.stderr)
        print(f"      run: freqtrade download-data --timeframes {args.timeframe} "
              "--days 365 --trading-mode futures", file=_sys.stderr)

    # default, but the running bot passes its own class-name-aware directory
    # via the new --result-dir flag so forks don't share state.
    if args.result_dir:
        result_dir = Path(args.result_dir)
    else:
        result_dir = ANF_DEFAULT_ROOT
    result_dir.mkdir(parents=True, exist_ok=True)
    result_txt = result_dir / "audit_last_result.txt"
    result_json = result_dir / "audit_last_result.json"

    auditor = ANFAudit(
        engine_cls=AdvancedPredictiveEngine,
        data_dir=data_dir,
        timeframe=args.timeframe,
    )
    print(f"[audit] auditing {len(pairs)} pairs on {args.timeframe} candles")
    print(f"[audit] window: train={ANFAudit.TRAIN_WIN}  embargo={ANFAudit.EMBARGO}  "
          f"test={ANFAudit.TEST_WIN}  step={ANFAudit.STEP}")
    print(f"[audit] results -> {result_txt}")
    print()

    def cli_progress(i, total, pair, result):
        decay = result.get('decay_pct', float('nan'))
        decay_s = f"{decay*100:.1f}%" if not np.isnan(decay) else "n/a"
        print(f"[{i:>2}/{total}] {pair:<22} "
              f"{result['n_folds']} folds, decay={decay_s} → {result['status']}",
              flush=True)

    results = auditor.run(pairs, progress_callback=cli_progress)
    report = ANFAudit.format_report(results)
    print()
    print(report)

    # Persist results (TXT for humans/Telegram, JSON for downstream tooling)
    try:
        result_txt.write_text(report)
        # Clean NaN/inf for JSON
        def _safe(r):
            out = {}
            for k, v in r.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    out[k] = None
                else:
                    out[k] = v
            return out
        result_json.write_text(_json.dumps([_safe(r) for r in results], indent=2))
        print(f"[audit] wrote {result_txt} and {result_json}")
    except Exception as e:
        print(f"[audit] WARN: could not write result files: {e}", file=_sys.stderr)

    # Errors at the end
    errored = [r for r in results if r.get('errors')]
    if errored:
        print("Errors encountered (first 3 per pair):")
        for r in errored:
            print(f"  {r['pair']}:")
            for msg in r['errors'][:3]:
                print(f"    - {msg}")

    n_overfit = sum(1 for r in results if r['status'] == 'OVERFIT')
    return 1 if n_overfit > 0 else 0


def _validate_artifact(pkl_path):
    """Inspect/verify an HMM .pkl before deploying it to the VPS.

    EN: Checks the sha256 sidecar, loads the model through the same
        HMMRegimeDetector.load() the bot uses (so any reject reason surfaces
        here), and prints schema, labels, frozen flag and library fingerprint.
    ES: Comprueba el sidecar sha256, carga el modelo con el mismo
        HMMRegimeDetector.load() que usa el bot (así cualquier motivo de rechazo
        aparece aquí) e imprime esquema, etiquetas, flag frozen y huella de libs.
    """
    import sys as _sys
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        print(f"ERROR: artifact not found: {pkl_path}", file=_sys.stderr)
        return 2
    if not HMM_AVAILABLE:
        print("ERROR: hmmlearn not installed; cannot validate.", file=_sys.stderr)
        return 2
    # Raw metadata peek (without trusting the model object)
    try:
        blob = pkl_path.read_bytes()
        sig = pkl_path.with_suffix(pkl_path.suffix + '.sha256')
        if sig.exists():
            ok = sig.read_text().strip() == hashlib.sha256(blob).hexdigest()
            print(f"sha256 sidecar: {'OK' if ok else 'MISMATCH'}")
            if not ok:
                print("  -> file corrupted or partial copy; do NOT deploy.")
                return 1
        else:
            print("sha256 sidecar: absent (older model or not generated by this version)")
    except Exception as e:
        print(f"ERROR reading artifact: {e}", file=_sys.stderr)
        return 1
    # Load through the real detector path
    det = HMMRegimeDetector()
    loaded = det.load(pkl_path)
    if not loaded:
        print("RESULT: detector REJECTED the model (see warnings above).")
        return 1
    print("RESULT: model loads cleanly.")
    print(f"  frozen        : {det.frozen}")
    print(f"  n_states      : {det.n_states}")
    print(f"  observables   : {list(det.OBSERVABLE_COLUMNS)}")
    print(f"  state_labels  : {det.state_labels}")
    print(f"  last_train    : {det.last_train_time}")
    print(f"  current libs  : {_library_fingerprint()}")
    print("  (compare 'current libs' with the training machine before deploy)")
    return 0


def _train_hmm_standalone(years=3, pair='BTC/USDT:USDT', timeframe='1h',
                          exchange='binance', out_path=None, freeze=True,
                          strategy_name='ANF_v2'):
    """Download BTC history with ccxt and train a frozen HMM, no Freqtrade needed.

    EN: Standalone trainer. Fetches `years` of OHLCV for the anchor pair via
        ccxt, fits HMMRegimeDetector on the whole history (not the short rolling
        window the live bot uses), and saves a frozen .pkl that the live ANF_v2
        will load and never overwrite. Run this on any machine with
        numpy/pandas/hmmlearn/ccxt; copy the .pkl into the VPS's
        user_data/ml_models/ANF_v2/ directory.
    ES: Entrenador standalone. Descarga `years` de OHLCV del par ancla vía ccxt,
        ajusta HMMRegimeDetector sobre todo el histórico (no la ventana móvil
        corta del bot en vivo), y guarda un .pkl congelado que el ANF_v2 en vivo
        cargará y nunca sobrescribirá. Ejecútalo en cualquier máquina con
        numpy/pandas/hmmlearn/ccxt; copia el .pkl al directorio
        user_data/ml_models/ANF_v2/ del VPS.
    """
    import sys as _sys
    if not HMM_AVAILABLE:
        print("ERROR: hmmlearn not installed. Run: pip install hmmlearn",
              file=_sys.stderr)
        return 2
    try:
        import ccxt
    except ImportError:
        print("ERROR: ccxt not installed. Run: pip install ccxt", file=_sys.stderr)
        return 2

    tf_ms = {'1m': 60_000, '5m': 300_000, '15m': 900_000, '1h': 3_600_000,
             '4h': 14_400_000, '1d': 86_400_000}.get(timeframe, 3_600_000)
    since = int((datetime.now(timezone.utc).timestamp() - years * 365 * 86400) * 1000)

    ex = getattr(ccxt, exchange)({'enableRateLimit': True})
    # Some exchanges need the spot symbol for historical OHLCV; strip the
    # perpetual suffix (BTC/USDT:USDT -> BTC/USDT) for the data fetch.
    fetch_symbol = pair.split(':')[0]
    print(f"[train-hmm] fetching ~{years}y of {fetch_symbol} {timeframe} from {exchange} ...")

    all_rows = []
    cursor = since
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    while cursor < now_ms:
        try:
            batch = ex.fetch_ohlcv(fetch_symbol, timeframe=timeframe,
                                   since=cursor, limit=1000)
        except Exception as e:
            print(f"[train-hmm] fetch error: {e}; retrying once...", file=_sys.stderr)
            try:
                batch = ex.fetch_ohlcv(fetch_symbol, timeframe=timeframe,
                                       since=cursor, limit=1000)
            except Exception as e2:
                print(f"[train-hmm] fetch failed: {e2}", file=_sys.stderr)
                break
        if not batch:
            break
        all_rows.extend(batch)
        cursor = batch[-1][0] + tf_ms
        if len(batch) < 1000:
            break

    if not all_rows:
        print("ERROR: no data fetched.", file=_sys.stderr)
        return 2

    df = pd.DataFrame(all_rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
    print(f"[train-hmm] fetched {len(df)} candles "
          f"({len(df) / (24 if timeframe == '1h' else 1):.0f} days approx).")

    # Fit on the FULL downloaded history. fit() internally tails to
    # fit_lookback, so raise it to cover everything we just downloaded.
    detector = HMMRegimeDetector(fit_lookback=len(df) + 1)
    ok = detector.fit(df)
    if not ok:
        print("ERROR: HMM fit failed (insufficient/degenerate data).", file=_sys.stderr)
        return 1

    if out_path is None:
        out_path = Path("user_data/ml_models") / strategy_name / "hmm_regime.pkl"
    out_path = Path(out_path)
    detector.save(out_path, frozen=freeze)
    print(f"[train-hmm] saved {'frozen ' if freeze else ''}model -> {out_path}")
    print(f"[train-hmm] labels={detector.state_labels}")
    print(f"[train-hmm] lib fingerprint={_library_fingerprint()}")
    print("[train-hmm] copy this .pkl to your VPS user_data/ml_models/ANF_v2/")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(_cli_main())
