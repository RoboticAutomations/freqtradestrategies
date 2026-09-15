# VERSION 11.0.1 by Moutonneux.
from datetime import datetime, timedelta, timezone
import numpy as np
import math
from freqtrade.persistence import Trade
import os

import pandas as pd
from pandas import DataFrame, concat
from datetime import datetime
import time
import numpy as np
from technical import qtpylib
from scipy import stats, signal


import logging
logger = logging.getLogger(__name__)

import warnings

warnings.simplefilter("ignore")

# Installations commands : 
# pip install requests websockets
# git clone https://github.com/rongardF/tvdatafeed
# cd tvdatafeed
# pip install .

from tvDatafeed import TvDatafeed, Interval
from datetime import datetime, timedelta
from pytz import UTC


from freqtrade.strategy import CategoricalParameter

import ta as clean_ta
import talib.abstract as ta


def normalize_btc_dataframe_to_usd(self, dataframe: DataFrame, use_csv=False) -> DataFrame:
    if use_csv:
        if self.btc_dataframe.empty:
            self.btc_dataframe = pd.read_csv("btc_dataframe.csv")
        custom_btc_df = retime_dataframe(dataframe, self.btc_dataframe)
        for column in ["close", "high", "low", "open"]:
            dataframe[f"{column}_old_btc"] = dataframe[column]
            dataframe[column] = dataframe[column] * custom_btc_df[column]
    else :
        if f"{self.timeframe}_BTCUSD" not in dataframe.columns:
            dataframe = add_tv_graph(
                self,
                dataframe=dataframe,
                symbol="BTCUSD",
                timeframes=[self.timeframe],
                exchange="BITSTAMP"
            )
        for column in ["close", "high", "low", "open"]:
            dataframe[f"{column}_old_btc"] = dataframe[column]
            dataframe[column] = dataframe[column] * dataframe[f"{self.timeframe}_BTCUSD"]
    return dataframe


def restore_btc_dataframe_from_usd(dataframe: DataFrame) -> DataFrame:
    for column in ['close', 'high', 'low', 'open']:
        dataframe[column]=dataframe[f'{column}_old_btc']
    return dataframe


def gen_error(error: str, source=None):
    if not source:
        print(error)
    else:
        print(f"{source} - {error}")


LONG_SUFFIXES = {"3L/USDT", "5L/USDT"}
SHORT_SUFFIXES = {"3S/USDT", "5S/USDT"}
def check_gateio_long(pair: str) -> bool:
    return any(suffix in pair for suffix in LONG_SUFFIXES)


def check_gateio_short(pair: str) -> bool:
    return any(suffix in pair for suffix in SHORT_SUFFIXES)


# Cette fonction permet de redimensionner 2 dataframe aux memes dimensions pour les superposer.
def retime_dataframe(dataframe: DataFrame, dataframe_pair: DataFrame):
    df1_dates = set(dataframe.index)
    df2_dates = set(dataframe_pair.index)
    if len(df1_dates) < len(df2_dates):
        X = len(df2_dates) - len(df1_dates)
        df2 = dataframe_pair.iloc[X:]
        df2.reset_indessx(drop=True, inplace=True)
        return df2
    elif len(df1_dates) > len(df2_dates):
        X = len(df1_dates) - len(df2_dates)
        zeros_df = DataFrame(0, index=range(X), columns=dataframe_pair.columns)
        df2 = concat([zeros_df, dataframe_pair], ignore_index=True)
        return df2
    else:
        return dataframe_pair


# Renvoie tous les noms d'indicateurs d'un type spécifique (tendance, vol, over, momentum, ...)
def get_indicator_names_by_type(type: str, indicators: list):
    names=[]
    for indicator in indicators:
        if type in indicator.types:
            names.append(indicator.name)
    return names


def getSuppAndResistanceFromVolume(df, kde_factor: float, num_samples: int, src: int=1) :
    volume = df['volume']

    if src == 1:
        masrc=df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"]+ df["close"] + df["open"]) / 4

    kde = stats.gaussian_kde(masrc, weights=volume, bw_method=kde_factor)
    xr = np.linspace(masrc.min(),masrc.max(),num_samples)
    kdy = kde(xr)
    peaks,_ = signal.find_peaks(kdy)
    return xr[peaks]


def getSuppInformations(dataframe, supports, permissivity: float) :
    dataframe["vpState"] = "nothing"
    for support in supports :
        bottom=float(support-support*permissivity)
        top=float(support+support*permissivity)
        line=support
        dataframe.loc[
            (
                (
                    (dataframe['high'].ge(bottom) & dataframe['low'].le(top))
                    | (dataframe['high'].shift(1).ge(bottom) & dataframe['low'].shift(1).lt(top))
                ) & ((dataframe['high'].ge(bottom) & dataframe['low'].lt(top))==False)
                & (dataframe['high'].lt(bottom) & dataframe['high'].shift(1).ge(bottom))
            ),
            'vpState'] = f"quitte zone {line} en baisse"
        dataframe.loc[
            (
                (
                    (dataframe['high'].ge(bottom) & dataframe['low'].le(top))
                    | (dataframe['high'].shift(1).ge(bottom) & dataframe['low'].shift(1).lt(top))
                ) & ((dataframe['high'].ge(bottom) & dataframe['low'].lt(top))==False)
                & (dataframe['low'].gt(top) & dataframe['low'].shift(1).le(top))
            ),
            'vpState'] = f"quitte zone {line} en hausse"
        dataframe.loc[
            (
                (
                    (dataframe['high'].ge(bottom) & dataframe['low'].le(top))
                    | (dataframe['high'].shift(1).ge(bottom) & dataframe['low'].shift(1).lt(top))
                ) & ((dataframe['high'].shift(1).ge(bottom) & dataframe['low'].shift(1).lt(top))==False)
                & (dataframe['high'].gt(bottom) & dataframe['high'].shift(1).le(bottom))
            ),
            'vpState'] = f"rejoin zone {line} en hausse"
        dataframe.loc[
            (
                (
                    (dataframe['high'].ge(bottom) & dataframe['low'].le(top))
                    | (dataframe['high'].shift(1).ge(bottom) & dataframe['low'].shift(1).lt(top))
                ) & ((dataframe['high'].shift(1).ge(bottom) & dataframe['low'].shift(1).lt(top))==False)
                & (dataframe['low'].lt(top) & dataframe['low'].shift(1).ge(top))
            ),
            'vpState'] = f"rejoin zone {line} en baisse"
    return dataframe['vpState']


def divergence_function(dataframe: DataFrame, size: int, mode: int) -> DataFrame:
    short = int(size/3)
    medium = int(size/3*2)
    long = int(size)
    if mode == 1 :
        return (
            (dataframe['volume'] > dataframe['volume'].shift(+1).rolling(short).max()) &
            (dataframe['volume'].rolling(short).max() < dataframe['volume'].shift(+1).rolling(medium).max()) &
            (dataframe['volume'].rolling(medium).max() < dataframe['volume'].shift(+1).rolling(long).max())
        )
    elif mode == 2 :
        return (
            (dataframe['volume'] > dataframe['volume'].shift(+1).rolling(long).mean()) 
        )
    else :
        return (
            (dataframe['volume'] > dataframe['volume'].shift(+1).rolling(medium).mean()) &
            (dataframe['volume'] > dataframe['volume'].shift(+1).rolling(long).max())
        )

def TRIX(df, length=9, power=11, MAtype=1, src=3):
    if src == 1:
        masrc=df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"]+ df["close"] + df["open"]) / 4

    if MAtype == 1:
        # Triple EMA (Exponential Moving Average)
        df["TRIX"] = clean_ta.trend.ema_indicator(
            close=clean_ta.trend.ema_indicator(
                close=clean_ta.trend.ema_indicator(close=masrc, window=length),
                window=length
            ),
            window=length
        )
    elif MAtype == 2:
        # Triple DEMA (Double Exponential Moving Average) - Approximé via EMA x2
        df["TRIX"] = clean_ta.trend.ema_indicator(
            close=clean_ta.trend.ema_indicator(
                close=clean_ta.trend.ema_indicator(close=masrc, window=length) * 2,
                window=length
            ),
            window=length
        )
    elif MAtype == 3:
        # T3 (Triple Exponential Moving Average with Volume Factor) - Implémenté approximativement
        df["TRIX"] = clean_ta.trend.ema_indicator(
            close=clean_ta.trend.ema_indicator(
                close=clean_ta.trend.ema_indicator(close=masrc, window=length),
                window=length
            ),
            window=length
        ) * 1.3  # Facteur d'adoucissement approximatif pour simuler T3
    elif MAtype == 4:
        # Triple SMA (Simple Moving Average)
        df["TRIX"] = clean_ta.trend.sma_indicator(
            close=clean_ta.trend.sma_indicator(
                close=clean_ta.trend.sma_indicator(close=masrc, window=length),
                window=length
            ),
            window=length
        )
    elif MAtype == 5:
        # Triple TEMA (Triple Exponential Moving Average)
        tema = clean_ta.trend.ema_indicator(
            close=clean_ta.trend.ema_indicator(
                close=clean_ta.trend.ema_indicator(close=masrc, window=length),
                window=length
            ),
            window=length
        )
        df["TRIX"] = tema

    df[f'TRIX_PCT'] = df[f"TRIX"].pct_change()*100
    df[f'TRIX_SIGNAL'] = clean_ta.trend.sma_indicator(df[f'TRIX_PCT'], power)
    df[f'TRIX_HISTO'] = df[f'TRIX_PCT'] - df[f'TRIX_SIGNAL']
    return df


def custom_trix(dataframe, column_name="close", length=9, power=11, MAtype=1):
    masrc=dataframe[column_name]
    if MAtype == 1:
        # Triple EMA (Exponential Moving Average)
        dataframe["TRIX"] = clean_ta.trend.ema_indicator(
            close=clean_ta.trend.ema_indicator(
                close=clean_ta.trend.ema_indicator(close=masrc, window=length),
                window=length
            ),
            window=length
        )
    elif MAtype == 2:
        # Triple DEMA (Double Exponential Moving Average) - Approximé via EMA x2
        dataframe["TRIX"] = clean_ta.trend.ema_indicator(
            close=clean_ta.trend.ema_indicator(
                close=clean_ta.trend.ema_indicator(close=masrc, window=length) * 2,
                window=length
            ),
            window=length
        )
    elif MAtype == 3:
        # T3 (Triple Exponential Moving Average with Volume Factor) - Implémenté approximativement
        dataframe["TRIX"] = clean_ta.trend.ema_indicator(
            close=clean_ta.trend.ema_indicator(
                close=clean_ta.trend.ema_indicator(close=masrc, window=length),
                window=length
            ),
            window=length
        ) * 1.3  # Facteur d'adoucissement approximatif pour simuler T3
    elif MAtype == 4:
        # Triple SMA (Simple Moving Average)
        dataframe["TRIX"] = clean_ta.trend.sma_indicator(
            close=clean_ta.trend.sma_indicator(
                close=clean_ta.trend.sma_indicator(close=masrc, window=length),
                window=length
            ),
            window=length
        )
    elif MAtype == 5:
        # Triple TEMA (Triple Exponential Moving Average)
        tema = clean_ta.trend.ema_indicator(
            close=clean_ta.trend.ema_indicator(
                close=clean_ta.trend.ema_indicator(close=masrc, window=length),
                window=length
            ),
            window=length
        )
        dataframe["TRIX"] = tema

    dataframe[f'TRIX_PCT'] = dataframe[f"TRIX"].pct_change()*100
    dataframe[f'TRIX_SIGNAL'] = clean_ta.trend.sma_indicator(dataframe[f'TRIX_PCT'], power)
    dataframe[f'TRIX_HISTO'] = dataframe[f'TRIX_PCT'] - dataframe[f'TRIX_SIGNAL']
    return dataframe


def normalize_emas(reference: pd.Series, normalize: pd.Series, rolling_size: int) -> pd.Series:
    # Assurez-vous que les données sont des floats et remplacez NaN par 0
    reference = reference.astype(float).fillna(0)
    normalize = normalize.astype(float).fillna(0)

    # Appliquez le calcul de normalisation
    def scale_window(x):
        if x.max() == x.min():  # Évite la division par zéro
            return 0
        return (x[-1] - x.min()) / (x.max() - x.min())

    normalized = normalize.rolling(rolling_size, min_periods=1).apply(scale_window, raw=True)

    # Étendre la normalisation à l'échelle de la référence
    range_ref = reference.rolling(rolling_size, min_periods=1)
    scaled_result = (
        normalized * (range_ref.max() - range_ref.min()) + range_ref.min()
    )

    # Remplir les NaN restants avec backward fill
    return scaled_result.bfill()  # Utilise bfill() au lieu de fillna(method="bfill")


def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']


def lerp(a: float, b: float, t: float) -> float:
    return (1 - t) * a + t * b



def top_percent_change(dataframe: DataFrame, length: int) -> float:
    if length == 0:
        return (dataframe['open'] - dataframe['close']) / dataframe['close']
    else:
        return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']




# Cache manuel pour stocker les résultats
cache = {}
def get_cached_data(self, symbol: str, exchange: str, interval, n_bars: int, this_hour, backtest=True):
    if backtest==True:
        cache_key = (symbol, exchange, interval.value, n_bars)
    else:
        cache_key = (this_hour, symbol, exchange, interval.value, n_bars)
    if cache_key in cache:
        return cache[cache_key]
    try:
        cache[cache_key] = TvDatafeed().get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
    except:
        time.sleep(10)
        try:
            cache[cache_key] = TvDatafeed().get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=int(n_bars/2))
        except:
            time.sleep(20)
            cache[cache_key] = TvDatafeed().get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=int(n_bars/2))
    if not (backtest==True):
        try:
            cache.pop((this_hour-timedelta(hours=1), symbol, exchange, interval.value, n_bars), None)
        except:
            pass
    return cache[cache_key]


def get_from_trading_view(self, dataframe: pd.DataFrame, symbol: str, exchange: str, interval, column_name: str, n_bars=5000, backtest=True) -> pd.DataFrame:
    #this_hour=0
    this_hour=datetime.now().replace(minute=0, second=0, microsecond=0)
    ohlc_types=['close', 'open', 'high', 'low', 'volume']
    if f'{column_name}_close' in dataframe.columns:
        if dataframe[f'{column_name}_close'].max() != -1:
            return dataframe
    tradingview_df = pd.DataFrame(get_cached_data(self, symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars, this_hour=this_hour, backtest=backtest))
    if tradingview_df.empty:
        time.sleep(10)
        tradingview_df = pd.DataFrame(get_cached_data(self, symbol=symbol, exchange=exchange, interval=interval, n_bars=int(n_bars/2), this_hour=this_hour, backtest=backtest))
        if tradingview_df.empty:
            print(f"Aucune donnée de tradingview {symbol} récupérée.")
            dataframe[column_name] = dataframe['close']*0 - 1
            dataframe[f'{column_name}_close'] = dataframe['close']*0 - 1
            return dataframe
    if "symbol" in tradingview_df.columns:
        tradingview_df.drop(columns=["symbol"], inplace=True)
    tradingview_df.index = pd.to_datetime(tradingview_df.index).tz_localize(UTC)
    for ohlc in ohlc_types:
        if ohlc in tradingview_df.columns:
            tradingview_df[ohlc] = tradingview_df[ohlc].shift(1)
            tradingview_df.rename(columns={ohlc: f'{column_name}_{ohlc}'}, inplace=True)
    dataframe = dataframe.copy()
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    if dataframe['date'].dt.tz is None:
        dataframe['date'] = dataframe['date'].dt.tz_localize(UTC)
    dataframe.set_index('date', inplace=True)
    start_date = dataframe.index.min().normalize()
    tradingview_df = tradingview_df.loc[tradingview_df.index >= start_date]
    dataframe = dataframe.join(tradingview_df, how='left')
    for ohlc in ohlc_types:
        if ohlc in tradingview_df.columns:
            dataframe[f'{column_name}_{ohlc}'] = dataframe[f'{column_name}_{ohlc}'].ffill()
    dataframe.reset_index(inplace=True)
    return dataframe


def add_dominance_to_dataframe(self, dataframe: pd.DataFrame, coin:str = 'BTC', timeframes=["1h", "4h", "1d"], backtest=True) -> pd.DataFrame:
    symbol = f'{coin}.D'
    for timeframe in timeframes:
        column=f'{timeframe}_{coin}_dominance'
        if timeframe=="1h":
            interval = Interval.in_1_hour
        elif timeframe=="4h":
            interval = Interval.in_4_hour
        elif timeframe=="2h":
            interval = Interval.in_2_hour
        elif timeframe=="1d":
            interval = Interval.in_daily
    dataframe=get_from_trading_view(self, dataframe=dataframe, symbol=symbol, exchange='CRYPTOCAP', interval=interval, n_bars=5000, column_name=column, backtest=backtest)
    if f'{column}_close' in dataframe.columns:
        dataframe[f'{column}']=dataframe[f'{column}_close'].ffill()
    return dataframe



def add_tv_graph(self, dataframe: pd.DataFrame, symbol, timeframes=["1h", "4h", "1d"], exchange='CRYPTOCAP', backtest=True):
    for timeframe in timeframes:
        if timeframe=="1h":
            interval = Interval.in_1_hour
        elif timeframe=="4h":
            interval = Interval.in_4_hour
        elif timeframe=="2h":
            interval = Interval.in_2_hour
        elif timeframe=="1d":
            interval = Interval.in_daily
    dataframe=get_from_trading_view(self, dataframe=dataframe, symbol=symbol, exchange=exchange, interval=interval, n_bars=5000, column_name=f'{timeframe}_{symbol}', backtest=backtest)
    if f'{timeframe}_{symbol}_close' in dataframe.columns:
        dataframe[f'{timeframe}_{symbol}']=dataframe[f'{timeframe}_{symbol}_close'].ffill()
    return dataframe



def get_tv_instance():
    """Retourne une instance TvDatafeed - créée à chaque appel pour éviter les problèmes de pickle"""
    return TvDatafeed()
import time

def add_global_m2_complete(dataframe: pd.DataFrame, backtest=True):
    csv_cache_path = "global_m2_cache.csv"
    m2_config = {
        # EUROZONE
        'EUM2': {'symbol': 'EUM2', 'exchange': 'ECONOMICS', 'fx': 'EURUSD', 'fx_exchange': 'FX_IDC'},
        
        # NORTH AMERICA
        'USM2': {'symbol': 'USM2', 'exchange': 'ECONOMICS', 'fx': None},  # Déjà en USD
        'CAM2': {'symbol': 'CAM2', 'exchange': 'ECONOMICS', 'fx': 'CADUSD', 'fx_exchange': 'FX_IDC'},
        
        # NON-EU EUROPE
        'CHM2': {'symbol': 'CHM2', 'exchange': 'ECONOMICS', 'fx': 'CHFUSD', 'fx_exchange': 'FX_IDC'},
        'GBM2': {'symbol': 'GBM2', 'exchange': 'ECONOMICS', 'fx': 'GBPUSD', 'fx_exchange': 'FX'},
        'FIM2': {'symbol': 'FIM2', 'exchange': 'ECONOMICS', 'fx': 'USDFIM', 'fx_exchange': 'FX_IDC', 'fx_inverse': True},
        'RUM2': {'symbol': 'RUM2', 'exchange': 'ECONOMICS', 'fx': 'RUBUSD', 'fx_exchange': 'FX_IDC'},
        
        # PACIFIC
        'NZM2': {'symbol': 'NZM2', 'exchange': 'ECONOMICS', 'fx': 'NZDUSD', 'fx_exchange': 'FX_IDC'},
        
        # ASIA
        'CNM2': {'symbol': 'CNM2', 'exchange': 'ECONOMICS', 'fx': 'CNYUSD', 'fx_exchange': 'FX_IDC'},
        'TWM2': {'symbol': 'TWM2', 'exchange': 'ECONOMICS', 'fx': 'TWDUSD', 'fx_exchange': 'FX_IDC'},
        'HKM2': {'symbol': 'HKM2', 'exchange': 'ECONOMICS', 'fx': 'HKDUSD', 'fx_exchange': 'FX_IDC'},
        'INM2': {'symbol': 'INM2', 'exchange': 'ECONOMICS', 'fx': 'INRUSD', 'fx_exchange': 'FX_IDC'},
        'JPM2': {'symbol': 'JPM2', 'exchange': 'ECONOMICS', 'fx': 'JPYUSD', 'fx_exchange': 'FX_IDC'},
        'PHM2': {'symbol': 'PHM2', 'exchange': 'ECONOMICS', 'fx': 'PHPUSD', 'fx_exchange': 'FX_IDC'},
        'SGM2': {'symbol': 'SGM2', 'exchange': 'ECONOMICS', 'fx': 'SGDUSD', 'fx_exchange': 'FX_IDC'},
        
        # LATIN AMERICA
        'BRM2': {'symbol': 'BRM2', 'exchange': 'ECONOMICS', 'fx': 'BRLUSD', 'fx_exchange': 'FX_IDC'},
        'COM2': {'symbol': 'COM2', 'exchange': 'ECONOMICS', 'fx': 'COPUSD', 'fx_exchange': 'FX_IDC'},
        'MXM2': {'symbol': 'MXM2', 'exchange': 'ECONOMICS', 'fx': 'MXNUSD', 'fx_exchange': 'FX_IDC'},
        
        # MIDDLE EAST
        'AEM2': {'symbol': 'AEM2', 'exchange': 'ECONOMICS', 'fx': 'AEDUSD', 'fx_exchange': 'FX_IDC'},
        'TRM2': {'symbol': 'TRM2', 'exchange': 'ECONOMICS', 'fx': 'TRYUSD', 'fx_exchange': 'FX_IDC'},
        
        # AFRICA
        'ZAM2': {'symbol': 'ZAM2', 'exchange': 'ECONOMICS', 'fx': 'ZARUSD', 'fx_exchange': 'FX_IDC'},
    }
    if _check_if_update_needed(csv_cache_path):
        print("Mise à jour des données Global M2 nécessaire...")
        m2_data = _download_all_m2_data(m2_config, backtest)
        _save_m2_cache(m2_data, csv_cache_path)
    m2_data = _load_m2_cache(csv_cache_path)
    
    if m2_data is None or m2_data.empty:
        print("✗ Aucune donnée Global M2 disponible")
        dataframe['global_m2'] = dataframe['close'] * 0 - 1
        return dataframe
    dataframe = _merge_m2_data(dataframe, m2_data)
    
    return dataframe


def _check_if_update_needed(csv_path):
    import os
    if not os.path.exists(csv_path):
        print("Cache CSV n'existe pas, téléchargement nécessaire")
        return True
    file_mod_time = datetime.fromtimestamp(os.path.getmtime(csv_path), tz=timezone.utc)
    now_utc = datetime.now(timezone.utc)
    today_005_utc = now_utc.replace(hour=0, minute=5, second=0, microsecond=0)
    if file_mod_time < today_005_utc:
        print(f"Cache périmé (dernière MAJ: {file_mod_time.strftime('%Y-%m-%d %H:%M UTC')})")
        return True
    return False


def _download_all_m2_data(m2_config, backtest):
    all_data = {}
    failed_symbols = []
    print("Téléchargement des données M2...")
    for m2_name, config in m2_config.items():
        try:
            print(f"  Récupération {m2_name}...")
            m2_data = _download_single_symbol(config['symbol'], config['exchange'], backtest)
            all_data[config['symbol']] = m2_data
            if config['fx'] is not None:
                fx_data = _download_single_symbol(config['fx'], config['fx_exchange'], backtest)
                all_data[config['fx']] = fx_data
                print(f"    ✓ {m2_name} + {config['fx']} téléchargés")
            else:
                print(f"    ✓ {m2_name} téléchargé (USD)")
        except Exception as e:
            print(f"    ✗ Erreur {m2_name}: {e}")
            failed_symbols.append(m2_name)
    if failed_symbols:
        print(f"⚠️  Échecs: {', '.join(failed_symbols)}")
    return all_data


def _download_single_symbol(symbol, exchange, backtest):
    interval = Interval.in_daily
    try:
        tv=get_tv_instance()
        data = tv.get_hist(symbol, exchange, interval, n_bars=5000)
        df = pd.DataFrame(data)
        if not df.empty:
            df.drop(columns=["symbol"], errors="ignore", inplace=True)
            df.index = pd.to_datetime(df.index).tz_localize(UTC)
            df['close'] = df['close'].shift(1)  # Décalage comme dans add_tv_graph
            return df[['close']].rename(columns={'close': symbol})
    except Exception as e:
        print(f"      Erreur téléchargement {symbol}: {e}")
        time.sleep(5)
        try:
            data = tv.get_hist(symbol, exchange, interval, n_bars=5000)
            df = pd.DataFrame(data)
            if not df.empty:
                df.drop(columns=["symbol"], errors="ignore", inplace=True)
                df.index = pd.to_datetime(df.index).tz_localize(UTC)
                df['close'] = df['close'].shift(1)
                return df[['close']].rename(columns={'close': symbol})
        except Exception as e2:
            raise Exception(f"Échec définitif pour {symbol}: {e2}")
    raise Exception(f"Aucune donnée pour {symbol}")


def _save_m2_cache(all_data, csv_path):
    if not all_data:
        print("Aucune donnée à sauvegarder")
        return
    combined_df = None
    for symbol, data in all_data.items():
        if data is not None and not data.empty:
            if combined_df is None:
                combined_df = data.copy()
            else:
                combined_df = combined_df.join(data, how='outer')
    if combined_df is not None:
        combined_df.to_csv(csv_path)
        print(f"✓ Cache sauvegardé: {csv_path} ({len(combined_df)} lignes, {len(combined_df.columns)} colonnes)")
    else:
        print("✗ Aucune donnée valide à sauvegarder")


def _load_m2_cache(csv_path):
    """Charge les données depuis le CSV cache"""
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        
        # Gestion timezone : localiser seulement si pas déjà tz-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize(UTC)
        else:
            df.index = df.index.tz_convert(UTC)
        return df
    except Exception as e:
        print(f"✗ Erreur chargement cache: {e}")
        return None


def _merge_m2_data(dataframe, m2_data):
    m2_config = {
        'EUM2': {'symbol': 'EUM2', 'fx': 'EURUSD'},
        'USM2': {'symbol': 'USM2', 'fx': None},
        'CAM2': {'symbol': 'CAM2', 'fx': 'CADUSD'},
        'CHM2': {'symbol': 'CHM2', 'fx': 'CHFUSD'},
        'GBM2': {'symbol': 'GBM2', 'fx': 'GBPUSD'},
        'FIM2': {'symbol': 'FIM2', 'fx': 'USDFIM', 'fx_inverse': True},
        'RUM2': {'symbol': 'RUM2', 'fx': 'RUBUSD'},
        'NZM2': {'symbol': 'NZM2', 'fx': 'NZDUSD'},
        'CNM2': {'symbol': 'CNM2', 'fx': 'CNYUSD'},
        'TWM2': {'symbol': 'TWM2', 'fx': 'TWDUSD'},
        'HKM2': {'symbol': 'HKM2', 'fx': 'HKDUSD'},
        'INM2': {'symbol': 'INM2', 'fx': 'INRUSD'},
        'JPM2': {'symbol': 'JPM2', 'fx': 'JPYUSD'},
        'PHM2': {'symbol': 'PHM2', 'fx': 'PHPUSD'},
        'SGM2': {'symbol': 'SGM2', 'fx': 'SGDUSD'},
        'BRM2': {'symbol': 'BRM2', 'fx': 'BRLUSD'},
        'COM2': {'symbol': 'COM2', 'fx': 'COPUSD'},
        'MXM2': {'symbol': 'MXM2', 'fx': 'MXNUSD'},
        'AEM2': {'symbol': 'AEM2', 'fx': 'AEDUSD'},
        'TRM2': {'symbol': 'TRM2', 'fx': 'TRYUSD'},
        'ZAM2': {'symbol': 'ZAM2', 'fx': 'ZARUSD'},
    }
    dataframe_temp = dataframe.copy()
    dataframe_temp['date'] = pd.to_datetime(dataframe_temp['date'])
    if dataframe_temp['date'].dt.tz is None:
        dataframe_temp['date'] = dataframe_temp['date'].dt.tz_localize(UTC)
    dataframe_temp.set_index('date', inplace=True)

    start_date = dataframe_temp.index.min().normalize()
    m2_filtered = m2_data[m2_data.index >= start_date]
    dataframe_temp = dataframe_temp.join(m2_filtered, how='left')
    
    global_m2_components = []
    
    for m2_name, config in m2_config.items():
        m2_col = config['symbol']
        fx_col = config['fx']
        
        if m2_col in dataframe_temp.columns:
            if fx_col is None:
                global_m2_components.append(m2_col)
            elif fx_col in dataframe_temp.columns:
                usd_col = f'{m2_name}_USD'
                if config.get('fx_inverse', False):
                    dataframe_temp[usd_col] = dataframe_temp[m2_col] / dataframe_temp[fx_col]
                else:
                    dataframe_temp[usd_col] = dataframe_temp[m2_col] * dataframe_temp[fx_col]
                global_m2_components.append(usd_col)

    if global_m2_components:
        dataframe_temp['global_m2'] = dataframe_temp[global_m2_components].sum(axis=1, skipna=True)
        dataframe_temp['global_m2'] = dataframe_temp['global_m2'].ffill()
    else:
        dataframe_temp['global_m2'] = dataframe_temp.index.to_series() * 0 - 1
        print("✗ Aucune composante M2 disponible pour le calcul")
    
    dataframe_temp.reset_index(inplace=True)
    dataframe['global_m2'] = dataframe_temp['global_m2']
    
    return dataframe


# ========== SYSTÈME DE CACHE POUR US10Y ==========

def add_us10y_complete(dataframe: pd.DataFrame, backtest=True):
    csv_cache_path = "us10y_cache.csv"
    
    if _check_if_update_needed_us10y(csv_cache_path):
        print("Mise à jour des données US10Y nécessaire...")
        us10y_data = _download_us10y_data(backtest)
        _save_us10y_cache(us10y_data, csv_cache_path)
    
    us10y_data = _load_us10y_cache(csv_cache_path)
    
    if us10y_data is None or us10y_data.empty:
        print("✗ Aucune donnée US10Y disponible")
        dataframe['1d_US10Y'] = dataframe['close'] * 0 - 1
        return dataframe
    
    dataframe = _merge_us10y_data(dataframe, us10y_data)
    return dataframe

def _check_if_update_needed_us10y(csv_path):
    import os
    if not os.path.exists(csv_path):
        print("Cache US10Y n'existe pas, téléchargement nécessaire")
        return True
    file_mod_time = datetime.fromtimestamp(os.path.getmtime(csv_path), tz=timezone.utc)
    now_utc = datetime.now(timezone.utc)
    today_005_utc = now_utc.replace(hour=0, minute=5, second=0, microsecond=0)
    if file_mod_time < today_005_utc:
        print(f"Cache US10Y périmé (dernière MAJ: {file_mod_time.strftime('%Y-%m-%d %H:%M UTC')})")
        return True
    return False

def _download_us10y_data(backtest):
    # Liste des symboles US10Y possibles sur TradingView
    us10y_configs = [
        {'symbol': 'US10Y', 'exchange': 'TVC'},
        {'symbol': 'TNX', 'exchange': 'CBOE'},
        {'symbol': 'US10', 'exchange': 'TVC'},
        {'symbol': 'DGS10', 'exchange': 'FRED'},
    ]
    
    for config in us10y_configs:
        try:
            print(f"  Récupération US10Y via {config['symbol']}@{config['exchange']}...")
            tv = get_tv_instance()
            data = tv.get_hist(config['symbol'], config['exchange'], Interval.in_daily, n_bars=5000)
            df = pd.DataFrame(data)
            if not df.empty:
                df.drop(columns=["symbol"], errors="ignore", inplace=True)
                df.index = pd.to_datetime(df.index).tz_localize(UTC)
                df['close'] = df['close'].shift(1)
                print(f"    ✓ US10Y récupéré avec succès via {config['symbol']}@{config['exchange']}")
                return df[['close']].rename(columns={'close': 'US10Y'})
        except Exception as e:
            print(f"    ✗ Erreur US10Y {config['symbol']}@{config['exchange']}: {e}")
            time.sleep(2)
            continue
    
    # Dernier essai avec des paramètres réduits
    try:
        print("  Dernier essai US10Y avec paramètres réduits...")
        tv = get_tv_instance()
        data = tv.get_hist('US10Y', 'TVC', Interval.in_daily, n_bars=1000)
        df = pd.DataFrame(data)
        if not df.empty:
            df.drop(columns=["symbol"], errors="ignore", inplace=True)
            df.index = pd.to_datetime(df.index).tz_localize(UTC)
            df['close'] = df['close'].shift(1)
            print("    ✓ US10Y récupéré avec paramètres réduits")
            return df[['close']].rename(columns={'close': 'US10Y'})
    except Exception as e:
        print(f"    ✗ Échec définitif US10Y: {e}")
    
    print("    ✗ Tous les essais US10Y ont échoué")
    return None

def _save_us10y_cache(data, csv_path):
    if data is not None and not data.empty:
        data.to_csv(csv_path)
        print(f"✓ Cache US10Y sauvegardé: {csv_path} ({len(data)} lignes)")
    else:
        print("✗ Aucune donnée US10Y à sauvegarder")

def _load_us10y_cache(csv_path):
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize(UTC)
        else:
            df.index = df.index.tz_convert(UTC)
        return df
    except Exception as e:
        print(f"✗ Erreur chargement cache US10Y: {e}")
        return None

def _merge_us10y_data(dataframe, us10y_data):
    dataframe_temp = dataframe.copy()
    dataframe_temp['date'] = pd.to_datetime(dataframe_temp['date'])
    if dataframe_temp['date'].dt.tz is None:
        dataframe_temp['date'] = dataframe_temp['date'].dt.tz_localize(UTC)
    dataframe_temp.set_index('date', inplace=True)
    
    start_date = dataframe_temp.index.min().normalize()
    us10y_filtered = us10y_data[us10y_data.index >= start_date]
    dataframe_temp = dataframe_temp.join(us10y_filtered, how='left')
    try:
        dataframe_temp['US10Y'] = dataframe_temp['US10Y'].ffill()
    except Exception as e:
        print(f"Impossible de charger US1OY : dataframe_temp['US10Y'] = dataframe_temp['US10Y'].ffill() | error={e}")
        dataframe_temp['US10Y'] = 1
    
    dataframe_temp.reset_index(inplace=True)
    dataframe['1d_US10Y'] = dataframe_temp['US10Y']
    return dataframe

# ========== SYSTÈME DE CACHE POUR VIX ==========

def add_vix_complete(dataframe: pd.DataFrame, backtest=True):
    csv_cache_path = "vix_cache.csv"
    
    if _check_if_update_needed_vix(csv_cache_path):
        print("Mise à jour des données VIX nécessaire...")
        vix_data = _download_vix_data(backtest)
        _save_vix_cache(vix_data, csv_cache_path)
    
    vix_data = _load_vix_cache(csv_cache_path)
    
    if vix_data is None or vix_data.empty:
        print("✗ Aucune donnée VIX disponible")
        dataframe['1d_VIX'] = dataframe['close'] * 0 - 1
        return dataframe
    
    dataframe = _merge_vix_data(dataframe, vix_data)
    return dataframe

def _check_if_update_needed_vix(csv_path):
    import os
    if not os.path.exists(csv_path):
        print("Cache VIX n'existe pas, téléchargement nécessaire")
        return True
    file_mod_time = datetime.fromtimestamp(os.path.getmtime(csv_path), tz=timezone.utc)
    now_utc = datetime.now(timezone.utc)
    today_005_utc = now_utc.replace(hour=0, minute=5, second=0, microsecond=0)
    if file_mod_time < today_005_utc:
        print(f"Cache VIX périmé (dernière MAJ: {file_mod_time.strftime('%Y-%m-%d %H:%M UTC')})")
        return True
    return False

def _download_vix_data(backtest):
    # Liste des symboles VIX possibles sur TradingView
    vix_configs = [
        {'symbol': 'VIX', 'exchange': 'CBOE'},
        {'symbol': 'VIX', 'exchange': 'TVC'},
        {'symbol': 'VIXCLS', 'exchange': 'FRED'},
        {'symbol': 'VX1!', 'exchange': 'CBOE'},
    ]
    
    for config in vix_configs:
        try:
            print(f"  Récupération VIX via {config['symbol']}@{config['exchange']}...")
            tv = get_tv_instance()
            data = tv.get_hist(config['symbol'], config['exchange'], Interval.in_daily, n_bars=5000)
            df = pd.DataFrame(data)
            if not df.empty:
                df.drop(columns=["symbol"], errors="ignore", inplace=True)
                df.index = pd.to_datetime(df.index).tz_localize(UTC)
                df['close'] = df['close'].shift(1)
                print(f"    ✓ VIX récupéré avec succès via {config['symbol']}@{config['exchange']}")
                return df[['close']].rename(columns={'close': 'VIX'})
        except Exception as e:
            print(f"    ✗ Erreur VIX {config['symbol']}@{config['exchange']}: {e}")
            time.sleep(2)
            continue
    
    # Dernier essai avec des paramètres réduits
    try:
        print("  Dernier essai VIX avec paramètres réduits...")
        tv = get_tv_instance()
        data = tv.get_hist('VIX', 'CBOE', Interval.in_daily, n_bars=1000)
        df = pd.DataFrame(data)
        if not df.empty:
            df.drop(columns=["symbol"], errors="ignore", inplace=True)
            df.index = pd.to_datetime(df.index).tz_localize(UTC)
            df['close'] = df['close'].shift(1)
            print("    ✓ VIX récupéré avec paramètres réduits")
            return df[['close']].rename(columns={'close': 'VIX'})
    except Exception as e:
        print(f"    ✗ Échec définitif VIX: {e}")
    
    print("    ✗ Tous les essais VIX ont échoué")
    return None

def _save_vix_cache(data, csv_path):
    if data is not None and not data.empty:
        data.to_csv(csv_path)
        print(f"✓ Cache VIX sauvegardé: {csv_path} ({len(data)} lignes)")
    else:
        print("✗ Aucune donnée VIX à sauvegarder")

def _load_vix_cache(csv_path):
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize(UTC)
        else:
            df.index = df.index.tz_convert(UTC)
        return df
    except Exception as e:
        print(f"✗ Erreur chargement cache VIX: {e}")
        return None

def _merge_vix_data(dataframe, vix_data):
    dataframe_temp = dataframe.copy()
    dataframe_temp['date'] = pd.to_datetime(dataframe_temp['date'])
    if dataframe_temp['date'].dt.tz is None:
        dataframe_temp['date'] = dataframe_temp['date'].dt.tz_localize(UTC)
    dataframe_temp.set_index('date', inplace=True)
    
    start_date = dataframe_temp.index.min().normalize()
    vix_filtered = vix_data[vix_data.index >= start_date]
    dataframe_temp = dataframe_temp.join(vix_filtered, how='left')
    dataframe_temp['VIX'] = dataframe_temp['VIX'].ffill()
    
    dataframe_temp.reset_index(inplace=True)
    dataframe['1d_VIX'] = dataframe_temp['VIX']
    return dataframe

def add_walcl_complete(dataframe: pd.DataFrame, backtest=True):
    csv_cache_path = "walcl_cache.csv"
    
    if _check_if_update_needed_walcl(csv_cache_path):
        print("Mise à jour des données WALCL nécessaire...")
        walcl_data = _download_walcl_data(backtest)
        _save_walcl_cache(walcl_data, csv_cache_path)
    
    walcl_data = _load_walcl_cache(csv_cache_path)
    
    if walcl_data is None or walcl_data.empty:
        print("✗ Aucune donnée WALCL disponible")
        dataframe['1w_WALCL'] = dataframe['close'] * 0 - 1
        return dataframe
    
    # Vérifier que WALCL existe dans les données
    if 'WALCL' not in walcl_data.columns:
        print("✗ Colonne WALCL manquante dans les données")
        dataframe['1w_WALCL'] = dataframe['close'] * 0 - 1
        return dataframe
    
    dataframe = _merge_walcl_data(dataframe, walcl_data)
    return dataframe

def _check_if_update_needed_walcl(csv_path):
    import os
    if not os.path.exists(csv_path):
        print("Cache WALCL n'existe pas, téléchargement nécessaire")
        return True
    file_mod_time = datetime.fromtimestamp(os.path.getmtime(csv_path), tz=timezone.utc)
    now_utc = datetime.now(timezone.utc)
    today_005_utc = now_utc.replace(hour=0, minute=5, second=0, microsecond=0)
    if file_mod_time < today_005_utc:
        print(f"Cache WALCL périmé (dernière MAJ: {file_mod_time.strftime('%Y-%m-%d %H:%M UTC')})")
        return True
    return False

def _download_walcl_data(backtest):
    # Liste des symboles WALCL possibles sur TradingView
    walcl_configs = [
        {'symbol': 'WALCL', 'exchange': 'FRED'},
        {'symbol': 'WALCL', 'exchange': 'ECONOMICS'},
        {'symbol': 'FRED:WALCL', 'exchange': 'FRED'},
        {'symbol': 'USCBBALANCESHEET', 'exchange': 'ECONOMICS'},
    ]
    
    for config in walcl_configs:
        try:
            print(f"  Récupération WALCL via {config['symbol']}@{config['exchange']}...")
            tv = get_tv_instance()
            data = tv.get_hist(config['symbol'], config['exchange'], Interval.in_weekly, n_bars=5000)
            df = pd.DataFrame(data)
            if not df.empty:
                df.drop(columns=["symbol"], errors="ignore", inplace=True)
                df.index = pd.to_datetime(df.index).tz_localize(UTC)
                df['close'] = df['close'].shift(1)
                print(f"    ✓ WALCL récupéré avec succès via {config['symbol']}@{config['exchange']}")
                return df[['close']].rename(columns={'close': 'WALCL'})
        except Exception as e:
            print(f"    ✗ Erreur WALCL {config['symbol']}@{config['exchange']}: {e}")
            time.sleep(2)
            continue
    
    # Dernier essai avec des paramètres réduits
    try:
        print("  Dernier essai WALCL avec paramètres réduits...")
        tv = get_tv_instance()
        data = tv.get_hist('WALCL', 'FRED', Interval.in_weekly, n_bars=500)
        df = pd.DataFrame(data)
        if not df.empty:
            df.drop(columns=["symbol"], errors="ignore", inplace=True)
            df.index = pd.to_datetime(df.index).tz_localize(UTC)
            df['close'] = df['close'].shift(1)
            print("    ✓ WALCL récupéré avec paramètres réduits")
            return df[['close']].rename(columns={'close': 'WALCL'})
    except Exception as e:
        print(f"    ✗ Échec définitif WALCL: {e}")
    
    print("    ✗ Tous les essais WALCL ont échoué")
    return None

def _save_walcl_cache(data, csv_path):
    try:
        if data is not None and not data.empty:
            data.to_csv(csv_path)
            print(f"✓ Cache WALCL sauvegardé: {csv_path} ({len(data)} lignes)")
        else:
            print("✗ Aucune donnée WALCL à sauvegarder")
            # Créer un fichier vide pour éviter les retéléchargements constants
            import pandas as pd
            empty_df = pd.DataFrame({'WALCL': []})
            empty_df.to_csv(csv_path)
            print(f"✓ Cache WALCL vide créé: {csv_path}")
    except Exception as e:
        print(f"✗ Erreur sauvegarde cache WALCL: {e}")

def _load_walcl_cache(csv_path):
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize(UTC)
        else:
            df.index = df.index.tz_convert(UTC)
        
        # S'assurer que la colonne WALCL existe
        if 'WALCL' not in df.columns and len(df.columns) > 0:
            # Si la première colonne n'est pas nommée WALCL, la renommer
            df.rename(columns={df.columns[0]: 'WALCL'}, inplace=True)
        
        return df
    except Exception as e:
        print(f"✗ Erreur chargement cache WALCL: {e}")
        return None


def _merge_walcl_data(dataframe, walcl_data):
    dataframe_temp = dataframe.copy()
    dataframe_temp['date'] = pd.to_datetime(dataframe_temp['date'])
    if dataframe_temp['date'].dt.tz is None:
        dataframe_temp['date'] = dataframe_temp['date'].dt.tz_localize(UTC)
    dataframe_temp.set_index('date', inplace=True)
    
    start_date = dataframe_temp.index.min().normalize()
    walcl_filtered = walcl_data[walcl_data.index >= start_date]
    
    # Éviter les conflits de colonnes en supprimant les colonnes WALCL existantes
    walcl_columns = [col for col in dataframe_temp.columns if col.startswith('WALCL') or col == '1w_WALCL']
    if walcl_columns:
        dataframe_temp = dataframe_temp.drop(columns=walcl_columns)
    
    # S'assurer que walcl_filtered contient bien la colonne WALCL
    if walcl_filtered.empty or 'WALCL' not in walcl_filtered.columns:
        logger.warning("WALCL data is empty or missing WALCL column")
        dataframe_temp['WALCL'] = -1
    else:
        dataframe_temp = dataframe_temp.join(walcl_filtered[['WALCL']], how='left')
        # Remplir les valeurs manquantes
        if 'WALCL' in dataframe_temp.columns:
            dataframe_temp['WALCL'] = dataframe_temp['WALCL'].ffill()
        else:
            dataframe_temp['WALCL'] = -1
    
    dataframe_temp.reset_index(inplace=True)
    dataframe['1w_WALCL'] = dataframe_temp['WALCL'] if 'WALCL' in dataframe_temp.columns else -1
    
    # Si WALCL est toujours -1 ou NaN, on le remplit avec une valeur par défaut
    if dataframe['1w_WALCL'].isna().all() or (dataframe['1w_WALCL'] == -1).all():
        logger.warning("WALCL data could not be merged properly, using default values")
        dataframe['1w_WALCL'] = dataframe['close'] * 0 - 1
    
    return dataframe


# ========== FONCTIONS WALCL ==========

def fetch_walcl_from_tv():
    """
    Télécharge les données WALCL depuis TradingView
    Retourne un DataFrame avec les données WALCL ou None si échec
    """
    from tvDatafeed import TvDatafeed, Interval
    from pytz import UTC
    
    walcl_configs = [
        {'symbol': 'WALCL', 'exchange': 'FRED'},
        {'symbol': 'WALCL', 'exchange': 'ECONOMICS'},
        {'symbol': 'FRED:WALCL', 'exchange': 'FRED'},
        {'symbol': 'USCBBALANCESHEET', 'exchange': 'ECONOMICS'},
    ]
    
    for config in walcl_configs:
        tv = TvDatafeed()
        data = tv.get_hist(config['symbol'], config['exchange'], Interval.in_weekly, n_bars=1000)
        if data is not None and not data.empty:
            walcl_df = pd.DataFrame(data)
            if 'close' in walcl_df.columns:
                walcl_df.index = pd.to_datetime(walcl_df.index).tz_localize(UTC)
                walcl_df['WALCL'] = walcl_df['close'].shift(1)  # Shift pour éviter le look-ahead bias
                logger.info(f"✓ WALCL récupéré via {config['symbol']}@{config['exchange']}")
                return walcl_df[['WALCL']]
        else:
            logger.debug(f"Échec WALCL {config['symbol']}@{config['exchange']}")
            time.sleep(1)
            continue
    
    logger.warning("✗ Impossible de récupérer WALCL depuis TradingView")
    return None

# ========== FONCTIONS TOTAL2ES ==========

def fetch_total2es_from_tv(use_hourly=False):
    """
    Télécharge TOTAL2ES depuis TradingView
    Args:
        use_hourly: Si True, télécharge en 1h, sinon en 1d
    """
    from tvDatafeed import TvDatafeed, Interval
    from pytz import UTC

    tv = TvDatafeed()
    
    # Choisir l'intervalle selon le paramètre
    interval = Interval.in_1_hour if use_hourly else Interval.in_daily
    n_bars = 5000 if use_hourly else 5000
    
    data = tv.get_hist('TOTAL2ES', 'CRYPTOCAP', interval, n_bars=n_bars)
    if data is not None and not data.empty:
        df = pd.DataFrame(data).rename(str.lower, axis=1)

        if 'close' in df.columns:
            df.index = pd.to_datetime(df.index).tz_localize(UTC)
            df['TOTAL2ES'] = df['close'].shift(1)

            # S'il n'y a pas de volume, on remplit avec 1 pour que le VWAP fonctionne
            if 'volume' in df.columns:
                df['TOTAL2ES_volume'] = df['volume'].shift(1)
            else:
                df['TOTAL2ES_volume'] = 1.0

            interval_str = "1h" if use_hourly else "1d"
            logger.info(f"✓ TOTAL2ES récupéré en {interval_str}")
            return df[['TOTAL2ES', 'TOTAL2ES_volume']]

    logger.warning("✗ TOTAL2ES introuvable ou vide")
    return None
# ========== SYSTÈME DE CACHE POUR TOTAL2ES ==========

def add_total2es_complete(dataframe: pd.DataFrame, timeframe="4h"):
    """
    Ajoute les données TOTAL2ES au dataframe avec support multi-timeframe
    Args:
        dataframe: DataFrame principal
        timeframe: Timeframe de la stratégie (1h, 2h, 4h, 1d)
    """
    # Adapter le cache selon la timeframe
    csv_cache_path = f"total2es_{timeframe}_cache.csv"
    
    if _check_if_update_needed_total2es(csv_cache_path):
        print(f"Mise à jour des données TOTAL2ES nécessaire... (timeframe: {timeframe})")
        total2es_data = _download_total2es_data(timeframe=timeframe)
        _save_total2es_cache(total2es_data, csv_cache_path)
    
    total2es_data = _load_total2es_cache(csv_cache_path)
    
    if total2es_data is None or total2es_data.empty:
        print("✗ Aucune donnée TOTAL2ES disponible")
        dataframe['TOTAL2ES'] = -1
        dataframe['TOTAL2ES_volume'] = -1
        return dataframe
    
    dataframe = _merge_total2es_data(dataframe, total2es_data)
    return dataframe

def _check_if_update_needed_total2es(csv_path):
    import os
    if not os.path.exists(csv_path):
        print("Cache TOTAL2ES n'existe pas, téléchargement nécessaire")
        return True
    
    # Vérifier si le fichier est vide ou corrompu
    if os.path.getsize(csv_path) < 100:
        return True
    
    file_mod_time = datetime.fromtimestamp(os.path.getmtime(csv_path), tz=timezone.utc)
    now_utc = datetime.now(timezone.utc)
    # Mettre à jour une fois par jour après 00:05 UTC
    today_005_utc = now_utc.replace(hour=0, minute=5, second=0, microsecond=0)
    
    if file_mod_time < today_005_utc:
        print(f"Cache TOTAL2ES périmé (dernière MAJ: {file_mod_time.strftime('%Y-%m-%d %H:%M UTC')})")
        return True
    return False

def _download_total2es_data(timeframe="4h"):
    """
    Télécharge TOTAL2ES depuis TradingView
    Args:
        timeframe: Timeframe à utiliser (1h, 2h, 4h, 1d)
    """
    from tvDatafeed import TvDatafeed, Interval
    from pytz import UTC

    tv = get_tv_instance()
    
    # Mapper la timeframe vers l'intervalle TradingView
    # Note: TradingView n'a pas tous les intervalles, on prend le plus proche inférieur
    timeframe_mapping = {
        '1h': Interval.in_1_hour,
        '2h': Interval.in_1_hour,  # On téléchargera en 1h et resamplera
        '4h': Interval.in_4_hour,
        '1d': Interval.in_daily,
    }
    
    interval = timeframe_mapping.get(timeframe, Interval.in_4_hour)
    
    # Ajuster le nombre de barres selon la timeframe
    # Pour 2h, on double le nombre de barres 1h
    n_bars_mapping = {
        '1h': 5000,
        '2h': 10000,  # Double car on va resampler
        '4h': 5000,
        '1d': 1000,
    }
    n_bars = n_bars_mapping.get(timeframe, 5000)
    
    try:
        data = tv.get_hist('TOTAL2ES', 'CRYPTOCAP', interval, n_bars=n_bars)
        if data is not None and not data.empty:
            df = pd.DataFrame(data).rename(str.lower, axis=1)
            
            if 'close' in df.columns:
                df.index = pd.to_datetime(df.index).tz_localize(UTC)
                
                # Si on veut du 2h et qu'on a téléchargé en 1h, resampler
                if timeframe == '2h' and interval == Interval.in_1_hour:
                    df_resampled = pd.DataFrame()
                    df_resampled['close'] = df['close'].resample('2h').last()
                    if 'volume' in df.columns:
                        df_resampled['volume'] = df['volume'].resample('2h').sum()
                    df = df_resampled
                
                # Shift pour éviter le look-ahead bias
                df['TOTAL2ES'] = df['close'].shift(1)
                
                # S'il n'y a pas de volume, on remplit avec 1 pour que le VWAP fonctionne
                if 'volume' in df.columns:
                    df['TOTAL2ES_volume'] = df['volume'].shift(1)
                else:
                    df['TOTAL2ES_volume'] = 1.0
                
                print(f"✓ TOTAL2ES récupéré en {timeframe}")
                return df[['TOTAL2ES', 'TOTAL2ES_volume']]
    except Exception as e:
        print(f"✗ Erreur TOTAL2ES: {e}")
    
    print("✗ TOTAL2ES introuvable ou vide")
    return None

def _save_total2es_cache(data, csv_path):
    """Sauvegarde les données TOTAL2ES dans le cache CSV"""
    try:
        if data is not None and not data.empty:
            # Créer un fichier temporaire d'abord
            temp_path = csv_path + '.tmp'
            data.to_csv(temp_path)
            
            # Vérifier que le fichier temporaire est valide
            test_df = pd.read_csv(temp_path, index_col=0)
            if not test_df.empty:
                # Remplacer l'ancien fichier
                os.replace(temp_path, csv_path)
                print(f"✓ Cache TOTAL2ES sauvegardé: {csv_path} ({len(data)} lignes)")
                return True
            else:
                os.remove(temp_path)
                print(f"Fichier cache {csv_path} invalide")
                return False
    except Exception as e:
        print(f"✗ Erreur sauvegarde cache TOTAL2ES: {e}")
    return False



def _load_total2es_cache(csv_path):
    """Charge les données TOTAL2ES depuis le cache CSV"""
    try:
        from pytz import UTC
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        if df.empty:
            print(f"Cache {csv_path} vide")
            return None
        
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize(UTC)
        else:
            df.index = df.index.tz_convert(UTC)
        
        return df
    except Exception as e:
        print(f"✗ Erreur chargement cache TOTAL2ES: {e}")
        return None

def _merge_total2es_data(dataframe, total2es_data):
    """Merge les données TOTAL2ES avec le dataframe principal"""
    from pytz import UTC
    
    # Préparer le dataframe pour le merge
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    if dataframe['date'].dt.tz is None:
        dataframe['date'] = dataframe['date'].dt.tz_localize(UTC)
    
    # Sauvegarder l'ordre original
    dataframe['_original_index'] = dataframe.index
    
    # Trier par date pour merge_asof
    dataframe = dataframe.sort_values('date')
    total2es_data_copy = total2es_data.copy()
    total2es_data_copy['date'] = total2es_data_copy.index
    total2es_data_copy = total2es_data_copy.sort_values('date')
    
    # Utiliser merge_asof pour éviter le look-ahead bias
    merged = pd.merge_asof(
        dataframe,
        total2es_data_copy,
        on='date',
        direction='backward'  # Prendre la valeur la plus récente disponible dans le passé
    )
    
    # Restaurer l'ordre original
    merged = merged.sort_values('_original_index')
    merged.index = merged['_original_index']
    merged.drop('_original_index', axis=1, inplace=True)
    
    # Forward fill pour combler les gaps
    if 'TOTAL2ES' in merged.columns:
        merged['TOTAL2ES'] = merged['TOTAL2ES'].ffill()
    if 'TOTAL2ES_volume' in merged.columns:
        merged['TOTAL2ES_volume'] = merged['TOTAL2ES_volume'].ffill()
    
    return merged

def calculate_vwap(df: pd.DataFrame, price_col: str, volume_col: str, period: int) -> pd.Series:
    """
    Calcule le VWAP (Volume Weighted Average Price) sur une période donnée
    """
    # Calculer le produit prix × volume
    df['pv'] = df[price_col] * df[volume_col]
    
    # Calculer les sommes cumulées sur la période
    cumsum_pv = df['pv'].rolling(window=period).sum()
    cumsum_vol = df[volume_col].rolling(window=period).sum()
    
    # VWAP = Somme(Prix × Volume) / Somme(Volume)
    vwap = cumsum_pv / cumsum_vol
    
    # Nettoyer la colonne temporaire
    df.drop('pv', axis=1, inplace=True, errors='ignore')
    
    return vwap



def add_total2es_with_vwap(dataframe: pd.DataFrame, timeframe="4h", vwap_period=2, band_std=1.055):
    """
    Ajoute TOTAL2ES avec VWAP et bandes au dataframe
    Args:
        dataframe: DataFrame principal
        timeframe: Timeframe de la stratégie
        vwap_period: Période pour le calcul du VWAP (en nombre de périodes de la timeframe)
        band_std: Multiplicateur pour l'écart-type des bandes
    """
    # Ajouter les données TOTAL2ES de base
    dataframe = add_total2es_complete(dataframe, timeframe=timeframe)
    
    if 'TOTAL2ES' in dataframe.columns and dataframe['TOTAL2ES'].iloc[-1] > 0:
        # La période est déjà dans la bonne unité (nombre de bougies de la timeframe)
        adjusted_period = vwap_period
        
        # Calculer VWAP
        dataframe['TOTAL2ES_vwap'] = calculate_vwap(
            dataframe,
            'TOTAL2ES',
            'TOTAL2ES_volume',
            period=adjusted_period
        )
        
        # Calculer l'écart-type et la bande inférieure
        dataframe['TOTAL2ES_std'] = dataframe['TOTAL2ES'].rolling(adjusted_period).std()
        dataframe['TOTAL2ES_lower_band'] = (
            dataframe['TOTAL2ES_vwap'] - band_std * dataframe['TOTAL2ES_std']
        )
        
        # Nettoyer la colonne temporaire
        dataframe.drop('TOTAL2ES_std', axis=1, inplace=True, errors='ignore')
    else:
        dataframe['TOTAL2ES_vwap'] = -1
        dataframe['TOTAL2ES_lower_band'] = -1
    
    return dataframe

