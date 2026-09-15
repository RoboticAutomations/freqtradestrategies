import logging
from functools import reduce
import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)


class DarkProphetRL(IStrategy):

    """
    $$$$$$$  |  ______    ______  $$ |   __ $$$$$$$  |  ______    ______    ______    ______    ______  
    $$ |  $$ | /      \  /      \ $$ |  /  |$$ |__$$ | /      \  /      \  /      \  /      \  /      \ 
    $$ |  $$ | $$$$$$  |/$$$$$$  |$$ |_/$$/ $$    $$< /$$$$$$  | $$$$$$  |/$$$$$$  |/$$$$$$  |/$$$$$$  |
    $$ |  $$ | /    $$ |$$ |  $$/ $$   $$<  $$$$$$$  |$$    $$ | /    $$ |$$ |  $$ |$$    $$ |$$ |  $$/ 
    $$ |__$$ |/$$$$$$$ |$$ |      $$$$$$  \ $$ |  $$ |$$$$$$$$/ /$$$$$$$ |$$ |__$$ |$$$$$$$$/ $$ |      
    $$    $$/ $$    $$ |$$ |      $$ | $$  |$$ |  $$ |$$       |$$    $$ |$$    $$/ $$       |$$ |      
    $$$$$$$/   $$$$$$$/ $$/       $$/   $$/ $$/   $$/  $$$$$$$/  $$$$$$$/ $$$$$$$/   $$$$$$$/ $$/           
                                                                          $$$$$
                                                                          $$$$$
                                                                          $$$$$
    """

    INTERFACE_VERSION = 3
    process_only_new_candles = True
    stoploss = -0.15
    use_exit_signal = True
    startup_candle_count: int = 500
    can_short = True

    @staticmethod
    def _safe_div(num, den, default: float = 0.0):
        den = pd.Series(den).replace([0.0, np.inf, -np.inf], np.nan)
        out = pd.Series(num) / den
        return out.replace([np.inf, -np.inf], np.nan).fillna(default)

    @staticmethod
    def _zscore(series: pd.Series, window: int):
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return (series - mean) / (std + 1e-9)

    @staticmethod
    def _donchian(df: DataFrame, n: int):
        high_n = df["high"].rolling(n).max()
        low_n = df["low"].rolling(n).min()
        rng = (high_n - low_n).replace(0, np.nan)
        pos = (df["close"] - low_n) / rng
        return high_n, low_n, pos

    @staticmethod
    def _vwap_bands(df: DataFrame, window: int = 20, stdev_mult: float = 1.0):

        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        vol = df["volume"]
        pv = (tp * vol).rolling(window).sum()
        v = vol.rolling(window).sum()
        vwap = np.where(v != 0, pv / v, np.nan)

        var_num = (vol * (tp - vwap) ** 2).rolling(window).sum()
        vwapsd = np.sqrt(np.where(v != 0, var_num / v, np.nan))

        return (vwap - stdev_mult * vwapsd, vwap, vwap + stdev_mult * vwapsd)

    @staticmethod
    def _linreg_slope(series: pd.Series, window: int = 50):

        x = np.arange(window, dtype=float)
        x_c = x - x.mean()
        denom = float(np.sum(x_c * x_c) + 1e-12)

        def _calc(y):
            y = np.asarray(y, dtype=float)
            if np.sum(np.isfinite(y)) < window:
                return np.nan
            ymean = float(np.nanmean(y) + 1e-9)
            y_c = y - np.nanmean(y)
            b1 = float(np.nansum(x_c * y_c) / denom)
            return b1 / ymean

        return series.rolling(window).apply(lambda s: _calc(s.values), raw=False)

    @staticmethod
    def _linreg_r2(series: pd.Series, window: int = 50):

        x = np.arange(window, dtype=float)
        def _calc(y):
            y = np.asarray(y, dtype=float)
            if np.sum(np.isfinite(y)) < window:
                return np.nan
            b1, b0 = np.polyfit(x, y, 1)
            yhat = b0 + b1 * x
            sst = float(np.sum((y - y.mean()) ** 2) + 1e-9)
            ssr = float(np.sum((yhat - y.mean()) ** 2))
            return ssr / sst
        return series.rolling(window).apply(lambda s: _calc(s.values), raw=False)

    @staticmethod
    def _hurst_rs(series: pd.Series, window: int = 50):
        def _calc(y):
            y = pd.Series(y)
            y = y.replace([np.inf, -np.inf], np.nan).dropna()
            if len(y) < window:
                return np.nan
            r = y - y.mean()
            z = r.cumsum()
            R = z.max() - z.min()
            S = y.std(ddof=0)
            if S == 0:
                return np.nan
            return np.log(R / S) / np.log(len(y))
        return series.rolling(window).apply(lambda s: _calc(s.values), raw=False)

    @staticmethod
    def _autocorr(series: pd.Series, lag: int, window: int = 50):
        def _ac(y):
            s = pd.Series(y)
            return s.autocorr(lag=lag)
        return series.rolling(window).apply(lambda s: _ac(s.values), raw=False).fillna(0.0)

    @staticmethod
    def _entropy_norm(series: pd.Series, window: int = 100, bins: int = 20):

        def _ent(y):
            y = pd.Series(y).replace([np.inf, -np.inf], np.nan).dropna()
            if len(y) < 10:
                return np.nan
            hist, _ = np.histogram(y, bins=bins, density=True)
            p = hist / (np.sum(hist) + 1e-9)
            p = p[p > 0]
            H = -np.sum(p * np.log(p))
            return H / np.log(bins)
        return series.rolling(window).apply(lambda s: _ent(s.values), raw=False)

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs):

        df = dataframe

        df["%-raw_close"] = df["close"]
        df["%-raw_open"] = df["open"]
        df["%-raw_high"] = df["high"]
        df["%-raw_low"] = df["low"]
        df["%-raw_volume"] = df["volume"]

        if "date" in df.columns and hasattr(df["date"], "dt"):
            df["%-day_of_week"] = df["date"].dt.dayofweek
            df["%-hour_of_day"] = df["date"].dt.hour
            df["%-dow_sin"] = np.sin(2 * np.pi * df["%-day_of_week"] / 7.0)
            df["%-dow_cos"] = np.cos(2 * np.pi * df["%-day_of_week"] / 7.0)
            df["%-hod_sin"] = np.sin(2 * np.pi * df["%-hour_of_day"] / 24.0)
            df["%-hod_cos"] = np.cos(2 * np.pi * df["%-hour_of_day"] / 24.0)


        df["%-pct_change_1"] = df["close"].pct_change()
        df["%-pct_change_3"] = df["close"].pct_change(3)
        df["%-pct_change_6"] = df["close"].pct_change(6)
        df["%-logret_1"] = np.log(self._safe_div(df["close"], df["close"].shift(1), default=1.0))

        lower = np.minimum(df["open"], df["close"])
        upper = np.maximum(df["open"], df["close"])
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        df["%-body"] = self._safe_div(upper - lower, rng)
        df["%-tail"] = self._safe_div(lower - df["low"], rng)
        df["%-wick"] = self._safe_div(df["high"] - upper, rng)

        body_mid = (lower + upper) / 2.0
        df["%-microprice_proxy_dev"] = self._safe_div(df["close"] - body_mid, df["close"])

        return df

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs):

        df = dataframe

        # === MAs ===
        df["ema_8"] = ta.EMA(df["close"], timeperiod=8)
        df["ema_21"] = ta.EMA(df["close"], timeperiod=21)
        df["ema_50"] = ta.EMA(df["close"], timeperiod=50)
        df["ema_200"] = ta.EMA(df["close"], timeperiod=200)
        df["%-dist_ema8"] = (df["close"] - df["ema_8"]) / df["close"]
        df["%-dist_ema21"] = (df["close"] - df["ema_21"]) / df["close"]
        df["%-dist_ema50"] = (df["close"] - df["ema_50"]) / df["close"]
        df["%-dist_ema200"] = (df["close"] - df["ema_200"]) / df["close"]

        atr14 = ta.ATR(df, timeperiod=14)
        df["%-atr_pct_14"] = self._safe_div(atr14, df["close"])

        macd_vals = ta.MACD(df["close"])
        try:

            macd_series = macd_vals["macd"]
            macdsignal_series = macd_vals["macdsignal"]
            macdhist_series = macd_vals["macdhist"]
        except Exception:

            try:
                macd_series, macdsignal_series, macdhist_series = macd_vals
            except Exception:

                macd_series = pd.Series(np.nan, index=df.index)
                macdsignal_series = pd.Series(np.nan, index=df.index)
                macdhist_series = pd.Series(np.nan, index=df.index)

        df["macd"] = macd_series
        df["macdsignal"] = macdsignal_series
        df["macdhist"] = macdhist_series
        price = df["close"].replace(0, np.nan)
        macd_hist_pct = self._safe_div(df["macdhist"], price)
        df["%-macd_hist_tanh"] = np.tanh(10.0 * macd_hist_pct)

        df["%-rsi_7"] = ta.RSI(df["close"], timeperiod=7)
        rsi14 = ta.RSI(df["close"], timeperiod=14)
        df["%-rsi_14"] = rsi14
        df["%-rsi_c"] = (rsi14 - 50.0) / 50.0
        df["%-rsi_25"] = ta.RSI(df["close"], timeperiod=25)

        stoch = ta.STOCH(df)
        df["%-stochk_c"] = (stoch["slowk"] - 50.0) / 50.0
        df["%-stochd_c"] = (stoch["slowd"] - 50.0) / 50.0

        df["%-cci_14"] = ta.CCI(df, timeperiod=14)
        df["%-cci_20"] = ta.CCI(df, timeperiod=20)

        bb_u, bb_m, bb_l = ta.BBANDS(df["close"], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        den = (bb_u - bb_l)
        df["%-bb_pos_20"] = np.where(den != 0, (df["close"] - bb_l) / den, 0.5)
        df["%-bb_width_20"] = self._safe_div(den, bb_m)

        vwap_l, vwap, vwap_h = self._vwap_bands(df, window=20, stdev_mult=1.0)
        df["vwap"], df["vwap_low"], df["vwap_high"] = vwap, vwap_l, vwap_h
        df["%-z_vwap_20"] = self._zscore(df["close"] - vwap, 20)

        obv = ta.OBV(df)
        df["%-obv_z_20"] = self._zscore(obv, 20)
        df["%-vol_ratio_20"] = self._safe_div(
            df["volume"], df["volume"].rolling(20).mean(), default=1.0
        )
        df["%-mfi_14"] = ta.MFI(df, timeperiod=14) / 100.0

        ema20 = ta.EMA(df["close"], timeperiod=20)
        atr20 = ta.ATR(df, timeperiod=20)
        kel_u, kel_l = ema20 + 1.5 * atr20, ema20 - 1.5 * atr20
        df["squeeze_on"] = ((bb_u < kel_u) & (bb_l > kel_l)).astype(int)

        df["%-adx_14"] = ta.ADX(df, timeperiod=14)
        df["%-plus_di_14"] = ta.PLUS_DI(df, timeperiod=14)
        df["%-minus_di_14"] = ta.MINUS_DI(df, timeperiod=14)

        ema8_slope = df["ema_8"] - df["ema_8"].shift(1)
        ema21_slope = df["ema_21"] - df["ema_21"].shift(1)
        ema50_slope = df["ema_50"] - df["ema_50"].shift(1)
        slope_norm = self._safe_div(ema21_slope + ema50_slope, price)
        df["%-trend_direction"] = np.tanh(3.0 * slope_norm + 2.0 * df["%-macd_hist_tanh"])
        df["%-trend_strength"] = np.tanh(
            np.abs(4.0 * slope_norm) + 1.5 * np.abs(df["%-macd_hist_tanh"])
        )
        df["%-ema8_slope"] = ema8_slope / (df["close"] + 1e-9)
        df["%-ema21_slope"] = ema21_slope / (df["close"] + 1e-9)
        df["%-ema50_slope"] = ema50_slope / (df["close"] + 1e-9)

        _, _, don_pos = self._donchian(df, 20)
        df["%-don_pos_20"] = don_pos

        df["%-z_close_20"] = self._zscore(df["close"], 20)

        if "%-logret_1" not in df.columns:
            df["%-logret_1"] = np.log(
                self._safe_div(df["close"], df["close"].shift(1), default=1.0)
            )
        lr = df["%-logret_1"].fillna(0.0)
        df["rv10"] = lr.rolling(10).std()
        df["rv50"] = lr.rolling(50).std()
        df["%-rv_ratio"] = self._safe_div(df["rv10"], df["rv50"], default=0.0)


        df["adosc_3_10"] = ta.ADOSC(df, fastperiod=3, slowperiod=10)

        df["%-top_pct_20"] = (df["high"].rolling(20).max() - df["close"]) / df["close"]

        # SuperTrend ATR-based
        mult, n = 3.0, 10
        atr_n = ta.ATR(df, timeperiod=n)
        hl2 = (df['high'] + df['low']) / 2.0
        basic_u = hl2 + mult * atr_n
        basic_l = hl2 - mult * atr_n
        f_upper = basic_u.copy()
        f_lower = basic_l.copy()
        for i in range(1, len(df)):
            if df['close'].iat[i-1] > f_upper.iat[i-1]:
                f_upper.iat[i] = min(basic_u.iat[i], f_upper.iat[i-1])
            else:
                f_upper.iat[i] = basic_u.iat[i]
            if df['close'].iat[i-1] < f_lower.iat[i-1]:
                f_lower.iat[i] = max(basic_l.iat[i], f_lower.iat[i-1])
            else:
                f_lower.iat[i] = basic_l.iat[i]
        st = f_lower.where(df['close'] > f_upper, f_upper)
        df['%-supertrend_gap_pct'] = (df['close'] - st) / df['close']
        df['%-supertrend_dir'] = (df['close'] > st).astype(int) * 2 - 1

        tr = (df[['high','close']].max(axis=1) - df[['low','close']].min(axis=1))
        vm_plus  = (df['high'] - df['low'].shift(1)).abs()
        vm_minus = (df['low']  - df['high'].shift(1)).abs()
        nvi = 14
        vi_plus  = vm_plus.rolling(nvi).sum()  / tr.rolling(nvi).sum()
        vi_minus = vm_minus.rolling(nvi).sum() / tr.rolling(nvi).sum()
        df['%-vi_plus']  = vi_plus
        df['%-vi_minus'] = vi_minus

        # Anchored VWAP Daily & Weekly
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        if "date" in df.columns and hasattr(df["date"], "dt"):
            day = df['date'].dt.floor('D')
            num_d = (tp * df['volume']).groupby(day).cumsum()
            den_d = df['volume'].groupby(day).cumsum().replace(0, np.nan)
            avwap_d = num_d / den_d

            week = (df['date'] - pd.to_timedelta(df['date'].dt.weekday, unit='D')).dt.floor('D')
            num_w = (tp * df['volume']).groupby(week).cumsum()
            den_w = df['volume'].groupby(week).cumsum().replace(0, np.nan)
            avwap_w = num_w / den_w

            df['%-dist_avwap_d'] = (df['close'] - avwap_d) / df['close']
            df['%-dist_avwap_w'] = (df['close'] - avwap_w) / df['close']

        # Realized vol estimators + skew/kurt
        logh = np.log(df['high'])
        logl = np.log(df['low'])
        logo = np.log(df['open'])
        logc = np.log(df['close'])
        parkinson = (1.0/(4*np.log(2))) * (logh - logl)**2
        gk = 0.5*(logh - logl)**2 - (2*np.log(2)-1)*(logc - logo)**2
        rs = (np.log(df['high']/df['close'])*np.log(df['high']/df['open']) +
              np.log(df['low']/df['close'])*np.log(df['low']/df['open']))
        win = 20
        df['%-rv_parkinson_20'] = parkinson.rolling(win).mean().pow(0.5)
        df['%-rv_gk_20']        = gk.rolling(win).mean().pow(0.5)
        df['%-rv_rs_20']        = rs.rolling(win).mean().clip(lower=0).pow(0.5)
        lr2 = np.log(self._safe_div(df['close'], df['close'].shift(1), default=1.0)).fillna(0)
        df['%-skew_20'] = lr2.rolling(win).skew()
        df['%-kurt_20'] = lr2.rolling(win).kurt()

        # Trend quality & Hurst
        df['%-linreg_slope_50'] = self._linreg_slope(df['close'], window=50)
        df['%-linreg_r2_50']    = self._linreg_r2(df['close'], window=50)
        df['%-hurst_50']        = self._hurst_rs(df['close'], window=50)

        # Autocorrelation & Entropy
        df['%-autocorr_1'] = self._autocorr(lr2, lag=1, window=50)
        df['%-autocorr_3'] = self._autocorr(lr2, lag=3, window=50)
        df['%-autocorr_5'] = self._autocorr(lr2, lag=5, window=50)
        df['%-entropy_100'] = self._entropy_norm(lr2, window=100, bins=20)

        # Microstructure proxies OHLCV
        amihud = (lr2.abs()) / (df['volume'].replace(0, np.nan))
        df['%-amihud_20'] = amihud.rolling(20).mean().replace([np.inf, -np.inf], np.nan)
        def _roll_impact(y):
            y = pd.Series(y).dropna()
            if len(y) < 2:
                return np.nan
            return y.cov(y.shift(1))
        df['%-roll_impact_50'] = lr2.rolling(50).apply(lambda s: _roll_impact(s), raw=False)
        df['%-eff_spread'] = (df['high'] - df['low']) / (df['close'] + 1e-9)

        if 'funding_rate' in df.columns:
            df['%-funding']  = df['funding_rate']
            df['%-dFunding'] = df['funding_rate'].diff()
        if 'open_interest' in df.columns:
            oi = df['open_interest']
            df['%-oi_norm'] = oi / (oi.rolling(100).mean() + 1e-9)
            df['%-dOI']     = oi.pct_change()
        if {'perp_close','spot_close'}.issubset(df.columns):
            df['%-basis'] = (df['perp_close'] - df['spot_close']) / (df['spot_close'] + 1e-9)

        if '%-pct_change_1_BTC' in df.columns:
            r_pair = df['%-pct_change_1']
            r_btc  = df['%-pct_change_1_BTC']
            winb = 50
            cov = r_pair.rolling(winb).cov(r_btc)
            var = r_btc.rolling(winb).var()
            beta = cov / (var + 1e-9)
            df['%-beta_btc_50'] = beta
            df['%-corr_btc_50'] = r_pair.rolling(winb).corr(r_btc)
            df['%-resid_1']     = r_pair - beta * r_btc

        long_signals = 0.0
        short_signals = 0.0

        if 'ema_8' in df.columns and 'ema_21' in df.columns:
            long_signals += (df['ema_8'] > df['ema_21']).astype(float)
            short_signals += (df['ema_8'] < df['ema_21']).astype(float)

        if '%-rsi_7' in df.columns:
            long_signals += (df['%-rsi_7'] < 30).astype(float)
            short_signals += (df['%-rsi_7'] > 70).astype(float)

        if '%-cci_14' in df.columns:
            long_signals += (df['%-cci_14'] < -100).astype(float)  # Extreme oversold
            short_signals += (df['%-cci_14'] > 100).astype(float)  # Extreme overbought

        if '%-plus_di_14' in df.columns and '%-minus_di_14' in df.columns:
            long_signals += (df['%-plus_di_14'] > df['%-minus_di_14']).astype(float)
            short_signals += (df['%-minus_di_14'] > df['%-plus_di_14']).astype(float)

        if 'ema_50' in df.columns:
            long_signals += (df['close'] > df['ema_50']).astype(float)
            short_signals += (df['close'] < df['ema_50']).astype(float)

        total_signals = long_signals + short_signals
        df['%-directional_bias'] = np.where(
            total_signals > 0,
            (long_signals - short_signals) / total_signals,
            0.0
        )

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        return df

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict
    ):

        df = dataframe
        p = int(period)

        df[f"%-rsi-period_{p}"] = ta.RSI(df["close"], timeperiod=p)
        df[f"%-mfi-period_{p}"] = ta.MFI(df, timeperiod=p)
        df[f"%-adx-period_{p}"] = ta.ADX(df, timeperiod=p)
        df[f"%-plus_di-period_{p}"] = ta.PLUS_DI(df, timeperiod=p)
        df[f"%-minus_di-period_{p}"] = ta.MINUS_DI(df, timeperiod=p)

        atr = ta.ATR(df, timeperiod=p)
        df[f"%-atr_pct-period_{p}"] = self._safe_div(atr, df["close"])

        ema_p = ta.EMA(df["close"], timeperiod=p)
        df[f"%-dist_ema-period_{p}"] = (df["close"] - ema_p) / df["close"]

        bb_u, bb_m, bb_l = ta.BBANDS(df["close"], timeperiod=p, nbdevup=2.0, nbdevdn=2.0, matype=0)
        den = (bb_u - bb_l)
        df[f"%-bb_width-period_{p}"] = self._safe_div(den, bb_m)
        df[f"%-bb_pos-period_{p}"] = np.where(den != 0, (df["close"] - bb_l) / den, 0.5)

        _, _, dpos = self._donchian(df, p)
        df[f"%-don_pos-period_{p}"] = dpos

        obv = ta.OBV(df)
        df[f"%-obv_z-period_{p}"] = self._zscore(obv, p)

        return df

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs):
        """
        Asigură coloana &-action (agentul RL scrie în ea).
        """
        if "&-action" not in dataframe.columns:
            dataframe["&-action"] = 0
        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            reduce(lambda x, y: x & y, [df["do_predict"] == 1, df["&-action"] == 1]),
            ["enter_long", "enter_tag"]
        ] = (1, "RL_long")

        df.loc[
            reduce(lambda x, y: x & y, [df["do_predict"] == 1, df["&-action"] == 3]),
            ["enter_short", "enter_tag"]
        ] = (1, "RL_short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            reduce(lambda x, y: x & y, [df["do_predict"] == 1, df["&-action"] == 2]),
            "exit_long"
        ] = 1

        df.loc[
            reduce(lambda x, y: x & y, [df["do_predict"] == 1, df["&-action"] == 4]),
            "exit_short"
        ] = 1

        return df
