from __future__ import annotations

import talib as ta
import numpy as np
import pandas as pd
from functools import reduce
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy


class MyRLStrat(IStrategy):
    timeframe = "5m"
    can_short = True
    startup_candle_count = 300

    stoploss = -0.30
    minimal_roi = {"0": 0.999}

    @property
    def plot_config(self):
        """
        Define what to plot in Freqtrade backtest/plot mode.
        main_plot overlays on price chart, subplots are below.
        """
        return {
            "main_plot": {
                # Show Bollinger Bands around price
                "%-bb_upperband-period": {"color": "blue"},
                "%-bb_middleband-period": {"color": "gray"},
                "%-bb_lowerband-period": {"color": "blue"},
            },
            "subplots": {
                "RSI": {
                    "%-rsi-period": {"color": "purple"},
                },
                "ADX / DI": {
                    "%-adx-period": {"color": "orange"},
                    "%-pdi-period": {"color": "green"},
                    "%-mdi-period": {"color": "red"},
                },
                "Volume": {
                    "volume": {"type": "bar", "color": "gray"},
                    "%-relative_volume-period": {"color": "blue"},
                },
            },
        }


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def feature_engineering_expand_all(self, dataframe: DataFrame, period, metadata, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `indicator_periods_candles`, `include_timeframes`, `include_shifted_candles`, and
        `include_corr_pairs`. In other words, a single feature defined in this function
        will automatically expand to a total of
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` numbers of features added to the model.

        All features must be prepended with `%` to be recognized by FreqAI internals.

        Access metadata such as the current pair/timeframe/period with:

        `metadata["pair"]` `metadata["tf"]`  `metadata["period"]`

        :param df: strategy dataframe which will receive the features
        :param period: period of the indicator - usage example:
        :param metadata: metadata of current pair
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        """
        # ---- Multi-input indicators (H, L, C) ----
        dataframe["%-dx-period"]   = ta.DX(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=period)
        dataframe["%-pdi-period"]  = ta.PLUS_DI(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=period)
        dataframe["%-mdi-period"]  = ta.MINUS_DI(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=period)
        dataframe["%-adx-period"]  = ta.ADX(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=period)

        # ---- Windowed highs/lows ----
        dataframe["%-max-period"] = ta.MAX(dataframe["high"], timeperiod=period)
        dataframe["%-min-period"] = ta.MIN(dataframe["low"],  timeperiod=period)

        # ---- Single-series indicators (use close) ----
        dataframe["%-std-period"] = ta.STDDEV(dataframe["close"], timeperiod=period)
        dataframe["%-rsi-period"] = ta.RSI(dataframe["close"],    timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe["close"],     timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe["close"],     timeperiod=period)
        dataframe["%-roc-period"] = ta.ROC(dataframe["close"],     timeperiod=period)

        # ---- Volume-aware indicator ----
        dataframe["%-mfi-period"] = ta.MFI(
            dataframe["high"], dataframe["low"], dataframe["close"], dataframe["volume"], timeperiod=period
        )

        # ---- Bollinger Bands (Series in/out). Prefix with "%-" so FreqAI expands them too ----
        tp = qtpylib.typical_price(dataframe)  # (H+L+C)/3
        bb = qtpylib.bollinger_bands(tp, window=period, stds=2.2)
        dataframe["%-bb_lowerband-period"] = bb["lower"]
        dataframe["%-bb_middleband-period"] = bb["mid"]
        dataframe["%-bb_upperband-period"] = bb["upper"]

        # Safe BB width (avoid divide-by-zero)
        dataframe["%-bb_width-period"] = (
            (dataframe["%-bb_upperband-period"] - dataframe["%-bb_lowerband-period"])
            / dataframe["%-bb_middleband-period"].replace(0, np.nan)
        )

        # Distance of close to lower band
        dataframe["%-close-bb_lower-period"] = dataframe["close"] / dataframe["%-bb_lowerband-period"].replace(0, np.nan)

        # ---- Relative volume with a proper rolling mean (avoid early bias) ----
        vol_ma = dataframe["volume"].rolling(window=period, min_periods=period).mean()
        dataframe["%-relative_volume-period"] = dataframe["volume"] / vol_ma.replace(0, np.nan)

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `include_timeframes`, `include_shifted_candles`, and `include_corr_pairs`.
        In other words, a single feature defined in this function
        will automatically expand to a total of
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`
        numbers of features added to the model.

        Features defined here will *not* be automatically duplicated on user defined
        `indicator_periods_candles`

        Access metadata such as the current pair/timeframe with:

        `metadata["pair"]` `metadata["tf"]`

        All features must be prepended with `%` to be recognized by FreqAI internals.

        :param df: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-raw_low"] = dataframe["low"]
        dataframe["%-trange"] = ta.TRANGE(dataframe["high"], dataframe["low"], dataframe["close"])

        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This optional function will be called once with the dataframe of the base timeframe.
        This is the final function to be called, which means that the dataframe entering this
        function will contain all the features and columns created by all other
        freqai_feature_engineering_* functions.

        This function is a good place to do custom exotic feature extractions (e.g. tsfresh).
        This function is a good place for any feature that should not be auto-expanded upon
        (e.g. day of the week).

        Access metadata such as the current pair with:

        `metadata["pair"]`

        All features must be prepended with `%` to be recognized by FreqAI internals.

        :param df: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        usage example: dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        """
        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-raw_low"] = dataframe["low"]

        dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 25
        dataframe["%-min_of_day"]  = (dataframe["date"].dt.minute + 1) / 60

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        Access metadata such as the current pair with:

        `metadata["pair"]`

        :param df: strategy dataframe which will receive the targets
        :param metadata: metadata of current pair
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        dataframe["&-action"] = 0
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        cond_long = [df["do_predict"] == 1, df["&-action"] == 1]
        if cond_long:
            df.loc[reduce(lambda a, b: a & b, cond_long), ["enter_long", "enter_tag"]] = (1, "RL-long")

        cond_short = [df["do_predict"] == 1, df["&-action"] == 3]
        if cond_short:
            df.loc[reduce(lambda a, b: a & b, cond_short), ["enter_short", "enter_tag"]] = (1, "RL-short")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        cond_exit_long = [df["do_predict"] == 1, df["&-action"] == 2]
        if cond_exit_long:
            df.loc[reduce(lambda a, b: a & b, cond_exit_long), "exit_long"] = 1

        cond_exit_short = [df["do_predict"] == 1, df["&-action"] == 4]
        if cond_exit_short:
            df.loc[reduce(lambda a, b: a & b, cond_exit_short), "exit_short"] = 1
        return df
