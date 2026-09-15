import numpy as np
import pandas as pd
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import IntParameter, DecimalParameter
import talib.abstract as ta


class LongStrat2h(IStrategy):
    # Configurações gerais
    timeframe = '3m'
    process_only_new_candles = True
    leverage_value = 10
    can_short = True

    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    stoploss = -0.342
    trailing_stop = True
    trailing_stop_positive = 0.027
    trailing_stop_positive_offset = 0.047
    trailing_only_offset_is_reached = True

    minimal_roi = {
        "0": 0.648,
        "560": 0.247,
        "1400": 0.084,
        "3409": 0
    }

    # Parâmetros ajustáveis
    per = IntParameter(2, 200, default=189, space="buy", optimize=True)
    mult = DecimalParameter(1.0, 5.0, default=2.161, space="buy", optimize=True)
    visited_window_long = IntParameter(1, 50, default=10, space="buy", optimize=True)
    visited_window_short = IntParameter(1, 50, default=10, space="sell", optimize=True)

    def smooth_rng(self, src: np.ndarray, t: int, m: float) -> np.ndarray:
        """
        Calcula o intervalo suavizado para o filtro de preços.
        """
        wper = t * 2 - 1
        avrng = ta.EMA(np.abs(src - np.roll(src, 1)), t)
        smooth_rng = ta.EMA(avrng, wper) * m
        return smooth_rng

    def range_filter(self, src: np.ndarray, smoothrng: np.ndarray) -> np.ndarray:
        """
        Aplica o filtro de intervalo para suavizar os valores.
        """
        filt = src.copy()
        for i in range(1, len(src)):
            prev = filt[i - 1]
            if src[i] > prev:
                filt[i] = src[i]
            elif src[i] + smoothrng[i] > prev:
                filt[i] = prev
            else:
                filt[i] = src[i] + smoothrng[i]
        return filt

    def leverage(self, pair: str, current_time, current_rate: float, proposed_leverage: float, **kwargs) -> float:
        """
        Define a alavancagem fixa.
        """
        return self.leverage_value

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calcula indicadores necessários para a estratégia.
        """
        src = dataframe['close'].values
        smooth_rng = self.smooth_rng(src, self.per.value, self.mult.value)

        dataframe['smooth_rng'] = smooth_rng
        dataframe['filter'] = self.range_filter(src, smooth_rng)
        dataframe['upward'] = ((dataframe['filter'] > np.roll(dataframe['filter'], 1)) &
                               (dataframe['filter'] > dataframe['filter'].shift(1))).astype(int)
        dataframe['downward'] = ((dataframe['filter'] < np.roll(dataframe['filter'], 1)) &
                                 (dataframe['filter'] < dataframe['filter'].shift(1))).astype(int)

        # Corrigir lógica de visited_long e visited_short
        dataframe['visited_long'] = dataframe['close'] > dataframe['close'].rolling(self.visited_window_long.value).min()
        dataframe['visited_short'] = dataframe['close'] < dataframe['close'].rolling(self.visited_window_short.value).max()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define as condições de entrada para posições longas e curtas.
        """
        # Condições para entrada longa
        dataframe.loc[
            (dataframe['close'] > dataframe['filter'] * 0.99) &
            (dataframe['filter'] > dataframe['filter'].shift(1)) &
            (dataframe['upward'] > 0) &
            (dataframe['visited_short']),  # Corrigida lógica de "visited_short"
            'enter_long'] = 1

        # Condições para entrada curta
        dataframe.loc[
            (dataframe['close'] < dataframe['filter'] * 1.01) &
            (dataframe['filter'] < dataframe['filter'].shift(1)) &
            (dataframe['downward'] > 0) &
            (dataframe['visited_long']),  # Corrigida lógica de "visited_long"
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define as condições de saída para posições longas e curtas.
        """
        dataframe.loc[
            (dataframe['close'] < dataframe['filter'] * 0.98),
            'exit_long'] = 1

        dataframe.loc[
            (dataframe['close'] > dataframe['filter'] * 1.02),
            'exit_short'] = 1

        return dataframe
