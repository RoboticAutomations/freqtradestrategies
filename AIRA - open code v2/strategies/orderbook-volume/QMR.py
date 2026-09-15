# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
import numpy as np
import pandas as pd
from datetime import datetime
from pandas import DataFrame
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, RealParameter
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

# QuantitativeMeanReversion
class QMR(IStrategy):
    """
    Estrategia: Mean Reversion + ICT Liquidity Sweeps
    Timeframe: 15m
    Conceptos: POC (Point of Control), Range High/Low, Dynamic Stake (Kelly)
    """
    INTERFACE_VERSION = 3
    timeframe = '15m'
    can_short = True

    lookback_window = IntParameter(1, 100, default=50, space="buy")
    rsi_threshold_long = IntParameter(20, 40, default=30, space="buy")
    rsi_threshold_short = IntParameter(60, 80, default=70, space="sell")
    
    leverage_value = IntParameter(1, 5, default=3, space="buy")
    kelly_fraction = DecimalParameter(0.05, 0.2, default=0.1, space="buy")

    minimal_roi = {} 
    stoploss = -0.99
    exit_profit_only = True

    @property
    def plot_config(self):
        return {
            "main_plot": {
                "typical_price": {"color": "purple", "style": "line"},
                "poc": {"color": "red", "style": "dash"},
                "vwap": {"color": "orange", "style": "dash"},
            },
            "subplots": {
                "RSI": {
                    "rsi": {"color": "yellow"},
                },
                "ATR": {
                    "atr": {"color": "yellow"},
                },
            },
        }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1. RSI para Exhaustión
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # 2. Definición de Rango (ICT Liquidity Pools)
        # Identificamos el máximo y mínimo de las últimas N velas
        dataframe['range_hi'] = dataframe['high'].rolling(window=self.lookback_window.value).max()
        dataframe['range_lo'] = dataframe['low'].rolling(window=self.lookback_window.value).min()

        # 3. Point of Control (POC) Simplificado - Reversión a la Media
        # En modelos cuantitativos, el Typical Price es una aproximación robusta al valor justo
        dataframe['typical_price'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['poc'] = ta.SMA(dataframe['typical_price'], timeperiod=self.lookback_window.value)

        # 4. Volatilidad para Gestión de Riesgo (ATR)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # 5. VWAP rolling = POC proxy
        dataframe['vwap'] = (
            (dataframe['typical_price'] * dataframe['volume'])
            .rolling(self.lookback_window.value)
            .sum()
            / dataframe['volume'].rolling(self.lookback_window.value).sum()
        )

        # Distance to value (mean deviation)
        dataframe['dev_from_vwap'] = (dataframe['close'] - dataframe['vwap']) / dataframe['atr']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Lógica de Entrada basada en Barrido de Liquidez (Liquidity Sweep)
        """
        # LONG: El precio barre el mínimo del rango (Discount) + RSI sobrevendido
        dataframe.loc[
            (
                (dataframe['low'] <= dataframe['range_lo']) &
                (dataframe['rsi'] < self.rsi_threshold_long.value) &
                (dataframe['close'] > dataframe['low']) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'long_sweep_reversion')

        # SHORT: El precio barre el máximo del rango (Premium) + RSI sobrecomprado
        dataframe.loc[
            (
                (dataframe['high'] >= dataframe['range_hi']) &
                (dataframe['rsi'] > self.rsi_threshold_short.value) &
                (dataframe['close'] < dataframe['high']) &
                (dataframe['volume'] > 0)
            ),
            ['enter_short', 'enter_tag']] = (1, 'short_sweep_reversion')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Lógica de Salida: Reversión Completa al POC
        """
        # Salida Long: El precio alcanza o supera el POC (Fair Value)
        dataframe.loc[
            (dataframe['close'] >= dataframe['poc']),
            ['exit_long', 'exit_tag']] = (1, 'exit_at_poc')

        # Salida Short: El precio cae por debajo del POC
        dataframe.loc[
            (dataframe['close'] <= dataframe['poc']),
            ['exit_short', 'exit_tag']] = (1, 'exit_at_poc')

        return dataframe

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:
        """
        GESTIÓN DINÁMICA: Implementación del Criterio de Kelly
        Calcula el tamaño de la posición basado en la ventaja estadística.
        """
        # Estimación de Win Rate (p) y Ratio Riesgo/Beneficio (R) basada en backtest previo
        win_rate = 0.55 
        risk_reward = 1.5
        
        # Fórmula de Kelly: f* = (p*R - q) / R
        q = 1 - win_rate
        kelly_f = ((win_rate * risk_reward) - q) / risk_reward
        
        # Aplicamos la fracción de Kelly y reserva de balance (Hedging Reserve)
        # Mantenemos un 30% del balance libre para Cross Margin/Funding
        effective_stake = max_stake * kelly_f * float(self.kelly_fraction.value)
        
        return min(max(effective_stake, min_stake or 0), max_stake * 0.7)

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str | None,
                            side: str, **kwargs) -> bool:
        """
        Configuración de Apalancamiento Dinámico (Cross Margin Simulation)
        """
        # Aquí se podría integrar lógica para ajustar el leverage en el exchange
        # según la volatilidad (ATR). A mayor ATR, menor leverage.
        return True