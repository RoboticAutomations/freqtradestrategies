# freqtrade download-data -c user_data/config.MoMy5.json --exchange binance --pair `cat pairs33` --timeframe 5m --timerange=20240901-20241208
# freqtrade hyperopt --hyperopt-loss ProfitDrawDownHyperOptLoss --spaces buy sell -c user_data/config.MoMy5.json --strategy MoMy5 --timerange=20240910-20241208 -e 200 --timeframe-detail 1m -p `cat pairs33` --timeframe 5m
import talib.abstract as ta
import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
from typing import Optional
from datetime import datetime
import logging


class MoMy6(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 0.10,
        "30": 0.05,
        "60": 0.03,
        "120": 0.01,
    }
    can_short = True
    stoploss = -0.15
    use_exit_signal = True
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    mri_threshold_high = DecimalParameter(1.0, 100.0, default=50, space="buy")
    mri_threshold_low = DecimalParameter(-100.0, -1.0, default=-50, space="sell")
    timeframe = "5m"

    # Parámetros ajustables de apalancamiento (optimizables con Hyperopt)
    leverage_input_buy = DecimalParameter(1.0, 10.0, default=3.0, space="buy")
    leverage_input_sell = DecimalParameter(1.0, 10.0, default=3.0, space="sell")

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
        "force_entry": "limit",
        "force_exit": "limit",
    }

    order_time_in_force = {
        "entry": "gtc",
        "exit": "gtc",
    }

    # Configuración del gráfico
    plot_config = {
        "main_plot": {
            "mri": {
                "color": "blue",
                "title": "Índice de Reversión de Mercado (MRI)",
            },
            "momentum_change": {
                "color": "red",
                "title": "Cambio de momentum",
            },
            "wma": {  # Configuración de WMA para añadirlo al gráfico principal
                "color": "green",
                "title": "Weighted Moving Average (WMA)",
            },
        },
        "subplots": {
            "candlestick_patterns": {
                "hammer": {
                    "color": "green",
                    "title": "Vela martillo",
                    "plot": "bar",
                },
                "shooting_star": {
                    "color": "orange",
                    "title": "Vela estrella fugaz",
                    "plot": "bar",
                },
            },
            "volume_deviations": {
                "volume_deviation": {
                    "color": "purple",
                    "title": "Desviación del volumen",
                },
            },
        },
    }

    def leverage(
            self,
            pair: str,
            current_time: datetime,
            current_rate: float,
            proposed_leverage: float,
            max_leverage: float,
            entry_tag: Optional[str],
            side: str,
            **kwargs,
    ) -> float:
        """
            Ajusta el apalancamiento dinámicamente según la volatilidad (ATR).
            """
        # Verificar si el DataFrame tiene suficientes datos
        if not hasattr(self, 'dataframe') or self.dataframe is None:
            logging.warning("Dataframe no está disponible. Usando apalancamiento por defecto.")
            return min(self.leverage_input_buy.value if side == "long" else self.leverage_input_sell.value,
                       max_leverage)

        # Determinamos el tamaño mínimo necesario basado en el período del ATR
        required_periods = 14  # Este es el "timeperiod=14" usado para ATR
        if len(self.dataframe) < required_periods:
            logging.warning(
                f"Datos insuficientes para calcular ATR. Se necesitan al menos {required_periods} períodos.")
            return min(self.leverage_input_buy.value if side == "long" else self.leverage_input_sell.value,
                       max_leverage)

        try:
            # Cálculo del ATR (Average True Range)
            atr_series = ta.ATR(
                self.dataframe['high'],
                self.dataframe['low'],
                self.dataframe['close'],
                timeperiod=14,
            )

            # Obtenemos el último valor del ATR usando `.iloc[-1]`
            atr = atr_series.iloc[-1]

            # Validación para evitar errores si ATR no es calculable
            if atr is None or atr <= 0:
                logging.warning("ATR no se pudo calcular correctamente. Usando apalancamiento por defecto.")
                return min(self.leverage_input_buy.value if side == "long" else self.leverage_input_sell.value,
                           max_leverage)

            # Ajuste del apalancamiento basado en la volatilidad
            if atr > 0.03:
                calculated_leverage = 1.0  # Alta volatilidad -> Bajo apalancamiento
            elif atr > 0.01:
                calculated_leverage = 3.0  # Volatilidad media -> Apalancamiento moderado
            else:
                calculated_leverage = 5.0  # Baja volatilidad -> Alto apalancamiento

            # Restricción según el apalancamiento máximo permitido por el exchange
            return min(calculated_leverage, max_leverage)

        except Exception as e:
            logging.error(f"Error durante el cálculo del ATR: {str(e)}. Usando apalancamiento por defecto.")
            return min(self.leverage_input_buy.value if side == "long" else self.leverage_input_sell.value,
                       max_leverage)

    def calculate_mri(self, dataframe: DataFrame) -> DataFrame:
        """
        Calcula el Índice de Reversión de Mercado (MRI) utilizando patrones de velas y volumen.
        """
        dataframe['volatility'] = dataframe['close'].rolling(window=20).std()
        window_size = max(10, int(dataframe['volatility'].mean()))

        rolling_mean = dataframe['volume'].rolling(window=window_size).mean()
        rolling_std = dataframe['volume'].rolling(window=window_size).std()

        dataframe["volume_deviation"] = (
                (dataframe["volume"] - rolling_mean) / rolling_std
        ).fillna(0)

        dataframe["hammer"] = np.where(
            (dataframe["close"] > dataframe["open"])
            & ((dataframe["high"] - dataframe["low"]) > 2 * (dataframe["open"] - dataframe["low"]))
            & ((dataframe["close"] - dataframe["open"]) / (dataframe["high"] - dataframe["low"]).replace(0,
                                                                                                         np.nan) > 0.6),
            1, 0
        )

        dataframe["shooting_star"] = np.where(
            (dataframe["close"] < dataframe["open"])
            & ((dataframe["high"] - dataframe["low"]) > 2 * (dataframe["close"] - dataframe["high"]))
            & ((dataframe["open"] - dataframe["close"]) / (dataframe["high"] - dataframe["low"]).replace(0,
                                                                                                         np.nan) > 0.75),
            1, 0
        )

        # Calcolo componenti MRI
        dataframe['mri_components'] = (
            dataframe["hammer"] - dataframe["shooting_star"] + 
            np.sign(dataframe['momentum_change']) +
            np.where(dataframe["volume_deviation"] > 1, 1, 
                    np.where(dataframe["volume_deviation"] < -1, -1, 0))
        )
        
        def normalize_series(series):
            def rolling_zscore(x, window=50):
                # Calcola z-score usando solo dati precedenti
                rolling_mean = x.rolling(window=window, min_periods=1).mean()
                rolling_std = x.rolling(window=window, min_periods=1).std()
                
                # Calcola z-score usando solo dati fino a quel punto
                return (x - rolling_mean) / rolling_std

            # Calcola lo z-score senza guardare avanti
            normalized = np.clip(rolling_zscore(series) * 50, -100, 100)
            
            return normalized
        
        # Applica normalizzazione
        dataframe['mri'] = normalize_series(dataframe['mri_components'])

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Agrega los indicadores necesarios a los datos del mercado.
        """
        dataframe['tsf']   = ta.TSF(dataframe)
        dataframe['momentum_change'] = dataframe['close'].diff().fillna(0)
        dataframe['wma'] = ta.WMA(dataframe['close'], timeperiod=14)  # Calcula el indicador WMA
        dataframe = self.calculate_mri(dataframe)
        self.dataframe = dataframe  # Guarda el dataframe para usarlo en apalancamiento
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define las condiciones para abrir posiciones (entry trend).
        """
        # Condición de entrada para long
        dataframe.loc[
            (dataframe["mri"] > self.mri_threshold_high.value) & 
            (dataframe["mri"] < 80) &(
                    dataframe['close'] > dataframe['tsf']),  # Precio > WMA = Momentum alcista
            ["enter_long", "enter_tag"]
        ] = (1, "reversal_up")

        # Condición de entrada para short
        dataframe.loc[
            (dataframe["mri"] < self.mri_threshold_low.value) &
            (dataframe["mri"] > -80) &(
                    dataframe['close'] < dataframe['tsf']),  # Precio < WMA = Momentum bajista
            ["enter_short", "enter_tag"]
        ] = (1, "reversal_down")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define las condiciones para salir de posiciones (exit trend).
        """
        dataframe.loc[
            (dataframe["mri"] < self.mri_threshold_low.value),
            ["exit_long", "exit_tag"],
        ] = (1, "exit_reversal_down")
        dataframe.loc[
            (dataframe["mri"] > self.mri_threshold_high.value),
            ["exit_short", "exit_tag"],
        ] = (1, "exit_reversal_up")
        return dataframe

    def confirm_trade_entry(
            self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs
    ) -> bool:
        logging.info(
            f"Confirmando entrada de trade para el par {pair} con tipo de orden {order_type} y cantidad {amount} a la tasa {rate}."
        )
        return True

    def confirm_trade_exit(
            self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs
    ) -> bool:
        logging.info(
            f"Confirmando salida de trade para el par {pair} con tipo de orden {order_type} y cantidad {amount} a la tasa {rate}."
        )
        return True