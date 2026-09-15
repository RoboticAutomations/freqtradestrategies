
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce
from pandas import DataFrame
import warnings
import pandas as pd
# --------------------------------
import talib.abstract as ta
from talib import MA_Type
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, informative
from freqtrade.strategy import DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter
from freqtrade.exchange import date_minus_candles
import technical.indicators as ftt
import math
import logging
from scipy.signal import find_peaks, find_peaks_cwt
import warnings
from math import ceil
from datetime import datetime, timezone, timedelta
from pmdarima import auto_arima
from pmdarima import model_selection
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf
import time

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

pd.set_option('display.float_format', lambda x: '%.7f' % x)
logger = logging.getLogger(__name__)

class ARIMA_MTF_CS(IStrategy):
    INTERFACE_VERSION = 3

    # Stoploss:
    stoploss = -0.08

    # Trailing stop:
    use_custom_stoploss = True
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # Initialize dicts for arima storage
    last_run_time = {}
    arima_model = {}
    last_run_time_1h = {}
    arima_model_1h = {}
    last_run_time_4h = {}
    arima_model_4h = {}

    # Sell signal
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False
    can_short = True

    ## Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 60,
        "stoploss_on_exchange_limit_ratio": 0.99
    }

    # Optimal timeframe for the strategy
    timeframe = '15m'
    startup_candle_count = 480
    process_only_new_candles = True

    # Custom Entry
    last_entry_price = None

    position_adjustment_enable = True

    # Protection
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0, default=-12.0, space='buy', optimize=True)
    ewo_high = DecimalParameter(2.0, 12.0, default=8.0, space='buy', optimize=True)
    ewo_high_2 = DecimalParameter(-6.0, 12.0, default=-8, space='buy', optimize=True)

    # Hyper-opt parameters
    base_nb_candles_buy = IntParameter(150, 200, default=180, space='buy', optimize=True)
    up = DecimalParameter(low=1.020, high=1.025, default=1.02, decimals=3 ,space='buy', optimize=True, load=True)
    dn = DecimalParameter(low=0.983, high=0.987, default=0.984, decimals=3 ,space='buy', optimize=True, load=True)
    increment = DecimalParameter(low=1.0005, high=1.001, default=1.0007, decimals=4 ,space='buy', optimize=True, load=True)
    atr_length = IntParameter(10, 30, default=14, space='buy', optimize=True, load=True)
    window = IntParameter(10, 30, default=21, space='buy', optimize=True, load=True)
    window_1h = IntParameter(10, 30, default=21, space='buy', optimize=True, load=True)
    window_4h = IntParameter(10, 30, default=5, space='buy', optimize=True, load=True)
    x = DecimalParameter(low=1.2, high=1.75, default=1.6, decimals=2 ,space='buy', optimize=True, load=True)
    x_1h = DecimalParameter(low=1.2, high=1.75, default=1.6, decimals=2 ,space='buy', optimize=True, load=True)
    x_4h = DecimalParameter(low=1.2, high=1.75, default=1.6, decimals=2 ,space='buy', optimize=True, load=True)

    predicted_profit_percentage = DecimalParameter(low=0.5, high=1, default=1, decimals=3,  space='buy', optimize=True, load=True)
    predicted_profit_percentage_4h = DecimalParameter(low=0.5, high=2, default=1.5, decimals=3,  space='buy', optimize=True, load=True)
    predicted_profit_percentage_4h_big = DecimalParameter(low=0.5, high=3, default=2, decimals=3,  space='buy', optimize=True, load=True)

    ### trailing stop loss optimiziation ###
    tsl_target3 = DecimalParameter(low=0.10, high=0.15, default=0.15, decimals=2,  space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, decimals=3,  space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.06, high=0.10, default=0.1, decimals=3, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, decimals=3, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.04, high=0.08, default=0.06, decimals=3, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, decimals=3, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.03, high=0.06, default=0.04, decimals=3, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.005, high=0.012, default=0.01, decimals=3, space='sell', optimize=True, load=True)
    moon = IntParameter(80, 90, default=85, space='sell', optimize=True)

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]

    def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)

        entry_price = (dataframe['close'].iat[-1] + dataframe['open'].iat[-1] + proposed_rate + proposed_rate) / 4
        logger.info(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}") 

        # Check if there is a stored last entry price and if it matches the proposed entry price
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.0001:  # Tolerance for floating-point comparison
            entry_price *= self.increment.value # Increment by 0.2%
            logger.info(f"{pair} Incremented entry price: {entry_price} based on previous entry price : {self.last_entry_price}.")

        # Update the last entry price
        self.last_entry_price = entry_price

        return entry_price

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                            rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        # Handle freak events
        if exit_reason == 'roi' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} ROI is below 0")
            self.dp.send_msg(f'{trade.pair} ROI is below 0')
            return False

        if exit_reason == 'partial_exit' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} partial exit is below 0")
            self.dp.send_msg(f'{trade.pair} partial exit is below 0')
            return False

        if exit_reason == 'trailing_stop_loss' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} trailing stop price is below 0")
            self.dp.send_msg(f'{trade.pair} trailing stop price is below 0')
            return False

        return True

    @informative('4h')
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair_4h = metadata['pair']
        current_time_4h = time.time()

        dataframe['OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        size_4h = len(dataframe) - 10
        train_4h, test_4h = model_selection.train_test_split(dataframe['OHLC4'], train_size=size_4h)

        # Initialize values for the current pair if not already done
        if pair_4h not in self.last_run_time_4h:
            self.last_run_time_4h[pair_4h] = current_time_4h

            logger.info(f"Initial ARIMA 4h Model Training for {pair_4h}")
            # Fit ARIMA model
            start_time_4h = time.time()
            self.arima_model_4h[pair_4h] = auto_arima(train_4h, start_p=1, start_q=1, start_P=1, start_Q=1,
                                                max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=False,
                                                stepwise=True, suppress_warnings=True, D=10, max_D=20,
                                                error_action='ignore')
            fitting_time_4h = time.time() - start_time_4h
            logger.info(f"{pair_4h} - ARIMA 4h Model fitted in {fitting_time_4h:.2f} seconds")

        # Check if it's time to retrain for the current pair
        if current_time_4h - self.last_run_time_4h[pair_4h] >= 14400:  # Check if an 4 hours has passed
            logger.info(f"Auto Fitting 4h ARIMA Model for {pair_4h}")
            self.last_run_time_4h[pair_4h] = current_time_4h

            # Fit ARIMA model
            start_time_4h = time.time()
            self.arima_model_4h[pair_4h] = auto_arima(train_4h, start_p=1, start_q=1, start_P=1, start_Q=1,
                                                max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=False,
                                                stepwise=True, suppress_warnings=True, D=10, max_D=20,
                                                error_action='ignore')
            fitting_time_4h = time.time() - start_time_4h
            logger.info(f"{pair_4h} - ARIMA 4h Model fitted in {fitting_time_4h:.2f} seconds")

        # Use the previously fitted ARIMA model for forecasting
        if self.arima_model_4h[pair_4h] is not None:
            start_time_4h = time.time()
            future_forecast_4h, conf_int_4h = self.arima_model_4h[pair_4h].predict(n_periods=test_4h.shape[0], return_conf_int=True)
            inference_time_4h = time.time() - start_time_4h

        timeleft_4h = current_time_4h - self.last_run_time_4h[pair_4h]
        if timeleft_4h <= 43200 and timeleft_4h != 0:
            logger.info(f"{pair_4h} - ARIMA 4h Model re-optimized in {timeleft_4h:.2f} seconds")

        # Extract upper and lower confidence intervals
        lower_confidence_4h, upper_confidence_4h = conf_int_4h[:, 0], conf_int_4h[:, 1]
        logger.info(f"{pair_4h} - Inference time: {inference_time_4h:.2f} seconds | " \
            f"Current Price: {dataframe['OHLC4'].iloc[-1]:.7f} | 4h Future Forecast: {future_forecast_4h.iloc[-1]:.7f}")

        dataframe['rmse'] = 0
        dataframe['accuracy_perc'] = 0
        dataframe['reward'] = 0
        dataframe['rmse'] = np.sqrt(mean_squared_error(test_4h, future_forecast_4h))
        dataframe['accuracy_perc'] = 100 * (1 - (dataframe['rmse'].iloc[-1] / dataframe['OHLC4'].iloc[-1]))
        dataframe['reward'] = ((future_forecast_4h.iloc[-1] / dataframe['OHLC4'].iloc[-1]) - 1) * 100

        rmse_4h = dataframe['rmse'].iloc[-1]
        accuracy_perc_4h = dataframe['accuracy_perc'].iloc[-1]
        reward_4h = dataframe['reward'] .iloc[-1]
        # Apply rolling window operation to the 'OHLC4' column
        rolling_window_4h = dataframe['OHLC4'].rolling(self.window_4h.value)

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_value_4h = rolling_window_4h.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['move'] = ptp_value_4h / dataframe['OHLC4']
        dataframe['move_mean'] = dataframe['move'].mean()
        dataframe['move_mean_x'] = dataframe['move'].mean() * self.x_4h.value
        move_4h = '{:.2f}'.format(dataframe['move'].iloc[-1] * 100)
        move_mean_4h = '{:.2f}'.format(dataframe['move_mean'].iloc[-1] * 100)

        if future_forecast_4h.iloc[-1] > dataframe['OHLC4'].iloc[-1]:
            direction_4h = 'Up'
            dataframe['decision'] = 1
        else:
            direction_4h = 'Down'
            dataframe['decision'] = -1

        logger.info(f"{pair_4h} - Test RMSE: {rmse_4h:.3f} | Accuracy: {accuracy_perc_4h:.2f}% | Potential Profit: {move_4h}% | Avg. Profit: {move_mean_4h}% | 4h Trend: {direction_4h}")

        dataframe['arima_predictions'] = pd.Series(future_forecast_4h)
        dataframe['lower_confidence'] = pd.Series(lower_confidence_4h)
        dataframe['upper_confidence'] = pd.Series(upper_confidence_4h)

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_length.value)
        dataframe['vol_z_score'] = (dataframe['volume'] - dataframe['volume'].rolling(window=30).mean()) / dataframe['volume'].rolling(window=30).std()
        dataframe['vol_anomaly'] = np.where(dataframe['vol_z_score'] > 3, 1, 0)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['sma'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']
        dataframe['sma_up'] = dataframe['sma'] * self.up.value
        dataframe['sma_dn'] = dataframe['sma'] * self.dn.value

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # Calculate Bollinger Bands
        dataframe['bb_upperband'], dataframe['bb_middleband'], dataframe['bb_lowerband'] = ta.BBANDS(dataframe['close'], timeperiod=20)


        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        return dataframe

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair_1h = metadata['pair']
        current_time_1h = time.time()

        dataframe['OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        size_1h = len(dataframe) - 10
        train_1h, test_1h = model_selection.train_test_split(dataframe['OHLC4'], train_size=size_1h)

        # Initialize values for the current pair if not already done
        if pair_1h not in self.last_run_time_1h:
            self.last_run_time_1h[pair_1h] = current_time_1h

            logger.info(f"Initial ARIMA 1h Model Training for {pair_1h}")
            # Fit ARIMA model
            start_time_1h = time.time()
            self.arima_model[pair_1h] = auto_arima(train_1h, start_p=1, start_q=1, start_P=1, start_Q=1,
                                                max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=False,
                                                stepwise=True, suppress_warnings=True, D=10, max_D=20,
                                                error_action='ignore')
            fitting_time_1h = time.time() - start_time_1h
            logger.info(f"{pair_1h} - ARIMA 1h Model fitted in {fitting_time_1h:.2f} seconds")

        # Check if it's time to retrain for the current pair
        if current_time_1h - self.last_run_time_1h[pair_1h] >= 3600:  # Check if an 1 hour has passed
            logger.info(f"Auto Fitting ARIMA 1h Model for {pair_1h}")
            self.last_run_time_1h[pair_1h] = current_time_1h

            # Fit ARIMA model
            start_time_1h = time.time()
            self.arima_model[pair_1h] = auto_arima(train_1h, start_p=1, start_q=1, start_P=1, start_Q=1,
                                                max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=False,
                                                stepwise=True, suppress_warnings=True, D=10, max_D=20,
                                                error_action='ignore')
            fitting_time_1h = time.time() - start_time_1h
            logger.info(f"{pair_1h} - ARIMA 1h Model fitted in {fitting_time_1h:.2f} seconds")

        # Use the previously fitted ARIMA model for forecasting
        if self.arima_model[pair_1h] is not None:
            start_time_1h = time.time()
            future_forecast_1h, conf_int_1h = self.arima_model[pair_1h].predict(n_periods=test_1h.shape[0], return_conf_int=True)
            inference_time_1h = time.time() - start_time_1h

        # print(self.last_run_time_1h)
        timeleft_1h = current_time_1h - self.last_run_time_1h[pair_1h]
        if timeleft_1h <= 14400 and timeleft_1h != 0:
            logger.info(f"{pair_1h} - ARIMA 1h Model re-optimized in {timeleft_1h:.2f} seconds")

        # Calculate autocorrelation and add to dataframe
        acf1 = acf(dataframe['OHLC4'], fft=False)[1]
        dataframe['acf1'] = acf1

        # Assuming you want to use the autocorrelation value for the last candle
        acf1_last = dataframe['acf1'].iloc[-1]

        logger.info(f"Autocorrelation for {pair_1h}: {acf1_last:.6f}")

        # Extract upper and lower confidence intervals
        lower_confidence_1h, upper_confidence_1h = conf_int_1h[:, 0], conf_int_1h[:, 1]
        logger.info(f"{pair_1h} - Inference time: {inference_time_1h:.2f} seconds | " \
            f"Current Price: {dataframe['OHLC4'].iloc[-1]:.7f} | 1h Future Forecast: {future_forecast_1h.iloc[-1]:.7f}")

        dataframe['rmse'] = 0
        dataframe['accuracy_perc'] = 0
        dataframe['reward'] = 0
        dataframe['rmse'] = np.sqrt(mean_squared_error(test_1h, future_forecast_1h))
        dataframe['accuracy_perc'] = 100 * (1 - (dataframe['rmse'].iloc[-1] / dataframe['OHLC4'].iloc[-1]))
        dataframe['reward'] = ((future_forecast_1h.iloc[-1] / dataframe['OHLC4'].iloc[-1]) - 1) * 100

        rmse_1h = dataframe['rmse'].iloc[-1]
        accuracy_perc_1h = dataframe['accuracy_perc'].iloc[-1]
        reward_1h = dataframe['reward'] .iloc[-1]
        # Apply rolling window operation to the 'OHLC4' column
        rolling_window_1h = dataframe['OHLC4'].rolling(self.window_1h.value) # 21 hours

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_value_1h = rolling_window_1h.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['move'] = ptp_value_1h / dataframe['OHLC4']
        dataframe['move_mean'] = dataframe['move'].mean()
        dataframe['move_mean_x'] = dataframe['move'].mean() * self.x_1h.value
        move_1h = '{:.2f}'.format(dataframe['move'].iloc[-1] * 100)
        move_mean_1h = '{:.2f}'.format(dataframe['move_mean'].iloc[-1] * 100)

        if future_forecast_1h.iloc[-1] > dataframe['OHLC4'].iloc[-1]:
            direction_1h = 'Up'
            dataframe['decision'] = 1
        else:
            direction_1h = 'Down'
            dataframe['decision'] = -1

        logger.info(f"{pair_1h} - Test RMSE: {rmse_1h:.3f} | Accuracy: {accuracy_perc_1h:.2f}% | Potential Profit: {move_1h}% | Avg. Profit: {move_mean_1h}% | 1h Trend: {direction_1h}")

        dataframe['arima_predictions'] = pd.Series(future_forecast_1h)
        dataframe['lower_confidence'] = pd.Series(lower_confidence_1h)
        dataframe['upper_confidence'] = pd.Series(upper_confidence_1h)
        
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_length.value)
        dataframe['vol_z_score'] = (dataframe['volume'] - dataframe['volume'].rolling(window=30).mean()) / dataframe['volume'].rolling(window=30).std()
        dataframe['vol_anomaly'] = np.where(dataframe['vol_z_score'] > 3, 1, 0)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['sma'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']
        dataframe['sma_up'] = dataframe['sma'] * self.up.value
        dataframe['sma_dn'] = dataframe['sma'] * self.dn.value

        # Calculate Bollinger Bands
        dataframe['bb_upperband'], dataframe['bb_middleband'], dataframe['bb_lowerband'] = ta.BBANDS(dataframe['close'], timeperiod=20)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        pair = metadata['pair']
        current_time = time.time()

        dataframe['OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        size = len(dataframe) - 10
        train, test = model_selection.train_test_split(dataframe['OHLC4'], train_size=size)

        # Initialize values for the current pair if not already done
        if pair not in self.last_run_time:
            self.last_run_time[pair] = current_time

            logger.info(f"Initial ARIMA {self.timeframe} Model Training for {pair}")
            # Fit ARIMA model with error handling and performance tracking
            start_time = time.time()
            try:
                model = auto_arima(train, start_p=1, start_q=1, start_P=1, start_Q=1,
                                 max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=False,
                                 stepwise=True, suppress_warnings=True, D=10, max_D=20,
                                 error_action='ignore')
                
                # Validate model
                if not hasattr(model, 'order'):
                    raise ValueError("Model fitting failed - no order attribute")
                    
                # Store model and track performance
                self.arima_model[pair] = model
                fitting_time = time.time() - start_time
                logger.info(f"{pair} - ARIMA {self.timeframe} Model fitted in {fitting_time:.2f} seconds")
                logger.info(f"Model Order: {model.order} | AIC: {model.aic():.2f}")
                
                # Track historical performance
                if not hasattr(self, 'model_history'):
                    self.model_history = {}
                if pair not in self.model_history:
                    self.model_history[pair] = []
                    
                # Calculate in-sample performance
                preds = model.predict_in_sample()
                rmse = np.sqrt(mean_squared_error(train, preds))
                self.model_history[pair].append({
                    'timestamp': current_time,
                    'order': model.order,
                    'aic': model.aic(),
                    'rmse': rmse,
                    'fitting_time': fitting_time
                })
                
            except Exception as e:
                logger.error(f"ARIMA model fitting failed for {pair}: {str(e)}")
                # Use previous model if available
                if pair in self.arima_model:
                    logger.warning(f"Using previous model for {pair}")
                    return
                else:
                    raise ValueError(f"Initial model fitting failed for {pair}")

        # Check if it's time to retrain for the current pair
        if current_time - self.last_run_time[pair] >= 3600:  # Check if an hour has passed
            logger.info(f"Auto Fitting ARIMA {self.timeframe} Model for {pair}")
            self.last_run_time[pair] = current_time

            # Fit ARIMA model
            start_time = time.time()
            self.arima_model[pair] = auto_arima(train, start_p=1, start_q=1, start_P=1, start_Q=1,
                                                max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=False,
                                                stepwise=True, suppress_warnings=True, D=10, max_D=20,
                                                error_action='ignore')
            fitting_time = time.time() - start_time
            logger.info(f"{pair} - ARIMA {self.timeframe} Model fitted in {fitting_time:.2f} seconds")

        # Use the previously fitted ARIMA model for forecasting
        if self.arima_model[pair] is not None:
            start_time = time.time()
            future_forecast, conf_int = self.arima_model[pair].predict(n_periods=test.shape[0], return_conf_int=True)
            inference_time = time.time() - start_time

        timeleft = current_time - self.last_run_time[pair]
        if timeleft <= 3600 and timeleft != 0:
            logger.info(f"{pair} - ARIMA {self.timeframe} Model re-optimized in {timeleft:.2f} seconds")

        # Extract upper and lower confidence intervals
        lower_confidence, upper_confidence = conf_int[:, 0], conf_int[:, 1]
        logger.info(f"{pair} - Inference time: {inference_time:.2f} seconds | " \
            f"Current Price: {dataframe['OHLC4'].iloc[-1]:.7f} | {self.timeframe} Future Forecast: {future_forecast.iloc[-1]:.7f}")

        dataframe['rmse'] = 0
        dataframe['accuracy_perc'] = 0
        dataframe['reward'] = 0
        dataframe['rmse'] = np.sqrt(mean_squared_error(test, future_forecast))
        dataframe['accuracy_perc'] = 100 * (1 - (dataframe['rmse'].iloc[-1] / dataframe['OHLC4'].iloc[-1]))
        dataframe['reward'] = ((future_forecast.iloc[-1] / dataframe['OHLC4'].iloc[-1]) - 1) * 100

        rmse = dataframe['rmse'].iloc[-1]
        accuracy_perc = dataframe['accuracy_perc'].iloc[-1]
        reward = dataframe['reward'] .iloc[-1]
        # Apply rolling window operation to the 'OHLC4' column
        rolling_window = dataframe['OHLC4'].rolling(self.window.value) # 5.25 hrs

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_value = rolling_window.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['move'] = ptp_value / dataframe['OHLC4']
        dataframe['move_mean'] = dataframe['move'].mean()
        dataframe['move_mean_x'] = dataframe['move'].mean() * self.x.value
        move = '{:.2f}'.format(dataframe['move'].iloc[-1] * 100)
        move_mean = '{:.2f}'.format(dataframe['move_mean'].iloc[-1] * 100)

        if future_forecast.iloc[-1] > dataframe['OHLC4'].iloc[-1]:
            direction = 'Up'
            dataframe['decision'] = 1
        else:
            direction = 'Down'
            dataframe['decision'] = -1

        logger.info(f"{pair} - Test RMSE: {rmse:.3f} | Accuracy: {accuracy_perc:.2f}% | Potential Profit: {move}% | Avg. Profit: {move_mean}% | {self.timeframe} Trend: {direction}")

        dataframe['arima_predictions'] = pd.Series(future_forecast)
        dataframe['lower_confidence'] = pd.Series(lower_confidence)
        dataframe['upper_confidence'] = pd.Series(upper_confidence)
        
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_length.value)
        dataframe['atr_pcnt'] = (ta.ATR(dataframe, timeperiod=self.atr_length.value) / dataframe['OHLC4'])
        dataframe['vol_z_score'] = (dataframe['volume'] - dataframe['volume'].rolling(window=30).mean()) / dataframe['volume'].rolling(window=30).std()
        dataframe['vol_anomaly'] = np.where(dataframe['vol_z_score'] > 3, 1, 0)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['sma'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']
        dataframe['sma_up'] = dataframe['sma'] * self.up.value
        dataframe['sma_dn'] = dataframe['sma'] * self.dn.value

        dataframe['max_l'] = dataframe['OHLC4'].rolling(120).max() / dataframe['OHLC4'] - 1
        dataframe['min_l'] = abs(dataframe['OHLC4'].rolling(120).min() / dataframe['OHLC4'] - 1)

        dataframe['max'] = dataframe['OHLC4'].rolling(4).max() / dataframe['OHLC4'] - 1
        dataframe['min'] = abs(dataframe['OHLC4'].rolling(4).min() / dataframe['OHLC4'] - 1)

        dataframe['rsi_overbought'] = (dataframe['rsi'] > 70).astype('int')
        dataframe['volume_increase'] = (dataframe['volume'] > dataframe['volume'].shift()).astype('int')
        dataframe['price_below_sma'] = (dataframe['close'] < dataframe['sma']).astype('int')
        dataframe['price_drop'] = (dataframe['close'] < dataframe['close'].shift()).astype('int')
        dataframe['williamr'] = ta.WILLR(dataframe, timeperiod=10)

        # Calculate Bollinger Bands
        dataframe['bb_upperband'], dataframe['bb_middleband'], dataframe['bb_lowerband'] = ta.BBANDS(dataframe['close'], timeperiod=20)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------

        dataframe['CDL3STARSINSOUTH'] = ta.CDL3STARSINSOUTH(dataframe) # values [0, 100]
        dataframe['CDLDOJISTAR'] = ta.CDLDOJISTAR(dataframe) # values [0, -100, 100]
        dataframe['CDLTAKURI'] = ta.CDLTAKURI(dataframe) # values [0, 100]
        dataframe['CDLTRISTAR'] = ta.CDLTRISTAR(dataframe) # values [0, -100, 100]
        dataframe['CDLLONGLEGGEDDOJI'] = ta.CDLLONGLEGGEDDOJI(dataframe) # values [0, -100, 100]
        dataframe['CDLMARUBOZU'] = ta.CDLMARUBOZU(dataframe) # values [0, -100, 100]
        dataframe['CDLMORNINGDOJISTAR'] = ta.CDLMORNINGDOJISTAR(dataframe)

        dataframe['CDL2CROWS'] = ta.CDL2CROWS(dataframe) # values [0, 100]
        dataframe['CDLHARAMICROSS'] = ta.CDLHARAMICROSS(dataframe) # values [0, -100, 100]
        dataframe['CDLRISEFALL3METHODS'] = ta.CDLRISEFALL3METHODS(dataframe) # values [0, 100]
        dataframe['CDLSEPARATINGLINES'] = ta.CDLSEPARATINGLINES(dataframe) # values [0, -100, 100]
        dataframe['CDLSTALLEDPATTERN'] = ta.CDLSTALLEDPATTERN(dataframe) # values [0, 100]
        dataframe['CDLTHRUSTING'] = ta.CDLTHRUSTING(dataframe) # values [0, 100]

        # # Hammer: values [0, 100]
        dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        # # Inverted Hammer: values [0, 100]
        dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
        # # Dragonfly Doji: values [0, 100]
        dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
        # # Piercing Line: values [0, 100]
        dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe)  # values [0, 100]
        # # Morningstar: values [0, 100]
        dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe)  # values [0, 100]
        # # Three White Soldiers: values [0, 100]
        dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe)  # values [0, 100]

        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        # # Hanging Man: values [0, 100]
        dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
        # # Shooting Star: values [0, 100]
        dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
        # # Gravestone Doji: values [0, 100]
        dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
        # # Dark Cloud Cover: values [0, 100]
        dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
        # # Evening Doji Star: values [0, 100]
        dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
        # # Evening Star: values [0, 100]
        dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)

        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------
        # # Three Inside Up/Down: values [0, -100, 100]
        dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe)  # values [0, -100, 100]
        # # Three Line Strike: values [0, -100, 100]
        dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
        # # Spinning Top: values [0, -100, 100]
        dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe)  # values [0, -100, 100]
        # # Engulfing: values [0, -100, 100]
        dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe)  # values [0, -100, 100]
        # # Harami: values [0, -100, 100]
        dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe)  # values [0, -100, 100]
        # # Three Outside Up/Down: values [0, -100, 100]
        dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe)  # values [0, -100, 100]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        bollinger_bullish_divergence = (
            (dataframe['close'].shift(1) > dataframe['close']) &
            (dataframe['bb_lowerband'].shift(1) <= dataframe['bb_lowerband'])
        )

        bollinger_bearish_divergence = (
            (dataframe['close'].shift(1) < dataframe['close']) &
            (dataframe['bb_upperband'].shift(1) >= dataframe['bb_upperband'])
        )

        macd_bullish_divergence = (
            (dataframe['close'].shift(1) > dataframe['close']) &
            (dataframe['macd'].shift(1) <= dataframe['macd'])
        )

        macd_bearish_divergence = (
            (dataframe['close'].shift(1) < dataframe['close']) &
            (dataframe['macd'].shift(1) >= dataframe['macd'])
        )

        long_condition = (
            (dataframe['close'] > dataframe['bb_lowerband']) &
            (qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])) &
            (dataframe['rsi'] < 50) &
            (bollinger_bullish_divergence | macd_bullish_divergence)
        )

        dataframe.loc[long_condition, 'enter_long'] = 1
        dataframe.loc[long_condition, 'enter_tag'] = 'L - Bullish Divergence'

        short_condition = (
            (dataframe['close'] < dataframe['bb_upperband']) &
            (qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])) &
            (dataframe['rsi'] > 50) &
            (bollinger_bearish_divergence | macd_bearish_divergence)
        )

        dataframe.loc[short_condition, 'enter_short'] = 1
        dataframe.loc[short_condition, 'enter_tag'] = 'S - Bearish Divergence'

        condition1 = (
            (dataframe['decision_1h'] == 1) &
            (dataframe['EWO_4h'] > self.ewo_high.value) &
            (dataframe['move_1h'] >= dataframe['move_mean_1h']) &
            (dataframe['move_1h'].shift(5) < dataframe['move_mean_1h'].shift(5)) &
            (dataframe['volume_1h'] > 0)
        )
        dataframe.loc[condition1, 'enter_long'] = 1
        dataframe.loc[condition1, 'enter_tag'] = 'EWO High 1ne'

        condition2 = (
            (dataframe['decision_4h'] == 1) &
            (dataframe['EWO_4h'] > self.ewo_high_2.value) &
            (dataframe['move_4h'] >= dataframe['move_mean_4h']) &
            (dataframe['move_4h'].shift(5) < dataframe['move_mean_4h'].shift(5)) &
            (dataframe['volume_4h'] > 0)
        )
        dataframe.loc[condition2, 'enter_long'] = 1
        dataframe.loc[condition2, 'enter_tag'] = 'EWO High 2wo'

        condition3 = (
            (dataframe['decision_4h'] == 1) &
            (dataframe['EWO_4h'] < self.ewo_low.value) &
            (dataframe['move_4h'] <= dataframe['move_mean_4h']) &
            (dataframe['move_4h'].shift(5) > dataframe['move_mean_4h'].shift(5)) &
            (dataframe['volume_4h'] > 0)
        )
        dataframe.loc[condition3, 'enter_long'] = 1
        dataframe.loc[condition3, 'enter_tag'] = 'EWO Low-Low'

        condition4 = (
            (dataframe['decision'] == 1) &
            (dataframe['move'] >= dataframe['move_mean']) &
            (dataframe['move'].shift(5) < dataframe['move_mean'].shift(5)) &
            (qtpylib.crossed_above(dataframe['macd_4h'], dataframe['macdsignal_4h'])) &
            (dataframe['rsi_4h'] < 50) &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[condition4, 'enter_long'] = 1
        dataframe.loc[condition4, 'enter_tag'] = 'Up Trend Soon 15m (4h MACD + RSI)'

        condition5 = (
            (dataframe['decision_1h'] == 1) &
            (dataframe['move_1h'] >= dataframe['move_mean_1h']) &
            (dataframe['move_1h'].shift(2) < dataframe['move_mean_1h'].shift(2)) &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[condition5, 'enter_long'] = 1
        dataframe.loc[condition5, 'enter_tag'] = 'Up Trend Soon 1h'

        condition6 = (
            (dataframe['decision_4h'] == 1) &
            (dataframe['move_4h'] >= dataframe['move_mean_4h']) &
            (dataframe['move_4h'].shift(2) < dataframe['move_mean_4h'].shift(2)) &
            (dataframe['volume_4h'] > 0)
        )
        dataframe.loc[condition6, 'enter_long'] = 1
        dataframe.loc[condition6, 'enter_tag'] = 'Up Trend Soon 4h'

        condition7 = (
            (dataframe['decision_4h'] == 1) &
            (dataframe['move'] >= dataframe['move_mean_x']) &
            (dataframe['min'] < dataframe['max']) &
            (dataframe['min_l'] < dataframe['max_l']) &
            (dataframe['max_l'] < dataframe['atr_pcnt']) &
            (dataframe['OHLC4'] < dataframe['sma']) &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[condition7, 'enter_long'] = 1
        dataframe.loc[condition7, 'enter_tag'] = 'Move Mean Fib below sma'

        condition8 = (
            (dataframe['decision_4h'] == 1) &
            (dataframe['move'] >= dataframe['move_mean']) &
            (dataframe['min'] < dataframe['max']) &
            (dataframe['OHLC4'] > dataframe['sma']) &
            (dataframe['sma_up'].shift() < dataframe['sma']) &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[condition8, 'enter_long'] = 1
        dataframe.loc[condition8, 'enter_tag'] = 'Hope this works...'

        condition9 = (
            (dataframe['decision_1h'] == -1) &
            (dataframe['decision_4h'] == -1) &
            (dataframe['move'] <= dataframe['move_mean_x']) &
            (dataframe['max'] < dataframe['min']) &
            (dataframe['min_l'] > dataframe['max_l']) &
            (dataframe['max_l'] > dataframe['atr_pcnt']) &
            (dataframe['OHLC4'] > dataframe['sma_up']) &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[condition9, 'enter_short'] = 1
        dataframe.loc[condition9, 'enter_tag'] = 'Move Mean Fib above sma_up'

        condition10 = (
            (dataframe['decision_1h'] == -1) &
            (dataframe['decision_4h'] == -1) &
            (dataframe['move'] <= dataframe['move_mean']) &
            (dataframe['move'].shift(6) > dataframe['move_mean'].shift(6)) &
            (dataframe['max'] < dataframe['min']) &
            (dataframe['min_l'] > dataframe['max_l']) &
            (dataframe['OHLC4'] > dataframe['sma']) &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[condition10, 'enter_short'] = 1
        dataframe.loc[condition10, 'enter_tag'] = 'Down Trend Soon above sma'

        condition11 = (
            (dataframe['reward'] >= 0.08) &
            (dataframe['reward_4h'] >= abs(self.predicted_profit_percentage_4h_big.value)) &
            (dataframe['accuracy_perc'] >= 0.9) &
            (dataframe['accuracy_perc_4h'] >= 98.0) &
            (dataframe['move'] >= dataframe['move_mean']) &
            (dataframe['move'].shift(6) < dataframe['move_mean'].shift(6)) &
            (dataframe['min'] < dataframe['max']) &
            (dataframe['min'] != 0) &
            (dataframe['volume'] > 0) &
            (dataframe['OHLC4'] < dataframe['sma'])
        )
        dataframe.loc[condition11, 'enter_long'] = 1
        dataframe.loc[condition11, 'enter_tag'] = 'Big Up Trend 4h'

        condition12 = (
            (dataframe['reward'] <= -0.10) &
            (dataframe['reward_4h'] <= -abs(self.predicted_profit_percentage_4h_big.value)) &
            (dataframe['accuracy_perc'] >= 98.0) &
            (dataframe['accuracy_perc_4h'] >= 98.0) &
            (dataframe['move'] <= dataframe['move_mean']) &
            (dataframe['move'].shift(6) > dataframe['move_mean'].shift(6)) &
            (dataframe['max'] < dataframe['min']) &
            (dataframe['max'] != 0) &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[condition12, 'enter_short'] = 1
        dataframe.loc[condition12, 'enter_tag'] = 'Big Down Trend 4h'

        condition13 = (
            (dataframe['reward'] >= self.predicted_profit_percentage.value) &
            (dataframe['reward_4h'] >= self.predicted_profit_percentage_4h.value) &
            (dataframe['accuracy_perc'] >= 0.9) &
            (dataframe['accuracy_perc_4h'] >= 98.0) &
            (dataframe['decision_4h'] == 1) &
            (dataframe['volume'] > 0) &
            (dataframe['williamr'] <= -90) &
            (dataframe['williamr'].shift(10) <= -85)
        )
        dataframe.loc[condition13, 'enter_long'] = 1
        dataframe.loc[condition13, 'enter_tag'] = 'W Up Trend Soon'

        condition14 = (
            (dataframe['reward'] <= -abs(self.predicted_profit_percentage.value)) &
            (dataframe['reward_4h'] <= -abs(self.predicted_profit_percentage_4h.value)) &
            (dataframe['accuracy_perc'] >= 0.9) &
            (dataframe['accuracy_perc_4h'] >= 98.0) &
            (dataframe['decision_4h'] == -1) &
            (dataframe['williamr'] <= -10) &
            (dataframe['williamr'].shift(10) <= -25) &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[condition14, 'enter_short'] = 1
        dataframe.loc[condition14, 'enter_tag'] = 'W Down Trend Soon'

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        condition1 = (
            (dataframe['decision'] == -1) &
            (dataframe['move'] >= dataframe['move_mean']) &
            (dataframe['move'].shift(5) < dataframe['move_mean'].shift(5)) &
            (qtpylib.crossed_below(dataframe['macd_4h'], dataframe['macdsignal_4h'])) &
            (dataframe['rsi_4h'] > 50) &
            (dataframe['volume'] > 0)
            )

        dataframe.loc[condition1, 'exit_long'] = 1
        dataframe.loc[condition1, 'exit_tag'] = 'Down Trend Soon 15m (4h MACD + RSI)'

        condition2 = (
            (dataframe['decision_1h'] == -1) &
            (dataframe['move_1h'] >= dataframe['move_mean_1h']) &
            (dataframe['move_1h'].shift(2) < dataframe['move_mean_1h'].shift(2)) &
            (dataframe['volume'] > 0)
            )

        dataframe.loc[condition2, 'exit_long'] = 1
        dataframe.loc[condition2, 'exit_tag'] = 'Down Trend Soon 1h'

        condition3 = (
            (dataframe['decision_4h'] == -1) &
            (dataframe['move_4h'] >= dataframe['move_mean_4h']) &
            (dataframe['move_4h'].shift(2) < dataframe['move_mean_4h'].shift(2)) &
            (dataframe['volume'] > 0)
            )

        dataframe.loc[condition3, 'exit_long'] = 1
        dataframe.loc[condition3, 'exit_tag'] = 'Down Trend Soon 4h'

        condition4 = (

            (dataframe['decision'] == 1) &
            (dataframe['move'] <= dataframe['move_mean']) &
            (dataframe['move'].shift(5) > dataframe['move_mean'].shift(5)) &
            (dataframe['volume'] > 0)
        )

        dataframe.loc[condition4, 'exit_short'] = 1
        dataframe.loc[condition4, 'exit_tag'] = 'Up Trend Soon 15m'

        condition5 = (
            (dataframe['decision_4h'] == 1) &
            (dataframe['move_4h'] <= dataframe['move_mean_4h']) &
            (dataframe['move_4h'].shift(5) > dataframe['move_mean_4h'].shift(5)) &
            (dataframe['volume'] > 0)
        )

        dataframe.loc[condition5, 'exit_short'] = 1
        dataframe.loc[condition5, 'exit_tag'] = 'Up Trend Soon 4h'

        if self.use_exit_signal:
            dataframe.loc[
                (
                    (dataframe['close_1h'] < dataframe['bb_middleband_1h']) |
                    (dataframe['close_4h'] < dataframe['bb_middleband_4h']) |
                    (dataframe['rsi_1h'] > 70)
                ),
                'exit_long'
            ] = 1

            dataframe.loc[
                (
                    (dataframe['close_1h'] > dataframe['bb_middleband_1h']) |
                    (dataframe['close_4h'] > dataframe['bb_middleband_4h']) |
                    (dataframe['rsi_1h'] < 30)
                ),
                'exit_short'
            ] = 1

        return dataframe

# # ADD BEST RMSE & MAPE TO DECIDE TREND UP OR DOWN OR ENTRY SIGNAL - correct values for arima paramters depeding on the timeframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float, 
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], 
                side: str, **kwargs) -> float:
        """Dynamic leverage adjustment based on market conditions"""
        
        # Get volatility measure
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        atr = dataframe['atr'].iloc[-1]
        volatility_factor = min(max(atr / current_rate, 0.5), 2.0)
        
        # Adjust base leverage based on volatility
        base_leverage = min(5 / volatility_factor, max_leverage)

        if max_leverage is None:
            raise ValueError('Max leverage cannot be None')
        if not isinstance(max_leverage, (int, float)):
            raise ValueError('Max leverage must be a number')

        base_leverage = 5

        proposed_leverage = 3

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        bullish_patterns = ['CDLRISEFALL3METHODS', 'CDLMORNINGDOJISTAR', 'CDLTAKURI', 'CDL3STARSINSOUTH', 'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLPIERCING', 'CDLMORNINGSTAR', 'CDL3WHITESOLDIERS']
        bearish_patterns = ['CDLTHRUSTING', 'CDLSTALLEDPATTERN', 'CDL2CROWS', 'CDLHANGINGMAN', 'CDLSHOOTINGSTAR', 'CDLGRAVESTONEDOJI', 'CDLDARKCLOUDCOVER', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR']
        bullish_bearish_patterns = ['CDLSEPARATINGLINES', 'CDLHARAMICROSS', 'CDLMARUBOZU', 'CDLLONGLEGGEDDOJI', 'CDLTRISTAR', 'CDLDOJISTAR', 'CDL3LINESTRIKE', 'CDLSPINNINGTOP', 'CDLENGULFING', 'CDLHARAMI', 'CDL3OUTSIDE', 'CDL3INSIDE']
        pattern_found = False

        if side == 'long':
            for pattern in bullish_patterns:
                pattern_value = current_candle.get(pattern)
                if pattern_value is not None and pattern_value == 100:

                    # Bullish signal
                    base_leverage = max(1.0, min(base_leverage * 1.5, max_leverage))
                    print(f'Bullish Pattern: {pattern} Value: {pattern_value} Adjusted Leverage: {base_leverage}')
                    self.dp.send_msg(f'Found {pattern} pattern for {pair} Leverage updated to {base_leverage}')
                    pattern_found = True
                    break

        elif side == 'short':
            for pattern in bearish_patterns:
                pattern_value = current_candle.get(pattern)
                if pattern_value is not None and pattern_value == 100:

                    # Bearish signal
                    base_leverage = max(1.0, min(base_leverage * 1.5, max_leverage))
                    print(f'Bearish Pattern: {pattern} Value: {pattern_value} Adjusted Leverage: {base_leverage}')
                    self.dp.send_msg(f'Found {pattern} pattern for {pair} Leverage updated to {base_leverage}')
                    pattern_found = True
                    break

        if not pattern_found and side == 'long':
            for pattern in bullish_bearish_patterns:
                pattern_value = current_candle.get(pattern)
                if pattern_value is not None and pattern_value == 100:

                    # Bullish signal
                    base_leverage = max(1.0, min(base_leverage * 2, max_leverage))
                    print(f'Bullish Pattern: {pattern} Value: {pattern_value} Adjusted Leverage: {base_leverage}')
                    self.dp.send_msg(f'Found {pattern} bullish pattern for {pair} Leverage updated to {base_leverage}')
                    pattern_found = True
                    break

        elif not pattern_found and side == 'short':
            for pattern in bullish_bearish_patterns:
                pattern_value = current_candle.get(pattern)
                if pattern_value is not None and pattern_value == -100:

                    # Bearish signal
                    base_leverage = max(1.0, min(base_leverage * 2, max_leverage))
                    print(f'Bearish Pattern: {pattern} Value: {pattern_value} Adjusted Leverage: {base_leverage}')
                    self.dp.send_msg(f'Found {pattern} bearish pattern for {pair} Leverage updated to {base_leverage}')
                    pattern_found = True
                    break

        # Apply maximum and minimum limits if any pattern is found
        if pattern_found:
            adjusted_leverage = max(min(base_leverage, max_leverage), 1.0)  # Apply max and min limits
            print(f'Base Leverage: {proposed_leverage}')
            print(f'Adjusted Leverage: {adjusted_leverage}')
            return adjusted_leverage  # Return the adjusted leverage

        # If no pattern matches, return the proposed_leverage
        print(f'No Pattern Matched for {pair}. Using Proposed Leverage: {proposed_leverage}')
        self.dp.send_msg(f'No Pattern Matched for {pair}. Using Proposed Leverage: {proposed_leverage}')
        return proposed_leverage

    def adjust_trade_position(self, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float,
                            min_stake: Optional[float], max_stake: float, current_entry_rate: float,
                            current_exit_rate: float, current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Adjusts trade position size based on leverage and market conditions.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        if len(dataframe) > 2:
            last_candle = dataframe.iloc[-1].squeeze()
            previous_candle = dataframe.iloc[-2].squeeze()
            signal_name = 'enter_long' if not trade.is_short else 'enter_short'
            prior_date = date_minus_candles(self.timeframe, 1, current_time)

            if last_candle.get(signal_name) == 1 and previous_candle.get(signal_name) != 1 \
                    and trade.nr_of_successful_entries < 2 \
                    and trade.orders[-1].order_date_utc < prior_date:

                proposed_leverage = trade.leverage
                side = 'long' if not trade.is_short else 'short'
                default_max_leverage = 20
                adjusted_leverage = self.leverage(trade.pair, current_time, current_rate, proposed_leverage,
                                                default_max_leverage, trade.enter_tag, side)

                new_position_size = trade.stake_amount * (adjusted_leverage / proposed_leverage)

                # Ensure new_position_size adheres to limits
                if min_stake is not None:
                    new_position_size = max(new_position_size, min_stake)
                    new_position_size = min(new_position_size, max_stake)

                self.dp.send_msg(f"Adjusting position size for {trade.pair}. Previous Leverage: {proposed_leverage} for {trade.pair}, "
                                f"Adjusted Leverage for {trade.pair}: {adjusted_leverage}, Previous Stake for {trade.pair}: {trade.stake_amount}, "
                                f"New Stake: {new_position_size}")
                return new_position_size

        return None

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        adjusted_leverage = getattr(self, 'adjusted_leverage', 1)

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if current_candle['max_l'] > 0.0035:
            if (current_profit > self.tsl_target3.value):
                self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl3 {self.tsl_target3.value}/{self.ts3.value} activated')
                logger.info(f'*** {pair} *** Profit {current_profit} - lvl3 {self.tsl_target3.value}/{self.ts3.value} activated')
                return self.ts3.value * adjusted_leverage
            if (current_profit > self.tsl_target2.value):
                self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl2 {self.tsl_target2.value}/{self.ts2.value} activated')
                logger.info(f'*** {pair} *** Profit {current_profit} - lvl2 {self.tsl_target2.value}/{self.ts2.value} activated')
                return self.ts2.value * adjusted_leverage
            if (current_profit > self.tsl_target1.value):
                self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl1 {self.tsl_target1.value}/{self.ts1.value} activated')
                logger.info(f'*** {pair} *** Profit {current_profit} - lvl1 {self.tsl_target1.value}/{self.ts1.value} activated')
                return self.ts1.value * adjusted_leverage
            if (current_profit > self.tsl_target0.value):
                self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl0 {self.tsl_target0.value}/{self.ts0.value} activated')
                logger.info(f'*** {pair} *** Profit {current_profit} - lvl0 {self.tsl_target0.value}/{self.ts0.value} activated')
                return self.ts0.value * adjusted_leverage
        else:
            if (current_profit > self.tsl_target0.value):
                self.dp.send_msg(f'*** {pair} *** Profit {current_profit} SWINGING FOR THE MOON!!!')
                return 0.99 * adjusted_leverage

        return self.stoploss

def EWO(dataframe, ema_length=6, ema2_length=42):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif
