# install autogluon
# pip install -U pip
# pip install -U setuptools wheel
# pip install autogluon
# Warning GPU ONLY
import os
import time
import numpy as np
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import IStrategy, IntParameter
from freqtrade.persistence import Trade
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import torch
from datetime import datetime, timedelta
import logging
import pickle
from technical import qtpylib
import warnings
from pandas.errors import PerformanceWarning
import threading
import shutil

warnings.simplefilter(action="ignore", category=PerformanceWarning)

logger = logging.getLogger(__name__)

class AutoGluonTankAIHybridV2(IStrategy):
    timeframe = '1h'
    startup_candle_count = 600
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True  # Only trail after offset
    can_short = False  # Enable short-selling

    prediction_length = 3
    train_interval = 24 * 60 * 60  # 24 hours in seconds
    min_train_size = 500
    model_path = "user_data/AutoGluonTankAIHybridV2"
    model_path_temp = model_path + "_temp"

    use_dynamic_stoploss = False
    stoploss_atr_multiplier = 2.0
    process_only_new_candles = True
    forecast_uncertainty_threshold = 0.05
    min_profit_threshold = 0.01
    min_exit_signal_interval = 300  # 5-minute cooldown for exit signals (in seconds)

    min_drop_threshold = -0.01  # 1% drop threshold for exiting long positions
    min_rise_threshold = 0.01  # 1% rise threshold for exiting short positions

    u_window_size = IntParameter(70, 300, default=120, space='buy', optimize=True, load=True)
    l_window_size = IntParameter(20, 50, default=42, space='buy', optimize=True, load=True)

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 5},
            {"method": "StoplossGuard", "lookback_period_candles": 24, "trade_limit": 4, "stop_duration_candles": 2, "only_per_pair": False},
            {"method": "LowProfitPairs", "lookback_period_candles": 6, "trade_limit": 2, "stop_duration_candles": 60, "required_profit": 0.02},
            {"method": "LowProfitPairs", "lookback_period_candles": 24, "trade_limit": 4, "stop_duration_candles": 2, "required_profit": 0.01}
        ]

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.last_train_time = None
        self.last_known_train_time = None
        self.predictor = None
        self.exit_order_generated = {}
        self.last_exit_price = {}
        self.last_exit_signal_time = {}
        self.training_thread = None
        self.load_predictor()

    def plot_config(self):
        return {
            "main_plot": {"forecast_median": {"color": "green"}, "forecast_upper": {"color": "lightgreen"}, "forecast_lower": {"color": "lightgreen"}},
            "subplots": {"RSI": {"rsi": {"color": "blue"}}}
        }

    def load_predictor(self):
        if os.path.exists(self.model_path):
            try:
                self.predictor = TimeSeriesPredictor.load(self.model_path)
                if not self.predictor.model_names():
                    logger.warning("Loaded predictor has no models. Will train a new one.")
                    self.predictor = None
                    self.last_train_time = None
                else:
                    with open(os.path.join(self.model_path, "last_train_time.pkl"), "rb") as f:
                        self.last_train_time = pickle.load(f)
                        self.last_known_train_time = self.last_train_time
                    logger.info(f"Loaded model from {self.model_path}. Last trained: {self.last_train_time}")
            except Exception as e:
                logger.warning(f"Error loading model: {e}. Will train a new one when needed.")
                self.predictor = None
                self.last_train_time = None
        else:
            logger.info("No existing model found. Will train when conditions are met.")
            self.predictor = None
            self.last_train_time = None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'close' in df.columns:
            df['close'] = df['close'].rolling(window=3, min_periods=1).median()
        return df

    def prepare_ts_data(self, dataframe: pd.DataFrame):
        if dataframe.empty or dataframe[['close']].isnull().all().all():
            logger.warning("Dataframe is empty or contains all NaN values. Skipping preparation.")
            return None
        df = dataframe.copy().reset_index(drop=True)
        df = self.clean_data(df)
        if 'date' in df.columns:
            ts = pd.to_datetime(df['date'], errors='coerce')
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert('UTC').dt.tz_localize(None)
            else:
                ts = ts.dt.tz_localize(None)
            df['timestamp'] = ts
        else:
            ts = pd.to_datetime(df.index, errors='coerce').dt.tz_localize(None)
            df['timestamp'] = ts
        df['item_id'] = 'crypto_pair'
        df['hour'] = df['timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        price_cols = ['close', 'lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'cp', 'h0', 'h1', 'h2', 'atr', 'ma5', 'ma20', 'ma50']
        for col in price_cols:
            if col in df.columns:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        if 'rsi' in df.columns:
            df['rsi'] = (df['rsi'] - df['rsi'].mean()) / df['rsi'].std()
        return TimeSeriesDataFrame.from_data_frame(
            df[['item_id', 'timestamp', 'close', 'lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'cp', 'h0', 'h1', 'h2', 'rsi', 'atr', 'ma5', 'ma20', 'ma50', 'hour_sin', 'hour_cos']],
            id_column="item_id",
            timestamp_column="timestamp"
        )

    def _check_and_reload_predictor(self):
        last_train_time_file = os.path.join(self.model_path, "last_train_time.pkl")
        if os.path.exists(last_train_time_file):
            try:
                with open(last_train_time_file, "rb") as f:
                    new_train_time = pickle.load(f)
                if self.last_known_train_time is None or new_train_time > self.last_known_train_time:
                    logger.info(f"New model detected, last trained at {new_train_time}. Reloading predictor.")
                    self.load_predictor()
                    self.last_known_train_time = new_train_time
            except Exception as e:
                logger.warning(f"Failed to check last train time: {e}")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        self._check_and_reload_predictor()

        dataframe['forecast_median'] = np.nan
        dataframe['forecast_upper'] = np.nan
        dataframe['forecast_lower'] = np.nan
        dataframe['ma5'] = dataframe['close'].rolling(window=5).mean()
        dataframe['ma20'] = dataframe['close'].rolling(window=20).mean()
        dataframe['ma50'] = dataframe['close'].rolling(window=50).mean()
        pair = metadata['pair']
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        cycle_period = 80
        harmonics = [40, 27, 20]
        freq, power = perform_fft(dataframe['ha_close'], window_size=self.u_window_size.value)
        positive_mask = (1 / freq > self.l_window_size.value) & (1 / freq < self.u_window_size.value)
        positive_freqs = freq[positive_mask]
        positive_power = power[positive_mask]
        power_threshold = 0.01 * np.max(positive_power) if len(positive_power) > 0 else 0
        significant_indices = positive_power > power_threshold
        dominant_freq_index = np.argmax(positive_power[significant_indices])
        dominant_freq = positive_freqs[significant_indices][dominant_freq_index]
        cycle_period = int(np.abs(1 / dominant_freq)) if dominant_freq != 0 else 100
        harmonics = [cycle_period / (i + 1) for i in range(1, 4)]
        self.cp = int(cycle_period)
        self.h0 = int(harmonics[0])
        self.h1 = int(harmonics[1])
        self.h2 = int(harmonics[2])

        dataframe['cp'] = dataframe['ha_close'].ewm(span=int(cycle_period)).mean()
        dataframe['h0'] = dataframe['ha_close'].ewm(span=int(harmonics[0])).mean()
        dataframe['h1'] = dataframe['ha_close'].ewm(span=int(harmonics[1])).mean()
        dataframe['h2'] = dataframe['ha_close'].ewm(span=int(harmonics[2])).mean()

        for i in range(1, 6):
            dataframe[f'lag{i}'] = dataframe['close'].shift(i)
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.h2)
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=self.h2)

        if self.should_retrain(dataframe) and (self.training_thread is None or not self.training_thread.is_alive()):
            logger.info(f"Training required for {pair} at {datetime.now()}. Starting background training.")
            self.execute_training(dataframe.copy())

        if self.predictor is not None and self.predictor.model_names():
            median_forecast, upper_bound, lower_bound = self.generate_forecast(dataframe)
            if median_forecast is not None:
                for i in range(min(self.prediction_length, len(dataframe))):
                    dataframe.loc[dataframe.index[-1 - i], 'forecast_median'] = median_forecast[-1 - i]
                    dataframe.loc[dataframe.index[-1 - i], 'forecast_upper'] = upper_bound[-1 - i]
                    dataframe.loc[dataframe.index[-1 - i], 'forecast_lower'] = lower_bound[-1 - i]
        return dataframe

    def generate_forecast(self, dataframe: pd.DataFrame):
        if self.predictor is None or dataframe.empty:
            return None, None, None
        start_time = time.time()
        try:
            mean_close = dataframe['close'].mean()
            std_close = dataframe['close'].std()
            if std_close == 0:
                return None, None, None
            ts_data = self.prepare_ts_data(dataframe)
            if ts_data is None:
                return None, None, None
            model_to_use = self.predictor.model_best if self.predictor.model_best else self.predictor.model_names()[0]
            forecast = self.predictor.predict(ts_data, model=model_to_use)
            if isinstance(forecast.columns, pd.MultiIndex):
                median_forecast = forecast.loc[:, ('crypto_pair', '0.5')].values[-self.prediction_length:]
                upper_bound = forecast.loc[:, ('crypto_pair', '0.9')].values[-self.prediction_length:]
                lower_bound = forecast.loc[:, ('crypto_pair', '0.1')].values[-self.prediction_length:]
            else:
                median_forecast = forecast['0.5'].values[-self.prediction_length:]
                upper_bound = forecast['0.9'].values[-self.prediction_length:]
                lower_bound = forecast['0.1'].values[-self.prediction_length:]
            median_forecast = median_forecast * std_close + mean_close
            upper_bound = upper_bound * std_close + mean_close
            lower_bound = lower_bound * std_close + mean_close
            logger.info(f"Forecast generated in {time.time() - start_time:.2f} seconds: Median={median_forecast[-1]:.2f}")
            return median_forecast, upper_bound, lower_bound
        except Exception as e:
            logger.error(f"Forecasting failed: {e}")
            return None, None, None

    def should_retrain(self, dataframe: pd.DataFrame) -> bool:
        if self.predictor is None or self.last_train_time is None:
            return True
        current_time = datetime.now()
        time_elapsed = (current_time - self.last_train_time).total_seconds()
        return len(dataframe) >= self.min_train_size and time_elapsed >= self.train_interval

    def execute_training(self, dataframe: pd.DataFrame):
        self.training_thread = threading.Thread(target=self._background_train, args=(dataframe,))
        self.training_thread.start()
        logger.info("Background training thread started.")

    def _background_train(self, dataframe: pd.DataFrame):
        if dataframe.empty:
            logger.warning("Empty dataframe provided for training.")
            return

        train_data = self.prepare_ts_data(dataframe)
        if train_data is None:
            logger.warning("Prepared training data is None. Skipping training.")
            return

        try:
            # Create temporary directory for training
            os.makedirs(self.model_path_temp, exist_ok=True)

            predictor = TimeSeriesPredictor(
                path=self.model_path_temp,
                prediction_length=self.prediction_length,
                eval_metric="MAPE",
                target="close"  # Specify the target column
            )

            predictor.fit(
                train_data=train_data,
                presets="best_quality",  # Use the best_quality preset for optimal performance
                time_limit=1200,  # Increase time limit for best_quality training
            )
            last_train_time = datetime.now()
            predictor.save()
            with open(os.path.join(self.model_path_temp, "last_train_time.pkl"), "wb") as f:
                pickle.dump(last_train_time, f)
            if os.path.exists(self.model_path):
                shutil.rmtree(self.model_path)
            os.rename(self.model_path_temp, self.model_path)
            logger.info(f"TS model trained and saved to {self.model_path}")
            self.log_feature_importance(predictor)

        except Exception as e:
            logger.error(f"Background training failed: {e}")
            if os.path.exists(self.model_path_temp):
                shutil.rmtree(self.model_path_temp, ignore_errors=True)

    def log_feature_importance(self, predictor):
        try:
            feature_importance = predictor.feature_importance()
            if feature_importance is not None:
                logger.info("Feature Importance:")
                for feature, importance in feature_importance.items():
                    logger.info(f"{feature}: {importance}")
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata['pair']
        if self.predictor is None or dataframe.empty:
            return dataframe
        # Check for open trades
        open_trades = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True)]).all()
        if open_trades:
            return dataframe
        # Generate forecast
        median_forecast, upper_bound, lower_bound = self.generate_forecast(dataframe)
        if median_forecast is None:
            return dataframe
        last_close = dataframe['close'].iloc[-1]
        median_val = np.ravel(median_forecast)[-1]
        upper_val = np.ravel(upper_bound)[-1]
        lower_val = np.ravel(lower_bound)[-1]
        forecast_uncertainty = (upper_val - lower_val) / last_close
        if forecast_uncertainty > self.forecast_uncertainty_threshold:
            return dataframe
        # Calculate potential profit/loss for long and short positions
        potential_profit_long = (median_val - last_close) / last_close
        potential_profit_short = (last_close - median_val) / last_close
        # Entry conditions
        if potential_profit_long > self.min_profit_threshold and median_val > last_close:
            dataframe.loc[dataframe.index[-1], 'enter_long'] = 1
            logger.info(f"Entry long signal for {pair}: Median {median_val:.2f} > Close {last_close:.2f}")
        elif potential_profit_short > self.min_profit_threshold and median_val < last_close:
            dataframe.loc[dataframe.index[-1], 'enter_short'] = 1
            logger.info(f"Entry short signal for {pair}: Median {median_val:.2f} < Close {last_close:.2f}")
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata['pair']
        if self.predictor is None or dataframe.empty:
            logger.info(f"No predictor or empty dataframe for {pair}. Skipping exit logic.")
            return dataframe

        # Check for open trades
        open_trades = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True)]).all()
        if not open_trades:
            self.exit_order_generated[pair] = False
            self.last_exit_price[pair] = None
            self.last_exit_signal_time[pair] = 0
            return dataframe

        # Cooldown check
        current_time = time.time()
        last_signal_time = self.last_exit_signal_time.get(pair, 0)
        if current_time - last_signal_time < self.min_exit_signal_interval:
            logger.info(f"Exit signal cooldown active for {pair}. Last signal at {datetime.fromtimestamp(last_signal_time)}.")
            return dataframe

        # Generate forecast
        median_forecast, _, _ = self.generate_forecast(dataframe)
        if median_forecast is None:
            return dataframe

        last_close = dataframe['close'].iloc[-1]
        if pd.isna(last_close) or last_close <= 0:
            return dataframe

        median_val = np.ravel(median_forecast)[-1]

        # Price change check to avoid redundant signals
        if self.last_exit_price.get(pair) is not None:
            price_change = abs(median_val - self.last_exit_price[pair]) / last_close
            if price_change < 0.015:
                logger.info(f"Exit price change for {pair} too small ({price_change:.2%} < 1.5%). Skipping.")
                return dataframe

        # Determine trade direction and apply exit logic
        trade = open_trades[0]  # Assuming one open trade per pair
        if trade.is_short:
            # Exit short if projected rise exceeds threshold
            projected_rise = (median_val - last_close) / last_close
            if projected_rise > self.min_rise_threshold:
                dataframe.loc[dataframe.index[-1], 'exit_short'] = 1
                self.exit_order_generated[pair] = True
                self.last_exit_price[pair] = median_val
                self.last_exit_signal_time[pair] = current_time
                logger.info(f"Exit short signal for {pair}: Projected rise {projected_rise:.2%} > Threshold {self.min_rise_threshold:.2%}")
            else:
                logger.info(f"No exit short for {pair}: Projected rise {projected_rise:.2%} <= Threshold {self.min_rise_threshold:.2%}")
        else:
            # Exit long if projected drop exceeds threshold
            drop_percentage = (last_close - median_val) / last_close
            if drop_percentage < self.min_drop_threshold:
                dataframe.loc[dataframe.index[-1], 'exit_long'] = 1
                self.exit_order_generated[pair] = True
                self.last_exit_price[pair] = median_val
                self.last_exit_signal_time[pair] = current_time
                logger.info(f"Exit long signal for {pair}: Projected drop {drop_percentage:.2%} > Threshold {self.min_drop_threshold:.2%}")
            else:
                logger.info(f"No exit long for {pair}: Projected drop {drop_percentage:.2%} <= Threshold {self.min_drop_threshold:.2%}")

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        if not self.use_dynamic_stoploss:
            return self.stoploss
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss
        last_candle = dataframe.iloc[-1]
        atr = last_candle['atr']
        return -abs(atr * self.stoploss_atr_multiplier / current_rate)

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return self.populate_entry_trend(dataframe, metadata)

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return self.populate_exit_trend(dataframe, metadata)

def perform_fft(price_data, window_size=None):
    if window_size is not None:
        price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
    normalized_data = (price_data - np.mean(price_data)) / np.std(price_data)
    n = len(normalized_data)
    fft_data = np.fft.fft(normalized_data)
    freq = np.fft.fftfreq(n)
    power = np.abs(fft_data) ** 2
    power[np.isinf(power)] = 0
    return freq, power