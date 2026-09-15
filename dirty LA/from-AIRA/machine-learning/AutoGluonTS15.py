import os
import time
import numpy as np
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
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
from scipy.signal import savgol_filter

warnings.simplefilter(action="ignore", category=PerformanceWarning)

logger = logging.getLogger(__name__)

def version(self) -> str:
    return "1.04"

def timeframe_to_minutes(timeframe: str) -> int:
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == 'm':
        return value
    elif unit == 'h':
        return value * 60
    elif unit == 'd':
        return value * 60 * 24
    elif unit == 'w':
        return value * 60 * 24 * 7
    else:
        raise ValueError(f"Unsupported timeframe unit: {timeframe}")

class AutoGluonTS15(IStrategy):
    INTERFACE_VERSION = 3
    
    timeframe = '15m'
    startup_candle_count = 600
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = True
    sell_profit_only = True

    prediction_length = 4
    train_interval = 24 * 60 * 60
    min_train_size = 500
    model_path = "user_data/AutoGluonTS15"
    model_path_temp = model_path + "_temp"
    forecast_history_file = "user_data/forecast_history_15m.pkl"

    use_dynamic_stoploss = True
    stoploss_atr_multiplier = 2.0
    process_only_new_candles = True
    forecast_uncertainty_threshold = 0.05
    min_profit_threshold = 0.01
    min_exit_signal_interval = 75
    min_drop_threshold = -0.02

    min_atr_multiplier = DecimalParameter(0.1, 5.0, default=1.0, space='buy', optimize=True, load=True)
    max_uncertainty_threshold = DecimalParameter(0.01, 0.1, default=0.05, space='buy', optimize=True, load=True)
    max_uncertainty_for_holding = DecimalParameter(0.05, 0.2, default=0.1, space='sell', optimize=True, load=True)
    stoploss_uncertainty_multiplier = DecimalParameter(1.0, 5.0, default=2.0, space='sell', optimize=True, load=True)
    use_uncertainty_based_stoploss = BooleanParameter(default=True, space='sell', optimize=True, load=True)
    take_profit_uncertainty_factor = DecimalParameter(0.0, 2.0, default=0.5, space='sell', optimize=True, load=True)
    fft_days = IntParameter(10, 60, default=30, space='buy', optimize=True, load=True)
    min_cycle_days = DecimalParameter(0.5, 5, default=1, space='buy', optimize=True, load=True)
    max_cycle_days = DecimalParameter(5, 30, default=4, space='buy', optimize=True, load=True)

    tema_period = IntParameter(5, 50, default=10, space='buy', optimize=True, load=True)
    macd_fast = IntParameter(5, 20, default=12, space='buy', optimize=True, load=True)
    macd_slow = IntParameter(20, 50, default=26, space='buy', optimize=True, load=True)
    macd_signal = IntParameter(5, 15, default=9, space='buy', optimize=True, load=True)
    roc_period = IntParameter(5, 20, default=10, space='buy', optimize=True, load=True)
    adosc_fast = IntParameter(3, 10, default=3, space='buy', optimize=True, load=True)
    adosc_slow = IntParameter(10, 20, default=10, space='buy', optimize=True, load=True)
    ria_efficiency_threshold = DecimalParameter(0.3, 0.7, default=0.5, space='buy', optimize=True, load=True)  # New RIA parameter

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 5},
            {"method": "StoplossGuard", "lookback_period_candles": 96, "trade_limit": 4, "stop_duration_candles": 8, "only_per_pair": False},
            {"method": "LowProfitPairs", "lookback_period_candles": 24, "trade_limit": 2, "stop_duration_candles": 240, "required_profit": 0.02},
            {"method": "LowProfitPairs", "lookback_period_candles": 96, "trade_limit": 4, "stop_duration_candles": 8, "required_profit": 0.01}
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
        self.forecast_history = {}
        self.load_forecast_history()
        self.load_predictor()

    def plot_config(self):
        return {
            "main_plot": {
                "forecast_median_extended": {"color": "green"},
                "forecast_upper_extended": {"color": "lightgreen"},
                "forecast_lower_extended": {"color": "lightgreen"},
                "tema": {"color": "blue"},
                "sar": {"color": "orange"},
                "medprice": {"color": "purple"}
            },
            "subplots": {
                "MACD": {
                    "macd": {"color": "blue"},
                    "macd_signal": {"color": "orange"},
                    "macd_hist": {"color": "gray", "type": "bar"}
                },
                "BOP": {"bop": {"color": "green"}},
                "ROCR100": {"rocr100": {"color": "red"}},
                "ADOSC": {"adosc": {"color": "purple"}},
                "TRANGE": {"trange": {"color": "brown"}},
                "RIA": {"ria_efficiency": {"color": "cyan"}}  # Added RIA to subplots
            }
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
                logger.error(f"Error loading model: {e}. Will train a new one when needed.")
                self.predictor = None
                self.last_train_time = None
        else:
            logger.info("No existing model found. Will train when conditions are met.")
            self.predictor = None
            self.last_train_time = None

    def load_forecast_history(self):
        if os.path.exists(self.forecast_history_file):
            try:
                with open(self.forecast_history_file, 'rb') as f:
                    self.forecast_history = pickle.load(f)
                logger.info(f"Loaded forecast history from {self.forecast_history_file}")
            except Exception as e:
                logger.error(f"Failed to load forecast history: {e}. Starting fresh.")
                self.forecast_history = {}
    
    def save_forecast_history(self):
        try:
            with open(self.forecast_history_file, 'wb') as f:
                pickle.dump(self.forecast_history, f)
            logger.info(f"Saved forecast history to {self.forecast_history_file}")
        except Exception as e:
            logger.error(f"Failed to save forecast history: {e}")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'close' in df.columns:
            df['close'] = df['close'].rolling(window=3, min_periods=1).median()
        return df

    def calculate_ria_efficiency(self, dataframe: pd.DataFrame, lookback: int = 20) -> pd.Series:
        price_change = dataframe['close'].diff().abs()
        net_change = dataframe['close'].diff(lookback).abs()
        total_range = price_change.rolling(window=lookback).sum()
        efficiency = net_change / total_range
        efficiency = efficiency.fillna(0).replace([np.inf, -np.inf], 0)
        return efficiency

    def prepare_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df = dataframe.copy()
        df['sar'] = ta.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
        df['tema'] = ta.TEMA(df['close'], timeperiod=self.tema_period.value)
        df['bop'] = ta.BOP(df['open'], df['high'], df['low'], df['close'])
        macd = ta.MACD(df['close'], fastperiod=self.macd_fast.value, 
                      slowperiod=self.macd_slow.value, signalperiod=self.macd_signal.value)
        df['macd'] = macd[0]
        df['macd_signal'] = macd[1]
        df['macd_hist'] = macd[2]
        df['rocr100'] = ta.ROCR100(df['close'], timeperiod=self.roc_period.value)
        df['adosc'] = ta.ADOSC(df['high'], df['low'], df['close'], df['volume'],
                             fastperiod=self.adosc_fast.value, slowperiod=self.adosc_slow.value)
        df['trange'] = ta.TRANGE(df['high'], df['low'], df['close'])
        df['medprice'] = ta.MEDPRICE(df['high'], df['low'])
        df['ria_efficiency'] = self.calculate_ria_efficiency(df, lookback=20)  # Added RIA feature
        return df

    def prepare_ts_data(self, dataframe: pd.DataFrame):
        if dataframe.empty or dataframe[['close']].isnull().all().all():
            logger.warning("Dataframe is empty or contains all NaN values for 'close'. Skipping preparation.")
            return None

        df = dataframe.copy().reset_index(drop=True)
        df = self.clean_data(df)
        df = self.prepare_features(df)
        
        if 'date' in df.columns:
            ts = pd.to_datetime(df['date'], errors='coerce')
        else:
            ts = pd.to_datetime(df.index, errors='coerce')
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert('UTC').dt.tz_localize(None)
        else:
            ts = ts.dt.tz_localize(None)
        df['timestamp'] = ts
        
        time_diffs = df['timestamp'].diff().dropna()
        expected_diff = pd.Timedelta(minutes=timeframe_to_minutes(self.timeframe))
        if not time_diffs.empty and not (time_diffs == expected_diff).all():
            logger.warning("Timestamp discontinuities detected. Attempting to reindex.")
            full_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='15T')
            df = df.set_index('timestamp').reindex(full_range).reset_index().rename(columns={'index': 'timestamp'})
        
        df['item_id'] = 'crypto_pair'
        
        initial_len = len(df)
        df = df.dropna(subset=['close'])
        logger.info(f"Dropped {initial_len - len(df)} rows with NaN in 'close'. Remaining rows: {len(df)}")
        
        feature_cols = ['close', 'sar', 'tema', 'bop', 'macd', 'macd_signal', 'macd_hist', 'rocr100', 'adosc', 'trange', 'medprice', 'ria_efficiency']
        available_cols = [col for col in feature_cols if col in df.columns]
        
        df[available_cols] = df[available_cols].ffill().bfill()
        
        if df[available_cols].isnull().any().any():
            nan_cols = df[available_cols].columns[df[available_cols].isnull().any()].tolist()
            logger.warning(f"NaNs persist in features after fill: {nan_cols}. Filling with 0.")
            df[nan_cols] = df[nan_cols].fillna(0)
        
        cols_to_normalize = ['close', 'sar', 'tema', 'macd', 'macd_signal', 'macd_hist', 'adosc', 'trange', 'medprice', 'ria_efficiency']
        for col in cols_to_normalize:
            if col in df.columns:
                std = df[col].std()
                if pd.isna(std) or std < 1e-10:
                    logger.warning(f"Column {col} has near-zero standard deviation ({std}). Setting to constant.")
                    df[col] = 0
                else:
                    df[col] = (df[col] - df[col].mean()) / std
        
        min_length = self.startup_candle_count + (self.prediction_length * 5)
        if len(df) < min_length:
            logger.warning(f"Data length {len(df)} is less than minimum required {min_length}. Skipping preparation.")
            return None
        
        ts_data = TimeSeriesDataFrame.from_data_frame(
            df[['item_id', 'timestamp'] + available_cols],
            id_column="item_id",
            timestamp_column="timestamp"
        )
        
        if ts_data.isnull().any().any():
            logger.error("NaNs still present in TimeSeriesDataFrame. Skipping preparation.")
            return None
        
        logger.info(f"Prepared TimeSeriesDataFrame with {len(ts_data)} rows and columns: {ts_data.columns.tolist()}")
        return ts_data

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
                logger.error(f"Failed to check last train time: {e}")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        self._check_and_reload_predictor()
        dataframe = self.prepare_features(dataframe)
        
        dataframe['forecast_median'] = np.nan
        dataframe['forecast_upper'] = np.nan
        dataframe['forecast_lower'] = np.nan
        dataframe['forecast_median_extended'] = np.nan
        dataframe['forecast_upper_extended'] = np.nan
        dataframe['forecast_lower_extended'] = np.nan
        
        pair = metadata['pair']

        if self.should_retrain(dataframe) and (self.training_thread is None or not self.training_thread.is_alive()):
            logger.info(f"Training required for {pair} at {datetime.now()}. Starting background training.")
            self.execute_training(dataframe.copy())

        if self.predictor is not None and self.predictor.model_names():
            median_forecast, upper_bound, lower_bound = self.generate_forecast(dataframe, metadata)
            if median_forecast is not None:
                try:
                    forecast_len = min(self.prediction_length, len(median_forecast))
                    for i in range(forecast_len):
                        idx = dataframe.index[-1 - i]
                        dataframe.loc[idx, 'forecast_median'] = median_forecast[-1 - i]
                        dataframe.loc[idx, 'forecast_upper'] = upper_bound[-1 - i]
                        dataframe.loc[idx, 'forecast_lower'] = lower_bound[-1 - i]
                    
                    last_date = pd.to_datetime(dataframe['date'].iloc[-1] if 'date' in dataframe.columns else dataframe.index[-1])
                    tf_delta = timedelta(minutes=timeframe_to_minutes(self.timeframe))
                    extended_dates = [last_date + tf_delta * i for i in range(self.prediction_length)]
                    
                    if pair not in self.forecast_history:
                        self.forecast_history[pair] = {}
                    for i, date in enumerate(extended_dates):
                        if i < len(median_forecast):
                            self.forecast_history[pair][date] = {
                                'median': float(median_forecast[i]),
                                'upper': float(upper_bound[i]),
                                'lower': float(lower_bound[i])
                            }
                    
                    for date in list(self.forecast_history[pair].keys()):
                        if date in (dataframe['date'].values if 'date' in dataframe.columns else dataframe.index):
                            idx = dataframe[dataframe['date'] == date].index[0] if 'date' in dataframe.columns else dataframe.index.get_loc(date)
                            forecast = self.forecast_history[pair][date]
                            dataframe.loc[idx, 'forecast_median_extended'] = forecast['median']
                            dataframe.loc[idx, 'forecast_upper_extended'] = forecast['upper']
                            dataframe.loc[idx, 'forecast_lower_extended'] = forecast['lower']

                    self.save_forecast_history()
                except Exception as e:
                    logger.error(f"Error processing forecast for {pair}: {e}")

        return dataframe

    def generate_forecast(self, dataframe: pd.DataFrame, metadata: dict):
        if self.predictor is None or dataframe.empty:
            logger.warning("No predictor or empty dataframe. Skipping forecast generation.")
            return None, None, None
        start_time = time.time()
        try:
            mean_close = dataframe['close'].mean()
            std_close = dataframe['close'].std()
            if pd.isna(std_close) or std_close < 1e-10:
                logger.warning("Close price standard deviation is zero or NaN")
                return None, None, None
            ts_data = self.prepare_ts_data(dataframe)
            if ts_data is None:
                logger.warning("TimeSeriesDataFrame preparation failed.")
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
            logger.info(f"Pair {metadata['pair']}: Forecast generated in {time.time() - start_time:.2f} seconds: Median={median_forecast[-1]:.2f}")
            return median_forecast, upper_bound, lower_bound
        except Exception as e:
            logger.error(f"Forecasting failed for {metadata['pair']}: {e}")
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
            if os.path.exists(self.model_path_temp):
                shutil.rmtree(self.model_path_temp, ignore_errors=True)

            os.makedirs(os.path.join(self.model_path_temp, "logs"), exist_ok=True)
            predictor = TimeSeriesPredictor(
                prediction_length=self.prediction_length,
                target="close",
                eval_metric="WQL",
                path=self.model_path_temp
            )
            predictor.fit(
                train_data=train_data,
                num_val_windows=5,
                presets="chronos2",
                time_limit=600,  # time limit in seconds
                enable_ensemble=True
            )
            last_train_time = datetime.now()

            self.log_feature_importance(predictor)
            predictor.save()

            with open(os.path.join(self.model_path_temp, "last_train_time.pkl"), "wb") as f:
                pickle.dump(last_train_time, f)

            max_attempts = 5
            attempt = 0
            base_delay = 1

            while attempt < max_attempts:
                try:
                    if os.path.exists(self.model_path):
                        shutil.rmtree(self.model_path, ignore_errors=True)
                    os.rename(self.model_path_temp, self.model_path)
                    logger.info(f"Model successfully moved from {self.model_path_temp} to {self.model_path} using rename")
                    break
                except (OSError, PermissionError) as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_attempts} - Rename failed: {e}. Falling back to copy.")
                    try:
                        if os.path.exists(self.model_path):
                            shutil.rmtree(self.model_path, ignore_errors=True)
                        shutil.copytree(self.model_path_temp, self.model_path)
                        shutil.rmtree(self.model_path_temp, ignore_errors=True)
                        logger.info(f"Model successfully copied from {self.model_path_temp} to {self.model_path}")
                        break
                    except Exception as copy_error:
                        attempt += 1
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Copy attempt {attempt}/{max_attempts} failed: {copy_error}. Retrying in {delay} seconds...")
                        time.sleep(delay)
            else:
                raise RuntimeError(f"Failed to move/copy model from {self.model_path_temp} to {self.model_path} after {max_attempts} attempts")

            self.predictor = predictor
            self.last_train_time = last_train_time
            logger.info(f"TS model trained and saved to {self.model_path}")

        except Exception as e:
            logger.error(f"Background training failed: {e}")
            if os.path.exists(self.model_path_temp):
                shutil.rmtree(self.model_path_temp, ignore_errors=True)

    def log_feature_importance(self, predictor):
        log_file = os.path.join(self.model_path_temp, "logs", "feature_importance.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        predictor_logger = logging.getLogger('FeatureImportanceLogger')
        if not predictor_logger.handlers:
            predictor_logger.addHandler(file_handler)
        predictor_logger.setLevel(logging.INFO)

        try:
            logger.info("Starting feature importance calculation...")
            predictor_logger.info("Starting feature importance calculation...")
            
            subsample_size = 1
            logger.info(f"Using subsample_size: {subsample_size} (single time series detected)")
            predictor_logger.info(f"Using subsample_size: {subsample_size} (single time series detected)")

            feature_importance = predictor.feature_importance(subsample_size=subsample_size)
            
            if feature_importance is not None and not feature_importance.empty:
                logger.info("=== Feature Importance Analysis ===")
                predictor_logger.info("=== Feature Importance Analysis ===")
                logger.info(f"Number of features evaluated: {len(feature_importance)}")
                predictor_logger.info(f"Number of features evaluated: {len(feature_importance)}")
                
                feature_importance_sorted = feature_importance.sort_values('importance', ascending=False)
                
                if feature_importance['importance'].max() == 0:
                    logger.warning("All feature importance scores are 0. This may indicate:")
                    logger.warning("- Insufficient validation data (need at least prediction_length * num_val_windows points)")
                    logger.warning("- Model type (e.g., ETS) may not support feature importance for covariates")
                    logger.warning("- Consider increasing num_val_windows or using a tabular model")
                    predictor_logger.warning("All feature importance scores are 0. This may indicate:")
                    predictor_logger.warning("- Insufficient validation data (need at least prediction_length * num_val_windows points)")
                    predictor_logger.warning("- Model type (e.g., ETS) may not support feature importance for covariates")
                    predictor_logger.warning("- Consider increasing num_val_windows or using a tabular model")
                
                logger.info("Detailed Feature Importance Scores:")
                predictor_logger.info("Detailed Feature Importance Scores:")
                feature_table = []
                for feature, row in feature_importance_sorted.iterrows():
                    feature_line = (f"Feature: {feature:<15} | Importance: {row['importance']:.6f} "
                                  f"| Std Dev: {row.get('std', 0):.6f} "
                                  f"| P-value: {row.get('p_value', 'N/A')}")
                    feature_table.append(feature_line)
                logger.info("\n".join(feature_table))
                predictor_logger.info("\n".join(feature_table))
                
                logger.info("Summary Statistics:")
                predictor_logger.info("Summary Statistics:")
                logger.info(f"Mean Importance: {feature_importance['importance'].mean():.6f}")
                logger.info(f"Max Importance: {feature_importance['importance'].max():.6f}")
                logger.info(f"Min Importance: {feature_importance['importance'].min():.6f}")
                logger.info(f"Std Dev of Importance: {feature_importance['importance'].std():.6f}")
                predictor_logger.info(f"Mean Importance: {feature_importance['importance'].mean():.6f}")
                predictor_logger.info(f"Max Importance: {feature_importance['importance'].max():.6f}")
                predictor_logger.info(f"Min Importance: {feature_importance['importance'].min():.6f}")
                predictor_logger.info(f"Std Dev of Importance: {feature_importance['importance'].std():.6f}")
                
                logger.info("Top 3 Most Important Features:")
                predictor_logger.info("Top 3 Most Important Features:")
                top_3 = feature_importance_sorted.head(3)
                for feature, row in top_3.iterrows():
                    logger.info(f"  {feature}: {row['importance']:.6f}")
                    predictor_logger.info(f"  {feature}: {row['importance']:.6f}")
                
                best_model = predictor.model_best or predictor.model_names()[0]
                logger.info(f"Best model used: {best_model}")
                predictor_logger.info(f"Best model used: {best_model}")
                if hasattr(predictor, 'leaderboard'):
                    leaderboard = predictor.leaderboard()
                    if not leaderboard.empty:
                        logger.info("Model Leaderboard:")
                        leaderboard_str = leaderboard[['model', 'score_val']].to_string(index=False)
                        logger.info(f"\n{leaderboard_str}")
                        predictor_logger.info("Model Leaderboard:")
                        predictor_logger.info(f"\n{leaderboard_str}")
                
                logger.info("Feature importance calculation completed successfully")
                predictor_logger.info("Feature importance calculation completed successfully")
            else:
                logger.warning("No feature importance data available from predictor")
                predictor_logger.warning("No feature importance data available from predictor")
                
        except FileNotFoundError as e:
            logger.error(f"Cannot calculate feature importance: Training data not found - {e}")
            predictor_logger.error(f"Cannot calculate feature importance: Training data not found - {e}")
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            logger.debug("Stack trace:", exc_info=True)
            predictor_logger.error(f"Error calculating feature importance: {e}")
        finally:
            predictor_logger.removeHandler(file_handler)
            file_handler.close()

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is not None and not dataframe.empty:
            current_candle = dataframe.iloc[-1]
            Trade.custom_info[pair] = {
                'entry_forecast_median': float(current_candle['forecast_median']) if not pd.isna(current_candle['forecast_median']) else None,
                'entry_forecast_upper': float(current_candle['forecast_upper']) if not pd.isna(current_candle['forecast_upper']) else None,
                'entry_forecast_lower': float(current_candle['forecast_lower']) if not pd.isna(current_candle['forecast_lower']) else None,
                'entry_ria_efficiency': float(current_candle['ria_efficiency']) if not pd.isna(current_candle['ria_efficiency']) else None  # Added RIA
            }
        return True

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_tag'] = None
        
        required_cols = {'forecast_median', 'forecast_upper', 'forecast_lower', 'tema', 'macd', 'macd_signal', 'rocr100', 'ria_efficiency'}
        if not required_cols.issubset(dataframe.columns):
            logger.warning(f"Missing required columns for entry in {metadata['pair']}: {required_cols - set(dataframe.columns)}")
            return dataframe
        
        current = dataframe.iloc[-1]
        
        if any(pd.isna(current[col]) for col in required_cols):
            logger.warning(f"NaN values in critical columns for {metadata['pair']}")
            return dataframe
        
        reference_price = current['close']
        forecast_return = (current['forecast_median'] - reference_price) / reference_price if reference_price != 0 else 0
        uncertainty = (current['forecast_upper'] - current['forecast_lower']) / reference_price if reference_price != 0 else float('inf')
        
        dynamic_min_profit = self.min_profit_threshold
        
        price_above_tema = current['close'] > current['tema']
        macd_bullish = current['macd'] > current['macd_signal'] and current['macd_hist'] > 0
        roc_positive = current['rocr100'] > 100
        ria_efficient = current['ria_efficiency'] > self.ria_efficiency_threshold.value
        
        if all([
            forecast_return > dynamic_min_profit,
            uncertainty < self.max_uncertainty_threshold.value,
            price_above_tema,
            macd_bullish,
            roc_positive,
            ria_efficient
        ]):
            dataframe.loc[dataframe.index[-1], 'enter_long'] = 1
            dataframe.loc[dataframe.index[-1], 'enter_tag'] = 'forecast_tema_macd_roc_ria'
            logger.info(
                f"Entry signal for {metadata['pair']}: "
                f"Forecast: {current['forecast_median']:.2f} > Ref: {reference_price:.2f} "
                f"(Threshold: {dynamic_min_profit:.2%}, Uncertainty: {uncertainty:.2%}) "
                f"TEMA: {price_above_tema}, MACD: {macd_bullish}, ROCR100: {roc_positive}, RIA: {ria_efficient} ({current['ria_efficiency']:.2f})"
            )

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_tag'] = None
        
        required_cols = {'forecast_median', 'forecast_upper', 'forecast_lower', 'tema', 'macd', 'macd_signal', 'bop', 'ria_efficiency'}
        if not required_cols.issubset(dataframe.columns):
            logger.warning(f"Missing required columns for exit in {metadata['pair']}: {required_cols - set(dataframe.columns)}")
            return dataframe
        
        current = dataframe.iloc[-1]
        if any(pd.isna(current[col]) for col in required_cols):
            logger.warning(f"NaN values in critical columns for {metadata['pair']}")
            return dataframe
        
        reference_price = current['close']
        forecast_reversal = current['forecast_median'] < reference_price
        stoploss_trigger = current['close'] < current['forecast_lower']
        uncertainty = (current['forecast_upper'] - current['forecast_lower']) / reference_price if reference_price != 0 else float('inf')
        
        price_below_tema = current['close'] < current['tema']
        macd_bearish = current['macd'] < current['macd_signal'] and current['macd_hist'] < 0
        bop_negative = current['bop'] < 0
        ria_inefficient = current['ria_efficiency'] < (self.ria_efficiency_threshold.value * 0.6)  # Lower threshold for exit (e.g., 60% of entry)
        
        if any([
            forecast_reversal,
            stoploss_trigger,
            uncertainty > self.max_uncertainty_for_holding.value,
            price_below_tema and macd_bearish,
            bop_negative,
            ria_inefficient
        ]):
            dataframe.loc[dataframe.index[-1], 'exit_long'] = 1
            if forecast_reversal:
                tag = 'forecast_reversal'
            elif stoploss_trigger:
                tag = 'stoploss'
            elif uncertainty > self.max_uncertainty_for_holding.value:
                tag = 'high_uncertainty'
            elif price_below_tema and macd_bearish:
                tag = 'tema_macd_exit'
            elif bop_negative:
                tag = 'bop_exit'
            else:
                tag = 'ria_inefficient'
            dataframe.loc[dataframe.index[-1], 'exit_tag'] = tag
            logger.info(
                f"Exit signal for {metadata['pair']}: "
                f"Reversal: {forecast_reversal}, "
                f"Stoploss: {current['close']:.2f} < {current['forecast_lower']:.2f}, "
                f"Uncertainty: {uncertainty:.2%}, "
                f"TEMA: {price_below_tema}, MACD: {macd_bearish}, BOP: {bop_negative}, RIA: {ria_inefficient} ({current['ria_efficiency']:.2f}), Tag: {tag}"
            )

        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> dict:
        entry_data = trade.custom_info.get(pair, {})
        entry_forecast_median = entry_data.get('entry_forecast_median')
        entry_forecast_upper = entry_data.get('entry_forecast_upper')
        entry_forecast_lower = entry_data.get('entry_forecast_lower')
        entry_ria_efficiency = entry_data.get('entry_ria_efficiency')

        if all(x is not None for x in [entry_forecast_median, entry_forecast_upper, entry_forecast_lower]):
            uncertainty_at_entry = entry_forecast_upper - entry_forecast_lower
            take_profit_level = entry_forecast_median + self.take_profit_uncertainty_factor.value * uncertainty_at_entry
            if current_rate >= take_profit_level:
                logger.info(f"Custom exit for {pair}: Take profit triggered at {current_rate:.2f} >= {take_profit_level:.2f}")
                return {'exit_reason': 'take_profit_forecast'}

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is not None and not dataframe.empty:
            current_candle = dataframe.iloc[-1]
            if not pd.isna(current_candle['forecast_upper']) and not pd.isna(current_candle['forecast_lower']):
                uncertainty = (current_candle['forecast_upper'] - current_candle['forecast_lower']) / current_rate if current_rate != 0 else float('inf')
                if uncertainty > self.max_uncertainty_for_holding.value:
                    logger.info(f"Custom exit for {pair}: High uncertainty {uncertainty:.2%} > {self.max_uncertainty_for_holding.value:.2%}")
                    return {'exit_reason': 'high_uncertainty'}
            if not pd.isna(current_candle['ria_efficiency']) and entry_ria_efficiency is not None:
                if current_candle['ria_efficiency'] < (entry_ria_efficiency * 0.6):  # Significant drop in efficiency
                    logger.info(f"Custom exit for {pair}: RIA efficiency dropped from {entry_ria_efficiency:.2f} to {current_candle['ria_efficiency']:.2f}")
                    return {'exit_reason': 'ria_efficiency_drop'}
        return None

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            logger.warning(f"No dataframe available for stoploss calculation in {pair}. Using default stoploss.")
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        if self.use_uncertainty_based_stoploss.value:
            if not pd.isna(last_candle['forecast_upper']) and not pd.isna(last_candle['forecast_lower']):
                uncertainty = (last_candle['forecast_upper'] - last_candle['forecast_lower']) / current_rate if current_rate != 0 else float('inf')
                dynamic_stoploss = -abs(uncertainty * self.stoploss_uncertainty_multiplier.value)
                logger.debug(f"Dynamic stoploss for {pair}: {dynamic_stoploss:.4f} (Uncertainty: {uncertainty:.2%})")
                return dynamic_stoploss
        return self.stoploss

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return self.populate_entry_trend(dataframe, metadata)

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return self.populate_exit_trend(dataframe, metadata)

def perform_fft(price_data, window_size=None):
    if window_size is not None and window_size > 1 and len(price_data) >= window_size:
        price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
    mean = np.mean(price_data)
    std = np.std(price_data)
    if pd.isna(std) or std < 1e-10:
        normalized_data = np.zeros_like(price_data)
    else:
        normalized_data = (price_data - mean) / std
    n = len(normalized_data)
    fft_data = np.fft.fft(normalized_data)
    freq = np.fft.fftfreq(n)
    power = np.abs(fft_data) ** 2
    power[np.isinf(power)] = 0
    return freq, power