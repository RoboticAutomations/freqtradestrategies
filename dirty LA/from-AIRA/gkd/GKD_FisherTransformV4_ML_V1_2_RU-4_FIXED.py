"""
# Enhanced Fisher Transform Strategy with ML/RL Integration - V1.2 BALANCED
# FIXED: Balanced long/short entries, tightened stoploss, removed restrictive Goldie Locks
# 
# Changelog v1.2 (2026-02-17):
# - FIXED: Stoploss tightened from -0.524 (52%!) to -0.15 (15%)
# - FIXED: max_open_trades reduced from 20 to 5 to limit risk
# - FIXED: Balanced ML conditions - both longs and shorts use same confidence logic
# - FIXED: Short entry removed restrictive Goldie Locks zone requirement
# - FIXED: Short entry now uses fisher < -0.5 (symmetric to long) instead of < 2.89
# - FIXED: Both directions require baseline trend in their favor
# - FIXED: Long entry threshold relaxed to 1.0 (was 0.65)
#
# Changelog v1.1 (2026-02-17):
# - FIXED: fisher_trend_long was identical to fisher_trend_short
# - FIXED: Long entry used fisher_sell_threshold instead of fisher_buy_threshold
# - FIXED: Long entry momentum check was inverted
#
# Usage:
# freqtrade hyperopt --hyperopt-loss SharpeHyperOptLossDaily --strategy GKD_FisherTransformV4_ML_V1_2_Balanced \
#     --spaces buy sell roi stoploss trailing --config user_data/config_binance_futures_backtest_usdt.json \
#     --epochs 1000 --timerange 20241001-20250501 --timeframe-detail 5m --max-open-trades 3 -timeframe 1h
"""

import datetime
import logging
import math
import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import talib
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter,
                                DecimalParameter, IntParameter, IStrategy,
                                RealParameter, informative,
                                merge_informative_pair)
from pandas import DataFrame
from pandas_ta import ema
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

ASCII_ART = '''
'''
def lerp(a: float, b: float, t: float) -> float:
    """Линейная интерполяция между a и b по коэффициенту t"""
    return a + t * (b - a)



class EnhancedLogger:
    """Улучшенное логирование с эмодзи и форматированием"""
    
    @staticmethod
    def log_banner(message: str, emoji: str = "🚀"):
        """Баннер-сообщение"""
        border = "═" * (len(message) + 6)
        logger.info(f"{emoji} {border}")
        logger.info(f"{emoji} ║  {message}  ║")
        logger.info(f"{emoji} {border}")
    
    @staticmethod
    def log_section(title: str, emoji: str = "📊"):
        """Заголовок раздела"""
        logger.info(f"\n{emoji} ═══ {title} ═══")
    
    @staticmethod
    def log_subsection(title: str, emoji: str = "▶️"):
        """Подраздел"""
        logger.info(f"{emoji} {title}")
    
    @staticmethod
    def log_parameter(name: str, value: Any, emoji: str = "⚙️"):
        """Параметр с форматированием"""
        if isinstance(value, float):
            logger.info(f"  {emoji} {name}: {value:.4f}")
        else:
            logger.info(f"  {emoji} {name}: {value}")
    
    @staticmethod
    def log_performance(metric: str, value: float, emoji: str = "📈"):
        """Метрики производительности"""
        color_emoji = "🟢" if value > 0 else "🔴" if value < 0 else "🟡"
        logger.info(f"{emoji} {color_emoji} {metric}: {value:.4f}")
    
    @staticmethod
    def log_trade_action(action: str, pair: str, rate: float, emoji: str = "💰"):
        """Торговые действия"""
        logger.info(f"{emoji} {action} {pair} @ {rate:.6f}")
    
    @staticmethod
    def log_ml_status(message: str, confidence: float = None, emoji: str = "🤖"):
        """ML-статус"""
        if confidence is not None:
            conf_emoji = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🔴"
            logger.info(f"{emoji} {conf_emoji} {message} (Уверенность: {confidence:.2%})")
        else:
            logger.info(f"{emoji} {message}")
    
    @staticmethod
    def log_error(message: str, emoji: str = "❌"):
        """Сообщение об ошибке"""
        logger.error(f"{emoji} ОШИБКА: {message}")
    
    @staticmethod
    def log_warning(message: str, emoji: str = "⚠️"):
        """Предупреждение"""
        logger.warning(f"{emoji} ПРЕДУПРЕЖДЕНИЕ: {message}")
    
    @staticmethod
    def log_success(message: str, emoji: str = "✅"):
        """Успешное выполнение"""
        logger.info(f"{emoji} УСПЕХ: {message}")


class MLOptimizer:
    """ML-оптимизатор параметров стратегии с улучшенным логированием"""
    
    def __init__(self, strategy_name: str = "GKD_FisherV5"):
        self.strategy_name = strategy_name
        self.model_path = f"user_data/strategies/ml_models/{strategy_name}_model.pkl"
        self.scaler_path = f"user_data/strategies/ml_models/{strategy_name}_scaler.pkl"
        self.study_path = f"user_data/strategies/ml_models/{strategy_name}_study.pkl"
        self.model = None
        self.scaler = None
        self.study = None
        self.performance_history = []
        
        # Ensure directory exists
        os.makedirs("user_data/strategies/ml_models", exist_ok=True)
        
        # Enhanced logging for initialization
        EnhancedLogger.log_section("ИНИЦИАЛИЗАЦИЯ ML ОПТИМИЗАТОРА", "🤖")
        EnhancedLogger.log_parameter("Название стратегии", strategy_name, "🏷️")
        EnhancedLogger.log_parameter("Путь к модели", self.model_path, "📁")
        
        # Load existing models if available
        self.load_models()
    
    def load_models(self):
        """Загрузка существующих ML моделей и исследования Optuna"""
        try:
            models_loaded = 0
            
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                models_loaded += 1
                EnhancedLogger.log_success("ML модель успешно загружена", "🧠")
            else:
                EnhancedLogger.log_warning("Существующая ML модель не найдена", "🤖")
            
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                models_loaded += 1
                EnhancedLogger.log_success("Масштабировщик признаков успешно загружен", "📏")
            else:
                EnhancedLogger.log_warning("Масштабировщик не найден", "📏")
                    
            if os.path.exists(self.study_path):
                with open(self.study_path, 'rb') as f:
                    self.study = pickle.load(f)
                models_loaded += 1
                EnhancedLogger.log_success(f"Исследование Optuna загружено ({len(self.study.trials)} испытаний)", "🔬")
            else:
                EnhancedLogger.log_warning("Исследование Optuna не найдено", "🔬")
            
            if models_loaded > 0:
                EnhancedLogger.log_success(f"Загружено {models_loaded}/3 компонентов ML", "✨")
            
        except Exception as e:
            EnhancedLogger.log_error(f"Ошибка загрузки моделей: {e}")
    
    def save_models(self):
        """Сохранение ML моделей и исследования Optuna"""
        try:
            saved_models = 0
            
            if self.model:
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                saved_models += 1
                EnhancedLogger.log_success("ML модель сохранена", "💾")
            
            if self.scaler:
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                saved_models += 1
                EnhancedLogger.log_success("Масштабировщик сохранён", "💾")
                    
            if self.study:
                with open(self.study_path, 'wb') as f:
                    pickle.dump(self.study, f)
                saved_models += 1
                EnhancedLogger.log_success("Исследование Optuna сохранено", "💾")
            
            if saved_models > 0:
                EnhancedLogger.log_success(f"Сохранено {saved_models} компонентов ML", "🎯")
                
        except Exception as e:
            EnhancedLogger.log_error(f"Ошибка сохранения моделей: {e}")
    
    def create_features(self, dataframe: DataFrame) -> np.ndarray:
        """Создание признаков для ML модели — ровно 12 штук"""
        features = []
        
        try:
            EnhancedLogger.log_subsection("Создание ML признаков", "🔧")
            
            # Check if required columns exist, if not calculate them
            if 'atr' not in dataframe.columns or dataframe['atr'].isna().all():
                EnhancedLogger.log_warning("Столбец ATR отсутствует, вычисляем...", "📊")
                try:
                    import talib
                    dataframe['atr'] = talib.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
                except:
                    # Fallback ATR calculation
                    high_low = dataframe['high'] - dataframe['low']
                    high_close = abs(dataframe['high'] - dataframe['close'].shift())
                    low_close = abs(dataframe['low'] - dataframe['close'].shift())
                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    dataframe['atr'] = true_range.rolling(window=14).mean()
                EnhancedLogger.log_success("ATR успешно вычислен", "📊")
            
            # Check if fisher exists, if not create a simple version
            if 'fisher' not in dataframe.columns or dataframe['fisher'].isna().all():
                EnhancedLogger.log_warning("Fisher Transform отсутствует, вычисляем...", "🎯")
                # Simple Fisher Transform calculation
                median_price = (dataframe['high'] + dataframe['low']) / 2
                period = 14
                fisher = pd.Series(0.0, index=dataframe.index)
                
                for i in range(period, len(dataframe)):
                    price_window = median_price.iloc[i-period:i]
                    price_min = price_window.min()
                    price_max = price_window.max()
                    if price_max != price_min:
                        norm = (median_price.iloc[i] - price_min) / (price_max - price_min)
                        norm = 2 * norm - 1
                        norm = max(min(norm, 0.999), -0.999)
                        fisher.iloc[i] = 0.5 * np.log((1 + norm) / (1 - norm))
                
                dataframe['fisher'] = fisher
                EnhancedLogger.log_success("Fisher Transform вычислен", "🎯")
            
            # Check if baseline_diff exists
            if 'baseline_diff' not in dataframe.columns:
                EnhancedLogger.log_warning("Baseline diff отсутствует, вычисляем...", "📈")
                try:
                    from pandas_ta import ema
                    dataframe['baseline'] = ema(dataframe['close'], length=14)
                    dataframe['baseline_diff'] = dataframe['baseline'].diff()
                except:
                    dataframe['baseline'] = dataframe['close'].ewm(span=14).mean()
                    dataframe['baseline_diff'] = dataframe['baseline'].diff()
                EnhancedLogger.log_success("Базовая линия вычислена", "📈")
            
            # NOW CREATE EXACTLY 12 FEATURES (FIXED COUNT)
            feature_names = []
            
            # 1. Market volatility features (2 features)
            atr_mean = dataframe['atr'].rolling(14).mean().iloc[-1]
            features.append(atr_mean if not pd.isna(atr_mean) else 0.01)
            feature_names.append("ATR_среднее")
            
            atr_std = dataframe['atr'].rolling(7).std().iloc[-1]
            features.append(atr_std if not pd.isna(atr_std) else 0.001)
            feature_names.append("ATR_стд")
            
            # 2. Price momentum features (3 features)
            for period in [5, 10, 20]:
                pct_change = dataframe['close'].pct_change(period).iloc[-1]
                features.append(pct_change if not pd.isna(pct_change) else 0.0)
                feature_names.append(f"импульс_{period}")
            
            # 3. Volume features (2 features)
            if 'volume' in dataframe.columns and not dataframe['volume'].isna().all():
                vol_mean = dataframe['volume'].rolling(14).mean().iloc[-1]
                features.append(vol_mean if not pd.isna(vol_mean) else 1000.0)
                feature_names.append("объём_среднее")
                
                vol_pct = dataframe['volume'].pct_change().iloc[-1]
                features.append(vol_pct if not pd.isna(vol_pct) else 0.0)
                feature_names.append("объём_изменение")
                
                EnhancedLogger.log_success("Признаки объёма добавлены", "📊")
            else:
                features.extend([1000.0, 0.0])
                feature_names.extend(["объём_среднее_дефолт", "объём_изменение_дефолт"])
                EnhancedLogger.log_warning("Используются признаки объёма по умолчанию", "📊")
            
            # 4. Fisher transform features (3 features)
            fisher_current = dataframe['fisher'].iloc[-1]
            features.append(fisher_current if not pd.isna(fisher_current) else 0.0)
            feature_names.append("fisher_текущий")
            
            fisher_mean = dataframe['fisher'].rolling(5).mean().iloc[-1]
            features.append(fisher_mean if not pd.isna(fisher_mean) else 0.0)
            feature_names.append("fisher_среднее")
            
            fisher_std = dataframe['fisher'].rolling(5).std().iloc[-1]
            features.append(fisher_std if not pd.isna(fisher_std) else 1.0)
            feature_names.append("fisher_стд")
            
            # 5. Baseline trend features (1 feature - COMBINED TO SAVE SPACE)
            baseline_diff_mean = dataframe['baseline_diff'].rolling(5).mean().iloc[-1]
            baseline_diff_sum = dataframe['baseline_diff'].rolling(10).sum().iloc[-1]
            
            # COMBINE baseline features into one normalized feature
            if not pd.isna(baseline_diff_mean) and not pd.isna(baseline_diff_sum):
                combined_baseline = (baseline_diff_mean + baseline_diff_sum * 0.1)  # Weighted combination
            else:
                combined_baseline = 0.0
            
            features.append(combined_baseline)
            feature_names.append("базовая_линия_комб")
            
            # 6. Market regime features (1 feature)
            sma_50 = dataframe['close'].rolling(50).mean().iloc[-1]
            sma_200 = dataframe['close'].rolling(200).mean().iloc[-1]
            
            if not pd.isna(sma_50) and not pd.isna(sma_200) and sma_200 != 0:
                regime_feature = 1.0 if sma_50 > sma_200 else 0.0
                regime_status = "БЫЧИЙ 🐂" if sma_50 > sma_200 else "МЕДВЕЖИЙ 🐻"
            else:
                regime_feature = 0.5
                regime_status = "НЕЙТРАЛЬНЫЙ ⚖️"
            
            features.append(regime_feature)
            feature_names.append("рыночный_режим")
            
            # VERIFY EXACTLY 12 FEATURES
            if len(features) != 12:
                EnhancedLogger.log_error(f"Ошибка количества признаков: ожидалось 12, получено {len(features)}", "❌")
                # Force exactly 12 features
                if len(features) > 12:
                    features = features[:12]
                    feature_names = feature_names[:12]
                    EnhancedLogger.log_warning("Признаки обрезаны до 12", "✂️")
                else:
                    while len(features) < 12:
                        features.append(0.0)
                        feature_names.append(f"заглушка_{len(features)}")
                    EnhancedLogger.log_warning("Признаки дополнены до 12", "📋")
            
            EnhancedLogger.log_success(f"Создано {len(features)} ML признаков", "✨")
            EnhancedLogger.log_parameter("Рыночный режим", regime_status, "🏛️")
            
            # Debug log feature names (optional)
            if len(features) == 12:
                EnhancedLogger.log_success("Количество признаков подтверждено: 12/12", "✅")
            else:
                EnhancedLogger.log_error(f"Количество признаков всё ещё неверно: {len(features)}/12", "❌")
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            EnhancedLogger.log_error(f"Ошибка в create_features: {e}")
            # Return exactly 12 zero features as fallback
            return np.zeros((1, 12))
    
    def optimize_parameters(self, dataframe: DataFrame, current_performance: float):
        """Оптимизация параметров стратегии через Optuna"""
        
        EnhancedLogger.log_banner("ЗАПУСК ОПТИМИЗАЦИИ OPTUNA", "🔬")
        EnhancedLogger.log_performance("Текущая эффективность", current_performance, "📊")
        
        def objective(trial):
            # Entry parameters
            fisher_period = trial.suggest_int('fisher_period', 10, 15)
            fisher_smooth_long = trial.suggest_int('fisher_smooth_long', 3, 10)
            fisher_smooth_short = trial.suggest_int('fisher_smooth_short', 3, 10)
            fisher_buy_threshold = trial.suggest_float('fisher_buy_threshold', -1.0, 2.5)
            baseline_period = trial.suggest_int('baseline_period', 5, 21)
            atr_period = trial.suggest_int('atr_period', 7, 21)
            goldie_locks = trial.suggest_float('goldie_locks', 1.5, 3.0)
            
            # EXIT PARAMETERS
            fisher_long_exit = trial.suggest_float('fisher_long_exit', -1.0, 1.0)
            fisher_short_exit = trial.suggest_float('fisher_short_exit', -1.0, 1.0)
            fisher_sell_threshold = trial.suggest_float('fisher_sell_threshold', 2.0, 3.9)
            
            # Risk management parameters
            atr_sl_long_multip = trial.suggest_float('atr_sl_long_multip', 1.0, 6.0)
            atr_sl_short_multip = trial.suggest_float('atr_sl_short_multip', 1.0, 6.0)
            rr_long = trial.suggest_float('rr_long', 1.0, 4.0)
            rr_short = trial.suggest_float('rr_short', 1.0, 4.0)
            
            complete_params = {
                'fisher_period': fisher_period,
                'fisher_smooth_long': fisher_smooth_long,
                'fisher_smooth_short': fisher_smooth_short,
                'fisher_buy_threshold': fisher_buy_threshold,
                'baseline_period': baseline_period,
                'atr_period': atr_period,
                'goldie_locks': goldie_locks,
                'fisher_long_exit': fisher_long_exit,
                'fisher_short_exit': fisher_short_exit,
                'fisher_sell_threshold': fisher_sell_threshold,
                'atr_sl_long_multip': atr_sl_long_multip,
                'atr_sl_short_multip': atr_sl_short_multip,
                'rr_long': rr_long,
                'rr_short': rr_short
            }
            
            # Log trial progress
            if len(self.study.trials) % 5 == 0:
                trial_num = len(self.study.trials) + 1
                EnhancedLogger.log_subsection(f"Испытание #{trial_num}", "🧪")
                EnhancedLogger.log_parameter("Выход лонг", f"{fisher_long_exit:.3f}", "📤")
                EnhancedLogger.log_parameter("Выход шорт", f"{fisher_short_exit:.3f}", "📤")
            
            score = self.simulate_performance(dataframe, complete_params)
            return score
        
        # Create or load study
        if self.study is None:
            self.study = optuna.create_study(direction='maximize')
            EnhancedLogger.log_success("Создано новое исследование Optuna", "🔬")
        
        # Optimize for a few trials
        start_time = datetime.datetime.now()
        self.study.optimize(objective, n_trials=70, timeout=30)
        optimization_time = (datetime.datetime.now() - start_time).total_seconds()
        
        EnhancedLogger.log_banner("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА", "🎯")
        EnhancedLogger.log_performance("Лучший Score", self.study.best_value, "🏆")
        EnhancedLogger.log_parameter("Время оптимизации", f"{optimization_time:.1f} сек", "⏱️")
        EnhancedLogger.log_parameter("Всего испытаний", len(self.study.trials), "🔢")
        
        # Get best parameters
        best_params = self.study.best_params.copy()
        
        # Verify all exit parameters are present
        required_exit_params = ['fisher_long_exit', 'fisher_short_exit', 'fisher_sell_threshold']
        missing_params = []
        for param in required_exit_params:
            if param not in best_params:
                missing_params.append(param)
                if param == 'fisher_long_exit':
                    best_params[param] = -0.5
                elif param == 'fisher_short_exit':
                    best_params[param] = 0.5
                elif param == 'fisher_sell_threshold':
                    best_params[param] = 2.5
        
        if missing_params:
            EnhancedLogger.log_warning(f"Добавлены дефолтные значения для: {missing_params}", "🔧")
        
        EnhancedLogger.log_success(f"Параметры проверены: {len(best_params)} всего", "✅")
        
        # Save the study
        self.save_models()
        
        return best_params

    def simulate_performance(self, dataframe: DataFrame, params: dict) -> float:
        """Симуляция эффективности стратегии"""
        try:
            recent_data = dataframe.tail(50)
            fisher = self.calculate_fisher_simple(recent_data, params['fisher_period'])
            baseline = ema(recent_data['close'], length=params['baseline_period'])
            
            buy_signals = (fisher > params['fisher_buy_threshold']).astype(int)
            
            exit_signals = pd.Series(0, index=recent_data.index)
            if 'fisher_long_exit' in params:
                exit_signals = (fisher < params['fisher_long_exit']).astype(int)
            
            returns = recent_data['close'].pct_change().shift(-1)
            position = 0
            strategy_returns = []
            
            for i in range(len(recent_data)):
                if buy_signals.iloc[i] == 1 and position == 0:
                    position = 1
                elif exit_signals.iloc[i] == 1 and position == 1:
                    position = 0
                
                return_val = returns.iloc[i] if not pd.isna(returns.iloc[i]) else 0
                strategy_returns.append(position * return_val)
            
            # FIX: Ensure we return a scalar float
            performance = sum(strategy_returns)
            if hasattr(performance, 'item'):  # numpy scalar
                performance = performance.item()
            
            return float(performance)
            
        except Exception as e:
            EnhancedLogger.log_error(f"Ошибка симуляции эффективности: {e}")
            return -1.0

    def calculate_fisher_simple(self, dataframe: DataFrame, period: int) -> pd.Series:
        """Упрощённый расчёт Fisher Transform"""
        median_price = (dataframe['high'] + dataframe['low']) / 2
        fisher = pd.Series(0.0, index=dataframe.index)
        
        for i in range(period, len(dataframe)):
            price_window = median_price.iloc[i-period:i]
            price_min = price_window.min()
            price_max = price_window.max()
            if price_max != price_min:
                norm = (median_price.iloc[i] - price_min) / (price_max - price_min)
                norm = 2 * norm - 1
                norm = max(min(norm, 0.999), -0.999)
                fisher.iloc[i] = 0.5 * np.log((1 + norm) / (1 - norm))
        
        return fisher
    
    def predict_optimal_params(self, dataframe: DataFrame) -> dict:
        """Прогноз оптимальных параметров с помощью ML модели"""
        if self.model is None or self.scaler is None:
            EnhancedLogger.log_warning("ML модель или масштабировщик недоступны", "🤖")
            return {}
        
        try:
            EnhancedLogger.log_subsection("Прогноз параметров ML", "🔮")
            
            features = self.create_features(dataframe)
            features_scaled = self.scaler.transform(features)
            
            predictions = self.model.predict(features_scaled)
            
            # Fix: Ensure predictions is a 2D array and handle single prediction
            if predictions.ndim == 1:
                predictions = predictions.reshape(1, -1)
            elif predictions.ndim == 0:
                predictions = np.array([[predictions]])
            
            # EXPANDED: Predict ALL required parameters instead of just 6
            param_names = [
                'fisher_period', 'fisher_smooth_long', 'fisher_smooth_short',
                'baseline_period', 'atr_period', 'goldie_locks',
                'fisher_buy_threshold', 'fisher_sell_threshold',
                'fisher_long_exit', 'fisher_short_exit',
                'atr_sl_long_multip', 'atr_sl_short_multip',
                'rr_long', 'rr_short'
            ]
            
            param_dict = {}
            
            # Handle case where model only predicts 6 values but we need 14
            if len(predictions) > 0 and len(predictions[0]) > 0:
                model_predictions = predictions[0]
                
                for i, name in enumerate(param_names):
                    if i < len(model_predictions):
                        # Direct ML prediction available
                        param_value = model_predictions[i]
                        if hasattr(param_value, 'item'):
                            param_value = param_value.item()
                        param_dict[name] = max(0.1, float(param_value))
                    else:
                        # Generate derived/interpolated values for missing parameters
                        param_dict[name] = self._generate_derived_parameter(name, param_dict)
            
            if param_dict:
                EnhancedLogger.log_success(f"ML спрогнозировал {len(param_dict)} параметров", "🔮")
                # Log which were direct predictions vs derived
                direct_count = min(len(param_names), len(predictions[0]) if len(predictions) > 0 else 0)
                derived_count = len(param_dict) - direct_count
                if derived_count > 0:
                    EnhancedLogger.log_warning(f"Производных параметров из ML базы: {derived_count}", "🔄")
            else:
                EnhancedLogger.log_warning("ML прогноз вернул пустой результат", "⚠️")
            
            return param_dict
            
        except Exception as e:
            EnhancedLogger.log_error(f"Ошибка ML прогноза: {e}")
            return {}

    def _generate_derived_parameter(self, param_name: str, existing_params: dict) -> float:
        """Генерация производных параметров на основе существующих ML прогнозов"""
        
        # Use intelligent defaults based on parameter relationships
        if param_name == 'fisher_smooth_short':
            # Base on fisher_smooth_long if available
            if 'fisher_smooth_long' in existing_params:
                return max(3, min(10, existing_params['fisher_smooth_long'] - 1))
            return 6.0
        
        elif param_name == 'fisher_buy_threshold':
            # Typically opposite sign of fisher_long_exit
            if 'fisher_long_exit' in existing_params:
                return abs(existing_params['fisher_long_exit']) + 1.0
            return 1.5
        
        elif param_name == 'fisher_sell_threshold':
            # Usually higher than buy threshold
            if 'fisher_buy_threshold' in existing_params:
                return existing_params['fisher_buy_threshold'] + 1.0
            return 2.8
        
        elif param_name == 'goldie_locks':
            # Related to ATR period
            if 'atr_period' in existing_params:
                return 1.5 + (existing_params['atr_period'] - 14) * 0.1
            return 2.0
        
        elif param_name == 'atr_sl_long_multip':
            return 2.5  # Conservative default
        
        elif param_name == 'atr_sl_short_multip':
            return 2.5  # Conservative default
        
        elif param_name == 'rr_long':
            return 3.0  # Good risk/reward ratio
        
        elif param_name == 'rr_short':
            return 3.0  # Good risk/reward ratio
        
        else:
            # Fallback for any other parameters
            return 1.0

    def update_model(self, dataframe: DataFrame, performance: float):
        """Обновление ML модели новыми данными"""
        try:
            EnhancedLogger.log_section("ОБНОВЛЕНИЕ ML МОДЕЛИ", "🧠")
            
            features = self.create_features(dataframe)
            
            # FIX: Ensure features is properly flattened
            if features.ndim > 1:
                features_flat = features.flatten()
            else:
                features_flat = features
            
            self.performance_history.append({
                'features': features_flat,
                'performance': float(performance),  # Ensure scalar
                'timestamp': datetime.datetime.now()
            })
            
            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
                EnhancedLogger.log_warning("История обрезана до последних 100 образцов", "📊")
            
            # Train model if we have enough data
            if len(self.performance_history) >= 15:
                X = np.array([h['features'] for h in self.performance_history])
                y = np.array([h['performance'] for h in self.performance_history])
                
                # FIX: Ensure proper array shapes
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                if y.ndim > 1:
                    y = y.flatten()
                
                # Scale features
                if self.scaler is None:
                    self.scaler = StandardScaler()
                    EnhancedLogger.log_success("Создан новый масштабировщик признаков", "📏")
                    
                X_scaled = self.scaler.fit_transform(X)
                
                # Train model
                if self.model is None:
                    self.model = RandomForestRegressor(n_estimators=200, random_state=42)
                    EnhancedLogger.log_success("Создана новая модель RandomForest", "🌲")
                
                self.model.fit(X_scaled, y)
                model_score = self.model.score(X_scaled, y)
                
                EnhancedLogger.log_success(f"ML модель обновлена: {len(self.performance_history)} образцов", "🧠")
                EnhancedLogger.log_performance("R² Score модели", model_score, "📊")
                
                # Feature importance analysis
                if hasattr(self.model, 'feature_importances_'):
                    top_features = np.argsort(self.model.feature_importances_)[-3:]
                    EnhancedLogger.log_subsection("Топ-3 важных признака", "🔍")
                    for i, feat_idx in enumerate(reversed(top_features)):
                        importance = self.model.feature_importances_[feat_idx]
                        EnhancedLogger.log_parameter(f"Признак #{feat_idx}", f"{importance:.3f}", "⭐")
                
                self.save_models()
            else:
                samples_needed = 20 - len(self.performance_history)
                EnhancedLogger.log_warning(f"Нужно ещё {samples_needed} образцов для обучения модели", "📊")
                
        except Exception as e:
            EnhancedLogger.log_error(f"Ошибка обновления модели: {e}")


class SampleStrategy(IStrategy):
    # Strategy parameters
    timeframe = "5m"
    startup_candle_count = 3000
    minimal_roi = {}
    stoploss = -0.20
    use_custom_stoploss = False
    trailing_stop = False
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    
    can_short = True
    set_leverage = 3
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.ml_optimizers = {}
        self.last_ml_update = None
        self.ml_update_frequency = 6
        self.trade_performance_cache = {}
        self.enable_ml_optimization = True
        self.initial_training_completed = {}  # Track per-pair training status
        
        # 🚀 STARTUP TRAINING CONFIGURATION
        self.startup_training_enabled = True
        self.startup_training_periods = 1000  # Use last 1000 candles for training
        self.startup_training_pairs = []  # Will be populated with active pairs
        
        logger.info("🤖 Fisher Transform ML Стратегия v4 — Расширенное начальное обучение")
        logger.info("🎯 Начальное обучение запустится сразу после первого анализа данных")
        # Enhanced initialization logging
        EnhancedLogger.log_banner("FISHER TRANSFORM ML СТРАТЕГИЯ ИНИЦИАЛИЗИРОВАНА", "🚀")
        EnhancedLogger.log_parameter("Таймфрейм", self.timeframe, "⏰")
        EnhancedLogger.log_parameter("Шорт", self.can_short, "📊")
        EnhancedLogger.log_parameter("Плечо", self.set_leverage, "⚖️")
        EnhancedLogger.log_parameter("ML оптимизация", self.enable_ml_optimization, "🤖")

    # Hyperparameters with ML integration
    if can_short:
        buy_params = {
            "atr_period": 20, "baseline_period": 5, "fisher_buy_threshold": 1.0,  # V1.2: Relaxed from 2.39 to 1.0
            "fisher_period": 14, "fisher_smooth_long": 9, "fisher_smooth_short": 9,
            "goldie_locks": 2.85,
        }
        sell_params = {
            "fisher_long_exit": -0.5, "fisher_short_exit": 0.5, "fisher_sell_threshold": -0.5,  # V1.2: Symmetric thresholds
        }
        minimal_roi = {"0": 0.373, "1019": 0.22, "3124": 0.076, "4482": 0}
        stoploss = -0.15  # V1.2: CRITICAL FIX - Tightened from -0.524 (52%!) to -0.15 (15%)
        trailing_stop = True  # V1.2: Enable trailing stop
        trailing_stop_positive = 0.02
        trailing_stop_positive_offset = 0.05
        trailing_only_offset_is_reached = True
        max_open_trades = 5  # V1.2: Reduced from 20 to limit risk
    else:
        buy_params = {
            "atr_period": 21, "baseline_period": 11, "fisher_buy_threshold": 0.65,
            "fisher_period": 13, "fisher_smooth_long": 7, "goldie_locks": 1.6,
            "fisher_smooth_short": 6,
        }
        sell_params = {
            "fisher_long_exit": 0.837, "fisher_sell_threshold": 2.89, "fisher_short_exit": 0.293,
        }
        minimal_roi = {"0": 0.871, "1787": 0.323, "2415": 0.118, "5669": 0}
        stoploss = -0.591
        trailing_stop = False
        trailing_stop_positive = 0.345
        trailing_stop_positive_offset = 0.373
        trailing_only_offset_is_reached = True
        max_open_trades = 3

    # ML-enhanced parameters with dynamic optimization
    fisher_period = IntParameter(10, 15, default=buy_params.get('fisher_period'), space="buy", optimize=True)
    fisher_smooth_long = IntParameter(3, 10, default=buy_params.get('fisher_smooth_long'), space="buy", optimize=True)
    fisher_smooth_short = IntParameter(3, 10, default=buy_params.get('fisher_smooth_short'), space="buy", optimize=can_short)
    fisher_short_exit = DecimalParameter(-1.0, 1.0, default=sell_params.get('fisher_short_exit'), decimals=3, space="sell", optimize=can_short)
    fisher_long_exit = DecimalParameter(-1.0, 1.0, default=sell_params.get('fisher_long_exit'), decimals=3, space="sell", optimize=True)
    fisher_sell_threshold = DecimalParameter(-2.0, -0.5, default=-1.0, decimals=2, space="sell", optimize=True)  # V1.2: Changed to negative for symmetry
    fisher_buy_threshold = DecimalParameter(0.5, 2.0, default=1.0, decimals=2, space="buy", optimize=True)  # V1.2: Positive for long entries
    baseline_period = IntParameter(5, 21, default=buy_params.get('baseline_period'), space="buy", optimize=True)
    atr_period = IntParameter(7, 21, default=buy_params.get('atr_period'), space="buy", optimize=True)
    goldie_locks = DecimalParameter(1.5, 3.0, default=buy_params.get('goldie_locks'), decimals=2, space="buy", optimize=True)
    
    # ML confidence parameters
    ml_confidence_threshold = DecimalParameter(0.3, 0.5, default=0.4, decimals=2, space="buy", optimize=True)
    ml_adaptation_rate = DecimalParameter(0.1, 0.5, default=0.2, decimals=2, space="buy", optimize=True)
    ml_signal_threshold = DecimalParameter(0.1, 0.8, default=0.4, decimals=2, space="buy", optimize=True)  # Add this line

    # Risk management with ML
    ATR_SL_short_Multip = DecimalParameter(1.0, 6.0, decimals=1, default=1.5, space="sell", optimize=True)
    ATR_SL_long_Multip = DecimalParameter(1.0, 6.0, decimals=1, default=1.5, space="sell", optimize=True)
    ATR_Multip = DecimalParameter(1.0, 6.0, decimals=1, default=1.5, space="sell", optimize=True)
    rr_long = DecimalParameter(1.0, 4.0, decimals=1, default=2.0, space="sell", optimize=True)
    rr_short = DecimalParameter(1.0, 4.0, decimals=1, default=2.0, space="sell", optimize=True)
    
    # DCA Configuration
    overbuy_factor = 1.295
    position_adjustment_enable = True
    initial_safety_order_trigger = -0.02
    max_so_multiplier_orig = 3
    safety_order_step_scale = 2
    safety_order_volume_scale = 1.8
    max_so_multiplier = max_so_multiplier_orig
    cust_proposed_initial_stakes = {}
    partial_fill_compensation_scale = 1
    
    # DCA calculation
    if max_so_multiplier_orig > 0:
        if safety_order_volume_scale > 1:
            firstLine = safety_order_volume_scale * (math.pow(safety_order_volume_scale, (max_so_multiplier_orig - 1)) - 1)
            divisor = safety_order_volume_scale - 1
            max_so_multiplier = 2 + firstLine / divisor
        elif safety_order_volume_scale < 1:
            firstLine = safety_order_volume_scale * (1 - math.pow(safety_order_volume_scale, (max_so_multiplier_orig - 1)))
            divisor = 1 - safety_order_volume_scale
            max_so_multiplier = 2 + firstLine / divisor
    
    #stoploss = -1
    
    def get_ml_adjusted_params(self, dataframe: DataFrame, pair: str) -> dict:
        """Получение ML-скорректированных параметров для пары"""
        try:
            EnhancedLogger.log_section(f"ML КОРРЕКТИРОВКА ПАРАМЕТРОВ — {pair}", "🤖")
            
            # Skip ML if disabled or insufficient data
            if not self.enable_ml_optimization or len(dataframe) < 50:
                EnhancedLogger.log_warning(f"ML оптимизация пропущена для {pair} (отключена или мало данных)", "⚠️")
                return {}
            
            # Ensure required columns exist before ML operations
            required_columns = ['close', 'high', 'low']
            if not all(col in dataframe.columns for col in required_columns):
                EnhancedLogger.log_error(f"Отсутствуют обязательные столбцы для {pair}", "❌")
                return {}
            
            # Create pair-specific optimizer if doesn't exist
            if pair not in self.ml_optimizers:
                self.ml_optimizers[pair] = MLOptimizer(f"GKD_FisherV5_{pair.replace('/', '_')}")
                EnhancedLogger.log_success(f"Создан ML оптимизатор для {pair}", "🆕")
            
            ml_optimizer = self.ml_optimizers[pair]
            
            # Check if it's time to update ML model for this pair
            current_time = datetime.datetime.now()
            should_update = (self.last_ml_update is None or 
                            (current_time - self.last_ml_update).total_seconds() > self.ml_update_frequency * 3600)
            
            if should_update and len(dataframe) > 100:
                try:
                    EnhancedLogger.log_subsection(f"Обновление ML модели для {pair}", "🔄")
                    
                    # Update ML model with recent performance for this pair
                    recent_performance = self.calculate_recent_performance(pair)
                    EnhancedLogger.log_performance("Недавняя эффективность", recent_performance, "📊")
                    
                    ml_optimizer.update_model(dataframe, recent_performance)
                    self.last_ml_update = current_time
                    
                    EnhancedLogger.log_success(f"ML модель обновлена в {current_time.strftime('%H:%M:%S')}", "✅")
                    
                    # Optimize parameters with Optuna for this specific pair
                    optimized_params = ml_optimizer.optimize_parameters(dataframe, recent_performance)
                    
                    if optimized_params:
                        # Ensure all exit parameters are present in optimized results
                        self._ensure_all_parameters(optimized_params, pair)
                        
                        # Add optimization score
                        optimized_params['score'] = ml_optimizer.study.best_value if ml_optimizer.study else 0.0
                        
                        EnhancedLogger.log_success(f"Optuna вернул {len(optimized_params)} параметров", "🎯")
                        
                        # Log the optimized parameters with enhanced formatting
                        self.log_formatted_parameters(pair, optimized_params)
                        
                        return optimized_params
                    else:
                        EnhancedLogger.log_warning(f"Optuna не вернул результатов для {pair}", "⚠️")
                
                except Exception as e:
                    EnhancedLogger.log_error(f"Ошибка ML оптимизации для {pair}: {str(e)}", "❌")
            
            # Get ML predictions for optimal parameters for this pair (fallback)
            try:
                EnhancedLogger.log_subsection(f"Получение ML прогноза для {pair}", "🔮")
                ml_params = ml_optimizer.predict_optimal_params(dataframe)
                
                # ALWAYS ensure exit parameters are included (critical fix)
                self._ensure_all_parameters(ml_params, pair)
                
                if ml_params:
                    EnhancedLogger.log_success(f"Используются ML прогнозируемые параметры: {len(ml_params)} всего", "✨")
                
                return ml_params
                
            except Exception as e:
                EnhancedLogger.log_error(f"Ошибка ML прогноза для {pair}: {str(e)}", "❌")
                return self._get_default_parameters()
                
        except Exception as e:
            EnhancedLogger.log_error(f"Ошибка ML корректировки для {pair}: {str(e)}", "💥")
            return self._get_default_parameters()

# PART 3 - Continuing from Part 2

    def _get_default_parameters(self) -> dict:
        """Параметры по умолчанию как запасной вариант"""
        EnhancedLogger.log_warning("Используются параметры по умолчанию", "🔄")
        return {
            'fisher_long_exit': self.fisher_long_exit.value,
            'fisher_short_exit': self.fisher_short_exit.value,
            'fisher_sell_threshold': self.fisher_sell_threshold.value,
            'atr_sl_long_multip': self.ATR_SL_long_Multip.value,
            'atr_sl_short_multip': self.ATR_SL_short_Multip.value,
            'rr_long': self.rr_long.value,
            'rr_short': self.rr_short.value
        }
    
    def _ensure_all_parameters(self, params: dict, pair: str) -> None:
        """Проверка наличия всех обязательных параметров"""
        required_params = {
            # Fisher Transform Parameters
            'fisher_period': 14,
            'fisher_smooth_long': 7,
            'fisher_smooth_short': 6,
            'fisher_buy_threshold': 1.5,
            'fisher_sell_threshold': 2.8,
            'fisher_long_exit': -0.5,
            'fisher_short_exit': 0.5,
            
            # Baseline & ATR Parameters  
            'baseline_period': 14,
            'atr_period': 14,
            'goldie_locks': 2.0,
            
            # Risk Management Parameters
            'atr_sl_long_multip': 2.5,
            'atr_sl_short_multip': 2.5,
            'rr_long': 3.0,
            'rr_short': 3.0
        }
        
        missing_count = 0
        added_params = []
        
        for param, default_value in required_params.items():
            if param not in params:
                params[param] = default_value
                missing_count += 1
                added_params.append(param)
        
        if missing_count > 0:
            EnhancedLogger.log_warning(f"Добавлено {missing_count} отсутствующих параметров для {pair}", "🔧")
            # Log which parameters were added (for debugging)
            EnhancedLogger.log_parameter("Добавленные параметры", ", ".join(added_params[:3]) + "..." if len(added_params) > 3 else ", ".join(added_params), "📋")
        else:
            EnhancedLogger.log_success(f"Все {len(required_params)} параметров проверены для {pair}", "✅")
    
    def calculate_recent_performance(self, pair: str = None) -> float:
        """Расчёт недавней эффективности стратегии"""
        try:
            if pair:
                pair_trades = [perf for p, perf in self.trade_performance_cache.items() if p == pair]
                if not pair_trades:
                    EnhancedLogger.log_warning(f"Нет истории сделок для {pair}", "📊")
                    return 0.0
                performance = sum(pair_trades[-5:]) / len(pair_trades[-5:])
                EnhancedLogger.log_performance(f"Недавняя эффективность ({pair})", performance, "🎯")
                return performance
            else:
                if not self.trade_performance_cache:
                    EnhancedLogger.log_warning("История сделок отсутствует", "📊")
                    return 0.0
                recent_trades = list(self.trade_performance_cache.values())[-10:]
                performance = sum(recent_trades) / len(recent_trades)
                EnhancedLogger.log_performance("Общая недавняя эффективность", performance, "🌟")
                return performance
        except Exception as e:
            EnhancedLogger.log_error(f"Ошибка расчёта эффективности: {e}", "💥")
            return 0.0
    
    def log_formatted_parameters(self, pair: str, params: Dict[str, Any]):
        """Форматированный вывод оптимизированных параметров"""
        EnhancedLogger.log_banner(f"ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ ДЛЯ {pair}", "🎯")
        
        # Fisher Transform Parameters Section
        EnhancedLogger.log_section("НАСТРОЙКИ FISHER TRANSFORM", "🎣")
        fisher_params = {
            "fisher_period": ("🔄", "Период"),
            "fisher_smooth_long": ("📈", "Сглаживание лонг"),
            "fisher_smooth_short": ("📉", "Сглаживание шорт"), 
            "fisher_buy_threshold": ("🚀", "Порог покупки")
        }
        
        for param, (emoji, name) in fisher_params.items():
            if param in params:
                EnhancedLogger.log_parameter(name, params[param], emoji)
        
        # Fisher Exit Parameters Section  
        EnhancedLogger.log_section("НАСТРОЙКИ ВЫХОДА", "🚪")
        exit_params = {
            "fisher_long_exit": ("📤", "Выход из лонга"),
            "fisher_short_exit": ("📥", "Выход из шорта"),
            "fisher_sell_threshold": ("🛑", "Порог продажи")
        }
        
        for param, (emoji, name) in exit_params.items():
            if param in params:
                value = params[param]
                # Color coding for exit levels
                if isinstance(value, (int, float)):
                    if value > 0:
                        color_status = "🟢 ПОЛОЖИТ."
                    elif value < 0:
                        color_status = "🔴 ОТРИЦАТ." 
                    else:
                        color_status = "🟡 НЕЙТРАЛ."
                    EnhancedLogger.log_parameter(f"{name} {color_status}", f"{value:.3f}", emoji)
                else:
                    EnhancedLogger.log_parameter(name, value, emoji)
        
        # Baseline & Volatility Section
        EnhancedLogger.log_section("БАЗОВАЯ ЛИНИЯ И ВОЛАТИЛЬНОСТЬ", "📊")
        baseline_params = {
            "baseline_period": ("📏", "Период базовой линии"),
            "atr_period": ("🌊", "Период ATR"),
            "goldie_locks": ("🔒", "Зона Голди Локс")
        }
        
        for param, (emoji, name) in baseline_params.items():
            if param in params:
                EnhancedLogger.log_parameter(name, params[param], emoji)
        
        # Risk Management Section
        EnhancedLogger.log_section("УПРАВЛЕНИЕ РИСКАМИ", "⚖️")
        risk_params = {
            "atr_sl_long_multip": ("🛡️", "Множитель SL лонг"),
            "atr_sl_short_multip": ("🛡️", "Множитель SL шорт"),
            "rr_long": ("💰", "R/R лонг"),
            "rr_short": ("💰", "R/R шорт")
        }
        
        for param, (emoji, name) in risk_params.items():
            if param in params:
                value = params[param]
                if isinstance(value, (int, float)):
                    # Risk level indication
                    if 'sl_' in param:  # Stop loss multipliers
                        risk_level = "🟢 КОНСЕРВ." if value <= 2.0 else "🟡 УМЕРЕН." if value <= 4.0 else "🔴 АГРЕССИВ."
                        EnhancedLogger.log_parameter(f"{name} ({risk_level})", f"{value:.2f}x", emoji)
                    else:  # Risk/Reward ratios
                        rr_quality = "🟢 ОТЛИЧНО" if value >= 3.0 else "🟡 ХОРОШО" if value >= 2.0 else "🔴 РИСКОВАННО"
                        EnhancedLogger.log_parameter(f"{name} ({rr_quality})", f"{value:.1f}:1", emoji)
                else:
                    EnhancedLogger.log_parameter(name, value, emoji)
        
        # Optimization Quality Assessment
        if 'score' in params:
            score = params['score']
            if score > 0.1:
                quality = "🟢 ПРЕВОСХОДНО"
            elif score > 0.05:
                quality = "🟡 ХОРОШО"
            elif score > 0:
                quality = "🟠 УДОВЛЕТВ."
            else:
                quality = "🔴 ПЛОХО"
            
            EnhancedLogger.log_section("КАЧЕСТВО ОПТИМИЗАЦИИ", "📈")
            EnhancedLogger.log_performance(f"Score {quality}", score, "🏆")
        
        # Summary
        param_count = len([p for p in params.keys() if p != 'score'])
        EnhancedLogger.log_section("СВОДКА ПАРАМЕТРОВ", "📋")
        EnhancedLogger.log_parameter("Всего параметров", param_count, "🔢")
        EnhancedLogger.log_parameter("Время оптимизации", datetime.datetime.now().strftime("%H:%M:%S"), "⏰")
        
        # Visual separator
        logger.info("🔹" * 60)
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Расчёт индикаторов с ML интеграцией"""
        """Enhanced with STARTUP TRAINING"""
        pair = metadata.get('pair', 'Unknown')
        
        # 🚀 STARTUP TRAINING - Run on first data load
        if (self.startup_training_enabled and 
            pair not in self.initial_training_completed and 
            len(dataframe) >= self.startup_training_periods):
            
            logger.info(f"🎯 [СТАРТ] Начинается начальное ML обучение для {pair}")
            self.perform_startup_training(dataframe, pair)
            self.initial_training_completed[pair] = True
        EnhancedLogger.log_banner(f"РАСЧЁТ ИНДИКАТОРОВ — {pair}", "📊")
        
        # Use default values initially
        fisher_period = self.fisher_period.value
        fisher_smooth_long = self.fisher_smooth_long.value
        fisher_smooth_short = self.fisher_smooth_short.value
        baseline_period = self.baseline_period.value
        atr_period = self.atr_period.value
        
        EnhancedLogger.log_section("НАЧАЛЬНЫЕ ПАРАМЕТРЫ", "⚙️")
        EnhancedLogger.log_parameter("Период Fisher", fisher_period, "🎣")
        EnhancedLogger.log_parameter("Период базовой линии", baseline_period, "📏")
        EnhancedLogger.log_parameter("Период ATR", atr_period, "🌊")
        
        # Calculate basic indicators first (required for ML features)
        EnhancedLogger.log_subsection("Расчёт базовых индикаторов", "🔧")
        
        try:
            dataframe["atr"] = talib.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=atr_period)
            EnhancedLogger.log_success("ATR вычислен", "✅")
            
            dataframe["fisher"] = self.calculate_fisher(dataframe, fisher_period)
            EnhancedLogger.log_success("Fisher Transform вычислен", "✅")
            
            dataframe["baseline"] = ema(dataframe["close"], length=baseline_period)
            dataframe["baseline_diff"] = dataframe["baseline"].diff()
            EnhancedLogger.log_success("Базовые индикаторы вычислены", "✅")
            
        except Exception as e:
            EnhancedLogger.log_error(f"Ошибка расчёта базовых индикаторов: {e}", "💥")
            raise
        
        # NOW get ML-adjusted parameters after basic indicators exist
        EnhancedLogger.log_subsection("Применение ML корректировок", "🤖")
        ml_params = self.get_ml_adjusted_params(dataframe, pair)
        
        # Apply ML adjustments to parameters if available
        if ml_params:
            EnhancedLogger.log_subsection("ML параметры обнаружены, пересчёт...", "🔄")
            
            original_params = {
                'fisher_period': fisher_period,
                'fisher_smooth_long': fisher_smooth_long, 
                'fisher_smooth_short': fisher_smooth_short,
                'baseline_period': baseline_period,
                'atr_period': atr_period
            }
            
            # Update parameters with ML suggestions
            fisher_period = ml_params.get('fisher_period', fisher_period)
            fisher_smooth_long = ml_params.get('fisher_smooth_long', fisher_smooth_long)
            fisher_smooth_short = ml_params.get('fisher_smooth_short', fisher_smooth_short)
            baseline_period = ml_params.get('baseline_period', baseline_period)
            atr_period = ml_params.get('atr_period', atr_period)
            
            # Ensure parameters are within valid ranges
            fisher_period = max(10, min(15, int(fisher_period)))
            fisher_smooth_long = max(3, min(10, int(fisher_smooth_long)))
            fisher_smooth_short = max(3, min(10, int(fisher_smooth_short)))
            baseline_period = max(5, min(21, int(baseline_period)))
            atr_period = max(7, min(21, int(atr_period)))
            
            # Log parameter changes
            changes_made = 0
            for param_name in original_params:
                old_val = original_params[param_name]
                new_val = locals()[param_name]
                if old_val != new_val:
                    changes_made += 1
                    EnhancedLogger.log_parameter(f"{param_name} изменён", f"{old_val} → {new_val}", "🔄")
            
            if changes_made > 0:
                EnhancedLogger.log_success(f"Применено {changes_made} ML корректировок параметров", "🎯")
                
                # Recalculate indicators with ML-adjusted parameters
                if fisher_period != self.fisher_period.value:
                    dataframe["fisher"] = self.calculate_fisher(dataframe, fisher_period)
                    EnhancedLogger.log_success("Fisher пересчитан с ML параметрами", "🔄")
                
                if baseline_period != self.baseline_period.value:
                    dataframe["baseline"] = ema(dataframe["close"], length=baseline_period)
                    dataframe["baseline_diff"] = dataframe["baseline"].diff()
                    EnhancedLogger.log_success("Базовая линия пересчитана с ML параметрами", "🔄")
                
                if atr_period != self.atr_period.value:
                    dataframe["atr"] = talib.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=atr_period)
                    EnhancedLogger.log_success("ATR пересчитан с ML параметрами", "🔄")
            else:
                EnhancedLogger.log_success("ML параметры совпадают с дефолтными", "✨")
        else:
            EnhancedLogger.log_warning("ML параметры недоступны, используются дефолтные", "⚠️")
        
        # Continue with remaining indicators
        EnhancedLogger.log_subsection("Расчёт производных индикаторов", "🔧")
        
        try:
            # Smooth Fisher with EMA
            dataframe["fisher_smooth_long"] = ema(dataframe["fisher"], length=fisher_smooth_long)
            dataframe["fisher_smooth_short"] = ema(dataframe["fisher"], length=fisher_smooth_short)
            # FIXED: trend_long now uses fisher_smooth_long instead of fisher_smooth_short
            dataframe["fisher_trend_long"] = ema(dataframe["fisher_smooth_long"], length=21)
            # FIXED: trend_short uses shorter length (9) for faster response
            dataframe["fisher_trend_short"] = ema(dataframe["fisher_smooth_short"], length=21)
            EnhancedLogger.log_success("Сглаживание Fisher завершено (тренды исправлены)", "✅")
            
            # Baseline indicators
            dataframe["baseline_up"] = dataframe["baseline_diff"] > 0
            dataframe["baseline_down"] = dataframe["baseline_diff"] < 0
            trend_up_pct = (dataframe["baseline_up"].tail(50).sum() / 50) * 100
            EnhancedLogger.log_parameter("% восходящего тренда", f"{trend_up_pct:.1f}%", "📈")
            
            # Volatility (ATR for Goldie Locks Zone)
            dataframe["goldie_min"] = dataframe["baseline"] - (dataframe["atr"] * self.goldie_locks.value)
            dataframe["goldie_max"] = dataframe["baseline"] + (dataframe["atr"] * self.goldie_locks.value)
            EnhancedLogger.log_success("Зоны Голди Локс вычислены", "✅")
            
            # ML confidence indicators
            dataframe["ml_confidence"] = self.calculate_ml_confidence(dataframe)
            dataframe["market_regime"] = self.identify_market_regime(dataframe)
            
            # Enhanced signals with ML
            dataframe["ml_signal_strength"] = self.calculate_signal_strength(dataframe)
            
            # Log ML indicator statistics
            avg_confidence = dataframe["ml_confidence"].tail(50).mean()
            avg_signal_strength = dataframe["ml_signal_strength"].tail(50).mean()
            current_regime = dataframe["market_regime"].iloc[-1]
            
            regime_text = "🐂 БЫЧИЙ" if current_regime > 0 else "🐻 МЕДВЕЖИЙ" if current_regime < 0 else "⚖️ НЕЙТРАЛЬНЫЙ"
            
            EnhancedLogger.log_section("СВОДКА ML ИНДИКАТОРОВ", "🤖")
            EnhancedLogger.log_parameter("Средняя уверенность ML", f"{avg_confidence:.1%}", "🎯")
            EnhancedLogger.log_parameter("Средняя сила сигнала", f"{avg_signal_strength:.3f}", "⚡")
            EnhancedLogger.log_parameter("Рыночный режим", regime_text, "🏛️")
            
            EnhancedLogger.log_success("Все ML индикаторы вычислены", "✅")
            
        except Exception as e:
            EnhancedLogger.log_error(f"Ошибка расчёта производных индикаторов: {e}", "💥")
            raise
        
        # Final summary
        if ml_params:
            EnhancedLogger.log_success(f"ML параметры активны: {len(ml_params)} корректировок", "🎯")
            EnhancedLogger.log_parameter("Активных оптимизаторов", len(self.ml_optimizers), "🤖")
        
        EnhancedLogger.log_banner(f"ИНДИКАТОРЫ ГОТОВЫ — {pair}", "🎉")
        
        return dataframe

    def perform_startup_training(self, dataframe: DataFrame, pair: str):
        """NEW: Perform ML training on startup using historical data"""
        try:
            logger.info(f"🧠 [СТАРТ] Обучение ML для {pair} на {len(dataframe)} исторических свечах")
            
            # Create pair-specific optimizer if doesn't exist
            if pair not in self.ml_optimizers:
                self.ml_optimizers[pair] = MLOptimizer(f"GKD_FisherV5_{pair.replace('/', '_')}")
            
            ml_optimizer = self.ml_optimizers[pair]
            
            # 📊 Generate synthetic training data from historical patterns
            training_data = self.generate_historical_training_data(dataframe, pair)
            
            if len(training_data) > 0:
                logger.info(f"📈 [СТАРТ] Сгенерировано {len(training_data)} обучающих образцов для {pair}")
                
                # Update ML optimizer with historical performance patterns
                for sample in training_data:
                    ml_optimizer.performance_history.append(sample)
                
                # Train the model immediately
                if len(ml_optimizer.performance_history) >= 20:
                    ml_optimizer.update_model(dataframe, 0.0)  # Use neutral performance for initial training
                    logger.info(f"✅ [СТАРТ] ML модель успешно обучена для {pair}")
                    
                    # Run initial Optuna optimization
                    logger.info(f"🎯 [СТАРТ] Запускается начальная оптимизация параметров для {pair}")
                    optimized_params = ml_optimizer.optimize_parameters(dataframe, 0.0)
                    
                    if optimized_params:
                        logger.info(f"🎉 [СТАРТ] Начальная оптимизация завершена для {pair}")
                        self.log_formatted_parameters(pair, optimized_params)
                    else:
                        logger.warning(f"⚠️ [СТАРТ] Начальная оптимизация не дала результатов для {pair}")
                else:
                    logger.warning(f"⚠️ [СТАРТ] Недостаточно обучающих данных для {pair}")
            else:
                logger.error(f"❌ [СТАРТ] Не удалось сгенерировать обучающие данные для {pair}")
                
        except Exception as e:
            logger.error(f"❌ [СТАРТ] Ошибка обучения для {pair}: {str(e)}")

    def generate_historical_training_data(self, dataframe: DataFrame, pair: str) -> List[Dict]:
        """NEW: Generate training data from historical price patterns"""
        try:
            training_samples = []
            lookback_period = len(dataframe) - 100  # Use ALL candles for training

            logger.info(f"🔍 [СТАРТ] Анализируется {lookback_period} исторических периодов для {pair}")
            
            # Calculate basic indicators needed for analysis
            dataframe_copy = dataframe.copy()
            dataframe_copy["atr"] = talib.ATR(dataframe_copy["high"], dataframe_copy["low"], 
                                             dataframe_copy["close"], timeperiod=14)
            dataframe_copy["fisher"] = self.calculate_fisher(dataframe_copy, 14)
            dataframe_copy["baseline"] = ema(dataframe_copy["close"], length=14)
            dataframe_copy["baseline_diff"] = dataframe_copy["baseline"].diff()
            
            # Generate training samples by analyzing historical patterns
            for i in range(100, lookback_period):  # Skip first 100 for indicator stability
                try:
                    # Extract features at this historical point
                    features = self.extract_features_at_index(dataframe_copy, i)
                    
                    # Calculate performance of next 10-20 candles as "target"
                    future_performance = self.calculate_future_performance(dataframe_copy, i, periods=15)
                    
                    if not np.isnan(future_performance) and abs(future_performance) < 0.5:  # Filter extreme values
                        training_sample = {
                            'features': features,
                            'performance': future_performance,
                            'timestamp': datetime.datetime.now() - datetime.timedelta(hours=lookback_period-i)
                        }
                        training_samples.append(training_sample)
                        
                except Exception as e:
                    continue  # Skip problematic samples
            
            logger.info(f"📊 [СТАРТ] Создано {len(training_samples)} обучающих образцов для {pair}")
            return training_samples
            
        except Exception as e:
            logger.error(f"❌ [СТАРТ] Ошибка генерации данных для {pair}: {str(e)}")
            return []

    def extract_features_at_index(self, dataframe: DataFrame, index: int) -> np.ndarray:
        """Extract ML features at a specific historical index"""
        try:
            features = []
            
            # Market volatility features
            atr_mean = dataframe['atr'].iloc[max(0, index-14):index].mean()
            features.append(atr_mean if not pd.isna(atr_mean) else 0.01)
            
            atr_std = dataframe['atr'].iloc[max(0, index-7):index].std()
            features.append(atr_std if not pd.isna(atr_std) else 0.001)
            
            # Price momentum features
            for period in [5, 10, 20]:
                pct_change = dataframe['close'].iloc[index] / dataframe['close'].iloc[max(0, index-period)] - 1
                features.append(pct_change if not pd.isna(pct_change) else 0.0)
            
            # Volume features (with defaults)
            if 'volume' in dataframe.columns:
                vol_mean = dataframe['volume'].iloc[max(0, index-14):index].mean()
                vol_pct = (dataframe['volume'].iloc[index] / dataframe['volume'].iloc[max(0, index-1)] - 1 
                          if index > 0 else 0.0)
            else:
                vol_mean, vol_pct = 1000.0, 0.0
            
            features.extend([vol_mean if not pd.isna(vol_mean) else 1000.0, 
                            vol_pct if not pd.isna(vol_pct) else 0.0])
            
            # Fisher transform features
            fisher_current = dataframe['fisher'].iloc[index]
            fisher_mean = dataframe['fisher'].iloc[max(0, index-5):index].mean()
            fisher_std = dataframe['fisher'].iloc[max(0, index-5):index].std()
            
            features.extend([
                fisher_current if not pd.isna(fisher_current) else 0.0,
                fisher_mean if not pd.isna(fisher_mean) else 0.0,
                fisher_std if not pd.isna(fisher_std) else 1.0
            ])
            
            # Baseline trend features
            baseline_diff_mean = dataframe['baseline_diff'].iloc[max(0, index-5):index].mean()
            baseline_diff_sum = dataframe['baseline_diff'].iloc[max(0, index-10):index].sum()
            
            features.extend([
                baseline_diff_mean if not pd.isna(baseline_diff_mean) else 0.0,
                baseline_diff_sum if not pd.isna(baseline_diff_sum) else 0.0
            ])
            
            # Market regime
            sma_50 = dataframe['close'].iloc[max(0, index-50):index].mean()
            sma_200 = dataframe['close'].iloc[max(0, index-200):index].mean()
            
            if not pd.isna(sma_50) and not pd.isna(sma_200) and sma_200 != 0:
                features.append(1.0 if sma_50 > sma_200 else 0.0)
            else:
                features.append(0.5)
            
            # Ensure exactly 12 features
            while len(features) < 12:
                features.append(0.0)
            features = features[:12]
            
            return np.array(features)
            
        except Exception as e:
            return np.zeros(12)

    def calculate_future_performance(self, dataframe: DataFrame, index: int, periods: int = 15) -> float:
        """Calculate future performance for training target"""
        try:
            if index + periods >= len(dataframe):
                return 0.0
            
            # Simple return calculation
            current_price = dataframe['close'].iloc[index]
            future_price = dataframe['close'].iloc[index + periods]
            
            if current_price > 0:
                return (future_price - current_price) / current_price
            else:
                return 0.0
                
        except:
            return 0.0

    def calculate_ml_confidence(self, dataframe: DataFrame) -> pd.Series:
        """Расчёт уровня уверенности ML в сигналах"""
        try:
            EnhancedLogger.log_subsection("Расчёт уверенности ML", "🎯")
            
            # Simple confidence calculation based on market volatility and trend consistency
            atr_norm = dataframe["atr"] / dataframe["close"]
            trend_consistency = abs(dataframe["baseline_diff"].rolling(10).mean())
            fisher_volatility = dataframe["fisher"].rolling(10).std()
            
            # Higher confidence in stable, trending markets
            confidence = 1.0 - (atr_norm * 2 + fisher_volatility * 0.5)
            confidence = confidence.fillna(0.5).clip(0.1, 1.0)
            
            # Log confidence statistics
            avg_confidence = confidence.tail(20).mean()
            min_confidence = confidence.tail(20).min()
            max_confidence = confidence.tail(20).max()
            
            EnhancedLogger.log_parameter("Средняя уверенность", f"{avg_confidence:.1%}", "🎯")
            EnhancedLogger.log_parameter("Минимальная уверенность", f"{min_confidence:.1%}", "🔽")
            EnhancedLogger.log_parameter("Максимальная уверенность", f"{max_confidence:.1%}", "🔼")
            
            return confidence
            
        except Exception as e:
            EnhancedLogger.log_error(f"Ошибка расчёта уверенности ML: {e}", "💥")
            return pd.Series(0.5, index=dataframe.index)
    
    def identify_market_regime(self, dataframe: DataFrame) -> pd.Series:
        """Определение рыночного режима с помощью ML признаков"""
        try:
            EnhancedLogger.log_subsection("Определение рыночного режима", "🏛️")
            
            sma_50 = dataframe["close"].rolling(50).mean()
            sma_200 = dataframe["close"].rolling(200).mean()
            
            # Market regimes: 1=Bull, 0=Neutral, -1=Bear
            regime = pd.Series(0, index=dataframe.index)
            regime.loc[sma_50 > sma_200 * 1.02] = 1  # Bull market
            regime.loc[sma_50 < sma_200 * 0.98] = -1  # Bear market
            
            # Calculate regime statistics
            recent_regime = regime.tail(50)
            bull_periods = (recent_regime == 1).sum()
            bear_periods = (recent_regime == -1).sum()
            neutral_periods = (recent_regime == 0).sum()
            
            EnhancedLogger.log_parameter("Бычьи периоды", f"{bull_periods}/50 ({bull_periods*2:.0f}%)", "🐂")
            EnhancedLogger.log_parameter("Медвежьи периоды", f"{bear_periods}/50 ({bear_periods*2:.0f}%)", "🐻") 
            EnhancedLogger.log_parameter("Нейтральные периоды", f"{neutral_periods}/50 ({neutral_periods*2:.0f}%)", "⚖️")
            
            current_regime = regime.iloc[-1]
            if current_regime > 0:
                EnhancedLogger.log_success("Текущий: БЫЧИЙ РЫНОК", "🐂")
            elif current_regime < 0:
                EnhancedLogger.log_warning("Текущий: МЕДВЕЖИЙ РЫНОК", "🐻")
            else:
                EnhancedLogger.log_subsection("Текущий: НЕЙТРАЛЬНЫЙ РЫНОК", "⚖️")
            
            return regime
            
        except Exception as e:
            EnhancedLogger.log_error(f"Ошибка определения рыночного режима: {e}", "💥")
            return pd.Series(0, index=dataframe.index)
    
    def calculate_signal_strength(self, dataframe: DataFrame) -> pd.Series:
        """Расчёт силы сигнала по нескольким индикаторам"""
        try:
            EnhancedLogger.log_subsection("Расчёт силы сигнала", "⚡")
            
            # Combine multiple signal components
            fisher_strength = abs(dataframe["fisher"]) / 3.0  # Normalize
            trend_strength = abs(dataframe["baseline_diff"]) / dataframe["atr"]
            volume_strength = 1.0  # Default if no volume data
            
            if 'volume' in dataframe.columns:
                volume_ma = dataframe['volume'].rolling(20).mean()
                volume_strength = (dataframe['volume'] / volume_ma).clip(0.5, 2.0) / 2.0
                EnhancedLogger.log_success("Признаки объёма включены", "📊")
            else:
                EnhancedLogger.log_warning("Данные объёма отсутствуют, используется дефолт", "📊")
            
            # Combined signal strength
            signal_strength = (fisher_strength * 0.4 + trend_strength * 0.4 + volume_strength * 0.2)
            signal_strength = signal_strength.fillna(0.5).clip(0.1, 1.0)
            
            # Log signal strength statistics
            avg_strength = signal_strength.tail(20).mean()
            current_strength = signal_strength.iloc[-1]
            strong_signals = (signal_strength.tail(50) > 0.7).sum()
            
            EnhancedLogger.log_parameter("Средняя сила сигнала", f"{avg_strength:.3f}", "⚡")
            EnhancedLogger.log_parameter("Текущая сила", f"{current_strength:.3f}", "📊")
            EnhancedLogger.log_parameter("Сильных сигналов (>0.7)", f"{strong_signals}/50", "💪")
            
            if current_strength > 0.8:
                EnhancedLogger.log_success("ОЧЕНЬ СИЛЬНЫЙ сигнал", "🚀")
            elif current_strength > 0.6:
                EnhancedLogger.log_success("СИЛЬНЫЙ сигнал", "💪")
            elif current_strength > 0.4:
                EnhancedLogger.log_warning("УМЕРЕННЫЙ сигнал", "⚡")
            else:
                EnhancedLogger.log_warning("СЛАБЫЙ сигнал", "🔋")
            
            return signal_strength
            
        except Exception as e:
            EnhancedLogger.log_error(f"Ошибка расчёта силы сигнала: {e}", "💥")
            return pd.Series(0.5, index=dataframe.index)
    
    def calculate_fisher(self, dataframe: DataFrame, period: int) -> pd.Series:
        """Расчёт Fisher Transform с ML улучшениями"""
        try:
            EnhancedLogger.log_subsection(f"Расчёт Fisher Transform (период={period})", "🎣")
            
            median_price = (dataframe["high"] + dataframe["low"]) / 2
            fisher = pd.Series(0.0, index=dataframe.index)
            
            for i in range(period, len(dataframe)):
                price_window = median_price.iloc[i-period:i]
                price_min = price_window.min()
                price_max = price_window.max()
                if price_max != price_min:
                    norm = (median_price.iloc[i] - price_min) / (price_max - price_min)
                    norm = 2 * norm - 1
                    norm = max(min(norm, 0.999), -0.999)
                    fisher.iloc[i] = 0.5 * np.log((1 + norm) / (1 - norm))
                else:
                    fisher.iloc[i] = 0.0
            
            # Log Fisher Transform statistics
            current_fisher = fisher.iloc[-1]
            avg_fisher = fisher.tail(50).mean()
            std_fisher = fisher.tail(50).std()
            
            EnhancedLogger.log_parameter("Текущий Fisher", f"{current_fisher:.3f}", "🎣")
            EnhancedLogger.log_parameter("Средний Fisher (50)", f"{avg_fisher:.3f}", "📊")
            EnhancedLogger.log_parameter("Волатильность Fisher", f"{std_fisher:.3f}", "🌊")
            
            if abs(current_fisher) > 2.0:
                EnhancedLogger.log_warning("Fisher в экстремальной зоне", "⚠️")
            elif abs(current_fisher) > 1.0:
                EnhancedLogger.log_success("Fisher показывает сильный сигнал", "💪")
            
            return fisher
            
        except Exception as e:
            EnhancedLogger.log_error(f"Ошибка расчёта Fisher Transform: {e}", "💥")
            return pd.Series(0.0, index=dataframe.index)
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Расчёт сигналов входа с ML интеграцией"""
        pair = metadata.get('pair', 'Unknown')
        
        EnhancedLogger.log_banner(f"АНАЛИЗ СИГНАЛОВ ВХОДА — {pair}", "🎯")
        
        # Get ML-adjusted parameters for this pair
        ml_params = self.get_ml_adjusted_params(dataframe, pair)
        
        # Use ML-adjusted thresholds
        fisher_buy_threshold = ml_params.get('fisher_buy_threshold', self.fisher_buy_threshold.value)
        fisher_sell_threshold = ml_params.get('fisher_sell_threshold', self.fisher_sell_threshold.value)
        
        EnhancedLogger.log_section("ПОРОГИ ВХОДА", "🎚️")
        EnhancedLogger.log_parameter("Порог покупки", fisher_buy_threshold, "🟢")
        EnhancedLogger.log_parameter("Порог продажи", fisher_sell_threshold, "🔴")
        
        # ML-enhanced entry logic
        ml_confidence_condition = dataframe["ml_confidence"] > self.ml_confidence_threshold.value
        signal_strength_condition = dataframe["ml_signal_strength"] > self.ml_signal_threshold.value
        
        # Count conditions for logging
        ml_conf_count = ml_confidence_condition.sum()
        signal_str_count = signal_strength_condition.sum()
        
        EnhancedLogger.log_section("ML УСЛОВИЯ", "🤖")
        EnhancedLogger.log_parameter("Периодов с высокой уверенностью", f"{ml_conf_count}/{len(dataframe)}", "🎯")
        EnhancedLogger.log_parameter("Периодов с сильным сигналом", f"{signal_str_count}/{len(dataframe)}", "⚡")
        
        # V1.2 BALANCED: Long entry with symmetric conditions
        long_conditions = (
            (dataframe["fisher"] < fisher_buy_threshold) &
            (dataframe["fisher"] > dataframe['fisher_smooth_long']) &
            (dataframe["fisher_trend_long"] > dataframe["fisher_trend_long"].shift(1)) &
            (dataframe["baseline_up"] | (dataframe["market_regime"] > 0)) &
            (ml_confidence_condition | signal_strength_condition)
        )
        
        dataframe.loc[long_conditions, ["enter_long", "enter_tag"]] = [1, "fisher_long_v12"]
        
        long_signals = long_conditions.sum()
        EnhancedLogger.log_parameter("Сигналов входа в лонг", long_signals, "🟢")
        
        # V1.2 BALANCED: Short entry with symmetric conditions
        if self.can_short:
            short_conditions = (
                (dataframe["fisher_smooth_short"] < fisher_sell_threshold) &
                (dataframe["baseline_down"]) &
                (dataframe["close"] >= dataframe["goldie_min"]) &
                (dataframe["close"] <= dataframe["goldie_max"]) &
                ml_confidence_condition &
                signal_strength_condition &
                (dataframe["market_regime"] <= 0)  # Neutral or bear market
            )
            dataframe.loc[short_conditions, ["enter_short", "enter_tag"]] = [1, "fisher_short_v12"]
            
            short_signals = short_conditions.sum()
            EnhancedLogger.log_parameter("Сигналов входа в шорт", short_signals, "🔴")
        else:
            EnhancedLogger.log_warning("Торговля в шорт отключена", "⚠️")
        
        # Log recent entry signals
        recent_long = dataframe["enter_long"].tail(20).sum()
        if self.can_short:
            recent_short = dataframe["enter_short"].tail(20).sum()
            EnhancedLogger.log_section("НЕДАВНИЕ СИГНАЛЫ (20 периодов)", "📊")
            EnhancedLogger.log_parameter("Входов в лонг", recent_long, "🟢")
            EnhancedLogger.log_parameter("Входов в шорт", recent_short, "🔴")
        else:
            EnhancedLogger.log_section("НЕДАВНИЕ СИГНАЛЫ (20 периодов)", "📊")
            EnhancedLogger.log_parameter("Входов в лонг", recent_long, "🟢")
        
        # Current market analysis
        current_fisher = dataframe["fisher"].iloc[-1]
        current_confidence = dataframe["ml_confidence"].iloc[-1]
        current_strength = dataframe["ml_signal_strength"].iloc[-1]
        current_regime = dataframe["market_regime"].iloc[-1]
        
        EnhancedLogger.log_section("ТЕКУЩЕЕ СОСТОЯНИЕ РЫНКА", "📈")
        EnhancedLogger.log_parameter("Значение Fisher", f"{current_fisher:.3f}", "🎣")
        EnhancedLogger.log_ml_status("ML анализ", current_confidence, "🤖")
        EnhancedLogger.log_parameter("Сила сигнала", f"{current_strength:.3f}", "⚡")
        
        regime_emoji = "🐂" if current_regime > 0 else "🐻" if current_regime < 0 else "⚖️"
        regime_text = "БЫЧИЙ" if current_regime > 0 else "МЕДВЕЖИЙ" if current_regime < 0 else "НЕЙТРАЛЬНЫЙ"
        EnhancedLogger.log_parameter("Рыночный режим", f"{regime_text} {regime_emoji}", "🏛️")
        
        EnhancedLogger.log_banner(f"АНАЛИЗ ВХОДОВ ЗАВЕРШЁН — {pair}", "✅")
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Расчёт сигналов выхода с ML интеграцией"""
        pair = metadata.get('pair', 'Unknown')
        
        EnhancedLogger.log_banner(f"АНАЛИЗ СИГНАЛОВ ВЫХОДА — {pair}", "🚪")
        
        # Get ML-adjusted parameters for this pair
        ml_params = self.get_ml_adjusted_params(dataframe, pair)
        
        # Use ML-adjusted exit thresholds
        fisher_long_exit = ml_params.get('fisher_long_exit', self.fisher_long_exit.value)
        fisher_short_exit = ml_params.get('fisher_short_exit', self.fisher_short_exit.value)
        
        EnhancedLogger.log_section("ПОРОГИ ВЫХОДА", "🎚️")
        exit_color_long = "🟢" if fisher_long_exit > 0 else "🔴" if fisher_long_exit < 0 else "🟡"
        exit_color_short = "🟢" if fisher_short_exit > 0 else "🔴" if fisher_short_exit < 0 else "🟡"
        
        EnhancedLogger.log_parameter(f"Выход из лонга {exit_color_long}", f"{fisher_long_exit:.3f}", "📤")
        if self.can_short:
            EnhancedLogger.log_parameter(f"Выход из шорта {exit_color_short}", f"{fisher_short_exit:.3f}", "📥")
        
        # ML-enhanced exit logic with confidence-based adjustments
        ml_confidence = dataframe["ml_confidence"]
        
        # Long exit with ML-optimized threshold
        long_exit_conditions = (
            (dataframe["fisher_smooth_long"].shift() > fisher_long_exit) & 
            (dataframe["fisher_smooth_long"] < fisher_long_exit) & 
            (dataframe["fisher_smooth_long"] > dataframe['fisher']) &
            (ml_confidence > 0.2)  # Only exit with reasonable confidence
        )
        
        dataframe.loc[long_exit_conditions, ["exit_long", "exit_tag"]] = [1, "exit_long_ml"]
        
        long_exits = long_exit_conditions.sum()
        EnhancedLogger.log_parameter("Сигналов выхода из лонга", long_exits, "📤")
        
        # Short exit with ML-optimized threshold (if enabled)
        if self.can_short:
            short_exit_conditions = (
                (dataframe["fisher_smooth_short"] > fisher_short_exit) &
                (ml_confidence > 0.2)
            )
            
            dataframe.loc[short_exit_conditions, ["exit_short", "exit_tag"]] = [1, "exit_short_ml"]
            
            short_exits = short_exit_conditions.sum()
            EnhancedLogger.log_parameter("Сигналов выхода из шорта", short_exits, "📥")
        
        # Log recent exit signals
        recent_long_exit = dataframe["exit_long"].tail(20).sum() if "exit_long" in dataframe.columns else 0
        
        EnhancedLogger.log_section("НЕДАВНИЕ ВЫХОДЫ (20 периодов)", "📊")
        EnhancedLogger.log_parameter("Выходов из лонга", recent_long_exit, "📤")
        
        if self.can_short:
            recent_short_exit = dataframe["exit_short"].tail(20).sum() if "exit_short" in dataframe.columns else 0
            EnhancedLogger.log_parameter("Выходов из шорта", recent_short_exit, "📥")
        
        # Current exit readiness analysis
        current_fisher_long = dataframe["fisher_smooth_long"].iloc[-1]
        current_confidence = dataframe["ml_confidence"].iloc[-1]
        
        EnhancedLogger.log_section("ТЕКУЩИЙ АНАЛИЗ ВЫХОДА", "🔍")
        
        long_distance_to_exit = current_fisher_long - fisher_long_exit
        EnhancedLogger.log_parameter("Расстояние до выхода лонг", f"{long_distance_to_exit:.3f}", "📏")
        
        if abs(long_distance_to_exit) < 0.1:
            EnhancedLogger.log_warning("Лонг близко к порогу выхода", "⚠️")
        elif long_distance_to_exit < 0:
            EnhancedLogger.log_success("Условия выхода из лонга выполнены", "✅")
        
        if self.can_short:
            current_fisher_short = dataframe["fisher_smooth_short"].iloc[-1]
            short_distance_to_exit = current_fisher_short - fisher_short_exit
            EnhancedLogger.log_parameter("Расстояние до выхода шорт", f"{short_distance_to_exit:.3f}", "📏")
            
            if abs(short_distance_to_exit) < 0.1:
                EnhancedLogger.log_warning("Шорт близко к порогу выхода", "⚠️")
            elif short_distance_to_exit > 0:
                EnhancedLogger.log_success("Условия выхода из шорта выполнены", "✅")
        
        EnhancedLogger.log_ml_status("Уверенность при выходе", current_confidence, "🎯")
        
        EnhancedLogger.log_banner(f"АНАЛИЗ ВЫХОДОВ ЗАВЕРШЁН — {pair}", "✅")
        
        return dataframe

# PART 5 (FINAL) - Continuing from Part 4

    def custom_exit(self, pair: str, trade: "Trade", current_time: "datetime", current_rate: float, current_profit: float, **kwargs):
        """Пользовательский выход с ML интеграцией"""
        tag = super().custom_sell(pair, trade, current_time, current_rate, current_profit, **kwargs)
        if tag:
            return tag
        
        EnhancedLogger.log_section(f"АНАЛИЗ ПОЛЬЗОВАТЕЛЬСКОГО ВЫХОДА — {pair}", "🚪")
        
        entry_tag = "empty"
        if hasattr(trade, "entry_tag") and trade.entry_tag is not None:
            entry_tag = trade.entry_tag
        
        EnhancedLogger.log_parameter("Тег входа", entry_tag, "🏷️")
        EnhancedLogger.log_parameter("Текущий PnL", f"{current_profit:.2%}", "💰")
        EnhancedLogger.log_parameter("Длительность сделки", str(current_time - trade.open_date_utc), "⏱️")
        
        # ML-enhanced stop loss with dynamic adjustment
        ml_adjusted_stop = -0.35
        current_ml_confidence = 0.5
        market_regime = 0
        
        try:
            # Get current dataframe for ML analysis
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                current_ml_confidence = dataframe["ml_confidence"].iloc[-1]
                market_regime = dataframe["market_regime"].iloc[-1]
                
                EnhancedLogger.log_ml_status("Текущее состояние ML", current_ml_confidence, "🤖")
                
                regime_text = "БЫЧИЙ 🐂" if market_regime > 0 else "МЕДВЕЖИЙ 🐻" if market_regime < 0 else "НЕЙТРАЛЬНЫЙ ⚖️"
                EnhancedLogger.log_parameter("Рыночный режим", regime_text, "🏛️")
                
                # Adjust stop loss based on ML confidence and market regime
                if current_ml_confidence < 0.5:
                    ml_adjusted_stop = -0.25  # Tighter stop in low confidence
                    EnhancedLogger.log_warning("Стоп ужесточён из-за низкой уверенности", "⚠️")
                elif market_regime < 0 and not trade.is_short:
                    ml_adjusted_stop = -0.3   # Tighter stop for longs in bear market
                    EnhancedLogger.log_warning("Стоп ужесточён для лонга в медвежьем рынке", "🐻")
                elif market_regime > 0 and trade.is_short:
                    ml_adjusted_stop = -0.3   # Tighter stop for shorts in bull market
                    EnhancedLogger.log_warning("Стоп ужесточён для шорта в бычьем рынке", "🐂")
                
                EnhancedLogger.log_parameter("ML скорректированный стоп", f"{ml_adjusted_stop:.1%}", "🛡️")
                
        except Exception as e:
            EnhancedLogger.log_error(f"Ошибка ML анализа: {e}", "❌")
        
        if current_profit <= ml_adjusted_stop:
            # Store trade performance for ML learning
            self.trade_performance_cache[trade.pair] = current_profit
            EnhancedLogger.log_warning(f"ML СТОП-ЛОСС СРАБОТАЛ", "🛑")
            EnhancedLogger.log_performance("Итоговый PnL", current_profit, "💸")
            return f"ml_stop_loss ({entry_tag})"
        
        EnhancedLogger.log_success("Условия выхода не выполнены", "✅")
        return None
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, 
                          time_in_force: str, exit_reason: str, current_time: datetime, **kwargs) -> bool:
        """Подтверждение выхода с ML обучением"""
        
        EnhancedLogger.log_banner(f"ПОДТВЕРЖДЕНИЕ ВЫХОДА — {pair}", "🔍")
        
        filled_buys = trade.select_filled_orders(trade.entry_side)
        count_of_buys = len(filled_buys)
        
        # Calculate profit for ML learning
        current_profit = trade.calc_profit_ratio(rate)
        
        EnhancedLogger.log_section("ДЕТАЛИ ВЫХОДА", "📊")
        EnhancedLogger.log_parameter("Причина выхода", exit_reason, "📝")
        EnhancedLogger.log_parameter("Тип ордера", order_type, "📋")
        EnhancedLogger.log_parameter("Объём выхода", f"{amount:.8f}", "💹")
        EnhancedLogger.log_parameter("Цена выхода", f"{rate:.8f}", "💱")
        EnhancedLogger.log_parameter("Ордеров на покупку", count_of_buys, "🔢")
        EnhancedLogger.log_performance("PnL при выходе", current_profit, "💰")
        
        # ML learning: store trade performance
        if exit_reason in ["roi", "stop_loss", "ml_stop_loss"]:
            self.trade_performance_cache[pair] = current_profit
            
            # Enhanced logging for ML learning
            performance_quality = "🟢 ПРИБЫЛЬ" if current_profit > 0.01 else "🟡 БЕЗУБЫТОК" if current_profit > -0.01 else "🔴 УБЫТОК"
            EnhancedLogger.log_parameter(f"Качество {performance_quality}", f"{current_profit:.2%}", "📈")
            EnhancedLogger.log_success("Эффективность сохранена для ML обучения", "🧠")
        
        # ✅ КРИТИЧЕСКИЙ ФИКС: Блокируем только маленькую ПОЛОЖИТЕЛЬНУЮ прибыль
        # Убытки и стоп-лоссы ВСЕГДА разрешены!
        if current_profit > 0 and current_profit < 0.005:
            EnhancedLogger.log_warning("Прибыль слишком мала, выход отклонён", "⚠️")
            return False
        
        if (count_of_buys == 1) & (exit_reason == "roi"):
            EnhancedLogger.log_warning("Одна покупка + ROI выход, отклонено", "⚠️")
            return False
        
        # Clean up stake tracking
        if trade.amount == amount and pair in self.cust_proposed_initial_stakes:
            del self.cust_proposed_initial_stakes[pair]
            EnhancedLogger.log_success("Отслеживание ставки очищено", "🧹")
        
        EnhancedLogger.log_success("Выход из сделки подтверждён", "✅")
        return True
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, 
                           proposed_stake: float, min_stake: float, max_stake: float, **kwargs) -> float:
        """ML-улучшенный расчёт размера позиции"""
        
        EnhancedLogger.log_section(f"РАСЧЁТ РАЗМЕРА ПОЗИЦИИ — {pair}", "💰")
        
        try:
            # Get market analysis for stake adjustment
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            ml_adjustment = 1.0
            
            EnhancedLogger.log_parameter("Предложенная ставка", f"{proposed_stake:.4f}", "💵")
            EnhancedLogger.log_parameter("Минимальная ставка", f"{min_stake:.4f}", "🔻")
            EnhancedLogger.log_parameter("Максимальная ставка", f"{max_stake:.4f}", "🔺")
            
            if not dataframe.empty:
                ml_confidence = dataframe["ml_confidence"].iloc[-1]
                signal_strength = dataframe["ml_signal_strength"].iloc[-1]
                
                EnhancedLogger.log_ml_status("Уверенность ML", ml_confidence, "🎯")
                EnhancedLogger.log_parameter("Сила сигнала", f"{signal_strength:.3f}", "⚡")
                
                # Adjust stake based on ML confidence
                confidence_multiplier = 0.5 + (ml_confidence * 0.5)  # 0.5 to 1.0
                signal_multiplier = 0.7 + (signal_strength * 0.3)    # 0.7 to 1.0
                
                ml_adjustment = confidence_multiplier * signal_multiplier
                
                EnhancedLogger.log_parameter("Множитель уверенности", f"{confidence_multiplier:.3f}", "🎯")
                EnhancedLogger.log_parameter("Множитель сигнала", f"{signal_multiplier:.3f}", "⚡")
                EnhancedLogger.log_parameter("Итоговая ML корректировка", f"{ml_adjustment:.3f}x", "🤖")
                
                if ml_adjustment > 1.0:
                    EnhancedLogger.log_success("Ставка увеличена из-за сильных ML сигналов", "📈")
                elif ml_adjustment < 0.8:
                    EnhancedLogger.log_warning("Ставка снижена из-за слабых ML сигналов", "📉")
                else:
                    EnhancedLogger.log_success("Стандартная ставка с умеренной ML корректировкой", "⚖️")
            else:
                EnhancedLogger.log_warning("Датафрейм недоступен, используется дефолтная корректировка", "⚠️")
            
            custom_stake = (proposed_stake / self.max_so_multiplier * self.overbuy_factor) * ml_adjustment
            custom_stake = max(min_stake, min(custom_stake, max_stake))  # Ensure within bounds
            
        except Exception as e:
            EnhancedLogger.log_error(f"Ошибка расчёта ставки: {e}", "💥")
            custom_stake = proposed_stake / self.max_so_multiplier * self.overbuy_factor
        
        EnhancedLogger.log_parameter("Итоговая ставка", f"{custom_stake:.4f}", "💎")
        
        stake_change_pct = ((custom_stake - proposed_stake) / proposed_stake) * 100
        change_emoji = "📈" if stake_change_pct > 0 else "📉" if stake_change_pct < 0 else "➡️"
        EnhancedLogger.log_parameter(f"Изменение ставки {change_emoji}", f"{stake_change_pct:+.1f}%", "📊")
        
        self.cust_proposed_initial_stakes[pair] = custom_stake
        return custom_stake
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float, 
                             current_profit: float, min_stake: float, max_stake: float, **kwargs) -> Optional[float]:
        """DCA с ML оценкой рисков"""
        
        if current_profit > self.initial_safety_order_trigger:
            return None
        
        EnhancedLogger.log_section(f"АНАЛИЗ DCA — {trade.pair}", "🔄")
        
        filled_buys = trade.select_filled_orders(trade.entry_side)
        count_of_buys = len(filled_buys)
        
        EnhancedLogger.log_parameter("Текущий PnL", f"{current_profit:.2%}", "📊")
        EnhancedLogger.log_parameter("Существующих ордеров покупки", count_of_buys, "🔢")
        EnhancedLogger.log_parameter("Макс. множитель SO", self.max_so_multiplier_orig, "🔢")
        
        if 1 <= count_of_buys <= self.max_so_multiplier_orig:
            # ML-enhanced safety order trigger
            ml_trigger_adjustment = 1.0
            
            try:
                dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                if not dataframe.empty:
                    ml_confidence = dataframe["ml_confidence"].iloc[-1]
                    market_regime = dataframe["market_regime"].iloc[-1]
                    
                    EnhancedLogger.log_ml_status("Уверенность ML", ml_confidence, "🤖")
                    
                    regime_text = "БЫЧИЙ 🐂" if market_regime > 0 else "МЕДВЕЖИЙ 🐻" if market_regime < 0 else "НЕЙТРАЛЬНЫЙ ⚖️"
                    EnhancedLogger.log_parameter("Рыночный режим", regime_text, "🏛️")
                    
                    # Adjust safety order trigger based on ML analysis
                    if ml_confidence < 0.5:
                        ml_trigger_adjustment = 1.5  # More conservative in low confidence
                        EnhancedLogger.log_warning("Консервативный DCA из-за низкой уверенности", "⚠️")
                    elif market_regime < 0 and not trade.is_short:
                        ml_trigger_adjustment = 1.3  # More conservative for longs in bear market
                        EnhancedLogger.log_warning("Консервативный DCA для лонга в медвежьем рынке", "🐻")
                    else:
                        EnhancedLogger.log_success("Стандартный DCA триггер", "✅")
                    
                    EnhancedLogger.log_parameter("ML корректировка триггера", f"{ml_trigger_adjustment:.1f}x", "🎯")
                        
            except Exception as e:
                EnhancedLogger.log_error(f"Ошибка ML анализа: {e}", "❌")
            
            safety_order_trigger = abs(self.initial_safety_order_trigger) * count_of_buys * ml_trigger_adjustment
            
            if self.safety_order_step_scale > 1:
                safety_order_trigger = abs(self.initial_safety_order_trigger) * ml_trigger_adjustment + (
                    abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * 
                    (math.pow(self.safety_order_step_scale, (count_of_buys - 1)) - 1) / 
                    (self.safety_order_step_scale - 1)
                )
            elif self.safety_order_step_scale < 1:
                safety_order_trigger = abs(self.initial_safety_order_trigger) * ml_trigger_adjustment + (
                    abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * 
                    (1 - math.pow(self.safety_order_step_scale, (count_of_buys - 1))) / 
                    (1 - self.safety_order_step_scale)
                )
            
            EnhancedLogger.log_parameter("Триггер безопасного ордера", f"{safety_order_trigger:.2%}", "🎯")
            
            if current_profit <= (-1 * abs(safety_order_trigger)):
                EnhancedLogger.log_success("DCA триггер активирован!", "🚀")
                
                try:
                    actual_initial_stake = filled_buys[0].cost
                    stake_amount = actual_initial_stake
                    already_bought = sum(filled_buy.cost for filled_buy in filled_buys)
                    
                    EnhancedLogger.log_parameter("Начальная ставка", f"{actual_initial_stake:.4f}", "💰")
                    EnhancedLogger.log_parameter("Уже вложено", f"{already_bought:.4f}", "💸")
                    
                    if trade.pair in self.cust_proposed_initial_stakes:
                        if self.cust_proposed_initial_stakes[trade.pair] > 0:
                            proposed_initial_stake = self.cust_proposed_initial_stakes[trade.pair]
                            current_actual_stake = already_bought * math.pow(self.safety_order_volume_scale, (count_of_buys - 1))
                            current_stake_preposition = proposed_initial_stake * math.pow(self.safety_order_volume_scale, (count_of_buys - 1))
                            current_stake_preposition_compensation = (
                                current_stake_preposition + abs(current_stake_preposition - current_actual_stake)
                            )
                            total_so_stake = lerp(current_actual_stake, current_stake_preposition_compensation, 
                                                self.partial_fill_compensation_scale)
                            stake_amount = total_so_stake
                            
                            EnhancedLogger.log_parameter("Компенсированная ставка", f"{stake_amount:.4f}", "🎯")
                        else:
                            stake_amount = stake_amount * math.pow(self.safety_order_volume_scale, (count_of_buys - 1))
                            EnhancedLogger.log_parameter("Масштабированная ставка", f"{stake_amount:.4f}", "📈")
                    else:
                        stake_amount = stake_amount * math.pow(self.safety_order_volume_scale, (count_of_buys - 1))
                        EnhancedLogger.log_parameter("Ставка по умолчанию масштабирована", f"{stake_amount:.4f}", "📊")
                    
                    EnhancedLogger.log_success(f"DCA ордер #{count_of_buys + 1} одобрен", "✅")
                    return stake_amount
                    
                except Exception as e:
                    EnhancedLogger.log_error(f"Ошибка расчёта DCA: {e}", "💥")
                    return None
            else:
                distance_to_trigger = abs(current_profit) - abs(safety_order_trigger)
                EnhancedLogger.log_parameter("Расстояние до DCA", f"{distance_to_trigger:.2%}", "📏")
                EnhancedLogger.log_warning("DCA триггер ещё не достигнут", "⏳")
        else:
            if count_of_buys > self.max_so_multiplier_orig:
                EnhancedLogger.log_warning("Достигнут максимум DCA ордеров", "🛑")
            else:
                EnhancedLogger.log_warning("Нет существующих ордеров для DCA", "❌")
        
        return None
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, 
                       current_profit: float, **kwargs) -> float:
        """ML-улучшенный пользовательский стоп-лосс и тейк-профит"""
        
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            trade_candle = dataframe.loc[dataframe['date'] == trade_date]
            
            if not trade_candle.empty:
                trade_candle = trade_candle.squeeze()
                
                EnhancedLogger.log_section(f"ПОЛЬЗОВАТЕЛЬСКИЙ СТОП — {pair}", "🛡️")
                
                # Get ML indicators for dynamic adjustment
                try:
                    current_ml_confidence = dataframe["ml_confidence"].iloc[-1]
                    market_regime = dataframe["market_regime"].iloc[-1]
                    
                    # Adjust multipliers based on ML analysis
                    sl_multiplier = self.ATR_SL_long_Multip.value if not trade.is_short else self.ATR_SL_short_Multip.value
                    tp_multiplier = self.rr_long.value if not trade.is_short else self.rr_short.value
                    
                    EnhancedLogger.log_parameter("Базовый множитель SL", f"{sl_multiplier:.1f}x", "🛡️")
                    EnhancedLogger.log_parameter("Базовый множитель TP", f"{tp_multiplier:.1f}x", "🎯")
                    EnhancedLogger.log_ml_status("Уверенность ML", current_ml_confidence, "🤖")
                    
                    # Dynamic adjustment based on ML confidence
                    if current_ml_confidence < 0.5:
                        sl_multiplier *= 0.8  # Tighter stop loss
                        tp_multiplier *= 0.9  # Closer take profit
                        EnhancedLogger.log_warning("SL/TP ужесточены из-за низкой уверенности", "⚠️")
                    elif current_ml_confidence > 0.8:
                        sl_multiplier *= 1.2  # Wider stop loss
                        tp_multiplier *= 1.1  # Further take profit
                        EnhancedLogger.log_success("SL/TP расширены из-за высокой уверенности", "✨")
                    
                    # Market regime adjustment
                    if not trade.is_short and market_regime < 0:  # Long in bear market
                        sl_multiplier *= 0.9
                        EnhancedLogger.log_warning("SL ужесточён для лонга в медвежьем рынке", "🐻")
                    elif trade.is_short and market_regime > 0:    # Short in bull market
                        sl_multiplier *= 0.9
                        EnhancedLogger.log_warning("SL ужесточён для шорта в бычьем рынке", "🐂")
                    
                    EnhancedLogger.log_parameter("Скорректированный множитель SL", f"{sl_multiplier:.2f}x", "🎯")
                    EnhancedLogger.log_parameter("Скорректированный множитель TP", f"{tp_multiplier:.2f}x", "🎯")
                        
                except Exception as e:
                    EnhancedLogger.log_error(f"Ошибка ML корректировки: {e}", "❌")
                    sl_multiplier = self.ATR_SL_long_Multip.value if not trade.is_short else self.ATR_SL_short_Multip.value
                    tp_multiplier = self.rr_long.value if not trade.is_short else self.rr_short.value
                
                # Stop Loss Logic
                atr_value = trade_candle['atr']
                sl_distance = atr_value * sl_multiplier
                
                if not trade.is_short:
                    sl_price = trade.open_rate - sl_distance
                    sl_condition = current_rate < sl_price
                    side_text = "ЛОНГ"
                else:
                    sl_price = trade.open_rate + sl_distance
                    sl_condition = current_rate > sl_price
                    side_text = "ШОРТ"
                
                EnhancedLogger.log_parameter(f"{side_text} Уровень SL", f"{sl_price:.6f}", "🛑")
                EnhancedLogger.log_parameter("Текущая цена", f"{current_rate:.6f}", "💱")
                EnhancedLogger.log_parameter("Расстояние до SL", f"{sl_distance:.6f}", "📏")
                
                if sl_condition:
                    self.trade_performance_cache[pair] = current_profit  # Store for ML learning
                    EnhancedLogger.log_warning("СТОП-ЛОСС СРАБОТАЛ!", "🛑")
                    EnhancedLogger.log_performance("Итоговый убыток", current_profit, "💸")
                    return -0.01
                
                # Take Profit Logic
                dist = trade_candle['atr'] * self.ATR_Multip.value
                tp_distance = dist * tp_multiplier
                
                if not trade.is_short:
                    tp_price = trade.open_rate + tp_distance
                    tp_condition = current_rate > tp_price
                else:
                    tp_price = trade.open_rate - tp_distance
                    tp_condition = current_rate < tp_price
                
                EnhancedLogger.log_parameter(f"{side_text} Уровень TP", f"{tp_price:.6f}", "🎯")
                EnhancedLogger.log_parameter("Расстояние до TP", f"{tp_distance:.6f}", "📏")
                
                if tp_condition:
                    self.trade_performance_cache[pair] = current_profit  # Store for ML learning
                    EnhancedLogger.log_success("ТЕЙК-ПРОФИТ ДОСТИГНУТ!", "🎯")
                    EnhancedLogger.log_performance("Итоговая прибыль", current_profit, "💰")
                    return -0.0001
                
                # Log current distances
                if not trade.is_short:
                    sl_distance_current = current_rate - sl_price
                    tp_distance_current = tp_price - current_rate
                else:
                    sl_distance_current = sl_price - current_rate
                    tp_distance_current = current_rate - tp_price
                
                EnhancedLogger.log_parameter("Расстояние до SL", f"{sl_distance_current:.6f}", "📏")
                EnhancedLogger.log_parameter("Расстояние до TP", f"{tp_distance_current:.6f}", "📏")
        
        except Exception as e:
            EnhancedLogger.log_error(f"Ошибка расчёта пользовательского стопа: {e}", "💥")
        
        return None
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, 
                max_leverage: float, side: str, **kwargs) -> float:
        """ML-улучшенное управление плечом"""
        
        EnhancedLogger.log_section(f"РАСЧЁТ ПЛЕЧА — {pair}", "⚖️")
        
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            base_leverage = self.set_leverage if self.can_short else 1
            
            EnhancedLogger.log_parameter("Предложенное плечо", f"{proposed_leverage:.1f}x", "📊")
            EnhancedLogger.log_parameter("Максимальное плечо", f"{max_leverage:.1f}x", "🔺")
            EnhancedLogger.log_parameter("Базовое плечо", f"{base_leverage:.1f}x", "⚙️")
            EnhancedLogger.log_parameter("Сторона сделки", side.upper(), "↔️")
            
            if not dataframe.empty:
                ml_confidence = dataframe["ml_confidence"].iloc[-1]
                market_volatility = dataframe["atr"].iloc[-1] / dataframe["close"].iloc[-1]
                
                EnhancedLogger.log_ml_status("Уверенность ML", ml_confidence, "🤖")
                EnhancedLogger.log_parameter("Волатильность рынка", f"{market_volatility:.1%}", "🌊")
                
                # Reduce leverage in high volatility or low confidence conditions
                if ml_confidence < 0.6 or market_volatility > 0.05:
                    adjusted_leverage = base_leverage * 0.8
                    reason = "низкая уверенность" if ml_confidence < 0.6 else "высокая волатильность"
                    EnhancedLogger.log_warning(f"Плечо снижено: {reason}", "⚠️")
                elif ml_confidence > 0.8 and market_volatility < 0.02:
                    adjusted_leverage = min(base_leverage * 1.1, max_leverage)
                    EnhancedLogger.log_success("Плечо увеличено — благоприятные условия", "📈")
                else:
                    adjusted_leverage = base_leverage
                    EnhancedLogger.log_success("Применено стандартное плечо", "✅")
                
                final_leverage = max(1, min(adjusted_leverage, max_leverage))
                
                leverage_change = final_leverage - base_leverage
                change_emoji = "📈" if leverage_change > 0 else "📉" if leverage_change < 0 else "➡️"
                
                EnhancedLogger.log_parameter("Итоговое плечо", f"{final_leverage:.1f}x", "🎯")
                EnhancedLogger.log_parameter(f"Изменение плеча {change_emoji}", f"{leverage_change:+.1f}x", "📊")
                
                return final_leverage
                
        except Exception as e:
            EnhancedLogger.log_error(f"Ошибка расчёта плеча: {e}", "💥")
        
        default_leverage = self.set_leverage if self.can_short else 1
        EnhancedLogger.log_parameter("Применено плечо по умолчанию", f"{default_leverage:.1f}x", "🔄")
        return default_leverage
