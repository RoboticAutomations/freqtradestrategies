import logging
from typing import Any, Dict

from xgboost import XGBRegressor
import time
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import pandas as pd
import scipy as spy
import numpy as np


logger = logging.getLogger(__name__)


class XGBoostRegressorQuickAdapterV3_1(BaseRegressionModel):
    """
    The following freqaimodel is released to sponsors of the non-profit FreqAI open-source project.
    If you find the FreqAI project useful, please consider supporting it by becoming a sponsor.
    We use sponsor money to help stimulate new features and to pay for running these public
    experiments, with a an objective of helping the community make smarter choices in their
    ML journey.

    This strategy is experimental (as with all strategies released to sponsors). Do *not* expect
    returns. The goal is to demonstrate gratitude to people who support the project and to
    help them find a good starting point for their own creativity.

    If you have questions, please direct them to our discord: https://discord.gg/xE4RMg4QYw

    https://github.com/sponsors/robcaulk
    """

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        """

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]

        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            eval_set = None
            eval_weights = None
        else:
            eval_set = [(data_dictionary["test_features"], data_dictionary["test_labels"])]
            eval_weights = [data_dictionary['test_weights']]

        sample_weight = data_dictionary["train_weights"]

        xgb_model = self.get_init_model(dk.pair)

        model = XGBRegressor(**self.model_training_parameters)

        start = time.time()
        model.fit(X=X, y=y, sample_weight=sample_weight, eval_set=eval_set,
                  sample_weight_eval_set=eval_weights, xgb_model=xgb_model)
        time_spent = (time.time() - start)
        self.dd.update_metric_tracker('fit_time', time_spent, dk.pair)

        return model

    def fit_live_predictions(self, dk: FreqaiDataKitchen, pair: str) -> None:

        warmed_up = True

        num_candles = self.freqai_info.get('fit_live_predictions_candles', 100)
        if self.live:
            if not hasattr(self, 'exchange_candles'):
                self.exchange_candles = len(self.dd.model_return_values[pair].index)
            candle_diff = len(self.dd.historic_predictions[pair].index) - \
                (num_candles + self.exchange_candles)
            if candle_diff < 0:
                logger.warning(
                    f'Fit live predictions not warmed up yet. Still {abs(candle_diff)} candles to go')
                warmed_up = False

        pred_df_full = self.dd.historic_predictions[pair].tail(num_candles).reset_index(drop=True)
        pred_df_sorted = pd.DataFrame()
        for label in pred_df_full.keys():
            if pred_df_full[label].dtype == object:
                continue
            pred_df_sorted[label] = pred_df_full[label]

        # pred_df_sorted = pred_df_sorted
        for col in pred_df_sorted:
            pred_df_sorted[col] = pred_df_sorted[col].sort_values(
                ascending=False, ignore_index=True)
        frequency = num_candles / (self.freqai_info['feature_parameters']['label_period_candles'] * 2)
        max_pred = pred_df_sorted.iloc[:int(frequency)].mean()
        min_pred = pred_df_sorted.iloc[-int(frequency):].mean()

        if not warmed_up:
            dk.data['extra_returns_per_train']['&s-maxima_sort_threshold'] = 2
            dk.data['extra_returns_per_train']['&s-minima_sort_threshold'] = -2
        else:
            dk.data['extra_returns_per_train']['&s-maxima_sort_threshold'] = max_pred['&s-extrema']
            dk.data['extra_returns_per_train']['&s-minima_sort_threshold'] = min_pred['&s-extrema']

        dk.data["labels_mean"], dk.data["labels_std"] = {}, {}
        for ft in dk.label_list:
            # f = spy.stats.norm.fit(pred_df_full[ft])
            dk.data['labels_std'][ft] = 0  # f[1]
            dk.data['labels_mean'][ft] = 0  # f[0]

        # Handle DI_values: clean the data and fit the distribution
        try:
            # Convert DI_values to numeric, coercing errors to NaN
            di_values = pd.to_numeric(pred_df_full['DI_values'], errors='coerce')
            
            # Remove any NaN, inf, or -inf values
            di_values = di_values[np.isfinite(di_values)]
            
            # If we have valid values after cleaning, fit the distribution
            if len(di_values) > 0:
                if not warmed_up:
                    f = [0, 0, 0]
                    cutoff = 2
                else:
                    # Ensure all values are positive for Weibull distribution
                    min_val = di_values.min()
                    if min_val <= 0:
                        di_values = di_values - min_val + 1e-6  # Small positive offset
                    
                    try:
                        f = spy.stats.weibull_min.fit(di_values)
                        cutoff = spy.stats.weibull_min.ppf(0.999, *f)
                    except Exception as e:
                        logger.warning(f"Failed to fit Weibull distribution: {str(e)}. Using default values.")
                        f = [1, 0, 1]  # Default shape, loc, scale parameters
                        cutoff = 2
            else:
                logger.warning("No valid DI_values found after cleaning. Using default values.")
                f = [1, 0, 1]
                cutoff = 2
        except Exception as e:
            logger.warning(f"Error processing DI_values: {str(e)}. Using default values.")
            f = [1, 0, 1]
            cutoff = 2

        # Store the results
        dk.data["DI_value_mean"] = float(di_values.mean()) if len(di_values) > 0 else 0
        dk.data["DI_value_std"] = float(di_values.std()) if len(di_values) > 0 else 1
        dk.data['extra_returns_per_train']['DI_value_param1'] = float(f[0])
        dk.data['extra_returns_per_train']['DI_value_param2'] = float(f[1])
        dk.data['extra_returns_per_train']['DI_value_param3'] = float(f[2])
        dk.data['extra_returns_per_train']['DI_cutoff'] = float(cutoff)
