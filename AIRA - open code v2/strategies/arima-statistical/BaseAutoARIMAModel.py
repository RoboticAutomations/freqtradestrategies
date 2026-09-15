import logging
from time import time
from typing import Any, Tuple

import numpy as np
from pandas import DataFrame
from pmdarima import AutoARIMA

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel


logger = logging.getLogger(__name__)


class BaseAutoARIMAModel(IFreqaiModel):
    """
    ARIMA model for time series forecasting.
    """

    def __init__(self, model_training_parameters: dict):
        self.model_training_parameters = model_training_parameters
        self.model = None  # This will be set to the trained AutoARIMA model

    def train(
        self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        logger.info(f"-------------------- Starting training {pair} --------------------")

        start_time = time()

        # For ARIMA, we typically only need the target variable (univariate time series)
        _, labels_filtered = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            dk.label_list,
            training_filter=True,
        )

        start_date = unfiltered_df["date"].iloc[0].strftime("%Y-%m-%d")
        end_date = unfiltered_df["date"].iloc[-1].strftime("%Y-%m-%d")
        logger.info(f"-------------------- Training on data from {start_date} to {end_date} --------------------")

        # split data into train/test data.
        dd = dk.make_train_test_datasets(None, labels_filtered)  # No features for ARIMA, only labels

        # ARIMA doesn't use feature pipelines, only the target variable (label)
        dd["train_labels"], _, _ = dk.label_pipeline.fit_transform(dd["train_labels"])

        # Ensure that train_labels is a 1D array for univariate forecasting
        train_labels = dd["train_labels"].squeeze()

        logger.info(f"Training ARIMA model on {len(train_labels)} data points")

        # Initialize and fit the AutoARIMA model
        self.model = AutoARIMA(**self.model_training_parameters)
        self.model.fit(train_labels)

        end_time = time()

        logger.info(f"-------------------- Done training {pair} ({end_time - start_time:.2f} secs) --------------------")

        return self.model

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[DataFrame, np.ndarray]:
        # For ARIMA, we don't need to filter features, as it uses past values of the time series to predict future values

        # Number of periods to forecast is based on the length of the dataframe
        n_periods = len(unfiltered_df)

        # Make predictions using the trained AutoARIMA model
        predictions = self.model.predict(n_periods=n_periods)

        # Convert predictions to DataFrame
        pred_df = DataFrame(predictions, columns=[dk.label_list[0]])  # Assuming single target variable

        # No need for inverse transformation of predictions for ARIMA
        # Placeholder for outlier detection logic (if applicable)
        outliers = np.zeros(n_periods, dtype=bool)

        # Set dk.do_predict to the outliers array (or other logic as needed)
        dk.do_predict = outliers

        return pred_df, dk.do_predict

