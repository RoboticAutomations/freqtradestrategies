import logging
from typing import Any, Dict
from pmdarima import AutoARIMA
from time import time  # Corrected import
from freqtrade.freqai.base_models.BaseAutoARIMAModel import BaseAutoARIMAModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

class ARIMA(BaseAutoARIMAModel):
    """
    User-created prediction model using AutoARIMA.
    """

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        Set up the training data to fit the AutoARIMA model.
        :param data_dictionary: Dictionary holding all data for train, test, labels, weights.
        :param dk: The datakitchen object for the current coin/model.
        """
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        sample_weight = data_dictionary.get("train_weights")

        model = AutoARIMA(**self.model_training_parameters)

        start = time()
        model.fit(X, y, sample_weight=sample_weight,
                suppress_warnings=False, seasonal=True, m=52)  # m=52 for weekly data
        time_spent = time() - start
        self.dd.update_metric_tracker('fit_time', time_spent, dk.pair)

        return model
