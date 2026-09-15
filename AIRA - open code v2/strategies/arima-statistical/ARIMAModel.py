import logging
from typing import Any, Dict
import pandas as pd
from pandas import DataFrame
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from numpy.typing import NDArray
from typing import Tuple
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import numpy.typing as npt

logger = logging.getLogger(__name__)

class ARIMAModel(BaseRegressionModel):
    """
    User-created prediction model using ARIMA for time series forecasting.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = None 
    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        Fit an ARIMA model to the training data.
        :param data_dictionary: The dictionary holding all data for train, test, labels, and weights.
        :param dk: The data kitchen object for the current coin/model.
        :param kwargs: Additional keyword arguments.
        :return: Trained ARIMA model.
        """


    
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]


        # Fit an ARIMA model (you can adjust the order and other parameters)
        config = self.freqai_info.get("model_training_parameters", {})
        p,d,q = config.get("p",1),config.get("d",0),config.get("q",1)
        basemodel = ARIMA(endog=y,exog=X, order=(p, d, q))
        model = basemodel.fit()
        self.model = model
        return model
    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        Filter the prediction features data and predict with it.
        :param unfiltered_df: Full dataframe for the current backtest period.
        :return:
        :pred_df: dataframe containing the predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """

        dk.find_features(unfiltered_df)
        dk.data_dictionary["prediction_features"], _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )

        dk.data_dictionary["prediction_features"], outliers, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True)
        exog=dk.data_dictionary["prediction_features"]
        
        predictions = self.model.forecast(steps=len(exog),exog =exog)
        if self.CONV_WIDTH == 1:
            predictions = np.reshape(predictions, (-1, len(dk.label_list)))
        pred_df = DataFrame(predictions, columns=dk.label_list)

        pred_df, _, _ = dk.label_pipeline.inverse_transform(pred_df)
        if dk.feature_pipeline["di"]:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers

        return (pred_df, dk.do_predict)