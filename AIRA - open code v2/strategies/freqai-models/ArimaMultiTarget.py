# ARIMAMultiTarget
import logging
from typing import Any, Dict

from pmdarima import AutoARIMA
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.base_models.FreqaiMultiOutputRegressor import FreqaiMultiOutputRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

logger = logging.getLogger(__name__)

class ARIMAMultiTarget(BaseRegressionModel):

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        arima = AutoARIMA(**self.model_training_parameters)

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        sample_weight = data_dictionary["train_weights"]

        eval_weights = None
        eval_sets = [None] * y.shape[1]

        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            eval_weights = [data_dictionary["test_weights"]]
            eval_sets = [
                (
                    data_dictionary["test_features"],
                    data_dictionary["test_labels"].iloc[:, i]
                )
                for i in range(data_dictionary['test_labels'].shape[1])
            ]

        init_model = self.get_init_model(dk.pair)
        if init_model:
            init_models = init_model.estimators_
        else:
            init_models = [None] * y.shape[1]

        fit_params = []
        for i in range(len(eval_sets)):
            fit_params.append(
                {'exogenous': eval_sets[i], 'sample_weight': eval_weights,
                'init_model': init_models[i]}
            )

        model = FreqaiMultiOutputRegressor(estimator=arima)
        thread_training = self.freqai_info.get('multitarget_parallel_training', False)
        if thread_training:
            model.n_jobs = y.shape[1]
        model.fit(X=X, y=y, sample_weight=sample_weight, fit_params=fit_params)

        return model