import logging
import numpy as np
import pmdarima as pm
from typing import Dict
from pmdarima import pipeline, preprocessing as ppc, arima

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoARIMAModelWithRetrainFreqAI:
    """
    An AutoARIMA model that fits FreqAI's data dictionary format and supports retraining with new data.
    """
    
    def __init__(self, **pipeline_args):
        """
        Initializes the model pipeline with preprocessing steps and AutoARIMA.
        
        :param pipeline_args: Arguments for Pipeline components.
        """
        self.pipe = pipeline.Pipeline([
            ("fourier", ppc.FourierFeaturizer(m=12, k=pipeline_args.get('k', 3))),
            ("arima", arima.AutoARIMA(stepwise=pipeline_args.get('stepwise', True),
                                      trace=pipeline_args.get('trace', True),
                                      error_action=pipeline_args.get('error_action', 'ignore'),
                                      suppress_warnings=pipeline_args.get('suppress_warnings', True),
                                      seasonal=pipeline_args.get('seasonal', True),
                                      m=pipeline_args.get('m', 12)))
        ])
        self.last_train_idx = 0
    
    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        Fits an ARIMA model based on the provided training data and updates metrics in FreqaiDataKitchen.
        
        :param data_dictionary: Dictionary containing 'train_labels' for the time series data.
        :param dk: FreqaiDataKitchen object for context and metrics tracking.
        """
        # ARIMA directly uses the time series values (y) to fit the model, no X (features) needed
        y = data_dictionary["train_labels"]
        
        # Initialize and configure ARIMA model pipeline if needed
        self.model = pipeline.Pipeline([
            ("fourier", ppc.FourierFeaturizer(m=12, k=self.arima_args.get('k', 3))),  # Optional preprocessing
            ("arima", arima.AutoARIMA(**self.arima_args))
        ])

        start = time.time()
        self.model.fit(y)
        time_spent = time.time() - start
        
        # Update training fit time metric
        dk.update_metric_tracker('fit_time', time_spent, dk.pair)

        logger.info(f"ARIMA Model fitted in {time_spent:.2f} seconds.")
        return self.model

        
    def predict(self, data_dictionary: Dict[str, np.ndarray], n_periods: int) -> np.ndarray:
        """
        Forecast future values based on the FreqAI data dictionary format.
        
        :param data_dictionary: FreqAI data dictionary, used if needed for context.
        :param n_periods: Number of periods to forecast.
        :return: Array of forecasts.
        """
        return self.pipe.predict(n_periods=n_periods)

    def fit_live_predictions(self, dk, pair: str) -> None:
        """
        Update and possibly retrain the model based on live data available 
        through the FreqaiDataKitchen for a specific pair.
        
        :param dk: The FreqaiDataKitchen instance containing the data.
        :param pair: The trading pair (e.g., "BTC/USD") to predict for.
        """
        
        warmed_up = True
        num_candles = self.freqai_info.get('fit_live_predictions_candles', 100)
        weibull_outlier_thresh = self.freqai_info.get('weibull_outlier_threshold', 0.99)
        
        if not hasattr(self, 'exchange_candles'):
            self.exchange_candles = len(dk.model_return_values[pair].index)
        
        candle_diff = len(dk.historic_predictions[pair].index) - (num_candles + self.exchange_candles)
        
        if candle_diff < 0:
            logger.warning(f'Fit live predictions not warmed up yet. Still {abs(candle_diff)} candles to go')
            warmed_up = False

        if warmed_up:
            # Assuming dk.provide_data() is a method that retrieves the latest complete dataset for training, including new observations
            data_dictionary = dk.provide_data(pair)
            self.fit(data_dictionary)
            
            # Process predictions for setting thresholds or altering model parameters based on predictions
            pred_df_full = dk.historic_predictions[pair].tail(num_candles).reset_index(drop=True)
            # Continue processing according to the provided example...
            # E.g., Sorting predictions, updating according to extrema, setting up new cut-offs based on Weibull distribution

            # Example of updating dk with the new calculated thresholds
            dk.data['extra_returns_per_train'][f'{pair}-maxima_sort_threshold'] = 2  # Sample update, adapt based on actual logic needed

            # Example of fitting Weibull distribution and calculating cutoff
            if 'DI_values' in pred_df_full:
                f = spy.stats.weibull_min.fit(pred_df_full['DI_values'])
                cutoff = spy.stats.weibull_min.ppf(weibull_outlier_thresh, *f)
                dk.data['extra_returns_per_train'][f'{pair}-DI_cutoff'] = cutoff  # Updating DI cutoff for the pair based on the newly observed data
        else:
            logger.info('Waiting for more data to warm up fit_live_predictions.')
