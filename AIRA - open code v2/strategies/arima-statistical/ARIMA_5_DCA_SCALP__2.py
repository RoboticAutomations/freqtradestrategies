# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce
from pandas import DataFrame
import warnings
import pandas as pd
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, informative
from freqtrade.strategy import DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter
import technical.indicators as ftt
import math
import logging
from scipy.signal import find_peaks, find_peaks_cwt
import warnings
from math import ceil
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pmdarima import auto_arima
from pmdarima import model_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.set_option('display.float_format', lambda x: '%.7f' % x)
logger = logging.getLogger(__name__)

class ARIMA_5_DCA_SCALP(IStrategy):
    # Stoploss:
    stoploss = -0.20
    # Trailing stop:
    use_custom_stoploss = True
    # Initialize dicts for arima storage
    forecast_period = 5
    maxiter =  50
    forecasting_models = {}
    context_length = 10
    previous_predicted_close = {}
    # Sell signal
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False
    ## Optional order time in force.
    order_time_in_force = {'entry': 'gtc', 'exit': 'gtc'}

    # Optimal timeframe for the strategy
    timeframe = '5m'
    startup_candle_count = 400
    process_only_new_candles = True

    # DCA
    max_entry_position_adjustment = 2  # Number of ADDITIONAL entries
    max_dca_multiplier = DecimalParameter(1.0, 4.0, default=2.5, decimals=1, space='buy', optimize=True) # allocation divider
    position_adjustment_enable = True

    # Custom Entry
    last_entry_price = None

    # Hyper-opt parameters
    base_nb_candles_buy = IntParameter(150, 200, default=184, space='buy', optimize=True, load=True)
    up = DecimalParameter(low=1.02, high=1.025, default=1.02, decimals=3, space='buy', optimize=True, load=True)
    dn = DecimalParameter(low=0.983, high=0.987, default=0.984, decimals=3, space='buy', optimize=True, load=True)
    increment = DecimalParameter(low=1.0005, high=1.001, default=1.0007, decimals=4, space='buy', optimize=True, load=True)
    atr_length = IntParameter(5, 30, default=5, space='buy', optimize=True, load=True)
    window = IntParameter(10, 30, default=16, space='buy', optimize=True, load=True)
    x = DecimalParameter(low=1.2, high=1.75, default=1.6, decimals=2, space='buy', optimize=True, load=True)

    ### trailing stop loss optimiziation ###
    tsl_target3 = DecimalParameter(low=0.1, high=0.15, default=0.15, decimals=2, space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, decimals=3, space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.06, high=0.1, default=0.1, decimals=3, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, decimals=3, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.04, high=0.08, default=0.06, decimals=3, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, decimals=3, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.03, high=0.06, default=0.04, decimals=3, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.005, high=0.012, default=0.01, decimals=3, space='sell', optimize=True, load=True)
    moon = IntParameter(80, 90, default=85, space='sell', optimize=True)

    @property
    def protections(self):
        return [{'method': 'CooldownPeriod', 'stop_duration_candles': 5}, {'method': 'MaxDrawdown', 'lookback_period_candles': 48, 'trade_limit': 20, 'stop_duration_candles': 4, 'max_allowed_drawdown': 0.2}, {'method': 'StoplossGuard', 'lookback_period_candles': 24, 'trade_limit': 4, 'stop_duration_candles': 2, 'only_per_pair': False}, {'method': 'LowProfitPairs', 'lookback_period_candles': 6, 'trade_limit': 2, 'stop_duration_candles': 60, 'required_profit': 0.02}, {'method': 'LowProfitPairs', 'lookback_period_candles': 24, 'trade_limit': 4, 'stop_duration_candles': 2, 'required_profit': 0.01}]
 

    # This is called when placing the initial order (opening trade)
    # Let unlimited stakes leave funds open for DCA orders
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        # We need to leave most of the funds for possible further DCA orders
        proposed_stake = proposed_stake / self.max_dca_multiplier.value  #  Leaving some reserve incase the market dumps!!!

        # This also applies to fixed staked
        return proposed_stake 

   ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        if current_candle['max_l'] > 0.001:
            if current_profit > self.tsl_target3.value:
                self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl3 {self.tsl_target3.value}/{self.ts3.value} activated')
                logger.info(f'*** {pair} *** Profit {current_profit} - lvl3 {self.tsl_target3.value}/{self.ts3.value} activated')
                return self.ts3.value
            if current_profit > self.tsl_target2.value:
                self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl2 {self.tsl_target2.value}/{self.ts2.value} activated')
                logger.info(f'*** {pair} *** Profit {current_profit} - lvl2 {self.tsl_target2.value}/{self.ts2.value} activated')
                return self.ts2.value
            if current_profit > self.tsl_target1.value:
                self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl1 {self.tsl_target1.value}/{self.ts1.value} activated')
                logger.info(f'*** {pair} *** Profit {current_profit} - lvl1 {self.tsl_target1.value}/{self.ts1.value} activated')
                return self.ts1.value
            if current_profit > self.tsl_target0.value:
                self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl0 {self.tsl_target0.value}/{self.ts0.value} activated')
                logger.info(f'*** {pair} *** Profit {current_profit} - lvl0 {self.tsl_target0.value}/{self.ts0.value} activated')
                return self.ts0.value
        elif current_profit > self.tsl_target0.value:
            self.dp.send_msg(f'*** {pair} *** Profit {current_profit} SWINGING FOR THE MOON!!!')
            return 0.99
        return self.stoploss

    def adjust_trade_position(self, trade: Trade, current_time: datetime, 
                              current_rate: float, current_profit: float, 
                              min_stake: Optional[float], max_stake: float, 
                              current_entry_rate: float, current_exit_rate: float, 
                              current_entry_profit: float, current_exit_profit: float, 
                              **kwargs) -> Optional[float]: 
 
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe) 
        current_candle = dataframe.iloc[-1].squeeze() 
        DCA1 = current_candle['move_mean'] * 2
        DCA2 = current_candle['move_mean_x'] * 2
        enable = current_candle['enter_long'] 
 
        filled_entries = trade.select_filled_orders(trade.entry_side) 
        count_of_entries = trade.nr_of_successful_entries 
        trade_duration = (current_time - trade.open_date_utc).seconds / 60 
 
        if current_profit is not None: 
            logger.info(f"{trade.pair} - Current Profit: {current_profit:.4f} # of Entries: {trade.nr_of_successful_entries} DCA1: -{DCA1:.4f} DCA2: -{DCA2:.4f} DCA Enable:{enable}") 
        # Take Profit if m00n 
        if current_profit > (DCA1) and trade.nr_of_successful_exits == 0: 
            # Take half of the profit at +5% 
            return -(trade.stake_amount / 2) 
        if current_profit > (DCA2) and trade.nr_of_successful_exits == 1: 
            # Take half of the profit at +5% 
            return -(trade.stake_amount / 1) 
 
        # Profit Based DCA     
        if current_profit > -DCA1 and trade.nr_of_successful_entries == 1: 
            
            return None 
 
        if current_profit > -DCA2 and trade.nr_of_successful_entries == 2: 
            
            return None 
 
        try: 
            # This returns first order stake size  
            # Modify the following parameters to enable more levels or different buy size: 
            # max_entry_position_adjustment = 3  
            # max_dca_multiplier = 3.5  
 
            stake_amount = filled_entries[0].cost 
            # This then calculates current safety order size 
            if count_of_entries == 1:  
                stake_amount = stake_amount * 1.5 
            elif count_of_entries == 2: 
                stake_amount = stake_amount * 1.5 
            else: 
                stake_amount = stake_amount 
 
            return stake_amount 
        except Exception as exception: 
            return None 
 
        return None

    def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        entry_price = (dataframe['close'].iat[-1] + dataframe['open'].iat[-1] + proposed_rate + proposed_rate) / 4
        logger.info(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}")
        # Check if there is a stored last entry price and if it matches the proposed entry price
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.0001:  # Tolerance for floating-point comparison
            entry_price *= self.increment.value  # Increment by 0.2%
            logger.info(f'{pair} Incremented entry price: {entry_price} based on previous entry price : {self.last_entry_price}.')
        # Update the last entry price
        self.last_entry_price = entry_price
        return entry_price

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, exit_reason: str, current_time: datetime, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        if exit_reason == 'roi' and last_candle['min_l'] > last_candle['max_l'] * 3:
            return False
        # Handle freak events
        if exit_reason == 'roi' and last_candle['max'] < 0.002:
            logger.info(f'{trade.pair} ROI is below 0')
            self.dp.send_msg(f'{trade.pair} ROI is below 0')
            return False
        if exit_reason == 'roi' and trade.calc_profit_ratio(rate) < 0.003:
            logger.info(f'{trade.pair} ROI is below 0')
            self.dp.send_msg(f'{trade.pair} ROI is below 0')
            return False
        if exit_reason == 'partial_exit' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f'{trade.pair} partial exit is below 0')
            self.dp.send_msg(f'{trade.pair} partial exit is below 0')
            return False
        if exit_reason == 'trailing_stop_loss' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f'{trade.pair} trailing stop price is below 0')
            self.dp.send_msg(f'{trade.pair} trailing stop price is below 0')
            return False
        return True

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        distance = (last_candle['move_mean'] - current_profit) * 100
        display_profit = current_profit * 100
        target = last_candle['move_mean'] * 100

        logger.info(f'{trade.pair} profit: {display_profit:.4f}% target: {target:.4f}% distance to ROI: {distance:.4f}% ')
        # when above the mean
        if current_profit > last_candle['move_mean']:
            
            return 'Above Mean'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        #ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_length.value)

        # Apply rolling window operation to the 'OHLC4' column
        rolling_window = dataframe['OHLC4'].rolling(self.window.value)  # 5.25 hrs
        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_value = rolling_window.apply(lambda x: np.ptp(x))
        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['move'] = ptp_value / ta.SMA(dataframe['OHLC4'], 3)
        dataframe['move_mean'] = dataframe['move'].mean()
        dataframe['move_mean_x'] = dataframe['move'].mean() * self.x.value
        move = '{:.2f}'.format(dataframe['move'].iloc[-1] * 100)
        move_mean = '{:.2f}'.format(dataframe['move_mean'].iloc[-1] * 100)

        dataframe['atr_pcnt'] = ta.ATR(dataframe, timeperiod=self.atr_length.value) / dataframe['OHLC4']
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
        dataframe['OHLC_SELL'] = ta.SMA(dataframe['OHLC4'] * (1 + dataframe['move_mean']), 12)
        dataframe['OHLC_BUY'] = ta.SMA(dataframe['OHLC4'] * (1 - dataframe['move_mean']), 12)
        
        pair = metadata['pair']
        close_series = dataframe['OHLC4'][-self.context_length:]

        dataframe.loc[dataframe.index[-1], 'decision'] = 0
        dataframe.loc[dataframe.index[-1], 'prediction_quality'] = 0

        if pair not in self.forecasting_models:
            self.forecasting_models[pair] = auto_arima(close_series, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True, maxiter=self.maxiter, seasonal=False)
        else:
            self.forecasting_models[pair].update(close_series)

        forecast = self.forecasting_models[pair].predict(n_periods=self.forecast_period)
        forecast_index_start = dataframe.index[-1] + 1
        forecast_series = pd.Series(forecast, index=range(forecast_index_start, forecast_index_start + self.forecast_period))

        self.forecast_series = forecast_series
        
        # Logging predictions with the current close date
        last_date = dataframe['date'].tail(1).item()
        current_close = dataframe['OHLC4'].iloc[-1]
        formatted_current_close = f"{current_close:.8f}"
        
        # Check if the pair already exists in the dictionary
        if pair in self.previous_predicted_close:
            # Use the previous predicted close for this pair
            previous_predicted_close = self.previous_predicted_close[pair]
            # print(f"Previous Predicted Close: {previous_predicted_close}")
        
            # Convert prediction dates to pandas datetime outside the loop
            previous_predicted_close = {pd.to_datetime(k): v for k, v in previous_predicted_close.items()}

            # Add the previous predicted close values to the dataframe
            previous_predicted_close_dates = dataframe['date'].isin(previous_predicted_close.keys())
            dataframe.loc[previous_predicted_close_dates, 'forecasted_close'] = [previous_predicted_close[date] for date in dataframe.loc[previous_predicted_close_dates, 'date']]
        
            # Iterate through each prediction date
            for prediction_date, predicted_close in previous_predicted_close.items():
                # Get the corresponding rows in the dataframe for the prediction date
                matching_rows = dataframe[dataframe['date'] == prediction_date]
                if not matching_rows.empty:
                    # Convert 'predicted_close' to numeric explicitly
                    predicted_close = float(predicted_close)
                    
                    # Get the actual close for the prediction date
                    actual_close = matching_rows['OHLC4'].iloc[0]

                    # Error thresholds definition
                    threshold_mae = 0.01  # Example threshold for Mean Absolute Error
                    threshold_mse = 0.0001  # Example threshold for Mean Squared Error
                    threshold_rmse = 0.01  # Example threshold for Root Mean Squared Error
                    threshold_mape = 1.0  # Percentage threshold for Mean Absolute Percentage Error (e.g., 1.0 for 1%)
        
                    # Calculate metrics for the prediction
                    mae, mse, rmse = mean_absolute_error([actual_close], [predicted_close]), mean_squared_error([actual_close], [predicted_close]), np.sqrt(mean_squared_error([actual_close], [predicted_close]))
                    mape = (np.abs((predicted_close - actual_close) / actual_close) * 100) if actual_close != 0 else np.nan
        
                    # Determine prediction quality based on thresholds
                    is_error = mae > threshold_mae or mse > threshold_mse or rmse > threshold_rmse or (not np.isnan(mape) and mape > threshold_mape)
                    is_perfect = mae == 0 and mse == 0 and rmse == 0 and (np.isnan(mape) or mape == 0)
        
                    # Assign prediction quality value
                    if is_error:
                        dataframe.loc[matching_rows.index, 'prediction_quality'] = -1
                    elif is_perfect:
                        dataframe.loc[matching_rows.index, 'prediction_quality'] = 0
                    else:
                        dataframe.loc[matching_rows.index, 'prediction_quality'] = 1
        
            logger.info(f"{pair} {self.timeframe} | Prediction Date: {prediction_date} | Predicted Close: {predicted_close} | Actual Close: {actual_close}")
            logger.info(f"{pair} | MAE: {mae:.7f} | MSE: {mse:.14f} | RMSE: {rmse:.7f} | MAPE: {mape:.7f}%")
        else:
            # Initialize the previous predicted close for this pair as an empty dictionary
            self.previous_predicted_close[pair] = {}
    
        
        # Get the forecasted close prices
        forecasts_above_current_close = self.forecast_series
        # Convert all forecasted prices to formatted strings
        formatted_forecasts = {k: f"{v:.8f}" for k, v in forecasts_above_current_close.items()}
        # Print the first forecasted close price
        first_forecast_key = next(iter(formatted_forecasts))
        # print(f"Forecasted Close Prices: {formatted_forecasts[first_forecast_key]}")
        
        # Update the previous predicted close dictionary with the latest forecast
        self.previous_predicted_close[pair].update({last_date: formatted_forecasts[first_forecast_key]})       

        # Given previous steps concatenate forecast series to the existing dataframe
        actual_close = dataframe['OHLC4'][-self.forecast_period:]
        forecast_close = self.forecast_series.values

        # Check if, at any point, forecasted close is 5% above current close
        if (forecast_series - current_close).max() > (dataframe['move_mean'].iloc[-1] * 0.618) * current_close:
            direction = 'Up'
            dataframe.loc[dataframe.index[-1], 'decision'] = 1

        # Check if, at any point, forecasted close is below current close and down by 5%
        if (forecast_series - current_close).min() < -(dataframe['move_mean'].iloc[-1] * 0.618) * current_close:
            direction = 'Down'
            dataframe.loc[dataframe.index[-1], 'decision'] = -1

        # Concatenate all historical close prices with forecasted values
        combined_series = pd.concat([dataframe['OHLC4'], self.forecast_series])
        combined_series.reset_index(drop=True, inplace=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # condition1 = (
        #     (dataframe['decision'] == 1) & 
        #     (dataframe['move'] >= dataframe['move_mean']) & 
        #     (dataframe['move'].shift(3) < dataframe['move_mean'].shift(3)) & 
        #     # (dataframe['min'] < dataframe['max']) & (dataframe['min_l'] < dataframe['max_l']) & 
        #     # (dataframe['max_l'] < dataframe['atr_pcnt']) & (dataframe['OHLC4'] < dataframe['sma_dn']) & 
        #     # (dataframe['sma_dn'].shift() > dataframe['sma_dn']) & 
        #     (dataframe['volume'] > 0)
        # )

        # dataframe.loc[condition1, 'enter_long'] = 1
        # dataframe.loc[condition1, 'enter_tag'] = 'Up Trend Soon below sma_dn'

        # condition2 = (
        #     (dataframe['decision'] == 1) & 
        #     (dataframe['move'] >= dataframe['move_mean_x']) & 
        #     # (dataframe['min'] < dataframe['max']) & 
        #     # (dataframe['min_l'] < dataframe['max_l']) & 
        #     # (dataframe['max_l'] < dataframe['atr_pcnt']) & 
        #     # (dataframe['OHLC4'] < dataframe['sma_dn']) & 
        #     # (dataframe['max_l'] < dataframe['atr_pcnt']) & 
        #     (dataframe['volume'] > 0)
        #     )

        # dataframe.loc[condition2, 'enter_long'] = 1
        # dataframe.loc[condition2, 'enter_tag'] = 'Move Mean Fib below sma_dn'

        # condition3 = (
        #     (dataframe['decision'] == 1) & 
        #     (dataframe['move'] >= dataframe['move_mean']) & 
        #     (dataframe['move'].shift(3) < dataframe['move_mean'].shift(3)) & 
        #     (dataframe['min'] < dataframe['max']) & 
        #     (dataframe['min_l'] < dataframe['max_l']) & 
        #     (dataframe['OHLC4'] < dataframe['sma']) & 
        #     (dataframe['volume'] > 0)
        #     )

        # dataframe.loc[condition3, 'enter_long'] = 1
        # dataframe.loc[condition3, 'enter_tag'] = 'Up Trend Soon below sma'

        # condition4 = (
        #     (dataframe['decision'] == 1) & 
        #     (dataframe['move'] >= dataframe['move_mean_x']) & 
        #     (dataframe['min'] < dataframe['max']) & 
        #     (dataframe['min_l'] < dataframe['max_l']) & 
        #     (dataframe['max_l'] < dataframe['atr_pcnt']) & 
        #     (dataframe['OHLC4'] < dataframe['sma']) & 
        #     (dataframe['max_l'] < dataframe['atr_pcnt']) & 
        #     (dataframe['volume'] > 0)
        #     )

        # dataframe.loc[condition4, 'enter_long'] = 1
        # dataframe.loc[condition4, 'enter_tag'] = 'Move Mean Fib below sma'

        condition5150 = (
            (dataframe['decision'] == 1) & 
            (dataframe['move'] >= dataframe['move_mean']) & 
            (dataframe['move'].shift(2) > dataframe['move_mean'].shift(2)) & 
            (dataframe['move'].shift(3) < dataframe['move_mean'].shift(3)) & 
            (dataframe['min'] == 0) & 
            (dataframe['volume'] > 0)
            )

        dataframe.loc[condition5150, 'enter_long'] = 1
        dataframe.loc[condition5150, 'enter_tag'] = 'Hope this works...'

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        condition5 = (
            (dataframe['decision'] == -1) & 
            (dataframe['move'] >= dataframe['move_mean']) &
            (dataframe['move'].shift(3) < dataframe['move_mean'].shift(3)) & 
            (dataframe['min'] > dataframe['max']) & 
            (dataframe['min_l'] > dataframe['max_l']) & 
            (dataframe['volume'] > 0)
            )

        dataframe.loc[condition5, 'exit_long'] = 1
        dataframe.loc[condition5, 'exit_tag'] = 'Down Trend Soon'

        condition6 = (
            (dataframe['decision'] == -1) & 
            (dataframe['move'] >= dataframe['move_mean_x']) & 
            (dataframe['move'].shift(3) >= dataframe['move_mean_x'].shift(3)) & 
            (dataframe['min'] > dataframe['max']) & 
            (dataframe['min_l'] > dataframe['max_l']) & 
            (dataframe['volume'] > 0)
            )

        dataframe.loc[condition6, 'exit_long'] = 1
        dataframe.loc[condition6, 'exit_tag'] = 'Move Mean Fib'

        return dataframe


    
