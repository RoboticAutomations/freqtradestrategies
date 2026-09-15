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

'''



________/\\\\\\\\\__/\\\______________/\\\________________/\\\________/\\\_____/\\\\\\\\\\___/\\\\\_____/\\\__________________________/\\\______________________/\\\\\\\_________________        
 _____/\\\////////__\/\\\__________/\\\\\\\_______________\/\\\_____/\\\//____/\\\///////\\\_\/\\\\\\___\/\\\________________________/\\\\\____________________/\\\/////\\\_______________       
  ___/\\\/___________\/\\\_________\/////\\\_______________\/\\\__/\\\//______\///______/\\\__\/\\\/\\\__\/\\\_____/\\\_____________/\\\/\\\___________________/\\\____\//\\\______________      
   __/\\\_____________\/\\\_____________\/\\\_____/\\\\\\\\_\/\\\\\\//\\\_____________/\\\//___\/\\\//\\\_\/\\\__/\\\\\\\\\\\______/\\\/\/\\\________/\\\\\\\\_\/\\\_____\/\\\__/\\\\\\\\\\_     
    _\/\\\_____________\/\\\\\\\\\\______\/\\\___/\\\//////__\/\\\//_\//\\\___________\////\\\__\/\\\\//\\\\/\\\_\////\\\////_____/\\\/__\/\\\______/\\\//////__\/\\\_____\/\\\_\/\\\//////__    
     _\//\\\____________\/\\\/////\\\_____\/\\\__/\\\_________\/\\\____\//\\\_____________\//\\\_\/\\\_\//\\\/\\\____\/\\\_______/\\\\\\\\\\\\\\\\__/\\\_________\/\\\_____\/\\\_\/\\\\\\\\\\_   
      __\///\\\__________\/\\\___\/\\\_____\/\\\_\//\\\________\/\\\_____\//\\\___/\\\______/\\\__\/\\\__\//\\\\\\____\/\\\_/\\__\///////////\\\//__\//\\\________\//\\\____/\\\__\////////\\\_  
       ____\////\\\\\\\\\_\/\\\___\/\\\_____\/\\\__\///\\\\\\\\_\/\\\______\//\\\_\///\\\\\\\\\/___\/\\\___\//\\\\\____\//\\\\\_____________\/\\\_____\///\\\\\\\\__\///\\\\\\\/____/\\\\\\\\\\_ 
        _______\/////////__\///____\///______\///_____\////////__\///________\///____\/////////_____\///_____\/////______\/////______________\///________\////////_____\///////_____\//////////__





'''


class ARIMA_5_DCA_SCALP_HO(IStrategy):
    # Stoploss:
    stoploss = -0.20
    # Trailing stop:
    use_custom_stoploss = True
    # Initialize dicts for arima storage
    last_run_time = {} 
    arima_model = {} 
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
    moon = DecimalParameter(0.000, 0.003, default=0.003, decimals=3, space='sell', optimize=True)
    SLX = DecimalParameter(0.5, 0.67, default=0.618,  decimals=2, space='sell', optimize=True)
    SLS = DecimalParameter(0.25, 0.4, default=0.35,  decimals=2, space='sell', optimize=True)

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
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        SLT1 = current_candle['move_mean']
        SL1 = current_candle['move_mean'] / 3
        SLT2 = current_candle['move_mean_x']
        SL2 = current_candle['move_mean_x'] - current_candle['move_mean']
        display_profit = current_profit * 100
        slt1 = SLT1 * 100
        sl1 = SL1 * 100
        slt2 = SLT2 * 100
        sl2 = SL2 * 100


        if current_candle['max_l'] > self.moon.value: #ignore stoploss if setting new highs
            if SLT2 is not None and current_profit > SLT2:
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.2f}% - {slt2:.2f}%/{sl2:.2f}% activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.2f}% - {slt2:.2f}%/{sl2:.2f}% activated')
                return SL2
            if SLT1 is not None and current_profit > SLT1:
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.2f}% - {SLT1:.2f}%/{SL1:.2f}% activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.2f}% - {slt1:.2f}%/{sl1:.2f}% activated')
                return SL1
            if SLT1 is not None and current_profit > (SLT1 * self.SLX.value):
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.2f}% - {SLT1:.2f}%/{self.SLS.value:.2f}% activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.2f}% - {slt1:.2f}%/{self.SLS.value:.2f}% activated')
                return (SLT1 * self.SLS.value)

        else:
            if SLT1 is not None and current_profit > SL1:
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.2f}% SWINGING FOR THE MOON!!!')
                logger.info(f'*** {pair} *** Profit {display_profit:.2f}% SWINGING FOR THE MOON!!!')
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
        distance = (current_profit - last_candle['move_mean']) * 100
        display_profit = current_profit * 100
        target = last_candle['move_mean'] * 100

        logger.info(f'{trade.pair} profit: {display_profit:.4f}% target: {target:.4f}% distance to ROI: {distance:.4f}% ')

        # when above the mean x
        if current_profit > last_candle['move_mean_x']:
            if last_candle['move'] > last_candle['move_mean_x']:
                return 'Above Mean X'

        # when above the mean x
        if current_profit > last_candle['move_mean']:
            if last_candle['max'] > 0:
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
        
        dataframe['decision'] = 0
        pair = metadata['pair']
        current_time = time.time()

        dataframe['OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        size = len(dataframe) - 10
        train, test = model_selection.train_test_split(dataframe['OHLC4'], train_size=size)

        # Initialize values for the current pair if not already done
        if pair not in self.last_run_time:
            self.last_run_time[pair] = current_time

            logger.info(f"Initial ARIMA {self.timeframe} Model Training for {pair}")
            # Fit ARIMA model
            start_time = time.time()
            self.arima_model[pair] = auto_arima(train, start_p=1, start_q=1, start_P=1, start_Q=1,
                                                 max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=False,
                                                 stepwise=True, suppress_warnings=True, D=10, max_D=20,
                                                 error_action='ignore')
            fitting_time = time.time() - start_time
            logger.info(f"{pair} - ARIMA {self.timeframe} Model fitted in {fitting_time:.2f} seconds")


        # Check if it's time to retrain for the current pair
        if current_time - self.last_run_time[pair] >= 3600:  # Check if an hour has passed
            logger.info(f"Auto Fitting ARIMA {self.timeframe} Model for {pair}")
            self.last_run_time[pair] = current_time

            # Fit ARIMA model
            start_time = time.time()
            self.arima_model[pair] = auto_arima(train, start_p=1, start_q=1, start_P=1, start_Q=1,
                                                 max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=False,
                                                 stepwise=True, suppress_warnings=True, D=10, max_D=20,
                                                 error_action='ignore')
            fitting_time = time.time() - start_time
            logger.info(f"{pair} - ARIMA {self.timeframe} Model fitted in {fitting_time:.2f} seconds")

        # Use the previously fitted ARIMA model for forecasting
        if self.arima_model[pair] is not None:
            start_time = time.time()
            future_forecast, conf_int = self.arima_model[pair].predict(n_periods=test.shape[0], return_conf_int=True)
            inference_time = time.time() - start_time

        timeleft = current_time - self.last_run_time[pair]
        if timeleft <= 3600 and timeleft != 0:
            logger.info(f"{pair} - ARIMA {self.timeframe} Model re-optimized in {timeleft:.2f} seconds")

        # Extract upper and lower confidence intervals
        lower_confidence, upper_confidence = conf_int[:, 0], conf_int[:, 1]
        logger.info(f"{pair} - Inference time: {inference_time:.2f} seconds | " \
             f"Current Price: {dataframe['OHLC4'].iloc[-1]:.7f} | {self.timeframe} Future Forecast: {future_forecast.iloc[-1]:.7f}")

        dataframe['rmse'] = 0
        dataframe['accuracy_perc'] = 0
        dataframe['reward'] = 0
        dataframe['rmse'] = np.sqrt(mean_squared_error(test, future_forecast))
        dataframe['accuracy_perc'] = 100 * (1 - (dataframe['rmse'].iloc[-1] / dataframe['OHLC4'].iloc[-1]))
        dataframe['reward'] = ((future_forecast.iloc[-1] / dataframe['OHLC4'].iloc[-1]) - 1) * 100

        rmse = dataframe['rmse'].iloc[-1] 
        accuracy_perc = dataframe['accuracy_perc'].iloc[-1] 
        reward = dataframe['reward'] .iloc[-1] 
        # Apply rolling window operation to the 'OHLC4' column
        rolling_window = dataframe['OHLC4'].rolling(self.window.value) # 5.25 hrs

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_value = rolling_window.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['move'] = ptp_value / dataframe['OHLC4']
        dataframe['move_mean'] = dataframe['move'].mean()
        dataframe['move_mean_x'] = dataframe['move'].mean() * self.x.value
        move = '{:.2f}'.format(dataframe['move'].iloc[-1] * 100)
        move_mean = '{:.2f}'.format(dataframe['move_mean'].iloc[-1] * 100)


        if future_forecast.iloc[-1] > (1 - (dataframe['move_mean'].iloc[-1] * 0.618)) * dataframe['OHLC4'].iloc[-1]:
            direction = 'Up'
            dataframe['decision'] = 1
        if future_forecast.iloc[-1] < (1 + (dataframe['move_mean'].iloc[-1] * 0.618)) * dataframe['OHLC4'].iloc[-1]:
            direction = 'Down'
            dataframe['decision'] = -1
        

        logger.info(f"{pair} - Test RMSE: {rmse:.3f} | Accuracy: {accuracy_perc:.2f}% | Potential Profit: {move}% | Avg. Profit: {move_mean}% | {self.timeframe} Trend: {direction}")

        dataframe['arima_predictions'] = pd.Series(future_forecast)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        condition5150 = (
            (dataframe['decision'] == 1) & 
            (dataframe['move'] >= dataframe['move_mean']) & 
            (dataframe['move'].shift(3) > dataframe['move_mean'].shift(3)) & 
            (dataframe['move'].shift(4) < dataframe['move_mean'].shift(4)) & 
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
            (dataframe['move'].shift(4) > dataframe['move_mean'].shift(4)) & 
            (0 < dataframe['max']) & 
            (dataframe['volume'] > 0)
            )

        dataframe.loc[condition5, 'exit_long'] = 1
        dataframe.loc[condition5, 'exit_tag'] = '$$$...'



        return dataframe


    
