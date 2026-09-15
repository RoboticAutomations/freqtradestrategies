
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce
from typing import Optional
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
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter
import technical.indicators as ftt
import math
import logging
from scipy.signal import find_peaks
import warnings
import torch
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


logger = logging.getLogger(__name__)


class CusumMeanDetector():
        

    def __init__(self, t_warmup = 30, p_limit = 0.01, error_margin = 0.005) -> None:
        self._t_warmup = t_warmup
        self._p_limit = p_limit
        self._error_margin = error_margin
        self._reset()    
        

    def predict_next(self, y: torch.tensor) -> Tuple[float,bool]:
        self._update_data(y)

        if self.current_t == self._t_warmup:
            self._init_params()
        
        if self.current_t >= self._t_warmup:
            prob, is_changepoint = self._check_for_changepoint()
            if is_changepoint:
                self._reset()

            return (1-prob), is_changepoint
        
        else:
            return 0, False
            
    
    def _reset(self) -> None:
        self.current_t = torch.zeros(1)
                
        self.current_obs = []
        
        self.current_mean = None
        self.current_std = None
            
    
    def _update_data(self, y: torch.tensor) -> None:
        self.current_t += 1
        self.current_obs.append(y.reshape(1))

        
    
    def _init_params(self) -> None:
        self.current_mean = torch.mean(torch.concat(self.current_obs))
        self.current_std = torch.std(torch.concat(self.current_obs))
             
    
    def _check_for_changepoint(self) -> Tuple[float,bool]:
        standardized_sum = torch.sum(torch.concat(self.current_obs) - self.current_mean)/(self.current_std * self.current_t**0.5)
        prob = float(self._get_prob(standardized_sum).detach().numpy())
        
        return prob, self._p_limit - self._error_margin < prob < self._p_limit + self._error_margin
    
    
    def _get_prob(self, y: torch.tensor) -> bool:
        p = torch.distributions.normal.Normal(0,1).cdf(torch.abs(y))
        prob = 2*(1 - p)
        
        return prob

class PeekabooV4(IStrategy):
    INTERFACE_VERSION = 2


    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]




    # Stoploss:
    stoploss = -0.99

    # Trailing stop:
    use_custom_stoploss = True


    # exit signal
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False
    can_short = True

    ## Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    # Optimal timeframe for the strategy
    timeframe = '15m'
    startup_candle_count = 400
    process_only_new_candles = True
    
    # DCA
    position_adjustment_enable = True

    # Custom Entry
    last_entry_price = None

    # Hyper-opt parameters
    base_nb_candles_buy = IntParameter(150, 200, default=184, space='buy', optimize=True)
    up = DecimalParameter(low=1.020, high=1.025, default=1.02, decimals=3 ,space='buy', optimize=True, load=True)
    dn = DecimalParameter(low=0.983, high=0.987, default=0.984, decimals=3 ,space='buy', optimize=True, load=True)
    enable1 = BooleanParameter(default=True, space="buy", optimize=False)
    enable2 = BooleanParameter(default=True, space="buy", optimize=False)
    enable3 = BooleanParameter(default=True, space="buy", optimize=False)
    enable4 = BooleanParameter(default=True, space="buy", optimize=False)
    enable5 = BooleanParameter(default=True, space="buy", optimize=False)
    enable6 = BooleanParameter(default=True, space="buy", optimize=False)
    increment = DecimalParameter(low=1.0005, high=1.001, default=1.0007, decimals=4 ,space='buy', optimize=True, load=True)
    # quick = DecimalParameter(low=0.75, high=0.9, default=0.85, decimals=2 ,space='buy', optimize=True, load=True)
    m00n = DecimalParameter(low=1.60, high=1.80, default=1.68, decimals=2 ,space='buy', optimize=True, load=True)

    # Modulus    
    peaks = IntParameter(70, 200, default=120, space='buy', optimize=True) ### initial smallest window
    # bull_bear = IntParameter(120, 160, default=155, space='buy', optimize=True)
    # trend = DecimalParameter(low=25, high=40, default=29.6, decimals=1 ,space='buy', optimize=True, load=True)
    # volatility = DecimalParameter(low=30, high=50, default=38.4, decimals=1 ,space='buy', optimize=True, load=True)
    # sensitivity = IntParameter(7, 15, default=11, space='buy', optimize=True, load=True)
    # lookback_candles = IntParameter(30, 60, default=55, space='buy', optimize=True, load=True)
    # atr = IntParameter(3, 7, default=5, space='buy', optimize=True, load=True)
    perc_target = DecimalParameter(low=1, high=5, default=1.5, decimals=1 ,space='buy', optimize=True, load=True)

    # DCA
    initial_safety_order_trigger = DecimalParameter(low=-0.02, high=-0.015, default=-0.016, decimals=3 ,space='buy', optimize=True, load=True)
    max_safety_orders = IntParameter(1, 6, default=2, space='buy', optimize=True)
    max_dca_multiplier = IntParameter(1, 6, default=2, space='buy', optimize=True)
    safety_order_step_scale = DecimalParameter(low=1.1, high=1.4, default=1.3, decimals=2 ,space='buy', optimize=True, load=True)
    safety_order_volume_scale = DecimalParameter(low=1.1, high=1.5, default=1.2, decimals=1 ,space='buy', optimize=True, load=True)

    # Unclog Function
    days = IntParameter(2, 7, default=3, space='sell', optimize=True)
    loss = DecimalParameter(-0.07, -0.04, default=-0.055, space='sell', optimize=True)
    atr_length = IntParameter(5, 30, default=5, space='buy', optimize=True, load=True)

    ### trailing stop loss optimiziation ###
    tsl_target3 = DecimalParameter(low=0.10, high=0.15, default=0.15, decimals=2,  space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, decimals=3,  space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.06, high=0.10, default=0.1, decimals=3, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, decimals=3, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.04, high=0.06, default=0.06, decimals=3, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, decimals=3, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.04, high=0.05, default=0.04, decimals=3, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.005, high=0.012, default=0.01, decimals=3, space='sell', optimize=True, load=True)
    moon = IntParameter(80, 90, default=85, space='sell', optimize=True)
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
             proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
             side: str, **kwargs) -> float:
        return 20
    
    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:


        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if current_candle['rsi'] < self.moon.value:

            for stop3 in self.tsl_target3.range:
                if (current_profit > stop3):
                    for stop3a in self.ts3.range:
                        self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl3 {stop3}/{stop3a} activated')
                        return stop3a 
            for stop2 in self.tsl_target2.range:
                if (current_profit > stop2):
                    for stop2a in self.ts2.range:
                        self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl2 {stop2}/{stop2a} activated')
                        return stop2a 
            for stop1 in self.tsl_target1.range:
                if (current_profit > stop1):
                    for stop1a in self.ts1.range:
                        self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl1 {stop1}/{stop1a} activated')
                        return stop1a 
            for stop0 in self.tsl_target0.range:
                if (current_profit > stop0):
                    for stop0a in self.ts0.range:
                        self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl0 {stop0}/{stop0a} activated')
                        return stop0a 
        else:
            for stop0 in self.tsl_target0.range:
                if (current_profit > stop0):
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} SWINGING FOR THE MOON!!!')
                    return 0.99

        return self.stoploss

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        if current_profit > self.initial_safety_order_trigger.value:
            logger.info(f"{trade.pair} - Current Profit: {current_profit} Trigger: {self.initial_safety_order_trigger.value}")
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        count_of_buys = 0
        for order in trade.orders:
            if order.ft_is_open or order.ft_order_side != 'buy':
                continue
            if order.status == "closed":
                count_of_buys += 1

        if 1 <= count_of_buys <= self.max_safety_orders.value:
            
            safety_order_trigger = abs(self.initial_safety_order_trigger.value) + (abs(self.initial_safety_order_trigger.value) * self.safety_order_step_scale.value * (math.pow(self.safety_order_step_scale.value,(count_of_buys - 1)) - 1) / (self.safety_order_step_scale.value - 1))

            if current_profit <= (-1 * abs(safety_order_trigger)):
                try:
                    stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
                    stake_amount = stake_amount * math.pow(self.safety_order_volume_scale.value,(count_of_buys - 1))
                    amount = stake_amount / current_rate
                    logger.info(f"Initiating safety order buy #{count_of_buys} for {trade.pair} with stake amount of {stake_amount} which equals {amount}")
                    return stake_amount
                except Exception as exception:
                    logger.debug(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}') 
                    return None
            else:
                stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
                stake_amount = stake_amount * math.pow(self.safety_order_volume_scale.value,(count_of_buys - 1))
                logger.info(f"{trade.pair} Next Safety Order #{count_of_buys} @ Trigger -{safety_order_trigger} Current Profit: {current_profit}")    
                return None

        logger.info(f"{trade.pair} - Current Profit: {current_profit} All safety orders used")     
        return None

    ### Custom Functions ###
    # This is called when placing the initial order (opening trade)
    # Let unlimited stakes leave funds open for DCA orders
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        # We need to leave most of the funds for possible further DCA orders
        # This also applies to fixed stakes
        total_stake = self.max_dca_multiplier.value + self.max_safety_orders.value
        return proposed_stake / total_stake

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        # exit any positions at a loss if they are held for more than 7 days.
        if current_profit < self.loss.value and (current_time - trade.open_date_utc).days >= self.days.value:
            return 'unclog'

    def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)

        entry_price = (dataframe['close'].iat[-1] + dataframe['open'].iat[-1] + proposed_rate + proposed_rate) / 4
        logger.info(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}") 

        # Check if there is a stored last entry price and if it matches the proposed entry price
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.0001:  # Tolerance for floating-point comparison
            entry_price *= self.increment.value # Increment by 0.2%
            logger.info(f"{pair} Incremented entry price: {entry_price} based on previous entry price : {self.last_entry_price}.")

        # Update the last entry price
        self.last_entry_price = entry_price

        return entry_price


    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        # Handle freak events

        if exit_reason == 'partial_exit' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} partial exit is below 0")
            self.dp.send_msg(f'{trade.pair} partial exit is below 0')
            return False

        if exit_reason == 'trailing_stop_loss' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} trailing stop price is below 0")
            self.dp.send_msg(f'{trade.pair} trailing stop price is below 0')
            return False

        return True


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        pair = metadata['pair']

        # heikinashi = qtpylib.heikinashi(dataframe)
        # dataframe['ha_open'] = heikinashi['open']
        # dataframe['ha_close'] = heikinashi['close']
        # dataframe['ha_high'] = heikinashi['high']
        # dataframe['ha_low'] = heikinashi['low']
        # dataframe['ha_closedelta'] = (heikinashi['close'] - heikinashi['close'].shift())

        dataframe["&s-extrema"] = 0

        dataframe['sma'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']
        dataframe['sma_up'] = dataframe['sma'] * self.up.value
        dataframe['sma_dn'] = dataframe['sma'] * self.dn.value

        dataframe['sma_pc'] = abs((dataframe['sma'] - dataframe['sma'].shift(1)) / dataframe['sma']) * 100
        dataframe['atr_pcnt'] = (ta.ATR(dataframe, timeperiod=self.atr_length.value) / dataframe['close'])

        # dataframe['modulation'] = 1 + (dataframe['sma_pc'] * self.trend.value) + (dataframe['atr_pcnt'] * self.volatility.value)

        # min_window = self.peaks.value 
        # max_window = self.bull_bear.value

        # # Set minimum and maximum window size
        # dataframe['order'] = (dataframe['modulation'] * self.peaks.value).round().fillna(self.bull_bear.value).astype(int)
        # dataframe['order'] = np.where(dataframe['order'] > max_window, max_window, dataframe['order'])
        # dataframe['order'] = np.where(dataframe['order'] < min_window, min_window, dataframe['order'])
        
        # if not dataframe['order'].empty:
        #     order = dataframe['order'].iloc[-1]
        # else:
        #     order = self.bear.value

        # peak prominence
        ptc_target= self.perc_target.value
        dataframe['peak_prominance'] = float(0)
        dataframe['peak'] = float(0)

        high_peaks, high_properties = find_peaks(dataframe['high'].values, prominence=dataframe['high'].values / 100 * ptc_target,
                                                 wlen=120)

        lower_peaks, low_properties = find_peaks(-dataframe['low'].values, prominence=dataframe['low'].values / 100 * ptc_target,
                                                 wlen=120)

        dataframe.iloc[lower_peaks, dataframe.columns.get_loc('peak_prominance')] = low_properties["prominences"].astype('float')
        dataframe.iloc[high_peaks, dataframe.columns.get_loc('peak_prominance')] = -high_properties["prominences"].astype('float')

        dataframe.iloc[lower_peaks, dataframe.columns.get_loc('peak')] = 1
        dataframe.iloc[high_peaks, dataframe.columns.get_loc('peak')] = -1

        positive_values = dataframe['peak_prominance'].where(dataframe['peak_prominance'] > 0)
        negative_values = dataframe['peak_prominance'].where(dataframe['peak_prominance'] < 0)

        # Calculate the expanding mean for positive values
        dataframe['positive_expanding_mean'] = positive_values.expanding().mean()

        # Calculate the expanding mean for negative values
        dataframe['negative_expanding_mean'] = negative_values.expanding().mean()

        # Fill NaN values with the previous non-null values
        dataframe['positive_expanding_mean'].ffill(inplace=True)
        dataframe['negative_expanding_mean'].ffill(inplace=True)
        dataframe.reset_index(drop=True, inplace=True)
        dataframe['OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4

        dataframe['returns'] = dataframe['close'].pct_change()

        # SMA
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        
        dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['sma8'] = ta.SMA(dataframe, timeperiod=8)
        dataframe['sma13'] = ta.SMA(dataframe, timeperiod=13)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=6)
        
        dataframe['slowk'], dataframe['slowd'] = ta.STOCH(dataframe['high'], dataframe['low'], dataframe['close'], fastk_period=14, slowk_period=1, slowd_period=3)
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_percent'] = \
            (dataframe['close'] - dataframe['bb_lowerband']) / (dataframe['bb_upperband'] - dataframe['bb_lowerband'])
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']

        # Candlestick patterns bullish
        dataframe['cdl3inside'] = ta.CDL3INSIDE(dataframe)
        dataframe['cdl3outside'] = ta.CDL3OUTSIDE(dataframe)
        dataframe['cdl3starsinsouth'] = ta.CDL3STARSINSOUTH(dataframe)
        
        # Candlestick patterns bearish
        dataframe['cdl3blackcrows'] = ta.CDL3BLACKCROWS(dataframe)
        dataframe['cdl3whitesoldiers'] = ta.CDL3WHITESOLDIERS(dataframe)
        dataframe['cdl3linestrike'] = ta.CDL3LINESTRIKE(dataframe)
        # Calculate the standard deviation of the returns
        dataframe['volatility'] = dataframe['returns'].rolling(window=14).std()
        dataframe['rsi14'] = ta.RSI(dataframe, timeperiod=14)
        
        
        dataframe['max_l'] = dataframe['OHLC4'].rolling(120).max() / dataframe['OHLC4'] - 1
        dataframe['min_l'] = abs(dataframe['OHLC4'].rolling(120).min() / dataframe['OHLC4'] - 1)

        dataframe['max'] = dataframe['OHLC4'].rolling(4).max() / dataframe['OHLC4'] - 1
        dataframe['min'] = abs(dataframe['OHLC4'].rolling(4).min() / dataframe['OHLC4'] - 1)
     

        dataframe['rsi_overbought'] = (dataframe['rsi'] > 70).astype('int')
        dataframe['volume_increase'] = (dataframe['volume'] > dataframe['volume'].shift()).astype('int')
        dataframe['price_below_sma'] = (dataframe['close'] < dataframe['sma']).astype('int')
        dataframe['price_drop'] = (dataframe['close'] < dataframe['close'].shift()).astype('int')
        dataframe['williamr'] = ta.WILLR(dataframe, timeperiod=7)
        # prom = dataframe['peak_prominance'].iloc[-1]
        # if prom != 0:
        #     logger.info(f"*** {pair} *** Prominence 0: {prom}")
        # prom1 = dataframe['peak_prominance'].iloc[-2]
        # if prom1 != 0:
        #     logger.info(f"*** {pair} *** Prominence 1: {prom1}")
        # prom2 = dataframe['peak_prominance'].iloc[-3]
        # if prom2 != 0:
        #     logger.info(f"*** {pair} *** Prominence 2: {prom2}")
        # prom3 = dataframe['peak_prominance'].iloc[-4]
        # if prom3 != 0:
        #     logger.info(f"*** {pair} *** Prominence 3: {prom3}")
        # Debugging print statements
        # print("High Peaks Indices:", high_peaks)
        # print("Original High Values:", dataframe['high'].iloc[high_peaks])
        # print("Assigned Prominences:", -high_properties["prominences"].astype('float'))

   
        '''comment this out if you don't want signal messages to Telegram'''
        pair = metadata['pair']
        if dataframe['peak'].iloc[-1] < 0:
            self.dp.send_msg(f'*** {pair} *** Maxima Detected - Prominence: {prom}')
            logger.info(f"*** {pair} *** Maxima Detected - Prominence: {prom}")

        if dataframe['peak'].iloc[-1] > 0:
            self.dp.send_msg(f'*** {pair} *** Minima Detected - Prominence: {prom}')
            logger.info(f"*** {pair} *** Maxima Detected - Prominence: {prom}")

        # last_few_rows_column_A = dataframe['peak_prominance'].tail()
        # print(pair, prom, prom1, prom2, prom3 ,last_few_rows_column_A)

        
        # CUSUM
        close_prices = torch.tensor(dataframe['close'].values, dtype=torch.float)
        detector = CusumMeanDetector()
        
        # Calculate probabilities and changepoint flags
        probs = []
        is_changepoints = []
        for price in close_prices:
            prob, is_changepoint = detector.predict_next(price)
            probs.append(prob)
            is_changepoints.append(is_changepoint)

        # Add to dataframe
        dataframe['cusum_prob'] = probs
        dataframe['cusum_is_changepoint'] = is_changepoints
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        condition121 = (                
            (dataframe['cusum_is_changepoint'] == True) &

            (dataframe['peak_prominance'] <= dataframe['negative_expanding_mean']) &
            # (dataframe['order'] < (self.peaks.value * self.quick.value)) &
            (self.enable1.value == True) 
            # (dataframe['close'] < dataframe['sma_dn'])
        )

        dataframe.loc[condition121, 'enter_short'] = 1
        dataframe.loc[condition121, 'enter_tag'] = 'minima_check_quick_dn short'

        condition22 = (
            (dataframe['cusum_is_changepoint'] == True) &
            (dataframe['volume'] > 0) &
            (dataframe['williamr'] <= -90) & 
            (dataframe['williamr'].shift(10) >= -85)  
        )

        dataframe.loc[condition22, 'enter_long'] = 1
        dataframe.loc[condition22, 'enter_tag'] = '2 Trend Soon'

      
        
        condition101 = (
            (dataframe['cusum_is_changepoint'] == True) &
            (dataframe['volume'] > 0) &
            (dataframe['bb_percent'] < 0.1) &
            (dataframe['bb_width'] > 0.02) |
            # Candlestick patterns
            (dataframe['cdl3inside'] == 100) | # 3 Inside Up/Down
            (dataframe['cdl3outside'] == 100) | # 3 Outside Up/Down
            (dataframe['cdl3starsinsouth'] == 100) # 3 Stars In The South
        )
        
        dataframe.loc[condition101, 'enter_long'] = 1
        dataframe.loc[condition101, 'enter_tag'] = 'Big Up Trend 2'
         # Reverse of condition1 for entering short


        condition711 = (
            (dataframe['cusum_is_changepoint'] == True) &
            (dataframe['williamr'] <= -10) & 
            (dataframe['williamr'].shift(6) >= -20) & 
            (dataframe['OHLC4'] > dataframe['sma_up']) &
            (dataframe['volume'] > 0) &
            (dataframe['bb_percent'] > 0.9) &
            (dataframe['bb_width'] > 0.03) |
            # Candlestick patterns
            (dataframe['cdl3blackcrows'] == -100) | # 3 Black Crows
            (dataframe['cdl3whitesoldiers'] == -100) | # 3 White Soldiers
            (dataframe['cdl3linestrike'] == -100) # 3 Line Strike
            )

        dataframe.loc[condition711, 'enter_short'] = 1
        dataframe.loc[condition711, 'enter_tag'] = 'Down Trend Soon1'
        
        
        condition1 = (
            (dataframe['cusum_is_changepoint'] == True) &
            (dataframe['peak_prominance'] >= dataframe['positive_expanding_mean']) &
            # (dataframe['order'] < (self.peaks.value * self.quick.value)) &
            (self.enable1.value == True) &
            (dataframe['close'] < dataframe['sma_dn'])
        )

        dataframe.loc[condition1, 'enter_long'] = 1
        dataframe.loc[condition1, 'enter_tag'] = 'minima_check'

        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        condition7 = (
                (dataframe['cusum_is_changepoint'] == True) &
                (dataframe['peak_prominance'] <= dataframe['negative_expanding_mean']) 
            )

        dataframe.loc[condition7, 'exit_long'] = 1
        dataframe.loc[condition7, 'exit_tag'] = 'maxima_check'

        condition8 = (
                (dataframe['cusum_is_changepoint'] == True) &
                (dataframe['peak_prominance'] < 0)&
                (dataframe['close'] < dataframe['sma'])
                
            )

        dataframe.loc[condition8, 'exit_long'] = 1
        dataframe.loc[condition8, 'exit_tag'] = 'maxima_check_below_sma'

        condition9 = (
                (dataframe['cusum_is_changepoint'] == True) &
                (dataframe['peak_prominance'].shift() <= dataframe['negative_expanding_mean'].shift()) 
            )

        dataframe.loc[condition9, 'exit_long'] = 1
        dataframe.loc[condition9, 'exit_tag'] = 'maxima_check'

        condition10 = (
                (dataframe['cusum_is_changepoint'] == True) &
                (dataframe['peak_prominance'].shift() < 0)&
                (dataframe['close'].shift() < dataframe['sma'].shift())
                
            )

        dataframe.loc[condition10, 'exit_long'] = 1
        dataframe.loc[condition10, 'exit_tag'] = 'maxima_check_below_sma'

        return dataframe

