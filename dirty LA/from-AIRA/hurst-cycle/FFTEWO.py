import logging
from functools import reduce
import datetime
import talib.abstract as ta
import pandas_ta as pta
import logging
import numpy as np
import pandas as pd
import time
import math
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical import qtpylib
from datetime import timedelta, datetime, timezone
from pandas import DataFrame, Series
from technical import qtpylib
from typing import Optional
from freqtrade.strategy.interface import IStrategy
from technical.pivots_points import pivots_points
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
from scipy.signal import argrelextrema
from typing import Optional, Union, Tuple
from functools import reduce


logger = logging.getLogger(__name__)

class FFTEWO(IStrategy):
    exit_profit_only = False ### No selling at a loss
    use_custom_stoploss = True
    trailing_stop = False
    position_adjustment_enable = False
    ignore_roi_if_entry_signal = True
    process_only_new_candles = True
    can_short = False
    use_exit_signal = True
    startup_candle_count = 200
    stoploss = -0.15
    timeframe = '5m'

    locked_stoploss = {}
    minimal_roi = {}

    plot_config = {}

    # Threshold and Limits
    u_window_size = IntParameter(80, 140, default=140, space='buy', optimize=True)
    l_window_size = IntParameter(20, 60, default=60, space='buy', optimize=True)
    ewo_bear_x = DecimalParameter(1.62, 1.80, default=1.618, decimals=2, space='buy', optimize=True)
    ewo_bull_x = DecimalParameter(1.01, 1.30, default=1.15, decimals=2, space='buy', optimize=True)
    ewo_high_x = DecimalParameter(1.30, 1.40, default=1.35, decimals=2, space='buy', optimize=True)
    ewo_lo_limit = DecimalParameter(-5, 0, default=-2.5, decimals=1, space='buy', optimize=True)
    rsi_buy_low = IntParameter(30, 50, default=35, space='buy', optimize=True)
    rsi_buy_high = IntParameter(50, 70, default=58, space='buy', optimize=True)

    # Custom Entry
    increment = DecimalParameter(low=1.0005, high=1.002, default=1.001, decimals=4 ,space='buy', optimize=True, load=True)
    last_entry_price = None

    use0 = BooleanParameter(default=False, space="buy", optimize=True, load=True)
    use1 = BooleanParameter(default=False, space="buy", optimize=True, load=True)
    use2 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use3 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use4 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use5 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use6 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use7 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use8 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use9 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use10 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use11 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use12 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use13 = BooleanParameter(default=True, space="sell", optimize=True, load=True)

    # CooldownPeriod 
    cooldown_lookback = IntParameter(0, 48, default=5, space="protection", optimize=True)
    
    # StoplossGuard    
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    stop_protection_only_per_pair = BooleanParameter(default=False, space="protection", optimize=True)
    stop_protection_only_per_side = BooleanParameter(default=False, space="protection", optimize=True)
    stop_protection_trade_limit = IntParameter(1, 10, default=4, space="protection", optimize=True)
    stop_protection_required_profit = DecimalParameter(-1.0, 3.0, default=0.0, space="protection", optimize=True)

    # LowProfitPairs    
    use_lowprofit_protection = BooleanParameter(default=True, space="protection", optimize=True)
    lowprofit_protection_lookback = IntParameter(1, 10, default=6, space="protection", optimize=True)
    lowprofit_trade_limit = IntParameter(1, 10, default=4, space="protection", optimize=True)
    lowprofit_stop_duration = IntParameter(1, 100, default=60, space="protection", optimize=True)
    lowprofit_required_profit = DecimalParameter(-1.0, 3.0, default=0.0, space="protection", optimize=True)
    lowprofit_only_per_pair = BooleanParameter(default=False, space="protection", optimize=True)


    # MaxDrawdown    
    use_maxdrawdown_protection = BooleanParameter(default=True, space="protection", optimize=True)
    maxdrawdown_protection_lookback = IntParameter(1, 10, default=6, space="protection", optimize=True)
    maxdrawdown_trade_limit = IntParameter(1, 20, default=10, space="protection", optimize=True)
    maxdrawdown_stop_duration = IntParameter(1, 100, default=6, space="protection", optimize=True)
    maxdrawdown_allowed_drawdown = DecimalParameter(0.01, 0.10, default=0.0, space="protection", optimize=True)

    ### protections ###
    @property
    def protections(self):
    
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,
                "trade_limit": self.stop_protection_trade_limit.value,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": self.stop_protection_only_per_pair.value,
                "required_profit": self.stop_protection_required_profit.value,
            "only_per_side": self.stop_protection_only_per_side.value
        })

        if self.use_lowprofit_protection.value:
            prot.append({
                    "method": "LowProfitPairs",
                    "lookback_period_candles": self.lowprofit_protection_lookback.value,
                    "trade_limit": self.lowprofit_trade_limit.value,
                    "stop_duration_candles": self.lowprofit_stop_duration.value,
                    "required_profit": self.lowprofit_required_profit.value,
                    "only_per_pair": self.lowprofit_only_per_pair.value
        })

        if self.use_maxdrawdown_protection.value:
            prot.append({
                    "method": "MaxDrawdown",
                    "lookback_period_candles": self.maxdrawdown_protection_lookback.value,
                    "trade_limit": self.maxdrawdown_trade_limit.value,
                    "stop_duration_candles": self.maxdrawdown_stop_duration.value,
                    "max_allowed_drawdown": self.maxdrawdown_allowed_drawdown.value
        })

        return prot    

    def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)

        entry_price = (dataframe['close'].iat[-1] + dataframe['open'].iat[-1] + dataframe['low'].iat[-1] + proposed_rate) / 4
        logger.info(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}")
        self.dp.send_msg(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}")

        # Check if there is a stored last entry price and if it matches the proposed entry price
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.0001:  # Tolerance for floating-point comparison
            entry_price *= self.increment.value # Increment by 0.2%%
            logger.info(f"{pair} Incremented entry price: {entry_price} based on previous entry price : {self.last_entry_price}.")

        # Update the last entry price
        self.last_entry_price = entry_price

        return entry_price

    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()


        SLT0 = current_candle['h2_move_mean']
        SLT1 = current_candle['h1_move_mean']
        SLT2 = current_candle['h0_move_mean']
        SLT3 = current_candle['cycle_move_mean']

        
        display_profit = current_profit * 100

        if current_profit < -0.01:
            if pair in self.locked_stoploss:
                del self.locked_stoploss[pair]
                self.dp.send_msg(f'*** {pair} *** Stoploss reset.')
                logger.info(f'*** {pair} *** Stoploss reset.')
            return self.stoploss

        new_stoploss = None
        if SLT3 is not None and current_profit > SLT3:
            new_stoploss = (SLT2 - SLT1)
            level = 4
        elif SLT2 is not None and current_profit > SLT2:
            new_stoploss = (SLT2 - SLT1)
            level = 3

        # in the future toggle these on certain conditions with indicators.
        elif SLT1 is not None and current_profit > SLT1 and self.use8.value == True:
            new_stoploss = (SLT1 - SLT0)
            level = 2
        elif SLT0 is not None and current_profit > SLT0 and self.use9.value == True:
            new_stoploss = (SLT1 - SLT0)
            level = 1

        if new_stoploss is not None:
            if pair not in self.locked_stoploss or new_stoploss > self.locked_stoploss[pair]:
                self.locked_stoploss[pair] = new_stoploss
                self.dp.send_msg(f'*** {pair} *** Profit {level} {display_profit:.3f}%% - New stoploss: {new_stoploss:.4f} activated')
                logger.info(f'*** {pair} *** Profit {level} {display_profit:.3f}%% - New stoploss: {new_stoploss:.4f} activated')
            return self.locked_stoploss[pair]


        return self.stoploss

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                       rate: float, time_in_force: str, exit_reason: str,
                       current_time: datetime, **kwargs) -> bool:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_fill = (current_time - trade.date_last_filled_utc).seconds / 60 

        if exit_reason == 'roi' and trade.calc_profit_ratio(rate) < 0.002:
            logger.info(f"{trade.pair} ROI is below 0%")
            # self.dp.send_msg(f'{trade.pair} ROI is below 0')
            return False

        if exit_reason == 'partial_exit' and trade.calc_profit_ratio(rate) < 0.002:
            logger.info(f"{trade.pair} partial exit is below 0%")
            # self.dp.send_msg(f'{trade.pair} partial exit is below 0')
            return False

        ### May Need This ###
        # if exit_reason == 'trailing_stop_loss' and trade.calc_profit_ratio(rate) < 0.002:
        #     logger.info(f"{trade.pair} partial exit is below 0%")
        #     # self.dp.send_msg(f'{trade.pair} partial exit is below 0')
        #     return False


        return True


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        start_time = time.time()
        pair = metadata['pair']

        heikinashi = qtpylib.heikinashi(dataframe)

        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # Initialize Hurst Cycles for startup errors
        cycle_period = 80
        harmonics = [0, 0, 0]
        harmonics[0] = 40
        harmonics[1] = 27
        harmonics[2] = 20

        if len(dataframe) < self.u_window_size.value:
            raise ValueError(f"Insufficient data points for FFT: {len(dataframe)}. Need at least {self.u_window_size.value} data points.")

        # Perform FFT to identify cycles with a rolling window
        freq, power = perform_fft(dataframe['ha_close'], window_size=self.u_window_size.value)

        if len(freq) == 0 or len(power) == 0:
            raise ValueError("FFT resulted in zero or invalid frequencies. Check the data or the FFT implementation.")

        # Filter out the zero-frequency component and limit the frequency to below 500
        # positive_mask = (freq > 0) & (1 / freq < self.u_window_size.value)
        positive_mask = (1 / freq > self.l_window_size.value) & (1 / freq < self.u_window_size.value)
        positive_freqs = freq[positive_mask]
        positive_power = power[positive_mask]

        # Convert frequencies to periods
        cycle_periods = 1 / positive_freqs

        # Set a threshold to filter out insignificant cycles based on power
        power_threshold = 0.01 * np.max(positive_power)
        significant_indices = positive_power > power_threshold
        significant_periods = cycle_periods[significant_indices]
        significant_power = positive_power[significant_indices]

        # Identify the dominant cycle
        dominant_freq_index = np.argmax(significant_power)
        dominant_freq = positive_freqs[dominant_freq_index]
        # logger.info(f'{pair} Hurst Exponent: {dominant_freq}')
        cycle_period = int(np.abs(1 / dominant_freq)) if dominant_freq != 0 else 100

        if cycle_period == np.inf:
            raise ValueError("No dominant frequency found. Check the data or the method used.")

        # Calculate harmonics for the dominant cycle
        harmonics = [cycle_period / (i + 1) for i in range(1, 4)]
        # print(cycle_period, harmonics)
        self.cp = int(cycle_period)
        self.h0 = int(harmonics[0])
        self.h1 = int(harmonics[1])
        self.h2 = int(harmonics[2])

        dataframe['dc_EWM'] = dataframe['ha_close'].ewm(span=int(cycle_period)).mean()
        dataframe['dc_1/2'] = dataframe['ha_close'].ewm(span=int(harmonics[0])).mean()
        dataframe['dc_1/3'] = dataframe['ha_close'].ewm(span=int(harmonics[1])).mean()
        dataframe['dc_1/4'] = dataframe['ha_close'].ewm(span=int(harmonics[2])).mean()

        # Apply rolling window operation to the 'OHLC4' column
        rolling_windowc = dataframe['ha_close'].rolling(cycle_period) 
        rolling_windowh0 = dataframe['ha_close'].rolling(int(harmonics[0]))
        rolling_windowh1 = dataframe['ha_close'].rolling(int(harmonics[1])) 
        rolling_windowh2 = dataframe['ha_close'].rolling(int(harmonics[2])) 

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_valuec = rolling_windowc.apply(lambda x: np.ptp(x))
        ptp_valueh0 = rolling_windowh0.apply(lambda x: np.ptp(x))
        ptp_valueh1 = rolling_windowh1.apply(lambda x: np.ptp(x))
        ptp_valueh2 = rolling_windowh2.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['cycle_move'] = ptp_valuec / dataframe['ha_close']
        dataframe['h0_move'] = ptp_valueh0 / dataframe['ha_close']
        dataframe['h1_move'] = ptp_valueh1 / dataframe['ha_close']
        dataframe['h2_move'] = ptp_valueh2 / dataframe['ha_close']

        dataframe['cycle_move_mean'] = dataframe['cycle_move'].rolling(self.cp).mean()        
        dataframe['h0_move_mean'] = dataframe['h0_move'].rolling(self.cp).mean()
        dataframe['h1_move_mean'] = dataframe['h1_move'].rolling(self.cp).mean() 
        dataframe['h2_move_mean'] = dataframe['h2_move'].rolling(self.cp).mean()

        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.h2)
        dataframe["atrpercent"] = (dataframe["atr"] / dataframe['close']) * 100

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.h2, self.cp)

        # EMAs for Bear/Bull
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.h2)
        dataframe['ema2'] = ta.EMA(dataframe, timeperiod=self.cp)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.h2)  # 14)
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'], timeperiod=(int(self.h2 / 2)))  # 5)

        # absolute ewo
        dataframe["ABSEWO"] = dataframe["EWO"]
        dataframe.loc[dataframe["EWO"] < 0, "ABSEWO"] = dataframe["EWO"] * - 1
        # seperate pos and neg ewo
        dataframe.loc[dataframe["EWO"] > 0, "POSEWO"] = dataframe["EWO"]
        dataframe.loc[dataframe["EWO"] < 0, "NEGEWO"] = dataframe["EWO"]
        dataframe[["POSEWO", "NEGEWO"]] = dataframe[["POSEWO", "NEGEWO"]].ffill()

        # angle of the ema
        candlesBack = 3
        backQuote = dataframe['ema'].shift(candlesBack)
        deltaX = candlesBack
        deltaY = dataframe['ema'] - backQuote
        dataframe['angleRad'] = (deltaY / deltaX).apply(lambda x: math.atan(x))
        dataframe['angle_perc'] = (dataframe['angleRad'] / dataframe['close']) * 100
        threshold = 0  # .0000001
        dataframe["ema_angle_up"] = dataframe['angleRad'] > 0 + threshold
        dataframe["ema_angle_down"] = dataframe['angleRad'] < 0 - threshold

        # ewo normalized
        dataframe['min'] = dataframe["EWO"].rolling(self.cp).min()
        dataframe['max'] = dataframe["EWO"].rolling(self.cp).max()
        dataframe['dif'] = dataframe['max'].sub(dataframe['min'])
        dataframe["ewoNorm"] = dataframe["EWO"] / dataframe['dif']
        dataframe['ewoNormMa'] =  ta.SMA(dataframe["ewoNorm"], timeperiod=self.h2)
        dataframe['ewoNormMaHigh'] = 0.0618

        # angle of the ewoNorm
        candlesBack2 = 1
        backQuote2 = dataframe['ewoNorm'].shift()
        deltaX2 = candlesBack2
        deltaY2 = dataframe['ewoNorm'] - backQuote2
        dataframe['ewoAngleRad'] = (deltaY2 / deltaX2).apply(lambda x: math.atan(x))
        dataframe["ewo_angle_up"] = dataframe['ewoAngleRad'] > -0.0001
        dataframe["ewo_angle_down"] = dataframe['ewoAngleRad'] < 0.0001

        emaPercent = dataframe['ema'] / dataframe['ema2']

        avg_positive_ewo = dataframe["POSEWO"].rolling(self.cp).mean()
        avg_negative_ewo = dataframe["NEGEWO"].rolling(self.cp).mean()
        avg_abs_ewo = dataframe["ABSEWO"].rolling(self.cp).mean()

        # scale the buys. if we bear then push back the buys if we bull make more buys
        dataframe["ewo_low_mult"] = self.ewo_bull_x.value  # / emaPercent # 0.618
        dataframe.loc[dataframe["ema"] < dataframe["ema2"], "ewo_low_mult"] = self.ewo_bear_x.value

        # for buy below rolling ewo low
        ewo_low = avg_negative_ewo * dataframe["ewo_low_mult"] 
        dataframe['ewo_low'] = ewo_low
        # for rsi buy
        ewo_high = avg_positive_ewo * self.ewo_high_x.value
        dataframe['ewo_high'] = ewo_high

        dataframe['ewo_limit'] = self.ewo_lo_limit.value

        dataframe['ma_lo'] = dataframe['dc_EWM'] * (1 - dataframe['h2_move_mean'])
        dataframe['ma_hi'] = dataframe['dc_EWM'] * (1 + dataframe['h2_move_mean'])

        if not self.dp.runmode.value in ("backtest", "plot", "hyperopt"):
            logger.info(f'{pair} - DC: {cycle_period:.2f} | 1/2: {harmonics[0]:.2f} | 1/3: {harmonics[1]:.2f} | 1/4: {harmonics[2]:.2f}')
            end_time = time.time()
            logger.info(f"Indicators done for {pair} in {end_time - start_time:.2f} secs")

        return dataframe


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:


        df.loc[
            (
                (df['EWO'] > df['ewo_high']) &
                (df['close'] < df['ma_lo']) &
                (self.use0.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO > HIGH')

        df.loc[
            (
                (df['EWO'] < df['ewo_high']) &
                (df['close'] < df['ma_lo']) &
                (df["ewo_angle_up"]) &
                (self.use1.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO < HIGH')

        df.loc[
            (
                (qtpylib.crossed_above(df['EWO'], df['ewo_high'])) &
                (df['close'] < df['ma_lo']) &
                (df["ewo_angle_up"]) &
                (df['rsi'] < self.rsi_buy_high.value) &
                (self.use2.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO XO HIGH')

        df.loc[
            (
                (qtpylib.crossed_above(df['EWO'], df['ewo_low'])) &
                (df['close'] < df['ma_lo']) &
                (df['EWO'] < self.ewo_lo_limit.value) &
                (df['rsi'] < self.rsi_buy_low.value) &
                (self.use3.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO XO LOW')

        df.loc[
            (
                (df['EWO'] < df['ewo_low']) &
                (df['close'] < df['ma_lo']) &
                (df['EWO'] < self.ewo_lo_limit.value) &
                (df['rsi'] < self.rsi_buy_low.value) &
                (self.use4.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO < LOW')

        df.loc[
            (
                ((df['angle_perc'] - df['angle_perc'].shift(1)) > 0.07 ) &
                ((df['angle_perc'].shift(1) - df['angle_perc'].shift(2)) < 0 ) &
                (df['angle_perc'] < 0.17) &
                (df['angle_perc'] > 0) &
                (df['close'] > df['ma_hi']) &
                (self.use5.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'Angle Change Pos')

        df.loc[
            (
                ((df['angle_perc'] - df['angle_perc'].shift(1)) > 0.07 ) &
                ((df['angle_perc'].shift(1) - df['angle_perc'].shift(2)) < 0 ) &
                (df['angle_perc'] < 0) &
                (df['open'] <  df['ma_hi']) &
                (self.use6.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'Angle Change Neg')

        return df


    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['close'] > df['ma_hi']) &
                (df['EWO'] > df['ewo_high']) &
                # (df['angle_perc'] < 0.10) &
                (df['EWO'] > 0) &
                (self.use10.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'EWO High')

        df.loc[
            (
                (df['close'] > df['ma_hi']) &
                (df['rsi'] >= 89) &
                (df['EWO'] > 0) &
                (self.use11.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'EWO rsi limit')

        df.loc[
            (
                (df['close'] > df['ma_hi']) &
                (df['rsi'] >= 89) &
                (df['EWO'] < 0) &
                (self.use12.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'EWO low rsi limit')

        return df


def perform_fft(price_data, window_size=None):
    if window_size is not None:
        # Apply rolling window to smooth the data
        price_data = price_data.rolling(window=window_size, center=True).mean().dropna()

    normalized_data = (price_data - np.mean(price_data)) / np.std(price_data)
    n = len(normalized_data)
    fft_data = np.fft.fft(normalized_data)
    freq = np.fft.fftfreq(n)
    power = np.abs(fft_data) ** 2
    power[np.isinf(power)] = 0
    return freq, power


def EWO(dataframe, ema_length=8, ema2_length=89):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['ha_close'] * 100
    return emadif
