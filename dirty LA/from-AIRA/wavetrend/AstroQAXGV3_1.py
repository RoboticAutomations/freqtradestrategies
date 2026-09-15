import logging
from functools import reduce
import datetime
import ephem
import talib.abstract as ta
import pandas_ta as pta
import logging
import numpy as np
import pandas as pd
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
from typing import Optional
from functools import reduce

logger = logging.getLogger(__name__)

class AstroQAXGV3_1(IStrategy):
    exit_profit_only = True ### No selling at a loss
    use_custom_stoploss = True
    trailing_stop = True
    position_adjustment_enable = True
    ignore_roi_if_entry_signal = True
    position_adjustment_enable = True
    max_entry_position_adjustment = 3
    max_dca_multiplier = 4.75
    process_only_new_candles = True
    can_short = False
    use_exit_signal = True
    startup_candle_count: int = 50
    stoploss = -0.99
    timeframe = '5m'

    minimal_roi = {

        "48000": 0.01,
        "24000": 0.025,
        "12000": 0.05,
        "2400": 0.10,
        "300": 0.15,
        "180": 0.30,
        "120":0.40,
        "60": 0.45,
        "0": 0.50
    }

    plot_config = {
        "main_plot": {},
        "subplots": {
            "extrema": {
                "&s-extrema": {
                    "color": "#f53580",
                    "type": "line"
                },
                "&s-minima_sort_threshold": {
                    "color": "#4ae747",
                    "type": "line"
                },
                "&s-maxima_sort_threshold": {
                    "color": "#5b5e4b",
                    "type": "line"
                }
            },
            "min_max": {
                "maxima": {
                    "color": "#a29db9",
                    "type": "line"
                },
                "minima": {
                    "color": "#ac7fc",
                    "type": "bar"
                }
            }
        }
    }

    # protections
    cooldown_lookback = IntParameter(24, 48, default=12, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    #trailing stop loss optimiziation
    tsl_target5 = DecimalParameter(low=0.3, high=0.4, default=0.3, decimals=2, space='sell', optimize=True, load=True)
    ts5 = DecimalParameter(low=0.04, high=0.06, default=0.05,decimals=2, space='sell', optimize=True, load=True)
    tsl_target4 = DecimalParameter(low=0.18, high=0.3, default=0.2, decimals=2, space='sell', optimize=True, load=True)
    ts4 = DecimalParameter(low=0.03, high=0.05, default=0.045, decimals=3, space='sell', optimize=True, load=True)
    tsl_target3 = DecimalParameter(low=0.12, high=0.18, default=0.15, decimals=2,space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, decimals=3,space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.07, high=0.12, default=0.1, decimals=2,space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, decimals=2,space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.04, high=0.07, default=0.06, decimals=3,space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, decimals=3,space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.02, high=0.05, default=0.04, decimals=3 ,space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.008, high=0.015, default=0.01, decimals=3,space='sell', optimize=True, load=True)

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
                "trade_limit": 2,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False
            })

        return prot

    ### Dollar Cost Averaging ###
    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        # We need to leave most of the funds for possible further DCA orders
        # This also applies to fixed stakes
        return proposed_stake / self.max_dca_multiplier 

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        # Take Profit
        if current_profit > 0.10 and trade.nr_of_successful_exits == 0:
            # Take half of the profit at +10%
            return -(trade.stake_amount / 2)

        # Profit Based DCA    
        if current_profit > -0.0125 and trade.nr_of_successful_entries == 1:
            return None

        if current_profit > -0.03125 and trade.nr_of_successful_entries == 2:
            return None

        if current_profit > -0.09 and trade.nr_of_successful_entries == 3:
            return None

        if current_profit > -0.13 and trade.nr_of_successful_entries == 4:
            return None

        if current_profit > -0.18 and trade.nr_of_successful_entries == 5:
            return None

        # Time Based DCA
        if current_profit > -0.0125:
            #12 hrs
            if trade_duration > 720 and trade.nr_of_successful_entries == 1:
                return None
            #24    
            if trade_duration > 1440 and trade.nr_of_successful_entries == 2:
                return None
            #48    
            if trade_duration > 2880 and trade.nr_of_successful_entries == 3:
                return None
            #96
            if trade_duration > 5760 and trade.nr_of_successful_entries == 4:
                return None

            if trade_duration > 11520 and trade.nr_of_successful_entries == 5:
                return None


        try:
            # This returns first order stake size
            stake_amount = filled_entries[0].cost
            # This then calculates current safety order size
            if count_of_entries == 1: 
                stake_amount = stake_amount * 1.125
            elif count_of_entries == 2:
                stake_amount = stake_amount * 1.25
            elif count_of_entries == 3:
                stake_amount = stake_amount * 1.375
            elif count_of_entries == 4:
                stake_amount = stake_amount * 1.5
            elif count_of_entries == 5:
                stake_amount = stake_amount * 1.625
            else:
                stake_amount = stake_amount

            return stake_amount
        except Exception as exception:
            return None

        return None


    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        for stop5 in self.tsl_target5.range:
            if (current_profit > stop5):
                for stop5a in self.ts5.range:
                    self.dp.send_msg(f'*** {pair} *** Profit: {current_profit} - lvl5 {stop5}/{stop5a} activated')
                    return stop5a 
        for stop4 in self.tsl_target4.range:
            if (current_profit > stop4):
                for stop4a in self.ts4.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl4 {stop4}/{stop4a} activated')
                    return stop4a 
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

        return self.stoploss


    def compute_planet_position(self, planet_name, date):
        observer = ephem.Observer()
        observer.lat = '33.4484'  # Latitude of Phoenix, Arizona
        observer.lon = '-112.0740'  # Longitude of Phoenix, Arizona

        planet = getattr(ephem, planet_name)(observer)
        planet.compute(date)

        return planet.ra, planet.dec


    def feature_engineering_expand_all(self, dataframe, period, **kwargs):
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-tcp-period"] = top_percent_change(dataframe, period)
        dataframe["%-chop-period"] = qtpylib.chopiness(dataframe, period)
        dataframe["%-linear-period"] = ta.LINEARREG_ANGLE(
            dataframe['close'], timeperiod=period)
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        dataframe["%-atr-periodp"] = dataframe["%-atr-period"] / \
            dataframe['close'] * 1000
        return dataframe


    def feature_engineering_expand_basic(self, dataframe, **kwargs):
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-obv"] = ta.OBV(dataframe)
        # Fib EMA
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_34'] = ta.EMA(dataframe, timeperiod=34)
        dataframe['%-ewo'] = EWO(dataframe, dataframe['ema_8'], dataframe['ema_34'])
        dataframe['%-distema8'] = get_distance(dataframe['close'], dataframe['ema_8'])
        dataframe['%-distema34'] = get_distance(dataframe['close'], dataframe['ema_34'])

        dataframe['%-HLC3'] = (dataframe['high'] + dataframe['low'] + dataframe['close'])/3


        # TTM Squeeze
        ttm_Squeeze = pta.squeeze(high = dataframe['high'], low = dataframe['low'], close = dataframe["close"], lazybear = True)
        dataframe['%-ttm_Squeeze'] = ttm_Squeeze['SQZ_20_2.0_20_1.5_LB']
        dataframe['%-ttm_ema'] = ta.EMA(dataframe['%-ttm_Squeeze'], timeperiod = 4)
        dataframe['%-squeeze_ON'] = ttm_Squeeze['SQZ_ON']
        dataframe['%-squeeze_OFF'] = ttm_Squeeze['SQZ_OFF']
        dataframe['%-NO_squeeze'] = ttm_Squeeze['SQZ_NO']

        # Calculate the percentage change between the high and open prices for each 5-minute candle
        dataframe['%-perc_change'] = (dataframe['high'] / dataframe['open'] - 1) * 100

        # Create a custom indicator that checks if any of the past 100 5-minute candles' high price is 3% or more above the open price
        dataframe['%-candle_1perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x <= 1, 1, 0).sum()).shift()
        dataframe['%-candle_2perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x <= 2, 1, 0).sum()).shift()
        dataframe['%-candle_3perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x <= 3, 1, 0).sum()).shift()

        dataframe['%-candle_-1perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x <= -1, 1, 0).sum()).shift()
        dataframe['%-candle_-2perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x <= -2, 1, 0).sum()).shift()
        dataframe['%-candle_-3perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x <= -3, 1, 0).sum()).shift()

        # Calculate the percentage of the current candle's range where the close price is
        dataframe['%-close_percentage'] = (dataframe['close'] - dataframe['low']) / (dataframe['high'] - dataframe['low'])

        dataframe['%-body_size'] = abs(dataframe['open'] - dataframe['close'])
        dataframe['%-range_size'] = dataframe['high'] - dataframe['low']
        dataframe['%-body_range_ratio'] = dataframe['%-body_size'] / dataframe['%-range_size']

        dataframe['%-upper_wick_size'] = dataframe['high'] - dataframe[['open', 'close']].max(axis=1)
        dataframe['%-upper_wick_range_ratio'] = dataframe['%-upper_wick_size'] / dataframe['%-range_size']
        
        lookback_period = 10
        dataframe['%-max_high'] = dataframe['high'].rolling(50).max()
        dataframe['%-min_low'] = dataframe['low'].rolling(50).min()
        dataframe['%-close_position'] = (dataframe['close'] - dataframe['%-min_low']) / (dataframe['%-max_high'] - dataframe['%-min_low'])

        dataframe['%-current_candle_perc_change'] = (dataframe['high'] / dataframe['open'] - 1) * 100

        # Impulse Macd
        dataframe['%-hi'] = ta.SMA(dataframe['high'], timeperiod = 28)
        dataframe['%-lo'] = ta.SMA(dataframe['low'], timeperiod = 28)
        dataframe['%-ema1'] = ta.EMA(dataframe['%-HLC3'], timeperiod = 28)
        dataframe['%-ema2'] = ta.EMA(dataframe['%-ema1'], timeperiod = 28)
        dataframe['%-d'] = dataframe['%-ema1'] - dataframe['%-ema2']
        dataframe['%-mi'] = dataframe['%-ema1'] + dataframe['%-d']
        dataframe['%-md'] = np.where(dataframe['%-mi'] > dataframe['%-hi'], 
            dataframe['%-mi'] - dataframe['%-hi'], 
            np.where(dataframe['%-mi'] < dataframe['%-lo'], 
            dataframe['%-mi'] - dataframe['%-lo'], 0))
        dataframe['%-sb'] = ta.SMA(dataframe['%-md'], timeperiod = 8)
        dataframe['%-sh'] = dataframe['%-md'] - dataframe['%-sb']
        
        # WaveTrend using OHLC4 or HA close - 3/21
        ap = (0.25 * (dataframe['high'] + dataframe['low'] + dataframe["close"] + dataframe["open"]))
        
        dataframe['esa'] = ta.EMA(ap, timeperiod = 3)
        dataframe['d'] = ta.EMA(abs(ap - dataframe['esa']), timeperiod = 3)
        dataframe['%-wave_ci'] = (ap-dataframe['esa']) / (0.015 * dataframe['d'])
        dataframe['%-wave_t1'] = ta.EMA(dataframe['%-wave_ci'], timeperiod = 21)  
        dataframe['%-wave_t2'] = ta.SMA(dataframe['%-wave_t1'], timeperiod = 4)
        # VWAP
        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['%-vwap_upperband'] = vwap_high
        dataframe['%-vwap_middleband'] = vwap
        dataframe['%-vwap_lowerband'] = vwap_low
        dataframe['%-vwap_width'] = ((dataframe['%-vwap_upperband'] -
                                     dataframe['%-vwap_lowerband']) / dataframe['%-vwap_middleband']) * 100
        dataframe = dataframe.copy()
        dataframe['%-dist_to_vwap_upperband'] = get_distance(
            dataframe['close'], dataframe['%-vwap_upperband'])
        dataframe['%-dist_to_vwap_middleband'] = get_distance(
            dataframe['close'], dataframe['%-vwap_middleband'])
        dataframe['%-dist_to_vwap_lowerband'] = get_distance(
            dataframe['close'], dataframe['%-vwap_lowerband'])
        dataframe['%-tail'] = (dataframe['close'] - dataframe['low']).abs()
        dataframe['%-wick'] = (dataframe['high'] - dataframe['close']).abs()
        dataframe['rawclose'] = dataframe['close']
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_low"] = dataframe["low"]
        dataframe["%-raw_high"] = dataframe["high"]
    
        # Convert timestamp to ephem date format (UTC time)
        dataframe['e_date'] = dataframe['date'].apply(lambda x: ephem.Date(x.strftime("%Y/%m/%d %H:%M:%S")))

        # Calculate moon phase
        moon_phase = dataframe['e_date'].apply(lambda x: ephem.Moon(x).phase)

        # Calculate planet transits through astrological signs
        planets = {
            'Sun': ephem.Sun,
            'Mercury': ephem.Mercury,
            'Venus': ephem.Venus,
            'Mars': ephem.Mars,
            'Jupiter': ephem.Jupiter,
            'Saturn': ephem.Saturn,
            'Uranus': ephem.Uranus,
            'Neptune': ephem.Neptune,
            'Pluto': ephem.Pluto,
            'Moon': ephem.Moon,
        }

        for planet_name, planet in planets.items():
            dataframe[planet_name + '_ra'], dataframe[planet_name + '_dec'] = zip(*dataframe['e_date'].apply(
                lambda x: self.compute_planet_position(planet_name, x)
            ))
            dataframe[planet_name] = dataframe['e_date'].apply(lambda x: ephem.constellation(planet(x))[1])

        # Scale Orbits to 100
        dataframe['%-Moon_Phase'] = moon_phase
        dataframe['%-Sun_rax'] = dataframe['Sun_ra'] * 15.923566879
        dataframe['%-Mercury_rax'] = dataframe['Mercury_ra'] * 15.923566879
        dataframe['%-Venus_rax'] = dataframe['Venus_ra'] * 15.923566879
        dataframe['%-Mars_rax'] = dataframe['Mars_ra'] * 15.923566879
        dataframe['%-Jupiter_rax'] = dataframe['Jupiter_ra'] * 15.923566879
        dataframe['%-Saturn_rax'] = dataframe['Saturn_ra'] * 15.923566879
        dataframe['%-Uranus_rax'] = dataframe['Uranus_ra'] * 15.923566879
        dataframe['%-Neptune_rax'] = dataframe['Neptune_ra'] * 15.923566879
        dataframe['%-Pluto_rax'] = dataframe['Pluto_ra'] * 15.923566879

        # Declination
        dataframe['%-Moon_Dec'] = (dataframe['Moon_dec']+0.5) * 100
        dataframe['%-Sun_dec'] = (dataframe['Sun_dec']+0.5) * 100
        dataframe['%-Mercury_dec'] = (dataframe['Mercury_dec']+0.5) * 100
        dataframe['%-Venus_dec'] = (dataframe['Venus_dec']+0.5) * 100
        dataframe['%-Mars_dec'] = (dataframe['Mars_dec']+0.5) * 100
        dataframe['%-Jupiter_dec'] = (dataframe['Jupiter_dec']+0.5) * 100
        dataframe['%-Saturn_dec'] = (dataframe['Saturn_dec']+0.5) * 100
        dataframe['%-Uranus_dec'] = (dataframe['Uranus_dec']+0.5) * 100
        dataframe['%-Neptune_dec'] = (dataframe['Neptune_dec']+0.5) * 100
        dataframe['%-Pluto_dec'] = (dataframe['Pluto_dec']+0.5) * 100

        # Astrological House for each Celestial Body
        dataframe['%-Sun_house'] = celestial_house(dataframe['Sun'].iloc[0])
        dataframe['%-Mercury_house'] = celestial_house(dataframe['Mercury'].iloc[0])
        dataframe['%-Venus_house'] = celestial_house(dataframe['Venus'].iloc[0])
        dataframe['%-Mars_house'] = celestial_house(dataframe['Mars'].iloc[0]) 
        dataframe['%-Jupiter_house'] = celestial_house(dataframe['Jupiter'].iloc[0])
        dataframe['%-Saturn_house'] = celestial_house(dataframe['Saturn'].iloc[0])
        dataframe['%-Uranus_house'] = celestial_house(dataframe['Uranus'].iloc[0])
        dataframe['%-Neptune_house'] = celestial_house(dataframe['Neptune'].iloc[0])
        dataframe['%-Pluto_house'] = celestial_house(dataframe['Pluto'].iloc[0]) 

        return dataframe


    def feature_engineering_standard(self, dataframe, **kwargs):
        dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 25
        return dataframe


    def set_freqai_targets(self, dataframe, **kwargs):
        dataframe["&s-extrema"] = 0
        min_peaks = argrelextrema(
            dataframe["low"].values, np.less,
            order=self.freqai_info["feature_parameters"]["label_period_candles"]
        )
        max_peaks = argrelextrema(
            dataframe["high"].values, np.greater,
            order=self.freqai_info["feature_parameters"]["label_period_candles"]
        )
        for mp in min_peaks[0]:
            dataframe.at[mp, "&s-extrema"] = -1
        for mp in max_peaks[0]:
            dataframe.at[mp, "&s-extrema"] = 1
        dataframe["minima"] = np.where(dataframe["&s-extrema"] == -1, 1, 0)
        dataframe["maxima"] = np.where(dataframe["&s-extrema"] == 1, 1, 0)
        dataframe['&s-extrema'] = dataframe['&s-extrema'].rolling(
            window=5, win_type='gaussian', center=True).mean(std=0.5)
        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe["DI_catch"] = np.where(
            dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1,
        )

        dataframe["minima_sort_threshold"] = dataframe["&s-minima_sort_threshold"]
        dataframe["maxima_sort_threshold"] = dataframe["&s-maxima_sort_threshold"]
        return dataframe


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df["do_predict"] == 1) &  
                (df["DI_catch"] == 1) & 
                (df["&s-extrema"] < df["minima_sort_threshold"]) & 
                (df["&s-extrema"].shift(1) < df["minima_sort_threshold"].shift(1)) &
                (df["&s-extrema"].shift(2) < df["minima_sort_threshold"].shift(2)) &
                (df["close"].shift(1) < df["open"].shift(1)) &
                (df["close"].shift(2) < df["open"].shift(2)) &
                (df["maxima"] != 1) &
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'Extrema < Minima Threshold 3rd Descending')

        df.loc[
            (
                (df["do_predict"] == 1) &  
                (df["DI_catch"] == 1) & 
                (df["&s-extrema"] < df["minima_sort_threshold"]) & 
                (df["&s-extrema"].shift(1) < df["minima_sort_threshold"].shift(1)) &
                (df["&s-extrema"].shift(2) < df["minima_sort_threshold"].shift(2)) &
                (df["close"] > df["open"]) &
                (df["close"].shift(1) < df["open"].shift(1)) &
                (df["maxima"] != 1) &
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'Extrema < Minima Threshold Quick Reversal')

        df.loc[
            (
                (df["do_predict"] == 1) &  # Guard: tema is raising
                (df["DI_catch"] == 1) & 
                (df["&s-extrema"] < 0) & 
                (df["minima"] == 1) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'Minima')


        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df["do_predict"] == 1) &  # Guard: tema is raising
                (df["DI_catch"] == 1) & 
                (df["&s-extrema"] > df["maxima_sort_threshold"]) & 
                (df["maxima_sort_threshold"] > 0) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['exit_long', 'exit_tag']] = (1, 'Extrema > Maxima Threshold > 0')

        df.loc[
            (
                (df["do_predict"] == 1) &  # Guard: tema is raising
                (df["DI_catch"] == 1) & 
                (df["&s-extrema"] > (df["minima_sort_threshold"] * 10)) & 
                (df["minima_sort_threshold"] > 0) & 
                (df["maxima_sort_threshold"] == 0) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['exit_long', 'exit_tag']] = (1, 'Extrema > Minima Threshold * 10')

        df.loc[
            (
                (df["do_predict"] == 1) &  # Guard: tema is raising
                (df["DI_catch"] == 1) & 
                (df["&s-extrema"] > 0) & 
                (df["maxima"] == 1) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['exit_long', 'exit_tag']] = (1, 'Maxima')

        df.loc[
            (
                (df["do_predict"] == 1) &  # Guard: tema is raising
                (df["DI_catch"] == 0) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['exit_long', 'exit_tag']] = (1, 'Outlier')

        return df


def top_percent_change(dataframe: DataFrame, length: int) -> float:
    """
    Percentage change of the current close from the range maximum Open price
    :param dataframe: DataFrame The original OHLC dataframe
    :param length: int The length to look back
    """
    if length == 0:
        return (dataframe['open'] - dataframe['close']) / dataframe['close']
    else:
        return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']


def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']


def EWO(dataframe, sma1, sma2):
    df = dataframe.copy()
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif


def get_distance(p1, p2):
    return abs((p1) - (p2))


def celestial_house(planet_name):
    house = 0
    if planet_name == "Aries":
        house = 1
    elif planet_name == "Taurus":
        house = 2
    elif planet_name == "Gemini":
        house = 3
    elif planet_name == "Cancer":
        house = 4
    elif planet_name == "Leo":
        house = 5
    elif planet_name == "Virgo":
        house = 6
    elif planet_name == "Libra":
        house = 7
    elif planet_name == "Scorpio":
        house = 8
    elif planet_name == "Sagittarius":
        house = 9
    elif planet_name == "Capricorn":
        house = 10
    elif planet_name == "Aquarius":
        house = 11
    elif planet_name == "Pisces":
        house = 12
    return house