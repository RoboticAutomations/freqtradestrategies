import logging
from functools import reduce
from typing import Dict

import talib.abstract as ta # type: ignore
from pandas import DataFrame, Series
import pandas as pd
from technical import qtpylib # type: ignore

from datetime import timedelta, datetime
from freqtrade.persistence import Trade
from typing import *



from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, stoploss_from_absolute, stoploss_from_open, informative
from scipy.signal import argrelextrema
import numpy as np
import pandas_ta as pta # type: ignore
import math
from technical.pivots_points import pivots_points # type: ignore
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_minutes
from technical import indicators as ti # type: ignore
from pykalman import KalmanFilter # type: ignore
from scipy.stats import zscore
from scipy.signal import savgol_filter
from scipy.fft import fft
from hmmlearn import hmm # type: ignore
import legendary_ta as lta
import custom_indicators as ci
# from tsfresh import extract_features
# from tsfresh.utilities.dataframe_functions import make_forecasting_frame


logger = logging.getLogger(__name__)
'''
freqtrade trade --strategy gridAI --config user_data/Grid.json --freqaimodel XGBoostRegressorQuickAdapterV35x --logfile user_data/logs/Grid.log
    
'''

class gridAI(IStrategy):
    
    minimal_roi = {"0": 1}

    plot_config = {
        "main_plot": {
            "atr_sell1": {
            "color": "fuchsia",
            "type": "line"
            },
            "atr_sell2": {
            "color": "fuchsia",
            "type": "line"
            },
            "atr_sell3": {
            "color": "fuchsia",
            "type": "line"
            },
            "atr_sell4": {
            "color": "fuchsia",
            "type": "line"
            },
            "atr_sell5": {
            "color": "fuchsia",
            "type": "line"
            },
            "atr_buy1": {
            "color": "cyan",
            "type": "line"
            },
            "atr_buy2": {
            "color": "cyan",
            "type": "line"
            },
            "atr_buy3": {
            "color": "cyan",
            "type": "line"
            },
            "atr_buy4": {
            "color": "cyan",
            "type": "line"
            },
            "atr_buy5": {
            "color": "cyan",
            "type": "line"
            },
            "daily_open": {
            "color": "blue",
            "type": "line"
            },
            "deadzone_upper": {
            "color": "red",
            "type": "line"
            },
            "deadzone_lower": {
            "color": "red",
            "type": "line"
            }
        },
        "subplots": {
            "extrema": {
                "&s-extrema": {
                    "color": "#f53580",
                    "type": "line"
                },
                "&s-minima_sort_threshold": {
                    "color": "#f66151",
                    "type": "line"
                },
                "&s-maxima_sort_threshold": {
                    "color": "#8ff0a4",
                    "type": "line"
                }
            },
            "range_est": {
                "&-s_max": {
                    "color": "#a29db9",
                    "type": "line"
                },
                "&-s_min": {
                    "color": "#ac7fc",
                    "type": "line"
                }
            },
            "truth": {
                "maxima-exit": {
                    "color": "#8ff0a4",
                    "type": "bar"
                },
                "minima-exit": {
                    "color": "#f66151",
                    "type": "bar"
                }
            }
        }
    }
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        
        rounded_leverage = self.lev.value

        return rounded_leverage

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2,
            },

        ]
    
    process_only_new_candles = True
    stoploss = -0.99
    use_exit_signal = True
    position_adjustment_enable = True
    use_custom_stoploss = True
    # this is the maximum period fed to talib (timeframe independent)
    startup_candle_count: int = 100
    can_short = True

    timeframe = "15m"

    ####################### Parameters
    atr_multiplier = DecimalParameter(1.0, 3.0, default=1.5, space="buy", optimize=True)
    atr_rr = DecimalParameter(0.5, 3.0, default=2, space="buy", optimize=True)
    tpRate1 = IntParameter(2, 5, default=5, space='sell', optimize=True, load=True)
    tpRate2 = IntParameter(2, 5, default=5, space='sell', optimize=True, load=True)
    tpRate3 = IntParameter(2, 5, default=5, space='sell', optimize=True, load=True)
    tpRate4 = IntParameter(2, 5, default=5, space='sell', optimize=True, load=True)
    tpRate5 = IntParameter(2, 5, default=5, space='sell', optimize=True, load=True)
    risk_per_trade = DecimalParameter(0.01, 0.05, default=0.03, decimals=2, space='buy', optimize=True, load=True)
    atr_period = IntParameter(4, 28, default=5, space='buy', optimize=True, load=True)
    atr_distance = DecimalParameter(0.3, 4, default=4, decimals=2, space='sell', optimize=True, load=True)
    lev = IntParameter(low=1, high=5, default=1, space="buy", optimize=True, load=True)

    prediction_time = 90 # für 15m, 280 or so bei 5m
    

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: Dict, **kwargs
    ) -> DataFrame:
        
        # print("All Feature Engineering")
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, window=period)
        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=period)
        dataframe["%-er-period"] = pta.er(dataframe['close'], length=period)
        dataframe["%-rocr-period"] = ta.ROCR(dataframe, timeperiod=period)
        dataframe["%-cmf-period"] = chaikin_mf(dataframe, periods=period)
        dataframe["%-tcp-period"] = top_percent_change(dataframe, period)
        dataframe["%-cti-period"] = pta.cti(dataframe['close'], length=period)
        dataframe["%-chop-period"] = qtpylib.chopiness(dataframe, period)
        dataframe["%-linear-period"] = ta.LINEARREG_ANGLE(
            dataframe['close'], timeperiod=period)
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        dataframe["%-atr-periodp"] = dataframe[f"%-atr-period"] / \
            dataframe['close'] * 1000

        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )
        
        dataframe['%-VIDYA-period'] = ti.VIDYA(dataframe, length=period)

        dataframe["%-savgol-volume-period"] = savgol_filter(
            dataframe["volume"], period, 3)
        dataframe["%-savgol-close-period"] = savgol_filter(
            dataframe["close"], period, 3)
        dataframe["%-savgol-open-period"] = savgol_filter(
            dataframe["open"], period, 3)
        dataframe["%-savgol-high-period"] = savgol_filter(
            dataframe["high"], period, 3)
        dataframe["%-savgol-low-period"] = savgol_filter(
            dataframe["low"], period, 3)
        
        dataframe["%-savgol2-volume-period"] = savgol_filter(
            dataframe["volume"], period, 2)
        dataframe["%-savgol2-close-period"] = savgol_filter(
            dataframe["close"], period, 2)
        dataframe["%-savgol2-open-period"] = savgol_filter(
            dataframe["open"], period, 2)
        dataframe["%-savgol2-high-period"] = savgol_filter(
            dataframe["high"], period, 2)
        dataframe["%-savgol2-low-period"] = savgol_filter(
            dataframe["low"], period, 2)          
        
        # Fisher Stochastic Center of Gravity
        fishcg = lta.fisher_cg(dataframe, period)
        dataframe["fisher_cg-period"] = fishcg["fisher_cg"]
        dataframe["fisher_sig-period"] = fishcg["fisher_sig"]
        dataframe["%-dist-fisher_cg-period"] = get_distance(
            dataframe["close"], dataframe["fisher_cg-period"])
        dataframe["%-dist-fisher_sig-period"] = get_distance(
            dataframe["close"], dataframe["fisher_sig-period"])
        
        # Stochastic Momentum Index
        smi = lta.smi_momentum(dataframe, period)
        dataframe["smi-period"] = smi["smi"]
        dataframe["%-dist-smi-to-0"] = get_distance(
            0, dataframe["smi-period"])
        dataframe["%-dist-smi-to-40"] = get_distance(
            40, dataframe["smi-period"])
        dataframe["%-dist-smi-to_-40"] = get_distance(
            -40, dataframe["smi-period"]) 
        
        # Breakouts and Retests
        supres = lta.breakouts(dataframe, period)
        dataframe["support-period"] =  supres['support_level']
        dataframe["resistance-period"] = supres['resistance_level']

        dataframe["%-dist_to_support-period"] = get_distance(
            dataframe["close"], dataframe["support-period"])
        dataframe["%-dist_to_resistance-period"] = get_distance(
            dataframe["close"], dataframe["resistance-period"])
        
        dataframe["zema-period"] = ci.zema(dataframe, period)
        dataframe["%-dist_to_zema-period"] = get_distance(
            dataframe["close"], dataframe["zema-period"])
        
        dataframe["pcc_upper-period"], dataframe["pcc_rangema-period"], dataframe["pcc_lower-period"] = ci.pcc(dataframe, period)
        dataframe["%-dist_to_pcc_upper-period"] = get_distance(
            dataframe["pcc_rangema-period"], dataframe["pcc_upper-period"])
        dataframe["%-dist_to_pcc_lower-period"] = get_distance(
            dataframe["pcc_rangema-period"], dataframe["pcc_lower-period"])
        
        dataframe['rmi-period'] = ci.RMI(dataframe, length=period)
        dataframe['rmi-up'] = np.where(dataframe['rmi-period'] >= dataframe['rmi-period'].shift(), 1, 0)
        dataframe['%-rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() >= 3, 1, 0)

        dataframe['rmi-dn'] = np.where(dataframe['rmi-period'] <= dataframe['rmi-period'].shift(), 1, 0)
        dataframe['%-rmi-dn-count'] = dataframe['rmi-dn'].rolling(8).sum()

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: Dict, **kwargs
    ) -> DataFrame:
        # print("Basic Feature Engineering")
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-obv"] = ta.OBV(dataframe)
        dataframe['%-tail'] = (dataframe['close'] - dataframe['low']).abs()
        dataframe['%-wick'] = (dataframe['high'] - dataframe['close']).abs()
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_low"] = dataframe["low"]
        dataframe["%-raw_high"] = dataframe["high"]  
        # Added
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=14, stds=2.2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["%-bb_width"] = (dataframe["bb_upperband"] -
                                dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        dataframe["%-dist_bb_lowerband"] = get_distance(
            dataframe["close"], dataframe["bb_lowerband"])
        dataframe["%-dist_bb_upperband"] = get_distance(
            dataframe["close"], dataframe["bb_upperband"])
        dataframe["%-dist_bb_middleband"] = get_distance(
            dataframe["close"], dataframe["bb_middleband"])
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['%-distema50'] = get_distance(
            dataframe['close'], dataframe['ema_50'])
        dataframe['%-distema13'] = get_distance(
            dataframe['close'], dataframe['ema_13'])
        dataframe['%-distema21'] = get_distance(
            dataframe['close'], dataframe['ema_21'])
        
        # dataframe['%-b-ema13_21_crossed_above'] = qtpylib.crossed_above(dataframe['ema_13'], dataframe['ema_21']).astype(int)
        # dataframe['%-b-ema13_21_crossed_below'] = qtpylib.crossed_below(dataframe['ema_13'], dataframe['ema_21']).astype(int)

        macd = ta.MACD(dataframe)
        dataframe['%-macd'] = macd['macd']
        dataframe['%-macdsignal'] = macd['macdsignal']
        dataframe['%-macdhist'] = macd['macdhist']
        dataframe['%-dist_to_macdsignal'] = get_distance(
            dataframe['%-macd'], dataframe['%-macdsignal'])
        dataframe['%-dist_to_zerohist'] = get_distance(
            0, dataframe['%-macdhist'])
        # VWAP
        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['vwap_upperband'] = vwap_high
        dataframe['vwap_middleband'] = vwap
        dataframe['vwap_lowerband'] = vwap_low
        dataframe['%-vwap_width'] = ((dataframe['vwap_upperband'] -
                                    dataframe['vwap_lowerband']) / dataframe['vwap_middleband']) * 100
        dataframe = dataframe.copy()
        dataframe['%-dist_to_vwap_upperband'] = get_distance(
            dataframe['close'], dataframe['vwap_upperband'])
        dataframe['%-dist_to_vwap_middleband'] = get_distance(
            dataframe['close'], dataframe['vwap_middleband'])
        dataframe['%-dist_to_vwap_lowerband'] = get_distance(
            dataframe['close'], dataframe['vwap_lowerband'])
            

        dataframe['%-zscore_close'] = zscore(dataframe['close'])
        dataframe['%-zscore_volume'] = zscore(dataframe['volume'])
        dataframe['%-zscore_high'] = zscore(dataframe['high'])
        dataframe['%-zscore_low'] = zscore(dataframe['low'])
        dataframe['%-zscore_open'] = zscore(dataframe['open'])   
        
        dataframe['%-sar'] = ta.SAR(dataframe)
        hilbert = ta.HT_SINE(dataframe)
        dataframe['%-htsine'] = hilbert['sine']
        dataframe['%-htleadsine'] = hilbert['leadsine']
    
        kf = KalmanFilter(transition_matrices=[1],
                        observation_matrices=[1],
                        initial_state_mean=0,
                        initial_state_covariance=1,
                        observation_covariance=1,
                        transition_covariance=0.01)

        state_means, state_covariances = kf.filter(dataframe.close.values)
        kalman_avg = state_means.flatten()

        dataframe["%-kalman_avg"]=kalman_avg
        dataframe["%-zscore_kalman"]=zscore(dataframe["%-kalman_avg"]) 
        
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: Dict, **kwargs
    ) -> DataFrame:
        # print("Standard Feature Engineering")
        dataframe["day_of_week"] = (dataframe["date"].dt.dayofweek)
        dataframe["hour_of_day"] = (dataframe["date"].dt.hour)
        dataframe['day_of_week_norm'] = 2 * math.pi * \
            dataframe['day_of_week'] / dataframe['day_of_week'].max()
        dataframe['hour_of_day_norm'] = 2 * math.pi * \
            dataframe['hour_of_day'] / dataframe['hour_of_day'].max()

        dataframe['%%-day_of_week_cos'] = np.cos(dataframe['day_of_week_norm'])
        dataframe['%%-hour_of_day_cos'] = np.cos(dataframe['hour_of_day_norm'])
        dataframe['%%-day_of_week_sin'] = np.sin(dataframe['day_of_week_norm'])
        dataframe['%%-hour_of_day_sin'] = np.sin(dataframe['hour_of_day_norm'])

        
        # HMM Features
        n_states = 69
        price_bins = np.linspace(np.min(dataframe['close']), np.max(dataframe['close']), n_states + 1)
        price_states = np.digitize(dataframe['close'], price_bins) - 1

        model = hmm.MultinomialHMM(n_components=n_states, n_iter=1000)
        model.fit(price_states.reshape(-1, 1))
        predicted_states = model.predict(price_states.reshape(-1, 1))
        dataframe["%-hmm_state"] = predicted_states
        
        # FFT Features
        window_size = 20  # Adjust the window size as needed
        for i in range(len(dataframe) - window_size):
            window_data = dataframe['close'].iloc[i:i+window_size].values
            close_fft = fft(window_data)
            fft_features = np.abs(close_fft)
            
            for j in range(5):  # Adjust the number of components as needed
                dataframe.at[i + window_size, f"%-fft_{j}"] = fft_features[j]

        # Trends, Peaks and Crosses
        dataframe['%-candle-up'] = np.where(dataframe['close'] >= dataframe['open'], 1, 0)
        dataframe['%-candle-up-trend'] = np.where(dataframe['%-candle-up'].rolling(5).sum() >= 3, 1, 0)



        # Indicators used only for ROI and Custom Stoploss
        ssldown, sslup = ci.SSLChannels_ATR(dataframe, length=21)
        dataframe['%-sroc'] = ci.SROC(dataframe, roclen=21, emalen=13, smooth=21)
        dataframe['%-ssl-dir'] = np.where(sslup > ssldown, 1, -1)      

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:

        dataframe["&s-train_label"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .mean()
            / dataframe["close"]
            - 1
        )
        return dataframe

    @informative("1d")
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe["atr"] = calculate_atr(dataframe, atr_period=self.atr_period.value)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

        return dataframe
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        def get_daily_open(dataframe):
            first_open = dataframe.groupby(dataframe['date'].dt.date)['open'].transform('first')
            return first_open
        
        dataframe['daily_open'] = get_daily_open(dataframe)
        dataframe["atr_sell1"] = dataframe["daily_open"] + (dataframe["atr_1d"] * 0.1)
        dataframe["atr_sell2"] = dataframe["daily_open"] + (dataframe["atr_1d"] * 0.2)
        dataframe["atr_sell3"] = dataframe["daily_open"] + (dataframe["atr_1d"] * 0.3)
        dataframe["atr_sell4"] = dataframe["daily_open"] + (dataframe["atr_1d"] * 0.4)
        dataframe["atr_sell5"] = dataframe["daily_open"] + (dataframe["atr_1d"] * 0.5)
        dataframe["atr_buy1"] = dataframe["daily_open"] - (dataframe["atr_1d"] * 0.1)
        dataframe["atr_buy2"] = dataframe["daily_open"] - (dataframe["atr_1d"] * 0.2)
        dataframe["atr_buy3"] = dataframe["daily_open"] - (dataframe["atr_1d"] * 0.3)
        dataframe["atr_buy4"] = dataframe["daily_open"] - (dataframe["atr_1d"] * 0.4)
        dataframe["atr_buy5"] = dataframe["daily_open"] - (dataframe["atr_1d"] * 0.5)

        # distance from daily open to atr5 in %
        dataframe["dist_atr5"] = get_distance(dataframe["daily_open"], dataframe["atr_buy5"]) / dataframe["daily_open"] * 100
        # distance from daily open to atr5 sell in %
        dataframe["dist_atr5_sell"] = get_distance(dataframe["daily_open"], dataframe["atr_sell5"]) / dataframe["daily_open"] * 100


        dataframe["deadzone_upper"] = dataframe["daily_open"] + (dataframe["atr_1d"] * 1)
        dataframe["deadzone_lower"] = dataframe["daily_open"] - (dataframe["atr_1d"] * 1)
        
        
        dataframe["atr"] = calculate_atr(dataframe, atr_period=self.atr_period.value)

        dataframe["sl_long"] = dataframe["low"] - (self.atr_distance.value * dataframe['atr'])
        dataframe["sl_short"] = dataframe["high"] + (self.atr_distance.value * dataframe['atr'])

        dataframe = self.freqai.start(dataframe, metadata, self)
        dataframe["DI_catch"] = np.where(
            dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1,
        )

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['enter_tag'] = ''

        day_bullish = (
            # Trend is bullish
            (df["dist_atr5"] < df["&s-train_label"]) &
            (df['do_predict'] == 1) &
            # we have a new daily open
            (df["daily_open"] != df["daily_open"].shift(1)) &
            # volume is not 0
            (df["volume"] > 0)
        )
        df.loc[day_bullish, 'enter_long'] = 1
        df.loc[day_bullish, 'enter_tag'] = 'Bullish Day ahead'
        
        day_bearish = (
            # Trend is bearish
            (df['dist_atr5_sell'] > df["&s-train_label"]) &
            (df['do_predict'] == 1) &
            # we have a new daily open
            (df["daily_open"] != df["daily_open"].shift(1)) &
            # volume is not 0
            (df["volume"] > 0)
        )
        df.loc[day_bearish, 'enter_short'] = 1
        df.loc[day_bearish, 'enter_tag'] = 'Bearish Day ahead' 

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        

        return df
    
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ):
        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()
        trade_date = timeframe_to_prev_date(
            self.timeframe, (trade.open_date_utc - timedelta(minutes=int(self.timeframe[:-1])))
        )
        trade_candle = dataframe.loc[(dataframe["date"] == trade_date)]
        if trade_candle.empty:
            return None
        trade_candle = trade_candle.squeeze()

        entry_tag = trade.enter_tag

        if trade_duration > 720:
            return "Trade expired"

        if last_candle["DI_catch"] == 0 and current_profit < 0:
            return "Outlier detected"

        return None


    def custom_stake_amount(self, pair: str, current_rate: float, proposed_stake: float, side: str, **kwargs) -> float:
        
        stake_currency = "USDT"

        # Get total equity and available balance
        total_equity = self.wallets.get_total(currency=stake_currency)
        max_open_trades = self.max_open_trades
        def_stake = total_equity / max_open_trades

        return proposed_stake / 6
    

    def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)
        new_entryprice = dataframe['daily_open'].iat[-1]

        return new_entryprice

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs) -> float:
      
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        trade_open_candle = dataframe.loc[dataframe['date'] == trade_date]
        entry_price = trade.open_rate

        if trade_open_candle.empty:
            return None
        
        if trade.trade_direction == "long":
            sl_long = stoploss_from_open(trade_open_candle['deadzone_lower'].values[0], current_profit, False, leverage=self.lev.value)
            if current_rate < trade_open_candle['deadzone_lower'].values[0]:
                logger.info(f"Deadzone Long hit for {trade.pair} at {current_rate}")
                self.dp.send_msg(f"Deadzone Long hit for {trade.pair} at {current_rate}")
            return sl_long
        else: 
            sl_short = stoploss_from_open(trade_open_candle['deadzone_upper'].values[0], current_profit, True, leverage=self.lev.value)
            if current_rate > trade_open_candle['deadzone_upper'].values[0]:
                logger.info(f"Deadzone Short hit for {trade.pair} at {current_rate}")
                self.dp.send_msg(f"Deadzone Short hit for {trade.pair} at {current_rate}")
            return sl_short
            
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs
                              ) -> Union[Optional[float], Tuple[Optional[float], Optional[str]]]:
        
        # Unterscheidung zwischen Long- und Short-Positionen
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        trade_open_candle = dataframe.loc[dataframe['date'] == trade_date]
        entry_price = trade.open_rate
        proposed_stake = trade.stake_amount

        if trade_open_candle.empty:
            return None
        
        if current_profit < 0:
            if trade.trade_direction == "long":
                if current_rate <= trade_open_candle['atr_buy1'].values[0] and trade.nr_of_successful_entries == 1:
                    # add more to the position
                    logger.info(f"Adding more to the position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    self.dp.send_msg(f"Adding more to the position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    return proposed_stake / 6
                elif current_rate <= trade_open_candle['atr_buy2'].values[0] and trade.nr_of_successful_entries == 2:
                    # add more to the position
                    logger.info(f"Adding more to the position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    self.dp.send_msg(f"Adding more to the position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    return proposed_stake / 6
                elif current_rate <= trade_open_candle['atr_buy3'].values[0] and trade.nr_of_successful_entries == 3:
                    # add more to the position
                    logger.info(f"Adding more to the position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    self.dp.send_msg(f"Adding more to the position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    return proposed_stake / 6
                elif current_rate <= trade_open_candle['atr_buy4'].values[0] and trade.nr_of_successful_entries == 4:
                    # add more to the position
                    logger.info(f"Adding more to the position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    self.dp.send_msg(f"Adding more to the position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    return proposed_stake / 6
                elif current_rate <= trade_open_candle['atr_buy5'].values[0] and trade.nr_of_successful_entries == 5:
                    # add more to the position
                    logger.info(f"Adding more to the position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    self.dp.send_msg(f"Adding more to the position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    return proposed_stake / 6
                else:
                    return None
            else:
                if current_rate >= trade_open_candle['atr_sell1'].values[0] and trade.nr_of_successful_entries == 1:
                    # add more to the position
                    logger.info(f"Adding more to the short position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    self.dp.send_msg(f"Adding more to the short position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    return proposed_stake / 6
                elif current_rate >= trade_open_candle['atr_sell2'].values[0] and trade.nr_of_successful_entries == 2:
                    # add more to the position
                    logger.info(f"Adding more to the short position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    self.dp.send_msg(f"Adding more to the short position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    return proposed_stake / 6
                elif current_rate >= trade_open_candle['atr_sell3'].values[0] and trade.nr_of_successful_entries == 3:
                    # add more to the position
                    logger.info(f"Adding more to the short position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    self.dp.send_msg(f"Adding more to the short position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    return proposed_stake / 6
                elif current_rate >= trade_open_candle['atr_sell4'].values[0] and trade.nr_of_successful_entries == 4:
                    # add more to the position
                    logger.info(f"Adding more to the short position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    self.dp.send_msg(f"Adding more to the short position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    return proposed_stake / 6
                elif current_rate >= trade_open_candle['atr_sell5'].values[0] and trade.nr_of_successful_entries == 5:
                    # add more to the position
                    logger.info(f"Adding more to the short position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    self.dp.send_msg(f"Adding more to the short position for {trade.pair} at {current_rate}. Number of successful entries: {trade.nr_of_successful_entries}")
                    return proposed_stake / 6
                else:
                    return None
        
        else: # Logic for partial exits
            if trade.trade_direction == "long":
                if current_rate >= trade_open_candle['atr_sell1'].values[0] and trade.nr_of_successful_exits == 0:
                    # exit partial position
                    logger.info(f"Exiting partial position for {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    self.dp.send_msg(f"Exiting partial position for {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    return -(proposed_stake / 6)
                elif current_rate >= trade_open_candle['atr_sell2'].values[0] and trade.nr_of_successful_exits == 1:
                    # exit partial position
                    logger.info(f"Exiting partial position for {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    self.dp.send_msg(f"Exiting partial position for {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    return -(proposed_stake / 6)
                elif current_rate >= trade_open_candle['atr_sell3'].values[0] and trade.nr_of_successful_exits == 2:
                    # exit partial position
                    logger.info(f"Exiting partial position for {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    self.dp.send_msg(f"Exiting partial position for {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    return -(proposed_stake / 6)
                elif current_rate >= trade_open_candle['atr_sell4'].values[0] and trade.nr_of_successful_exits == 3:
                    # exit partial position
                    logger.info(f"Exiting partial position for {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    self.dp.send_msg(f"Exiting partial position for {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    return -(proposed_stake / 6)
                elif current_rate >= trade_open_candle['atr_sell5'].values[0] and trade.nr_of_successful_exits == 4:
                    # exit partial position
                    logger.info(f"Exiting partial position for {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    self.dp.send_msg(f"Exiting partial position for {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    return -(proposed_stake / 6)
                else:
                    return None
            else:
                if current_rate <= trade_open_candle['atr_buy1'].values[0] and trade.nr_of_successful_exits == 0:
                    # exit partial position
                    logger.info(f"Exiting partial position for short {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    self.dp.send_msg(f"Exiting partial position for short {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    return -(proposed_stake / 6)
                elif current_rate <= trade_open_candle['atr_buy2'].values[0] and trade.nr_of_successful_exits == 1:
                    # exit partial position
                    logger.info(f"Exiting partial position for short {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    self.dp.send_msg(f"Exiting partial position for short {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    return -(proposed_stake / 6)
                elif current_rate <= trade_open_candle['atr_buy3'].values[0] and trade.nr_of_successful_exits == 2:
                    # exit partial position
                    logger.info(f"Exiting partial position for short {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    self.dp.send_msg(f"Exiting partial position for short {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    return -(proposed_stake / 6)
                elif current_rate <= trade_open_candle['atr_buy4'].values[0] and trade.nr_of_successful_exits == 3:
                    # exit partial position
                    logger.info(f"Exiting partial position for short {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    self.dp.send_msg(f"Exiting partial position for short {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    return -(proposed_stake / 6)
                elif current_rate <= trade_open_candle['atr_buy5'].values[0] and trade.nr_of_successful_exits == 4:
                    # exit partial position
                    logger.info(f"Exiting partial position for short {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    self.dp.send_msg(f"Exiting partial position for short {trade.pair} at {current_rate}. Number of successful exits: {trade.nr_of_successful_exits}")
                    return -(proposed_stake / 6)
                else:
                    return None

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True

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


def chaikin_mf(df, periods=20):
    close = df['close']
    low = df['low']
    high = df['high']
    volume = df['volume']
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)
    mfv *= volume
    cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()
    return Series(cmf, name='cmf')

# VWAP bands


def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']


def EWO(dataframe, sma_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.EMA(df, timeperiod=sma_length)
    sma2 = ta.EMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif


def get_distance(p1, p2):
    return abs((p1) - (p2))

def calculate_atr(df, atr_period=28):
    # atr = pda.atr(df["high"], df["low"], df["close"], length=atr_period)
    atr = ta.ATR(df, timeperiod=atr_period)
    # round atr to 5 decimals
    atr = atr.apply(lambda x: round(x, 4))
    return atr

def detect_pullback(df: DataFrame, periods=30, method='pct_outlier'):
    """     
    Pullback & Outlier Detection
    Know when a sudden move and possible reversal is coming
    
    Method 1: StDev Outlier (z-score)
    Method 2: Percent-Change Outlier (z-score)
    Method 3: Candle Open-Close %-Change
    
    outlier_threshold - Recommended: 2.0 - 3.0
    
    df['pullback_flag']: 1 (Outlier Up) / -1 (Outlier Down) 
    """
    if method == 'stdev_outlier':
        outlier_threshold = 2.0
        df['dif'] = df['close'] - df['close'].shift(1)
        df['dif_squared_sum'] = (df['dif']**2).rolling(window=periods + 1).sum()
        df['std'] = np.sqrt((df['dif_squared_sum'] - df['dif'].shift(0)**2) / (periods - 1))
        df['z'] = df['dif'] / df['std']
        df['pullback_flag'] = np.where(df['z'] >= outlier_threshold, 1, 0)
        df['pullback_flag'] = np.where(df['z'] <= -outlier_threshold, -1, df['pullback_flag'])

    if method == 'pct_outlier':
        outlier_threshold = 2.0
        df["pb_pct_change"] = df["close"].pct_change()
        df['pb_zscore'] = qtpylib.zscore(df, window=periods, col='pb_pct_change')
        df['pullback_flag'] = np.where(df['pb_zscore'] >= outlier_threshold, 1, 0)
        df['pullback_flag'] = np.where(df['pb_zscore'] <= -outlier_threshold, -1, df['pullback_flag'])
    
    if method == 'candle_body':
        pullback_pct = 1.0
        df['change'] = df['close'] - df['open']
        df['pullback'] = (df['change'] / df['open']) * 100
        df['pullback_flag'] = np.where(df['pullback'] >= pullback_pct, 1, 0)
        df['pullback_flag'] = np.where(df['pullback'] <= -pullback_pct, -1, df['pullback_flag'])
    
    return df