# freqtrade hyperopt --hyperopt-loss ProfitDrawDownHyperOptLoss --spaces roi trailing protection buy sell trades --strategy AWATRCT  --config user_data/config_backtesting.json  -e 200 -p `cat pairs34`  --timerange  20240801-   -j 3 --timeframe-detail 5m
#
#       Hyperopt results
#       ┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
#       ┃ Best   ┃   Epoch ┃ Trades ┃  Win  Draw  Loss  Win% ┃ Avg profit ┃                  Profit ┃ Avg duration ┃  Objective ┃    Max Drawdown (Acct) ┃
#       ┡━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
#       │ * Best │   1/200 │    124 │   60    48    16  48.4 │      0.31% │  63.062 USDT    (6.31%) │     12:29:00 │  -57.58846 │ 96.888 USDT    (9.38%) │
#       │ * Best │   4/200 │    212 │   86   115    11  40.6 │      0.35% │ 110.718 USDT   (11.07%) │      6:02:00 │ -109.57862 │ 11.850 USDT    (1.11%) │
#       │ * Best │  13/200 │    112 │   56    42    14  50.0 │      0.49% │ 137.341 USDT   (13.73%) │     11:14:00 │ -128.90512 │ 68.666 USDT    (6.64%) │
#       │ Best   │  75/200 │     90 │   46    36     8  51.1 │      0.49% │ 152.257 USDT   (15.23%) │     10:02:00 │ -140.19980 │ 90.589 USDT    (8.56%) │
#       │ Best   │ 108/200 │     42 │   24    12     6  57.1 │      0.50% │ 174.542 USDT   (17.45%) │     11:52:00 │ -161.57066 │ 84.972 USDT    (8.03%) │
#       │ Best** │ 157/200 │     59 │   33    20     6  55.9 │      0.62% │ 184.604 USDT   (18.46%) │      9:48:00 │ -170.44132 │ 88.886 USDT    (8.29%) │
#       └────────┴─────────┴────────┴────────────────────────┴────────────┴─────────────────────────┴──────────────┴────────────┴────────────────────────┘
# --- Do not remove these libs ---
import math
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
import pandas_ta as pta
import pandas as pd
from typing import Optional
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import BooleanParameter, stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, \
    CategoricalParameter
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_minutes

# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 20,
    "ewof": 8,
    "ewos": 89,
    "ewo_high_mult": 1.3618,
    "ewo_bear_mult": 1.65,
    "ewo_bull_mult": 1.01618,
    "low_offset": 0.95,  # 34,  # 0.942 for BTC and   0.934 for USDT
    "rsi_buy": 61  # with regular ewo 45
}

# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 45,
    "high_offset": 1.01918  # 1.019
}


def EWO(dataframe, ema_length=8, ema2_length=89):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length.value)
    ema2 = ta.EMA(df, timeperiod=ema2_length.value)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

def calculate_murrey_math_levels(df, window_size=64):

    df = df.iloc[-window_size:]
    # Calculate rolling 64-bar maximum and minimum values
    rolling_max_H = df['high'].rolling(window=window_size).max()
    rolling_min_L = df['low'].rolling(window=window_size).min()

    max_H = rolling_max_H
    min_L = rolling_min_L
    range_HL = max_H - min_L

    def calculate_fractal(v2):
        fractal = 0
        if 25000 < v2 <= 250000:
            fractal = 100000
        elif 2500 < v2 <= 25000:
            fractal = 10000
        elif 250 < v2 <= 2500:
            fractal = 1000
        elif 25 < v2 <= 250:
            fractal = 100
        elif 12.5 < v2 <= 25:
            fractal = 12.5
        elif 6.25 < v2 <= 12.5:
            fractal = 12.5
        elif 3.125 < v2 <= 6.25:
            fractal = 6.25
        elif 1.5625 < v2 <= 3.125:
            fractal = 3.125
        elif 0.390625 < v2 <= 1.5625:
            fractal = 1.5625
        elif 0 < v2 <= 0.390625:
            fractal = 0.1953125
        return fractal

    def calculate_octave(v1, v2, mn, mx):
        range_ = v2 - v1
        sum_ = np.floor(np.log(calculate_fractal(v1) / range_) / np.log(2))
        octave = calculate_fractal(v1) * (0.5 ** sum_)
        mn = np.floor(v1 / octave) * octave
        if mn + octave > v2:
            mx = mn + octave
        else:
            mx = mn + (2 * octave)
        return mx

    def calculate_x_values(v1, v2, mn, mx):
        dmml = (v2 - v1) / 8
        x_values = []

        # Calculate the midpoints of each segment
        midpoints = [mn + i * dmml for i in range(8)]

        for i in range(7):
            x_i = (midpoints[i] + midpoints[i + 1]) / 2
            x_values.append(x_i)

        finalH = max(x_values)  # Maximum of the x_values is the finalH

        return x_values, finalH

    def calculate_y_values(x_values, mn):
        y_values = []

        for x in x_values:
            if x > 0:
                y = mn
            else:
                y = 0
            y_values.append(y)

        return y_values

    def calculate_mml(mn, finalH, mx):
        dmml = ((finalH - finalL) / 8) * 1.0699
        mml = (float([mx][0]) * 0.99875) + (dmml * 3) 
        # mml = (float([mx]) * 0.99875) + (dmml * 3) 

        ml = []
        for i in range(0, 16):
            calc = mml - (dmml * (i))
            ml.append(calc)

        murrey_math_levels = {
            "[-3/8]P": ml[14],
            "[-2/8]P": ml[13],
            "[-1/8]P": ml[12],
            "[0/8]P": ml[11],
            "[1/8]P": ml[10],
            "[2/8]P": ml[9],
            "[3/8]P": ml[8],
            "[4/8]P": ml[7],
            "[5/8]P": ml[6],
            "[6/8]P": ml[5],
            "[7/8]P": ml[4],
            "[8/8]P": ml[3],
            "[+1/8]P": ml[2],
            "[+2/8]P": ml[1],
            "[+3/8]P": ml[0]
        }
        

        return murrey_math_levels

    mn = np.min(min_L)
    mx = np.max(max_H)
    x_values, finalH = calculate_x_values(mn, mx, mn, mx)
    y_values = calculate_y_values(x_values, mn)
    finalL = np.min(y_values)
    mml = calculate_mml(finalL, finalH, mx)

    return mml


class AWATRCT(IStrategy):
    INTERFACE_VERSION = 3

    # ROI table:
    minimal_roi = {
        "0": 0.215,
        "40": 0.132,
        "87": 0.086,
        "360": 0.03
    }

    # Stoploss:
    stoploss = -0.318  # -0.042 # -0.318
    use_custom_stoploss = True

    # SMAOffset
    base_nb_candles_buy = IntParameter(20, 60, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(30, 60, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(0.96, 0.99, default=buy_params['low_offset'], space='buy', decimals=2, optimize=True)
    high_offset = DecimalParameter(0.99, 1.04, default=sell_params['high_offset'], decimals=2, space='sell', optimize=True)


    fast_ewo = IntParameter(5, 9, default=buy_params['ewof'], space='buy', optimize=True)
    fs_ewo = IntParameter(30, 50, default=35, space='buy', optimize=True)
    sf_ewo = IntParameter(35, 65, default=50, space='buy', optimize=True)
    ss_ewo = IntParameter(180, 200, default=200, space='buy', optimize=True)

    slow_ewo = IntParameter(80, 100, default=buy_params['ewos'], space='buy', optimize=True)
    ewo_lo_limit = DecimalParameter(-5, 0, default=-2.5, decimals=1, space='buy', optimize=True)
    ewo_xo_limit = DecimalParameter(-0.5, 0, default=-0.3, decimals=1, space='buy', optimize=True)

    ewo_bear_x = DecimalParameter(1.62, 1.80, default=buy_params['ewo_bear_mult'], decimals=2, space='buy', optimize=True)
    ewo_high_x = DecimalParameter(1.30, 1.40, default=buy_params['ewo_high_mult'], decimals=2, space='buy', optimize=True)
    ewo_bull_x = DecimalParameter(1.01, 1.3, default=buy_params['ewo_bull_mult'], decimals=3, space='buy', optimize=True)

    rsi_buy_low = IntParameter(30, 50, default=35, space='buy', optimize=True)
    rsi_buy_high = IntParameter(50, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)

    atr_buy = DecimalParameter(0.2, 4, default=1, decimals=1, space='buy', optimize=True)
    atr_sell = DecimalParameter(0.2, 4, default=1, decimals=1, space='sell', optimize=True)

    ### trailing stop loss optimiziation ###
    tsl_target5 = DecimalParameter(low=0.2, high=0.4, decimals=1, default=0.3, space='sell', optimize=True, load=True)
    ts5 = DecimalParameter(low=0.04, high=0.06, default=0.05, decimals=2,space='sell', optimize=True, load=True)
    tsl_target4 = DecimalParameter(low=0.15, high=0.2, default=0.2, decimals=2, space='sell', optimize=True, load=True)
    ts4 = DecimalParameter(low=0.03, high=0.05, default=0.045, decimals=2,  space='sell', optimize=True, load=True)
    tsl_target3 = DecimalParameter(low=0.10, high=0.15, default=0.15, decimals=2,  space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, decimals=3,  space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.06, high=0.10, default=0.1, decimals=3, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, decimals=3, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.04, high=0.06, default=0.06, decimals=3, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, decimals=3, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.02, high=0.04, default=0.03, decimals=3, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.008, high=0.015, default=0.013, decimals=3, space='sell', optimize=True, load=True)

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


    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = True

    ## Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'ioc'
    }

    # Optimal timeframe for the strategy
    timeframe = '15m'

    process_only_new_candles = True
    startup_candle_count = 89

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }


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


    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:


        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if current_candle['rsi'] < 80:

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
        else:
            for stop0 in self.tsl_target0.range:
                if (current_profit > stop0):
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} SWINGING FOR THE MOON!!!')
                    return 0.99

        return self.stoploss

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        dataframe['FEWO'] = EWO(dataframe, self.fast_ewo, self.fs_ewo)
        dataframe['SEWO'] = EWO(dataframe, self.sf_ewo, self.ss_ewo)
        dataframe['EWO_DIF'] = dataframe['FEWO'] - dataframe['SEWO']

        # EMAs for Bear/Bull
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.fast_ewo.value)
        dataframe['ema2'] = ta.EMA(dataframe, timeperiod=self.slow_ewo.value)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)  # 14)
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'], timeperiod=10)  # 5)

        # absolute ewo
        dataframe["ABSEWO"] = dataframe["EWO"]
        dataframe.loc[dataframe["EWO"] < 0, "ABSEWO"] = dataframe["EWO"] * - 1
        # seperate pos and neg ewo
        dataframe.loc[dataframe["EWO"] > 0, "POSEWO"] = dataframe["EWO"]
        dataframe.loc[dataframe["EWO"] < 0, "NEGEWO"] = dataframe["EWO"]
        dataframe["POSEWO"].fillna(method="ffill", inplace=True)
        dataframe["NEGEWO"].fillna(method="ffill", inplace=True) 
        

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
        dataframe['min'] = dataframe["EWO"].expanding(self.slow_ewo.value).min()
        dataframe['max'] = dataframe["EWO"].expanding(self.slow_ewo.value).max()

        dataframe['dif'] = dataframe['max'].sub(dataframe['min'])
        dataframe["ewoNorm"] = dataframe["EWO"] / dataframe['dif']
        dataframe['ewoNormMa'] =  ta.SMA(dataframe["ewoNorm"], timeperiod=self.fast_ewo.value)
        dataframe['ewoNormMaHigh'] = 0.0618

        # angle of the ewoNorm
        candlesBack2 = 1
        backQuote2 = dataframe['ewoNorm'].shift(candlesBack2)
        deltaX2 = candlesBack2
        deltaY2 = dataframe['ewoNorm'] - backQuote2
        dataframe['ewoAngleRad'] = (deltaY2 / deltaX2).apply(lambda x: math.atan(x))
        dataframe["ewo_angle_up"] = dataframe['ewoAngleRad'] > -0.0001
        dataframe["ewo_angle_down"] = dataframe['ewoAngleRad'] < 0.0001

        emaPercent = dataframe['ema'] / dataframe['ema2']

        avg_positive_ewo = dataframe["POSEWO"].rolling(self.slow_ewo.value).mean()
        avg_negative_ewo = dataframe["NEGEWO"].rolling(self.slow_ewo.value).mean()
        avg_abs_ewo = dataframe["ABSEWO"].rolling(self.slow_ewo.value).mean()

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

        #ATR active MA Offset adjustments
        dataframe['atr_pcnt'] = (qtpylib.atr(dataframe, window=5) / (dataframe['close'] + dataframe['open'] / 2))
        dataframe['atr_buy'] = dataframe['atr_pcnt'] / self.atr_buy.value
        dataframe['atr_sell'] = dataframe['atr_pcnt'] / self.atr_sell.value

        dataframe['ma_lo'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (self.low_offset.value - (dataframe['atr_buy']))
        dataframe['ma_hi'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (self.high_offset.value + (dataframe['atr_sell']))

        dataframe['lo_average'] = dataframe['ewo_low'].expanding().mean()
        dataframe['hi_average'] = dataframe['ewo_high'].expanding().mean()
        dataframe['min_of_mins'] = dataframe['ewo_low'].expanding(self.slow_ewo.value).min()
        dataframe['max_of_mins'] = dataframe['ewo_low'].expanding(self.slow_ewo.value).max()
        dataframe['min_of_maxs'] = dataframe['ewo_high'].expanding(self.slow_ewo.value).min()
        dataframe['max_of_maxs'] = dataframe['ewo_high'].expanding(self.slow_ewo.value).max()

        murrey_math_levels = calculate_murrey_math_levels(dataframe)
        for level, value in murrey_math_levels.items():
            dataframe[level] = value

        # Track Order Book Depth    
        ob = self.dp.orderbook(metadata['pair'], 100)
        levels = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
        df = pd.DataFrame(ob['asks'], columns=['price', 'volume'])
        df['mid_price'] = (ob['bids'][0][0] + ob['asks'][0][0]) / 2

        seller_data = []
        seller_df = pd.DataFrame(columns=['level', 'total_volume', 'average_price'])

        for level in levels:
            upper_bound = df['mid_price'] * (1 + level)
            lower_bound = df['mid_price'] * (1 - level)
            
            within_level = df[(df['price'] <= upper_bound) & (df['price'] >= lower_bound)]
            
            total_volume = within_level['volume'].sum()
            
            if total_volume > 0:
                average_price = (within_level['volume'] * within_level['price']).sum() / total_volume
            else:
                average_price = 0  # Set to 0 if no prices within the level
            
            seller_data.append({'level': level, 'total_volume': total_volume, 'average_price': average_price})

        seller_df = pd.DataFrame(seller_data)

        for level in levels:
            column_name = f'{level}_TV'
            total_volume = seller_df[seller_df['level'] == level]['total_volume'].values[0]
            most_recent_row_index = dataframe.index[-1]
            dataframe.at[most_recent_row_index, column_name] = total_volume

        df = pd.DataFrame(ob['bids'], columns=['price', 'volume'])
        df['mid_price'] = (ob['bids'][0][0] + ob['asks'][0][0]) / 2

        # levels = [-0.005, -0.01, -0.02, -0.03, -0.04, -0.05]
        buyer_data = []
        buyer_df = pd.DataFrame(columns=['level', 'total_volume', 'average_price'])

        for level in levels:
            upper_bound = df['mid_price'] * (1 + level)
            lower_bound = df['mid_price'] * (1 - level)
            
            within_level = df[(df['price'] <= upper_bound) & (df['price'] >= lower_bound)]
            
            total_volume = within_level['volume'].sum()
            
            if total_volume > 0:
                average_price = (within_level['volume'] * within_level['price']).sum() / total_volume
            else:
                average_price = 0  # Set to 0 if no prices within the level
            
            buyer_data.append({'level': level, 'total_volume': total_volume, 'average_price': average_price})

        buyer_df = pd.DataFrame(buyer_data)

        for level in levels:
            column_name = f'-{level}_TV'
            total_volume = buyer_df[buyer_df['level'] == level]['total_volume'].values[0]
            most_recent_row_index = dataframe.index[-1]
            dataframe.at[most_recent_row_index, column_name] = -total_volume

        return dataframe

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if current_candle['rsi'] < 40:
            self.dp.send_msg("Deploying Maximum Capital")
            if self.config['stake_amount'] == 'unlimited':
                # Use entire available wallet during favorable conditions when in compounding mode.
                return self.wallets.get_total_stake_amount() / self.config['max_open_trades']

            else:
                # Compound profits during favorable conditions instead of using a static stake.
                return self.wallets.get_total_stake_amount() / self.config['max_open_trades']

        else:
            self.dp.send_msg(f"Capital Conservation Mode Enabled for {pair}")
            return (self.wallets.get_total_stake_amount() / 1.5) / self.config['max_open_trades']
        print("PROPOSED STAKE:::::::::::::::::::::::::::::::::::::::::::::::")
        print(proposed_stake, pair)
        # Use default stake amount.
        return proposed_stake

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['EWO'] < df['ewo_high']) &
                (df['close'] < df['ma_lo']) &
                (df["ewo_angle_up"]) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO < HIGH')

        # df.loc[
        #     (
        #         (qtpylib.crossed_above(df['EWO'], df['hi_average'])) &
        #         # (df['close'] < df['ma_lo']) &
        #         # (df["ewo_angle_up"]) &
        #         (df['rsi'] < self.rsi_buy_high.value) &
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'EWO XO HIGH')

        df.loc[
            (
                (qtpylib.crossed_above(df['EWO'], df['ewo_low'])) &
                (df['close'] < df['ma_lo']) &
                (df['EWO'] < self.ewo_lo_limit.value) &
                (df['rsi'] < self.rsi_buy_low.value) &
                (df['[2/8]P'] > df['close']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO XO LOW')

        df.loc[
            (
                (df['EWO'] < df['ewo_low']) &
                (df['close'] < df['ma_lo']) &
                (df['EWO'] < self.ewo_lo_limit.value) &
                (df['rsi'] < self.rsi_buy_low.value) &
                (df['[2/8]P'] > df['close']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO < LOW')

        df.loc[
            (
                ((df['angle_perc'] - df['angle_perc'].shift(1)) > 0.07 ) &
                ((df['angle_perc'].shift(1) - df['angle_perc'].shift(2)) < 0 ) &
                (df['angle_perc'] < 0) &
                (df['[2/8]P'] > df['close']) &
                (df['open'] < (df[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'Angle Change Neg')

        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                # (df['close'].shift() > df['ma_hi']) &
                (df['close'] > df['ma_hi']) &
                (df['EWO'] > df['ewo_high']) &
                (df['rsi'] <= df['rsi_ma']) &
                (df['EWO'] > 0) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'EWO High')

        df.loc[
            (
                (df['close'] > df['ma_hi']) &
                (df['rsi'] >= 89) &
                (df['EWO'] > 0) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'EWO rsi limit')

        df.loc[
            (
                (df['close'] > df['ma_hi']) &
                (df['rsi'] >= 89) &
                (df['EWO'] < 0) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'EWO low rsi limit')


        return df

      # "0": 0.202,
      # "141": 0.094,
      # "380": 0.017,
      # "1459": 0


        # "pair_whitelist": [


        #     "ATOM/USDT",
        #     "XRP/USDT",
        #     "QNT/USDT",
        #     "AKT/USDT",
        #     "IOTA/USDT",
        #     "ADA/USDT",
        #     "OSMO/USDT",
        #     "BTC/USDT",
        #     "ETH/USDT",
        #     "CSPR/USDT",
        #     "ETC/USDT",
        #     "XDC/USDT",
        #     "LINK/USDT",
        #     "MATIC/USDT",
        #     "AVAX/USDT",
        #     "HBAR/USDT",
        #     "SCRT/USDT",
        #     "KAVA/USDT",
        #     "INJ/USDT",
        #     "NEAR/USDT",
        #     "DOT/USDT",
        #     "ALGO/USDT",
        #     "XLM/USDT",
        #     "RNDR/USDT",
        #     "ORAI/USDT",
        #     "AGIX/USDT",
        #     "GRT/USDT",
        #     "DOGE/USDT",
        #     "SOL/USDT",
        #     "TRX/USDT",
        #     "UNI/USDT",
        #     "LDO/USDT",
        #     "FIL/USDT",
        #     "VET/USDT",
        #     "EGLD/USDT",
        #     "THETA/USDT",
        #     "XTZ/USDT",
        #     "IMX/USDT",
        #     "ZEC/USDT",
        #     "GMX/USDT",
        #     "KLAY/USDT",
        #     "OP/USDT",
        #     "ROSE/USDT",
        #     "ENJ/USDT",
        #     "DYDX/USDT",
        #     "OCEAN/USDT",
        #     "FLR/USDT",
        #     "EWT/USDT",
        #     "JASMY/USDT",
        #     "APT/USDT",
        #     "APE/USDT",
        #     "EOS/USDT",
        #     "LUNC/USDT",
        #     "ANKR/USDT",
        #     "GALAX/USDT",
        #     "OPUL/USDT",
        #     "FET/USDT",
        #     "FTM/USDT",
        #     "EOSC/USDT",
        #     "VELO/USDT",
        #     "YFDAI/USDT",
        #     "BAT/USDT",
        #     "TRU/USDT",
        #     "YFI/USDT",
        #     "COMP/USDT",
        #     "AGLD/USDT",
        #     "SUSHI/USDT",
        #     "RLY/USDT",
        #     "CTI/USDT",

   # 164/1000:    230 trades. 145/1/84 Wins/Draws/Losses. Avg profit   0.44%. Median profit   0.63%. Total profit 501.08426833 USDT (  50.11%). Avg duration 7:29:00 min. Objective: -487.66331


   #  # Buy hyperspace params:
   #  buy_params = {
   #      "atr_buy": 2,  # value loaded from strategy
   #      "base_nb_candles_buy": 20,  # value loaded from strategy
   #      "ewo_bear_x": 1.62,  # value loaded from strategy
   #      "ewo_bull_x": 1.143,  # value loaded from strategy
   #      "ewo_high_x": 1.39,  # value loaded from strategy
   #      "ewo_lo_limit": -1.9,  # value loaded from strategy
   #      "fast_ewo": 7,  # value loaded from strategy
   #      "low_offset": 0.99,  # value loaded from strategy
   #      "rsi_buy_high": 70,  # value loaded from strategy
   #      "rsi_buy_low": 48,  # value loaded from strategy
   #      "slow_ewo": 96,  # value loaded from strategy
   #  }

   #  # Sell hyperspace params:
   #  sell_params = {
   #      "atr_sell": 2,  # value loaded from strategy
   #      "base_nb_candles_sell": 39,  # value loaded from strategy
   #      "high_offset": 0.99,  # value loaded from strategy
   #      "ts0": 0.008,  # value loaded from strategy
   #      "ts1": 0.01,  # value loaded from strategy
   #      "ts2": 0.02,  # value loaded from strategy
   #      "ts3": 0.04,  # value loaded from strategy
   #      "ts4": 0.05,  # value loaded from strategy
   #      "ts5": 0.04,  # value loaded from strategy
   #      "tsl_target0": 0.038,  # value loaded from strategy
   #      "tsl_target1": 0.048,  # value loaded from strategy
   #      "tsl_target2": 0.092,  # value loaded from strategy
   #      "tsl_target3": 0.14,  # value loaded from strategy
   #      "tsl_target4": 0.18,  # value loaded from strategy
   #      "tsl_target5": 0.2,  # value loaded from strategy
   #  }

   #  # Protection hyperspace params:
   #  protection_params = {
   #      "cooldown_lookback": 16,
   #      "lowprofit_only_per_pair": False,
   #      "lowprofit_protection_lookback": 5,
   #      "lowprofit_required_profit": 0.146,
   #      "lowprofit_stop_duration": 54,
   #      "lowprofit_trade_limit": 1,
   #      "maxdrawdown_allowed_drawdown": 0.074,
   #      "maxdrawdown_protection_lookback": 1,
   #      "maxdrawdown_stop_duration": 82,
   #      "maxdrawdown_trade_limit": 15,
   #      "stop_duration": 38,
   #      "stop_protection_only_per_pair": False,
   #      "stop_protection_only_per_side": True,
   #      "stop_protection_required_profit": 0.082,
   #      "stop_protection_trade_limit": 1,
   #      "use_lowprofit_protection": True,
   #      "use_maxdrawdown_protection": False,
   #      "use_stop_protection": True,
   #  }

   #  # ROI table:
   #  minimal_roi = {
   #      "0": 0.144,
   #      "131": 0.087,
   #      "582": 0.053,
   #      "1511": 0
   #  }

   #  # Stoploss:
   #  stoploss = -0.14

   #  # Trailing stop:
   #  trailing_stop = False  # value loaded from strategy
   #  trailing_stop_positive = None  # value loaded from strategy
   #  trailing_stop_positive_offset = 0.0  # value loaded from strategy
   #  trailing_only_offset_is_reached = False  # value loaded from strategy
