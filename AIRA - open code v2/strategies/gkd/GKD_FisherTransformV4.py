"""
# All tests made with 
# freqtrade - INFO - freqtrade 2025.6
# Python Version:		Python 3.12.8
# CCXT Version:		           4.4.91
#
###### future

freqtrade hyperopt --hyperopt-loss SharpeHyperOptLossDaily --strategy GKD_FisherTransformV2 --spaces buy sell roi stoploss trailing  \
    --config user_data/config_binance_futures_backtest_usdt.json --epochs 1000 --timerange 20241001-20250501 --timeframe-detail 5m --max-open-trades 3 -timeframe 1h

timeframe = "1h" 
can_short = True
set_leverage = 3
"""


from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                               IStrategy, IntParameter, RealParameter, merge_informative_pair, informative)
from pandas_ta import ema
import pandas as pd
import numpy as np
import talib
import datetime
import math
from typing import List, Tuple, Optional
from freqtrade.persistence import Trade
from freqtrade.exchange import timeframe_to_prev_date
from typing import Dict, List
from pandas import DataFrame
from freqtrade.persistence import Trade


class GKD_FisherTransformV4(IStrategy):
    # Strategy parameters
    timeframe = "4h"
    startup_candle_count = 30  
    minimal_roi = {}
    stoploss = -0.50
    use_custom_stoploss = True
    trailing_stop = False
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03

    can_short = True
    set_leverage = 3

    if can_short:
        # Buy hyperspace params:
        buy_params = {
            "atr_period": 20,
            "baseline_period": 5,
            "fisher_buy_threshold": 2.39,
            "fisher_period": 14,
            "fisher_smooth_long": 9,
            "fisher_smooth_short": 9,
            "goldie_locks": 2.85,
        }

        # Sell hyperspace params:
        sell_params = {
            "fisher_long_exit": -0.736,
            "fisher_short_exit": -0.548,
            "fisher_sell_threshold": 2.89,  # value loaded from strategy
        }

        # ROI table:
        minimal_roi = {
            "0": 0.373,
            "1019": 0.22,
            "3124": 0.076,
            "4482": 0
        }

        # Stoploss:
        stoploss = -0.524

        # Trailing stop:
        trailing_stop = False
        trailing_stop_positive = 0.127
        trailing_stop_positive_offset = 0.208
        trailing_only_offset_is_reached = True
        

        # Max Open Trades:
        max_open_trades = 3  # value loaded from Strategy

    else:    # spot
        # Buy hyperspace params:
        buy_params = {
            "atr_period": 21,
            "baseline_period": 11,
            "fisher_buy_threshold": 0.65,
            "fisher_period": 13,
            "fisher_smooth_long": 7,
            "goldie_locks": 1.6,
            "fisher_smooth_short": 6,  # value loaded from strategy
        }

        # Sell hyperspace params:
        sell_params = {
            "fisher_long_exit": 0.837,
            "fisher_sell_threshold": 2.89,  # value loaded from strategy
            "fisher_short_exit": 0.293,  # value loaded from strategy
        }

        # ROI table:
        minimal_roi = {
            "0": 0.871,
            "1787": 0.323,
            "2415": 0.118,
            "5669": 0
        }

        # Stoploss:
        stoploss = -0.591

        # Trailing stop:
        trailing_stop = False
        trailing_stop_positive = 0.345
        trailing_stop_positive_offset = 0.373
        trailing_only_offset_is_reached = True
        

        # Max Open Trades:
        max_open_trades = 3  # value loaded from Strategy


    # Custom parameters
    fisher_period = IntParameter(10, 15, default=buy_params.get('fisher_period'), space="buy", optimize=True)      # Lookback period for Fisher Transform
    fisher_smooth_long = IntParameter(3, 10, default=buy_params.get('fisher_smooth_long'), space="buy", optimize=True)         # EMA smoothing period for Fisher Transform
    fisher_smooth_short = IntParameter(3, 10, default=buy_params.get('fisher_smooth_short'), space="buy", optimize=can_short)         # EMA smoothing period for Fisher Transform
    fisher_short_exit = DecimalParameter(-1.0, 1.0, default=sell_params.get('fisher_short_exit'), decimals=3, space="sell", optimize=can_short)
    fisher_long_exit = DecimalParameter(-1.0, 1.0, default=sell_params.get('fisher_long_exit'), decimals=3, space="sell", optimize=True)
    fisher_sell_threshold = DecimalParameter(2.0, 3.9, default=sell_params.get('fisher_sell_threshold'), decimals=2, space="sell", optimize=False) # not used
    fisher_buy_threshold = DecimalParameter(-1.0, 2.5, default=buy_params.get('fisher_buy_threshold'), decimals=2, space="buy", optimize=True)
    baseline_period = IntParameter(5, 21, default=buy_params.get('baseline_period'), space="buy", optimize=True)        # Period for baseline EMA
    atr_period = IntParameter(7, 21, default=buy_params.get('atr_period'), space="buy", optimize=True)             # Period for ATR (volatility filter)
    goldie_locks = DecimalParameter(1.5, 3.0, default=buy_params.get('goldie_locks'), decimals=2, space="buy", optimize=True)    # Max multiplier for Goldie Locks Zone

    ATR_SL_short_Multip = DecimalParameter(1.0, 6.0, decimals=1, default=1.5, space="sell", optimize=True)
    ATR_SL_long_Multip = DecimalParameter(1.0, 6.0, decimals=1, default=1.5, space="sell", optimize=True)
    ATR_Multip = DecimalParameter(1.0, 6.0, decimals=1, default=1.5, space="sell", optimize=True)
    rr_long = DecimalParameter(1.0, 4.0, decimals=1, default=4.0, space="sell", optimize=True)
    rr_short = DecimalParameter(1.0, 4.0, decimals=1, default=4.0, space="sell", optimize=True)

    # Use this option with caution!
    # enables a full 1st slot and base+safety-order for the 2nd slot before you run out of money in your wallet.
    # (combo from the .ods-file: 4+3, see rows "Overbuy calculation")

    overbuy_factor = 1.295

    position_adjustment_enable = True
    initial_safety_order_trigger = -0.02
    max_so_multiplier_orig = 3
    safety_order_step_scale = 2
    safety_order_volume_scale = 1.8

    # just for initialization, now we calculate it...
    max_so_multiplier = max_so_multiplier_orig
    # We will store the size of stake of each trade's first order here
    cust_proposed_initial_stakes = {}
    # Amount the strategy should compensate previously partially filled orders for successive safety orders (0.0 - 1.0)
    partial_fill_compensation_scale = 1

    if max_so_multiplier_orig > 0:
        if safety_order_volume_scale > 1:
            # print(safety_order_volume_scale * (math.pow(safety_order_volume_scale,(max_so_multiplier - 1)) - 1))

            firstLine = safety_order_volume_scale * (
                math.pow(safety_order_volume_scale, (max_so_multiplier_orig - 1)) - 1
            )
            divisor = safety_order_volume_scale - 1
            max_so_multiplier = 2 + firstLine / divisor
            # max_so_multiplier = (2 +
            #                     (safety_order_volume_scale *
            #                      (math.pow(safety_order_volume_scale, (max_so_multiplier - 1)) - 1) /
            #                      (safety_order_volume_scale - 1)))
        elif safety_order_volume_scale < 1:
            firstLine = safety_order_volume_scale * (
                1 - math.pow(safety_order_volume_scale, (max_so_multiplier_orig - 1))
            )
            divisor = 1 - safety_order_volume_scale
            max_so_multiplier = 2 + firstLine / divisor
            # max_so_multiplier = (2 + (safety_order_volume_scale * (
            #        1 - math.pow(safety_order_volume_scale, (max_so_multiplier - 1))) / (
            #                                  1 - safety_order_volume_scale)))

    # Since stoploss can only go up and can't go down, if you set your stoploss here, your lowest stoploss will always be tied to the first buy rate
    # So disable the hard stoploss here, and use custom_sell or custom_stoploss to handle the stoploss trigger
    stoploss = -1

    def custom_exit(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        tag = super().custom_sell(pair, trade, current_time, current_rate, current_profit, **kwargs)
        if tag:
            return tag

        entry_tag = "empty"
        if hasattr(trade, "entry_tag") and trade.entry_tag is not None:
            entry_tag = trade.entry_tag

        if current_profit <= -0.35:
            return f"stop_loss ({entry_tag})"

        return None

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        filled_buys = trade.select_filled_orders(trade.entry_side)
        count_of_buys = len(filled_buys)

        if trade.calc_profit_ratio(rate) < 0.005:
            return False

        if (count_of_buys == 1) & (exit_reason == "roi"):
            return False
        # remove pair from custom initial stake dict only if full exit
        if trade.amount == amount and pair in self.cust_proposed_initial_stakes:
            del self.cust_proposed_initial_stakes[pair]
        return True

    # Let unlimited stakes leave funds open for DCA orders
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        **kwargs,
    ) -> float:
        custom_stake = proposed_stake / self.max_so_multiplier * self.overbuy_factor
        self.cust_proposed_initial_stakes[pair] = (
            custom_stake  # Setting of first stake size just before each first order of a trade
        )
        return custom_stake  # set to static 10 to simulate partial fills of 10$, etc

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float,
        max_stake: float,
        **kwargs,
    ) -> Optional[float]:
        if current_profit > self.initial_safety_order_trigger:
            return None

        filled_buys = trade.select_filled_orders(trade.entry_side)
        count_of_buys = len(filled_buys)

        if 1 <= count_of_buys <= self.max_so_multiplier_orig:
            # if (1 <= count_of_buys) and (open_trade_value < self.stake_amount * self.overbuy_factor):
            safety_order_trigger = abs(self.initial_safety_order_trigger) * count_of_buys
            if self.safety_order_step_scale > 1:
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (
                    abs(self.initial_safety_order_trigger)
                    * self.safety_order_step_scale
                    * (math.pow(self.safety_order_step_scale, (count_of_buys - 1)) - 1)
                    / (self.safety_order_step_scale - 1)
                )
            elif self.safety_order_step_scale < 1:
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (
                    abs(self.initial_safety_order_trigger)
                    * self.safety_order_step_scale
                    * (1 - math.pow(self.safety_order_step_scale, (count_of_buys - 1)))
                    / (1 - self.safety_order_step_scale)
                )

            if current_profit <= (-1 * abs(safety_order_trigger)):
                try:
                    # This returns first order actual stake size
                    actual_initial_stake = filled_buys[0].cost

                    # Fallback for when the initial stake was not set for whatever reason
                    stake_amount = actual_initial_stake

                    already_bought = sum(filled_buy.cost for filled_buy in filled_buys)
                    if trade.pair in self.cust_proposed_initial_stakes:
                        if self.cust_proposed_initial_stakes[trade.pair] > 0:
                            # This calculates the amount of stake that will get used for the current safety order,
                            # including compensation for any partial buys
                            proposed_initial_stake = self.cust_proposed_initial_stakes[trade.pair]
                            current_actual_stake = already_bought * math.pow(
                                self.safety_order_volume_scale, (count_of_buys - 1)
                            )
                            current_stake_preposition = proposed_initial_stake * math.pow(
                                self.safety_order_volume_scale, (count_of_buys - 1)
                            )
                            current_stake_preposition_compensation = (
                                current_stake_preposition
                                + abs(current_stake_preposition - current_actual_stake)
                            )
                            total_so_stake = lerp(
                                current_actual_stake,
                                current_stake_preposition_compensation,
                                self.partial_fill_compensation_scale,
                            )
                            # Set the calculated stake amount
                            stake_amount = total_so_stake
                        else:
                            # Fallback stake amount calculation
                            stake_amount = stake_amount * math.pow(
                                self.safety_order_volume_scale, (count_of_buys - 1)
                            )
                    else:
                        # Fallback stake amount calculation
                        stake_amount = stake_amount * math.pow(
                            self.safety_order_volume_scale, (count_of_buys - 1)
                        )

                    amount = stake_amount / current_rate
                    # logger.info(
                    #    f"Initiating safety order buy #{count_of_buys} "
                    #    f"for {trade.pair} with stake amount of {stake_amount}. "
                    #    f"which equals {amount}. "
                    #    f"Previously bought: {already_bought}. "
                    #    f"Now overall:{already_bought + stake_amount}. "
                    # )
                    return stake_amount
                except Exception as exception:
                    # logger.info(
                    #    f"Error occured while trying to get stake amount for {trade.pair}: {str(exception)}"
                    # )
                    # print(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}')
                    return None

        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate Fisher Transform
        dataframe["fisher"] = self.calculate_fisher(dataframe, self.fisher_period.value)
        
        # Smooth Fisher with EMA
        dataframe["fisher_smooth_long"] = ema(dataframe["fisher"], length=self.fisher_smooth_long.value)
        dataframe["fisher_smooth_short"] = ema(dataframe["fisher"], length=self.fisher_smooth_short.value)
        dataframe["fisher_trend_long"] = ema(dataframe["fisher_smooth_short"], length=21)
        dataframe["fisher_trend_short"] = ema(dataframe["fisher_smooth_short"], length=21)

        # Baseline (EMA)
        dataframe["baseline"] = ema(dataframe["close"], length=self.baseline_period.value)
        dataframe["baseline_diff"] = dataframe["baseline"].diff()
        dataframe["baseline_up"] = dataframe["baseline_diff"] > 0
        dataframe["baseline_down"] = dataframe["baseline_diff"] < 0

        # Volatility (ATR for Goldie Locks Zone)
        dataframe["atr"] = talib.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=self.atr_period.value
        )
        dataframe["goldie_min"] = dataframe["baseline"] - (dataframe["atr"] * self.goldie_locks.value)
        dataframe["goldie_max"] = dataframe["baseline"] + (dataframe["atr"] * self.goldie_locks.value)

        return dataframe

    def calculate_fisher(self, dataframe: DataFrame, period: int) -> pd.Series:
        # Fisher Transform calculation
        median_price = (dataframe["high"] + dataframe["low"]) / 2
        fisher = pd.Series(0.0, index=dataframe.index)

        for i in range(period, len(dataframe)):
            # Normalize price
            price_window = median_price.iloc[i-period:i]
            price_min = price_window.min()
            price_max = price_window.max()
            if price_max != price_min:
                norm = (median_price.iloc[i] - price_min) / (price_max - price_min)
                norm = 2 * norm - 1  # Scale to [-0.999, 0.999]
                norm = max(min(norm, 0.999), -0.999)  # Prevent division by zero
                # Apply Fisher Transform
                fisher.iloc[i] = 0.5 * np.log((1 + norm) / (1 - norm))
            else:
                fisher.iloc[i] = 0.0

        return fisher

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Standard Entry Logic (GKD-C Confirmation)
        dataframe.loc[
            # (dataframe["fisher_smooth"] > self.fisher_sell.value) & 
            # (dataframe["fisher_smooth"] < self.fisher_sell.value) &
            # (dataframe["fisher_smooth"] > 0) & 
            # (dataframe["fisher_smooth"].shift() < -0) &  
            (dataframe["fisher"] < self.fisher_sell_threshold.value) & 
            (dataframe["fisher_smooth_long"] < dataframe['fisher']), # &

             # Fisher indicates bullish reversal
            # (dataframe["baseline_up"]) &                              # Baseline confirms uptrend
            # (dataframe["close"] >= dataframe["goldie_min"]) &         # Within Goldie Locks Zone
            # (dataframe["close"] <= dataframe["goldie_max"]),
            ["enter_long", "enter_tag"]
            ] = [1, "fisher_long"]


        if self.can_short == True:
            dataframe.loc[
                (dataframe["fisher_smooth_short"] < self.fisher_sell_threshold.value) &  # Fisher indicates bearish reversal
                (dataframe["baseline_down"]) &                              # Baseline confirms downtrend
                (dataframe["close"] >= dataframe["goldie_min"]) &           # Within Goldie Locks Zone
                (dataframe["close"] <= dataframe["goldie_max"]),
            ["enter_short", "enter_tag"]
            ] = [1, "fisher_short"]

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit Logic: Fisher reverses or crosses neutral zone
        dataframe.loc[
            (
            (dataframe["fisher_smooth_long"].shift() > self.fisher_long_exit.value) & 
            (dataframe["fisher_smooth_long"] < self.fisher_long_exit.value) & 
            (dataframe["fisher_smooth_long"] > dataframe['fisher'])
            ),  # Fisher indicates weakening or bearish reversal
            ["exit_long", "exit_tag"]
            ] = [1, "exit_long"]

        #dataframe.loc[
        #    (
        #    (dataframe["fisher_smooth_long"] < self.fisher_sell_threshold.value)
        #    ),  # Fisher indicates weakening or bearish reversal
        #    "exit_long"
        #] = 1

        if self.can_short == True:
            dataframe.loc[
                (
                # (dataframe["fisher_smooth_short"].shift() < self.fisher_short_exit.value) & 
                (dataframe["fisher_smooth_short"] > 0)   # reduce short lost 
                # (dataframe["baseline_up"])                               # Baseline confirms downtrend
                ),
            ["exit_short", "exit_tag"]
            ] = [1, "exit_short"]

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Fonction de stop-loss personnalisée
        """
        # Récupération des données analysées pour la paire et le timeframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Conversion de la date d'ouverture du trade au format du timeframe
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        
        # Récupération de la bougie correspondant à l'ouverture du trade
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]

        # Logique de Stop Loss
        c2 = False
        if not trade_candle.empty:
            trade_candle = trade_candle.squeeze()
            if not trade.is_short:
                # Pour les positions longues, le SL est placé en dessous du prix d'entrée
                c2 = current_rate < trade.open_rate - trade_candle['atr'] * float(self.ATR_SL_long_Multip.value)
            else:
                # Pour les positions courtes, le SL est placé au-dessus du prix d'entrée
                c2 = current_rate > trade.open_rate + trade_candle['atr'] * float(self.ATR_SL_short_Multip.value)
            if c2:
                return -0.0001  # Déclenche le stop-loss

        # Logique de Take Profit
        c1 = False
        if not trade_candle.empty:
            trade_candle = trade_candle.squeeze()
            dist = trade_candle['atr'] * self.ATR_Multip.value
            if not trade.is_short:
                # Pour les positions longues, le TP est placé au-dessus du prix d'entrée
                c1 = current_rate > trade.open_rate + dist * float(self.rr_long.value)
            else:
                # Pour les positions courtes, le TP est placé en dessous du prix d'entrée
                c1 = current_rate < trade.open_rate - dist * float(self.rr_short.value)
            if c1:
                return -0.0001  # Déclenche le take-profit

        # Si aucune condition n'est remplie, retourne le stop-loss par défaut
        return self.stoploss


    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        return self.set_leverage if self.can_short else 1
