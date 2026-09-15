#v2:    incluye N periodos despues del cruce de pdi/mdi en ADX (ncross) de validez para la señal
#       incluye filtro de sma200 BTC 1h para no ir contra la corriente
#       incluye formateo
#       incluye custom_roi
#V2.1   incluye entry/exit por rsi crosses 
#       atr roi exit   
#V2.2   kill old trades (KOT custom exit) 
#       ichimoku filter
#       mfi exits
#       pasado atr roi exit a custom exits
#v4     Conversion a 5m
#4.2    Custom Stoploss: Ichimoku + n ATR's (1h) (no resultó, optimizar)
#       Custom Stoploss: Banda de Bollinger offseteada (no resultó, optimizar)
#       Custom Exit: BB (choque de banda de bollinger)
#5      Custom Exit: MFI Oversold / Overbought



################################################################################################################################
############################ IMPORTS ###########################################################################################
################################################################################################################################

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from technical import qtpylib
from pandas import DataFrame
from datetime import datetime, timezone, timedelta
from typing import Optional
from functools import reduce
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, RealParameter, informative, merge_informative_pair)
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.exchange import  timeframe_to_minutes #,timeframe_to_prev_date

logger = logging.getLogger(__name__)

################################################################################################################################
############################## CLASS DEFINITION ################################################################################
################################################################################################################################

class FTPv5w(IStrategy):

    leve = 3
    stakename = "USDT"              # PARA MENSAJES

    ### Strategy parameters ###
    exit_profit_only = True
    ignore_roi_if_entry_signal = False
    can_short = True
    use_exit_signal = True
    stoploss = -0.23*leve
    startup_candle_count: int = 200
    timeframe = '5m'

    # Custom exit
    use_custom_exit = True                  # Clear Old Trades (ver valores en hyperoptables) | atr roi
    use_custom_stoploss = False
    trailing_stop = False

    # DCA Parameters
    position_adjustment_enable = True       #enable ajust_trade_position() callback in the strategy
    max_entry_position_adjustment = 2       # 2 original
    max_dca_multiplier = 1                  # Maximum DCA multiplier sum_1_to_n(base_stake.(1+dca_increase)^n) where n is max_entry_position_adjustment
    dca_increase = 0.5                      # % increase on step, variar a 0.5 si dca_post_leverage = False
    dca_post_leverage = True                # True original
    step = 0.04                             # % steps, 
    step1 = -1*step
    step2 = -1.5*step
    step3 = -2*step
    step4 = -2.5*step    

    # ROI table:
    rm=leve
    minimal_roi = {
        "0": 0.1*leve,     #original=30% x1 (error?)
        #"30": 0.02*leve,   #original=2% x1
        #"120": 0.01*leve,  #original=1% x1
        "240": 0.008 
    }
################################################################################################################################
############################ custom methods ###################################################################################
################################################################################################################################
    
    def safe_shift(self, data, shift_periods=1):
        """
        Safely shift data whether it's numpy array or pandas Series
        """
        if isinstance(data, np.ndarray):
            # For numpy arrays, create a new array with NaN padding
            result = np.empty_like(data, dtype=float)
            result[:] = np.nan
            if shift_periods > 0:
                result[shift_periods:] = data[:-shift_periods]
            else:
                result[:shift_periods] = data[-shift_periods:]
            return result
        else:
            # For pandas Series, use built-in shift
            return data.shift(shift_periods)


################################################################################################################################
################################ CUSTOM ROI ###################################################################################
################################################################################################################################

    use_custom_roi = True
    atr_roi_coeff = 2.15            #actualizar para compatibilizar con 5m || ACT 5m

    def custom_roi(self, pair: str, trade: Trade, current_time: datetime, trade_duration: int,
                   entry_tag: str | None, side: str, **kwargs) -> float | None:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        atr_ratio = last_candle["atr_1h"] / last_candle["close_1h"]
        atr_roi = atr_ratio * self.atr_roi_coeff * self.leve

        return atr_roi # Returns the ATR value as ratio

# ##############################################################################################################################
################################ LEVERAGE ######################################################################################
################################################################################################################################

    #LEVERAGE CONFIG
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:
    #   """
    #   Customize leverage for each new trade. This method is only called in futures mode.
	#
    #    :param pair: Pair that's currently analyzed
    #    :param current_time: datetime object, containing the current datetime
    #    :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
    #    :param proposed_leverage: A leverage proposed by the bot.
    #    :param max_leverage: Max leverage allowed on this pair
    #    :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
    #    :param side: "long" or "short" - indicating the direction of the proposed trade
    #    :return: A leverage amount, which is between 1.0 and max_leverage.
    #    """
        return min(self.leve, max_leverage)
    
################################################################################################################################
################################ CONFIRM TRADE EXIT ############################################################################
################################################################################################################################

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Called right before placing a regular exit order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair for trade that's about to be exited.
        :param trade: trade object.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in base currency.
        :param rate: Rate that's going to be used when using limit orders
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param exit_reason: Exit reason.
            Can be any of ["roi", "stop_loss", "stoploss_on_exchange", "trailing_stop_loss",
                           "exit_signal", "force_exit", "emergency_exit"]
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True, then the exit-order is placed on the exchange.
            False aborts the process
        """

        trade_age = current_time - trade.open_date_utc
        profit_ratio = trade.calc_profit_ratio(rate)

        #logger.info(f"Trade age: {trade_age}, Profit ratio: {profit_ratio:.5f}, Exit reason: {exit_reason}")

        # Allow exit if small profit and trade is older than 4 hours
        if trade_age > timedelta(hours=4):
            return True

        # Otherwise, only allow exit if profit is at least n
        if profit_ratio >= 0.05:
            return True

        # Otherwise, block the exit
        return False

################################################################################################################################
################################ HYPEROPT RESULTS ##############################################################################
################################################################################################################################



################################################################################################################################
################################ HYPEROPTABLES #################################################################################
################################################################################################################################

    # protections
    Opt_prot = False
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=Opt_prot)
    stop_duration = IntParameter(12, 120, default=72, space="protection", optimize=Opt_prot)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=Opt_prot)

    #buyspace - Signal types
    optimize_buy_signals = True
    use_FTP_longs = BooleanParameter(default=True, space="buy", optimize=optimize_buy_signals)
    use_FTP_shorts = BooleanParameter(default=True, space="buy", optimize=optimize_buy_signals)
    #use_RSI_Cross_longs = BooleanParameter(default=False, space="buy", optimize=optimize_buy_signals)
    #use_RSI_Cross_shorts = BooleanParameter(default=False, space="buy", optimize=optimize_buy_signals)
    use_ichimoku_longs = BooleanParameter(default=True, space="buy", optimize=optimize_buy_signals)
    use_ichimoku_shorts = BooleanParameter(default=True, space="buy", optimize=optimize_buy_signals)

    #buyspace - general
    optimize_buy_general = True
    use_informative_pairs = BooleanParameter(default=True, space="buy", optimize=optimize_buy_general) #use btc_sma200_1h filter
    
    #buyspace - FTP (RSI, ADX, sma200, btc_sma200_1h)
    optimize_ftp = True
    ncrossed = IntParameter(1, 6, default=3, space="buy", optimize=optimize_ftp)  #n of candles after mdi or pdi cross on adx and signal is valid

    #sellspace - ATR Exit (in adjust_trade_position) (also not hyperoptable parameters in strat parameters for custom_roi)
    optimize_atr_exit = True
    use_atr_exit = BooleanParameter(default=True, space="sell", optimize=optimize_atr_exit)
    atr_exit_multiplier = DecimalParameter(0.5, 5, decimals=2, default=1.8, space="sell", optimize=optimize_atr_exit)                          #apem 0.25 ok (custom roi true) |0.08 ok (custom roi False)
    atr_exit_percent = DecimalParameter(0.1, 1, decimals=1, default=1, space="sell", optimize=optimize_atr_exit)                                #apep
    #atr_minimum_roi_threshold = DecimalParameter(0.005, 0.500, decimals=3, default=0.010, space="sell", optimize=optimize_atr_exit)             #amrt

    #sellspace - Signal types
    optimize_sell_signals = True
    use_FTP_longs_exit = BooleanParameter(default=True, space="sell", optimize=optimize_sell_signals)
    use_FTP_shorts_exit = BooleanParameter(default=True, space="sell", optimize=optimize_sell_signals)
    #use_RSI_Cross_longs_exit = BooleanParameter(default=False, space="sell", optimize=optimize_sell_signals)
    #use_RSI_Cross_shorts_exit = BooleanParameter(default=False, space="sell", optimize=optimize_sell_signals)
    use_mfi_exit_short = BooleanParameter(default=False, space="sell", optimize=optimize_sell_signals)                                 #mfi exit short & long (custom)
    use_mfi_exit_long = BooleanParameter(default=False, space="sell", optimize=optimize_sell_signals)                                  #mfi exit short & long (custom)
    use_bollinger_exit_long = BooleanParameter(default=True, space="sell", optimize=optimize_sell_signals)
    use_bollinger_exit_short = BooleanParameter(default=True, space="sell", optimize=optimize_sell_signals)

    #sellspace - Kill Old Trades (KOT)
    optimize_KOT = True
    use_kot =  BooleanParameter(default=False, space="sell", optimize=optimize_KOT)
    sell_clear_old_trade = IntParameter(120, 360, default=180, space="sell", optimize=optimize_KOT)             #edad en velas (120 velas son 10 horas)
    sell_clear_old_trade_profit = IntParameter(0, 10, default=1, space="sell", optimize=optimize_KOT)           #perdidas admisibles al cierre en -n%

    #sellspace - custom stoploss
    optimize_customSL = True
    use_ichiATR_csl = BooleanParameter(default=False, space="sell", optimize=optimize_customSL)                         #ichimoku + ATR csl
    csl_atr_mult = DecimalParameter(0.5, 3.0, default=3.0, space="sell", decimals=2, optimize=optimize_customSL)        #ichimoku + ATR csl
    use_BB_csl = BooleanParameter(default=True, space="sell", optimize=optimize_customSL)                               #bollinger + ATR
    csl_atr_mult_BB = DecimalParameter(0.5, 1.0, default=6.0, space="sell", decimals=2, optimize=optimize_customSL)





################################################################################################################################
################################ PROTECTIONS ###################################################################################
################################################################################################################################

    ### Protections ###
    @property
    def protections(self):
        """
            Defines the protections to apply during trading operations.
        """
        prot = []

        """  
        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value 
        })"""
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,
                "trade_limit": 1,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": True
            })

        return prot

################################################################################################################################
################################ DCA = DOLLAR COST AVERAGING ###################################################################
################################################################################################################################

    ### Dollar Cost Averaging (DCA) ###
    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        """
            Calculates the stake amount to use for a trade, adjusted dynamically based on the DCA multiplier.
            - The proposed stake is divided by the maximum DCA multiplier (`self.max_dca_multiplier`)
              to determine the adjusted stake.
            - If the adjusted stake is lower than the allowed minimum (`min_stake`), it is automatically increased
              to meet the minimum stake requirement.
        """
        # Calculates the adjusted stake amount based on the DCA multiplier.
        adjusted_stake = proposed_stake / self.max_dca_multiplier

        # Automatically adjusts to the minimum stake if it is too low.
        if adjusted_stake < min_stake:
            adjusted_stake = min_stake

        return adjusted_stake



    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be
        increased or decreased.
        This means extra buy or sell orders with additional fees.
        Only called when `position_adjustment_enable` is set to True.
        """
        # current_profit: ratio de profit actual - si se mult. x 100% da profit en %

        # Obtain pair dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        last_candle = dataframe.iloc[-1].squeeze()
            
        if current_profit > self.step1 and trade.nr_of_successful_entries == 1:         #filtros echadores para que no se active
            return None
        if current_profit > self.step2 and trade.nr_of_successful_entries == 2:  
            return None
        if current_profit > self.step3 and trade.nr_of_successful_entries == 3:
            return None

            """else:  # For shorts
            if current_profit > 0.10 and trade.nr_of_successful_exits == 0:
                return (-(trade.stake_amount / 2), "10%_profit_closing_half")
            
            if self.use_atr_exit.value and current_profit > atr_exit_profit:      #atr partial exit
                self.dp.send_msg(msg_atrExit)
                logger.info(msg_atrExit)
                return (-(trade.stake_amount * self.atr_exit_percent.value),  "ATR EXIT VIA ATP")  
            
            if current_profit > self.step1 and trade.nr_of_successful_entries == 1:      #filtros echadores para que no se active
                return None
            if current_profit > self.step2 and trade.nr_of_successful_entries == 2:  
                return None
            if current_profit > self.step3 and trade.nr_of_successful_entries == 3:
                return None"""
        
        try:
            if self.dca_post_leverage:
                stake_amount = filled_entries[0].cost                   #post leverage
            else:
                stake_amount = filled_entries[0].stake_amount_filled    #pre leverage
       
            # If already more than first entry, increase per your dca_increase
            if count_of_entries > 1:
                stake_amount = stake_amount * (1 + (count_of_entries - 1) * self.dca_increase)  # dca_increase % increase per additional entry
            
            # --- LOGGING GUARD: only log once per entry count ---
            entry_count = trade.nr_of_successful_entries
            last_logged = trade.get_custom_data(key='last_dca_log_count', default=None)
            if last_logged is None or last_logged != entry_count:
                msg = f"🟡 DCA: Increasing {trade.pair} - new stake {stake_amount:.2f} {self.stakename} - total dca entries {count_of_entries} ({current_time})"
                logger.info(msg)
                self.dp.send_msg(msg)
                # Also log wallet if you like:
                wallet_msg = f"Available funds: {self.wallets.get_free(self.stakename):.2f} / {self.wallets.get_total(self.stakename):.2f}"
                logger.info(wallet_msg)
                # Record that we logged for this entry count
                trade.set_custom_data(key='last_dca_log_count', value=entry_count)
            # --- end logging guard ---

            return stake_amount, "DCA INCREASING"
        
        except Exception as exception:
            logger.error(f"Error adjusting DCA position for the pair {trade.pair}: {exception}")
            return None
     

################################################################################################################################
######################################## CUSTOM EXIT LOGIC #####################################################################
################################################################################################################################
    
    #CUSTOM EXIT: INC KILL OLD TRADES [KOT] & ATR ROI
    timeframe_minutes = timeframe_to_minutes(timeframe)

    def custom_exit(self, pair: str, trade: Trade, 
                    current_time: datetime, current_rate: float, 
                    current_profit: float,
                    **kwargs) -> Optional[Union[str, bool]]:

        # current_profit: ratio de profit actual - si se mult. x 100% da profit en %
        # Obtain pair dataframe
        
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        last_candle = dataframe.iloc[-1].squeeze()
##########################################################################
        #❇️ATR CUSTOM EXIT
        atr_exit_profit = self.atr_exit_multiplier.value * self.leve * last_candle["atr_1h"]
        #atr_exit_profit = trade.stake_amount *  last_candle["atr_rel_1h"] * self.leve * self.atr_exit_multiplier.value  #base*atr%*lev*apem = variacion esperable de la stake en apem atr rels

          
        if self.use_atr_exit.value and current_profit > atr_exit_profit:      #atr exit
            #logger.info(f"atr exit profit limit for pair {trade.pair}: {atr_exit_profit:.2f} ({100*atr_exit_profit:.2f}%) at {current_time}")
            #logger.info(f"Info on trade: time {current_time} pair {trade.pair}: stake_amount {trade.stake_amount:.2f}")
            msg_atrExit = f"❇️ ATR: {trade.pair} {self.atr_exit_percent.value*100}% exit!! Profit {100*current_profit:.2f}% ({current_profit*trade.stake_amount:.2f} {self.stakename}) at {current_time}. Total stake was {trade.stake_amount:.2f} {self.stakename} ({count_of_entries} entry(s))."
            self.dp.send_msg(msg_atrExit)
            logger.info(msg_atrExit)
            #return (-(trade.stake_amount * self.atr_exit_percent.value), "ATR EXIT VIA ATP")
            return ("ATR CUSTOM EXIT")

##########################################################################
        #♾️BB CUSTOM EXIT
        min_bb_roi = 0.02    
        if trade.is_short and self.use_bollinger_exit_short.value and current_rate <= last_candle['BB_lower_1h'] and current_profit > min_bb_roi:
            msg_bbexit = f"♾️ BB: {trade.pair} short exit!! Profit {100*current_profit:.2f}% ({current_profit*trade.stake_amount:.2f} {self.stakename}) at {current_time}. Total stake was {trade.stake_amount:.2f} {self.stakename} ({count_of_entries} entry(s))."
            self.dp.send_msg(msg_bbexit)
            logger.info(msg_bbexit)            
            return("BOLLINGER SHORT EXIT")

        if not trade.is_short and self.use_bollinger_exit_long.value and current_rate >= last_candle['BB_upper_1h'] and current_profit > min_bb_roi:
            msg_bbexit = f"♾️ BB: {trade.pair} long exit!! Profit {100*current_profit:.2f}% ({current_profit*trade.stake_amount:.2f} {self.stakename}) at {current_time}. Total stake was {trade.stake_amount:.2f} {self.stakename} ({count_of_entries} entry(s))."
            self.dp.send_msg(msg_bbexit)
            logger.info(msg_bbexit)       
            return("BOLLINGER LONG EXIT")        
##########################################################################
        #🚀​MFI OVERSOLD/OVERBOUGHT CUSTOM EXIT
        min_mfi_roi = 0.06
        if trade.is_short and self.use_mfi_exit_short.value and last_candle['mfi'] <= 11 and current_profit > min_mfi_roi:
            msg_mfiexit = f"🚀​ MFI: {trade.pair} short exit!! Profit {100*current_profit:.2f}% ({current_profit*trade.stake_amount:.2f} {self.stakename}) at {current_time}. Total stake was {trade.stake_amount:.2f} {self.stakename} ({count_of_entries} entry(s))."
            self.dp.send_msg(msg_mfiexit)
            logger.info(msg_mfiexit)            
            return("MFI SHORT EXIT")

        if not trade.is_short and self.use_mfi_exit_long.value and last_candle['mfi'] >= 89 and current_profit > min_mfi_roi:
            msg_mfiexit = f"🚀​ MFI: {trade.pair} long exit!! Profit {100*current_profit:.2f}% ({current_profit*trade.stake_amount:.2f} {self.stakename}) at {current_time}. Total stake was {trade.stake_amount:.2f} {self.stakename} ({count_of_entries} entry(s))."
            self.dp.send_msg(msg_mfiexit)
            logger.info(msg_mfiexit)       
            return("MFI LONG EXIT") 
##########################################################################
        #👻KILL OLD TRADES (KOT)        
        
        if self.use_kot.value:

            trade_age = current_time - trade.open_date_utc                              #edad del trade
                    
            if (current_time - timedelta(minutes=int(self.timeframe_minutes)) >= trade.open_date_utc):  #si la trade es mas de 1 vela de edad

                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)     #obtener df
                current_candle = dataframe.iloc[-1].squeeze()                           #obtener ultima vela
                current_profit = trade.calc_profit_ratio(current_candle['close'])       #obtener profit al ultimo cierre de vela

                #   si la trade es mas vieja que velas=self.sell_clear_old_trade.value

            if current_time - timedelta(minutes=int(self.timeframe_minutes * self.sell_clear_old_trade.value)) >= trade.open_date_utc:
                if (current_profit >= (-0.01 * self.sell_clear_old_trade_profit.value)):       
                    #si el profit es igual o mayor que el valor mínimo de cierre
                    msgKOT=(f"👻KOT: at {current_time} {trade.pair} {trade.trade_direction} is being closed with age {trade_age}. Profit {current_profit*100:.2f}%")
                    logger.info(msgKOT)
                    self.dp.send_msg(msgKOT)
                    return f"KILL OLD TRADE"

##########################################################################
        #🦜FTP base exits
        """min_ftp_long_roi = 0.01
        if self.use_FTP_longs_exit.value:
            if last_candle['adx_1h'] < 25 and last_candle['rsi_1h'] > 50 and current_profit >= min_ftp_long_roi:
                msg_ftp_ex = f"🦜​ FTP: {trade.pair} long exit!! Profit {100*current_profit:.2f}% ({current_profit*trade.stake_amount:.2f} {self.stakename}) at {current_time}. Total stake was {trade.stake_amount:.2f} {self.stakename} ({count_of_entries} entry(s))."
                logger.info(msg_ftp_ex)
                self.dp.send_msg(msg_ftp_ex)
                return "FTP EXIT LONG"          

        min_ftp_short_roi = 0.01
        if self.use_FTP_shorts_exit.value:
            if last_candle['adx_1h'] < 25 and last_candle['rsi_1h'] < 50 and current_profit >= min_ftp_short_roi:
                msg_ftp_ex = f"🦜​ FTP: {trade.pair} short exit!! Profit {100*current_profit:.2f}% ({current_profit*trade.stake_amount:.2f} {self.stakename}) at {current_time}. Total stake was {trade.stake_amount:.2f} {self.stakename} ({count_of_entries} entry(s))."
                logger.info(msg_ftp_ex)
                self.dp.send_msg(msg_ftp_ex)
                return "FTP EXIT SHORT" """    

               

################################################################################################################################
######################################## CUSTOM STOPLOSS LOGIC #################################################################
################################################################################################################################

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, **kwargs) -> float | None:
        """
        Custom stoploss logic based on the Ichimoku cloud and ATR. Returns a stoploss ratio relative to current_rate.

        :param pair: Pair that's currently analyzed
        :param trade: Trade to calculate stoploss for
        :param current_time: Current timestamp
        :param current_rate: The current price rate
        :return: Stoploss ratio (fraction of current_rate) or None to keep existing stoploss
        """
        # Retrieve dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        last_candle = dataframe.iloc[-1]

        # ichimoku cloud support/resistanice + n atr_1d
        if self.use_ichiATR_csl.value:
            # Get the last candle and relevant indicators
            senkou_span_a = last_candle.get('senkou_span_a_1d', np.nan)
            senkou_span_b = last_candle.get('senkou_span_b_1d', np.nan)
            atr_1h = last_candle.get('atr_1d', np.nan)

            # Validate indicators
            if np.isnan(atr_1h) or np.isnan(senkou_span_a) or np.isnan(senkou_span_b):
                return None

            # Determine cloud boundary
            ichimoku_support = min(senkou_span_a, senkou_span_b)
            ichimoku_resistance = max(senkou_span_a, senkou_span_b)

            # Calculate absolute stoploss price
            if trade.is_short:
                raw_sl_price = ichimoku_resistance + (self.csl_atr_mult.value * atr_1h)
            else:
                raw_sl_price = ichimoku_support - (self.csl_atr_mult.value * atr_1h)


            # Ensure raw_sl_price moves in allowed direction
            if trade.is_short and raw_sl_price < current_rate:
                raw_sl_price = current_rate

            elif not trade.is_short and raw_sl_price > current_rate:
                raw_sl_price = current_rate


            # Convert absolute price to relative stoploss ratio
            # Ratio = absolute distance / current_rate
            # LEVERAGE: trade will see ratio * leverage, so we multiply that before sending
            sl_ratio = self.leve * abs((current_rate - raw_sl_price) / current_rate)

        #######################################################################################################
        # Bollinger Bands + ATR
        elif self.use_BB_csl.value:
            bb_upper_limit = last_candle.get('BB_upper_SL_1h', np.nan)
            bb_lower_limit = last_candle.get('BB_lower_SL_1h', np.nan)
            if np.isnan(bb_lower_limit) or np.isnan(bb_upper_limit):
                return None
            if trade.is_short:
                raw_sl_price = bb_upper_limit 
            else:
                raw_sl_price = bb_lower_limit 
            # Enforce direction
            if trade.is_short and raw_sl_price < current_rate:
                raw_sl_price = current_rate
            if not trade.is_short and raw_sl_price > current_rate:
                raw_sl_price = current_rate

            # Compute relative stoploss ratio
            sl_ratio = self.leve * abs((current_rate - raw_sl_price) / current_rate)
        
        else:
            return None
        #######################################################################################################
        # Send debug message once every x hours
        last_logged_time = trade.get_custom_data(key='last_sl_log_time', default=None)
        # Parse stored ISO timestamp
        if last_logged_time:
            try:
                last_time = datetime.fromisoformat(last_logged_time)
            except Exception:
                last_time = None
        else:
            last_time = None
        
        # If never logged or more than x hours passed
        if last_time is None or (current_time - last_time).total_seconds() >= 10 * 3600:
            tradedir = 'SHORT' if trade.is_short else 'LONG'
            msg_atrSL = (
                f"🛑 CSL: {trade.pair} Current Rate: {current_rate:.2f}"
                f" SL: {raw_sl_price:.2f} ({sl_ratio:.2f}%) ({tradedir}) ({current_time})"
            )
            self.dp.send_msg(msg_atrSL)
            logger.info(msg_atrSL)

            # Store ISO string of log time
            trade.set_custom_data(key='last_sl_log_time', value=current_time.isoformat())

            # Return as positive fraction (<1). Freqtrade uses absolute value, sign ignored.
        return sl_ratio

################################################################################################################################
######################################## INFORMATIVE PAIRS #####################################################################
################################################################################################################################

    #INFORMATIVE PAIRS

    # Define BTC/STAKE informative pair. Available in populate_indicators and other methods as
    # 'btc_rsi_1h'. Current stake currency should be specified as {stake} format variable 
    # instead of hard-coding actual stake currency. Available in populate_indicators and other 
    # methods as 'btc_usdt_sma200_1h' (when stake currency is USDT).
    @informative('1h', 'BTC/{stake}:{stake}')
    def populate_indicators_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        return dataframe
    
    # Define informative upper timeframe for each pair. Decorators can be stacked on same 
    # method. Available in populate_indicators as 'sma200_1h'
    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        #SMA 200
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        #ATR & ATR REL 1h
        dataframe['sma14'] = ta.SMA(dataframe, timeperiod=14)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_rel'] = ( dataframe['atr'] / dataframe['sma14'])

        #FTP: ADX, RSI
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)  # ADX for trend strength
        dataframe['pdi'] = ta.PLUS_DI(dataframe, timeperiod=14)  # Positive directional index
        dataframe['mdi'] = ta.MINUS_DI(dataframe, timeperiod=14)  # Negative directional index
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)  # RSI for momentum
        
        #MFI
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        
        #ichimoku
        high = dataframe['high']; low = dataframe['low']; close = dataframe['close']
        dataframe['tenkan_sen'] = (high.rolling(9).max() + low.rolling(9).min()) / 2
        dataframe['kijun_sen'] = (high.rolling(26).max() + low.rolling(26).min()) / 2
        dataframe['senkou_span_a'] = ((dataframe['tenkan_sen'] + dataframe['kijun_sen']) / 2).shift(26)
        dataframe['senkou_span_b'] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        dataframe['chikou_span'] = close.shift(-26)

        #bollinger
        sma = dataframe['close'].rolling(14).mean()
        std_dev = dataframe['close'].rolling(14).std()
        dataframe['BB_upper'] = sma + std_dev * 2.0
        dataframe['BB_lower'] = sma - std_dev * 2.0             

        #bollinger SL levels
        dataframe['BB_upper_SL'] = dataframe['BB_upper'] + self.csl_atr_mult_BB.value*dataframe['atr']
        dataframe['BB_lower_SL'] = dataframe['BB_lower'] - self.csl_atr_mult_BB.value*dataframe['atr']
        return dataframe

    @informative('1d')
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        #SMA 200
        #dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        #ATR & ATR REL 1h
        #dataframe['sma14'] = ta.SMA(dataframe, timeperiod=14)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        #dataframe['atr_rel'] = ( dataframe['atr'] / dataframe['sma14'])
        
        #FTP: ADX, RSI
        #dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)  # ADX for trend strength
        #dataframe['pdi'] = ta.PLUS_DI(dataframe, timeperiod=14)  # Positive directional index
        #dataframe['mdi'] = ta.MINUS_DI(dataframe, timeperiod=14)  # Negative directional index
        #dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)  # RSI for momentum
        
        #MFI
        #dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)"""
        
        #ichimoku
        #high = dataframe['high']; low = dataframe['low']; close = dataframe['close']
        #dataframe['tenkan_sen'] = (high.rolling(9).max() + low.rolling(9).min()) / 2
        #dataframe['kijun_sen'] = (high.rolling(26).max() + low.rolling(26).min()) / 2
        #dataframe['senkou_span_a'] = ((dataframe['tenkan_sen'] + dataframe['kijun_sen']) / 2).shift(26)
        #dataframe['senkou_span_b'] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        #dataframe['chikou_span'] = close.shift(-26)

        #bollinger
        #sma = dataframe['close'].rolling(14).mean()
        #std_dev = dataframe['close'].rolling(14).std()
        #dataframe['BB_upper'] = sma + std_dev * 2.0
        #dataframe['BB_lower'] = sma - std_dev * 2.0        
        
        return dataframe
    

    ################################################################################################################################
    ######################################## populate_indicators ###################################################################
    ################################################################################################################################

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
            Calculates technical indicators used to define entry and exit signals.
        """

        #FTP
        #dataframe['sma14'] = ta.SMA(dataframe, timeperiod=14)
        #dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        #dataframe['atr_rel'] = ( dataframe['atr'] / dataframe['sma14'])

        #dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)  # ADX for trend strength
        #dataframe['pdi'] = ta.PLUS_DI(dataframe, timeperiod=14)  # Positive directional index
        #dataframe['mdi'] = ta.MINUS_DI(dataframe, timeperiod=14)  # Negative directional index
        #dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)  # 200-period SMA for trend direction
        #dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)  # RSI for momentum
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)


        #RSI_CROSS
        #dataframe['rsi_7'] = ta.RSI(dataframe, timeperiod=7)  # RSI for momentum

        #ichimoku
        #high = dataframe['high']; low = dataframe['low']; close = dataframe['close']
        #dataframe['tenkan_sen'] = (high.rolling(9).max() + low.rolling(9).min()) / 2
        #dataframe['kijun_sen'] = (high.rolling(26).max() + low.rolling(26).min()) / 2
        #dataframe['senkou_span_a'] = ((dataframe['tenkan_sen'] + dataframe['kijun_sen']) / 2).shift(26)
        #dataframe['senkou_span_b'] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        #dataframe['chikou_span'] = close.shift(-26)        

        return dataframe
        
    ################################################################################################################################
    ######################################## populate_entry_trend ##################################################################
    ################################################################################################################################

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # GENERAL ##################################################################################################################
        
        # BTC trend condition (1h sma 200)
        if self.use_informative_pairs.value:
            df['btc_trend'] = np.where(
                df['btc_usdt_close_1h'] > df['btc_usdt_sma200_1h'], 'UP', 'DOWN'
            )
        else:
            df['btc_trend'] = 'NA'  # No filter



        # FTP (ADX CROSS PDI/MDI AND RSI ######################################################################################################################


        if self.use_FTP_longs.value or self.use_FTP_shorts.value:
            
            #ADX CROSS DIR for FTP ENTRY

            # Single-column ADX cross direction (1 / -1 / 0) with N-candle lookback
            N = self.ncrossed.value*12  # how many extra candles after the cross. x12: tf 1h a 5m
            # 1) compute raw cross events (Series of 0/1)
            cross_up   = qtpylib.crossed_above(df['pdi_1h'], df['mdi_1h']).astype(int)
            cross_down = qtpylib.crossed_above(df['mdi_1h'], df['pdi_1h']).astype(int)
            # 2) rolling-max over window (cross candle + next N candles)
            recent_up   = cross_up.rolling(window=N+1, min_periods=1).max()
            recent_down = cross_down.rolling(window=N+1, min_periods=1).max()
            # 3) one-shot column:  1 if recent up-cross AND pdi>mdi, -1 if recent down-cross AND mdi>pdi, else 0
            df['adx_cross_dir'] = 0
            # only mark long if recent up-cross AND pdi>mdi now
            mask_up   = (recent_up   == 1) & (df['pdi_1h'] > df['mdi_1h'])
            # only mark short if recent down-cross AND mdi>pdi now, and no up condition
            mask_down = (recent_down == 1) & (df['mdi_1h'] > df['pdi_1h']) & (~mask_up)
            df.loc[mask_up,   'adx_cross_dir'] =  1
            df.loc[mask_down, 'adx_cross_dir'] = -1


        if self.use_FTP_longs.value:
            # Long entry: PDI crosses above MDI and ADX > 25 (strong trend), RSI > 30 (not oversold)
            df.loc[
                (
                    (df['btc_trend'].isin(['UP', 'NA'])) &               #Check BTC trend 1H
                    (df['adx_cross_dir'] == 1) &                        #uptrend signal supera downtrend signal
                    (df['adx_1h'] > 25) &                                  #strong trend
                    (df['rsi_1h'] > 30) &                                  #not oversold (30 and below is oversold)
                    (df['close_1h'] > df['sma200_1h'])                       #Check for uptrend using 200 SMA
                ),
                ['enter_long', 'enter_tag']
            ] = (1, 'FTP__Long')

        if self.use_FTP_shorts.value:
            # Short entry: MDI crosses above PDI and ADX > 25 (strong trend), RSI < 70 (not overbought)
            df.loc[
                (
                    (df['btc_trend'].isin(['DOWN', 'NA'])) &                         #Check BTC trend 1H
                    (df['adx_cross_dir'] == -1) &                                   #downtrend signal supera uptrend signal
                    (df['adx_1h'] > 25) &                                              #strong trend
                    (df['rsi_1h'] < 70) &                                              #not overbought (70 and above is overbought) -> todavia no empieza a caer
                    (df['close_1h'] < df['sma200_1h'])                                   #downtrend
                ),
                ['enter_short', 'enter_tag']
            ] = (1, 'FTP__Short')

    # ރICHIMOKU #####################################################################################################################
    # Ichimoku base conditions 
    # nota: estrategia muy certera. Evaluar aumentar stake o apalancamiento en caso de entrada. (no he visto que falle aún)

        #tenkan es la rapida
        ichimoku_long = (
            (df['close_1h'] > df['senkou_span_a_1h']) &                                     #por arriba de nube
            (df['senkou_span_a_1h'] > df['senkou_span_b_1h']) &                             #nube alcista
            qtpylib.crossed_above(df['tenkan_sen_1h'], df['kijun_sen_1h']).astype(int) &    #rapida cruza sobre lenta
            (df['mfi_1h'] < 35)                                                             #filtro adicional: mfi "barato"
        )

        ichimoku_short = (
            (df['close_1h'] < df['senkou_span_b_1h']) &                                           #por debajo de nube
            (df['senkou_span_b_1h'] < df['senkou_span_a_1h']) &                                   #nube bajista
            qtpylib.crossed_above(df['kijun_sen_1h'], df['tenkan_sen_1h']).astype(int) &            #lenta sobre rápida
            (df['mfi_1h'] > 65)                                                                   #filtro adicional: mfi "caro"

        )
       
        if self.use_ichimoku_longs.value:
            df.loc[
                (
                    (ichimoku_long)                                    #ichimoku filter
                ),
                ['enter_long', 'enter_tag']
            ] = (1, 'ރIchimoku Long')

        if self.use_ichimoku_shorts.value:
            df.loc[
                (
                    (ichimoku_short)                                       #ichimoku filter

                ),
                ['enter_short', 'enter_tag']
            ] = (1, 'ރIchimoku Short')

        return df
    

    ################################################################################################################################
    ######################################## populate_exit_trend ###################################################################
    ################################################################################################################################

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        
        
        # FTP ######################################################################################################################
        """
            - Exit Long: Triggered when 'adx' drops below 25 and RSI > 50 (indicating weakening uptrend).
            - Exit Short: Triggered when 'adx' drops below 25 and RSI < 50 (indicating weakening downtrend).
        """
        # Long exit: ADX < 25 and RSI > 50 (indicating trend weakness or reversal)

        
        if self.use_FTP_longs_exit.value:
            df.loc[
                (
                    (df['adx_1h'] < 25) &
                    (df['rsi_1h'] > 50)  # RSI confirms weakening upward momentum
                ),
                ['exit_long', 'exit_tag']
            ] = (1, 'FTP__ExitLong')

        if self.use_FTP_shorts_exit.value:
            # Short exit: ADX < 25 and RSI < 50 (indicating trend weakness or reversal)
            df.loc[
                (
                    (df['adx_1h'] < 25) &
                    (df['rsi_1h'] < 50)  # RSI confirms weakening downward momentum
                ),
                ['exit_short', 'exit_tag']
            ] = (1, 'FTP__ExitShort')
               
        return df
    

    ################################################################################################################################
    ######################################## CALCULATE WIN RATES (AUX) #############################################################
    ################################################################################################################################    

    def calculate_win_rate_per_pair(trades):
        """
        Calculate the win rate for each trading pair from a list of trades.
        
        Parameters:
            trades (list): List of trade dictionaries or DataFrame from Freqtrade containing trade data.
                        Each trade should have 'pair' and 'profit_ratio' or 'profit_abs' fields.
        
        Returns:
            dict: Dictionary with trading pairs as keys and win rates (as percentages) as values.
        """
        # Initialize dictionaries to store trade counts and wins
        pair_trade_counts = {}
        pair_win_counts = {}
        
        # Iterate through trades
        for trade in trades:
            pair = trade['pair']
            # Determine if the trade is a win (positive profit)
            is_win = trade.get('profit_ratio', 0) > 0 or trade.get('profit_abs', 0) > 0
            
            # Update trade count for the pair
            pair_trade_counts[pair] = pair_trade_counts.get(pair, 0) + 1
            # Update win count if the trade is a win
            if is_win:
                pair_win_counts[pair] = pair_win_counts.get(pair, 0) + 1
        
        # Calculate win rate for each pair
        win_rates = {}
        for pair in pair_trade_counts:
            total_trades = pair_trade_counts[pair]
            wins = pair_win_counts.get(pair, 0)
            # Win rate = (wins / total trades) * 100
            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
            win_rates[pair] = round(win_rate, 2)  # Round to 2 decimal places
        
        return win_rates


    ################################################################################################################################
    ######################################## PLOT CONFIGURATION ###################################################################
    ################################################################################################################################
    
    plot_config = {
        'main_plot': {

                'sma200_1h': {'color': 'white','plotly': {'opacity': 0.9}},

                #'tenkan_sen_1h': {'color': 'blue', 'plotly': {'opacity': 1.0, 'width': 1.5}},
                #'kijun_sen_1h': {'color': 'red', 'plotly': {'opacity': 1.0, 'width': 1.5}},
                #'senkou_span_a_1h': {'color': 'green', 'plotly': {'opacity': 0.0, 'width': 1.0, 'fill': 'none', 'dash': 'dash'}},
                #'senkou_span_b_1h': {'color': 'brown', 'plotly': {'opacity': 0.0, 'width': 1.0, 'fill': 'senkou_span_a_1h', 'dash': 'dash'}},
                #'chikou_span_1h': {'color': 'gray', 'plotly': {'opacity': 0.9, 'width': 1.0, 'dash': 'dot'}},

                #'tenkan_sen': {'color': 'blue', 'plotly': {'opacity': 1.0, 'width': 1.5}},
                #'kijun_sen': {'color': 'red', 'plotly': {'opacity': 1.0, 'width': 1.5}},
                #'senkou_span_a': {'color': 'green', 'plotly': {'opacity': 0.0, 'width': 1.0, 'fill': 'none', 'dash': 'dash'}},
                #'senkou_span_b': {'color': 'brown', 'plotly': {'opacity': 0.0, 'width': 1.0, 'fill': 'senkou_span_a_1h', 'dash': 'dash'}},
                #'chikou_span': {'color': 'gray', 'plotly': {'opacity': 0.9, 'width': 1.0, 'dash': 'dot'}},     

                #'tenkan_sen_1d': {'color': 'blue', 'plotly': {'opacity': 1.0, 'width': 1.5}},
                #'kijun_sen_1d': {'color': 'red', 'plotly': {'opacity': 1.0, 'width': 1.5}},
                #'senkou_span_a_1d': {'color': 'green', 'plotly': {'opacity': 0.0, 'width': 1.0, 'fill': 'none', 'dash': 'dash'}},
                #'senkou_span_b_1d': {'color': 'brown', 'plotly': {'opacity': 0.0, 'width': 1.0, 'fill': 'senkou_span_a_1h', 'dash': 'dash'}},
                #'chikou_span_1d': {'color': 'gray', 'plotly': {'opacity': 0.9, 'width': 1.0, 'dash': 'dot'}},     
                
                'BB_upper_1h': {'color': 'white','plotly': {'opacity': 0.9}},
                'BB_lower_1h': {'color': 'white','plotly': {'opacity': 0.9}},
                #'BB_upper_SL_1h': {'color': 'red','plotly': {'opacity': 0.9}},
                #'BB_lower_SL_1h': {'color': 'red','plotly': {'opacity': 0.9}},

                                       

        },
        'subplots': {
   
            
            "ADX": {
                'adx_1h': {'color': 'yellow'},
                'pdi_1h': {'color': 'green', 'fill_to': 'adx'},
                'mdi_1h': {'color': 'red', 'fill_to': 'adx'}
            },
            "ADX_CROSSDIR": {
                'adx_cross_dir': {'color': 'yellow'}
            },

            "RSI": {
                'rsi_1h': {'color': 'rgba(94, 43, 126, 0.87)'},
            },
            
            "BTC 1h": {
                'btc_usdt_close_1h': {'color': 'orange'},
                'btc_usdt_sma200_1h': {'color': 'white','plotly': {'opacity': 0.9}}
            },
        },
    }
