#ESTRATEGIA NS1 "POS REBOTE BOLLINGER" 
#base: Vela cierra por encima/debajo de banda de bollinger, o la "pincha" y luego las siguientes 1-3 invierten dirección (cierra por dentro en dir opuesta)
#take profit: banda central de bollinger (custom exit)
#filtros adicionales:
#   long / short habilitado dependiendo ultimas 3 velas por encima/por debajo btc sma200 1h 
#   long / short habilitado dependiendo btc sma10_1h por encima sma30_1h / por debajo
#   long si rsi<30, short si rsi>70 en las ultimas 1-3 velas 1h
#   mfi misma logica rsi


################################################################################################################################
############################ IMPORTS ###########################################################################################
################################################################################################################################

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
#from technical import qtpylib
from pandas import DataFrame
from datetime import datetime, timezone, timedelta
from typing import Optional
from functools import reduce
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, RealParameter, informative, merge_informative_pair)
#import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.exchange import  timeframe_to_minutes #,timeframe_to_prev_date

logger = logging.getLogger(__name__)

################################################################################################################################
############################## CLASS DEFINITION ################################################################################
################################################################################################################################

class NS2(IStrategy):
    leve = 10

    stakename = "USDT"              # FOR PRINTING

    ### Strategy parameters ###
    exit_profit_only = True
    ignore_roi_if_entry_signal = False
    can_short = True
    use_exit_signal = True
    startup_candle_count: int = 12*31     #1h sma indicators
    timeframe = '5m'

    min_roi_cte = 0.01*leve     #%*leve
    min_hours_cte = 100           #hs    

    # ROI table:
    rm=leve
    """minimal_roi = {
        "0": min_roi_cte,     #original=30% x1 (error?)
        "60":min_roi_cte/2,
        "120":min_roi_cte/3,
        "180":min_roi_cte/4,
        "240":min_roi_cte/5,
        "300": min_roi_cte/10
    }"""


####################################################################################################################################
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
        return True

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

        #logger.info(f"Trade age: {trade_age}, Profit ratio: {profit_ratio:.5f}, Exit reason: {exit_reason}")
        
        # Allow force exit
        if exit_reason == "force_exit":
            return True
        
        if exit_reason == "stop_loss":
            return True

        if exit_reason == "trailing_stop_loss":
            return True
        
        if exit_reason == 'EL BB MIDDLE' or exit_reason == 'ES BB MIDDLE':
            return True
        
        if exit_reason == 'ExL(C): TREND REVERSAL' or exit_reason == 'ExS(C): TREND REVERSAL':
            return True        

        #exit conditions
        trade_age = current_time - trade.open_date_utc
        profit_ratio = trade.calc_profit_ratio(rate)
        exit_condition1 = (trade_age > timedelta(hours=self.min_hours_cte)) and (profit_ratio >= self.min_roi_cte/2)
        exit_condition2 = (profit_ratio >= self.min_roi_cte)
        exit_conditions = exit_condition1 or exit_condition2

        # Allow exit if small profit and trade is older than 4 hours
        if exit_conditions:
            return True

        # Otherwise, block the exit
        return False
    
################################################################################################################################
######################################## CUSTOM STOPLOSS LOGIC #################################################################
################################################################################################################################
    # Stoploss
    #rrratio = 4                 #risk reward ratio base
    stoploss = -0.9            #-1*min_roi_cte*rrratio

    trailing_stop = False
    #trailing_stop_positive_offset = 0.09*leve
    #trailing_stop_positive = 0.089*leve
    #trailing_only_offset_is_reached = True

    use_custom_stoploss = False
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, **kwargs) -> float | None:
        """
        :param pair: Pair that's currently analyzed
        :param trade: Trade to calculate stoploss for
        :param current_time: Current timestamp
        :param current_rate: The current price rate
        :return: Stoploss ratio (fraction of current_rate) or None to keep existing stoploss
        """
        trade_age = current_time - trade.open_date_utc

        # Retrieve dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        last_candle = dataframe.iloc[-1]
        
        # CSL para BOLL LARGO: BANDA LARGA + 1 ATR
        if True:#entry_tag == 'LONG BOLL LARGO' or  entry_tag == 'SHORT BOLL LARGO':            #ver otra forma de sacar entry tag, tal vez en trade object

            bb_upper_limit = last_candle.get('BB_upper_largo_sl_4h', np.nan)
            bb_lower_limit = last_candle.get('BB_lower_largo_sl_4h', np.nan)
            if np.isnan(bb_lower_limit) or np.isnan(bb_upper_limit):
                return None
            
        # Choose raw SL price: for short use upper bounds, for long use lower bounds
        if trade.is_short:
            # Tighter SL is the lower price
            raw_bb = bb_upper_limit
            #raw_dc = last_candle.get('dc_high', np.nan) 
            raw_sl_price = raw_bb #if (trade_age < timedelta(minutes=30)) else min(raw_bb, raw_dc)

        else:
            # Tighter SL is the higher price
            raw_bb = bb_lower_limit
            #raw_dc = last_candle.get('dc_low', np.nan)  
            raw_sl_price = raw_bb #if (trade_age < timedelta(minutes=30)) else max(raw_bb, raw_dc)

            # Enforce direction
            #if trade.is_short and raw_sl_price < current_rate:
            #    raw_sl_price = current_rate

            #if not trade.is_short and raw_sl_price > current_rate:
            #    raw_sl_price = current_rate

        # Compute relative stoploss ratio
        sl_ratio = self.leve * abs((trade.open_rate - raw_sl_price) / trade.open_rate) if trade_age > timedelta(minutes=5) else self.stoploss     #solo si trade age > x
        
        
        """# Bollinger Bands + ATR
        if self.use_BB_csl.value:
            bb_upper_limit = last_candle.get('BB_upper_sl_4h', np.nan)
            bb_lower_limit = last_candle.get('BB_lower_sl_4h', np.nan)
            if np.isnan(bb_lower_limit) or np.isnan(bb_upper_limit):
                return None
            
        # Choose raw SL price: for short use upper bounds, for long use lower bounds
        if trade.is_short:
            # Tighter SL is the lower price
            raw_bb = bb_upper_limit
            #raw_dc = last_candle.get('dc_high', np.nan) 
            raw_sl_price = raw_bb #if (trade_age < timedelta(minutes=30)) else min(raw_bb, raw_dc)

        else:
            # Tighter SL is the higher price
            raw_bb = bb_lower_limit
            #raw_dc = last_candle.get('dc_low', np.nan)  
            raw_sl_price = raw_bb #if (trade_age < timedelta(minutes=30)) else max(raw_bb, raw_dc)

            # Enforce direction
            #if trade.is_short and raw_sl_price < current_rate:
            #    raw_sl_price = current_rate

            #if not trade.is_short and raw_sl_price > current_rate:
            #    raw_sl_price = current_rate

        # Compute relative stoploss ratio
        sl_ratio = self.leve * abs((trade.open_rate - raw_sl_price) / trade.open_rate)"""

        return sl_ratio

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
        """if last_time is None or (current_time - last_time).total_seconds() >= 10 * 3600:        #10 hs
            tradedir = 'SHORT' if trade.is_short else 'LONG'
            msg_atrSL = (
                f"🛑 CSL: {trade.pair} Current Rate: {current_rate:.2f}"
                f" SL: {raw_sl_price:.2f} ({100*sl_ratio:.2f}%) ({tradedir}) ({current_time})"
            )
            self.dp.send_msg(msg_atrSL)
            logger.info(msg_atrSL)

            # Store ISO string of log time
            trade.set_custom_data(key='last_sl_log_time', value=current_time.isoformat())"""

            # Return as positive fraction (<1). Freqtrade uses absolute value, sign ignored.
        return sl_ratio
    
################################################################################################################################
################################ CUSTOM STAKE AMOUNT / RESIZE ##################################################################
################################################################################################################################
#     
    max_dca_multiplier = 1.0
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        """
            Calculates the stake amount to use for a trade, adjusted dynamically.
            - If the adjusted stake is lower than the allowed minimum (`min_stake`), it is automatically increased
              to meet the minimum stake requirement.
        """
        # Calculates the adjusted stake amount based on the DCA multiplier.
        adjusted_stake = proposed_stake / self.max_dca_multiplier

        # Automatically adjusts to the minimum stake if it is too low.
        if adjusted_stake < min_stake:
            adjusted_stake = min_stake

        return adjusted_stake
    
################################################################################################################################
################################ DCA / ADJUST POSITION #########################################################################
################################################################################################################################
    position_adjustment_enable = True
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
        # Obtain pair dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        #filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        count_of_exits = trade.nr_of_successful_exits
        max_order_id = max((o.id for o in trade.orders), default=0)
        orders = trade.orders
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Track which DCA step we've executed: 0=none, 1=middle band done, 2=upper band done
        step_done = trade.get_custom_data(key='step', default=0)
        #logger.info(f"{step_done} {count_of_exits} {orders}")
        try:
            # SALIDA 50 EN BANDA TAKE PROFIT "1ER"
            if (
                (current_rate > last_candle['BB_middle_largo_TPL_4h'] and not trade.is_short)           #LONGS
                or (current_rate < last_candle['BB_middle_largo_TPS_4h'] and trade.is_short)            #SHORTS
            ) and (step_done == 0 and count_of_exits == 0 and max_order_id >= 1 ):                                                                #1er ENTRADA
                
                prev_stake = trade.stake_amount
                reduction = -prev_stake/2

                avail_funds = self.wallets.get_free(self.stakename)
                total_funds = self.wallets.get_total(self.stakename)
            
                # --- LOGGING GUARD: only log once per entry count ---
                last_logged = trade.get_custom_data(key='last_dca_log_count', default=None)
                if last_logged is None or last_logged != count_of_entries:
                    msg = (f"🟢 MBR: {current_time} {trade.pair} Profit={100*current_profit:.2f}% - reducing {reduction:.2f}{self.stakename} - total orders {count_of_entries}"
                        f" Av.: {avail_funds:.2f}/{total_funds:.2f} {self.stakename}")              
                    logger.info(msg)
                    self.dp.send_msg(msg)
                    # Record that we logged for this entry count
                    trade.set_custom_data(key='last_dca_log_count', value=count_of_entries)
                # --- end logging guard ---

                trade.set_custom_data(key='step', value=1)

                return reduction, "MIDDLE BAND PROFIT"
            
            """            # SALIDA 50 EN 2NDA BANDA TAKE PROFIT "2NDA"
            if (
                (current_rate > last_candle['BB_upper_largo_4h'] and not trade.is_short)           #LONGS
                or (current_rate < last_candle['BB_lower_largo_4h'] and trade.is_short)            #SHORTS
            ) and (step_done == 1 and max_order_id >= 2):                                                           #2er ENTRADA
                
                prev_stake = trade.stake_amount
                reduction = -trade.stake_amount

                avail_funds = self.wallets.get_free(self.stakename)
                total_funds = self.wallets.get_total(self.stakename)
            
                # --- LOGGING GUARD: only log once per entry count ---
                last_logged = trade.get_custom_data(key='last_dca_log_count', default=None)
                if last_logged is None or last_logged != count_of_entries:
                    msg = (f"🟢🟢OBR: {current_time} {trade.pair} Profit={100*current_profit:.2f}% - total exit {reduction:.2f}{self.stakename} - total orders {count_of_entries}"
                        f" Av.: {avail_funds:.2f}/{total_funds:.2f} {self.stakename}")              
                    logger.info(msg)
                    self.dp.send_msg(msg)
                    # Record that we logged for this entry count
                    trade.set_custom_data(key='last_dca_log_count', value=count_of_entries)
                # --- end logging guard ---

                trade.set_custom_data(key='step', value=2)

                return reduction, "OTHER BAND PROFIT"""

        except Exception as exception:
            logger.error(f"Error in DCA module {trade.pair}: {exception}")
            return None

            """# If already more than first entry, increase per your dca_increase
            if count_of_entries > 1 and count_of_entries < 3:
                stake_amount = stake_amount * (1 + (count_of_entries - 1) * self.dca_increase)  # dca_increase % increase per additional entry
            
            #SUPERDCA 1
            if count_of_entries > 3 and count_of_entries < 4:
                stake_amount = trade.stake_amount

            #SUPERDCA 2
            if count_of_entries > 4 :
                stake_amount = trade.stake_amount*2

            additional = stake_amount - prev_stake
            avail_funds = self.wallets.get_free(self.stakename)
            total_funds = self.wallets.get_total(self.stakename)

            if avail_funds < additional:
                stake_amount = prev_stake + avail_funds

                # --- LOGGING GUARD: only log once per entry count ---
                entry_count = trade.nr_of_successful_entries
                last_logged = trade.get_custom_data(key='last_dca_log_count', default=None)
                if last_logged is None or last_logged != entry_count:
                    msg = (f"🟠 DCA: {current_time} Increasing {trade.pair} - new total {stake_amount:.2f}{self.stakename} - total dca entries {count_of_entries}"
                        f" Av.: {avail_funds:.2f}/{total_funds:.2f} {self.stakename} (INCOMPLETE DCA)")              
                    logger.info(msg)
                    self.dp.send_msg(msg)
                    # Record that we logged for this entry count
                    trade.set_custom_data(key='last_dca_log_count', value=entry_count)
                # --- end logging guard ---

                return stake_amount, "DCA PARTIAL"


            # --- LOGGING GUARD: only log once per entry count ---
            entry_count = trade.nr_of_successful_entries
            last_logged = trade.get_custom_data(key='last_dca_log_count', default=None)
            if last_logged is None or last_logged != entry_count:
                msg = (f"🟡 DCA: {current_time} Increasing {trade.pair} new total {stake_amount:.2f}{self.stakename} - total dca entries {count_of_entries}"
                    f" Av.: {avail_funds:.2f}/{total_funds:.2f} {self.stakename}")                
                logger.info(msg)
                self.dp.send_msg(msg)
                # Record that we logged for this entry count
                trade.set_custom_data(key='last_dca_log_count', value=entry_count)
            # --- end logging guard ---

            return stake_amount, "DCA INCREASING"
        
        except Exception as exception:
            logger.error(f"Error adjusting DCA position for the pair {trade.pair}: {exception}")
            return None"""


################################################################################################################################
################################ HYPEROPTABLES #################################################################################
################################################################################################################################
    # BUY
    optimize_bb_largo = True
    BB_largo_nstd = DecimalParameter(1.5, 3.0, decimals=1, default=2.9, space="buy", optimize=optimize_bb_largo)

    
    # SELL #############################################################################################

    # BB LARGO TP
    BB_largo_middle_tp_offset = DecimalParameter(0.001, 0.1, decimals=3, default=2.0, space="sell", optimize=optimize_bb_largo)                #offset de atr para take profit en middle_largo

    # BB LARGO SL
    atr_sl_mult_largo = DecimalParameter(0.1, 3.0, decimals=2, default=0.85, space="sell", optimize=optimize_bb_largo)                                 #atr sl offset for bbands stoploss

    # CSL
    optimize_csl = False
    use_BB_csl = BooleanParameter(default=True, space="sell", optimize=optimize_csl)                                                        #use or not bbands sl
    BB_sl_coeff = DecimalParameter(0.001, 0.20, decimals=3, default=0.001, space="sell", optimize=optimize_csl)                             #constant offset for bbands sl
    atr_sl_mult = DecimalParameter(0.1, 3.0, decimals=1, default=0.75, space="sell", optimize=optimize_csl)                                 #atr sl offset for bbands sl
    where_BB_csl = CategoricalParameter(['custom_stoploss', 'custom_exit'], default='custom_stoploss', space="sell", optimize=optimize_csl) #Not implemented
    dc_length = IntParameter(12, 48, default=36, space="sell", optimize=optimize_csl)                                                       #Donchian stoploss lenght in self.timeframe candles
    #atr_sl_mult = 1
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

          
        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": 48 
        })
        if False: #if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,
                "trade_limit": 1,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": True
            })

        return prot
    
        
################################################################################################################################
######################################## CUSTOM EXIT LOGIC #####################################################################
################################################################################################################################
    timeframe_minutes = timeframe_to_minutes(timeframe)
    use_custom_exit = True

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
        
        # STOPLOSS #############################################################################################################
        
        """ Simula trailing stop loss en las bandas de stop loss"""

        if (
            (not trade.is_short and current_rate < last_candle['BB_lower_largo_sl_4h'])     #longs
            or (trade.is_short and current_rate > last_candle['BB_upper_largo_sl_4h'])      #shorts
        ):
            
            sl_rate = trade.get_custom_data(key='sl_rate', default=None)                    #obtener valor de sl_rate

            if (
                (sl_rate is None)                                                              #primer valor
                or (not trade.is_short and last_candle['BB_lower_largo_sl_4h']> sl_rate)       #aumenta exigencia longs
            ):
                trade.set_custom_data(key='sl_rate', value=last_candle['BB_lower_largo_sl_4h']) #actualizar valor

            if (
                (sl_rate is None)                                                              #primer valor
                or (trade.is_short and last_candle['BB_upper_largo_sl_4h']< sl_rate)           #aumenta exigencia shorts
            ):
                trade.set_custom_data(key='sl_rate', value=last_candle['BB_upper_largo_sl_4h']) #actualizar valor

            if sl_rate is not None and (
                (not trade.is_short and current_rate < sl_rate)
                or (trade.is_short and current_rate > sl_rate)
            ):
                msg_Exit = f"🟥 CSL: {current_time} {trade.pair} STOPLOSS! Loss={100*current_profit:.2f}% ({current_profit*trade.stake_amount:.2f} {self.stakename}). Total stake was {trade.stake_amount:.2f} {self.stakename}"
                self.dp.send_msg(msg_Exit)
                logger.info(msg_Exit)

                return "C. STOP LOSS"
        #############################################################################################################
        # EXIT EN SEGUNDA BANDA VERDE ####################################################################################
        """ Take Profit: Salida en 2nda Banda Middle Offseteada por multiplo de atr (4h) """
        if (
            (current_rate > last_candle['BB_middle_largo_TPS_4h'] and (not trade.is_short))
            or (current_rate < last_candle['BB_middle_largo_TPL_4h'] and (trade.is_short))
        ):
            msg_Exit = f"❇️ 2BM: {current_time} {trade.pair} exit!! Profit {100*current_profit:.2f}% ({current_profit*trade.stake_amount:.2f} {self.stakename}). Total stake was {trade.stake_amount:.2f} {self.stakename}"
            self.dp.send_msg(msg_Exit)
            logger.info(msg_Exit)

            return "(C)EXIT 2NDA BANDA"

        #############################################################################################################
        # EXIT INDECISO ####################################################################################
        """ Take Profit: Salida en 1er Banda Middle Offseteada por multiplo de atr (4h) despues de tocar middle y volver """
        indeciso = trade.get_custom_data(key='indeciso', default=False)                    #obtener valor de sl_rate
        
        # marcar como indeciso
        if (
            (not trade.is_short and current_rate > last_candle['BB_middle_largo_4h'])           #long sube mas que middle
            or (trade.is_short and current_rate < last_candle['BB_middle_largo_4h'])            #short baja mas que middle
        ):
            trade.set_custom_data(key='indeciso', value=True) #actualizar valor

        # exit en middle a los indecisos
        if (
            (indeciso)
            and ((not trade.is_short and current_rate < last_candle['BB_middle_largo_TPL_4h'])      #long cae debajo de primera banda
                 or (trade.is_short and current_rate > last_candle['BB_middle_largo_TPS_4h']))      #short sube soibre primera banda
        ):

            msg_Exit = f"🪢​ IND: {current_time} {trade.pair} exit!! Profit {100*current_profit:.2f}% ({current_profit*trade.stake_amount:.2f} {self.stakename}). Total stake was {trade.stake_amount:.2f} {self.stakename}"
            self.dp.send_msg(msg_Exit)
            logger.info(msg_Exit)

            return "(C)EXIT INDECISO"
        
        return False
            

        """#exit conditions / confirm_trade_exit
        trade_age = current_time - trade.open_date_utc
        profit_ratio = trade.calc_profit_ratio(current_rate)        

        exit_condition1 = (trade_age > timedelta(hours=self.min_hours_cte)) & (profit_ratio >= self.min_roi_cte/2)
        exit_condition2 = (profit_ratio >= self.min_roi_cte)
        exit_conditions = exit_condition1 or exit_condition2"""



##########################################################################
        """#❇️ROI CUSTOM EXIT
          
        if exit_conditions:

            msg_Exit = f"❇️ ROI: {current_time} {trade.pair} exit!! Profit {100*current_profit:.2f}% ({current_profit*trade.stake_amount:.2f} {self.stakename}). Total stake was {trade.stake_amount:.2f} {self.stakename}"
            self.dp.send_msg(msg_Exit)
            logger.info(msg_Exit)

            return ("Ex(c): ROI")"""
##########################################################################
    """#🧣TREND REVERSAL CUSTOM EXIT
        TRL_conditions = not trade.is_short and last_candle['sma10'] > last_candle['sma30'] and current_profit > self.min_roi_cte/10
        TRS_conditions = trade.is_short and last_candle['sma10'] < last_candle['sma30'] and current_profit > self.min_roi_cte/10

        # Long trend reversal exit
        if TRL_conditions:      #exit long trend reversal
            msg_Exit = f"🧣 TRL: {current_time} {trade.pair} exit!! Profit {100*current_profit:.2f}% ({current_profit*trade.stake_amount:.2f} {self.stakename}). Total stake was {trade.stake_amount:.2f} {self.stakename}"
            self.dp.send_msg(msg_Exit)
            logger.info(msg_Exit)

            return 'ExL(C): TREND REVERSAL'
        
        # Short trend reversal exit
        if TRS_conditions:     #exit short trend reversal
            msg_Exit = f"🧣 TRS: {current_time} {trade.pair} exit!! Profit {100*current_profit:.2f}% ({current_profit*trade.stake_amount:.2f} {self.stakename}). Total stake was {trade.stake_amount:.2f} {self.stakename}"
            self.dp.send_msg(msg_Exit)
            logger.info(msg_Exit)

            return 'ExS(C): TREND REVERSAL'"""

################################################################################################################################
######################################## INFORMATIVE PAIRS #####################################################################
################################################################################################################################

    #INFORMATIVE PAIRS

    # Define BTC/STAKE informative pair. Available in populate_indicators and other methods as
    # 'btc_rsi_1h'. Current stake currency should be specified as {stake} format variable 
    # instead of hard-coding actual stake currency. Available in populate_indicators and other 
    # methods as 'btc_usdt_sma200_1h' (when stake currency is USDT).
    """@informative('1h', 'BTC/{stake}:{stake}')
    def populate_indicators_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        #BTC 1h SMA200
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        return dataframe
    
    # Define informative upper timeframe for each pair. Decorators can be stacked on same 
    # method. Available in populate_indicators as 'sma200_1h'"""


    """@informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
       
        #Bollinger
        #sma = dataframe['close'].rolling(20).mean()
        #std_dev = dataframe['close'].rolling(20).std()
        #dataframe['BB_middle'] = sma
        #dataframe['BB_upper'] = sma + std_dev * 2.0
        #dataframe['BB_lower'] = sma - std_dev * 2.0     

        # SMAs 
        #dataframe['sma10'] = ta.SMA(dataframe, timeperiod=10)
        #dataframe['sma30'] = ta.SMA(dataframe, timeperiod=30)
        #dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)            

        # ADX
        #dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)  # ADX for trend strength
        #dataframe['pdi'] = ta.PLUS_DI(dataframe, timeperiod=14)  # Positive directional index
        #dataframe['mdi'] = ta.MINUS_DI(dataframe, timeperiod=14)  # Negative directional index

        # RSI
        #dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)  # RSI for momentum

        # MFI
        #dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)

        # ATR
        #dataframe['atr'] = ta.ATR(dataframe, timeperiod=12)     #12: 1-hour on 5m tf

        # BB SL LEVELS
        #dataframe['BB_upper_sl'] = (dataframe['BB_upper'] + dataframe['atr'] * self.atr_sl_mult.value)
        #dataframe['BB_lower_sl'] = (dataframe['BB_lower'] - dataframe['atr'] * self.atr_sl_mult.value)
        #dataframe['BB_upper_sl'] = ((sma + std_dev * 2.0) * (1 + self.BB_sl_coeff.value))
        #dataframe['BB_lower_sl'] = ((sma - std_dev * 2.0) * (1 - self.BB_sl_coeff.value))

        return dataframe"""

    @informative('4h')
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=6)     #6: 1 day on 4h-tf     

        #Bollinger
        sma = dataframe['close'].rolling(20).mean()
        std_dev = dataframe['close'].rolling(20).std()
        dataframe['BB_middle'] = sma
        dataframe['BB_upper'] = sma + std_dev * 2.0
        dataframe['BB_lower'] = sma - std_dev * 2.0     
        # BB SL LEVELS
        dataframe['BB_upper_sl'] = (dataframe['BB_upper'] + dataframe['atr'] * self.atr_sl_mult.value)
        dataframe['BB_lower_sl'] = (dataframe['BB_lower'] - dataframe['atr'] * self.atr_sl_mult.value)
        
        # BOLLINGER LARGA ##########################################################################################
        sma = dataframe['close'].rolling(200).mean()
        std_dev = dataframe['close'].rolling(200).std()
        dataframe['BB_middle_largo'] = sma
        dataframe['BB_upper_largo'] = sma + std_dev * self.BB_largo_nstd.value # 2.9
        dataframe['BB_lower_largo'] = sma - std_dev * self.BB_largo_nstd.value # 

        dataframe['BB_middle_largo_TPL'] = (dataframe['BB_middle_largo'] - dataframe['atr']* self.BB_largo_middle_tp_offset.value)         # VALORES PARA BANDA OFFSETEADA TAKE PROFITS
        dataframe['BB_middle_largo_TPS'] = (dataframe['BB_middle_largo'] + dataframe['atr']* self.BB_largo_middle_tp_offset.value)        # 

        dataframe['BB_upper_largo_sl'] = (dataframe['BB_upper_largo'] + dataframe['atr'] * self.atr_sl_mult_largo.value)
        dataframe['BB_lower_largo_sl'] = (dataframe['BB_lower_largo'] - dataframe['atr'] * self.atr_sl_mult_largo.value)

        dataframe['BB_ancho_larga'] = (dataframe['BB_upper_largo'] - dataframe['BB_lower_largo'])                                               #ancho, para saber si estamos en squeeze
        dataframe['BB_ancho_larga_sma200'] = ta.SMA(dataframe['BB_upper_largo'] - dataframe['BB_lower_largo'], timeperiod=200)                  #media del ancho, para saber si estamos en squeeze
        #############################################################################################################
        # CANDLE PATTERNS

        # BULLISH

        # CDLMORNINGDOJISTAR # (+100) Bullish: Downtrend + doji + strong up (High)
        dataframe["morning_doji_star"] = ta.CDLMORNINGDOJISTAR(dataframe)
        # CDLMORNINGSTAR # (+100) Bullish: Downtrend + small + strong up (High)
        dataframe["morning_star"] = ta.CDLMORNINGSTAR(dataframe)
        # CDLMATHOLD # (+100) Bullish: Uptrend pause & resume (High)
        dataframe["mat_hold"] = ta.CDLMATHOLD(dataframe)
        # CDL3WHITESOLDIERS # (+100) Bullish: Three long white candles (High)
        dataframe["three_white_soldiers"] = ta.CDL3WHITESOLDIERS(dataframe)

        # CDLCONCEALBABYSWALL # (+100) Bullish: Four black candles trap & reverse (Medium)
        dataframe["concealing_baby_swallow"] = ta.CDLCONCEALBABYSWALL(dataframe)
        # CDLHAMMER # (+100) Bullish: Small body with long lower shadow (Medium)
        dataframe["hammer"] = ta.CDLHAMMER(dataframe)
        # CDLDRAGONFLYDOJI # (+100) Bullish: Doji with long lower shadow (Medium)
        dataframe["dragonfly_doji"] = ta.CDLDRAGONFLYDOJI(dataframe)
        # CDLTAKURI # (+100) Bullish: Long lower shadow (Medium)
        dataframe["takuri"] = ta.CDLTAKURI(dataframe)
        # CDL3STARSINSOUTH # (+100) Bullish: Three small candles in downtrend (Low)
        dataframe["three_stars_in_south"] = ta.CDL3STARSINSOUTH(dataframe)
        # CDLHOMINGPIGEON # (+100) Bullish: Two black candles inside downtrend (Medium)
        dataframe["homing_pigeon"] = ta.CDLHOMINGPIGEON(dataframe)
        # CDLUNIQUE3RIVER # (+100) Bullish: Black + small + hammer-like (Medium)
        dataframe["unique_three_river"] = ta.CDLUNIQUE3RIVER(dataframe)
        # CDLSTICKSANDWICH # (+100) Bullish: Black, white, black with same close (Medium)
        dataframe["stick_sandwich"] = ta.CDLSTICKSANDWICH(dataframe)
        # CDLLADDERBOTTOM # (+100) Bullish: Series of black & small then white (Medium)
        dataframe["ladder_bottom"] = ta.CDLLADDERBOTTOM(dataframe)
        # CDLMATCHINGLOW # (+100) Bullish: Two black candles closing at same low (Medium)
        dataframe["matching_low"] = ta.CDLMATCHINGLOW(dataframe)
        # CDLINVERTEDHAMMER # (+100) Bullish: Small body, long upper shadow at bottom (Medium)
        dataframe["inverted_hammer"] = ta.CDLINVERTEDHAMMER(dataframe)


        # BEARISH

        # CDL3BLACKCROWS # (-100) Bearish: Three long black candles (High)
        dataframe["three_black_crows"] = ta.CDL3BLACKCROWS(dataframe)
        # CDLEVENINGDOJISTAR # (-100) Bearish: Uptrend + doji + down (High)
        dataframe["evening_doji_star"] = ta.CDLEVENINGDOJISTAR(dataframe)
        # CDLEVENINGSTAR # (-100) Bearish: Uptrend + small + down (High)
        dataframe["evening_star"] = ta.CDLEVENINGSTAR(dataframe)
        # CDLIDENTICAL3CROWS # (-100) Bearish: Three black with same close (High)
        dataframe["identical_three_crows"] = ta.CDLIDENTICAL3CROWS(dataframe)

        # CDLDARKCLOUDCOVER # (-100) Bearish: Black opens above & closes below midpoint (High)
        dataframe["dark_cloud_cover"] = ta.CDLDARKCLOUDCOVER(dataframe)
        # CDLHANGINGMAN # (-100) Bearish: Small body with long lower shadow at top (Medium)
        dataframe["hanging_man"] = ta.CDLHANGINGMAN(dataframe)
        # CDLADVANCEBLOCK # (-100) Bearish: Three white candles weakening (Medium)
        dataframe["advance_block"] = ta.CDLADVANCEBLOCK(dataframe)
        # CDLSHOOTINGSTAR # (-100) Bearish: Small body, long upper shadow at top (Medium)
        dataframe["shooting_star"] = ta.CDLSHOOTINGSTAR(dataframe)
        # CDLGRAVESTONEDOJI # (-100) Bearish: Doji with long upper shadow (Medium)
        dataframe["gravestone_doji"] = ta.CDLGRAVESTONEDOJI(dataframe)
        # CDLSTALLEDPATTERN # (-100) Bearish: Three white weakening at top (Medium)
        dataframe["stalled_pattern"] = ta.CDLSTALLEDPATTERN(dataframe)
        # CDLTHRUSTING # (-100) Bearish: White opens high but closes below midpoint (Medium)
        dataframe["thrusting"] = ta.CDLTHRUSTING(dataframe)
        # CDLUPSIDEGAP2CROWS # (-100) Bearish: Uptrend with gap + two black (Medium)
        dataframe["upside_gap_two_crows"] = ta.CDLUPSIDEGAP2CROWS(dataframe)
        # CDLONNECK # (-100) Bearish: Similar to In-Neck but closes at low (Low)
        dataframe["on_neck"] = ta.CDLONNECK(dataframe)
        # CDLINNECK # (-100) Bearish: Black & white near prior low (Low)
        dataframe["in_neck"] = ta.CDLINNECK(dataframe)
        # CDL2CROWS # (-100) Bearish: Two black candles over white (Medium)
        dataframe["two_crows"] = ta.CDL2CROWS(dataframe)


        # BOTH

        # CDLENGULFING # (±100) Both: Candle engulfs prior (High)
        dataframe["engulfing"] = ta.CDLENGULFING(dataframe)
        # CDLRISEFALL3METHODS # (±100) Both: Trend pause & continuation (High)
        dataframe["rise_fall_three_methods"] = ta.CDLRISEFALL3METHODS(dataframe)
        # CDLSEPARATINGLINES # (±100) Both: Continuation after opposite candle (Medium)
        dataframe["separating_lines"] = ta.CDLSEPARATINGLINES(dataframe)
        # CDL3INSIDE # (±100) Both: Harami + confirmation (Medium)
        dataframe["three_inside"] = ta.CDL3INSIDE(dataframe)
        # CDLABANDONEDBABY # (±100) Both: Doji gap between two candles (High)
        dataframe["abandoned_baby"] = ta.CDLABANDONEDBABY(dataframe)
        # CDLBREAKAWAY # (±100) Both: Series showing reversal (Medium)
        dataframe["breakaway"] = ta.CDLBREAKAWAY(dataframe)
        # CDLBELTHOLD # (±100) Both: Large single reversal candle (Medium)
        dataframe["belt_hold"] = ta.CDLBELTHOLD(dataframe)
        # CDLCLOSINGMARUBOZU # (±100) Both: Strong close at extreme (Medium)
        dataframe["closing_marubozu"] = ta.CDLCLOSINGMARUBOZU(dataframe)
        # CDLCOUNTERATTACK # (±100) Both: Opposite candle closes at prior close (Medium)
        dataframe["counterattack"] = ta.CDLCOUNTERATTACK(dataframe)
        # CDLDOJI # (±100) Both: Open ≈ Close, indecision (Low)
        dataframe["doji"] = ta.CDLDOJI(dataframe)
        # CDLDOJISTAR # (±100) Both: Doji after long candle (Medium)
        dataframe["doji_star"] = ta.CDLDOJISTAR(dataframe)
        # CDLGAPSIDESIDEWHITE # (±100) Both: Two white candles straddle a gap (Low)
        dataframe["gap_side_side_white"] = ta.CDLGAPSIDESIDEWHITE(dataframe)
        # CDLHARAMI # (±100) Both: Small body within prior (Medium)
        dataframe["harami"] = ta.CDLHARAMI(dataframe)
        # CDLHARAMICROSS # (±100) Both: Same as harami with doji (Medium)
        dataframe["harami_cross"] = ta.CDLHARAMICROSS(dataframe)
        # CDLHIKKAKE # (±100) Both: Inside day breakout & trap (Medium)
        dataframe["hikkake"] = ta.CDLHIKKAKE(dataframe)
        # CDLHIKKAKEMOD # (±100) Both: Modified Hikkake (Medium)
        dataframe["hikkake_mod"] = ta.CDLHIKKAKEMOD(dataframe)
        # CDLKICKING # (±100) Both: Opposite marubozu gap (High)
        dataframe["kicking"] = ta.CDLKICKING(dataframe)
        # CDLKICKINGBYLENGTH # (±100) Both: Longer marubozu dictates (High)
        dataframe["kicking_by_length"] = ta.CDLKICKINGBYLENGTH(dataframe)
        # CDLLONGLINE # (±100) Both: Long body candle (Medium)
        dataframe["long_line"] = ta.CDLLONGLINE(dataframe)
        # CDLLONGLEGGEDDOJI # (±100) Both: Doji with long shadows (Low)
        dataframe["long_legged_doji"] = ta.CDLLONGLEGGEDDOJI(dataframe)
        # CDLMARUBOZU # (±100) Both: Long body no wicks (Medium)
        dataframe["marubozu"] = ta.CDLMARUBOZU(dataframe)
        # CDLRICKSHAWMAN # (±100) Both: Doji with equal shadows (Low)
        dataframe["rickshaw_man"] = ta.CDLRICKSHAWMAN(dataframe)
        # CDLSHORTLINE # (±100) Both: Short body candle (Low)
        dataframe["short_line"] = ta.CDLSHORTLINE(dataframe)
        # CDLSPINNINGTOP # (±100) Both: Small body, indecision (Low)
        dataframe["spinning_top"] = ta.CDLSPINNINGTOP(dataframe)
        # CDLTASUKIGAP # (±100) Both: Gap then opposite candle (Medium)
        dataframe["tasuki_gap"] = ta.CDLTASUKIGAP(dataframe)
        # CDL3LINESTRIKE # (±100) Both: Three in trend + opposite candle (Low)
        dataframe["three_line_strike"] = ta.CDL3LINESTRIKE(dataframe)
        # CDLXSIDEGAP3METHODS # (±100) Both: Gaps and small candles in trend (Medium)
        dataframe["xside_gap_three_methods"] = ta.CDLXSIDEGAP3METHODS(dataframe)
        # CDLTRISTAR # (±100) Both: Three dojis signal reversal (Medium)
        dataframe["tristar"] = ta.CDLTRISTAR(dataframe)

        # SMAs 
        dataframe['sma10'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['sma30'] = ta.SMA(dataframe, timeperiod=30)      
        
        return dataframe
    

    ################################################################################################################################
    ######################################## populate_indicators ###################################################################
    ################################################################################################################################

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
            Calculates technical indicators used to define entry and exit signals.
        """
        #Bollinger
        #sma = dataframe['close'].rolling(20).mean()
        #std_dev = dataframe['close'].rolling(20).std()
        #dataframe['BB_middle'] = sma
        #dataframe['BB_upper'] = sma + std_dev * 2.0
        #dataframe['BB_lower'] = sma - std_dev * 2.0     

        # SMAs 
        #dataframe['sma10'] = ta.SMA(dataframe, timeperiod=10)
        #dataframe['sma30'] = ta.SMA(dataframe, timeperiod=30)      

        # ADX
        #dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)  # ADX for trend strength
        #dataframe['pdi'] = ta.PLUS_DI(dataframe, timeperiod=14)  # Positive directional index
        #dataframe['mdi'] = ta.MINUS_DI(dataframe, timeperiod=14)  # Negative directional index

        # RSI
        #dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)  # RSI for momentum

        # MFI
        #dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)

        # ATR
        #dataframe['atr'] = ta.ATR(dataframe, timeperiod=12)     #12: 1-hour on 5m tf
    
        # BB SL LEVELS
        #dataframe['BB_upper_sl'] = (dataframe['BB_upper'] + dataframe['atr'] * self.atr_sl_mult.value)
        #dataframe['BB_lower_sl'] = (dataframe['BB_lower'] - dataframe['atr'] * self.atr_sl_mult.value)
        #dataframe['BB_upper_sl'] = ((sma + std_dev * 2.0) * (1 + self.BB_sl_coeff.value))
        #dataframe['BB_lower_sl'] = ((sma - std_dev * 2.0) * (1 - self.BB_sl_coeff.value))
        
        # DONCHIAN SL LEVELS
        #dataframe['dc_low'] = dataframe['low'].rolling(self.dc_length.value).min()
        #dataframe['dc_high'] = dataframe['high'].rolling(self.dc_length.value).max()


        """# TOQUE DE BOLLINGER #####################################################################################
        n = 3  # número de velas a verificar

        # Toque banda superior 4h
        dataframe['toque_BB_upper_4h'] = (
            dataframe['high_4h']
            .rolling(window=n, min_periods=1).max()
            > dataframe['BB_upper_4h']
        )

        # Toque banda inferior 4h
        dataframe['toque_BB_lower_4h'] = (
            dataframe['low_4h']
            .rolling(window=n, min_periods=1).min()
            < dataframe['BB_lower_4h']
        )
        ############################################################################################################"""

        # TOQUE DE BOLLINGER LARGO #################################################################################
        n = 3  # número de velas a verificar

        # Toque banda superior 4h
        dataframe['toque_BB_upper_largo_4h'] = (
            dataframe['high']
            .rolling(window=n, min_periods=1).max()
            > dataframe['BB_upper_largo_4h']
        )

        # Toque banda inferior 4h
        dataframe['toque_BB_lower_largo_4h'] = (
            dataframe['low']
            .rolling(window=n, min_periods=1).min()
            < dataframe['BB_lower_largo_4h']
        )
        ############################################################################################################

        """# Define rebote ############################################################################################
        #SHORTS
        df['rebote_en_upper'] = (
            (df['close'].shift(1) > df['BB_upper'].shift(1))  # previous close above upper band
            &(df['close'] < df['BB_upper'])                      # current close back inside upper band
        )
        #LONGS
        df['rebote_en_lower'] = (
            (df['close'].shift(1) < df['BB_lower'].shift(1))  # previous close below lower band
            &(df['close'] > df['BB_lower'])                      # current close back inside lower band
        )
        
        # Define rebote conditions 1h
        #SHORTS
        df['rebote_en_upper_1h'] = (
            (df['close_1h'].shift(1) > df['BB_upper_1h'].shift(1))  # previous close above upper band
            &(df['close_1h'] < df['BB_upper_1h'])                      # current close back inside upper band
        )
        #LONGS
        df['rebote_en_lower_1h'] = (
            (df['close_1h'].shift(1) < df['BB_lower_1h'].shift(1))  # previous close below lower band
            &(df['close_1h'] > df['BB_lower_1h'])                      # current close back inside lower band
        )
        ############################################################################################################"""
                                    
        return dataframe
        
    ################################################################################################################################
    ######################################## populate_entry_trend ##################################################################
    ################################################################################################################################

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
       
        """# BTC trend condition (1h sma 200 - 3 candles) #############################################################
        btc_1h_up3 =     ((df['btc_usdt_close_1h'] > df['btc_usdt_sma200_1h']).rolling(3).sum() == 3)
        btc_1h_down3 =   ((df['btc_usdt_close_1h'] < df['btc_usdt_sma200_1h']).rolling(3).sum() == 3)
        ############################################################################################################"""
        
        # PATTERNS ############################################################################################

        # Bullish: includes both bullish-only and both-direction patterns > 0
        """pattern_bullish_4h = (
            (df["morning_doji_star_4h"] > 0) |
            (df["morning_star_4h"] > 0) |
            (df["mat_hold_4h"] > 0) |
            (df["three_white_soldiers_4h"] > 0) |
            (df["concealing_baby_swallow_4h"] > 0) |
            (df["hammer_4h"] > 0) |
            (df["dragonfly_doji_4h"] > 0) |
            (df["takuri_4h"] > 0) |
            (df["three_stars_in_south_4h"] > 0) |
            (df["homing_pigeon_4h"] > 0) |
            (df["unique_three_river_4h"] > 0) |
            (df["stick_sandwich_4h"] > 0) |
            (df["ladder_bottom_4h"] > 0) |
            (df["matching_low_4h"] > 0) |
            (df["inverted_hammer_4h"] > 0) |
            (df["engulfing_4h"] > 0) |
            (df["rise_fall_three_methods_4h"] > 0) |
            (df["separating_lines_4h"] > 0) |
            (df["three_inside_4h"] > 0) |
            (df["abandoned_baby_4h"] > 0) |
            (df["breakaway_4h"] > 0) |
            (df["belt_hold_4h"] > 0) |
            (df["closing_marubozu_4h"] > 0) |
            (df["counterattack_4h"] > 0) |
            (df["doji_4h"] > 0) |
            (df["doji_star_4h"] > 0) |
            (df["gap_side_side_white_4h"] > 0) |
            (df["harami_4h"] > 0) |
            (df["harami_cross_4h"] > 0) |
            (df["hikkake_4h"] > 0) |
            (df["hikkake_mod_4h"] > 0) |
            (df["kicking_4h"] > 0) |
            (df["kicking_by_length_4h"] > 0) |
            (df["long_line_4h"] > 0) |
            (df["long_legged_doji_4h"] > 0) |
            (df["marubozu_4h"] > 0) |
            (df["rickshaw_man_4h"] > 0) |
            (df["short_line_4h"] > 0) |
            (df["spinning_top_4h"] > 0) |
            (df["tasuki_gap_4h"] > 0) |
            (df["three_line_strike_4h"] > 0) |
            (df["xside_gap_three_methods_4h"] > 0) |
            (df["tristar_4h"] > 0)
        )

        # Bearish: includes both bearish-only and both-direction patterns < 0
        pattern_bearish_4h = (
            (df["three_black_crows_4h"] < 0) |
            (df["evening_doji_star_4h"] < 0) |
            (df["evening_star_4h"] < 0) |
            (df["identical_three_crows_4h"] < 0) |
            (df["dark_cloud_cover_4h"] < 0) |
            (df["hanging_man_4h"] < 0) |
            (df["advance_block_4h"] < 0) |
            (df["shooting_star_4h"] < 0) |
            (df["gravestone_doji_4h"] < 0) |
            (df["stalled_pattern_4h"] < 0) |
            (df["thrusting_4h"] < 0) |
            (df["upside_gap_two_crows_4h"] < 0) |
            (df["on_neck_4h"] < 0) |
            (df["in_neck_4h"] < 0) |
            (df["two_crows_4h"] < 0) |
            (df["engulfing_4h"] < 0) |
            (df["rise_fall_three_methods_4h"] < 0) |
            (df["separating_lines_4h"] < 0) |
            (df["three_inside_4h"] < 0) |
            (df["abandoned_baby_4h"] < 0) |
            (df["breakaway_4h"] < 0) |
            (df["belt_hold_4h"] < 0) |
            (df["closing_marubozu_4h"] < 0) |
            (df["counterattack_4h"] < 0) |
            (df["doji_4h"] < 0) |
            (df["doji_star_4h"] < 0) |
            (df["gap_side_side_white_4h"] < 0) |
            (df["harami_4h"] < 0) |
            (df["harami_cross_4h"] < 0) |
            (df["hikkake_4h"] < 0) |
            (df["hikkake_mod_4h"] < 0) |
            (df["kicking_4h"] < 0) |
            (df["kicking_by_length_4h"] < 0) |
            (df["long_line_4h"] < 0) |
            (df["long_legged_doji_4h"] < 0) |
            (df["marubozu_4h"] < 0) |
            (df["rickshaw_man_4h"] < 0) |
            (df["short_line_4h"] < 0) |
            (df["spinning_top_4h"] < 0) |
            (df["tasuki_gap_4h"] < 0) |
            (df["three_line_strike_4h"] < 0) |
            (df["xside_gap_three_methods_4h"] < 0) |
            (df["tristar_4h"] < 0)
        )        """

        """# FILTROS VARIOS ###########################################################################################

        # Trend filter: sma10_1h > sma30_1h for last 3 candles
        sma_uptrend_4h = ((df['sma10_4h'] > df['sma30_4h']).rolling(3).sum() == 3)
        sma_downtrend_4h = ((df['sma10_4h'] < df['sma30_4h']).rolling(3).sum() == 3)

        sma_uptrend_1h = ((df['sma10_1h'] > df['sma30_1h']).rolling(3).sum() == 3)
        sma_downtrend_1h = ((df['sma10_1h'] < df['sma30_1h']).rolling(3).sum() == 3)

        sma_uptrend = ((df['sma10'] > df['sma30']).rolling(3).sum() == 3)
        sma_downtrend = ((df['sma10'] < df['sma30']).rolling(3).sum() == 3)

        mfi_down = (df['mfi'] < 30)
        mfi_up = (df['mfi'] > 70)

        rsi_down = (df['rsi'] < 30)
        rsi_up = (df['rsi'] > 70)
        """

        # Generate individual LONG and SHORT enters for each pattern ###############################################################################################

        # LONG entries: bullish and both-direction patterns > 0
        long_patterns = [
            "morning_doji_star", 
            "morning_star", 
            "mat_hold", 
            "three_white_soldiers",
            "concealing_baby_swallow", 
            "hammer", 
            "dragonfly_doji", 
            "takuri",
            "three_stars_in_south", 
            "homing_pigeon", 
            "unique_three_river", 
            "stick_sandwich",
            "ladder_bottom", 
            "matching_low", 
            "inverted_hammer", 
            "engulfing",
            "rise_fall_three_methods", 
            "separating_lines", 
            "three_inside", 
            "abandoned_baby",
            "breakaway", 
            "belt_hold", 
            "closing_marubozu", 
            "counterattack",
            "doji", 
            "doji_star", 
            "gap_side_side_white", 
            #"harami", 
            "harami_cross",
            #"hikkake",                     #anulado por malo
            #"hikkake_mod", 
            "kicking", 
            "kicking_by_length", 
            #"long_line",
            "long_legged_doji", 
            "marubozu", 
            "rickshaw_man", 
            #"short_line", 
            #"spinning_top",
            "tasuki_gap", 
            #"three_line_strike",           #anulado por malo
            "xside_gap_three_methods",
            "tristar"
        ]

        # SHORT entries: bearish and both-direction patterns < 0
        short_patterns = [
            "three_black_crows", 
            "evening_doji_star", 
            "evening_star", 
            "identical_three_crows",
            "dark_cloud_cover", 
            "hanging_man", 
            "advance_block", 
            "shooting_star",
            "gravestone_doji", 
            #"stalled_pattern", 
            "thrusting", 
            "upside_gap_two_crows",
            "on_neck", 
            "in_neck", 
            "two_crows", 
            #"engulfing", 
            "rise_fall_three_methods",
            "separating_lines", 
            "three_inside", 
            "abandoned_baby", 
            "breakaway", 
            "belt_hold",
            "closing_marubozu", 
            "counterattack", 
            "doji", 
            #"doji_star", 
            "gap_side_side_white",
            #"harami", 
            "harami_cross", 
            #"hikkake",                        #anulado por malo
            "hikkake_mod", 
            "kicking", 
            "kicking_by_length",
            #"long_line", 
            "long_legged_doji", 
            #"marubozu", 
            #"rickshaw_man", 
            "short_line",
            #"spinning_top", 
            "tasuki_gap", 
            #"three_line_strike",           #anulado por malo
            "xside_gap_three_methods", 
            "tristar"
        ]

        # LONG entries
        for pattern in long_patterns:
            df.loc[(
                (df[f"{pattern}_4h"] > 0)
                &(df['close'] > df['BB_lower_largo_4h'])                        # no en zona amarilla (bollinger larga)
                #&(df['close'] < (df['BB_middle_largo_TPL_4h'] - df['atr_4h']*3))  # no del lado superior de la primer banda verde + 1 atr
                &(df['BB_ancho_larga_4h'] < df['BB_ancho_larga_sma200_4h'])     # no tradear en banda amplia 
                &(df['toque_BB_lower_largo_4h'] == True)                        # tocando banda de bollinger 4h en utlmas n velas
            ),
                ["enter_long", "enter_tag"]
            ] = [1, f"LONG {pattern.upper()}"]

        # SHORT entries
        for pattern in short_patterns:
            df.loc[(
                (df[f"{pattern}_4h"] < 0)
                &(df['close'] < df['BB_upper_largo_4h'])                        # no en zona amarilla (bollinger larga)
                #&(df['close'] > (df['BB_middle_largo_TPS_4h'] + df['atr_4h']*3))  # no del lado inferior de la primer banda verde
                &(df['BB_ancho_larga_4h'] < df['BB_ancho_larga_sma200_4h'])     # no tradear en banda amplia  
                &(df['toque_BB_upper_largo_4h'] == True)                        # tocando banda de bollinger 4h en utlmas n velas

            ),
                ["enter_short", "enter_tag"]
            ] = [1, f"SHORT {pattern.upper()}"]


        # BB LARGO LONG & SHORT ####################################################################################
        """# LONG TOQUE BB LARGO 4H 
        df.loc[
            (
                (df['toque_BB_lower_largo_4h'] == True)                         # tocando banda de bollinger 4h en utlmas n velas
                &(df['close'] > df['BB_lower_largo_sl_4h'])                     # no en zona roja "stoploss"
                &(df['close'] > df['BB_lower_largo_4h'])                        # no en zona amarilla (bollinger larga)
                &(df['BB_ancho_larga_4h'] < df['BB_ancho_larga_sma200_4h'])     # no tradear en banda amplia  
                #&(pattern_bullish_4h)                                           # patrones de velas en 4h
            ),
            ['enter_long', 'enter_tag']] = (1, 'LONG BOLL LARGO')
        
        # SHORT TOQUE BB LARGO 4H 
        df.loc[
            (
                (df['toque_BB_upper_largo_4h'] == True)                         # tocando banda de bollinger 4h en utlmas n velas
                &(df['close'] < df['BB_upper_largo_sl_4h'])                     # no en zona roja "stoploss"
                &(df['close'] < df['BB_upper_largo_4h'])                        # no en zona amarilla (bollinger larga)
                &(df['BB_ancho_larga_4h'] < df['BB_ancho_larga_sma200_4h'])     # no tradear en banda amplia  
                #&(pattern_bearish_4h)                                           # patrones de velas en 4h

            ),
            ['enter_short', 'enter_tag']] = (1, 'SHORT BOLL LARGO')"""

        return df

    ################################################################################################################################
    ######################################## populate_exit_trend ###################################################################
    ################################################################################################################################

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:        
    # bollinger largo exit: bb middle offset ###################################################################
        """df.loc[
            (df['close'] >= df['BB_middle_largo_TPS_4h']),
            ['exit_long', 'exit_tag']
            ] = (1, 'EXL BB MIDDLE LARGO')

        df.loc[
            (df['close'] <= df['BB_middle_largo_TPL_4h']),
            ['exit_short', 'exit_tag']
            ] = (1, 'EXS BB MIDDLE LARGO')"""

    ############################################################################################################
    ############################################################################################################
        """#  BOLLINGER EXITS (MIDDLE) ################################################################################
        df.loc[
            (df['close'] >= df['BB_middle']),
            ['exit_long', 'exit_tag']
            ] = (1, 'EL BB MIDDLE')

        df.loc[
            (df['close'] <= df['BB_middle']),
            ['exit_short', 'exit_tag']
            ] = (1, 'ES BB MIDDLE')
        ############################################################################################################"""
        """#  PDI / MDI EXITS #########################################################################################

        df.loc[
            (
                (df['mdi'] >= df['pdi'])
                &(df['adx']>25)
            ),
            ['exit_long', 'exit_tag']
            ] = (1, 'EL MDI>PDI')

        df.loc[
            (
                (df['pdi'] >= df['mdi'])
                &(df['adx']>25)
            ),
            ['exit_short', 'exit_tag']
            ] = (1, 'ES MDI<PDI')
        
         #   sma exits
        df.loc[
            (
                (df['sma30'] >= df['sma10'])
                &(df['sma30_1h'] >= df['sma10_1h'])
            ),
            ['exit_long', 'exit_tag']
            ] = (1, 'EL SMA REVERSAL')

        df.loc[
            (
                (df['sma30'] <= df['sma10'])
                &(df['sma30_1h'] <= df['sma10_1h'])
            ),
            ['exit_short', 'exit_tag']
            ] = (1, 'ES SMA REVERSAL')
        ############################################################################################################"""

    #si pdi cruza sobre mdi exit short, si mdi cruza sobre pdi exit long

        return df
    
    ################################################################################################################################
    ######################################## PLOT CONFIGURATION ###################################################################
    ################################################################################################################################
    
    plot_config = {
        'main_plot': {

                #'sma200_1h': {'color': 'white','plotly': {'opacity': 0.9}},
                #'sma10_1h': {'color': 'red','plotly': {'opacity': 0.9}},
                #'sma30_1h': {'color': 'blue','plotly': {'opacity': 0.9}},
                
                #'BB_upper_4h': {'color': 'yellow','plotly': {'opacity': 0.9}},
                #'BB_lower_4h': {'color': 'yellow','plotly': {'opacity': 0.9}},
                #'BB_middle_4h': {'color': 'yellow','plotly': {'opacity': 0.9}},

                #'BB_upper_sl_4h': {'color': 'orange','plotly': {'opacity': 0.5}},
                #'BB_lower_sl_4h': {'color': 'orange','plotly': {'opacity': 0.5}},
                #'dc_high': {'color': '#c927a6','plotly': {'opacity': 0.5}},
                #'dc_low': {'color': '#c927a6','plotly': {'opacity': 0.5}},

                #BB LARGO 4H
                'BB_upper_largo_4h': {'color': 'yellow','plotly': {'opacity': 0.9}},
                'BB_lower_largo_4h': {'color': 'yellow','plotly': {'opacity': 0.9}},
                'BB_middle_largo_4h': {'color': 'yellow','plotly': {'opacity': 0.9}},

                'BB_upper_largo_sl_4h': {'color': 'red','plotly': {'opacity': 0.5}},
                'BB_lower_largo_sl_4h': {'color': 'red','plotly': {'opacity': 0.5}},

                'BB_middle_largo_TPL_4h': {'color': 'green','plotly': {'opacity': 0.5}},
                'BB_middle_largo_TPS_4h': {'color': 'green','plotly': {'opacity': 0.5}},

        },
        'subplots': {
   
            "ANCHO BB": {
                'BB_ancho_larga_4h': {'color': "#28b8e7",'plotly': {'opacity': 0.5}},
                'BB_ancho_larga_sma200_4h': {'color': 'white','plotly': {'opacity': 0.5}},
            },
            
            #"ATR": {
            #    'atr': {'color': 'yellow'},
            #},
            
            #"ADX": {
            #    'adx': {'color': 'yellow'},
            #    'pdi': {'color': 'green', 'fill_to': 'adx'},
            #    'mdi': {'color': 'red', 'fill_to': 'adx'}
            #},

            #"RSI/MFI": {
            #    'rsi': {'color': 'purple'},
            #    'mfi': {'color': 'red'},
            #},
            
            #"BTC 1h": {
            #    'btc_usdt_close_1h': {'color': 'orange'},
            #    'btc_usdt_sma200_1h': {'color': 'white','plotly': {'opacity': 0.9}}
            #},
        },
    }


    #idea 2: banda de bollinger en 4h con periodo 200 y 2.9 std devs, long cuando pega en banda y take profit en la media (o casi en la media)
    #opc. desactivar en tendencias fuertes (banda muy ancha)
        #como saber "bada ancha": tomar la media de banda (upper-lower) y solo operar debajo de la media
    #hyperoptable n° std devs bollingers
