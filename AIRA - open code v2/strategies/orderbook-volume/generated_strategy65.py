
# =============================================================
#  Fichier généré par Moutonneux pour Freqtrade France 🇫🇷
#  ➤ Soutenez-nous sur : https://coff.ee/freqtrade_france et obtenez davantage de stratégies
#
#  📌 Ce code est fourni à des fins personnelles ou éducatives.
#  ❌ Il ne doit pas être partagé, distribué, ni publié dans un autre cadre.
# =============================================================


import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.persistence import Trade
import math
from scipy import stats, signal
from functools import reduce
from datetime import datetime
from pandas import DataFrame, concat
from typing import Optional, Union, Dict
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from freqtrade.optimize.hyperopt import IHyperOptLoss
from external_indicators_v11 import *
from ta.volatility import AverageTrueRange
import talib.abstract as ta

import logging
logger = logging.getLogger(__name__)

from freqtrade.strategy import (timeframe_to_minutes, BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, IStrategy, merge_informative_pair)

import ta as clean_ta
import pandas_ta as pta
from technical import qtpylib


# La stratégie a été trouvée avec les paramètres suivants :
#   max positions = 10
#   timeframe = 4h
#   config = backtest_configs/futures_binance.json
#   timerange = 20220101-
#   hyperoptloss = MultiMetricHyperOptLoss
#   spaces = buy sell

#command = freqtrade backtesting --strategy generated_strategy65_v1 --config backtest_configs/futures_binance.json --timerange 20220101- --timeframe 4h --max-open-trades 10 --cache none --dry-run-wallet 1000

def lerp(a: float, b: float, t: float) -> float:
    return (1 - t) * a + t * b

class generated_strategy65_v1(IStrategy):    
    # Fonction appelée au démarrage du bot (en backtest ou hyperopt uniquement)
    # Initialise une liste interne pour garder en mémoire les trades ouverts, utilisé notamment
    # dans confirm_trade_exit() pour purger les données de suivi des stakes personnalisés.
    # (Lorsque le bot est en live, on peut consulter les trades ouverts sans que ce type d'action soit nécessaire)
    def bot_start(self, **kwargs) -> None:
        if self.dp.runmode.value in ("backtest", "hyperopt"):
            self._open_trades = []
            self.scaled_entries = {}
            self.trailing_buy_data = {}
            self.daily_profit_tracker = {}
            self.daily_trades_closed = {}
            self.partial_exits = {}
    
    
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 10
    position_adjustment_enable = True

    INTERFACE_VERSION = 3
    can_short: bool = False

    minimal_roi = {'0': 100000.0}
    stoploss = -1
    trailing_stop = False
    trailing_stop_positive = 0.0
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    timeframe = '4h'

    stoploss = -1



    profit_ratio_needed = DecimalParameter(-0.100, 0.200, decimals=3, default=0.067, space="sell", optimize=True)
    exit_only_profit = BooleanParameter(default=True, space="sell", optimize=True)
    max_hold_hours = CategoricalParameter([8, 16, 24, 48, 72, 96, 120, 150, 200, 250, 300, 350, 400, 450, 500, 750, 1000, 1250, 1500], default=1250, space="sell", optimize=True)

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, exit_reason: str, current_time: datetime, **kwargs) -> bool:
        if 'stop_loss' in exit_reason or exit_reason == 'force_exit':
            if trade.amount == amount and pair in self.cust_proposed_initial_stakes:
                del self.cust_proposed_initial_stakes[pair]
                if self.dp.runmode.value in ("backtest", "hyperopt"):
                    for trade in self._open_trades:
                        if trade.get('pair') == pair:
                            self._open_trades.remove(trade)
                            break
            return True
        
        if self.exit_only_profit.value:
            trade_duration_hours = (current_time - trade.open_date_utc).total_seconds() / 3600
            if trade_duration_hours >= self.max_hold_hours.value:
                if self.dp.runmode.value not in ("hyperopt"):
                    logger.info(f"Exit for {pair} autorisé après {trade_duration_hours:.1f}h (max: {self.max_hold_hours.value}h)")
            elif trade.calc_profit_ratio(rate) < self.profit_ratio_needed.value:
                if self.dp.runmode.value not in ("hyperopt"):
                    logger.info(f"Exit for {pair} bloqué par profit ratio ({trade.calc_profit_ratio(rate):.3f} < {self.profit_ratio_needed.value:.3f})")
                return False

        if trade.amount == amount and pair in self.cust_proposed_initial_stakes:
            del self.cust_proposed_initial_stakes[pair]
        return True



    ################################################
    # Fonctions pour gérer la sortie d'une position
    # Globalement on applique juste un stoploss manuellement
    ################################################
    # Stoploss personnalisé exprimé en ratio de perte (ex : -0.8 = -80 %).
    # Si le profit courant descend en dessous de ce seuil, la fonction custom_exit() déclenche une sortie immédiate.
    my_custom_stoploss = DecimalParameter(-0.9, -0.1, decimals=1, default=-0.11, space="sell", optimize=True)
    
    use_partial_profit = CategoricalParameter([True, False], default=False, space="sell", optimize=True)
    partial_profit_levels = CategoricalParameter([2, 3, 4], default=2, space="sell", optimize=True)
    partial_profit_pct_1 = DecimalParameter(0.005, 0.50, default=0.147, space="sell", optimize=True, decimals=3)
    partial_profit_pct_2 = DecimalParameter(0.01, 1.00, default=0.512, space="sell", optimize=True, decimals=3)
    partial_profit_pct_3 = DecimalParameter(0.02, 1.50, default=0.537, space="sell", optimize=True, decimals=3)
    partial_profit_pct_4 = DecimalParameter(0.05, 2.00, default=1.858, space="sell", optimize=True, decimals=3)
    
    use_parabolic_exit = CategoricalParameter([True, False], default=False, space="sell", optimize=True)
    parabolic_lookback = CategoricalParameter([3, 5, 7, 10, 15], default=7, space="sell", optimize=True)
    parabolic_gain_threshold = DecimalParameter(0.10, 0.50, default=0.24, space="sell", optimize=True, decimals=2)
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        if self.use_partial_profit.value:
            if pair not in self.partial_exits:
                self.partial_exits[pair] = 0
            
            levels = [
                self.partial_profit_pct_1.value, 
                self.partial_profit_pct_2.value, 
                self.partial_profit_pct_3.value,
                self.partial_profit_pct_4.value
            ]
            
            for i, level in enumerate(levels[:self.partial_profit_levels.value]):
                if current_profit >= level and self.partial_exits[pair] <= i:
                    self.partial_exits[pair] = i + 1
                    if i == self.partial_profit_levels.value - 1:
                        if pair in self.partial_exits:
                            del self.partial_exits[pair]
                        return f'partial_take_final_{i+1}'
        
        if self.use_parabolic_exit.value:
            try:
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if dataframe is not None and len(dataframe) >= self.parabolic_lookback.value:
                    recent_gain = (dataframe['close'].iloc[-1] / dataframe['close'].iloc[-self.parabolic_lookback.value] - 1)
                    if recent_gain > self.parabolic_gain_threshold.value:
                        return 'parabolic_exit'
            except Exception as e:
                logger.debug(f"Error checking parabolic exit for {pair}: {e}")
        
        tag = super().custom_sell(pair, trade, current_time, current_rate, current_profit, **kwargs)
        if tag:
            return tag
        if current_profit <= self.my_custom_stoploss.value:
            return 'stop_loss'
        return None
        
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, exit_reason: str, current_time: datetime, **kwargs) -> bool:
        if 'stop_loss' in exit_reason or exit_reason == 'force_exit':
            if trade.amount == amount and pair in self.cust_proposed_initial_stakes:
                del self.cust_proposed_initial_stakes[pair]
                if pair in self.scaled_entries:
                    del self.scaled_entries[pair]
                if pair in self.partial_exits:
                    del self.partial_exits[pair]
                    
                if self.dp.runmode.value in ("backtest", "hyperopt"):
                    for trade in self._open_trades:
                        if trade.get('pair') == pair:
                            self._open_trades.remove(trade)
                            break
            return True
        
        if self.exit_only_profit.value:
            trade_duration_hours = (current_time - trade.open_date_utc).total_seconds() / 3600
            if trade_duration_hours >= self.max_hold_hours.value:
                if self.dp.runmode.value not in ("hyperopt"):
                    logger.info(f"Exit for {pair} autorisé après {trade_duration_hours:.1f}h (max: {self.max_hold_hours.value}h)")
            elif trade.calc_profit_ratio(rate) < self.profit_ratio_needed.value:
                if self.dp.runmode.value not in ("hyperopt"):
                    logger.info(f"Exit for {pair} bloqué par profit ratio ({trade.calc_profit_ratio(rate):.3f} < {self.profit_ratio_needed.value:.3f})")
                return False

        if trade.amount == amount and pair in self.cust_proposed_initial_stakes:
            del self.cust_proposed_initial_stakes[pair]
            if pair in self.scaled_entries:
                del self.scaled_entries[pair]
            if pair in self.partial_exits:
                del self.partial_exits[pair]
        return True

    trading_time_mode = CategoricalParameter(["always", "weekdays_only", "weekend_only"], default="always", space="buy", optimize=True)
    
    use_max_daily_loss = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    max_daily_loss_pct = DecimalParameter(0.01, 0.10, default=0.039, space="buy", optimize=True, decimals=3)
    
    entry_mode = CategoricalParameter(["none", "entry_scaling", "trailing_buy"], default="none", space="buy", optimize=True)
    
    entry_scaling_orders = CategoricalParameter([2, 3, 4, 5], default=3, space="buy", optimize=True)
    entry_scaling_mode = CategoricalParameter(["equal", "increasing", "decreasing"], default="equal", space="buy", optimize=True)
    
    trailing_buy_pct = DecimalParameter(0.001, 0.10, default=0.043, space="buy", optimize=True, decimals=3)
    trailing_buy_max_distance = DecimalParameter(0.01, 0.20, default=0.037, space="buy", optimize=True, decimals=3)
    trailing_buy_max_hours = CategoricalParameter([1, 2, 4, 8, 12, 24, 48], default=4, space="buy", optimize=True)
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, current_time: datetime, entry_tag: str, side: str, **kwargs) -> bool:
        if self.trading_time_mode.value != "always":
            day_of_week = current_time.weekday()
            if self.trading_time_mode.value == "weekdays_only":
                if day_of_week >= 5:
                    return False
            elif self.trading_time_mode.value == "weekend_only":
                if day_of_week < 5:
                    return False
        
        if self.use_max_daily_loss.value:
            today = current_time.date()
            daily_pnl = 0.0
            
            if self.dp.runmode.value in ("backtest", "hyperopt"):
                if today not in self.daily_profit_tracker:
                    self.daily_profit_tracker[today] = 0.0
                daily_pnl = self.daily_profit_tracker[today]
            else:
                try:
                    closed_trades = Trade.get_trades([Trade.is_open.is_(False), Trade.close_date >= datetime.combine(today, datetime.min.time())])
                    for t in closed_trades:
                        if t.close_profit:
                            daily_pnl += t.close_profit
                except Exception as e:
                    logger.debug(f"Error calculating daily P&L: {e}")
            
            if daily_pnl < -self.max_daily_loss_pct.value:
                logger.info(f"Max daily loss reached for {today}: {daily_pnl:.2%} < -{self.max_daily_loss_pct.value:.2%}")
                return False
        
        if self.entry_mode.value == "trailing_buy":
            if pair not in self.trailing_buy_data:
                self.trailing_buy_data[pair] = {
                    'signal_price': rate,
                    'best_price': rate,
                    'signal_time': current_time
                }
                return False
            
            trailing_data = self.trailing_buy_data[pair]
            
            if rate < trailing_data['best_price']:
                trailing_data['best_price'] = rate
                return False
            
            price_bounce = (rate - trailing_data['best_price']) / trailing_data['best_price']
            if price_bounce >= self.trailing_buy_pct.value:
                del self.trailing_buy_data[pair]
                return True
            
            price_distance = (rate - trailing_data['signal_price']) / trailing_data['signal_price']
            time_elapsed = (current_time - trailing_data['signal_time']).total_seconds() / 3600
            
            if price_distance > self.trailing_buy_max_distance.value or time_elapsed > self.trailing_buy_max_hours.value:
                del self.trailing_buy_data[pair]
                return False
            
            return False
        
        return True


    
    ################################################
    # Fonctions pour gérer l'investissement initial d'une position lors de son ouverture
    ################################################
    
    # Active ou non la personnalisation du stake initial. Si False, on utilise le stake proposé tel quel.
    use_custom_stake = CategoricalParameter([True, False], default=False, space="buy", optimize=True)
    
    # Facteur de réduction du stake initial. Une valeur < 1 permet de garder des fonds pour les safety orders.
    # Valeurs recommandées : overbuy_factor = 0.6 à 0.95 pour DCA actif, 1.0 si stratégie sans rechargements.
    overbuy_factor = CategoricalParameter([0.8, 0.9, 0.995, 1.0], default=0.9, space="buy", optimize=True)
    
    use_volatility_stake = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    volatility_size = CategoricalParameter([4,6,8,10,12,14,16,20, 25, 30, 35, 40], default=10, space="buy", optimize=True)
    volatility_max_filter = DecimalParameter(0.01, 0.12, default=0.03, space="buy", optimize=True, decimals=3)
    
    use_walcl_stake = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    walcl_size = CategoricalParameter([7, 14, 50, 72, 100, 200, 300, 45, 46], default=100, space="buy", optimize=True)
    walcl_ema_size = CategoricalParameter([14,42,72,84,126,168,212,256,300,400,500], default=126, space="buy", optimize=True)
    walcl_stake_mode = CategoricalParameter([1,2,3,4,5,6], default=1, space="buy", optimize=True)
    walcl_adjustment = CategoricalParameter([0.25, 0.5, 0.75], default=0.25, space="buy", optimize=True)
    
    volatility_size_stake = CategoricalParameter([4, 6, 8, 10, 12, 14, 16, 20, 25], default=14, space="buy", optimize=True)
    volatility_max_filter_stake = DecimalParameter(0.01, 0.30, default=0.132, space="buy", optimize=True, decimals=3)
    
    use_volume_stake = CategoricalParameter([True, False], default=False, space="buy", optimize=True)
    volume_stake_min_threshold = CategoricalParameter([10_000, 50_000, 100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 700_000, 800_000, 1_000_000, 1_500_000], default=400000, space="buy", optimize=True)
    volume_stake_max_threshold = CategoricalParameter([25_000, 50_000, 75_000, 100_000, 250_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000, 5_000_000, 7_500_000, 10_000_000], default=1000000, space="buy", optimize=True)
    volume_stake_min_ratio = DecimalParameter(0.5, 1.0, default=0.89, space="buy", optimize=True, decimals=2)
    volume_stake_max_ratio = DecimalParameter(0.5, 1.0, default=0.8, space="buy", optimize=True, decimals=2)
    
    use_drawdown_sizing = CategoricalParameter([True, False], default=False, space="buy", optimize=True)
    drawdown_threshold = DecimalParameter(0.05, 0.30, default=0.24, space="buy", optimize=True, decimals=2)
    drawdown_reduction = DecimalParameter(0.3, 0.8, default=0.64, space="buy", optimize=True, decimals=2)
    drawdown_lookback = CategoricalParameter([10, 20, 30, 50], default=20, space="buy", optimize=True)
    
    use_portfolio_rebalancing = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    rebalance_threshold = DecimalParameter(0.20, 0.50, default=0.39, space="buy", optimize=True, decimals=2)
    max_position_pct = DecimalParameter(0.15, 0.40, default=0.36, space="buy", optimize=True, decimals=2)
    
    # Plafond en % du capital disponible (capital initial + gain) qui sera utilisé pour ouvrir des positions
    # Une position ne pourra pas dépasser ((available_capital + gain) * tradable_balance_ratio) / max_open_trades
    # 0.25 = 25% du capital maximum
    tradable_balance_ratio = CategoricalParameter([0.25, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], default=1.0, space="buy", optimize=True)
    
    # Détermine la taille personnalisée du premier ordre d'achat.
    # Utilisé uniquement si use_custom_stake est activé.
    # Divise le stake proposé par le multiplicateur de safety orders, puis applique un facteur de réduction (overbuy_factor).
    # Permet de ne pas investir tout le capital dès le départ pour garder de la marge pour les rechargements.
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, proposed_stake: float, min_stake: float, max_stake: float, **kwargs) -> float:
        
        if self.use_custom_stake.value:
            base_stake = proposed_stake / self.get_max_so_multiplier() * self.overbuy_factor.value
            self.cust_proposed_initial_stakes[pair] = base_stake
        else:
            base_stake = proposed_stake
        
        if self.entry_mode.value == "entry_scaling":
            if pair not in self.scaled_entries:
                self.scaled_entries[pair] = 1
            else:
                self.scaled_entries[pair] += 1
            
            if self.entry_scaling_mode.value == "equal":
                base_stake = base_stake / self.entry_scaling_orders.value
            elif self.entry_scaling_mode.value == "increasing":
                total_weight = sum(range(1, self.entry_scaling_orders.value + 1))
                current_weight = self.scaled_entries[pair]
                base_stake = base_stake * (current_weight / total_weight)
            elif self.entry_scaling_mode.value == "decreasing":  
                total_weight = sum(range(1, self.entry_scaling_orders.value + 1))
                current_weight = self.entry_scaling_orders.value - self.scaled_entries[pair] + 1
                base_stake = base_stake * (current_weight / total_weight)
        
        if self.use_walcl_stake.value:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            walcl_adjustment=1.0
            if self.walcl_stake_mode.value == 1:
                if (dataframe['WALCL_rolling'].iloc[-1] > dataframe['WALCL_ema'].iloc[-1]):
                    walcl_adjustment = self.walcl_adjustment.value
            if self.walcl_stake_mode.value == 2:
                if (dataframe['WALCL_rolling'].iloc[-1] < dataframe['WALCL_ema'].iloc[-1]):
                    walcl_adjustment = self.walcl_adjustment.value
            if self.walcl_stake_mode.value == 3:
                if (dataframe['WALCL_rolling'].iloc[-1] < dataframe['WALCL_rolling'].iloc[-2]):
                    walcl_adjustment = self.walcl_adjustment.value
            if self.walcl_stake_mode.value == 4:
                if (dataframe['WALCL_rolling'].iloc[-1] > dataframe['WALCL_rolling'].iloc[-2]):
                    walcl_adjustment = self.walcl_adjustment.value
            if self.walcl_stake_mode.value == 5:
                if (dataframe['WALCL_ema'].iloc[-1] < dataframe['WALCL_ema'].iloc[-2]):
                    walcl_adjustment = self.walcl_adjustment.value
            if self.walcl_stake_mode.value == 6:
                if (dataframe['WALCL_ema'].iloc[-1] > dataframe['WALCL_ema'].iloc[-2]):
                    walcl_adjustment = self.walcl_adjustment.value
            base_stake *= walcl_adjustment
        
        
        if self.use_volatility_stake.value:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is not None and not dataframe.empty:
                # Calculer l'ATR récent
                recent_atr_pct = (dataframe['high'].iloc[-int(self.volatility_size_stake.value):] - dataframe['low'].iloc[-int(self.volatility_size_stake.value):]).mean() / dataframe['close'].iloc[-1]
                
                # Réduire la taille si volatilité élevée
                if recent_atr_pct > self.volatility_max_filter_stake.value:
                    volatility_adjustment = 0.5
                elif recent_atr_pct > self.volatility_max_filter_stake.value*0.6:
                    volatility_adjustment = 0.75
                else:
                    volatility_adjustment = 1.0
                    
                base_stake *= volatility_adjustment
        
        if self.use_volume_stake.value:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is not None and not dataframe.empty:
                if 'volume_mean_rolling' in dataframe.columns:
                    current_volume = dataframe['volume_mean_rolling'].iloc[-1]
                    
                    min_threshold = float(self.volume_stake_min_threshold.value)
                    max_threshold = float(self.volume_stake_max_threshold.value)
                    
                    # S'assurer que max > min
                    if max_threshold <= min_threshold:
                        max_threshold = min_threshold * 2
                    
                    if current_volume <= min_threshold:
                        volume_adjustment = self.volume_stake_min_ratio.value
                    elif current_volume >= max_threshold:
                        volume_adjustment = self.volume_stake_max_ratio.value
                    else:
                        # Interpolation linéaire
                        volume_position = (current_volume - min_threshold) / (max_threshold - min_threshold)
                        volume_adjustment = self.volume_stake_min_ratio.value + (
                            (self.volume_stake_max_ratio.value - self.volume_stake_min_ratio.value) * volume_position
                        )
                    
                    base_stake *= volume_adjustment
        
        if self.use_drawdown_sizing.value:
            try:
                recent_profit = 0.0
                trade_count = 0
                
                if self.dp.runmode.value in ("backtest", "hyperopt"):
                    if hasattr(self, 'daily_trades_closed'):
                        all_trades = []
                        for date_trades in self.daily_trades_closed.values():
                            all_trades.extend(date_trades)
                        recent_trades = all_trades[-self.drawdown_lookback.value:] if all_trades else []
                        for t in recent_trades:
                            if 'profit_ratio' in t:
                                recent_profit += t['profit_ratio']
                                trade_count += 1
                else:
                    closed_trades = Trade.get_trades([Trade.is_open.is_(False)]).order_by(Trade.close_date.desc()).limit(self.drawdown_lookback.value)
                    for t in closed_trades:
                        if t.close_profit:
                            recent_profit += t.close_profit  
                            trade_count += 1
                
                if trade_count > 0 and recent_profit < -self.drawdown_threshold.value:
                    logger.info(f"Drawdown detected ({recent_profit:.2%}), reducing stake by {1-self.drawdown_reduction.value:.0%}")
                    base_stake *= self.drawdown_reduction.value
                    
            except Exception as e:
                logger.debug(f"Error calculating drawdown sizing: {e}")
        
        if self.use_portfolio_rebalancing.value:
            try:
                open_trades = []
                total_stake = 0.0
                
                if self.dp.runmode.value in ("backtest", "hyperopt"):
                    open_trades = self._open_trades if hasattr(self, '_open_trades') else []
                    total_stake = sum(t.get('stake_amount', 0) for t in open_trades)
                else:
                    open_trades = Trade.get_open_trades()
                    total_stake = sum(t.stake_amount for t in open_trades)
                
                if total_stake > 0:
                    max_stake_found = 0
                    if self.dp.runmode.value in ("backtest", "hyperopt"):
                        for t in open_trades:
                            stake_pct = t.get('stake_amount', 0) / total_stake if total_stake > 0 else 0
                            max_stake_found = max(max_stake_found, stake_pct)
                    else:
                        for t in open_trades:
                            stake_pct = t.stake_amount / total_stake
                            max_stake_found = max(max_stake_found, stake_pct)
                    
                    if max_stake_found > self.max_position_pct.value:
                        logger.info(f"Portfolio imbalance detected, reducing stake by {self.rebalance_threshold.value:.0%}")
                        base_stake *= (1 - self.rebalance_threshold.value)
                        
            except Exception as e:
                logger.debug(f"Error in portfolio rebalancing: {e}")
        
        ratio      = self.tradable_balance_ratio.value        # ex. 0.35
        max_trades = self.config.get("max_open_trades", 1)    # ex. 8

        # Capital initial défini ?
        available_wallet = self.wallets.get_total_stake_amount()
        quota = (available_wallet * ratio) / max_trades               # ex. 200*0.35/8 = 8.75

        # On ne dépasse pas ce quota
        base_stake = min(base_stake, quota)
            
        stake_final = max(15.0, base_stake)           # min requis par Hyperliquid
        stake_final = min(stake_final, max_stake)     # limite fournie par Freqtrade

        return stake_final
    
    ################################################
    # Fonctions pour recharger une position (DCA)
    ################################################
    
    # Déclencheur initial de safety order. Si le profit d'une position est inférieur à cette valeur,
    # adjust_trade_position() considère un premier rechargement.
    initial_safety_order_trigger = DecimalParameter(-0.5, 0.0, decimals=3, default=-0.099, space="buy", optimize=True)
    
    # Si activé, un signal d’entrée est requis en plus du seuil de perte (initial_safety_order_trigger) pour autoriser un rechargement.
    # Permet d’éviter de recharger dans une tendance baissière sans reprise détectée.
    adjust_need_entry_signal = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    
    # Active le rechargement uniquement s’il y a un pic de volume anormalement élevé.
    # Cela permet de ne recharger que lorsque le marché montre une impulsion.
    adjust_require_volume_spike = CategoricalParameter([True, False], default=False, space="buy", optimize=True)

    # Taille de la fenêtre utilisée pour le volume moyen (rolling).
    # Une fenêtre plus large donne une moyenne plus lissée.
    volume_spike_window = IntParameter(3, 50, default=15, space="buy", optimize=True)

    # Facteur multiplicatif appliqué à la moyenne de volume pour définir un "pic".
    # Exemple : si le volume actuel > 1.5 × volume moyen, on considère qu’il y a un pic.
    volume_spike_threshold = DecimalParameter(0.7, 3.0, decimals=1, default=1.25, space="buy", optimize=True)

    
    # Influence la rapidité de déclenchement des safety orders suivants.
    # Plus cette valeur est grande, plus les rechargements sont espacés.
    safety_order_step_scale = IntParameter(1, 20, default=4, space="buy", optimize=True)
    
    # Facteur utilisé pour compenser les différences entre le montant prévu et le montant réellement rempli
    # lors d’un rechargement partiel. Appliqué dans adjust_trade_position().
    partial_fill_compensation_scale = DecimalParameter(0.4, 1, decimals=1, default=0.9, space="buy", optimize=True)
    
    use_time_based_dca = CategoricalParameter([True, False], default=False, space="buy", optimize=True)
    time_based_dca_candles = CategoricalParameter([1, 2, 4, 8, 12, 24, 48], default=2, space="buy", optimize=True)
    
    # Active ou non l'ajustement automatique des positions via adjust_trade_position().
    # Si False, aucun rechargement ne sera effectué, quelle que soit la perte.
    use_position_adjustment = CategoricalParameter([True,False], default=True, space="buy", optimize=True)

    
    # Fonction d'ajustement dynamique de la position — appelée automatiquement par Freqtrade
    # lorsqu’un trade est en cours et qu’un ajustement est autorisé (use_position_adjustment=True).
    # Elle décide si la position doit être rechargée et calcule le montant à ajouter, en fonction du
    # profit actuel, du nombre d’ordres précédents (count_of_buys), et des paramètres de la stratégie.
    # Utilise get_max_so_multiplier() pour limiter les rechargements et applique un système de compensation
    # en cas de remplissage partiel (via partial_fill_compensation_scale).

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, min_stake: float, max_stake: float, **kwargs) -> Optional[float]:
        if self.use_position_adjustment.value:
            if current_profit > self.initial_safety_order_trigger.value:
                return None

            if self.adjust_need_entry_signal.value or self.adjust_require_volume_spike.value:
                dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            
            # Vérifie s'il y a un signal d'entrée à la bougie courante
            if self.adjust_need_entry_signal.value:
                if dataframe is None or dataframe.empty:
                    return None
                if dataframe.iloc[-1].get("enter_long", 0) != 1:
                    return None
            

            if self.adjust_require_volume_spike.value:
                if dataframe is None or dataframe.empty:
                    return None

                vol_window = int(self.volume_spike_window.value)
                vol_threshold = self.volume_spike_threshold.value

                if len(dataframe) < vol_window + 1:
                    return None  # pas assez de données pour le rolling

                recent_volume = dataframe['volume'].iloc[-1]
                avg_volume = dataframe['volume'].rolling(vol_window).mean().iloc[-1]

                if recent_volume < avg_volume * vol_threshold:
                    return None  # pas de pic de volume => on ne recharge pas

            
            filled_buys = trade.select_filled_orders(trade.entry_side)
            count_of_buys = len(filled_buys)
            
            if self.entry_mode.value == "entry_scaling":
                if trade.pair in self.scaled_entries:
                    if self.scaled_entries[trade.pair] >= self.entry_scaling_orders.value:
                        return None
                    
                    if self.adjust_need_entry_signal.value:
                        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                        if dataframe is None or dataframe.empty:
                            return None
                        if dataframe.iloc[-1].get("enter_long", 0) != 1:
                            return None
                    
                    try:
                        actual_initial_stake = filled_buys[0].cost if filled_buys else trade.stake_amount
                        
                        if self.entry_scaling_mode.value == "equal":
                            return actual_initial_stake
                        elif self.entry_scaling_mode.value == "increasing":
                            return actual_initial_stake * (self.scaled_entries[trade.pair] + 1) / 2
                        elif self.entry_scaling_mode.value == "decreasing":
                            remaining = self.entry_scaling_orders.value - self.scaled_entries[trade.pair]
                            return actual_initial_stake * remaining / self.entry_scaling_orders.value
                    except Exception as e:
                        logger.debug(f"Error in entry scaling: {e}")
                        return None
                        
            if self.use_time_based_dca.value:
                tf_minutes = timeframe_to_minutes(self.timeframe)
                minutes_in_trade = (current_time - trade.open_date_utc).total_seconds() / 60
                candles_in_trade = int(minutes_in_trade / tf_minutes)
                
                expected_buys = int(candles_in_trade / self.time_based_dca_candles.value) + 1
                
                if count_of_buys < expected_buys and count_of_buys <= self.max_so_multiplier_orig.value:
                    try:
                        actual_initial_stake = filled_buys[0].cost
                        stake_amount = actual_initial_stake * math.pow(self.safety_order_volume_scale.value, (count_of_buys - 1))
                        
                        if trade.pair in self.cust_proposed_initial_stakes:
                            proposed_initial_stake = self.cust_proposed_initial_stakes[trade.pair]
                            if proposed_initial_stake > 0:
                                already_bought = sum(filled_buy.cost for filled_buy in filled_buys)
                                current_actual_stake = already_bought * math.pow(self.safety_order_volume_scale.value, (count_of_buys - 1))
                                current_stake_preposition = proposed_initial_stake * math.pow(self.safety_order_volume_scale.value, (count_of_buys - 1))
                                current_stake_preposition_compensation = current_stake_preposition + abs(current_stake_preposition - current_actual_stake)
                                total_so_stake = lerp(current_actual_stake, current_stake_preposition_compensation, self.partial_fill_compensation_scale.value)
                                stake_amount = total_so_stake
                        
                        return stake_amount
                    except Exception as e:
                        logger.debug(f"Error in time-based DCA: {e}")

            if 1 <= count_of_buys <= self.max_so_multiplier_orig.value:
                safety_order_trigger = (abs(self.initial_safety_order_trigger.value) * count_of_buys)
                if self.safety_order_step_scale.value > 1:
                    safety_order_trigger = abs(self.initial_safety_order_trigger.value) + (abs(self.initial_safety_order_trigger.value) * self.safety_order_step_scale.value * (math.pow(self.safety_order_step_scale.value, (count_of_buys - 1)) - 1) / (self.safety_order_step_scale.value - 1))
                elif self.safety_order_step_scale.value < 1:
                    safety_order_trigger = abs(self.initial_safety_order_trigger.value) + (abs(self.initial_safety_order_trigger.value) * self.safety_order_step_scale.value * (1 - math.pow(self.safety_order_step_scale.value, (count_of_buys - 1))) / (1 - self.safety_order_step_scale.value))

                if current_profit <= (-1 * abs(safety_order_trigger)):
                    try:
                        actual_initial_stake = filled_buys[0].cost
                        stake_amount = actual_initial_stake
                        already_bought = sum(filled_buy.cost for filled_buy in filled_buys)
                        if trade.pair in self.cust_proposed_initial_stakes:
                            if self.cust_proposed_initial_stakes[trade.pair] > 0:
                                proposed_initial_stake = self.cust_proposed_initial_stakes[trade.pair]
                                current_actual_stake = already_bought * math.pow(self.safety_order_volume_scale.value, (count_of_buys - 1))
                                current_stake_preposition = proposed_initial_stake * math.pow(self.safety_order_volume_scale.value, (count_of_buys - 1))
                                current_stake_preposition_compensation = current_stake_preposition + abs(current_stake_preposition - current_actual_stake)
                                total_so_stake = lerp(current_actual_stake, current_stake_preposition_compensation, self.partial_fill_compensation_scale.value)
                                stake_amount = total_so_stake
                            else:
                                stake_amount = stake_amount * math.pow(self.safety_order_volume_scale.value, (count_of_buys - 1))
                        else:
                            stake_amount = stake_amount * math.pow(self.safety_order_volume_scale.value, (count_of_buys - 1))
                        amount = stake_amount / current_rate
                        logger.info(
                            f"Initiating safety order buy #{count_of_buys} "
                            f"for {trade.pair} with stake amount of {stake_amount}. "
                            f"which equals {amount}. "
                            f"Previously bought: {already_bought}. "
                            f"Now overall:{already_bought}{stake_amount}. ")
                        return stake_amount
                    except Exception as exception:
                        logger.info(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}')
                        return None
        else:
            return None
            
            
            
    # Paramètre définissant le nombre maximum de safety orders autorisés.
    # Plus cette valeur est élevée, plus la stratégie pourra recharger une position perdante
    # sur une longue série de baisse. Ce paramètre est utilisé dans get_max_so_multiplier() et
    # adjust_trade_position() pour limiter le nombre d’ajustements successifs autorisés.
    max_so_multiplier_orig = IntParameter(1, 10, default=4, space="buy", optimize=True)
    max_so_multiplier = max_so_multiplier_orig.value
    
    
    # Détermine le facteur multiplicateur pour chaque nouveau safety order.
    # Une valeur > 1 augmente exponentiellement le volume de chaque ordre supplémentaire,
    # ce qui permet une récupération plus rapide mais augmente le risque.
    # Utilisé dans get_max_so_multiplier() et adjust_trade_position().
    safety_order_volume_scale = DecimalParameter(0.5, 9.0, decimals=1, default=1.5, space="buy", optimize=True)
    
    cust_proposed_initial_stakes = {}
    # Calcule le multiplicateur total de stake nécessaire pour couvrir tous les safety orders.
    # Utilisé pour répartir le capital dès le premier ordre (via custom_stake_amount) et pour limiter le nombre total de rechargements.
    # Plus safety_order_volume_scale est grand, plus les ordres suivants sont gros, donc le multiplicateur augmente vite.
    # Valeur typique : entre 2 et 10 selon la stratégie ; attention au risque de surexposition si trop élevé.
    def get_max_so_multiplier(self):
        if (self.max_so_multiplier_orig.value > 0):
            if (self.safety_order_volume_scale.value > 1):
                firstLine = (self.safety_order_volume_scale.value * (math.pow(self.safety_order_volume_scale.value, (self.max_so_multiplier_orig.value - 1)) - 1))
                divisor = (self.safety_order_volume_scale.value - 1)
                max_so_multiplier = (2 + firstLine / divisor)
            elif (self.safety_order_volume_scale.value < 1):
                firstLine = self.safety_order_volume_scale.value * (1 - math.pow(self.safety_order_volume_scale.value, (self.max_so_multiplier_orig.value - 1)))
                divisor = 1 - self.safety_order_volume_scale.value
                max_so_multiplier = (2 + firstLine / divisor)
            else:
                return self.max_so_multiplier_orig.value
            return max_so_multiplier
        else:
            return self.max_so_multiplier_orig.value
            
    def order_filled(self, pair: str, trade: Trade, order, **kwargs) -> None:
        if self.dp.runmode.value in ("backtest", "hyperopt") and order.ft_order_side == "sell":
            current_date = trade.close_date.date() if trade.close_date else datetime.now().date()
            if current_date not in self.daily_trades_closed:
                self.daily_trades_closed[current_date] = []
            
            self.daily_trades_closed[current_date].append({
                'pair': pair,
                'profit_ratio': trade.close_profit if trade.close_profit else 0,
                'close_date': trade.close_date
            })
            
            if current_date not in self.daily_profit_tracker:
                self.daily_profit_tracker[current_date] = 0.0
            self.daily_profit_tracker[current_date] += trade.close_profit if trade.close_profit else 0
    
    
    class HyperOpt:
        def max_open_trades_space():
            return [Categorical([8, 12, 15, 20, 30], name='max_open_trades'),]

        def roi_space():
            return [
                Integer(2, 40, name='roi_t1'),
                Integer(2, 40, name='roi_t2'),
                Integer(2, 40, name='roi_t3'),
                Integer(2, 40, name='roi_t4'),
                SKDecimal(0.02, 1.5, decimals=3, name='roi_p1'),
                SKDecimal(0.02, 1.5, decimals=3, name='roi_p2'),
                SKDecimal(0.02, 1.5, decimals=3, name='roi_p3'),
                SKDecimal(-0.30, 0.70, decimals=3, name='roi_last'),
                SKDecimal(-0.30, 0.30, decimals=3, name='roi_last2'),
            ]

        def generate_roi_table(params: Dict) -> Dict[int, float]:
            tf_value = timeframe_to_minutes('1d')
            roi_table = {}
            roi_table[0] = params['roi_p1'] + params['roi_p2'] + params['roi_p3']
            roi_table[params['roi_t3']*tf_value] = params['roi_p1'] + params['roi_p2']
            roi_table[params['roi_t3']*tf_value + params['roi_t2']*tf_value] = params['roi_p1']
            roi_table[params['roi_t3']*tf_value + params['roi_t2']*tf_value + params['roi_t1']*tf_value] = params['roi_last']
            roi_table[params['roi_t4']*tf_value + params['roi_t3']*tf_value + params['roi_t2']*tf_value + params['roi_t1']*tf_value] = params['roi_last2']
            return roi_table

        def trailing_space():
            return [
                Categorical([True, False], name='trailing_stop'),
                SKDecimal(0.01, 0.35, decimals=2, name='trailing_stop_positive'),
                SKDecimal(0.001, 0.1, decimals=2, name='trailing_stop_positive_offset_p1'),
                Categorical([True, False], name='trailing_only_offset_is_reached'),
        ]
    
    order_types = {'entry': 'limit','exit': 'limit','stoploss': 'market','stoploss_on_exchange': False}
    order_time_in_force = {'entry': 'GTC', 'exit': 'GTC'}

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        if self.use_custom_leverage.value:
            return self.leverage_value.value
        return 1.0

    use_custom_leverage = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    leverage_value = CategoricalParameter([1, 2, 3], default=5, space="buy", optimize=True)

    use_min_volume_filter = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    min_volume_filter = CategoricalParameter([5_000, 10_000, 20_000, 50_000, 75_000, 100_000, 150_000, 200_000, 300_000, 400_000, 500_000, 750_000, 1_000_000], default=200000, space="buy", optimize=True)
    volume_rolling = CategoricalParameter([24, 48, 72, 96], default=96, space="buy", optimize=True)

    use_total2es_crash_exit = CategoricalParameter([True, False], default=True, space="sell", optimize=True)
    total2es_vwap_size = CategoricalParameter([2, 3, 7, 14, 21, 28, 35, 42, 49, 56], default=2, space="buy", optimize=True)
    total2es_vwap_band_std = DecimalParameter(1.0, 1.10, default=1.069, decimals=3, space="sell", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['volume_mean_rolling'] = dataframe['volume'].rolling(self.volume_rolling.value).mean()
        dataframe = add_total2es_with_vwap(dataframe, timeframe=self.timeframe, vwap_period=int(self.total2es_vwap_size.value), band_std=self.total2es_vwap_band_std.value)
        dataframe = add_walcl_complete(dataframe, backtest=True)
        dataframe['WALCL_rolling'] = dataframe['1w_WALCL'].rolling(self.walcl_size.value).mean()
        dataframe['WALCL_ema'] = dataframe['WALCL_rolling'].ewm(span=self.walcl_ema_size.value).mean()
        dataframe['trend_TRIX_130_20_2'] = TRIX(dataframe, length=130, power=20, MAtype=2)['TRIX_HISTO']
        # Fear & Greed Momentum Composite
        rsi = clean_ta.momentum.rsi(close=dataframe['close'], window=int(30))
        rsi_score = (rsi - 50) / 50 * 100
        atr = ta.ATR(dataframe, timeperiod=int(30))
        atr_pct = (atr / dataframe['close']) * 100
        atr_max = atr_pct.rolling(int(3)).max()
        atr_score = np.where(atr_max != 0, (1 - atr_pct / atr_max) * 100, 0)
        volume_sma = dataframe['volume'].rolling(int(30)).mean()
        volume_score = ((dataframe['volume'] / volume_sma) - 1) * 50
        ema_long = clean_ta.trend.ema_indicator(close=dataframe['close'], window=int(30 * 12))
        price_score = ((dataframe['close'] / ema_long) - 1) * 100
        weights = [0.3, 0.2, 0.2, 0.3]
        dataframe['over_FearGreedMomentum_30_3_12'] = (rsi_score * weights[0] + atr_score * weights[1] + volume_score * weights[2] + price_score * weights[3])
        dataframe['over_FearGreedMomentum_30_3_12_smooth'] = dataframe['over_FearGreedMomentum_30_3_12'].rolling(int(3)).mean()
        dataframe["volatility_VWAP_3_7_0.6"] = clean_ta.volume.VolumeWeightedAveragePrice(dataframe["high"], dataframe["low"], dataframe["close"], dataframe["volume"], window=int(1)).volume_weighted_average_price()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(((dataframe['trend_TRIX_130_20_2'] > 0) & (dataframe['trend_TRIX_130_20_2'] > dataframe['trend_TRIX_130_20_2'].shift(25))))
        conditions.append((dataframe['over_FearGreedMomentum_30_3_12'] > dataframe['over_FearGreedMomentum_30_3_12'].shift(25)))
        conditions.append((dataframe['volatility_VWAP_3_7_0.6'].shift(75) < dataframe['close']))
        conditions.append((dataframe['volume_mean_rolling'] > self.min_volume_filter.value))
        atr = AverageTrueRange(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], window=int(self.volatility_size.value)).average_true_range()
        atr_pct = atr / dataframe['close']
        conditions.append((atr_pct < 0.02))
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
            'enter_long'] = 1
            # Si cette option est activée, on ouvrira une position uniquement lorsque la paire renverra au moins X signaux d'ouvertures
            dataframe['enter_long'] = dataframe['enter_long'].rolling(window=3).min()
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        total2es_conditions = []
        if ('TOTAL2ES' in dataframe.columns and 'TOTAL2ES_lower_band' in dataframe.columns and
            dataframe['TOTAL2ES'].iloc[-1] > 0 and dataframe['TOTAL2ES_lower_band'].iloc[-1] > 0):
            cond_total2es = ((dataframe['TOTAL2ES'] < dataframe['TOTAL2ES_lower_band']) & 
                            (dataframe['TOTAL2ES'].shift(1) < dataframe['TOTAL2ES_lower_band'].shift(1)))
            total2es_conditions.append(cond_total2es)
        conditions.append(((dataframe['over_FearGreedMomentum_30_3_12'] > 20) & (dataframe['close'] < dataframe['close'].shift(40))))
        if conditions:
            if total2es_conditions:
                total2es_mask = reduce(lambda x, y: x | y, total2es_conditions)
                dataframe.loc[reduce(lambda x, y: x | y, conditions) | total2es_mask, 'exit_long'] = 1
            else:
                dataframe.loc[reduce(lambda x, y: x | y, conditions), 'exit_long'] = 1
        elif total2es_conditions:
            total2es_mask = reduce(lambda x, y: x | y, total2es_conditions)
            dataframe.loc[total2es_mask, 'exit_long'] = 1
        return dataframe

# Contenu du strategy_generator_json (pour comparer avec external_indicators_v*.py et strategies_writer_v*.py en cas de souci)
"""
strategy_generator_json = {
    "strategy_name": "strategies_generator_v13_long",
    "params": {
        "roi": {
            "0": 100000.0
        },
        "stoploss": {
            "stoploss": -1.0
        },
        "trailing": {
            "trailing_stop": false,
            "trailing_stop_positive": 0.0,
            "trailing_stop_positive_offset": 0.0,
            "trailing_only_offset_is_reached": false
        },
        "max_open_trades": {
            "max_open_trades": 10
        },
        "buy": {
            "filter_buy_conditions": 0,
            "filter_buy_value": "p5",
            "filter_indicator": "PCT_TOTAL2_MARKETCAP_1d",
            "filter_size1": "p5",
            "filter_size2": "p5",
            "filter_size3": "p5",
            "filter_use_buy": false,
            "momentum_buy_conditions": 0,
            "momentum_buy_value": "p5",
            "momentum_indicator": "RSI",
            "momentum_size1": "p5",
            "momentum_size2": "p5",
            "momentum_size3": "p5",
            "ref_rolling_size": 1,
            "ref_rolling_type": 1,
            "ref_use_comparaison": "none",
            "supports_buy_conditions": 0,
            "supports_buy_value": "p5",
            "supports_indicator": "SR_PEAKS_VOLUME",
            "supports_size1": "p5",
            "supports_size2": "p5",
            "supports_size3": "p5",
            "supports_use_buy": false,
            "adjust_need_entry_signal": true,
            "adjust_require_volume_spike": false,
            "cumulative_entry_call": 3,
            "drawdown_lookback": 20,
            "drawdown_reduction": 0.64,
            "drawdown_threshold": 0.24,
            "entry_mode": "none",
            "entry_scaling_mode": "equal",
            "entry_scaling_orders": 3,
            "exit_conditions_type": "OR",
            "initial_safety_order_trigger": -0.099,
            "leverage_value": 5,
            "max_daily_loss_pct": 0.039,
            "max_position_pct": 0.36,
            "max_so_multiplier_orig": 4,
            "min_volume_filter": 200000,
            "need_cumulative_entry_call": true,
            "over_buy_conditions": 2,
            "over_buy_value": "p9",
            "over_indicator": "FearGreedMomentum",
            "over_size1": "p6",
            "over_size2": "p1",
            "over_size3": "p13",
            "overbuy_factor": 0.9,
            "partial_fill_compensation_scale": 0.9,
            "rebalance_threshold": 0.39,
            "safety_order_step_scale": 4,
            "safety_order_volume_scale": 1.5,
            "time_based_dca_candles": 2,
            "total2es_vwap_size": 2,
            "tradable_balance_ratio": 1.0,
            "trading_time_mode": "always",
            "trailing_buy_max_distance": 0.037,
            "trailing_buy_max_hours": 4,
            "trailing_buy_pct": 0.043,
            "trend_buy_conditions": 1,
            "trend_buy_value": "p7",
            "trend_indicator": "TRIX",
            "trend_size1": "p14",
            "trend_size2": "p11",
            "trend_size3": "p4",
            "use_cumulative_end_entry": false,
            "use_custom_leverage": true,
            "use_custom_stake": false,
            "use_drawdown_sizing": false,
            "use_max_daily_loss": true,
            "use_min_volume_filter": true,
            "use_portfolio_rebalancing": true,
            "use_position_adjustment": true,
            "use_time_based_dca": false,
            "use_volatility_filter": true,
            "use_volatility_stake": true,
            "use_volume_stake": false,
            "use_walcl_stake": true,
            "volatility_buy_conditions": 7,
            "volatility_buy_value": "p10",
            "volatility_indicator": "VWAP",
            "volatility_max_filter_stake": 0.132,
            "volatility_size": 10,
            "volatility_size1": "p3",
            "volatility_size2": "p7",
            "volatility_size3": "p10",
            "volatility_size_stake": 14,
            "volatility_threshold": 0.02,
            "volume_rolling": 96,
            "volume_spike_threshold": 1.25,
            "volume_spike_window": 15,
            "volume_stake_max_ratio": 0.8,
            "volume_stake_max_threshold": 1000000,
            "volume_stake_min_ratio": 0.89,
            "volume_stake_min_threshold": 400000,
            "walcl_adjustment": 0.25,
            "walcl_ema_size": 126,
            "walcl_size": 100,
            "walcl_stake_mode": 1
        },
        "sell": {
            "filter_sell_conditions": 0,
            "filter_sell_value": "p5",
            "filter_use_sell": false,
            "momentum_sell_conditions": 0,
            "momentum_sell_value": "p5",
            "momentum_use_sell": false,
            "supports_sell_conditions": 0,
            "supports_sell_value": "p5",
            "supports_use_sell": false,
            "exit_only_profit": true,
            "max_hold_hours": 1250,
            "my_custom_stoploss": -0.11,
            "over_sell_conditions": 6,
            "over_sell_value": "p6",
            "over_use_sell": true,
            "parabolic_gain_threshold": 0.24,
            "parabolic_lookback": 7,
            "partial_profit_levels": 2,
            "partial_profit_pct_1": 0.147,
            "partial_profit_pct_2": 0.512,
            "partial_profit_pct_3": 0.537,
            "partial_profit_pct_4": 1.858,
            "profit_ratio_needed": 0.067,
            "total2es_vwap_band_std": 1.069,
            "trend_sell_conditions": 3,
            "trend_sell_value": "p6",
            "trend_use_sell": false,
            "use_parabolic_exit": false,
            "use_partial_profit": false,
            "use_total2es_crash_exit": true,
            "volatility_sell_conditions": 5,
            "volatility_sell_value": "p8",
            "volatility_use_sell": false
        },
        "protection": {}
    },
    "ft_stratparam_v": 1,
    "export_time": "2025-09-03 21:53:17.081793+00:00"
}
"""
