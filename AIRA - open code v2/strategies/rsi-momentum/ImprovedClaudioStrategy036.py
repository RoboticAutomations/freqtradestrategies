from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timezone
from freqtrade.persistence import Trade

class ImprovedClaudioStrategy036(IStrategy):
    INTERFACE_VERSION = 3
    
    # Strategy settings
    minimal_roi = {
        "0": 0.05  # 5% profit
    }
    
    stoploss = -0.20  # 20% stop loss
    
    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.0005 
    trailing_stop_positive_offset = 0.015 # 1.5%
    trailing_only_offset_is_reached = True
    
    # Timeframe settings
    timeframe = '1h'
    
    # Buy hyperparameters
    buy_params = {
        "bb_length": 20,
        "bb_mult": 2.0,
        "kc_length": 20,
        "kc_mult": 1.5,
    }

    # DCA Parameters - DCA Simples com valores fixos
    max_dca_orders = 6  # 5 ordens DCA
    
    # Multiplicadores fixos para garantir valores mínimos executáveis
    dca_multipliers = {
        1: 0.35,    # €5 (primeira DCA)
        2: 0.67,    # €10 (segunda DCA) 
        3: 1.00,    # €15 (terceira DCA - igual ao inicial)
        4: 1.33,    # €20 (quarta DCA)
        5: 1.67,     # €25 (quinta DCA)
	6: 1.67     # 25 euros para o ultimo DCA 
    }

    # Thresholds para DCA Simples - mais espaçados
    dca_thresholds = {
        1: -0.05,   # -5% para 1ª DCA (€5)
        2: -0.10,   # -10% para 2ª DCA (€10)
        3: -0.18,   # -18% para 3ª DCA (€15)
        4: -0.28,   # -28% para 4ª DCA (€20)
        5: -0.40,   # -40% para 5ª DCA (€25)
	6: -0.50    # -60% para o 6 e ultimo DCA
    }


    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                            current_rate: float, current_profit: float,
                            min_stake: float, max_stake: float,
                            current_entry_profit: float, current_exit_profit: float,
                            **kwargs) -> Optional[float]:
        
        # Verifica se já atingiu o número máximo de ordens DCA
        current_order_number = trade.nr_of_successful_entries
        if current_order_number >= self.max_dca_orders:
            return None

        # Verifica se atingiu o threshold para DCA
        threshold = self.dca_thresholds.get(current_order_number, -1.0)
        if current_profit <= threshold:
            # Calcula o novo valor de entrada
            multiplier = self.dca_multipliers.get(current_order_number, 1.0)
            new_stake = trade.stake_amount * multiplier

            # Verifica se está dentro dos limites
            if new_stake <= max_stake :
                return new_stake

        return None

    # [Resto do código permanece igual...]
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # BB Calculation
        bb_length = self.buy_params['bb_length']
        bb_mult = self.buy_params['bb_mult']
        
        dataframe['bb_basis'] = ta.SMA(dataframe['close'], timeperiod=bb_length)
        dataframe['bb_dev'] = bb_mult * ta.STDDEV(dataframe['close'], timeperiod=bb_length)
        dataframe['bb_upper'] = dataframe['bb_basis'] + dataframe['bb_dev']
        dataframe['bb_lower'] = dataframe['bb_basis'] - dataframe['bb_dev']
        
        # KC Calculation
        kc_length = self.buy_params['kc_length']
        kc_mult = self.buy_params['kc_mult']
        
        dataframe['kc_ma'] = ta.SMA(dataframe['close'], timeperiod=kc_length)
        dataframe['tr'] = ta.TRANGE(dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['kc_range'] = ta.SMA(dataframe['tr'], timeperiod=kc_length)
        dataframe['kc_upper'] = dataframe['kc_ma'] + dataframe['kc_range'] * kc_mult
        dataframe['kc_lower'] = dataframe['kc_ma'] - dataframe['kc_range'] * kc_mult
        
        # Squeeze Conditions
        dataframe['sqz_on'] = (dataframe['bb_lower'] > dataframe['kc_lower']) & \
                             (dataframe['bb_upper'] < dataframe['kc_upper'])
        dataframe['sqz_off'] = (dataframe['bb_lower'] < dataframe['kc_lower']) & \
                              (dataframe['bb_upper'] > dataframe['kc_upper'])
        dataframe['no_sqz'] = ~(dataframe['sqz_on'] | dataframe['sqz_off'])
        
        # Momentum Value calculation
        highest_high = dataframe['high'].rolling(window=kc_length).max()
        lowest_low = dataframe['low'].rolling(window=kc_length).min()
        avg_hl = (highest_high + lowest_low) / 2
        avg_close = ta.SMA(dataframe['close'], timeperiod=kc_length)
        final_avg = (avg_hl + avg_close) / 2
        
        # Linear regression calculation
        source = dataframe['close'] - final_avg
        dataframe['momentum'] = ta.LINEARREG(source, timeperiod=kc_length)
        
        # Previous momentum for comparison
        dataframe['momentum_prev'] = dataframe['momentum'].shift(1)
        
        return dataframe
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['momentum'] > 0) &  # Positive momentum
                (dataframe['momentum'] > dataframe['momentum_prev']) &  # Increasing momentum
                (dataframe['sqz_off'])  # Squeeze is off (expansion)
            ),
            'buy'] = 1
        return dataframe
    
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['momentum'] < 0) &  # Negative momentum
                (dataframe['momentum'] < dataframe['momentum_prev'])  # Decreasing momentum
            ),
            'sell'] = 1
        return dataframe
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, sell_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """
        Override to prevent selling when in negative profit
        """
        current_profit = trade.calc_profit_ratio(rate)
        
        # Don't sell if we're in a loss
        if current_profit < 0:
            return False
            
        return True
