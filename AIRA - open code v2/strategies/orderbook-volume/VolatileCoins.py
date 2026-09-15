import logging
from freqtrade.strategy import IStrategy
from typing import Dict, List, Optional
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
from functools import reduce
from datetime import datetime
import pandas_ta as pta
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)

class VolatileCoins(IStrategy):
    timeframe = '5m'
    startup_candle_count = 100
    minimal_roi = {"0": 0.3}
    
    # Stoploss configuration
    stoploss = -0.3
    trailing_stop = True
    trailing_stop_positive = 0.18
    trailing_stop_positive_offset = 0.24
    trailing_only_offset_is_reached = True

    # Risk management
    position_adjustment_enable = True
    max_entry_position_adjustment = 3
    use_custom_stoploss = True
    use_exit_signal = True

    # Strategy parameters
    buy_params = {
        'volume_multiplier': 3.5,
        'rsi_buy': 32,
        'adx_threshold': 22,
        'vwap_deviation': 0.025,
        'atr_multiplier': 1.8,
        'max_listing_age': 72,
        'min_volume_growth': 1.8,
        'price_drop_threshold': 0.25,
        'recovery_factor': 0.4
    }

    sell_params = {
        'rsi_sell': 68,
        'trail_atr_multiplier': 3.2,
        'profit_threshold': 0.15
    }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.pattern_db = {}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Core indicators
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['bollinger_upper'] = ta.BBANDS(dataframe, timeperiod=20)['upperband']
        dataframe['bollinger_lower'] = ta.BBANDS(dataframe, timeperiod=20)['lowerband']
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = ta.MACD(dataframe)
        
        # Volume analysis
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma']
        
        # VWAP calculations
        vwap = qtpylib.rolling_vwap(dataframe, window=20)
        dataframe['vwap_deviation'] = (dataframe['close'] - vwap) / vwap
        
        # Historical pattern analysis
        hist_analysis = self.analyze_pair_history(dataframe)
        for key, value in hist_analysis.items():
            dataframe[key] = value
            
        # Time since listing calculations
        dataframe['listing_age'] = (len(dataframe) - dataframe.index) * 5 / 60  # Hours
        
        # Price deviation from listing price
        dataframe['first_close'] = dataframe['close'].iloc[0]
        dataframe['price_deviation'] = (dataframe['close'] - dataframe['first_close']) / dataframe['first_close']
        
        # Data quality checks
        dataframe['clean_data'] = (
            (dataframe['volume'] > 0) &
            (dataframe['high'] != dataframe['low']) &
            (dataframe['close'].pct_change().abs() < 0.5)
        )
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        
        # Historical pattern conditions
        conditions.append(
            (dataframe['listing_age'] < self.buy_params['max_listing_age']) &
            (dataframe['volume_growth'] > self.buy_params['min_volume_growth']) &
            (dataframe['price_deviation'] < -self.buy_params['price_drop_threshold']) &
            (dataframe['price_deviation'] > -self.buy_params['price_drop_threshold'] * self.buy_params['recovery_factor']) &
            (dataframe['hist_volatility'] > 2.0) &
            (dataframe['consolidation'])
        )

        # Technical indicators
        conditions.append(
            (dataframe['volume_ratio'] > self.buy_params['volume_multiplier']) &
            (dataframe['close'] < dataframe['bollinger_upper']) &
            (dataframe['vwap_deviation'].abs() > self.buy_params['vwap_deviation']) &
            (dataframe['adx'] > self.buy_params['adx_threshold']) &
            (dataframe['rsi'] < self.buy_params['rsi_buy']) &
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['clean_data'])
        )

        # Log conditions for debugging
        if len(dataframe) > 0:
            logger.info(f"Conditions for {metadata['pair']}:")
            logger.info(f"Volume ratio: {dataframe['volume_ratio'].iloc[-1]}")
            logger.info(f"RSI: {dataframe['rsi'].iloc[-1]}")
            logger.info(f"ADX: {dataframe['adx'].iloc[-1]}")
            logger.info(f"MACD: {dataframe['macd'].iloc[-1]}")
            logger.info(f"VWAP deviation: {dataframe['vwap_deviation'].iloc[-1]}")
            logger.info(f"Price deviation: {dataframe['price_deviation'].iloc[-1]}")

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        
        # Technical exit conditions
        conditions.append(
            (dataframe['rsi'] > self.sell_params['rsi_sell']) &
            (dataframe['macd'] < dataframe['macdsignal'])
        )

        # Profit target condition
        conditions.append(
            (dataframe['close'] >= (1 + self.sell_params['profit_threshold']) * dataframe['open'])
        )

        # Trailing stop calculation
        dataframe['trailing_stop'] = dataframe['close'] * (1 - (dataframe['atr'] * self.sell_params['trail_atr_multiplier'] / dataframe['close']))
        
        dataframe.loc[
            reduce(lambda x, y: x | y, conditions),
            'exit_long'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Dynamic stoploss calculation
        base_sl = last_candle['hist_volatility'] * 0.5
        if last_candle['recent_recovery']:
            base_sl *= 0.8
        if last_candle['consolidation']:
            base_sl *= 1.2
            
        return min(-0.15, -abs(base_sl))

    def notify_trade(self, trade: Trade) -> None:
        """Store successful trade patterns"""
        if trade.is_successful:
            pair_data = self.dp.get_pair_dataframe(trade.pair, self.timeframe)
            analysis = self.analyze_pair_history(pair_data)
            self.pattern_db[trade.pair] = {
                'entry_candle': trade.entry_data[0],
                'patterns': analysis
            }

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration": 3
            },
            {
                "method": "MaxDrawdown",
                "lookback_period": 24,
                "trade_limit": 4,
                "stop_duration": 12,
                "max_allowed_drawdown": 0.25
            },
            {
                "method": "StoplossGuard",
                "lookback_period": 24,
                "trade_limit": 2,
                "stop_duration": 6,
                "only_per_pair": True
            }
        ]