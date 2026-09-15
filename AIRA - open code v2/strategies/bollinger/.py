from freqtrade.strategy import IStrategy
import talib.abstract as ta
import datetime
from pandas import DataFrame, Series
from typing import Optional

class AdvancedMultiIndicatorStrategy(IStrategy):
    # 优化后的最小回报率
    minimal_roi = {
        "0": 0.15,
        "10": 0.1,
        "30": 0
    }

    stoploss = -0.08
    timeframe = '5m'
    startup_candle_count = 50
    can_short = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # Calculate MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        
        # Calculate EMA
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        
        # Calculate Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_lower'] = bollinger['lowerband']
        
        # Calculate ATR for volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # Calculate Stochastic Oscillator
        stoch = ta.STOCH(dataframe)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']
        
        # Calculate ADX (Average Directional Index)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        
        # Calculate Money Flow Index (MFI)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < 35) &  # RSI 超卖
                (dataframe['macd'] > dataframe['macd_signal']) &  # MACD 金叉
                (dataframe['stoch_k'] < 20) &  # Stochastic 超卖
                (dataframe['close'] < dataframe['bb_lower']) &  # 价格低于布林带下轨
                (dataframe['close'] > dataframe['ema_20']) &  # 价格高于 EMA 20
                (dataframe['adx'] > 25) &  # ADX 表示强趋势
                (dataframe['mfi'] < 20) &  # MFI 显示资金流入减少
                (dataframe['atr'] > 1.5 * dataframe['atr'].rolling(window=14).mean())  # 波动率增加
            ),
            'enter_long'] = 1
        dataframe.loc[
            (
                (dataframe['rsi'] > 65) &  # RSI 超买
                (dataframe['macd'] < dataframe['macd_signal']) &  # MACD 死叉
                (dataframe['stoch_k'] > 80) &  # Stochastic 超买
                (dataframe['close'] > dataframe['bb_upper']) &  # 价格高于布林带上轨
                (dataframe['adx'] > 25) &  # ADX 表示强趋势
                (dataframe['mfi'] > 80) &  # MFI 显示资金流入增加
                (dataframe['atr'] < dataframe['atr'].rolling(window=14).mean())  # 波动率回落
            ),
            'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 65) &  # RSI 超买
                (dataframe['macd'] < dataframe['macd_signal']) &  # MACD 死叉
                (dataframe['stoch_k'] > 80) &  # Stochastic 超买
                (dataframe['close'] > dataframe['bb_upper']) &  # 价格高于布林带上轨
                (dataframe['adx'] > 25) &  # ADX 表示强趋势
                (dataframe['mfi'] > 80) &  # MFI 显示资金流入增加
                (dataframe['atr'] < dataframe['atr'].rolling(window=14).mean())  # 波动率回落
            ),
            'exit_long'] = 1
        dataframe.loc[
            (
                (dataframe['rsi'] < 35) &  # RSI 超卖
                (dataframe['macd'] > dataframe['macd_signal']) &  # MACD 金叉
                (dataframe['stoch_k'] < 20) &  # Stochastic 超卖
                (dataframe['close'] < dataframe['bb_lower']) &  # 价格低于布林带下轨
                (dataframe['close'] > dataframe['ema_20']) &  # 价格高于 EMA 20
                (dataframe['adx'] > 25) &  # ADX 表示强趋势
                (dataframe['mfi'] < 20) &  # MFI 显示资金流入减少
                (dataframe['atr'] > 1.5 * dataframe['atr'].rolling(window=14).mean())  # 波动率增加
            ),
            'exit_short'] = 1
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return 3

