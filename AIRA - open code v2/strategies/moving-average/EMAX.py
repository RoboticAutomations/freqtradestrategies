from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class EMAX(IStrategy):
    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.1,
        "5": 0.07,
        "10": 0.05,
        "15": 0.04,
        "20": 0.03,
        "25": 0.02,
        "30": 0.01,
        "50": 0
    }

    # Optimal stoploss
    stoploss = -0.3

    # Trailing stoploss settings
    trailing_stop = True
    trailing_stop_positive = 0.01  # 2% profit to start trailing
    trailing_stop_positive_offset = 0.05  # 3% offset for trailing stop

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # Define the parameters for the strategy
    ema_length = 15
    lsma_offset = 5
    rsi_length = 2  # Increased to 14 for better accuracy
    rsi_overbought = 80
    rsi_oversold = 20
    adx_length = 5  # Increased ADX length for stability
    adx_threshold = 30  # ADX threshold to determine trend readiness

    # Specify how many startup candles are needed to avoid NaN errors
    startup_candles = 50  # Make sure enough data is loaded

    def informative_pairs(self):
        return []

    def dirmov(self, dataframe, len):
        # 计算 +DI 和 -DI
        up = dataframe['high'].diff()
        down = -dataframe['low'].diff()
        plusDM = (up > down) & (up > 0)  # Positive Directional Movement
        minusDM = (down > up) & (down > 0)  # Negative Directional Movement
        truerange = ta.TRANGE(dataframe)  # True Range

        # 计算 +DI 和 -DI
        plus = ta.EMA(plusDM.astype(float), len) / ta.EMA(truerange, len) * 100
        minus = ta.EMA(minusDM.astype(float), len) / ta.EMA(truerange, len) * 100

        return plus, minus

    def adx(self, dataframe, len):
        plus, minus = self.dirmov(dataframe, len)
        # 计算 ADX
        adx = ta.EMA(abs(plus - minus) / (plus + minus), len) * 100
        return adx

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Ensure that we have enough data points to calculate the indicators
        if len(dataframe) < self.startup_candles:
            return dataframe  # If there isn't enough data, skip calculation

        dataframe = dataframe.fillna(method='ffill')  # Forward-fill missing data
  
        # Calculate ZLSMA (Zero Lag Linear Regression)
        lsma = ta.LINEARREG(dataframe['close'], timeperiod=self.ema_length)
        lsma2 = ta.LINEARREG(lsma, timeperiod=self.ema_length)
        eq = lsma - lsma2
        dataframe['zlsma'] = lsma + eq  # Zero Lag Linear Regression

        # Calculate RSI
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.rsi_length)

        # Calculate ADX
        dataframe['adx'] = self.adx(dataframe, self.adx_length)

        return dataframe


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 添加入场信号：检查入场条件并标记 'enter_long' 标签
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['zlsma']) &  # Close above ZLSMA
                (dataframe['rsi'] <= self.rsi_oversold) &  # RSI below oversold threshold
                (dataframe['adx'] > self.adx_threshold) &  # ADX above threshold
                (dataframe['close'] > dataframe['high'].shift(1))  # Close above the previous high
            ),
            'enter_long'] = 1  # Mark entry for long position
        
        # 仅在产生信号时添加标签
        dataframe.loc[
            dataframe['enter_long'] == 1,
            'entry_label'] = 'Long Signal'  # 为入场信号添加标签
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 添加出场信号：检查出场条件并标记 'exit_long' 标签
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['zlsma']) &  # Close below ZLSMA
                (dataframe['rsi'] >= self.rsi_overbought) &  # RSI above overbought threshold
                (dataframe['adx'] > self.adx_threshold) &  # ADX above threshold
                (dataframe['close'] < dataframe['low'].shift(1))  # Close below the previous low
            ),
            'exit_long'] = 1  # Mark exit for long position
        
        # 仅在产生信号时添加标签
        dataframe.loc[
            dataframe['exit_long'] == 1,
            'exit_label'] = 'Exit Signal'  # 为出场信号添加标签
        
        return dataframe
