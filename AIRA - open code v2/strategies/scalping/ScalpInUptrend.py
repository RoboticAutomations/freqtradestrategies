from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
import pandas as pd
import talib.abstract as ta

class ScalpInUptrend(IStrategy):
    # Strategy parameters
    timeframe = '5m'
    minimal_roi = {
        "0": 0.01,  # 1% profit
    }
    stoploss = -0.05  # Fixed stop loss of 0.5%
    startup_candle_count = 200  # Skip initial candles where indicators aren't fully computed

    # Define parameters for optimization
    rsi_overbought = IntParameter(60, 80, default=65, space='buy')
    adx_threshold = IntParameter(10, 45, default=25, space='buy')
    profit_target = DecimalParameter(0.01, 0.10, default=0.01, space='sell')

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Calculate indicators
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['sma50'] = ta.SMA(dataframe['close'], timeperiod=50)
        dataframe['sma200'] = ta.SMA(dataframe['close'], timeperiod=200)
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Define buy conditions
        buy_conditions = (
            (dataframe['sma50'] > dataframe['sma200']) &
            (dataframe['adx'] > self.adx_threshold.value) &
            (dataframe['rsi'] < self.rsi_overbought.value)
        )
        dataframe.loc[buy_conditions, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Define sell conditions (take profit)
        take_profit_condition = (
            (dataframe['close'] >= dataframe['open'] * (1 + self.profit_target.value))
        )
        dataframe.loc[take_profit_condition, 'exit_long'] = 1
        return dataframe