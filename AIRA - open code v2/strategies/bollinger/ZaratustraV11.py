# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
from datetime import datetime
from pandas import DataFrame
from freqtrade.strategy import IStrategy, informative, IntParameter
import talib.abstract as ta
from technical import qtpylib



class ZaratustraV11(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = True
    use_exit_signal = False
    exit_profit_only = True
    exit_profit_offset = 0.5
    
    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 1.0
    }

    # Stoploss:
    stoploss = -0.3

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = True

    # Max Open Trades:
    max_open_trades = 10

    def leverage(self, pair: str, current_time: "datetime", current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return 10
    
    @informative('5m')
    @informative('15m')
    @informative('30m')
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['pdi'] = ta.PLUS_DI(dataframe)
        dataframe['mdi'] = ta.MINUS_DI(dataframe)
        dataframe[['bbl', 'bbm', 'bbu']] = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)[['lower', 'mid', 'upper']]
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['close_30m'] > dataframe['bbm_30m']) &
                (dataframe['close_15m'] > dataframe['bbm_15m']) &
                (dataframe['close_5m']  > dataframe['bbm_5m']) &

                (dataframe['adx_30m'] > dataframe['mdi_30m']) &
                (dataframe['adx_15m'] > dataframe['mdi_15m']) &
                (dataframe['adx_5m']  > dataframe['mdi_5m']) &

                (dataframe['pdi_30m'] > dataframe['mdi_30m']) &
                (dataframe['pdi_15m'] > dataframe['mdi_15m']) &
                (dataframe['pdi_5m']  > dataframe['mdi_5m'])
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Bullish trend')

        dataframe.loc[
            (
                (dataframe['close_30m'] < dataframe['bbm_30m']) &
                (dataframe['close_15m'] < dataframe['bbm_15m']) &
                (dataframe['close_5m']  < dataframe['bbm_5m']) &

                (dataframe['adx_30m'] > dataframe['pdi_30m']) &
                (dataframe['adx_15m'] > dataframe['pdi_15m']) &
                (dataframe['adx_5m']  > dataframe['pdi_5m']) &

                (dataframe['mdi_30m'] > dataframe['pdi_30m']) &
                (dataframe['mdi_15m'] > dataframe['pdi_15m']) &
                (dataframe['mdi_5m']  > dataframe['pdi_5m'])
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Bearish trend')
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe