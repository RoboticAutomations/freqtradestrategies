# ======================================================
# BEST PROFITABLE STRATEGY WITH EXTENDED HYPEROPTS
#    - LONG/SHORT
#    - TRAILING STOP
#    - LEVERAGE
#    - EXPERIMENTAL TIMEFRAME
#    - CORRECTED confirm_trade_exit()
# ======================================================
#
# 1) SMA/RSI for momentum and trend identification.
# 2) Hyperparameters for trailing stops, leverage, timeframe (conceptual),
#    and other indicators (SMA lengths, RSI thresholds).
# 3) can_short = True to allow short trades (ensure your exchange supports it).
# 4) Corrected confirm_trade_exit() method to avoid multiple 'exit_reason' errors.
#
# Save this file as BestProfitableStrategyHyperoptExtended.py
# in your user_data/strategies folder.
#
# Example hyperopt usage:
#   freqtrade hyperopt \
#       --strategy BestProfitableStrategyHyperoptExtended \
#       --timeframe 5m \
#       --hyperopt-loss ShortHyperOptLossDaily \
#       --spaces all
#
# ======================================================

import logging
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter, CategoricalParameter
from freqtrade.persistence import Trade
from pandas import DataFrame
import talib.abstract as ta

logger = logging.getLogger(__name__)

class Best5m(IStrategy):
    """
    Demonstrates a 5-minute (by default) strategy using two SMAs and RSI for entries
    (long or short) and exits. Hyperparameters control trailing stops, leverage, 
    and an experimental timeframe switch.

    NOTE ON TIMEFRAME:
    Freqtrade does not dynamically change timeframes within a single run by default. 
    The 'timeframe_opt' parameter is shown for conceptual or advanced usage. 
    In standard usage, set your timeframe via CLI/config (e.g., '--timeframe 5m').

    MAKE SURE:
    1) 'can_short' is True, and your exchange/config supports margin/futures for shorting.
    2) You update your config accordingly (e.g., margin_mode, can_short settings).
    3) You run thorough backtesting and (ideally) separate forward-testing 
       or paper trading before going live.
    """

    # -------------------------------------------------------------------------
    # 1) Timeframe Hyperparameter (Experimental)
    # -------------------------------------------------------------------------
    # Freqtrade normally uses a single timeframe set via CLI/config.
    # We'll set up a hyperparameter for demonstration purposes.
    timeframe_opt = CategoricalParameter(
        ['1m', '3m', '5m', '15m', '1h'],
        default='5m',
        space='buy',
        optimize=True
    )

    # By default, set the strategy timeframe to 5m.
    timeframe = '5m'

    # -------------------------------------------------------------------------
    # 2) Basic Settings
    # -------------------------------------------------------------------------
    can_short = True
    stoploss = -0.99  # 99% stoploss
    minimal_roi = {
        "0": 0.01 # 1% ROI
    }
    exit_profit_only = True  # Only exit when in profit
    exit_profit_offset = 0.0  # Offset for profit-only exits

    # -------------------------------------------------------------------------
    # 3) Trailing Stop Settings
    # -------------------------------------------------------------------------
    trailing_stop = False  # This should be a direct boolean
    trailing_stop_positive = 0.02  # This should be a direct number
    trailing_stop_positive_offset = 0.03  # This should be a direct number
    trailing_only_offset_is_reached = False  # This should be a direct boolean

    # Store the parameters for hyperopt but don't use them directly in the above settings
    trailing_stop_opt = CategoricalParameter([True, False], default=False, space='buy', optimize=True)
    trailing_stop_positive_opt = DecimalParameter(0.01, 0.10, default=0.02, decimals=3, space='buy', optimize=True)
    trailing_stop_positive_offset_opt = DecimalParameter(0.01, 0.15, default=0.03, decimals=3, space='buy', optimize=True)
    trailing_only_offset_is_reached_opt = CategoricalParameter([True, False], default=False, space='buy', optimize=True)

    # -------------------------------------------------------------------------
    # 4) Leverage Hyperparameter
    # -------------------------------------------------------------------------
    leverage_opt = IntParameter(low=1, high=10, default=1, space='buy', optimize=True)

    # -------------------------------------------------------------------------
    # 5) Other Technical Hyperparameters (SMA, RSI)
    # -------------------------------------------------------------------------
    fast_ma_length = IntParameter(low=5, high=30, default=10, space='buy', optimize=True)
    slow_ma_length = IntParameter(low=31, high=100, default=30, space='buy', optimize=True)
    rsi_length = IntParameter(low=7, high=30, default=14, space='buy', optimize=True)

    # RSI thresholds for oversold (long) and overbought (short)
    rsi_buy_threshold = IntParameter(low=10, high=50, default=30, space='buy', optimize=True)
    rsi_sell_threshold = IntParameter(low=50, high=90, default=70, space='sell', optimize=True)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        """
        Attempt to override the timeframe with timeframe_opt. 
        In most cases, Freqtrade loads a single timeframe from your CLI/config. 
        This assignment might only work if you have a custom logic or tool that 
        reloads the strategy per timeframe. 
        """
        self.timeframe = self.timeframe_opt.value
        logger.info(f"Using timeframe: {self.timeframe}")

    # -------------------------------------------------------------------------
    # 6) Leverage Method (for Margin / Futures)
    # -------------------------------------------------------------------------
    def leverage(
        self, pair: str, current_time, current_rate: float, proposed_leverage: float, max_leverage: float,
        side: str, **kwargs
    ) -> float:
        """
        Dynamically determine leverage for each trade. In margin/futures, 
        ensure you do not exceed the exchange's max leverage for the pair.
        """
        selected_leverage = min(self.leverage_opt.value, max_leverage)
        return float(selected_leverage)

    # -------------------------------------------------------------------------
    # 7) Populate Indicators
    # -------------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate technical indicators: SMA (fast/slow), RSI.
        """
        dataframe['fast_sma'] = ta.SMA(dataframe, timeperiod=int(self.fast_ma_length.value))
        dataframe['slow_sma'] = ta.SMA(dataframe, timeperiod=int(self.slow_ma_length.value))
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=int(self.rsi_length.value))

        return dataframe

    # -------------------------------------------------------------------------
    # 8) Entry Signal Logic (Long / Short)
    # -------------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Determine when to open a long or short position based on SMA and RSI signals.
        """
        long_conditions = (
            (dataframe['fast_sma'] > dataframe['slow_sma']) &
            (dataframe['rsi'] < self.rsi_buy_threshold.value)
        )

        short_conditions = (
            (dataframe['fast_sma'] < dataframe['slow_sma']) &
            (dataframe['rsi'] > self.rsi_sell_threshold.value)
        )

        dataframe.loc[long_conditions, 'enter_long'] = 1
        dataframe.loc[short_conditions, 'enter_short'] = 1

        return dataframe

    # -------------------------------------------------------------------------
    # 9) Exit Signal Logic (Sell / Cover)
    # -------------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Determine when to exit a long or short position based on reversed conditions.
        """
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        long_exit_conditions = (
            (dataframe['fast_sma'] < dataframe['slow_sma']) &
            (dataframe['rsi'] > self.rsi_sell_threshold.value)
        )

        short_exit_conditions = (
            (dataframe['fast_sma'] > dataframe['slow_sma']) &
            (dataframe['rsi'] < self.rsi_buy_threshold.value)
        )

        dataframe.loc[long_exit_conditions, 'exit_long'] = 1
        dataframe.loc[short_exit_conditions, 'exit_short'] = 1

        return dataframe

    # -------------------------------------------------------------------------
    # 10) confirm_trade_exit (Corrected to avoid exit_reason conflict)
    # -------------------------------------------------------------------------
    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        **kwargs
    ) -> bool:
        """
        Confirm the exit before placing the order. 
        Overridden to avoid duplicate 'exit_reason' errors.
        """
        # Optionally add custom checks here.
        # Example: If you wanted to disallow exit under certain conditions,
        # you'd return False. For standard usage, just call the parent method.
        return super().confirm_trade_exit(
            pair=pair,
            trade=trade,
            order_type=order_type,
            amount=amount,
            rate=rate,
            time_in_force=time_in_force,
            exit_reason=exit_reason,
            **kwargs
        )

    # -------------------------------------------------------------------------
    # 11) custom_stoploss (Optional Trailing Logic)
    # -------------------------------------------------------------------------
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> float:
        """
        Customize the stoploss if trailing_stop_opt is enabled. Otherwise, use a fixed stop.
        Returning 1 indicates no immediate stoploss override (Freqtrade will handle trailing).
        """
        if self.trailing_stop_opt.value:
            # Defer to built-in trailing stop parameters.
            return 1
        else:
            # Use a fixed stoploss of -0.02.
            return float(self.stoploss)

    # -------------------------------------------------------------------------
    # 12) bot_start (Optional Logging)
    # -------------------------------------------------------------------------
    def bot_start(self, **kwargs) -> None:
        """
        Called at bot start. Use this to assign the hyperopt values to the actual parameters.
        """
        super().bot_start(**kwargs)
        
        # Assign hyperopt values to actual trailing stop parameters
        self.trailing_stop = self.trailing_stop_opt.value
        self.trailing_stop_positive = self.trailing_stop_positive_opt.value
        self.trailing_stop_positive_offset = self.trailing_stop_positive_offset_opt.value
        self.trailing_only_offset_is_reached = self.trailing_only_offset_is_reached_opt.value

        # Log parameters
        logger.info("Final Strategy Parameters:")
        logger.info(f"  timeframe            = {self.timeframe}")
        logger.info(f"  trailing_stop        = {self.trailing_stop}")
        if self.trailing_stop:
            logger.info(f"    trailing_stop_positive        = {self.trailing_stop_positive}")
            logger.info(f"    trailing_stop_positive_offset = {self.trailing_stop_positive_offset}")
            logger.info(f"    trailing_only_offset_is_reached = {self.trailing_only_offset_is_reached}")
        logger.info(f"  leverage            = {self.leverage_opt.value}")
        logger.info(f"  fast_ma_length      = {self.fast_ma_length.value}")
        logger.info(f"  slow_ma_length      = {self.slow_ma_length.value}")
        logger.info(f"  rsi_length          = {self.rsi_length.value}")
        logger.info(f"  rsi_buy_threshold   = {self.rsi_buy_threshold.value}")
        logger.info(f"  rsi_sell_threshold  = {self.rsi_sell_threshold.value}")
