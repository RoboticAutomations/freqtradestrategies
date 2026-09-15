import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
from pandas import DataFrame
import pandas as pd
from datetime import datetime, timedelta
from CompleteOrderStateManager import TradingState, CompleteOrderStateManager
from freqtrade.strategy.informative_decorator import informative

logger = logging.getLogger(__name__)


class ComprehensiveDeepSeekStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short = True

    trailing_stop = True
    trailing_stop_positive = 0.1
    trailing_stop_positive_offset = 0.2
    trailing_only_offset_is_reached = True

    position_adjustment_enable = True

    timeframe = '5m'
    startup_candle_count = 500
    stoploss = -0.99  # Will be overridden by custom stoploss

    # Enable custom stoploss and take profit
    use_custom_stoploss = True

    leverage_value = 20

    # Order types - Set entry to limit orders
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
    }

    # Order time in force
    order_time_in_force = {
        'entry': 'gtc',  # Good Till Canceled
        'exit': 'gtc',
    }

    @informative('15m')
    @informative('1h')
    @informative('4h')
    def populate_indicators_informative(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # Initialize components
        self.order_manager = CompleteOrderStateManager()

        # Tracking
        self.comprehensive_decisions = {}
        self.last_check_time = {}
        self.active_signals = {}  # Track active signals with TP/SL

        # Configuration
        self.check_interval = 900  # 15 minutes
        self.signal_timeout = 3600  # 1 hour

        # Configure timeout from config or use defaults
        self.batch_timeout = config.get('batch_timeout', 30)  # seconds
        self.max_cache_size = config.get('max_cache_size', 10)  # pairs
        self.last_batch_process_time = None

        self.executor = ThreadPoolExecutor(max_workers=10)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Comprehensive indicator population"""

        self._store_pair_data(metadata['pair'], dataframe, metadata)
        return dataframe

    def determine_current_state(self, pair: str) -> TradingState:
        """Determine current trading state"""

        # Check for active positions
        open_trades = Trade.get_open_trades()
        if any(trade.pair == pair for trade in open_trades):
            return TradingState.ACTIVE_POSITION

        # Check for pending orders (simplified - would need exchange API)
        # This is a placeholder - implement based on your exchange

        # Check for last signal
        last_signal = self.order_manager.last_signals.get(pair)
        if last_signal:
            signal_age = (datetime.now() - last_signal['detected_time']).total_seconds()
            if signal_age > self.signal_timeout:
                return TradingState.SIGNAL_EXPIRED
            elif not last_signal['used']:
                return TradingState.PENDING_LIMIT

        return TradingState.NO_SIGNAL

    def prepare_market_data(self, dataframe: DataFrame, pair: str, current_price: float, timeframes: list = None):
        """Prepare market data for analysis with multiple timeframes"""

        if timeframes is None:
            timeframes = ['15m', '1h', '4h']

        market_data = {
            'pair': pair,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat(),
            'timeframes': {}
        }

        # Add base timeframe data
        market_data['timeframes'][self.timeframe] = {
            'ohlcv_data': dataframe[['open', 'high', 'low', 'close', 'volume']].tail(200).to_dict('records'),
            'volume_avg': dataframe['volume'].tail(200).mean(),
            'price_change_24h': (current_price - dataframe['close'].iloc[-24]) / dataframe['close'].iloc[-24] * 100
        }

        # Add informative timeframe data
        for tf in timeframes:
            try:
                if self.dp:
                    inf_dataframe = self.dp.get_pair_dataframe(pair=pair, timeframe=tf)
                    if not inf_dataframe.empty:
                        market_data['timeframes'][tf] = {
                            'ohlcv_data': inf_dataframe[['open', 'high', 'low', 'close', 'volume']].tail(200).to_dict(
                                'records'),
                            'volume_avg': inf_dataframe['volume'].tail(200).mean(),
                            'current_price': inf_dataframe['close'].iloc[-1]
                        }
            except Exception as e:
                print(f"Error getting {tf} data for {pair}: {e}")

        return market_data

    # def populate_decision_indicators(self, dataframe: DataFrame, pair: str, current_state: TradingState):
    #     """Populate indicators based on decision"""
    #
    #     # Initialize indicators
    #     dataframe['trading_state'] = current_state.value
    #     dataframe['decision_action'] = 0
    #     dataframe['decision_urgency'] = 0
    #     dataframe['has_new_signal'] = 0
    #
    #     if pair in self.comprehensive_decisions:
    #         decision = self.comprehensive_decisions[pair]
    #
    #         # Map decisions to numeric values
    #         action_map = {
    #             'WAIT': 0,
    #             'ENTER_NEW': 1,
    #             'CANCEL': 2,
    #             'KEEP_WAITING': 3,
    #             'HOLD': 4,
    #             'CLOSE_NOW': 5,
    #             'PARTIAL_EXIT': 6
    #         }
    #
    #         urgency_map = {
    #             'LOW': 1,
    #             'MEDIUM': 2,
    #             'HIGH': 3
    #         }
    #
    #         dataframe.loc[dataframe.index[-1], 'decision_action'] = action_map.get(decision['decision'], 0)
    #         dataframe.loc[dataframe.index[-1], 'decision_urgency'] = urgency_map.get(decision['urgency'], 1)
    #         dataframe.loc[dataframe.index[-1], 'has_new_signal'] = 1 if decision.get('new_signal') else 0

    def log_decision(self, pair: str, current_state: TradingState, decision: dict):
        """Log comprehensive decision"""

        logger.info(f"📊 COMPREHENSIVE DECISION for {pair}")
        logger.info(f"{'=' * 60}")
        logger.info(f"Current State: {current_state.value}")
        logger.info(f"Decision: {decision['decision']}")
        logger.info(f"Confidence: {decision['confidence']}")
        logger.info(f"Urgency: {decision['urgency']}")
        logger.info(f"Reason: {decision['reason']}")

        # self.dp.send_msg(f"""```
        # COMPREHENSIVE DECISION for {pair}
        # {'=' * 60}
        # Current State: {current_state.value}
        # Decision: {decision['decision']}
        # Confidence: {decision['confidence']}
        # Urgency: {decision['urgency']}
        # Reason: {decision['reason']}
        # ```""")

        if decision.get('new_signal'):
            logger.info(f"🆕 NEW SIGNAL DETECTED: :")
            logger.info(f"   Action: {decision['new_signal'].get('action')}")
            logger.info(f"   Entry: ${decision['new_signal'].get('entry')}")
            logger.info(f"   SL: ${decision['new_signal'].get('sl')}")
            logger.info(f"   TP: ${decision['new_signal'].get('tp')}")

            self.dp.send_msg(f"""🆕 NEW SIGNAL DETECTED: {pair}
Action: {decision['new_signal'].get('action')}
Entry: ${decision['new_signal'].get('entry')}
SL: ${decision['new_signal'].get('sl')}
TP: ${decision['new_signal'].get('tp')}""")

        logger.info(f"{'=' * 60}")

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Entry logic with limit orders based on comprehensive decisions"""

        pair = metadata['pair']

        # Enter new positions only when decision is ENTER_NEW
        if pair in self.comprehensive_decisions:
            decision = self.comprehensive_decisions[pair]

            if (decision['decision'] == 'ENTER_NEW' and
                    decision.get('new_signal') and
                    decision['confidence'] in ['HIGH', 'MEDIUM']):

                new_signal = decision['new_signal']

                # Validate signal has required TP/SL
                if not new_signal.get('sl') or not new_signal.get('tp'):
                    logger.warning(f"⚠️ Signal for {pair} missing SL or TP - skipping entry")
                    return dataframe

                # Long entries with limit order
                if new_signal.get('action') == 'LONG':
                    dataframe.loc[
                        dataframe.index[-1:],
                        ['enter_long', 'enter_tag']
                    ] = (1,
                         f'deepseek_long_limit_{new_signal.get("entry", 0)}_{new_signal.get("tp", 0)}_{new_signal.get("sl", 0)}')

                # Short entries with limit order
                elif new_signal.get('action') == 'SHORT':
                    dataframe.loc[
                        dataframe.index[-1:],
                        ['enter_short', 'enter_tag']
                    ] = (1,
                         f'deepseek_short_limit_{new_signal.get("entry", 0)}_{new_signal.get("tp", 0)}_{new_signal.get("sl", 0)}')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit logic with TP/SL based on comprehensive decisions"""

        pair = metadata['pair']
        current_price = dataframe['close'].iloc[-1]

        # Check for TP exits from active signals
        if pair in self.active_signals:
            signal = self.active_signals[pair]
            tp_price = signal.get('tp')

            # Get current open trades for this pair
            open_trades = Trade.get_open_trades()
            pair_trades = [trade for trade in open_trades if trade.pair == pair]

            for trade in pair_trades:
                if tp_price:
                    # Check Take Profit
                    if trade.entry_side == "buy" and current_price >= tp_price:
                        dataframe.loc[
                            dataframe.index[-1:],
                            ['exit_long', 'exit_tag']
                        ] = (1, f'tp_hit_{tp_price}')
                        logger.info(f"🎯 TP hit for {pair} at ${tp_price}")

                    elif trade.entry_side == "sell" and current_price <= tp_price:
                        dataframe.loc[
                            dataframe.index[-1:],
                            ['exit_short', 'exit_tag']
                        ] = (1, f'tp_hit_{tp_price}')
                        logger.info(f"🎯 TP hit for {pair} at ${tp_price}")

        # Handle comprehensive decisions
        if pair in self.comprehensive_decisions:
            decision = self.comprehensive_decisions[pair]

            # Close now - immediate exit
            # if decision['decision'] == 'CLOSE_NOW':
            #     dataframe.loc[
            #         dataframe.index[-1:],
            #         ['exit_long', 'exit_short', 'exit_tag']
            #     ] = (1, 1, 'deepseek_close_now')

            # Partial exit
            # elif decision['decision'] == 'PARTIAL_EXIT':
            #     dataframe.loc[
            #         dataframe.index[-1:],
            #         ['exit_long', 'exit_short', 'exit_tag']
            #     ] = (1, 1, 'deepseek_partial_exit')

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """Dynamic stoploss based on DeepSeek signal"""

        if pair in self.active_signals:
            signal = self.active_signals[pair]
            sl_price = signal.get('sl')

            if sl_price and trade.open_rate:
                if trade.entry_side == "buy":
                    # Long position: SL below entry
                    sl_ratio = (trade.open_rate - sl_price) / trade.open_rate
                    if sl_ratio > 0:
                        # logger.info(f"🛑 Using DeepSeek SL for {pair}: {sl_ratio:.4f}")
                        return -sl_ratio * self.leverage_value
                    else:
                        logger.warning(f"⚠️ Invalid SL for long {pair}: SL {sl_price} >= Entry {trade.open_rate}")
                        return -0.01  # Fallback 1%
                else:
                    # Short position: SL above entry  
                    sl_ratio = (sl_price - trade.open_rate) / trade.open_rate
                    if sl_ratio > 0:
                        # logger.info(f"🛑 Using DeepSeek SL for {pair}: {sl_ratio:.4f}")
                        return -sl_ratio * self.leverage_value
                    else:
                        logger.warning(f"⚠️ Invalid SL for short {pair}: SL {sl_price} <= Entry {trade.open_rate}")
                        return -0.01  # Fallback 1%

        # Fallback to default stoploss
        return self.stoploss

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                            rate: float, time_in_force: str, current_time,
                            entry_tag: str, side: str, **kwargs) -> bool:
        """Confirm entry with comprehensive validation"""

        if pair in self.comprehensive_decisions:
            decision = self.comprehensive_decisions[pair]

            # Block entry if decision is not ENTER_NEW
            if decision['decision'] != 'ENTER_NEW':
                logger.info(f"🚫 Blocking entry for {pair} - Decision: {decision['decision']}")
                self.dp.send_msg(f"🚫 Blocking entry for {pair} - Decision: {decision['decision']}")
                return False

            # Validate signal has required TP/SL
            new_signal = decision.get('new_signal', {})
            if not new_signal.get('sl') or not new_signal.get('tp'):
                logger.warning(f"🚫 Blocking entry for {pair} - Missing SL or TP")
                self.dp.send_msg(f"🚫 Blocking entry for {pair} - Missing SL or TP")
                return False

            # Log the entry with TP/SL levels
            logger.info(f"✅ Confirming {side} entry for {pair}")
            logger.info(f"   Entry: ${rate}")
            logger.info(f"   SL: ${new_signal.get('sl')}")
            logger.info(f"   TP: ${new_signal.get('tp')}")

            self.dp.send_msg(f"""✅ Confirming {side} entry for {pair}
Entry: ${rate}
SL: ${new_signal.get('sl')}
TP: ${new_signal.get('tp')}""")

            # Mark signal as used
            if pair in self.order_manager.last_signals:
                self.order_manager.last_signals[pair]['used'] = True

            return True

        return False

    def custom_entry_price(self, pair: str, current_time, proposed_rate: float,
                           entry_tag: str, side: str, **kwargs) -> float:
        """Custom entry price for limit orders from DeepSeek signal"""

        if pair in self.comprehensive_decisions:
            decision = self.comprehensive_decisions[pair]

            if (decision['decision'] == 'ENTER_NEW' and
                    decision.get('new_signal') and
                    decision['new_signal'].get('entry')):

                entry_price = decision['new_signal']['entry']
                current_price = proposed_rate

                # Validate limit price (should be better than current market)
                if side == 'long' and entry_price >= current_price:
                    logger.warning(f"⚠️ {pair} - Long limit price {entry_price} not below market {current_price}")
                    return current_price * 0.999  # Slightly below market
                elif side == 'short' and entry_price <= current_price:
                    logger.warning(f"⚠️ {pair} - Short limit price {entry_price} not above market {current_price}")
                    return current_price * 1.001  # Slightly above market

                logger.info(f"🎯 {pair} - Using DeepSeek limit entry: ${entry_price}")
                return entry_price

        return proposed_rate

    def custom_exit_price(self, pair: str, trade: Trade, current_time,
                          proposed_rate: float, current_profit: float,
                          exit_tag: str, **kwargs) -> float:
        """Custom exit price for TP orders"""

        if exit_tag and 'tp_hit' in exit_tag and pair in self.active_signals:
            signal = self.active_signals[pair]
            tp_price = signal.get('tp')

            if tp_price:
                logger.info(f"🎯 {pair} - Using TP exit price: ${tp_price}")
                return tp_price

        return proposed_rate

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str,
                           amount: float, rate: float, time_in_force: str,
                           exit_reason: str, current_time, **kwargs) -> bool:
        """Clean up signals when trade exits"""

        # Remove active signal when trade is fully closed
        if pair in self.active_signals:
            logger.info(f"🧹 Cleaning up signal for {pair} - {exit_reason}")
            del self.active_signals[pair]
            del self.comprehensive_decisions[pair]

        return True

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        decision = self.comprehensive_decisions[trade.pair]
        if decision['decision'] == 'PARTIAL_EXIT':
            return -(trade.stake_amount / 2), 'deepseek_partial_exit'

        return None

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if current_time - timedelta(hours=24) > trade.open_date_utc:
            return f'exit_24h'
        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        return self.leverage_value

    def _store_pair_data(self, pair: str, dataframe: DataFrame, metadata: dict):
        """Store pair data for async batch processing with timeout logic"""
        if not hasattr(self, '_pair_data_cache'):
            self._pair_data_cache = {}

        current_time = datetime.now()

        self._pair_data_cache[pair] = {
            'dataframe': dataframe.copy(),
            'metadata': metadata,
            'timestamp': current_time
        }

        # Check if we should trigger processing
        should_process = self._should_trigger_batch_processing(current_time)

        if should_process:
            self._execute_batch_processing()

    def _should_trigger_batch_processing(self, current_time: datetime) -> bool:
        """
        Determine if we should trigger batch processing based on:
        1. Multiple pairs available (> 1)
        """
        if len(self._pair_data_cache) >= self.max_cache_size:
            logger.info(f"Triggering batch processing: {len(self._pair_data_cache)} pairs available")
            return True

        return False

    def _execute_batch_processing(self):
        """Execute the actual batch processing"""
        try:
            if not hasattr(self, '_pair_data_cache') or not self._pair_data_cache:
                return

            current_time = datetime.now()
            pairs_to_process = list(self._pair_data_cache.keys())

            logger.info(f"Processing {len(pairs_to_process)} pairs asynchronously")

            start_time = datetime.now()
            futures = {
                self.executor.submit(
                        self._process_single_pair,
                        pair,
                        self._pair_data_cache[pair]['dataframe'],
                        self._pair_data_cache[pair]['metadata'],
                        current_time
                    ): pair for pair in pairs_to_process
            }

            for future in as_completed(futures):
                pair = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing pair {pair}: {e}")
            end_time = datetime.now()
            logger.info(f"Finished processing {len(pairs_to_process)} pairs asynchronously in "
                        f"{(end_time - start_time).total_seconds()} seconds")

            # Clean up processed pairs
            self._cleanup_processed_pairs(pairs_to_process)

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")

    def _cleanup_processed_pairs(self, processed_pairs: list):
        """Clean up processed pairs from cache"""
        for pair in processed_pairs:
            if pair in self._pair_data_cache:
                del self._pair_data_cache[pair]

        logger.info(f"Cleaned up {len(processed_pairs)} pairs from cache")

    def _process_single_pair(self, pair: str, dataframe: DataFrame, metadata: dict, current_time: datetime):
        """Process a single pair asynchronously"""
        try:
            pair = metadata['pair']
            current_time = datetime.now()
            current_price = dataframe['close'].iloc[-1]

            # Determine current trading state
            current_state = self.determine_current_state(pair)

            # Update state in manager
            self.order_manager.update_trading_state(pair, current_state, {
                'current_price': current_price,
                'timestamp': current_time
            })

            # Get comprehensive decision
            if (pair not in self.last_check_time or
                    (current_time - self.last_check_time[pair]).seconds > self.check_interval):

                market_data = self.prepare_market_data(dataframe, pair, current_price)
                decision = self.order_manager.get_comprehensive_decision(pair, market_data)

                self.comprehensive_decisions[pair] = decision
                self.last_check_time[pair] = current_time

                # Log decision
                self.log_decision(pair, current_state, decision)

                # Handle new signal if provided
                if decision.get('new_signal'):
                    self.order_manager.update_last_signal(pair, decision['new_signal'])
                    # Store active signal for TP/SL management
                    self.active_signals[pair] = decision['new_signal']

            # Populate indicators
            # self.populate_decision_indicators(dataframe, pair, current_state)

        except Exception as e:
            logger.error(f"Error processing pair {pair}: {e}")
