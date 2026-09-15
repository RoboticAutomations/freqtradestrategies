# --- Do not remove these libs ---
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from freqtrade.strategy import (IStrategy, BooleanParameter, DecimalParameter,
                                IntParameter, merge_informative_pair)
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class POCStrategy(IStrategy):
    """
    Point of Control (PoC) Strategy based on Volume Profile
    Use PoC, RSI and Rolling range
    """
    
    # --- Strategy Settings ---
    INTERFACE_VERSION = 3
    timeframe = '15m'
    informative_timeframe = '1h'
    can_short = True
    use_custom_stoploss = False
    use_custom_exit = True
    process_only_new_candles = True
    exit_profit_only = True
    
    # Minimal ROI - let the strategy manage exits
    minimal_roi = {"0": 0.20}  # 20% max, but custom exit handles most
    
    # Base stoploss (will be overridden by custom_stoploss)
    stoploss = -0.99
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.025
    trailing_stop_positive_offset = 0.06
    trailing_only_offset_is_reached = True

    # ============ HYPEROPT PARAMETERS ============
    
    # --- Point of Control (PoC) ---
    poc_window = IntParameter(48, 192, default=96, space='buy', optimize=True)  # 24h on 15m
    poc_bins = IntParameter(10, 50, default=20, space='buy', optimize=True)
    
    # --- Rolling Range ---
    range_window = IntParameter(10, 40, default=20, space='buy', optimize=True)
    
    # --- RSI ---
    rsi_period = IntParameter(7, 21, default=14, space='buy', optimize=True)
    rsi_oversold = IntParameter(20, 35, default=30, space='buy', optimize=True)
    rsi_overbought = IntParameter(65, 80, default=70, space='buy', optimize=True)
    
    # --- PoC Divergence (ATR multiples) ---
    poc_divergence_atr = DecimalParameter(0.5, 3.0, default=1.5, decimals=1, space='buy', optimize=True)
    
    # --- Re-Entry System ---
    max_reentries = IntParameter(1, 5, default=3, space='buy', optimize=True)
    reentry_multiplier = DecimalParameter(1.2, 2.0, default=1.5, decimals=1, space='buy', optimize=True)
    
    # --- Stop Loss / Take Profit (ATR multiples) ---
    atr_sl_mult = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space='sell', optimize=True)
    atr_tp_mult = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space='sell', optimize=True)
    
    # --- Capital Management ---
    base_leverage = IntParameter(3, 10, default=5, space='protection', optimize=True)
    reserve_pct = DecimalParameter(0.20, 0.40, default=0.30, decimals=2, space='protection', optimize=True)
    max_position_pct = DecimalParameter(0.20, 0.40, default=0.30, decimals=2, space='protection', optimize=True)
    
    # --- Volatility Thresholds for Dynamic Leverage ---
    vol_low_threshold = DecimalParameter(0.005, 0.015, default=0.01, decimals=3, space='protection', optimize=True)
    vol_high_threshold = DecimalParameter(0.02, 0.04, default=0.025, decimals=3, space='protection', optimize=True)

    # --- Trend Filter ---
    use_trend_filter = BooleanParameter(default=True, space='buy', optimize=True)
    adx_threshold = IntParameter(20, 35, default=25, space='buy', optimize=True)

    # --- Volume Oscillator ---
    vol_osc_fast = IntParameter(3, 10, default=5, space="buy", optimize=True)
    vol_osc_slow = IntParameter(15, 30, default=20, space="buy", optimize=True)
    vol_osc_threshold = DecimalParameter(-50, 50, default=0, decimals=0, space="buy", optimize=True)


    cum_delta_window = IntParameter(10, 50, default=20, space="buy", optimize=True)
    cum_delta_threshold = DecimalParameter(
        -30,
        30,
        default=0,
        decimals=0,
        space="buy",
        optimize=True,
    )

    # --- Plot Config ---
    plot_config = {
        "main_plot": {
            "poc": {"color": "yellow"},
            "range_high": {"color": "red"},
            "range_low": {"color": "green"},
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "purple"},
            },
            "Distance": {
                "dist_poc_atr": {"color": "blue"},
            },
        },
    }

    # ============ LEVERAGE ============
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str, 
                 side: str, **kwargs) -> float:
        """
        Dynamic leverage based on volatility.
        Lower volatility = Higher leverage (more confidence in mean reversion)
        Higher volatility = Lower leverage (protect against breakouts)
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is not None and not dataframe.empty:
                volatility = dataframe.iloc[-1].get('volatility', 0.02)
                
                if volatility < self.vol_low_threshold.value:
                    # Low volatility: use higher leverage
                    leverage = min(self.base_leverage.value + 3, max_leverage)
                elif volatility > self.vol_high_threshold.value:
                    # High volatility: use lower leverage  
                    leverage = max(self.base_leverage.value - 2, 3)
                else:
                    leverage = self.base_leverage.value
                
                logger.info(f"Leverage for {pair}: Volatility={volatility:.4f}, Leverage={leverage}x")
                return min(leverage, max_leverage)
        except Exception as e:
            logger.warning(f"Error calculating leverage for {pair}: {e}")
        
        return float(self.base_leverage.value)

    # ============ STAKE AMOUNT ============
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        """
        Dynamic stake sizing with:
        1. Reserve protection (never use full balance)
        2. Re-entry scaling (increase stake on re-entries)
        3. Max position size cap
        """
        # Get available balance (respecting reserve)
        try:
            available = self.wallets.get_total_stake_amount() if hasattr(self, 'wallets') else proposed_stake * 5
            reserve = available * self.reserve_pct.value
            usable = available - reserve
            
            # Base stake is 5% of usable balance
            base_stake = usable * 0.05
            
            # Check for re-entry scaling
            reentry_count = self._get_reentry_count(pair, side)
            if reentry_count > 0:
                # Scale up stake for re-entries
                multiplier = self.reentry_multiplier.value ** reentry_count
                base_stake = base_stake * multiplier
            
            # Cap at max position size
            max_allowed = usable * self.max_position_pct.value
            stake = min(base_stake, max_allowed, max_stake)
            
            return max(stake, min_stake or 0)
            
        except Exception as e:
            logger.warning(f"Error calculating stake for {pair}: {e}")
            return proposed_stake

    def _get_reentry_count(self, pair: str, side: str) -> int:
        """Get current re-entry count for a pair and direction."""
        try:
            trades = Trade.get_trades_proxy(pair=pair, is_open=True)
            count = 0
            for trade in trades:
                if (side == 'long' and not trade.is_short) or (side == 'short' and trade.is_short):
                    reentries = trade.get_custom_data('reentry_count')
                    if reentries:
                        count = max(count, int(reentries))
            return count
        except Exception:
            return 0

    # ============ INFORMATIVE PAIRS ============
    def informative_pairs(self):
        pairs = [(pair, self.informative_timeframe) for pair in self.dp.current_whitelist()]
        return pairs

    # ============ INDICATORS ============
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate all technical indicators:
        - Rolling PoC (Point of Control)
        - RSI
        - Rolling Range (high/low)
        - ATR for volatility and normalization
        - ADX for trend filtering
        """
        
        # --- RSI ---
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=int(self.rsi_period.value))
        
        # --- ATR for volatility ---
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['volatility'] = dataframe['atr'] / dataframe['close']
        
        # --- Rolling Range ---
        dataframe['range_high'] = dataframe['high'].rolling(window=int(self.range_window.value)).max()
        dataframe['range_low'] = dataframe['low'].rolling(window=int(self.range_window.value)).min()
        dataframe['range_mid'] = (dataframe['range_high'] + dataframe['range_low']) / 2
        
        # --- Rolling PoC (Volume Profile Point of Control) ---
        dataframe['poc'] = self._calculate_rolling_poc(dataframe)
        
        # --- ATR-normalized distance from PoC ---
        dataframe['dist_poc'] = dataframe['close'] - dataframe['poc']
        dataframe['dist_poc_atr'] = dataframe['dist_poc'] / dataframe['atr']
        
        # --- ADX for trend filter ---
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)
        
        # --- Range breach detection ---
        dataframe['below_range'] = dataframe['close'] < dataframe['range_low']
        dataframe['above_range'] = dataframe['close'] > dataframe['range_high']

        dataframe["volume_osc_norm"] = self._volume_oscillator_normalized(
            dataframe,
            self.vol_osc_fast.value,
            self.vol_osc_slow.value
        )

        dataframe["cum_delta_norm"] = self._cumulative_delta_normalized(
            dataframe,
            self.cum_delta_window.value
        )
        
        # --- Higher timeframe trend filter ---
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
        if informative is not None and not informative.empty and len(informative) > 50:
            informative['ema_50'] = ta.EMA(informative['close'], timeperiod=50)
            informative['htf_trend'] = np.where(informative['close'] > informative['ema_50'], 1, -1)
            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, 
                                               self.informative_timeframe, ffill=True)
        else:
            dataframe['htf_trend_1h'] = 0
            dataframe['ema_50_1h'] = dataframe['close']
        
        return dataframe

    def _calculate_rolling_poc(self, dataframe: DataFrame) -> np.ndarray:
        """
        Calculate rolling Volume Profile Point of Control (PoC).
        PoC is the price level with the highest volume concentration.
        """
        poc = np.full(len(dataframe), np.nan)
        window = int(self.poc_window.value)
        bins = int(self.poc_bins.value)
        
        for i in range(window, len(dataframe)):
            # Extract price and volume for the window
            prices = dataframe['close'].iloc[i - window:i].values
            volumes = dataframe['volume'].iloc[i - window:i].values
            
            if len(prices) == 0 or np.sum(volumes) == 0:
                continue
            
            # Create volume-weighted histogram
            try:
                hist, edges = np.histogram(prices, bins=bins, weights=volumes)
                # Find bin with maximum volume
                max_idx = hist.argmax()
                poc_price = (edges[max_idx] + edges[max_idx + 1]) / 2
                poc[i] = poc_price
            except Exception:
                continue
        
        return poc
    
    def _ema(self, s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=int(n), adjust=False).mean()

    def _volume_oscillator_normalized(self, df: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.Series:
        """
        Volume Oscillator Normalized: (EMA_fast(vol) - EMA_slow(vol)) / EMA_slow(vol) x 100
        Normalizat între -100 si 100
        """
        volume = df["volume"].astype(float)
        ema_fast = self._ema(volume, fast)
        ema_slow = self._ema(volume, slow).replace(0, np.nan)

        vol_osc = ((ema_fast - ema_slow) / ema_slow * 100).replace([np.inf, -np.inf], np.nan).fillna(0)

        vol_osc_norm = vol_osc.clip(-100, 100).astype(float)

        return vol_osc_norm


    def _cumulative_delta_normalized(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Cumulative Delta Normalized: aproximare bazată pe close position în range
        Delta = (close - low) / (high - low) - 0.5, scalat
        Pozitiv = buying pressure
        Negativ = selling pressure
        Normalizat pe fereastra rulanta.
        """
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        # Calculăm poziția close în range sunt calcule grele :)
        range_size = (high - low).replace(0, np.nan)
        close_position = (close - low) / range_size  # 0=low, 1=high

        # Delta per bară: pozitiv dacă close spre high, negativ dacă spre low
        delta_per_bar = (close_position - 0.5) * 2 * volume  # Scalat cu volum

        # Cumulative delta pe fereastră
        cum_delta = delta_per_bar.rolling(window, min_periods=1).sum()

        # Normalizare
        cum_delta_max = cum_delta.rolling(window * 2, min_periods=1).max().abs()
        cum_delta_min = cum_delta.rolling(window * 2, min_periods=1).min().abs()
        normalizer = pd.concat([cum_delta_max, cum_delta_min], axis=1).max(axis=1).replace(0, 1)

        cum_delta_norm = (cum_delta / normalizer * 100).clip(-100, 100).fillna(0).astype(float)

        return cum_delta_norm

    # ============ ENTRY LOGIC ============
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Tripartite Confluence Entry:
        1. Price breaches Rolling Range boundary
        2. RSI in overbought/oversold zone
        3. Price diverges from PoC by ATR threshold
        """

        rsi_oversold = dataframe["rsi"] < float(self.rsi_oversold.value)
        rsi_overbought = dataframe["rsi"] > float(self.rsi_overbought.value)
        rsi_neutral_low = dataframe["rsi"] < 50  # RSI sub 50 = momentum descendent
        rsi_neutral_high = dataframe["rsi"] > 50  # RSI peste 50 = momentum ascendent

        vol_positive = dataframe["volume_osc_norm"] > float(self.vol_osc_threshold.value)
        vol_negative = dataframe["volume_osc_norm"] < -float(self.vol_osc_threshold.value)

        delta_bullish = dataframe["cum_delta_norm"] > float(self.cum_delta_threshold.value)
        delta_bearish = dataframe["cum_delta_norm"] < -float(self.cum_delta_threshold.value)
        
        # --- LONG ENTRY CONDITIONS ---
        long_conditions = [
            # Price at or below range low
            dataframe['close'] <= dataframe['range_low'],
            # Price diverges below PoC
            dataframe['dist_poc_atr'] < -self.poc_divergence_atr.value,
            # Volume confirmation
            dataframe['volume'] > 0,
            # RSI oversold
            (rsi_oversold | rsi_neutral_low),
            # Volume confirmation
            vol_positive,
            # Delta bullish
            delta_bullish
        ]
        
        # Optional trend filter: don't long in strong downtrend
        if self.use_trend_filter.value:
            long_conditions.append(
                (dataframe['adx'] < self.adx_threshold.value) | 
                (dataframe['htf_trend_1h'] >= 0)
            )
        
        dataframe.loc[
            np.all(long_conditions, axis=0),
            ['enter_long', 'enter_tag']
        ] = (1, 'mean_reversion_long')
        
        # --- SHORT ENTRY CONDITIONS ---
        short_conditions = [
            # Price at or above range high
            dataframe['close'] >= dataframe['range_high'],
            # Price diverges above PoC
            dataframe['dist_poc_atr'] > self.poc_divergence_atr.value,
            # Volume confirmation
            dataframe['volume'] > 0,
            # RSI overbought
            (rsi_overbought | rsi_neutral_high),
            # Volume confirmation
            vol_negative,
            # Delta bearish
            delta_bearish
        ]
        
        # Optional trend filter: don't short in strong uptrend
        if self.use_trend_filter.value:
            short_conditions.append(
                (dataframe['adx'] < self.adx_threshold.value) | 
                (dataframe['htf_trend_1h'] <= 0)
            )
        
        dataframe.loc[
            np.all(short_conditions, axis=0),
            ['enter_short', 'enter_tag']
        ] = (1, 'mean_reversion_short')
        
        return dataframe

    # ============ EXIT LOGIC ============
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit signals handled by custom_exit for more control."""
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                current_profit: float, **kwargs) -> Optional[Union[str, bool]]:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        range_hi = last_candle['range_high']
        range_lo = last_candle['range_low']

        # --- Basic safety exits ---
        if current_profit > 0.20:
            return "profit"  # Lock in massive wins

        # Detect position direction
        is_short = trade.is_short

        # --- Get HTF context (you already populate this via informative pairs) ---
        htf_trend = last_candle.get('htf_trend_1h', 0)  # Assume: 1=uptrend, -1=downtrend, 0=neutral
        ema_50_htf = last_candle.get('ema_50_1h', current_rate)

        # Define strong trend conditions
        strong_uptrend = (htf_trend == 1) and (current_rate > ema_50_htf * 1.005)
        strong_downtrend = (htf_trend == -1) and (current_rate < ema_50_htf * 0.995)

        in_strong_trend = (strong_uptrend and not is_short) or (strong_downtrend and is_short)

        # --- ATR for volatility-based buffer ---
        atr = last_candle.get('atr', 0.005 * current_rate)  # fallback to 0.5% if missing

        # --- EXIT LOGIC ---

        # 1. If in STRONG TREND → DO NOT exit early. Let trailing_stop handle it.
        if in_strong_trend:
            # Optionally: add a very wide ATR-based emergency stop
            if not is_short:
                if current_rate < (range_lo - atr):
                    return "trend_break_long"
            else:
                if current_rate > (range_hi + atr):
                    return "trend_break_short"
            # Otherwise: HOLD → trailing_stop will manage exit
            return None

        # 2. If NOT in strong trend → allow conservative profit-taking
        if current_profit > 0:
            if current_profit >= 0.04:  # Only 4% (lower than before) and only if no trend
                return "profit_secure_no_trend"
            # Also exit if RSI shows reversal + price stalls
            rsi = last_candle['rsi']
            rsi_threshold = 65 if not is_short else 35
            if (not is_short and rsi > rsi_threshold and current_rate < last_candle['close']) or \
            (is_short and rsi < rsi_threshold and current_rate > last_candle['close']):
                return "rsi_reversal_exit"

        # Default: no exit signal → hold
        return None

    # ============ CUSTOM STOPLOSS ============
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        ATR-based dynamic stoploss with hard floor.
        """
        hard_stop = -0.99  # 10% absolute max loss
        
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or dataframe.empty:
                return hard_stop
            
            last_candle = dataframe.iloc[-1]
            atr = last_candle.get('atr')
            
            if atr is None or atr == 0:
                return hard_stop
            
            entry_price = trade.open_rate
            sl_distance = atr * self.atr_sl_mult.value
            
            if trade.is_short:
                sl_price = entry_price + sl_distance
                if current_rate >= sl_price:
                    return -0.001  # Trigger stop
            else:
                sl_price = entry_price - sl_distance
                if current_rate <= sl_price:
                    return -0.001  # Trigger stop
            
            # Trail after profit
            if current_profit > 0.03:
                return -0.015  # Tighten to 1.5%
            elif current_profit > 0.02:
                return -0.02  # Tighten to 2%
            
            return -1  # Use default/no change
            
        except Exception as e:
            logger.error(f"Error in custom_stoploss for {pair}: {e}")
            return hard_stop

    # ============ ADJUST TRADE POSITION (MARTINGALE RE-ENTRY) ============
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Martingale-style position adjustment.
        
        Adds to position when:
        1. Price moves further against us (DCA into losing position)
        2. Re-entry conditions are met (RSI + range boundary)
        3. Max re-entries not reached
        
        Returns positive stake to increase position, negative to decrease, None for no change.
        """
        try:
            pair = trade.pair
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            
            if dataframe is None or dataframe.empty:
                return None
            
            last_candle = dataframe.iloc[-1]
            entry_price = trade.open_rate
            rsi = last_candle.get('rsi', 50)
            atr = last_candle.get('atr', 0)
            range_high = last_candle.get('range_high', current_rate)
            range_low = last_candle.get('range_low', current_rate)
            current_close = last_candle.get('close', current_rate)
            poc = last_candle.get('poc', current_close)
            
            # Get current re-entry count
            reentry_count = trade.get_custom_data('reentry_count') or 0
            
            # Check if max re-entries reached
            if reentry_count >= self.max_reentries.value:
                return None
            
            # Calculate time since last adjustment to prevent rapid re-entries
            last_adjustment_time = trade.get_custom_data('last_adjustment_time')
            if last_adjustment_time:
                time_since_adjustment = (current_time - datetime.fromisoformat(last_adjustment_time)).total_seconds() / 60
                if time_since_adjustment < 15:  # At least 15 minutes between adjustments (1 candle)
                    return None
            
            # Calculate stake for adjustment
            filled_entries = trade.nr_of_successful_entries
            
            should_adjust = False
            adjustment_reason = ""
            
            # === LONG POSITION: Add when price drops further ===
            if not trade.is_short:
                # Condition: Price at lower range + RSI oversold + moving away from PoC
                if (current_close <= range_low and 
                    rsi < self.rsi_oversold.value and
                    current_profit < -0.01):  # Position is losing at least 1%
                    
                    # Check if price diverges from PoC (confirms mean reversion opportunity)
                    if atr > 0:
                        dist_poc_atr = (current_close - poc) / atr
                        if dist_poc_atr < -self.poc_divergence_atr.value:
                            should_adjust = True
                            adjustment_reason = f"LONG DCA: RSI={rsi:.1f}, Dist_PoC={dist_poc_atr:.2f}ATR, Loss={current_profit*100:.2f}%"
            
            # === SHORT POSITION: Add when price rises further ===
            else:
                # Condition: Price at upper range + RSI overbought + moving away from PoC
                if (current_close >= range_high and 
                    rsi > self.rsi_overbought.value and
                    current_profit < -0.01):  # Position is losing at least 1%
                    
                    # Check if price diverges from PoC (confirms mean reversion opportunity)
                    if atr > 0:
                        dist_poc_atr = (current_close - poc) / atr
                        if dist_poc_atr > self.poc_divergence_atr.value:
                            should_adjust = True
                            adjustment_reason = f"SHORT DCA: RSI={rsi:.1f}, Dist_PoC={dist_poc_atr:.2f}ATR, Loss={current_profit*100:.2f}%"
            
            if should_adjust:
                # Calculate adjustment stake with martingale multiplier
                try:
                    available = self.wallets.get_total_stake_amount() if hasattr(self, 'wallets') else max_stake
                    reserve = available * self.reserve_pct.value
                    usable = available - reserve
                    
                    # Base adjustment is 5% of usable, scaled by multiplier^reentry_count
                    multiplier = self.reentry_multiplier.value ** (reentry_count + 1)
                    adjustment_stake = usable * 0.05 * multiplier
                    
                    # Cap at max position size and available stake
                    max_allowed = usable * self.max_position_pct.value
                    current_stake = trade.stake_amount
                    remaining_allowed = max(0, max_allowed - current_stake)
                    
                    adjustment_stake = min(adjustment_stake, remaining_allowed, max_stake)
                    
                    if adjustment_stake >= (min_stake or 0):
                        # Update re-entry tracking
                        trade.set_custom_data('reentry_count', reentry_count + 1)
                        trade.set_custom_data('last_adjustment_time', current_time.isoformat())
                        
                        # ========== CONSOLE PRINT FOR SUCCESSFUL ADJUSTMENT ==========
                        direction = "SHORT" if trade.is_short else "LONG"
                        print(f"")
                        print(f"========== 🔄 POSITION ADJUSTMENT SUCCESSFUL ==========")
                        print(f"  📊 Pair: {pair}")
                        print(f"  📈 Direction: {direction}")
                        print(f"  💰 Adjustment Stake: {adjustment_stake:.4f}")
                        print(f"  🔢 Re-entry #{reentry_count + 1} of {self.max_reentries.value}")
                        print(f"  📉 Current Profit: {current_profit * 100:.2f}%")
                        print(f"  💵 Current Rate: {current_rate:.6f}")
                        print(f"  📍 Entry Rate: {trade.open_rate:.6f}")
                        print(f"  🎯 Reason: {adjustment_reason}")
                        print(f"  ⏰ Time: {current_time}")
                        print(f"  💼 Total Entries: {filled_entries + 1}")
                        print(f"  🏦 New Position Size: {current_stake + adjustment_stake:.4f}")
                        print(f"========================================================")
                        print(f"")
                        
                        logger.info(f"✅ Position adjustment: {pair} {direction} +{adjustment_stake:.4f} (re-entry #{reentry_count + 1})")
                        
                        return adjustment_stake
                        
                except Exception as e:
                    logger.warning(f"Error calculating adjustment stake for {pair}: {e}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error in adjust_trade_position for {trade.pair}: {e}")
            return None

    # ============ TRADE CONFIRMATION ============
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                            rate: float, time_in_force: str, current_time: datetime,
                            entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """
        Final confirmation before entering trade.
        Check re-entry limits and position sizing constraints.
        """
        try:
            # Check re-entry count
            reentry_count = self._get_reentry_count(pair, side)
            if reentry_count >= self.max_reentries.value:
                logger.info(f"Skipping {pair} {side}: max re-entries ({self.max_reentries.value}) reached")
                return False
            
            # Check if we have too many open trades in same direction
            trades = Trade.get_trades_proxy(is_open=True)
            same_direction = sum(1 for t in trades if 
                                 (side == 'long' and not t.is_short) or 
                                 (side == 'short' and t.is_short))
            if same_direction >= 10:  # Max 10 trades per direction
                logger.info(f"Skipping {pair}: too many {side} positions open")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error in confirm_trade_entry for {pair}: {e}")
            return True  # Allow trade on error
