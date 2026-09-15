import sqlite3
import time
import logging
import os
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade
from pandas import DataFrame, to_datetime
import pandas as pd
from datetime import datetime
import talib.abstract as ta
import numpy as np

logger = logging.getLogger(__name__)

class RustOrderbookStrategy(IStrategy):
    INTERFACE_VERSION = 3

    plot_config = {
        'main_plot': {
            # ═══════════════════════════════════════════════════════════════════
            # 1️⃣ PRICE & ZONES
            # ═══════════════════════════════════════════════════════════════════
            # Price is plotted by default.
        },
        'subplots': {
            # ═══════════════════════════════════════════════════════════════════
            # 2️⃣ NORMALIZED DELTA (Primary Signal - 1h)
            # ═══════════════════════════════════════════════════════════════════
            "Normalized Delta (1h)": {
                'norm_delta': {'color': '#E91E63', 'type': 'line'},
                'zero': {'color': '#888888', 'type': 'line'},
                'zone_high': {'color': 'green', 'type': 'line', 'dash': 'dash'},
                'zone_low': {'color': 'red', 'type': 'line', 'dash': 'dash'},
                'zone_high_extreme': {'color': 'green', 'type': 'line', 'dash': 'dot'},
                'zone_low_extreme': {'color': 'red', 'type': 'line', 'dash': 'dot'},
                'recent_max': {'color': '#cccccc', 'type': 'line', 'dash': 'dot'},
                'recent_min': {'color': '#cccccc', 'type': 'line', 'dash': 'dot'},
            },
            
            # ═══════════════════════════════════════════════════════════════════
            # 3️⃣ CVD & WHALE ACTIVITY
            # ═══════════════════════════════════════════════════════════════════
            "CVD Analysis": {
                'cvd_absolute': {'color': '#2196F3', 'type': 'line'},
                'whale_cvd_absolute': {'color': '#9C27B0', 'type': 'line'},
                'cvd_ema': {'color': '#FFA000', 'type': 'line'},
            },
            
            # ═══════════════════════════════════════════════════════════════════
            # 4️⃣ DELTA COMPONENTS (Z-Scores)
            # ═══════════════════════════════════════════════════════════════════
            "Z-Scores (Context)": {
                'v_norm': {'color': '#4CAF50', 'type': 'line'},
                'w_norm': {'color': '#F44336', 'type': 'line'},
            },
            
            # ═══════════════════════════════════════════════════════════════════
            # 5️⃣ ORDER COUNTS
            # ═══════════════════════════════════════════════════════════════════
            "Order Counts": {
                'buy_count': {'color': '#66BB6A', 'type': 'bar'},
                'sell_count': {'color': '#EF5350', 'type': 'bar'},
            },
            
            # ═══════════════════════════════════════════════════════════════════
            # 6️⃣ VOLUME
            # ═══════════════════════════════════════════════════════════════════
            "Volume": {
                'volume': {'color': '#9E9E9E', 'type': 'bar'},
            },
        }
    }

    # --- CONFIGURAZIONE STRATEGIA ---
    minimal_roi = {"0": 0.1, "5": 0.05} 
    stoploss = -0.2
    
    # --- HYPEROPT PARAMETERS ---
    # Optimization for Trend Switch Logic
    # lookback: How many candles back to check for the switch pivot (1 = check previous candle)
    trend_lookback = IntParameter(1, 5, default=1, space='buy', optimize=True)
    
    # min_delta_change: Threshold for confirming a directional switch (Noise Filter)
    min_delta_change = DecimalParameter(0.0, 50.0, default=50.0, decimals=1, space='buy', optimize=True)
    
    # Trailing Stop
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = False
    
    # TIMEFRAME CONFIGURATION
    # Supports '1h', '15m', etc.
    timeframe = '1h'
    
    # Dynamic Parameter Calculation Helper
    def get_window_factor(self):
        # Base is 1h (60 mins)
        tf_minutes = int(self.timeframe.replace('m', '').replace('h', '60')) if 'h' not in self.timeframe or self.timeframe == '1h' else 60
        if 'h' in self.timeframe and self.timeframe != '1h':
            tf_minutes = int(self.timeframe.replace('h', '')) * 60
            
        # Factor: 1h / Current TF
        # e.g. 15m -> 60/15 = 4. We need 4x larger windows.
        return 60 / tf_minutes
    
    # FONDAMENTALE: Reattività immediata
    process_only_new_candles = False
    
    can_short = True

    # DCA / Position Adjustment
    position_adjustment_enable = True
    max_entry_position_adjustment = 5
    
    # Path del DB visto da dentro il container Freqtrade
    DB_PATH = "/freqtrade/user_data/orderbook_cache.db"
    
    # Parameters
    cvd_lookback = 12 # hours previous was 24
    cvd_ema_period = 12
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              **kwargs) -> float | None | tuple[float | None, str | None]:
        
        # 1. Check if we have room for more entries
        if trade.nr_of_successful_entries >= (self.max_entry_position_adjustment + 1):
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        current_delta = last_candle['norm_delta']
        
        # 2. DCA Logic: Average entry when price returns to extreme zones
        # Use consistent position sizing (same stake amount as initial)
        stake_amount = trade.stake_amount / trade.nr_of_successful_entries # Or fixed stake? User said "consistent position sizing". Usually means fixed amount per entry.
        # Ideally, we should use the initial stake amount.
        # Freqtrade's adjust_trade_position returns the *additional* stake to add.
        # We can assume we want to add the same amount as the initial stake.
        # But we don't have easy access to initial stake here unless we store it or deduce it.
        # trade.stake_amount is the CURRENT total stake.
        # So additional stake = trade.stake_amount / trade.nr_of_successful_entries
        
        additional_stake = trade.stake_amount / trade.nr_of_successful_entries

        # Long DCA:
        # Trigger: Red Extreme Zone (<-400)
        # Condition: Price is dropping (profit < 0) and we are in extreme red
        if trade.is_short == False:
            if (current_profit < -0.02) and (current_delta < -400):
                 # Ensure we don't DCA too close to the last one? 
                 # For now, rely on the candle close (1h timeframe limits frequency naturally)
                 return additional_stake, "dca_long_extreme"

        # Short DCA:
        # Trigger: Green Extreme Zone (>500)
        # Condition: Price is rising (profit < 0) and we are in extreme green
        else:
            if (current_profit < -0.02) and (current_delta > 500):
                return additional_stake, "dca_short_extreme"

        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1. Load CVD Data
        cvd_df = self.load_cvd_history(metadata['pair'])
        
        if not cvd_df.empty:
            # 2. Resample to Strategy Timeframe (1h)
            # Ensure index is datetime and UTC aware
            cvd_df['timestamp'] = pd.to_datetime(cvd_df['timestamp'], unit='s', utc=True)
            cvd_df.set_index('timestamp', inplace=True)
            
            # Resample: Sum Delta, Volume, Counts
            resampled = cvd_df[['delta', 'whale_delta', 'volume', 'buy_count', 'sell_count']].resample(self.timeframe).sum()
            
            # 3. Calculate Cumulative Delta
            # We want the cumulative sum of the 1h deltas
            cvd_series = resampled['delta'].cumsum()
            whale_cvd_series = resampled['whale_delta'].cumsum()
            
            # 4. Merge into Main DataFrame
            # We merge on date/timestamp. Freqtrade dataframe has 'date' column.
            dataframe = dataframe.merge(cvd_series.rename('cvd_absolute'), left_on='date', right_index=True, how='left')
            dataframe = dataframe.merge(whale_cvd_series.rename('whale_cvd_absolute'), left_on='date', right_index=True, how='left')
            
            # Merge auxiliary metrics for Normalization
            dataframe = dataframe.merge(resampled[['delta', 'whale_delta', 'volume', 'buy_count', 'sell_count']], left_on='date', right_index=True, how='left', suffixes=('', '_res'))
            
            # Fill NaNs (if any missing data, forward fill or 0)
            dataframe['cvd_absolute'] = dataframe['cvd_absolute'].ffill().fillna(0)
            dataframe['whale_cvd_absolute'] = dataframe['whale_cvd_absolute'].ffill().fillna(0)
            
            # Fill 0 for volume/delta/counts if missing
            cols_to_fill = ['delta', 'whale_delta', 'volume', 'buy_count', 'sell_count']
            dataframe[cols_to_fill] = dataframe[cols_to_fill].fillna(0)

            # --- ROBUST NORMALIZATION (LONG SWINGS) ---
            
            # Dynamic Window Calculation
            # We want to maintain ~6 hours of smoothing and ~24 hours of context regardless of timeframe
            factor = self.get_window_factor()
            
            ema_period = max(2, int(6 * factor))  # 1h=6, 15m=24
            vol_window = max(2, int(24 * factor)) # 1h=24, 15m=96
            reversal_window = max(2, int(12 * factor)) # 1h=12, 15m=48

            # 1. Smooth Inputs (EMA ~6 hours) - High stability for trend following
            dataframe['delta_sm'] = ta.EMA(dataframe['delta'], timeperiod=ema_period)
            dataframe['whale_sm'] = ta.EMA(dataframe['whale_delta'], timeperiod=ema_period)
            
            # 2. Count Imbalance (Smoothed)
            total_counts = dataframe['buy_count'] + dataframe['sell_count']
            dataframe['count_ratio'] = (dataframe['buy_count'] - dataframe['sell_count']) / total_counts.replace(0, 1)
            dataframe['count_sm'] = ta.EMA(dataframe['count_ratio'], timeperiod=ema_period)

            # 3. Robust Z-Score for Volume Delta
            # Window ~24 hours
            v_mean = dataframe['delta_sm'].rolling(window=vol_window).mean()
            v_std = dataframe['delta_sm'].rolling(window=vol_window).std().replace(0, 1)
            # Clip Z-score to +/- 3 to handle extreme outliers
            v_z = ((dataframe['delta_sm'] - v_mean) / v_std).clip(-3, 3)
            # Increased sensitivity: tanh(z) instead of tanh(z/2)
            # This makes the indicator reach higher values faster (1 sigma = 0.76 instead of 0.46)
            dataframe['v_norm'] = np.tanh(v_z) 

            # 4. Robust Z-Score for Whale Delta
            w_mean = dataframe['whale_sm'].rolling(window=vol_window).mean()
            w_std = dataframe['whale_sm'].rolling(window=vol_window).std().replace(0, 1)
            w_z = ((dataframe['whale_sm'] - w_mean) / w_std).clip(-3, 3)
            dataframe['w_norm'] = np.tanh(w_z)

            # 5. Combined Score
            # Weighting: 40% Volume, 40% Whale, 20% Trade Counts
            # DYNAMIC WEIGHTING: If whale/count data is missing (V1), shift weight to Volume.
            
            has_whale = dataframe['whale_delta'].abs().sum() > 0
            has_counts = dataframe['buy_count'].sum() > 0
            
            w_vol = 0.4
            w_whale = 0.4
            w_count = 0.2
            
            if not has_whale:
                # If no whale data, distribute whale weight to volume
                w_vol += w_whale
                w_whale = 0.0
                
            if not has_counts:
                # If no count data, distribute count weight to volume
                w_vol += w_count
                w_count = 0.0
                
            # raw_score = (0.4 * dataframe['v_norm']) + (0.4 * dataframe['w_norm']) + (0.2 * dataframe['count_sm'])
            raw_score = (w_vol * dataframe['v_norm']) + (w_whale * dataframe['w_norm']) + (w_count * dataframe['count_sm'])
            
            # 6. Final Smoothing & Scaling
            # Scale to -1000..1000 (Zoomed In) and smooth output (EMA 2 for faster reaction)
            # EMA 2 on 1h = 2h reaction. On 15m we need EMA 8 to match.
            final_ema = max(2, int(2 * factor))
            dataframe['norm_delta'] = ta.EMA(raw_score * 1000, timeperiod=final_ema)
            
            # Calculate recent extremes for Reversal Logic (12h lookback)
            dataframe['recent_min'] = dataframe['norm_delta'].rolling(window=reversal_window).min()
            dataframe['recent_max'] = dataframe['norm_delta'].rolling(window=reversal_window).max()
            
            # 5. Smoothing (EMA)
            dataframe['cvd_ema'] = ta.EMA(dataframe['cvd_absolute'], timeperiod=self.cvd_ema_period)
            
            # 6. Divergence Detection (Simplified)
            # Price Higher High vs CVD Lower High (Bearish)
            # Price Lower Low vs CVD Higher Low (Bullish)
            
            # Helper for local peaks (requires more complex logic, using simple slope for now)
            dataframe['cvd_slope'] = dataframe['cvd_ema'].diff()
            dataframe['price_slope'] = dataframe['close'].diff()
            
            # --- PLOT HELPERS ---
            dataframe['zero'] = 0.0
            dataframe['zone_high'] = 600.0
            dataframe['zone_low'] = -600.0
            dataframe['zone_high_extreme'] = 950.0
            dataframe['zone_low_extreme'] = -950.0
            
        else:
            dataframe['cvd_absolute'] = 0.0
            dataframe['whale_cvd_absolute'] = 0.0
            dataframe['norm_delta'] = 0.0 # Default neutral
            dataframe['w_norm'] = 0.0 # Default neutral
            dataframe['recent_min'] = 0.0 # Initial value
            dataframe['recent_max'] = 0.0 # Initial value
            dataframe['cvd_ema'] = 0.0
            dataframe['cvd_slope'] = 0.0
            dataframe['price_slope'] = 0.0
            dataframe['zero'] = 0.0
            dataframe['zone_high'] = 600.0
            dataframe['zone_low'] = -600.0
            dataframe['zone_high_extreme'] = 950.0
            dataframe['zone_low_extreme'] = -950.0

        return dataframe

    def load_cvd_history(self, pair):
        db_path = self.DB_PATH
        try:
            coin_name = pair.split('/')[0]
            target_key = f"{coin_name}/USDT"
            
            # Dynamic Path Resolution
            if not os.path.exists(db_path):
                # Check local user_data (fallback for local execution outside Docker)
                local_path = os.path.join(os.getcwd(), 'user_data', 'orderbook_cache.db')
                if os.path.exists(local_path):
                    db_path = local_path
            
            # Connection String Construction
            # We use URI to enable Read-Only mode (mode=ro) which is crucial for concurrency
            if os.name == 'nt':
                # Windows: needs file:///Drive:/Path
                abs_path = os.path.abspath(db_path).replace('\\', '/')
                conn_str = f"file:///{abs_path}?mode=ro"
            else:
                # Linux/Docker: file:/absolute/path
                conn_str = f"file:{db_path}?mode=ro"

            conn = sqlite3.connect(conn_str, timeout=0.1, uri=True)
            
            # Load last N days of 1m data
            # Reverted to V1 (trade_candles) as requested.
            # Note: V1 lacks whale_delta, buy_count, sell_count, so we default them to 0.
            query = f"""
                SELECT timestamp, delta, 0.0 as whale_delta, volume, 0 as buy_count, 0 as sell_count
                FROM trade_candles 
                WHERE symbol='{target_key}' 
                ORDER BY timestamp ASC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            logger.warning(f"Could not load CVD data for {pair} using {db_path}: {e}")
            return DataFrame(columns=['timestamp', 'delta', 'whale_delta', 'volume', 'buy_count', 'sell_count'])

    def get_imbalance(self, current_pair):
        """
        1. Prende la coppia attuale (es. BTC/USDC:USDC)
        2. Trova il nome della moneta (BTC)
        3. Cerca nel DB i dati della liquidità USDT (BTC/USDT)
        """
        try:
            # Logica di Mapping:
            # Input: BTC/USDC:USDC oppure BTC/USDC
            # Step 1: Prendi "BTC"
            coin_name = current_pair.split('/')[0]
            
            # Step 2: Costruisci la chiave per il DB (dove salviamo sempre COIN/USDT)
            target_key = f"{coin_name}/USDT"

            conn = sqlite3.connect(f"file:{self.DB_PATH}?mode=ro", timeout=0.1, uri=True)
            cursor = conn.cursor()
            cursor.execute("SELECT imbalance, updated_at FROM orderbook_metrics WHERE symbol=?", (target_key,))
            row = cursor.fetchone()
            conn.close()

            if row:
                imbalance = row[0]
                updated_at = row[1]
                
                # Se i dati sono vecchi di 10 secondi, Rust potrebbe essere fermo -> Ignora
                if time.time() - updated_at > 10:
                    return None
                
                return imbalance
            return None

        except Exception as e:
            return None

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # --- NEW LOGIC: EXTREME ZONES & TREND SWITCHES ---
        
        # Zone Definitions
        EXTREME_GREEN = 500
        EXTREME_RED = -400
        
        norm_delta = dataframe['norm_delta']
        prev_delta = dataframe['norm_delta'].shift(1)
        prev_prev_delta = dataframe['norm_delta'].shift(2)
        
        # -------------------------------------------------------------------------
        # 1. EXTREME TO TREND ZONE CROSSOVERS
        # -------------------------------------------------------------------------
        # "anything passes from extreme red zone to trend red zone should generate a long signal"
        extreme_to_trend_long = (
            (prev_delta < EXTREME_RED) &
            (norm_delta > EXTREME_RED)
        )
        
        # "anything passes from extreme green zone to trend green zone should generate a short signal"
        extreme_to_trend_short = (
            (prev_delta > EXTREME_GREEN) &
            (norm_delta < EXTREME_GREEN)
        )
        
        # -------------------------------------------------------------------------
        # 2. TREND ZONE DIRECTIONAL SWITCHES
        # -------------------------------------------------------------------------
        # "In the trend zone, any switch of direction should generate a signal based on that switch"
        
        # Check if the turning point (prev_delta) was inside the Trend Zone (between extremes)
        in_trend_zone = (prev_delta >= EXTREME_RED) & (prev_delta <= EXTREME_GREEN)
        
        # NOISE FILTER: Require a minimum bounce magnitude to confirm the switch
        # Range is approx -1000 to 1000. 
        # A 15 unit move (approx 0.75% of range) filters 1h noise.
        # This parameter is OPTIMIZABLE via self.min_delta_change
        MIN_DELTA_CHANGE = self.min_delta_change.value
        
        # Switch to Long (Local Bottom: \/)
        # Logic: Was dropping (prev_prev > prev), now rising (prev < current)
        trend_switch_long = (
            in_trend_zone &
            (prev_prev_delta > prev_delta) &
            (norm_delta > (prev_delta + MIN_DELTA_CHANGE))
        )
        
        # Switch to Short (Local Top: /\)
        # Logic: Was rising (prev_prev < prev), now dropping (prev > current)
        trend_switch_short = (
            in_trend_zone &
            (prev_prev_delta < prev_delta) &
            (norm_delta < (prev_delta - MIN_DELTA_CHANGE))
        )
        
        # -------------------------------------------------------------------------
        # APPLY SIGNALS
        # -------------------------------------------------------------------------
        
        # TEMPORARILY DISABLED TREND SWITCH ENTRIES
        long_raw = extreme_to_trend_long # | trend_switch_long
        short_raw = extreme_to_trend_short # | trend_switch_short
        up_move = norm_delta > prev_delta
        down_move = norm_delta < prev_delta
        both_mask = long_raw & short_raw
        
        final_long = (long_raw & ~short_raw) | (both_mask & up_move)
        final_short = (short_raw & ~long_raw) | (both_mask & down_move)
        
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = ''
        
        dataframe.loc[final_long, 'enter_long'] = 1
        dataframe.loc[final_short, 'enter_short'] = 1
        
        dataframe.loc[final_long & trend_switch_long, 'enter_tag'] = 'trend_switch_long'
        dataframe.loc[final_long & (~trend_switch_long) & extreme_to_trend_long, 'enter_tag'] = 'extreme_trend_long'
        
        dataframe.loc[final_short & trend_switch_short, 'enter_tag'] = 'trend_switch_short'
        dataframe.loc[final_short & (~trend_switch_short) & extreme_to_trend_short, 'enter_tag'] = 'extreme_trend_short'

        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Get last 3 candles to detect switches
        # We need at least 3 rows
        if len(dataframe) < 3:
            return None
            
        # Look at the last closed candle pattern (iloc[-1] is current, -2 is prev, -3 is prev_prev)
        # Assuming we want to react as soon as the candle closes and the pattern is confirmed.
        # current_delta is iloc[-1] (current open candle or last closed?)
        # In Freqtrade backtesting, iloc[-1] is the current candle.
        # If process_only_new_candles=False, we are checking the forming candle.
        # A "switch" is only confirmed when the candle closes (V-shape).
        # So we should look at -1 (current), -2 (prev), -3 (prev_prev) ?
        # Actually, if we want to exit ON the signal, and the signal is generated by populate_entry_trend
        # which uses closed candle logic (shift(1), shift(2) usually referring to closed candles relative to prediction),
        # we should align.
        
        # Let's simply replicate the logic on the last closed sequence.
        # last_candle = dataframe.iloc[-1] (Current forming)
        # prev = dataframe.iloc[-2]
        # prev_prev = dataframe.iloc[-3]
        
        c = dataframe.iloc[-1]
        p = dataframe.iloc[-2]
        pp = dataframe.iloc[-3]
        
        EXTREME_GREEN = 500
        EXTREME_RED = -400
        
        # Current Delta Values
        curr_d = c['norm_delta']
        prev_d = p['norm_delta']
        pp_d = pp['norm_delta']
        
        # ---------------------------------------------------------------------
        # EXTREME EXITS (Force exit at opposite extreme)
        # ---------------------------------------------------------------------
        if trade.is_short == False:
            if curr_d > EXTREME_GREEN:
                return "long_exit_extreme_green"
        else:
            if curr_d < EXTREME_RED:
                return "short_exit_extreme_red"

        # ---------------------------------------------------------------------
        # TREND SWITCH EXITS
        # ---------------------------------------------------------------------
        # "In the trend zone, any switch of direction should generate... a long exit [if switch to short]"
        
        in_trend_zone = (prev_d >= EXTREME_RED) & (prev_d <= EXTREME_GREEN)

        # Noise Filter for Exits (Match Entry Logic)
        MIN_DELTA_CHANGE = self.min_delta_change.value
        
        # Trend Switch Short (Top / A-Shape) -> Exit Long
        # Rising then Falling
        is_switch_short = in_trend_zone and (pp_d < prev_d) and (curr_d < (prev_d - MIN_DELTA_CHANGE))
        
        if trade.is_short == False and is_switch_short:
            return "long_exit_trend_switch"
            
        # Trend Switch Long (Bottom / V-Shape) -> Exit Short
        # Falling then Rising
        is_switch_long = in_trend_zone and (pp_d > prev_d) and (curr_d > (prev_d + MIN_DELTA_CHANGE))
        
        if trade.is_short == True and is_switch_long:
            return "short_exit_trend_switch"
            
        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
