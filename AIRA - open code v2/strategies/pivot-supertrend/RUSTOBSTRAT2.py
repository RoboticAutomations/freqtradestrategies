import sqlite3
import time
import logging
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade
from pandas import DataFrame, to_datetime
import pandas as pd
from datetime import datetime
import talib.abstract as ta
import numpy as np

logger = logging.getLogger(__name__)

class RustOrderbookStrategy2(IStrategy):
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

    # --- STRATEGY CONFIGURATION ---
    minimal_roi = {"0": 0.1, "5": 0.05} 
    stoploss = -0.2
    
    # --- HYPEROPT PARAMETERS ---
    # trend_lookback: How many candles back to check for the switch pivot
    trend_lookback = IntParameter(1, 5, default=1, space='buy', optimize=True)
    
    # min_delta_change: Threshold for confirming a directional switch (Noise Filter)
    min_delta_change = DecimalParameter(0.0, 50.0, default=50.0, decimals=1, space='buy', optimize=True)
    
    # Trailing Stop
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = False
    
    # TIMEFRAME CONFIGURATION
    timeframe = '1h'
    
    # Dynamic Parameter Calculation Helper
    def get_window_factor(self):
        # Base is 1h (60 mins)
        tf_minutes = int(self.timeframe.replace('m', '').replace('h', '60')) if 'h' not in self.timeframe or self.timeframe == '1h' else 60
        if 'h' in self.timeframe and self.timeframe != '1h':
            tf_minutes = int(self.timeframe.replace('h', '')) * 60
            
        # Factor: 1h / Current TF
        return 60 / tf_minutes
    
    # Process only new candles must be False to allow partial candle analysis if needed, 
    # though for this logic, closed candles are safer.
    process_only_new_candles = False
    
    can_short = True

    # DCA / Position Adjustment
    position_adjustment_enable = True
    max_entry_position_adjustment = 5
    
    # Path del DB seen from inside the Freqtrade container
    DB_PATH = "/freqtrade/user_data/orderbook_cache.db"
    
    # Parameters
    cvd_lookback = 12 
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
        additional_stake = trade.stake_amount / trade.nr_of_successful_entries

        # Long DCA:
        # Trigger: Red Extreme Zone (<-400)
        # Condition: Price is dropping (profit < 0) and we are in extreme red
        if trade.is_short == False:
            if (current_profit < -0.02) and (current_delta < -400):
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
            cvd_df['timestamp'] = pd.to_datetime(cvd_df['timestamp'], unit='s', utc=True)
            cvd_df.set_index('timestamp', inplace=True)
            
            # Resample: Sum Delta, Volume, Counts
            resampled = cvd_df[['delta', 'whale_delta', 'volume', 'buy_count', 'sell_count']].resample(self.timeframe).sum()
            
            # 3. Calculate Cumulative Delta
            cvd_series = resampled['delta'].cumsum()
            whale_cvd_series = resampled['whale_delta'].cumsum()
            
            # 4. Merge into Main DataFrame
            dataframe = dataframe.merge(cvd_series.rename('cvd_absolute'), left_on='date', right_index=True, how='left')
            dataframe = dataframe.merge(whale_cvd_series.rename('whale_cvd_absolute'), left_on='date', right_index=True, how='left')
            
            # Merge auxiliary metrics for Normalization
            dataframe = dataframe.merge(resampled[['delta', 'whale_delta', 'volume', 'buy_count', 'sell_count']], left_on='date', right_index=True, how='left', suffixes=('', '_res'))
            
            # Fill NaNs
            dataframe['cvd_absolute'] = dataframe['cvd_absolute'].ffill().fillna(0)
            dataframe['whale_cvd_absolute'] = dataframe['whale_cvd_absolute'].ffill().fillna(0)
            
            cols_to_fill = ['delta', 'whale_delta', 'volume', 'buy_count', 'sell_count']
            dataframe[cols_to_fill] = dataframe[cols_to_fill].fillna(0)

            # --- ROBUST NORMALIZATION ---
            
            # Dynamic Window Calculation
            factor = self.get_window_factor()
            
            ema_period = max(2, int(6 * factor))  
            vol_window = max(2, int(24 * factor)) 
            reversal_window = max(2, int(12 * factor)) 

            # 1. Smooth Inputs (EMA ~6 hours)
            dataframe['delta_sm'] = ta.EMA(dataframe['delta'], timeperiod=ema_period)
            dataframe['whale_sm'] = ta.EMA(dataframe['whale_delta'], timeperiod=ema_period)
            
            # 2. Count Imbalance (Smoothed)
            total_counts = dataframe['buy_count'] + dataframe['sell_count']
            dataframe['count_ratio'] = (dataframe['buy_count'] - dataframe['sell_count']) / total_counts.replace(0, 1)
            dataframe['count_sm'] = ta.EMA(dataframe['count_ratio'], timeperiod=ema_period)

            # 3. Robust Z-Score for Volume Delta
            v_mean = dataframe['delta_sm'].rolling(window=vol_window).mean()
            v_std = dataframe['delta_sm'].rolling(window=vol_window).std().replace(0, 1)
            v_z = ((dataframe['delta_sm'] - v_mean) / v_std).clip(-3, 3)
            dataframe['v_norm'] = np.tanh(v_z) 

            # 4. Robust Z-Score for Whale Delta
            w_mean = dataframe['whale_sm'].rolling(window=vol_window).mean()
            w_std = dataframe['whale_sm'].rolling(window=vol_window).std().replace(0, 1)
            w_z = ((dataframe['whale_sm'] - w_mean) / w_std).clip(-3, 3)
            dataframe['w_norm'] = np.tanh(w_z)

            # 5. Combined Score
            raw_score = (0.4 * dataframe['v_norm']) + (0.4 * dataframe['w_norm']) + (0.2 * dataframe['count_sm'])
            
            # 6. Final Smoothing & Scaling
            final_ema = max(2, int(2 * factor))
            dataframe['norm_delta'] = ta.EMA(raw_score * 1000, timeperiod=final_ema)
            
            # Calculate recent extremes for Reversal Logic
            dataframe['recent_min'] = dataframe['norm_delta'].rolling(window=reversal_window).min()
            dataframe['recent_max'] = dataframe['norm_delta'].rolling(window=reversal_window).max()
            
            # 5. Smoothing (EMA)
            dataframe['cvd_ema'] = ta.EMA(dataframe['cvd_absolute'], timeperiod=self.cvd_ema_period)
            
            # Helpers
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
            dataframe['norm_delta'] = 0.0
            dataframe['w_norm'] = 0.0
            dataframe['recent_min'] = 0.0
            dataframe['recent_max'] = 0.0
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
        try:
            coin_name = pair.split('/')[0]
            target_key = f"{coin_name}/USDT"
            
            conn = sqlite3.connect(f"file:{self.DB_PATH}?mode=ro", timeout=0.1, uri=True)
            
            # Load last N days of 1m data
            query = f"""
                SELECT timestamp, delta, whale_delta, volume, buy_count, sell_count
                FROM trade_candles_v2 
                WHERE symbol='{target_key}' 
                ORDER BY timestamp ASC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            return DataFrame(columns=['timestamp', 'delta', 'whale_delta', 'volume', 'buy_count', 'sell_count'])

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Zone Definitions
        EXTREME_GREEN = 500
        EXTREME_RED = -400
        
        # Current and Previous Deltas
        norm_delta = dataframe['norm_delta']
        prev_delta = dataframe['norm_delta'].shift(1)
        prev_prev_delta = dataframe['norm_delta'].shift(2)
        
        # -------------------------------------------------------------------------
        # 1. EXTREME REVERSION (The "Bounce")
        # -------------------------------------------------------------------------
        # Long: Price was below -400, now crosses back above -400
        extreme_to_trend_long = (
            (prev_delta < EXTREME_RED) &
            (norm_delta > EXTREME_RED)
        )
        
        # Short: Price was above 500, now crosses back below 500
        extreme_to_trend_short = (
            (prev_delta > EXTREME_GREEN) &
            (norm_delta < EXTREME_GREEN)
        )
        
        # -------------------------------------------------------------------------
        # 2. TREND ZONE SWITCHES (The "Pivot")
        # -------------------------------------------------------------------------
        # Only valid if the previous candle was inside the "Normal" zone
        in_trend_zone = (prev_delta >= EXTREME_RED) & (prev_delta <= EXTREME_GREEN)
        
        # Noise Filter: The pivot must have a magnitude > min_delta_change
        MIN_DELTA_CHANGE = self.min_delta_change.value
        
        # Switch to Long (V-Shape: Down -> Up)
        # Logic: (PrevPrev > Prev) AND (Current > Prev + Threshold)
        trend_switch_long = (
            in_trend_zone &
            (prev_prev_delta > prev_delta) &
            (norm_delta > (prev_delta + MIN_DELTA_CHANGE))
        )
        
        # Switch to Short (A-Shape: Up -> Down)
        # Logic: (PrevPrev < Prev) AND (Current < Prev - Threshold)
        trend_switch_short = (
            in_trend_zone &
            (prev_prev_delta < prev_delta) &
            (norm_delta < (prev_delta - MIN_DELTA_CHANGE))
        )
        
        # -------------------------------------------------------------------------
        # APPLY SIGNALS
        # -------------------------------------------------------------------------
        # Combine triggers: Either we bounced off extreme OR we switched trend
        long_signal = extreme_to_trend_long | trend_switch_long
        short_signal = extreme_to_trend_short | trend_switch_short
        
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = ''
        
        # Apply Long
        dataframe.loc[long_signal, 'enter_long'] = 1
        dataframe.loc[extreme_to_trend_long, 'enter_tag'] += 'long_extreme_rev '
        dataframe.loc[trend_switch_long, 'enter_tag'] += 'long_trend_switch '

        # Apply Short
        dataframe.loc[short_signal, 'enter_short'] = 1
        dataframe.loc[extreme_to_trend_short, 'enter_tag'] += 'short_extreme_rev '
        dataframe.loc[trend_switch_short, 'enter_tag'] += 'short_trend_switch '

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Zone Definitions
        EXTREME_GREEN = 500
        EXTREME_RED = -400
        
        norm_delta = dataframe['norm_delta']
        prev_delta = dataframe['norm_delta'].shift(1)
        prev_prev_delta = dataframe['norm_delta'].shift(2)
        
        MIN_DELTA_CHANGE = self.min_delta_change.value
        in_trend_zone = (prev_delta >= EXTREME_RED) & (prev_delta <= EXTREME_GREEN)

        # -------------------------------------------------------------------------
        # 1. EXTREME TARGET REACHED
        # -------------------------------------------------------------------------
        # If Long: Exit when we hit the Short Extreme (Green Zone)
        exit_long_extreme = (norm_delta > EXTREME_GREEN)
        
        # If Short: Exit when we hit the Long Extreme (Red Zone)
        exit_short_extreme = (norm_delta < EXTREME_RED)

        # -------------------------------------------------------------------------
        # 2. TREND FAILURE (Opposite Switch)
        # -------------------------------------------------------------------------
        # If Long: Exit if Delta makes a "Short Switch" (A-Shape)
        exit_long_switch = (
            in_trend_zone &
            (prev_prev_delta < prev_delta) &
            (norm_delta < (prev_delta - MIN_DELTA_CHANGE))
        )
        
        # If Short: Exit if Delta makes a "Long Switch" (V-Shape)
        exit_short_switch = (
            in_trend_zone &
            (prev_prev_delta > prev_delta) &
            (norm_delta > (prev_delta + MIN_DELTA_CHANGE))
        )

        # -------------------------------------------------------------------------
        # APPLY EXITS
        # -------------------------------------------------------------------------
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = ''

        # Long Exits
        long_exit_cond = exit_long_extreme | exit_long_switch
        dataframe.loc[long_exit_cond, 'exit_long'] = 1
        dataframe.loc[exit_long_extreme, 'exit_tag'] += 'long_exit_extreme '
        dataframe.loc[exit_long_switch, 'exit_tag'] += 'long_exit_switch '

        # Short Exits
        short_exit_cond = exit_short_extreme | exit_short_switch
        dataframe.loc[short_exit_cond, 'exit_short'] = 1
        dataframe.loc[exit_short_extreme, 'exit_tag'] += 'short_exit_extreme '
        dataframe.loc[exit_short_switch, 'exit_tag'] += 'short_exit_switch '

        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        # Technical exits are now handled in populate_exit_trend.
        # This prevents mid-candle repainting and ensures consistent backtesting.
        return None