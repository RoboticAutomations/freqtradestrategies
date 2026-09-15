import sqlite3
import matplotlib.pyplot as plt
import argparse
import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime

def read_data(db_path, symbol):
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return [], [], [], [], [], [], []

    # Copy DB to temp to avoid WAL locking/visibility issues on Windows/Docker
    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "temp_cvd.db")
    
    conn = None
    try:
        shutil.copy2(db_path, temp_db_path)
        if os.path.exists(db_path + "-wal"):
            shutil.copy2(db_path + "-wal", temp_db_path + "-wal")
        if os.path.exists(db_path + "-shm"):
            shutil.copy2(db_path + "-shm", temp_db_path + "-shm")

        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        # Handle symbol format (e.g. BTC/USDT)
        symbol = symbol.upper()
        if "/" not in symbol and "USDT" in symbol:
            symbol = symbol.replace("USDT", "/USDT")
        
        # Read from trade_candles (v1) ONLY
        rows = []
        try:
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trade_candles'")
            if cursor.fetchone():
                # v1 columns: symbol, timestamp, open, high, low, close, volume, delta, buy_vol, sell_vol
                # We need: timestamp, close, delta, whale_delta, volume, buy_count, sell_count
                # Note: v1 does not have whale_delta, buy_count, sell_count usually. 
                # We map them to 0.0/0 to match the expected format.
                query_v1 = """
                    SELECT timestamp, close, delta, 0.0 as whale_delta, volume, 0 as buy_count, 0 as sell_count
                    FROM trade_candles 
                    WHERE symbol = ?
                """
                cursor.execute(query_v1, (symbol,))
                rows = cursor.fetchall()
        except Exception as e:
            print(f"Error reading v1: {e}")

        
    except Exception as e:
        print(f"Error querying database: {e}")
        return [], [], [], [], [], [], []
    finally:
        if conn:
            conn.close()
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    
    times = []
    prices = []
    deltas = []
    whale_deltas = []
    volumes = []
    buy_counts = []
    sell_counts = []
    
    for r in rows:
        times.append(datetime.fromtimestamp(r[0]))
        prices.append(r[1])
        deltas.append(r[2])
        whale_deltas.append(r[3] if r[3] is not None else 0.0)
        volumes.append(r[4])
        buy_counts.append(r[5])
        sell_counts.append(r[6])
        
    return times, prices, deltas, whale_deltas, volumes, buy_counts, sell_counts

def plot_cvd(times, prices, deltas, whale_deltas, volumes, buy_counts, sell_counts, symbol, out_path, min_delta_change=15.0, lookback=60):
    if not times:
        print(f"No data found for {symbol}")
        return

    # Create DataFrame for easier calculation
    df = pd.DataFrame({
        'time': times,
        'price': prices,
        'delta': deltas,
        'whale_delta': whale_deltas,
        'volume': volumes,
        'buy_count': buy_counts,
        'sell_count': sell_counts
    })
    
    # --- CALCULATE INDICATORS (MATCHING STRATEGY LOGIC) ---
    df['cvd'] = df['delta'].cumsum()
    df['whale_cvd'] = df['whale_delta'].cumsum()
    df['cvd_ema'] = df['cvd'].ewm(span=12, adjust=False).mean()
    
    # --- ROBUST NORMALIZATION (LONG SWINGS) ---
    # 1. Smooth Inputs (STABLE TREND)
    # Strategy EMA 6 on 1h data = 6 hours.
    # On 1m data, span=360 is approx 6h smoothing. 
    # This filters out all intraday noise and focuses on the multi-day trend.
    df['delta_sm'] = df['delta'].ewm(span=360, adjust=False).mean()
    df['whale_sm'] = df['whale_delta'].ewm(span=360, adjust=False).mean()
    
    # 2. Count Imbalance (Smoothed)
    total_counts = df['buy_count'] + df['sell_count']
    df['count_ratio'] = (df['buy_count'] - df['sell_count']) / total_counts.replace(0, 1)
    df['count_sm'] = df['count_ratio'].ewm(span=360, adjust=False).mean()

    # 3. Robust Z-Score for Volume Delta
    # Window = 1440 (24 hours context)
    vol_window = 1440
    v_mean = df['delta_sm'].rolling(window=vol_window).mean()
    v_std = df['delta_sm'].rolling(window=vol_window).std().replace(0, 1)
    
    # Clip Z-score to +/- 3
    v_z = ((df['delta_sm'] - v_mean) / v_std).clip(-3, 3)
    df['v_norm'] = np.tanh(v_z) 

    # 4. Robust Z-Score for Whale Delta
    w_mean = df['whale_sm'].rolling(window=vol_window).mean()
    w_std = df['whale_sm'].rolling(window=vol_window).std().replace(0, 1)
    w_z = ((df['whale_sm'] - w_mean) / w_std).clip(-3, 3)
    df['w_norm'] = np.tanh(w_z)

    # 5. Combined Score
    # Weighting: 40% Volume, 40% Whale, 20% Trade Counts
    
    # DYNAMIC WEIGHTING (Match Strategy Logic)
    has_whale = df['whale_delta'].abs().sum() > 0
    has_counts = df['buy_count'].sum() > 0
    
    w_vol = 0.4
    w_whale = 0.4
    w_count = 0.2
    
    if not has_whale:
        w_vol += w_whale
        w_whale = 0.0
        
    if not has_counts:
        w_vol += w_count
        w_count = 0.0
        
    raw_score = (w_vol * df['v_norm']) + (w_whale * df['w_norm']) + (w_count * df['count_sm'])
    
    # 6. Final Smoothing & Scaling
    # Scale to -1000..1000 and smooth output
    # Strategy uses EMA(2) on 1h data (~2h influence).
    # On 1m data, span=120 is approx 2h.
    df['norm_delta'] = (raw_score * 1000).ewm(span=120, adjust=False).mean()

    # --- SIMULATE SIGNALS (SMART EXTREMES V2) ---
    
    # 1. EXTREME ZONES (Visualization & Filters)
    # Green Zone: > 600 (Extreme Buying / Overbought)
    # Red Zone: < -600 (Extreme Selling / Oversold)
    
    # 2. TREND SIGNALS (Middle Band Only)
    # Define recent min/max for Trend Logic
    # Strategy uses 12h window. On 1m data, that's 720 periods.
    df['recent_min'] = df['norm_delta'].rolling(window=720).min()
    df['recent_max'] = df['norm_delta'].rolling(window=720).max()

    # ═══════════════════════════════════════════════════════════════════
    # REPLICATE STRATEGY LOGIC EXACTLY (ZONE RECOVERY)
    # ═══════════════════════════════════════════════════════════════════
    
    # Zones
    ZONE_EXTREME_HIGH = 500
    ZONE_HIGH = 400
    ZONE_ZERO = 0
    ZONE_LOW = -300
    ZONE_EXTREME_LOW = -400
    
    # Data
    norm_delta = df['norm_delta']
    LOOKBACK = lookback
    prev_delta = df['norm_delta'].shift(LOOKBACK)
    prev_prev_delta = df['norm_delta'].shift(LOOKBACK * 2)
    w_norm = df['w_norm'] # Whale Z-Score
    
    # -------------------------------------------------------------------------
    # 1. EXTREME TO TREND ZONE CROSSOVERS
    # -------------------------------------------------------------------------
    extreme_to_trend_long = (
        (prev_delta < ZONE_EXTREME_LOW) &
        (norm_delta >= ZONE_EXTREME_LOW)
    )
    
    extreme_to_trend_short = (
        (prev_delta > ZONE_EXTREME_HIGH) &
        (norm_delta <= ZONE_EXTREME_HIGH)
    )
    
    # -------------------------------------------------------------------------
    # 2. TREND ZONE DIRECTIONAL SWITCHES
    # -------------------------------------------------------------------------
    in_trend_zone = (prev_delta >= ZONE_EXTREME_LOW) & (prev_delta <= ZONE_EXTREME_HIGH)
    
    # NOISE FILTER
    MIN_DELTA_CHANGE = float(min_delta_change)
    
    # Switch to Long (Local Bottom: \/)
    trend_switch_long = (
        in_trend_zone &
        (prev_prev_delta > prev_delta) &
        (norm_delta > (prev_delta + MIN_DELTA_CHANGE))
    )
    
    # Switch to Short (Local Top: /\)
    trend_switch_short = (
        in_trend_zone &
        (prev_prev_delta < prev_delta) &
        (norm_delta < (prev_delta - MIN_DELTA_CHANGE))
    )

    # -------------------------------------------------------------------------
    # APPLY SIGNALS
    # -------------------------------------------------------------------------
    df['reversal_long'] = 0
    df['trend_long'] = 0
    df['reversal_short'] = 0
    df['trend_short'] = 0
    
    df['enter_long_signal'] = 0
    df['enter_short_signal'] = 0
    df['exit_long_signal'] = 0
    df['exit_short_signal'] = 0
    
    # Enforce mutual exclusivity per candle
    # TEMPORARILY DISABLED TREND SWITCH ENTRIES
    long_raw = extreme_to_trend_long # | trend_switch_long
    short_raw = extreme_to_trend_short # | trend_switch_short
    up_move = norm_delta > prev_delta
    down_move = norm_delta < prev_delta
    both_mask = long_raw & short_raw
    
    final_long = (long_raw & ~short_raw) | (both_mask & up_move)
    final_short = (short_raw & ~long_raw) | (both_mask & down_move)
    
    df.loc[:, 'reversal_long'] = 0
    df.loc[:, 'trend_long'] = 0
    df.loc[:, 'reversal_short'] = 0
    df.loc[:, 'trend_short'] = 0
    df.loc[:, 'enter_long_signal'] = 0
    df.loc[:, 'enter_short_signal'] = 0
    
    df.loc[final_long, 'enter_long_signal'] = 1
    df.loc[final_short, 'enter_short_signal'] = 1
    
    # Tag classification
    df.loc[final_long & trend_switch_long, 'trend_long'] = 1
    df.loc[final_long & extreme_to_trend_long & (~trend_switch_long), 'reversal_long'] = 1
    
    df.loc[final_short & trend_switch_short, 'trend_short'] = 1
    df.loc[final_short & extreme_to_trend_short & (~trend_switch_short), 'reversal_short'] = 1
    
    # -------------------------------------------------------------------------
    # DYNAMIC EXITS (Risk Management)
    # -------------------------------------------------------------------------
    
    # Exit Longs: 
    # 1. Extreme Profit (> 500) (Green Extreme)
    df.loc[
        (norm_delta > 500),
        'exit_long_signal'
    ] = 1
    # 2. Trend Switch Short (Top) -> Exit Long
    df.loc[trend_switch_short, 'exit_long_signal'] = 1
    
    # Exit Shorts: 
    # 1. Extreme Profit (< -400) (Red Extreme)
    df.loc[
        (norm_delta < -400),
        'exit_short_signal'
    ] = 1
    # 2. Trend Switch Long (Bottom) -> Exit Short
    df.loc[trend_switch_long, 'exit_short_signal'] = 1

    # COMBINE SIGNALS & SIMULATE TRADES (Loop for Accuracy)
    # We need to simulate ROI exits and Logic exits properly.
    # Vectorized approaches fail for "Stateful" logic like ROI.
    
    positions = np.zeros(len(df))
    exit_longs = np.zeros(len(df))
    exit_shorts = np.zeros(len(df))
    enter_longs = np.zeros(len(df))
    enter_shorts = np.zeros(len(df))
    
    # State Variables
    current_pos = 0 # 0, 1, -1
    entry_price = 0.0
    entry_type = None # 'trend', 'reversal'
    
    # Config
    ROI_TARGET = 0.03 # 3% Profit Target for Reversals
    STOPLOSS = -0.10 # 10% Stoploss (Safety)

    # Trailing Stop Config
    TRAILING_STOP = False
    TRAILING_STOP_POSITIVE = 0.01
    TRAILING_STOP_POSITIVE_OFFSET = 0.02
    TRAILING_ONLY_OFFSET_IS_REACHED = True
    
    # Arrays for speed
    closes = df['price'].values
    norm_delta_vals = df['norm_delta'].values # Need this for custom exits
    enter_long_sig = df['enter_long_signal'].values
    enter_short_sig = df['enter_short_signal'].values
    exit_long_sig = df['exit_long_signal'].values
    exit_short_sig = df['exit_short_signal'].values
    
    # For Classification
    is_rev_long = df['reversal_long'].values
    # is_zone_long = df['zone_long'].values # Removed
    is_rev_short = df['reversal_short'].values
    # is_zone_short = df['zone_short'].values # Removed

    # State for Trailing Stop
    max_profit = -999.0
    
    for i in range(1, len(df)):
        # 1. CHECK EXITS IF IN POSITION
        if current_pos == 1:
            roi = (closes[i] - entry_price) / entry_price
            
            # Update Max Profit
            if roi > max_profit:
                max_profit = roi
                
            # Trailing Stop Logic
            trailing_exit = False
            if TRAILING_STOP:
                # If offset required, check if we hit it
                if TRAILING_ONLY_OFFSET_IS_REACHED:
                    if max_profit >= TRAILING_STOP_POSITIVE_OFFSET:
                        # Triggered. Stop is max_profit - callback
                        if roi <= (max_profit - TRAILING_STOP_POSITIVE):
                            trailing_exit = True
                else:
                    # Always trailing
                    if roi <= (max_profit - TRAILING_STOP_POSITIVE):
                        trailing_exit = True
            
            # Logic Exit (Split Logic)
            # Standard Exit
            logic_exit = (exit_long_sig[i] == 1)
            
            # Execute Exit
            if roi >= ROI_TARGET or roi <= STOPLOSS or logic_exit or trailing_exit:
                exit_longs[i] = 1
                current_pos = 0
                entry_price = 0.0
                entry_type = None
                max_profit = -999.0
        
        elif current_pos == -1:
            roi = (entry_price - closes[i]) / entry_price
            
            # Update Max Profit
            if roi > max_profit:
                max_profit = roi
                
            # Trailing Stop Logic
            trailing_exit = False
            if TRAILING_STOP:
                if TRAILING_ONLY_OFFSET_IS_REACHED:
                    if max_profit >= TRAILING_STOP_POSITIVE_OFFSET:
                        if roi <= (max_profit - TRAILING_STOP_POSITIVE):
                            trailing_exit = True
                else:
                    if roi <= (max_profit - TRAILING_STOP_POSITIVE):
                        trailing_exit = True
            
            # Logic Exit (Split Logic)
            # Standard Exit
            logic_exit = (exit_short_sig[i] == 1)
            
            # Execute Exit
            if roi >= ROI_TARGET or roi <= STOPLOSS or logic_exit or trailing_exit:
                exit_shorts[i] = 1
                current_pos = 0
                entry_price = 0.0
                entry_type = None
                max_profit = -999.0

        # 2. CHECK ENTRIES IF FLAT (Priority to Reversals if both trigger)
        if current_pos == 0:
            if enter_long_sig[i] == 1:
                current_pos = 1
                entry_price = closes[i]
                max_profit = -999.0 # Reset
                # Determine Type
                if is_rev_long[i]:
                    entry_type = 'reversal'
                else:
                    entry_type = 'trend'
                
                enter_longs[i] = 1
                
            elif enter_short_sig[i] == 1:
                current_pos = -1
                entry_price = closes[i]
                max_profit = -999.0 # Reset
                # Determine Type
                if is_rev_short[i]:
                    entry_type = 'reversal'
                else:
                    entry_type = 'trend'
                    
                enter_shorts[i] = 1
        
        positions[i] = current_pos

    df['position'] = positions
    df['enter_long'] = enter_longs
    df['enter_short'] = enter_shorts
    df['exit_long'] = exit_longs
    df['exit_short'] = exit_shorts
    
    # Classify Entries for Plotting
    df['is_reversal_long'] = (df['enter_long'] == 1) & (df['reversal_long'] == 1)
    df['is_trend_long'] = (df['enter_long'] == 1) & (df['trend_long'] == 1)
    
    df['is_reversal_short'] = (df['enter_short'] == 1) & (df['reversal_short'] == 1)
    df['is_trend_short'] = (df['enter_short'] == 1) & (df['trend_short'] == 1)
    
    # Propagate 'is_reversal' flags for visualization (stateful)
    # (Leaving original logic commented out or removed as loop handles state)

    print(f"Total Long Entries: {df['enter_long'].sum()}")
    print(f"  - Reversal Longs: {df['is_reversal_long'].sum()}")
    print(f"  - Trend Longs: {df['is_trend_long'].sum()}")
    print(f"Total Short Entries: {df['enter_short'].sum()}")
    print(f"  - Reversal Shorts: {df['is_reversal_short'].sum()}")
    print(f"  - Trend Shorts: {df['is_trend_short'].sum()}")
    
    # Exits for Visualization (Matched to Custom Exit)
    # 1. Initialize Exits
    df['exit_long'] = 0
    df['exit_short'] = 0
    
    # 2. Identify Entry Type for current position state
    # We need to know if the current active position was triggered by Reversal or Trend.
    # We can forward fill the 'is_reversal' flag while position is active.
    
    # Propagate 'is_reversal' flags
    # Use float for ffill to avoid object dtype deprecation
    df['active_reversal_long'] = df['is_reversal_long'].astype(float).replace(0.0, np.nan)
    df['active_reversal_long'] = df['active_reversal_long'].ffill()
    df['active_reversal_long'] = np.where(df['position'] == 1, df['active_reversal_long'].fillna(0.0), 0.0).astype(bool)
    
    df['active_reversal_short'] = df['is_reversal_short'].astype(float).replace(0.0, np.nan)
    df['active_reversal_short'] = df['active_reversal_short'].ffill()
    df['active_reversal_short'] = np.where(df['position'] == -1, df['active_reversal_short'].fillna(0.0), 0.0).astype(bool)
    
    # 3. Apply Exit Logic (Handled in Loop)
    # The loop already populated exit_long and exit_short.
    # We just need to ensure the plotting logic uses these columns.
    
    # (Previous vectorized logic removed)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # --- Price Chart ---
    ax1.plot(df['time'], df['price'], label='Price', color='#333333', linewidth=1.5)
    
    # Plot Entries
    rev_long = df[df['is_reversal_long']]
    trend_long = df[df['is_trend_long']]
    
    rev_short = df[df['is_reversal_short']]
    trend_short = df[df['is_trend_short']]
    
    # Plot Exits
    exits_long = df[df['exit_long'] == 1]
    exits_short = df[df['exit_short'] == 1]
    
    # Make Reversals VERY distinct (Big Hollow Markers with Thick Edges)
    ax1.scatter(rev_long['time'], rev_long['price'], marker='^', facecolors='none', edgecolors='#00FF00', linewidth=3, s=300, label='Reversal Long (Extreme)', zorder=10)
    ax1.scatter(trend_long['time'], trend_long['price'], marker='^', color='green', s=100, label='Trend Long (Mid)', zorder=5)
    
    ax1.scatter(rev_short['time'], rev_short['price'], marker='v', facecolors='none', edgecolors='#FF0000', linewidth=3, s=300, label='Reversal Short (Extreme)', zorder=10)
    ax1.scatter(trend_short['time'], trend_short['price'], marker='v', color='red', s=100, label='Trend Short (Mid)', zorder=5)
    
    # Plot Exits
    ax1.scatter(exits_long['time'], exits_long['price'], marker='x', color='black', s=100, label='Exit Long', zorder=20)
    ax1.scatter(exits_short['time'], exits_short['price'], marker='x', color='black', s=100, label='Exit Short', zorder=20)

    ax1.set_title(f'{symbol} Smart Extremes Analysis', fontsize=14)
    ax1.set_ylabel('Price (USDT)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # --- CVD Chart ---
    ax2.plot(df['time'], df['cvd'], label='Total CVD', color='#2196F3', linewidth=1.5)
    ax2.plot(df['time'], df['whale_cvd'], label='Whale CVD (>10k)', color='#9C27B0', linewidth=1.5, linestyle='-')
    ax2.set_ylabel('Cumulative Delta')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # --- Normalized Delta Chart ---
    ax3.plot(df['time'], df['norm_delta'], label='Normalized Delta', color='#E91E63', linewidth=1.5)
    
    # Add Band & Zones (Restored & Enhanced)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.axhline(y=500, color='green', linestyle='--', alpha=0.5)
    ax3.axhline(y=-400, color='red', linestyle='--', alpha=0.5)
    
    # 1. Extreme Zones Shading (Strongest)
    ax3.fill_between(df['time'], df['norm_delta'], 500, where=(df['norm_delta'] > 500), facecolor='green', alpha=0.4, label='Extreme Buy (Reverse Short)')
    ax3.fill_between(df['time'], df['norm_delta'], -400, where=(df['norm_delta'] < -400), facecolor='red', alpha=0.4, label='Extreme Sell (Reverse Long)')
    
    # 2. Trend Zones Shading (The "Visual Zone" Request)
    # Long Trend Zone: 0 to 500
    ax3.fill_between(df['time'], 0, df['norm_delta'], where=((df['norm_delta'] > 0) & (df['norm_delta'] <= 500)), facecolor='green', alpha=0.1, label='Trend Long Zone')
    
    # Short Trend Zone: 0 to -400
    ax3.fill_between(df['time'], 0, df['norm_delta'], where=((df['norm_delta'] < 0) & (df['norm_delta'] >= -400)), facecolor='red', alpha=0.1, label='Trend Short Zone')
    
    ax3.set_ylim(-1000, 1000) # Zoom out slightly to show extremes
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    plt.tight_layout()
    
    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Cumulative Volume Delta from SQLite DB")
    parser.add_argument("--db", default="user_data/orderbook_cache.db", help="Path to SQLite DB")
    parser.add_argument("--symbol", required=True, help="Symbol (e.g., BTC/USDT)")
    parser.add_argument("--out", help="Output file path (e.g., plot/cvd.png)")
    parser.add_argument("--mindc", type=float, default=1.0, help="Min delta change for switches")
    parser.add_argument("--lookback", type=int, default=60, help="Minutes to look back for prev candles")
    args = parser.parse_args()
    
    print(f"Reading data for {args.symbol} from {args.db}...")
    times, prices, deltas, whale_deltas, volumes, buy_counts, sell_counts = read_data(args.db, args.symbol)
    if times:
        print(f"Found {len(times)} candles. Plotting...")
        plot_cvd(times, prices, deltas, whale_deltas, volumes, buy_counts, sell_counts, args.symbol, args.out, args.mindc, args.lookback)
    else:
        print("No data to plot.")
