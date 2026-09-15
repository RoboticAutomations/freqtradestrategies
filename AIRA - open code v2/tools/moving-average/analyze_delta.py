import sqlite3
import pandas as pd
import numpy as np

def analyze_delta_stats():
    db_path = "user_data/orderbook_cache.db"
    symbol = "BTC/USDT"
    
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    query = f"""
        SELECT timestamp, delta, whale_delta, volume, buy_count, sell_count
        FROM trade_candles_v2 
        WHERE symbol='{symbol}' 
        ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No data found.")
        return

    # --- REPLICATE RUSTOBSTRAT LOGIC EXACTLY (1h Resampling) ---
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df.set_index('timestamp', inplace=True)
    
    # Resample to 1h (Strategy Timeframe)
    resampled = df[['delta', 'whale_delta', 'volume', 'buy_count', 'sell_count']].resample('1h').sum()
    
    # Drop empty bins
    resampled = resampled.dropna()
    
    # 1. Smooth Inputs (EMA 6)
    resampled['delta_sm'] = resampled['delta'].ewm(span=6, adjust=False).mean()
    resampled['whale_sm'] = resampled['whale_delta'].ewm(span=6, adjust=False).mean()
    
    # 2. Count Imbalance
    total_counts = resampled['buy_count'] + resampled['sell_count']
    resampled['count_ratio'] = (resampled['buy_count'] - resampled['sell_count']) / total_counts.replace(0, 1)
    resampled['count_sm'] = resampled['count_ratio'].ewm(span=6, adjust=False).mean()
    
    # 3. Z-Score (Window 24)
    vol_window = 24
    v_mean = resampled['delta_sm'].rolling(window=vol_window).mean()
    v_std = resampled['delta_sm'].rolling(window=vol_window).std().replace(0, 1)
    v_z = ((resampled['delta_sm'] - v_mean) / v_std).clip(-3, 3)
    resampled['v_norm'] = np.tanh(v_z)
    
    w_mean = resampled['whale_sm'].rolling(window=vol_window).mean()
    w_std = resampled['whale_sm'].rolling(window=vol_window).std().replace(0, 1)
    w_z = ((resampled['whale_sm'] - w_mean) / w_std).clip(-3, 3)
    resampled['w_norm'] = np.tanh(w_z)
    
    # 4. Combined Score
    raw_score = (0.4 * resampled['v_norm']) + (0.4 * resampled['w_norm']) + (0.2 * resampled['count_sm'])
    
    # 5. Final Norm Delta
    resampled['norm_delta'] = (raw_score * 1000).ewm(span=2, adjust=False).mean()
    
    # --- STATS ---
    nd = resampled['norm_delta'].dropna()
    print("--- Norm Delta Stats (1h Candles) ---")
    print(nd.describe())
    print(f"\n10th Percentile: {nd.quantile(0.10):.2f}")
    print(f"90th Percentile: {nd.quantile(0.90):.2f}")
    print(f"5th Percentile:  {nd.quantile(0.05):.2f}")
    print(f"95th Percentile: {nd.quantile(0.95):.2f}")
    print(f"1st Percentile:  {nd.quantile(0.01):.2f}")
    print(f"99th Percentile: {nd.quantile(0.99):.2f}")

if __name__ == "__main__":
    analyze_delta_stats()
