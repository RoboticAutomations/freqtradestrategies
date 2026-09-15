import sqlite3
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

DB_PATH = 'user_data/orderbook_cache.db'

def seed_database():
    print(f"Seeding database at {DB_PATH}...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create V2 Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_candles_v2 (
            symbol TEXT,
            timestamp INTEGER,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            delta REAL,
            buy_vol REAL,
            sell_vol REAL,
            buy_count INTEGER,
            sell_count INTEGER,
            whale_buy_vol REAL,
            whale_sell_vol REAL,
            whale_delta REAL,
            PRIMARY KEY (symbol, timestamp)
        )
    """)
    
    # Generate 5 days of 1-minute data for ETH/USDT and BTC/USDT
    symbols = ['ETH/USDT', 'BTC/USDT']
    end_time = int(time.time())
    start_time = end_time - (5 * 24 * 60 * 60) # 5 days ago
    
    # Create time range (every minute)
    timestamps = range(start_time, end_time, 60)
    
    for symbol in symbols:
        print(f"Generating data for {symbol}...")
        data = []
        
        # Simple random walk for price and delta
        price = 2000.0 if 'ETH' in symbol else 60000.0
        
        for ts in timestamps:
            # Random price movement
            change = np.random.normal(0, price * 0.0005) # 0.05% volatility
            price += change
            
            # Random volume
            volume = np.abs(np.random.normal(100, 50))
            
            # Random Delta (correlated with price change slightly)
            delta_bias = 1 if change > 0 else -1
            delta = (volume * 0.3 * delta_bias) + np.random.normal(0, volume * 0.1)
            
            # Bound delta
            delta = max(min(delta, volume), -volume)
            
            buy_vol = (volume + delta) / 2
            sell_vol = (volume - delta) / 2
            
            # Whale Delta (correlated but more sparse/exaggerated)
            whale_delta = delta * 0.6 + np.random.normal(0, volume * 0.05)
            
            data.append((
                symbol, ts, 
                price, price + abs(change), price - abs(change), price, # OHLC (simplified)
                volume, delta, buy_vol, sell_vol,
                int(volume/10), int(volume/10), # buy/sell count
                buy_vol*0.5, sell_vol*0.5, whale_delta
            ))
            
        # Bulk insert
        cursor.executemany("""
            INSERT OR REPLACE INTO trade_candles_v2 
            (symbol, timestamp, open, high, low, close, volume, delta, buy_vol, sell_vol,
             buy_count, sell_count, whale_buy_vol, whale_sell_vol, whale_delta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        
        conn.commit()
        
    print("Seeding complete.")
    conn.close()

if __name__ == "__main__":
    seed_database()
