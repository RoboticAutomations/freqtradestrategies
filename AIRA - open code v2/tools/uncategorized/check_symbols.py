import sqlite3
import pandas as pd

def inspect_symbols():
    db_path = "user_data/orderbook_cache.db"
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        
        # Check trade_candles_v2 symbols
        print("--- Symbols in trade_candles_v2 ---")
        query_candles = "SELECT DISTINCT symbol FROM trade_candles_v2"
        df_candles = pd.read_sql_query(query_candles, conn)
        print(df_candles)
        
        # Check orderbook_metrics symbols
        print("\n--- Symbols in orderbook_metrics ---")
        query_metrics = "SELECT DISTINCT symbol FROM orderbook_metrics"
        df_metrics = pd.read_sql_query(query_metrics, conn)
        print(df_metrics)
        
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_symbols()
