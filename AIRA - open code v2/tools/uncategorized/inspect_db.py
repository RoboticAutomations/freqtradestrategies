import sqlite3
import pandas as pd

def check_trades():
    try:
        conn = sqlite3.connect('user_data/tradesv3.sqlite')
        query = "SELECT * FROM trades ORDER BY open_date DESC LIMIT 5"
        df = pd.read_sql_query(query, conn)
        print("Trades Columns:", df.columns.tolist())
        if not df.empty:
            # Print available columns that look like profit/exit info
            cols = ['pair', 'open_date', 'close_date', 'exit_reason', 'close_profit', 'profit_ratio', 'open_rate', 'close_rate']
            available_cols = [c for c in cols if c in df.columns]
            print(df[available_cols])
        else:
            print("No trades found.")
        conn.close()
    except Exception as e:
        print(f"Error reading trades: {e}")

def check_ob_schema():
    try:
        conn = sqlite3.connect('user_data/orderbook_cache.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("\nTables in orderbook_cache.db:", tables)
        
        for table in tables:
            table_name = table[0]
            print(f"\nSchema for {table_name}:")
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            for col in columns:
                print(col)
                
        # Check if we can get historical imbalance
        # Assuming orderbook_metrics might have history?
        cursor.execute("SELECT count(*) FROM orderbook_metrics")
        count = cursor.fetchone()[0]
        print(f"\nRow count in orderbook_metrics: {count}")
        
        cursor.execute("SELECT * FROM orderbook_metrics LIMIT 5")
        rows = cursor.fetchall()
        print("\nSample rows from orderbook_metrics:")
        for row in rows:
            print(row)
            
        conn.close()
    except Exception as e:
        print(f"Error reading orderbook_cache: {e}")

if __name__ == "__main__":
    check_trades()
    check_ob_schema()
