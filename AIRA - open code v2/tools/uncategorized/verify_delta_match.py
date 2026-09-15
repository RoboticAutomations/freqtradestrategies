import sqlite3
import shutil
import tempfile
import os

db_path = "user_data/orderbook_cache.db"
if not os.path.exists(db_path):
    print("DB not found")
    exit(1)

temp_dir = tempfile.mkdtemp()
temp_db = os.path.join(temp_dir, "temp_match.db")

try:
    shutil.copy2(db_path, temp_db)
    if os.path.exists(db_path + "-wal"):
        shutil.copy2(db_path + "-wal", temp_db + "-wal")
        
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    
    ts = 1767712020
    query = f"SELECT delta FROM trade_candles WHERE symbol='ETH/USDT' AND timestamp={ts}"
    cursor.execute(query)
    row = cursor.fetchone()
    if row:
        print(f"Timestamp {ts}: Delta = {row[0]}")
    else:
        print(f"Timestamp {ts} not found in trade_candles")
        
finally:
    try:
        conn.close()
    except:
        pass
    shutil.rmtree(temp_dir)
