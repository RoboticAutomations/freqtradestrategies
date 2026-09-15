import sqlite3
import shutil
import tempfile
import os

db_path = "user_data/orderbook_cache.db"

if not os.path.exists(db_path):
    print("DB not found")
    exit(1)

temp_dir = tempfile.mkdtemp()
temp_db = os.path.join(temp_dir, "temp_verify.db")

try:
    print("Copying DB to temp...")
    shutil.copy2(db_path, temp_db)
    if os.path.exists(db_path + "-wal"):
        shutil.copy2(db_path + "-wal", temp_db + "-wal")
        
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    
    print("\n--- Schema of trade_candles ---")
    cursor.execute("PRAGMA table_info(trade_candles)")
    columns = cursor.fetchall()
    for col in columns:
        print(col)

    print("\n--- Schema of trade_candles_v2 ---")
    cursor.execute("PRAGMA table_info(trade_candles_v2)")
    columns = cursor.fetchall()
    for col in columns:
        print(col)
        
finally:
    try:
        conn.close()
    except:
        pass
    shutil.rmtree(temp_dir)
