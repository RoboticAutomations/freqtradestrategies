import unittest
import sqlite3
import os
import sys
import shutil
import time

# Add root to path to import plot script
sys.path.append(os.getcwd())
from plot.cvd_plot import plot_cvd, read_data

class TestCVDIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_output"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        self.db_path = os.path.join(self.test_dir, "test_db.sqlite")
        self.conn = sqlite3.connect(self.db_path)
        self.create_tables()
        
    def tearDown(self):
        self.conn.close()
        # if os.path.exists(self.test_dir):
        #     shutil.rmtree(self.test_dir)

    def create_tables(self):
        cursor = self.conn.cursor()
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
            );
        """)
        self.conn.commit()

    def test_end_to_end_plot(self):
        # 1. Generate Sample Data
        cursor = self.conn.cursor()
        symbol = "BTC/USDT"
        base_price = 50000.0
        start_time = int(time.time()) - 3600 # 1 hour ago
        
        print("Generating sample data...")
        for i in range(60): # 60 minutes
            t = start_time + (i * 60)
            
            # Simulate price movement and delta
            if i < 30:
                price = base_price + (i * 10)
                delta = 100.0 + (i * 5)
            else:
                price = base_price + 300 - ((i-30) * 10)
                delta = -100.0 - ((i-30) * 5)
                
            cursor.execute("""
                INSERT INTO trade_candles_v2 (symbol, timestamp, open, high, low, close, volume, delta, buy_vol, sell_vol, buy_count, sell_count, whale_buy_vol, whale_sell_vol, whale_delta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 10, 10, 0, 0, 0)
            """, (symbol, t, price, price+5, price-5, price, 1000, delta, 500+delta/2, 500-delta/2))
        
        self.conn.commit()
        
        # 2. Verify Data Read
        times, prices, deltas, whale_deltas = read_data(self.db_path, symbol)
        self.assertEqual(len(times), 60)
        self.assertEqual(prices[0], 50000.0)
        
        # 3. Verify Plot Generation
        out_plot = os.path.join(self.test_dir, "cvd_test.png")
        plot_cvd(times, prices, deltas, whale_deltas, symbol, out_plot)
        
        self.assertTrue(os.path.exists(out_plot))
        print(f"Test plot generated at {out_plot}")

if __name__ == '__main__':
    unittest.main()
