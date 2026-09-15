#!/usr/bin/env python3
"""
Quick backtest importer - handles both SQLite and JSON files.
Uses environment-driven database and results paths.
"""

import json
import os
import sqlite3
import glob
import psycopg2
from datetime import datetime
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_DIR = os.environ.get("BACKTEST_DIR", str(APP_ROOT / "results" / "backtest"))
DB_HOST = os.environ.get("DB_HOST", "postgres")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "dashboard")
DB_USER = os.environ.get("DB_USER", "dashboard")
DB_PASS = os.environ.get("DB_PASSWORD", "dashboard")

def get_existing(pg_conn):
    cur = pg_conn.cursor()
    cur.execute("SELECT strategy_name FROM backtest_results")
    existing = {r[0] for r in cur.fetchall()}
    cur.close()
    return existing

def safe_float(v, default=None):
    try:
        return float(v) if v is not None else default
    except:
        return default

def import_from_sqlite(pg_conn, filepath, existing):
    """Import from another SQLite backtest_results.db file."""
    added = 0
    try:
        sq = sqlite3.connect(filepath)
        rows = sq.execute("SELECT strategy_name, timeframe, timerange, start_balance, final_balance, total_profit_pct, total_profit_abs, total_trades, win_rate, avg_profit_pct, max_drawdown_pct, max_drawdown_abs, sharpe, sortino, calmar, sqn, profit_factor, expectancy, avg_trade_duration, best_pair, best_pair_profit, worst_pair, worst_pair_profit, market_change, cagr_pct, backtest_date FROM backtest_results").fetchall()
        sq.close()
        
        cur = pg_conn.cursor()
        for r in rows:
            if r[0] in existing:
                print(f"SKIP: {r[0]} (exists)")
                continue
            cur.execute("""
                INSERT INTO backtest_results VALUES (DEFAULT, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, r)
            print(f"ADD:  {r[0]} ({r[5]:+.2f}% profit)")
            added += 1
        pg_conn.commit()
        cur.close()
    except Exception as e:
        print(f"ERR:  SQLite import failed: {e}")
    return added

def import_from_json(pg_conn, filepath, existing):
    """Import from a JSON backtest result file."""
    try:
        with open(filepath) as f:
            data = json.load(f)
        
        # Extract strategy name
        basename = os.path.basename(filepath).replace('.json', '')
        strategy = basename.replace('backtest-result-', '').replace('backtest-', '')
        
        # Try to get from data
        if 'strategy' in data and isinstance(data['strategy'], dict):
            strat_key = list(data['strategy'].keys())[0]
            metrics = data['strategy'][strat_key]
            strategy = strat_key
        else:
            metrics = data
        
        if strategy in existing:
            print(f"SKIP: {strategy} (exists)")
            return 0
        
        # Extract metrics
        profit_pct = safe_float(metrics.get('profit_total_pct'))
        profit_abs = safe_float(metrics.get('profit_total_abs'))
        trades = int(safe_float(metrics.get('trade_count'), 0) or 0)
        win_rate = safe_float(metrics.get('win_rate'))
        if win_rate and win_rate <= 1:
            win_rate *= 100
        
        # Simple insert with just the key fields
        cur = pg_conn.cursor()
        cur.execute("""
            INSERT INTO backtest_results (strategy_name, timeframe, total_profit_pct, total_profit_abs, total_trades, win_rate, backtest_date)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """, (strategy, data.get('timeframe', 'unknown'), profit_pct, profit_abs, trades, win_rate))
        pg_conn.commit()
        cur.close()
        print(f"ADD:  {strategy} ({profit_pct:+.2f}% profit, {trades} trades)")
        return 1
        
    except Exception as e:
        print(f"ERR:  {filepath} - {e}")
        return 0

def main():
    print("Backtest Importer")
    print("=" * 50)
    
    pg = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
    existing = get_existing(pg)
    print(f"DB has {len(existing)} strategies")
    print()
    
    added = 0
    
    # Check for SQLite files
    for dbfile in glob.glob(os.path.join(BACKTEST_DIR, "*.db")):
        print(f"\nSQLite: {dbfile}")
        added += import_from_sqlite(pg, dbfile, existing)
    
    # Check for JSON files
    json_files = glob.glob(os.path.join(BACKTEST_DIR, "*.json"))
    json_files += glob.glob(os.path.join(BACKTEST_DIR, "**/*.json"), recursive=True)
    
    if json_files:
        print(f"\nJSON files: {len(json_files)}")
        for jfile in set(json_files):
            added += import_from_json(pg, jfile, existing)
    
    print()
    print(f"Added {added} new strategies")
    pg.close()

if __name__ == "__main__":
    main()
