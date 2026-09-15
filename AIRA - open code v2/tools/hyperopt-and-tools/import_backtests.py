#!/usr/bin/env python3
"""
Import backtest results from JSON files into dashboard PostgreSQL database.
Only adds NEW strategies (skips existing ones).
Uses environment-driven database and results paths.
"""

import json
import os
import glob
import psycopg2
from datetime import datetime
from pathlib import Path

# Config
APP_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_DIR = os.environ.get("BACKTEST_DIR", str(APP_ROOT / "results" / "backtest"))
DB_HOST = os.environ.get("DB_HOST", "postgres")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "dashboard")
DB_USER = os.environ.get("DB_USER", "dashboard")
DB_PASS = os.environ.get("DB_PASSWORD", "dashboard")

def parse_backtest_file(filepath):
    """Extract metrics from a freqtrade backtest result file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Get strategy name from filename or data
    basename = os.path.basename(filepath)
    strategy_name = basename.replace('.json', '').replace('backtest-result-', '')
    
    # Extract key metrics
    strategy_data = data.get('strategy', {})
    if not strategy_data:
        # Try alternative format
        strategy_data = data
    
    # Get the first strategy if multiple
    if isinstance(strategy_data, dict) and len(strategy_data) > 0:
        first_key = list(strategy_data.keys())[0]
        if isinstance(strategy_data[first_key], dict):
            metrics = strategy_data[first_key]
            strategy_name = first_key
        else:
            metrics = strategy_data
    else:
        metrics = strategy_data
    
    # Parse timeframe and timerange from filename or data
    timeframe = data.get('timeframe', 'unknown')
    timerange = data.get('timerange', data.get('daterange', 'unknown'))
    
    # Extract numeric values safely
    def safe_float(val, default=None):
        if val is None or val == '':
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default
    
    def safe_int(val, default=0):
        if val is None or val == '':
            return default
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return default
    
    # Backtest stats
    total_profit_pct = safe_float(metrics.get('profit_total_pct'))
    total_profit_abs = safe_float(metrics.get('profit_total_abs'))
    total_trades = safe_int(metrics.get('trade_count') or metrics.get('total_trades'))
    win_rate = safe_float(metrics.get('win_rate'))
    
    if win_rate and win_rate <= 1:
        win_rate = win_rate * 100  # Convert to percentage if stored as decimal
    
    # Drawdown
    max_drawdown = metrics.get('max_drawdown', {})
    max_drawdown_pct = safe_float(max_drawdown.get('drawdown_relative') or max_drawdown.get('pct'))
    max_drawdown_abs = safe_float(max_drawdown.get('drawdown_abs') or max_drawdown.get('abs'))
    
    if max_drawdown_pct and max_drawdown_pct <= 1:
        max_drawdown_pct = max_drawdown_pct * 100
    
    # Balance
    start_balance = safe_float(metrics.get('starting_balance') or data.get('starting_balance'))
    final_balance = safe_float(metrics.get('final_balance') or data.get('final_balance'))
    
    if not final_balance and total_profit_abs and start_balance:
        final_balance = start_balance + total_profit_abs
    
    # Risk metrics
    sharpe = safe_float(metrics.get('sharpe'))
    sortino = safe_float(metrics.get('sortino'))
    calmar = safe_float(metrics.get('calmar'))
    sqn = safe_float(metrics.get('sqn'))
    profit_factor = safe_float(metrics.get('profit_factor'))
    expectancy = safe_float(metrics.get('expectancy'))
    
    # Trade duration
    avg_trade_duration = metrics.get('avg_trade_duration') or metrics.get('holding_avg', '')
    if isinstance(avg_trade_duration, (int, float)):
        avg_trade_duration = str(avg_trade_duration)
    
    # Best/worst pair
    best_pair = ''
    worst_pair = ''
    best_pair_profit = None
    worst_pair_profit = None
    
    pair_results = metrics.get('pair_performance', metrics.get('results_per_pair', {}))
    if pair_results and isinstance(pair_results, dict):
        profits = []
        for pair, pdata in pair_results.items():
            pprofit = safe_float(pdata.get('profit_pct') or pdata.get('profit_total_pct'))
            if pprofit is not None:
                profits.append((pair, pprofit))
        if profits:
            profits.sort(key=lambda x: x[1], reverse=True)
            best_pair = profits[0][0]
            best_pair_profit = profits[0][1]
            worst_pair = profits[-1][0]
            worst_pair_profit = profits[-1][1]
    
    # Market change / CAGR
    market_change = safe_float(metrics.get('market_change'))
    cagr_pct = safe_float(metrics.get('cagr'))
    
    # Avg profit per trade
    avg_profit_pct = safe_float(metrics.get('avg_profit_pct') or metrics.get('profit_mean_pct'))
    
    return {
        'strategy_name': strategy_name,
        'timeframe': timeframe,
        'timerange': timerange if isinstance(timerange, str) else str(timerange),
        'start_balance': start_balance,
        'final_balance': final_balance,
        'total_profit_pct': total_profit_pct,
        'total_profit_abs': total_profit_abs,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_profit_pct': avg_profit_pct,
        'max_drawdown_pct': max_drawdown_pct,
        'max_drawdown_abs': max_drawdown_abs,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'sqn': sqn,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'avg_trade_duration': avg_trade_duration[:50] if avg_trade_duration else None,
        'best_pair': best_pair[:50] if best_pair else None,
        'best_pair_profit': best_pair_profit,
        'worst_pair': worst_pair[:50] if worst_pair else None,
        'worst_pair_profit': worst_pair_profit,
        'market_change': market_change,
        'cagr_pct': cagr_pct,
    }

def get_existing_strategies(pg_conn):
    """Get list of strategies already in the database."""
    cur = pg_conn.cursor()
    cur.execute("SELECT strategy_name FROM backtest_results")
    existing = set(row[0] for row in cur.fetchall())
    cur.close()
    return existing

def insert_backtest(pg_conn, data):
    """Insert a backtest result into the database."""
    cur = pg_conn.cursor()
    cur.execute("""
        INSERT INTO backtest_results (
            strategy_name, timeframe, timerange, start_balance, final_balance,
            total_profit_pct, total_profit_abs, total_trades, win_rate, avg_profit_pct,
            max_drawdown_pct, max_drawdown_abs, sharpe, sortino, calmar, sqn,
            profit_factor, expectancy, avg_trade_duration, best_pair, best_pair_profit,
            worst_pair, worst_pair_profit, market_change, cagr_pct, backtest_date
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
        )
    """, (
        data['strategy_name'], data['timeframe'], data['timerange'],
        data['start_balance'], data['final_balance'], data['total_profit_pct'],
        data['total_profit_abs'], data['total_trades'], data['win_rate'],
        data['avg_profit_pct'], data['max_drawdown_pct'], data['max_drawdown_abs'],
        data['sharpe'], data['sortino'], data['calmar'], data['sqn'],
        data['profit_factor'], data['expectancy'], data['avg_trade_duration'],
        data['best_pair'], data['best_pair_profit'], data['worst_pair'],
        data['worst_pair_profit'], data['market_change'], data['cagr_pct']
    ))
    pg_conn.commit()
    cur.close()

def main():
    print("=" * 60)
    print("Backtest Results Importer")
    print("=" * 60)
    print(f"Source: {BACKTEST_DIR}")
    print(f"Target: {DB_HOST}/{DB_NAME}")
    print()
    
    # Connect to PostgreSQL
    try:
        pg_conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        print(f"✓ Connected to dashboard database")
    except Exception as e:
        print(f"✗ Failed to connect to database: {e}")
        return
    
    # Get existing strategies
    existing = get_existing_strategies(pg_conn)
    print(f"✓ Found {len(existing)} existing strategies in database")
    
    # Find backtest files
    patterns = [
        os.path.join(BACKTEST_DIR, "*.json"),
        os.path.join(BACKTEST_DIR, "backtest-*.json"),
        os.path.join(BACKTEST_DIR, "**", "*.json"),
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates
    files = list(set(files))
    print(f"✓ Found {len(files)} backtest result files")
    print()
    
    # Process each file
    added = 0
    skipped = 0
    errors = 0
    
    for filepath in sorted(files):
        basename = os.path.basename(filepath)
        try:
            data = parse_backtest_file(filepath)
            strategy_name = data['strategy_name']
            
            if strategy_name in existing:
                print(f"SKIP: {strategy_name} (already exists)")
                skipped += 1
                continue
            
            insert_backtest(pg_conn, data)
            print(f"ADD:  {strategy_name} ({data['total_profit_pct']:+.2f}% profit, {data['total_trades']} trades)")
            added += 1
            existing.add(strategy_name)  # Mark as added
            
        except Exception as e:
            print(f"ERR:  {basename} - {e}")
            errors += 1
    
    print()
    print("=" * 60)
    print(f"Done! Added: {added}, Skipped: {skipped}, Errors: {errors}")
    print("=" * 60)
    
    # Show final count
    cur = pg_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM backtest_results")
    total = cur.fetchone()[0]
    cur.close()
    print(f"Total strategies in database: {total}")
    
    pg_conn.close()

if __name__ == "__main__":
    main()
