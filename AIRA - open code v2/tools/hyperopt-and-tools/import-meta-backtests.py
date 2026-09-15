#!/usr/bin/env python3
"""
Backtest Result Importer - Parses raw log files from freqtrade backtests
"""

import re
import os
import glob
from datetime import datetime
from pathlib import Path
import psycopg2

APP_ROOT = Path(__file__).resolve().parents[1]

# Database config
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'postgres'),
    'port': int(os.environ.get('DB_PORT', '5432')),
    'database': os.environ.get('DB_NAME', 'dashboard'),
    'user': os.environ.get('DB_USER', 'dashboard'),
    'password': os.environ.get('DB_PASSWORD', 'dashboard')
}

RESULTS_DIR = os.environ.get("BACKTEST_RESULTS_DIR", str(APP_ROOT / "results" / "backtest"))


def parse_raw_log(filepath):
    """Parse freqtrade raw log file with unicode table format"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract strategy name from filename (AlexStrategy_1234567890_raw.log)
        filename = os.path.basename(filepath)
        strategy_match = re.match(r'(.+?)_\d+_raw\.log', filename)
        strategy = strategy_match.group(1) if strategy_match else 'Unknown'
        
        # Parse metrics from unicode table format
        # Total profit %: look for │ Total profit % │ -97.01% │
        profit_match = re.search(r'Total profit %\s*│\s*([-?\d.]+)%', content)
        if not profit_match:
            profit_match = re.search(r'Total profit[\s%]*[:\-]?\s*([-?\d.]+)%', content, re.IGNORECASE)
        profit_pct = float(profit_match.group(1)) if profit_match else 0.0
        
        # Absolute profit
        abs_match = re.search(r'Absolute profit\s*│\s*([-?\d.]+)\s*USDT', content)
        if not abs_match:
            abs_match = re.search(r'Absolute profit[\s:]*([-?\d.]+)\s*USDT', content, re.IGNORECASE)
        profit_abs = float(abs_match.group(1)) if abs_match else 0.0
        
        # Total trades from TOTAL row
        trades_match = re.search(r'TOTAL\s*│\s*(\d+)', content)
        if not trades_match:
            trades_match = re.search(r'Total/Daily Avg Trades\s*[:/\-]?\s*(\d+)', content, re.IGNORECASE)
        total_trades = int(trades_match.group(1)) if trades_match else 0
        
        # Win rate - last number in TOTAL row
        win_match = re.search(r'TOTAL\s*│[^│]+│[^│]+│[^│]+│[^│]+│[^│]+│\s*\d+\s+\d+\s+\d+\s+([\d.]+)', content)
        if not win_match:
            win_match = re.search(r'Win[\s%]*[:\-]?\s*([\d.]+)', content, re.IGNORECASE)
        win_rate = float(win_match.group(1)) if win_match else 0.0
        
        # Initialize drawdown variables
        max_drawdown_pct = 0.0
        max_drawdown_abs = 0.0

        # Max Drawdown - format: "800.53 USDT 97.10%" (percentage is LAST)
        # Look for pattern: number USDT number%
        drawdown_match = re.search(r'(\d+\.?\d*)\s*USDT\s*([\d.]+)%', content)
        if drawdown_match:
            max_drawdown_abs = float(drawdown_match.group(1))
            max_drawdown_pct = float(drawdown_match.group(2))
        else:
            # Fallback to old patterns
            drawdown_match = re.search(r'Max Drawdown\s*│\s*[-?\d.]+\s*│\s*│\s*[-?\d.]+\s*│\s*([\d.]+)%', content)
            if drawdown_match:
                max_drawdown_pct = float(drawdown_match.group(1))
            else:
                drawdown_match = re.search(r'Max [Dd]rawdown[\s%]*[:\-]?\s*([\d.]+)', content, re.IGNORECASE)
                if drawdown_match:
                    max_drawdown_pct = float(drawdown_match.group(1))

        # Starting balance
        start_balance_match = re.search(r'Starting balance\s*│\s*([\d.]+)\s*USDT', content)
        if not start_balance_match:
            start_balance_match = re.search(r'Starting balance[\s:]*([\d.]+)\s*USDT', content, re.IGNORECASE)
        start_balance = float(start_balance_match.group(1)) if start_balance_match else 0.0

        # Final balance
        final_balance_match = re.search(r'Final balance\s*│\s*([\d.]+)\s*USDT', content)
        if not final_balance_match:
            final_balance_match = re.search(r'Final balance[\s:]*([\d.]+)\s*USDT', content, re.IGNORECASE)
        final_balance = float(final_balance_match.group(1)) if final_balance_match else 0.0
        
        # Sharpe
        sharpe_match = re.search(r'Sharpe\s*│\s*([-?\d.]+)', content)
        if not sharpe_match:
            sharpe_match = re.search(r'Sharpe[\s:]*([-?\d.]+)', content, re.IGNORECASE)
        sharpe = float(sharpe_match.group(1)) if sharpe_match else 0.0
        
        # Sortino
        sortino_match = re.search(r'Sortino\s*│\s*([-?\d.]+)', content)
        if not sortino_match:
            sortino_match = re.search(r'Sortino[\s:]*([-?\d.]+)', content, re.IGNORECASE)
        sortino = float(sortino_match.group(1)) if sortino_match else 0.0
        
        # Profit factor
        profit_factor_match = re.search(r'Profit Factor\s*│\s*([\d.]+)', content)
        if not profit_factor_match:
            profit_factor_match = re.search(r'Profit [Ff]actor[\s:]*([\d.]+)', content, re.IGNORECASE)
        profit_factor = float(profit_factor_match.group(1)) if profit_factor_match else 0.0
        
        # Avg profit
        avg_match = re.search(r'TOTAL\s*│\s*\d+\s*│\s*([-?\d.]+)', content)
        avg_profit = float(avg_match.group(1)) if avg_match else 0.0
        
        # Timeframe - extract from timerange if available
        timeframe = "15m"  # Default
        tf_match = re.search(r'Timeframe\s*│\s*(\w+)', content)
        if tf_match:
            timeframe = tf_match.group(1)
        
        metrics = {
            'strategy_name': strategy,
            'timeframe': timeframe,
            'timerange': '20241001-20260221',
            'total_profit_pct': profit_pct,
            'total_profit_abs': profit_abs,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'max_drawdown_pct': max_drawdown_pct,
            'max_drawdown_abs': max_drawdown_abs,
            'start_balance': start_balance,
            'final_balance': final_balance,
            'sharpe': sharpe,
            'sortino': sortino,
            'profit_factor': profit_factor,
            'avg_profit_pct': avg_profit,
            'backtest_date': datetime.now()
        }
        
        return metrics
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def import_to_database(metrics):
    """Insert metrics into backtest_results table"""
    if not metrics:
        return False
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Check if already imported
        cur.execute("""
            SELECT id FROM backtest_results 
            WHERE strategy_name = %s 
            AND backtest_date > NOW() - INTERVAL '1 hour'
        """, (metrics['strategy_name'],))
        
        if cur.fetchone():
            print(f"Already imported: {metrics['strategy_name']}")
            cur.close()
            conn.close()
            return False
        
        # Insert
        cur.execute("""
            INSERT INTO backtest_results (
                strategy_name, timeframe, timerange,
                total_profit_pct, total_profit_abs, total_trades,
                win_rate, max_drawdown_pct, max_drawdown_abs,
                start_balance, final_balance,
                sharpe, sortino, profit_factor, avg_profit_pct,
                backtest_date
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            metrics['strategy_name'],
            metrics['timeframe'],
            metrics['timerange'],
            metrics['total_profit_pct'],
            metrics['total_profit_abs'],
            metrics['total_trades'],
            metrics['win_rate'],
            metrics['max_drawdown_pct'],
            metrics['max_drawdown_abs'],
            metrics['start_balance'],
            metrics['final_balance'],
            metrics['sharpe'],
            metrics['sortino'],
            metrics['profit_factor'],
            metrics['avg_profit_pct']
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"✅ Imported: {metrics['strategy_name']} - Profit: {metrics['total_profit_pct']:.2f}%, Trades: {metrics['total_trades']}, Win: {metrics['win_rate']:.1f}%, DD: {metrics['max_drawdown_pct']:.2f}%")
        return True
        
    except Exception as e:
        print(f"❌ DB error: {e}")
        import traceback
        traceback.print_exc()
        return False


def scan_and_import():
    """Scan for raw log files and import"""
    imported = 0
    errors = []
    
    # Find all raw log files
    pattern = os.path.join(RESULTS_DIR, "*_raw.log")
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} raw log files")
    
    for filepath in files:
        try:
            metrics = parse_raw_log(filepath)
            if metrics:
                if import_to_database(metrics):
                    imported += 1
            else:
                errors.append(os.path.basename(filepath))
        except Exception as e:
            errors.append(f"{os.path.basename(filepath)}: {e}")
    
    return {"imported": imported, "errors": errors}


if __name__ == "__main__":
    result = scan_and_import()
    print(f"\n=== Import Complete ===")
    print(f"Imported: {result['imported']}")
    if result['errors']:
        print(f"Errors: {len(result['errors'])}")
        for err in result['errors'][:5]:
            print(f"  - {err}")
