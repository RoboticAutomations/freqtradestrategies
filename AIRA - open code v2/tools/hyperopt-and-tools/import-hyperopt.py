#!/usr/bin/env python3
"""
Hyperopt Result Importer - Parses raw log files from freqtrade hyperopt
"""

import re
import os
import glob
import json
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

HYPEROPT_DIR = os.environ.get("HYPEROPT_RESULTS_DIR", str(APP_ROOT / "results" / "hyperopt"))


def parse_hyperopt_rawlog(filepath):
    """Parse hyperopt raw log file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract strategy name from filename (AlexStrategy_1234567890_raw.log)
        filename = os.path.basename(filepath)
        strategy_match = re.match(r'(.+?)_\d+_raw\.log', filename)
        strategy = strategy_match.group(1) if strategy_match else 'Unknown'
        
        # Look for "Best result:" or "Best epoch:" in stdout
        best_match = re.search(r'Best\s+(?:result|epoch):?\s*Epoch\s*(\d+)', content, re.IGNORECASE)
        if not best_match:
            print(f"No 'Best result' found in {filepath}")
            return None
        
        epoch_num = int(best_match.group(1))
        
        # Parse epoch details from table
        # Pattern: epoch │ avg_profit │ total_profit │ winrate │ trades
        epoch_pattern = rf'{epoch_num}\s*│\s*([\d.]+)\s*│\s*([\d.]+)\s*│\s*([\d.]+)\s*│\s*(\d+)'
        epoch_match = re.search(epoch_pattern, content)
        
        if epoch_match:
            metrics = {
                'strategy_name': strategy,
                'epoch': epoch_num,
                'avg_profit': float(epoch_match.group(1)),
                'profit_pct': float(epoch_match.group(2)),
                'win_rate': float(epoch_match.group(3)),  # Already %
                'total_trades': int(epoch_match.group(4)),
                'sharpe': 0.0,
                'sortino': 0.0,
                'params': '{}',
                'is_best': True,
                'created_at': datetime.now()
            }
        else:
            # Fallback: just use epoch number
            metrics = {
                'strategy_name': strategy,
                'epoch': epoch_num,
                'avg_profit': 0.0,
                'profit_pct': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'sharpe': 0.0,
                'sortino': 0.0,
                'params': '{}',
                'is_best': True,
                'created_at': datetime.now()
            }
        
        # Look for sharpe/sortino in summary
        sharpe_match = re.search(r'Sharpe\s*[:\-]?\s*([-?\d.]+)', content, re.IGNORECASE)
        if sharpe_match:
            metrics['sharpe'] = float(sharpe_match.group(1))
        
        sortino_match = re.search(r'Sortino\s*[:\-]?\s*([-?\d.]+)', content, re.IGNORECASE)
        if sortino_match:
            metrics['sortino'] = float(sortino_match.group(1))
        
        # Try to get params from fthypt file
        try:
            fthypt_pattern = os.path.join(HYPEROPT_DIR, f".fthypt*{strategy}*")
            fthypt_files = glob.glob(fthypt_pattern)
            if fthypt_files:
                latest_file = max(fthypt_files, key=os.path.getmtime)
                with open(latest_file, 'r') as f:
                    for line in f:
                        try:
                            epoch_data = json.loads(line)
                            if epoch_data.get('epoch') == epoch_num:
                                metrics['params'] = json.dumps(epoch_data.get('params', {}))
                                # Update other metrics if available
                                if 'profit_mean_pct' in epoch_data:
                                    metrics['avg_profit'] = epoch_data['profit_mean_pct']
                                if 'profit_total_pct' in epoch_data:
                                    metrics['profit_pct'] = epoch_data['profit_total_pct']
                                if 'winrate' in epoch_data:
                                    metrics['win_rate'] = epoch_data['winrate'] * 100
                                if 'trade_count' in epoch_data:
                                    metrics['total_trades'] = epoch_data['trade_count']
                                break
                        except:
                            pass
        except Exception as e:
            print(f"Could not read fthypt: {e}")
        
        return metrics
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def import_to_database(metrics):
    """Insert metrics into hyperopt_epochs table"""
    if not metrics:
        return False
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Check if already imported
        cur.execute("""
            SELECT id FROM hyperopt_epochs 
            WHERE strategy_name = %s AND epoch = %s
            AND created_at > NOW() - INTERVAL '1 hour'
        """, (metrics['strategy_name'], metrics['epoch']))
        
        if cur.fetchone():
            print(f"Already imported: {metrics['strategy_name']} epoch {metrics['epoch']}")
            cur.close()
            conn.close()
            return False
        
        # Insert
        cur.execute("""
            INSERT INTO hyperopt_epochs (
                strategy_name, epoch, profit_pct, win_rate,
                avg_profit, total_trades, sharpe, sortino,
                params, is_best, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            metrics['strategy_name'],
            metrics['epoch'],
            metrics['profit_pct'],
            metrics['win_rate'],
            metrics['avg_profit'],
            metrics['total_trades'],
            metrics['sharpe'],
            metrics['sortino'],
            metrics['params'],
            metrics['is_best']
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"✅ Imported: {metrics['strategy_name']} - Epoch {metrics['epoch']} - Profit: {metrics['profit_pct']:.2f}%, Win: {metrics['win_rate']:.1f}%")
        return True
        
    except Exception as e:
        print(f"❌ DB error: {e}")
        import traceback
        traceback.print_exc()
        return False


def scan_and_import():
    """Scan for hyperopt raw log files and import"""
    imported = 0
    errors = []
    
    # Find all raw log files
    pattern = os.path.join(HYPEROPT_DIR, "*_raw.log")
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} hyperopt raw log files")
    
    for filepath in files:
        try:
            metrics = parse_hyperopt_rawlog(filepath)
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
