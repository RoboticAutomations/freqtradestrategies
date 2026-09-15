#!/usr/bin/env python3
"""
Import pairlist results from JSON files into database
Run: docker exec multibotdashboard-backend python /app/scripts/import_pairlist_results.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
import psycopg2
from psycopg2.extras import Json

APP_ROOT = Path(__file__).resolve().parents[1]

# Database connection
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'postgres'),
    'port': int(os.environ.get('DB_PORT', '5432')),
    'database': os.environ.get('DB_NAME', 'dashboard'),
    'user': os.environ.get('DB_USER', 'dashboard'),
    'password': os.environ.get('DB_PASSWORD', 'dashboard')
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def import_pairlist_result(json_path: str):
    """Import a pairlist JSON result file into database"""
    
    print(f"Importing {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract data from your file structure
    strategy = data.get('strategy', 'unknown')
    timestamp = data.get('generated_at', datetime.now().isoformat())
    timeframe = data.get('timeframe', 'unknown')
    evaluation_mode = data.get('evaluation_mode', 'unknown')
    total_pairs = data.get('total_evaluated', 0)
    pairs = data.get('pairs', [])
    detailed_metrics = data.get('detailed_metrics', [])
    
    print(f"  Strategy: {strategy}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Mode: {evaluation_mode}")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Detailed metrics: {len(detailed_metrics)}")
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Generate job_id from filename
        job_id = Path(json_path).stem
        
        # Insert job
        cur.execute("""
            INSERT INTO pairlist_jobs (job_id, strategy, mode, status, created_at, updated_at)
            VALUES (%s, %s, %s, 'completed', NOW(), NOW())
            ON CONFLICT (job_id) DO NOTHING
        """, (job_id, strategy, evaluation_mode))
        
        # Get summary
        summary = data.get('summary', {})
        
        # Insert result
        cur.execute("""
            INSERT INTO pairlist_results 
            (job_id, strategy, timeframe, evaluation_mode, total_pairs, 
             best_pair, best_profit, best_sharpe, avg_profit, avg_win_rate, results_json, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (job_id) DO NOTHING
        """, (
            job_id, strategy, timeframe, evaluation_mode, total_pairs,
            summary.get('best_pair'),
            summary.get('best_profit', 0),
            summary.get('best_sharpe', 0),
            summary.get('avg_profit_total', 0),
            summary.get('avg_win_rate', 0),
            Json(data)
        ))
        
        # Insert individual pairs
        for rank, pair_data in enumerate(detailed_metrics, 1):
            cur.execute("""
                INSERT INTO pairlist_pair_results 
                (job_id, pair, rank, profit_total, win_rate, max_drawdown, 
                 sharpe_ratio, trade_count, score, metrics_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                job_id,
                pair_data.get('pair', ''),
                rank,
                pair_data.get('profit_total', 0),
                pair_data.get('win_rate', 0),
                pair_data.get('max_drawdown', 0),
                pair_data.get('sharpe_ratio', 0),
                pair_data.get('trade_count', 0),
                pair_data.get('score', 0),
                Json(pair_data)
            ))
        
        conn.commit()
        print(f"✅ Imported {len(detailed_metrics)} pairs for {strategy}")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    results_dir = os.environ.get("PAIRLIST_RESULTS_DIR", str(APP_ROOT / "results" / "pairlists"))
    
    print(f"Looking for JSON files in: {results_dir}")
    
    # Check if directory exists
    if not Path(results_dir).exists():
        print(f"❌ Directory {results_dir} does not exist!")
        sys.exit(1)
    
    # List ALL files
    all_files = list(Path(results_dir).iterdir())
    print(f"All files in directory: {[f.name for f in all_files]}")
    
    # Look for JSON files with any name pattern
    json_files = [f for f in all_files if f.suffix == '.json']
    print(f"Found {len(json_files)} JSON files: {[f.name for f in json_files]}")
    
    # Import all JSON files
    for json_file in json_files:
        try:
            import_pairlist_result(str(json_file))
        except Exception as e:
            print(f"Failed to import {json_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Import complete!")
    
    # Verify import
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM pairlist_jobs")
        job_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM pairlist_results")
        result_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM pairlist_pair_results")
        pair_count = cur.fetchone()[0]
        print(f"\n📊 Database counts:")
        print(f"   Jobs: {job_count}")
        print(f"   Results: {result_count}")
        print(f"   Pairs: {pair_count}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Could not verify: {e}")
