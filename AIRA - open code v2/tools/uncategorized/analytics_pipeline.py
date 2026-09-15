#!/usr/bin/env python3
"""
Freqtrade Analytics Pipeline — V12 Community Edition
=====================================================
Extracts data from all Freqtrade bots and loads into PostgreSQL.
Runs on a schedule (default: every 5 minutes).

Configuration priority:
  1. data_collector/config.json (bots array)
  2. Environment variables: BOTS_JSON, API_USER, API_PASS
  3. Auto-discovery via docker labels (optional, future)

Environment variables:
  DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME  — Database connection
  API_USER, API_PASS                                — Freqtrade API credentials
  BOTS_JSON                                         — JSON array of bot configs
  COLLECTOR_INTERVAL_SECONDS                        — Run interval (default: 300)
"""

import json
import requests
import psycopg2
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper()),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Database config from environment
# =============================================================================
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "postgres"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "database": os.environ.get("DB_NAME", "dashboard"),
    "user": os.environ.get("DB_USER", "dashboard"),
    "password": os.environ.get("DB_PASSWORD", "dashboard"),
}

# =============================================================================
# Freqtrade API credentials from environment
# =============================================================================
API_USER = os.environ.get("API_USER", os.environ.get("FREQTRADE_USERNAME", ""))
API_PASS = os.environ.get("API_PASS", os.environ.get("FREQTRADE_PASSWORD", ""))

# =============================================================================
# Bot configuration loader
# =============================================================================
def load_bot_config():
    """Load bot list from config.json or environment."""
    bots = []
    
    # Try config.json first
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            bots = cfg.get("bots", [])
            if bots:
                logger.info(f"📋 Loaded {len(bots)} bots from config.json")
        except Exception as e:
            logger.warning(f"Could not read config.json: {e}")
    
    # Fall back to BOTS_JSON environment variable
    if not bots and os.environ.get("BOTS_JSON"):
        try:
            bots = json.loads(os.environ["BOTS_JSON"])
            logger.info(f"📋 Loaded {len(bots)} bots from BOTS_JSON env var")
        except Exception as e:
            logger.warning(f"Could not parse BOTS_JSON: {e}")
    
    if not bots:
        logger.warning(
            "⚠️ No bots configured! Create data_collector/config.json from config.json.example "
            "or set BOTS_JSON environment variable."
        )
    
    return bots

# Load bot list
BOTS = load_bot_config()

# =============================================================================
# Helper functions
# =============================================================================
def get_db_connection():
    """Get PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)


def fetch_bot_name(url, headers):
    """Fetch real bot name from freqtrade API."""
    try:
        port = url.split(':')[-1].split('/')[0]
    except Exception:
        port = "unknown"
    
    # Try show_config endpoint
    try:
        resp = requests.get(f"{url}/api/v1/show_config", headers=headers, timeout=30)
        if resp.status_code == 200:
            cfg = resp.json()
            bot_name = cfg.get('bot_name')
            if bot_name and bot_name != "freqtrade":
                return bot_name
            strategy = cfg.get('strategy', 'Unknown')
            return f"{strategy}-{port}"
    except Exception as e:
        logger.debug(f"show_config failed for {url}: {e}")
    
    # Try status endpoint
    try:
        resp = requests.get(f"{url}/api/v1/status", headers=headers, timeout=30)
        if resp.status_code == 200:
            status = resp.json()
            if isinstance(status, dict):
                bot_name = status.get('bot_name')
                if bot_name and bot_name != "freqtrade":
                    return bot_name
    except Exception:
        pass
    
    return f"Bot-{port}"


def make_auth_headers():
    """Create auth headers with Basic auth."""
    if not API_USER or not API_PASS:
        return {}
    import base64
    auth = base64.b64encode(f"{API_USER}:{API_PASS}".encode()).decode()
    return {"Authorization": f"Basic {auth}"}


def fetch_bot_data(bot):
    """Fetch data from a single bot."""
    url = bot.get('url')
    if not url:
        logger.warning(f"Bot missing URL: {bot}")
        return None
    
    # Per-bot credentials override global
    user = bot.get('username') or API_USER
    passwd = bot.get('password') or API_PASS
    
    headers = {}
    if user and passwd:
        import base64
        auth = base64.b64encode(f"{user}:{passwd}".encode()).decode()
        headers = {"Authorization": f"Basic {auth}"}
    
    try:
        # Fetch bot name
        real_name = fetch_bot_name(url, headers)
        if real_name:
            bot['name'] = real_name
            logger.info(f"📝 Bot name: {real_name}")
        else:
            logger.warning(f"⚠️ Using fallback name: {bot.get('name', 'Unknown')}")
        
        # Fetch endpoints
        profit_resp = requests.get(f"{url}/api/v1/profit", headers=headers, timeout=30)
        status_resp = requests.get(f"{url}/api/v1/status", headers=headers, timeout=30)
        balance_resp = requests.get(f"{url}/api/v1/balance", headers=headers, timeout=30)
        
        return {
            'profit': profit_resp.json() if profit_resp.status_code == 200 else {},
            'status': status_resp.json() if status_resp.status_code == 200 else [],
            'balance': balance_resp.json() if balance_resp.status_code == 200 else {}
        }
        
    except Exception as e:
        logger.error(f"Error fetching {bot.get('name', url)}: {e}")
        return None


def store_snapshot(conn, bot, data):
    """Store bot snapshot."""
    try:
        profit = data.get('profit', {})
        balance = data.get('balance', {})
        bot_name = bot.get('name') or bot.get('fallback_name', 'Unknown')
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO bot_snapshots 
                (bot_name, status, mode, profit_all, profit_closed, winrate,
                 trade_count, open_trades, balance, max_drawdown, current_drawdown)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                bot_name,
                'running',
                bot.get('mode', 'unknown'),
                profit.get('profit_all_fiat', 0),
                profit.get('profit_closed_fiat', 0),
                (profit.get('winrate', 0) or 0) * 100,
                profit.get('trade_count', 0),
                len(data.get('status', [])),
                balance.get('total', 0),
                (profit.get('max_drawdown', 0) or 0) * 100,
                (profit.get('current_drawdown', 0) or 0) * 100
            ))
            conn.commit()
            logger.info(f"✓ Snapshot stored: {bot_name}")
            
    except Exception as e:
        logger.error(f"Error storing snapshot: {e}")
        try:
            conn.rollback()
        except Exception:
            pass


def store_daily_performance(conn, bot, data):
    """Store or update daily performance."""
    try:
        profit = data.get('profit', {})
        today = datetime.now().date()
        bot_name = bot.get('name') or bot.get('fallback_name', 'Unknown')
        
        winning_trades = profit.get('winning_trades')
        losing_trades = profit.get('losing_trades')
        
        win_count = int(winning_trades) if winning_trades is not None else 0
        loss_count = int(losing_trades) if losing_trades is not None else 0
        
        total = win_count + loss_count
        win_rate = profit.get('win_rate', 0)
        if not win_rate and total > 0:
            win_rate = round((win_count / total) * 100, 2)
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO daily_performance 
                (date, bot_name, strategy, profit, total_trades, balance, win_count, loss_count, win_rate)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date, bot_name) 
                DO UPDATE SET
                    profit = EXCLUDED.profit,
                    total_trades = EXCLUDED.total_trades,
                    balance = EXCLUDED.balance,
                    win_count = EXCLUDED.win_count,
                    loss_count = EXCLUDED.loss_count,
                    win_rate = EXCLUDED.win_rate
            """, (
                today,
                bot_name,
                bot.get('strategy', 'unknown'),
                profit.get('profit_all_fiat', 0),
                profit.get('trade_count', 0),
                data.get('balance', {}).get('total', 0),
                win_count,
                loss_count,
                win_rate
            ))
            conn.commit()
            logger.info(f"✓ {bot_name}: {win_count}W/{loss_count}L ({win_rate}% WR)")
            
    except Exception as e:
        logger.error(f"Error storing daily performance: {e}")
        try:
            conn.rollback()
        except Exception:
            pass


# =============================================================================
# Main
# =============================================================================
def main():
    """Main pipeline function."""
    logger.info("🚀 Freqtrade Analytics Pipeline V12 starting...")
    logger.info(f"   Database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    logger.info(f"   Bots configured: {len(BOTS)}")
    
    if not BOTS:
        logger.error("❌ No bots configured — pipeline exiting. See config.json.example")
        return
    
    conn = None
    try:
        conn = get_db_connection()
        logger.info("✓ Database connected")
        
        for bot in BOTS:
            logger.info(f"📡 Fetching: {bot.get('name', bot.get('url', 'Unknown'))}")
            data = fetch_bot_data(bot)
            if data:
                store_snapshot(conn, bot, data)
                store_daily_performance(conn, bot, data)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        if conn:
            conn.close()
            logger.info("🔌 Database connection closed")
    
    logger.info("✅ Pipeline complete")


if __name__ == "__main__":
    main()
