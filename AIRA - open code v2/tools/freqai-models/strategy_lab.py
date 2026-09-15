"""
Strategy Lab API for Multibotdashboard V6
Integrates ftmanager workflow engine with dashboard
"""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import asyncio
import json
import os
import logging
import re

import time
from datetime import datetime

logger = logging.getLogger(__name__)

from src.models import get_db
from src.models.bot import Bot
from src.schemas.bot import BotResponse

router = APIRouter(prefix="/strategy-lab", tags=["strategy-lab"])

# Import ftmanager components
from src.services.ftmanager.state import AppState, ProcessType, ProcessStatus
from src.services.ftmanager.workflow import Workflow
from src.services.ftmanager.hyperopt_monitor import HyperoptMonitor
from src.services.ftmanager.config import AppConfig, StrategyConfig
from src.services.runtime_settings import get_runtime_paths

# Global state (initialized on startup)
app_state: Optional[AppState] = None
workflow_engine: Optional[Workflow] = None
hyperopt_monitor: Optional[HyperoptMonitor] = None


@router.on_event("startup")
async def startup_strategy_lab():
    """Initialize ftmanager components on startup"""
    global app_state, workflow_engine, hyperopt_monitor
    paths = await get_runtime_paths()
    
    # Create default config (customize as needed)
    config = AppConfig(
        freqtrade_dir=paths["freqtrade_root_dir"],
        strategies=[]
    )
    
    app_state = AppState()
    hyperopt_monitor = HyperoptMonitor(config, app_state)
    workflow_engine = Workflow(config, app_state, None, hyperopt_monitor)
    
    # Start hyperopt file watcher
    # Note: start_monitoring requires strategy_name, called per-strategy when needed
    # asyncio.create_task(hyperopt_monitor.start_monitoring("default"))


@router.get("/strategies")
async def get_strategies(
    session: AsyncSession = Depends(get_db)
) -> List[dict]:
    """Get all strategies by scanning Strategies folder"""
    strategies = []
    paths = await get_runtime_paths()
    strategies_path = paths["strategies_dir"]
    
    print(f"DEBUG: Looking for strategies at: {strategies_path}", flush=True)
    
    if not os.path.exists(strategies_path):
        print(f"DEBUG: Path does not exist: {strategies_path}", flush=True)
        return []
    
    py_files_found = 0
    for root, dirs, files in os.walk(strategies_path):
        print(f"DEBUG: Scanning {root}, files: {len(files)}", flush=True)
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                py_files_found += 1
                file_path = os.path.join(root, file)
                strategy_name = file[:-3]  # Remove .py
                
                # Extract family from path (e.g., AlexKasuari from AlexKasuari/V1)
                rel_path = os.path.relpath(file_path, strategies_path)
                path_parts = rel_path.split(os.sep)
                family = path_parts[0] if len(path_parts) > 1 else 'Other'
                
                # Try to extract version from path (e.g., V2, V3)
                version = ""
                for part in path_parts:
                    if part.upper().startswith('V') and part[1:].isdigit():
                        version = part
                        break
                
                strategies.append({
                    "name": strategy_name,
                    "family": family,
                    "file_path": file_path,
                    "version": version,
                    "description": "",
                    "entry_type": "unknown",
                    "exit_type": "unknown",
                    "has_custom_exit": False,
                    "has_trailing_stop": False,
                    "indicators": [],
                    "optimizable_params": [],
                    "ml_features": []
                })
    
    print(f"DEBUG: Total .py files found: {py_files_found}, strategies added: {len(strategies)}", flush=True)
    strategies.sort(key=lambda x: x["name"])
    return strategies


@router.post("/strategies/upload")
async def upload_strategy(
    file: UploadFile = File(...),
) -> dict:
    """Upload a new strategy file into the Strategy Lab directory."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    original_name = os.path.basename(file.filename)
    if not original_name.endswith(".py"):
        raise HTTPException(status_code=400, detail="Only .py strategy files are supported")

    sanitized_name = re.sub(r"[^A-Za-z0-9_.-]", "_", original_name)
    if sanitized_name.startswith("__") or sanitized_name == ".py":
        raise HTTPException(status_code=400, detail="Invalid strategy filename")

    paths = await get_runtime_paths()
    strategies_path = paths["strategies_dir"]
    os.makedirs(strategies_path, exist_ok=True)

    destination_path = os.path.join(strategies_path, sanitized_name)
    if os.path.exists(destination_path):
        raise HTTPException(
            status_code=409,
            detail=f"Strategy file {sanitized_name} already exists",
        )

    content = await file.read()
    if not content.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Strategy file must be UTF-8 text") from exc

    if "class " not in decoded:
        raise HTTPException(status_code=400, detail="Strategy file does not look like a Python strategy")

    with open(destination_path, "wb") as strategy_file:
        strategy_file.write(content)

    return {
        "message": "Strategy uploaded successfully",
        "filename": sanitized_name,
        "path": destination_path,
    }



@router.get("/bots/{bot_id}/workflow-status")
async def get_workflow_status(
    bot_id: str,
    session: AsyncSession = Depends(get_db)
) -> dict:
    """Get current workflow status for a bot"""
    if not app_state:
        raise HTTPException(status_code=503, detail="Strategy Lab not initialized")
    
    # Get bot to find its strategy name
    result = await session.get(Bot, bot_id)
    if not result:
        return {"status": "idle", "message": "Bot not found"}
    
    bot = result
    strategy_name = bot.strategy or bot.name
    
    # Check workflow status for this strategy
    process = app_state.get_process(ProcessType.HYPEROPT, strategy_name)
    if not process:
        process = app_state.get_process(ProcessType.BACKTEST, strategy_name)
    
    if not process:
        return {"status": "idle", "message": "No active workflow"}
    
    return {
        "status": process.status.value,
        "type": process.type.value,
        "started_at": process.started_at.isoformat() if process.started_at else None,
        "pid": process.pid,
        "stats": process.stats.to_dict() if process.stats else None
    }


@router.post("/workflow/start")
async def start_workflow(
    config: dict,  # Full config from frontend
    session: AsyncSession = Depends(get_db)
) -> dict:
    """Start optimization workflow - spawn Docker container for backtest/hyperopt"""
    
    strategy_name = config.get('strategy')
    if not strategy_name:
        raise HTTPException(status_code=400, detail="Strategy name required")
    
    # Optional: associate with bot if provided
    bot_id = config.get('bot_id')
    bot_name = None
    if bot_id and bot_id != 'default':
        result = await session.get(Bot, bot_id)
        if result:
            bot_name = result.name
    steps = config.get('steps', [])
    epochs = config.get('epochs', 30)
    
    steps = config.get('steps', [])
    epochs = config.get('epochs', 30)
    
    logger.info(f"Starting workflow for strategy {strategy_name}, steps: {steps}")
    
    # Start workflow in background (non-blocking)
    import asyncio
    import subprocess
    import shutil
    
    async def run_workflow():
        try:
            logger.info(f"Workflow started for {strategy_name}")
            paths = await get_runtime_paths()
            
            # Find strategy file and copy to freqtrade strategies folder
            freqtrade_strat_dir = paths["freqtrade_strategies_dir"]
            os.makedirs(freqtrade_strat_dir, exist_ok=True)
            
            strategy_file_path = None
            strategy_class = strategy_name
            available_spaces = ["buy"]  # INITIALIZE HERE
            
            for root, dirs, files in os.walk(paths["strategies_dir"]):
                for file in files:
                    if file == f"{strategy_name}.py":
                        strategy_file_path = os.path.join(root, file)
                        dest_path = os.path.join(freqtrade_strat_dir, file)
                        shutil.copy(strategy_file_path, dest_path)
                        logger.info(f"Copied strategy to {dest_path}")
                        try:
                            with open(strategy_file_path, 'r') as f:
                                content = f.read()
                                import re
                                # FIX: Find class that inherits from IStrategy, not just any class
                                # Pattern: class StrategyName(IStrategy) or class StrategyName(IStrategy, ...)
                                match = re.search(r'class\s+(\w+)\s*\(\s*IStrategy', content)
                                if match:
                                    strategy_class = match.group(1)
                                    logger.info(f"Found strategy class name: {strategy_class}")
                                else:
                                    # Fallback: look for class containing strategy name
                                    match = re.search(rf'class\s+({re.escape(strategy_name)}\w*)\s*\(', content)
                                    if match:
                                        strategy_class = match.group(1)
                                        logger.info(f"Found class by name: {strategy_class}")
                                    else:
                                        # Last resort: find any class that looks like a strategy
                                        matches = re.findall(r'class\s+(\w+)\s*\(', content)
                                        for m in matches:
                                            if 'Config' not in m and 'Plot' not in m:
                                                strategy_class = m
                                                logger.info(f"Found strategy class (fallback): {strategy_class}")
                                                break
                        except Exception as e:
                            logger.warning(f"Could not read strategy file: {e}")
                        break
                if strategy_file_path:
                    break
            
            if not strategy_file_path:
                logger.error(f"Strategy file not found: {strategy_name}.py")
                return
            
            # Get timerange and timeframe from config or use defaults
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)  # 6 months default
            default_timerange = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
            timerange = config.get('timerange', default_timerange)
            timeframe = config.get('timeframe', '15m')
            logger.info(f"Using timerange: {timerange}, timeframe: {timeframe}")
            
            # Build Docker command based on steps
            if 'backtest' in steps:
                logger.info(f"Running backtest for {strategy_name}")
                
                results_dir = paths["backtest_results_dir"]
                os.makedirs(results_dir, exist_ok=True)
                timestamp = int(time.time())
                # Use a path inside the container that freqtrade can write to
                container_export_path = f"/freqtrade/user_data/backtest_results/{strategy_name}_{timestamp}_export.json"
                host_export_path = f"{results_dir}/{strategy_name}_{timestamp}_export.json"
                
                cmd = [
                    "docker", "run",
                    "--name", f"backtest_{strategy_name}_{timestamp}",
                    "-v", f"{freqtrade_strat_dir}:/freqtrade/user_data/strategies",
                    "-v", f"{paths['freqtrade_data_dir']}:/freqtrade/user_data/data",
                    "-v", f"{paths['freqtrade_config_dir']}:/freqtrade/config",
                    "-v", f"{paths['freqtrade_logs_dir']}:/freqtrade/user_data/logs",
                    "-v", f"{results_dir}:/freqtrade/user_data/backtest_results",
                    "freqtradeorg/freqtrade:stable_freqaitorch",
                    "backtesting",
                    "--strategy", strategy_class,
                    "--config", "/freqtrade/config/config-torch.json",
                    "--timerange", timerange,
                    "--timeframe", timeframe,
                    "--cache", "none",
                    "--export", "trades",
                    "--export-filename", container_export_path
                ]
                logger.info(f"Docker command: {' '.join(cmd)}")
                process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                stdout, stderr = await process.communicate()
                result = type('Result', (), {'stdout': stdout.decode(), 'stderr': stderr.decode(), 'returncode': process.returncode})()

                
                # ALWAYS save raw results first (for debugging)
                try:
                    raw_log_file = f"{results_dir}/{strategy_name}_{timestamp}_raw.log"
                    logger.info(f"Attempting to save raw log to: {raw_log_file}")
                    with open(raw_log_file, 'w') as f:
                        f.write("=== STDOUT ===\n")
                        f.write(result.stdout if result.stdout else "")
                        f.write("\n=== STDERR ===\n")
                        f.write(result.stderr if result.stderr else "")
                        f.write(f"\n=== RETURN CODE ===\n{result.returncode}")
                    logger.info(f"✅ Saved raw backtest log to {raw_log_file}")
                except Exception as e:
                    logger.error(f"❌ FAILED to save raw log: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # Parse results and save to database
                # Initialize with defaults (in case parsing fails)
                profit_pct = 0.0
                profit_abs = 0.0
                total_trades = 0
                win_rate = 0.0
                max_drawdown = 0.0
                sharpe = 0.0
                sortino = 0.0
                profit_factor = 0.0
                avg_profit = 0.0
                export_data = {}
                db_inserted = False
                
                try:
                    stdout = result.stdout if result.stdout else ""
                    
                    # Parse key metrics from stdout - handle unicode table format
                    # The format uses │ as column separators
                    
                    # Total profit % from SUMMARY METRICS table
                    profit_match = re.search(r'Total profit %\s*│\s*([-?\d.]+)%', stdout)
                    if not profit_match:
                        profit_match = re.search(r'Total profit[\s%]*[:\-]?\s*([-?\d.]+)%', stdout, re.IGNORECASE)
                    profit_pct = float(profit_match.group(1)) if profit_match else 0.0
                    
                    # Absolute profit USDT
                    abs_match = re.search(r'Absolute profit\s*│\s*([-?\d.]+)\s*USDT', stdout)
                    if not abs_match:
                        abs_match = re.search(r'Absolute profit[\s:]*([-?\d.]+)\s*USDT', stdout, re.IGNORECASE)
                    profit_abs = float(abs_match.group(1)) if abs_match else 0.0
                    
                    # Total trades - look for TOTAL row: TOTAL │ 7367 │ ...
                    trades_match = re.search(r'TOTAL\s*│\s*(\d+)', stdout)
                    if not trades_match:
                        trades_match = re.search(r'Total/Daily Avg Trades\s*[:/\-]?\s*(\d+)', stdout, re.IGNORECASE)
                    total_trades = int(trades_match.group(1)) if trades_match else 0
                    
                    # Win rate - last number in TOTAL row before │
                    # Pattern: TOTAL │ 7367 │ ... │ 6420     0   947  87.1 │
                    win_match = re.search(r'TOTAL\s*│[^│]+│[^│]+│[^│]+│[^│]+│[^│]+│\s*\d+\s+\d+\s+\d+\s+([\d.]+)', stdout)
                    if not win_match:
                        win_match = re.search(r'Win[\s%]*[:\-]?\s*([\d.]+)', stdout, re.IGNORECASE)
                    win_rate = float(win_match.group(1)) if win_match else 0.0
                    
                    # Max Drawdown
                    drawdown_match = re.search(r'Max Drawdown\s*│\s*([-?\d.]+)%', stdout)
                    if not drawdown_match:
                        drawdown_match = re.search(r'Max [Dd]rawdown[\s%]*[:\-]?\s*([\d.]+)', stdout, re.IGNORECASE)
                    max_drawdown = float(drawdown_match.group(1)) if drawdown_match else 0.0
                    
                    # Sharpe
                    sharpe_match = re.search(r'Sharpe\s*│\s*([-?\d.]+)', stdout)
                    if not sharpe_match:
                        sharpe_match = re.search(r'Sharpe[\s:]*([-?\d.]+)', stdout, re.IGNORECASE)
                    sharpe = float(sharpe_match.group(1)) if sharpe_match else 0.0
                    
                    # Sortino
                    sortino_match = re.search(r'Sortino\s*│\s*([-?\d.]+)', stdout)
                    if not sortino_match:
                        sortino_match = re.search(r'Sortino[\s:]*([-?\d.]+)', stdout, re.IGNORECASE)
                    sortino = float(sortino_match.group(1)) if sortino_match else 0.0
                    
                    # Profit factor
                    profit_factor_match = re.search(r'Profit Factor\s*│\s*([\d.]+)', stdout)
                    if not profit_factor_match:
                        profit_factor_match = re.search(r'Profit [Ff]actor[\s:]*([\d.]+)', stdout, re.IGNORECASE)
                    profit_factor = float(profit_factor_match.group(1)) if profit_factor_match else 0.0
                    
                    # Avg profit - from TOTAL row, second column
                    avg_match = re.search(r'TOTAL\s*│\s*\d+\s*│\s*([-?\d.]+)', stdout)
                    if avg_match:
                        avg_profit = float(avg_match.group(1))
                    avg_profit_match = re.search(r'Avg profit[:\s]+([-?\d.]+)%', result.stdout, re.IGNORECASE)
                    avg_profit = float(avg_profit_match.group(1)) if avg_profit_match else 0.0
                    
                    logger.info(f"Parsed: profit={profit_pct}%, abs={profit_abs} USDT, trades={total_trades}, win={win_rate}%, dd={max_drawdown}%, sharpe={sharpe}")
                    
                    # Try to read the export file for more details
                    if os.path.exists(host_export_path):
                        try:
                            with open(host_export_path, 'r') as f:
                                export_data = json.load(f)
                                logger.info(f"Loaded export file: {host_export_path}")
                        except Exception as e:
                            logger.warning(f"Could not read export file: {e}")
                    
                    # Insert into backtest_results table using SQLAlchemy (async)
                    try:
                        import psycopg2
                        from src.config import settings
                        
                        # FIX: Use settings.database.url not settings.DATABASE_URL
                        db_url = settings.database.url.replace('postgresql+asyncpg://', 'postgresql://')
                        conn = psycopg2.connect(db_url)
                        cur = conn.cursor()
                        cur.execute("""
                            INSERT INTO backtest_results (
                                strategy_name, timeframe, timerange,
                                total_profit_pct, total_profit_abs, total_trades,
                                win_rate, max_drawdown_pct, sharpe, sortino, profit_factor, avg_profit_pct,
                                backtest_date
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        """, (
                            strategy_name, timeframe, timerange,
                            profit_pct, profit_abs, total_trades,
                            win_rate, max_drawdown, sharpe, sortino, profit_factor, avg_profit,
                            host_export_path
                        ))
                        conn.commit()
                        cur.close()
                        conn.close()
                        db_inserted = True
                        logger.info(f"✅ INSERTED backtest result into database for {strategy_name}")
                    except ImportError:
                        logger.warning("psycopg2 not available - results saved to file only, will be picked up by scanner")
                    except Exception as e:
                        logger.error(f"❌ DB insert failed: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                    
                except Exception as e:
                    logger.error(f"❌ Parsing failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # ALWAYS save result summary JSON (even if parsing failed)
                try:
                    result_data = {
                        'strategy': strategy_name,
                        'type': 'backtest',
                        'status': 'completed' if result.returncode == 0 else 'error',
                        'timestamp': datetime.now().isoformat(),
                        'profit_pct': profit_pct,
                        'profit_abs': profit_abs,
                        'total_trades': total_trades,
                        'win_rate': win_rate,
                        'max_drawdown': max_drawdown,
                        'avg_profit_pct': avg_profit,
                        'sharpe': sharpe,
                        'sortino': sortino,
                        'profit_factor': profit_factor,
                        'export_file': host_export_path,
                        'parsed': True,
                        'db_inserted': db_inserted
                    }
                    
                    result_file = f"{results_dir}/{strategy_name}_{timestamp}.json"
                    logger.info(f"Attempting to save JSON to: {result_file}")
                    with open(result_file, 'w') as f:
                        json.dump(result_data, f, indent=2)
                    logger.info(f"✅ Saved backtest result to {result_file}")
                except Exception as e:
                    logger.error(f"❌ FAILED to save JSON: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                logger.info(f"Backtest stdout: {result.stdout[:2000]}")
                logger.error(f"Backtest stderr: {result.stderr[:2000]}")
                logger.info(f"Backtest return code: {result.returncode}")
                if result.returncode != 0:
                    logger.error(f"Backtest FAILED for {strategy_name}")
                else:
                    logger.info(f"Backtest completed for {strategy_name}")
                    
                    # AUTO-IMPORT: Use raw log importer
                    try:
                        logger.info(f"Auto-importing backtest results for {strategy_name}")
                        result = subprocess.run(
                            ['python3', os.path.join(paths["scripts_dir"], 'import-meta-backtests.py')],
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        logger.info(f"Auto-import output: {result.stdout[:500]}")
                        if result.stderr:
                            logger.error(f"Auto-import stderr: {result.stderr[:500]}")
                    except Exception as e:
                        logger.error(f"Auto-import failed: {e}")
            
            if 'hyperopt' in steps:
                logger.info(f"Running hyperopt for {strategy_name}, epochs={epochs}")
                results_dir2 = paths["hyperopt_results_dir"]
                os.makedirs(results_dir2, exist_ok=True)
                timestamp = int(time.time())
                
                # Read strategy content
                try:
                    with open(dest_path, 'r') as f:
                        strategy_content = f.read()
                except Exception as e:
                    logger.warning(f"Could not read strategy: {e}")
                    strategy_content = ""
                
                # RESET and detect available hyperopt spaces
                available_spaces = ["buy"]  # RESET here, not append!
                
                try:
                    content_lower = strategy_content.lower()
                    if "sell" in content_lower and "parameter" in content_lower:
                        available_spaces.append("sell")
                    if "protection" in content_lower and "parameter" in content_lower:
                        available_spaces.append("protection")
                    if "stoploss" in content_lower and "parameter" in content_lower:
                        available_spaces.append("stoploss")
                    if "roi" in content_lower and "parameter" in content_lower:
                        available_spaces.append("roi")
                    if "trailing" in content_lower and "parameter" in content_lower:
                        available_spaces.append("trailing")
                except Exception as e:
                    logger.warning(f"Space detection error: {e}")
                
                logger.info(f"Hyperopt spaces: {available_spaces}")

                # Use same timerange as backtest
                cmd = [
                    "docker", "run",
                    "--name", f"hyperopt_{strategy_name}_{timestamp}",
                    "-v", f"{freqtrade_strat_dir}:/freqtrade/user_data/strategies",
                    "-v", f"{paths['freqtrade_data_dir']}:/freqtrade/user_data/data",
                    "-v", f"{paths['freqtrade_config_dir']}:/freqtrade/config",
                    "-v", f"{results_dir2}:/freqtrade/user_data/hyperopt_results",
                    "-v", f"{paths['freqtrade_logs_dir']}:/freqtrade/user_data/logs",
                    "freqtradeorg/freqtrade:stable_freqaitorch",
                    "hyperopt",
                    "--strategy", strategy_class,
                    "--config", "/freqtrade/config/config-torch.json",
                    "--timerange", timerange,
                    "--timeframe", timeframe,
                    "--epochs", str(epochs),
                    "--spaces", "all",
                    "--hyperopt-loss", "SharpeHyperOptLoss",
                    "-j", "4"
                ]
                logger.info(f"Docker command: {' '.join(cmd)}")
                process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                stdout, stderr = await process.communicate()
                result = type('Result', (), {'stdout': stdout.decode(), 'stderr': stderr.decode(), 'returncode': process.returncode})()

                
                # ALWAYS save raw results (like backtest)
                try:
                    raw_log_file = f"{results_dir2}/{strategy_name}_{timestamp}_raw.log"
                    logger.info(f"Attempting to save hyperopt raw log to: {raw_log_file}")
                    with open(raw_log_file, 'w') as f:
                        f.write("=== STDOUT ===\n")
                        f.write(result.stdout if result.stdout else "")
                        f.write("\n=== STDERR ===\n")
                        f.write(result.stderr if result.stderr else "")
                        f.write(f"\n=== RETURN CODE ===\n{result.returncode}")
                    logger.info(f"✅ Saved hyperopt raw log to {raw_log_file}")
                except Exception as e:
                    logger.error(f"❌ Failed to save hyperopt raw log: {e}")
                
                # Parse hyperopt results from stdout (like backtest)
                best_epoch = None
                try:
                    stdout = result.stdout if result.stdout else ""
                    
                    # Look for "Best result:" or "Best epoch:" in stdout
                    # Pattern: Best result: Epoch 42 - Profit 15.2%
                    best_match = re.search(r'Best\s+(?:result|epoch):?\s*Epoch\s*(\d+)', stdout, re.IGNORECASE)
                    if best_match:
                        epoch_num = int(best_match.group(1))
                        logger.info(f"Found best epoch from stdout: {epoch_num}")
                        
                        # Try to parse epoch details from table
                        # Look for epoch row in results table
                        epoch_pattern = rf'{epoch_num}\s*│\s*([\d.]+)\s*│\s*([\d.]+)\s*│\s*([\d.]+)\s*│\s*(\d+)'
                        epoch_match = re.search(epoch_pattern, stdout)
                        if epoch_match:
                            best_epoch = {
                                'epoch': epoch_num,
                                'profit_mean_pct': float(epoch_match.group(1)),
                                'profit_total_pct': float(epoch_match.group(2)),
                                'winrate': float(epoch_match.group(3)) / 100,  # Convert to decimal
                                'trade_count': int(epoch_match.group(4))
                            }
                            logger.info(f"Parsed epoch {epoch_num} details from stdout")
                    
                    # Also look for sharpe/sortino in summary
                    sharpe_match = re.search(r'Sharpe\s*[:\-]?\s*([-?\d.]+)', stdout, re.IGNORECASE)
                    if sharpe_match and best_epoch:
                        best_epoch['sharpe'] = float(sharpe_match.group(1))
                    
                    sortino_match = re.search(r'Sortino\s*[:\-]?\s*([-?\d.]+)', stdout, re.IGNORECASE)
                    if sortino_match and best_epoch:
                        best_epoch['sortino'] = float(sortino_match.group(1))
                        
                except Exception as e:
                    logger.error(f"Could not parse hyperopt stdout: {e}")
                    
                # Fallback: try fthypt file
                if not best_epoch:
                    try:
                        import glob
                        fthypt_files = glob.glob(f"{results_dir2}/.fthypt*{strategy_name}*")
                        if fthypt_files:
                            latest_file = max(fthypt_files, key=os.path.getmtime)
                            logger.info(f"Found fthypt file: {latest_file}")
                            with open(latest_file, 'r') as f:
                                lines = f.readlines()
                                for line in reversed(lines):
                                    if '"epoch"' in line:
                                        try:
                                            epoch_data = json.loads(line)
                                            if epoch_data.get('is_best'):
                                                best_epoch = epoch_data
                                                logger.info(f"Found best epoch from fthypt: {epoch_data.get('epoch')}")
                                                break
                                        except:
                                            pass
                    except Exception as e2:
                        logger.error(f"Could not parse fthypt: {e2}")
                
                # Save results to JSON file in Multibotdashboard
                try:
                    import json
                    from datetime import datetime
                    
                    result_data = {
                        'strategy': strategy_name,
                        'type': 'hyperopt',
                        'status': 'completed' if result.returncode == 0 else 'error',
                        'timestamp': datetime.now().isoformat(),
                        'epochs': epochs,
                        'profit_pct': best_epoch.get('profit_total_pct') if best_epoch else None,
                        'drawdown': best_epoch.get('max_drawdown') if best_epoch else None,
                        'best_epoch': best_epoch,
                        'export_file': host_export_path,  # ADD EXPORT FILE PATH
                        'stdout': result.stdout[-5000:] if result.stdout else None,
                        'stderr': result.stderr[-2000:] if result.stderr else None
                    }
                    
                    # Save to Multibotdashboard results folder
                    # Use SAME timestamp as above
                    result_file = f"{results_dir2}/{strategy_name}_{timestamp}.json"
                    logger.info(f"Attempting to save hyperopt JSON to: {result_file}")
                    with open(result_file, 'w') as f:
                        json.dump(result_data, f, indent=2)
                    logger.info(f"✅ Saved hyperopt result to {result_file}")
                    
                    # INSERT INTO DATABASE
                    if best_epoch:
                        try:
                            import psycopg2
                            from src.config import settings
                            
                            db_url = settings.database.url.replace('postgresql+asyncpg://', 'postgresql://')
                            conn = psycopg2.connect(db_url)
                            cur = conn.cursor()
                            
                            cur.execute("""
                                INSERT INTO hyperopt_epochs (
                                    strategy_name, epoch, profit_pct, win_rate, 
                                    avg_profit, total_trades, sharpe, sortino,
                                    params, is_best, created_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                            """, (
                                strategy_name,
                                best_epoch.get('epoch', 0),
                                best_epoch.get('profit_total_pct', 0),
                                best_epoch.get('winrate', 0) * 100,
                                best_epoch.get('profit_mean_pct', 0),
                                best_epoch.get('trade_count', 0),
                                best_epoch.get('sharpe', 0),
                                best_epoch.get('sortino', 0),
                                json.dumps(best_epoch.get('params', {})),
                                True
                            ))
                            conn.commit()
                            cur.close()
                            conn.close()
                            logger.info(f"✅ INSERTED hyperopt result into database for {strategy_name}")
                        except Exception as e:
                            logger.error(f"❌ Hyperopt DB insert failed: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                    
                    # AUTO-IMPORT: Also run import script (for consistency)
                    try:
                        logger.info(f"Auto-importing hyperopt results for {strategy_name}")
                        result_import = await async_subprocess_run(
                            ['python3', './scripts/import-hyperopt.py'],
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        logger.info(f"Hyperopt auto-import output: {result_import.stdout[:500]}")
                        if result_import.stderr:
                            logger.error(f"Hyperopt auto-import stderr: {result_import.stderr[:500]}")
                    except Exception as e:
                        logger.error(f"Hyperopt auto-import failed: {e}")
                except Exception as e:
                    logger.error(f"Failed to save hyperopt result: {e}")
                
                logger.info(f"Hyperopt stdout: {result.stdout[:2000]}")
                logger.error(f"Hyperopt stderr: {result.stderr[:2000]}")
                logger.info(f"Hyperopt return code: {result.returncode}")
                if result.returncode != 0:
                    logger.error(f"Hyperopt FAILED for {strategy_name}")
                else:
                    logger.info(f"Hyperopt completed for {strategy_name}")
                
        except Exception as e:
            logger.error(f"Workflow error: {e}")
        finally:
            # Cleanup copied strategy file
            try:
                copied_file = os.path.join(freqtrade_strat_dir, f"{strategy_name}.py")
                if os.path.exists(copied_file):
                    os.remove(copied_file)
                    logger.info(f"Cleaned up strategy file: {copied_file}")
            except Exception as e:
                logger.warning(f"Could not cleanup strategy file: {e}")
    
    # Start background task (non-blocking)
    asyncio.create_task(run_workflow())
    
    return {
        "status": "started",
        "strategy": strategy_name,
        "bot_id": bot_id,
        "bot_name": bot_name,
        "steps": steps,
        "message": f"Workflow started: {', '.join(steps)}"
    }


@router.post("/bots/{bot_id}/workflow/stop")
async def stop_workflow(
    bot_id: str,
    session: AsyncSession = Depends(get_db)
) -> dict:
    """Stop running workflow for a bot"""
    if not app_state:
        raise HTTPException(status_code=503, detail="Strategy Lab not initialized")
    
    # Get bot to find its strategy name
    result = await session.get(Bot, bot_id)
    if not result:
        return {"status": "idle", "message": "Bot not found"}
    
    bot = result
    strategy_name = bot.strategy or bot.name
    
    process = app_state.get_process(ProcessType.HYPEROPT, strategy_name)
    if not process:
        process = app_state.get_process(ProcessType.BACKTEST, strategy_name)
    
    if not process:
        return {"status": "idle", "message": "No workflow to stop"}
    
    # Stop the process
    await workflow_engine.stop_process(strategy_name)
    
    return {
        "status": "stopped",
        "bot_id": bot_id,
        "message": "Workflow stopped"
    }


@router.post("/import-backtest-results")
async def import_backtest_results(
    session: AsyncSession = Depends(get_db)
) -> dict:
    """Import backtest results from JSON files into database"""
    imported = 0
    errors = []
    paths = await get_runtime_paths()
    results_dir = paths["backtest_results_dir"]
    if not os.path.exists(results_dir):
        return {"status": "error", "message": "Results directory not found"}
    
    for filename in os.listdir(results_dir):
        if not filename.endswith('.json') or filename.endswith('_export.json'):
            continue  # Skip export files and non-JSON
            
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Skip if already has 'db_imported' flag
            if data.get('db_imported'):
                continue
                
            strategy = data.get('strategy') or data.get('strategy_name')
            if not strategy:
                continue
                
            # Check if already in database
            result = await session.execute(
                text("""
                    SELECT id FROM backtest_results 
                    WHERE strategy_name = :strategy 
                    AND backtest_date > NOW() - INTERVAL '1 hour'
                """),
                {"strategy": strategy}
            )
            if result.fetchone():
                continue  # Already imported recently
            
            # Insert into database (without export_file_path - column doesn't exist)
            await session.execute(
                text("""
                    INSERT INTO backtest_results (
                        strategy_name, timeframe, timerange,
                        total_profit_pct, total_profit_abs, total_trades,
                        win_rate, max_drawdown_pct, sharpe, avg_profit_pct,
                        backtest_date
                    ) VALUES (
                        :strategy_name, :timeframe, :timerange,
                        :profit_pct, :profit_abs, :total_trades,
                        :win_rate, :max_drawdown, :sharpe, :avg_profit_pct,
                        NOW()
                    )
                """),
                {
                    "strategy_name": strategy,
                    "timeframe": data.get('timeframe', '15m'),
                    "timerange": data.get('timerange', '20241001-20260224'),
                    "profit_pct": data.get('profit_pct', 0) or 0,
                    "profit_abs": data.get('profit_abs', 0) or 0,
                    "total_trades": data.get('total_trades', 0) or 0,
                    "win_rate": data.get('win_rate', 0) or 0,
                    "max_drawdown": data.get('max_drawdown', 0) or 0,
                    "sharpe": data.get('sharpe', 0) or 0,
                    "avg_profit_pct": data.get('avg_profit_pct', 0) or 0
                }
            )
            await session.commit()
            
            # Mark as imported
            data['db_imported'] = True
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            imported += 1
            logger.info(f"Imported {filename} to database")
            
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
            logger.error(f"Failed to import {filename}: {e}")
    
    return {
        "status": "success",
        "imported": imported,
        "errors": errors
    }


@router.post("/hyperopt/{strategy_name}/extract/{epoch_number}")
async def extract_epoch_params(
    strategy_name: str,
    epoch_number: int,
    session: AsyncSession = Depends(get_db)
) -> dict:
    """Extract parameters from a specific epoch"""
    if not workflow_engine:
        raise HTTPException(status_code=503, detail="Strategy Lab not initialized")
    
    # Run hyperopt-show command
    success = await workflow_engine.extract_epoch(strategy_name, epoch_number)
    
    if success:
        return {
            "status": "success",
            "strategy": strategy_name,
            "epoch": epoch_number,
            "message": f"Parameters extracted and saved"
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to extract parameters")


@router.get("/optimization-results")
async def get_optimization_results(
    limit: int = 50,
    session: AsyncSession = Depends(get_db)
) -> List[dict]:
    """Get optimization results from BOTH optimization_runs AND backtest_results tables"""
    results = []
    
    # FIRST: Get from optimization_runs table (primary for workflow runs)
    try:
        opt_results = await session.execute(
            text("""
                SELECT 
                    id, bot_id, strategy_name, process_type, status,
                    started_at, completed_at, duration_seconds,
                    result_profit_pct, result_drawdown, result_trade_count,
                    config, error_message
                FROM optimization_runs
                ORDER BY started_at DESC
                LIMIT :limit
            """),
            {"limit": limit}
        )
        
        for row in opt_results.fetchall():
            results.append({
                "id": f"opt_{row[0]}",
                "bot_id": str(row[1]) if row[1] else None,
                "strategy_name": row[2],
                "process_type": row[3],
                "type": row[3],
                "status": row[4],
                "started_at": row[5].isoformat() if row[5] else None,
                "completed_at": row[6].isoformat() if row[6] else None,
                "duration_seconds": row[7],
                "result_profit_pct": float(row[8]) if row[8] is not None else None,
                "result_drawdown": float(row[9]) if row[9] is not None else None,
                "result_trade_count": row[10],
                "config": row[11],
                "error_message": row[12],
            })
    except Exception as e:
        logger.warning(f"optimization_runs query failed: {e}")
    
    # SECOND: Also get from backtest_results table (for backtests run via workflow)
    # Only add if we have fewer than limit results
    if len(results) < limit:
        remaining = limit - len(results)
        try:
            db_results = await session.execute(
                text("""
                    SELECT 
                        id, strategy_name, timeframe, timerange,
                        total_profit_pct, total_profit_abs, total_trades,
                        win_rate, avg_profit_pct, max_drawdown_pct, sharpe,
                        backtest_date
                    FROM backtest_results
                    ORDER BY backtest_date DESC
                    LIMIT :limit
                """),
                {"limit": remaining}
            )
            
            for row in db_results.fetchall():
                backtest_date = row[11]
                
                results.append({
                    "id": f"backtest_{row[0]}",
                    "strategy_name": row[1],
                    "process_type": "backtest",
                    "type": "backtest",
                    "status": "completed",
                    "started_at": backtest_date.isoformat() if backtest_date else None,
                    "completed_at": backtest_date.isoformat() if backtest_date else None,
                    "result_profit_pct": float(row[4]) if row[4] is not None else None,
                    "result_drawdown": float(row[9]) if row[9] is not None else None,
                    "result_trade_count": row[6],
                    "win_rate": float(row[7]) if row[7] is not None else None,
                    "sharpe": float(row[10]) if row[10] is not None else None,
                    "timeframe": row[2],
                    "timerange": row[3],
                })
        except Exception as e:
            logger.warning(f"backtest_results query failed: {e}")
    
    # Sort by started_at (newest first)
    results.sort(key=lambda x: x.get('started_at') or '', reverse=True)
    
    return results[:limit]


@router.post("/optimization-results/import")
async def import_results_to_db(
    session: AsyncSession = Depends(get_db)
) -> dict:
    """Import all JSON results into database"""
    imported = 0
    paths = await get_runtime_paths()
    
    # Import backtest results
    backtest_dir = paths["backtest_results_dir"]
    if os.path.exists(backtest_dir):
        for filename in os.listdir(backtest_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(backtest_dir, filename), 'r') as f:
                        data = json.load(f)
                        # TODO: Insert into optimization_runs table
                        imported += 1
                except Exception as e:
                    logger.warning(f"Could not import {filename}: {e}")
    
    # Import hyperopt results
    hyperopt_dir = paths["hyperopt_results_dir"]
    if os.path.exists(hyperopt_dir):
        for filename in os.listdir(hyperopt_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(hyperopt_dir, filename), 'r') as f:
                        data = json.load(f)
                        # TODO: Insert into optimization_runs table
                        imported += 1
                except Exception as e:
                    logger.warning(f"Could not import {filename}: {e}")
    
    return {"status": "success", "imported": imported}


# ── Replay workflow integration ────────────────────────────────────────

@router.post("/workflow/replay")
async def start_replay_workflow(
    config: dict,
    session: AsyncSession = Depends(get_db)
) -> dict:
    """Start a replay run as a workflow step. Delegates to /replay/run under the hood."""
    strategy_name = config.get("strategy")
    timerange = config.get("timerange")
    if not strategy_name or not timerange:
        raise HTTPException(status_code=400, detail="strategy and timerange are required")

    # Re-use the replay service logic via a direct DB insert + background task
    from src.services.replay_service import get_replay_service
    replay_svc = get_replay_service()

    bot_id = config.get("bot_id")
    timeframe = config.get("timeframe")

    result = await session.execute(
        text(
            """
            INSERT INTO replay_runs
                (bot_id, strategy_name, timerange, timeframe, status, notes)
            VALUES
                (:bot_id, :strategy_name, :timerange, :timeframe, 'pending', :notes)
            RETURNING id
            """
        ),
        {
            "bot_id": bot_id,
            "strategy_name": strategy_name,
            "timerange": timerange,
            "timeframe": timeframe,
            "notes": config.get("notes", ""),
        },
    )
    row = result.fetchone()
    replay_id = row[0] if row else None
    await session.commit()

    # Kick off async background execution
    asyncio.create_task(
        replay_svc.run_replay(
            db=session,
            replay_id=replay_id,
            strategy_name=strategy_name,
            timerange=timerange,
            bot_id=bot_id,
            timeframe=timeframe,
        )
    )

    return {
        "status": "started",
        "replay_id": replay_id,
        "strategy": strategy_name,
        "timerange": timerange,
        "message": "Replay queued via Strategy Lab workflow",
    }


# WebSocket for real-time updates
@router.websocket("/ws")
async def strategy_lab_websocket(websocket: WebSocket):
    """WebSocket for real-time workflow and hyperopt updates"""
    await websocket.accept()
    
    if not app_state:
        await websocket.close(code=1011, reason="Strategy Lab not initialized")
        return
    
    # Subscribe to state updates
    async def state_callback(update: dict):
        await websocket.send_json({
            "type": "state_update",
            "data": update
        })
    
    app_state.subscribe(state_callback)
    
    try:
        while True:
            # Keep connection alive, handle client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("action") == "ping":
                await websocket.send_json({"type": "pong"})
            
    except WebSocketDisconnect:
        app_state.unsubscribe(state_callback)
