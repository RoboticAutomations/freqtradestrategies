# user_data/strategies/ml_pairlist_selector.py

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

# Add freqtrade to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from freqtrade.configuration import Configuration
from freqtrade.resolvers import StrategyResolver
from pathlib import Path
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
RESULTS_HISTORY_FILE = Path("user_data/pairlist/ml_pairlist_history.json")
ASCII_ART = '''
    ╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                                                        ║
    ║  ██████╗ ██╗     ███████╗██╗  ██╗ ██████╗██████╗ ██╗   ██╗██████╗ ████████╗ ██████╗ ██╗  ██╗██╗███╗   ██╗ ██████╗      ║
    ║  ██╔══██╗██║     ██╔════╝╚██╗██╔╝██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗╚══██╔══╝██╔═══██╗██║ ██╔╝██║████╗  ██║██╔════╝      ║
    ║  ███████║██║     █████╗   ╚███╔╝ ██║     ██████╔╝ ╚████╔╝ ██████╔╝   ██║   ██║   ██║█████╔╝ ██║██╔██╗ ██║██║  ███╗     ║
    ║  ██╔══██║██║     ██╔══╝   ██╔██╗ ██║     ██╔══██╗  ╚██╔╝  ██╔══╝     ██║   ██║   ██║██╔═██╗ ██║██║╚██╗██║██║   ██║     ║
    ║  ██║  ██║███████╗███████╗██╔╝ ██╗╚██████╗██║  ██║   ██║   ██║        ██║   ╚██████╔╝██║  ██╗██║██║ ╚████║╚██████╔╝     ║
    ║  ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝      ║
    ║                                                                                                                        ║
    ║                                       PAIRLIST SELECTOR                                                                ║
    ║                                       OPTIMIZING PAIRLISTS FOR STRATEGYS                                               ║
    ║                                       WORKS FOR ML AND STANDARD STRATEGIES                                             ║
    ║                                                                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
'''
class MLPairlistOptimizer:
    def __init__(self, config_path: str, strategy_name: str, pairs_file: str = None, download_data: bool = True, download_days: int = 60, fullbacktest: bool = False, individual_backtest: bool = False, backtest_days: int = None):
        self.test_mode = True
        self.fullbacktest_mode = fullbacktest
        self.individual_backtest = individual_backtest
        self.backtest_days = backtest_days
        """Initialize with Freqtrade config and strategy"""
        # Load config
        self.config = Configuration.from_files([config_path])
        
        # Add strategy to config for StrategyResolver
        self.config['strategy'] = strategy_name
        
        # Store parameters
        self.pairs_file = pairs_file
        self.download_data = download_data
        self.download_days = download_days
        
        # Set backtest timerange if specified
        if self.backtest_days:
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.backtest_days)
            timerange_str = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
            self.config['timerange'] = timerange_str
            logger.info(f"📅 Backtest timerange set: {self.backtest_days} days ({timerange_str})")
        
        # Load strategy
        self.strategy = StrategyResolver.load_strategy(self.config)
        
        # Get data handler
        from freqtrade.data.history import get_datahandler
        self.datahandler = get_datahandler(
            self.config['datadir'],
            self.config.get('dataformat_ohlcv', 'feather')
        )
        
        logger.info(f"✅ Loaded strategy: {strategy_name}")
        logger.info(f"✅ Timeframe: {self.strategy.timeframe}")
        logger.info(f"✅ Data directory: {self.config['datadir']}")
        if self.fullbacktest_mode:
            mode_desc = "INDIVIDUAL (Each pair separately)" if self.individual_backtest else "BATCH (All pairs at once - 3-5x faster!)"
            logger.info(f"🎯 Mode: FULL BACKTEST - {mode_desc}")
        else:
            logger.info(f"🎯 Mode: {'ML TRAINING' if self._strategy_has_ml() else 'STANDARD BACKTEST'}")
        
    def disable_optuna_runtime(self):
        """
        Disable Optuna by neutralizing RealOptunaManager
        without modifying the strategy.
        """

        # Walk all attributes on the strategy
        for attr in dir(self.strategy):
            try:
                obj = getattr(self.strategy, attr)
            except Exception:
                continue

            # Case 1: instance of RealOptunaManager
            if obj.__class__.__name__ == "RealOptunaManager":
                logger.info("🔕 Disabling RealOptunaManager (runtime)")

                # Replace all public methods with no-ops
                for method_name in dir(obj):
                    if method_name.startswith("_"):
                        continue

                    try:
                        method = getattr(obj, method_name)
                        if callable(method):
                            setattr(obj, method_name, lambda *a, **k: 0.0)
                    except Exception:
                        pass

                # Optional: hard-disable flag if present
                for flag in [
                    "enabled",
                    "optuna_enabled",
                    "active",
                    "running",
                ]:
                    if hasattr(obj, flag):
                        try:
                            setattr(obj, flag, False)
                        except Exception:
                            pass


    def _strategy_has_ml(self) -> bool:
        return hasattr(self.strategy, 'train_ml_from_backtest_single_pair')
    
    def append_results_history(self, results: List[Dict[str, Any]]):
        """Append ML pair results to a cumulative history file"""

        entry = {
            "timestamp": datetime.now().isoformat(),
            "strategy": self.strategy.__class__.__name__,
            "timeframe": self.strategy.timeframe,
            "mode": "fullbacktest" if self.fullbacktest_mode else ("ml" if self._strategy_has_ml() else "standard"),
            "results": results,
        }

        # Load existing history
        if RESULTS_HISTORY_FILE.exists():
            with open(RESULTS_HISTORY_FILE, "r") as f:
                history = json.load(f)
        else:
            history = []

        history.append(entry)

        with open(RESULTS_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(
            f"📚 ML history updated: {RESULTS_HISTORY_FILE} "
            f"(runs stored: {len(history)})"
        )

    def generate_dynamic_pairlist(self, output_file: str = None) -> List[str]:
        """
        Generate dynamic pairlist by calling freqtrade list-pairs command.
        If the output file already exists, it will be reused (no regeneration).
        """
        import subprocess
        from pathlib import Path

        # ------------------------------------------------------------------
        # Determine output file
        # ------------------------------------------------------------------
        if not output_file:
            output_file = 'user_data/pairlist/dynamic_pairs.json'

        output_path = Path(output_file)

        # ------------------------------------------------------------------
        # 🛑 CACHE GUARD: reuse existing dynamic pairlist
        # ------------------------------------------------------------------
        if output_path.exists():
            logger.warning(
                f"🧪 Using existing dynamic pairlist (skipping regeneration): "
                f"{output_path}"
            )
            try:
                with open(output_path, 'r') as f:
                    data = json.load(f)

                pairs = data.get('pairs', [])
                logger.info(f"📄 Loaded {len(pairs)} pairs from cached dynamic pairlist")
                return pairs

            except Exception as e:
                logger.error(
                    f"❌ Failed to load existing dynamic pairlist "
                    f"(will regenerate): {e}"
                )
                # fall through to regeneration

        # ------------------------------------------------------------------
        # Generate dynamic pairlist via freqtrade
        # ------------------------------------------------------------------
        logger.info(f"\n{'=' * 70}")
        logger.info("🔍 GENERATING DYNAMIC PAIRLIST")
        logger.info(f"{'=' * 70}")

        try:
            config_file = self.config.get('config_files', ['config.json'])[0]

            logger.info("📊 Running freqtrade list-pairs command...")
            logger.info(f"   Config: {config_file}")
            logger.info(f"   Output: {output_file}")

            cmd = [
                'freqtrade',
                'list-pairs',
                '--config', config_file,
                '--quote', 'USDT',
                '--print-json'
            ]

            logger.info(f"\n   Executing: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                logger.error(f"❌ Command failed with return code {result.returncode}")
                logger.error(f"   Error output: {result.stderr}")
                return []

            # ------------------------------------------------------------------
            # Parse JSON output
            # ------------------------------------------------------------------
            try:
                pairs_data = json.loads(result.stdout)

                if isinstance(pairs_data, dict) and 'pairs' in pairs_data:
                    pairs = pairs_data['pairs']
                elif isinstance(pairs_data, list):
                    pairs = pairs_data
                else:
                    logger.error(f"❌ Unexpected JSON format: {type(pairs_data)}")
                    return []

                logger.info(f"\n✅ Generated {len(pairs)} pairs from freqtrade list-pairs")

                # Log applied filters
                if 'pairlists' in self.config:
                    logger.info("\n   Filters applied:")
                    for i, pairlist in enumerate(self.config['pairlists'], 1):
                        logger.info(f"      {i}. {pairlist.get('method', 'Unknown')}")

                # Preview pairs
                if pairs:
                    preview_count = min(20, len(pairs))
                    logger.info(f"\n   Preview ({preview_count} of {len(pairs)}):")
                    for i, pair in enumerate(pairs[:preview_count], 1):
                        logger.info(f"      {i:2d}. {pair}")
                    if len(pairs) > preview_count:
                        logger.info(f"      ... and {len(pairs) - preview_count} more")
                else:
                    logger.warning("\n⚠️  No pairs returned! Check your pairlist filters.")

                # ------------------------------------------------------------------
                # Save to file
                # ------------------------------------------------------------------
                if pairs:
                    output_data = {
                        'generated_at': datetime.now().isoformat(),
                        'total_pairs': len(pairs),
                        'pairs': pairs,
                        'filters_applied': [
                            p.get('method', 'Unknown')
                            for p in self.config.get('pairlists', [])
                        ]
                    }

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'w') as f:
                        json.dump(output_data, f, indent=2)

                    logger.info(f"\n✅ Saved pairlist to: {output_path}")

                return pairs

            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse JSON output: {e}")
                logger.error(f"   Raw output: {result.stdout[:500]}")
                return []

        except subprocess.TimeoutExpired:
            logger.error("❌ Command timed out after 60 seconds")
            return []
        except Exception as e:
            logger.error(f"❌ Failed to generate dynamic pairlist: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    
    def download_missing_data(self, pairs: List[str]) -> None:
        """Download historical data for pairs that are missing data"""
        from freqtrade.resolvers import ExchangeResolver
        from freqtrade.data.history import refresh_backtest_ohlcv_data
        from freqtrade.configuration import TimeRange
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📥 DOWNLOADING MISSING DATA")
        logger.info(f"{'='*70}")
        
        if not pairs:
            logger.info("✅ No pairs need downloading")
            return
        
        try:
            # Load exchange
            logger.info("🔌 Connecting to exchange...")
            exchange = ExchangeResolver.load_exchange(self.config, validate=False)
            logger.info(f"✅ Connected to {exchange.name}")
            
            # Determine timeframes to download
            timeframes = [self.strategy.timeframe]
            
            # Add informative timeframes from strategy
            if hasattr(self.strategy, 'informative_pairs'):
                try:
                    for pair_tuple in self.strategy.informative_pairs():
                        if isinstance(pair_tuple, tuple) and len(pair_tuple) >= 2:
                            tf = pair_tuple[1]
                            if tf not in timeframes:
                                timeframes.append(tf)
                except:
                    pass
            
            # Add informative timeframes from config
            if 'informative_timeframes' in self.config:
                for tf in self.config['informative_timeframes'].keys():
                    if tf not in timeframes:
                        timeframes.append(tf)
            
            logger.info(f"   📊 Timeframes: {', '.join(timeframes)}")
            logger.info(f"   🔢 Pairs to download: {len(pairs)}")
            logger.info(f"   📅 Days of history: {self.download_days}")
            logger.info(f"   💾 Data directory: {self.config['datadir']}")
            
            # Calculate timerange - FIX: Create proper TimeRange object
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.download_days)
            
            # Create TimeRange object (format: YYYYMMDD-YYYYMMDD)
            timerange_str = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
            timerange = TimeRange.parse_timerange(timerange_str)
            
            logger.info(f"   📆 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            logger.info(f"   📆 TimeRange: {timerange_str}")
            logger.info(f"\n📥 Starting download... This may take 5-30 minutes depending on pairs count\n")
            
            # Download data - FIX: Pass TimeRange object, not string
            refresh_backtest_ohlcv_data(
                exchange=exchange,
                pairs=pairs,
                timeframes=timeframes,
                datadir=Path(self.config['datadir']),
                timerange=timerange,  # FIX: Pass TimeRange object
                new_pairs_days=self.download_days,
                erase=False,
                data_format=self.config.get('dataformat_ohlcv', 'feather'),
                trading_mode=self.config.get('trading_mode', 'spot')
            )
            
            logger.info(f"\n{'='*70}")
            logger.info(f"✅ DATA DOWNLOAD COMPLETE!")
            logger.info(f"{'='*70}")
            logger.info(f"   Downloaded {len(pairs)} pairs x {len(timeframes)} timeframes")
            logger.info(f"   Total: {len(pairs) * len(timeframes)} datasets\n")
            
        except Exception as e:
            logger.error(f"\n{'='*70}")
            logger.error(f"❌ DATA DOWNLOAD FAILED")
            logger.error(f"{'='*70}")
            logger.error(f"Error: {str(e)}")
            logger.warning(f"⚠️  Continuing with available data...")
            import traceback
            logger.error(traceback.format_exc())
    def analyze_existing_backtest(self, json_path: str = None) -> List[Dict[str, Any]]:
        """
        Analyze an existing backtest JSON file without re-running backtest.
        This is useful when you've already run a backtest and just want to extract/rank pairs.
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 ANALYZING EXISTING BACKTEST JSON")
        logger.info(f"{'='*70}\n")
        
        from pathlib import Path
        
        # ──────────────────────────────────────────────────────────────────
        # FIND JSON FILE
        # ──────────────────────────────────────────────────────────────────
        if json_path:
            json_file = Path(json_path)
        else:
            # Auto-detect most recent backtest file
            results_dir = Path('user_data/backtest_results')
            
            if not results_dir.exists():
                logger.error(f"❌ Backtest results directory not found: {results_dir}")
                return []
            
            json_files = list(results_dir.glob('backtest-result*.json'))
            
            if not json_files:
                logger.error(f"❌ No backtest JSON files found in: {results_dir}")
                logger.info("💡 Run a backtest first, or specify --from-json <path>")
                return []
            
            # Get most recent file
            json_file = max(json_files, key=lambda p: p.stat().st_mtime)
        
        if not json_file.exists():
            logger.error(f"❌ Backtest JSON file not found: {json_file}")
            return []
        
        logger.info(f"📂 Reading: {json_file}")
        logger.info(f"📅 Modified: {datetime.fromtimestamp(json_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # ──────────────────────────────────────────────────────────────────
        # LOAD AND PARSE JSON
        # ──────────────────────────────────────────────────────────────────
        try:
            with open(json_file, 'r') as f:
                backtest_data = json.load(f)
        except Exception as e:
            logger.error(f"❌ Failed to read JSON file: {e}")
            return []
        
        # Extract strategy results
        strategy_name = self.strategy.__class__.__name__
        
        # Try to find strategy data (handle different JSON formats)
        strategy_data = None
        
        # Format 1: strategy → strategy_name → results_per_pair
        if 'strategy' in backtest_data:
            strategy_data = backtest_data['strategy'].get(strategy_name)
        
        # Format 2: Direct results_per_pair at root
        if not strategy_data and 'results_per_pair' in backtest_data:
            strategy_data = backtest_data
        
        if not strategy_data:
            logger.error(f"❌ Strategy '{strategy_name}' not found in JSON")
            logger.info(f"💡 Available strategies: {list(backtest_data.get('strategy', {}).keys())}")
            return []
        
        # Get per-pair results
        pairlist_results = strategy_data.get('results_per_pair', [])
        
        if not pairlist_results:
            logger.error(f"❌ No 'results_per_pair' found in JSON")
            return []
        
        logger.info(f"✅ Found {len(pairlist_results)} pairs in backtest results\n")
        
        # ──────────────────────────────────────────────────────────────────
        # EXTRACT METRICS FOR EACH PAIR
        # ──────────────────────────────────────────────────────────────────
        results = []
        
        for pair_data in pairlist_results:
            pair = pair_data.get('key')
            
            if not pair or pair == 'TOTAL':
                continue
            
            trades = pair_data.get('trades', 0)
            
            # Skip pairs with insufficient trades (unless test mode)
            if trades < 5:
                if self.test_mode:
                    logger.debug(f"⚠️  {pair}: Only {trades} trades – keeping in TEST MODE")
                else:
                    logger.debug(f"⚠️  {pair}: Only {trades} trades – SKIPPING")
                    continue

            # Core performance metrics
            profit_total = pair_data.get('profit_total_pct', 0)
            profit_abs = pair_data.get('profit_total_abs', 0)
            wins = pair_data.get('wins', 0)
            win_rate = (wins / trades * 100) if trades > 0 else 0
            
            # Risk metrics  
            max_dd = abs(pair_data.get('max_drawdown_account', 0))
            
            # Trade distribution
            avg_profit = pair_data.get('profit_mean_pct', 0)
            median_profit = pair_data.get('profit_median_pct', 0)
            
            # Duration (convert to minutes)
            avg_duration_str = pair_data.get('duration_avg', '0:00:00')
            try:
                parts = avg_duration_str.split(':')
                if len(parts) == 3:
                    hours, minutes, seconds = map(float, parts)
                    avg_duration = hours * 60 + minutes + seconds / 60
                else:
                    avg_duration = 0
            except:
                avg_duration = 0
            
            # Advanced metrics
            sharpe_ratio = pair_data.get('sharpe', 0)
            sortino_ratio = pair_data.get('sortino', 0)
            calmar_ratio = pair_data.get('calmar', 0)
            expectancy = pair_data.get('expectancy', 0)
            
            # Streaks
            max_win_streak = pair_data.get('max_streak_win', 0)
            max_loss_streak = pair_data.get('max_streak_loss', 0)

            # ──────────────────────────────────────────────────────────────────
            # CALCULATE COMPOSITE SCORE
            # ──────────────────────────────────────────────────────────────────
            score = (
                (profit_total / 100) * 0.40 +      # 40% weight on total profit
                (win_rate / 100) * 0.25 +          # 25% weight on win rate
                (-max_dd / 100) * 0.20 +           # 20% weight on drawdown (inverted)
                (sharpe_ratio / 3) * 0.10 +        # 10% weight on Sharpe ratio
                (expectancy / 10) * 0.05           # 5% weight on expectancy
            )

            metrics = {
                'pair': pair,
                'trade_count': trades,
                
                # Performance metrics
                'profit_total': profit_total,
                'profit_abs': profit_abs,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'median_profit': median_profit,
                
                # Risk metrics
                'max_drawdown': max_dd,
                'max_drawdown_abs': profit_abs * (max_dd / 100) if profit_abs else 0,
                
                # Advanced metrics
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'expectancy': expectancy,
                
                # Duration
                'avg_duration': avg_duration,
                
                # Streaks
                'max_win_streak': max_win_streak,
                'max_loss_streak': max_loss_streak,
                
                # Composite score for ranking
                'score': score,
                
                # ML-compatible fields (for unified handling)
                'train_accuracy': None,
                'val_accuracy': None,
                'accuracy_gap': None,
            }

            results.append(metrics)
            
            logger.info(
                f"✅ {pair:20s} | Trades={trades:3d} | Profit={profit_total:6.2f}% | "
                f"WR={win_rate:5.1f}% | DD={max_dd:5.2f}% | Sharpe={sharpe_ratio:5.2f}"
            )

        logger.info(f"\n{'='*70}")
        logger.info(f"✅ Extracted metrics for {len(results)} pairs")
        logger.info(f"{'='*70}\n")
        
        return results

    def evaluate_all_pairs_batch_backtest(self, pairs: List[str]) -> List[Dict[str, Any]]:
        """
        Run ONE comprehensive backtest with ALL pairs at once.
        Calculate per-pair metrics from the trades DataFrame.
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🎯 [BATCH-BT] Running comprehensive backtest on ALL {len(pairs)} pairs at once")
        logger.info(f"{'='*70}")
        
        from freqtrade.optimize.backtesting import Backtesting
        import pandas as pd

        bt_config = self.config.copy()
        bt_config['exchange']['pair_whitelist'] = pairs
        bt_config['pairlists'] = [{"method": "StaticPairList", "pairs": pairs}]
        bt_config['max_open_trades'] = min(20, len(pairs))
        bt_config['stake_amount'] = 100
        bt_config['dry_run_wallet'] = 10000
        
        if 'timerange' not in bt_config:
            bt_config['timerange'] = None

        try:
            logger.info(f"⚙️  Initializing backtest engine...")
            logger.info(f"   Max open trades: {bt_config['max_open_trades']}")
            logger.info(f"   Total pairs: {len(pairs)}\n")
            
            backtesting = Backtesting(bt_config)
            backtesting.strategy = self.strategy
            
            logger.info(f"🚀 Running backtest... (this may take 5-15 minutes)\n")
            backtesting.start()
            logger.info(f"✅ Backtest complete! Extracting metrics...\n")
            
            # 🔍 TRY MULTIPLE METHODS TO FIND TRADES DATA
            all_trades = None
            
            # Method 1: Try backtesting.all_results
            if hasattr(backtesting, 'all_results') and backtesting.all_results is not None:
                logger.info("📊 Found trades in backtesting.all_results")
                all_trades = backtesting.all_results
            
            # Method 2: Try backtesting.results (look for trades list/df)
            elif hasattr(backtesting, 'results') and backtesting.results:
                results = backtesting.results
                logger.info(f"🔍 Checking backtesting.results structure...")
                
                # Check if results contains a trades list
                if isinstance(results, dict):
                    # Try to find trades in various locations
                    for key in ['trades', 'results_per_pair', 'all_results', 'pairlist']:
                        if key in results and results[key]:
                            logger.info(f"📊 Found data in results['{key}']")
                            if isinstance(results[key], pd.DataFrame):
                                all_trades = results[key]
                                break
                            elif isinstance(results[key], list) and len(results[key]) > 0:
                                all_trades = pd.DataFrame(results[key])
                                break
            
            # Method 3: Check for processed attribute
            if all_trades is None and hasattr(backtesting, 'processed'):
                logger.info("🔍 Checking backtesting.processed...")
                # This might contain the processed data
            
            if all_trades is None or len(all_trades) == 0:
                logger.error(f"❌ Could not find trades data in backtest results")
                logger.error(f"   Available attributes: {[a for a in dir(backtesting) if not a.startswith('_')]}")
                logger.error(f"   Results type: {type(backtesting.results) if hasattr(backtesting, 'results') else 'N/A'}")
                
                # 🚨 FALLBACK: Extract from stats instead
                logger.warning("⚠️  Attempting to extract from stats instead...")
                return self._extract_from_stats(backtesting, pairs)
            
            logger.info(f"📊 Processing {len(all_trades)} trades across pairs...")
            
            # Continue with per-pair processing...
            results = []
            
            for pair in pairs:
                pair_trades = all_trades[all_trades['pair'] == pair]
                
                if len(pair_trades) == 0:
                    logger.debug(f"⚠️  {pair}: No trades")
                    continue
                
                trade_count = len(pair_trades)
                
                if trade_count < 5 and not self.test_mode:
                    logger.debug(f"⚠️  {pair}: Only {trade_count} trades - SKIPPING")
                    continue
                
                # [Rest of the metrics calculation code - same as before]
                profit_total = pair_trades['profit_ratio'].sum() * 100
                profit_abs = pair_trades['profit_abs'].sum()
                avg_profit = pair_trades['profit_ratio'].mean() * 100
                median_profit = pair_trades['profit_ratio'].median() * 100
                
                wins = len(pair_trades[pair_trades['profit_ratio'] > 0])
                win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
                
                # ... rest of metrics calculation ...
                
                metrics = {
                    'pair': pair,
                    'trade_count': trade_count,
                    'profit_total': profit_total,
                    # ... etc
                }
                
                results.append(metrics)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch backtest failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _extract_from_stats(self, backtesting, pairs: List[str]) -> List[Dict[str, Any]]:
        """
        FALLBACK: Extract metrics from stats dict when trades DataFrame is unavailable
        """
        logger.info("📊 Extracting metrics from backtesting.results stats...")
        
        stats = getattr(backtesting, "results", None)
        if not stats:
            logger.error("❌ backtesting.results is None!")
            return []
        
        logger.info(f"🔍 Stats keys: {list(stats.keys())}")
        
        # Get strategy-level stats
        strategy_dict = stats.get("strategy", {})
        if not strategy_dict:
            logger.error("❌ No 'strategy' key in stats!")
            return []
        
        # Get the strategy name
        strategy_name = list(strategy_dict.keys())[0] if strategy_dict else None
        if not strategy_name:
            logger.error("❌ No strategy name found!")
            return []
        
        logger.info(f"✅ Found strategy: {strategy_name}")
        
        strategy_stats = strategy_dict[strategy_name]
        logger.info(f"🔍 Strategy stats keys: {list(strategy_stats.keys())[:20]}")
        
        # 🎯 THE FIX: Use 'results_per_pair' instead of 'pairlist'!
        results_per_pair = strategy_stats.get("results_per_pair", [])
        
        if not results_per_pair:
            logger.warning(f"⚠️  No 'results_per_pair' data!")
            return []
        
        logger.info(f"✅ Found results_per_pair with {len(results_per_pair)} entries")
        
        # 🔍 DEBUG: Check structure
        if results_per_pair:
            first_entry = results_per_pair[0]
            logger.info(f"🔍 First entry type: {type(first_entry)}")
            if isinstance(first_entry, dict):
                logger.info(f"   Keys: {list(first_entry.keys())}")
        
        # Convert to dict for easy lookup
        pairlist_dict = {}
        for entry in results_per_pair:
            if isinstance(entry, dict):
                # Try different possible key names
                pair_name = entry.get('key') or entry.get('pair') or entry.get('name')
                if pair_name:
                    pairlist_dict[pair_name] = entry
        
        logger.info(f"   Converted to dict with {len(pairlist_dict)} pairs")
        if pairlist_dict:
            logger.info(f"   Sample pairs: {list(pairlist_dict.keys())[:5]}")
        
        results = []
        for pair in pairs:
            pair_stats = pairlist_dict.get(pair)
            if not pair_stats:
                logger.debug(f"⚠️  {pair}: No stats found")
                continue
            
            trades = pair_stats.get('trades', 0)
            if trades < 5 and not self.test_mode:
                logger.debug(f"⚠️  {pair}: Only {trades} trades - SKIPPING")
                continue
            
            wins = pair_stats.get('wins', 0)
            win_rate = (wins / trades * 100) if trades > 0 else 0
            
            metrics = {
                'pair': pair,
                'trade_count': trades,
                'profit_total': pair_stats.get('profit_total', 0),
                'profit_abs': pair_stats.get('profit_total_abs', 0),
                'win_rate': win_rate,
                'avg_profit': pair_stats.get('profit_mean', 0),
                'median_profit': pair_stats.get('profit_median', 0),
                'max_drawdown': pair_stats.get('max_drawdown', 0),
                'max_drawdown_abs': pair_stats.get('max_drawdown_abs', 0),
                'sharpe_ratio': pair_stats.get('sharpe', 0),
                'sortino_ratio': pair_stats.get('sortino', 0),
                'calmar_ratio': pair_stats.get('calmar', 0),
                'expectancy': pair_stats.get('expectancy', 0),
                'avg_duration': pair_stats.get('duration_avg', 0),
                'max_win_streak': pair_stats.get('max_winning_streak', 0),
                'max_loss_streak': pair_stats.get('max_losing_streak', 0),
                'score': 0,
                'train_accuracy': None,
                'val_accuracy': None,
                'accuracy_gap': None,
            }
            
            # Calculate composite score
            metrics['score'] = (
                (metrics['profit_total'] / 100) * 0.40 +
                (metrics['win_rate'] / 100) * 0.25 +
                (-metrics['max_drawdown'] / 100) * 0.20 +
                (metrics['sharpe_ratio'] / 3) * 0.10 +
                (metrics['expectancy'] / 10) * 0.05
            )
            
            results.append(metrics)
            
            logger.info(
                f"✅ {pair:20s} | Trades={trades:3d} | Profit={metrics['profit_total']:6.2f}% | "
                f"WR={win_rate:5.1f}% | Sharpe={metrics['sharpe_ratio']:5.2f}"
            )
        
        logger.info(f"📊 Extracted {len(results)} pair metrics from stats")
        return results

    def evaluate_pair_full_backtest(self, pair: str):
        """
        Run COMPREHENSIVE backtest for a single pair with detailed metrics extraction.
        This is the --fullbacktest mode that provides complete trading simulation.
        """
        logger.info(f"🎯 [FULL-BT] Comprehensive backtest for {pair}")

        from freqtrade.optimize.backtesting import Backtesting
        from freqtrade.configuration import TimeRange

        # Clone config to avoid side effects
        bt_config = self.config.copy()

        # 🔒 REQUIRED for Backtesting engine
        bt_config['exchange']['pair_whitelist'] = [pair]
        bt_config['pairlists'] = [{
            "method": "StaticPairList",
            "pairs": [pair]
        }]

        # Optimize for single-pair analysis
        bt_config['max_open_trades'] = 3
        bt_config['stake_amount'] = 100
        bt_config['dry_run_wallet'] = 1000

        # Optional: restrict timerange if specified
        if 'timerange' not in bt_config:
            bt_config['timerange'] = None

        try:
            # Initialize backtesting engine
            backtesting = Backtesting(bt_config)
            backtesting.strategy = self.strategy

            # Run complete backtest
            backtesting.start()

            # ✅ Extract comprehensive stats
            stats = getattr(backtesting, "results", None)

            if not stats:
                logger.warning(f"⚠️  {pair}: Backtest produced no results object")
                return None

            # Try to get pair-specific stats first
            pair_stats = stats.get("pairlist", {}).get(pair)

            # Fallback: single-pair run → strategy stats
            if not pair_stats:
                strategy_stats = stats.get("strategy")
                if not strategy_stats:
                    logger.warning(f"⚠️  {pair}: No usable backtest stats")
                    return None

                logger.info(f"ℹ️  {pair}: Using strategy-level stats (single-pair backtest)")
                pair_stats = strategy_stats

            # ──────────────────────────────────────────────────
            # 📊 EXTRACT COMPREHENSIVE METRICS
            # ──────────────────────────────────────────────────
            trades = pair_stats.get('trades', 0)
            
            # Skip pairs with insufficient trades (unless test mode)
            if trades < 5:
                if self.test_mode:
                    logger.warning(f"⚠️  {pair}: Only {trades} trades – keeping in TEST MODE")
                else:
                    logger.warning(f"⚠️  {pair}: Only {trades} trades – SKIPPING")
                    return None

            # Core performance metrics
            profit_total = pair_stats.get('profit_total', 0)
            profit_abs = pair_stats.get('profit_total_abs', 0)
            win_rate = pair_stats.get('wins', 0) / trades * 100 if trades > 0 else 0
            
            # Risk metrics
            max_dd = pair_stats.get('max_drawdown', 0)
            max_dd_abs = pair_stats.get('max_drawdown_abs', 0)
            
            # Trade distribution
            avg_profit = pair_stats.get('profit_mean', 0)
            median_profit = pair_stats.get('profit_median', 0)
            
            # Duration metrics
            avg_duration = pair_stats.get('duration_avg', 0)
            
            # Advanced metrics
            sharpe_ratio = pair_stats.get('sharpe', 0)
            sortino_ratio = pair_stats.get('sortino', 0)
            calmar_ratio = pair_stats.get('calmar', 0)
            
            # Expectancy (average profit per trade)
            expectancy = pair_stats.get('expectancy', 0)
            
            # Winning/Losing streaks
            max_win_streak = pair_stats.get('max_winning_streak', 0)
            max_loss_streak = pair_stats.get('max_losing_streak', 0)

            # ──────────────────────────────────────────────────
            # 🎯 CALCULATE COMPOSITE SCORE
            # ──────────────────────────────────────────────────
            # Weighted scoring system for ranking pairs
            score = (
                (profit_total / 100) * 0.40 +      # 40% weight on total profit
                (win_rate / 100) * 0.25 +          # 25% weight on win rate
                (-max_dd / 100) * 0.20 +           # 20% weight on drawdown (inverted)
                (sharpe_ratio / 3) * 0.10 +        # 10% weight on Sharpe ratio
                (expectancy / 10) * 0.05           # 5% weight on expectancy
            )

            metrics = {
                'pair': pair,
                'trade_count': trades,
                
                # Performance metrics
                'profit_total': profit_total,
                'profit_abs': profit_abs,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'median_profit': median_profit,
                
                # Risk metrics
                'max_drawdown': max_dd,
                'max_drawdown_abs': max_dd_abs,
                
                # Advanced metrics
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'expectancy': expectancy,
                
                # Duration
                'avg_duration': avg_duration,
                
                # Streaks
                'max_win_streak': max_win_streak,
                'max_loss_streak': max_loss_streak,
                
                # Composite score for ranking
                'score': score,
                
                # ML-compatible fields (for unified handling)
                'train_accuracy': None,
                'val_accuracy': None,
                'accuracy_gap': None,
            }

            logger.info(f"✅ {pair}: Trades={trades} | Profit={profit_total:.2f}% | "
                       f"WR={win_rate:.1f}% | DD={max_dd:.2f}% | Sharpe={sharpe_ratio:.2f} | Score={score:.3f}")

            return metrics

        except Exception as e:
            logger.error(f"❌ {pair}: Full backtest failed - {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def evaluate_pair_real_backtest(self, pair: str):
        """Run REAL backtest for a single pair (non-ML strategies) - LEGACY MODE"""
        logger.info(f"📊 [BT] Real backtest for {pair}")

        from freqtrade.optimize.backtesting import Backtesting
        from freqtrade.configuration import TimeRange

        # Clone config to avoid side effects
        bt_config = self.config.copy()

        # 🔒 REQUIRED for Backtesting engine
        bt_config['exchange']['pair_whitelist'] = [pair]

        # Optional but safe
        bt_config['pairlists'] = [{
            "method": "StaticPairList",
            "pairs": [pair]
        }]

        # Limit scope for speed
        bt_config['max_open_trades'] = 1
        bt_config['stake_amount'] = 100
        bt_config['dry_run_wallet'] = 1000

        # Optional: restrict timerange
        if 'timerange' not in bt_config:
            bt_config['timerange'] = None

        # Run backtest
        backtesting = Backtesting(bt_config)
        backtesting.strategy = self.strategy

        # Run backtest (return value is unreliable)
        backtesting.start()

        # ✅ THIS is where stats actually are
        stats = getattr(backtesting, "results", None)

        if not stats:
            logger.warning(f"⚠️  {pair}: Backtest produced no results object")
            return None


        pair_stats = stats.get("pairlist", {}).get(pair)

        # Fallback: single-pair run → strategy stats
        if not pair_stats:
            strategy_stats = stats.get("strategy")
            if not strategy_stats:
                logger.warning(f"⚠️  {pair}: No usable backtest stats")
                return None

            logger.info(
                f"ℹ️  {pair}: Using strategy-level stats (single-pair backtest)"
            )
            pair_stats = strategy_stats


        trades = pair_stats.get('trades', 0)
        profit_total = pair_stats.get('profit_total', 0)
        profit_abs = pair_stats.get('profit_total_abs', 0)
        win_rate = pair_stats.get('winrate', 0)
        max_dd = pair_stats.get('max_drawdown', 0)

        if trades < 5:
            if self.test_mode:
                logger.warning(
                    f"⚠️  {pair}: Only {trades} trades – keeping in TEST MODE"
                )
            else:
                return None

        metrics = {
            'pair': pair,
            'trade_count': trades,
            'profit_total': profit_total,
            'win_rate': win_rate,
            'max_drawdown': max_dd,

            # 🔒 ML-compatible fields (required by ranking / saving)
            'train_accuracy': None,
            'val_accuracy': None,
            'accuracy_gap': None,

            # normalized score (used for ranking)
            'score': (profit_total / 100) - (max_dd / 100),
        }

        logger.info(
            f"✅ {pair}: "
            f"Trades={trades} | "
            f"Profit={profit_total:.2f}% | "
            f"WR={win_rate:.1f}% | "
            f"DD={max_dd:.1f}%"
        )

        return metrics

    def evaluate_pair_backtest(self, pair: str):
        """Evaluate pair by classic backtest metrics (non-ML strategies)"""

        logger.info(f"📈 [BT] Backtesting {pair}")

        dataframe = self.load_pair_data(pair)
        if dataframe is None or len(dataframe) < 500:
            return None

        dataframe = self.strategy.populate_indicators(dataframe, {'pair': pair})
        dataframe = self.strategy.populate_entry_trend(dataframe, {'pair': pair})
        dataframe = self.strategy.populate_exit_trend(dataframe, {'pair': pair})

        # Simple metrics (fast, no Trade DB)
        trades = dataframe[dataframe['enter_long'] == 1]
        trade_count = len(trades)

        if trade_count < 5:
            return None

        # Proxy metrics (no real wallet simulation)
        win_rate = (
            (trades['enter_long'] == 1).sum() / trade_count
        ) * 100

        metrics = {
            'pair': pair,
            'trade_count': trade_count,
            'win_rate': win_rate,
            'score': win_rate / 100.0,  # normalized
        }

        logger.info(
            f"✅ {pair}: Trades={trade_count} | WR={win_rate:.1f}%"
        )

        return metrics


    def _extract_ml_metrics(self, pair: str):
        """
        Extract ML metrics from strategy after backtest ML training
        """
        # Training data stored by your strategy
        samples = self.strategy.ml_training_data_per_pair.get(pair, [])
        model = self.strategy.ml_models.get(pair)

        if not model or len(samples) < 50:
            return None

        metrics = {
            'train_accuracy': getattr(model, 'train_accuracy', 50.0),
            'val_accuracy': getattr(model, 'val_accuracy', 50.0),
            'sample_count': len(samples),
        }

        metrics['accuracy_gap'] = (
            metrics['train_accuracy'] - metrics['val_accuracy']
        )

        return metrics

    def get_candidate_pairs(self) -> List[str]:
        """Get initial pool of pairs to evaluate"""
        pairs = []
        
        # Method 0: Generate dynamic pairlist if configured (NEW!)
        if self.config.get('pairlists') and len(self.config['pairlists']) > 0:
            # Check if pairlist is dynamic (not just StaticPairList)
            has_dynamic_pairlist = any(
                p.get('method') != 'StaticPairList' 
                for p in self.config['pairlists']
            )
            
            if has_dynamic_pairlist:
                logger.info("🔍 Config has dynamic pairlist, generating fresh pairlist...")
                
                # Generate and optionally save
                save_path = self.pairs_file if self.pairs_file else None
                pairs = self.generate_dynamic_pairlist(output_file=save_path)
                
                if pairs:
                    logger.info(f"✅ Generated {len(pairs)} pairs from dynamic pairlist\n")
                else:
                    logger.warning("⚠️  Dynamic pairlist generation failed, trying other methods...\n")
        
        # Method 1: Try to load from external pairs file (dynamic_pairs.json)
        if not pairs and self.pairs_file and Path(self.pairs_file).exists():
            logger.info(f"📊 Loading pairs from {self.pairs_file}...")
            try:
                with open(self.pairs_file, 'r') as f:
                    pairs_data = json.load(f)
                    # Handle different JSON formats
                    if isinstance(pairs_data, dict) and 'pairs' in pairs_data:
                        pairs = pairs_data['pairs']
                    elif isinstance(pairs_data, list):
                        pairs = pairs_data
                    
                    if pairs:
                        logger.info(f"✅ Loaded {len(pairs)} pairs from {self.pairs_file}")
            except Exception as e:
                logger.warning(f"⚠️  Could not load pairs file: {str(e)}")
        
        # Method 2: Try static pair_whitelist from config
        if not pairs:
            pairs = self.config.get('exchange', {}).get('pair_whitelist', [])
            if pairs:
                logger.info(f"📊 Using {len(pairs)} pairs from config pair_whitelist")
        
        # Method 3: Scan data directory for available pairs
        if not pairs:
            logger.info("⚠️  No pairs configured, scanning data directory...")
            pairs = self._scan_data_directory()
            if pairs:
                logger.info(f"✅ Found {len(pairs)} pairs with downloaded data")
        
        if not pairs:
            logger.error("❌ No pairs found by any method!")
            raise ValueError("No pairs configured or available")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 TOTAL CANDIDATE PAIRS: {len(pairs)}")
        logger.info(f"{'='*70}\n")
        
        # Download missing data if enabled
        if self.download_data:
            logger.info("🔍 Checking which pairs need data download...")
            pairs_needing_download = []
            
            for pair in pairs:
                data = self.load_pair_data(pair)
                if data is None or len(data) < 500:
                    pairs_needing_download.append(pair)
                    logger.debug(f"   📥 {pair} needs download ({len(data) if data is not None else 0} candles)")
            
            if pairs_needing_download:
                logger.info(f"\n📥 {len(pairs_needing_download)}/{len(pairs)} pairs need data download")
                self.download_missing_data(pairs_needing_download)
            else:
                logger.info("✅ All pairs already have sufficient data!\n")
        else:
            logger.info("⏭️  Auto-download disabled (--no-download flag)\n")
        
        # NOW filter pairs that have sufficient data (after download attempt)
        logger.info("🔍 Verifying data availability for each pair...")
        available_pairs = []
        
        for i, pair in enumerate(pairs, 1):
            data = self.load_pair_data(pair)
            if data is not None and len(data) >= 500:
                available_pairs.append(pair)
                if i % 20 == 0:  # Progress update every 20 pairs
                    logger.info(f"   Progress: {i}/{len(pairs)} pairs checked, {len(available_pairs)} valid")
            else:
                candle_count = len(data) if data is not None else 0
                logger.debug(f"   ⚠️  {pair}: Insufficient data ({candle_count} candles), skipping")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 PAIRS WITH SUFFICIENT DATA: {len(available_pairs)}/{len(pairs)}")
        logger.info(f"{'='*70}")
        
        if available_pairs:
            preview_count = min(10, len(available_pairs))
            logger.info(f"   Preview: {', '.join(available_pairs[:preview_count])}")
            if len(available_pairs) > preview_count:
                logger.info(f"   ... and {len(available_pairs) - preview_count} more\n")
        else:
            logger.error("\n❌ No pairs with sufficient data!")
            logger.error("💡 TROUBLESHOOTING:")
            logger.error("   1. Check if download actually ran (look for download logs above)")
            logger.error("   2. Verify exchange connection is working")
            logger.error("   3. Try manual download:")
            logger.error(f"      freqtrade download-data --config {self.config.get('config_files', ['config.json'])[0]} \\")
            logger.error(f"          --timerange 20231201-20260105 --timeframe 15m 1h --trading-mode futures")
            raise ValueError("No pairs have enough historical data for ML training")
        
        return available_pairs
    
    def _scan_data_directory(self) -> List[str]:
        """Scan data directory for available pair files"""
        data_dir = Path(self.config['datadir'])
        timeframe = self.strategy.timeframe
        
        # Determine candle type directory
        if self.config.get('trading_mode') == 'futures':
            candle_dir = data_dir / 'futures'
        else:
            candle_dir = data_dir / 'spot'
        
        if not candle_dir.exists():
            logger.warning(f"⚠️  Data directory not found: {candle_dir}")
            return []
        
        # Find all pair files for this timeframe
        pairs = []
        data_format = self.config.get('dataformat_ohlcv', 'feather')
        pattern = f"*-{timeframe}.{data_format}"
        
        for file in candle_dir.glob(pattern):
            # Extract pair name from filename
            # Format: BTC_USDT_USDT-15m.json -> BTC/USDT:USDT
            pair_str = file.stem.replace(f'-{timeframe}', '')
            pair_parts = pair_str.split('_')
            
            if len(pair_parts) >= 3:
                # Futures format: BTC_USDT_USDT -> BTC/USDT:USDT
                pair = f"{pair_parts[0]}/{pair_parts[1]}:{pair_parts[2]}"
            elif len(pair_parts) == 2:
                # Spot format: BTC_USDT -> BTC/USDT
                pair = f"{pair_parts[0]}/{pair_parts[1]}"
            else:
                continue
            
            pairs.append(pair)
        
        logger.info(f"   Found {len(pairs)} pair files in {candle_dir}")
        return pairs
    
    def load_pair_data(self, pair: str) -> pd.DataFrame:
        """Load historical data for a pair"""
        from freqtrade.configuration import TimeRange  # ✅ Import TimeRange
        
        timeframe = self.strategy.timeframe
        timerange_config = self.config.get('timerange', None)
        
        # ✅ Convert string timerange to TimeRange object if needed
        if timerange_config and isinstance(timerange_config, str):
            timerange = TimeRange.parse_timerange(timerange_config)
        else:
            timerange = timerange_config
        
        # Determine candle type
        if self.config.get('trading_mode') == 'futures':
            candle_type = 'futures'
        else:
            candle_type = 'spot'
        
        # Load data
        try:
            data = self.datahandler.ohlcv_load(
                pair=pair,
                timeframe=timeframe,
                timerange=timerange,  # ✅ Now passing TimeRange object!
                candle_type=candle_type
            )
            
            return data
        except Exception as e:
            logger.debug(f"   Failed to load {pair}: {str(e)}")
            return None

    def evaluate_pair(self, pair: str):
        """
        Main evaluation dispatcher - routes to appropriate evaluation method
        based on mode and strategy type
        """
        # Priority 1: Full backtest mode (if enabled)
        if self.fullbacktest_mode:
            return self.evaluate_pair_full_backtest(pair)
        
        # Priority 2: ML training mode (if strategy supports it)
        if self._strategy_has_ml():
            return self.evaluate_pair_ml(pair)
        
        # Priority 3: Standard backtest (fallback)
        return self.evaluate_pair_real_backtest(pair)


    def evaluate_pair_ml(self, pair: str):
        """Train ML model for one pair using backtest ML only"""
        logger.info(f"🔄 Processing {pair}...")
        self.disable_optuna_runtime()
        
        if not hasattr(self.strategy, 'train_ml_from_backtest_single_pair'):
            logger.error(
                f"❌ Strategy {self.strategy.__class__.__name__} "
                f"does not support backtest ML training"
            )
            return None

        setattr(self.strategy, 'RUNNING_PAIRLIST_OPTIMIZER', True)
        logger.info(f"🎯 [ML] Backtest training for {pair}")
        success = self.strategy.train_ml_from_backtest_single_pair(pair)

        if not success:
            logger.warning(f"⚠️  {pair}: ML training failed")
            return None

        # Read metrics FROM STRATEGY
        samples = self.strategy.ml_training_data_per_pair.get(pair, [])
        model = self.strategy.ml_models.get(pair)

        if not model or len(samples) < 50:
            logger.warning(f"⚠️  {pair}: ML model or samples missing")
            return None

        train_acc = getattr(model, 'train_accuracy', 50.0)
        val_acc = getattr(model, 'val_accuracy', 50.0)
        
        # ✅ NEW: Calculate win rate from training samples
        win_count = sum(1 for s in samples if s.get('profit', 0) > 0)
        loss_count = len(samples) - win_count
        win_rate = (win_count / len(samples)) * 100 if samples else 0.0

        metrics = {
            'pair': pair,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'accuracy_gap': train_acc - val_acc,
            'sample_count': len(samples),
            'win_rate': win_rate,  # ✅ NEW: Add win rate
            'win_count': win_count,  # ✅ NEW: Add win count
            'loss_count': loss_count,  # ✅ NEW: Add loss count
        }

        logger.info(
            f"✅ {pair}: "
            f"Train={train_acc:.1f}% | "
            f"Val={val_acc:.1f}% | "
            f"Gap={metrics['accuracy_gap']:.1f}% | "
            f"WR={win_rate:.1f}% | "  # ✅ NEW: Log win rate
            f"Samples={metrics['sample_count']}"
        )

        return metrics

    def select_best_pairs(self, n_pairs: int = 100) -> List[Dict[str, Any]]:
        """Evaluate all pairs and return top performers"""
        candidate_pairs = self.get_candidate_pairs()
        
        if not candidate_pairs:
            logger.error("❌ No candidate pairs with sufficient data!")
            return []
        
        results = []
        
        logger.info(f"\n{'='*70}")
        if self.fullbacktest_mode:
            if self.individual_backtest:
                logger.info(f"🚀 STARTING FULL BACKTEST EVALUATION (INDIVIDUAL MODE)")
            else:
                logger.info(f"🚀 STARTING FULL BACKTEST EVALUATION (BATCH MODE)")
        elif self._strategy_has_ml():
            logger.info(f"🚀 STARTING ML EVALUATION")
        else:
            logger.info(f"🚀 STARTING STANDARD BACKTEST EVALUATION")
        logger.info(f"{'='*70}")
        logger.info(f"Total pairs to evaluate: {len(candidate_pairs)}")
        logger.info(f"Target top pairs: {n_pairs}\n")
        
        # Determine effective limit
        limit = (
            len(candidate_pairs)
            if self.max_backtest_pairs is None
            else min(len(candidate_pairs), self.max_backtest_pairs)
        )

        # ──────────────────────────────────────────────────
        # 🎯 BATCH BACKTEST MODE (DEFAULT - FAST!)
        # ──────────────────────────────────────────────────
        if self.fullbacktest_mode and not self.individual_backtest:
            logger.info(f"🎯 Using BATCH BACKTEST mode - processing all {limit} pairs in ONE backtest")
            logger.info(f"   ⚡ This is MUCH faster than individual backtests! (3-5x speedup)")
            logger.info(f"   💡 Use --individual flag to backtest each pair separately\n")
            
            # Run batch backtest on all pairs at once
            results = self.evaluate_all_pairs_batch_backtest(candidate_pairs[:limit])
            
            if not results:
                logger.error("❌ Batch backtest produced no results")
                return []
            
            logger.info(f"✅ Batch backtest complete: {len(results)} pairs evaluated")
        
        # ──────────────────────────────────────────────────
        # 🔬 INDIVIDUAL BACKTEST MODE (DETAILED!)
        # ──────────────────────────────────────────────────
        elif self.fullbacktest_mode and self.individual_backtest:
            logger.info(f"🔬 Using INDIVIDUAL BACKTEST mode - backtesting each pair separately")
            logger.info(f"   ⏱️  This is slower but provides more isolated metrics")
            logger.info(f"   💡 Remove --individual flag for faster batch processing\n")
            
            for i, pair in enumerate(candidate_pairs[:limit], 1):
                logger.info(f"\n{'='*70}")
                logger.info(f"Progress: {i}/{limit} ({i / limit * 100:.1f}%)")
                logger.info(f"{'='*70}")

                metrics = self.evaluate_pair_full_backtest(pair)
                if metrics:
                    results.append(metrics)
                    logger.info(f"✅ Successfully evaluated: {len(results)} pairs so far")
            
            logger.info(f"✅ Individual backtest complete: {len(results)} pairs evaluated")
        
        # ──────────────────────────────────────────────────
        # OTHER EVALUATION MODES (ML / Standard)
        # ──────────────────────────────────────────────────
        else:
            for i, pair in enumerate(candidate_pairs[:limit], 1):
                logger.info(f"\n{'='*70}")
                logger.info(f"Progress: {i}/{limit} ({i / limit * 100:.1f}%)")
                logger.info(f"{'='*70}")

                metrics = self.evaluate_pair(pair)
                if metrics:
                    results.append(metrics)
                    logger.info(f"✅ Successfully evaluated: {len(results)} pairs so far")

            logger.info(f"🧪 Raw evaluated pairs = {len(results)}")
            if results:
                logger.info(f"🧪 Example result keys = {list(results[0].keys())}")

        # ──────────────────────────────────────────────────
        # RANK & RETURN RESULTS
        # ──────────────────────────────────────────────────
        if results:
            results = self.rank_pairs(results)
            logger.info(f"\n{'='*70}")
            logger.info(f"✅ EVALUATION COMPLETE")
            logger.info(f"{'='*70}")
            logger.info(f"Successfully evaluated: {len(results)}/{len(candidate_pairs)} pairs\n")
            return results[:n_pairs]

        else:
            if self.test_mode:
                logger.warning(
                    "⚠️ TEST MODE: No pairs passed ranking filters – ranking without filtering"
                )

                # ✅ Rank manually by score (NO FILTERING)
                ranked = sorted(
                    results,
                    key=lambda r: r.get('score', 0),
                    reverse=True
                )

                return ranked[:n_pairs]

            else:
                logger.error("\n❌ No pairs successfully evaluated")
                return []


    
    def rank_pairs(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank pairs based on evaluation mode:
        - Fullbacktest: comprehensive trading metrics
        - ML: hybrid score (win rate + validation accuracy + gap)
        - Standard: win rate
        """
        if not results:
            return []

        # ────────────────────────────────────────────────────────────
        # FULLBACKTEST MODE
        # ────────────────────────────────────────────────────────────
        if self.fullbacktest_mode and 'sharpe_ratio' in results[0]:
            logger.info("📊 Ranking pairs using FULLBACKTEST metrics")
            for r in results:
                if 'score' not in r:
                    profit_total = r.get('profit_total', 0)
                    win_rate = r.get('win_rate', 0)
                    max_dd = r.get('max_drawdown', 0)
                    sharpe = r.get('sharpe_ratio', 0)
                    expectancy = r.get('expectancy', 0)
                    
                    r['score'] = (
                        (profit_total / 100) * 0.40 +
                        (win_rate / 100) * 0.25 +
                        (-max_dd / 100) * 0.20 +
                        (sharpe / 3) * 0.10 +
                        (expectancy / 10) * 0.05
                    )
            return sorted(results, key=lambda x: x['score'], reverse=True)

        # ────────────────────────────────────────────────────────────
        # NON-ML STRATEGY
        # ────────────────────────────────────────────────────────────
        if 'accuracy_gap' not in results[0]:
            logger.info("📊 Ranking pairs using NON-ML metrics")
            for r in results:
                r.setdefault('score', r.get('win_rate', 0) / 100.0)
            return sorted(results, key=lambda x: x['score'], reverse=True)

        # ────────────────────────────────────────────────────────────
        # ✅ ML STRATEGY - HYBRID SCORING
        # ────────────────────────────────────────────────────────────
        logger.info("📊 Ranking pairs using HYBRID ML metrics (Win Rate + Val Accuracy + Gap)")

        for r in results:
            val_acc = r.get('val_accuracy', 0)
            win_rate = r.get('win_rate', 0)  # ✅ NEW: Use win rate
            acc_gap = r.get('accuracy_gap', 0)
            
            # Normalize components to 0-1 range
            val_acc_score = val_acc / 100.0
            win_rate_score = win_rate / 100.0
            gap_score = max(0.0, 1.0 - (acc_gap / 20.0))
            
            # ✅ HYBRID SCORE: 40% Win Rate + 40% Val Accuracy + 20% Gap
            r['score'] = (
                win_rate_score * 0.40 +      # Strategy baseline performance
                val_acc_score * 0.40 +       # ML predictive power
                gap_score * 0.20             # Model stability
            )

        return sorted(results, key=lambda x: x['score'], reverse=True)

        # ────────────────────────
        # NON-ML STRATEGY
        # ────────────────────────
        if 'accuracy_gap' not in results[0]:
            logger.info("📊 Ranking pairs using NON-ML metrics")

            # Ensure score exists
            for r in results:
                r.setdefault('score', r.get('win_rate', 0) / 100.0)

            return sorted(results, key=lambda x: x['score'], reverse=True)

        # ────────────────────────
        # ML STRATEGY
        # ────────────────────────
        logger.info("📊 Ranking pairs using ML metrics")

        for r in results:
            # ──────────────────────────────────
            # SAFE ML METRICS HANDLING
            # ──────────────────────────────────
            val_acc = r.get('val_accuracy')
            acc_gap = r.get('accuracy_gap')

            # Accuracy score
            if val_acc is None:
                acc_score = 0.0
            else:
                acc_score = val_acc / 100.0

            # Gap score
            if acc_gap is None:
                gap_score = 0.0
            else:
                gap_score = max(0.0, 1.0 - (acc_gap / 20.0))

            # Sample size bonus (optional)
            sample_score = 1.0 if r.get('sample_count', 0) >= 500 else 0.5

            # Final score (ML-aware, non-ML safe)
            r['score'] = (
                acc_score * 0.6 +
                gap_score * 0.2 +
                sample_score * 0.2
            )

        return sorted(results, key=lambda x: x['score'], reverse=True)

    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save results to JSON file (supports all evaluation modes)"""

        if not results:
            logger.warning("⚠️ No results to save")
            return

        # Detect evaluation mode from results
        is_fullbacktest = self.fullbacktest_mode and 'sharpe_ratio' in results[0]
        is_ml = results[0].get('val_accuracy') is not None

        output = {
            'generated_at': datetime.now().isoformat(),
            'strategy': self.strategy.__class__.__name__,
            'timeframe': self.strategy.timeframe,
            'evaluation_mode': 'fullbacktest' if is_fullbacktest else ('ml' if is_ml else 'standard'),
            'total_evaluated': len(results),
            'pairs': [r['pair'] for r in results],
            'detailed_metrics': results,
        }

        # ─────────────────────────────────────────────────
        # FULLBACKTEST MODE SUMMARY
        # ─────────────────────────────────────────────────
        if is_fullbacktest:
            output['summary'] = {
                'avg_profit_total': np.mean([r['profit_total'] for r in results]),
                'avg_win_rate': np.mean([r['win_rate'] for r in results]),
                'avg_max_drawdown': np.mean([r['max_drawdown'] for r in results]),
                'avg_sharpe_ratio': np.mean([r['sharpe_ratio'] for r in results]),
                'avg_expectancy': np.mean([r['expectancy'] for r in results]),
                'best_pair': results[0]['pair'],
                'best_profit': results[0]['profit_total'],
                'best_sharpe': results[0]['sharpe_ratio'],
                'best_score': results[0]['score'],
            }

        # ─────────────────────────────────────────────────
        # ML STRATEGY SUMMARY
        # ─────────────────────────────────────────────────
        elif is_ml:
            output['summary'] = {
                'avg_val_accuracy': np.mean([r['val_accuracy'] for r in results]),
                'avg_train_accuracy': np.mean([r['train_accuracy'] for r in results]),
                'avg_gap': np.mean([r['accuracy_gap'] for r in results]),
                'avg_win_rate': np.mean([r['win_rate'] for r in results]),  # ✅ NEW
                'best_pair': results[0]['pair'],
                'best_val_accuracy': results[0]['val_accuracy'],
                'best_win_rate': results[0]['win_rate'],  # ✅ NEW
                'best_score': results[0]['score'],
            }

        # ─────────────────────────────────────────────────
        # STANDARD BACKTEST SUMMARY
        # ─────────────────────────────────────────────────
        else:
            output['summary'] = {
                'avg_win_rate': np.mean([r['win_rate'] for r in results]),
                'best_pair': results[0]['pair'],
                'best_win_rate': results[0]['win_rate'],
                'best_score': results[0]['score'],
            }

        # Save JSON
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        # ─────────────────────────────────────────────────
        # LOGGING (COMMON HEADER)
        # ─────────────────────────────────────────────────
        logger.info(f"\n{'='*70}")
        logger.info("📊 FINAL RESULTS SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Evaluation Mode: {output['evaluation_mode'].upper()}")
        logger.info(f"Total pairs evaluated: {len(results)}")

        # ─────────────────────────────────────────────────
        # FULLBACKTEST LOGGING
        # ─────────────────────────────────────────────────
        if is_fullbacktest:
            logger.info(f"Average total profit: {output['summary']['avg_profit_total']:.2f}%")
            logger.info(f"Average win rate: {output['summary']['avg_win_rate']:.1f}%")
            logger.info(f"Average max drawdown: {output['summary']['avg_max_drawdown']:.2f}%")
            logger.info(f"Average Sharpe ratio: {output['summary']['avg_sharpe_ratio']:.2f}")
            logger.info(
                f"Best pair: {output['summary']['best_pair']} "
                f"(Profit: {output['summary']['best_profit']:.2f}%, "
                f"Sharpe: {output['summary']['best_sharpe']:.2f})"
            )

        # ────────────────────────────────────────────────────────────
        # ✅ ML LOGGING (WITH WIN RATE)
        # ────────────────────────────────────────────────────────────
        elif is_ml:
            logger.info(
                f"Average validation accuracy: "
                f"{output['summary']['avg_val_accuracy']:.1f}%"
            )
            logger.info(
                f"Average win rate: "
                f"{output['summary']['avg_win_rate']:.1f}%"  # ✅ NEW
            )
            logger.info(
                f"Average train/val gap: "
                f"{output['summary']['avg_gap']:.1f}%"
            )
            logger.info(
                f"Best pair: {output['summary']['best_pair']} "
                f"(Val={output['summary']['best_val_accuracy']:.1f}%, "
                f"WR={output['summary']['best_win_rate']:.1f}%)"  # ✅ NEW
            )

        # ─────────────────────────────────────────────────
        # STANDARD LOGGING
        # ─────────────────────────────────────────────────
        else:
            logger.info(
                f"Average win rate: "
                f"{output['summary']['avg_win_rate']:.1f}%"
            )
            logger.info(
                f"Best pair: {output['summary']['best_pair']} "
                f"({output['summary']['best_win_rate']:.1f}%)"
            )

        logger.info(f"\n✅ Results saved to: {output_path}")

        # ─────────────────────────────────────────────────
        # TOP 20 TABLE
        # ─────────────────────────────────────────────────
        logger.info(f"\n{'='*70}")
        
        if is_fullbacktest:
            logger.info("🏆 TOP 20 PAIRS BY COMPREHENSIVE BACKTEST SCORE")
        elif is_ml:
            logger.info("🏆 TOP 20 PAIRS BY ML VALIDATION ACCURACY")
        else:
            logger.info("🏆 TOP 20 PAIRS BY SIGNAL WIN RATE")
            
        logger.info(f"{'='*70}")

        top_count = min(20, len(results))
        for i, r in enumerate(results[:top_count], 1):
            if is_fullbacktest:
                logger.info(
                    f"  {i:2d}. {r['pair']:20s} "
                    f"Profit={r['profit_total']:6.2f}%  "
                    f"WR={r['win_rate']:5.1f}%  "
                    f"DD={r['max_drawdown']:5.2f}%  "
                    f"Sharpe={r['sharpe_ratio']:5.2f}  "
                    f"Score={r['score']:.3f}"
                )
            elif is_ml:
                logger.info(
                    f"  {i:2d}. {r['pair']:20s} "
                    f"Val={r['val_accuracy']:5.1f}%  "
                    f"WR={r['win_rate']:5.1f}%  "  # ✅ NEW: Show win rate
                    f"Gap={r['accuracy_gap']:5.1f}%  "
                    f"Score={r['score']:.3f}"
                )
            else:
                logger.info(
                    f"  {i:2d}. {r['pair']:20s} "
                    f"WR={r['win_rate']:5.1f}%  "
                    f"Trades={r.get('trade_count', 0):4d}  "
                    f"Score={r['score']:.3f}"
                )

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ML Pairlist Optimizer - Fully automated: Generate pairlist → Download data → Evaluate → Rank',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ML Training mode (default):
  python ml_pairlist_selector.py --config config-pairlist.json --strategy AlexBandSniperV11MLAI --n-pairs 50
  
  # Full Backtest mode (comprehensive metrics):
  python ml_pairlist_selector.py --config config-pairlist.json --strategy MyStrategy --fullbacktest --n-pairs 50
  
  # Analyze existing backtest JSON (NO RE-RUN - FAST!):
  python ml_pairlist_selector.py --config config-pairlist.json --strategy MyStrategy --from-json --n-pairs 50
  python ml_pairlist_selector.py --config config-pairlist.json --strategy MyStrategy --from-json user_data/backtest_results/backtest-result-2025-01-09.json
  
  # Custom download period:
  python ml_pairlist_selector.py --config config-pairlist.json --strategy AlexBandSniperV11MLAI --download-days 90
  
  # Skip download (use existing data):
  python ml_pairlist_selector.py --config config-pairlist.json --strategy AlexBandSniperV11MLAI --no-download
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to Freqtrade config file with pairlist configuration')
    parser.add_argument('--strategy', type=str, required=True,
                       help='Strategy name (e.g., AlexBandSniperV11MLAI)')
    parser.add_argument('--pairs-file', type=str, default='user_data/pairlist/dynamic_pairs.json',
                       help='Path to save/load pairs JSON file (default: user_data/pairlist/dynamic_pairs.json)')
    parser.add_argument('--n-pairs', type=int, default=50,
                       help='Number of top pairs to select (default: 50)')

    parser.add_argument('--output', type=str, default='user_data/pairlist/optimal_pairs.json',
                       help='Output JSON path for optimal pairs (default: user_data/pairlist/optimal_pairs.json)')
    
    # ✅ NEW: JSON analysis mode
    parser.add_argument('--from-json', type=str, nargs='?', const='auto', default=None,
                       help='Analyze existing backtest JSON without re-running. '
                            'Auto-detects most recent file if no path given. '
                            'Example: --from-json (auto) or --from-json path/to/backtest.json')
    
    parser.add_argument('--no-download', action='store_true',
                       help='Skip automatic data download (use existing data only)')
    parser.add_argument('--download-days', type=int, default=60,
                       help='Days of historical data to download (default: 60)')
    parser.add_argument('--backtest-days', type=int, default=None,
                       help='Days to use for backtesting (default: use all available data). '
                            'Example: --backtest-days 30 will backtest only the last 30 days. '
                            'Useful to speed up backtests or test recent performance.')
    parser.add_argument('--test-mode', action='store_true',
                       help='Enable test mode (do not filter losing pairs, for research/debugging)')
    parser.add_argument('--max-backtest-pairs', type=int, default=500,
                       help='Maximum number of pairs to backtest (performance control)')
    parser.add_argument('--fullbacktest', action='store_true',
                       help='Enable FULL BACKTEST mode: Run comprehensive backtests with detailed metrics '
                            '(Sharpe, Sortino, Calmar, Expectancy, etc.) instead of ML training. '
                            'By default uses BATCH mode (all pairs in one backtest - FAST). '
                            'Ranks pairs by composite trading performance score.')
    parser.add_argument('--individual', action='store_true',
                       help='When used with --fullbacktest, backtest each pair INDIVIDUALLY instead of batch. '
                            'Slower but more isolated metrics. Batch mode is default (3-5x faster).')

    args = parser.parse_args()
    
    # ══════════════════════════════════════════════════════════════════════
    # ✅ EARLY EXIT: JSON ANALYSIS MODE (NO BACKTEST RE-RUN)
    # ══════════════════════════════════════════════════════════════════════
    if args.from_json:
        logger.info("\n" + "="*70)
        logger.info("📊 JSON ANALYSIS MODE")
        logger.info("="*70)
        logger.info("Mode: Analyze existing backtest JSON (no re-run)")
        logger.info("="*70)
        logger.info(f"Config: {args.config}")
        logger.info(f"Strategy: {args.strategy}")
        
        if args.from_json == 'auto':
            logger.info(f"JSON file: Auto-detect (most recent)")
        else:
            logger.info(f"JSON file: {args.from_json}")
        
        logger.info(f"Target pairs: {args.n_pairs}")
        logger.info(f"Output: {args.output}")
        logger.info(f"Test mode: {'ENABLED' if args.test_mode else 'DISABLED'}")
        logger.info("="*70 + "\n")
        
        try:
            # Initialize optimizer (minimal setup, no download)
            optimizer = MLPairlistOptimizer(
                config_path=args.config,
                strategy_name=args.strategy,
                pairs_file=None,
                download_data=False,
                download_days=0,
                fullbacktest=False,
                individual_backtest=False,
                backtest_days=None
            )
            
            optimizer.test_mode = args.test_mode
            optimizer.fullbacktest_mode = True  # Treat as fullbacktest for ranking
            
            # Analyze JSON file
            json_path = None if args.from_json == 'auto' else args.from_json
            results = optimizer.analyze_existing_backtest(json_path)
            
            if results:
                # Rank pairs
                results = optimizer.rank_pairs(results)
                
                # Take top N
                results = results[:args.n_pairs]
                
                # Save results
                output_file = args.output.replace('.json', f'_{args.strategy}.json') if args.output == 'user_data/pairlist/optimal_pairs.json' else args.output
                optimizer.save_results(results, output_file)
                
                logger.info("\n" + "="*70)
                logger.info("✅ JSON ANALYSIS COMPLETE!")
                logger.info("="*70)
                logger.info(f"📁 Output saved to: {args.output}")
                logger.info(f"🎯 Top {len(results)} pairs extracted and ranked")
                
                best = results[0]
                logger.info(
                    f"⭐ Best pair: {best['pair']} "
                    f"(Profit: {best['profit_total']:.2f}%, "
                    f"Sharpe: {best['sharpe_ratio']:.2f}, "
                    f"WR: {best['win_rate']:.1f}%, "
                    f"Score: {best['score']:.3f})"
                )
                logger.info("="*70 + "\n")
                
                sys.exit(0)
            else:
                logger.error("\n" + "="*70)
                logger.error("❌ JSON ANALYSIS FAILED")
                logger.error("="*70)
                logger.error("No results extracted from JSON file")
                logger.error("💡 Make sure the JSON file contains valid backtest results")
                logger.error("="*70 + "\n")
                sys.exit(1)
                
        except Exception as e:
            logger.error("\n" + "="*70)
            logger.error("❌ JSON ANALYSIS ERROR")
            logger.error("="*70)
            logger.error(f"Error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            logger.error("="*70 + "\n")
            sys.exit(1)
    
    # ══════════════════════════════════════════════════════════════════════
    # NORMAL EXECUTION PATH (FULL WORKFLOW)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("🚀 ML PAIRLIST OPTIMIZER")
    logger.info("="*70)
    
    if args.fullbacktest:
        if args.individual:
            logger.info("Mode: FULL BACKTEST - INDIVIDUAL (Each pair separately)")
            logger.info("Workflow:")
            logger.info("  1. Generate dynamic pairlist from exchange")
            logger.info("  2. Download historical data for all pairs")
            logger.info("  3. Run individual comprehensive backtest on EACH pair")
            logger.info("  4. Extract comprehensive metrics for each pair")
            logger.info("  5. Rank pairs by composite performance score")
            logger.info("  6. Output top performers")
            logger.info("Note: This is slower but provides isolated per-pair analysis")
        else:
            logger.info("Mode: FULL BACKTEST - BATCH (All pairs at once - RECOMMENDED)")
            logger.info("Workflow:")
            logger.info("  1. Generate dynamic pairlist from exchange")
            logger.info("  2. Download historical data for all pairs")
            logger.info("  3. Run ONE comprehensive backtest with ALL pairs")
            logger.info("  4. Extract per-pair metrics from batch results")
            logger.info("  5. Rank pairs by composite performance score")
            logger.info("  6. Output top performers")
            logger.info("Note: This is 3-5x faster than individual mode!")
    else:
        logger.info("Mode: ML TRAINING / STANDARD BACKTEST")
        logger.info("Workflow:")
        logger.info("  1. Generate dynamic pairlist from exchange")
        logger.info("  2. Download historical data for all pairs")
        logger.info("  3. Train ML models / Run backtests on each pair")
        logger.info("  4. Rank pairs by validation accuracy / win rate")
        logger.info("  5. Output top performers")
    
    logger.info("="*70)
    logger.info(f"Config: {args.config}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Pairs file: {args.pairs_file}")
    logger.info(f"Target pairs: {args.n_pairs}")
    logger.info(f"Auto-download: {'Disabled' if args.no_download else 'Enabled'}")
    if not args.no_download:
        logger.info(f"Download days: {args.download_days}")
    if args.backtest_days:
        logger.info(f"Backtest period: Last {args.backtest_days} days")
    else:
        logger.info(f"Backtest period: All available data")
    logger.info(f"Max backtest pairs: {args.max_backtest_pairs}")
    logger.info("="*70 + "\n")
    
    try:
        optimizer = MLPairlistOptimizer(
            config_path=args.config,
            strategy_name=args.strategy,
            pairs_file=args.pairs_file,
            download_data=not args.no_download,
            download_days=args.download_days,
            fullbacktest=args.fullbacktest,
            individual_backtest=args.individual,
            backtest_days=args.backtest_days
        )
        optimizer.max_backtest_pairs = args.max_backtest_pairs
        optimizer.test_mode = args.test_mode
        
        logger.info(f"🧪 Test mode: {'ENABLED' if optimizer.test_mode else 'DISABLED'}")
        if args.fullbacktest:
            mode_str = "INDIVIDUAL BACKTEST" if args.individual else "BATCH BACKTEST (FAST)"
            logger.info(f"🎯 Evaluation mode: FULL BACKTEST - {mode_str}")
        else:
            logger.info(f"🎯 Evaluation mode: {'ML TRAINING' if optimizer._strategy_has_ml() else 'STANDARD'}\n")
        
        results = optimizer.select_best_pairs(n_pairs=args.n_pairs)
        
        if results:
            optimizer.append_results_history(results)
            output_file = args.output.replace('optimal_pairs.json', f'optimal_pairs_{args.strategy}.json')
            optimizer.save_results(results, output_file)

            logger.info("\n" + "="*70)
            logger.info("✅ OPTIMIZATION COMPLETE!")
            logger.info("="*70)
            logger.info(f"📁 Output saved to: {args.output}")
            logger.info(f"🎯 Top {len(results)} pairs ready for trading!")

            best = results[0]

            if args.fullbacktest:
                logger.info(
                    f"⭐ Best pair: {best['pair']} "
                    f"(Profit: {best['profit_total']:.2f}%, "
                    f"Sharpe: {best['sharpe_ratio']:.2f}, "
                    f"Score: {best['score']:.3f})"
                )
            elif best.get('val_accuracy') is not None:
                logger.info(
                    f"⭐ Best pair: {best['pair']} "
                    f"(Val Acc: {best['val_accuracy']:.1f}%)"
                )
            else:
                logger.info(
                    f"⭐ Best pair: {best['pair']} "
                    f"(Win Rate: {best.get('win_rate', 0.0):.1f}%)"
                )

            logger.info("="*70 + "\n")
        else:
            logger.error("\n" + "="*70)
            logger.error("❌ OPTIMIZATION FAILED")
            logger.error("="*70)
            logger.error("No valid results - check logs above for errors")
            sys.exit(1)
            
    except Exception as e:
        logger.error("\n" + "="*70)
        logger.error("❌ FATAL ERROR")
        logger.error("="*70)
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()