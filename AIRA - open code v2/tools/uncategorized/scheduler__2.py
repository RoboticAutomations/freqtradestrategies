"""
Finance data collector scheduler for MultibotdashboardV7
Runs all collectors on scheduled intervals
"""

import asyncio
from datetime import datetime
import logging

from .crypto_collector import crypto_collector
from .stock_collector import stock_collector
from .news_collector import news_collector
from .economic_collector import economic_collector
from .bybit_collector import bybit_collector
from .binance_collector import binance_collector

logger = logging.getLogger(__name__)

class FinanceScheduler:
    """Schedules and runs all finance data collectors."""
    
    def __init__(self):
        self.running = False
        self.tasks = []
        # Collection intervals (in seconds)
        self.intervals = {
            'crypto': 300,      # Every minute
            'stocks': 3600,     # Every 5 minutes
            'news': 600,       # Every 10 minutes
            'economic': 3600,  # Every hour
            'bybit': 30,       # Every 30 seconds
            'binance': 30,     # Every 30 seconds
        }
    
    async def start(self):
        """Start all collector schedules."""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting Finance Data Collector Scheduler...")
        
        # Create tasks for each collector
        self.tasks = [
            asyncio.create_task(self._run_collector('crypto', crypto_collector.run, self.intervals['crypto'])),
            asyncio.create_task(self._run_collector('stocks', stock_collector.run, self.intervals['stocks'])),
            asyncio.create_task(self._run_collector('news', news_collector.run, self.intervals['news'])),
            asyncio.create_task(self._run_collector('economic', economic_collector.run, self.intervals['economic'])),
            asyncio.create_task(self._run_collector('bybit', bybit_collector.run, self.intervals['bybit'])),
            asyncio.create_task(self._run_collector('binance', binance_collector.run, self.intervals['binance'])),
        ]
        
        logger.info(f"Started {len(self.tasks)} collector tasks")
    
    async def stop(self):
        """Stop all collectors."""
        self.running = False
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks = []
        logger.info("Finance Data Collector Scheduler stopped")
    
    async def _run_collector(self, name: str, collector_func, interval: int):
        """Run a collector at specified interval."""
        logger.info(f"Starting {name} collector (interval: {interval}s)")
        
        while self.running:
            try:
                await collector_func()
            except Exception as e:
                logger.error(f"Error in {name} collector: {e}")
            
            # Wait for next interval
            await asyncio.sleep(interval)
    
    async def run_once(self):
        """Run all collectors once (for manual trigger)."""
        logger.info("Running all collectors once...")
        await asyncio.gather(
            crypto_collector.run(),
            stock_collector.run(),
            news_collector.run(),
            economic_collector.run(),
            bybit_collector.run(),
            binance_collector.run(),
            return_exceptions=True
        )

# Singleton instance
finance_scheduler = FinanceScheduler()
