"""
Alex_AgentStrategy.py - Multibotdashboard V8 Dynamic Weight Trading Agent

Integrates with AlexFinanceData/Multibotdashboard V7 API
Uses dynamic weights based on market regime (Bull/Bear/Ranging/HighVol)
Learns from trade outcomes and adjusts weights automatically

Author: Alex Kasuari
Version: 1.0.0 (V8)
"""

import datetime
import json
import logging
import requests
from typing import Dict, Optional

import numpy as np
import pandas as pd
import talib
from freqtrade.persistence import Trade
from freqtrade.strategy import (CategoricalParameter, DecimalParameter, IStrategy, IntParameter)
from pandas import DataFrame

logger = logging.getLogger(__name__)

# Dashboard API configuration
DASHBOARD_API = "http://your-dashboard-host:8000/api/v1"  # Adjust to your API
API_TIMEOUT = 5

# Regime detection settings
SMA_FAST = 50
SMA_SLOW = 200
ATR_PERIOD = 14
VOLATILITY_THRESHOLD = 0.03  # 3% ATR as high vol threshold

# Weight learning settings
MIN_TRADES_BEFORE_ADJUST = 20
MAX_WEIGHT_PER_SIGNAL = 0.50
LEARNING_RATE = 0.05
MIN_CONFIDENCE = 75


class Alex_AgentStrategy(IStrategy):
    """
    Dynamic Weight Agent Strategy
    
    Fetches signals from Multibotdashboard V7 API
    Weights signals based on current market regime
    Adjusts weights based on trade performance
    """
    
    # Strategy config
    timeframe = "15m"
    startup_candle_count = 300
    can_short = True
    
    # Risk management
    minimal_roi = {"0": 0.10}
    stoploss = -0.05
    use_custom_stoploss = True
    
    # Position sizing
    position_adjustment_enable = False
    max_open_trades = 3
    
    # Strategy parameters (for hyperopt if needed)
    confidence_threshold = IntParameter(60, 90, default=MIN_CONFIDENCE, space="buy")
    signal_timeout = IntParameter(10, 60, default=30, space="buy")
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.current_regime = "ranging"
        self.weights = None
        self.last_weights_update = None
        self.signals_cache = {}
        self.cache_timestamp = None
        
        logger.info("🤖 Alex_AgentStrategy initialized")
        logger.info(f"   Dashboard API: {DASHBOARD_API}")
        logger.info(f"   Min confidence: {MIN_CONFIDENCE}")
        logger.info(f"   Learning rate: {LEARNING_RATE}")
    
    def fetch_dashboard_data(self, endpoint: str) -> Optional[Dict]:
        """Fetch data from Multibotdashboard V7 API"""
        try:
            url = f"{DASHBOARD_API}{endpoint}"
            response = requests.get(url, timeout=API_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                return data.get('data', data)  # Handle wrapped/unwrapped
            else:
                logger.warning(f"API error {response.status_code}: {url}")
                return None
        except Exception as e:
            logger.warning(f"API fetch failed: {e}")
            return None
    
    def fetch_weights(self, regime: str) -> Dict:
        """Fetch current weights for market regime from dashboard"""
        try:
            # Try to fetch from API if available
            data = self.fetch_dashboard_data(f'/agent/weights?regime={regime}')
            if data:
                return {
                    'price': data.get('price_momentum_weight', 0.25),
                    'volume': data.get('volume_weight', 0.20),
                    'sentiment': data.get('sentiment_weight', 0.20),
                    'macro': data.get('macro_weight', 0.20),
                    'orderbook': data.get('orderbook_weight', 0.15)
                }
        except Exception as e:
            logger.warning(f"Failed to fetch weights: {e}")
        
        # Fallback defaults based on regime
        defaults = {
            'bull': {'price': 0.30, 'volume': 0.25, 'sentiment': 0.20, 'macro': 0.15, 'orderbook': 0.10},
            'bear': {'price': 0.35, 'volume': 0.20, 'sentiment': 0.15, 'macro': 0.15, 'orderbook': 0.15},
            'ranging': {'price': 0.25, 'volume': 0.30, 'sentiment': 0.20, 'macro': 0.15, 'orderbook': 0.10},
            'high_vol': {'price': 0.20, 'volume': 0.35, 'sentiment': 0.15, 'macro': 0.20, 'orderbook': 0.10}
        }
        return defaults.get(regime, defaults['ranging'])
    
    def detect_regime(self, dataframe: DataFrame) -> str:
        """Detect current market regime based on BTC trend and volatility"""
        try:
            # Calculate SMAs
            sma50 = dataframe['close'].rolling(window=SMA_FAST).mean().iloc[-1]
            sma200 = dataframe['close'].rolling(window=SMA_SLOW).mean().iloc[-1]
            
            # Calculate ATR for volatility
            atr = talib.ATR(dataframe['high'], dataframe['low'], 
                          dataframe['close'], timeperiod=ATR_PERIOD).iloc[-1]
            atr_pct = atr / dataframe['close'].iloc[-1]
            
            # Determine regime
            if atr_pct > VOLATILITY_THRESHOLD:
                regime = "high_vol"
            elif sma50 > sma200 * 1.02:
                regime = "bull"
            elif sma50 < sma200 * 0.98:
                regime = "bear"
            else:
                regime = "ranging"
                
            if regime != self.current_regime:
                logger.info(f"Regime changed: {self.current_regime} -> {regime}")
                self.current_regime = regime
                self.weights = None  # Force weight refresh
                
            return regime
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return self.current_regime or "ranging"
    
    def fetch_signals(self) -> Dict:
        """Fetch all signals from dashboard"""
        # Cache signals for 30 seconds
        if self.cache_timestamp and \
           (datetime.datetime.now() - self.cache_timestamp).seconds < 30:
            return self.signals_cache
        
        signals = {
            'price': 0,      # -10 to +10
            'volume': 0,     # 0 to 10
            'sentiment': 0,  # -5 to +5
            'macro': 0,      # -5 to +5
            'orderbook': 0   # -10 to +10
        }
        
        # Fetch crypto price data
        crypto_data = self.fetch_dashboard_data('/finance/crypto/prices?limit=20')
        if crypto_data and len(crypto_data) > 0:
            btc = next((c for c in crypto_data if c.get('symbol') == 'BTC'), None)
            if btc:
                change_24h = btc.get('change_24h_pct', 0)
                volume = btc.get('volume_24h', 0)
                
                # Normalize price momentum: -10 to +10
                signals['price'] = max(-10, min(10, change_24h * 2))
                
                # Normalize volume: 0 to 10 (rough)
                signals['volume'] = min(10, volume / 1e10)
        
        # Fetch news sentiment
        news_data = self.fetch_dashboard_data('/finance/news?limit=50')
        if news_data:
            positive_keywords = ['bull', 'surge', 'gain', 'rally', 'moon', 'breakout']
            negative_keywords = ['bear', 'crash', 'dump', 'fall', 'bearish', 'fud']
            
            sentiment_score = 0
            for news in news_data[:10]:
                title = news.get('title', '').lower()
                pos_count = sum(1 for kw in positive_keywords if kw in title)
                neg_count = sum(1 for kw in negative_keywords if kw in title)
                sentiment_score += (pos_count - neg_count)
            
            signals['sentiment'] = max(-5, min(5, sentiment_score))
        
        # Fetch macro data
        macro_data = self.fetch_dashboard_data('/finance/economic')
        if macro_data:
            # Simple macro signal based on Fed rate trend
            fed_data = next((m for m in macro_data if m.get('indicator_id') == 'FEDFUNDS'), None)
            if fed_data:
                rate = fed_data.get('value', 5.0)
                # Higher rates = bearish
                signals['macro'] = max(-5, min(5, (5.5 - rate) * 2))
        
        # Fetch orderbook data
        orderbook_data = self.fetch_dashboard_data('/finance/bybit/orderbook')
        if orderbook_data:
            btc_ob = next((o for o in orderbook_data if o.get('symbol') == 'BTCUSDT'), None)
            if btc_ob:
                imbalance = btc_ob.get('imbalance', 0)
                signals['orderbook'] = max(-10, min(10, imbalance * 10))
        
        self.signals_cache = signals
        self.cache_timestamp = datetime.datetime.now()
        
        return signals
    
    def check_agent_enabled(self) -> bool:
        """Check if agent is enabled from dashboard"""
        try:
            status = self.fetch_dashboard_data('/agent/status')
            if status:
                return status.get('enabled', False)
        except Exception as e:
            logger.warning(f"Failed to check agent status: {e}")
        return False

    # ============================================================================
    # REQUIRED FREQTRADE METHODS
    # ============================================================================
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        REQUIRED: Add indicators to dataframe
        This is the main indicator calculation method
        """
        # Add basic indicators
        dataframe['sma50'] = dataframe['close'].rolling(window=SMA_FAST).mean()
        dataframe['sma200'] = dataframe['close'].rolling(window=SMA_SLOW).mean()
        dataframe['atr'] = talib.ATR(dataframe['high'], dataframe['low'], 
                                     dataframe['close'], timeperiod=ATR_PERIOD)
        
        # Detect regime and calculate score
        regime = self.detect_regime(dataframe)
        
        # Get weights for current regime
        if not self.weights or regime != self.current_regime:
            self.weights = self.fetch_weights(regime)
        
        # Fetch signals
        signals = self.fetch_signals()
        
        # Calculate weighted score
        score = (
            signals['price'] * self.weights['price'] +
            signals['volume'] * self.weights['volume'] +
            signals['sentiment'] * self.weights['sentiment'] +
            signals['macro'] * self.weights['macro'] +
            signals['orderbook'] * self.weights['orderbook']
        )
        
        # Normalize to 0-100 range
        dataframe['agent_score'] = (score + 50) / 100 * 100
        dataframe['agent_confidence'] = abs(dataframe['agent_score'])
        
        # Add regime info
        dataframe['regime'] = regime
        
        # Check if agent is enabled
        enabled = self.check_agent_enabled()
        dataframe['agent_enabled'] = enabled
        
        logger.info(f"Regime: {regime}, Score: {score:.2f}, Enabled: {enabled}")
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        REQUIRED: Define entry signals
        """
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Only trade if agent is enabled
        if not dataframe['agent_enabled'].iloc[-1]:
            logger.info("Agent disabled - no entries")
            return dataframe
        
        score = dataframe['agent_score'].iloc[-1]
        confidence = dataframe['agent_confidence'].iloc[-1]
        
        # Long entry: score > 75 (bullish)
        if confidence > MIN_CONFIDENCE and score > 0:
            dataframe.loc[:, 'enter_long'] = 1
            logger.info(f"LONG signal - Score: {score:.2f}, Confidence: {confidence:.2f}")
        
        # Short entry: score < -75 (bearish)
        if self.can_short and confidence > MIN_CONFIDENCE and score < 0:
            dataframe.loc[:, 'enter_short'] = 1
            logger.info(f"SHORT signal - Score: {score:.2f}, Confidence: {confidence:.2f}")
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        REQUIRED: Define exit signals
        """
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        score = dataframe['agent_score'].iloc[-1]
        
        # Exit long when score turns negative (bearish)
        if score < -25:
            dataframe.loc[:, 'exit_long'] = 1
        
        # Exit short when score turns positive (bullish)
        if self.can_short and score > 25:
            dataframe.loc[:, 'exit_short'] = 1
        
        return dataframe
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, 
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """Dynamic stoploss based on market regime"""
        try:
            # Use ATR-based stoploss
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                atr = dataframe['atr'].iloc[-1]
                stop_pct = -abs(atr / current_rate * 2)  # 2x ATR
                return max(stop_pct, -0.10)  # Max 10% stop
        except Exception as e:
            logger.error(f"Custom stoploss error: {e}")
        
        return -0.05  # Default 5% stoploss
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str,
                           side: str, **kwargs) -> bool:
        """Confirm entry - check if agent still enabled"""
        enabled = self.check_agent_enabled()
        if not enabled:
            logger.info(f"Trade rejected - agent disabled: {pair}")
            return False
        
        logger.info(f"Trade confirmed: {pair} {side} @ {rate}")
        return True