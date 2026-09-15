"""
Agent Models - Multibotdashboard V8
Dynamic Weight Trading Agent Models
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class SignalWeights(Base):
    """Signal weights per market regime"""
    __tablename__ = "signal_weights"
    
    id = Column(Integer, primary_key=True)
    regime = Column(String(20), unique=True, nullable=False)
    price_momentum_weight = Column(Float, default=0.25)
    volume_weight = Column(Float, default=0.20)
    sentiment_weight = Column(Float, default=0.20)
    macro_weight = Column(Float, default=0.20)
    orderbook_weight = Column(Float, default=0.15)
    total_trades = Column(Integer, default=0)
    win_count = Column(Integer, default=0)
    win_rate = Column(Float, default=50.0)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SignalPerformance(Base):
    """Performance tracking for signal learning"""
    __tablename__ = "signal_performance"
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    pair = Column(String(20), nullable=False)
    regime = Column(String(20), nullable=False)
    direction = Column(String(10))
    
    # Signal values at entry
    price_signal = Column(Float)
    volume_signal = Column(Float)
    sentiment_signal = Column(Float)
    macro_signal = Column(Float)
    orderbook_signal = Column(Float)
    
    combined_score = Column(Float)
    confidence = Column(Float)
    
    # Trade result
    outcome = Column(String(10))  # win, loss, breakeven, open
    profit_pct = Column(Float)
    duration_minutes = Column(Integer)
    executed = Column(Boolean, default=False)
    approved_by_user = Column(Boolean, default=True)


class RegimeHistory(Base):
    """Market regime history"""
    __tablename__ = "regime_history"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    regime = Column(String(20), nullable=False)
    btc_price = Column(Float)
    btc_sma50 = Column(Float)
    btc_sma200 = Column(Float)
    atr_14 = Column(Float)
    vix = Column(Float, nullable=True)
    notes = Column(String(500))


class AgentTrade(Base):
    """Agent trade log"""
    __tablename__ = "agent_trades"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    pair = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)
    confidence = Column(Float)
    stake_amount = Column(Float)
    entry_price = Column(Float)
    stoploss = Column(Float)
    take_profit = Column(Float)
    status = Column(String(20), default='pending')  # pending, open, closed
    freqtrade_trade_id = Column(String(50))
    closed_at = Column(DateTime)
    final_profit = Column(Float)
    signals = Column(JSON)


class AgentConfig(Base):
    """Agent configuration"""
    __tablename__ = "agent_config"
    
    id = Column(Integer, primary_key=True)
    key = Column(String(50), unique=True, nullable=False)
    value = Column(String, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Pydantic Schemas
class WeightsResponse(BaseModel):
    regime: str
    price_momentum_weight: float
    volume_weight: float
    sentiment_weight: float
    macro_weight: float
    orderbook_weight: float
    total_weight: float
    win_rate: float
    total_trades: int
    last_updated: Optional[datetime]


class WeightsUpdate(BaseModel):
    price_momentum_weight: float = Field(..., ge=0.05, le=0.50)
    volume_weight: float = Field(..., ge=0.05, le=0.50)
    sentiment_weight: float = Field(..., ge=0.05, le=0.50)
    macro_weight: float = Field(..., ge=0.05, le=0.50)
    orderbook_weight: float = Field(..., ge=0.05, le=0.50)


class SignalEntry(BaseModel):
    pair: str
    direction: str
    price_signal: float
    volume_signal: float
    sentiment_signal: float
    macro_signal: float
    orderbook_signal: float
    combined_score: float
    confidence: float


class TradeResult(BaseModel):
    trade_id: str
    outcome: str
    profit_pct: float
    duration_minutes: int


class PerformanceMetrics(BaseModel):
    date: str
    total_signals: int
    wins: int
    losses: int
    win_rate: float
    avg_profit: float
    total_profit: float


class AgentConfigUpdate(BaseModel):
    min_confidence: Optional[int] = Field(None, ge=50, le=95)
    min_trades_before_adjust: Optional[int] = Field(None, ge=10, le=100)
    max_weight_per_signal: Optional[float] = Field(None, ge=0.30, le=0.80)
    learning_rate: Optional[float] = Field(None, ge=0.01, le=0.20)
    position_size_pct: Optional[float] = Field(None, ge=0.005, le=0.05)
    max_concurrent_trades: Optional[int] = Field(None, ge=1, le=10)
    enabled: Optional[bool] = None
    paper_trading: Optional[bool] = None