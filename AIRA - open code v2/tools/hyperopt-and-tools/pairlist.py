"""
Database models for Pairlist Selector
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship

from src.models import Base

class PairlistJob(Base):
    """Track pairlist selector jobs"""
    __tablename__ = "pairlist_jobs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(100), unique=True, index=True, nullable=False)
    strategy = Column(String(100), nullable=False)
    mode = Column(String(50), nullable=False)  # ml_training, fullbacktest_batch, fullbacktest_individual
    status = Column(String(20), nullable=False)  # running, completed, failed, stopped
    n_pairs = Column(Integer, default=50)
    download_days = Column(Integer, default=60)
    backtest_days = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationship to results
    results = relationship("PairlistResult", back_populates="job", uselist=False)

class PairlistResult(Base):
    """Store pairlist selector results summary"""
    __tablename__ = "pairlist_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(100), ForeignKey("pairlist_jobs.job_id"), unique=True, nullable=False)
    strategy = Column(String(100), nullable=False)
    timeframe = Column(String(20), nullable=True)
    evaluation_mode = Column(String(50), nullable=False)
    total_pairs = Column(Integer, default=0)
    
    # Best pair metrics
    best_pair = Column(String(50), nullable=True)
    best_profit = Column(Float, default=0)
    best_sharpe = Column(Float, default=0)
    
    # Average metrics
    avg_profit = Column(Float, default=0)
    avg_win_rate = Column(Float, default=0)
    avg_max_drawdown = Column(Float, default=0)
    
    # Full results JSON
    results_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    job = relationship("PairlistJob", back_populates="results")
    pair_details = relationship("PairlistPairResult", back_populates="result")

class PairlistPairResult(Base):
    """Individual pair results from pairlist selector"""
    __tablename__ = "pairlist_pair_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(100), ForeignKey("pairlist_results.job_id"), nullable=False)
    pair = Column(String(50), nullable=False)
    rank = Column(Integer, nullable=False)
    
    # Performance metrics
    profit_total = Column(Float, default=0)
    win_rate = Column(Float, default=0)
    max_drawdown = Column(Float, default=0)
    sharpe_ratio = Column(Float, default=0)
    sortino_ratio = Column(Float, nullable=True)
    calmar_ratio = Column(Float, nullable=True)
    expectancy = Column(Float, nullable=True)
    trade_count = Column(Integer, default=0)
    avg_profit = Column(Float, nullable=True)
    avg_duration = Column(Float, nullable=True)
    
    # ML metrics (if applicable)
    train_accuracy = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)
    accuracy_gap = Column(Float, nullable=True)
    
    # Composite score
    score = Column(Float, default=0)
    
    # Full metrics JSON
    metrics_json = Column(JSON, nullable=True)
    
    # Relationship
    result = relationship("PairlistResult", back_populates="pair_details")
