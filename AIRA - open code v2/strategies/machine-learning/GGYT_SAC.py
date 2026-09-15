"""
GGYT_SAC - Recurrent Soft Actor-Critic Trading Strategy
freqtrade version: 2025.6
Python Version: 3.12.8

This strategy uses a Recurrent SAC (Soft Actor-Critic) reinforcement learning
agent with LSTM networks for decision making.

Usage:
    freqtrade trade -s GGYT_SAC
    freqtrade backtesting -s GGYT_SAC
"""

import copy
import datetime
import json
import logging
import math
import os
import pickle
import threading
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import talib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pandas import DataFrame
from pandas_ta import ema
from sklearn.preprocessing import StandardScaler
from torch.distributions import Normal

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy
from freqtrade.strategy.strategy_helper import timeframe_to_prev_date

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════════
# ENHANCED LOGGER
# ════════════════════════════════════════════════════════════════════════════════
class EnhancedLogger:
    """Enhanced logging with emojis and formatting"""
    
    @staticmethod
    def log_banner(message: str, emoji: str = "🚀"):
        border = "═" * 60
        logger.info(f"\n{emoji} {border}")
        logger.info(f"{emoji}   {message.upper()}")
        logger.info(f"{emoji} {border}")
    
    @staticmethod
    def log_section(title: str, emoji: str = "📊"):
        logger.info(f"\n{emoji} ══════ {title} ══════")
    
    @staticmethod
    def log_parameter(name: str, value: Any, emoji: str = "⚙️"):
        logger.info(f"  {emoji} {name}: {value}")
    
    @staticmethod
    def log_error(message: str, emoji: str = "❌"):
        logger.error(f"{emoji} {message}")
    
    @staticmethod
    def log_warning(message: str, emoji: str = "⚠️"):
        logger.warning(f"{emoji} {message}")
    
    @staticmethod
    def log_success(message: str, emoji: str = "✅"):
        logger.info(f"{emoji} {message}")


# ════════════════════════════════════════════════════════════════════════════════
# SEQUENCE REPLAY BUFFER
# ════════════════════════════════════════════════════════════════════════════════
class SequenceReplayBuffer:
    """Experience replay buffer for sequential data"""
    
    def __init__(self, capacity: int, sequence_length: int, obs_dim: int, 
                 action_dim: int, device: str = 'cpu'):
        self.capacity = capacity
        self.seq_len = sequence_length
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.obs_dim = obs_dim
        
        self.obs_buf = np.zeros((capacity + sequence_length, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity + sequence_length, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity + sequence_length, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros((capacity + sequence_length, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity + sequence_length, 1), dtype=np.float32)

    def push(self, obs: np.ndarray, act: np.ndarray, rew: float, 
             next_obs: np.ndarray, done: bool):
        """Add experience to buffer"""
        obs_step = obs[-1] if len(obs.shape) > 1 else obs
        next_obs_step = next_obs[-1] if len(next_obs.shape) > 1 else next_obs

        self.obs_buf[self.ptr] = obs_step
        self.next_obs_buf[self.ptr] = next_obs_step
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        
        if self.ptr < self.seq_len:
            self.obs_buf[self.capacity + self.ptr] = obs_step
            self.next_obs_buf[self.capacity + self.ptr] = next_obs_step
            self.act_buf[self.capacity + self.ptr] = act
            self.rew_buf[self.capacity + self.ptr] = rew
            self.done_buf[self.capacity + self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of sequences"""
        valid_indices = []
        attempts = 0
        while len(valid_indices) < batch_size and attempts < batch_size * 10:
            idx = np.random.randint(0, min(self.size, self.capacity - self.seq_len))
            if idx < self.ptr and idx + self.seq_len >= self.ptr:
                attempts += 1
                continue
            valid_indices.append(idx)
            attempts += 1
        
        if len(valid_indices) < batch_size:
            valid_indices = np.random.randint(0, max(1, self.size - self.seq_len), batch_size)
            
        indices = np.array(valid_indices[:batch_size])
        batch_indices = indices[:, None] + np.arange(self.seq_len)[None, :]
        
        return (
            torch.as_tensor(self.obs_buf[batch_indices], device=self.device),
            torch.as_tensor(self.act_buf[batch_indices], device=self.device),
            torch.as_tensor(self.rew_buf[batch_indices], device=self.device),
            torch.as_tensor(self.next_obs_buf[batch_indices], device=self.device),
            torch.as_tensor(self.done_buf[batch_indices], device=self.device)
        )


# ════════════════════════════════════════════════════════════════════════════════
# RECURRENT ACTOR NETWORK
# ════════════════════════════════════════════════════════════════════════════════
class RecurrentActor(nn.Module):
    """LSTM-based policy network for action generation"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super(RecurrentActor, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs_seq: torch.Tensor, hidden_state: Optional[Tuple] = None):
        """Forward pass through actor network"""
        features = self.feature_layer(obs_seq)
        lstm_out, new_hidden_state = self.lstm(features, hidden_state)
        
        mu = self.mu_layer(lstm_out)
        log_std = self.log_std_layer(lstm_out)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, new_hidden_state


# ════════════════════════════════════════════════════════════════════════════════
# RECURRENT CRITIC NETWORK
# ════════════════════════════════════════════════════════════════════════════════
class RecurrentCritic(nn.Module):
    """Twin LSTM-based Q-value networks"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super(RecurrentCritic, self).__init__()
        # Q1
        self.l1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out1 = nn.Linear(hidden_dim, 1)
        # Q2
        self.l2 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out2 = nn.Linear(hidden_dim, 1)

    def forward(self, obs_seq: torch.Tensor, action_seq: torch.Tensor):
        """Forward pass through critic networks"""
        xu = torch.cat([obs_seq, action_seq], dim=-1)
        
        # Q1
        x1 = F.relu(self.l1(xu))
        x1, _ = self.lstm1(x1)
        q1 = self.out1(x1)
        
        # Q2
        x2 = F.relu(self.l2(xu))
        x2, _ = self.lstm2(x2)
        q2 = self.out2(x2)
        
        return q1, q2


# ════════════════════════════════════════════════════════════════════════════════
# RECURRENT SAC AGENT
# ════════════════════════════════════════════════════════════════════════════════
class RecurrentSACAgent:
    """Recurrent Soft Actor-Critic agent for trading"""
    
    FEATURE_COUNT = 20
    SEQUENCE_LENGTH = 20
    
    def __init__(self, pair: str, obs_dim: int = 20, action_dim: int = 1,
                 buffer_size: int = 50000, batch_size: int = 32, 
                 seq_len: int = 20, device: str = 'cpu'):
        self.pair = pair
        self.device = device
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        
        # Networks
        self.actor = RecurrentActor(obs_dim, action_dim).to(device)
        self.critic = RecurrentCritic(obs_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # Memory
        self.memory = SequenceReplayBuffer(buffer_size, seq_len, obs_dim, action_dim, device)
        
        # Hidden state for online inference
        self.hidden_state = None
        
        # Feature scaler
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
        # Training state
        self.is_trained = False
        self.training_steps = 0
        self.is_training = False
        self._training_lock = threading.Lock()
        
        # Model paths
        pair_safe = pair.replace('/', '_')
        self.model_dir = "user_data/strategies/sac_models"
        self.actor_path = f"{self.model_dir}/sac_actor_{pair_safe}.pt"
        self.critic_path = f"{self.model_dir}/sac_critic_{pair_safe}.pt"
        self.scaler_path = f"{self.model_dir}/sac_scaler_{pair_safe}.pkl"
        
        os.makedirs(self.model_dir, exist_ok=True)
        self.load_models()
    
    def load_models(self):
        """Load saved models if they exist"""
        try:
            if os.path.exists(self.actor_path):
                self.actor.load_state_dict(torch.load(self.actor_path, map_location=self.device))
                self.is_trained = True
                EnhancedLogger.log_success(f"Loaded actor model for {self.pair}", "📂")
            
            if os.path.exists(self.critic_path):
                self.critic.load_state_dict(torch.load(self.critic_path, map_location=self.device))
                self.critic_target = copy.deepcopy(self.critic)
                EnhancedLogger.log_success(f"Loaded critic model for {self.pair}", "📂")
            
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                    self.scaler_fitted = True
                    
        except Exception as e:
            EnhancedLogger.log_warning(f"Could not load models: {e}", "⚠️")
    
    def save_models(self):
        """Save models to disk"""
        try:
            torch.save(self.actor.state_dict(), self.actor_path)
            torch.save(self.critic.state_dict(), self.critic_path)
            
            if self.scaler_fitted:
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                    
            EnhancedLogger.log_success(f"Saved models for {self.pair}", "💾")
        except Exception as e:
            EnhancedLogger.log_error(f"Failed to save models: {e}", "❌")
    
    def get_action(self, obs_seq: np.ndarray, evaluate: bool = False) -> Tuple[float, float]:
        """Get action from policy network"""
        if not self.scaler_fitted:
            return 0.0, 0.5  # Neutral action, medium confidence
        
        try:
            # Normalize observation
            obs_flat = obs_seq.reshape(-1, self.FEATURE_COUNT)
            obs_scaled = self.scaler.transform(obs_flat)
            obs_reshaped = obs_scaled.reshape(1, -1, self.FEATURE_COUNT)
            
            obs_tensor = torch.FloatTensor(obs_reshaped).to(self.device)
            
            with torch.no_grad():
                action, log_prob, new_hidden = self.actor(obs_tensor, self.hidden_state)
                self.hidden_state = new_hidden
                
                action_value = action.cpu().numpy()[0, -1, 0]
                
                # Calculate confidence from log_prob
                confidence = 1.0 / (1.0 + np.exp(-log_prob.cpu().numpy()[0, -1, 0]))
                
                return float(action_value), float(confidence)
                
        except Exception as e:
            EnhancedLogger.log_error(f"Action prediction failed: {e}", "❌")
            return 0.0, 0.5
    
    def store_transition(self, obs: np.ndarray, action: float, reward: float,
                        next_obs: np.ndarray, done: bool):
        """Store transition in replay buffer"""
        self.memory.push(obs, np.array([action]), reward, next_obs, done)
    
    def update_parameters(self) -> float:
        """Update actor and critic networks"""
        if self.memory.size < self.batch_size * 2:
            return 0.0
        
        try:
            # Sample sequences
            obs, act, rew, next_obs, done = self.memory.sample(self.batch_size)
            
            # Calculate target Q
            with torch.no_grad():
                next_action, next_log_prob, _ = self.actor(next_obs)
                target_q1, target_q2 = self.critic_target(next_obs, next_action)
                target_q_min = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
                target_q_val = rew + (1 - done) * self.gamma * target_q_min

            # Update critic
            current_q1, current_q2 = self.critic(obs, act)
            critic_loss = F.mse_loss(current_q1, target_q_val) + F.mse_loss(current_q2, target_q_val)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Update actor
            new_action, log_prob, _ = self.actor(obs)
            q1_new, q2_new = self.critic(obs, new_action)
            q_new_min = torch.min(q1_new, q2_new)
            
            actor_loss = (self.alpha * log_prob - q_new_min).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            self.training_steps += 1
            self.is_trained = True
            
            return critic_loss.item()
            
        except Exception as e:
            EnhancedLogger.log_error(f"Update failed: {e}", "❌")
            return 0.0
    
    def fit_scaler(self, features: np.ndarray):
        """Fit the feature scaler"""
        self.scaler.fit(features)
        self.scaler_fitted = True
    
    def reset_hidden_state(self):
        """Reset LSTM hidden state"""
        self.hidden_state = None


# ════════════════════════════════════════════════════════════════════════════════
# MAIN STRATEGY CLASS
# ════════════════════════════════════════════════════════════════════════════════
class GGYT_SAC(IStrategy):
    """
    GGYT_SAC - Recurrent Soft Actor-Critic Trading Strategy
    
    Uses LSTM-based SAC agent for generating long/short signals
    with dynamic leverage and stop-loss based on agent confidence.
    """
    
    # Strategy settings
    timeframe = "1h"
    startup_candle_count = 200
    minimal_roi = {}
    stoploss = -0.50
    use_custom_stoploss = True
    trailing_stop = False
    can_short = True
    
    # Position sizing
    position_adjustment_enable = True
    max_entry_position_adjustment = 3
    
    # SAC Parameters
    sac_action_threshold = DecimalParameter(0.2, 0.5, default=0.3, space="buy")
    sac_confidence_min = DecimalParameter(0.3, 0.7, default=0.5, space="buy")
    
    # Risk Parameters
    ATR_Multip = DecimalParameter(1.0, 5.0, default=2.5, space="sell")
    ATR_SL_long_Multip = DecimalParameter(1.0, 5.0, default=2.0, space="sell")
    ATR_SL_short_Multip = DecimalParameter(1.0, 5.0, default=2.0, space="sell")
    rr_long = DecimalParameter(1.0, 4.0, default=2.0, space="sell")
    rr_short = DecimalParameter(1.0, 4.0, default=2.0, space="sell")
    
    # Feature columns
    FEATURE_COLS = [
        'atr', 'atr_std',
        'momentum_5', 'momentum_10', 'momentum_20',
        'volume_sma', 'volume_change',
        'fisher', 'fisher_mean', 'fisher_std',
        'baseline_combined',
        'market_regime',
        'rsi_14',
        'macd', 'macd_hist',
        'bb_width', 'bb_percent',
        'close_norm', 'high_norm', 'low_norm'
    ]
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        
        EnhancedLogger.log_banner("GGYT_SAC STRATEGY INITIALIZATION", "🤖")
        
        # SAC agents per pair
        self.sac_agents: Dict[str, RecurrentSACAgent] = {}
        
        # Device selection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        EnhancedLogger.log_parameter("Device", self.device, "💻")
        
        # Trade tracking
        self.trade_rewards: Dict[str, List[float]] = {}
        
        EnhancedLogger.log_success("Strategy initialized", "✅")
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate all indicators and features for SAC agent"""
        pair = metadata.get("pair", "UNKNOWN")
        
        EnhancedLogger.log_section(f"INDICATORS - {pair}", "📊")
        
        # ════════════════════════════════════════════════════════════════
        # VOLATILITY FEATURES
        # ════════════════════════════════════════════════════════════════
        dataframe['atr'] = talib.ATR(
            dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14
        )
        dataframe['atr_std'] = dataframe['atr'].rolling(7).std()
        
        # ════════════════════════════════════════════════════════════════
        # MOMENTUM FEATURES
        # ════════════════════════════════════════════════════════════════
        dataframe['momentum_5'] = dataframe['close'].pct_change(5)
        dataframe['momentum_10'] = dataframe['close'].pct_change(10)
        dataframe['momentum_20'] = dataframe['close'].pct_change(20)
        
        # ════════════════════════════════════════════════════════════════
        # VOLUME FEATURES
        # ════════════════════════════════════════════════════════════════
        dataframe['volume_sma'] = dataframe['volume'].rolling(14).mean()
        dataframe['volume_change'] = dataframe['volume'].pct_change()
        
        # ════════════════════════════════════════════════════════════════
        # FISHER TRANSFORM
        # ════════════════════════════════════════════════════════════════
        period = 14
        highest = dataframe['high'].rolling(period).max()
        lowest = dataframe['low'].rolling(period).min()
        value = 2 * ((dataframe['close'] - lowest) / (highest - lowest + 1e-10)) - 1
        value = value.clip(-0.999, 0.999)
        dataframe['fisher'] = 0.5 * np.log((1 + value) / (1 - value + 1e-10))
        dataframe['fisher'] = dataframe['fisher'].ewm(span=3).mean()
        dataframe['fisher_mean'] = dataframe['fisher'].rolling(5).mean()
        dataframe['fisher_std'] = dataframe['fisher'].rolling(5).std()
        
        # ════════════════════════════════════════════════════════════════
        # BASELINE FEATURES
        # ════════════════════════════════════════════════════════════════
        dataframe['baseline'] = ema(dataframe['close'], length=14)
        dataframe['baseline_diff'] = dataframe['baseline'].diff()
        baseline_diff_mean = dataframe['baseline_diff'].rolling(5).mean()
        baseline_diff_sum = dataframe['baseline_diff'].rolling(10).sum()
        dataframe['baseline_combined'] = baseline_diff_mean + baseline_diff_sum * 0.1
        
        # ════════════════════════════════════════════════════════════════
        # MARKET REGIME
        # ════════════════════════════════════════════════════════════════
        sma_50 = dataframe['close'].rolling(50).mean()
        sma_200 = dataframe['close'].rolling(200).mean()
        dataframe['market_regime'] = np.where(sma_50 > sma_200, 1.0, -1.0)
        
        # ════════════════════════════════════════════════════════════════
        # RSI
        # ════════════════════════════════════════════════════════════════
        dataframe['rsi_14'] = talib.RSI(dataframe['close'], timeperiod=14)
        
        # ════════════════════════════════════════════════════════════════
        # MACD
        # ════════════════════════════════════════════════════════════════
        macd, macdsignal, macdhist = talib.MACD(
            dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        dataframe['macd'] = macd
        dataframe['macd_hist'] = macdhist
        
        # ════════════════════════════════════════════════════════════════
        # BOLLINGER BANDS
        # ════════════════════════════════════════════════════════════════
        upper, middle, lower = talib.BBANDS(
            dataframe['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        dataframe['bb_width'] = (upper - lower) / (middle + 1e-10)
        dataframe['bb_percent'] = (dataframe['close'] - lower) / (upper - lower + 1e-10)
        
        # ════════════════════════════════════════════════════════════════
        # NORMALIZED PRICE FEATURES
        # ════════════════════════════════════════════════════════════════
        close_mean = dataframe['close'].rolling(50).mean()
        close_std = dataframe['close'].rolling(50).std()
        dataframe['close_norm'] = (dataframe['close'] - close_mean) / (close_std + 1e-10)
        dataframe['high_norm'] = (dataframe['high'] - close_mean) / (close_std + 1e-10)
        dataframe['low_norm'] = (dataframe['low'] - close_mean) / (close_std + 1e-10)
        
        # Fill NaN values
        dataframe.fillna(0, inplace=True)
        
        EnhancedLogger.log_success(f"Calculated {len(self.FEATURE_COLS)} features", "✅")
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate entry signals using SAC agent"""
        pair = metadata.get("pair", "UNKNOWN")
        
        EnhancedLogger.log_section(f"SAC ENTRY ANALYSIS - {pair}", "🤖")
        
        # Initialize SAC agent for this pair
        if pair not in self.sac_agents:
            self.sac_agents[pair] = RecurrentSACAgent(
                pair=pair,
                obs_dim=len(self.FEATURE_COLS),
                device=self.device
            )
            EnhancedLogger.log_success(f"Initialized SAC agent for {pair}", "🆕")
        
        agent = self.sac_agents[pair]
        
        # Prepare features
        available_cols = [c for c in self.FEATURE_COLS if c in dataframe.columns]
        features = dataframe[available_cols].values
        features = np.nan_to_num(features, nan=0.0)
        
        # Fit scaler if needed
        if not agent.scaler_fitted and len(features) > 100:
            agent.fit_scaler(features)
            EnhancedLogger.log_success("Fitted feature scaler", "📐")
        
        # Get SAC action for last candle
        seq_length = RecurrentSACAgent.SEQUENCE_LENGTH
        if len(dataframe) >= seq_length and agent.scaler_fitted:
            seq_start = len(dataframe) - seq_length
            sequence = features[seq_start:]
            
            action, confidence = agent.get_action(sequence)
            
            EnhancedLogger.log_parameter("SAC Action", f"{action:.3f}", "🎮")
            EnhancedLogger.log_parameter("SAC Confidence", f"{confidence:.3f}", "📊")
            
            action_threshold = self.sac_action_threshold.value
            confidence_min = self.sac_confidence_min.value
            
            # Long signal
            if action > action_threshold and confidence > confidence_min:
                dataframe.loc[dataframe.index[-1], ["enter_long", "enter_tag"]] = [
                    1,
                    f"sac_long_a{action:.2f}_c{confidence:.2f}",
                ]
                EnhancedLogger.log_success(f"LONG ENTRY: action={action:.3f}", "🟢")
            
            # Short signal
            elif action < -action_threshold and confidence > confidence_min:
                if self.can_short:
                    dataframe.loc[dataframe.index[-1], ["enter_short", "enter_tag"]] = [
                        1,
                        f"sac_short_a{action:.2f}_c{confidence:.2f}",
                    ]
                    EnhancedLogger.log_success(f"SHORT ENTRY: action={action:.3f}", "🔴")
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate exit signals using SAC agent"""
        pair = metadata.get("pair", "UNKNOWN")
        
        # Get SAC agent
        if pair not in self.sac_agents:
            return dataframe
        
        agent = self.sac_agents[pair]
        
        available_cols = [c for c in self.FEATURE_COLS if c in dataframe.columns]
        features = dataframe[available_cols].values
        features = np.nan_to_num(features, nan=0.0)
        
        seq_length = RecurrentSACAgent.SEQUENCE_LENGTH
        if len(dataframe) >= seq_length and agent.scaler_fitted:
            seq_start = len(dataframe) - seq_length
            sequence = features[seq_start:]
            
            action, confidence = agent.get_action(sequence)
            
            # Exit long if action goes negative
            if action < -0.1:
                dataframe.loc[dataframe.index[-1], "exit_long"] = 1
            
            # Exit short if action goes positive
            if action > 0.1:
                dataframe.loc[dataframe.index[-1], "exit_short"] = 1
        
        return dataframe
    
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime.datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """SAC confidence-based dynamic stop loss"""
        
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            trade_candle = dataframe.loc[dataframe["date"] == trade_date]

            if not trade_candle.empty:
                trade_candle = trade_candle.squeeze()

                EnhancedLogger.log_section(f"SAC STOPLOSS - {pair}", "🛡️")

                # Get SAC confidence
                confidence = 0.5
                if pair in self.sac_agents:
                    agent = self.sac_agents[pair]
                    available_cols = [c for c in self.FEATURE_COLS if c in dataframe.columns]
                    features = dataframe[available_cols].values
                    features = np.nan_to_num(features, nan=0.0)
                    
                    seq_length = RecurrentSACAgent.SEQUENCE_LENGTH
                    if len(features) >= seq_length and agent.scaler_fitted:
                        sequence = features[-seq_length:]
                        _, confidence = agent.get_action(sequence)
                
                # Base SL multiplier
                base_sl_multiplier = (
                    self.ATR_SL_long_Multip.value
                    if not trade.is_short
                    else self.ATR_SL_short_Multip.value
                )
                
                # SAC confidence adjustment (0.5x to 1.5x)
                sac_sl_factor = 0.5 + confidence
                sl_multiplier = base_sl_multiplier * sac_sl_factor
                
                EnhancedLogger.log_parameter("SAC Confidence", f"{confidence:.3f}", "📊")
                EnhancedLogger.log_parameter("SL Multiplier", f"{sl_multiplier:.2f}x", "🛡️")
                
                # Calculate SL
                atr_value = trade_candle["atr"]
                sl_distance = atr_value * sl_multiplier

                if not trade.is_short:
                    sl_price = trade.open_rate - sl_distance
                    if current_rate < sl_price:
                        EnhancedLogger.log_warning("STOP LOSS TRIGGERED!", "🛑")
                        return -0.0001
                else:
                    sl_price = trade.open_rate + sl_distance
                    if current_rate > sl_price:
                        EnhancedLogger.log_warning("STOP LOSS TRIGGERED!", "🛑")
                        return -0.0001

        except Exception as e:
            EnhancedLogger.log_error(f"Stoploss calculation failed: {e}", "❌")

        return self.stoploss
    
    def leverage(
        self,
        pair: str,
        current_time: datetime.datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        side: str,
        **kwargs,
    ) -> float:
        """SAC confidence-based dynamic leverage 1x-50x"""
        
        EnhancedLogger.log_section(f"SAC LEVERAGE - {pair}", "⚖️")
        
        MIN_LEVERAGE = 1.0
        MAX_LEVERAGE = 50.0
        
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            
            # Get SAC confidence
            confidence = 0.5
            if pair in self.sac_agents:
                agent = self.sac_agents[pair]
                available_cols = [c for c in self.FEATURE_COLS if c in dataframe.columns]
                features = dataframe[available_cols].values
                features = np.nan_to_num(features, nan=0.0)
                
                seq_length = RecurrentSACAgent.SEQUENCE_LENGTH
                if len(features) >= seq_length and agent.scaler_fitted:
                    sequence = features[-seq_length:]
                    action, confidence = agent.get_action(sequence)
                    
                    EnhancedLogger.log_parameter("SAC Action", f"{action:.3f}", "🎮")
                    EnhancedLogger.log_parameter("SAC Confidence", f"{confidence:.3f}", "📊")
            
            # Linear leverage calculation
            leverage = MIN_LEVERAGE + (MAX_LEVERAGE - MIN_LEVERAGE) * confidence
            
            # Volatility cap
            if not dataframe.empty:
                volatility = dataframe["atr"].iloc[-1] / dataframe["close"].iloc[-1]
                if volatility > 0.05:
                    leverage = min(leverage, 20.0)
                    EnhancedLogger.log_warning("High volatility - capping at 20x", "⚠️")
            
            final_leverage = min(leverage, max_leverage)
            
            leverage_emoji = "🚀" if final_leverage >= 30 else "📈" if final_leverage >= 10 else "📊"
            EnhancedLogger.log_parameter(f"LEVERAGE {leverage_emoji}", f"{final_leverage:.1f}x", "🎯")
            
            return final_leverage

        except Exception as e:
            EnhancedLogger.log_error(f"Leverage calculation failed: {e}", "❌")

        return MIN_LEVERAGE
    
    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime.datetime,
        **kwargs,
    ) -> bool:
        """Learn from trade result"""
        
        EnhancedLogger.log_section(f"TRADE EXIT - {pair}", "📤")
        
        # Calculate reward
        profit = trade.calc_profit_ratio(rate)
        reward = profit * 10  # Scale reward
        
        EnhancedLogger.log_parameter("Profit", f"{profit:.2%}", "💰")
        EnhancedLogger.log_parameter("Reward", f"{reward:.3f}", "🎁")
        
        # Update SAC agent
        if pair in self.sac_agents:
            agent = self.sac_agents[pair]
            
            # Perform learning update
            if agent.memory.size > agent.batch_size:
                loss = agent.update_parameters()
                EnhancedLogger.log_parameter("Training Loss", f"{loss:.4f}", "📉")
                
                # Save models periodically
                if agent.training_steps % 100 == 0:
                    agent.save_models()
        
        return True
