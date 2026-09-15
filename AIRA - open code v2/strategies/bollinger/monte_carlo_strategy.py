# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
import talib.abstract as ta
from freqtrade.strategy import IStrategy, informative, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MonteCarloStrategy(IStrategy):
    """
    Monte Carlo Enhanced Trading Strategy
    
    This strategy uses Monte Carlo simulation to:
    1. Assess risk before entering trades
    2. Optimize position sizing based on simulated outcomes
    3. Set dynamic stop losses and take profits
    4. Evaluate strategy performance under different market conditions
    """

    INTERFACE_VERSION = 3

    # Strategy parameters
    timeframe = '5m'
    
    # ROI table - Conservative approach due to Monte Carlo risk management
    minimal_roi = {
        "0": 0.15,   # 15% at any time
        "10": 0.10,  # 10% after 10 minutes
        "30": 0.05,  # 5% after 30 minutes
        "60": 0.02   # 2% after 1 hour
    }

    # Stoploss
    stoploss = -0.08  # 8% stop loss

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # Monte Carlo Parameters
    mc_simulations = DecimalParameter(500, 2000, default=1000, space="buy", optimize=True)
    mc_lookback_period = IntParameter(20, 100, default=50, space="buy", optimize=True)
    confidence_threshold = DecimalParameter(0.6, 0.9, default=0.75, space="buy", optimize=True)
    risk_tolerance = DecimalParameter(0.01, 0.05, default=0.02, space="buy", optimize=True)
    
    # Technical Indicator Parameters
    rsi_period = IntParameter(10, 30, default=14, space="buy", optimize=True)
    macd_fast = IntParameter(8, 15, default=12, space="buy", optimize=True)
    macd_slow = IntParameter(20, 30, default=26, space="buy", optimize=True)
    macd_signal = IntParameter(7, 12, default=9, space="buy", optimize=True)
    bb_period = IntParameter(15, 25, default=20, space="buy", optimize=True)
    bb_std = DecimalParameter(1.8, 2.2, default=2.0, space="buy", optimize=True)

    # Buy/Sell thresholds
    rsi_buy_threshold = IntParameter(20, 40, default=30, space="buy", optimize=True)
    rsi_sell_threshold = IntParameter(60, 80, default=70, space="sell", optimize=True)
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate technical indicators and Monte Carlo metrics
        """
        
        # Basic Technical Indicators
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        
        # MACD
        macd = ta.MACD(dataframe, fastperiod=self.macd_fast.value, 
                      slowperiod=self.macd_slow.value, signalperiod=self.macd_signal.value)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = dataframe['macd'] - dataframe['macdsignal']
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=self.bb_period.value, stds=self.bb_std.value)
        dataframe['bb_lower'] = bollinger['lower']
        dataframe['bb_middle'] = bollinger['mid']
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])
        
        # Volume indicators
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # ATR for volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = dataframe['atr'] / dataframe['close']
        
        # Price momentum
        dataframe['momentum'] = ta.MOM(dataframe, timeperiod=10)
        dataframe['price_change'] = dataframe['close'].pct_change()
        
        # Monte Carlo Analysis
        dataframe = self.add_monte_carlo_indicators(dataframe)
        
        return dataframe

    def add_monte_carlo_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Add Monte Carlo simulation results as indicators
        """
        lookback = int(self.mc_lookback_period.value)
        simulations = int(self.mc_simulations.value)
        
        # Initialize Monte Carlo columns
        dataframe['mc_expected_return'] = 0.0
        dataframe['mc_risk_score'] = 0.0
        dataframe['mc_confidence'] = 0.0
        dataframe['mc_var_95'] = 0.0  # Value at Risk at 95% confidence
        dataframe['mc_sharpe_sim'] = 0.0
        dataframe['mc_win_probability'] = 0.0
        
        # Calculate rolling Monte Carlo metrics
        for i in range(lookback, len(dataframe)):
            try:
                # Get historical returns for the lookback period
                returns = dataframe['price_change'].iloc[i-lookback+1:i+1].dropna()
                
                if len(returns) < 10:  # Need minimum data
                    continue
                
                # Monte Carlo simulation
                mc_results = self.run_monte_carlo_simulation(returns, simulations)
                
                # Store results
                dataframe.loc[dataframe.index[i], 'mc_expected_return'] = mc_results['expected_return']
                dataframe.loc[dataframe.index[i], 'mc_risk_score'] = mc_results['risk_score']
                dataframe.loc[dataframe.index[i], 'mc_confidence'] = mc_results['confidence']
                dataframe.loc[dataframe.index[i], 'mc_var_95'] = mc_results['var_95']
                dataframe.loc[dataframe.index[i], 'mc_sharpe_sim'] = mc_results['sharpe_ratio']
                dataframe.loc[dataframe.index[i], 'mc_win_probability'] = mc_results['win_probability']
                
            except Exception as e:
                logger.warning(f"Monte Carlo calculation error at index {i}: {e}")
                continue
        
        return dataframe

    def run_monte_carlo_simulation(self, returns: pd.Series, n_simulations: int) -> Dict:
        """
        Run Monte Carlo simulation on historical returns
        """
        try:
            # Calculate return statistics
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Handle edge cases
            if std_return == 0 or np.isnan(std_return):
                return {
                    'expected_return': 0.0,
                    'risk_score': 1.0,
                    'confidence': 0.0,
                    'var_95': 0.0,
                    'sharpe_ratio': 0.0,
                    'win_probability': 0.5
                }
            
            # Generate random scenarios
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            
            # Calculate metrics
            expected_return = np.mean(simulated_returns)
            var_95 = np.percentile(simulated_returns, 5)  # 5th percentile (95% VaR)
            
            # Risk score (higher = riskier)
            risk_score = min(abs(var_95) / abs(mean_return) if mean_return != 0 else 1.0, 2.0)
            
            # Confidence in positive returns
            positive_returns = np.sum(simulated_returns > 0)
            confidence = positive_returns / n_simulations
            
            # Sharpe ratio estimation
            risk_free_rate = 0.0  # Assume 0 for crypto
            sharpe_ratio = (expected_return - risk_free_rate) / std_return if std_return > 0 else 0.0
            
            # Win probability (probability of positive return)
            win_probability = confidence
            
            return {
                'expected_return': expected_return,
                'risk_score': risk_score,
                'confidence': confidence,
                'var_95': var_95,
                'sharpe_ratio': sharpe_ratio,
                'win_probability': win_probability
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo simulation error: {e}")
            return {
                'expected_return': 0.0,
                'risk_score': 1.0,
                'confidence': 0.0,
                'var_95': 0.0,
                'sharpe_ratio': 0.0,
                'win_probability': 0.5
            }

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define buy signals with Monte Carlo risk assessment
        """
        
        # Technical conditions for potential buy
        technical_buy = (
            # RSI oversold
            (dataframe['rsi'] < self.rsi_buy_threshold.value) &
            
            # MACD showing momentum
            (dataframe['macdhist'] > 0) &
            (dataframe['macd'] > dataframe['macdsignal']) &
            
            # Price near lower Bollinger Band (oversold)
            (dataframe['bb_percent'] < 0.2) &
            
            # Volume confirmation
            (dataframe['volume_ratio'] > 1.1) &
            
            # Positive momentum
            (dataframe['momentum'] > 0)
        )
        
        # Monte Carlo risk assessment
        monte_carlo_buy = (
            # Expected positive return
            (dataframe['mc_expected_return'] > 0) &
            
            # High confidence in positive outcome
            (dataframe['mc_confidence'] > self.confidence_threshold.value) &
            
            # Acceptable risk level
            (dataframe['mc_risk_score'] < (1 / self.risk_tolerance.value)) &
            
            # Positive Sharpe ratio
            (dataframe['mc_sharpe_sim'] > 0) &
            
            # High win probability
            (dataframe['mc_win_probability'] > 0.6)
        )
        
        # Combined buy signal
        dataframe.loc[
            technical_buy & monte_carlo_buy,
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define sell signals with Monte Carlo guidance
        """
        
        # Technical conditions for sell
        technical_sell = (
            # RSI overbought
            (dataframe['rsi'] > self.rsi_sell_threshold.value) |
            
            # MACD bearish crossover
            (
                (dataframe['macdhist'] < 0) &
                (dataframe['macd'] < dataframe['macdsignal'])
            ) |
            
            # Price near upper Bollinger Band
            (dataframe['bb_percent'] > 0.9)
        )
        
        # Monte Carlo risk-based sell
        monte_carlo_sell = (
            # Expected negative return
            (dataframe['mc_expected_return'] < 0) |
            
            # Low confidence
            (dataframe['mc_confidence'] < 0.4) |
            
            # High risk
            (dataframe['mc_risk_score'] > (2 / self.risk_tolerance.value)) |
            
            # Poor risk-adjusted returns
            (dataframe['mc_sharpe_sim'] < -0.5)
        )
        
        # Combined sell signal
        dataframe.loc[
            technical_sell | monte_carlo_sell,
            'exit_long'
        ] = 1

        return dataframe

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: Optional[float], max_stake: float,
                           leverage: float, entry_tag: Optional[str], side: str,
                           **kwargs) -> float:
        """
        Dynamic position sizing based on Monte Carlo risk assessment
        """
        try:
            # Get current dataframe
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return proposed_stake
            
            # Get latest Monte Carlo metrics
            latest_data = dataframe.iloc[-1]
            
            mc_confidence = latest_data.get('mc_confidence', 0.5)
            mc_risk_score = latest_data.get('mc_risk_score', 1.0)
            mc_sharpe = latest_data.get('mc_sharpe_sim', 0.0)
            
            # Calculate risk-adjusted position size
            base_risk_factor = 1.0
            
            # Adjust based on confidence
            confidence_multiplier = min(mc_confidence * 1.5, 1.2)
            
            # Adjust based on risk score (lower risk = higher position)
            risk_multiplier = max(0.3, 1.0 / (1.0 + mc_risk_score))
            
            # Adjust based on Sharpe ratio
            sharpe_multiplier = max(0.5, min(1.0 + mc_sharpe * 0.2, 1.3))
            
            # Combined multiplier
            final_multiplier = base_risk_factor * confidence_multiplier * risk_multiplier * sharpe_multiplier
            final_multiplier = max(0.2, min(final_multiplier, 1.5))  # Cap between 20% and 150%
            
            # Calculate final stake
            adjusted_stake = proposed_stake * final_multiplier
            
            # Ensure within bounds
            if min_stake:
                adjusted_stake = max(adjusted_stake, min_stake)
            adjusted_stake = min(adjusted_stake, max_stake)
            
            logger.info(f"Monte Carlo position sizing for {pair}: "
                       f"Confidence: {mc_confidence:.3f}, Risk: {mc_risk_score:.3f}, "
                       f"Sharpe: {mc_sharpe:.3f}, Multiplier: {final_multiplier:.3f}, "
                       f"Stake: {adjusted_stake:.6f}")
            
            return adjusted_stake
            
        except Exception as e:
            logger.error(f"Error in custom_stake_amount: {e}")
            return proposed_stake

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Dynamic stop loss based on Monte Carlo VaR
        """
        try:
            # Get current dataframe
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return self.stoploss
            
            # Get latest Monte Carlo VaR
            latest_data = dataframe.iloc[-1]
            mc_var_95 = latest_data.get('mc_var_95', -0.05)
            
            # Use VaR as dynamic stop loss (but cap it for safety)
            dynamic_stoploss = max(mc_var_95 * 1.5, -0.15)  # Max 15% loss
            dynamic_stoploss = min(dynamic_stoploss, self.stoploss)  # Don't be more aggressive than base stoploss
            
            return dynamic_stoploss
            
        except Exception as e:
            logger.error(f"Error in custom_stoploss: {e}")
            return self.stoploss

    def informative_pairs(self):
        """
        Define additional pairs for analysis
        """
        return []

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], 
                side: str, **kwargs) -> float:
        """
        Conservative leverage based on Monte Carlo risk assessment
        """
        try:
            # Get current dataframe
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return 1.0
            
            # Get latest Monte Carlo metrics
            latest_data = dataframe.iloc[-1]
            mc_confidence = latest_data.get('mc_confidence', 0.5)
            mc_risk_score = latest_data.get('mc_risk_score', 1.0)
            
            # Conservative leverage calculation
            if mc_confidence > 0.8 and mc_risk_score < 0.5:
                return min(2.0, max_leverage)  # Max 2x leverage for high confidence, low risk
            elif mc_confidence > 0.6 and mc_risk_score < 1.0:
                return min(1.5, max_leverage)  # 1.5x for moderate confidence
            else:
                return 1.0  # No leverage for uncertain conditions
                
        except Exception as e:
            logger.error(f"Error in leverage calculation: {e}")
            return 1.0