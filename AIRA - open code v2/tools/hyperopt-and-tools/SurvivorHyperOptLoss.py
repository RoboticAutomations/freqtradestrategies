"""
SurvivorHyperOptLoss — Custom loss for leveraged futures strategies.

The problem with standard loss functions:
- SharpeDaily: rewards high-frequency low-quality trades, ignores total ruin
- OnlyProfit: ignores drawdown, selects overleveraged configs that blow up
- ProfitDrawDown: drawdown penalty too weak, still selects losers
- Calmar: only uses max drawdown, misses sustained underwater periods

SurvivorHyperOptLoss solves this by requiring SURVIVAL first, THEN optimizing:

1. HARD GATES (instant penalty if violated):
   - Must be net profitable (profit > 0)
   - Must have minimum trade count (avoid lucky few-trade flukes)
   - Profit factor must be > 1.0 (wins must exceed losses)
   - Max drawdown must not exceed ruin threshold

2. SCORE (weighted combination for survivors):
   - Profit (log-scaled to prevent chasing outlier returns)
   - Profit Factor (quality of edge)
   - Win Rate (consistency)
   - Drawdown penalty (exponential — 10% DD is fine, 40% DD is devastating)
   - Trade count bonus (more trades = more statistical confidence)
   - Expectancy (expected $ per trade)

Lower return = better result (freqtrade convention).
"""

import numpy as np
from pandas import DataFrame

from freqtrade.data.metrics import calculate_expectancy, calculate_max_drawdown
from freqtrade.optimize.hyperopt import IHyperOptLoss


# === HARD GATES ===
MIN_TRADES = 100           # Minimum trades for statistical significance
MAX_DRAWDOWN_RUIN = 0.50   # 50% account drawdown = instant reject
MIN_PROFIT_FACTOR = 1.0    # Must have positive edge

# === SCORING WEIGHTS ===
W_PROFIT = 0.25            # Weight for log-scaled profit
W_PROFIT_FACTOR = 0.20     # Weight for profit factor quality
W_DRAWDOWN = 0.20          # Weight for drawdown penalty
W_WINRATE = 0.10           # Weight for consistency
W_EXPECTANCY = 0.10        # Weight for per-trade expectancy
W_LINEARITY = 0.15         # Weight for equity curve smoothness (R² of cumulative P&L)

# === TUNING ===
DRAWDOWN_STEEPNESS = 4.0   # Exponential penalty steepness (higher = harsher on DD)
IDEAL_TRADES_PER_MONTH = 50  # Trade count bonus saturates here
LARGE_PENALTY = 1e6        # Returned for gated-out results


class SurvivorHyperOptLoss(IHyperOptLoss):
    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date,
        max_date,
        starting_balance: float,
        backtest_stats: dict = None,
        **kwargs,
    ) -> float:
        if len(results) == 0:
            return LARGE_PENALTY

        total_profit = results["profit_abs"].sum()
        profit_ratio = total_profit / starting_balance

        # --- HARD GATE 1: Must be profitable ---
        if total_profit <= 0:
            # Still differentiate among losers (less negative = less bad)
            return LARGE_PENALTY + abs(total_profit)

        # --- HARD GATE 2: Minimum trade count ---
        if trade_count < MIN_TRADES:
            # Scale penalty by how far below minimum
            return LARGE_PENALTY * (1 - trade_count / MIN_TRADES)

        # --- Calculate metrics ---
        winning_profit = results.loc[results["profit_abs"] > 0, "profit_abs"].sum()
        losing_profit = abs(results.loc[results["profit_abs"] < 0, "profit_abs"].sum())
        profit_factor = winning_profit / (losing_profit + 1e-10)

        # --- HARD GATE 3: Profit factor ---
        if profit_factor < MIN_PROFIT_FACTOR:
            return LARGE_PENALTY * (1 / (profit_factor + 0.01))

        # --- Drawdown ---
        try:
            dd = calculate_max_drawdown(
                results, starting_balance=starting_balance, value_col="profit_abs"
            )
            max_drawdown_pct = dd.relative_account_drawdown
        except (ValueError, Exception):
            max_drawdown_pct = 0.0

        # --- HARD GATE 4: Ruin prevention ---
        if max_drawdown_pct > MAX_DRAWDOWN_RUIN:
            return LARGE_PENALTY * (1 + max_drawdown_pct)

        # --- Win rate ---
        wins = len(results[results["profit_abs"] > 0])
        winrate = wins / trade_count

        # --- Expectancy ---
        try:
            _, expectancy_ratio = calculate_expectancy(results)
            expectancy_ratio = min(expectancy_ratio, 10.0)  # cap outliers
        except Exception:
            expectancy_ratio = 0.0

        # --- Duration (days) ---
        duration_days = (max_date - min_date).days or 1

        # --- Equity curve linearity (R² of cumulative P&L) ---
        # Steady upward curve -> R² near 1.0, spiky/lucky -> R² near 0
        cum_profits = results["profit_abs"].cumsum().values
        if len(cum_profits) >= 2:
            x = np.arange(len(cum_profits), dtype=np.float64)
            # Fast R² without scipy: correlation coefficient squared
            x_mean = x.mean()
            y_mean = cum_profits.mean()
            ss_xy = ((x - x_mean) * (cum_profits - y_mean)).sum()
            ss_xx = ((x - x_mean) ** 2).sum()
            ss_yy = ((cum_profits - y_mean) ** 2).sum()
            r_squared = (ss_xy ** 2) / (ss_xx * ss_yy + 1e-10)
        else:
            r_squared = 0.0

        # ============================================
        # SCORING — all components normalized to ~[0, 1]
        # ============================================

        # 1. PROFIT SCORE: log-scaled to prevent chasing 10000% outliers
        #    log(1 + profit_ratio) gives ~0.7 for 100%, ~2.4 for 1000%
        profit_score = np.log1p(profit_ratio)
        # Annualize to compare across different segment lengths
        annual_factor = 365.0 / duration_days
        annualized_profit_score = profit_score * min(annual_factor, 2.0)  # cap at 2x

        # 2. PROFIT FACTOR SCORE: log-scaled, PF=1.5 is good, PF=3 is great
        pf_score = np.log(profit_factor)  # 0 at PF=1, 0.4 at PF=1.5, 1.1 at PF=3

        # 3. DRAWDOWN SCORE: exponential penalty
        #    0% DD -> 1.0, 10% DD -> 0.67, 25% DD -> 0.37, 40% DD -> 0.20
        dd_score = np.exp(-DRAWDOWN_STEEPNESS * max_drawdown_pct)

        # 4. WINRATE SCORE: linear, 80% WR = 0.8
        wr_score = winrate

        # 5. EXPECTANCY SCORE: log-scaled
        exp_score = np.log1p(expectancy_ratio)

        # 6. LINEARITY SCORE: R² of equity curve (0 = random, 1 = perfect line)
        linearity_score = r_squared

        # 7. TRADE COUNT BONUS: more trades = more confidence
        #    Saturates at IDEAL_TRADES_PER_MONTH * months
        ideal_trades = IDEAL_TRADES_PER_MONTH * (duration_days / 30.0)
        trade_confidence = min(trade_count / ideal_trades, 1.0)

        # Combine weighted score
        score = (
            W_PROFIT * annualized_profit_score
            + W_PROFIT_FACTOR * pf_score
            + W_DRAWDOWN * dd_score
            + W_WINRATE * wr_score
            + W_EXPECTANCY * exp_score
            + W_LINEARITY * linearity_score
        ) * trade_confidence

        # Return negative (freqtrade convention: lower = better)
        return -score
