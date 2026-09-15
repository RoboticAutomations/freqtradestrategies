# freqtrade/user_data/freqaimodels/MyRLLearner.py
from __future__ import annotations

from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions


class MyRLLearner(ReinforcementLearner):
    """
    Minimal, rule-driven RL learner.

    Rules implemented:
      1) Invalid actions → strong penalty
      2) Opening a trade → small reward
      3) Staying neutral for too long → small (increasing) penalty
      4) Holding a trade for too long → increasing penalty
      5) Closing a profitable trade → big reward (scaled to profit)
      6) Being in a trade → small time-decay penalty
      7) Everything else → neutral (0)
    """

    class MyRLEnv(Base5ActionRLEnv):
        # ---- helper: read knobs from rl_config with defaults ----
        def _rp(self, key: str, default: float) -> float:
            rp = (self.rl_config or {}).get("model_reward_parameters") or {}
            try:
                return float(rp.get(key, default))
            except Exception:
                return default

        def calculate_reward(self, action: int) -> float:
            # Ensure per-episode counters exist
            if not hasattr(self, "_neutral_streak"):
                self._neutral_streak = 0

            # 0) Common context
            pnl = self.get_unrealized_profit()
            max_trade_duration = int((self.rl_config or {}).get("max_trade_duration_candles", 300))
            trade_duration = self._current_tick - self._last_trade_tick if self._last_trade_tick is not None else 0
            frac = min(1.0, max(0.0, trade_duration / max_trade_duration)) if max_trade_duration > 0 else 1.0

            # Knobs (tweak in config.freqai.rl_config.model_reward_parameters)
            invalid_action_penalty = self._rp("invalid_action_penalty", -2.0)
            open_reward             = self._rp("open_reward", 0.25)
            neutral_penalty_base    = self._rp("neutral_penalty_base", -0.05)
            inpos_decay_penalty     = self._rp("inpos_decay_penalty", -0.001)  # per step, scaled by time-in-trade
            hold_penalty_max        = self._rp("hold_penalty_max", -0.5)       # cap for “too long” hold penalty
            profit_scale            = self._rp("profit_scale", 100.0)          # scales pnl when closing
            win_reward_factor       = self._rp("win_reward_factor", 2.0)       # extra boost on “good wins”
            profit_aim              = float(getattr(self, "profit_aim", 0.01))
            rr                      = float(getattr(self, "rr", 1.0))

            # 1) Invalid actions → strong penalty
            if not self._is_valid(action):
                # reset neutral streak on any non-neutral step
                if action != Actions.Neutral.value:
                    self._neutral_streak = 0
                return float(invalid_action_penalty)

            # 2) Opening a trade → small reward
            if self._position == Positions.Neutral and action in (Actions.Long_enter.value, Actions.Short_enter.value):
                self._neutral_streak = 0
                return float(open_reward)

            # 3) Staying neutral for too long → small (increasing) penalty
            if self._position == Positions.Neutral and action == Actions.Neutral.value:
                # increase penalty with consecutive neutral ticks (softly capped)
                self._neutral_streak += 1
                scale = min(3.0, 1.0 + self._neutral_streak / 50.0)  # grows slowly, capped x3
                return float(neutral_penalty_base * scale)

            # 6) Being in a trade → small time-decay penalty (when doing nothing)
            if self._position in (Positions.Long, Positions.Short) and action == Actions.Neutral.value:
                # proportional to time spent in trade
                time_decay = inpos_decay_penalty * (1.0 + frac)  # slightly stronger as time goes on

                # 4) Holding a trade for too long → increasing penalty
                too_long_pen = 0.0
                if max_trade_duration > 0 and trade_duration > max_trade_duration:
                    # ramp from 0 to |hold_penalty_max| as duration doubles the threshold
                    over = trade_duration - max_trade_duration
                    ramp = min(1.0, over / max_trade_duration)
                    too_long_pen = hold_penalty_max * ramp

                self._neutral_streak = 0
                return float(time_decay + too_long_pen)

            # 5) Closing a profitable trade → big reward (scaled to profit)
            if action == Actions.Long_exit.value and self._position == Positions.Long:
                self._neutral_streak = 0
                reward = pnl * profit_scale
                if pnl > profit_aim * rr:
                    reward *= win_reward_factor
                return float(reward)

            if action == Actions.Short_exit.value and self._position == Positions.Short:
                self._neutral_streak = 0
                reward = pnl * profit_scale
                if pnl > profit_aim * rr:
                    reward *= win_reward_factor
                return float(reward)

            # 7) Everything else → neutral (0)
            self._neutral_streak = 0
            return 0.0
