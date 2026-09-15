import logging
import gc
import copy
import numpy as np
import torch as th
import pandas as pd
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Any, Dict, Type, Callable, List
from pandas import DataFrame
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam, Figure
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from freqtrade.strategy import timeframe_to_minutes
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions
from freqtrade.freqai.RL.BaseEnvironment import BaseEnvironment
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel, make_env
from freqtrade.freqai.tensorboard.TensorboardCallback import TensorboardCallback


logger = logging.getLogger(__name__)


class ReforceXModel(BaseReinforcementLearningModel):
    """
    Reinforcement learning prediction model.

    "rl_config": {
        // ReforceX
        "force_exits": false,
        "lr_schedule": false,
        "tensorboard_plot": false,
        "frame_staking": false,
        "frame_staking_n_stack": 2,
        "multiproc": false
    }
    """

    def __init__(self, **kwargs) -> None:
        """
        Model specific config
        """
        super().__init__(**kwargs)

        # Enable action masking for MaskablePPO
        self.use_masking: bool = self.model_type == "MaskablePPO"

        # Enable learning rate linear schedule
        self.lr_schedule: bool = self.rl_config.get('lr_schedule', False)

        # Enable tensorboard logging
        self.activate_tensorboard: bool = self.rl_config.get('activate_tensorboard', True)
        # TENSORBOARD CALLBACK DOES NOT RECOMMENDED TO USE WITH MULTIPLE ENVS,
        # IT WILL RETURN FALSE INFORMATIONS, NEVERTHLESS NOT THREAD SAFE WITH SB3!!!

        # Enable tensorboard rollout plot
        self.tensorboard_plot: bool = self.rl_config.get('tensorboard_plot', False)

        # [Experimental]
        # Demonstration of how to build vectorized environments:
        # Enable frame stacking
        self.frame_staking: bool = self.rl_config.get('frame_staking', False)
        self.frame_staking_n_stack: int = self.rl_config.get('frame_staking_n_stack', 2)
        # OR
        # Enable multiproc
        self.multiproc: bool = self.rl_config.get('multiproc', False)

        self.unset_unsupported()

    def unset_unsupported(self):
        if self.use_masking and self.multiproc:
            logger.warning(
                "User tried to use MaskablePPO with multiproc. "
                "Deactivating multiproc")
            self.multiproc = False

        if self.frame_staking and self.multiproc:
            logger.warning(
                "User tried to use frame_staking with multiproc. "
                "Deactivating multiproc")
            self.multiproc = False

        if self.continual_learning and self.frame_staking:
            logger.warning(
                "User tried to use continual_learning with frame_staking. "
                "Deactivating continual_learning")
            self.continual_learning = False

    def _set_envs(self, train_df: DataFrame, prices_train: DataFrame,
                  test_df: DataFrame, prices_test: DataFrame,
                  env_dict: Dict):
        """
        Set single training and evaluation environments
        """
        self.train_env = self.MyRLEnv(df=train_df, prices=prices_train, **env_dict)
        self.eval_env = Monitor(self.MyRLEnv(df=test_df, prices=prices_test, **env_dict))

    def _set_envs_frame_staking(self, train_df: DataFrame, prices_train: DataFrame,
                                test_df: DataFrame, prices_test: DataFrame,
                                env_dict: Dict, n_stack: int):
        """
        Set dummy vectorized frame stacked training and evaluation environments
        """
        logger.info(f"Frame staking enabled, rank: {self.max_threads}, n_stack: {n_stack}")
        self.train_env = DummyVecEnv([
            make_env(
                self.MyRLEnv, "train_env", i, 42,
                train_df, prices_train,
                env_info=env_dict
            ) for i in range(self.max_threads)
        ])
        self.eval_env = DummyVecEnv([
            make_env(
                self.MyRLEnv, "eval_env", i, 42,
                test_df, prices_test,
                env_info=env_dict
            ) for i in range(self.max_threads)
        ])
        self.train_env = VecMonitor(VecFrameStack(self.train_env, n_stack=n_stack))
        self.eval_env = VecMonitor(VecFrameStack(self.eval_env, n_stack=n_stack))

    def _set_envs_multiproc(self, train_df: DataFrame, prices_train: DataFrame,
                            test_df: DataFrame, prices_test: DataFrame,
                            env_dict: Dict):
        """
        Set vectorized subproc training and evaluation environments
        """
        logger.info(f"Multiproc enabled, rank: {self.max_threads}")
        self.train_env = SubprocVecEnv([
                make_env(
                    self.MyRLEnv, "train_env", i, 42,
                    train_df, prices_train,
                    env_info=env_dict
                ) for i in range(self.max_threads)
        ])
        self.eval_env = SubprocVecEnv([
                make_env(
                    self.MyRLEnv, 'eval_env', i, 42,
                    test_df, prices_test,
                    env_info=env_dict
                ) for i in range(self.max_threads)
        ])
        self.train_env = VecMonitor(self.train_env)
        self.eval_env = VecMonitor(self.eval_env)

    def set_train_and_eval_environments(self, data_dictionary: Dict[str, DataFrame],
                                        prices_train: DataFrame, prices_test: DataFrame,
                                        dk: FreqaiDataKitchen):
        """
        Set training and evaluation environments
        """
        self.close_envs()
        logger.info("Creating environments...")
        env_data = {
            "train_df": data_dictionary["train_features"],
            "prices_train": prices_train,
            "test_df": data_dictionary["test_features"],
            "prices_test": prices_test,
            "env_dict": self.pack_env_dict(dk.pair)
        }
        if self.frame_staking:
            self._set_envs_frame_staking(n_stack=self.frame_staking_n_stack, **env_data)
        elif self.multiproc:
            self._set_envs_multiproc(**env_data)
        else:
            self._set_envs(**env_data)

    def get_model_params(self):
        """
        Get the model specific parameters
        """
        model_params = copy.deepcopy(self.freqai_info["model_training_parameters"])

        if self.lr_schedule:
            _lr = model_params.get('learning_rate', 0.0003)
            model_params["learning_rate"] = linear_schedule(_lr)
            logger.info(f"Learning rate linear schedule enabled, initial value: {_lr}")

        model_params["policy_kwargs"] = dict(
            net_arch=dict(vf=self.net_arch, pi=self.net_arch),
            activation_fn=th.nn.ReLU,
            optimizer_class=th.optim.Adam
        )
        return model_params

    def get_callbacks(self, eval_freq, data_path) -> list:
        """
        Get the model specific callbacks
        """
        callbacks = []
        callbacks.append(MaskableEvalCallback(
            self.eval_env, deterministic=True,
            render=False, eval_freq=eval_freq,
            best_model_save_path=data_path,
            use_masking=self.use_masking
        ))
        if self.activate_tensorboard:
            callbacks.append(CustomTensorboardCallback())
        if self.tensorboard_plot:
            callbacks.append(FigureRecorderCallback())
        return callbacks

    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        """
        User customizable fit method
        :param data_dictionary: dict = common data dictionary containing all train/test
            features/labels/weights.
        :param dk: FreqaiDatakitchen = data kitchen for current pair.
        :return:
        model Any = trained model to be used for inference in dry/live/backtesting
        """
        train_df = data_dictionary["train_features"]
        total_timesteps = int(self.freqai_info["rl_config"]["train_cycles"] * len(train_df))
        tensorboard_log_path = Path(dk.full_path / "tensorboard" / dk.pair.split('/')[0])
        model_params = self.get_model_params()

        logger.info(f"Action masking enabled") if self.use_masking else None
        logger.info(f"Total timesteps: {total_timesteps}")
        logger.info(f"Params: {model_params}")

        if dk.pair not in self.dd.model_dictionary or not self.continual_learning:
            model = self.MODELCLASS(
                self.policy_type,
                self.train_env,
                tensorboard_log=tensorboard_log_path if self.activate_tensorboard else None,
                **model_params
            )
        else:
            logger.info('Continual training activated - starting training from previously '
                        'trained agent.')
            model = self.dd.model_dictionary[dk.pair]
            model.set_env(self.train_env)

        model.learn(
            total_timesteps=total_timesteps,
            callback=self.get_callbacks(len(train_df), str(dk.data_path)),
            progress_bar=self.rl_config.get('progress_bar', False)
        )

        if Path(dk.data_path / "best_model.zip").is_file():
            logger.info('Callback found a best model.')
            best_model = self.MODELCLASS.load(dk.data_path / "best_model")
            return best_model

        logger.info('Couldnt find best model, using final model instead.')

        return model

    def _action_masks_predict(self, market_side: float) -> List[bool]:
        if self.frame_staking or self.multiproc:
            _is_valid_func = self.train_env.envs[0]._is_valid
        else:
            _is_valid_func = self.train_env._is_valid
        return [_is_valid_func(action.value, market_side) for action in Actions]

    def rl_model_predict(self, dataframe: DataFrame, dk: FreqaiDataKitchen, model: Any) -> DataFrame:
        """
        A helper function to make predictions in the Reinforcement learning module.
        :param dataframe: DataFrame = the dataframe of features to make the predictions on
        :param dk: FreqaiDatakitchen = data kitchen for the current pair
        :param model: Any = the trained model used to inference the features.
        """
        output = pd.DataFrame(np.zeros(len(dataframe)), columns=dk.label_list)

        def _predict(window):
            observations = dataframe.iloc[window.index]
            action_masks_param = {}

            if self.live and self.rl_config.get('add_state_info', False):
                market_side, current_profit, trade_duration = self.get_state_info(dk.pair)
                observations['current_profit_pct'] = current_profit
                observations['position'] = market_side
                observations['trade_duration'] = trade_duration

                if self.use_masking:
                    action_masks_param["action_masks"] = self._action_masks_predict(market_side)

            if self.frame_staking:
                observations = observations.to_numpy()
                observations = np.repeat(observations, repeats=self.frame_staking_n_stack, axis=1)

            action, _states = model.predict(observations, deterministic=True, **action_masks_param)
            return action

        output = output.rolling(window=self.CONV_WIDTH).apply(_predict)

        return output

    def close_envs(self):
        if self.train_env:
            self.train_env.close()
        if self.eval_env:
            self.eval_env.close()

    MyRLEnv: Type[BaseEnvironment]

    class MyRLEnv(Base5ActionRLEnv):  # type: ignore[no-redef]
        """
        User can override any function in BaseRLEnv and gym.Env.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # Enable force trade exits by take profit, stop loss and timeout
            self.force_exits: bool = self.rl_config.get('force_exits', False)
            self.take_profit = round(self.profit_aim * self.rr, 3)
            self.stop_loss = round(self.profit_aim * -1, 3)
            self.timeout_candles: int = self.rl_config.get('max_trade_duration_candles', 100)
            self.timeout_minutes: int = (self.timeout_candles *
                                         timeframe_to_minutes(self.config['timeframe']))
            if self.force_exits:
                logger.info(
                    f"Force exits enabled, take_profit: {self.take_profit}, "
                    f"stop_loss: {self.stop_loss}, "
                    f"timeout: {self.timeout_candles} candles ({self.timeout_minutes} min)"
                )

        def action_masks(self) -> list[bool]:
            return [self._is_valid(action.value) for action in self.actions]

        def calculate_reward(self, action):
            """
            An example reward function. This is the one function that users will likely
            wish to inject their own creativity into.

                        Warning!
            This is function is a showcase of functionality designed to show as many possible
            environment control features as possible. It is also designed to run quickly
            on small computers. This is a benchmark, it is *not* for live production.

            :param action: int = The action made by the agent for the current candle.
            :return:
            float = the reward to give to the agent for current step (used for optimization
                of weights in NN)
            """
            # first, penalize if the action is not valid
            if not self._is_valid(action):
                self.tensorboard_log("Invalid", category="actions")
                return -1

            factor = 100.

            # you can use feature values from dataframe
            # rsi_now = self.get_feature_value("rsi", 16)

            if self._position == Positions.Neutral:

                self.tensorboard_log("idle_duration", self.get_idle_duration())

                # Neutral/Idle
                if action == Actions.Neutral.value:
                    self.tensorboard_log("Neutral/Idle", category="actions")
                    # discourage agent from not entering trades
                    return -1

                # Entry
                elif action in (Actions.Long_enter.value, Actions.Short_enter.value):
                    # reward agent for entering trades
                    # factor = 40 / rsi_now if rsi_now < 40 else 1
                    return 25 * factor

            elif self._position != Positions.Neutral:

                pnl = self.get_unrealized_profit()
                # mrr = self.get_most_recent_return()
                # mrp = self.get_most_recent_profit()
                # mpp = self.get_max_possible_profit()

                trade_duration = self.get_trade_duration()
                max_trade_duration = self.rl_config.get('max_trade_duration_candles', 300)

                if trade_duration <= max_trade_duration:
                    factor *= 1.5
                elif trade_duration > max_trade_duration:
                    factor *= 0.5

                # Neutral/Hold
                if action == Actions.Neutral.value:
                    self.tensorboard_log("Neutral/Hold", category="actions")
                    # discourage sitting in position
                    return -1 * trade_duration / max_trade_duration

                # Exit long
                if action == Actions.Long_exit.value and self._position == Positions.Long:
                    if pnl > self.profit_aim:
                        factor *= self.rr
                    return float(pnl * factor)

                # Exit short
                elif action == Actions.Short_exit.value and self._position == Positions.Short:
                    if pnl > self.profit_aim:
                        factor *= self.rr
                    return float(pnl * factor)

            return 0.

        def reset(self, seed=None) -> DataFrame:
            obs, history = super().reset()
            self._last_closed_trade_tick: int = None
            self.max_trades_count: int = (self._end_tick - self._start_tick) / len(Positions)
            self.max_possible_profit = self.get_max_possible_profit(self.can_short)
            self.tensorboard_log("max_possible_profit", self.max_possible_profit)
            self.tensorboard_log("max_trades_count", int(self.max_trades_count))
            return obs, history

        def step(self, action: int):
            """
            Logic for a single step by the agent
            """
            self._current_tick += 1
            self._update_unrealized_total_profit()
            self.tensorboard_log(self.actions._member_names_[action], category="actions")

            reward = self.calculate_reward(action)
            info = self.get_info()

            executed_trade_type = self.execute_trade(action)

            if self.force_exits and executed_trade_type is None:
                executed_trade_type = self.execute_force_exit_trade()

            self.total_reward += reward
            self.update_trade_history(executed_trade_type)
            self._position_history.append(self._position)
            self._update_history(info)
            truncated = False

            return self._get_observation(), reward, self.is_done(), truncated, info

        def execute_trade(self, action) -> str:
            if not self.is_tradesignal(action):
                return None

            executed_trade_type = None
            if action == Actions.Long_enter.value:
                self._position = Positions.Long
                executed_trade_type = "enter_long"
                self._last_trade_tick = self._current_tick
            elif action == Actions.Short_enter.value:
                self._position = Positions.Short
                executed_trade_type = "enter_short"
                self._last_trade_tick = self._current_tick
            elif action == Actions.Long_exit.value:
                self._update_total_profit()
                self._position = Positions.Neutral
                executed_trade_type = "exit_long"
                self._last_closed_trade_tick = self._last_trade_tick
                self._last_trade_tick = None
            elif action == Actions.Short_exit.value:
                self._update_total_profit()
                self._position = Positions.Neutral
                executed_trade_type = "exit_short"
                self._last_closed_trade_tick = self._last_trade_tick
                self._last_trade_tick = None
            return executed_trade_type

        def execute_force_exit_trade(self) -> str:
            if self._position == Positions.Neutral:
                return None

            executed_trade_type = None
            pnl = self.get_unrealized_profit()
            trade_duration = self.get_trade_duration()

            if pnl >= self.take_profit:
                executed_trade_type = "take_profit"
            elif pnl <= self.stop_loss:
                executed_trade_type = "stop_loss"
            elif trade_duration >= self.timeout_candles:
                executed_trade_type = "timeout_profit" if pnl > 0 else "timeout_loss"

            if executed_trade_type:
                self._update_total_profit()
                self._position = Positions.Neutral
                self._last_closed_trade_tick = self._last_trade_tick
                self._last_trade_tick = None
                self.tensorboard_log(executed_trade_type, category="force_exit")
                return executed_trade_type

        def update_trade_history(self, executed_trade_type) -> None:
            if executed_trade_type is not None:
                self.trade_history.append({
                    'tick': self._current_tick,
                    'price': self.current_price(),
                    'type': executed_trade_type,
                    'profit': self.get_unrealized_profit()
                })

        def get_info(self) -> dict:
            return dict(
                tick=self._current_tick,
                total_reward=self.total_reward,
                total_profit=self._total_profit,
                position=self._position.value,
                trade_duration=self.get_trade_duration(),
                current_profit_pct=self.get_unrealized_profit(),
                trades_count=len(self.trade_history)
            )

        def is_done(self) -> bool:
            return True if (
                self._current_tick == self._end_tick or
                self._total_profit <= self.max_drawdown or
                self._total_unrealized_profit <= self.max_drawdown or
                len(self.trade_history) >= self.max_trades_count or
                self._total_profit >= self.max_possible_profit
            ) else False

        def is_tradesignal(self, action: int) -> bool:
            """
            Determine if the action is entry or exit
            """
            if action in (Actions.Short_enter.value, Actions.Long_enter.value):
                if self._position == Positions.Neutral:
                    return True

            elif action == Actions.Short_exit.value:
                if self._position == Positions.Short:
                    return True

            elif action == Actions.Long_exit.value:
                if self._position == Positions.Long:
                    return True

            return False

        def _is_valid(self, action: int, position: int = None) -> bool:
            """
            Determine if the action is valid for the step
            """
            if not position:
                position = self._position.value

            if action in (Actions.Short_enter.value, Actions.Long_enter.value):
                if position != Positions.Neutral.value:
                    return False

            elif action == Actions.Short_exit.value:
                if position != Positions.Short.value:
                    return False

            elif action == Actions.Long_exit.value:
                if position != Positions.Long.value:
                    return False

            return True

        def get_feature_value(self, name: str,
                              period: int = None, shift: int = None, timeframe: str = None,
                              normalized=False) -> float:
            """
            Get the raw or normalized value of the feature on the current tick
            """
            period = f"-period_{period}" if period else ""
            shift = f"_shift-{shift}" if shift else ""
            pair = self.pair.replace(":", "")
            timeframe = self.config["timeframe"] if not timeframe else timeframe

            feature_col = f"%-{name}{period}{shift}_{pair}_{timeframe}"

            if normalized:
                return self.signal_features[feature_col].iloc[self._current_tick]
            return self.raw_features[feature_col].iloc[self._current_tick]

        def get_idle_duration(self) -> int:
            """
            Get idle ticks since the last closed trade tick
            """
            if self._last_closed_trade_tick is None:
                return self._current_tick - self._start_tick
            return self._current_tick - self._last_closed_trade_tick

        def get_most_recent_return(self) -> float:
            """
            Calculate the tick to tick return if in a trade.
            Return is generated from rising prices in Long
            and falling prices in Short positions.
            The actions Sell/Buy or Hold during a Long position trigger the sell/buy-fee.
            """
            if self._position == Positions.Long:
                previous_price = self.previous_price()
                if (self._position_history[self._current_tick - 1] == Positions.Short
                        or self._position_history[self._current_tick - 1] == Positions.Neutral):
                    previous_price = self.add_entry_fee(previous_price)
                return np.log(self.current_price()) - np.log(previous_price)
            elif self._position == Positions.Short:
                previous_price = self.previous_price()
                if (self._position_history[self._current_tick - 1] == Positions.Long
                        or self._position_history[self._current_tick - 1] == Positions.Neutral):
                    previous_price = self.add_exit_fee(previous_price)
                return np.log(previous_price) - np.log(self.current_price())
            return 0.

        def get_most_recent_profit(self):
            """
            Calculate the tick to tick unrealized profit if in a trade
            """
            if self._position == Positions.Long:
                current_price = self.add_exit_fee(self.current_price())
                previous_price = self.add_entry_fee(self.previous_price())
                return (current_price - previous_price) / previous_price
            elif self._position == Positions.Short:
                current_price = self.add_entry_fee(self.current_price())
                previous_price = self.add_exit_fee(self.previous_price())
                return (previous_price - current_price) / previous_price
            return 0

        def get_max_possible_profit(self, can_short: bool = True) -> float:
            """
            The maximum possible profit that an RL agent can obtain
            Modified from https://github.com/AminHP/gym-anytrading
            """
            current_tick = self._start_tick
            last_trade_tick = current_tick - 1
            max_possible_profit = 1.

            while current_tick <= self._end_tick:
                position = None

                if self.prices.iloc[current_tick].open < self.prices.iloc[current_tick - 1].open:
                    while (current_tick <= self._end_tick and
                            self.prices.iloc[current_tick].open < self.prices.iloc[current_tick - 1].open):
                        current_tick += 1
                    position = Positions.Short
                else:
                    while (current_tick <= self._end_tick and
                            self.prices.iloc[current_tick].open >= self.prices.iloc[current_tick - 1].open):
                        current_tick += 1
                    position = Positions.Long

                if position == Positions.Short and can_short:
                    current_price = self.add_entry_fee(self.prices.iloc[current_tick - 1].open)
                    last_trade_price = self.add_exit_fee(self.prices.iloc[last_trade_tick].open)
                    pnl = (last_trade_price - current_price) / last_trade_price
                    max_possible_profit += pnl

                elif position == Positions.Long:
                    current_price = self.add_exit_fee(self.prices.iloc[current_tick - 1].open)
                    last_trade_price = self.add_entry_fee(self.prices.iloc[last_trade_tick].open)
                    pnl = (current_price - last_trade_price) / last_trade_price
                    max_possible_profit += pnl

                last_trade_tick = current_tick - 1

            return max_possible_profit

        def previous_price(self) -> float:
            return self.prices.iloc[self._current_tick - 1].open

        def get_rollout_history(self) -> DataFrame:
            """
            Get environment data from the first to the last trade
            """
            _history_df = pd.DataFrame.from_dict(self.history)
            _trade_history_df = pd.DataFrame.from_dict(self.trade_history)
            _rollout_history = _history_df.merge(_trade_history_df, on="tick", how="left")
            _price_history = self.prices.iloc[_rollout_history.tick].copy().reset_index()

            history = pd.merge(
                _rollout_history,
                _price_history,
                left_index=True, right_index=True
            )
            return history

        def get_rollout_plot(self):
            """
            Plot trades and environment data
            """
            def transform_y_offset(ax, offset):
                return mtransforms.offset_copy(ax.transData, fig=fig, x=0, y=offset, units="inches")

            def plot_markers(ax, ticks, marker, color, size, offset):
                ax.plot(ticks, marker=marker, color=color, markersize=size, fillstyle="full",
                        transform=transform_y_offset(ax, offset), linestyle="none")

            plt.style.use("dark_background")
            fig, axs = plt.subplots(
                nrows=5, ncols=1,
                figsize=(16, 9),
                height_ratios=[6, 1, 1, 1, 1],
                sharex=True
            )

            # Return empty fig if no trades
            if len(self.trade_history) == 0:
                return fig

            history = self.get_rollout_history()
            enter_long_prices = history.loc[history["type"] == "enter_long"]["price"]
            enter_short_prices = history.loc[history["type"] == "enter_short"]["price"]
            exit_long_prices = history.loc[history["type"] == "exit_long"]["price"]
            exit_short_prices = history.loc[history["type"] == "exit_short"]["price"]

            axs[0].plot(history["open"], linewidth=1, color="#c28ce3")
            plot_markers(axs[0], enter_long_prices, "^", "#4ae747", 5, -0.05)
            plot_markers(axs[0], enter_short_prices, "v", "#f53580", 5, 0.05)
            plot_markers(axs[0], exit_long_prices, "o", "#73b3e4", 3, 0)
            plot_markers(axs[0], exit_short_prices, "o", "#73b3e4", 3, 0)

            axs[1].set_ylabel("pnl")
            axs[1].plot(history["current_profit_pct"], linewidth=1, color="#a29db9")
            axs[1].axhline(y=0, label='0', alpha=0.33)
            axs[2].set_ylabel("duration")
            axs[2].plot(history["trade_duration"], linewidth=1, color="#a29db9")
            axs[3].set_ylabel("total_reward")
            axs[3].plot(history["total_reward"], linewidth=1, color="#a29db9")
            axs[3].axhline(y=0, label='0', alpha=0.33)
            axs[4].set_ylabel("total_profit")
            axs[4].set_xlabel("tick")
            axs[4].plot(history["total_profit"], linewidth=1, color="#a29db9")
            axs[4].axhline(y=1, label='1', alpha=0.33)

            for _ax in axs:
                for _border in ["top", "right", "bottom", "left"]:
                    _ax.spines[_border].set_color("#5b5e4b")

            fig.suptitle(
                "Total Reward: %.6f" % self.total_reward + " ~ " +
                "Total Profit: %.6f" % self._total_profit
            )
            fig.tight_layout()

            return fig

        def close(self) -> None:
            gc.collect()
            th.cuda.empty_cache()


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


class CustomTensorboardCallback(TensorboardCallback):
    """
    Tensorboard callback
    """

    def _on_training_start(self) -> None:
        _lr = self.model.learning_rate
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning_rate": _lr if _lr is float else "lr_schedule",
            "gamma": self.model.gamma,
            "gae_lambda": self.model.gae_lambda,
            "n_steps": self.model.n_steps,
            "batch_size": self.model.batch_size,
        }
        metric_dict = {
            "eval/mean_reward": 0,
            "rollout/ep_rew_mean": 0,
            "rollout/ep_len_mean": 0,
            "info/total_profit": 1,
            "info/trades_count": 0,
            "info/trade_duration": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:

        local_info = self.locals["infos"][0]
        if self.training_env is None:
            return True
        tensorboard_metrics = self.training_env.get_attr("tensorboard_metrics")[0]

        for metric in local_info:
            if metric not in ["episode", "terminal_observation", "TimeLimit.truncated"]:
                self.logger.record(f"info/{metric}", local_info[metric])

        for category in tensorboard_metrics:
            for metric in tensorboard_metrics[category]:
                self.logger.record(f"{category}/{metric}", tensorboard_metrics[category][metric])

        return True


class FigureRecorderCallback(BaseCallback):
    """
    Tensorboard figures callback
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        try:
            figures = [env.get_rollout_plot() for env in self.training_env.envs]
        except AttributeError:
            figures = self.training_env.env_method("get_rollout_plot")

        for i, fig in enumerate(figures):
            self.logger.record(
                f"rollout/env_{i}",
                Figure(fig, close=True),
                exclude=("stdout", "log", "json", "csv")
            )
            plt.close(fig)
        return True
