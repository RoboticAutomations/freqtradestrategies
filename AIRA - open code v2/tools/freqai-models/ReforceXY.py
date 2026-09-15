import copy
import gc
import json
import logging
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import torch as th
from gymnasium import Env
from gymnasium.spaces import Box
from optuna import Trial, TrialPruned, create_study
from optuna.exceptions import ExperimentalWarning
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.study import Study, StudyDirection
from pandas import DataFrame, concat, merge
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import (
    BaseCallback,
    ProgressBarCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import Figure, HParam
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions
from freqtrade.freqai.RL.BaseEnvironment import BaseEnvironment
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel
from freqtrade.freqai.tensorboard.TensorboardCallback import TensorboardCallback
from freqtrade.strategy import timeframe_to_minutes


matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ExperimentalWarning)
logger = logging.getLogger(__name__)


class ForceActions(Enum):
    Take_profit = 1
    Stop_loss = 2
    Timeout = 3


class ReforceXY(BaseReinforcementLearningModel):
    """
    Custom Freqtrade Freqai reinforcement learning prediction model.
    Model specific config:
    {
        "freqaimodel": "ReforceXY",
        "strategy": "RLAgentStrategy",
        "minimal_roi": {"0": 0.03},                 // Take_profit exit value used with force_actions
        "stoploss": -0.03,                          // Stop_loss exit value used with force_actions
        ...
        "freqai": {
            ...
            "rl_config": {
                ...
                "max_trade_duration_candles": 96,   // Timeout exit value used with force_actions
                "force_actions": false,             // Utilize minimal_roi, stoploss, and max_trade_duration_candles as TP/SL/Timeout in the environment
                "n_envs": 1,                        // Number of DummyVecEnv environments
                "frame_staking": 0,                 // Number of VecFrameStack stacks (set > 1 to use)
                "lr_schedule": false,               // Enable learning rate linear schedule
                "cr_schedule": false,               // Enable clip range linear schedule
                "max_no_improvement_evals": 0,      // Maximum consecutive evaluations without a new best model
                "min_evals": 0,                     // Number of evaluations before start to count evaluations without improvements
                "check_envs": true,                 // Check that an environment follows Gym API
                "plot_new_best": false,             // Enable tensorboard rollout plot upon finding a new best model
            },
            "rl_config_optuna": {
                "enabled": false,                   // Enable optuna hyperopt
                "n_trials": 100,
                "n_startup_trials": 10,
                "timeout_hours": 0,
            }
        }
    }
    Requirements:
        - pip install optuna

    Optional:
        - pip install optuna-dashboard
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_maskable: bool = self.model_type == "MaskablePPO" # Enable action masking
        self.lr_schedule: bool = self.rl_config.get("lr_schedule", False)
        self.cr_schedule: bool = self.rl_config.get("cr_schedule", False)
        self.n_envs: int = self.rl_config.get("n_envs", 1)
        self.frame_staking: int = self.rl_config.get("frame_staking", 0)
        self.frame_staking += 1 if self.frame_staking == 1 else 0
        self.max_no_improvement_evals: int = self.rl_config.get("max_no_improvement_evals", 0)
        self.min_evals: int = self.rl_config.get("min_evals", 0)
        self.plot_new_best: bool = self.rl_config.get("plot_new_best", False)
        self.check_envs: bool = self.rl_config.get("check_envs", True)
        self.progressbar_callback: Optional[ProgressBarCallback] = None
        # Optuna hyperopt hyperopt
        self.rl_config_optuna: dict = self.freqai_info.get("rl_config_optuna", {})
        self.hyperopt: bool = self.rl_config_optuna.get("enabled", False)
        self.optuna_timeout_hours: float = self.rl_config_optuna.get("timeout_hours", 0)
        self.optuna_n_trials: int = self.rl_config_optuna.get("n_trials", 100)
        self.optuna_n_startup_trials: int = self.rl_config_optuna.get("n_startup_trials", 10)
        self.optuna_trial_params: list = []
        self.optuna_callback: Optional[MaskableTrialEvalCallback] = None
        self.unset_unsupported()

    def unset_unsupported(self) -> None:
        """
        If user has activated any custom function that may conflict, this
        function will set them to false and warn them
        """
        if self.continual_learning and self.frame_staking:
            logger.warning(
                "User tried to use continual_learning with frame_staking. \
                Deactivating continual_learning"
            )
            self.continual_learning = False        

    def set_train_and_eval_environments(
        self,
        data_dictionary: Dict[str, DataFrame],
        prices_train: DataFrame,
        prices_test: DataFrame,
        dk: FreqaiDataKitchen,
    ) -> None:
        """
        Set training and evaluation environments
        """
        self.close_envs()
        
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]
        env_dict = self.pack_env_dict(dk.pair)
        seed = self.freqai_info["model_training_parameters"].get("seed", 42)

        if self.check_envs:
            logger.info("Checking environments...")
            check_env(self.MyRLEnv(id="train_env_check", df=train_df, prices=prices_train, **env_dict))
            check_env(self.MyRLEnv(id="eval_env_check", df=test_df, prices=prices_test, **env_dict))

        logger.info("Populating environments: %s", self.n_envs)
        train_env = DummyVecEnv(
            [make_env(
                self.MyRLEnv, "train_env", i, seed,
                train_df, prices_train, env_info=env_dict
            ) for i in range(self.n_envs)]
        )
        eval_env = DummyVecEnv(
            [make_env(
                self.MyRLEnv, "eval_env", i, seed,
                test_df, prices_test, env_info=env_dict
            ) for i in range(self.n_envs)]
        )
        if self.frame_staking:
            logger.info("Frame staking: %s", self.frame_staking)
            train_env = VecFrameStack(train_env, n_stack=self.frame_staking)
            eval_env = VecFrameStack(eval_env, n_stack=self.frame_staking)

        self.train_env = VecMonitor(train_env)
        self.eval_env = VecMonitor(eval_env)


    def get_model_params(self) -> Dict:
        """
        Get model parameters
        """
        model_params: Dict = copy.deepcopy(self.freqai_info["model_training_parameters"])

        if self.lr_schedule:
            _lr = model_params.get("learning_rate", 0.0003)
            model_params["learning_rate"] = linear_schedule(_lr)
            logger.info("Learning rate linear schedule enabled, initial value: %s", _lr)

        if self.cr_schedule:
            _cr = model_params.get("clip_range", 0.2)
            model_params["clip_range"] = linear_schedule(_cr)
            logger.info("Clip range linear schedule enabled, initial value: %s", _cr)

        model_params["policy_kwargs"] = {
            "net_arch": self.net_arch,
            "activation_fn": th.nn.ReLU,
            "optimizer_class": th.optim.Adam,
            # "ortho_init": True
        }
        if "PPO" in self.model_type:
            model_params["policy_kwargs"]["net_arch"] = {"pi": self.net_arch, "vf": self.net_arch}

        return model_params

    def get_callbacks(self, eval_freq, data_path, trial: Trial = None) -> list:
        """
        Get the model specific callbacks
        """
        callbacks: list[BaseCallback] = []
        no_improvement_callback = None
        rollout_plot_callback = None
        verbose = self.freqai_info["model_training_parameters"].get("verbose", 0)

        if self.n_envs > 1:
            eval_freq //= self.n_envs

        if self.plot_new_best:
            rollout_plot_callback = RolloutPlotCallback(verbose=verbose)

        if self.max_no_improvement_evals:
            no_improvement_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=self.max_no_improvement_evals,
                min_evals=self.min_evals,
                verbose=1,
            )
        
        if self.activate_tensorboard:
            info_callback = InfoMetricsCallback(actions=Actions, verbose=verbose)
            callbacks.append(info_callback)

        if self.rl_config.get("progress_bar", False):
            self.progressbar_callback = ProgressBarCallback()
            callbacks.append(self.progressbar_callback)

        self.eval_callback = MaskableEvalCallback(
            self.eval_env,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            best_model_save_path=str(data_path),
            use_masking=self.is_maskable,
            callback_on_new_best=rollout_plot_callback,
            callback_after_eval=no_improvement_callback,
            verbose=verbose,
        )
        callbacks.append(self.eval_callback)

        if not trial:
            return callbacks

        self.optuna_callback = MaskableTrialEvalCallback(
            self.eval_env,
            trial,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            best_model_save_path=data_path,
            use_masking=self.is_maskable,
            verbose=verbose,
        )
        callbacks.append(self.optuna_callback)
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
        train_timesteps = len(train_df)
        test_timestaps = len(data_dictionary["test_features"])
        train_cycles = int(self.freqai_info["rl_config"]["train_cycles"])
        total_timesteps = train_timesteps * train_cycles
        train_days = steps_to_days(train_timesteps, self.config["timeframe"])
        total_days = steps_to_days(total_timesteps, self.config["timeframe"])

        logger.info("Action masking: %s", self.is_maskable)
        logger.info("Train: %s steps (%s days) * %s cycles = Total %s (%s days)",
                    train_timesteps, train_days, train_cycles, total_timesteps, total_days
        )
        logger.info("Test: %s steps (%s days)", test_timestaps, steps_to_days(test_timestaps, self.config["timeframe"]))
        logger.info("Hyperopt: %s", self.hyperopt)

        if self.hyperopt:
            model_params = self.study(train_df, total_timesteps, dk)
        else:
            model_params = self.get_model_params()
        logger.info("%s params: %s", self.model_type, model_params)

        if self.activate_tensorboard:
            tb_path = Path(dk.full_path / "tensorboard" / dk.pair.split("/")[0])
        else:
            tb_path = None

        if dk.pair not in self.dd.model_dictionary or not self.continual_learning:
            model = self.MODELCLASS(
                self.policy_type, self.train_env, tensorboard_log=tb_path, **model_params
            )
        else:
            logger.info(
                "Continual training activated - starting training from previously trained agent."
            )
            model = self.dd.model_dictionary[dk.pair]
            model.set_env(self.train_env)

        callbacks = self.get_callbacks(train_timesteps, str(dk.data_path))
        try:
            model.learn(total_timesteps=total_timesteps, callback=callbacks)
        finally:
            if self.progressbar_callback:
                self.progressbar_callback.on_training_end()

        if Path(dk.data_path / "best_model.zip").is_file():
            logger.info("Callback found a best model.")
            best_model = self.MODELCLASS.load(dk.data_path / "best_model")
            return best_model

        logger.info("Couldnt find best model, using final model instead.")

        return model
    
    def get_state_info(self, pair: str) -> Tuple[float, float, int]:
        """
        State info during dry/live (not backtesting) which is fed back
        into the model.
        :param pair: str = COIN/STAKE to get the environment information for
        :return:
        :market_side: float = representing short, long, or neutral for
            pair
        :current_profit: float = unrealized profit of the current trade
        :trade_duration: int = the number of candles that the trade has
            been open for
        """
        #STATE_INFO
        position, pnl, trade_duration = super().get_state_info(pair)
        return position, pnl, int(trade_duration)

    def rl_model_predict(self, dataframe: DataFrame, dk: FreqaiDataKitchen, model: Any) -> DataFrame:
        """
        A helper function to make predictions in the Reinforcement learning module.
        :param dataframe: DataFrame = the dataframe of features to make the predictions on
        :param dk: FreqaiDatakitchen = data kitchen for the current pair
        :param model: Any = the trained model used to inference the features.
        """

        def _is_valid(action: int, position: float) -> bool:
            return not (
                (
                    action in (Actions.Short_enter.value, Actions.Long_enter.value)
                    and position != Positions.Neutral.value
                )
                or (action == Actions.Long_exit.value and position != Positions.Long.value)
                or (action == Actions.Short_exit.value and position != Positions.Short.value)
            )

        def _action_masks(position: float):
            return [_is_valid(action.value, position) for action in Actions]

        def _predict(window):
            observations: DataFrame = dataframe.iloc[window.index]
            action_masks_param: dict = {}

            if self.live and self.rl_config.get("add_state_info", False):

                position, pnl, trade_duration = self.get_state_info(dk.pair)
                observations["pnl"] = pnl
                observations["position"] = position
                observations["trade_duration"] = trade_duration
                #STATE_INFO

                if self.is_maskable:
                    action_masks_param = {"action_masks": _action_masks(position)}

            observations = observations.to_numpy(dtype=np.float32)

            if self.frame_staking:
                observations = np.repeat(observations, axis=1, repeats=self.frame_staking)

            action, _ = model.predict(observations, deterministic=True, **action_masks_param)
            return action

        output = DataFrame(np.zeros(len(dataframe)), columns=dk.label_list)
        output = output.rolling(window=self.CONV_WIDTH).apply(_predict)
        return output

    def study(self, train_df, total_timesteps: int, dk: FreqaiDataKitchen) -> Dict:
        """
        Runs hyperparameter optimization using Optuna and
        returns the best hyperparameters found
        """
        storage_dir, study_name = str(dk.full_path).rsplit("/", 1)
        study: Study = create_study(
            study_name=study_name,
            sampler=TPESampler(
                n_startup_trials=self.optuna_n_startup_trials,
                multivariate=True, group=True
            ),
            pruner=HyperbandPruner(
                min_resource=1, max_resource=self.optuna_n_trials,
                reduction_factor=3
            ),
            direction=StudyDirection.MAXIMIZE,
            storage=f"sqlite:///{storage_dir}/optuna.sqlite",
            load_if_exists=True,
        )
        try:
            study.optimize(
                lambda trial: self.objective(trial, train_df, total_timesteps, dk),
                n_trials=self.optuna_n_trials,
                timeout=(
                    hours_to_seconds(self.optuna_timeout_hours)
                    if self.optuna_timeout_hours
                    else None
                ),
                gc_after_trial=True,
                show_progress_bar=self.rl_config.get("progress_bar", False),
                n_jobs=1,
            )
        except KeyboardInterrupt:
            pass

        logger.info("------------ Hyperopt results ------------")
        logger.info("Best trial: %s. Score: %s", study.best_trial.number, study.best_trial.value)
        logger.info("Best trial params: %s", self.optuna_trial_params[study.best_trial.number])
        logger.info("-----------------------------------------")

        best_trial_path = Path(dk.full_path / "hyperopt_best_trial.json")
        logger.info("dumping json to %s", best_trial_path)
        with best_trial_path.open("w", encoding="utf-8") as write_file:
            json.dump(study.best_trial.params, write_file, indent=4)

        return self.optuna_trial_params[study.best_trial.number]

    def objective(self, trial: Trial, train_df, total_timesteps: int, dk: FreqaiDataKitchen) -> float:
        """
        Defines a single trial for hyperparameter optimization using Optuna
        """
        if "PPO" in self.model_type:
            params = sample_params_ppo(trial)
        elif "QRDQN" in self.model_type:
            params = sample_params_qrdqn(trial)
        elif "DQN" in self.model_type:
            params = sample_params_dqn(trial)
        else:
            raise NotImplementedError

        params.update(self.freqai_info["model_training_parameters"])

        nan_encountered = False
        tensorboard_log_path = Path(dk.full_path / "tensorboard" / dk.pair.split("/")[0])

        logger.info("------------ Hyperopt ------------")
        logger.info("------------ Trial %s ------------", trial.number)
        logger.info("Trial %s params: %s", trial.number, params)
        self.optuna_trial_params.append(params)

        model = self.MODELCLASS(
            self.policy_type, self.train_env,
            tensorboard_log=tensorboard_log_path, **params
        )
        callbacks = self.get_callbacks(len(train_df), str(dk.data_path), trial)

        try:
            model.learn(total_timesteps=total_timesteps, callback=callbacks)
        except AssertionError:
            logger.warning("Optuna encountered NaN")
            nan_encountered = True
        finally:
            if self.progressbar_callback:
                self.progressbar_callback.on_training_end()
            self.close_envs()
            model.env.close()

        if nan_encountered:
            return float("nan")

        if self.optuna_callback.is_pruned:
            raise TrialPruned()

        return self.optuna_callback.best_mean_reward

    def close_envs(self):
        """
        Closes the training and evaluation environments if they are open
        """
        if self.train_env:
            self.train_env.close()
        if self.eval_env:
            self.eval_env.close()

    class MyRLEnv(Base5ActionRLEnv):
        """
        Env
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.force_actions: bool = self.rl_config.get("force_actions", False)
            self._force_action: ForceActions = None
            self.take_profit: float = self.config["minimal_roi"]["0"]
            self.stop_loss: float = self.config["stoploss"]
            self.timeout: int = self.rl_config.get("max_trade_duration_candles", 128)
            self._last_closed_position: Positions = None
            self._last_closed_trade_tick: int = 0
            # self.reward_range = (-1, 1)
            if self.force_actions:
                logger.info(
                    "%s - take_profit: %s, stop_loss: %s, timeout: %s candles (%s days), observation_space: %s",
                    self.id, self.take_profit, self.stop_loss, self.timeout,
                    steps_to_days(self.timeout, self.config["timeframe"]), self.observation_space
                )

        def reset_env(
            self,
            df: DataFrame,
            prices: DataFrame,
            window_size: int,
            reward_kwargs: dict,
            starting_point=True,
        ):
            """
            Resets the environment when the agent fails
            """
            super().reset_env(df, prices, window_size, reward_kwargs, starting_point)
            self.state_features = ["pnl", "position", "trade_duration"] #STATE_INFO
            if self.add_state_info:
                self.total_features = self.signal_features.shape[1] + len(self.state_features)
                self.shape = (window_size, self.total_features)

            self.observation_space = Box(
                low=-np.inf,
                high=np.inf,
                shape=self.shape, dtype=np.float32
            )

        def reset(self, seed=None, **kwargs):
            """
            Reset is called at the beginning of every episode
            """
            _, history = super().reset(seed, **kwargs)
            self._force_action: ForceActions = None
            self._last_closed_position: Positions = None
            self._last_closed_trade_tick: int = 0
            return self._get_observation(), history

        def calculate_reward(self, action) -> float:
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
            pnl = self.get_unrealized_profit()
            # mrr = self.get_most_recent_return()
            # mrp = self.get_most_recent_profit()

            factor = 100.

            # Force exits
            if self._force_action in (
                ForceActions.Take_profit,
                ForceActions.Stop_loss,
                ForceActions.Timeout,
            ):
                return pnl * factor

            # first, penalize if the action is not valid
            if not self._is_valid(action):
                return -2

            # # you can use feature values from dataframe
            # rsi_now = self.get_feature_value(
            #     name="%-rsi",
            #     period=8,
            #     pair=self.pair,
            #     timeframe=self.config["timeframe"],
            #     raw=True
            # )

            # # reward agent for entering trades
            # if (action in (Actions.Long_enter.value, Actions.Short_enter.value)
            #         and self._position == Positions.Neutral):
            #     if rsi_now < 40:
            #         factor = 40 / rsi_now
            #     else:
            #         factor = 1
            #     return 25 * factor

            # discourage agent from not entering trades
            if action == Actions.Neutral.value and self._position == Positions.Neutral:
                return -1

            max_trade_duration = self.rl_config.get('max_trade_duration_candles', 300)
            trade_duration = self.get_trade_duration()

            if trade_duration <= max_trade_duration:
                factor *= 1.5
            elif trade_duration > max_trade_duration:
                factor *= 0.5

            # discourage sitting in position
            if (self._position in (Positions.Short, Positions.Long) and
               action == Actions.Neutral.value):
                return -1 * trade_duration / max_trade_duration

            # close long
            if action == Actions.Long_exit.value and self._position == Positions.Long:
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
                return float(pnl * factor)

            # close short
            if action == Actions.Short_exit.value and self._position == Positions.Short:
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
                return float(pnl * factor)

            return 0.

        def _get_observation(self):
            """
            This may or may not be independent of action types, user can inherit
            this in their custom "MyRLEnv"
            """
            features_window = self.signal_features[
                (self._current_tick - self.window_size) : self._current_tick
            ]
            if self.add_state_info:
                features_and_state = DataFrame(
                    np.zeros((len(features_window), len(self.state_features)), dtype=np.float32),
                    columns=self.state_features,
                    index=features_window.index,
                )
                features_and_state["pnl"] = self.get_unrealized_profit()
                features_and_state["position"] = self._position.value
                features_and_state["trade_duration"] = self.get_trade_duration()
                #STATE_INFO

                features_and_state = concat([features_window, features_and_state], axis=1)
                return features_and_state.to_numpy(dtype=np.float32)
            else:
                return features_window.to_numpy(dtype=np.float32)

        def _get_force_action(self):
            if not self.force_actions or self._position == Positions.Neutral:
                return None

            trade_duration = self.get_trade_duration()
            if trade_duration <= 1:
                return None
            if trade_duration >= self.timeout:
                return ForceActions.Timeout

            pnl = self.get_unrealized_profit()
            if pnl >= self.take_profit:
                return ForceActions.Take_profit
            if pnl <= self.stop_loss:
                return ForceActions.Stop_loss

        def _get_new_position(self, action: int) -> Positions:
            return {
                Actions.Long_enter.value: Positions.Long,
                Actions.Short_enter.value: Positions.Short,
            }[action]

        def _enter_trade(self, action):
            self._position = self._get_new_position(action)
            self._last_trade_tick = self._current_tick

        def _exit_trade(self):
            self._update_total_profit()
            self._last_closed_position = self._position
            self._position = Positions.Neutral
            self._last_closed_trade_tick = self._last_trade_tick
            self._last_trade_tick = None

        def execute_trade(self, action: int) -> None:
            """
            Execute trade based on the given action
            """
            # Force exit trade
            if self._force_action:
                self._exit_trade()  # Exit trade due to force action
                self.append_trade_history(f"{self._force_action.name}")
                self.tensorboard_log(f"{self._force_action.name}", category="actions/force")
                return None

            if not self.is_tradesignal(action):
                return None

            # Enter trade based on action
            if action in (Actions.Long_enter.value, Actions.Short_enter.value):
                self._enter_trade(action)
                self.append_trade_history(f"{self._position.name}_enter")

            # Exit trade based on action
            if action in (Actions.Long_exit.value, Actions.Short_exit.value):
                self._exit_trade()
                self.append_trade_history(f"{self._last_closed_position.name}_exit")

        def step(self, action: int):
            """
            Take a step in the environment based on the provided action
            """
            self.tensorboard_log(Actions._member_names_[action], category="actions")
            self._current_tick += 1
            self._update_unrealized_total_profit()
            self._force_action = self._get_force_action()
            reward = self.calculate_reward(action)
            self.total_reward += reward
            info = {
                "tick": self._current_tick,
                "position": self._position.value,
                "action": action,
                "force_action": self._get_force_action(),
                "pnl": self.get_unrealized_profit(),
                "reward": round(reward, 5),
                "total_reward": round(self.total_reward, 5),
                "total_profit": round(self._total_profit, 5),
                "idle_duration": self.get_idle_duration(),
                "trade_duration": self.get_trade_duration(),
                "trade_count": len(self.trade_history),
            }
            self.execute_trade(action)
            self._position_history.append(self._position)
            self._update_history(info)
            return (
                self._get_observation(),
                reward,
                self.is_terminated(),
                self.is_truncated(),
                info,
            )

        def append_trade_history(self, trade_type: str):
            self.trade_history.append(
                {
                    "tick": self._current_tick,
                    "type": trade_type.lower(),
                    "price": self.current_price(),
                    "profit": self.get_unrealized_profit(),
                }
            )

        def is_terminated(self) -> bool:
            return bool(
                self._current_tick == self._end_tick
                or self._total_profit <= self.max_drawdown
                or self._total_unrealized_profit <= self.max_drawdown
            )

        def is_truncated(self) -> bool:
            return False

        def is_tradesignal(self, action: int) -> bool:
            """
            Determine if the action is entry or exit
            """
            return (
                (
                    action in (Actions.Short_enter.value, Actions.Long_enter.value)
                    and self._position == Positions.Neutral
                )
                or (action == Actions.Long_exit.value and self._position == Positions.Long)
                or (action == Actions.Short_exit.value and self._position == Positions.Short)
            )

        def _is_valid(self, action: int) -> bool:
            """
            Determine if the action is valid for the step
            """
            return not (
                (
                    action in (Actions.Short_enter.value, Actions.Long_enter.value)
                    and self._position != Positions.Neutral
                )
                or (action == Actions.Long_exit.value and self._position != Positions.Long)
                or (action == Actions.Short_exit.value and self._position != Positions.Short)
            )

        def get_feature_value(
                self, name: str, period: int = 0, shift: int = 0,
                pair: str = "", timeframe: str = "", raw: bool = True
            ) -> float:
            """
            Get feature value
            """
            feature_parts = [name]
            if period:
                feature_parts.append(f"_{period}")
            if shift:
                feature_parts.append(f"_shift-{shift}")
            if pair:
                feature_parts.append(f"_{pair.replace(':', '')}")
            if timeframe:
                feature_parts.append(f"_{timeframe}")
            feature_col = "".join(feature_parts)

            if not raw:
                return self.signal_features[feature_col].iloc[self._current_tick]
            return self.raw_features[feature_col].iloc[self._current_tick]

        def get_idle_duration(self) -> int:
            if self._position != Positions.Neutral:
                return 0
            if not self._last_closed_trade_tick:
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
                current_price = self.prices.iloc[self._current_tick].open
                previous_price = self.prices.iloc[self._current_tick - 1].open
                if (
                    self._position_history[self._current_tick - 1] == Positions.Short
                    or self._position_history[self._current_tick - 1] == Positions.Neutral
                ):
                    previous_price = self.add_entry_fee(previous_price)
                return np.log(current_price) - np.log(previous_price)
            if self._position == Positions.Short:
                current_price = self.prices.iloc[self._current_tick].open
                previous_price = self.prices.iloc[self._current_tick - 1].open
                if (
                    self._position_history[self._current_tick - 1] == Positions.Long
                    or self._position_history[self._current_tick - 1] == Positions.Neutral
                ):
                    previous_price = self.add_exit_fee(previous_price)
                return np.log(previous_price) - np.log(current_price)
            return 0.0

        def get_most_recent_profit(self) -> float:
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
            return 0.0

        def previous_price(self) -> float:
            return self.prices.iloc[self._current_tick - 1].open

        def get_env_history(self) -> DataFrame:
            """
            Get environment data from the first to the last trade
            """
            _history_df = DataFrame.from_dict(self.history)
            _trade_history_df = DataFrame.from_dict(self.trade_history)
            _rollout_history = _history_df.merge(_trade_history_df, on="tick", how="left")
            _price_history = self.prices.iloc[_rollout_history.tick].copy().reset_index()
            history = merge(_rollout_history, _price_history, left_index=True, right_index=True)
            return history

        def get_env_plot(self):
            """
            Plot trades and environment data
            """

            def transform_y_offset(ax, offset):
                return mtransforms.offset_copy(ax.transData, fig=fig, y=offset)

            def plot_markers(ax, ticks, marker, color, size, offset):
                ax.plot(
                    ticks,
                    marker=marker,
                    color=color,
                    markersize=size,
                    fillstyle="full",
                    transform=transform_y_offset(ax, offset),
                    linestyle="none",
                )

            plt.style.use("dark_background")
            fig, axs = plt.subplots(
                nrows=5, ncols=1, figsize=(14, 8), height_ratios=[5, 1, 1, 1, 1], sharex=True
            )

            # Return empty fig if no trades
            if len(self.trade_history) == 0:
                return fig

            history = self.get_env_history()

            enter_long_prices = history.loc[history["type"] == "long_enter"]["price"]
            enter_short_prices = history.loc[history["type"] == "short_enter"]["price"]
            exit_long_prices = history.loc[history["type"] == "long_exit"]["price"]
            exit_short_prices = history.loc[history["type"] == "short_exit"]["price"]
            take_profit_prices = history.loc[history["type"] == "take_profit"]["price"]
            stop_loss_prices = history.loc[history["type"] == "stop_loss"]["price"]
            timeout_prices = history.loc[history["type"] == "timeout"]["price"]
            axs[0].plot(history["open"], linewidth=1, color="orchid")

            plot_markers(axs[0], enter_long_prices, "^", "forestgreen", 5, -0.1)
            plot_markers(axs[0], enter_short_prices, "v", "firebrick", 5, 0.1)
            plot_markers(axs[0], exit_long_prices, ".", "cornflowerblue", 4, 0.1)
            plot_markers(axs[0], exit_short_prices, ".", "thistle", 4, -0.1)
            plot_markers(axs[0], take_profit_prices, "*", "lime", 8, 0.1)
            plot_markers(axs[0], stop_loss_prices, "x", "red", 8, -0.1)
            plot_markers(axs[0], timeout_prices, "1", "yellow", 8, 0)

            axs[1].set_ylabel("pnl")
            axs[1].plot(history["pnl"], linewidth=1, color="gray")
            axs[1].axhline(y=0, label="0", alpha=0.33, color="gray")
            axs[1].axhline(y=self.take_profit, label="tp", alpha=0.33, color="green")
            axs[1].axhline(y=self.stop_loss, label="sl", alpha=0.33, color="red")

            axs[2].set_ylabel("reward")
            axs[2].plot(history["reward"], linewidth=1, color="gray")
            axs[2].axhline(y=0, label="0", alpha=0.33)

            axs[3].set_ylabel("total_profit")
            axs[3].plot(history["total_profit"], linewidth=1, color="gray")
            axs[3].axhline(y=1, label="0", alpha=0.33)

            axs[4].set_ylabel("total_reward")
            axs[4].plot(history["total_reward"], linewidth=1, color="gray")
            axs[4].axhline(y=0, label="0", alpha=0.33)
            axs[4].set_xlabel("tick")

            _borders = ["top", "right", "bottom", "left"]
            for _ax in axs:
                for _border in _borders:
                    _ax.spines[_border].set_color("#5b5e4b")

            fig.suptitle(
                f"Total Reward: {self.total_reward:.2f} ~ " +
                f"Total Profit: {self._total_profit:.2f} ~ " +
                f"Trades: {len(self.trade_history)}"
            )
            fig.tight_layout()
            return fig

        def close(self) -> None:
            plt.close()
            gc.collect()
            th.cuda.empty_cache()


class InfoMetricsCallback(TensorboardCallback):
    """
    Tensorboard callback
    """

    def _on_training_start(self) -> None:
        _lr = self.model.learning_rate
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning_rate": _lr if _lr is float else "lr_schedule",
            # "gamma": self.model.gamma,
            # "gae_lambda": self.model.gae_lambda,
            # "n_steps": self.model.n_steps,
            # "batch_size": self.model.batch_size,
        }
        metric_dict = {
            "info/total_reward": 0,
            "info/total_profit": 1,
            "info/trade_count": 0,
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


class RolloutPlotCallback(BaseCallback):
    """
    Tensorboard plot callback
    """

    def record_env(self):
        figures = self.training_env.env_method("get_env_plot")
        for i, fig in enumerate(figures):
            figure = Figure(fig, close=True)
            self.logger.record(
                f"best/train_env_{i}", figure, exclude=("stdout", "log", "json", "csv")
            )
        return True

    def _on_step(self) -> bool:
        return self.record_env()


class MaskableTrialEvalCallback(MaskableEvalCallback):
    """
    Optuna maskable trial eval callback
    """

    def __init__(
        self,
        eval_env,
        trial,
        n_eval_episodes: int = 10,
        eval_freq: int = 2048,
        deterministic: bool = True,
        use_masking: bool = True,
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            use_masking=use_masking,
            **kwargs,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)

            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False

        return True


def make_env(
    MyRLEnv: Type[BaseEnvironment],
    env_id: str,
    rank: int,
    seed: int,
    train_df: DataFrame,
    price: DataFrame,
    env_info: Dict[str, Any],
) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :param env_info: (dict) all required arguments to instantiate the environment.
    :return: (Callable)
    """

    def _init() -> Env:
        env = MyRLEnv(df=train_df, prices=price, id=env_id, seed=seed + rank, **env_info)
        return env

    set_random_seed(seed)
    return _init

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func

def hours_to_seconds(hours):
    """
    Converts hours to seconds
    """
    seconds = hours * 3600
    return seconds

def steps_to_days(steps: int, timeframe: str) -> float:
    """
    Calculate the number of days based on the given number of steps
    """
    total_minutes = steps * timeframe_to_minutes(timeframe)
    days = total_minutes / (24 * 60)
    return round(days, 1)

def sample_params_ppo(trial: Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams
    """
    batch_size = trial.suggest_categorical("batch_size", [64, 256, 512, 1024, 10240])
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 10240])
    gamma = trial.suggest_float("gamma", 0.1, 0.99, step=0.01)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 0.01, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4, 0.5])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_float("gae_lambda", 0.1, 0.99, step=0.01)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 5, step=0.01)
    vf_coef = trial.suggest_float("vf_coef", 0, 1, step=0.01)
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    # ortho_init = True
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    activation_fn_name = trial.suggest_categorical(
        "activation_fn", ["tanh", "relu", "elu", "leaky_relu"]
    )
    lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])
    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)
    
    cr_schedule = trial.suggest_categorical("cr_schedule", ["linear", "constant"])
    if cr_schedule == "linear":
        clip_range = linear_schedule(clip_range)
    if batch_size > n_steps:
        batch_size = n_steps
    net_arch = {
        "small":  {"pi": [128, 128], "vf": [128, 128]},
        "medium": {"pi": [256, 256], "vf": [256, 256]},
        "large":  {"pi": [512, 512], "vf": [512, 512]},
    }[net_arch_type]

    activation_fn = {
        "tanh": th.nn.Tanh,
        "relu": th.nn.ReLU,
        "elu": th.nn.ELU,
        "leaky_relu": th.nn.LeakyReLU,
    }[activation_fn_name]
    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }

def sample_params_dqn(trial: Trial) -> Dict[str, Any]:
    """
    Sampler for DQN hyperparams
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 0.01, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5)])
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0, 0.2, step=0.1)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0, 0.5, step=0.1)
    target_update_interval = trial.suggest_categorical(
        "target_update_interval", [1000, 5000, 10000]
    )
    learning_starts = trial.suggest_categorical("learning_starts", [1000, 5000, 10000])
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 128, 256, 1000])
    subsample_steps = trial.suggest_categorical("subsample_steps", [1, 2, 4, 8])
    gradient_steps = max(train_freq // subsample_steps, 1)
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    net_arch = {
        "small":  [128, 128],
        "medium": [256, 256],
        "large":  [512, 512],
    }[net_arch_type]
    return {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "target_update_interval": target_update_interval,
        "learning_starts": learning_starts,
        "policy_kwargs": dict(net_arch=net_arch),
    }

def sample_params_qrdqn(trial: Trial) -> Dict[str, Any]:
    """
    Sampler for QRDQN hyperparams
    """
    hyperparams = sample_params_dqn(trial)
    n_quantiles = trial.suggest_int("n_quantiles", 5, 200)
    hyperparams["policy_kwargs"].update({"n_quantiles": n_quantiles})
    return hyperparams
