# trialevalcallback.py

# gymnasium
import gymnasium as gym

# callback
from stable_baselines3.common.callbacks import EvalCallback

# hyperparameter tuning
import optuna

class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """

    def __init__(self,
                 eval_env: gym.Env,
                 trial: optuna.Trial,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 deterministic: bool = True,
                 verbose: int = 0,
                 ):
        """
        Initialize a TrialEvalCallBack.
        """

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        """
        Every so eval_freq, make a new
        report and prune trial if necessary.
        """

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
