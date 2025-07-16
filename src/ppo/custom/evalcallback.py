from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import os

class CustomEvalCallback(EvalCallback):
    def __init__(self,stdlimit=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stdlimit = stdlimit

    def _on_step(self) -> bool:
        best_mean = self.best_mean_reward
        self.best_mean_reward = float('inf')
        result = super()._on_step()
        self.best_mean_reward = best_mean

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            std = np.std(self.evaluations_results[-1])
            mean_reward = self.last_mean_reward

            if std > self.stdlimit: 
                print(f"[무시됨] std={std:.2f} > 기준치. 평가 결과 무시.")
            elif mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.callback is not None:
                continue_training = continue_training and self._on_event()
            
        return result