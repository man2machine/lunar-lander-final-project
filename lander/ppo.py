
from lander.lunar_lander import LunarLander
from stable_baselines3 import PPO

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

class ProgressBar(BaseCallback):
    def __init__(self, verbose=0):
        super(ProgressBar, self).__init__(verbose)
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm()

    def _on_rollout_start(self):
        self.pbar.refresh()

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        self.pbar.update(1)

    def _on_training_end(self):
        self.pbar.close()
        self.pbar = None

def get_ppo_model(env):
    model = PPO("MlpPolicy", env)
    return model


class RLPolicyWrapper:
    def __init__(self, model):
        self.model = model
    
    def predict(self, s):
        return self.model.predict(s, deterministic=True)[0]