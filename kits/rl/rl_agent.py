from lux.utils import direction_to
import sys
import numpy as np

from stable_baselines3.ppo import PPO
from wrapper import ObservationTransformer
from luxai_s3.params import EnvParams

# Load the saved model

class Agent():

    def __init__(self, player: str, env_cfg) -> None:

        env_cfg = EnvParams(**env_cfg)
        self.transformer = ObservationTransformer(player, env_cfg)
        self.env_cfg = env_cfg
        self.model = PPO.load("logs/exp_2/models/latest_model.zip")


    def act(self, step: int, obs, remainingOverageTime: int = 60):
        
        new_obs = self.transformer.transform(obs)
        action, _ = self.model.predict(new_obs)

        actions = np.zeros((self.env_cfg.max_units, 3), dtype=int)
        actions[:, 0] = action

        return actions
