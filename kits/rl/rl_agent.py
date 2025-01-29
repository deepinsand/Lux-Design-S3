from lux.utils import direction_to
import sys
import numpy as np

from stable_baselines3.common.utils import set_random_seed
from sb3_contrib.ppo_mask import MaskablePPO
from wrapper import ObservationTransformer
from luxai_s3.params import EnvParams
from wrapper import SB3Wrapper
# Load the saved model
from train import experiment_number


class Agent():

    def __init__(self, player: str, env_cfg) -> None:

        env_cfg = EnvParams(**env_cfg)
        self.transformer = ObservationTransformer(player, env_cfg)
        self.env_cfg = env_cfg
        self.model = MaskablePPO.load(f"logs/exp_{experiment_number}/models/latest_model.zip")

        #DEBUG NEED TO DELETE
        set_random_seed(1)
        

    def act(self, step: int, obs, remainingOverageTime: int = 60):  
        action_mask = ObservationTransformer.action_mask(self.env_cfg, obs)
        new_obs = self.transformer.transform_observation(obs)
        action, _ = self.model.predict(new_obs, action_masks=action_mask)

        actions = np.zeros((self.env_cfg.max_units, 3), dtype=int)
        actions[:, 0] = action

        return actions
