import sys
import numpy as np

import numpy as np
import jax
import dacite
from packages.gymnax_blines.utils.models import get_model_ready
from packages.gymnax_blines.utils.helpers import load_pkl_object, load_config
from packages.gymnax_blines.gymnax_wrapper import LuxaiS3GymnaxWrapper

from luxai_s3.params import EnvParams
from luxai_s3.state import EnvObs
from luxai_s3.env import LuxAIS3Env

def load_neural_network(config, agent_path, env, env_params):
    rng = jax.random.PRNGKey(0)
    model, _ = get_model_ready(rng, config, env, env_params)

    params = load_pkl_object(agent_path)["network"]
    return model, params

class Agent():

    def __init__(self, player: str, env_cfg) -> None:
        env_cfg = EnvParams(**env_cfg)
        self.env_cfg = env_cfg
        underlying_env = LuxAIS3Env(auto_reset=False, fixed_env_params=env_cfg)
        self.env = LuxaiS3GymnaxWrapper(underlying_env, "player_0")
        configs = load_config("/Users/sandeepjain/src/Lux-Design-S3/src/packages/gymnax_blines/ppo.yaml")
        self.model, self.model_params = load_neural_network(
            configs.train_config,
            "/Users/sandeepjain/src/Lux-Design-S3/src/packages/gymnax_blines/agents/LuxAIS3/ppo.pkl",
            self.env, self.env_cfg)

        rng = jax.random.PRNGKey(0)
        self.rng, self.rng_reset = jax.random.split(rng)
        self.env_state = self.env.empty_stateful_env_state()


    def act(self, step: int, obs, remainingOverageTime: int = 60): 
        env_obs = dacite.from_dict(data_class=EnvObs, data=obs)
        new_obs, self.env_state = self.env.transform_obs(env_obs, self.env_state, self.env_cfg)
        self.rng, rng_act = jax.random.split(self.rng)

        v, pi = self.model.apply(self.model_params, new_obs, rng_act)
        action = pi.sample(seed=rng_act)
        
        actions = np.zeros((self.env_cfg.max_units, 3), dtype=int)
        actions[:, 0] = np.array(action)

        return actions
