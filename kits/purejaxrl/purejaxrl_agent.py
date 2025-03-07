import sys
import numpy as np
import pickle 
import jax
import jax.numpy as jnp
import flax
import dacite
from purejaxrl_wrapper import LuxaiS3GymnaxWrapper, WrappedEnvObs, init_empty_obs
from purejaxrl_config import config
from functools import partial
from luxai_s3.params import EnvParams
from luxai_s3_local.state import EnvObs # kaggle enviornment has a different one?
from luxai_s3.env import LuxAIS3Env
import time
import os
import sys

from purejaxrl_ppo import ActorCritic

def load_model_for_inference(rng, network_cls, env, env_params):

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_file_dir, "models/latest_model.pkl")

    with open(model_path, 'rb') as f: # Binary read mode for pickle
        loaded_params = pickle.load(f) # Load parameters directly using pickle.load

    action_space = env.action_space(env_params)
    network = network_cls(
        [action_space.shape[0], action_space.n], activation=config["ACTIVATION"], quick=(not config["CONVOLUTIONS"])
    )

    init_obs = init_empty_obs(env_params, 1)

    network_params = network.init(rng, init_obs)
    loaded_params = flax.serialization.from_state_dict(network_params['params'], loaded_params['params'])


    return network, loaded_params

class Agent():

    def __init__(self, player: str, env_cfg_dict) -> None:
        env_cfg = EnvParams(**env_cfg_dict)
        self.env_cfg = env_cfg
        underlying_env = LuxAIS3Env(auto_reset=False, fixed_env_params=env_cfg)
        self.env = LuxaiS3GymnaxWrapper(underlying_env, total_updates=1)
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0

        # 3. Load the model

        t0 = time.time()
        rng = jax.random.PRNGKey(0)
        self.rng, rng_reset = jax.random.split(rng)
        self.model, self.model_params= load_model_for_inference(rng_reset, ActorCritic, self.env, self.env_cfg) # Or path to your saved .npz file
   
        self.env_state = self.env.empty_stateful_env_state()

    @partial(jax.jit, static_argnums=(0,4))
    def get_action(self, env_state, env_obs, rng_act, should_use_solver):

        new_obs, env_state, _ = self.env.transform_obs(env_obs, env_state, self.env_cfg, self.team_id, 
                                                       self.opp_team_id, update_count=1, use_solver=should_use_solver)

        new_obs_with_new_axis = jax.tree_util.tree_map(lambda x: jnp.array(x)[None, ...], new_obs)

        pi, v = self.model.apply({'params': self.model_params}, new_obs_with_new_axis)
        action = pi.sample(seed=rng_act)
        return (action, new_obs, env_state)

    def act(self, step: int, obs, remainingOverageTime: int = 60): 
        should_use_solver = config["TRANSFER_LEARNING"] and remainingOverageTime > 10

        actions = np.zeros((self.env_cfg.max_units, 3), dtype=int)
        t0 = time.time()

        env_obs = dacite.from_dict(data_class=EnvObs, data=obs)
        #print(f"dacite.from_dict: {time.time() - t0:.2f} s")
        self.rng, rng_act = jax.random.split(self.rng)


        
        #print(f"apply_and_sample: {time.time() - t0:.2f} s")
        
        action, new_obs, self.env_state = jax.block_until_ready(self.get_action(self.env_state, env_obs, rng_act, should_use_solver))

        actions[:, 0] = np.array(action)
        actions[:, 1:3] = np.array(self.env_state.candidate_sap_locations)

        return actions, new_obs, env_obs, self.env_state
