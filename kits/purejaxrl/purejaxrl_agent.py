import sys
import numpy as np
import pickle 
import jax
import jax.numpy as jnp
import flax
import dacite
from purejaxrl_ppo import ActorCritic
from purejaxrl_wrapper import LuxaiS3GymnaxWrapper, WrappedEnvObs
from purejaxrl_train import config

from luxai_s3.params import EnvParams
from luxai_s3.state import EnvObs
from luxai_s3.env import LuxAIS3Env
import time

def load_model_for_inference(rng, network_cls, env, env_params):

    with open("models/latest_model.pkl", 'rb') as f: # Binary read mode for pickle
        loaded_params = pickle.load(f) # Load parameters directly using pickle.load

    action_space = env.action_space(env_params)
    network = network_cls(
        [action_space.shape[0], action_space.n]
    )

    def fill_zeroes(shape, dtype=jnp.int16):
        return jnp.zeros((1, *shape), dtype=dtype)
    
    init_obs = WrappedEnvObs(
        relic_map=fill_zeroes((env_params.map_width, env_params.map_height)),
        unit_counts_player_0=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        tile_type=fill_zeroes((env_params.map_width, env_params.map_height)),
        normalized_reward_last_round=fill_zeroes((), dtype=jnp.float32),
        unit_positions_player_0=fill_zeroes((env_params.max_units, 2)),
        unit_mask_player_0=fill_zeroes((env_params.max_units,)),
        grid_probability_of_being_energy_point_based_on_relic_positions=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        grid_probability_of_being_an_energy_point_based_on_positive_rewards=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        grid_probability_of_being_an_energy_point_based_on_no_reward=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
    )

    network_params = network.init(rng, init_obs)

    loaded_params = flax.serialization.from_state_dict(network_params['params'], loaded_params['params'])


    return network, loaded_params

class Agent():

    def __init__(self, player: str, env_cfg) -> None:
        env_cfg = EnvParams(**env_cfg)
        self.env_cfg = env_cfg
        underlying_env = LuxAIS3Env(auto_reset=False, fixed_env_params=env_cfg)
        self.env = LuxaiS3GymnaxWrapper(underlying_env, "player_0")
     
        # 3. Load the model

        t0 = time.time()
        rng = jax.random.PRNGKey(0)
        self.rng, rng_reset = jax.random.split(rng)
        self.model, self.model_params= load_model_for_inference(rng_reset, ActorCritic, self.env, self.env_cfg) # Or path to your saved .npz file
        print(f"model load: {time.time() - t0:.2f} s")

        self.env_state = self.env.empty_stateful_env_state()


    def act(self, step: int, obs, remainingOverageTime: int = 60): 
        t0 = time.time()

        env_obs = dacite.from_dict(data_class=EnvObs, data=obs)
        #print(f"dacite.from_dict: {time.time() - t0:.2f} s")

        new_obs, self.env_state = self.env.transform_obs(env_obs, self.env_state, self.env_cfg)
        #print(f"transform_obs: {time.time() - t0:.2f} s")

        self.rng, rng_act = jax.random.split(self.rng)

        new_obs_with_new_axis = jax.tree_util.tree_map(lambda x: jnp.array(x)[None, ...], new_obs)

        pi, v = self.model.apply({'params': self.model_params}, new_obs_with_new_axis)
        action = pi.sample(seed=rng_act)
        #print(f"apply_and_sample: {time.time() - t0:.2f} s")
        
        actions = np.zeros((self.env_cfg.max_units, 3), dtype=int)
        actions[:, 0] = np.array(action)
        #print(f"turn to np: {time.time() - t0:.2f} s")

        return actions
