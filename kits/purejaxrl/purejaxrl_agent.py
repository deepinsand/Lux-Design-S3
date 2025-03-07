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
from luxai_s3_local.state import EnvObs, UnitState, MapTile # kaggle enviornment has a different one?
from luxai_s3.env import LuxAIS3Env
import time
import os
import sys

from purejaxrl_ppo import ActorCritic

def create_empty_env_obs() -> EnvObs:
    T = 2  # Number of teams
    N = 16  # Max units per team
    W = 24 # Map width
    H = 24 # Map height
    N_relic = 6 # Max relic nodes

    return EnvObs(
        units=UnitState(position=jnp.zeros((T, N, 2), dtype=jnp.int16), energy=jnp.zeros((T, N), dtype=jnp.int16)),
        units_mask=jnp.zeros((T, N), dtype=jnp.int16),  # Or dtype=int if you want numerical mask
        sensor_mask=jnp.zeros((W, H), dtype=jnp.int16), # Or dtype=int
        map_features=MapTile(energy=jnp.zeros((W, H), dtype=jnp.int16), tile_type=jnp.zeros((W, H), dtype=jnp.int16)),
        relic_nodes=jnp.zeros((N_relic, 2), dtype=jnp.int16),
        relic_nodes_mask=jnp.zeros((N_relic), dtype=jnp.int16), # Or dtype=int
        team_points=jnp.zeros((T), dtype=jnp.int16),
        team_wins=jnp.zeros((T), dtype=jnp.int16),
        steps=0,
        match_steps=0
    )

class Agent():

    def __init__(self, player: str, env_cfg_dict) -> None:
        env_cfg = EnvParams(**env_cfg_dict)
        self.env_cfg = env_cfg
        underlying_env = LuxAIS3Env(auto_reset=False, fixed_env_params=env_cfg)
        self.env = LuxaiS3GymnaxWrapper(underlying_env)
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0

        # 3. Load the model

        t0 = time.time()
        rng = jax.random.PRNGKey(0)
        self.rng, rng_reset = jax.random.split(rng)
   
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_file_dir, "models/latest_model.pkl")

        with open(model_path, 'rb') as f: # Binary read mode for pickle
            loaded_params = pickle.load(f) # Load parameters directly using pickle.load

        action_space = self.env.action_space(env_cfg)
        network = ActorCritic(
            [action_space.shape[0], action_space.n], activation=config["ACTIVATION"], quick=(not config["CONVOLUTIONS"])
        )

        init_obs = init_empty_obs(env_cfg, 1)

        network_params = network.init(rng, init_obs)
        loaded_params = flax.serialization.from_state_dict(network_params['params'], loaded_params['params'])

        self.model = network
        self.model_params = loaded_params
        self.env_state = self.env.empty_stateful_env_state()
        self.jit_get_action = self.get_action_func()

        # do two to get it started
        self.jit_get_action(self.env_state, create_empty_env_obs(), rng_reset, True)
        self.jit_get_action(self.env_state, create_empty_env_obs(), rng_reset, False)
    

    def get_action_func(self):

        def to_jit(env_state, env_obs, rng_act, should_use_solver):
            new_obs, env_state, _ = self.env.transform_obs(env_obs, env_state, self.env_cfg, self.team_id, 
                                                        self.opp_team_id, use_solver=should_use_solver)

            new_obs_with_new_axis = jax.tree_util.tree_map(lambda x: jnp.array(x)[None, ...], new_obs)

            pi, v = self.model.apply({'params': self.model_params}, new_obs_with_new_axis)
            action = pi.sample(seed=rng_act)
            return (action, new_obs, env_state)
        return jax.jit(to_jit, static_argnums=(3,))

    def act(self, step: int, obs, remainingOverageTime: int = 60): 
        should_use_solver = config["TRANSFER_LEARNING"] and remainingOverageTime > 2

        actions = np.zeros((self.env_cfg.max_units, 3), dtype=int)
        t0 = time.time()

        env_obs = dacite.from_dict(data_class=EnvObs, data=obs)
        #print(f"dacite.from_dict: {time.time() - t0:.2f} s")
        self.rng, rng_act = jax.random.split(self.rng)


        
        #print(f"apply_and_sample: {time.time() - t0:.2f} s")
        
        action, new_obs, self.env_state = jax.block_until_ready(self.jit_get_action(self.env_state, env_obs, rng_act, should_use_solver))

        actions[:, 0] = np.array(action)
        actions[:, 1:3] = np.array(self.env_state.candidate_sap_locations)

        return actions, new_obs, env_obs, self.env_state
