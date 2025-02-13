import sys
import numpy as np
import pickle 
import jax
import jax.numpy as jnp
import flax
import dacite
from nn import ActorCriticRNN
from xland_wrapper import LuxaiS3GymnaxWrapper, WrappedEnvObs
from xland_train import TrainConfig

from luxai_s3.params import EnvParams
from luxai_s3.state import EnvObs
from luxai_s3.env import LuxAIS3Env
import time

config = TrainConfig()
def load_model_for_inference(rng, env, env_params):

    with open("models/latest_model.pkl", 'rb') as f: # Binary read mode for pickle
        loaded_params = pickle.load(f) # Load parameters directly using pickle.load

    action_space = env.action_space(env_params)
    network = ActorCriticRNN(
        action_dim=[action_space.shape[0], action_space.n],
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_num_layers=config.rnn_num_layers,
        head_hidden_dim=config.head_hidden_dim,
        dtype=jnp.bfloat16 if config.enable_bf16 else None,
    )

    
    def fill_zeroes(shape, dtype=jnp.int16):
        return jnp.zeros((config.num_envs_per_device, 1, *shape), dtype=dtype)
    
    init_obs = WrappedEnvObs(
        relic_map=fill_zeroes((env_params.map_width, env_params.map_height)),
        unit_counts_player_0=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        tile_type=fill_zeroes((env_params.map_width, env_params.map_height)),
        normalized_reward_last_round=fill_zeroes((), dtype=jnp.float32),
        unit_positions_player_0=fill_zeroes((env_params.max_units, 2)),
        unit_mask_player_0=fill_zeroes((env_params.max_units,)),
    )

    init_hstate = network.initialize_carry(batch_size=config.num_envs_per_device)
    network_params = network.init(rng, init_obs, init_hstate)

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
        self.model, self.model_params = load_model_for_inference(rng_reset, self.env, self.env_cfg) # Or path to your saved .npz file
        print(f"model load: {time.time() - t0:.2f} s")

        self.env_state = self.env.empty_stateful_env_state()
        self.hstate = jnp.zeros((1, config.rnn_num_layers, config.rnn_hidden_dim))

    def act(self, step: int, obs, remainingOverageTime: int = 60): 
        t0 = time.time()

        env_obs = dacite.from_dict(data_class=EnvObs, data=obs)
        #print(f"dacite.from_dict: {time.time() - t0:.2f} s")

        new_obs, self.env_state = self.env.transform_obs(env_obs, self.env_state, self.env_cfg)
        new_obs_with_new_axis = jax.tree_util.tree_map(lambda x: jnp.array(x)[None, None, ...], new_obs)

        #print(f"transform_obs: {time.time() - t0:.2f} s")

        self.rng, rng_act = jax.random.split(self.rng)

        pi, v, self.hstate = self.model.apply({'params': self.model_params}, new_obs_with_new_axis, self.hstate)
        action = pi.sample(seed=rng_act)
        #print(f"apply_and_sample: {time.time() - t0:.2f} s")
        
        actions = np.zeros((self.env_cfg.max_units, 3), dtype=int)
        actions[:, 0] = np.array(action)
        #print(f"turn to np: {time.time() - t0:.2f} s")

        return actions
