import sys
import numpy as np
import pickle 
import jax
import jax.numpy as jnp
import flax
import dacite
from purejaxrl_ppo_rnn import ActorCriticRNN, ScannedRNN
from purejaxrl_wrapper import LuxaiS3GymnaxWrapper
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
        [action_space.shape[0], action_space.n], config=config
    )
    init_x = (
        jnp.zeros(
            (1, 1, * env.observation_space(env_params).shape)
        ),
        jnp.zeros((1, 1)),
    )
    init_hstate = ScannedRNN.initialize_carry(1, 128)

    network_params = network.init(rng, init_hstate, init_x)

    loaded_params = flax.serialization.from_state_dict(network_params['params'], loaded_params['params'])


    return network, loaded_params, init_hstate




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
        self.model, self.model_params, self.hstate = load_model_for_inference(rng_reset, ActorCriticRNN, self.env, self.env_cfg) # Or path to your saved .npz file
        print(f"model load: {time.time() - t0:.2f} s")

        self.env_state = self.env.empty_stateful_env_state()


    def act(self, step: int, obs, remainingOverageTime: int = 60): 
        t0 = time.time()

        env_obs = dacite.from_dict(data_class=EnvObs, data=obs)
        #print(f"dacite.from_dict: {time.time() - t0:.2f} s")

        new_obs, self.env_state = self.env.transform_obs(env_obs, self.env_state, self.env_cfg)
        #print(f"transform_obs: {time.time() - t0:.2f} s")

        self.rng, rng_act = jax.random.split(self.rng)

        ac_in = (new_obs[np.newaxis, np.newaxis, :], jnp.array([[False]]))
        self.hstate, pi, v = self.model.apply({'params': self.model_params}, self.hstate, ac_in)
        action = pi.sample(seed=rng_act)
        #print(f"apply_and_sample: {time.time() - t0:.2f} s")
        
        actions = np.zeros((self.env_cfg.max_units, 3), dtype=int)
        actions[:, 0] = np.array(action)
        #print(f"turn to np: {time.time() - t0:.2f} s")

        return actions
