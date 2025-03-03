from dataclasses import dataclass
from functools import partial
from typing import Annotated
import jax

# Update the default device to the CPU
#jax.config.update("jax_default_device", jax.devices("cpu")[0])


import jax.numpy as jnp
import pickle
from luxai_s3.params import EnvParams
from luxai_s3.env import LuxAIS3Env
from purejaxrl_wrapper import LuxaiS3GymnaxWrapper, WrappedEnvObs
from luxai_s3.state import EnvObs


if __name__ == "__main__":
    jnp.set_printoptions(linewidth=500, suppress=True, precision=4)
    
    underlying_env = LuxAIS3Env(auto_reset=False, fixed_env_params=EnvParams())
    env = LuxaiS3GymnaxWrapper(underlying_env)
    

    with open("logs/obs.pkl", 'rb') as f: # Binary read mode for pickle
        file = pickle.load(f) # Load parameters directly using pickle.load

    all_new_obs = file["new_obs"]
    all_original_obs = file["original_obs"]
    all_state = file["state"]
    all_action = file["actions"]

    for step in range(26, 505):
        original_obs = all_original_obs[step]
        state = all_state[step]
        prev_state = all_state[step - 1]
        new_obs = all_new_obs[step]
        action = all_action[step]

        next_obs, next_env_state = env.transform_obs(original_obs, prev_state, env.fixed_env_params, 0, 1)

        print(next_obs)