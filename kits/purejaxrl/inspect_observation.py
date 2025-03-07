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
from purejaxrl_config import config

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
    params = file["params"]

    for step in range(179, 505):
        original_obs = all_original_obs[step]
        state = all_state[step]
        prev_state = all_state[step - 1]
        new_obs = all_new_obs[step]
        action = all_action[step]

        total_solved_spots = state.solved_energy_points_grid_mask.sum()
        total_knowns_spots = state.known_energy_points_grid_mask.sum()

        jax.debug.print("step: {}, total_solved_spots: {}, total_knowns_spots: {}", step, total_solved_spots, total_knowns_spots)

        next_obs, next_env_state, _ = env.transform_obs(original_obs, prev_state, params, 0, 1, use_solver=config["TRANSFER_LEARNING"])