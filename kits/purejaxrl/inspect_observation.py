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
from purejaxrl_wrapper import LuxaiS3GymnaxWrapper


if __name__ == "__main__":
    jnp.set_printoptions(linewidth=500)

    with open("logs/obs.pkl", 'rb') as f: # Binary read mode for pickle
        file = pickle.load(f) # Load parameters directly using pickle.load

    obs = file["obs"]
    state = file["state"]
    print(obs[0])