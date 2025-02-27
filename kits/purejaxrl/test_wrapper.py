from dataclasses import dataclass
from functools import partial
from typing import Annotated
import jax

# Update the default device to the CPU
#jax.config.update("jax_default_device", jax.devices("cpu")[0])


import jax.numpy as jnp
import tyro
from luxai_s3.params import EnvParams
from luxai_s3.env import LuxAIS3Env
from purejaxrl_wrapper import LuxaiS3GymnaxWrapper

from luxai_s3.params import env_params_ranges
from luxai_s3.profiler import Profiler

@dataclass
class Args:
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    trials_per_benchmark: Annotated[int, tyro.conf.arg(aliases=["-t"])] = 5
    verbose: Annotated[int, tyro.conf.arg(aliases=["-v"])] = 0
    seed: int = 0

if __name__ == "__main__":
    import numpy as np
    jax.config.update('jax_numpy_dtype_promotion', 'standard')
    args = tyro.cli(Args)

    np.random.seed(args.seed)

    # the first env params is not batched and is used to initialize any static / unchaging values
    # like map size, max units etc.
    # note auto_reset=False for speed reasons. If True, the default jax code will attempt to reset each time and discard the reset if its not time to reset
    # due to jax branching logic. It should be kept false and instead lax.scan followed by a reset after max episode steps should be used when possible since games
    # can't end early.
    env = LuxAIS3Env(auto_reset=True, fixed_env_params=EnvParams())
    env = LuxaiS3GymnaxWrapper(env)


    seed = args.seed
    rng_key = jax.random.key(seed)

    # sample random params initially
    def sample_params(rng_key):
        randomized_game_params = dict()
        for k, v in env_params_ranges.items():
            rng_key, subkey = jax.random.split(rng_key)
            if isinstance(v[0], int):
                randomized_game_params[k] = jax.random.choice(subkey, jax.numpy.array(v, dtype=jnp.int16))
            else:
                randomized_game_params[k] = jax.random.choice(subkey, jax.numpy.array(v, dtype=jnp.float32))
        params = EnvParams(**randomized_game_params)
        return params

    rng_key, subkey = jax.random.split(rng_key)
    env_params = sample_params(subkey)
    action_space = env.action_space() # note that this can generate sap actions beyond range atm
    sample_action = action_space.sample
    rng_key, subkey = jax.random.split(rng_key)

    obs, state = env.reset(subkey, env_params)

    max_episode_steps = (env.fixed_env_params.max_steps_in_match + 1) * env.fixed_env_params.match_count_per_episode
    previous = (obs, state, None, None)

    for simulations in range(10000):
        for step in range(max_episode_steps):
            rng_key, subkey = jax.random.split(rng_key)
            obs, state, reward, done, info = env.step(
                subkey, 
                state, 
                sample_action(subkey), 
                env_params
            )
            prev_obs, prev_state, prev_reward, prev_done = previous


            unit_mask = obs.unit_mask_player_0[0]
            unit_position = obs.unit_positions_player_0[0]

            positive_reward_in_map = obs.grid_probability_of_being_an_energy_point_based_on_positive_rewards[unit_position[0]][unit_position[1]]
            normalized_reward_last_round = obs.normalized_reward_last_round

            match = normalized_reward_last_round == positive_reward_in_map

            # compute relic node energy positions
            original_state = state.original_state
            relic_points = (original_state.relic_nodes_map_weights <= original_state.relic_nodes_mask.sum() // 2) & (original_state.relic_nodes_map_weights > 0)

            if not match.item():
                print(f"simulation: {simulations} step: {step}")
            previous = (obs, state, reward, done)



    

    