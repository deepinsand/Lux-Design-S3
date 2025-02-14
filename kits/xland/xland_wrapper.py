import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces

from luxai_s3.params import EnvParams
from luxai_s3.state import (
    ASTEROID_TILE,
    ENERGY_NODE_FNS,
    NEBULA_TILE,
    EnvObs,
    EnvState,
    MapTile,
    UnitState,
    gen_state
)

from packages.purejaxrl.purejaxrl.jax_debug import debuggable_vmap


class MultiDiscrete:
    """A minimal jittable MultiDiscrete space for Gymnax."""

    def __init__(self, nvec, n):
        # Convert to a JAX array (if not already)
        self.nvec = jnp.array(nvec, dtype=jnp.int_)
        self.shape = self.nvec.shape
        self.n = n
        self.dtype = jnp.int_

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        flat_nvec = self.nvec.flatten()
        num_elements = flat_nvec.shape[0]
        # Split the RNG for each element
        keys = jax.random.split(rng, num_elements)
        # Use vmap to sample for each discrete space in a vectorized way.
        sample_fn = jax.vmap(lambda key, n: jax.random.randint(key, shape=(), minval=0, maxval=n))
        samples_flat = sample_fn(keys, flat_nvec)
        return samples_flat.reshape(self.shape)

    def contains(self, x: chex.Array) -> jnp.ndarray:
        # Check that all entries are within their respective bounds.
        return jnp.all((x >= 0) & (x < self.nvec))

    
class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

@struct.dataclass
class WrappedEnvObs:
    normalized_reward_last_round: float
    tile_type: chex.Array
    relic_map: chex.Array
    unit_counts_player_0: chex.Array
    unit_positions_player_0: chex.Array
    unit_mask_player_0: chex.Array
            
@struct.dataclass
class StatefulEnvState:
    discovered_relic_nodes_mask: chex.Array
    discovered_relic_node_positions: chex.Array
    team_wins: chex.Array
    team_points: chex.Array

@struct.dataclass
class WrappedEnvState:
    original_state: EnvState
    stateful_data: StatefulEnvState

class LuxaiS3GymnaxWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment, player: str):
        super().__init__(env)
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return None # this is deeply coupled into train later
    
    def state_space(self, params: EnvParams):
        raise NotImplementedError("This function has not been implemented yet.")
    
    @property
    def num_actions(self):
        return self.fixed_env_params.max_units * 5
    
    def action_space(self, params: Optional[EnvParams] = None):
        high = jnp.ones((self.fixed_env_params.max_units,)) * 5
        return MultiDiscrete(high, 5)
    
    def empty_stateful_env_state(self):
        empty_data = StatefulEnvState(
            discovered_relic_nodes_mask=jnp.zeros(self.fixed_env_params.max_relic_nodes, dtype=jnp.int16),
            discovered_relic_node_positions=jnp.full((self.fixed_env_params.max_relic_nodes, 2), -1, dtype=jnp.int16),
            team_wins=jnp.zeros(2, dtype=jnp.int16),
            team_points=jnp.zeros(2, dtype=jnp.int16),
        )
        return empty_data
    
    #@partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        obs, stateful_data = self.transform_obs(obs[self.player], self.empty_stateful_env_state(), params)
        return obs, WrappedEnvState(original_state=state, stateful_data=stateful_data)
    
    


    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, wrapped_state, action, params=None):

        single_player_lux_action = jnp.zeros((self.fixed_env_params.max_units, 3), dtype=jnp.int16)
        single_player_lux_action = single_player_lux_action.at[:, 0].set(action)
        lux_action = {
            "player_0": single_player_lux_action, 
            "player_1": jnp.zeros((self.fixed_env_params.max_units, 3), dtype=jnp.int16)
        }
        
        obs, state, reward, terminated, truncated, info = self._env.step(key, wrapped_state.original_state, lux_action, params)
        obs, stateful_data = self.transform_obs(obs[self.player], wrapped_state.stateful_data, params)

        manufactured_reward = self.extract_differece_points_player_0(stateful_data, wrapped_state.stateful_data)

        #jax.debug.print("team_points: {}, reward: {}, cond: {}", state.team_points, manufactured_reward, jnp.any(state.team_points))
        #debuggable_conditional_breakpoint(jnp.any(state.team_points))

        done = terminated[self.player] | truncated[self.player]
        return obs, WrappedEnvState(original_state=state, stateful_data=stateful_data), manufactured_reward, done, info
    
    def compute_counts_map(self, position, mask):
        unit_counts_map = jnp.zeros(
            (self.fixed_env_params.map_width, 
             self.fixed_env_params.map_height), dtype=jnp.int16
        )

        def update_unit_counts_map(unit_position, unit_mask, unit_counts_map):
            mask = unit_mask
            unit_counts_map = unit_counts_map.at[
                unit_position[0], unit_position[1]
            ].add(mask.astype(jnp.int16))
            return unit_counts_map

        unit_counts_map = jnp.sum(
                debuggable_vmap(update_unit_counts_map, in_axes=(0, 0, None), out_axes=0)(
                    position, mask, unit_counts_map
                ),
                axis=0,
                dtype=jnp.int16
            )
        
        return unit_counts_map
    
    @partial(jax.jit, static_argnums=(0,))
    def transform_obs(self, obs: EnvObs, state: StatefulEnvState, params):
        observed_relic_node_positions = jnp.array(obs.relic_nodes) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = jnp.array(obs.relic_nodes_mask) # shape (max_relic_nodes, )

        discovered_relic_nodes_mask = state.discovered_relic_nodes_mask | observed_relic_nodes_mask
        discovered_relic_node_positions = jnp.maximum(state.discovered_relic_node_positions, observed_relic_node_positions)

        # relics
        # No need to normalize this since I don't think relics can occupy the same space
        relic_map = self.compute_counts_map(discovered_relic_node_positions, discovered_relic_nodes_mask)

        unit_mask = jnp.array(obs.units_mask[self.team_id]) # shape (max_units, )
        unit_positions = jnp.array(obs.units.position[self.team_id]) # shape (max_units, 2)
        #unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)

        unit_counts_player_0 = self.compute_counts_map(unit_positions, unit_mask) / float(self.fixed_env_params.max_units)
        norm_last_diff_player_0_points = self.extract_differece_points_player_0(obs, state) / self.fixed_env_params.max_units


        new_observation = WrappedEnvObs(
            relic_map=relic_map,
            unit_counts_player_0=unit_counts_player_0,
            normalized_reward_last_round=norm_last_diff_player_0_points,
            tile_type=jnp.array(obs.map_features.tile_type),
            unit_positions_player_0=unit_positions,
            unit_mask_player_0=unit_mask
        )
        
        new_state = StatefulEnvState(discovered_relic_node_positions=discovered_relic_node_positions,
                                     discovered_relic_nodes_mask=discovered_relic_nodes_mask,
                                     team_wins=obs.team_wins,
                                     team_points=obs.team_points,
                                     )
        
        return new_observation, new_state
    
    def extract_differece_points_player_0(self, obs, old_state):
        

        new_team_points = jnp.array(obs.team_points)
        new_team_wins = jnp.array(obs.team_wins)

        old_team_points = jnp.array(old_state.team_points)
        old_team_wins = jnp.array(old_state.team_wins)

        diff_points = new_team_points - old_team_points
        diff_wins = new_team_wins - old_team_wins

        # Game or match is over so reset diff
        # Match is reset so diff_wins should be 0
        diff_points_player_0 = jax.lax.cond(
            jnp.any(diff_wins),
            lambda: 0,
            lambda: jnp.dot(diff_points, jnp.array([1,0]))
        )

        return diff_points_player_0
    

@struct.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: environment.EnvState


class NormalizeVecReward(GymnaxWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = 1
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = 1

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info
