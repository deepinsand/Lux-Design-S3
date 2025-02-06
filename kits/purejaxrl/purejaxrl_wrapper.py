import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces
from jax.experimental import host_callback

from purejaxrl_util import MultiDiscrete
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

from packages.purejaxrl.purejaxrl.jax_debug import debuggable_conditional_breakpoint

    
class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

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
        observation_space_size = (
            params.max_units * 3  + # one for masks, one for x pos, one for y pos
            params.max_relic_nodes * 3  + # one for masks, one for x pos, one for y pos
            1 # points
        )
        return spaces.Box(
            low=0,
            high=1,
            shape=(observation_space_size,),
            dtype=jnp.float32
        )
    
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

    
    #@partial(jax.jit, static_argnums=(0,))
    def transform_obs(self, obs: EnvObs, state: StatefulEnvState, params):
        unit_mask = jnp.array(obs.units_mask[self.team_id]) # shape (max_units, )
        unit_positions = jnp.array(obs.units.position[self.team_id]) # shape (max_units, 2)
        #unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        observed_relic_node_positions = jnp.array(obs.relic_nodes) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = jnp.array(obs.relic_nodes_mask) # shape (max_relic_nodes, )
        #team_points = np.array(obs["team_points"]) # points of each team, team_points[self.team_id] is the points of the your team
        
        # ids of units you can control at this timestep

        discovered_relic_nodes_mask = state.discovered_relic_nodes_mask | observed_relic_nodes_mask
        discovered_relic_node_positions = jnp.maximum(state.discovered_relic_node_positions, observed_relic_node_positions)

        norm_and_flat_unit_positions = (unit_positions / self.fixed_env_params.map_width).flatten()
        norm_and_flat_relic_positions = (discovered_relic_node_positions / self.fixed_env_params.map_width).flatten()
        
        norm_last_diff_player_0_points = self.extract_differece_points_player_0(obs, state) / self.fixed_env_params.max_units

        new_observation = jnp.concatenate(
                [unit_mask, norm_and_flat_unit_positions, 
                discovered_relic_nodes_mask, norm_and_flat_relic_positions, 
                jnp.array([norm_last_diff_player_0_points])]
                
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