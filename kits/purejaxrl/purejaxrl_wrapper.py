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
    tile_type: chex.Array
    relic_map: chex.Array
    unit_counts_player_0: chex.Array
    unit_positions_player_0: chex.Array
    unit_mask_player_0: chex.Array
    grid_probability_of_being_energy_point_based_on_relic_positions: chex.Array
    grid_probability_of_being_an_energy_point_based_on_no_reward: chex.Array
    grid_max_probability_of_being_an_energy_point_based_on_positive_rewards: chex.Array
    grid_min_probability_of_being_an_energy_point_based_on_positive_rewards: chex.Array
    grid_avg_probability_of_being_an_energy_point_based_on_positive_rewards: chex.Array

@struct.dataclass
class StatefulEnvState:
    discovered_relic_nodes_mask: chex.Array
    discovered_relic_node_positions: chex.Array
    team_wins: chex.Array
    team_points: chex.Array
    sensor_last_visit: chex.Array
    grid_probability_of_being_an_energy_point_based_on_no_reward: chex.Array
    grid_max_probability_of_being_an_energy_point_based_on_positive_rewards: chex.Array
    grid_min_probability_of_being_an_energy_point_based_on_positive_rewards: chex.Array
    total_rewards_when_positions_are_occupied: chex.Array
    total_times_positions_are_occupied: chex.Array

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
            sensor_last_visit=jnp.full((self.fixed_env_params.map_width, self.fixed_env_params.map_height), -1, dtype=jnp.int16),
            grid_probability_of_being_an_energy_point_based_on_no_reward=jnp.full((self.fixed_env_params.map_width, self.fixed_env_params.map_height), 1., dtype=jnp.float32),
            grid_max_probability_of_being_an_energy_point_based_on_positive_rewards=jnp.full((self.fixed_env_params.map_width, self.fixed_env_params.map_height), 0., dtype=jnp.float32),
            grid_min_probability_of_being_an_energy_point_based_on_positive_rewards=jnp.full((self.fixed_env_params.map_width, self.fixed_env_params.map_height), 1., dtype=jnp.float32),
            total_rewards_when_positions_are_occupied=jnp.full((self.fixed_env_params.map_width, self.fixed_env_params.map_height), 0., dtype=jnp.float32),
            total_times_positions_are_occupied=jnp.full((self.fixed_env_params.map_width, self.fixed_env_params.map_height), 1., dtype=jnp.int16), # start at 1 to avoid div 0
            team_wins=jnp.zeros(2, dtype=jnp.int32),
            team_points=jnp.zeros(2, dtype=jnp.int32),
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
        done = terminated[self.player] | truncated[self.player]

        if self._env.auto_reset:
            prev_stateful_data = jax.lax.cond(
                done,
                lambda: self.empty_stateful_env_state(),
                lambda: wrapped_state.stateful_data
            )
        else:
            prev_stateful_data = wrapped_state.stateful_data

        obs, stateful_data = self.transform_obs(obs[self.player], prev_stateful_data, params)

        manufactured_reward = self.extract_differece_points_player_0(stateful_data, prev_stateful_data)

        #jax.debug.print("team_points: {}, reward: {}, cond: {}", state.team_points, manufactured_reward, jnp.any(state.team_points))
        #debuggable_conditional_breakpoint(jnp.any(state.team_points))

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
    

    def multiply_5x5_mask_log_conv(self, arr):
        mask_size = 5
        kernel = jnp.ones((mask_size, mask_size), dtype=arr.dtype) # 5x5 kernel of ones

        # --- Logarithm and Convolution ---

        # Add a small epsilon to avoid log(0) errors.  Important if input can be exactly 0.
        epsilon = 1e-7  # Or a suitably small value for your probability range
        log_arr = jnp.log(arr + epsilon)  # Take logarithm of each element

        # Perform convolution. "SAME" padding ensures output is same size as input.
        # Feature group count and batch group count are both 1 for standard 2D convolution.
        convolved_log_arr = jax.lax.conv_general_dilated(
            lhs=log_arr.reshape(1, 1, arr.shape[0], arr.shape[1]), # Input in NCHW format (Batch=1, Channel=1)
            rhs=kernel.reshape(1, 1, mask_size, mask_size),        # Kernel in OIHW format (OutputChannel=1, InputChannel=1)
            window_strides=(1, 1),
            padding="SAME",
            lhs_dilation=(1, 1),
            rhs_dilation=(1, 1),
            dimension_numbers=jax.lax.ConvDimensionNumbers(
                lhs_spec=(0, 1, 2, 3),  # NCHW: Batch, Channel, Height, Width
                rhs_spec=(0, 1, 2, 3),  # OIHW: Output Channel, Input Channel, Kernel Height, Kernel Width
                out_spec=(0, 1, 2, 3), # NCHW: Batch, Channel, Height, Width
            )
        ).reshape(arr.shape) # Reshape back to 2D

        # --- Exponentiate to get the product ---
        result = jnp.exp(convolved_log_arr)

        return result

    
    @partial(jax.jit, static_argnums=(0,))
    def transform_obs(self, obs: EnvObs, state: StatefulEnvState, params):
        total_spaces = self.fixed_env_params.map_width * self.fixed_env_params.map_height
        observed_relic_node_positions = jnp.array(obs.relic_nodes) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = jnp.array(obs.relic_nodes_mask) # shape (max_relic_nodes, )

        discovered_relic_nodes_mask = state.discovered_relic_nodes_mask | observed_relic_nodes_mask
        discovered_relic_node_positions = jnp.maximum(state.discovered_relic_node_positions, observed_relic_node_positions)
        
        num_relics_discovered = discovered_relic_nodes_mask.sum()

        # relics stop spawning after match
        num_relics_undiscovered = self.fixed_env_params.max_relic_nodes  - num_relics_discovered
        relic_map = self.compute_counts_map(discovered_relic_node_positions, discovered_relic_nodes_mask)
        
        # 1) calculate last visit history
        current_step_last_visit = obs.sensor_mask * obs.steps
        sensor_mask_inverse = 1 - obs.sensor_mask
        
        sensor_last_visit = state.sensor_last_visit * sensor_mask_inverse # will set last_visit  to 0 wherever the sensors are visible
        sensor_last_visit = sensor_last_visit + current_step_last_visit # will set last_visit to obs.match_steps wherever the sensors are visible

        last_relic_spawn_point = self.fixed_env_params.max_steps_in_match * 3 # state.py:348 shows spawn schedule
        has_seen_everywhere_after_match_1 = jnp.all(sensor_last_visit > last_relic_spawn_point)
        num_relics_undiscovered = num_relics_undiscovered * (1 - has_seen_everywhere_after_match_1) # once we've seen everywhere we know there are no more relics to discover

        # 3) calculate probabilty of non-observable spaces having a relic:
        #       # of relics undiscovered: discovered relics - max_relic_nodes
        #       # probability a relic spawned there since visiting: / (undiscovered  / 100-time_since_last_visit) / # unobserved spaces
        # This calculates how much time has passed since we visited this space, capped to the 100 time step.  If a space was visited after 100, we clip it to 0 since there's
        # no chance a relic has spawned after that
        time_passed_since_last_visit_until_final_relic_spawn = jnp.min(jnp.array([last_relic_spawn_point, obs.steps])) - sensor_last_visit
        time_passed_since_last_visit_until_final_relic_spawn = jnp.clip(time_passed_since_last_visit_until_final_relic_spawn, min=0, max=last_relic_spawn_point)

        grid_expected_num_of_relics_spawned_since_last_observation = (num_relics_undiscovered * (time_passed_since_last_visit_until_final_relic_spawn.astype(jnp.float32) / last_relic_spawn_point))
        
        num_unobserved_spaces = sensor_mask_inverse.sum()
        grid_probability_of_relic_spawned_since_last_observation = grid_expected_num_of_relics_spawned_since_last_observation / num_unobserved_spaces

        # 100% chance of spawning if 
        grid_probability_of_relic_spawned_since_last_observation = jnp.where(relic_map > 0, relic_map, grid_probability_of_relic_spawned_since_last_observation)
        grid_probability_of_relic_spawned_and_contributing_to_energy_point_since_last_observation = grid_probability_of_relic_spawned_since_last_observation / 5. # staet.py:304 makes 20% of tiles valid 

        # 4) calculate probability of every space being an energy point
        #       # 5x5 covulution of 1-p of all spaces around it
        grid_probability_of_relic_not_spawned_and_contributing_to_energy_point_since_last_observation = (1. - grid_probability_of_relic_spawned_and_contributing_to_energy_point_since_last_observation)
        grid_probability_of_being_energy_point_based_on_relic_positions = 1 - self.multiply_5x5_mask_log_conv(grid_probability_of_relic_not_spawned_and_contributing_to_energy_point_since_last_observation)

        # 5) calculate probabily of not being an energy point based on reward obs
        #       init: 1
        #       (1-(reward/total units)))^units_on_spot) * prev value

        unit_mask = jnp.array(obs.units_mask[self.team_id]) # shape (max_units, )
        unit_positions = jnp.array(obs.units.position[self.team_id]) # shape (max_units, 2)
        #unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)

        unit_counts_player_0 = self.compute_counts_map(unit_positions, unit_mask)
        normalized_unit_counts_player_0 = unit_counts_player_0 / float(self.fixed_env_params.max_units)
        last_diff_player_0_points = self.extract_differece_points_player_0(obs, state)

        match_over = self.is_match_over(obs, state)
        


        
        # If there are no points, we need to do probability all relics being spawned
        grid_unit_mask = jnp.clip(unit_counts_player_0, max=1)
        grid_unit_mask_float = grid_unit_mask.astype(jnp.float32)
        inverse_grid_unit_mask_float = 1. - grid_unit_mask_float
   
        # MAYBE: account for actual relic discovereies
        
        total_unique_positions_occupied = jnp.max(jnp.array([grid_unit_mask_float.sum(), 1.]))
        probability_of_any_unit_being_on_energy_point = jax.lax.cond(match_over, lambda: 0., lambda: last_diff_player_0_points / total_unique_positions_occupied)
        grid_probability_of_being_an_energy_point_based_on_this_turns_positions = (grid_unit_mask_float * probability_of_any_unit_being_on_energy_point)

        total_rewards_when_positions_are_occupied = state.total_rewards_when_positions_are_occupied + grid_probability_of_being_an_energy_point_based_on_this_turns_positions
        grid_unit_mask_if_match_not_over = jax.lax.cond(match_over, lambda: jnp.zeros_like(grid_unit_mask), lambda: grid_unit_mask)
        total_times_positions_are_occupied = state.total_times_positions_are_occupied + grid_unit_mask_if_match_not_over # should this account for relic spawning?

        grid_max_probability_of_being_an_energy_point_based_on_positive_rewards = jnp.max(jnp.array([state.grid_max_probability_of_being_an_energy_point_based_on_positive_rewards, grid_probability_of_being_an_energy_point_based_on_this_turns_positions]), axis=0)

        # Chance of a spot not being an EP when there is no reward is based on how far into spawning you are.  0 when you start and 100% when done
        percent_spawning_left = 1. - (jnp.min(jnp.array([last_relic_spawn_point, obs.steps])) / float(last_relic_spawn_point))
        grid_probability_of_being_an_energy_point_based_on_no_reward = inverse_grid_unit_mask_float + (grid_unit_mask_float * percent_spawning_left)

        grid_probability_of_being_an_energy_point_with_spawning_accounted_for = jax.lax.cond(
            last_diff_player_0_points, 
            lambda: grid_probability_of_being_an_energy_point_based_on_this_turns_positions + inverse_grid_unit_mask_float, 
            lambda: grid_probability_of_being_an_energy_point_based_on_no_reward
        )        
        
        grid_min_probability_of_being_an_energy_point_based_on_positive_rewards =  jax.lax.cond(
            match_over, 
            lambda: state.grid_min_probability_of_being_an_energy_point_based_on_positive_rewards,
            lambda: jnp.min(jnp.array([state.grid_min_probability_of_being_an_energy_point_based_on_positive_rewards, grid_probability_of_being_an_energy_point_with_spawning_accounted_for]), axis=0)
        )
        grid_avg_probability_of_being_an_energy_point_based_on_positive_rewards = total_rewards_when_positions_are_occupied / total_times_positions_are_occupied
        

        # 6) combines by multiplying?


        # TODO: account for units getting killed, check energy?



        new_observation = WrappedEnvObs(
            relic_map=relic_map, # not used
            unit_counts_player_0=normalized_unit_counts_player_0,
            tile_type=jnp.array(obs.map_features.tile_type),
            unit_positions_player_0=unit_positions,
            unit_mask_player_0=unit_mask,
            grid_probability_of_being_energy_point_based_on_relic_positions=grid_probability_of_being_energy_point_based_on_relic_positions,
            grid_max_probability_of_being_an_energy_point_based_on_positive_rewards=grid_max_probability_of_being_an_energy_point_based_on_positive_rewards,
            grid_min_probability_of_being_an_energy_point_based_on_positive_rewards=grid_min_probability_of_being_an_energy_point_based_on_positive_rewards,
            grid_avg_probability_of_being_an_energy_point_based_on_positive_rewards=grid_avg_probability_of_being_an_energy_point_based_on_positive_rewards,
            grid_probability_of_being_an_energy_point_based_on_no_reward=grid_probability_of_being_an_energy_point_based_on_no_reward

        )
        
        new_state = StatefulEnvState(discovered_relic_node_positions=discovered_relic_node_positions,
                                     discovered_relic_nodes_mask=discovered_relic_nodes_mask,
                                     team_wins=obs.team_wins,
                                     team_points=obs.team_points,
                                     grid_probability_of_being_an_energy_point_based_on_no_reward=grid_probability_of_being_an_energy_point_based_on_no_reward,
                                     grid_max_probability_of_being_an_energy_point_based_on_positive_rewards=grid_max_probability_of_being_an_energy_point_based_on_positive_rewards,
                                     grid_min_probability_of_being_an_energy_point_based_on_positive_rewards=grid_min_probability_of_being_an_energy_point_based_on_positive_rewards,
                                     total_rewards_when_positions_are_occupied=total_rewards_when_positions_are_occupied,
                                     total_times_positions_are_occupied=total_times_positions_are_occupied,
                                     sensor_last_visit=sensor_last_visit
                                     )
        
        return new_observation, new_state
    


    def is_match_over(self, obs, old_state):
        new_team_wins = jnp.array(obs.team_wins)
        old_team_wins = jnp.array(old_state.team_wins)
        diff_wins = new_team_wins - old_team_wins

        return jnp.any(diff_wins)
    
    def extract_differece_points_player_0(self, obs, old_state):
        

        new_team_points = jnp.array(obs.team_points)
        new_team_wins = jnp.array(obs.team_wins)

        old_team_points = jnp.array(old_state.team_points)
        old_team_wins = jnp.array(old_state.team_wins)

        diff_points = new_team_points - old_team_points
        diff_wins = new_team_wins - old_team_wins

        # On steps mod 101, the points are wiped out but the wins change.
        # when the wins change we have no idea how many points we collected that round, or if they mattered, so count them as 0 ...
        diff_points_player_0 = jax.lax.cond(
            jnp.any(diff_wins),
            lambda: 0,#jnp.dot(diff_wins, jnp.array([1,0])),
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
