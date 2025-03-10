import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces
import os

from benchmark_bitwise_and import create_centered_mask, extract_32bit_from_grid_mask, get_certain_positions, reconstruct_grid_from_subsection_bit_mask
from luxai_s3.params import EnvParams, env_params_ranges
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

from jax_debug import debuggable_vmap


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
    normalized_energy_field: chex.Array
    relic_map: chex.Array
    normalized_unit_counts: chex.Array
    normalized_unit_counts_opp: chex.Array
    normalized_unit_energys_max_grid: chex.Array
    normalized_unit_energys_max_grid_opp: chex.Array
    unit_positions: chex.Array
    normalized_unit_positions: chex.Array
    normalized_unit_energys: chex.Array
    unit_mask: chex.Array
    grid_probability_of_being_energy_point_based_on_relic_positions: chex.Array
    grid_probability_of_being_an_energy_point_based_on_no_reward: chex.Array
    grid_max_probability_of_being_an_energy_point_based_on_positive_rewards: chex.Array
    grid_min_probability_of_being_an_energy_point_based_on_positive_rewards: chex.Array
    grid_avg_probability_of_being_an_energy_point_based_on_positive_rewards: chex.Array
    value_of_sapping_grid: chex.Array
    sensor_mask: chex.Array
    sensor_last_visit_normalized: chex.Array
    action_mask: chex.Array
    param_list: chex.Array
    solved_energy_points_grid_mask: chex.Array

def init_empty_obs(env_params, num_envs):
    def fill_zeroes(shape, dtype=jnp.int16):
        return jnp.zeros((num_envs, *shape), dtype=dtype)
    
    return WrappedEnvObs(
        relic_map=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        normalized_unit_counts=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        normalized_unit_counts_opp=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        normalized_unit_energys_max_grid=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        normalized_unit_energys_max_grid_opp=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        tile_type=fill_zeroes((env_params.map_width, env_params.map_height)),
        normalized_energy_field=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        unit_positions=fill_zeroes((env_params.max_units, 2)),
        normalized_unit_energys=fill_zeroes((env_params.max_units,), dtype=jnp.float32),
        normalized_unit_positions=fill_zeroes((env_params.max_units, 2), dtype=jnp.float32),
        unit_mask=fill_zeroes((env_params.max_units,)),
        grid_probability_of_being_energy_point_based_on_relic_positions=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        grid_probability_of_being_an_energy_point_based_on_no_reward=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        grid_max_probability_of_being_an_energy_point_based_on_positive_rewards=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        grid_min_probability_of_being_an_energy_point_based_on_positive_rewards=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        grid_avg_probability_of_being_an_energy_point_based_on_positive_rewards=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        value_of_sapping_grid=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        sensor_mask=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        sensor_last_visit_normalized=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        action_mask=fill_zeroes((env_params.max_units, 6), dtype=jnp.bool),
        param_list=fill_zeroes((9,), dtype=jnp.float32),
        solved_energy_points_grid_mask=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
    )
@struct.dataclass
class StatefulEnvState:
    discovered_relic_nodes_mask: chex.Array
    discovered_relic_node_positions: chex.Array
    team_wins: chex.Array
    team_points: chex.Array
    sensor_last_visit: chex.Array
    last_time_sure_no_energy_point: chex.Array
    grid_max_probability_of_being_an_energy_point_based_on_positive_rewards: chex.Array
    grid_min_probability_of_being_an_energy_point_based_on_positive_rewards: chex.Array
    total_rewards_when_positions_are_occupied: chex.Array
    total_times_positions_are_occupied: chex.Array
    unit_positions_opp_last_round: chex.Array
    unit_mask_opp_last_round: chex.Array
    symmetrical_tile_type_last_round: chex.Array
    candidate_sap_locations: chex.Array
    drift_speed_guess: float
    stored_unit_masks_around_relics: chex.Array
    stored_rewards: chex.Array
    solved_energy_points_grid_mask: chex.Array
    max_solved_certainty_params: chex.Array


@struct.dataclass
class RewardInfo:
    unit_counts: chex.Array
    accumulated_solved_energy_points_grid_mask: chex.Array


@struct.dataclass
class WrappedEnvState:
    original_state: EnvState
    stateful_data_0: StatefulEnvState
    stateful_data_1: StatefulEnvState


class LuxaiS3GymnaxWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment, total_updates=None, use_solver=False):
        super().__init__(env)
        self.total_updates = total_updates
        self.use_solver = use_solver
      
    def observation_space(self, params: EnvParams) -> spaces.Box:
        return None # this is deeply coupled into train later
    
    def state_space(self, params: EnvParams):
        raise NotImplementedError("This function has not been implemented yet.")
    
    @property
    def num_actions(self):
        return self.fixed_env_params.max_units * 6
    
    def action_space(self, params: Optional[EnvParams] = None):
        high = jnp.ones((self.fixed_env_params.max_units,)) * 6
        return MultiDiscrete(high, 6)
    
    def empty_stateful_env_state(self):
        empty_data = StatefulEnvState(
            discovered_relic_nodes_mask=jnp.zeros(self.fixed_env_params.max_relic_nodes, dtype=jnp.int16),
            discovered_relic_node_positions=jnp.full((self.fixed_env_params.max_relic_nodes, 2), -1, dtype=jnp.int16),
            sensor_last_visit=jnp.full((self.fixed_env_params.map_width, self.fixed_env_params.map_height), -1, dtype=jnp.int16),
            last_time_sure_no_energy_point=jnp.full((self.fixed_env_params.map_width, self.fixed_env_params.map_height), -1, dtype=jnp.int16),
            grid_max_probability_of_being_an_energy_point_based_on_positive_rewards=jnp.full((self.fixed_env_params.map_width, self.fixed_env_params.map_height), 0., dtype=jnp.float32),
            grid_min_probability_of_being_an_energy_point_based_on_positive_rewards=jnp.full((self.fixed_env_params.map_width, self.fixed_env_params.map_height), 1., dtype=jnp.float32),
            total_rewards_when_positions_are_occupied=jnp.full((self.fixed_env_params.map_width, self.fixed_env_params.map_height), 0., dtype=jnp.float32),
            total_times_positions_are_occupied=jnp.full((self.fixed_env_params.map_width, self.fixed_env_params.map_height), 1, dtype=jnp.int16), # start at 1 to avoid div 0
            team_wins=jnp.zeros(2, dtype=jnp.int32),
            team_points=jnp.zeros(2, dtype=jnp.int32),
            unit_positions_opp_last_round=jnp.full((self.fixed_env_params.max_units, 2), -1, dtype=jnp.int16),
            unit_mask_opp_last_round=jnp.full(self.fixed_env_params.max_units, -1, dtype=jnp.bool),
            symmetrical_tile_type_last_round=jnp.full((self.fixed_env_params.map_width, self.fixed_env_params.map_height), -1, dtype=jnp.int16),
            candidate_sap_locations=jnp.full((self.fixed_env_params.max_units, 2), -1, dtype=jnp.int32),
            drift_speed_guess=jnp.zeros((), dtype=jnp.float32),
            solved_energy_points_grid_mask=jnp.zeros((self.fixed_env_params.max_relic_nodes // 2, self.fixed_env_params.map_width, self.fixed_env_params.map_height), dtype=jnp.int32),
            stored_unit_masks_around_relics=jnp.zeros((self.fixed_env_params.max_relic_nodes // 2, 50, 25), dtype=jnp.int32),
            stored_rewards=jnp.zeros((self.fixed_env_params.max_relic_nodes // 2, 50), dtype=jnp.int32),
            max_solved_certainty_params=jnp.zeros((self.fixed_env_params.max_relic_nodes // 2), dtype=jnp.int32),
        )
        return empty_data
    
    #@partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        obs_0, stateful_data_0, _ = self.transform_obs(obs["player_0"], self.empty_stateful_env_state(), params, 0, 1, 0, use_solver=False)
        obs_1, stateful_data_1, _ = self.transform_obs(obs["player_1"], self.empty_stateful_env_state(), params, 1, 0, 0, use_solver=False)
        return (obs_0, obs_1), WrappedEnvState(original_state=state, stateful_data_0=stateful_data_0, stateful_data_1=stateful_data_1)
    
    


    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, wrapped_state, action, params=None, update_count=None):

        sap_action_0 = jnp.zeros((self.fixed_env_params.max_units, 3), dtype=jnp.int16)
        sap_action_0 = sap_action_0.at[:, 0].set(action[0])
        sap_action_0 = sap_action_0.at[:, 1:3].set(wrapped_state.stateful_data_0.candidate_sap_locations)
        
        sap_action_1 = jnp.zeros((self.fixed_env_params.max_units, 3), dtype=jnp.int16)
        sap_action_1 = sap_action_1.at[:, 0].set(action[1])
        sap_action_1 = sap_action_1.at[:, 1:3].set(wrapped_state.stateful_data_1.candidate_sap_locations)

        lux_action = {
            "player_0": sap_action_0, 
            "player_1": sap_action_1
        }
        
        obs, state, reward, terminated, truncated, info = self._env.step(key, wrapped_state.original_state, lux_action, params)
        done = terminated["player_0"] | truncated["player_0"] | terminated["player_1"] | truncated["player_1"] # i dont think this happens

        if self._env.auto_reset:
            prev_stateful_data_0 = jax.lax.cond(
                done,
                lambda: self.empty_stateful_env_state(),
                lambda: wrapped_state.stateful_data_0
            )
            prev_stateful_data_1 = jax.lax.cond(
                done,
                lambda: self.empty_stateful_env_state(),
                lambda: wrapped_state.stateful_data_1
            )
        else:
            prev_stateful_data_0 = wrapped_state.stateful_data_0
            prev_stateful_data_1 = wrapped_state.stateful_data_1

        new_obs_0, stateful_data_0, reward_info_0 = self.transform_obs(obs["player_0"], prev_stateful_data_0, params, 0, 1, update_count=update_count, use_solver=self.use_solver)
        new_obs_1, stateful_data_1, reward_info_1 = self.transform_obs(obs["player_1"], prev_stateful_data_1, params, 1, 0, update_count=update_count, use_solver=self.use_solver)

        info["real_reward"] = self.extract_differece_points_for_player(stateful_data_0, prev_stateful_data_0, 0)

        manufactured_reward_0 = self.extract_manufactured_reward(stateful_data_0, prev_stateful_data_0, 0, 1, update_count, reward_info_0)
        manufactured_reward_1 = self.extract_manufactured_reward(stateful_data_1, prev_stateful_data_1, 1, 0, update_count, reward_info_1)

        #jax.debug.print("team_points: {}, reward: {}, cond: {}", state.team_points, manufactured_reward, jnp.any(state.team_points))
        #debuggable_conditional_breakpoint(jnp.any(state.team_points))

        return (new_obs_0, new_obs_1), WrappedEnvState(original_state=state, stateful_data_0=stateful_data_0, stateful_data_1=stateful_data_1), (manufactured_reward_0, manufactured_reward_1), done, info
    

    def compute_map(self, position, values, accumulator, dtype):
        unit_counts_map = jnp.zeros(
            (self.fixed_env_params.map_width, 
             self.fixed_env_params.map_height), dtype=dtype
        )

        def update_unit_counts_map(unit_position, value, unit_counts_map):
            unit_counts_map = unit_counts_map.at[
                unit_position[0], unit_position[1]
            ].add(value)
            return unit_counts_map

        unit_counts_map = accumulator(
                debuggable_vmap(update_unit_counts_map, in_axes=(0, 0, None), out_axes=0)(
                    position, values, unit_counts_map
                ),
                axis=0
            )
        
        return unit_counts_map

    def compute_energy_map_max(self, position, mask, energies):
        energies = energies * mask.astype(jnp.int16)
        return self.compute_map(position, energies, jnp.max, jnp.int16)
        
    def compute_energy_map_sum(self, position, mask, energies):
        partial_sum = partial(jnp.sum, dtype=jnp.int16)
        energies = energies * mask.astype(jnp.int16)
        return self.compute_map(position, energies, partial_sum, jnp.int16)
    

    def compute_counts_map(self, position, mask):
        partial_sum = partial(jnp.sum, dtype=jnp.int16)
        return self.compute_map(position, mask.astype(jnp.int16), partial_sum, jnp.int16)

    def extract_values_from_map(self, map, positions, mask, default_value):

        def extract_feature(pos, valid):
            row, col = pos  # extract row and col indices
            value = map[row, col]  # extract feature vector at that spatial location
            # If the agent doesn't exist (valid == False), return zeros.
            return jnp.where(valid, value, default_value)
        # Vectorize over the num_agents dimension.
        return debuggable_vmap(extract_feature)(positions, mask)
        
    def find_max_index_subsection(self, grid, pos, subsection_size):

        half_subsection = subsection_size // 2
        padding_size = half_subsection
        padding = ((padding_size, padding_size), (padding_size, padding_size))
        padded_grid = jnp.pad(grid, padding, mode='constant', constant_values=0)

        center_x, center_y = pos
        # Calculate start and end indices for rows and columns, handling edges
        start_row = center_x - half_subsection + padding_size
        start_col = center_y - half_subsection + padding_size

        # Extract the subsection using lax.dynamic_slice
        subsection = jax.lax.dynamic_slice(padded_grid, (start_row, start_col), (subsection_size, subsection_size))

        # Find the index of the maximum value within the subsection (flattened)
        max_index_flat = jnp.argmax(subsection)

        # Convert the flattened index to 2D indices within the subsection
        max_index_subsection = jnp.unravel_index(max_index_flat, subsection.shape)

        # Adjust the subsection indices to get the indices in the original grid
        max_row_index_original = start_row + max_index_subsection[0] - padding_size
        max_col_index_original = start_col + max_index_subsection[1] - padding_size

        valid_action = (subsection > 0).any()
        return (max_row_index_original, max_col_index_original, valid_action)
    

    def find_max_index_subsection_for_sap_ranges(self, grid, pos, subsection_size):
        possible_sizes = env_params_ranges['unit_sap_range']
        index = subsection_size - possible_sizes[0]

        # Use jax.switch to select the correct branch based on subsection_size
        return jax.lax.switch(
            index,
            [lambda g, p: self.find_max_index_subsection(g, p, ((size * 2) + 1)) for size in possible_sizes], # size is like radius here
            grid, 
            pos
        )
    

    def convolution_for_sap_actions_3x3(self, arr, adjacent_value, direct_value, dtype):
        mask_size = 3
        kernel = jnp.full((mask_size, mask_size), fill_value=adjacent_value, dtype=dtype)
        kernel = kernel.at[1, 1].set(direct_value)

        # Perform convolution. "SAME" padding ensures output is same size as input.
        # Feature group count and batch group count are both 1 for standard 2D convolution.
        convolved_arr = jax.lax.conv_general_dilated(
            lhs=arr.reshape(1, 1, arr.shape[0], arr.shape[1]), # Input in NCHW format (Batch=1, Channel=1)
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

        return convolved_arr    

    def convolution_for_relic_nodes_5x5(self, arr):
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
    
    def drift_map(self, map, steps, drift_speed):
                # Code taken from underlying env
        # Shift objects around in space
        # Move the nebula tiles in state.map_features.tile_types up by 1 and to the right by 1
        # this is also symmetric nebula tile movement # 
        # Drift speed is not known!!!
        drifted_map = jnp.roll(
            map,
            shift=(
                1 * jnp.sign(drift_speed),
                -1 * jnp.sign(drift_speed),
            ),
            axis=(0, 1),
        )
        drifted_map = jnp.where(
            (steps - 1) * abs(drift_speed) % 1 > (steps) * abs(drift_speed) % 1,
            drifted_map,
            map,
        )
        return drifted_map
    
    @partial(jax.jit, static_argnums=(0,7))
    def transform_obs(self, obs: EnvObs, state: StatefulEnvState, params, team_id, opp_team_id, update_count, use_solver=False):
        observed_relic_node_positions = jnp.array(obs.relic_nodes) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = jnp.array(obs.relic_nodes_mask) # shape (max_relic_nodes, )
        
        reshaped_observed_relic_nodes_mask = jnp.stack([observed_relic_nodes_mask, observed_relic_nodes_mask]).T.astype(jnp.int16) 
        inverse_negative_reshaped_observed_relic_nodes_mask = reshaped_observed_relic_nodes_mask - 1
        sym_observed_relic_node_positions = self.fixed_env_params.map_width - 1 - observed_relic_node_positions[:, [1,0]]
        sym_observed_relic_node_positions = sym_observed_relic_node_positions * reshaped_observed_relic_nodes_mask + inverse_negative_reshaped_observed_relic_nodes_mask

        half_max_num_relics = self.fixed_env_params.max_relic_nodes // 2
        flipped_sym_observed_relic_node_positions = jnp.concatenate([sym_observed_relic_node_positions[half_max_num_relics:, :], sym_observed_relic_node_positions[:half_max_num_relics, :]], axis=0)
        observed_relic_node_positions_with_symmetry = jnp.maximum(observed_relic_node_positions, flipped_sym_observed_relic_node_positions)
        
        flipped_observed_relic_nodes_mask = jnp.concatenate([observed_relic_nodes_mask[half_max_num_relics:], observed_relic_nodes_mask[:half_max_num_relics]], axis=0)
        observed_relic_nodes_mask_with_symmetry = observed_relic_nodes_mask | flipped_observed_relic_nodes_mask

        discovered_relic_nodes_mask = state.discovered_relic_nodes_mask | observed_relic_nodes_mask_with_symmetry
        discovered_relic_node_positions = jnp.maximum(state.discovered_relic_node_positions, observed_relic_node_positions_with_symmetry)

        num_relics_discovered = discovered_relic_nodes_mask.sum()

        # state.py:348 shows spawn schedule.  2 spawn 0.50 with 100%, 2 spawn 100-150 with 66%, 2 spawn 200-250 with 33%
        max_steps_in_episode = self.fixed_env_params.max_steps_in_match * self.fixed_env_params.match_count_per_episode
        normalized_steps =  obs.steps / float(max_steps_in_episode + self.fixed_env_params.match_count_per_episode)
        normalized_match_steps =  obs.match_steps / float(max_steps_in_episode + 1)

        spawn_steps = obs.steps // (self.fixed_env_params.max_steps_in_match // 2)
        max_relics = jnp.min(jnp.array([(1 + (obs.steps // self.fixed_env_params.max_steps_in_match)) * 2, self.fixed_env_params.max_relic_nodes]))
        num_relics_undiscovered = max_relics - num_relics_discovered
        relic_map = self.compute_counts_map(discovered_relic_node_positions, discovered_relic_nodes_mask)
        
        # 1) calculate last visit history
        current_step_last_visit = obs.sensor_mask * obs.steps
        sensor_mask_inverse = 1 - obs.sensor_mask
        sensor_mask = jnp.array(obs.sensor_mask)
        sensor_mask_symmetrical = jnp.logical_or(sensor_mask , sensor_mask[::-1, ::-1].T)
        
        sensor_last_visit = state.sensor_last_visit * sensor_mask_inverse # will set last_visit  to 0 wherever the sensors are visible
        sensor_last_visit = sensor_last_visit + current_step_last_visit # will set last_visit to obs.match_steps wherever the sensors are visible
        sensor_last_visit_symmetrical = jnp.maximum(sensor_last_visit, sensor_last_visit[::-1, ::-1].T) # Relics are spawned symetrricallys, so the sensor visit from the POV of the relics can be symmetrical
        sensor_last_visit_normalized = sensor_last_visit / float(max_steps_in_episode + self.fixed_env_params.match_count_per_episode)

        last_relic_spawn_point = (self.fixed_env_params.max_steps_in_match * 5) // 2 # state.py:348 shows spawn schedule
        has_seen_everywhere_after_match_1 = jnp.all(sensor_last_visit_symmetrical > last_relic_spawn_point)
        num_relics_undiscovered = num_relics_undiscovered * (1 - has_seen_everywhere_after_match_1) # once we've seen everywhere we know there are no more relics to discover

        # 3) calculate probabilty of non-observable spaces having a relic:
        #       # of relics undiscovered: discovered relics - max_relic_nodes
        #       # probability a relic spawned there since visiting: / (undiscovered  / 100-time_since_last_visit) / # unobserved spaces
        # This calculates how much time has passed since we visited this space, capped to the 100 time step.  If a space was visited after 100, we clip it to 0 since there's
        # no chance a relic has spawned after that
        # for i in range(2):
        #     steps_in_match = self.fixed_env_params.max_steps_in_match
        #     half_steps_in_match = steps_in_match // 2
        #     first_relic_possible_spawn_time = steps_in_match * i
        #     last_relic_possible_spawn_time = steps_in_match * i + half_steps_in_match
        #     time_elapsed_during_this_spawn =  jnp.max(jnp.array([jnp.min(jnp.array([last_relic_possible_spawn_time, obs.step])) - first_relic_possible_spawn_time, 0]))  # obs.step capped at 150, - 100, then minimum 0

        #     # effectively repeat the same caclulation but on a grid
        #     sensor_last_visit_relevant_during_this_spawn_time = jnp.max(jnp.array([sensor_last_visit, first_relic_possible_spawn_time]), axis=0) # Will this broadcast correctly?
        #     unobserved_time_during_n_relic_spawn = jnp.min(jnp.array([last_relic_possible_spawn_time, obs.step])) - sensor_last_visit_relevant_during_this_spawn_time # capped at 50 because unobserved time after that is irrellevant
        #     unobserved_time_during_n_relic_spawn = jnp.clip(unobserved_time_during_n_relic_spawn, min=0, max=half_steps_in_match) # capped at 50
            

        #     probabilty_relic_has_spawned = ((3-i) / 3.) * (time_elapsed_during_this_spawn.astype(jnp.float32) / half_steps_in_match) # each spawn period has 1/3 less chance of spawning, multiplied by how much time is in this period
        #     expected_relics_during_this_spawn_period = 2. * probabilty_relic_has_spawned

        #     relative_probability_of_space_spawning_relic = unobserved_time_during_n_relic_spawn / (unobserved_time_during_n_relic_spawn.sum() + 1e-9) # normalize 
        #     grid_expected_num_of_relics_spawned_since_last_observation = relative_probability_of_space_spawning_relic * num_relics_undiscovered
        #     grid_probability_of_relic_spawned_and_contributing_to_energy_point_since_last_observation = grid_expected_num_of_relics_spawned_since_last_observation / 5.



        unobserved_time_during_first_relic_spawn = jnp.min(jnp.array([50, obs.steps])) - sensor_last_visit_symmetrical # capped at 50 because unobserved time after that is irrellevant
        unobserved_time_during_first_relic_spawn = jnp.clip(unobserved_time_during_first_relic_spawn, min=0, max=50) # capped at 50

        # if obs.steps > 100, make this 0
        second_relic_spawn_conditional_start_time = jax.lax.cond(obs.steps > 100, lambda: obs.steps, lambda: -1)
        unobserved_time_during_second_relic_spawn = jnp.min(jnp.array([150, second_relic_spawn_conditional_start_time])) - sensor_last_visit_symmetrical # capped at 150 because unobserved time after that is irrellevant
        unobserved_time_during_second_relic_spawn = jnp.clip(unobserved_time_during_second_relic_spawn, min=0, max=50) # capped at 50

        # if obs.steps > 200, make this 0
        third_relic_spawn_conditional_start_time = jax.lax.cond(obs.steps > 200, lambda: obs.steps, lambda: -1)
        unobserved_time_during_third_relic_spawn = jnp.min(jnp.array([250, third_relic_spawn_conditional_start_time])) - sensor_last_visit_symmetrical # capped at 150 because unobserved time after that is irrellevant
        unobserved_time_during_third_relic_spawn = jnp.clip(unobserved_time_during_third_relic_spawn, min=0, max=50) # capped at 50
        
        # Lower chances of later relics being spawned
        probability_weighted_unobserved_spawn_time_since_last_observation = (unobserved_time_during_first_relic_spawn) + (0.66 * unobserved_time_during_second_relic_spawn) + (0.33 * unobserved_time_during_third_relic_spawn)
        relative_probability_of_space_spawning_relic = probability_weighted_unobserved_spawn_time_since_last_observation / (probability_weighted_unobserved_spawn_time_since_last_observation.sum() + 1e-9) # normalize 
        grid_expected_num_of_relics_spawned_since_last_observation = relative_probability_of_space_spawning_relic * num_relics_undiscovered

        # 100% chance of spawning if 
        grid_expected_num_of_relics_spawned_since_last_observation = jnp.where(relic_map > 0, relic_map, grid_expected_num_of_relics_spawned_since_last_observation)
        grid_probability_of_relic_spawned_and_contributing_to_energy_point_since_last_observation = grid_expected_num_of_relics_spawned_since_last_observation / 5. # staet.py:304 makes 20% of tiles valid 

        # 4) calculate probability of every space being an energy point
        #       # 5x5 covulution of 1-p of all spaces around it
        grid_probability_of_relic_not_spawned_and_contributing_to_energy_point_since_last_observation = (1. - grid_probability_of_relic_spawned_and_contributing_to_energy_point_since_last_observation)
        grid_probability_of_being_energy_point_based_on_relic_positions = 1 - self.convolution_for_relic_nodes_5x5(grid_probability_of_relic_not_spawned_and_contributing_to_energy_point_since_last_observation)

        # 5) calculate probabily of not being an energy point based on reward obs
        #       init: 1
        #       (1-(reward/total units)))^units_on_spot) * prev value

        unit_mask = jnp.array(obs.units_mask[team_id]) # shape (max_units, )
        unit_positions = jnp.array(obs.units.position[team_id]) # shape (max_units, 2)
        unit_energys = jnp.array(obs.units.energy[team_id]) # shape (max_units, 1)
        unit_mask = unit_mask & (unit_energys >= 0)

        normalized_unit_positions = unit_positions / float(self.fixed_env_params.map_width)
        normalized_unit_energys = unit_energys / float(self.fixed_env_params.max_unit_energy)
        unit_energys_max_grid = self.compute_energy_map_max(unit_positions, unit_mask, unit_energys) 
        normalized_unit_energys_max_grid = unit_energys_max_grid.astype(jnp.float32) / self.fixed_env_params.max_unit_energy

        unit_counts = self.compute_counts_map(unit_positions, unit_mask)
        grid_unit_mask = jnp.clip(unit_counts, max=1)
        inverse_grid_unit_mask = 1 - grid_unit_mask
        normalized_unit_counts = unit_counts / float(self.fixed_env_params.max_units)
        accumulated_points_last_round = self.extract_differece_points_for_player(obs, state, team_id)
        unit_mask_opp = jnp.array(obs.units_mask[opp_team_id]) # shape (max_units, )
        unit_energys_opp = jnp.array(obs.units.energy[opp_team_id]) # shape (max_units, 1)
        unit_mask_opp = unit_mask_opp & (unit_energys_opp >= 0)

        unit_positions_opp = jnp.array(obs.units.position[opp_team_id]) # shape (max_units, 2)        
        unit_counts_opp = self.compute_counts_map(unit_positions_opp, unit_mask_opp)
        normalized_unit_counts_opp = unit_counts_opp / float(self.fixed_env_params.max_units)
        accumulated_points_last_round_opp = self.extract_differece_points_for_player(obs, state, opp_team_id)
        unit_energys_max_grid_opp = self.compute_energy_map_max(unit_positions_opp, unit_mask_opp, unit_energys_opp)

        diff_dot_product = jnp.full(2, -1, dtype=jnp.int16)
        diff_dot_product = diff_dot_product.at[team_id].set(1)
 
        difference_points_normalized = jnp.dot(obs.team_points, diff_dot_product) / (self.fixed_env_params.max_units * self.fixed_env_params.max_steps_in_match / 8)
        difference_wins_normalized = jnp.dot(obs.team_wins, diff_dot_product) / self.fixed_env_params.match_count_per_episode
        
    
        normalized_unit_energys_max_grid_opp = unit_energys_max_grid_opp.astype(jnp.float32) / self.fixed_env_params.max_unit_energy

        # energys can be negative, but unknown is coded as -1.  Make them even more negative to help differentiate and avoid confusion
        energy_field_with_extra_mask = obs.map_features.energy - (sensor_mask_inverse * self.fixed_env_params.max_energy_per_tile)
        #symmetrical_energy_field = jnp.maximum(energy_field_with_extra_mask, energy_field_with_extra_mask[::-1, ::-1].T) 
        #we don't pass in a symmetric sensor mask, so it could be confusing.  I dont see how we'd use the symmetry though so whatever
        normalized_energy_field = energy_field_with_extra_mask.astype(jnp.float32) / self.fixed_env_params.max_energy_per_tile

        tile_type = jnp.array(obs.map_features.tile_type)
        symmetrical_tile_type = jnp.maximum(tile_type, tile_type[::-1, ::-1].T)



        # Code taken from underlying env
        # Shift objects around in space
        # Move the nebula tiles in state.map_features.tile_types up by 1 and to the right by 1
        # this is also symmetric nebula tile movement # 
        # Drift speed is not known!!!
        

        def two_maps_agree(map1, map2):
            map1_mask = map1 != -1
            map2_mask = map2 != -1
            return ~(((map1 != map2) & map1_mask & map2_mask).any())
        
        observed_different = ~(two_maps_agree(state.symmetrical_tile_type_last_round, symmetrical_tile_type))

        # obs.steps == 41 with 0.025, 21 with 0.05, 8 with 0.15, 11 with 0.1, 
        def calculate_drift_speed():
            # if it misses the first drift then the entire thing goes to hell
            drift_speed = (40 // (jnp.maximum(1, (obs.steps - 2)) % 40)) * 0.025  
            positive_drifted_map = self.drift_map(state.symmetrical_tile_type_last_round, obs.steps - 1, drift_speed)
            negative_drifted_map = self.drift_map(state.symmetrical_tile_type_last_round, obs.steps - 1, -1. * drift_speed)
            positive_drift_confirmation = two_maps_agree(positive_drifted_map, symmetrical_tile_type).astype(jnp.float32)
            negative_drift_confirmation = two_maps_agree(negative_drifted_map, symmetrical_tile_type).astype(jnp.float32)
            not_both_confirmed = 1. - (positive_drift_confirmation * negative_drift_confirmation)
            # drift speed is not 0, so this helps us with the unknown case, represented here as 0
            return ((drift_speed * positive_drift_confirmation) + (drift_speed * negative_drift_confirmation * -1.)) * not_both_confirmed

        
        drift_speed_guess = jax.lax.cond((obs.steps > 7) & observed_different, calculate_drift_speed, lambda: state.drift_speed_guess)
        
        # Only drift the map from last round if we guessed a new drift speed
        drifted_symmetrical_tile_type_last_round = jax.lax.cond((obs.steps > 7) & observed_different, 
                                                                lambda: self.drift_map(state.symmetrical_tile_type_last_round, obs.steps - 1, drift_speed_guess), 
                                                                lambda: state.symmetrical_tile_type_last_round)

        drifted_confirmed_with_observed = two_maps_agree(drifted_symmetrical_tile_type_last_round, symmetrical_tile_type)
        combined_stateful_map_and_current_obs = jnp.maximum(symmetrical_tile_type, drifted_symmetrical_tile_type_last_round)
        symmetrical_tile_type_next_round =  self.drift_map(combined_stateful_map_and_current_obs, obs.steps, drift_speed_guess)

        #jax.debug.print("drifted_confirmed_with_observed: {}, step: {}, params: {}", drifted_confirmed_with_observed, obs.steps, drift_speed_guess)

        # this should only happen if somehow on the 8th step, there was a drift but it was unobserved.  
        symmetrical_tile_type, symmetrical_tile_type_next_round  = jax.lax.cond(
            drifted_confirmed_with_observed, 
            lambda: (symmetrical_tile_type.astype(jnp.int16), symmetrical_tile_type_next_round.astype(jnp.int16)),
            lambda: (symmetrical_tile_type.astype(jnp.int16), symmetrical_tile_type.astype(jnp.int16))
        )

        match_over = self.is_match_over(obs, state)
        
        # If there are no points, we need to do probability all relics being spawned



        grid_unit_mask_float = grid_unit_mask.astype(jnp.float32)
        inverse_grid_unit_mask_float = 1. - grid_unit_mask_float
   
        # MAYBE: account for actual relic discovereies
        
        total_unique_positions_occupied = jnp.max(jnp.array([grid_unit_mask_float.sum(), 1.]))
        probability_of_any_unit_being_on_energy_point = jax.lax.cond(match_over, lambda: 0., lambda: accumulated_points_last_round / total_unique_positions_occupied)
        grid_probability_of_being_an_energy_point_based_on_this_turns_positions = (grid_unit_mask_float * probability_of_any_unit_being_on_energy_point)

        total_rewards_when_positions_are_occupied = state.total_rewards_when_positions_are_occupied + grid_probability_of_being_an_energy_point_based_on_this_turns_positions
        grid_unit_mask_if_match_not_over = jax.lax.cond(match_over, lambda: jnp.zeros_like(grid_unit_mask), lambda: grid_unit_mask)
        total_times_positions_are_occupied = state.total_times_positions_are_occupied + grid_unit_mask_if_match_not_over # should this account for relic spawning?

        grid_max_probability_of_being_an_energy_point_based_on_positive_rewards = jnp.max(jnp.array([state.grid_max_probability_of_being_an_energy_point_based_on_positive_rewards, grid_probability_of_being_an_energy_point_based_on_this_turns_positions]), axis=0)


        last_time_sure_no_energy_point = state.last_time_sure_no_energy_point * inverse_grid_unit_mask
        last_time_sure_no_energy_point = last_time_sure_no_energy_point + (grid_unit_mask * obs.steps)
        last_time_sure_no_energy_point = jax.lax.cond(
            jnp.logical_or(accumulated_points_last_round, match_over), 
            lambda: state.last_time_sure_no_energy_point, 
            lambda: last_time_sure_no_energy_point
        )

        grid_probability_of_being_an_energy_point_based_on_no_reward = jnp.clip(last_time_sure_no_energy_point.astype(jnp.float32) / last_relic_spawn_point, max=1., min=0.)

        # grid_probability_of_being_an_energy_point_based_on_no_reward = jnp.zeros_like(last_time_sure_no_energy_point, dtype=jnp.float32)
        # for i in range(2):
        #     steps_in_match = self.fixed_env_params.max_steps_in_match
        #     half_steps_in_match = steps_in_match // 2
        #     first_relic_possible_spawn_time = steps_in_match * i
        #     last_relic_possible_spawn_time = steps_in_match * i + half_steps_in_match

        #     # effectively repeat the same caclulation but on a grid
        #     last_time_sure_no_energy_point_relevant_during_this_spawn_time = jnp.clip(last_time_sure_no_energy_point, min=first_relic_possible_spawn_time)
        #     unsure_time_of_no_energy_point_during_this_spawn_time = jnp.min(jnp.array([last_relic_possible_spawn_time, obs.steps])) - last_time_sure_no_energy_point_relevant_during_this_spawn_time # capped at 50 because unobserved time after that is irrellevant
        #     unsure_time_of_no_energy_point_during_this_spawn_time = jnp.clip(unsure_time_of_no_energy_point_during_this_spawn_time, min=0, max=half_steps_in_match) # capped at 50
        #     probability_of_no_energy_point_during_this_spawn_time = unsure_time_of_no_energy_point_during_this_spawn_time.astype(jnp.float32) / half_steps_in_match
        #     probability_adjustment_of_no_energy_point_during_this_spawn_time = probability_of_no_energy_point_during_this_spawn_time * (((3-i) / 3.) / 2.)
        #     grid_probability_of_being_an_energy_point_based_on_no_reward = grid_probability_of_being_an_energy_point_based_on_no_reward + probability_adjustment_of_no_energy_point_during_this_spawn_time


        grid_min_probability_of_being_an_energy_point_based_on_positive_rewards =  jax.lax.cond(
            jnp.logical_or(jnp.logical_not(accumulated_points_last_round), match_over), 
            lambda: state.grid_min_probability_of_being_an_energy_point_based_on_positive_rewards,
            lambda: jnp.min(jnp.array([state.grid_min_probability_of_being_an_energy_point_based_on_positive_rewards, grid_probability_of_being_an_energy_point_based_on_this_turns_positions]), axis=0)
        )
        
        grid_avg_probability_of_being_an_energy_point_based_on_positive_rewards = total_rewards_when_positions_are_occupied / total_times_positions_are_occupied
        

        # 6) combines by multiplying?
       

        # SOLVER!!!!
        
        stored_unit_masks_around_relics = state.stored_unit_masks_around_relics
        stored_rewards = state.stored_rewards
        solved_energy_points_grid_mask = state.solved_energy_points_grid_mask
        max_solved_certainty_params = state.max_solved_certainty_params


        accumulated_solved_energy_points_grid_mask = jnp.full_like(grid_unit_mask, False) # this is False where a spot is unknown.  We not it when we send to obs so we can 0 it out during pretraining

        if use_solver:
            for relic_number in range(half_max_num_relics):
                mask_around_relic = create_centered_mask(discovered_relic_node_positions[relic_number], n=self.fixed_env_params.map_width, dtype=jnp.bool) & discovered_relic_nodes_mask[relic_number] 
                mask_around_relic2 = create_centered_mask(discovered_relic_node_positions[relic_number+3], n=self.fixed_env_params.map_width, dtype=jnp.bool) & discovered_relic_nodes_mask[relic_number+3] 

                mask_overlap = mask_around_relic & mask_around_relic2
                grid_units_in_overlap = mask_overlap & grid_unit_mask



                # want to get the union of unit counts, but not double counting overlap

                mask_around_relic_symmetrical = mask_around_relic | mask_around_relic2
                
                
                grid_unit_count_for_rewards_calculation_symmetrical = grid_unit_mask + (grid_unit_mask[::-1, ::-1].T)
                # subtract overlap
                grid_unit_count_for_rewards_calculation_symmetrical = grid_unit_count_for_rewards_calculation_symmetrical - grid_units_in_overlap.astype(jnp.int32)

                inverse_mask_around_relic = 1 - mask_around_relic_symmetrical
                
                units_not_around_relic = inverse_mask_around_relic * grid_unit_mask
                # TODO: if a unit is definitely on an ep, we can still solve and subtract

                #probabilty_units_not_around_relic_are_on_ep = grid_probability_of_being_energy_point_based_on_relic_positions * units_not_around_relic # TODO: check other solved states!
                valid_to_store = discovered_relic_nodes_mask[relic_number] & (~match_over)# & (probabilty_units_not_around_relic_are_on_ep.sum() < 0.01) 
                valid_to_store = valid_to_store.astype(jnp.int32)

                positions_as_32_bit = extract_32bit_from_grid_mask(grid_unit_count_for_rewards_calculation_symmetrical, discovered_relic_node_positions[relic_number])

                def solver():
                    # TODO: LOOK FOR THE BIGGEST NUMBER AND LOG IT FOR THE NUMBER OF PERMS
                    # TODO: remove solved units.  Either mask them or do it later
    
                    stored_unit_masks_around_relics_updated = stored_unit_masks_around_relics.at[relic_number, obs.steps % 50, :].set(positions_as_32_bit * valid_to_store)
                    stored_rewards_updated = stored_rewards.at[relic_number, obs.steps % 50].set((accumulated_points_last_round) * valid_to_store)

                    # count number of items stored
                    number_of_stored_unit_masks = (stored_unit_masks_around_relics_updated[relic_number, :, :].sum(axis=1) > 0).sum()
                    unique_positions_occupied = (stored_unit_masks_around_relics_updated[relic_number, :, :].sum(axis=0) > 0).sum()

                    # a = stored_unit_masks_around_relics_updated[relic_number, :] > 0
                    # jax.device_get(stored_rewards_updated[relic_number, :][a])


                    solution, _, _, _ = jnp.linalg.lstsq(stored_unit_masks_around_relics_updated[relic_number, :, :], stored_rewards_updated[relic_number, :])
                    
                    # check if 

                    solved_certainty_params = number_of_stored_unit_masks * unique_positions_occupied * valid_to_store

                    solved_energy_points_grid_mask_for_this_relic = reconstruct_grid_from_subsection_bit_mask(solution, discovered_relic_node_positions[relic_number], 
                                                                                            self.fixed_env_params.map_width)
                    
                    solved_energy_points_grid_mask_for_this_relic = (solved_energy_points_grid_mask_for_this_relic > 0.99) & (solved_energy_points_grid_mask_for_this_relic < 1.01)
                    
                    # TODO, check if this solution is valid by seeing if anything is over 1, if there is enough values passed in, if enough unique bit positions are covered ...
                    solved_energy_points_grid_mask_for_this_relic_symmetrical = jnp.logical_or(solved_energy_points_grid_mask_for_this_relic , solved_energy_points_grid_mask_for_this_relic[::-1, ::-1].T)
                    
                    
                    solved_energy_points_grid_mask_updated, max_solved_certainty_params_updated = jax.lax.cond(
                        solved_certainty_params > max_solved_certainty_params[relic_number],
                        lambda: (solved_energy_points_grid_mask.at[relic_number, :, :].set(solved_energy_points_grid_mask_for_this_relic_symmetrical), 
                                 max_solved_certainty_params.at[relic_number].set(solved_certainty_params)),
                        lambda: (solved_energy_points_grid_mask, max_solved_certainty_params)

                    )
                                    
                    
                    return (stored_unit_masks_around_relics_updated, stored_rewards_updated, solved_energy_points_grid_mask_updated, max_solved_certainty_params_updated)

                stored_unit_masks_around_relics, stored_rewards, solved_energy_points_grid_mask, max_solved_certainty_params = jax.lax.cond(
                    valid_to_store,
                    solver,
                    lambda: (stored_unit_masks_around_relics, stored_rewards, solved_energy_points_grid_mask, max_solved_certainty_params)
                )

                # This accumulated in the for loop since later relics can have unsolved spots that overlap with previously relics.  
                # True where we are certain of the value, false otherwise
                # If a later relic gets spawned, it never removes a previous relic
                accumulated_solved_energy_points_grid_mask = accumulated_solved_energy_points_grid_mask | solved_energy_points_grid_mask[relic_number]
                

        # TODO: account for units getting killed, check energy?

        # Figure out sapping locations
        unit_mask_opp_last_round_stacked = jnp.stack([state.unit_mask_opp_last_round, state.unit_mask_opp_last_round], axis=1).astype(jnp.int16)
        unit_positions_opp_diff_last_round = unit_mask_opp_last_round_stacked * (unit_positions_opp - state.unit_positions_opp_last_round)
        unit_positions_opp_predicted = unit_positions_opp + unit_positions_opp_diff_last_round
       
       # TODO: account for units moving towards EP?
       # TODO: remove extra kill parameter, let it learn ...
        unit_positions_opp_predicted = jnp.clip(unit_positions_opp_predicted, min=0, max=self.fixed_env_params.map_width)
        # number_units_killed_direct_hit = (unit_energys_opp < params.unit_sap_cost) & unit_mask_opp
        # number_units_killed_direct_hit_grid = self.compute_counts_map(unit_positions_opp_predicted, number_units_killed_direct_hit)

        # number_units_killed_adjacent_hit = (unit_energys_opp < (params.unit_sap_cost * params.unit_sap_dropoff_factor)) & unit_mask_opp
        # number_units_killed_adjacent_hit_grid = self.compute_counts_map(unit_positions_opp_predicted, number_units_killed_adjacent_hit)
        # number_units_killed_adjacent_hit_summed_grid = self.convolution_for_sap_actions_3x3(number_units_killed_adjacent_hit_grid.astype(jnp.float32), 1, 0, jnp.float32) #conv on nvidia must be float32?

        potential_energy_taken_opp = jnp.clip(unit_energys_opp, max=params.unit_sap_cost, min=0)
        unit_energy_potentially_taken_grid_opp = self.compute_energy_map_sum(unit_positions_opp_predicted, unit_mask_opp, potential_energy_taken_opp)
        unit_energy_potentially_taken_sum_grid_opp = self.convolution_for_sap_actions_3x3(unit_energy_potentially_taken_grid_opp.astype(jnp.float32), params.unit_sap_dropoff_factor, 1., jnp.float32)
        unit_energy_potentially_taken_sum_grid_opp_normalized = unit_energy_potentially_taken_sum_grid_opp / params.unit_sap_cost

        # effectively killing a unit is 1 value, and reducing points is 1/params.unit_sap_cost value per point
        # todo: normalize
        value_of_sapping_grid = (unit_energy_potentially_taken_sum_grid_opp_normalized) / 9.

        candidate_sap_locations_x, candidate_sap_locations_y, valid_saps = debuggable_vmap(self.find_max_index_subsection_for_sap_ranges, in_axes=(None, 0, None), out_axes=0)(
            value_of_sapping_grid, unit_positions, params.unit_sap_range
        )
        candidate_sap_locations = jnp.column_stack((candidate_sap_locations_x, candidate_sap_locations_y))
        candidate_sap_locations = candidate_sap_locations - unit_positions # relative to current position
        #go through each unit
        # mask the range they can sap
        # find the higest point        
        action_mask_without_saps = self.action_mask(unit_positions, symmetrical_tile_type == ASTEROID_TILE)
        action_mask = jnp.concatenate([action_mask_without_saps, valid_saps[:, jnp.newaxis]], axis=1)

        #known params: ['unit_move_cost', 'unit_sap_cost', 'unit_sap_range', 'unit_sensor_range'])
        param_list = jnp.array([
            params.unit_move_cost / float(env_params_ranges["unit_move_cost"][-1]),
            params.unit_sensor_range / float(env_params_ranges["unit_sensor_range"][-1]),
            params.unit_sap_cost / float(env_params_ranges["unit_sap_cost"][-1]),
            params.unit_sap_range / float(env_params_ranges["unit_sap_range"][-1]),
            difference_points_normalized,
            difference_wins_normalized,
            normalized_steps,
            normalized_match_steps,
            drift_speed_guess # already normalized ...
        ])


        def anneal(full_strength):
            stopping_point = jnp.ceil(float(self.total_updates) * full_strength)
            return jnp.minimum(update_count, stopping_point) / stopping_point.astype(jnp.float32)

        annealed_solved_energy_points_grid_mask = (accumulated_solved_energy_points_grid_mask).astype(jnp.float32) * anneal(0.05)

        # add sensor last visit?
        new_observation = WrappedEnvObs(
            relic_map=relic_map.astype(jnp.float32), # not used
            normalized_unit_counts=normalized_unit_counts,
            normalized_unit_counts_opp=normalized_unit_counts_opp,
            normalized_unit_energys_max_grid=normalized_unit_energys_max_grid,
            normalized_unit_energys_max_grid_opp=normalized_unit_energys_max_grid_opp,
            tile_type=symmetrical_tile_type_next_round, #have movements be forward looking
            normalized_energy_field=normalized_energy_field,
            unit_positions=unit_positions,
            normalized_unit_positions=normalized_unit_positions,
            normalized_unit_energys=normalized_unit_energys,
            unit_mask=unit_mask,
            sensor_mask=sensor_mask.astype(jnp.float32),
            sensor_last_visit_normalized=sensor_last_visit_normalized,
            grid_probability_of_being_energy_point_based_on_relic_positions=grid_probability_of_being_energy_point_based_on_relic_positions,
            grid_max_probability_of_being_an_energy_point_based_on_positive_rewards=grid_max_probability_of_being_an_energy_point_based_on_positive_rewards,
            grid_min_probability_of_being_an_energy_point_based_on_positive_rewards=grid_min_probability_of_being_an_energy_point_based_on_positive_rewards,
            grid_avg_probability_of_being_an_energy_point_based_on_positive_rewards=grid_avg_probability_of_being_an_energy_point_based_on_positive_rewards,
            grid_probability_of_being_an_energy_point_based_on_no_reward=grid_probability_of_being_an_energy_point_based_on_no_reward,
            value_of_sapping_grid=value_of_sapping_grid,
            action_mask=action_mask,
            param_list=param_list,
            solved_energy_points_grid_mask=annealed_solved_energy_points_grid_mask,
        )
        
        new_state = StatefulEnvState(discovered_relic_node_positions=discovered_relic_node_positions,
                                     discovered_relic_nodes_mask=discovered_relic_nodes_mask,
                                     team_wins=obs.team_wins,
                                     team_points=obs.team_points,
                                     last_time_sure_no_energy_point=last_time_sure_no_energy_point,
                                     grid_max_probability_of_being_an_energy_point_based_on_positive_rewards=grid_max_probability_of_being_an_energy_point_based_on_positive_rewards,
                                     grid_min_probability_of_being_an_energy_point_based_on_positive_rewards=grid_min_probability_of_being_an_energy_point_based_on_positive_rewards,
                                     total_rewards_when_positions_are_occupied=total_rewards_when_positions_are_occupied,
                                     total_times_positions_are_occupied=total_times_positions_are_occupied,
                                     sensor_last_visit=sensor_last_visit,
                                     symmetrical_tile_type_last_round=symmetrical_tile_type,
                                     unit_positions_opp_last_round=unit_positions_opp,
                                     unit_mask_opp_last_round=unit_mask_opp,
                                     candidate_sap_locations=candidate_sap_locations,
                                     drift_speed_guess=drift_speed_guess,
                                     stored_unit_masks_around_relics=stored_unit_masks_around_relics,
                                     stored_rewards=stored_rewards,
                                     solved_energy_points_grid_mask=solved_energy_points_grid_mask,
                                     max_solved_certainty_params=max_solved_certainty_params,
                                    )
        
        reward_info = RewardInfo(unit_counts=unit_counts, accumulated_solved_energy_points_grid_mask=accumulated_solved_energy_points_grid_mask)
        
        return new_observation, new_state, reward_info
    
    
    def action_mask(self, unit_positions, asteroid_grid):
        
        asteroid_pos = jnp.argwhere(asteroid_grid, size=self.fixed_env_params.map_height * self.fixed_env_params.map_height)


        action_diffs = jnp.array([
           # [0, 0],  # Do nothing
            [0, -1],  # Move up
            [1, 0],  # Move right
            [0, 1],  # Move down
            [-1, 0],  # Move left
        ])

        # Each row represents a unit, then 5 copies of the coordinates each representing an action
        unit_3d = jnp.stack([unit_positions, unit_positions, unit_positions, unit_positions], axis=1) + action_diffs 
        
        off_the_map = jnp.isin(unit_3d, jnp.array([-1, self.fixed_env_params.map_height]))
        action_mask = ~off_the_map.any(axis=2)

        # Each row represents a unit, then 5 copies of the coordinates each representing an action

        # Broadcast the two, adding dimensions for combinatorial explosion.  Axis were added to dimensions that I thought needed adding to?
        unit_4d = unit_3d[:, :, jnp.newaxis, :] == asteroid_pos[jnp.newaxis, jnp.newaxis, :, :]
        any_collisions = unit_4d.all(axis=3) # last dimension is of shape 2, representing x,y coordinates.  Need all to match for collision
        asteroid_mask = ~any_collisions.any(axis=2) # last dimension is of shape num_asteriods, and the action is invalid if it hits any asteroid

        action_mask = action_mask & asteroid_mask

        # do nothing is always valid
        return jnp.concatenate([jnp.full((self.fixed_env_params.max_units,1), True), action_mask], axis=1)


    def is_match_over(self, obs, old_state):
        new_team_wins = jnp.array(obs.team_wins)
        old_team_wins = jnp.array(old_state.team_wins)
        diff_wins = new_team_wins - old_team_wins

        return jnp.any(diff_wins)
    
    def extract_differece_points_for_player(self, obs, old_state, team_id):
        
        new_team_points = jnp.array(obs.team_points)
        new_team_wins = jnp.array(obs.team_wins)

        old_team_points = jnp.array(old_state.team_points)
        old_team_wins = jnp.array(old_state.team_wins)

        diff_points = new_team_points - old_team_points
        diff_wins = new_team_wins - old_team_wins

        point_dot_product = jnp.zeros(2, dtype=jnp.int16)
        point_dot_product = point_dot_product.at[team_id].set(1)


        # On steps mod 101, the points are wiped out but the wins change.
        # when the wins change we have no idea how many points we collected that round, or if they mattered, so count them as 0 ...
        diff_points_summed = jax.lax.cond(
            jnp.any(diff_wins),
            lambda: 0,
            lambda: jnp.dot(diff_points, point_dot_product)
        )

        return diff_points_summed
    
    
    def extract_manufactured_reward(self, new_state, old_state, team_id, opp_team_id, update_count, reward_info):

        new_team_points = jnp.array(new_state.team_points)
        new_team_wins = jnp.array(new_state.team_wins)

        old_team_points = jnp.array(old_state.team_points)
        old_team_wins = jnp.array(old_state.team_wins)

        diff_points = new_team_points - old_team_points
        diff_wins = new_team_wins - old_team_wins

        diff_dot_product = jnp.full(2, -1, dtype=jnp.int16)
        diff_dot_product = diff_dot_product.at[team_id].set(1)

        episode_over = new_team_wins.sum() == 5
        has_won = new_state.team_wins[team_id] > new_state.team_wins[opp_team_id]

        # On steps mod 101, the points are wiped out but the wins change.
        diff_points_summed = jax.lax.cond(
            jnp.any(diff_wins),
            lambda: 0,
            lambda: jnp.dot(diff_points, diff_dot_product)
        )
        diff_wins_summed = jax.lax.cond(
            episode_over,
            lambda: 0,
            lambda: jnp.dot(diff_wins, diff_dot_product)
        )

        match_win_summed = (episode_over).astype(jnp.int16) * (has_won.astype(jnp.int16) * 2 - 1) # win is 1, loss is -1

        # should you get awarded for jsut spawning?
        new_spaces_visited = (new_state.sensor_last_visit > -1).astype(jnp.int16).sum() - (old_state.sensor_last_visit > -1).astype(jnp.int16).sum()
        occupied_same_space = (reward_info.unit_counts > 1).astype(jnp.int16).sum()

        number_relics_discovered = new_state.discovered_relic_nodes_mask.sum() - old_state.discovered_relic_nodes_mask.sum()
        number_times_visited_unknown_spot = ((~reward_info.accumulated_solved_energy_points_grid_mask) & reward_info.unit_counts).sum()

        def progress_shape_rate(cutoff):
            stopping_point = float(self.total_updates) * cutoff
            return 1. - (jnp.minimum(update_count, stopping_point) / stopping_point)

        
        match_win_summed = match_win_summed * 20
        diff_wins_summed = diff_wins_summed * 0 #* (progress_shape_rate(0.8))
        diff_points_summed = diff_points_summed * 1# * (progress_shape_rate(0.9))
        number_relics_summed = number_relics_discovered * 10 * (progress_shape_rate(1))
        
        new_spaces_visited_summed = new_spaces_visited * 0.4 * (progress_shape_rate(0.5))
        occupied_same_space_summed = occupied_same_space * -0.4 * (progress_shape_rate(0.5))
        number_times_visited_unknown_spot_summed = number_times_visited_unknown_spot * (progress_shape_rate(0.7))
        
        reward = diff_points_summed + diff_wins_summed# + diff_points_summed + new_spaces_visited_summed + occupied_same_space_summed + number_relics_summed + number_times_visited_unknown_spot_summed


        return reward

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

    def reset(self, key, params=None, old_state=None):
        batch_count = 1
        if old_state is not None:
            obs, state = self._env.reset(key, params, old_state.env_state)
            state = NormalizeVecRewEnvState(env_state=state, mean=old_state.mean, var=old_state.var, count=old_state.count, return_val=old_state.return_val)
        else:
            obs, state = self._env.reset(key, params)
            state = NormalizeVecRewEnvState(
                mean=0.0,
                var=1.0,
                count=1e-4,
                return_val=jnp.zeros((batch_count,)),
                env_state=state,
            )
        return obs, state

    def step(self, key, state, action, params=None, update_count=None):
        obs, env_state, (reward_0, reward_1), done, info = self._env.step(
            key, state.env_state, action, params, update_count
        )

        reward_sum = jnp.absolute(jnp.array([reward_0, reward_1])).sum()
        return_val = state.return_val * self.gamma * (1 - done) + (reward_sum / 2.)

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

        divisor = jnp.sqrt(state.var + 1e-8)
        return obs, state, (reward_0 / divisor, reward_1 / divisor), done, info



@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None, old_state: Optional[LogEnvState] = None,
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        if old_state is not None:
             state = LogEnvState(
                env_state=env_state, # importantly keep the new env_state from the reset
                episode_returns=old_state.episode_returns,
                episode_lengths=old_state.episode_lengths,
                returned_episode_returns=old_state.returned_episode_returns,
                returned_episode_lengths=old_state.returned_episode_lengths,
                timestep=old_state.timestep,
             )
        else:
            state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
        update_count=None
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params, update_count
        )

        real_reward = info["real_reward"]
        new_episode_return = state.episode_returns + real_reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info
