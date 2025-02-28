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
    normalized_energy_field: chex.Array
    relic_map: chex.Array
    normalized_unit_counts: chex.Array
    normalized_unit_counts_opp: chex.Array
    normalized_unit_energys_max_grid: chex.Array
    normalized_unit_energys_max_grid_opp: chex.Array
    unit_positions: chex.Array
    unit_mask: chex.Array
    normalized_steps: float
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
    last_time_sure_no_energy_point: chex.Array
    grid_max_probability_of_being_an_energy_point_based_on_positive_rewards: chex.Array
    grid_min_probability_of_being_an_energy_point_based_on_positive_rewards: chex.Array
    total_rewards_when_positions_are_occupied: chex.Array
    total_times_positions_are_occupied: chex.Array

@struct.dataclass
class WrappedEnvState:
    original_state: EnvState
    stateful_data_0: StatefulEnvState
    stateful_data_1: StatefulEnvState

class LuxaiS3GymnaxWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)


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
            last_time_sure_no_energy_point=jnp.full((self.fixed_env_params.map_width, self.fixed_env_params.map_height), -1, dtype=jnp.int16),
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
        obs_0, stateful_data_0 = self.transform_obs(obs["player_0"], self.empty_stateful_env_state(), params, 0, 1)
        obs_1, stateful_data_1 = self.transform_obs(obs["player_1"], self.empty_stateful_env_state(), params, 1, 0)
        return (obs_0, obs_1), WrappedEnvState(original_state=state, stateful_data_0=stateful_data_0, stateful_data_1=stateful_data_1)
    
    


    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, wrapped_state, action, params=None):

        no_sap_action_0 = jnp.zeros((self.fixed_env_params.max_units, 3), dtype=jnp.int16)
        no_sap_action_0 = no_sap_action_0.at[:, 0].set(action[0])

        no_sap_action_1 = jnp.zeros((self.fixed_env_params.max_units, 3), dtype=jnp.int16)
        no_sap_action_1 = no_sap_action_1.at[:, 0].set(action[1])

        lux_action = {
            "player_0": no_sap_action_0, 
            "player_1": no_sap_action_1
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

        new_obs_0, stateful_data_0 = self.transform_obs(obs["player_0"], prev_stateful_data_0, params, 0, 1)
        new_obs_1, stateful_data_1 = self.transform_obs(obs["player_1"], prev_stateful_data_1, params, 1, 0)

        info["real_reward"] = self.extract_differece_points_for_player(stateful_data_0, prev_stateful_data_0, 0)

        manufactured_reward_0 = self.extract_manufactured_reward(stateful_data_0, prev_stateful_data_0, 0)
        manufactured_reward_1 = self.extract_manufactured_reward(stateful_data_1, prev_stateful_data_1, 1)

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
    

    def compute_counts_map(self, position, mask):
        partial_sum = partial(jnp.sum, dtype=jnp.int16)
        return self.compute_map(position, mask.astype(jnp.int16), partial_sum, jnp.int16)
    

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
    def transform_obs(self, obs: EnvObs, state: StatefulEnvState, params, team_id, opp_team_id):
        total_spaces = self.fixed_env_params.map_width * self.fixed_env_params.map_height
        observed_relic_node_positions = jnp.array(obs.relic_nodes) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = jnp.array(obs.relic_nodes_mask) # shape (max_relic_nodes, )

        discovered_relic_nodes_mask = state.discovered_relic_nodes_mask | observed_relic_nodes_mask
        discovered_relic_node_positions = jnp.maximum(state.discovered_relic_node_positions, observed_relic_node_positions)
        
        num_relics_discovered = discovered_relic_nodes_mask.sum()

        # state.py:348 shows spawn schedule.  2 spawn 0.50 with 100%, 2 spawn 100-150 with 66%, 2 spawn 200-250 with 33%
        normalized_steps =  obs.steps / float(self.fixed_env_params.max_steps_in_episode + self.fixed_env_params.match_count_per_episode)
        spawn_steps = obs.steps // (self.fixed_env_params.max_steps_in_match // 2)
        max_relics = jnp.min(jnp.array([(1 + (obs.steps // self.fixed_env_params.max_steps_in_match)) * 2, self.fixed_env_params.max_relic_nodes]))
        num_relics_undiscovered = max_relics - num_relics_discovered
        relic_map = self.compute_counts_map(discovered_relic_node_positions, discovered_relic_nodes_mask)
        
        # 1) calculate last visit history
        current_step_last_visit = obs.sensor_mask * obs.steps
        sensor_mask_inverse = 1 - obs.sensor_mask
        
        sensor_last_visit = state.sensor_last_visit * sensor_mask_inverse # will set last_visit  to 0 wherever the sensors are visible
        sensor_last_visit = sensor_last_visit + current_step_last_visit # will set last_visit to obs.match_steps wherever the sensors are visible

        last_relic_spawn_point = (self.fixed_env_params.max_steps_in_match * 5) // 2 # state.py:348 shows spawn schedule
        has_seen_everywhere_after_match_1 = jnp.all(sensor_last_visit > last_relic_spawn_point)
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



        unobserved_time_during_first_relic_spawn = jnp.min(jnp.array([50, obs.steps])) - sensor_last_visit # capped at 50 because unobserved time after that is irrellevant
        unobserved_time_during_first_relic_spawn = jnp.clip(unobserved_time_during_first_relic_spawn, min=0, max=50) # capped at 50

        # if obs.steps > 100, make this 0
        second_relic_spawn_conditional_start_time = jax.lax.cond(obs.steps > 100, lambda: obs.steps, lambda: -1)
        unobserved_time_during_second_relic_spawn = jnp.min(jnp.array([150, second_relic_spawn_conditional_start_time])) - sensor_last_visit # capped at 150 because unobserved time after that is irrellevant
        unobserved_time_during_second_relic_spawn = jnp.clip(unobserved_time_during_second_relic_spawn, min=0, max=50) # capped at 50

        # if obs.steps > 200, make this 0
        third_relic_spawn_conditional_start_time = jax.lax.cond(obs.steps > 200, lambda: obs.steps, lambda: -1)
        unobserved_time_during_third_relic_spawn = jnp.min(jnp.array([250, third_relic_spawn_conditional_start_time])) - sensor_last_visit # capped at 150 because unobserved time after that is irrellevant
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
        grid_probability_of_being_energy_point_based_on_relic_positions = 1 - self.multiply_5x5_mask_log_conv(grid_probability_of_relic_not_spawned_and_contributing_to_energy_point_since_last_observation)

        # 5) calculate probabily of not being an energy point based on reward obs
        #       init: 1
        #       (1-(reward/total units)))^units_on_spot) * prev value

        unit_mask = jnp.array(obs.units_mask[team_id]) # shape (max_units, )
        unit_positions = jnp.array(obs.units.position[team_id]) # shape (max_units, 2)
        unit_energys = jnp.array(obs.units.energy[team_id]) # shape (max_units, 1)
        unit_energys_max_grid = self.compute_map(unit_positions, unit_energys, jnp.max, jnp.int16)
        normalized_unit_energys_max_grid = unit_energys_max_grid.astype(jnp.float32) / self.fixed_env_params.max_unit_energy

        unit_counts = self.compute_counts_map(unit_positions, unit_mask)
        normalized_unit_counts = unit_counts / float(self.fixed_env_params.max_units)
        accumulated_points_last_round = self.extract_differece_points_for_player(obs, state, team_id)


        unit_mask_opp = jnp.array(obs.units_mask[opp_team_id]) # shape (max_units, )
        unit_positions_opp = jnp.array(obs.units.position[opp_team_id]) # shape (max_units, 2)
        unit_counts_opp = self.compute_counts_map(unit_positions_opp, unit_mask_opp)
        normalized_unit_counts_opp = unit_counts_opp / float(self.fixed_env_params.max_units)
        accumulated_points_last_round_opp = self.extract_differece_points_for_player(obs, state, opp_team_id)
        unit_energys_opp = jnp.array(obs.units.energy[opp_team_id]) # shape (max_units, 1)
        unit_energys_max_grid_opp = self.compute_map(unit_positions_opp, unit_energys_opp, jnp.max, jnp.int16)
        normalized_unit_energys_max_grid_opp = unit_energys_max_grid_opp.astype(jnp.float32) / self.fixed_env_params.max_unit_energy

        normalized_energy_field = obs.map_features.energy.astype(jnp.float32) / self.fixed_env_params.max_energy_per_tile

        match_over = self.is_match_over(obs, state)
        
        # If there are no points, we need to do probability all relics being spawned
        grid_unit_mask = jnp.clip(unit_counts, max=1)
        inverse_grid_unit_mask = 1 - grid_unit_mask


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


        # TODO: account for units getting killed, check energy?



        new_observation = WrappedEnvObs(
            relic_map=relic_map, # not used
            normalized_unit_counts=normalized_unit_counts,
            normalized_unit_counts_opp=normalized_unit_counts_opp,
            normalized_unit_energys_max_grid=normalized_unit_energys_max_grid,
            normalized_unit_energys_max_grid_opp=normalized_unit_energys_max_grid_opp,
            tile_type=jnp.array(obs.map_features.tile_type),
            normalized_energy_field=normalized_energy_field,
            unit_positions=unit_positions,
            unit_mask=unit_mask,
            normalized_steps=normalized_steps,
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
                                     last_time_sure_no_energy_point=last_time_sure_no_energy_point,
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
    
    
    def extract_manufactured_reward(self, new_state, old_state, team_id):

        new_team_points = jnp.array(new_state.team_points)
        new_team_wins = jnp.array(new_state.team_wins)

        old_team_points = jnp.array(old_state.team_points)
        old_team_wins = jnp.array(old_state.team_wins)

        diff_points = new_team_points - old_team_points
        diff_wins = new_team_wins - old_team_wins

        #match = new_team_wins.sum()

        # TODO: Subtract points from other team from your score
        point_dot_product = jnp.full(2, -1, dtype=jnp.int16)
        point_dot_product = point_dot_product.at[team_id].set(1)

        # On steps mod 101, the points are wiped out but the wins change.
        diff_points_summed = jax.lax.cond(
            jnp.any(diff_wins),
            lambda: 0,
            lambda: jnp.dot(diff_points, point_dot_product)
        )

        return diff_points_summed
    

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
        obs, env_state, (reward_0, reward_1), done, info = self._env.step(
            key, state.env_state, action, params
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
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
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
