import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from luxai_s3.wrappers import LuxAIS3GymEnv

class SB3Wrapper(gym.Wrapper):
    def __init__(
        self,
        player: str,
        env: LuxAIS3GymEnv, 
        transformer
    ) -> None:
        gym.Wrapper.__init__(self, env)
        self.player = player
        self.env = env        
        self.action_space = spaces.MultiDiscrete([ 5 ] * env.env_params.max_units)
        self.transformer = transformer

    def step(self, action: npt.NDArray):
        
        # here, for each agent in the game we translate their action into a Lux S2 action
        
        single_player_lux_action = np.zeros((self.env.env_params.max_units, 3), dtype=int)
        single_player_lux_action[:, 0] = action
        lux_action = {
            "player_0": single_player_lux_action, 
            "player_1": np.zeros((self.env.env_params.max_units, 3), dtype=int)
        }
        
        # Completely ignore sapping
        obs, reward, terminated, truncated, info  = self.env.step(lux_action)

        manufactured_reward = float(self.transformer.extract_differece_points_player_0(obs[self.player]))

        single_player_terminated = terminated[self.player]
        single_player_truncated = truncated[self.player]

        info = dict()
        metrics = dict()
        #metrics["rewards"] = manufactured_reward

        # we can save the metrics to info so we can use tensorboard to log them to get a glimpse into how our agent is behaving
        info["metrics"] = metrics

        return obs, manufactured_reward, single_player_terminated, single_player_truncated, info
    

    @staticmethod
    def training_mask_wrapper(env):
        if (env.transformer.last_obs is None):
            return np.ones((env.env.env_params.max_units, 5), dtype=int)

        return ObservationTransformer.action_mask(env.env.env_params, env.transformer.last_obs)   
    
    



class ObservationWrapper(gym.ObservationWrapper):

    def __init__(self, player: str, env: gym.Env, transformer):
        super().__init__(env)
        self.player = player
        self.transformer = transformer
        self.observation_space = spaces.Box(0, 1, shape=(self.transformer.observation_space_size,))

    def observation(self, obs):
        player_obs = obs[self.player]
        return self.transformer.transform_observation(player_obs)

class ObservationTransformer:

    def __init__(self, player: str, env_cfg):
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        
        self.params = env_cfg

        self.discovered_relic_nodes_mask = np.zeros(self.params.max_relic_nodes, dtype=bool)
        self.discovered_relic_node_positions =  np.full((self.params.max_relic_nodes, 2), -1, dtype=np.int16)

        self.observation_space_size = (
            self.params.max_units * 3  + # one for masks, one for x pos, one for y pos
            self.params.max_relic_nodes * 3  + # one for masks, one for x pos, one for y pos
            1 # Difference in points from last round
        )
        self.last_obs = None
        self.last_reward = None
        self.last_diff_player_0_points = 0

        self.old_team_points = 0
        self.old_team_wins = 0
        
        
    def record_observation(self, obs):
        self.last_obs = obs

    def extract_differece_points_player_0(self, obs):
        new_team_points = obs["team_points"]
        new_team_wins = obs["team_wins"]

        old_team_points = self.last_obs["team_points"] if (self.last_obs is not None) else 0
        old_team_wins = self.last_obs["team_wins"] if (self.last_obs is not None) else 0
        
        diff_points = new_team_points - old_team_points
        diff_wins = new_team_wins - old_team_wins
        diff_wins_sum = np.sum(diff_wins)

        # Game or match is over so reset diff
        if diff_wins_sum != 0:
            diff_points = np.zeros(2)

        # Match is reset so diff_wins should be 0
        if diff_wins_sum < 0:
            diff_wins = np.zeros(2)
        
        # only record player 0 points
        # TODO(fix if trying to do multi agents)
        diff_points_player_0 = np.dot(diff_points, [1,0])

        return diff_points_player_0

    def transform_observation(self, obs):
        unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        #unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        #team_points = np.array(obs["team_points"]) # points of each team, team_points[self.team_id] is the points of the your team
        
        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        # visible relic nodes

        self.discovered_relic_nodes_mask = self.discovered_relic_nodes_mask | observed_relic_nodes_mask
        self.discovered_relic_node_positions = np.maximum(self.discovered_relic_node_positions, observed_relic_node_positions)

        norm_and_flat_unit_positions = ObservationTransformer.normalize_positions(unit_positions, self.params).flatten()
        norm_and_flat_relic_positions = ObservationTransformer.normalize_positions(
            self.discovered_relic_node_positions, self.params).flatten()

        # Pass in how many points we earned in the last round, normalized by max_units (total points possible) becaues this 
        # gives a signal that the last spot we were on might be a good spot.
        #if (obs["team_points"][0] > 0):
        #    print(f'points: {obs["team_points"][0]}')

        norm_last_diff_player_0_points = self.extract_differece_points_player_0(obs) / self.params.max_units
        #if norm_last_diff_player_0_points != 0:
        #    print(norm_last_diff_player_0_points)

        new_observation = np.concatenate(
                [unit_mask, norm_and_flat_unit_positions, self.discovered_relic_nodes_mask, norm_and_flat_relic_positions, [norm_last_diff_player_0_points]])
        
        self.record_observation(obs)
        return new_observation
    
    @staticmethod
    def action_mask(env_params, obs):
        
        unit_mask = np.array(obs["units_mask"][0]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][0]) # shape (max_units, 2)
        sensor_mask = np.array(obs["sensor_mask"]) # shape (W, H)
        asteroid_grid = np.array(obs["map_features"]["tile_type"]) == 2 # shape (W, H)
        asteroid_pos = np.argwhere(asteroid_grid)
        num_asteroids = len(asteroid_pos)

        action_diffs = [
            [0, 0],  # Do nothing
            [0, -1],  # Move up
            [1, 0],  # Move right
            [0, 1],  # Move down
            [-1, 0],  # Move left
        ]

        # Each row represents a unit, then 5 copies of the coordinates each representing an action
        unit_3d = np.stack([unit_positions] * 5, axis=1) + action_diffs 
        
        off_the_map = np.isin(unit_3d, [-1, env_params.map_height])
        mask = ~off_the_map.any(axis=2)

        # Each row represents a unit, then 5 copies of the coordinates each representing an action
        if num_asteroids > 0:
            # Broadcast the two, adding dimensions for combinatorial explosion.  Axis were added to dimensions that I thought needed adding to?
            unit_4d = unit_3d[:, :, np.newaxis, :] == asteroid_pos[np.newaxis, np.newaxis, :, :]
            any_collisions = unit_4d.all(axis=3) # last dimension is of shape 2, representing x,y coordinates.  Need all to match for collision
            asteroid_mask = ~any_collisions.any(axis=2) # last dimension is of shape num_asteriods, and the action is invalid if it hits any asteroid
            mask = mask & asteroid_mask

        return mask

    @staticmethod
    def normalize_positions(input, env_cfg):
        return input / env_cfg.map_width