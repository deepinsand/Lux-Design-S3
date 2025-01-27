import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from luxai_s3.wrappers import LuxAIS3GymEnv

class SB3Wrapper(gym.Wrapper):
    def __init__(
        self,
        env: LuxAIS3GymEnv
    ) -> None:
        gym.Wrapper.__init__(self, env)
        self.env = env        
        self.action_space = spaces.MultiDiscrete([ 5 ] * env.env_params.max_units)
        self.old_team_points = 0
        self.old_team_wins = 0

    def step(self, action: npt.NDArray):
        
        # here, for each agent in the game we translate their action into a Lux S2 action
        
        single_player_lux_action = np.zeros((self.env.env_params.max_units, 3), dtype=int)
        single_player_lux_action[:, 0] = action
        lux_action = {
            "player_0": single_player_lux_action, 
            "player_1": np.zeros((self.env.env_params.max_units, 3), dtype=int)
        }
        
        # Completely ignore sapping
        obs, reward, terminated, truncated, _  = self.env.step(lux_action)
        new_team_points = obs["player_0"]["team_points"]
        new_team_wins = obs["player_0"]["team_wins"]
        
        diff_points = new_team_points - self.old_team_points
        diff_wins = new_team_wins - self.old_team_wins
        diff_wins_sum = np.sum(diff_wins)

        # Game or match is over so reset diff
        if diff_wins_sum != 0:
            diff_points = np.zeros(2)

        # Match is reset so diff_wins should be 0
        if diff_wins_sum < 0:
            diff_wins = np.zeros(2)
        
        manufactured_reward = np.dot(diff_points, [100,0])
        self.old_team_points = new_team_points
        self.old_team_wins = new_team_wins

        single_player_terminated = terminated["player_0"]
        single_player_truncated = truncated["player_0"]

        info = dict()
        metrics = dict()
        metrics["rewards"] = manufactured_reward

        # we can save the metrics to info so we can use tensorboard to log them to get a glimpse into how our agent is behaving
        info["metrics"] = metrics

        return obs, manufactured_reward, single_player_terminated, single_player_truncated, info


class ObservationWrapper(gym.ObservationWrapper):

    def __init__(self, player: str, env: gym.Env, env_cfg):
        super().__init__(env)
        self.player = player
        self.transformer = ObservationTransformer(player, env_cfg)
        self.observation_space = spaces.Box(0, 1, shape=(self.transformer.observation_space_size,))

    def observation(self, obs):
        player_obs = obs[self.player]
        return self.transformer.transform(player_obs)

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
            self.params.max_relic_nodes * 3 # one for masks, one for x pos, one for y pos
        )

    @staticmethod
    def normalize_positions(input, env_cfg):
        return input / env_cfg.map_width

    def transform(self, obs):
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

        new_observation = np.concatenate(
                [unit_mask, norm_and_flat_unit_positions, self.discovered_relic_nodes_mask, norm_and_flat_relic_positions])
            
        return new_observation