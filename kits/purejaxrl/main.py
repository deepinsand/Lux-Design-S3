import json
from typing import Dict
import sys
from argparse import Namespace
import os
import numpy as np

from purejaxrl_agent import Agent
# from lux.config import EnvConfig
from lux.kit import from_json
### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = dict() # store potentially multiple dictionaries as kaggle imports code directly
agent_prev_obs = dict()
import time

def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    global agent_dict
    obs = observation.obs
    if type(obs) == str:
        obs = json.loads(obs)
    step = observation.step
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        agent_dict[player] = Agent(player, configurations["env_cfg"])
    if "__raw_path__" in configurations:
        dirname = os.path.dirname(configurations["__raw_path__"])
    else:
        dirname = os.path.dirname(__file__)

    sys.path.append(os.path.abspath(dirname))

    agent = agent_dict[player]
    actions = agent.act(step, from_json(obs), remainingOverageTime)
    return dict(action=actions.tolist())


if __name__ == "__main__":
    
    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)
    step = 0
    player_id = 0
    env_cfg = None
    i = 0
    t0 = time.time()

    while True:
        #print(f"t0: {time.time() - t0:.2f} s")

        inputs = read_input()
        #print(f"t1: {time.time() - t0:.2f} s")
        raw_input = json.loads(inputs)
        #print(f"t2: {time.time() - t0:.2f} s")
        observation = Namespace(**dict(step=raw_input["step"], obs=raw_input["obs"], remainingOverageTime=raw_input["remainingOverageTime"], player=raw_input["player"], info=raw_input["info"]))

        #print(f"t3: {time.time() - t0:.2f} s")

        if i == 0:
            env_cfg = raw_input["info"]["env_cfg"]
            player_id = raw_input["player"]
        i += 1
        actions = agent_fn(observation, dict(env_cfg=env_cfg))
        #print(f"t4: {time.time() - t0:.2f} s")

        # send actions to engine
        print(json.dumps(actions))