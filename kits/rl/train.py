import os.path as osp
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from wrapper import SB3Wrapper, ObservationWrapper, ObservationTransformer
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

import gymnasium as gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO



import multiprocessing
import platform
import sys

experiment_number = 4

def in_debugger():
    return sys.gettrace() is not None

class TensorboardCallback(BaseCallback):
    def __init__(self, tag: str, verbose=0):
        super().__init__(verbose)
        self.tag = tag

    def _on_step(self) -> bool:
        c = 0

        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                c += 1
                for k in info["metrics"]:
                    stat = info["metrics"][k]
                    self.logger.record_mean(f"{self.tag}/{k}", stat)
        return True
    
def make_env(seed: int = 0):
    def _init():

        env = LuxAIS3GymEnv(numpy_output=True)
        transformer = ObservationTransformer("player_0", env.env_params)
        env = SB3Wrapper("player_0", env, transformer)
        env = ActionMasker(env, SB3Wrapper.training_mask_wrapper)  # Wrap to enable masking


        env = ObservationWrapper("player_0", env, transformer)        


        
        env = Monitor(env) # for SB3 to allow it to record metrics
        env.reset(seed=seed)
        set_random_seed(seed)
        return env

    return _init

if __name__ == "__main__":

    set_random_seed(42)
    log_path = "logs/exp_" + str(experiment_number)
    num_envs = min(4, multiprocessing.cpu_count())

    # set max episode steps to 200 for training environments to train faster
    VecEnv = DummyVecEnv if in_debugger() else SubprocVecEnv

    env = VecEnv([make_env(i) for i in range(num_envs)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True) # Observation normalization is handled by the custom wrappers
    env.reset()

    # set max episode steps to 1000 to match original environment
    eval_env = VecEnv([make_env(i) for i in range(num_envs)])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False) # Don't normalize rewards so the eval statistics can be used for something
    eval_env.training = False

    eval_env.reset()
    n_steps = 505
    policy_kwargs = dict(net_arch=(256, 256))

    model = MaskablePPO(MaskableActorCriticPolicy, env, 
        n_steps=n_steps,
        batch_size=n_steps * num_envs // 8,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_epochs=2,
        target_kl=0.05,
        gamma=0.99,
        tensorboard_log=osp.join(log_path),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=osp.join(log_path, "models"),
        log_path=osp.join(log_path, "eval_logs"),
        eval_freq=24_000,
        deterministic=False,
        render=False,
        n_eval_episodes=5,
    )

    total_timesteps = n_steps * num_envs * 1000
    model.learn(
        total_timesteps,
        callback=[TensorboardCallback(tag="train_metrics"), eval_callback],
    )
    model.save(osp.join(log_path, "models/latest_model"))