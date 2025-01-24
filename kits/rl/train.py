import os.path as osp
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import TimeLimit
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from wrapper import SB3Wrapper, ObservationWrapper
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import multiprocessing
import platform
import sys

def in_debugger():
    return sys.gettrace() is not None

class TensorboardCallback(BaseCallback):
    def __init__(self, tag: str, verbose=0):
        super().__init__(verbose)
        self.tag = tag

    def _on_step(self) -> bool:
        c = 0

        # for i, done in enumerate(self.locals["dones"]):
        #     if done:
        #         info = self.locals["infos"][i]
        #         c += 1
        #         for k in info["metrics"]:
        #             stat = info["metrics"][k]
        #             self.logger.record_mean(f"{self.tag}/{k}", stat)
        return True
    
def make_env(seed: int = 0):
    def _init():

        env = LuxAIS3GymEnv(numpy_output=True)

        # Add a SB3 wrapper to make it work with SB3 and simplify the action space with the controller
        # this will remove the bidding phase and factory placement phase. For factory placement we use
        # the provided place_near_random_ice function which will randomly select an ice tile and place a factory near it.
        env = SB3Wrapper(env)
        
        # changes observation to include a few simple features
        env = ObservationWrapper("player_0", env)        
        
        #env = Monitor(env) # for SB3 to allow it to record metrics
        env.reset(seed=seed)
        set_random_seed(seed)
        return env

    return _init

if __name__ == "__main__":

    set_random_seed(42)
    log_path = "logs/exp_1"
    num_envs = min(4, multiprocessing.cpu_count())

    # set max episode steps to 200 for training environments to train faster
    VecEnv = DummyVecEnv if in_debugger() else SubprocVecEnv

    env = VecEnv([make_env(i) for i in range(num_envs)])
    env.reset()
    # set max episode steps to 1000 to match original environment
    eval_env = VecEnv([make_env(i) for i in range(num_envs)])
    eval_env.reset()
    n_steps = 400
    policy_kwargs = dict(net_arch=(128, 128))
    model = PPO(
        "MlpPolicy",
        env,
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

    total_timesteps = n_steps * num_envs * 100
    model.learn(
        total_timesteps,
        callback=[TensorboardCallback(tag="train_metrics"), eval_callback],
    )
    model.save(osp.join(log_path, "models/latest_model"))