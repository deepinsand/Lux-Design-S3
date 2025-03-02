import os

import jax.experimental
import jax.experimental.checkify
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
import jax
import time
import datetime
import shutil  # For copying files
import flax
import pickle

from luxai_s3.params import EnvParams
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import env_params_ranges
from purejaxrl_ppo import make_train

from purejaxrl_wrapper import LuxaiS3GymnaxWrapper

from flax.metrics import tensorboard

config = {
    "LR": 2.5e-4,
    "NUM_ENVS": 2,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 200_000,
    "UPDATE_EPOCHS": 1,
    "NUM_MINIBATCHES": 1, # must be less than num_envs since RNN shuffles environemnts
    "GAMMA": 0.995,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ANNEAL_LR": True,
    "DEBUG": True,
    "PROFILE": False,
    "CONVOLUTIONS": False,
}

if __name__ == "__main__":

    rng = jax.random.PRNGKey(42)

    env_params = EnvParams()
    env = LuxAIS3Env(auto_reset=True, fixed_env_params=env_params)
    wrapped_env = LuxaiS3GymnaxWrapper(env)


    log_dir = "logs"

    # Generate a unique subdirectory name using timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_subdir = os.path.join(log_dir, timestamp)

    os.makedirs(log_subdir, exist_ok=True)
    summary_writer = tensorboard.SummaryWriter(log_subdir)
    summary_writer.hparams(dict(config))

    if config["PROFILE"]:
        jax.profiler.start_trace(log_subdir)
    
    train_fn = make_train(config, summary_writer, wrapped_env, env_params)

    #train_fn = jax.experimental.checkify.checkify(train_fn)

    train_jit = jax.jit(train_fn)
    t0 = time.time()
    out = jax.block_until_ready(train_jit(rng))
    #print(err.get())

    print(f"time: {time.time() - t0:.2f} s")

    if config["PROFILE"]:
        jax.profiler.stop_trace()

    summary_writer.close()
    save_dir = "models"
    timestamped_filename = f"model_{timestamp}.pkl"
    timestamped_filepath = os.path.join(save_dir, timestamped_filename)
    latest_model_filepath = os.path.join(save_dir, "latest_model.pkl")
    
    os.makedirs(save_dir, exist_ok=True)


   # Save model with timestamped filename using pickle
    with open(timestamped_filepath, 'wb') as f: # Binary write mode for pickle
        pickle.dump(flax.serialization.to_state_dict(out["runner_state"][0].params), f) # Directly pickle train_state.params
    print(f"Model parameters saved to (pickle): {timestamped_filepath}")

    shutil.copy2(timestamped_filepath, latest_model_filepath)  # copy2 preserves metadata
