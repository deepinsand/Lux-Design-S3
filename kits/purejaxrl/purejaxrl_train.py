
import os
import sys

# Get the absolute path to the sub_repo directory
sub_repo_path = os.path.join(os.path.dirname(__file__), 'distrax')
sys.path.append(sub_repo_path)

import jax.experimental
import jax.experimental.checkify
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
import jax
import time
import datetime
import shutil  # For copying files
import flax
import pickle

from purejaxrl_ppo import make_train


from flax.metrics import tensorboard

from purejaxrl_config import config

if __name__ == "__main__":

    transfer_learning = config["TRANSFER_LEARNING"]
    transfer_learning_model = "models/model_20250306-200833.pkl"

    assert((not transfer_learning) or (config["NUM_ENVS"] == 1)), "NUM_ENVS should be 1 when doing transfer learning"

    if transfer_learning:
        with open(transfer_learning_model, 'rb') as f: # Binary read mode for pickle
            transfer_learning_model = pickle.load(f) # Load parameters directly using pickle.load


    rng = jax.random.PRNGKey(42)



    log_dir = "logs"

    # Generate a unique subdirectory name using timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_subdir = os.path.join(log_dir, timestamp)

    os.makedirs(log_subdir, exist_ok=True)
    summary_writer = tensorboard.SummaryWriter(log_subdir)
    summary_writer.hparams(dict(config))

    if config["PROFILE"]:
        jax.profiler.start_trace(log_subdir)
    
    train_fn = make_train(config, summary_writer, transfer_learning, transfer_learning_model)

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
