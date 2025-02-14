# Adapted from PureJaxRL implementation and minigrid baselines, source:
# https://github.com/lupuandr/explainable-policies/blob/50acbd777dc7c6d6b8b7255cd1249e81715bcb54/purejaxrl/ppo_rnn.py#L4
# https://github.com/lcswillems/rl-starter-files/blob/master/model.py
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional
import os
import datetime
import shutil  # For copying files
import pickle

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import pyrallis
import flax
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState
from xland_util import Transition, calculate_gae, ppo_update_networks, rollout

from xminigrid.environment import Environment, EnvParams
from packages.purejaxrl.purejaxrl.jax_debug import debuggable_vmap, debuggable_pmap

from nn import ActorCriticRNN
# must come after import nn since that import distrax, which will require keras if this is imported first
from flax.metrics.tensorboard import SummaryWriter 

# this will be default in new jax versions anyway
jax.config.update("jax_threefry_partitionable", True)


from luxai_s3.params import EnvParams
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import env_params_ranges

from xland_wrapper import LuxaiS3GymnaxWrapper, WrappedEnvObs, NormalizeVecReward
from luxai_s3.state import (
    EnvObs,
    MapTile,
    UnitState,
)


@dataclass
class TrainConfig:
    # agent
    obs_emb_dim: int = 16
    rnn_hidden_dim: int = 1024
    rnn_num_layers: int = 1
    head_hidden_dim: int = 64
    # training
    enable_bf16: bool = False
    num_envs: int = 8
    num_steps: int = 128
    update_epochs: int = 2
    num_minibatches: int = 2
    total_timesteps: int = 200_000
    lr: float = 2.5e-4
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_episodes: int = 80
    seed: int = 42
    profile: bool = False

    def __post_init__(self):
        num_devices = jax.local_device_count()
        # splitting computation across all available devices
        self.num_envs_per_device = self.num_envs // num_devices
        self.total_timesteps_per_device = self.total_timesteps // num_devices
        self.eval_episodes_per_device = self.eval_episodes // num_devices
        assert self.num_envs % num_devices == 0
        self.num_updates = self.total_timesteps_per_device // self.num_steps // self.num_envs_per_device
        print(f"Num devices: {num_devices}, Num updates: {self.num_updates}")


def make_states(config: TrainConfig):
    # for learning rate scheduling
    def linear_schedule(count):
        frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates
        return config.lr * frac

    # setup environment

    env_params = EnvParams()
    env = LuxAIS3Env(auto_reset=True, fixed_env_params=env_params)
    env = LuxaiS3GymnaxWrapper(env, "player_0")
    env = NormalizeVecReward(env, config.gamma)
    # setup training state
    rng = jax.random.key(config.seed)
    rng, _rng = jax.random.split(rng)

        
    action_space = env.action_space(env_params)

    network = ActorCriticRNN(
        action_dim=[action_space.shape[0], action_space.n],
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_num_layers=config.rnn_num_layers,
        head_hidden_dim=config.head_hidden_dim,
        dtype=jnp.bfloat16 if config.enable_bf16 else None,
    )

    def fill_zeroes(shape, dtype=jnp.int16):
        return jnp.zeros((config.num_envs_per_device, 1, *shape), dtype=dtype)
    
    init_obs = WrappedEnvObs(
        relic_map=fill_zeroes((env_params.map_width, env_params.map_height)),
        unit_counts_player_0=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
        tile_type=fill_zeroes((env_params.map_width, env_params.map_height)),
        normalized_reward_last_round=fill_zeroes((), dtype=jnp.float32),
        unit_positions_player_0=fill_zeroes((env_params.max_units, 2)),
        unit_mask_player_0=fill_zeroes((env_params.max_units,)),
    )

    init_hstate = network.initialize_carry(batch_size=config.num_envs_per_device)

    network_params = network.init(_rng, init_obs, init_hstate)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=linear_schedule, eps=1e-8),  # eps=1e-5
    )
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    return rng, env, env_params, init_hstate, train_state


def make_train(
    env: Environment,
    env_params: EnvParams,
    config: TrainConfig,
    writer: SummaryWriter
):
    
    #@partial(debuggable_pmap, axis_name="devices")
    def train(
        rng: jax.Array,
        train_state: TrainState,
        init_hstate: jax.Array,
    ):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs_per_device)

        action_dim = env.action_space(env_params)
        reset_obs, reset_env_state = debuggable_vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        prev_action = jnp.zeros((config.num_envs_per_device, action_dim.shape[0]), dtype=jnp.int32)
        prev_reward = jnp.zeros(config.num_envs_per_device, dtype=jnp.float32) 

        # TRAIN LOOP
        def _update_step(runner_state, step_count):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, _):
                rng, train_state, prev_timestep, prev_action, prev_reward, prev_hstate = runner_state

                # prev reward is already unrolled from runner_state.  These should be the same in theory ...
                prev_obs, prev_env_state, _ = prev_timestep
                # Add the seq_len dimension?
                prev_obs_with_new_axis = jax.tree_util.tree_map(lambda x: x[:, None], prev_obs)

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                dist, value, hstate = train_state.apply_fn(
                    train_state.params,
                    prev_obs_with_new_axis, 
                    prev_hstate,
                )
                action, log_prob = dist.sample_and_log_prob(seed=_rng)
                # squeeze seq_len where possible
                action, value, log_prob = action.squeeze(1), value.squeeze(1), log_prob.squeeze(1)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, action.shape[0])
                obs, env_state, reward, done, _ = debuggable_vmap(env.step, in_axes=(0, 0, 0, None))(rng_step, prev_env_state, action, env_params)

                timestep = (obs, env_state, reward)

                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    log_prob=log_prob,
                    obs=obs,
                    prev_action=prev_action,
                    prev_reward=prev_reward,
                )
                runner_state = (rng, train_state, timestep, action, reward, hstate)
                return runner_state, transition

            initial_hstate = runner_state[-1]
            # transitions: [seq_len, batch_size, ...]
            runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config.num_steps)

            # CALCULATE ADVANTAGE
            rng, train_state, timestep, prev_action, prev_reward, hstate = runner_state
            # calculate value of the last step for bootstrapping

            last_obs, _, _ = timestep
            # Add the seq_len dimension?
            last_obs_with_new_axis = jax.tree_util.tree_map(lambda x: x[:, None], last_obs)
            
            _, last_val, _ = train_state.apply_fn(
                train_state.params,
                last_obs_with_new_axis,
                hstate,
            )
            advantages, targets = calculate_gae(transitions, last_val.squeeze(1), config.gamma, config.gae_lambda)

            # UPDATE NETWORK
            def _update_epoch(update_state, _):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, transitions, advantages, targets = batch_info
                    new_train_state, update_info = ppo_update_networks(
                        train_state=train_state,
                        transitions=transitions,
                        init_hstate=init_hstate.squeeze(1),
                        advantages=advantages,
                        targets=targets,
                        clip_eps=config.clip_eps,
                        vf_coef=config.vf_coef,
                        ent_coef=config.ent_coef,
                    )
                    return new_train_state, update_info

                rng, train_state, init_hstate, transitions, advantages, targets = update_state

                # MINIBATCHES PREPARATION
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config.num_envs_per_device)
                # [seq_len, batch_size, ...]
                batch = (init_hstate, transitions, advantages, targets)
                # [batch_size, seq_len, ...], as our model assumes
                batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

                shuffled_batch = jtu.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                # [num_minibatches, minibatch_size, ...]
                minibatches = jtu.tree_map(
                    lambda x: jnp.reshape(x, (config.num_minibatches, -1) + x.shape[1:]), shuffled_batch
                )
                train_state, update_info = jax.lax.scan(_update_minbatch, train_state, minibatches)

                update_state = (rng, train_state, init_hstate, transitions, advantages, targets)
                return update_state, update_info

            # [seq_len, batch_size, num_layers, hidden_dim]
            init_hstate = initial_hstate[None, :]
            update_state = (rng, train_state, init_hstate, transitions, advantages, targets)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)

            # averaging over minibatches then over epochs
            loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)

            rng, train_state = update_state[:2]
            # EVALUATE AGENT
            rng, _rng = jax.random.split(rng)
            eval_rng = jax.random.split(_rng, num=config.eval_episodes_per_device)

            # vmap only on rngs
            eval_stats = debuggable_vmap(rollout, in_axes=(0, None, None, None, None, None))(
                eval_rng,
                env,
                env_params,
                train_state,
                # TODO: make this as a static method mb?
                jnp.zeros((1, config.rnn_num_layers, config.rnn_hidden_dim)),
                1,
            )
            #eval_stats = jax.lax.pmean(eval_stats, axis_name="devices")
            loss_info.update(
                {
                    "eval/returns": eval_stats.reward.mean(0),
                    "eval/lengths": eval_stats.length.mean(0),
                    "lr": train_state.opt_state[-1].hyperparams["learning_rate"],
                }
            )

            if writer is not None:
                def callback(loss_info, step_count):
                    for key in ["total_loss", "value_loss", "actor_loss", "entropy", "eval/returns", "eval/lengths", "lr"]:
                        writer.scalar(key, loss_info[key], step_count)
                jax.debug.callback(callback, loss_info, step_count)


            runner_state = (rng, train_state, timestep, prev_action, prev_reward, hstate)
            return runner_state, loss_info

        reset_timestep = (reset_obs, reset_env_state, prev_reward)
        runner_state = (rng, train_state, reset_timestep, prev_action, prev_reward, init_hstate)
        step_counts = jnp.arange(config.num_updates)
        runner_state, loss_info = jax.lax.scan(_update_step, runner_state, step_counts, config.num_updates)
        return {"runner_state": runner_state, "loss_info": loss_info}

    return train


@pyrallis.wrap()
def train(config: TrainConfig):


    
    log_dir = "logs"

    # Generate a unique subdirectory name using timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_subdir = os.path.join(log_dir, timestamp)

    os.makedirs(log_subdir, exist_ok=True)
    summary_writer = SummaryWriter(log_subdir)
    summary_writer.hparams(asdict(config))


    rng, env, env_params, init_hstate, train_state = make_states(config)
    # replicating args across devices
    # rng = jax.random.split(rng, num=jax.local_device_count())
    # train_state = replicate(train_state, jax.local_devices())
    # init_hstate = replicate(init_hstate, jax.local_devices())

    print("Compiling...")
    t = time.time()
    train_fn = jax.jit(make_train(env, env_params, config, summary_writer))
    
    if os.environ.get("JAX_DISABLE_JIT", "").lower() != "true":
        train_fn = train_fn.lower(rng, train_state, init_hstate).compile()
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s.")

    print("Training...")
    t = time.time()

    if config.profile:
        jax.profiler.start_trace(log_subdir)

    train_info = jax.block_until_ready(train_fn(rng, train_state, init_hstate))
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s")

    if config.profile:
        jax.profiler.stop_trace()
    
    print("Logging...")
    loss_info = train_info["loss_info"]#unreplicate(train_info["loss_info"])

    total_transitions = 0
    for i in range(config.num_updates):
        # summing total transitions per update from all devices
        total_transitions += config.num_steps * config.num_envs_per_device * jax.local_device_count()
        info = jtu.tree_map(lambda x: x[i].item(), loss_info)
        info["transitions"] = total_transitions


    print("Final return: ", float(loss_info["eval/returns"][-1]))

    summary_writer.close()
    save_dir = "models"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_filename = f"model_{timestamp}.pkl"
    timestamped_filepath = os.path.join(save_dir, timestamped_filename)
    latest_model_filepath = os.path.join(save_dir, "latest_model.pkl")
    
    os.makedirs(save_dir, exist_ok=True)

   # Save model with timestamped filename using pickle
    with open(timestamped_filepath, 'wb') as f: # Binary write mode for pickle
        pickle.dump(flax.serialization.to_state_dict(train_info["runner_state"][1].params), f) # Directly pickle train_state.params
    print(f"Model parameters saved to (pickle): {timestamped_filepath}")

    shutil.copy2(timestamped_filepath, latest_model_filepath)  # copy2 preserves metadata



if __name__ == "__main__":
    train()



