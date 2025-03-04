import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict, Optional, TypedDict
from flax.training.train_state import TrainState
import distrax
import gymnax
from packages.purejaxrl.purejaxrl.wrappers import LogWrapper
import math
from kits.purejaxrl.jax_debug import debuggable_vmap, debuggable_scan
import functools
from flax.typing import Dtype

from luxai_s3.params import EnvParams
from luxai_s3.state import (
    EnvObs,
    MapTile,
    UnitState,
)
from purejaxrl_wrapper import WrappedEnvObs, NormalizeVecReward

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        hidden_size = rnn_state[0].shape[0]
        new_rnn_state, y = nn.GRUCell(features=hidden_size)(rnn_state, ins)

        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )
    
class EmbeddingEncoder(nn.Module):
    map_tile_emb_dim: int = 4
    embedding_dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    fixed_env_params: EnvParams = EnvParams()
    unit_cardinal_emb_dim: int = 8

    def compute_counts_map(self, position, mask):
        unit_counts_map = jnp.zeros(
            (self.fixed_env_params.map_width, 
             self.fixed_env_params.map_height), dtype=jnp.int16
        )

        def update_unit_counts_map(unit_position, unit_mask, unit_counts_map):
            mask = unit_mask
            unit_counts_map = unit_counts_map.at[
                unit_position[0], unit_position[1]
            ].add(mask.astype(jnp.int16))
            return unit_counts_map

        unit_counts_map = jnp.sum(
                debuggable_vmap(update_unit_counts_map, in_axes=(0, 0, None), out_axes=0)(
                    position, mask, unit_counts_map
                ),
                axis=0,
                dtype=jnp.int16
            )
        
        return unit_counts_map
    
    def vmap_compute_counts_map(self, batched_positions, batched_masks):
        def compute_counts_map_over_minibatches(position, mask):
            debug_return = debuggable_vmap(self.compute_counts_map, in_axes=(0, 0), out_axes=0)(
                position, mask
            )
            return debug_return
    
        debug_return = debuggable_vmap(compute_counts_map_over_minibatches, in_axes=(0, 0), out_axes=0)(
            batched_positions, batched_masks
        )

        return debug_return
                        

    def compute_unit_embeddings_map(self, positions, masks, unit_cardinals_embedding_for_all_teams):
        unit_embeddings_map = jnp.zeros(
            (self.fixed_env_params.num_teams, self.fixed_env_params.map_width, 
             self.fixed_env_params.map_height, self.unit_cardinal_emb_dim), dtype=self.embedding_dtype
        )

        def update_unit_embeddings_map(unit_position, unit_mask, unit_index, unit_cardinals_embeddings, unit_embeddings_map):
            mask = unit_mask.astype(self.embedding_dtype)
            
            unit_embeddings_map = unit_embeddings_map.at[
                unit_position[0], unit_position[1]
            ].add(mask * unit_cardinals_embeddings[unit_index])
            return unit_embeddings_map

        for t in range(self.fixed_env_params.num_teams):
            unit_indexes = jnp.arange(self.fixed_env_params.max_units)

            unit_embeddings_map = unit_embeddings_map.at[t].add(
                jnp.sum(
                    debuggable_vmap(update_unit_embeddings_map, in_axes=(0, 0, 0, None, None), out_axes=0)(
                        positions[t], masks[t], unit_indexes, unit_cardinals_embedding_for_all_teams[t], unit_embeddings_map[t]
                    ),
                    axis=0,
                    dtype=self.embedding_dtype
                )
            )
        return unit_embeddings_map    
    
    def vmap_compute_unit_embeddings_map(self, obs, unit_cardinals_embedding_for_all_teams):
        # Remove the first dimension since it's minibatches and for whatever reason always 0
        # TODO: above might be true only during step inference, not during training
        def vmap_compute_unit_embeddings_map_over_minibatches(positions, masks, unit_cardinals_embedding_for_all_teams):
            debug_return = debuggable_vmap(self.compute_unit_embeddings_map, in_axes=(0, 0, None), out_axes=0)(
                positions, masks, unit_cardinals_embedding_for_all_teams
            )
        
            return debug_return
        
        debug_return = debuggable_vmap(vmap_compute_unit_embeddings_map_over_minibatches, in_axes=(0, 0, None), out_axes=0)(
            obs.units.position, obs.units_mask, unit_cardinals_embedding_for_all_teams
        )
    
        return debug_return

    
    @nn.compact
    def __call__(self, obs):
        # 3 tiles types, empty, nebula, asteriod
        tile_type_embeder = nn.Embed(3, self.map_tile_emb_dim, self.embedding_dtype, self.param_dtype) # B x ENV x w x h x map_tile_emb_dim

        # Encode each cardinal unit into an embedding
        unit_cardinality_embeder = nn.Embed(self.fixed_env_params.max_units, self.unit_cardinal_emb_dim, self.embedding_dtype, self.param_dtype)
        unit_cardinals = jnp.tile(jnp.arange(self.fixed_env_params.max_units), (self.fixed_env_params.num_teams, 1)) # t x num_teams
        unit_cardinals_embedding = unit_cardinality_embeder(unit_cardinals) # t x num_teams x 8
        unit_embeddings_map = self.vmap_compute_unit_embeddings_map(obs.original_obs, unit_cardinals_embedding) # # B x ENV x t x w x h x 8

        # counts
        unit_counts_player_0 = self.vmap_compute_counts_map(obs.original_obs.units.position[:, :, 0, :, :], obs.original_obs.units_mask[:, :, 0, :])
        unit_counts_player_1 = self.vmap_compute_counts_map(obs.original_obs.units.position[:, :, 1, :, :], obs.original_obs.units_mask[:, :, 1, :])
        unit_counts_map = jnp.stack([unit_counts_player_0, unit_counts_player_1], axis=2) / float(self.fixed_env_params.max_units)  # B x ENV x t x w x h
        
        # relics
        relic_map = self.vmap_compute_counts_map(obs.discovered_relic_node_positions, obs.discovered_relic_nodes_mask)  # B x ENV x w x h

        # Ignore energy for now ...

        # Shape to countour

        # Reshape inputs to have (w, h) as first two axes:
        unit_embeddings_reshaped = jnp.transpose(unit_embeddings_map, (0, 1, 3, 4, 2, 5)) # (B, ENV, w, h, t, 8)
        countoured_shape = unit_embeddings_reshaped.shape[:-2] + (-1,)

        unit_embeddings_reshaped = jnp.reshape(unit_embeddings_reshaped, countoured_shape) # (B, ENV, w, h, t * 8)
        tile_type_embeddings = tile_type_embeder(obs.original_obs.map_features.tile_type) # (B, ENV, w, h, 4)
        unit_counts_map_reshaped = jnp.transpose(unit_counts_map, (0, 1, 3, 4, 2)) # (B, ENV, w, h, t)
        relic_map_reshaped = relic_map[..., jnp.newaxis] # (B, ENV, w, h, 1)

        
        normalized_reward_last_round_reshaped = jnp.array(obs.normalized_reward_last_round)
        normalized_reward_last_round_reshaped =  jnp.reshape(normalized_reward_last_round_reshaped, normalized_reward_last_round_reshaped.shape + (1,1,1)) # (B, Env) -> # (B, Env, 1,1,1)
        normalized_reward_last_round_reshaped = jnp.tile(normalized_reward_last_round_reshaped, reps=(1, 1) + relic_map_reshaped.shape[-3:])  # (B, Env, w,h,1)

        grid_embedding = jnp.concatenate(
            [
                tile_type_embeddings,
                unit_counts_map_reshaped,
                unit_embeddings_reshaped,
                relic_map_reshaped,
                normalized_reward_last_round_reshaped

            ],
            axis=-1, # Concatenate along the last axis (channels after wxh)
        ) # grid_embedding shape (w, h, t*(self.dim + 1)+4+1) , made 13 so 28  + 4 = 32 channels

        return grid_embedding


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    fixed_env_params: EnvParams = EnvParams()

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        grid_encoder = nn.Sequential(
            [
                # For small dims nn.Embed is extremely slow in bf16, so we leave everything in default dtypes
                EmbeddingEncoder(embedding_dtype=jnp.float32, fixed_env_params=self.fixed_env_params),
                nn.Conv(
                    16,
                    (2, 2),
                    padding="VALID",
                    kernel_init=orthogonal(math.sqrt(2)),
                    param_dtype=jnp.float32
                ),
                nn.relu,
                nn.Conv(
                    32,
                    (3, 3),
                    padding="VALID",
                    kernel_init=orthogonal(math.sqrt(2)),
                    param_dtype=jnp.float32
                ),
                nn.relu,
                nn.Conv(
                    64,
                    (5, 5),
                    padding="VALID",
                    kernel_init=orthogonal(math.sqrt(2)),
                    param_dtype=jnp.float32
                ),
                nn.relu,
            ]
        )
        
        grid_embedding = grid_encoder(obs)
        embedding = grid_embedding.reshape(grid_embedding.shape[:-3] + (-1,)) # flatten the last 3 layers, which are the w, h, and channels
        
        # Reshape to 128 for rnn
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            math.prod(self.action_dim), kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        #pi = distrax.Categorical(logits=actor_mean)

        # Replace the last dimension (the flattened action space) with the actual action_dimension
        actor_mean = actor_mean.reshape(actor_mean.shape[:-1] + self.action_dim)
        pi = distrax.Independent(
            distrax.Categorical(logits=actor_mean),
            reinterpreted_batch_ndims=1 # last dimension is not independent
        )
        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config, writer, env=None, env_params=None):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    if env is None and env_params is None:
        env, env_params = gymnax.make(config["ENV_NAME"])

    
    env = LogWrapper(env)  # Log rewards before normalizing 
    env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        action_space = env.action_space(env_params)
        network = ActorCriticRNN([action_space.shape[0], action_space.n], config=config, fixed_env_params=env_params)
        rng, _rng = jax.random.split(rng)

        def fill_zeroes(shape, dtype=jnp.int16):
            return jnp.zeros((1, config["NUM_ENVS"], *shape), dtype=dtype)
        
        init_obs = WrappedEnvObs(
            original_obs=EnvObs(
                units=UnitState(
                    position=fill_zeroes((env_params.num_teams, env_params.max_units, 2)),
                    energy=fill_zeroes((env_params.num_teams, env_params.max_units), dtype=jnp.float32),
                ),
                units_mask=fill_zeroes((env_params.num_teams, env_params.max_units)),
                sensor_mask=fill_zeroes((env_params.map_width, env_params.map_height)),
                map_features=MapTile(
                    energy=fill_zeroes((env_params.map_width, env_params.map_height), dtype=jnp.float32),
                    tile_type=fill_zeroes((env_params.map_width, env_params.map_height)),
                ),
                team_points=fill_zeroes((env_params.num_teams,)),
                team_wins=fill_zeroes((env_params.num_teams,)),
                steps=fill_zeroes(()),
                match_steps=fill_zeroes(()),
                relic_nodes=fill_zeroes((env_params.max_relic_nodes,2)),
                relic_nodes_mask=fill_zeroes((env_params.max_relic_nodes,)),
            ),
            discovered_relic_node_positions=fill_zeroes((env_params.max_relic_nodes,2)),
            discovered_relic_nodes_mask=fill_zeroes((env_params.max_relic_nodes,)),
            normalized_reward_last_round=fill_zeroes((), dtype=jnp.float32)
        )
        init_x = (
            init_obs,
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)
        network_params = network.init(_rng, init_hstate, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"]) # VMAP OUT TO NUM_ENV HERE, 2nd param generally
        obsv, env_state = debuggable_vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)

        # TRAIN LOOP
        def _update_step(runner_state, update_count):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                # Everything but train_state and rng has first dim as NUM_ENVS
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                last_obs_with_new_axis = jax.tree_util.tree_map(lambda x: x[np.newaxis, :], last_obs)

                ac_in = (last_obs_with_new_axis, last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # first dim is minibatches so squeeze it out during step
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                # Step occurs without minibatches or envs
                obsv, env_state, reward, done, info = debuggable_vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, done, hstate, rng)
                return runner_state, transition
            
            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state

            # CHANGED THIS since I'm not sending arrays
            last_obs_with_new_axis = jax.tree_util.tree_map(lambda x: x[np.newaxis, :], last_obs)

            ac_in = (last_obs_with_new_axis, last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)
            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = transition.done, transition.value, transition.reward 
                    delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    return (gae, value, done), gae
                _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val, last_done), traj_batch, reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value
            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae_norm
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae_norm
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        # Additional metrics:
                        clip_fraction = jnp.mean((ratio < (1.0 - config["CLIP_EPS"])) | (ratio > (1.0 + config["CLIP_EPS"])))
                        ratio_mean = jnp.mean(ratio)
                        # Compute explained variance:
                        explained_var = 1.0 - jnp.var(targets - value) / (jnp.var(targets) + 1e-8)
                        return total_loss, (value_loss, loss_actor, entropy, clip_fraction, ratio_mean, explained_var)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss_val, aux), grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    # Compute gradient norm.
                    grad_norm = optax.global_norm(grads)

                    # Append grad_norm to the auxiliary tuple.
                    aux = aux + (grad_norm,)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss_val, aux)

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                # first dim is steps, second is num_envs
                # this shuffles the second dimension, giving a random ordering of environments
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                # Creates second dimension here!
                # this reshapes into (num_minibatches, steps, and then environment of size (NUM_ENV / NUM_MINIBATCHES)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, loss_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            # Updating Training State and Metrics:
            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            # --- TensorBoard Logging ---
            if writer is not None:
                def callback(loss_info, metric, update_count):
                    # loss_info is a nested tuple from all update epochs and minibatches.
                    # We unpack the auxiliary metrics as:
                    #   [0]: value_loss,
                    #   [1]: actor_loss,
                    #   [2]: entropy,
                    #   [3]: clip_fraction,
                    #   [4]: ratio_mean,
                    #   [5]: explained_var,
                    #   [6]: grad_norm.
                    avg_total_loss = loss_info[0].mean()
                    avg_value_loss = loss_info[1][0].mean()
                    avg_actor_loss = loss_info[1][1].mean()
                    avg_entropy = loss_info[1][2].mean()
                    avg_clip_fraction = loss_info[1][3].mean()
                    avg_ratio_mean = loss_info[1][4].mean()
                    avg_explained_var = loss_info[1][5].mean()
                    avg_grad_norm = loss_info[1][6].mean()

                    writer.scalar("losses/total_loss", avg_total_loss, update_count)
                    writer.scalar("losses/value_loss", avg_value_loss, update_count)
                    writer.scalar("losses/actor_loss", avg_actor_loss, update_count)
                    writer.scalar("losses/entropy", avg_entropy, update_count)
                    writer.scalar("losses/clip_fraction", avg_clip_fraction, update_count)
                    writer.scalar("losses/ratio_mean", avg_ratio_mean, update_count)
                    writer.scalar("losses/explained_variance", avg_explained_var, update_count)
                    writer.scalar("grad/grad_norm", avg_grad_norm, update_count)
                    
                    # If using an annealed LR, log the current learning rate.
                    if config.get("ANNEAL_LR"):
                        current_lr = linear_schedule(update_count)
                        writer.scalar("lr", current_lr, update_count)
                    
                    # Log episodic returns and lengths (and optionally standard deviation of returns).
                    if (metric is not None and 
                        metric["returned_episode_returns"].shape[0] > 0 and 
                        metric["returned_episode_returns"][-1].shape[0] > 0):
                        returned_episode_returns = metric["returned_episode_returns"][-1]
                        returned_episode_lengths = metric["returned_episode_lengths"][-1]
                        if len(returned_episode_returns) > 0:
                            avg_return = returned_episode_returns.mean()
                            avg_length = returned_episode_lengths.mean()
                            std_return = returned_episode_returns.std()
                            writer.scalar("episodic/avg_return", avg_return, update_count)
                            writer.scalar("episodic/avg_length", avg_length, update_count)
                            writer.scalar("episodic/std_return", std_return, update_count)
                jax.debug.callback(callback, loss_info, metric, update_count)

            # Debugging mode
            if config.get("DEBUG"):
                def callback(info):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"]), config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train
