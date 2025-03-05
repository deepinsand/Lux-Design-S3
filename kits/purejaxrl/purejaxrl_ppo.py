import jax
import jax.experimental
import jax.experimental.checkify
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
import math
from jax_debug import debuggable_vmap, debuggable_conditional_breakpoint
import functools
from luxai_s3.params import EnvParams, env_params_ranges
from purejaxrl_wrapper import LuxaiS3GymnaxWrapper, WrappedEnvObs, NormalizeVecReward, LogWrapper
from luxai_s3.env import LuxAIS3Env



# Re-use the ResNet block and convolutional encoder from before.
class ResNetBlock(nn.Module):
    features: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
    
        residual = x
        x = activation(x)
        x = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME',
                    kernel_init=orthogonal(math.sqrt(2))
                )(x)
        x = activation(x)
        x = nn.Conv(self.features, kernel_size=(5, 5), padding='SAME',
                    kernel_init=orthogonal(math.sqrt(2))
                )(x)        
        return x + residual

# Agent-specific feature extraction using direct indexing
class SpatialFeatureExtractor(nn.Module):
    @nn.compact
    def __call__(self, feature_map, agent_positions, mask):

        def extract_features_for_timestep(fmap, positions, mask_t):
            # fmap: [H, W, C]
            # positions: [num_agents, 2]
            # mask_t: [num_agents] (boolean)
            def extract_feature(pos, valid):
                row, col = pos  # extract row and col indices
                feat = fmap[row, col, :]  # extract feature vector at that spatial location
                # If the agent doesn't exist (valid == False), return zeros.
                return jnp.where(valid, feat, jnp.zeros_like(feat))
            # Vectorize over the num_agents dimension.
            return debuggable_vmap(extract_feature)(positions, mask_t)
        
        # First, map over the time dimension.
        #extract_over_time = debuggable_vmap(extract_features_for_timestep, in_axes=(0, 0, 0))
        # Then, map over the batch dimension.
        agent_features = debuggable_vmap(extract_features_for_timestep, in_axes=(0, 0, 0))(
            feature_map, agent_positions, mask
        )
        return agent_features
        
        
class EmbeddingEncoder(nn.Module):
    map_tile_emb_dim: int = 4
    fixed_env_params: EnvParams = EnvParams()
    unit_cardinal_emb_dim: int = 8
    
    @nn.compact
    def __call__(self, obs):
        tile_type_embeder = nn.Embed(4, self.map_tile_emb_dim + 1, jnp.float32) # B x ENV x w x h x map_tile_emb_dim
        tile_type_embeddings =  tile_type_embeder(obs.tile_type)
    
        map_shape = tile_type_embeddings.shape[-3:-1]
        normalized_steps_reshaped = jnp.array(obs.normalized_steps)
        normalized_steps_reshaped =  jnp.reshape(normalized_steps_reshaped, normalized_steps_reshaped.shape + (1,1,1)) # (Env) -> # (Env, 1,1,1)
        normalized_steps_reshaped = jnp.tile(normalized_steps_reshaped, reps=(1,) + map_shape + (1,))  # (Env, w,h,1)

        param_list_reshaped = jnp.array(obs.param_list)
        param_list_reshaped =  jnp.reshape(param_list_reshaped, (param_list_reshaped.shape[0],) + (1,1) + (param_list_reshaped.shape[1],)) # (Env, 11) -> # (Env, 1,1,11)
        param_list_reshaped = jnp.tile(param_list_reshaped, reps=(1,) + map_shape + (1,))  # (Env, w,h,11)


        grid_embedding = jnp.concatenate(
            [
                tile_type_embeddings,
                normalized_steps_reshaped,
                param_list_reshaped,
                obs.sensor_mask[..., jnp.newaxis],
                obs.normalized_unit_counts[..., jnp.newaxis],
                obs.normalized_unit_counts_opp[..., jnp.newaxis],
                obs.normalized_unit_energys_max_grid[..., jnp.newaxis],
                obs.normalized_unit_energys_max_grid_opp[..., jnp.newaxis],
                obs.grid_probability_of_being_an_energy_point_based_on_no_reward[..., jnp.newaxis],
                obs.grid_max_probability_of_being_an_energy_point_based_on_positive_rewards[..., jnp.newaxis],
                #obs.grid_min_probability_of_being_an_energy_point_based_on_positive_rewards[..., jnp.newaxis],
                obs.grid_avg_probability_of_being_an_energy_point_based_on_positive_rewards[..., jnp.newaxis],
                obs.grid_probability_of_being_energy_point_based_on_relic_positions[..., jnp.newaxis],
                obs.value_of_sapping_grid[..., jnp.newaxis],
            ],
            axis=-1, # Concatenate along the last axis (channels after wxh)
        ) # grid_embedding shape (w, h, t*(self.dim + 1)+4+1) , made 13 so 28  + 4 = 32 channels

        return grid_embedding
    

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    features_dim: int = 16
    quick: bool = True
    

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        grid_embeddings = EmbeddingEncoder()(x)
        convolutions = nn.Sequential(
            [
                nn.Conv(
                    16,
                    (2, 2),
                    padding="SAME",
                    kernel_init=orthogonal(math.sqrt(2)),
                ),
                activation,
                nn.Conv(
                    32,
                    (3, 3),
                    padding="SAME",
                    kernel_init=orthogonal(math.sqrt(2)),
                ),
                activation,
                nn.Conv(
                    32,
                    (5, 5),
                    padding="SAME",
                    kernel_init=orthogonal(math.sqrt(2)),
                ),
                activation,

                #ResNetBlock(features=self.features_dim, activation=self.activation),
                #ResNetBlock(features=self.features_dim, activation=self.activation),
            ]
        )

        local_agent_embeddings = SpatialFeatureExtractor()(grid_embeddings, x.unit_positions, x.unit_mask) # [mb_size  x num_agents, features]

        if self.quick:
            local_agent_features = jnp.concatenate(
                [
                    x.unit_mask[..., jnp.newaxis],
                    local_agent_embeddings,
                ],
                axis=-1, # Add the mask to understand if the agent is actually there
            )
        else:
            convoluted_features = convolutions(grid_embeddings)
            local_agent_convoluted_features = SpatialFeatureExtractor()(convoluted_features, x.unit_positions, x.unit_mask) # [mb_size  x num_agents, features]

            # ---- Global Branch ----
            # Compute a global context vector using a global average pooling over the spatial dims.

            num_agents = x.unit_mask.shape[-1]

            global_context = jnp.mean(convoluted_features, axis=(1,2))  # mb_size x features
            global_context = jnp.reshape(global_context, global_context.shape[:-1] + (1,) + global_context.shape[-1:]) # mb_size x 1 x features
            global_context = jnp.tile(global_context, (num_agents, 1))
            
            local_agent_features = jnp.concatenate(
                [
                    x.unit_mask[..., jnp.newaxis],
                    local_agent_embeddings,
                    local_agent_convoluted_features,
                    global_context
                ],
                axis=-1, # Add the mask to understand if the agent is actually there
            )


        actor_mean = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(local_agent_features)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        #pi = distrax.Categorical(logits=actor_mean)

        actor_mean_masked = jnp.where(x.action_mask, actor_mean, -1e9)

        pi = distrax.Independent(
            distrax.Categorical(logits=actor_mean_masked),
            reinterpreted_batch_ndims=1
        )

        critic = local_agent_features.reshape(local_agent_features.shape[:-2] + (-1,)) # flatten num_agents and features into one vector for value guess
        #critic = critic[jnp.newaxis, :] # add a leading dimenion
        critic = nn.Dense(
            512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config, writer):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    # Double mini batch sizes since we now have 2 obs per env step
    config["MINIBATCH_SIZE"] = (
        2 * config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    fixed_env_params = EnvParams()
    env = LuxAIS3Env(auto_reset=True, fixed_env_params=fixed_env_params)
    env = LuxaiS3GymnaxWrapper(env, config["NUM_UPDATES"])
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
        action_space = env.action_space(fixed_env_params)
        network = ActorCritic(
            [action_space.shape[0], action_space.n], activation=config["ACTIVATION"], quick=(not config["CONVOLUTIONS"])
        )
        rng, _rng = jax.random.split(rng)
        def fill_zeroes(shape, dtype=jnp.int16):
            return jnp.zeros((config["NUM_ENVS"], *shape), dtype=dtype)
    
        init_x = WrappedEnvObs(
            relic_map=fill_zeroes((fixed_env_params.map_width, fixed_env_params.map_height)),
            normalized_unit_counts=fill_zeroes((fixed_env_params.map_width, fixed_env_params.map_height), dtype=jnp.float32),
            normalized_unit_counts_opp=fill_zeroes((fixed_env_params.map_width, fixed_env_params.map_height), dtype=jnp.float32),
            normalized_unit_energys_max_grid=fill_zeroes((fixed_env_params.map_width, fixed_env_params.map_height), dtype=jnp.float32),
            normalized_unit_energys_max_grid_opp=fill_zeroes((fixed_env_params.map_width, fixed_env_params.map_height), dtype=jnp.float32),
            tile_type=fill_zeroes((fixed_env_params.map_width, fixed_env_params.map_height)),
            normalized_energy_field=fill_zeroes((fixed_env_params.map_width, fixed_env_params.map_height), dtype=jnp.float32),
            unit_positions=fill_zeroes((fixed_env_params.max_units, 2)),
            unit_mask=fill_zeroes((fixed_env_params.max_units,)),
            normalized_steps=fill_zeroes((), dtype=jnp.float32),
            grid_probability_of_being_energy_point_based_on_relic_positions=fill_zeroes((fixed_env_params.map_width, fixed_env_params.map_height), dtype=jnp.float32),
            grid_probability_of_being_an_energy_point_based_on_no_reward=fill_zeroes((fixed_env_params.map_width, fixed_env_params.map_height), dtype=jnp.float32),
            grid_max_probability_of_being_an_energy_point_based_on_positive_rewards=fill_zeroes((fixed_env_params.map_width, fixed_env_params.map_height), dtype=jnp.float32),
            grid_min_probability_of_being_an_energy_point_based_on_positive_rewards=fill_zeroes((fixed_env_params.map_width, fixed_env_params.map_height), dtype=jnp.float32),
            grid_avg_probability_of_being_an_energy_point_based_on_positive_rewards=fill_zeroes((fixed_env_params.map_width, fixed_env_params.map_height), dtype=jnp.float32),
            value_of_sapping_grid=fill_zeroes((fixed_env_params.map_width, fixed_env_params.map_height), dtype=jnp.float32),
            sensor_mask=fill_zeroes((fixed_env_params.map_width, fixed_env_params.map_height), dtype=jnp.float32),
            action_mask=fill_zeroes((fixed_env_params.max_units, 6), dtype=jnp.bool),
            param_list=fill_zeroes((11,), dtype=jnp.float32),
        )


        network_params = network.init(_rng, init_x)
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

        def sample_params(rng_key):
            randomized_game_params = dict()
            for k, v in env_params_ranges.items():
                rng_key, subkey = jax.random.split(rng_key)
                if isinstance(v[0], int):
                    randomized_game_params[k] = jax.random.choice(subkey, jax.numpy.array(v, dtype=jnp.int16))
                else:
                    randomized_game_params[k] = jax.random.choice(subkey, jax.numpy.array(v, dtype=jnp.float32))
            params = EnvParams(**randomized_game_params)
            return params

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        env_params_randomized = debuggable_vmap(sample_params)(jax.random.split(_rng, config["NUM_ENVS"]))

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = debuggable_vmap(env.reset, in_axes=(0, 0))(reset_rng, env_params_randomized)

        # TRAIN LOOP
        def _update_step(update_count, runner_state):
            
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, (last_obs_0, last_obs_1), rng = runner_state


                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi_0, value_0 = network.apply(train_state.params, last_obs_0)
                action_0 = pi_0.sample(seed=_rng)
                log_prob_0 = pi_0.log_prob(action_0)

                rng, _rng = jax.random.split(rng)
                pi_1, value_1 = network.apply(train_state.params, last_obs_1)
                action_1 = pi_1.sample(seed=_rng)
                log_prob_1 = pi_1.log_prob(action_1)

                action = (action_0, action_1)
                

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, (reward_0, reward_1), done, info = debuggable_vmap(
                    env.step, in_axes=(0, 0, 0, 0, None)
                )(rng_step, env_state, action, env_params_randomized, update_count)

                transition_0 = Transition(
                    done, action_0, value_0, reward_0, log_prob_0, last_obs_0, info
                )                
                transition_1 = Transition(
                    done, action_1, value_1, reward_1, log_prob_1, last_obs_1, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, (transition_0, transition_1)

            runner_state, (traj_batch_0, traj_batch_1) = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, (last_obs_0, last_obs_1), rng = runner_state
            _, last_val_0 = network.apply(train_state.params, last_obs_0)
            _, last_val_1 = network.apply(train_state.params, last_obs_1)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages_0, targets_0 = _calculate_gae(traj_batch_0, last_val_0)
            advantages_1, targets_1 = _calculate_gae(traj_batch_1, last_val_1)

            advantages = jnp.concatenate([advantages_0, advantages_1])
            targets = jnp.concatenate([targets_0, targets_1])
            traj_batch = jax.tree.map(lambda x, y: jnp.concatenate([x, y]), traj_batch_0, traj_batch_1)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
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
                    (loss_val, aux), grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    # Compute gradient norm.
                    grad_norm = jnp.sqrt(
                        sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)])
                    )
                    # Append grad_norm to the auxiliary tuple.
                    aux = aux + (grad_norm,)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss_val, aux)

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == 2 * config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs times number of teams"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, loss_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, loss_info

            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch_0.info
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

            runner_state = (train_state, env_state, (last_obs_0, last_obs_1), rng)
            return runner_state

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng) # env_state and obsv come from the reset
        runner_state = jax.lax.fori_loop(0, config["NUM_UPDATES"], _update_step, runner_state)

        return {"runner_state": runner_state}

    return train
