# Model adapted from minigrid baselines:
# https://github.com/lcswillems/rl-starter-files/blob/master/model.py
import math
from typing import Optional, TypedDict, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.dtypes import promote_dtype
from flax.linen.initializers import glorot_normal, orthogonal, zeros_init
from flax.typing import Dtype

from xland_wrapper import WrappedEnvObs
from luxai_s3.params import EnvParams
from packages.purejaxrl.purejaxrl.jax_debug import debuggable_vmap

class GRU(nn.Module):
    hidden_dim: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, xs, init_state):
        seq_len, input_dim = xs.shape
        # this init might not be optimal, for example bias for reset gate should be -1 (for now ok)
        Wi = self.param("Wi", glorot_normal(in_axis=1, out_axis=0), (self.hidden_dim * 3, input_dim), self.param_dtype)
        Wh = self.param("Wh", orthogonal(column_axis=0), (self.hidden_dim * 3, self.hidden_dim), self.param_dtype)
        bi = self.param("bi", zeros_init(), (self.hidden_dim * 3,), self.param_dtype)
        bn = self.param("bn", zeros_init(), (self.hidden_dim,), self.param_dtype)

        def _step_fn(h, x):
            igates = jnp.split(Wi @ x + bi, 3)
            hgates = jnp.split(Wh @ h, 3)

            reset = nn.sigmoid(igates[0] + hgates[0])
            update = nn.sigmoid(igates[1] + hgates[1])
            new = nn.tanh(igates[2] + reset * (hgates[2] + bn))
            next_h = (1 - update) * new + update * h

            return next_h, next_h

        # cast to the computation dtype
        xs, init_state, Wi, Wh, bi, bn = promote_dtype(xs, init_state, Wi, Wh, bi, bn, dtype=self.dtype)

        last_state, all_states = jax.lax.scan(_step_fn, init=init_state, xs=xs)
        return all_states, last_state


class RNNModel(nn.Module):
    hidden_dim: int
    num_layers: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, xs, init_state):
        # xs: [seq_len, input_dim]
        # init_state: [num_layers, hidden_dim]
        outs, states = [], []
        for layer in range(self.num_layers):
            xs, state = GRU(self.hidden_dim, self.dtype, self.param_dtype)(xs, init_state[layer])
            outs.append(xs)
            states.append(state)

        # sum outputs from all layers, kinda like in ResNet
        return jnp.array(outs).sum(0), jnp.array(states)


BatchedRNNModel = flax.linen.vmap(
    RNNModel, variable_axes={"params": None}, split_rngs={"params": False}, axis_name="batch"
)

class EmbeddingEncoder(nn.Module):
    map_tile_emb_dim: int = 4
    embedding_dtype: Optional[Dtype] = jnp.float32
    param_dtype: Dtype = jnp.float32
    fixed_env_params: EnvParams = EnvParams()
    unit_cardinal_emb_dim: int = 8
    
    @nn.compact
    def __call__(self, obs):
        # 3 tiles types, empty, nebula, asteriod, and -1 for unknwon
        tile_type_embeder = nn.Embed(4, self.map_tile_emb_dim, self.embedding_dtype, self.param_dtype) # B x ENV x w x h x map_tile_emb_dim

        # Encode each cardinal unit into an embedding
        tile_type_embeddings = tile_type_embeder(obs.tile_type) # (B, ENV, w, h, 4)
        unit_counts_map_reshaped = obs.unit_counts_player_0[..., jnp.newaxis] # (B, ENV, w, h, 1)
        relic_map_reshaped = obs.relic_map[..., jnp.newaxis].astype(jnp.float32) # (B, ENV, w, h, 1)
    
        normalized_reward_last_round_reshaped = jnp.array(obs.normalized_reward_last_round)
        normalized_reward_last_round_reshaped =  jnp.reshape(normalized_reward_last_round_reshaped, normalized_reward_last_round_reshaped.shape + (1,1,1)) # (B, Env) -> # (B, Env, 1,1,1)
        normalized_reward_last_round_reshaped = jnp.tile(normalized_reward_last_round_reshaped, reps=(1, 1) + relic_map_reshaped.shape[-3:])  # (B, Env, w,h,1)

        grid_embedding = jnp.concatenate(
            [
                tile_type_embeddings,
                unit_counts_map_reshaped,
                relic_map_reshaped,
                normalized_reward_last_round_reshaped
            ],
            axis=-1, # Concatenate along the last axis (channels after wxh)
        ) # grid_embedding shape (w, h, t*(self.dim + 1)+4+1) , made 13 so 28  + 4 = 32 channels

        return grid_embedding

# Re-use the ResNet block and convolutional encoder from before.
class ResNetBlock(nn.Module):
    features: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.relu(x)
        x = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME',
                    kernel_init=orthogonal(math.sqrt(2)),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, kernel_size=(5, 5), padding='SAME',
                    kernel_init=orthogonal(math.sqrt(2)),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )(x)        
        return x + residual

# Agent-specific feature extraction using direct indexing
class SpatialFeatureExtractor(nn.Module):
    @nn.compact
    def __call__(self, feature_map, agent_positions, mask):
        """
        Args:
          feature_map: jnp.array of shape [B, T, H, W, C]
          agent_positions: jnp.array of shape [B, T, num_agents, 2] where each entry is (row, col)
          mask: jnp.array of shape [B, T, num_agents] of booleans indicating if an agent exists.

        Returns:
          agent_features: jnp.array of shape [B, T, num_agents, C]
        """
        def extract_features_for_timestep(fmap, positions, mask_t):
            # fmap: [H, W, C]
            # positions: [num_agents, 2]
            # mask_t: [num_agents] (boolean)
            def extract_feature(pos, valid):
                row, col = pos  # extract row and col indices
                feat = fmap[row, col, :]  # extract feature vector at that spatial location
                # If the agent doesn't exist (valid == False), return zeros.
                return feat#jnp.where(valid, feat, jnp.zeros_like(feat))
            # Vectorize over the num_agents dimension.
            return debuggable_vmap(extract_feature)(positions, mask_t)
        
        # First, map over the time dimension.
        extract_over_time = debuggable_vmap(extract_features_for_timestep, in_axes=(0, 0, 0))
        # Then, map over the batch dimension.
        agent_features = debuggable_vmap(extract_over_time, in_axes=(0, 0, 0))(
            feature_map, agent_positions, mask
        )
        return agent_features
        

class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int] = None
    rnn_hidden_dim: int = 64
    rnn_num_layers: int = 1
    head_hidden_dim: int = 64
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    features_dim = 16

    @nn.compact
    # doesn't get done calls, how does it reset memory?
    def __call__(self, obs: WrappedEnvObs, hidden: jax.Array) -> tuple[distrax.Categorical, jax.Array, jax.Array]:
        B, S = jax.tree_util.tree_leaves(obs)[0].shape[:2]

       
        grid_encoder = nn.Sequential(
            [
                # For small dims nn.Embed is extremely slow in bf16, so we leave everything in default dtypes
                EmbeddingEncoder(),
                nn.Conv(
                    self.features_dim,
                    (3, 3),
                    padding="SAME",
                    kernel_init=orthogonal(math.sqrt(2)),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                ResNetBlock(features=self.features_dim, dtype=self.dtype, param_dtype=self.param_dtype),
                ResNetBlock(features=self.features_dim, dtype=self.dtype, param_dtype=self.param_dtype),
            ]
        )

        rnn_core = BatchedRNNModel(
            self.rnn_hidden_dim, self.rnn_num_layers, dtype=self.dtype, param_dtype=self.param_dtype
        )
        actor = nn.Sequential(
            [
                nn.Dense(
                    self.head_hidden_dim, kernel_init=orthogonal(2), dtype=self.dtype, param_dtype=self.param_dtype
                ),
                nn.tanh,
                nn.Dense(
                    math.prod(self.action_dim), kernel_init=orthogonal(0.01), dtype=self.dtype, param_dtype=self.param_dtype
                ),
            ]
        )
        critic = nn.Sequential(
            [
                nn.Dense(
                    self.head_hidden_dim, kernel_init=orthogonal(2), dtype=self.dtype, param_dtype=self.param_dtype
                ),
                nn.tanh,
                nn.Dense(1, kernel_init=orthogonal(1.0), dtype=self.dtype, param_dtype=self.param_dtype),
            ]
        )

        # [batch_size, seq_len, ...]
        obs_emb = grid_encoder(obs)

        local_features = SpatialFeatureExtractor()(obs_emb, obs.unit_positions_player_0, obs.unit_mask_player_0)  # [b x s x num_agents, features]
        local_features = local_features.reshape(local_features.shape[:-2] + (-1,)) # flatten num_agents and features

        # [batch_size, seq_len, hidden_dim + 2 * act_emb_dim + 1]
        #out = jnp.concatenate([obs_emb, dir_emb, act_emb, inputs["prev_reward"][..., None]], axis=-1)

        # core networks
        out, new_hidden = rnn_core(local_features, hidden)

        # casting to full precision for the loss, as softmax/log_softmax
        # (inside Categorical) is not stable in bf16
        logits = actor(out).astype(jnp.float32)
        logits = logits.reshape(logits.shape[:-1] + self.action_dim) 

        dist = distrax.Independent(
            distrax.Categorical(logits=logits),
            reinterpreted_batch_ndims=1 # last dimension is not independent
        )

        values = critic(out)

        return dist, jnp.squeeze(values, axis=-1), new_hidden

    def initialize_carry(self, batch_size):
        return jnp.zeros((batch_size, self.rnn_num_layers, self.rnn_hidden_dim), dtype=self.dtype)
