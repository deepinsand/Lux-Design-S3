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
    action_dim: Sequence[int] = None
    rnn_hidden_dim: int = 64
    rnn_num_layers: int = 1
    head_hidden_dim: int = 64
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    # doesn't get done calls, how does it reset memory?
    def __call__(self, obs: WrappedEnvObs, hidden: jax.Array) -> tuple[distrax.Categorical, jax.Array, jax.Array]:
        B, S = jax.tree_util.tree_leaves(obs)[0].shape[:2]

        # encoder from https://github.com/lcswillems/rl-starter-files/blob/master/model.py
       
        img_encoder = nn.Sequential(
            [
                # For small dims nn.Embed is extremely slow in bf16, so we leave everything in default dtypes
                EmbeddingEncoder(),
                nn.Conv(
                    16,
                    (2, 2),
                    padding="VALID",
                    kernel_init=orthogonal(math.sqrt(2)),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                nn.relu,
                nn.Conv(
                    32,
                    (2, 2),
                    padding="VALID",
                    kernel_init=orthogonal(math.sqrt(2)),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                nn.relu,
                nn.Conv(
                    64,
                    (2, 2),
                    padding="VALID",
                    kernel_init=orthogonal(math.sqrt(2)),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                nn.relu,
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
        obs_emb = img_encoder(obs).reshape(B, S, -1)

        # [batch_size, seq_len, hidden_dim + 2 * act_emb_dim + 1]
        #out = jnp.concatenate([obs_emb, dir_emb, act_emb, inputs["prev_reward"][..., None]], axis=-1)

        # core networks
        out, new_hidden = rnn_core(obs_emb, hidden)

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
