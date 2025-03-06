import time
import jax
#jax.config.update("jax_disable_jit", True)
import jax.numpy as jnp
import jax
import jax.numpy as jnp

def create_permutation_mask_vectorized(n_cols):
  
  num_rows = 2**n_cols
  row_indices = jnp.arange(num_rows, dtype=jnp.int32)
  col_indices = jnp.arange(n_cols, dtype=jnp.int32)

  # Create a grid for row indices (but not for column indices this time)
  rows = jnp.tile(row_indices[:, None], (1, n_cols)) # Tile row_indices to match column count

  # Use bitwise operations with correct broadcasting against 1D col_indices
  mask = (rows >> col_indices[None, :]) & 1

  return mask # No reshape needed now as the shape is already (num_rows, n_cols)



def get_certain_positions(permutations, masks, rewards):
    storage_size = permutations.shape[1]
    permutation_rewards = jnp.matmul(permutations, masks.T, preferred_element_type=jnp.int32)
    equals_rewards = permutation_rewards == rewards
    valid_permutation_mask = jnp.all(equals_rewards, axis=1)
    first_masked_index = jnp.argmax(valid_permutation_mask) # Index of first True in mask
    first_valid_mask = permutations[first_masked_index]

    valid_permutation_mask = jnp.tile(valid_permutation_mask[:, jnp.newaxis], reps=(1, storage_size))
  

    mismatched_mask = valid_permutation_mask & (permutations != first_valid_mask)
    all_columns_are_the_same = ~jnp.any(mismatched_mask, axis=0)
    certain_positions = first_valid_mask & all_columns_are_the_same
    return (all_columns_are_the_same, certain_positions)

jit_bitwise_and = jax.jit(get_certain_positions)

# Create input data (as before)

large_vector = create_permutation_mask_vectorized(25)
stored_masks = create_permutation_mask_vectorized(2) #jnp.array([0, 1, 2, 3])
stored_masks = jnp.pad(stored_masks, ((0,0),(0,23)), mode='constant', constant_values=0)
stored_rewards = jnp.array([0, 1, 1, 2])

# JIT compile the function (first run will be slower)
jit_bitwise_and(large_vector, stored_masks, stored_rewards)

# Time the JIT-compiled function
start_time = time.time()
result = jax.block_until_ready(jit_bitwise_and(large_vector, stored_masks, stored_rewards))
end_time = time.time()

print(f"Time taken: {end_time - start_time:.4f} seconds")

print(f"Result: {result}")