import time
import jax
#jax.config.update("jax_disable_jit", True)
import jax.numpy as jnp

def mask_to_32_bit_integer(mask):

    # Create an array of bit positions (0, 1, 2, ...)
    positions = jnp.arange(mask.size, dtype=jnp.int32)

    # Calculate powers of 2 for each bit position (2^0, 2^1, 2^2, ...)
    powers_of_2 = 2**positions

    # Multiply each bit (0 or 1) by its corresponding power of 2
    weighted_bits = mask * powers_of_2

    # Sum the weighted bits to get the final integer
    return jnp.sum(weighted_bits).astype(jnp.int32) # Explicitly cast to int32 for clarity


def create_centered_mask(pos, mask_size=5, n=24, dtype=jnp.int32):
  half_mask = mask_size // 2
  center_x, center_y = pos
  rows = jnp.arange(n)
  cols = jnp.arange(n)
  row_grid, col_grid = jnp.meshgrid(rows, cols)

  row_start = center_y - half_mask
  row_end = center_y + half_mask + 1 # +1 because range is exclusive at the end
  col_start = center_x - half_mask
  col_end = center_x + half_mask + 1

  # Create the mask by checking if each cell's row and column
  # are within the mask boundaries.
  mask = (row_grid >= row_start) & (row_grid < row_end) & \
         (col_grid >= col_start) & (col_grid < col_end)

  return mask.astype(dtype) # Convert boolean mask to integer (0 and 1)


def extract_32bit_from_grid_mask(grid, pos, subsection_size=5):

  half_subsection = subsection_size // 2
  padding_size = half_subsection
  padding = ((padding_size, padding_size), (padding_size, padding_size))
  padded_grid = jnp.pad(grid, padding, mode='constant', constant_values=0)

  center_x, center_y = pos
  # Calculate start and end indices for rows and columns, handling edges
  start_row = center_x - half_subsection + padding_size
  start_col = center_y - half_subsection + padding_size

  # Extract the subsection using lax.dynamic_slice
  subsection = jax.lax.dynamic_slice(padded_grid, (start_row, start_col), (subsection_size, subsection_size))
  return mask_to_32_bit_integer(subsection.reshape((-1,)))
    

def reconstruct_grid_from_subsection_bit_mask(subsection, pos, grid_size, subsection_size=5):

  half_subsection = subsection_size // 2
  padding_size = half_subsection

  # 1. Initialize and Pad the reconstructed grid
  padded_reconstructed_grid = jnp.zeros((grid_size + 2 * padding_size, grid_size + 2 * padding_size), dtype=subsection.dtype)

  center_x, center_y = pos

  # 2. Calculate the start position in the PADDED reconstructed grid
  start_row = center_x - half_subsection + padding_size
  start_col = center_y - half_subsection + padding_size

  subsection = subsection.reshape((subsection_size, subsection_size))
  # 3. Place the subsection into the PADDED reconstructed grid
  padded_reconstructed_grid = jax.lax.dynamic_update_slice(
      padded_reconstructed_grid, subsection, (start_row, start_col)
  )

  # 4. Unpad to get back to the original grid size
  unpadded_reconstructed_grid = jax.lax.dynamic_slice(
      padded_reconstructed_grid, (padding_size, padding_size), (grid_size, grid_size)
  )

  return unpadded_reconstructed_grid

   

def count_set_bits(n):
  """Counts set bits (1s) in the bitwise representation of an integer."""
  n = (n & 0x55555555) + ((n >> 1) & 0x55555555)  # Sum pairs of bits
  n = (n & 0x33333333) + ((n >> 2) & 0x33333333)  # Sum quartets
  n = (n & 0x0F0F0F0F) + ((n >> 4) & 0x0F0F0F0F)  # Sum octets
  n = (n & 0x00FF00FF) + ((n >> 8) & 0x00FF00FF)  # Sum 16-bit sequences
  n = (n & 0x0000FFFF) + ((n >> 16) & 0x0000FFFF) # Sum 32-bit sequences (for 32-bit integers)
  return n

def find_uniform_bit_positions(arr, mask):
  num_bits = 25
  bits = jnp.arange(num_bits, dtype=jnp.int32)
  
  def do_all_permutations_share_bit(bit_position):
    # Shift right to bring the bit at bit_position to the least significant bit
    shifted_arr = arr >> bit_position
    # Isolate the least significant bit using bitwise AND with 1
    bit_values = shifted_arr & 1

    # what if mask has nothing?? deal with it later
    first_masked_index = jnp.argmax(mask) # Index of first True in mask
    first_bit_value = bit_values[first_masked_index]

    mismatched_mask = mask & (bit_values != first_bit_value)
    return (~jnp.any(mismatched_mask), first_bit_value)


  return jax.vmap(do_all_permutations_share_bit)(bits)


def get_certain_positions(permutations, masks, rewards):
    storage_size = masks.size
    tiled_permutations = jnp.tile(permutations[:, jnp.newaxis], reps=(1, storage_size))

    bitwise_anded = tiled_permutations & masks
    bit_counted = count_set_bits(bitwise_anded)
    equals_rewards = bit_counted == rewards
    valid_permutation_mask = jnp.all(equals_rewards, axis=1).astype(jnp.int32)
    return find_uniform_bit_positions(permutations, valid_permutation_mask)

if __name__ == "__main__":

  jit_bitwise_and = jax.jit(get_certain_positions)

  # Create input data (as before)
  vector_size = 2**25
  large_vector = jnp.arange(vector_size, dtype=jnp.int32)
  #stored_masks = jnp.array([0, 1, 2, 3]) 
  #stored_rewards = jnp.array([0, 1, 1, 2])



  # stored at timestamps array([18, 19, 20, 22, 23, 29, 31, 32]),)
  # The inclusion of the last mask causes the 5th and 25th bits to flip.  This is wrong since 
  # based on 16777240, only one of 5th and 25th can be true. 
  #stored_masks = jnp.array([     128,      256,      512,    16384,   524288,   524296, 16777240,   524304]) 
  #stored_rewards = jnp.array([1, 1, 0, 0, 1, 1, 1, 1]) 


  # happened at [36, 37, 38, 39],
  stored_rewards = jnp.array([3, 2, 2, 2]) 
  stored_masks = jnp.array([2120, 4233, 8458, 8520])

  #solved_energy_points_mask.astype(jnp.int32)
#Array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

#Array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)

  # JIT compile the function (first run will be slower)
  jit_bitwise_and(large_vector, stored_masks, stored_rewards)

  # Time the JIT-compiled function
  start_time = time.time()
  result_mask, result_values = jax.block_until_ready(jit_bitwise_and(large_vector, stored_masks, stored_rewards))

  result = (result_mask.astype(jnp.int16) * result_values) - (1 - result_mask.astype(jnp.int16))

  end_time = time.time()

  print(f"Time taken: {end_time - start_time:.4f} seconds")

  print(f"Result: {result},")