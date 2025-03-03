
import jax

# Update the default device to the CPU
jax.config.update("jax_disable_jit", True)
jax.config.update("jax_default_device", jax.devices("cpu")[0])


import jax.numpy as jnp


def make_anti_diagonal_symmetric_or(mask):
  reflected_mask = mask[::-1, ::-1].T  # Flip both dims and transpose
  symmetric_mask = jnp.logical_or(mask, reflected_mask).astype(mask.dtype) # Ensure same dtype as input
  return symmetric_mask

if __name__ == "__main__":

    mask2 = jnp.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0]
    ])

    enumerated = jnp.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

    pos = jnp.array([
        [0, 0], [1, 1],
        [-1, -1], [-1, -1],
        [2, 2], [2, 1],
    ])

    mask = jnp.array([
        True, True, False, False, True, True
    ])

    num_relics = 6
    half_num_relics = 3


    reshaped_mask = jnp.stack([mask, mask]).T.astype(jnp.int16) 
    inverse_negative_reshaped_mask = reshaped_mask - 1
    sym_pos = 3 - pos[:, [1,0]]
    sym_pos = sym_pos * reshaped_mask + inverse_negative_reshaped_mask

    half_num_relics = 3
    flipped_sym_pos = jnp.concatenate([sym_pos[half_num_relics:, :], sym_pos[:half_num_relics, :]], axis=0)

    mask7 =  jnp.maximum(pos, flipped_sym_pos)
    symmetric_mask2 = make_anti_diagonal_symmetric_or(mask2)
    print("\nOriginal Mask 2:\n", mask2)
    print("Symmetric Mask 2:\n", symmetric_mask2)