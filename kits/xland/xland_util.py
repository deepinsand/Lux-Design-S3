import jax
import jax.numpy as jnp
import chex

class MultiDiscrete:
    """A minimal jittable MultiDiscrete space for Gymnax."""

    def __init__(self, nvec, n):
        # Convert to a JAX array (if not already)
        self.nvec = jnp.array(nvec, dtype=jnp.int_)
        self.shape = self.nvec.shape
        self.n = n
        self.dtype = jnp.int_

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        flat_nvec = self.nvec.flatten()
        num_elements = flat_nvec.shape[0]
        # Split the RNG for each element
        keys = jax.random.split(rng, num_elements)
        # Use vmap to sample for each discrete space in a vectorized way.
        sample_fn = jax.vmap(lambda key, n: jax.random.randint(key, shape=(), minval=0, maxval=n))
        samples_flat = sample_fn(keys, flat_nvec)
        return samples_flat.reshape(self.shape)

    def contains(self, x: chex.Array) -> jnp.ndarray:
        # Check that all entries are within their respective bounds.
        return jnp.all((x >= 0) & (x < self.nvec))
