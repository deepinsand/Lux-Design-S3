import jax
import jax.numpy as jnp

@jax.jit
def breakpoint_if_nonfinite_using_isfinite(x):
    cond = jnp.isfinite(x).all()
    jax.lax.cond(cond, lambda x: None, lambda x: jax.debug.breakpoint(), x)

@jax.jit
def breakpoint_if_nonfinite_using_not_equals(x):
    cond = jnp.all(x != jnp.inf)
    jax.lax.cond(cond, lambda x: None, lambda x: jax.debug.breakpoint(), x)

x = jnp.array([0, 0])
x_vectorized = jnp.array([x])

print("--- Experiment 1 - testing isfinite(x).all using jit ---")
breakpoint_if_nonfinite_using_isfinite(x)

print("--- Experiment 2 - testing jnp.all(x != jnp.inf) using jit ---")
breakpoint_if_nonfinite_using_not_equals(x)

print("--- Experiment 3 - testing isfinite(x).all using vmap ---")
jax.vmap(breakpoint_if_nonfinite_using_isfinite)(x_vectorized)

print("--- Experiment 4 - testing jnp.all(x != jnp.inf) using vmap ---")
jax.vmap(breakpoint_if_nonfinite_using_not_equals)(x_vectorized)