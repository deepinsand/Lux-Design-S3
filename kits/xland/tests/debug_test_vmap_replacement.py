import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from kits.purejaxrl.jax_debug import loop_based_vmap_replacement

class TestClass:

    @partial(jit, static_argnums=(0,))
    def breakpoint_if_nonfinite_a_mod(self, x): # Modified function name
       # jax.debug.print("Input x (Exp A - Mod): {}", x)
        cond1 = jnp.all(x != 1)
       # jax.debug.print("Condition 1 (x != jnp.inf).all() Pre-cond (Exp A - Mod): {}", cond1)
       # jax.debug.print("Condition 1 shape: {}, dtype: {}", cond1.shape, cond1.dtype)
        def true_fn1(x):
            # jax.debug.print("true_fn1 (Exp A - Mod): team_points: {}, cond1: {}", x, cond1) # Removed print
            return x
        def false_fn1(x):
            #jax.debug.breakpoint() # Replaced breakpoint with return value
            return jnp.array([-1, -1]) # Return -1 array, matching input shape
        return jax.lax.cond(cond1, true_fn1, false_fn1, x) # Use bool cast condition


test_instance = TestClass()
x_input = jnp.array([[0, 0], [0, 1], [0, 2]])

print("--- Experiment A - JIT + vmap Modified with x = [[0, 0]] ---")
output_a_mod = loop_based_vmap_replacement(test_instance.breakpoint_if_nonfinite_a_mod, in_axes=(None,))(x_input)
print("Output of Experiment A - Mod:", output_a_mod) # Print the output


# print("--- Experiment B - JIT + with x = [[0, 0]] ---")
# output_b_mod = test_instance.breakpoint_if_nonfinite_a_mod(x_input)
# print("Output of Experiment B - Mod:", output_b_mod) # Print the output