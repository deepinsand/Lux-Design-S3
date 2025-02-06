import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from packages.purejaxrl.purejaxrl.jax_debug import loop_based_vmap_replacement, debuggable_vmap
from typing import NamedTuple

class TestNamedTuple(NamedTuple):
    a: int
    b: int

class TestClass:

    @partial(jit, static_argnums=(0,))
    def turn_1_negative(self, x): # Modified function name
        cond1 = jnp.all(x != 1)
        def true_fn1(x):
            return x
        def false_fn1(x):
            return jnp.array([-1, -1]) # Return -1 array, matching input shape
        return jax.lax.cond(cond1, true_fn1, false_fn1, x) 


    @partial(jit, static_argnums=(0,))
    def turn_1_negative_tuple(self, x): # Modified function name
        cond1 = jnp.all(x != 1)
        def true_fn1(x):
            return tuple(x)
        def false_fn1(x):
            return (-1, -1)
        return jax.lax.cond(cond1, true_fn1, false_fn1, x) 
    
    @partial(jit, static_argnums=(0,))
    def turn_1_negative_named_tuple(self, x): # Modified function name
        cond1 = jnp.all(x != 1)
        def true_fn1(x):
            return TestNamedTuple(x[0], x[1])
        def false_fn1(x):
            return TestNamedTuple(-1, -1)
        return jax.lax.cond(cond1, true_fn1, false_fn1, x) 

test_instance = TestClass()
x_input = jnp.array([[0, 0], [0, 1], [0, 2]])

print("--- Experiment A - JIT + vmap Modified with x = [[0, 0], [0, 1], [0, 2]] ---")
output_a_mod = debuggable_vmap(test_instance.turn_1_negative, in_axes=(0,))(x_input)
print("Output of Experiment A - Mod:", output_a_mod) # Print the output


print("--- Experiment B - JIT + tuple vmap Modified with x = [[0, 0], [0, 1], [0, 2]] ---")
output_b_mod = debuggable_vmap(test_instance.turn_1_negative_tuple, in_axes=(0,))(x_input)
print("Output of Experiment B - Mod:", output_b_mod) # Print the output


print("--- Experiment C - JIT + named tuple vmap Modified with x = [[0, 0], [0, 1], [0, 2]] ---")
output_c_mod = debuggable_vmap(test_instance.turn_1_negative_named_tuple, in_axes=(0,))(x_input)
print("Output of Experiment C - Mod:", output_c_mod) # Print the output