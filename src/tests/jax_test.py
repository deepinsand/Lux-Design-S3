import jax
import jax.numpy as jnp
import jax.lax as lax

@jax.jit
def f(x):
  def true_fun(x):
    return x * 2

  def false_fun(x):
    return x * 3

  pred = x > 0  # This now creates a JAX boolean tracer
  return lax.cond(pred, true_fun, false_fun, x)  # Important: Pass x as an argument

print(f(jnp.array(5)))   # Output: 10
print(f(jnp.array(-2)))  # Output: -6

# This will still not work because the condition depends on the value of x
# print(f(5))  # Error: Can't convert tracer to bool