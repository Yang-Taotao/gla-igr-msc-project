import jax
import jax.numpy as jnp
from jax import vmap

# Example function that takes a (4,) array and returns a float
def single_function(arr):
    return jnp.sum(arr)

# Create a (n, 4) array using jnp.column_stack()
n = 5  # Number of rows
array_2d = jnp.column_stack([jnp.arange(1, 5) for _ in range(n)])

# Vectorize the function using vmap
vectorized_function = vmap(single_function)

# Apply the vectorized function to the (n, 4) array
result = vectorized_function(array_2d)
print(result)
print(type(result))