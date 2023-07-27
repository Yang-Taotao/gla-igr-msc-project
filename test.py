# %%
import jax.numpy as jnp

x = jnp.arange(1, 5, 2)
y = jnp.arange(1, 10, 3)

X, Y = jnp.meshgrid(x, y, indexing='ij')
