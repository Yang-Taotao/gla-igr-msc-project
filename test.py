# %%
import jax
import jax.numpy as jnp
from data import gw_fim, gw_rpl

data_mc = jnp.linspace(1, 21, 1000)
data_eta = jnp.linspace(0.05, 0.25, 1000)

res = jnp.mean(data_mc-data_eta)
res.shape

# %%
