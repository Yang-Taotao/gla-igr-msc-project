# %%
import jax
import jax.numpy as jnp
from data import gw_fim, gw_rpl

data_mc = jnp.linspace(1, 21, 1000)
data_eta = jnp.linspace(0.05, 0.25, 1000)

# Get results
data_tc = 0.0*jnp.ones_like(data_mc)
data_phic = 0.0*jnp.ones_like(data_mc)
# Param build with shape (n, 4)
param = jnp.column_stack((data_mc, data_eta, data_tc, data_phic))

num_batch = int(param.shape[0] // (param.shape[0]/10))

# %%
