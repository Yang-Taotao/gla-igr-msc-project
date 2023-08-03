"""
This is the master script for MSc project.

Created on Thu August 03 2023
"""
# %%
# Library import
# Set XLA resource allocation
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# Other imports
import itertools as it
from tqdm import tqdm
# Use jax and persistent cache
import jax
import jax.numpy as jnp
from jax.experimental.compilation_cache import compilation_cache as cc
cc.initialize_cache("./data/__jaxcache__")
# Plotter style customization
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'notebook', 'grid'])
# Custom packages
from data.gw_config import f_sig
from data.gw_fisher import log10_sqrt_det
from data.gw_ripple import gradient_plus, waveform_plus_restricted

# %%
# First compilation
# Parameters assignment - t~1min20s
mc, eta = 20, 0.2
fisher_params = jnp.array([mc, eta, 0.0, 0.0])
# First compile if no persistent cache
hp = waveform_plus_restricted(fisher_params, f_sig)
gp = gradient_plus(fisher_params)
detp = log10_sqrt_det(fisher_params)
# First compilation - results checker
print(f"{'First Compilation Test Results':<30}")
print("=" * 30)
print(f"{'Test waveform.shape:':<20}{hp.shape}")
print(f"{'Test gradient.shape:':<20}{gp.shape}")
print(f"{'Test log10.sqrt.det.FIM:':<20}{detp:>10.4g}")
print("=" * 30)

# %%
# FIM density calc params
# Cached shape - (100, 100, 1, 1)
mcs = jnp.linspace(1.000, 21.00, num=100, dtype=jnp.float32)
etas = jnp.linspace(0.050, 0.250, num=100, dtype=jnp.float32)


def fim_param_build(mcs, etas):
    """
    Build 4-D FIM_PARAM grid with mc and eta entries:
    [mc, eta, tc, phic]
    """
    # Set (1, ) shape zero value array
    zeros = jnp.zeros(1, dtype=jnp.float32)
    # Param array - mc, eta, tc, phic
    param_arr = [mcs, etas, zeros, zeros]
    # Build 4-d mesh
    ND_PARAM = jnp.meshgrid(*param_arr, indexing='ij')
    # Stack and reshape into (n, 4) shape FIM_PARAM array
    FIM_PARAM = jnp.stack(ND_PARAM, axis=-1).reshape(-1, len(param_arr))
    # Func return
    return FIM_PARAM


FIM_PARAM = fim_param_build(mcs, etas)
print("=" * 30)
print(f"{'FIM_PARAM.shape:':<20}{FIM_PARAM.shape}")
print("=" * 30)

# %%
# Density matrix batching
# t~43.1s for (100, 100) shape


def density_batch_calc(data, mcs, etas, batch_size=50):
    """
    Calculate DENSITY grid values with batching
    """
    # Define batch numbers
    num_batch = data.shape[0] // batch_size
    density_list = []
    # Batching
    for i in range(num_batch):
        # Split batches
        batch_fim_param = data[i * batch_size: (i + 1) * batch_size]
        # Call jax.vmap
        batch_density = jax.vmap(log10_sqrt_det)(batch_fim_param)
        # Add to results
        density_list.append(batch_density)
    # Concatenate the results from smaller batches
    DENSITY = jnp.concatenate(density_list).reshape([len(mcs), len(etas)])
    # Func return
    return DENSITY

DENSITY = density_batch_calc(FIM_PARAM, mcs, etas, batch_size=50)

print("=" * 30)
print(f"{'Metric Density.shape:':<20}{DENSITY.shape}")
print("=" * 30)

# %% 
# Plot Generation


def fim_mc_mr_contour_log10(data_x: jnp.ndarray, data_y: jnp.ndarray, data_z: jnp.ndarray):
    # Plot init
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plotter
    cs = ax.contourf(
        data_x, 
        data_y, 
        data_z.T,
        alpha=0.8,
        levels=20,
        cmap='gist_heat',
    )
    # Plot customization
    ax.set(
        xlabel=r'Chirp Mass ($M_\odot$)', 
        ylabel=r'Symmetric Mass Ratio $\eta$',
    )
    cb = plt.colorbar(cs, ax=ax)
    cb.ax.set_ylabel(r'$\log_{10}$ Template Bank Density')
    # Plot admin
    fig.savefig("./figures/fig_06_fim_mc_mr_contour_log10.png")

fim_mc_mr_contour_log10(mcs, etas, DENSITY)

# %%
