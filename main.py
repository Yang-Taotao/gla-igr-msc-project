"""
This is the master script for MSc project.

Created on Thu August 03 2023
"""
# %%
# Library import
# Set XLA resource allocation
import os
# Use jax and persistent cache
import jax.numpy as jnp
from jax.experimental.compilation_cache import compilation_cache as cc
# Plotter style customization
import matplotlib.pyplot as plt
import scienceplots
# Custom packages
from data import gw_fim, gw_plt, gw_rpl
from data.gw_cfg import f_sig, mcs, etas
# Setup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
cc.initialize_cache("./data/__jaxcache__")
plt.style.use(['science', 'notebook', 'grid'])

# %%
# First compilation
# Parameters assignment - t~1min20s
mc, eta = 20, 0.2
fisher_params = jnp.array([mc, eta, 0.0, 0.0])
# First compile if no persistent cache
hp = gw_rpl.waveform_plus_restricted(fisher_params, f_sig)
gp = gw_rpl.gradient_plus(fisher_params)
detp = gw_fim.log10_sqrt_det(fisher_params)
# First compilation - results checker
print(f"Test waveform.shape:{hp.shape}")
print(f"Test gradient.shape:{gp.shape}")
print(f"Test log10.sqrt.det.FIM:{detp:.4g}")

# %%
# FIM density calc params
# Cached shape - (100, 100, 1, 1)
fim_param = gw_fim.fim_param_build(mcs, etas)
print(f"fim_param.shape:{fim_param.shape}")

# %%
# Density matrix batching
# t~43.1s for (100, 100) shape
density = gw_fim.density_batch_calc(fim_param, mcs, etas, batch_size=50)
print(f"Metric Density.shape:{density.shape}")

# %%
# Plot Generation
gw_plt.fim_contour_mc_eta_log10(mcs, etas, density)

# %%
