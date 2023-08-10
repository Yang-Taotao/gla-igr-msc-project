"""
This is the master script for MSc project.

Created on Thu August 03 2023
"""
# %%
# Library import
# Set XLA resource allocation
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
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
from data.gw_cfg import f_sig, mcs, etas
from data import gw_fim, gw_plt, gw_rpl

# %%
# Config setup
# ============================================================ #
# Frequency - min, max, step
f_min, f_max, f_del = 24.0, 512.0, 0.5
# Chirp mass - min, max, step
mc_min, mc_max, mc_num = 1.000, 21.00, 100
# Mass ratio - min, max, step
eta_min, eta_max, eta_num = 0.050, 0.250, 100
# Base theta - mc, eta, s1, s2, dist_mpc, c_time, c_phas, ang_inc, and_pol
theta_base = jnp.array(
    [28.0956, 0.2471, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0]
)
# ============================================================ #

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
print(f"Test waveform.shape:'{hp.shape}")
print(f"Test gradient.shape:'{gp.shape}")
print(f"Test log10.sqrt.det.FIM:'{detp:.4g}")

# %%
# FIM density calc params
# Cached shape - (100, 100, 1, 1)
FIM_PARAM = gw_fim.fim_param_build(mcs, etas)
print(f"FIM_PARAM.shape:{FIM_PARAM.shape}")

# %%
# Density matrix batching
# t~43.1s for (100, 100) shape
batch_size = 50
DENSITY = gw_fim.density_batch_calc(FIM_PARAM, mcs, etas, batch_size)
print(f"Metric Density.shape:':{DENSITY.shape}")

# %% 
# Plot Generation
gw_plt.fim_contour_mc_eta_log10(mcs, etas, DENSITY)

# %%
