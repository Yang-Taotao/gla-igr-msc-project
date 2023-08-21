"""
This is the master script for MSc project.

Created on Thu August 03 2023
"""
# %%
# Library import
# Set XLA resource allocation
import os
# Use jax and persistent cache
from jax.experimental.compilation_cache import compilation_cache as cc
# Custom packages
from data import gw_fim, gw_plt, gw_rpl
from data.gw_cfg import f_sig, mcs, etas, test_params
# Setup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
cc.initialize_cache("./data/__jaxcache__")

# %%
# First compilation
# t~1min20s
# Wavefor generation
hp = gw_rpl.waveform_plus_restricted(test_params, f_sig)
hc = gw_rpl.waveform_cros_restricted(test_params, f_sig)
# Gradient calculation
gp = gw_rpl.gradient_plus(test_params)
gc = gw_rpl.gradient_cros(test_params)
# FIM test statistics calculation
detp = gw_fim.log10_sqrt_det_plus(test_params)
detc = gw_fim.log10_sqrt_det_cros(test_params)
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
density = gw_fim.density_batch_calc(fim_param, mcs, etas, batch_size=100)
print(f"Metric Density.shape:{density.shape}")

# %%
# Plot Generation
gw_plt.fim_contour_mc_eta_log10(mcs, etas, density)

# %%
