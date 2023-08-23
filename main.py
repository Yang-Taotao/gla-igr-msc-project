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
from data.gw_cfg import f_sig, f_psd, mcs, etas, param_test
# Setup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
cc.initialize_cache("./data/__jaxcache__")

# %%
# First compilation
# t~1min20s
# Wavefor generation
hp = gw_rpl.waveform_plus_restricted(param_test, f_sig)
hc = gw_rpl.waveform_cros_restricted(param_test, f_sig)
# Gradient calculation
gp = gw_rpl.gradient_plus(param_test)
gc = gw_rpl.gradient_cros(param_test)
# FIM test statistics calculation
detp = gw_fim.log10_sqrt_det_plus(param_test)
detc = gw_fim.log10_sqrt_det_cros(param_test)
# First compilation - results checker
print(f"Test waveform hp.shape:{hp.shape} hc.shape:{hc.shape}")
print(f"Test gradient gp.shape:{gp.shape} gc.shape:{gc.shape}")
print(f"Test log10 density detp:{detp:.4g} detc:{detp:.4g}")

# %%
# FIM density calc params
# Cached shape - (100, 100, 1, 1)
fim_param = gw_fim.fim_param_build(mcs, etas)
print(f"fim_param.shape:{fim_param.shape}")

# %%
# Density matrix batching
# t~43.1s for (100, 100) shape
density_p = gw_fim.density_batch_calc(
    fim_param, mcs, etas, batch_size=100, waveform="hp")
density_c = gw_fim.density_batch_calc(
    fim_param, mcs, etas, batch_size=100, waveform="hc")
print(f"Metric density_p.shape:{density_p.shape}")
print(f"Metric density_c.shape:{density_c.shape}")

# %%
# Plot Generation
gw_plt.ripple_waveform(f_sig, hp, waveform="hp")
gw_plt.ripple_waveform(f_sig, hc, waveform="hc")
gw_plt.ripple_gradient(f_sig, hp, hc, param="mc")
gw_plt.ripple_gradient(f_sig, hp, hc, param="eta")
gw_plt.bilby_noise_psd(f_sig, f_psd)
gw_plt.log_fim_contour(mcs, etas, density_p, waveform="hp")
gw_plt.log_fim_contour(mcs, etas, density_c, waveform="hc")

# %%
