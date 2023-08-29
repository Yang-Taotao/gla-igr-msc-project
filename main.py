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
from data import gw_fim, gw_plt, gw_rpl, vi_dat
from data.gw_cfg import MCS, ETAS, PARAM_TEST, F_SIG, F_PSD
# Setup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
cc.initialize_cache("./data/__jaxcache__")

# %%
# First compilation test for sub modules
# Wavefor generation
HP = gw_rpl.waveform_plus_restricted(PARAM_TEST, F_SIG)
HC = gw_rpl.waveform_cros_restricted(PARAM_TEST, F_SIG)
# Gradient calculation
GP = gw_rpl.gradient_plus(PARAM_TEST)
GC = gw_rpl.gradient_cros(PARAM_TEST)
# FIM test statistics calculation
DETP = gw_fim.log_sqrt_det_plus(PARAM_TEST)
DETC = gw_fim.log_sqrt_det_cros(PARAM_TEST)
# First compilation - results checker
print(f"Test waveform HP.shape:{HP.shape} hc.shape:{HC.shape}")
print(f"Test gradient gp.shape:{GP.shape} gc.shape:{GC.shape}")
print(f"Test log density detp:{DETP:.4g} detc:{DETC:.4g}")

# %%
# FIM density calc params
FIM_PARAM = gw_fim.fim_param_build(MCS, ETAS)
print(f"fim_param.shape:{FIM_PARAM.shape}")

# %%
# New compilation for vectorized operaions
DENSITY_P = gw_fim.log_density_plus(FIM_PARAM).reshape([len(MCS), len(ETAS)])
DENSITY_C = gw_fim.log_density_cros(FIM_PARAM).reshape([len(MCS), len(ETAS)])

# %%
# Plot Generation
gw_plt.ripple_waveform(F_SIG, HP, waveform="hp")
gw_plt.ripple_waveform(F_SIG, HC, waveform="hc")
gw_plt.ripple_gradient(F_SIG, HP, HC, param="mc")
gw_plt.ripple_gradient(F_SIG, HP, HC, param="eta")
gw_plt.bilby_noise_psd(F_SIG, F_PSD)
gw_plt.log_fim_contour(MCS, ETAS, DENSITY_P, waveform="hp")
gw_plt.log_fim_contour(MCS, ETAS, DENSITY_C, waveform="hc")
gw_plt.log_fim_param(MCS, DENSITY_P, waveform= "hp",param= "mc")
gw_plt.log_fim_param(ETAS, DENSITY_P, waveform= "hp",param= "eta")
gw_plt.log_fim_param(MCS, DENSITY_C, waveform= "hc",param= "mc")
gw_plt.log_fim_param(ETAS, DENSITY_C, waveform= "hc",param= "eta")

# %%
# Flow training
vi_dat.train_flow()

# %%
