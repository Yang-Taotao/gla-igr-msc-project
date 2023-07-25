"""
This is the master script for MSc project.

Created on Thu Jul 11 2023

@author: Yang-Taotao
"""
# %%
# Section 0 - Library import
import jax.numpy as jnp
from data import gw_ripple, gw_fisher, gw_plotter

# %%
# Section 1.a -  Define GW data theta
# m1, m2, s1, s2, dist_mpc, c_time, c_phas, ang_inc, ang_pol
data_theta = (36.0, 29.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0)
# f_min, f_max, f_del
data_freq = (24.0, 512.0, 0.5)

# %%
# Section 1.b -  Generate GW data
# t~17.8s first compile
f_sig, f_ref = gw_ripple.freq_build(data_freq)
data_hp, data_hc = gw_ripple.waveform(data_theta)

# %%
# Section 1.c -  Plot GW data - GW170817
gw_plotter.ripple_waveform_plot(data_hp, data_hc, f_sig)

# %%
# Section 2.a -  Generate mapped grad data
# t~178.0s first compile
data_hp_grad = gw_ripple.grad_plus(data_theta)
data_hc_grad = gw_ripple.grad_cros(data_theta)

# %%
# Section 2.b -  Plot GW data - grad wrt data_idx
data_idx = 0, 1
gw_plotter.ripple_grad_plot_idx(data_hp_grad, data_hc_grad, f_sig, *data_idx)

# %%
# Section 3.a - FIM psd plot
gw_plotter.bilby_plot(f_sig, gw_fisher.bilby_psd(data_freq))

# %%
# Section 3.b - FIM and sqrt of det of matrix 
data_idx_test = tuple(range(9))
data_fim = gw_fisher.mat(data_hp_grad, data_freq, data_idx_test)
data_fim_sqrt_det = gw_fisher.sqrtdet(data_hp_grad, data_freq, data_idx)

# %%
# Section 3.c - FIM plot
gw_plotter.fim_plot(data_fim)


#======================================================================#
#======================================================================#
#======================================================================#

# %%
# Section 4.a - FIM sqrtdet wrt mc eta
# t~20.5s - unoptimized
# Define mass repo config
data_mass_config = (1.0, 101.0, 10.0)
# Get base ripple theta
data_theta_ripple_repo = gw_ripple.theta_m1_m2_repo(data_mass_config)
# Get calc ripple theta
data_theta_ripple_calc_repo = [
    gw_ripple.theta_build(theta)
    for theta in data_theta_ripple_repo
]
# Get mc, eta from calc ripple theta
data_mc_repo = [
    data[0]
    for data in data_theta_ripple_calc_repo
]
data_mr_repo = [
   data[1]
    for data in data_theta_ripple_calc_repo
]
# Generate waveform repo for hp
data_waveform_repo = [
    gw_ripple.waveform(theta)
    for theta in data_theta_ripple_repo
]

# %%
# Generate grad waveform repo for hp and hc
# t ~188.4s for 100x waveform grad - not optimized
data_hp_grad_repo = [
    gw_ripple.grad_plus(theta)
    for theta in data_theta_ripple_repo
]
data_hc_grad_repo = [
    gw_ripple.grad_cros(theta)
    for theta in data_theta_ripple_repo
]

# %%
# Generate FIM sqrt det repo - hp
data_fim_sqrtdet_hp_repo = [
    gw_fisher.sqrtdet(hp_grad, data_freq, data_idx)
    for hp_grad in data_hp_grad_repo
]
# Generate FIM sqrt det repo - hc
data_fim_sqrtdet_hc_repo = [
    gw_fisher.sqrtdet(hc_grad, data_freq, data_idx)
    for hc_grad in data_hc_grad_repo
]
# %%
data_fim_hp_repo = jnp.array([item for item in data_fim_sqrtdet_hp_repo])
data_fim_hc_repo = jnp.array([item for item in data_fim_sqrtdet_hc_repo])
data_mc_repo = jnp.array([item for item in data_mc_repo])
data_mr_repo = jnp.array([item for item in data_mr_repo])

# %%
gw_plotter.fim_param_plot(data_fim_hp_repo, data_fim_hc_repo, data_mc_repo, data_mr_repo)
gw_plotter.fim_contour_plot(data_fim_hp_repo, data_fim_hc_repo, data_mc_repo, data_mr_repo)
# %%
