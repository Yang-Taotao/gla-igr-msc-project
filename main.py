"""
This is the master script for MSc project.

Created on Thu Jul 11 2023

@author: Yang-Taotao
"""
# %%
# Section 0 - Library import
from data_ripple import (
    ripple_waveform,
    ripple_freq_builder, 
    ripple_waveform_plus, 
    ripple_waveform_cros, 
    ripple_grad_vmap,
)
from data_plotter import (
    ripple_waveform_plot, 
    ripple_grad_plot_idx,
)
import jax.numpy as jnp

# %%
# Section 1.a -  Define GW data theta
# m1, m2, s1, s2, dist_mpc, c_time, c_phas, ang_inc, ang_pol
data_theta = (36.0, 29.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0)
# f_min, f_max, f_del
data_freq = (24.0, 512.0, 0.5)

# %%
# Section 1.b -  Generate GW data
# t ~ 37.6s
data_hp, data_hc = ripple_waveform(data_theta)
f_sig, _ = ripple_freq_builder(data_freq)
# Save array to local
jnp.save("./data/data_170817_plus.npy", data_hp)
jnp.save("./data/data_170817_cros.npy", data_hc)

# %%
# Section 1.c -  Plot GW data - GW170817
data_theta_plot = data_hp, data_hc, f_sig
ripple_waveform_plot(data_theta_plot)

# %%
# Section 2.a -  Generate mapped grad data
# t ~ 1m51.8s
data_hp_grad = ripple_grad_vmap(ripple_waveform_plus, data_theta)
data_hc_grad = ripple_grad_vmap(ripple_waveform_cros, data_theta)

# %%
# Section 2.b -  Plot GW data - grad wrt data_idx
data_idx = 0, 4
data_theta_grad_plot = data_hp_grad, data_hc_grad, f_sig
ripple_grad_plot_idx(data_theta_grad_plot, *data_idx)
