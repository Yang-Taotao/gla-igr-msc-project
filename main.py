"""
This is the master script for MSc project.

Created on Thu Jul 11 2023

@author: Yang-Taotao
"""
# %%
# Section 0 - Library import
from data import gw_ripple, gw_fisher, gw_plotter

# %%
# Section 1.a -  Define GW data theta
# m1, m2, s1, s2, dist_mpc, c_time, c_phas, ang_inc, ang_pol
data_theta = (36.0, 29.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0)
# f_min, f_max, f_del
data_freq = (24.0, 512.0, 0.5)

# %%
# Section 1.b -  Generate GW data
# t ~ 37.6s
data_hp, data_hc = gw_ripple.waveform(data_theta)
f_sig, _ = gw_ripple.freq_build(data_freq)

# %%
# Section 1.c -  Plot GW data - GW170817
data_theta_plot = data_hp, data_hc, f_sig
gw_plotter.ripple_waveform_plot(data_theta_plot)

# %%
# Section 2.a -  Generate mapped grad data
# t ~ 1m51.8s
data_hp_grad = gw_ripple.grad_vmap(gw_ripple.waveform_plus, data_theta)
data_hc_grad = gw_ripple.grad_vmap(gw_ripple.waveform_cros, data_theta)

# %%
# Section 2.b -  Plot GW data - grad wrt data_idx
data_idx = 0, 1
data_theta_grad_plot = data_hp_grad, data_hc_grad, f_sig
gw_plotter.ripple_grad_plot_idx(data_theta_grad_plot, *data_idx)

# %%
# Section 3.a - FIM psd plot
data_theta_bilby = data_freq, gw_fisher.bilby_psd(data_freq)
gw_plotter.bilby_plot(data_theta_bilby)

# %%
# Section 3.b - FIM and sqrt of det of matrix 
data_idx_test = tuple(range(9))
data_fim = gw_fisher.mat(data_hp_grad, data_idx_test)
data_fim_sqrt_det = gw_fisher.sqrtdet(data_hp_grad, data_idx)

# %%
# Section 3.c - FIM plot
gw_plotter.fim_plot(data_fim)
