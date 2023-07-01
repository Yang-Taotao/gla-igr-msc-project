"""
This is the probabilistic density handler module for MSc project.

Created on Thu Jun 14 2023

@author: Yang-Taotao
"""

# %% 
# Section 0 - Library import
import jax
import jax.numpy as jnp
from ripple import ms_to_Mc_eta
from data_waveform import ripple_plot #, ripple_waveform, ripple_signal

# %%
# Section 1.a - Ripple waveform arg val entry
# Define stellar mass in units of solar mass, using GW170817 val
m_1, m_2 = 36.0, 29.0
# Define waveform generation parameters as tuples
ripple_arg = (
    # Mass - chirp
    jnp.asarray(ms_to_Mc_eta(jnp.array([m_1, m_2])))[0],
    # Mass - ratio
    jnp.asarray(ms_to_Mc_eta(jnp.array([m_1, m_2])))[1],
    # Other params
    0,          # Spin 1: no spin
    0,          # Spin 2: no spin
    440,        # Distance to source in Mpc
    0.0,        # Time of coalescence in seconds,
    0.0,        # Phase of coalescence
    0.0,        # Inclination angle
    0.2,        # Polarization angle
    24,         # Lower freq
    512,        # Upper freq
    0.5,        # Freq step -> delta_f = 1/total_t
)
# Assign ripple waveform args to individual vars
(
    m_c,        # Mass - chirp
    eta,        # Mass - ratio
    s_1,        # Spin 1: no spin
    s_2,        # Spin 2: no spin
    dist,       # Distance to source in Mpc
    c_time,     # Time of coalescence in seconds,
    c_phas,     # Phase of coalescence
    ang_inc,    # Inclination angle
    ang_pol,    # Polarization angle
    f_l,        # Freq - lower
    f_h,        # Freq - upper
    f_s,        # Freq step -> delta_f = 1/total_t
) = ripple_arg

# %%
# Section 1.b - Get ripple waveform result and plot

# Get ripple results
# ripple_res = ripple_waveform(*ripple_arg)
# Build plotter arg
# ripple_plot_arg = (ripple_signal(ripple_arg[-3:])[0], ) + ripple_res
# Get tipple waveform plot
# ripple_plot(ripple_plot_arg)
ripple_res = ripple_plot(ripple_arg)

# %% 
# Section 1.c - Ripple derivative calculator

# ripple_grad_func = jax.vjp(ripple_waveform, *ripple_arg)

# simple_waveform = lambda m_c, eta, s_1: ripple_waveform(m_c, eta, s_1, *ripple_arg[3:])
# simple_grad = jax.vjp(simple_waveform, m_c, eta, s_1)

# %% Section 2.a - Fisher matrix calculator


# %% Section 3 - Probability density
