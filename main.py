"""
This is the probabilistic density handler module for MSc project.

Created on Thu Jun 14 2023

@author: Yang-Taotao
"""

# %% Section 0 - Library import
import jax.numpy as jnp
from ripple import ms_to_Mc_eta
from data_waveform import data_ripple, data_grad, plot_ripple

# %% Section 1 - Waveform generator
# Define stellar mass
m_1, m_2 = 36.0, 29.0
# Define waveform generation parameters
(
    m_c,        # Mass - chirp: in units of solar masses
    eta,        # eta
    s_1,        # Spin 1: no spin
    s_2,        # Spin 2: no spin
    dist,       # Distance to source in Mpc
    c_time,     # Time of coalescence in seconds,
    c_phas,     # Phase of coalescence
    ang_inc,    # Inclination angle
    ang_pol,    # Polarization angle
    f_l,        # Lower freq
    f_h,        # Upper freq
    f_s,        # Freq step -> delta_f = 1/total_t
) = (
    # Mass - chirp: in units of solar masses
    ms_to_Mc_eta(jnp.array([m_1, m_2]))[0],
    # eta
    ms_to_Mc_eta(jnp.array([m_1, m_2]))[1],
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
# Gather ripple waveform
result_ripple = data_ripple(
    m_c,        # Mass - chirp: in units of solar masses
    eta,        # eta
    s_1,        # Spin 1: no spin
    s_2,        # Spin 2: no spin
    dist,       # Distance to source in Mpc
    c_time,     # Time of coalescence in seconds,
    c_phas,     # Phase of coalescence
    ang_inc,    # Inclination angle
    ang_pol,    # Polarization angle
    f_l,        # Lower freq
    f_h,        # Upper freq
    f_s,        # Freq step -> delta_f = 1/total_t
)
# Generate ripple plot
plot_ripple(result_ripple)
# Calculate ripple grad
data_grad(result_ripple)

# %% Section 2 - Fisher matrix calculator


# %% Section 3 - Probability density

