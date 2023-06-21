"""
This is the probabilistic density handler module for MSc project.

Created on Thu Jun 14 2023

@author: Yang-Taotao
"""

# %% Section 0 - Library import
from data_waveform import func_ripple

# %% Section 1 - Waveform generator
# Define waveform generation parameters
theta_waveform = (
    36.0,   # Mass 1: in units of solar masses
    29.0,   # Mass 2: in units of solar masses
    0,      # Spin 1: no spin
    0,      # Spin 2: no spin
    440,    # Distance to source in Mpc
    0.0,    # Time of coalescence in seconds,
    0.0,    # Phase of coalescence
    0.0,    # Inclination angle
    0.2,    # Polarization angle
)
# Define waveform plotter parameters
theta_waveform_config = (
    24,     # Lower freq
    512,    # Upper freq
    0.5,    # Freq step -> delta_f = 1/total_t
)
# Call ripple waveform generator and plotter
result_ripple = func_ripple(theta_waveform, theta_waveform_config)

# %% Section 2 - Fisher matrix calculator


# %% Section 3 - Probability density

