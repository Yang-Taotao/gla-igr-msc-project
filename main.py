"""
This is the probabilistic density handler module for MSc project.

Created on Thu Jun 14 2023

@author: Yang-Taotao
"""

# %% Section 0 - Library import
from data_waveform import data_ripple

# %% Section 1 - Waveform generator
# Define waveform generation parameters
theta_waveform_gen = (
    (36.0, 29.0),   # Mass: in units of solar masses
    (0, 0),         # Spin: no spin
    (0.0, 0.0),     # Time of coalescence in seconds, and Time of coalescence
    440,            # Distance to source in Mpc
    (0.0, 0.2),     # Inclination angle, and Polarization angle
)
# Define waveform plotter parameters
theta_waveform_plot = (
    24,         # 
    512,        # 
    0.01,       # 
)
# Combined waveform parameters
theta_waveform = theta_waveform_gen + theta_waveform_plot
# Call ripple waveform generator and plotter
data_ripple(theta_waveform)

# %% Section 2 - Fisher matrix calculator


# %% Section 3 - Probability density

