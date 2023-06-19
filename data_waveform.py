"""
This is the GW waveform generator module for MSc project.

Created on Thu Jun 14 2023

@author: Yang-Taotao
"""

# %% Section 0 - Library import
# JAX related import
import jax.numpy as jnp
from jax import grad, vmap
# RIPPLE related import
from ripple.waveforms import IMRPhenomXAS
from ripple import ms_to_Mc_eta
# Other import
from functools import partial
from math import pi
import matplotlib.pyplot as plt
import scienceplots

# %% Section 1 - Waveform param repo
theta = (
    (36.0, 29.0),   # Mass: in unit of solar masses
    (0, 0),         # Spin: no spin
    (0.0, 0.0),     # Time of coalescence in seconds, and Time of coalescence
    440,            # Distance to source in Mpc
    (0.0, 0.2),     # Inclination angle, and Polarization angle
)

# Invoke ms_to_Mc_eta
Mc, eta = ms_to_Mc_eta(jnp.array([theta[0][0], theta[0][1]]))

# %% Section 2 - Ripple
# Ripple
theta_ripple = jnp.array(
    [
        Mc,             
        eta,            
        theta[1][0],    
        theta[1][1],    
        theta[3],       
        theta[2][0],    
        theta[2][1],    
        theta[4][0], 
        theta[4][1],
    ]
)

# %% Section 3 - Additional param
# Freq grid generator
theta_grid = (
    24,         # Grid length
    512,        # Grid depth
    0.01,       # Grid step
)

# Signal freq and ref
f_sig, f_ref = (
    jnp.arange(theta_grid[0], theta_grid[1], theta_grid[2]),
    theta_grid[0],
)

# %% Section 4 - Waveform generator
hp_ripple, hc_ripple = IMRPhenomXAS.gen_IMRPhenomXAS_polar(f_sig, theta_ripple, f_ref)

# %% Section 5 - Waveform plotter
# Plot init
plt.figure(figsize=(15, 5))
# Plotting
plt.plot(f_sig, hp_ripple.real, label="h+ ripple", alpha=0.3)
plt.plot(f_sig, hc_ripple.imag, label="hx ripple", alpha=0.3)
# Plot customizer
plt.legend()
plt.xlabel("Freq")
plt.ylabel("Strain")
plt.savefig("./media/fig_01_ripple_waveform.png")
plt.close()
