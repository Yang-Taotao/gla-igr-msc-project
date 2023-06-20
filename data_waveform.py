"""
This is the GW waveform generator module for MSc project.

Created on Thu Jun 14 2023

@author: Yang-Taotao
"""

# %% Section 0 - Library import
# Module - jax related import
import jax.numpy as jnp
# Module - ripple related import
from ripple.waveforms import IMRPhenomXAS
from ripple import ms_to_Mc_eta
# Module - other import
import matplotlib.pyplot as plt
import scienceplots

# %% Section 0 - Plotter style customization
plt.style.use(["science", "notebook", "grid"])

# %% Section 1 - Waveform generation function
def data_ripple(theta):
    # Waveform generator
    # Invoke ms_to_Mc_eta method
    theta_mc, theta_eta = ms_to_Mc_eta(jnp.array([theta[0][0], theta[0][1]]))
    # Intermidiate calculations - ripple
    theta_ripple = jnp.array(
        [
            theta_mc,             
            theta_eta,            
            theta[1][0],    
            theta[1][1],    
            theta[3],       
            theta[2][0],    
            theta[2][1],    
            theta[4][0], 
            theta[4][1],
        ]
    )
    # Signal freq and ref freq
    freq_sig, freq_ref = (
        jnp.arange(theta[5], theta[6], theta[7]),
        theta[5],
    )
    # Strain signal generator
    strain_plus, strain_cros = IMRPhenomXAS.gen_IMRPhenomXAS_polar(freq_sig, theta_ripple, freq_ref)
    
    # Plotter
    # Plot init
    plt.figure(figsize=(15, 5))
    # Plotting
    plt.plot(freq_sig, strain_plus.real, label="h_+ ripple", alpha=0.3)
    plt.plot(freq_sig, strain_cros.imag, label="h_x ripple", alpha=0.3)
    # Plot customizer
    plt.legend()
    plt.xlabel("Frequency")
    plt.ylabel("Signal Strain")
    plt.savefig("./media/fig_01_ripple_waveform.png")
    plt.close()

    # Function result gather
    result = (
        freq_sig,
        freq_ref,
        strain_plus,
        strain_cros,
    )
    # Function return
    return result
