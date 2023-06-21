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
    f_sig, f_ref = (
        jnp.arange(theta[5], theta[6], theta[7]),
        theta[5],
    )
    # Strain signal generator
    h_plus, h_cros = IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        f_sig, 
        theta_ripple, 
        f_ref,
    )

    # INSERT JIT STUFF

    # Plotter
    # Plot init
    plt.figure(figsize=(15, 5))
    # Plotting for alingned spin sys
    # Modulation for angular momentum precession
    plt.plot(
        f_sig, 
        h_plus.real, 
        label=r"$h_+$ ripple", 
        alpha=0.3,
    )
    plt.plot(
        f_sig, 
        h_cros.imag,
        label=r"$h_x$ ripple", 
        alpha=0.3,
    )
    # Plot customizer
    plt.legend()
    plt.xlabel("Frequency")
    plt.ylabel("Signal Strain")
    plt.savefig("./media/fig_01_ripple_waveform.png")
    plt.close()

    # Secondary plot - Inverse FFT - preliminary
    h_plus_irfft = jnp.fft.irfft(h_plus) #inverse real fft, need to start from zero freqs
    plt.plot(jnp.arange(0, 1/f_sig[1], 1/(2*f_sig[-1])), jnp.roll(h_plus_irfft, -50))
    plt.savefig("./media/tfft.png")
    plt.close()

    # Function result gather
    result = (
        f_sig,
        f_ref,
        h_plus,
        h_cros,
    )
    # Function return
    return result
