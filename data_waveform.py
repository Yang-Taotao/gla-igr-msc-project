"""
This is the GW waveform generator module for MSc project.

Created on Thu Jun 14 2023

@author: Yang-Taotao
"""

# %% Section 0 - Library import
# Module - jax related import
import jax
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

# Dummy func
# def waveform(chirp_mass, mass_ratio, a1, a2, inclination, psi, distance, tc, phi):
#     # Build the theta tuple for ripple
#     # Build the frequency vector for ripple
#     f_ref = f_min
#     hp, hc = ripple.IMRPhenon(f_sig, theta, f_ref)
#     return hp, hc
# jax.grad(waveform, 0) # derivative w.r.t Mc
# jax.grad(waveform, 1) # deriv w.r.t mass_ratio

def data_ripple(theta, config):
    # Invoke ms_to_Mc_eta method: chirp mass, eta
    mc, eta = ms_to_Mc_eta(jnp.array([theta[0], theta[1]]))
    # Intermidiate calculations - ripple
    theta_ripple = jnp.array(
        [
            mc,         # Mass - chirp
            eta,        # eta
            theta[2],   # Spin - 1
            theta[3],   # Spin - 2
            theta[4],   # Distance - Mpc
            theta[5],   # Coalescence - time
            theta[6],   # Coalescence - phase
            theta[7],   # Angle - inclination
            theta[8],   # Angle - ploarization
        ]
    )
    # Signal freq and ref freq
    f_sig, f_ref = (
        # Construct sig freq
        jnp.arange(config[0], config[1], config[2]),
        # Set refrence sig freq as min freq
        config[0],
    )
    # Strain signal generator
    h_plus, h_cros = IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        f_sig, 
        theta_ripple, 
        f_ref,
    )

    # INSERT JIT STUFF

    # Function result gather
    result = (
        f_sig,
        h_plus,
        h_cros,
    )
    # Function return
    return result, theta_ripple

# %% Section 2 - Ripple waveform plotter
def plot_ripple(theta):
    # Local variable repo
    f_sig, h_plus, h_cros = theta[0]
    # Plot init
    plt.figure(figsize=(15, 5))
    # Plotting for alingned spin sys
    # No modulation for angular momentum precession
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
    # h_plus_irfft = jnp.fft.irfft(h_plus) #inverse real fft, need to start from zero freqs
    # plt.plot(jnp.arange(0, 1/f_sig[1], 1/(2*f_sig[-1])), jnp.roll(h_plus_irfft, -50))
    # plt.savefig("./media/tfft.png")
    # plt.close()

# %% Section 3 - Derivative calculator
def data_grad(data_ripple, theta_ripple):
    # Grad func assignment
    grad_func = jax.grad(data_ripple)
    # Build results
    result = jax.vmap(grad_func)(theta_ripple)
    # Func return
    return result

# %% Section 4 - Simple function calls
def func_ripple(theta, config):
    # Call all other functions
    output_wave, output_theta = data_ripple(theta, config)
    output_grad = data_grad(data_ripple(theta, config), output_theta)
    plot_ripple(output_wave)
    # Results tally
    result = output_wave, output_grad
    # Func return
    return result
