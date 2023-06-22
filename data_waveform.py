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


def data_ripple(
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
):
    # Build the theta tuple for ripple
    theta_ripple = jnp.array(
        [
            m_c,        # Mass - chirp
            eta,        # eta
            s_1,        # Spin - 1
            s_2,        # Spin - 2
            dist,       # Distance - Mpc
            c_time,     # Coalescence - time
            c_phas,     # Coalescence - phase
            ang_inc,    # Angle - inclination
            ang_pol,    # Angle - ploarization
        ]
    )
    # Build the frequency vector for ripple
    f_sig, f_ref = (
        # Construct sig freq
        jnp.arange(f_l, f_h, f_s),
        # Set refrence sig freq as min freq
        f_l,
    )
    # Strain signal generator
    h_plus, h_cros = IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        f_sig,
        theta_ripple,
        f_ref,
    )
    # INSERT JIT STUFF - here
    # Function result gather
    result = (
        f_sig,
        h_plus,
        h_cros,
    )
    # Function return
    return result

# %% Section 2 - Ripple waveform plotter


def plot_ripple(arg):
    # Local variable repo
    f_sig, h_plus, h_cros = arg
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


def data_grad(waveform):
    # Grad func assignment
    grad_m_c = jax.grad(waveform, 0)
    grad_dist = jax.grad(waveform, 4)
    # Func return
    return grad_m_c, grad_dist

# %% Section 4 - Derivative plotter


def plot_grad(array):
    # Plot init
    plt.figure(figsize=(15, 5))
    # Plotter
    plt.plot(array.real, jnp.arange(1, len(array))+1)
    plt.savefig("./media/fig_02_waveform_grad.png")
    plt.close()
