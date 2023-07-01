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

# %% Section 1.a - Plotter customization
plt.style.use(['science', 'notebook', 'grid'])

# %% Section 2.a - Waveform generation function


def ripple_waveform(
    m_c,        # Mass - chirp
    eta,        # Mass - ratio
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
            eta,        # Mass - ratio
            s_1,        # Spin - 1
            s_2,        # Spin - 2
            dist,       # Distance - Mpc
            c_time,     # Coalescence - time
            c_phas,     # Coalescence - phase
            ang_inc,    # Angle - inclination
            ang_pol,    # Angle - ploarization
        ]
    )
    # Build frequency vector arg tuple
    theta_signal = (f_l, f_h, f_s)
    # Call frequency vector builder for ripple
    f_sig, f_ref = ripple_signal(theta_signal)
    # Strain signal generator
    h_plus, h_cros = IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        f_sig, theta_ripple, f_ref,
    )
    # Function return
    return h_plus, h_cros

# %% Section2.b - Waveform signal vector builder


def ripple_signal(arg):
    # Local arg repo
    f_l, f_h, f_s = arg
    # Signal vector builder
    f_sig, f_ref = (
        # Construct signal freq
        jnp.arange(f_l, f_h, f_s),
        # Set refrence signal freq
        f_l,
    )
    # Function return
    return f_sig, f_ref

# %% Section 2.c - Waveform plotter function


def ripple_plot(arg):
    # Get local result returns
    h_plus, h_cros = ripple_waveform(*arg)
    f_sig, _ = ripple_signal(arg[-3:])
    # Plot init
    plt.figure(figsize=(15, 5))
    # Plotting for alingned spin sys
    # No modulation for angular momentum precession
    plt.plot(
        f_sig,
        h_plus.real,
        label=r"$h_+$ ripple",
        alpha=0.5,
    )
    plt.plot(
        f_sig,
        h_cros.imag,
        label=r"$h_\times$ ripple",
        alpha=0.5,
    )
    # Plot customizer
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Signal Strain")
    plt.legend()
    plt.show()
    plt.savefig("./media/fig_01_ripple_waveform.png")
    plt.close()

    # Optional return
    return h_plus, h_cros

    # Secondary plot - Inverse FFT - preliminary
    # h_plus_irfft = jnp.fft.irfft(h_plus) #inverse real fft, need to start from zero freqs
    # plt.plot(jnp.arange(0, 1/f_sig[1], 1/(2*f_sig[-1])), jnp.roll(h_plus_irfft, -50))
    # plt.savefig("./media/tfft.png")
    # plt.close()

# %% Section 3 - Gradiant calculator