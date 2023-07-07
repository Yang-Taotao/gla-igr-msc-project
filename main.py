"""
This is the probabilistic density handler module for MSc project.

Created on Thu Jun 14 2023

@author: Yang-Taotao
"""

# %%
# Section 0.a - Library import
# Jax related
import jax
import jax.numpy as jnp
# Ripple related
from ripple import ms_to_Mc_eta
from ripple.waveforms import IMRPhenomXAS
# Plotter related
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'notebook', 'grid'])

# %%
# Section 0.b - Jax GPU enable

# %%
# Section 1.a - Global varibles repo
# Variables - Primary
m1, m2 = 36.0, 29.0                                 # Mass - solar mass
f_min, f_max, f_del = 24.0, 512.0, 0.5              # GW waveform freq domain
# Variables - Primary calculated
mc, mr = ms_to_Mc_eta(jnp.array([m1, m2]))          # Mass - chirp, ratio
# Freq - signal, reference
f_sig, f_ref = jnp.arange(f_min, f_max, f_del), f_min
# Variables - Secondary
s1, s2 = 0.0, 0.0                                   # Spin
# Coalescence - time, phase
c_time, c_phas = 0.0, 0.0
# Angle - incline, polar
ang_inc, ang_pol = 0.0, 0.0
dist_mpc = 440.0                                    # Distance - Mpc
# Arguments - Built
arg_ripple = jnp.array(
    [mc, mr, s1, s2, dist_mpc, c_time, c_phas, ang_inc, ang_pol]
)

# %%
# Section 2.a - Ripple waveform generator
h_plus, h_cros = IMRPhenomXAS.gen_IMRPhenomXAS_polar(f_sig, arg_ripple, f_ref)


def h_real(theta, freq):
    return IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        jnp.array([freq]), theta, f_ref,
    )[0].real[0]


def h_imag(theta, freq):
    return IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        jnp.array([freq]), theta, f_ref,
    )[0].imag[0]


# %%
# Section 2.b - Ripple waveform plotter
# Plot init
fig, ax = plt.subplots()
# Plotter
ax.plot(f_sig, h_plus.real, label=r"$h_+$ ripple", alpha=0.5)
ax.plot(f_sig, h_cros.imag, label=r"$h_\times$ ripple", alpha=0.5)
ax.set(xlabel="Frequency (Hz)", ylabel="Signal Strain")
ax.legend()
# Plot admin
fig.savefig("./media/fig_01_ripple_waveform.png")

# %%
# Section 3.a - Derivatives calculator
# Get real and imag grad func
grad_real = jax.jit(jax.vmap(jax.grad(h_real), in_axes=(None, 0)))(arg_ripple, f_sig)
grad_imag = jax.jit(jax.vmap(jax.grad(h_imag), in_axes=(None, 0)))(arg_ripple, f_sig)
# Result recombine
grad_wave = grad_real + grad_imag * 1j

# %%
# Section 3.b - Grad plotter
# Plot init
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# Plotter
ax1.plot(f_sig, grad_wave.real, alpha=0.5)
ax2.plot(f_sig, grad_wave.imag, alpha=0.5)
ax3.plot(f_sig, grad_wave.real[:, 4], alpha=0.5)
ax4.plot(f_sig, grad_wave.imag[:, 4], alpha=0.5)
# Plot customization
ax1.set(xlabel="Frequency (Hz)", ylabel="Grad - real")
ax2.set(xlabel="Frequency (Hz)", ylabel="Grad - imag")
ax3.set(xlabel="Frequency (Hz)", ylabel="Grad dist - real")
ax4.set(xlabel="Frequency (Hz)", ylabel="Grad dist - imag")
# Plot admin
fig.tight_layout()
fig.savefig("./media/fig_02_ripple_waveform_grad.png")

# %%
# Section 3.c - Grad dist plotter
# Plot init
fig, (ax1, ax2) = plt.subplots(2, 1)
# Plotter
ax1.plot(f_sig, h_plus.real, alpha=0.5, label="Strain - real", color="r")
ax1.plot(f_sig, h_cros.imag, alpha=0.5, label="Strain - imag", color="b")
ax2.plot(f_sig, grad_wave.real[:, 4], alpha=0.5, label="Grad dist - real", color="y")
ax2.plot(f_sig, grad_wave.imag[:, 4], alpha=0.5, label="Grad dist - imag", color="g")
# Plot customization
ax1.set(xlabel="Frequency (Hz)", ylabel="GW Strain")
ax2.set(xlabel="Frequency (Hz)", ylabel="GW Grad - dist")
# Plot admin
fig.tight_layout()
fig.savefig("./media/fig_03_ripple_waveform_grad_dist.png")

# %%
# Section 4.a - Inner product with Power Spectral Density
