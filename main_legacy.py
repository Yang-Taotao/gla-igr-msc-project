"""
This is the master script - legacy for MSc project.

Created on Thu Jun 14 2023

@author: Yang-Taotao
"""
# %%
# Section 0 - Library import
# XLA GPU resource setup
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# Package - jax
import jax
import jax.numpy as jnp
# Package - ripple
import ripple
from ripple.waveforms import IMRPhenomXAS
# Plotter related
import matplotlib.pyplot as plt
import scienceplots
# Plotter style customization
plt.style.use(['science', 'notebook', 'grid'])

# %%
# Section 1.a - Global varibles repo
# Mass - solar mass - GW170817
m1, m2 = 36.0, 29.0
# GW waveform freq domain
f_min, f_max, f_del = 24.0, 512.0, 0.5
# Spin
s1, s2 = 0.0, 0.0
# Coalescence - time, phase
c_time, c_phas = 0.0, 0.0
# Angle - incline, polar
ang_inc, ang_pol = 0.0, 0.0
# Distance - Mpc - GW170817
dist_mpc = 40.0

# %%
# Section 1.b - Build theta_ripple
# Mass - chirp, ratio
mc, mr = ripple.ms_to_Mc_eta(jnp.array([m1, m2]))
# Freq - signal, reference
f_sig, f_ref = jnp.arange(f_min, f_max, f_del), f_min
# Arguments - Built
theta_ripple = jnp.array(
    [mc, mr, s1, s2, dist_mpc, c_time, c_phas, ang_inc, ang_pol]
)

# %%
# Section 2.a - Ripple waveform func


def waveform(theta):
    return IMRPhenomXAS.gen_IMRPhenomXAS_polar(f_sig, theta, f_ref)


def hp_real(theta, freq):
    return IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        jnp.array([freq]), theta, f_ref,
    )[0].real[0]


def hp_imag(theta, freq):
    return IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        jnp.array([freq]), theta, f_ref,
    )[0].imag[0]


def hc_real(theta, freq):
    return IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        jnp.array([freq]), theta, f_ref,
    )[1].real[0]


def hc_imag(theta, freq):
    return IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        jnp.array([freq]), theta, f_ref,
    )[1].imag[0]


# %%
# Section 2.b - Ripple waveform generator call
h_plus, h_cros = waveform(theta_ripple)

# %%
# Section 2.c - Ripple waveform plotter
# Plot init
fig, ax = plt.subplots()
# Plotter
ax.plot(f_sig, h_plus.real, label=r"$h_+$ ripple", alpha=0.5)
ax.plot(f_sig, h_cros.imag, label=r"$h_\times$ ripple", alpha=0.5)
ax.set(xlabel=r"Frequency (Hz)", ylabel=r"Signal Strain")
ax.legend()
# Plot admin
fig.savefig("./media/fig_01_ripple_waveform.png")

# %%
# Section 3.a - Derivatives calculator
# Get real and imag grad func for h_plus
grad_hp_real = jax.jit(jax.vmap(jax.grad(hp_real), in_axes=(None, 0)))(
    theta_ripple, f_sig)
grad_hp_imag = jax.jit(jax.vmap(jax.grad(hp_imag), in_axes=(None, 0)))(
    theta_ripple, f_sig)
grad_hc_real = jax.jit(jax.vmap(jax.grad(hc_real), in_axes=(None, 0)))(
    theta_ripple, f_sig)
grad_hc_imag = jax.jit(jax.vmap(jax.grad(hc_imag), in_axes=(None, 0)))(
    theta_ripple, f_sig)
# Result recombine
grad_hp_wave = grad_hp_real + grad_hp_imag * 1j
grad_hc_wave = grad_hc_real + grad_hc_imag * 1j

# %%
# Section 3.b - Grad plotter
# Plot init
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# Plotter
ax1.plot(f_sig, grad_hp_wave.real, alpha=0.5)
ax2.plot(f_sig, grad_hp_wave.imag, alpha=0.5)
ax3.plot(f_sig, grad_hc_wave.real, alpha=0.5)
ax4.plot(f_sig, grad_hc_wave.imag, alpha=0.5)
# Plot customization
ax1.set(xlabel=r"Frequency (Hz)", ylabel=r"Grad $h_+$ - real")
ax2.set(xlabel=r"Frequency (Hz)", ylabel=r"Grad $h_+$ - imag")
ax3.set(xlabel=r"Frequency (Hz)", ylabel=r"Grad $h_\times$ - real")
ax4.set(xlabel=r"Frequency (Hz)", ylabel=r"Grad $h_\times$ - imag")
# Plot admin
fig.tight_layout()
fig.savefig("./media/fig_02_ripple_waveform_grad.png")

# %%
# Section 3.c - Grad dist plotter
# Plot init
fig, (ax1, ax2) = plt.subplots(2, 1)
# Plotter
ax1.plot(f_sig, grad_hp_wave.real[:, 4],
         alpha=0.5, label=r"Grad $h_+$ dist - real")
ax1.plot(f_sig, grad_hp_wave.imag[:, 4],
         alpha=0.5, label=r"Grad $h_+$ dist - imag")
ax2.plot(f_sig, grad_hc_wave.real[:, 4],
         alpha=0.5, label=r"Grad $h_\times$ dist - real")
ax2.plot(f_sig, grad_hc_wave.imag[:, 4],
         alpha=0.5, label=r"Grad $h_\times$ dist - imag")
# Plot customization
ax1.set(xlabel=r"Frequency (Hz)", ylabel=r"Grad $h_+$ - dist")
ax2.set(xlabel=r"Frequency (Hz)", ylabel=r"Grad $h_\times$ - dist")
# Plot admin
ax1.legend()
ax2.legend()
fig.tight_layout()
fig.savefig("./media/fig_03_ripple_waveform_grad_dist.png")
