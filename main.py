"""
This is the probabilistic density handler module for MSc project.

Created on Thu Jun 14 2023

@author: Yang-Taotao
"""

# %%
# Section 0 - Library import
# XLA GPU resource setup
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# Jax related
import jax
import jax.numpy as jnp
# Ripple related
from ripple import ms_to_Mc_eta
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
# Mass - chirp, ratio
mc, mr = ms_to_Mc_eta(jnp.array([m1, m2]))          
# Freq - signal, reference
f_sig, f_ref = jnp.arange(f_min, f_max, f_del), f_min
# Spin
s1, s2 = 0.0, 0.0                                   
# Coalescence - time, phase
c_time, c_phas = 0.0, 0.0
# Angle - incline, polar
ang_inc, ang_pol = 0.0, 0.0
# Distance - Mpc - GW170817
dist_mpc = 40.0
# Arguments - Built
arg_ripple = jnp.array(
    [mc, mr, s1, s2, dist_mpc, c_time, c_phas, ang_inc, ang_pol]
)

# %%
# Section 2.a - Ripple waveform func jit
@jax.jit
def waveform(arg):
    return IMRPhenomXAS.gen_IMRPhenomXAS_polar(f_sig, arg, f_ref)

# %%
# Section 2.b - Ripple waveform func for real and imag
def hp_real(theta, freq):
    return IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        jnp.array([freq]), theta, f_ref,
    )[0].real[0]


def hp_imag(theta, freq):
    return IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        jnp.array([freq]), theta, f_ref,
    )[0].imag[0]


# %%
# Section 2.c - Ripple waveform generator call
h_plus, h_cros = waveform(arg_ripple)

# %%
# Section 2.d - Ripple waveform plotter
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
# Get real and imag grad func for h_plus
grad_real = jax.jit(jax.vmap(jax.grad(hp_real), in_axes=(None, 0)))(arg_ripple, f_sig)
grad_imag = jax.jit(jax.vmap(jax.grad(hp_imag), in_axes=(None, 0)))(arg_ripple, f_sig)
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
ax1.plot(f_sig, h_plus.real, alpha=0.5, label="Strain - real")
ax1.plot(f_sig, h_cros.imag, alpha=0.5, label="Strain - imag")
ax2.plot(f_sig, grad_wave.real[:, 4], alpha=0.5, label="Grad dist - real")
ax2.plot(f_sig, grad_wave.imag[:, 4], alpha=0.5, label="Grad dist - imag")
# Plot customization
ax1.set(xlabel="Frequency (Hz)", ylabel="GW Strain")
ax2.set(xlabel="Frequency (Hz)", ylabel="GW Grad - dist")
# Plot admin
ax1.legend(); ax2.legend()
fig.tight_layout()
fig.savefig("./media/fig_03_ripple_waveform_grad_dist.png")

# %%
# Section 4.a - Fisher information matrix
# Stellar mass definer - min, max, step 
m_arg = (1.0, 100.0, 1.0)
m_array = jnp.arange(*m_arg)
# Assign m1, m2 param guess arrays
m1_array, m2_array = m_array, m_array

# %%
