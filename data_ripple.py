"""
This is the ripple waveform handler script for MSc project.

Created on Thu Jul 11 2023

@author: Yang-Taotao
"""
# %%
# Library import
# XLA GPU resource setup
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# Package - jax
import jax
import jax.numpy as jnp
# Package - ripple
import ripple
from ripple.waveforms import IMRPhenomXAS

# %%
# Ripple - theta builder


def ripple_theta_builder(theta=(36.0, 29.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0)):
    # Local variable repo
    m1, m2, s1, s2, dist_mpc, c_time, c_phas, ang_inc, ang_pol = theta
    # Calculate mass - chirp, ratio
    mc, mr = ripple.ms_to_Mc_eta(jnp.array([m1, m2]))
    # Pack ripple theta tuple
    result = jnp.array(
        [mc, mr, s1, s2, dist_mpc, c_time, c_phas, ang_inc, ang_pol]
    )
    # Func return
    return result

# %%
# Ripple - freq builder


def ripple_freq_builder(theta=(24.0, 512.0, 0.5)):
    # Local variable repo
    f_min, f_max, f_del = theta
    # Calculate freq - signal, reference
    f_sig, f_ref = jnp.arange(f_min, f_max, f_del), f_min
    # Pack ripple freq tuple
    result = f_sig, f_ref
    # Func return
    return result

# %%
# Ripple - waveform generator


def ripple_waveform(theta):
    # Local variable repo
    f_sig, f_ref = ripple_freq_builder()
    theta_ripple = ripple_theta_builder(theta)
    # Generate strain waveform - plus, cross
    hp, hc = IMRPhenomXAS.gen_IMRPhenomXAS_polar(f_sig, theta_ripple, f_ref)
    # Pack ripple waveform tuple
    result = hp, hc
    # Func return
    return result


def ripple_waveform_plus(theta, freq):
    # Generate strain waveform - plus, cross
    _, f_ref = ripple_freq_builder()
    # Generate strain waveform - plus, cross
    hp, _ = IMRPhenomXAS.gen_IMRPhenomXAS_polar(jnp.array([freq]), theta, f_ref)
    # Pack ripple waveform tuple
    result = hp.real[0]
    # Func return
    return result


def ripple_waveform_cros(theta, freq):
    # Generate strain waveform - plus, cross
    _, f_ref = ripple_freq_builder()
    # Generate strain waveform - plus, cross
    _, hc = IMRPhenomXAS.gen_IMRPhenomXAS_polar(jnp.array([freq]), theta, f_ref)
    # Pack ripple waveform tuple
    result = hc.imag[0]
    # Func return
    return result

# %%
# Ripple - gradient calculator
def ripple_grad_vmap(func, theta):
    # Local variable repo
    f_sig, _ = ripple_freq_builder()
    # Map grad results
    result = jax.vmap(jax.grad(func), in_axes=(None, 0))(theta, f_sig)
    # Func return
    return result
