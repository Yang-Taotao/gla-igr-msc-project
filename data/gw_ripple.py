"""
This is the ripple waveform handler script for MSc project.

Created on Thu Jul 11 2023

@author: Yang-Taotao
"""
# %%
# Library import
import os
from typing import Callable
# Package - jax
import jax
import jax.numpy as jnp
# Package - ripple
import ripple
from ripple.waveforms import IMRPhenomXAS
# XLA GPU resource setup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# %%
# Ripple - theta build


def theta_build(theta: tuple=(36.0, 29.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0)):
    # Local variable repo
    m_1, m_2, s_1, s_2, dist_mpc, c_time, c_phas, ang_inc, ang_pol = theta
    # Calculate mass - chirp, ratio
    m_c, m_r = ripple.ms_to_Mc_eta(jnp.array([m_1, m_2]))
    # Pack and return ripple theta tuple
    return jnp.array(
        [m_c, m_r, s_1, s_2, dist_mpc, c_time, c_phas, ang_inc, ang_pol]
    )

# %%
# Ripple - freq build


def freq_build(theta: tuple=(24.0, 512.0, 0.5)):
    # Local variable repo
    f_min, f_max, f_del = theta
    # Calculate freq - signal, reference
    f_sig, f_ref = jnp.arange(f_min, f_max, f_del), f_min
    # Return ripple freq tuple
    return f_sig, f_ref

# %%
# Ripple - waveform generator


@jax.jit
def waveform(theta: tuple):
    # Local variable repo
    f_sig, f_ref = freq_build()
    theta_ripple = theta_build(theta)
    # Generate and return strain waveform - plus, cross
    return IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        f_sig, theta_ripple, f_ref)


def waveform_plus(theta: tuple, freq: jnp.ndarray):
    # Generate strain waveform - plus, cross
    _, f_ref = freq_build()
    # Generate strain waveform - plus, cross
    h_plus, _ = IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        jnp.array([freq]), theta, f_ref)
    # Pack and return ripple waveform tuple
    return h_plus.real[0]


def waveform_cros(theta: tuple, freq: jnp.ndarray):
    # Generate strain waveform - plus, cross
    _, f_ref = freq_build()
    # Generate strain waveform - plus, cross
    _, h_cros = IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        jnp.array([freq]), theta, f_ref)
    # Pack and return ripple waveform tuple
    return h_cros.imag[0]

# %%
# Ripple - gradient calculator


@jax.jit
def grad_plus(theta: tuple):
    # Local variable repo
    f_sig, _ = freq_build()
    theta_ripple = theta_build(theta)
    # Return grad
    return jax.vmap(jax.grad(waveform_plus), in_axes=(
            None, 0))(theta_ripple, f_sig)


@jax.jit
def grad_cros(theta: tuple):
    # Local variable repo
    f_sig, _ = freq_build()
    theta_ripple = theta_build(theta)
    # Return grad
    return jax.vmap(jax.grad(waveform_cros), in_axes=(
            None, 0))(theta_ripple, f_sig)

