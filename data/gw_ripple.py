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
# Ripple - theta repo


def theta_m1_m2_repo(theta: tuple, theta_resid: tuple=(0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0)):
    # Local variable repo
    m_min, m_max, m_del = theta
    # Mass repo config
    mass_config_1 = jnp.arange(m_min, m_max, m_del)
    mass_config_2 = jnp.arange(m_min, m_max, m_del)
    # Get mass repo
    m1_repo, m2_repo = mass_config_1, mass_config_2
    # Get results
    result = [
        (m1, m2) + theta_resid
        for m1 in m1_repo
        for m2 in m2_repo
    ]
    # Func return
    return result

# %%
# Ripple - theta build


def theta_build(theta: tuple):
    # Local variable repo
    m_1, m_2, s_1, s_2, dist_mpc, c_time, c_phas, ang_inc, ang_pol = theta
    # Calculate mass - chirp, ratio
    m_c, m_r = ripple.ms_to_Mc_eta(jnp.array([m_1, m_2]))
    # Pack ripple theta tuple
    result = jnp.array(
        [m_c, m_r, s_1, s_2, dist_mpc, c_time, c_phas, ang_inc, ang_pol]
    )
    # Func return
    return result

# %%
# Ripple - freq build


def freq_build(theta: tuple=(24.0, 512.0, 0.5)):
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
    # Pack ripple waveform tuple
    result = h_plus.real[0]
    # Func return
    return result


def waveform_cros(theta: tuple, freq: jnp.ndarray):
    # Generate strain waveform - plus, cross
    _, f_ref = freq_build()
    # Generate strain waveform - plus, cross
    _, h_cros = IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        jnp.array([freq]), theta, f_ref)
    # Pack ripple waveform tuple
    result = h_cros.imag[0]
    # Func return
    return result

# %%
# Ripple - gradient calculator


@jax.jit
def grad_plus(theta: tuple):
    # Local variable repo
    f_sig, _ = freq_build()
    theta_ripple = theta_build(theta)
    # Calculate gradient and return
    return jax.vmap(jax.grad(waveform_plus), in_axes=(
            None, 0))(theta_ripple, f_sig)


@jax.jit
def grad_cros(theta: tuple):
    # Local variable repo
    f_sig, _ = freq_build()
    theta_ripple = theta_build(theta)
    # Calculate gradient and return
    return jax.vmap(jax.grad(waveform_cros), in_axes=(
            None, 0))(theta_ripple, f_sig)
