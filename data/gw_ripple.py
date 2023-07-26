"""
GW waveform and gradient calculator with @jax.jit added.

Created on Thu Jul 11 2023
@author: Yang-Taotao
"""
# %%
# Library import
import os
# Package - jax
import jax
import jax.numpy as jnp
# Package - ripple
from ripple.waveforms import IMRPhenomXAS
# Custom config import
from data import gw_config
# XLA GPU resource setup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# %%
# Config import
# Freq - signal, reference
f_sig, f_ref = gw_config.f_sig, gw_config.f_ref

# %%
# Ripple - waveform generator


@jax.jit
def waveform(theta: jnp.ndarray):
    # Generate and return strain waveform - plus, cross
    return IMRPhenomXAS.gen_IMRPhenomXAS_polar(
            f_sig, theta, f_ref)


def waveform_plus(theta: jnp.ndarray, freq: jnp.ndarray):
    # Generate strain waveform - plus, cross
    h_plus, _ = IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        jnp.array([freq]), theta, f_ref)
    # Return ripple waveform tuple
    return h_plus.real[0]


def waveform_cros(theta: jnp.ndarray, freq: jnp.ndarray):
    # Generate strain waveform - plus, cross
    _, h_cros = IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        jnp.array([freq]), theta, f_ref)
    # Return ripple waveform tuple
    return h_cros.imag[0]

# %%
# Ripple - gradient calculator


def gradient_plus(theta: jnp.ndarray):
    # Calculate gradient and return
    return jax.vmap(jax.grad(waveform_plus), in_axes=(None, 0))(theta, f_sig)


def gradient_cros(theta: jnp.ndarray):
    # Calculate gradient and return
    return jax.vmap(jax.grad(waveform_cros), in_axes=(None, 0))(theta, f_sig)

# %%
# Ripple - gradient calculator with matrix support


@jax.jit
def grad_plus(theta: jnp.ndarray):
    # Return matrix form of gradients
    return jax.vmap(gradient_plus)(theta)


@jax.jit
def grad_cros(theta: jnp.ndarray):
    # Return matrix form of gradients
    return jax.vmap(gradient_cros)(theta)
