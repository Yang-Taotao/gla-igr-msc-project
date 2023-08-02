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
from data.gw_config import f_sig, f_ref, theta_base, f_psd, f_diff
# XLA GPU resource setup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# %%
# Ripple - waveform generator

def innerprod(a,b):
    """
    Noise weighted inner product between vectors a and b
    """
    numerator = jnp.real(a.conj()*b)
    #assert jnp.isnan(numerator).sum() == 0
    integrand = numerator / f_psd
    #assert jnp.isnan(integrand).sum() == 0
    result = 4 * f_diff * integrand.sum(axis=-1)
    #assert not jnp.isnan(result)
    return result

@jax.jit
def waveform(theta: jnp.ndarray):
    # Generate and return strain waveform - plus, cross
    return IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        f_sig, theta, f_ref)


def waveform_plus_mceta(mceta: jnp.ndarray, freq: jnp.ndarray):
    # Generate strain waveform - plus, cross
    theta = theta_base.at[0:2].set(mceta)
    h_plus, _ = IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        jnp.array([freq]), theta, f_ref)
    # Return ripple waveform tuple
    return h_plus

def waveform_plus_normed(mceta: jnp.ndarray, freq: jnp.ndarray):
    wf = waveform_plus_mceta(mceta, freq)
    N = innerprod(wf, wf)
    return wf[0]/jnp.complex128(jnp.sqrt(N))

def waveform_cros(theta: jnp.ndarray, freq: jnp.ndarray):
    # Generate strain waveform - plus, cross
    _, h_cros = IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        jnp.array([freq]), theta, f_ref)
    # Return ripple waveform tuple
    return h_cros.imag[0]

# %%
# Ripple - gradient calculator

@jax.jit
def gradient_plus_mceta(theta: jnp.ndarray):
    # Calculate gradient and return
    mceta = jnp.array(theta, dtype=jnp.complex128)
    return jax.vmap(jax.grad(waveform_plus_normed, holomorphic=True),in_axes=(None, 0))(mceta, f_sig)


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
