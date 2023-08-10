"""
GW waveform and gradient calculator functions.
"""
# %%
# Library import
import os
# XLA GPU resource setup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# Package - jax
import jax
import jax.numpy as jnp
# Package - ripple
from ripple.waveforms import IMRPhenomXAS
# Custom config import
from data.gw_cfg import f_sig, f_ref, theta_base, f_psd, f_diff

# %%
# Ripple - Inner Product Handler


@jax.jit
def inner_prod(a, b):
    """
    Noise weighted inner product between vectors a and b
    """
    numerator = jnp.abs(a.conj()*b)
    # assert jnp.isnan(numerator).sum() == 0
    integrand = numerator / f_psd
    # assert jnp.isnan(integrand).sum() == 0
    result = 4 * f_diff * integrand.sum(axis=-1)
    # assert not jnp.isnan(result)
    return result


# %%
# Ripple - Get waveform_plus -> restricted and normalized


@jax.jit
def waveform_plus_restricted(params: jnp.ndarray, freq: jnp.ndarray):
    '''
    Function to return restricted waveform_plus where params are:
    [Mc, eta, t_c, phi_c]
    '''
    theta = theta_base.at[0:2].set(params[0:2]).at[5:7].set(params[2:4])
    h_plus, _ = IMRPhenomXAS.gen_IMRPhenomXAS_polar(freq, theta, f_ref)
    # Func return
    return h_plus


@jax.jit
def waveform_plus_normed(params: jnp.ndarray, freq: jnp.ndarray):
    '''
    Produce waveform normalization for restricted waveoform_plus
    '''
    wf = waveform_plus_restricted(params, freq)
    N = inner_prod(wf, wf)
    return wf / jnp.sqrt(N)


# %%
# Ripple - Get waveform_cros -> restricted and normalized


@jax.jit
def waveform_cros_restricted(params: jnp.ndarray, freq: jnp.ndarray):
    '''
    Function to return restricted waveform_cros where params are:
    [Mc, eta, t_c, phi_c]
    '''
    theta = theta_base.at[0:2].set(params[0:2]).at[5:7].set(params[2:4])
    _, h_cros = IMRPhenomXAS.gen_IMRPhenomXAS_polar(freq, theta, f_ref)
    # Func return
    return h_cros


@jax.jit
def waveform_cros_normed(params: jnp.ndarray, freq: jnp.ndarray):
    '''
    Produce waveform normalization for restricted waveoform_cros
    '''
    wf = waveform_cros_restricted(params, freq)
    N = inner_prod(wf, wf)
    return wf / jnp.sqrt(N)


# %%
# Ripple - Gradient Calculator


@jax.jit
def gradient_plus(theta: jnp.ndarray):
    '''
    Map normalized waveform_plus gradients to signal frequency
    '''
    # Calculate gradient and return
    params = jnp.array(theta, dtype=jnp.complex128)
    return jax.vmap(jax.grad(waveform_plus_normed, holomorphic=True),
                    in_axes=(None, 0))(params, f_sig)


@jax.jit
def gradient_cros(theta: jnp.ndarray):
    '''
    Map normalized waveform_cros gradients to signal frequency
    '''
    # Calculate gradient and return
    params = jnp.array(theta, dtype=jnp.complex128)
    return jax.vmap(jax.grad(waveform_cros_normed, holomorphic=True),
                    in_axes=(None, 0))(params, f_sig)
