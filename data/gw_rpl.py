"""
GW waveform and gradient calculator functions.
"""
# Library import
import os
# Package - jax
import jax
import jax.numpy as jnp
# Package - ripple
from ripple.waveforms import IMRPhenomXAS
# Custom config import
from data.gw_cfg import F_SIG, F_REF, PARAM_BASE, F_PSD, F_DIFF
# XLA GPU resource setup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
jax.config.update("jax_enable_x64", True)

# Ripple - Inner Product Handler


def inner_prod(vec_a: jnp.ndarray, vec_b: jnp.ndarray):
    """
    Noise weighted inner product between vectors a and b
    """
    # Get components
    numerator = jnp.abs(vec_a.conj() * vec_b)
    integrand = numerator / F_PSD
    # Return one side noise weighted inner products
    return 4 * F_DIFF * integrand.sum(axis=-1)


# Ripple - Get waveform_plus -> restricted and normalized


def waveform_plus_restricted(params: jnp.ndarray, freq: jnp.ndarray):
    '''
    Function to return restricted waveform_plus where params are:
    [Mc, eta, t_c, phi_c]
    '''
    # Set complete ripple_theta
    theta = PARAM_BASE.at[0:2].set(params[0:2]).at[5:7].set(params[2:4])
    # Generate plus polarized waveform
    h_plus, _ = IMRPhenomXAS.gen_IMRPhenomXAS_polar(freq, theta, F_REF)
    # Func return
    return h_plus


def waveform_plus_normed(params: jnp.ndarray, freq: jnp.ndarray):
    '''
    Produce waveform normalization for restricted waveoform_plus
    '''
    # Get restricted waveform
    waveform = waveform_plus_restricted(params, freq)
    # Calculate normalization factor
    norm_factor_squared = inner_prod(waveform, waveform)
    # Return normalized waveform
    return waveform / jnp.sqrt(norm_factor_squared)


# Ripple - Get waveform_cros -> restricted and normalized


def waveform_cros_restricted(params: jnp.ndarray, freq: jnp.ndarray):
    '''
    Function to return restricted waveform_cros where params are:
    [Mc, eta, t_c, phi_c]
    '''
    # Set complete ripple_theta
    theta = PARAM_BASE.at[0:2].set(params[0:2]).at[5:7].set(params[2:4])
    # Generate cross polarized waveform
    _, h_cros = IMRPhenomXAS.gen_IMRPhenomXAS_polar(freq, theta, F_REF)
    # Func return
    return h_cros


def waveform_cros_normed(params: jnp.ndarray, freq: jnp.ndarray):
    '''
    Produce waveform normalization for restricted waveoform_cros
    '''
    # Get restricted waveform
    waveform = waveform_cros_restricted(params, freq)
    # Calculate normalization factor
    norm_factor_squared = inner_prod(waveform, waveform)
    # Return normalized waveform
    return waveform / jnp.sqrt(norm_factor_squared)


# Ripple - Gradient Calculator


def gradient_plus(theta: jnp.ndarray):
    '''
    Map normalized waveform_plus gradients to signal frequency
    '''
    # Assemble params -- FutureWarning: dtype complex128 -> float64 imcompatible
    params = jnp.array(theta, dtype=jnp.complex128)
    # Return gradiant func mapped to signal frequency array
    return jax.vmap(
        jax.grad(waveform_plus_normed, holomorphic=True),
        in_axes=(None, 0),
    )(params, F_SIG)


def gradient_cros(theta: jnp.ndarray):
    '''
    Map normalized waveform_cros gradients to signal frequency
    '''
    # Assemble params
    # FutureWarning: dtype complex128 -> float64 imcompatible
    params = jnp.array(theta, dtype=jnp.complex128)
    # Return gradiant func mapped to signal frequency array
    return jax.vmap(
        jax.grad(waveform_cros_normed, holomorphic=True),
        in_axes=(None, 0),
    )(params, F_SIG)
