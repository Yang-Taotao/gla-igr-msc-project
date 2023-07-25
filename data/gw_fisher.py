"""
This is the fisher information matrix handler script for MSc project.

Created on Thu Jul 20 2023

@author: Yang-Taotao
"""
# %%
# Library import
import os
# Package - jax
import jax.numpy as jnp
# Package - bilby
import bilby
# XLA GPU resource setup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# %%
# Notes
# consider just h+ for inner product for now
# construct FIM with mc and mr -> 2x2 mat for now

# %%
# FIM - freq assignment


def fim_freq(theta: tuple=(24.0, 512.0, 0.5)):
    # Local varibale repo
    f_min, f_max, f_del = theta
    # Get calcultions
    diff, sampling, duration = (
        f_max - f_min,
        (f_max - f_min) / f_del,
        1 / f_del,
    )
    # Return result tuple
    return diff, sampling, duration

# %%
# FIM - bilby psd


def bilby_psd(freq: tuple):
    # Local varibale repo
    _, sampling, duration = fim_freq(freq)
    # Get detector
    detector = bilby.gw.detector.get_empty_interferometer("H1")
    # Get sampling freq
    detector.sampling_frequency = sampling
    # Get dectector duration
    detector.duration = duration
    # Return psd as func result
    return detector.power_spectral_density_array[1:]

# %%
# FIM - inner prod


def inner_prod(data: jnp.ndarray, freq: tuple, theta: tuple):
    # Local variable repo
    idx_i, idx_j = theta
    diff, _, _ = fim_freq(freq)
    # Get grad array
    grad_i, grad_j = jnp.conj(data[:, idx_i]), data[:, idx_j]
    # Get grad product element
    grad_prod = grad_i * grad_j
    # Get psd array
    psd = bilby_psd(freq)
    # Get inner product - raw
    inner_prod = jnp.sum(grad_prod / psd)
    # Return inner product reult - real part
    return 4 * diff * jnp.real(inner_prod)

# %%
# FIM - matrix handler


def mat(data: jnp.ndarray, freq: tuple, theta: tuple):
    # Build local matrix
    n_idx = len(theta)
    # Return matrix entey parser
    return jnp.array([
        inner_prod(data, freq, (theta[i], theta[j]))
        for i in range(n_idx)
        for j in range(n_idx)
    ]).reshape((n_idx, n_idx))


def sqrtdet(data: jnp.ndarray, freq: tuple, theta: tuple):
    # Import results
    matrix = mat(data, freq, theta)
    # Return sqrt(det(FIM))
    return jnp.sqrt(jnp.linalg.det(matrix))
