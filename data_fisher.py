"""
This is the fisher information matrix handler script for MSc project.

Created on Thu Jul 20 2023

@author: Yang-Taotao
"""
# %%
# Library import
import os
from typing import Callable
# Package - jax
import jax
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
    # Build result tuple
    result = diff, sampling, duration
    # Func return
    return result


# %%
# FIM - bilby psd


def fim_bilby_psd(theta: tuple=(24.0, 512.0, 0.5)):
    # Local varibale repo
    _, sampling, duration = fim_freq(theta)
    # Get detector
    detector = bilby.gw.detector.get_empty_interferometer("H1")
    # Get sampling freq
    detector.sampling_frequency = sampling
    # Get dectector duration
    detector.duration = duration
    # Get psd as func result
    result = detector.power_spectral_density_array[1:]
    # Func return
    return result

# %%
# FIM - inner prod


def fim_inner_prod(data: jnp.ndarray, theta: tuple=(0,1)):
    # Local variable repo
    idx_i, idx_j = theta
    diff, _, _ = fim_freq()
    # Get grad array
    grad_i, grad_j = jnp.conj(data[:, idx_i]), data[:, idx_j]
    # Get grad product element
    grad_prod = grad_i * grad_j
    # Get psd array
    psd = fim_bilby_psd()
    # Get inner product - raw
    inner_prod = jnp.sum(grad_prod / psd)
    # Convert inner product to its real part
    result = 4 * diff * jnp.real(inner_prod)
    # Func return
    return result

# %%
# FIM - matrix entry


def fim_mat(data: jnp.ndarray, theta: tuple=(0,1)):
    # Build local matrix
    n_idx = len(theta)
    # Matrix entey parser
    matrix = jnp.array([
        fim_inner_prod(data, (theta[i], theta[j]))
        for i in range(n_idx)
        for j in range(n_idx)
    ]).reshape((n_idx, n_idx))
    # Matrix - square root of determinanat calculator
    result = jnp.sqrt(jnp.linalg.det(matrix))
    # Func return
    return result
