"""
Fisher Information Matrix calculator.

Created on Thu Jul 20 2023
@author: Yang-Taotao
"""
# %%
# Library import
import os
# Package - jax
import jax.numpy as jnp
# Custom config import
from data.gw_config import f_diff, f_psd
# XLA GPU resource setup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# %%
# FIM - inner prod


def inner_prod(data: jnp.ndarray, idx: tuple):
    # Local variable repo
    idx_i, idx_j = idx
    # Get grad array
    grad_i, grad_j = jnp.conj(data[:, idx_i]), data[:, idx_j]
    # Get grad product element
    grad_prod = grad_i * grad_j
    # Get inner product - raw
    inner_product = jnp.sum(grad_prod / f_psd)
    # Return inner product reult - real part
    return 4 * f_diff * jnp.real(inner_product)

# %%
# FIM - matrix handler


def build_fim(data: jnp.ndarray, idx: tuple):
    # Build local matrix
    n_idx = len(idx)
    # Return matrix entey parser
    return jnp.array([
        inner_prod(data, (idx[i], idx[j]))
        for i in range(n_idx)
        for j in range(n_idx)
    ]).reshape((n_idx, n_idx))


def sqrtdet_fim(data: jnp.ndarray, idx: tuple):
    # Import results
    matrix = build_fim(data, idx)
    # Return sqrt(det(FIM))
    return jnp.sqrt(jnp.linalg.det(matrix))
