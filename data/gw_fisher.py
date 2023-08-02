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
from data.gw_config import f_diff, f_psd, f_sig
from data.gw_ripple import gradient_plus_mceta, innerprod
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
    # Get matrix entey parser
    matrix = jnp.array([
        inner_prod(data, (idx[i], idx[j]))
        for i in range(n_idx)
        for j in range(n_idx)
    ]).reshape((n_idx, n_idx))
    # Normalize matrix entries
    fim_min, fim_max = jnp.min(matrix), jnp.max(matrix)
    # Return normalized result
    return (matrix-fim_min)/(fim_max-fim_min)


def sqrtdet_fim(data: jnp.ndarray, idx: tuple):
    # Import results
    matrix = build_fim(data, idx)
    # Return sqrt(det(FIM))
    return jnp.sqrt(jnp.linalg.det(matrix))

def fim(mc_eta):
    """Returns the fisher information matrix
    at a general value of mc, eta

    Args:
        mc_eta (array): chirp mass and eta array. Shape 1x2
    """
    # Generate the waveform derivatives
    assert mc_eta.shape[-1] == 2
    grads = gradient_plus_mceta(mc_eta)
    assert grads.shape[-2] == f_psd.shape[0]

    print("Computed gradients, shape ",grads.shape)
    Nd = grads.shape[-1]
    # There should be no nans
    assert jnp.isnan(grads).sum()==0
    # Compute their inner product

    
    fim = jnp.array([innerprod(grads[:,i],grads[:,j])
                     for j in range(Nd) for i in range(Nd)]).reshape([Nd,Nd])
    # Return FIM
    return fim

    