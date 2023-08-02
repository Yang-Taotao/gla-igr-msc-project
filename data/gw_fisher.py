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
from data.gw_config import f_diff, f_psd, f_sig, theta_base
from data.gw_ripple import gradient_plus, innerprod, waveform_plus_normed
# XLA GPU resource setup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def projected_fim(params):
    """
    Return the Fisher matrix projected onto the mc,eta space
    """
    full_fim = fim(params)
    Nd = params.shape[-1]
    
    # Calculate the conditioned matrix for phase
    # Equation 16 from Dent & Veitch
    gamma = jnp.array([
        full_fim[i,j] - full_fim[i,-1]*full_fim[-1,j]/full_fim[-1,-1]
        for i in range(Nd-1) for j in range(Nd-1)
    ]).reshape([Nd-1,Nd-1])

    # Calculate the conditioned matrix for time
    # Equation 18 from Dent & Veitch
    G = jnp.array([
        gamma[i,j] - gamma[i,-1]*gamma[-1,j]/gamma[-1,-1]
        for i in range(Nd-2) for j in range(Nd-2)
    ]).reshape([Nd-2,Nd-2])
    
    return G


def fim(params):
    """Returns the fisher information matrix
    at a general value of mc, eta, tc, phic

    Args:
        params (array): [Mc, eta, t_c, phi_c]. Shape 1x4
    """
    # Generate the waveform derivatives
    assert params.shape[-1] == 4
    grads = gradient_plus(params)
    assert grads.shape[-2] == f_psd.shape[0]

    #print("Computed gradients, shape ",grads.shape)
    Nd = grads.shape[-1]
    # There should be no nans
    assert jnp.isnan(grads).sum()==0
    #if jnp.isnan(grads).sum()>0:
    #    print(f"NaN encountered in FIM calculation for ",mceta)
    
    # Compute their inner product
    # Calculate the independent matrix entries
    entries = {
            (i,j) : innerprod(grads[:,i],grads[:,j])
        for j in range(Nd) for i in range(j+1)}

    # Fill the matrix from the precalculated entries
    fim = jnp.array([entries[tuple(sorted([i,j]))]
                     for j in range(Nd) for i in range(Nd)]).reshape([Nd,Nd])

    return fim

