"""
Fisher Information Matrix calculator functions.
"""
# Library import
import os
# Package - jax
import jax
import jax.numpy as jnp
# Other imports
from tqdm import trange
# Custom config import
from data import gw_rpl
# XLA GPU resource setup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
jax.config.update("jax_enable_x64", True)

# FIM - Parameter assembler


def fim_param_build(mcs: jnp.ndarray, etas: jnp.ndarray):
    """
    Build 4-D FIM_PARAM grid with mc and eta entries:
    [mc, eta, tc, phic]
    With input array shape (n, ) and (n, )
    Yield param array shape (n**2, 4)
    """
    # Set (1, ) shape zero value array
    zeros = jnp.zeros(1)
    # Param array - mc, eta, tc, phic
    param_arr = [mcs, etas, zeros, zeros]
    # Build 4-d mesh with matrix indexing
    nd_param = jnp.meshgrid(*param_arr, indexing='ij')
    # Stack and reshape into (n, 4) shape fim_param array
    fim_param = jnp.stack(nd_param, axis=-1).reshape(-1, len(param_arr))
    # Func return - (n**2, 4) param array
    return fim_param


def fim_param_stack(mcs: jnp.ndarray, etas: jnp.ndarray):
    """
    Build FIM_PARAM with column_stack method to get
    [mc, eta, tc, phic]
    With input array shape (n, ) and (n, )
    Yield param array shape (n, 4)
    """
    # Build tc, phic zeros entry array
    zeros = jnp.zeros_like(mcs)
    # Func return - stacked (n, 4) param array
    return jnp.column_stack((mcs, etas, zeros, zeros))


# FIM mapped


@jax.jit
def map_density_plus(param: jnp.ndarray):
    """
    Return the vectorized template density function for hp
    Param follows shape (n, 4)
    """
    return jax.vmap(log_sqrt_det_plus)(param)


@jax.jit
def map_density_cros(param: jnp.ndarray):
    """
    Return the vectorized template density function for hc
    Param follows shape (n, 4)
    """
    return jax.vmap(log_sqrt_det_cros)(param)


# @jax.jit
def log_density_plus(param: jnp.ndarray):
    """
    Return the vmap generated log density array for hp based param
    """
    # Local resources
    num_param = param.shape[0]
    batch_size = int(num_param * 0.1)
    num_batch = num_param // batch_size
    # Init
    density_list = []
    # Batching
    for i in range(num_batch):
        # Split batches
        batch_fim_param = param[i * batch_size: (i + 1) * batch_size]
        # Call jax.vmap
        batch_density = map_density_plus(batch_fim_param)
        # Add to results
        density_list.append(batch_density)
    # Concatenate the results from smaller batches
    density = jnp.concatenate(density_list)
    return density


# @jax.jit
def log_density_cros(param: jnp.ndarray):
    """
    Return the vmap generated log density array for hc based param
    """
    # Local resources
    num_param = param.shape[0]
    batch_size = int(num_param * 0.1)
    num_batch = num_param // batch_size
    # Init
    density_list = []
    # Batching
    for i in range(num_batch):
        # Split batches
        batch_fim_param = param[i * batch_size: (i + 1) * batch_size]
        # Call jax.vmap
        batch_density = map_density_cros(batch_fim_param)
        # Add to results
        density_list.append(batch_density)
    # Concatenate the results from smaller batches
    density = jnp.concatenate(density_list)
    return density


# FIM - Main ==> Batching


@jax.jit
def density_batch_calc(
    data: jnp.ndarray,
    mcs: jnp.ndarray,
    etas: jnp.ndarray,
    batch_size: int = 100,
    waveform: str = "hp",
):
    """
    Calculate metric density values with default batching size 100
    Default at waveform hp results
    Not actively used at the moment
    """
    # Select waveform
    if waveform == 'hp':
        wf_func = log_sqrt_det_plus
    elif waveform == 'hc':
        wf_func = log_sqrt_det_cros
    # Define batch numbers
    num_batch = data.shape[0] // batch_size
    density_list = []
    # Batching
    with trange(data.shape[0], desc="Processing Params") as param_range:
        for i in trange(num_batch):
            # Split batches
            batch_fim_param = data[i * batch_size: (i + 1) * batch_size]
            # Call jax.vmap
            batch_density = jax.vmap(wf_func)(batch_fim_param)
            # Add to results
            density_list.append(batch_density)
            # Update progress bar
            param_range.update(batch_fim_param.shape[0])
    # Concatenate the results from smaller batches
    density = jnp.concatenate(density_list).reshape([len(mcs), len(etas)])
    # Func return
    return density


# FIM - Main ==> log.sqrt.det.FIM


def log_sqrt_det_plus(param: jnp.ndarray):
    """
    Return the log based square root of the determinant of
    Fisher matrix projected onto the mc, eta space
    for hp waveform results
    """
    # Calculation
    #try:
    data_fim = projected_fim_plus(param)
    #except AssertionError:
    #    data_fim = jnp.nan
    # Func return - log density
    return jnp.log(jnp.sqrt(jnp.linalg.det(data_fim)))


def log_sqrt_det_cros(param: jnp.ndarray):
    """
    Return the log based square root of the determinant of
    Fisher matrix projected onto the mc, eta space
    for hc waveform results
    """
    # Calculation
    #try:
    data_fim = projected_fim_cros(param)
    #except AssertionError:
    #    data_fim = jnp.nan
    # Func return - log density
    return jnp.log(jnp.sqrt(jnp.linalg.det(data_fim)))


# FIM projection sub func


def fim_phic(full_fim: jnp.ndarray, nd_val: int):
    """
    Calculate the conditioned matrix projected onto coalecense phase
    """
    # Equation 16 from Dent & Veitch
    fim_result = jnp.array([
        full_fim[i, j] - full_fim[i, -1] * full_fim[-1, j] / full_fim[-1, -1]
        for i in range(nd_val-1)
        for j in range(nd_val-1)
    ]).reshape([nd_val-1, nd_val-1])
    # Func return
    return fim_result


def fim_tc(gamma: jnp.ndarray, nd_val: int):
    """
    Project the conditional matrix back onto coalecense time
    """
    # Equation 18 Dent & Veitch
    fim_result = jnp.array([
        gamma[i, j] - gamma[i, -1] * gamma[-1, j] / gamma[-1, -1]
        for i in range(nd_val-2)
        for j in range(nd_val-2)
    ]).reshape([nd_val-2, nd_val-2])
    # Func return
    return fim_result


# %%
# FIM - Projected and simple FIM


def projected_fim_plus(params: jnp.ndarray):
    """
    Return the Fisher matrix projected onto the mc, eta space
    for hp waveform results
    """
    # Get full FIM and dimensions
    full_fim = fim_plus(params)
    nd_val = params.shape[-1]
    # Calculate the conditioned matrix for phase
    gamma = fim_phic(full_fim, nd_val)
    # Calculate the conditioned matrix for time
    metric = fim_tc(gamma, nd_val)
    # Func return
    return metric


def projected_fim_cros(params: jnp.ndarray):
    """
    Return the Fisher matrix projected onto the mc, eta space
    for hc waveform results
    """
    # Get full FIM and dimensions
    full_fim = fim_cros(params)
    nd_val = params.shape[-1]
    # Calculate the conditioned matrix for phase
    gamma = fim_phic(full_fim, nd_val)
    # Calculate the conditioned matrix for time
    metric = fim_tc(gamma, nd_val)
    # Func return
    return metric


# FIM packers


def fim_plus(params: jnp.ndarray):
    """
    Returns the fisher information matrix
    at a general value of mc, eta, tc, phic
    for hp waveform

    Args:
        params (array): [Mc, eta, t_c, phi_c]. Shape 1x4
    """
    # Generate the waveform derivatives
    grads = gw_rpl.gradient_plus(params)
    # Get dimensions
    nd_val = grads.shape[-1]
    # Get FIM result
    fim_result = fim_base(grads, nd_val)
    # Func return
    return fim_result


def fim_cros(params: jnp.ndarray):
    """
    Returns the fisher information matrix
    at a general value of mc, eta, tc, phic
    for hc waveform

    Args:
        params (array): [Mc, eta, t_c, phi_c]. Shape 1x4
    """
    # Generate the waveform derivatives
    grads = gw_rpl.gradient_cros(params)
    # Get dimensions
    nd_val = grads.shape[-1]
    # Get FIM result
    fim_result = fim_base(grads, nd_val)
    # Func return
    return fim_result


def fim_base(grads: jnp.ndarray, nd_val: int):
    """
    Basic FIM entry packer
    """
    # Get FIM entries from inner products calculations
    entries = {
        (i, j): gw_rpl.inner_prod(grads[:, i], grads[:, j])
        for j in range(nd_val)
        for i in range(j+1)
    }
    # Fill the matrix from the precalculated entries
    fim_result = jnp.array([
        entries[tuple(sorted([i, j]))]
        for j in range(nd_val)
        for i in range(nd_val)
    ]).reshape([nd_val, nd_val])
    # Func return
    return fim_result
