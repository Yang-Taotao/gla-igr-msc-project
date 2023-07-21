"""
This is the ripple waveform handler script for MSc project.

Created on Thu Jul 11 2023

@author: Yang-Taotao
"""
# %%
# Library import
import os
from typing import Callable
# Package - jax
import jax
import jax.numpy as jnp
# Package - ripple
import ripple
from ripple.waveforms import IMRPhenomXAS
# XLA GPU resource setup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# %%
# Ripple - theta build


def theta_build(theta: tuple=(36.0, 29.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0)):
    # Local variable repo
    m_1, m_2, s_1, s_2, dist_mpc, c_time, c_phas, ang_inc, ang_pol = theta
    # Calculate mass - chirp, ratio
    m_c, m_r = ripple.ms_to_Mc_eta(jnp.array([m_1, m_2]))
    # Pack ripple theta tuple
    result = jnp.array(
        [m_c, m_r, s_1, s_2, dist_mpc, c_time, c_phas, ang_inc, ang_pol]
    )
    # Func return
    return result

# %%
# Ripple - freq build


def freq_build(theta: tuple=(24.0, 512.0, 0.5)):
    # Local variable repo
    f_min, f_max, f_del = theta
    # Calculate freq - signal, reference
    f_sig, f_ref = jnp.arange(f_min, f_max, f_del), f_min
    # Pack ripple freq tuple
    result = f_sig, f_ref
    # Func return
    return result

# %%
# Ripple - file path build


def file_path(theta: tuple, check: str):
    # Local variable repo
    path_data, path_grad = "./data_cache/data", "./data_cache/grad"
    path_hp, path_hc = "plus.npy", "cros.npy"
    # For waveform data
    if check == "data":
        data_header = path_data
    # For gradient data
    elif check == "grad":
        data_header = path_grad
    # Generate path string
    data_path = "_"+"_".join([str(i) for i in theta]) + "_"
    # Build hp. hc file_path
    path_file_hp, path_file_hc = (
        data_header + data_path + path_hp,
        data_header + data_path + path_hc,
    )
    # Pack file_data into results
    result = path_file_hp, path_file_hc
    # Func return
    return result

# %%
# Ripple - file manager


def file_check(theta: tuple):
    # Local varibale repo
    path_hp, path_hc = theta
    # Check both local files exist
    result = os.path.exists(path_hp) and os.path.exists(path_hc)
    # Func return
    return result


def file_load(theta: tuple):
    # Local variable repo
    path_hp, path_hc = theta
    # Load file
    h_plus, h_cros = jnp.load(path_hp), jnp.load(path_hc)
    # Pack result
    result = h_plus, h_cros
    # Func return
    return result


def file_save(theta: tuple):
    # Local variable repo
    path_hp, path_hc, data_hp, data_hc = theta
    # Save file
    jnp.save(path_hp, data_hp)
    jnp.save(path_hc, data_hc)
    # Report save status
    print("File: data_ripple saved to local.")

# %%
# Ripple - waveform generator


def waveform(theta: tuple):
    # Local variable repo
    f_sig, f_ref = freq_build()
    theta_ripple = theta_build(theta)
    # Get file path
    theta_path = file_path(theta, "data")
    # Pass in file data check
    check_file = file_check(theta_path)
    # Conditional npy array access
    if check_file is True:
        # Load hp, hc
        h_plus, h_cros = file_load(theta_path)
    # If no local stored file
    elif check_file is False:
        # Generate strain waveform - plus, cross
        h_plus, h_cros = IMRPhenomXAS.gen_IMRPhenomXAS_polar(
            f_sig, theta_ripple, f_ref)
        # Save array to local
        file_save((*theta_path, h_plus, h_cros))
    # Pack ripple waveform tuple
    result = h_plus, h_cros
    # Func return
    return result


def waveform_plus(theta: tuple, freq: jnp.ndarray):
    # Generate strain waveform - plus, cross
    _, f_ref = freq_build()
    # Generate strain waveform - plus, cross
    h_plus, _ = IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        jnp.array([freq]), theta, f_ref)
    # Pack ripple waveform tuple
    result = h_plus.real[0]
    # Func return
    return result


def waveform_cros(theta: tuple, freq: jnp.ndarray):
    # Generate strain waveform - plus, cross
    _, f_ref = freq_build()
    # Generate strain waveform - plus, cross
    _, h_cros = IMRPhenomXAS.gen_IMRPhenomXAS_polar(
        jnp.array([freq]), theta, f_ref)
    # Pack ripple waveform tuple
    result = h_cros.imag[0]
    # Func return
    return result

# %%
# Ripple - gradient calculator


def grad_vmap(func: Callable, theta: tuple):
    # Local variable repo
    f_sig, _ = freq_build()
    theta_ripple = theta_build(theta)
    # Get file path
    theta_path = file_path(theta, "grad")
    # Pass in file data check
    check_file = file_check(theta_path)
    # If local file exists
    if check_file is True:
        if func.__name__ == "waveform_plus":
            result, _ = file_load(theta_path)
        elif func.__name__ == "waveform_cros":
            _, result = file_load(theta_path)
    # If no local file
    elif check_file is False:
        # Map grad results
        result = jax.vmap(jax.grad(func), in_axes=(
            None, 0))(theta_ripple, f_sig)
        # Save to loacl
        if func.__name__ == "waveform_plus":
            jnp.save(theta_path[0], result)
        elif func.__name__ == "waveform_cros":
            jnp.save(theta_path[1], result)
        # Report save status
        print("File: grad_vmap saved to local.")
    # Func return
    return result
