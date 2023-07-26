"""
Configuration setup script.

Created on Thu Jul 26 2023
@author: Yang-Taotao
"""
# %%
# Library import
import jax
import jax.numpy as jnp
import bilby

# %%
# Config setup
# ============================================================ #
# Frequency - min, max, step
f_min, f_max, f_del = 24.0, 512.0, 0.5
# Chirp mass - min, max, step
mc_min, mc_max, mc_del = 5.0, 10.0, 0.5
# Mass ratio - min, max, step
mr_min, mr_max, mr_del = 0.10, 0.25, 0.01
# Base theta - s1, s2, dist_mpc, c_time, c_phas, ang_inc, and_pol
theta_base = jnp.array([0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0])
# ============================================================ #

# %%
# Frequency array builder


def freq_ripple(f_min: float, f_max: float, f_del: float):
    # Calculate and return freq - signal, reference
    return jnp.arange(f_min, f_max, f_del), f_min


def freq_fisher(f_min: float, f_max: float, f_del: float):
    # Calculate and return freq - difference, sampling, duration
    return (
        f_max - f_min,
        (f_max - f_min) / f_del,
        1 / f_del,
    )


def freq_psd(f_samp: float, f_dura: float):
    # Get detector
    detector = bilby.gw.detector.get_empty_interferometer("H1")
    # Get sampling freq
    detector.sampling_frequency = f_samp
    # Get dectector duration
    detector.duration = f_dura
    # Return psd as func result
    return detector.power_spectral_density_array[1:]

# %%
# Theta tuple builder


def theta_ripple(mc_repo: jnp.ndarray, mr_repo: jnp.ndarray, theta_base: jnp.ndarray):
    # Custom concatenater
    def theta_join(matrix):
        # Return joined matrix
        return jnp.concatenate((matrix, theta_base))
    # Build mc and mr grid 
    mc_grid, mr_grid = jnp.meshgrid(mc_repo, mr_repo)
    # Construct (mc, mr) matrix
    matrix = jnp.stack((mc_grid.flatten(), mr_grid.flatten()), axis=-1)
    # Return joined matrix
    return jax.vmap(theta_join)(matrix)

# %%
# Generate results
# Freq - signal, reference
f_sig, f_ref = freq_ripple(f_min, f_max, f_del)
# Freq - difference, sampling, duration
f_diff, f_samp, f_dura = freq_fisher(f_min, f_max, f_del)
# Freq - bilby PSD results
f_psd = freq_psd(f_samp, f_dura)
# Chirp mass repo
mc_repo = jnp.arange(mc_min, mc_max, mc_del)
# Mass ratio repo
mr_repo = jnp.arange(mr_min, mr_max, mr_del)
# Theta matrix result
theta_repo = theta_ripple(mc_repo, mr_repo, theta_base)
