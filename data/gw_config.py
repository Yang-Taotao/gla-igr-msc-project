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
f_min, f_max, f_del = 24.0, 512.0, 0.50
# Chirp mass - min, max, step
mc_min, mc_max, mc_del = 6.00, 31.0, 0.5
# Mass ratio - min, max, step
mr_min, mr_max, mr_del = 0.06, 0.26, 0.01
# Base theta - s1, s2, dist_mpc, c_time, c_phas, ang_inc, and_pol
theta_base = jnp.array([0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0])
# ============================================================ #

# %%
# Frequency array builder


def freq_ripple(data_min: float, data_max: float, data_del: float):
    # Calculate and return freq - signal, reference
    return jnp.arange(data_min, data_max, data_del), data_min


def freq_fisher(data_min: float, data_max: float, data_del: float):
    # Calculate and return freq - difference, sampling, duration
    return (
        data_max - data_min,
        (data_max - data_min) / data_del,
        1 / data_del,
    )


def freq_psd(data_samp: float, data_dura: float):
    # Get detector
    detector = bilby.gw.detector.get_empty_interferometer("H1")
    # Get sampling freq
    detector.sampling_frequency = data_samp
    # Get dectector duration
    detector.duration = data_dura
    # Return psd as func result
    return detector.power_spectral_density_array[1:]

# %%
# Theta tuple builder


def theta_ripple(
        data_mc_repo: jnp.ndarray,
        data_mr_repo: jnp.ndarray,
        data_theta_base: jnp.ndarray
    ):
    # Custom concatenater
    def theta_join(matrix):
        # Return joined matrix
        return jnp.concatenate((matrix, data_theta_base))
    # Build mc and mr grid
    mc_grid, mr_grid = jnp.meshgrid(data_mc_repo, data_mr_repo)
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
