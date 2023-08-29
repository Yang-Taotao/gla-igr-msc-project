"""
Configuration setup script.
"""
# Library import
import jax
import jax.numpy as jnp
import bilby

# Config setup
# =========================================================================== #
# Frequency - min, max, step
F_MIN, F_MAX, F_DEL = 24.0, 512.0, 0.5
# Chirp mass - min, max, step
MC_MIN, MC_MAX, MC_NUM = 1.000, 21.00, 100
# Mass ratio - min, max, step
ETA_MIN, ETA_MAX, ETA_NUM = 0.050, 0.250, 100
# Base param - mc, eta, s1, s2, dl, tc, phic, theta, phi
PARAM_BASE = jnp.array([28.0956, 0.2471, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0])
# Test param for FIM compilation
MC, ETA = 28.0956, 0.2471
PARAM_TEST = jnp.array([MC, ETA, 0.0, 0.0])
# =========================================================================== #

# Frequency array builder


def freq_ripple(data_min: float, data_max: float, data_del: float):
    '''
    Build signal and reference frequency array
    '''
    return jnp.arange(data_min, data_max, data_del), data_min


def freq_fisher(data_min: float, data_max: float, data_del: float):
    '''
    Calculate frequency difference, sampling size, and duration
    '''
    return (
        data_max - data_min,
        (data_max - data_min) / data_del,
        1 / data_del,
    )


def freq_psd(data_samp: float, data_dura: float):
    '''
    Produce bilby based PSD noise array
    '''
    # Get detector
    detector = bilby.gw.detector.get_empty_interferometer("H1")
    # Get sampling freq
    detector.sampling_frequency = data_samp
    # Get dectector duration
    detector.duration = data_dura
    # Return psd as func result
    return detector.power_spectral_density_array[1:]


# Theta tuple builder


def theta_ripple(
    mc_repo: jnp.ndarray,
    mr_repo: jnp.ndarray,
    theta: jnp.ndarray,
):
    '''
    Create matrix of ripplegw theta arguments
    '''
    # Custom concatenater
    def theta_join(matrix):
        # Return joined matrix
        return jnp.concatenate((matrix, theta))
    # Build mc and mr grid
    mc_grid, mr_grid = jnp.meshgrid(mc_repo, mr_repo)
    # Construct (mc, mr) matrix
    matrix = jnp.stack((mc_grid.flatten(), mr_grid.flatten()), axis=-1)
    # Return joined matrix
    return jax.vmap(theta_join)(matrix)


# Generate results
# Freq - signal, reference
F_SIG, F_REF = freq_ripple(F_MIN, F_MAX, F_DEL)
# Freq - difference, sampling, duration
F_DIFF, F_SAMP, F_DURA = freq_fisher(F_MIN, F_MAX, F_DEL)
# Freq - bilby PSD results
F_PSD = freq_psd(F_SAMP, F_DURA)
# Chirp mass repo
MCS = jnp.linspace(MC_MIN, MC_MAX, MC_NUM)
# Mass ratio repo
ETAS = jnp.linspace(ETA_MIN, ETA_MAX, ETA_NUM)
# Theta matrix result
theta_repo = theta_ripple(MCS, ETAS, PARAM_BASE)
