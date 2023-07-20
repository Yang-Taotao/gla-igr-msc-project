"""
This is the fisher information matrix handler script for MSc project.

Created on Thu Jul 20 2023

@author: Yang-Taotao
"""
# %%
# Library import
import bilby

# %%
# Notes
# consider just h+ for inner product for now
# construct FIM with mc and mr -> 2x2 mat for now

# %%
# Bilby - psd parser


def bibly_psd(theta: tuple=(24.0, 512.0, 0.5)):
    # Local varibale repo
    f_min, f_max, f_del = theta
    # Get detector
    detector = bilby.gw.detector.get_empty_interferometer("H1")
    # Get sampling freq
    detector.sampling_frequency = (f_max - f_min) / f_del
    # Get dectector duration
    detector.duration = 1 / f_del
    # Get psd as func result
    result = detector.power_spectral_density_array
    # Func return
    return result

# %%