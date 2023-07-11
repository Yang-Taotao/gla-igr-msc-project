"""
This is the plotter script for MSc project.

Created on Thu Jul 11 2023

@author: Yang-Taotao
"""
# %%
# Library import
import matplotlib.pyplot as plt
import scienceplots
# Plotter style customization
plt.style.use(['science', 'notebook', 'grid'])

# %%
# Ripple - waveform plotter


def ripple_waveform_plot(theta):
    # Local variable repo
    hp, hc, f_sig = theta
    # Plot init
    fig, ax = plt.subplots()
    # Plotter
    ax.plot(f_sig, hp.real, label=r"$h_+$ ripple", alpha=0.5)
    ax.plot(f_sig, hc.imag, label=r"$h_\times$ ripple", alpha=0.5)
    ax.set(xlabel=r"Frequency (Hz)", ylabel=r"Signal Strain")
    ax.legend()
    # Plot admin
    fig.savefig("./media/fig_01_ripple_waveform.png")

# %%
