"""
This is the plotter script for MSc project.

Created on Thu Jul 11 2023

@author: Yang-Taotao
"""
# %%
# Library import
import matplotlib.pyplot as plt
import scienceplots
from data_ripple import ripple_freq_build
# Plotter style customization
plt.style.use(['science', 'notebook', 'grid'])

# %%
# Ripple - plotter resources
lbl_p, lbl_c, lbl_d, lbl_r, lbl_i, lbl_f, lbl_h = (
    r"$h_+$ ",
    r"$h_\times$ ",
    r"$\partial$ ",
    r"$\Re$ ",
    r"$\Im$ ",
    r"Frequency ",
    r"GW Strain ",
)

# %%
# Ripple - theta label


def ripple_theta_label(idx: int=0):
    # Build label tuple
    label = (
        r"mc",
        r"mr",
        r"s1",
        r"s2",
        r"dist_mpc",
        r"c_time",
        r"c_phas",
        r"ang_inc",
        r"ang_pol",
    )
    # Build result string
    result = label[idx]
    # Func return
    return result

# %%
# Ripple - waveform plotter


def ripple_waveform_plot(theta):
    # Local variable repo
    h_plus, h_cros, f_sig = theta
    # Plot init
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # Plotter
    ax1.plot(f_sig, h_plus.real, label=f"{lbl_r}{lbl_p}", alpha=0.5)
    ax1.plot(f_sig, h_plus.imag, label=f"{lbl_i}{lbl_p}", alpha=0.5)
    ax2.plot(f_sig, h_cros.real, label=f"{lbl_r}{lbl_c}", alpha=0.5)
    ax2.plot(f_sig, h_cros.imag, label=f"{lbl_i}{lbl_c}", alpha=0.5)
    # Plot customization
    ax1.set(xlabel=f"{lbl_f}", ylabel=f"{lbl_h}{lbl_p}")
    ax2.set(xlabel=f"{lbl_f}", ylabel=f"{lbl_h}{lbl_c}")
    # Plot admin
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    fig.savefig("./figures/fig_01_ripple_waveform.png")

# %%
# Ripple - grad plotter


def ripple_grad_plot_idx(theta, idx1: int=0, idx2: int=1):
    # Local variable repo
    grad_hp, grad_hc, f_sig = theta
    label1 = ripple_theta_label(idx1)
    label2 = ripple_theta_label(idx2)
    # Plot init
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # Plotter
    ax1.plot(f_sig, grad_hp[:, idx1],
             alpha=0.5, label=f"{lbl_r}{lbl_p}")
    ax1.plot(f_sig, grad_hc[:, idx1],
             alpha=0.5, label=f"{lbl_i}{lbl_c}")
    ax2.plot(f_sig, grad_hp[:, idx2],
             alpha=0.5, label=f"{lbl_r}{lbl_p}")
    ax2.plot(f_sig, grad_hc[:, idx2],
             alpha=0.5, label=f"{lbl_i}{lbl_c}")
    # Plot customization
    ax1.set(xlabel=f"{lbl_f}", ylabel=f"{lbl_d}/{lbl_d}{label1}")
    ax2.set(xlabel=f"{lbl_f}", ylabel=f"{lbl_d}/{lbl_d}{label2}")
    # Plot admin
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    fig.savefig("./figures/fig_02_ripple_waveform_grad.png")

# %%
# Bilby - psd plotter


def bilby_plot(theta):
    # Local variable repo
    freq_base, strain = theta
    # Build freq
    freq, _ = ripple_freq_build(freq_base)
    # Plot init
    fig, ax = plt.subplots()
    # Plotter
    ax.plot(freq, strain, label="H1 PSD", alpha=0.5)
    # Plot customization
    ax.set(xlabel=f"{lbl_f}", ylabel=f"{lbl_h}", xscale='log')
    # Plot admin
    ax.legend()
    fig.tight_layout()
    fig.savefig("./figures/fig_03_bilby_psd.png")

# %%
# FIM - matrix plotter


def fim_plot(theta):
    # Plot init
    fig, ax = plt.subplots()
    # Plotter
    im = ax.imshow(theta, cmap='plasma')
    # Plot customization
    ax.figure.colorbar(im, ax=ax)
    ax.set(xlabel="Columns", ylabel="Rows")
    # Plot admin
    fig.savefig("./figures/fig_04_fim.png")
