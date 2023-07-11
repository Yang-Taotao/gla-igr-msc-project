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
# Ripple - theta label


def ripple_theta_label(idx=0):
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
    hp, hc, f_sig = theta
    # Plot init
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # Plotter
    ax1.plot(f_sig, hp.real, label=r"$h_+$ real", alpha=0.5)
    ax1.plot(f_sig, hp.imag, label=r"$h_+$ imag", alpha=0.5)
    ax2.plot(f_sig, hc.real, label=r"$h_\times$ real", alpha=0.5)
    ax2.plot(f_sig, hc.imag, label=r"$h_\times$ imag", alpha=0.5)
    # Plot customization
    ax1.set(xlabel=r"Frequency (Hz)", ylabel=r"GW Strain $h_+$")
    ax2.set(xlabel=r"Frequency (Hz)", ylabel=r"GW Strain $h_\times$")
    # Plot admin
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    fig.savefig("./figures/fig_01_ripple_waveform.png")

# %%
# Ripple - grad plotter


def ripple_grad_plot_idx(theta, idx1=0, idx2=1):
    # Local variable repo
    grad_hp, grad_hc, f_sig = theta
    label1 = ripple_theta_label(idx1)
    label2 = ripple_theta_label(idx2)
    # Plot init
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # Plotter
    ax1.plot(f_sig, grad_hp[idx1],
            alpha=0.5, label=r"$h_+$ real {}".format(label1))
    ax1.plot(f_sig, grad_hc[idx1],
            alpha=0.5, label=r"$h_\times$ imag {}".format(label1))
    ax2.plot(f_sig, grad_hp[idx2],
            alpha=0.5, label=r"$h_+$ real {}".format(label2))
    ax2.plot(f_sig, grad_hc[idx2],
            alpha=0.5, label=r"$h_\times$ imag {}".format(label2))
    # Plot customization
    ax1.set(xlabel=r"Frequency (Hz)", ylabel=r"$\partial h_+/ \partial$")
    ax2.set(xlabel=r"Frequency (Hz)", ylabel=r"$\partial h_\times/ \partial$")
    # Plot admin
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    fig.savefig("./figures/fig_02_ripple_waveform_grad_dist.png")
