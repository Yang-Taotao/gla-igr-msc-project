"""
This is the plotter script for MSc project.

Created on Thu Jul 11 2023

@author: Yang-Taotao
"""
# %%
# Library import
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scienceplots
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


def ripple_theta_label(idx: int):
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
    # Return result string
    return label[idx]

# %%
# Ripple - waveform plotter


def ripple_waveform_plot(
        h_plus: jnp.ndarray, 
        h_cros: jnp.ndarray, 
        f_sig: jnp.ndarray,
    ):
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
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    # Plot admin
    fig.savefig("./figures/fig_01_ripple_waveform.png")

# %%
# Ripple - grad plotter


def ripple_grad_plot_idx(
        grad_hp: jnp.ndarray, 
        grad_hc: jnp.ndarray, 
        f_sig: jnp.ndarray, 
        idx1: int, 
        idx2: int,
    ):
    # Local variable repo
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
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    # Plot admin
    fig.savefig("./figures/fig_02_ripple_waveform_grad.png")

# %%
# Bilby - psd plotter


def bilby_plot(
        f_sig: jnp.ndarray, 
        data: jnp.ndarray,
    ):
    # Plot init
    fig, ax = plt.subplots()
    # Plotter
    ax.plot(f_sig, data, label="H1 PSD", alpha=0.5)
    # Plot customization
    ax.set(xlabel=f"{lbl_f}", ylabel=f"{lbl_h}", xscale='log')
    ax.legend()
    fig.tight_layout()
    # Plot admin
    fig.savefig("./figures/fig_03_bilby_psd.png")

# %%
# FIM - matrix plotter


def fim_plot(data: jnp.ndarray):
    # Plot init
    fig, ax = plt.subplots()
    # Plotter
    im = ax.imshow(data, cmap='plasma')
    # Plot customization
    ax.figure.colorbar(im, ax=ax)
    ax.set(xlabel="Columns", ylabel="Rows")
    # Plot admin
    fig.savefig("./figures/fig_04_fim.png")

# %%
# FIM - hp mc mr


def fim_param_plot(
        data_fim_hp_repo: jnp.ndarray, 
        data_fim_hc_repo: jnp.ndarray, 
        data_mc_repo: jnp.ndarray, 
        data_mr_repo: jnp.ndarray,
    ):
    # Sort data - get idx
    idx_hp_sort = jnp.argsort(data_fim_hp_repo)
    idx_hc_sort = jnp.argsort(data_fim_hc_repo)
    # Sort data
    mc_hp_repo, mr_hp_repo, fim_hp_repo = (
        data_mc_repo[idx_hp_sort],
        data_mr_repo[idx_hp_sort],
        data_fim_hp_repo[idx_hp_sort],
    )
    mc_hc_repo, mr_hc_repo, fim_hc_repo = (
        data_mc_repo[idx_hc_sort],
        data_mr_repo[idx_hc_sort],
        data_fim_hc_repo[idx_hc_sort],
    )
    # Plot init
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    # Plotter
    ax1.plot(mc_hp_repo, fim_hp_repo, label="Mass - chirp", alpha=0.5)
    ax2.plot(mr_hp_repo, fim_hp_repo, label="Mass - ratio", alpha=0.5)
    ax3.plot(mc_hc_repo, fim_hc_repo, label="Mass - chirp", alpha=0.5)
    ax4.plot(mr_hc_repo, fim_hc_repo, label="Mass - ratio", alpha=0.5)
    # Plot custmoization
    ax1.set(xlabel="Mass - chirp", ylabel="FIM - Sqrt of Det", title="fim_hp-mc")
    ax2.set(xlabel="Mass - ratio", ylabel="FIM - Sqrt of Det", title="fim_hp-mr")
    ax3.set(xlabel="Mass - chirp", ylabel="FIM - Sqrt of Det", title="fim_hc-mc")
    ax4.set(xlabel="Mass - ratio", ylabel="FIM - Sqrt of Det", title="fim_hc-mr")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    # Plot admin
    fig.tight_layout()
    fig.savefig("./figures/fig_05_fim_hp_mc_mr.png")

# %%
# FIM - hp mc mr contour


def fim_contour_plot(
        data_fim_hp_repo: jnp.ndarray, 
        data_fim_hc_repo: jnp.ndarray, 
        data_mc_repo: jnp.ndarray, 
        data_mr_repo: jnp.ndarray,
    ):
    # Reshape data
    mc_repo_rs, mr_repo_rs, fim_hp_repo_rs, fim_hc_repo_rs = (
        data_mc_repo.reshape((5, 5)),
        data_mr_repo.reshape((5, 5)),
        data_fim_hp_repo.reshape((5, 5)),
        data_fim_hc_repo.reshape((5, 5)),
    )
    # Plot init
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # Plotter
    cs1 = ax1.contourf(mc_repo_rs, mr_repo_rs, fim_hp_repo_rs)
    cs2 = ax2.contourf(mc_repo_rs, mr_repo_rs, fim_hc_repo_rs)
    # Plot customization
    ax1.set(xlabel="Mass - chirp", ylabel="Mass - ratio", title="FIM-sqrtdet grad hp")
    ax2.set(xlabel="Mass - chirp", ylabel="Mass - ratio", title="FIM-sqrtdet grad hc")
    plt.colorbar(cs1, ax=ax1)
    plt.colorbar(cs2, ax=ax2)
    # Plot admin
    fig.tight_layout()
    fig.savefig("./figures/fig_06_fim_mc_mr_contour.png")
