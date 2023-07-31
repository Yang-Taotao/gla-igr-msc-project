"""
Plotter function repo.

Created on Thu Jul 11 2023
@author: Yang-Taotao
"""
# %%
# Library import
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots
# Plotter style customization
plt.style.use(['science', 'notebook', 'grid'])

# %%
# Plotter resources
lbl_plus, lbl_cros, lbl_real, lbl_imag = (
    r"$h_+$ ", 
    r"$h_\times$ ",
    r"$\Re$ ",
    r"$\Im$ ",
)
lbl_freq, lbl_strain, lbl_noise, lbl_param, lbl_grad, lbl_fim, lbl_sqrtdet = (
    r"Frequency [Hz]",
    r"GW Strain ",
    r"GW Strain Noise [Hz^($-1/2$)]",
    r"$\vec{\Theta}$",
    r"$\vec{\Delta}$",
    r"$\mathcal{I}$",
    r"$\sqrt{\det{\mathcal{I}}}$",
)
# Plot customization resources
xscale, yscale, norm, lw, alpha, size, levels, cmap, cmap_r, cr, cb = (
    'log',
    'log',
    'log',
    2.5,
    0.8,
    36,
    300,
    'gist_heat',
    'gist_heat_r',
    '#B30C00',
    '#005398',
)
# Param label resources
label_repo = (
    r"$m_c$",
    r"$m_r$",
    r"$s_1$",
    r"$s_2$",
    r"$dist\_mpc$",
    r"$c_{\text{time}}$",
    r"$c_{\text{phas}}$",
    r"$\theta$",
    r"$\phi$",
)

# %%
# Ripple - waveform plotter


def ripple_waveform_plot(
    h_plus: jnp.ndarray,
    h_cros: jnp.ndarray,
    f_sig: jnp.ndarray,
):
    # Plot init
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    # Plotter
    ax1.plot(f_sig, h_plus.real, label=f"{lbl_real}{lbl_plus}{(lbl_param)}",
             alpha=alpha, lw=lw, color=cr)
    ax1.plot(f_sig, h_plus.imag, label=f"{lbl_imag}{lbl_plus}{(lbl_param)}",
             alpha=alpha, lw=lw, color=cb)
    ax2.plot(f_sig, h_cros.real, label=f"{lbl_real}{lbl_cros}{(lbl_param)}",
             alpha=alpha, lw=lw, color=cr)
    ax2.plot(f_sig, h_cros.imag, label=f"{lbl_imag}{lbl_cros}{(lbl_param)}",
             alpha=alpha, lw=lw, color=cb)
    # Plot customization
    ax1.set(xlabel=f"{lbl_freq}", ylabel=f"{lbl_strain}", xscale=xscale)
    ax2.set(xlabel=f"{lbl_freq}", ylabel=f"{lbl_strain}", xscale=xscale)
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
    label1, label2 = label_repo[idx1], label_repo[idx2]
    # Plot init
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    # Plotter
    ax1.plot(f_sig, grad_hp[:, idx1], label=f"{lbl_real}{lbl_plus}", 
             alpha=alpha, lw=lw, color=cr)
    ax1.plot(f_sig, grad_hc[:, idx1], label=f"{lbl_imag}{lbl_cros}", 
             alpha=alpha, lw=lw, color=cb)
    ax2.plot(f_sig, grad_hp[:, idx2], label=f"{lbl_real}{lbl_plus}", 
             alpha=alpha, lw=lw, color=cr)
    ax2.plot(f_sig, grad_hc[:, idx2], label=f"{lbl_imag}{lbl_cros}", 
             alpha=alpha, lw=lw, color=cb)
    # Plot customization
    ax1.set(xlabel=f"{lbl_freq}", ylabel=f"{lbl_grad}{label1}")
    ax2.set(xlabel=f"{lbl_freq}", ylabel=f"{lbl_grad}{label2}")
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    # Plot admin
    fig.savefig("./figures/fig_02_ripple_waveform_grad.png")

# %%
# Bilby - psd plotter


def bilby_plot(f_sig: jnp.ndarray, data: jnp.ndarray):
    # Get artificial data min
    data_min = 0.0
    # Plot init
    fig, ax = plt.subplots(figsize=(16, 12))
    # Plotter
    ax.plot(f_sig, data, label="H1 PSD", alpha=alpha, lw=lw, color=cr)
    ax.fill_between(f_sig, data, data_min, alpha=0.6*alpha, color=cb)
    # Plot customization
    ax.set(xlabel=f"{lbl_freq}", ylabel=f"{lbl_noise}", xscale=xscale, yscale=yscale)
    ax.legend()
    fig.tight_layout()
    # Plot admin
    fig.savefig("./figures/fig_03_bilby.png")

# %%
# FIM - matrix plotter


def fim_plot(data: jnp.ndarray):
    # Plot init
    fig, ax = plt.subplots(figsize=(16, 12))
    # Plotter
    im = ax.imshow(data, cmap=cmap_r, alpha=alpha, norm=norm)
    # Plot customization
    ax.figure.colorbar(im, ax=ax)
    ax.set(xlabel=r"Columns: $\Theta_i$", ylabel=r"Rows: $\Theta_j$")
    # Plot admin
    fig.savefig("./figures/fig_04_fim.png")

# %%
# FIM - hp mc mr


def fim_param_plot(
    fim_hp_repo: jnp.ndarray,
    fim_hc_repo: jnp.ndarray,
    mc_repo: jnp.ndarray,
    mr_repo: jnp.ndarray,
):
    # Grid - mc, mr
    mc_grid, mr_grid = jnp.meshgrid(mc_repo, mr_repo)
    # Format mc, mr data
    mc_data, mr_data = mc_grid.flatten(), mr_grid.flatten()
    # Plot init
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    # Plotter
    cs1 = ax1.scatter(mc_data, fim_hp_repo, alpha=alpha, s=size, c=fim_hp_repo, cmap=cmap)
    cs2 = ax2.scatter(mr_data, fim_hp_repo, alpha=alpha, s=size, c=fim_hp_repo, cmap=cmap)
    cs3 = ax3.scatter(mc_data, fim_hc_repo, alpha=alpha, s=size, c=fim_hc_repo, cmap=cmap)
    cs4 = ax4.scatter(mr_data, fim_hc_repo, alpha=alpha, s=size, c=fim_hc_repo, cmap=cmap)
    # Plot custmoization
    ax1.set(xlabel=f"{label_repo[0]}", ylabel=f"{lbl_sqrtdet}",
            title=f"Normalized {lbl_sqrtdet} for {lbl_plus} wrt. {label_repo[0]}")
    ax2.set(xlabel=f"{label_repo[1]}", ylabel=f"{lbl_sqrtdet}",
            title=f"Normalized {lbl_sqrtdet} for {lbl_plus} wrt. {label_repo[1]}")
    ax3.set(xlabel=f"{label_repo[0]}", ylabel=f"{lbl_sqrtdet}",
            title=f"Normalized {lbl_sqrtdet} for {lbl_cros} wrt. {label_repo[0]}")
    ax4.set(xlabel=f"{label_repo[1]}", ylabel=f"{lbl_sqrtdet}",
            title=f"Normalized {lbl_sqrtdet} for {lbl_cros} wrt. {label_repo[1]}")
    plt.colorbar(cs1, ax=ax1)
    plt.colorbar(cs2, ax=ax2)
    plt.colorbar(cs3, ax=ax3)
    plt.colorbar(cs4, ax=ax4)
    # Plot admin
    fig.tight_layout()
    fig.savefig("./figures/fig_05_fim_hp_mc_mr.png")

# %%
# FIM - hp mc mr contour


def fim_contour_plot(
    fim_hp_repo: jnp.ndarray,
    fim_hc_repo: jnp.ndarray,
    mc_repo: jnp.ndarray,
    mr_repo: jnp.ndarray,
):
    # Grid - mc, mr
    mc_grid, mr_grid = jnp.meshgrid(mc_repo, mr_repo)
    # Plot data process
    mat_hp = fim_hp_repo.reshape((mc_repo.shape[0], mr_repo.shape[0]))
    mat_hc = fim_hc_repo.reshape((mc_repo.shape[0], mr_repo.shape[0]))
    # Plot init
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # Plotter
    cs1 = ax1.contourf(mc_grid, mr_grid, mat_hp, alpha=alpha, levels=levels, cmap=cmap)
    cs2 = ax2.contourf(mc_grid, mr_grid, mat_hc, alpha=alpha, levels=levels, cmap=cmap)
    # Plot customization
    ax1.set(xlabel=f"{label_repo[0]}", ylabel=f"{label_repo[1]}",
            title=f"Normalized {lbl_sqrtdet} for {lbl_grad}{lbl_plus}", xscale=xscale)
    ax2.set(xlabel=f"{label_repo[0]}", ylabel=f"{label_repo[1]}",
            title=f"Normalized {lbl_sqrtdet} for {lbl_grad}{lbl_cros}", xscale=xscale)
    plt.colorbar(cs1, ax=ax1)
    plt.colorbar(cs2, ax=ax2)
    # Plot admin
    fig.tight_layout()
    fig.savefig("./figures/fig_06_fim_mc_mr_contour.png")
