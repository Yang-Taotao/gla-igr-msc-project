"""
Plotter functions repository.
"""
# %%
# Library import
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scienceplots
# Plotter style customization
plt.style.use(['science', 'notebook', 'grid'])

# %%
# FIM - Contour


def fim_contour_mc_eta_log10(data_x: jnp.ndarray, data_y: jnp.ndarray, data_z: jnp.ndarray):
    """
    Generate contourf plots for log10 based FIM wrt mc, eta
    """
    # Plot init
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plotter
    cs = ax.contourf(
        data_x,
        data_y,
        data_z.T,
        alpha=0.8,
        levels=20,
        cmap='gist_heat',
    )
    # Plot customization
    ax.set(
        xlabel=r'Chirp Mass ($M_\odot$)',
        ylabel=r'Symmetric Mass Ratio $\eta$',
    )
    cb = plt.colorbar(cs, ax=ax)
    cb.ax.set_ylabel(r'$\log_{10}$ Template Bank Density')
    # Plot admin
    fig.savefig("./figures/fig_01_fim_contour_mc_eta_log10.png")
