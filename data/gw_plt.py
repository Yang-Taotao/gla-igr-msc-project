"""
Plotter functions repository.
"""
# Library import
import io
import jax.numpy as jnp
import matplotlib.pyplot as plt
import corner
from PIL import Image
import scienceplots
# Plotter style customization
plt.style.use(['science', 'notebook', 'grid'])

# FIM plots


def log_fim_contour(
    data_x: jnp.ndarray,
    data_y: jnp.ndarray,
    data_z: jnp.ndarray,
    waveform: str = 'hp',
):
    """
    Generate contourf plots for log density wrt mc, eta
    Defaulted at waveform hp results
    """
    # Local plotter resources
    xlabel, ylabel, cblabel = (
        r'Chirp Mass $\mathcal{M} [M_\odot$]',
        r'Symmetric Mass Ratio $\eta$',
        r'$\log$ Template Bank Density',
    )
    save_path = f'./figures/log_fim_contour_{waveform}.png'
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
    ax.set(xlabel=xlabel, ylabel=ylabel)
    cb = plt.colorbar(cs, ax=ax)
    cb.ax.set_ylabel(cblabel)
    # Plot admin
    fig.savefig(save_path)


def log_fim_param(
    data_x: jnp.ndarray,
    data_y: jnp.ndarray,
    waveform: str = "hp",
    param: str = "mc",
):
    """
    1-D param plot for log density wrt param entry
    """
    # Local plotter resources
    xlabel_dict = {
        'mc': r'Chirp Mass $\mathcal{M} [M_\odot$]',
        'eta': r'Symmetric Mass Ratio $\eta$',
    }
    ylabel = r'$\log$ Template Bank Density'
    save_path = f'./figures/log_fim_param_{param}_{waveform}.png'
    # Plot init
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plotter
    plot = [
        ax.scatter(
            data_x,
            data_y[:, i],
            alpha=0.8,
            s=1,
            cmap='gist_heat',
            c=data_y[:, i],
        )
        for i in range(int(data_y.shape[0]))
    ]
    # Plot customization
    ax.set(xlabel=xlabel_dict[param], ylabel=ylabel)
    fig.tight_layout()
    # Plot admin
    fig.savefig(save_path)

# Waveform and gradient


def ripple_waveform(
    data_x: jnp.ndarray,
    data_y: jnp.ndarray,
    waveform: str = 'hp',
):
    """
    Generate plots for ripple generated waveforms
    Defaulted at hp waveform
    """
    # Label plotter resources
    if waveform == "hp":
        label1, label2, xlabel, ylabel = (
            r'$\Re h_+$',
            r'$\Im h_+$',
            r'Freqency $f$ [Hz]',
            r'Signal Strain $h_+$',
        )
    elif waveform == "hc":
        label1, label2, xlabel, ylabel = (
            r'$\Re h_\times$',
            r'$\Im h_\times$',
            r'Freqency $f$ [Hz]',
            r'Signal Strain $h_\times$',
        )
    save_path = f'./figures/ripple_{waveform}.png'
    # Plot init
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plotter
    ax.plot(data_x, data_y.real, label=label1)
    ax.plot(data_x, data_y.imag, label=label2)
    # Plot customization
    ax.set(xlabel=xlabel, ylabel=ylabel, xscale="log")
    ax.legend()
    fig.tight_layout()
    # Plot admin
    fig.savefig(save_path)


def ripple_gradient(
    data_x: jnp.ndarray,
    data_y1: jnp.ndarray,
    data_y2: jnp.ndarray,
    param: str = 'mc',
):
    """
    Generate plots for gradients of ripple generated waveforms
    Defaulted at gradient wrt mc param
    """
    # Local label dict
    label_dict = {
        'real' : r'$\Re$',
        'imag' : r'$\Im$',
        'hp' : r'$h_+$',
        'hc' : r'$h_\times$',
        'freq' : r'Frequency $f$ [Hz]',
        'grad' : r'Gradient wrt. ',
        'mc': r'Chirp Mass $\tilde{h}_{\mathcal{M}}$',
        'eta': r'Symmetric Mass Ratio $\tilde{h}_{\eta}$',
        's1': r'Spin of $m_1$ $\tilde{h}_{s_1}$',
        's2': r'Spin of $m_2$ $\tilde{h}_{s_2}$',
        'dl': r'Distance $\tilde{h}_{d_L}$',
        'tc': r'Coalescence Time $\tilde{h}_{t_c}$',
        'phic': r'Coalescence Phase $\tilde{h}_{\phi_c}$',
        'theta': r'Inclination Angle $\tilde{h}_{\theta}$',
        'phi': r'Polarization Angle $\tilde{h}_{\phi}$',
    }
    # Local plotter resources
    label1, label2, xlabel, ylabel = (
        f"{label_dict['real']}{label_dict['hp']}",
        f"{label_dict['imag']}{label_dict['hc']}",
        f"{label_dict['freq']}",
        f"{label_dict['grad']}{label_dict[param]}",
    )
    save_path = f'./figures/ripple_{param}.png'
    # Plot init
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plotter
    ax.plot(data_x, data_y1.real, label=label1)
    ax.plot(data_x, data_y2.imag, label=label2)
    # Plot customization
    ax.set(xlabel=xlabel, ylabel=ylabel, xscale="log")
    ax.legend()
    fig.tight_layout()
    # Plot admin
    fig.savefig(save_path)


# PSD from bilby


def bilby_noise_psd(data_x: jnp.ndarray, data_y: jnp.ndarray):
    """
    Plot the PSD obtained from bilby
    """
    # Local plotter resources
    data_min, color_red, color_blue = (
        0.0,
        '#B30C00',
        '#005398',
    )
    label, xlabel, ylabel,  = (
        r'H1 Power Spectral Density $S(f)$',
        r'Frequency $f$ [Hz]',
        r'GW Strain Noise [Hz^($-1/2$)]',
    )
    save_path = './figures/bilby_psd.png'
    # Plot init
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plotter
    ax.plot(data_x, data_y, label=label, alpha=0.8, lw=2, color=color_red)
    ax.fill_between(data_x, data_y, data_min, alpha=0.6, color=color_blue)
    # Plot customization
    ax.set(xlabel=xlabel, ylabel=ylabel, xscale='log', yscale='log')
    ax.legend()
    fig.tight_layout()
    # Plot admin
    fig.savefig(save_path)


# Flow results gif plotter


def make_gif(data_flow):
    """
    GIF generator for flow results
    """
    # Frame repo init
    frames = []
    # Frame generation
    # for i in range(len(data_flow)):
    for _, flow in enumerate(data_flow):
        # Plot epoch related flow results
        corner.corner(flow)
        # Create frame buffer
        img_buf = io.BytesIO()
        # Save frames to buffer
        plt.savefig(img_buf, format='png')
        # Re-init
        plt.close()
        # Add to frame repo
        image = Image.open(img_buf)
        frames.append(image)
    # Get first frame
    frame_one = frames[0]
    # Save fig
    frame_one.save(
        #f'./results/{RUN_NAME}_animation.gif',
        './results/flow_animation.gif',
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=100,
        loop=0,
    )
    # Terminate buffer
    img_buf.close()
