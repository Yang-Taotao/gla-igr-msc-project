
"""
Plotter functions repository for VI related tasks.
"""
# Library import
import io
import matplotlib.pyplot as plt
import corner
import numpy as np
from PIL import Image


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


def flow_posterior(x_gen):
    """
    Generate posterior distribution approximated by NF
    """
    corner.corner(
        np.array(x_gen),
        labels=[r'$\psi$', r'$\phi$'],
        plot_density=True,
        plot_datapoints=True,
    )
    # plt.savefig(f'./results/{RUN_NAME}_posterior.png')
    plt.savefig('./results/flow_posterior.png')
    plt.close()


def training_loss(losses):
    """
    Plot training loss
    """
    plt.plot(losses, lw='2', alpha=0.8, color='black')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    # plt.savefig(f'./results/{RUN_NAME}_loss.png')
    plt.savefig('../results/flow_loss.png')
    plt.close()
