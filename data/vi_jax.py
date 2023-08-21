"""
This is the variational inference script for MSc project.

Created on Thu August 14 2023
Source: https://github.com/dominika-zieba/VI/
"""
# %%
# Libnrary import
import os
from typing import Any, Tuple
from tqdm import trange
import haiku as hk  # neural network library for JAX
import optax
import lal
import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config
import matplotlib.pyplot as plt
import corner
# Custom import
from data import vi_routines
# Config setup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
config.update("jax_enable_x64", True)
# Aliasing
Array = jnp.ndarray
PRNGKey = jnp.ndarray
OptState = Any

# %%
# LogLikelihood

# Import results of the log10 based density

# It will have to be written in a way that
# it can take an array of samples from the parameter space
# and evaluates your density for each of them

# Do it ideally as operations on vectors and not a for loop
# because the for loop will not work well in jax

# %%
# Training setup


@hk.without_apply_rng
@hk.transform
def sample_and_log_prob(prng_key: PRNGKey, n: int) -> Tuple[Any, Array]:
    """
    Missing docstring
    """
    model = vi_routines.make_flow_model(  # this is the flow distribution (a distrax object)
        event_shape=(n_params,),
        num_layers=flow_num_layers,
        hidden_sizes=[hidden_size] * mlp_num_layers,
        num_bins=num_bins
    )

    return model.sample_and_log_prob(seed=prng_key, sample_shape=(n,))
    # returns x (sample from the flow q),
    # and model.log_prob(x) (array of log(q) of th sampled points)


# Computes reverse KL-divergence for the sample x_flow
# between the flow and gw loglikelihood.
def loss_fn(params: hk.Params, prng_key: PRNGKey, n: int) -> Array:
    """
    Missing docstring
    """
    # Gets sample from the flow and computes log_q for the sampled points.
    x_flow, log_q = sample_and_log_prob.apply(params, prng_key, n)
    # log_p = log_likelihood(x_flow)

    log_p = # insert log density here

    # Gets gw loglikelihood for the sampled points
    # (after transforming them into physical space..)
    loss_result = jnp.mean(log_q - log_p)
    # Func return
    return loss_result


@jax.jit
def update(
        params: hk.Params,
        prng_key: PRNGKey,
        opt_state: OptState,
    ) -> Tuple[hk.Params, OptState]:
    """
    Single SGD update step
    """
    # Gradient w.r.t. params, evalueated at params, prng_key, Nsamps.
    grads = jax.grad(loss_fn)(params, prng_key, Nsamps)
    updates, new_opt_state = optimiser.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    # Func return
    return new_params, new_opt_state


# %%
# Call and train
if __name__ == '__main__':

    #Problem setup
    f_min = 0.  # minimum frequency cut-off
    T = 1.  # data segment duration (seconds)
    Fs = 100.  # samplingfrequency  (Hz)
    t_start = 0.

    #initialise the detectors
    H1 = Interferometer('H1', 'O3', f_min, T, Fs, t_start, 20011997)
    L1 = Interferometer('L1', 'O3', f_min, T, Fs, t_start, 27071994)

    true_params = dict(
        A=0.5,
        t0=0.5,
        f0=20.,
        tau=1/10,
        ra=jnp.atleast_1d(0.5),
        dec=jnp.atleast_1d(0.5),
        psi=jnp.atleast_1d(0.5),
    )

    H1_response = make_detector_response(H1.tensor, H1.location)
    L1_response = make_detector_response(L1.tensor, L1.location)
    response = {'H1': H1_response, 'L1': L1_response}

    # Compute signal SNR
    gps = lal.LIGOTimeGPS(true_params['t0'])
    gmst_rad = lal.GreenwichMeanSiderealTime(gps)

    L1.signal = L1_response(
        L1.freqs,
        hp,
        hc,
        true_params['ra'],
        true_params['dec'],
        gmst_rad,
        true_params['psi'],
    )
    H1.signal = H1_response(
        H1.freqs,
        hp,
        hc,
        true_params['ra'],
        true_params['dec'],
        gmst_rad,
        true_params['psi'],
    )

    H1.psd = 1.
    L1.psd = 1.
    network = Network([H1, L1])
    print('Network SNR of the injected signal: ', network.snr())

    # Flow parameters
    n_params = 4
    flow_num_layers = 4
    hidden_size = 16
    mlp_num_layers = 2
    num_bins = 6

    # Training parameters
    epochs = 9000
    Nsamps = 1000

    learning_rate = 0.01
    optimiser = optax.adam(learning_rate)  # stochastic gradient descent

    prng_seq = hk.PRNGSequence(42)
    key = next(prng_seq)
    params = sample_and_log_prob.init(key, prng_key=key, n=Nsamps)
    opt_state = optimiser.init(params)

    ldict = dict(loss=[])
    training_stats = dict(loss=[], mean_loss=np.inf)

    log_l = LogL(true_params)

    # Test likelihood
    l = log_l(true_params)
    print(f'log likelihood of true params = {l}')

    truths = [0.5, 0.5, 0.5, 0.5]

    with trange(epochs) as tepochs:
        for epoch in tepochs:
            # Update NN params
            # (stochastic gradient descent with Adam optimiser)
            prng_key = next(prng_seq)
            loss = loss_fn(params,  prng_key, Nsamps)
            ldict['loss'] = f'{loss:.2f}'
            training_stats['loss'].append(loss)

            tepochs.set_postfix(ldict, refresh=True)
            # Take a step in direction of stepest descent (negative gradient)
            params, opt_state = update(params, prng_key, opt_state)

            # Print results every 100 iterations
            # (first one is plotted after 1st update.)
            if (epoch) % 500 == 0:
                print(f'Epoch {epoch}, loss {loss}')
                x_gen, log_prob_gen = sample_and_log_prob.apply(
                    params, next(prng_seq), 10*Nsamps,
                )
                log_posterior = log_prob(x_gen)
                x_gen = np.array(x_gen, copy=False)
                p_gen = np.vstack(list(log_l.array_to_phys(x_gen).values()))
                data = np.concatenate([p_gen, [log_posterior]])
                fig = corner.corner(
                    data.T,
                    labels=['A', 'ra', 'dec', 'psi', 'log_p'],
                    truths=[0.5, 0.5, 0.5, 0.5, 0.],
                )
                plt.show()
                plt.savefig(f'flow_${epoch}.png')

            # if (epoch+1)%500 == 0:
                # Stopping criterion
                # new_mean_loss = np.mean(training_stats['loss'][-500:])
                # if np.abs(
                #   training_stats['mean_loss']-(new_mean_loss)
                #   ) / training_stats['mean_loss'] < 0.001:
                     # Break here if the training loss
                     # has decreased by less then 1% in the last 500 steps.
                #    break
                # else:
                #    training_stats['mean_loss'] = new_mean_loss
    print("Done!")

    x_gen, log_prob_gen = sample_and_log_prob.apply(
        params, next(prng_seq), 100*Nsamps,
    )
    x_gen = np.array(x_gen, copy=False)
    p_gen = np.vstack(list(log_l.array_to_phys(x_gen).values()))
    fig = corner.corner(p_gen.T, truths=truths)
    plt.show()
    plt.savefig(f'posterior_${epochs}.png')

    f = open('samples_emcee_cosprior.npy', 'rb')
    samples = np.load(f)

    kwargs = dict(
        bins=32,
        smooth=0.9,
        quantiles=[0.16, 0.5, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
        plot_density=True,
        plot_datapoints=True,
        fill_contours=True,
        show_titles=False,
        hist_kwargs=dict(density=True),
        range=[(0.2, 1.2), (-0.5, 1.6), (-1.2, 1.4), (0, 1.8), ],
    )

    fig = None

    # Need to update colours for 1D and 2D plots
    kwargs["color"] = "C0"
    kwargs["hist_kwargs"]["color"] = "C0"
    fig = corner.corner(samples, fig=fig, **kwargs)

    kwargs["color"] = "C1"
    kwargs["hist_kwargs"]["color"] = "C1"
    fig = corner.corner(
        p_gen.T,
        labels=log_l.params,
        truths=truths,
        fig=fig,
        **kwargs,
    )

    plt.show()
    plt.savefig(f'posterior_comparison_${epochs}.png')

# %%
