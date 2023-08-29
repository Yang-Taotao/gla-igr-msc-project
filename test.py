"""
This is the variational inference normalizing flow model for MSc project.

Created on Thu August 23 2023
"""
# Library import
import os
from typing import Any, Sequence, Tuple
import haiku as hk
from tqdm import trange
# Package - jax, numpy
import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
from jax.experimental.compilation_cache import compilation_cache as cc
# Package - plotters
import matplotlib.pyplot as plt
import corner
import scienceplots
# Other imports
from data import gw_fim, gw_plt

# Setup options
# XLA GPU resource setup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
cc.initialize_cache("./data/__jaxcache__")
# jax.config.update("jax_enable_x64", True)
# Plotter style customization
plt.style.use(['science', 'notebook', 'grid'])
# Aliasing
PRNGKey = jnp.ndarray
OptState = Any

# Dist - BVM distribution


class BivariateVonMises:
    """
    Original BVM class
    """
    def __init__(self, loc, concentration, correlation):
        self.data_mu, self.data_nu = loc
        self.data_k1, self.data_k2 = concentration
        self.data_k3 = correlation


    # Target dist, need to switch to working dist of fim density
    def log_prob(self, data_x):
        '''
        Get the log probability distribution

        Take in 2d param
        Feed 2d param to external func for some density result (dtype: float ish)
        The float is the return of log_prob()
        '''
        # 2-D parameters
        phi, psi = data_x.T
        # phi = 2*jnp.pi*phi
        # psi = 2*jnp.pi*psi
        # Get result
        result = (
            self.data_k1*jnp.cos(phi - self.data_mu)
            + self.data_k2*jnp.cos(psi - self.data_nu)
            - self.data_k3*jnp.cos(phi - self.data_mu - psi + self.data_nu)
        )
        # Func return
        return result


    def prob(self, data_x):
        """
        Get probability distribution
        """
        # Func return - examine log_prob(input), the input may be inverted
        return jnp.exp(self.log_prob(data_x))


# Dist - GW Metric Density


class TemplateDensity:
    """
    GW template density class for hp based results
    """
    def __init__(self, param_ripple):
        self.data_tc, self.data_phic = param_ripple


    def log_prob(self, data_x):
        '''
        Get the log template density

        Take in 2d param
        Feed 2d param to external func for some density result (dtype: float ish)
        The float is the return of log_prob()
        '''
        # Local assignment, 2-d param
        # data_x.shape (n, 2)
        data_mc, data_eta = data_x.T
        data_mc = data_mc + 1.0 # 1.0 - 2.0
        data_eta = data_eta * 0.24 + 0.01 # 0.05 - 0.25
        # Param build with shape (n, 4)
        param = gw_fim.fim_param_stack(data_mc, data_eta)
        # Get results with shape (n, )
        result = gw_fim.map_density_plus(param)
        # Func return
        return result


    def prob(self, data_x):
        """
        Get probability distribution
        """
        # Func return - examine log_prob(input), the input may be inverted
        return jnp.exp(self.log_prob(data_x))


# Flow model


def make_conditioner(
    event_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    num_bijector_params: int
    ) -> hk.Sequential:
    """
    Creates a conditioner
    The conditiiner is the Neural Network (parameters of the spline)
    """
    return hk.Sequential([
        hk.Flatten(preserve_dims=-len(event_shape)),
        hk.nets.MLP(hidden_sizes, activate_final=True),
        # We initialize this linear layer to zero so that the flow is initialized
        # to the identity function.
        hk.Linear(
            np.prod(event_shape) * num_bijector_params,
            w_init=jnp.zeros,
            b_init=jnp.zeros),
        hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1),
    ])


def make_flow_model(
    event_shape: Sequence[int],
    num_layers: int = 4,
    hidden_sizes: Sequence[int] = [250, 250],
    num_bins: int = 4,
    ) -> distrax.Transformed:
    """
    Creates the normalizing flow model
    """
    # Alternating binary mask.
    mask = np.arange(0, np.prod(event_shape)) % 2
    mask = np.reshape(mask, event_shape)
    mask = mask.astype(bool)
    # Param range definer
    range_min, range_max = 0.0, 2*jnp.pi

    # Bijector
    def bijector_fn(params: jnp.ndarray):
        """
        Missing docstrings
        """
        return distrax.RationalQuadraticSpline(
            # Regular spline
            # This defines the domain of the flow parameters
            params, range_min=0.0, range_max=2*jnp.pi
        )


    # Number of parameters for the rational-quadratic spline:
    # - `num_bins` bin widths
    # - `num_bins` bin heights
    # - `num_bins + 1` knot slopes
    # for a total of `3 * num_bins + 1` parameters.
    num_bijector_params = 3 * num_bins + 1

    layers = []
    for _ in range(num_layers):
        layer = distrax.MaskedCoupling(
            mask=mask,
            bijector=bijector_fn,
            conditioner=make_conditioner(
                event_shape,
                hidden_sizes,
                num_bijector_params,
            )
        )
        layers.append(layer)
        # Flip the mask after each layer.
        mask = jnp.logical_not(mask)

    # We invert the flow so that the `forward` method is called with `log_prob`.
    #bijective transformation from base (normal) to parameter space
    flow = distrax.Inverse(distrax.Chain(layers))
    base_distribution = distrax.Independent(
        #distrax.Uniform(low=jnp.ones(event_shape)*-1, high=jnp.ones(event_shape)*1),
        distrax.Uniform(low=jnp.ones(event_shape)*range_min, high=jnp.ones(event_shape)*range_max),
        #distrax.Normal(loc=jnp.zeros(event_shape), scale=jnp.ones(event_shape)),
        reinterpreted_batch_ndims=len(event_shape)
    )

    return distrax.Transformed(base_distribution, flow)


@hk.without_apply_rng
@hk.transform
def sample_and_log_prob(prng_key: PRNGKey, data_n: int) -> Tuple[Any, jnp.ndarray]:
    """
    This does ...
    """
    # Shape
    event_shape=(NUM_PARAMS,)
    # Model
    model = make_flow_model(
        event_shape=event_shape,
        num_layers=NUM_FLOW_LAYERS,
        hidden_sizes=[HIDDEN_SIZE] * NUM_MLP_LAYERS,
        num_bins=NUM_BINS,
    )
    # Func return
    return model.sample_and_log_prob(seed=prng_key, sample_shape=(data_n,))


@hk.without_apply_rng
@hk.transform
def flow_prob(data_x: jnp.ndarray) -> jnp.ndarray:
    """
    This is what
    """
    # Get shape
    event_shape=(NUM_PARAMS,)
    # Model
    model = make_flow_model(
        event_shape=event_shape,
        num_layers=NUM_FLOW_LAYERS,
        hidden_sizes=[HIDDEN_SIZE] * NUM_MLP_LAYERS,
        num_bins=NUM_BINS,
    )
    # Func return
    return model.prob(data_x)


def loss_fn(params: hk.Params, prng_key: PRNGKey, data_n: int) -> jnp.ndarray:
    """
    Calculate the expected value of Kullback-Leibler (KL) divergence 
    """
    # Local calculation resources
    x_flow, log_q = sample_and_log_prob.apply(params, prng_key, data_n)
    log_p = dist.log_prob(x_flow)
    # Get the KL divergence as loss
    data_loss = jnp.mean(log_q - log_p)
    # Func return
    return data_loss


@jax.jit
def update(
    params: hk.Params,
    prng_key: PRNGKey,
    opt_state: OptState,
    ) -> Tuple[hk.Params, OptState]:
    """
    Single SGD update step
    """
    grads = jax.grad(loss_fn)(params, prng_key, NUM_SAMPLES)
    updates, new_opt_state = optimiser.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    # Func return
    return new_params, new_opt_state


# Flow Class - Training


if __name__ == '__main__':

    # Name of the run, require more info for usage
    # RUN_NAME = sys.argv[1]
# =========================================================================== #
    # Target distribution. Bivariate von Mises distribution on a 2-Torus.
    LOC = [0.0, 0.0]
    CONCENTRATION = [4.0, 4.0]
    CORRELATION = 0.0
    # Target density params tc, phic
    PARAM_RIPPLE = [0.0, 0.0]

    # Flow parameters
    NUM_PARAMS = 2
    NUM_FLOW_LAYERS = 2
    HIDDEN_SIZE = 8
    NUM_MLP_LAYERS = 2
    NUM_BINS = 4

    # Perform variational inference
    TOTAL_EPOCHS = 3000 #reduce this for testing purpose, original val = 10000
    NUM_SAMPLES = 1000 #1000
    LEARNING_RATE = 0.001 #0.001
    print("Main: Local param assigned")
# =========================================================================== #
    dist = BivariateVonMises(LOC, CONCENTRATION, CORRELATION)
    # dist = TemplateDensity(PARAM_RIPPLE)
    optimiser = optax.adam(LEARNING_RATE)

    prng_seq = hk.PRNGSequence(42)
    key = next(prng_seq)
    data_param = sample_and_log_prob.init(key, prng_key=key, data_n=NUM_SAMPLES)
    data_opt_state = optimiser.init(data_param)

    loss = {"train": [], "val": []}
    ldict = {"loss": 0}
    losses = []
    flows = []

    print()
    print("Training: Initiated")
    print("=" * 30)

    with trange(TOTAL_EPOCHS) as tepochs:
        for epoch in tepochs:
            data_prng_key = next(prng_seq)
            loss = loss_fn(data_param, data_prng_key, NUM_SAMPLES)
            ldict['loss'] = f'{loss:.2f}'
            losses.append(loss)
            tepochs.set_postfix(ldict, refresh=True)
            # Problematic grad(loss_fn)
            data_param, data_opt_state = update(data_param, data_prng_key, data_opt_state) 
            # Results
            if epoch%100 == 0:
                x_gen, log_prob_gen = sample_and_log_prob.apply(
                    data_param,
                    next(prng_seq),
                    10*NUM_SAMPLES,
                )
                samples = np.array(x_gen)
                flows.append(samples)
                # Print results
                print(f'At epoch: {epoch}, with loss: {loss}')
    # Print if complete
    print("Training accomplished.")
    print("=" * 30)
    print()

    # Save plot of the final posterior
    x_gen, log_prob_gen = sample_and_log_prob.apply(
        data_param,
        next(prng_seq),
        100*NUM_SAMPLES,
    )
    fig = corner.corner(
        np.array(x_gen),
        labels=[r'$\psi$', r'$\phi$'],
        plot_density=True,
        plot_datapoints=True,
    )
    # plt.savefig(f'./results/{RUN_NAME}_posterior.png')
    plt.savefig('./results/flow_posterior.png')
    plt.close()

    # Save plot of the loss
    plt.plot(losses, lw='2', alpha=0.8, color='black')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    # plt.savefig(f'./results/{RUN_NAME}_loss.png')
    plt.savefig('./results/flow_loss.png')
    plt.close()

    # Save loss array
    # f = open(f'./results/{RUN_NAME}_loss.npy', 'wb')
    f = open('./results/flow_loss.npy', 'wb')
    np.save(f, np.array(losses))
    f.close()

    # Plot animation of the flows
    gw_plt.make_gif(flows)
