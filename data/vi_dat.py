"""
VI functions and flow model.
"""
# Library import
from typing import Any, Sequence, Tuple
import haiku as hk
# Package - jax, numpy
import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
from tqdm import trange
# Other imports
from data import vi_plt
from data.vi_cfg import (
    NUM_PARAMS,
    NUM_FLOW_LAYERS,
    HIDDEN_SIZE,
    NUM_MLP_LAYERS,
    NUM_BINS,
    NUM_SAMPLES,
    LEARNING_RATE,
    TOTAL_EPOCHS,
    PRNG_SEQ,
    DIST_BVM,
    #DIST_GW,
)
# Aliasing
PRNGKey = jnp.ndarray
OptState = Any


# Get target distribution from config
DIST = DIST_BVM

# Other configs import
OPTIMISER = optax.adam(LEARNING_RATE)
key = next(PRNG_SEQ)

# Flow training function


def train_flow():
    """
    Preliminary flow training script
    Generate and save training loss
    Plot training loss
    Plot approximated posterior
    """
    # Local init
    loss = {"train": [], "val": []}
    ldict = {"loss": 0}
    losses = []
    flows = []

    data_param = sample_and_log_prob.init(key, prng_key=key, data_n=NUM_SAMPLES)
    data_opt_state = OPTIMISER.init(data_param)

    # Start training print
    print()
    print("Training: Initiated")
    print("=" * 30)

    # Training
    with trange(TOTAL_EPOCHS) as tepochs:
        for epoch in tepochs:
            data_prng_key = next(PRNG_SEQ)
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
                    next(PRNG_SEQ),
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
        next(PRNG_SEQ),
        100*NUM_SAMPLES,
    )
    # Save plot of the loss
    vi_plt.flow_posterior(x_gen)
    vi_plt.training_loss(losses)
    # Plot animation of the flows
    vi_plt.make_gif(flows)

    # Save loss array
    file_loss = open('./results/flow_loss.npy', 'wb')
    np.save(file_loss, np.array(losses))
    file_loss.close()


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
    log_p = DIST.log_prob(x_flow)
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
    updates, new_opt_state = OPTIMISER.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    # Func return
    return new_params, new_opt_state
