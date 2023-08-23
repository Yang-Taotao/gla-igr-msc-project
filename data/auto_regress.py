import os
import numpy as np

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import optax
import corner
import matplotlib
import matplotlib.pyplot as pl
import corner
import jax
#jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
from functools import partial

import sys

import distrax
import haiku as hk
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple
Array = jnp.ndarray
PRNGKey = Array
OptState = Any

from PIL import Image, ImageDraw
import io

#routines for plotting the gif of flows
def make_gif(flows):
    frames = []
    for i in range(len(flows)):
        fig = corner.corner(flows[i])
        img_buf = io.BytesIO()
        pl.savefig(img_buf, format='png')
        pl.close()
        im = Image.open(img_buf)
        frames.append(im)

        
    frame_one = frames[0]
    frame_one.save(f'results/{run_name}/{run_name}_animation.gif', format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)
    img_buf.close()


class Bivariate_von_Mises:

    def __init__(self, loc,concentration,correlation):
        self.mu, self.nu = loc
        self.k1, self.k2 = concentration
        self.k3 = correlation

    def log_prob(self, x):
        phi, psi = x.T
        return self.k1*jnp.cos(phi-self.mu)+self.k2*jnp.cos(psi-self.nu)-self.k3*jnp.cos(phi-self.mu-psi+self.nu)
    
    def prob(self, x):
        phi, psi = x.T
        return jnp.exp(self(log_prob(phi, psi)))


#conditiiner is the NN (parameters of the spline)
def make_conditioner(
    event_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    num_bijector_params: int
) -> hk.Sequential:
  """Creates an."""
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
    """Creates the flow model."""
    # Alternating binary mask.
    mask = np.arange(0, np.prod(event_shape)) % 2
    mask = np.reshape(mask, event_shape)
    mask = mask.astype(bool)

    range_min=0.0
    range_max=2*np.pi

    def bijector_fn(params: Array):
        return distrax.RationalQuadraticSpline(
        #    params, range_min=0.0, range_max=2*np.pi, boundary_slopes = 'circular'   #circular spline
            params, range_min=0.0, range_max=2*np.pi    #regular spline
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
            conditioner=make_conditioner(event_shape, hidden_sizes,
                                        num_bijector_params))
        layers.append(layer)
        # Flip the mask after each layer.
        mask = jnp.logical_not(mask)

    # We invert the flow so that the `forward` method is called with `log_prob`.
    flow = distrax.Inverse(distrax.Chain(layers))                                   #bijective transformation from base (normal) to parameter space 
    base_distribution = distrax.Independent(
        #distrax.Uniform(low=jnp.ones(event_shape)*-1, high=jnp.ones(event_shape)*1),
        distrax.Uniform(low=jnp.ones(event_shape)*range_min, high=jnp.ones(event_shape)*range_max),
        #distrax.Normal(loc=jnp.zeros(event_shape), scale=jnp.ones(event_shape)),
        reinterpreted_batch_ndims=len(event_shape)
    )

    return distrax.Transformed(base_distribution, flow)


@hk.without_apply_rng
@hk.transform
def sample_and_log_prob(prng_key: PRNGKey, n: int) -> Tuple[Any, Array]:
    event_shape=(n_params,)

    model = make_flow_model(
        event_shape=event_shape,
        num_layers=flow_num_layers,
        hidden_sizes=[hidden_size] * mlp_num_layers,
        num_bins=num_bins
    )
    return model.sample_and_log_prob(seed=prng_key, sample_shape=(n,))

@hk.without_apply_rng
@hk.transform
def flow_prob(x: Array) -> Array:
    event_shape=(n_params,)

    model = make_flow_model(
        event_shape=event_shape,
        num_layers=flow_num_layers,
        hidden_sizes=[hidden_size] * mlp_num_layers,
        num_bins=num_bins
    )
    return model.prob(x)

def loss_fn(params: hk.Params, prng_key: PRNGKey, n: int) -> Array:

    x_flow, log_q = sample_and_log_prob.apply(params, prng_key, n)
    log_p = dist.log_prob(x_flow)
      
    loss = jnp.mean(log_q - log_p)
    return loss


#@jax.jit
def update(
    params: hk.Params,
    prng_key: PRNGKey,
    opt_state: OptState,
) -> Tuple[hk.Params, OptState]:
    """Single SGD update step."""
    grads = jax.grad(loss_fn)(params, prng_key, Nsamps)
    updates, new_opt_state = optimiser.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


if __name__ == '__main__':

    run_name = sys.argv[1] #name of the run

    #target distribution. Bivariate von Mises distribution on a 2-Torus.
    loc=[0.0, 0.0]
    concentration=[4.0, 4.0]
    correlation = 0.
    dist = Bivariate_von_Mises(loc, concentration, correlation)

    n_params = 2
    flow_num_layers = 2
    hidden_size = 8
    mlp_num_layers = 2
    num_bins = 4

    # perform variational inference
    epochs = 10000
    loss = dict(train=[], val=[])
    Nsamps = 1000

    learning_rate = 0.001
    optimiser = optax.adam(learning_rate)

    prng_seq = hk.PRNGSequence(42)
    key = next(prng_seq)
    params = sample_and_log_prob.init(key, prng_key=key, n=Nsamps)
    opt_state = optimiser.init(params)

    from tqdm import tqdm, trange
    ldict = dict(loss = 0)
    losses = []
    flows = []
    with trange(epochs) as tepochs:
        for epoch in tepochs:
            prng_key = next(prng_seq)
            loss = loss_fn(params,  prng_key, Nsamps)
            ldict['loss'] = f'{loss:.2f}'
            losses.append(loss)
            tepochs.set_postfix(ldict, refresh=True)
            params, opt_state = update(params, prng_key, opt_state)

            if epoch%100 == 0:
                print(f'Epoch {epoch}, loss {loss}')
                x_gen, log_prob_gen = sample_and_log_prob.apply(params, next(prng_seq), 10*Nsamps)
                samples = np.array((x_gen+np.pi)%(2*np.pi))
                #fig = corner.corner(samples)
                flows.append(samples)
                #pl.savefig(f'results/{run_name}/flow_{epoch}.png')
                #pl.close()
            


    print("Done!")

      
    #save plot of the final posterior
    x_gen, log_prob_gen = sample_and_log_prob.apply(params, next(prng_seq), 100*Nsamps)
    fig = corner.corner(np.array(x_gen))
    pl.savefig(f'results/{run_name}/{run_name}_posterior.png')
    pl.close()

    #save plot of the loss
    pl.plot(losses)
    pl.xlabel("Iteration")
    pl.ylabel("loss")
    pl.savefig(f'results/{run_name}/{run_name}_loss.png')
    pl.close()

    #save loss array
    f = open(f'results/{run_name}/{run_name}_loss.npy', 'wb')
    np.save(f,np.array(losses))
    f.close()

    #plot animation of the flows
    make_gif(flows)
