"""
This is the variational inference normalizing flow script for MSc project.

Created on Thu August 14 2023
Source: https://github.com/dominika-zieba/VI/
"""
# %%
# Library import
from typing import Any, Sequence
import jax.numpy as jnp
import numpy as np
import haiku as hk
import distrax
# Assign alias
Array = jnp.ndarray
PRNGKey = Array
OptState = Any

# %%
# Normalizing flow model


def make_conditioner(
        event_shape: Sequence[int],
        hidden_sizes: Sequence[int],
        num_bijector_params: int,
    ) -> hk.Sequential:
    """
    Creates a conditioner
    """
    # Func return
    return hk.Sequential([
        hk.Flatten(preserve_dims=-len(event_shape)),
        hk.nets.MLP(hidden_sizes, activate_final=True),
        # Init this linear layer to zero -> make flow init at identity func
        hk.Linear(
            np.prod(event_shape) * num_bijector_params,
            w_init=jnp.zeros,
            b_init=jnp.zeros,
        ),
        hk.Reshape(
            tuple(event_shape) + (num_bijector_params,),
            preserve_dims=-1,
        ),
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

    def bijector_fn(params: Array):
        """
        Create bijector func
        """
        # Func return
        return distrax.RationalQuadraticSpline(
            params, range_min=0.0, range_max=1.0,
        )

    # Number of parameters for the rational-quadratic spline:
    # - "num_bins" bin widths
    # - "num_bins" bin heights
    # - "num_bins + 1" knot slopes
    # for a total of "3 * num_bins + 1" parameters.
    num_bijector_params = 3 * num_bins + 1

    layers = []
    for _ in range(num_layers):
        layer = distrax.MaskedCoupling(
            mask=mask,
            bijector=bijector_fn,
            conditioner=make_conditioner(
                event_shape, hidden_sizes, num_bijector_params,
            ),
        )
        # Add new layer
        layers.append(layer)
        # Flip the mask after each layer.
        mask = jnp.logical_not(mask)

    # Invert the flow so that the "forward" method is called with "log_prob"
    # bijective transformation from base (normal) to parameter space
    flow = distrax.Inverse(distrax.Chain(layers))
    base_distribution = distrax.Independent(
        #distrax.Uniform(low=jnp.ones(event_shape)*-1, high=jnp.ones(event_shape)*1),
        distrax.Uniform(
            low=jnp.zeros(event_shape), high=jnp.ones(event_shape)*1,
        ),
        #distrax.Normal(loc=jnp.zeros(event_shape), scale=jnp.ones(event_shape)),
        reinterpreted_batch_ndims=len(event_shape),
    )

    # Func return
    return distrax.Transformed(base_distribution, flow)
