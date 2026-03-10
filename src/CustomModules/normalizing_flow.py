import jax
from numpyro.distributions.transforms import ComposeTransform, PermuteTransform
from numpyro.distributions.flows import InverseAutoregressiveTransform
from numpyro.nn import AutoregressiveNN
import jax.numpy as jnp

def flow_transform(input_dim, hidden_dims, inv):
    arn_init, arn_apply = AutoregressiveNN(input_dim=input_dim, hidden_dims=hidden_dims)
    
    def apply_fn(params):

        def temp_arn_apply(x):
            mu, log_scale = arn_apply(params, x)
            return mu, log_scale
        if inv:
            t = InverseAutoregressiveTransform(temp_arn_apply)
        else:
            t = InverseAutoregressiveTransform(temp_arn_apply).inv
        return t
        
    return arn_init, apply_fn

def normalizing_flow(input_dim, hidden_dims=[16,16], steps=10, inv =True):
    arn_init, single_layer_apply = flow_transform(input_dim, hidden_dims, inv)
    
    def init_fun(rng, input_shape):
        params = []
        for i in range(steps):
            rng, sub = jax.random.split(rng)
            _, p = arn_init(sub, input_shape)
            params.append(p)
        return (), params

    def apply_fun(params, *args, **kwargs):
        transforms = []
        permutation = jnp.arange(input_dim)[::-1]
        for i in range(steps):
            transforms.append(single_layer_apply(params[i]))

            if i < steps - 1:
                transforms.append(PermuteTransform(permutation))
            
        return ComposeTransform(transforms)

    return init_fun, apply_fun