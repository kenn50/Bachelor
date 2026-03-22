from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from jax import grad, numpy as jnp, random, vmap
from jax.lax import stop_gradient
import jax.scipy.linalg
import jax.scipy.stats

from numpyro.contrib.einstein.stein_util import median_bandwidth
from numpyro.distributions import biject_to
from numpyro.infer.autoguide import AutoNormal


from numpyro.contrib.einstein import RBFKernel



class SingleSiteRBFKernel(RBFKernel):
    def __init__(
        self,
        site_name,
        bandwidth_factor: Callable[[float], float] = lambda n: 1 / jnp.log(n),
    ):
        super().__init__("matrix", "norm_diag", bandwidth_factor)
        self.site_name = site_name
    
    def compute(self, rng_key, particles, particle_info, loss_fn):
        site_slice = particle_info[self.site_name]

        particles = particles[:, site_slice[0]:site_slice[1]]

        bandwidth = median_bandwidth(particles, self.bandwidth_factor)

        def kernel(x, y):
            x_in = x[site_slice[0]:site_slice[1]]
            y_in = y[site_slice[0]:site_slice[1]]

            
            reduce = jnp.sum if self._normed() else lambda x: x
            kernel_res = jnp.exp(-reduce((x_in - y_in) ** 2) / stop_gradient(bandwidth))
            
            diag_vals = jnp.ones(x.shape[0])
            diag_vals = diag_vals.at[site_slice[0]:site_slice[1]].set(kernel_res) 

            return jnp.diag(diag_vals)

        return kernel
