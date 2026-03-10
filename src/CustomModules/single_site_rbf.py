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
        mode="norm",
        matrix_mode="norm_diag",
        bandwidth_factor: Callable[[float], float] = lambda n: 1 / jnp.log(n),
    ):
        self.site_name = site_name
        super().__init__(mode, matrix_mode, bandwidth_factor)
    
    def compute(self, rng_key, particles, particle_info, loss_fn):
        site_slice = particle_info[self.site_name]
        particles = particles[:, site_slice[0]:site_slice[1]]
        return super().compute(rng_key, particles, particle_info, loss_fn)
