import jax.numpy as jnp
import jax
import numpyro
import numpyro.distributions as dist
from typing import Literal
from numpyro.handlers import trace, seed, substitute

import numpyro
from numpyro import optim
import numpyro.distributions as dist
from numpyro.examples.datasets import MNIST, load_dataset
from numpyro.infer import SVI, Trace_ELBO

from CustomModules.stein_impl_source import SteinVI
from CustomModules.single_site_rbf import SingleSiteRBFKernel
from CustomModules.custom_messengers import scale_sites


from tqdm import trange

class BaseVAE:
    def __init__(
            self, 
            encoder_stax,
            encoder_args,
            decoder_stax,
            decoder_args,
            z_dim,
            model_mode: Literal["n", "b"] = "n", 
            normal_scale=0.1):
        
        self.encoder = encoder_stax
        self.encoder_args = encoder_args
        self.decoder = decoder_stax
        self.decoder_args = decoder_args
        self.z_dim = z_dim
        self.model_mode = model_mode
        self.normal_scale = normal_scale
        self.inference = False


    def train_mode(self):
        self.inference = False

    def eval_mode(self):
        self.inference = True

    def get_guide(self):
        def guide(batch, **kwargs):
            batch = jnp.reshape(batch, (batch.shape[0], -1))
            batch_dim, out_dim = jnp.shape(batch)
            encode = numpyro.module("encoder", self.encoder(**self.encoder_args), (batch_dim, out_dim))
            z_loc, z_std = encode(batch)
            
            
            
            numpyro.deterministic("z_loc", z_loc)
            numpyro.deterministic("z_std", z_std)

            plate_size = self.total_size if not self.inference else batch_dim
            with numpyro.plate("batch", size=plate_size, subsample_size=batch_dim):
                return numpyro.sample("z", dist.Normal(z_loc, z_std).to_event(1))
            
        return guide
    
    def get_generative_model(self):
        
        def generative_model(num_samples, **kwargs):
            decode = numpyro.module("decoder", self.decoder(**self.decoder_args), (num_samples, self.z_dim))


            plate_size = self.total_size if not self.inference else num_samples
            with numpyro.plate("batch", size=plate_size, subsample_size=num_samples):

                z = numpyro.sample("z", dist.Normal(0, 1).expand([self.z_dim]).to_event(1))
                x = numpyro.deterministic("x", decode(z))

                if self.model_mode=="n":    
                    return numpyro.sample(
                        "obs", 
                        dist.Normal(x, scale=self.normal_scale).to_event(1)
                    )
                elif self.model_mode=="b":
                    return numpyro.sample("obs", dist.Bernoulli(x).to_event(1))
                
        return generative_model

    def get_training_model(self):
        generative_model = self.get_generative_model()
        
        def training_model(batch, **kwargs):
            batch = jnp.reshape(batch, (batch.shape[0], -1))
            batch_dim, out_dim = jnp.shape(batch)
            with numpyro.handlers.condition(data={"obs": batch}):
                generative_model(num_samples=batch_dim)
            
        return training_model
    
    def set_params(self, params):
        self.params = params
    
    def encode_batch(self, batch, rng_key):
        self.eval_mode()
        guide = self.get_guide()
        guide = substitute(guide, data=self.params)
        guide = seed(guide, rng_seed=rng_key)
        guide = trace(guide)
        raw_trace = guide.get_trace(batch)
        values = {k: v["value"] for k, v in raw_trace.items() if v["type"] in ["sample", "deterministic"]}
        return values
    
    def decode_latent(self, sites, rng_key):
        self.eval_mode()

        model = self.get_generative_model()
        model = substitute(model, data={**sites, **self.params})
        model = seed(model, rng_seed=rng_key)
        model = trace(model)
        raw_trace = model.get_trace(num_samples = sites["z"].shape[0])
        values = {k: v["value"] for k, v in raw_trace.items() if v["type"] in ["sample", "deterministic"]}
        return values
    
    def sample(self, rng_key, num_samples=100):
        self.eval_mode()

        model = self.get_generative_model()
        model = substitute(model, data=self.params)
        model = seed(model, rng_seed=rng_key)
        model = trace(model)
        raw_trace = model.get_trace(num_samples=num_samples)
        values = {k: v["value"] for k, v in raw_trace.items() if v["type"] in ["sample", "deterministic"]}
        return values
    

    def train(self, dataloader, total_size, optim, num_epochs, rng_key, annealed_sites=["z"], annealing_epochs=100):
        assert annealing_epochs <= num_epochs, "Annealing epochs cannot be greater than total epochs"
        self.train_mode()
        self.total_size = total_size

        svi = SVI(self.get_training_model(), self.get_guide(), optim, Trace_ELBO())
        

        dummy_batch = next(iter(dataloader))
        svi_state = svi.init(rng_key, dummy_batch)


        @jax.jit
        def annealed_update_step(svi_state, batch, beta):
            with scale_sites(scale=beta, sites=annealed_sites):
                return svi.update(svi_state, batch)
        
        
            
        for epoch in trange(num_epochs):
            for batch in dataloader:
                svi_state, loss = annealed_update_step(svi_state, batch, beta=min(1.0, 1.0*epoch/annealing_epochs))
                
        self.params = svi.get_params(svi_state)


class FlowBasicVAE(BaseVAE):
    def __init__(
        self, 
        encoder_stax,
        encoder_args,
        decoder_stax,
        decoder_args,
        z_dim,
        flow,
        flow_args,
        model_mode: Literal["n", "b"] = "n", 
        normal_scale=0.1):
        
        super().__init__(
            encoder_stax,
            encoder_args,
            decoder_stax,
            decoder_args,
            z_dim,
            model_mode, 
            normal_scale
        )

        self.flow = flow
        self.flow_args = flow_args
    
    def get_generative_model(self):
        
        def generative_model(num_samples, **kwargs):
            
            decode = numpyro.module("decoder", self.decoder(**self.decoder_args), (num_samples, self.z_dim))
            
            flow_transform = numpyro.module(
                "flow", 
                self.flow(**self.flow_args),
                input_shape=(num_samples, self.z_dim)
            )()

            def get_flow_dist():
                d = dist.Normal(0, 1).expand([self.z_dim]).to_event(1)
                m = numpyro.sample("m", d)
                m_dist = dist.Normal(m, 1).to_event(1)
                flow_dist = dist.TransformedDistribution(m_dist, flow_transform)
                return flow_dist
            

            
            plate_size = self.total_size if not self.inference else num_samples
            with numpyro.plate("batch", size=plate_size, subsample_size=num_samples):

                z = numpyro.sample("z", get_flow_dist())

                img_loc = decode(z)
                
                numpyro.deterministic("clean", img_loc)
                
                # P(x|z,m)
                return numpyro.sample(
                    "obs", 
                    dist.Normal(img_loc, scale=0.1).to_event(1)
                )
                
        return generative_model



class GlobalVAE(BaseVAE):
    def get_guide_m(self):
        def guide_m(**kwargs):
            def a_init(key):
                return dist.Normal(0, 5).expand([self.z_dim]).sample(key)
            a = numpyro.param("a", a_init)
            B = numpyro.param("B", 0.1*jnp.ones(self.z_dim), constraint=dist.constraints.positive)
            m = numpyro.sample("m", dist.Normal(a, B).to_event(1))
            return m

        return guide_m

    def get_guide_z(self):
        def guide_z(batch, m, **kwargs):
            batch = jnp.reshape(batch, (batch.shape[0], -1))
            batch_dim, out_dim = jnp.shape(batch)
            encode = numpyro.module("encoder", self.encoder(**self.encoder_args), (batch_dim, out_dim + self.z_dim))

            # Concatanate it for the encoder
            broadcasted = m + jnp.zeros((batch_dim, self.z_dim))
            concat_input = jnp.concat([batch, broadcasted], axis=1)

            z_loc, z_std = encode(concat_input)

            plate_size = self.total_size if not self.inference else batch_dim
            with numpyro.plate("batch", size=plate_size, subsample_size=batch_dim):
                d = dist.Normal(z_loc, z_std).to_event(1)
                z = numpyro.sample("z", d)
                return z
        return guide_z
        
    def get_guide(self):
        def guide(batch, **kwargs):
            m = self.get_guide_m()()
            zs = self.get_guide_z()(batch, m)
            return zs
            
        return guide
    
    def get_generative_model(self):
        
        def generative_model(num_samples, **kwargs):
            
            decode = numpyro.module("decoder", self.decoder(**self.decoder_args), (num_samples, 2*self.z_dim))


            def get_global_dist():

                m = numpyro.sample("m", dist.Normal(0, 1).expand([self.z_dim]).to_event(1))

                global_dist = dist.Normal(m, 1)
                return m, global_dist
            
            if not self.inference:
                m, global_dist = get_global_dist()
            
            plate_size = self.total_size if not self.inference else num_samples
            with numpyro.plate("batch", size=plate_size, subsample_size=num_samples):

                if self.inference:
                    m, global_dist = get_global_dist()

                z = numpyro.sample("z", global_dist.to_event(1))

                concat_input = jnp.concatenate([z, m + jnp.zeros((num_samples, self.z_dim))], axis=1)
                
                img_loc = decode(concat_input)
                
                numpyro.deterministic("clean", img_loc)
                
                #P(x|z, m)
                return numpyro.sample(
                    "obs", 
                    dist.Normal(img_loc, scale=0.1).to_event(1)
                )
                
        return generative_model
    

    def get_ms(self, rng_key, n):
        guide_m = self.get_guide_m()
        def m_single(rng_key):
                g = numpyro.handlers.seed(guide_m, rng_seed=rng_key)
                g = numpyro.handlers.substitute(g, data=self.params)
                return g()

        batch_keys = jax.random.split(rng_key, n)

        ms = jax.vmap(m_single)(batch_keys)

        return ms


    
    def encode_batch(self, batch, rng_key):
        self.eval_mode()
        
        ms_key, seed_key = jax.random.split(rng_key)

        ms = self.get_ms(ms_key, batch.shape[0])
        

        guide = self.get_guide_z()
        guide = substitute(guide, data=self.params)
        guide = seed(guide, rng_seed=seed_key)
        guide = trace(guide)
        raw_trace = guide.get_trace(batch, ms)
        values = {k: v["value"] for k, v in raw_trace.items() if v["type"] in ["sample", "deterministic"]}
        values["m"] = ms
        return values
    
    def sample(self, rng_key, num_samples=100):
        self.eval_mode()

        ms_key, seed_key = jax.random.split(rng_key)

        ms = self.get_ms(ms_key, num_samples)

        model = self.get_generative_model()
        model = substitute(model, data={"m": ms, **self.params})
        model = seed(model, rng_seed=seed_key)
        model = trace(model)
        raw_trace = model.get_trace(num_samples=num_samples)
        values = {k: v["value"] for k, v in raw_trace.items() if v["type"] in ["sample", "deterministic"]}
        return values
    
    def decode_latent(self, variables, rng_key):
        self.eval_mode()

        ms_key, seed_key = jax.random.split(rng_key)

        if not "m" in variables.keys():
            ms = self.get_ms(ms_key, variables["z"].shape[0])
            variables["m"] = ms 
        

        model = self.get_generative_model()
        model = substitute(model, data={**variables, **self.params})
        model = seed(model, rng_seed=seed_key)
        model = trace(model)
        raw_trace = model.get_trace(num_samples = variables["z"].shape[0])
        values = {k: v["value"] for k, v in raw_trace.items() if v["type"] in ["sample", "deterministic"]}
        return values
    


class GuideFlowGlobalVAE(GlobalVAE):
    def __init__(
        self, 
        encoder_stax,
        encoder_args,
        decoder_stax,
        decoder_args,
        z_dim,
        flow,
        flow_args,
        model_mode: Literal["n", "b"] = "n", 
        normal_scale=0.1):
        
        super().__init__(
            encoder_stax,
            encoder_args,
            decoder_stax,
            decoder_args,
            z_dim,
            model_mode, 
            normal_scale
        )

        self.flow = flow
        self.flow_args = flow_args
    
    def get_guide_m(self):
        def guide_m(**kwargs):
            flow_transform = numpyro.module(
                "flow", 
                self.flow(**self.flow_args),
                input_shape=(self.z_dim,)
            )()
            def a_init(key):
                return dist.Normal(0, 5).expand([self.z_dim]).sample(key)
            a = numpyro.param("a", a_init)
            B = numpyro.param("B", 0.1*jnp.ones(self.z_dim), constraint=dist.constraints.positive)

            m_dist =  dist.Normal(a, B).to_event(1)
            flow_dist = dist.TransformedDistribution(m_dist, flow_transform)
            m = numpyro.sample("m", flow_dist)
            return m

        return guide_m
    

class LearnedMeanGuideFlowGlobalVAE(GuideFlowGlobalVAE):
    def __init__(
        self, 
        encoder_stax,
        encoder_args,
        decoder_stax,
        decoder_args,
        mean_network_stax,
        mean_network_args,
        z_dim,
        flow,
        flow_args,
        model_mode: Literal["n", "b"] = "n", 
        normal_scale=0.1):
        
        super().__init__(
            encoder_stax,
            encoder_args,
            decoder_stax,
            decoder_args,
            z_dim,
            flow,
            flow_args,
            model_mode, 
            normal_scale
        )

        self.mean_network = mean_network_stax
        self.mean_network_args = mean_network_args

    def get_generative_model(self):
        
        def generative_model(num_samples, **kwargs):
            
            decode = numpyro.module("decoder", self.decoder(**self.decoder_args), (num_samples, 2*self.z_dim))
            

            def get_global_dist():

                m = numpyro.sample("m", dist.Normal(0, 1).expand([self.z_dim]).to_event(1))
                mean_network = numpyro.module("mean_network", self.mean_network(**self.mean_network_args), m.shape)
                mean, var = mean_network(m) 
                mean = mean + m


                global_dist = dist.Normal(mean, var)
                return m, global_dist
            
            if not self.inference:
                m, global_dist = get_global_dist()
            
            plate_size = self.total_size if not self.inference else num_samples
            with numpyro.plate("batch", size=plate_size, subsample_size=num_samples):

                if self.inference:
                    m, global_dist = get_global_dist()

                z = numpyro.sample("z", global_dist.to_event(1))

                concat_input = jnp.concatenate([z, m + jnp.zeros((num_samples, self.z_dim))], axis=1)
                
                img_loc = decode(concat_input)
                
                numpyro.deterministic("clean", img_loc)
                
                #P(x|z, m)
                return numpyro.sample(
                    "obs", 
                    dist.Normal(img_loc, scale=0.1).to_event(1)
                )
                
        return generative_model


class SteinGlobalVAE(GlobalVAE):
    def train(self, dataloader, total_size, optim, num_epochs, rng_key, num_stein_particles, repulsion_temperature=1, bandwidth_scaler=1, annealed_sites=["z"], annealing_epochs=100):
        assert annealing_epochs <= num_epochs, "Annealing epochs cannot be greater than total epochs"
        
        self.train_mode()
        self.inference = False
        self.total_size = total_size

        kernel = SingleSiteRBFKernel("a", bandwidth_factor=lambda n: bandwidth_scaler / jnp.log(n))
        self.num_stein_particles = num_stein_particles
    
        def non_stein(name):
            if name in ["a", "B"]:
                return False
            else:
                return True
        

        stein = SteinVI(self.get_training_model(), 
                        self.get_guide(),
                          optim, 
                          kernel, 
                          num_stein_particles=num_stein_particles, 
                          non_mixture_guide_params_fn=non_stein, 
                          repulsion_temperature=repulsion_temperature)

        
        dummy_batch = next(iter(dataloader))
        svi_state = stein.init(rng_key, dummy_batch)


        @jax.jit
        def annealed_update_step(svi_state, batch, beta):
            with scale_sites(scale=beta, sites=annealed_sites):
                return stein.update(svi_state, batch)
        
        t = trange(num_epochs)
            
        for epoch in t:
            for batch in dataloader:
                svi_state, loss = annealed_update_step(svi_state, batch, beta=min(1.0, 1.0*epoch/annealing_epochs))
            if epoch % 25 == 0:
                t.set_description(f"Epoch {epoch}, Loss: {loss:.2f}")

            self.params = stein.get_params(svi_state)

        


    
    def get_ms(self, rng_key, n):
        guide_m = self.get_guide_m()
        def m_single(rng_key):
                rng_key, sub_key, m_key = jax.random.split(rng_key, 3)
                particle = jax.random.randint(sub_key, (), 0, self.num_stein_particles)
                params = dict(self.params)

                params["a"] = jax.tree.map(lambda x: x[particle], params["a"])
                params["B"] = jax.tree.map(lambda x: x[particle], params["B"])

                g = numpyro.handlers.seed(guide_m, rng_seed=m_key)
                g = numpyro.handlers.substitute(g, data=params)
                return g(), particle

        batch_keys = jax.random.split(rng_key, n)

        ms, pidx = jax.vmap(m_single)(batch_keys)

        return ms, pidx

    def encode_batch(self, batch, rng_key):
        self.eval_mode()
        
        ms_key, seed_key = jax.random.split(rng_key)

        ms, pidx = self.get_ms(ms_key, batch.shape[0])
        

        guide = self.get_guide_z()
        guide = substitute(guide, data=self.params)
        guide = seed(guide, rng_seed=seed_key)
        guide = trace(guide)
        raw_trace = guide.get_trace(batch, ms)
        values = {k: v["value"] for k, v in raw_trace.items() if v["type"] in ["sample", "deterministic"]}
        values["m"] = ms
        values["pidx"] = pidx
        return values
    
    def sample(self, rng_key, num_samples=100):
        self.eval_mode()

        ms_key, seed_key = jax.random.split(rng_key)

        ms, pidx = self.get_ms(ms_key, num_samples)

        model = self.get_generative_model()
        model = substitute(model, data={"m": ms, **self.params})
        model = seed(model, rng_seed=seed_key)
        model = trace(model)
        raw_trace = model.get_trace(num_samples=num_samples)
        values = {k: v["value"] for k, v in raw_trace.items() if v["type"] in ["sample", "deterministic"]}
        values["pidx"] = pidx
        return values
    
    def decode_latent(self, variables, rng_key):
        self.eval_mode()

        ms_key, seed_key = jax.random.split(rng_key)

        pidx = None

        if not "m" in variables.keys():
            ms, pidx = self.get_ms(ms_key, variables["z"].shape[0])
            variables["m"] = ms 
        

        model = self.get_generative_model()
        model = substitute(model, data={**variables, **self.params})
        model = seed(model, rng_seed=seed_key)
        model = trace(model)
        raw_trace = model.get_trace(num_samples = variables["z"].shape[0])
        values = {k: v["value"] for k, v in raw_trace.items() if v["type"] in ["sample", "deterministic"]}
        values["pidx"] = pidx
        return values
        

class SteinGuideFlowVAE(SteinGlobalVAE, GuideFlowGlobalVAE):
    pass


class LearnedMeanSteinGuideFlowVAE(SteinGlobalVAE, LearnedMeanGuideFlowGlobalVAE):
    pass
