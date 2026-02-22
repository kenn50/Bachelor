# Week 08

## Notes from supervisor conversation:
- [x] Get SMI to work on a simple variational autoencoder.
- [x] Note any potential errors in the the STEINVI library
- [ ] Write article on SMI VAE


## Potential errors in the STEINVI Library
I added `self.static_kwargs` as input to the `self.find_init_params` method. Otherwise the dimensions would be wrong in my case.
```python
guide_init_params = self._find_init_params(
            particle_seed, self._init_guide, args, {**kwargs, **self.static_kwargs}
        )
```




## The problem last week
I managed to get the inference working last week but the `MixtureGuidePredictive` interface was not working. I spent some time reading the source code and found two errors. 
First:
```python
self.num_mixture_components = jnp.shape(tree.flatten(self.guide_params)[0][0])[0]
```
Here i changed it to find the num_mixture_components from the `guide_params` instead of `params` since the parameters for the decoder were not duplicated across multiple particles and would lead to error in my case.

Next i also discovered some bugs in the following code:
```python
predictive_model = self.model_predictive(
    posterior_samples=predictive_assign
    )

samples_model = predictive_model(model_key, *args, **kwargs)
```
Here, the predictive_assign would also contain the encoder parameters, and this would cause errors if those parameters were in a complex pytree. 

I decided to move away from the MixtureGuidePredictive and just do the predictions manually for the following reasons:
- It seems the predictive class is more for models with no dependency on data. I could not get inputting the batch to work, since it would just return the same batch in the `obs=batch` part of the decoder.
- Doing the predictions manually would lead to more control.


## Getting the VAE to work
First i load in the VAE example from <https://num.pyro.ai/en/0.6.0/examples/vae.html>

I want to work in jupyter notebooks for testing so i changed some stuff, for example the CLI parse. I also moved some stuff out function scopes so i can more quickly iterate and keep stuff in memory for testing.

The `STEINVI` interface is designed to easily replace the `SVI` interface. I decided to try to keep everything mostly the same and just replace with `STEIVI.
I also saw the recommendation to use the Ada optimizer so i keep that in mind and switch between adam and ada in my testing.

### Problem
The Losses (both train and test) go to NaN very quickly in the training. Keeping the `step_size` of Ada below $0.05$ and the learing rate of Adam small seems to prevent this from happening, but the loss does not seem to converge but instead bounces around. 
This might sugges that there is some kind of very large gradient calculated in the STEINVI step that is somewhat cancelled out by the very low `step_size`.

Other observations with the optimizers:
- The adam optimizer does not generalize at all and every image is the same, suggesting some failure mode, making the decoder output the same value each time. 
- The images when using the ada optimizer are actually good. The loss still bounces.


#### Analyzing the gradient explosion for Ada
I tried printing the maximum gradient of each update step for SMI using the Ada optimizer:
```python
leaf_maxes = jax.tree.map(jnp.max, grads)
all_max  =jnp.max(jnp.array(jax.tree.leaves(leaf_maxes)))
jax.debug.print("max gradient: {x}", x = all_max)
```

Some results for the first epoch:
```
max gradient: 78.14973449707031
max gradient: 440.06146240234375
max gradient: 314.31719970703125
max gradient: 26.300952911376953
max gradient: 304.7109375
max gradient: 262.7981262207031
max gradient: 199.69679260253906
max gradient: 166.2333526611328
max gradient: 147.82730102539062
max gradient: 270.4846496582031
max gradient: 386.1473693847656
max gradient: 981.373779296875
max gradient: 1185.8668212890625
max gradient: 1367.3203125
max gradient: 869.558837890625
max gradient: 841.095703125
max gradient: 544.7709350585938
```
As can be seen it seems to completely explode. Compared to the adam which actually has the max gradient fall slowly.


After spending a lot of time trying to debug this, i have come to the current conclusion: It might just be that this is what is meant to happen. I still get to quite good images from sampling. I think the reason the loss is jumping around is because the loss is evaluated as the norm of the gradients:

```python
    def evaluate(self, state: SteinVIState, *args, **kwargs):
        """Take a single step of Stein (possibly on a batch / minibatch of data).

        :param SteinVIState state: Current state of inference.
        :param args: Positional arguments to the model and guide.
        :param kwargs: Keyword arguments to the model and guide.
        :return: Normed Stein force given by :data:`SteinVIState`.
        """
        # we split to have the same seed as `update_fn` given a state
        _, _, rng_key_eval = random.split(state.rng_key, num=3)
        params = self.optim.get_params(state.optim_state)
        normed_stein_force, _ = self._svgd_loss_and_grads(
            rng_key_eval,
            params,
            state.loss_temperature,
            state.repulsion_temperature,
            *args,
            **kwargs,
            **self.static_kwargs,
        )
        return normed_stein_force
```
rather than the evaluate in SVI:

```
    def evaluate(self, svi_state, *args, **kwargs):
        """
        Take a single step of SVI (possibly on a batch / minibatch of data).

        :param svi_state: current state of SVI.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide.
        :return: evaluate ELBO loss given the current parameter values
            (held within `svi_state.optim_state`).
        """
        # we split to have the same seed as `update_fn` given an svi_state
        _, rng_key_eval = random.split(svi_state.rng_key)
        params = self.get_params(svi_state)
        return self.loss.loss(
            rng_key_eval,
            params,
            self.model,
            self.guide,
            *args,
            **kwargs,
            **self.static_kwargs,
        )
```
that just returns the loss, that is the Trace Elbo in the Vae Example.

### MSE Loss
To be able to compare the different methods, i need a loss that is similar for them all. I experiment with a standard MSE loss between the images and their reconstructed counterpart.

Using the mse, we can see that the loss generally does go down in a more stable manner (I included the end, where it somehow still explodes):
```
Epoch 0: loss = 66.79725646972656 (27.56 s.), mse_loss = 0.039099857211112976
Epoch 1: loss = 65.52764892578125 (3.55 s.), mse_loss = 0.031138921156525612
Epoch 2: loss = 75.78877258300781 (3.06 s.), mse_loss = 0.03007563203573227
Epoch 3: loss = 81.09504699707031 (3.56 s.), mse_loss = 0.029837289825081825
Epoch 4: loss = 91.08635711669922 (3.07 s.), mse_loss = 0.030330300331115723
Epoch 5: loss = 92.05186462402344 (3.07 s.), mse_loss = 0.02872023917734623
Epoch 6: loss = 132.95187377929688 (3.58 s.), mse_loss = 0.029911357909440994
Epoch 7: loss = 120.78569793701172 (3.08 s.), mse_loss = 0.031850121915340424
Epoch 8: loss = 82.37232208251953 (3.07 s.), mse_loss = 0.029768113046884537
Epoch 9: loss = 2562.911865234375 (3.59 s.), mse_loss = 0.030467070639133453
Epoch 10: loss = 113.9303207397461 (3.12 s.), mse_loss = 0.027449093759059906
Epoch 11: loss = 123.33576202392578 (3.12 s.), mse_loss = 0.027190884575247765
Epoch 12: loss = 7971818.0 (3.61 s.), mse_loss = 0.027505170553922653
Epoch 13: loss = nan (3.06 s.), mse_loss = nan
```

EDIT: After setting latent dimension and hidden dimension to the same as the VAE example, it indeed performs better:
```python
Epoch 0: loss = 292.33502197265625 (46.42 s.), mse_loss = 0.03159962221980095
Epoch 1: loss = 572.9500732421875 (0.98 s.), mse_loss = 0.020180899649858475
Epoch 2: loss = 411.0743713378906 (3.28 s.), mse_loss = 0.017833076417446136
Epoch 3: loss = 515.7774047851562 (3.29 s.), mse_loss = 0.015704844146966934
Epoch 4: loss = 575.5299682617188 (3.29 s.), mse_loss = 0.013818115927278996
Epoch 5: loss = 552.0779418945312 (3.32 s.), mse_loss = 0.013038775883615017
Epoch 6: loss = 720.43017578125 (3.29 s.), mse_loss = 0.013359283097088337
Epoch 7: loss = 488.77978515625 (3.29 s.), mse_loss = 0.011911974288523197
Epoch 8: loss = 564.1526489257812 (3.33 s.), mse_loss = 0.012914521619677544
Epoch 9: loss = 492.6183776855469 (3.30 s.), mse_loss = 0.012693608179688454
Epoch 10: loss = 520.7996826171875 (0.85 s.), mse_loss = 0.012004649266600609
Epoch 11: loss = 612.8050537109375 (3.29 s.), mse_loss = 0.012090461328625679
Epoch 12: loss = 511.09857177734375 (3.27 s.), mse_loss = 0.01125369779765606
Epoch 13: loss = 561.8724365234375 (3.30 s.), mse_loss = 0.011793434619903564
Epoch 14: loss = 506.2474670410156 (3.28 s.), mse_loss = 0.011394042521715164
```

#### MSE Loss compared to standard SVI
Running the same mse compare logic on the "VAE with SVI" example i get:
```
Epoch 0: loss = 143.7112579345703 (14.26 s.), mse_loss = 0.03674609214067459
Epoch 1: loss = 121.81253814697266 (0.46 s.), mse_loss = 0.02517949976027012
Epoch 2: loss = 115.04705047607422 (0.12 s.), mse_loss = 0.02047920599579811
Epoch 3: loss = 111.99622344970703 (0.12 s.), mse_loss = 0.019579891115427017
Epoch 4: loss = 110.09611511230469 (0.12 s.), mse_loss = 0.018653705716133118
Epoch 5: loss = 109.02849578857422 (0.12 s.), mse_loss = 0.01928989216685295
Epoch 6: loss = 107.8291015625 (0.12 s.), mse_loss = 0.017941094934940338
Epoch 7: loss = 107.57439422607422 (0.12 s.), mse_loss = 0.017174717038869858
Epoch 8: loss = 107.30595397949219 (0.12 s.), mse_loss = 0.01743471622467041
Epoch 9: loss = 106.43675994873047 (0.12 s.), mse_loss = 0.018033649772405624
Epoch 10: loss = 106.08879089355469 (0.12 s.), mse_loss = 0.018783163279294968
Epoch 11: loss = 105.81175994873047 (0.12 s.), mse_loss = 0.017368517816066742
Epoch 12: loss = 104.96485900878906 (0.12 s.), mse_loss = 0.01674378477036953
Epoch 13: loss = 105.13158416748047 (0.12 s.), mse_loss = 0.016703523695468903
Epoch 14: loss = 104.62661743164062 (0.12 s.), mse_loss = 0.015447555109858513
```

~~As can be seen, the mse results are generally better for the standard SVI engine.~~
It is slightly worse, showing slight improvement using the SMI engine.


## Takeaways from this week
- SMI works in python. ~~Produces okay-ish results, though not outperforming a regular VAE.~~ It does outperform the VAE
- There are some parts of the optimization that sometimes "explode" which produces very large gradients and sometimes gives "NaN" values. 
- I have tried to solve/investigate those issues. Even though i feel i am beginning to understand the source better, i have not managed to figure it out.
- Some potential ideas could be changing optimizer/kernel.
- I discovered the Ola Rønnings [PHD thesis](https://di.ku.dk/english/research/phd/phd-theses/2023/Ola_PhD_Thesis.pdf) which describes the main ideas of the code base, which was interesting and might be helpful later on.
- The VAE example i have been building on does everything very manually. SMI might work better just doing the basic approach: Define model, define guide, run.
- The code feels a little unorganized and chaotic, might be a good idea to clean up.


## Next week:
- Hopefully get past this SMI obstacle in some way, since i feel like i am wasting a lot of time on weird unintuitive stuff, like reading through source code.










