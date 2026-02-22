# Week 08 Logs

## Notes from supervisor conversation:
- [ ] Get SMI to work on a simple variational autoencoder.
- [ ] Note any potential errors in the the STEINVI library


## Potential errors in the STEINVI Library
I added `self.static_kwargs` as input to the `self.find_init_params` method. Otherwise the dimensions would be wrong in my case.
```python
guide_init_params = self._find_init_params(
            particle_seed, self._init_guide, args, {**kwargs, **self.static_kwargs}
        )
```

## How the 
