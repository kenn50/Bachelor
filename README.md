
# The Stein Mixture VAE

This is the source code for the bachelor thesis: "The Stein Mixture Variational Autoencoder:  A study of treating generation in VAEs as a posterior predictive" 

Name: Kevin Mark Lock
Supervisor: Thomas Hamelryck
Help from: Ola Rønning
Date: 12/06/2026


# Structure of the code
The project is primarily built on JAX and NumPyro together with the contributed SteinVI library, which was slighly edited in `src/CustomModules/stein_impl_source.py`

## Main contributions
- Four experiments are found in `Experiments/Final Experiments`, which are those referenced in the thesis. They are jupyter notebooks, that use the different methods architectures described in the thesis. Older (potentially not working anymore) experiments can be found in `Experiments/Old`
- The architecturees worked on in `src/CustomModules/architectures.py`
This file contains classes for a range of probabilistic architectures worked on. 


## Custom Src code modules
To acces custom modules and changes to libraries everywhere in the codebase, we put everything in a CustomModoules folder.
Run `pip install -e .` to get it it working.






