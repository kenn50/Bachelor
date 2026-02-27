# Notes from meeting
- I accidentally created amortized SMI. This was surprising and not intentional.
- We spent an hour developing an architecture that uses a global variable m. This global variable was the only one to be affected by SMI. We then use a normalizing flow to go from m to z.
- The architecture is descriped in [Extra Documents/MI_VAE_outline.pdf](>../Extra%20Documents/SMI_VAE_outline.pdf)


## Normalizing flows
I spent some time learning about Normalizing flows and how to implement them in numpyro. I read chapter 3 in [Introduction to VAEs](https://arxiv.org/abs/1906.02691) about normalizing flows as Autoregressive neural networks such as Made. Luckily this is implemented in numpyro as AutoregressiveNN and InverseAutoregressiveTransform. I also found the BlockNeuralAutoregressiveNN which apparently only requires on pass through because it is more complex. I think i will use this since it is easier to implement. They are also similar to stax, which make it easy to import them via `numpyro.module`





