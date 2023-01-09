# jaxtomo: tomographic projectors in JAX

jaxtomo implements tomographic projectors with [JAX](https://github.com/google/jax). 

They are implemented purely in Python, which makes the code readable and hackable. Because JAX offers just-in-time compilation to GPU, the projectors are reasonably fast. They don't use texture memory and are therefore slower than optimized implementations such as [torch-radon](https://github.com/matteo-ronchetti/torch-radon).

## Disclaimer

This is a personal project and very work-in-progress. It is meant as a learning exercise for me, a pedagogical implementation for others, and maybe even a tool for implementing proof-of-concept pipelines.

## Features:

* Parallel beam
* Fan beam
* FBP
* ... all with a flat detector

## Todo: 

* Cone beam
* Registering FP and BP as their respective transpose for autodiff with JAX
* Curved detector

## Proof of concept

![image](https://user-images.githubusercontent.com/5190547/211376095-c383e8ed-dc1d-40be-8869-0261c1830ccb.png)

