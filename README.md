# jaxtomo: tomographic projectors in JAX

jaxtomo implements tomographic projectors with [JAX](https://github.com/google/jax). 

They are implemented purely in Python, which makes the code readable and hackable. Because JAX offers just-in-time compilation to GPU, the projectors are reasonably fast. They don't use texture memory and are slower than optimized implementations such as [torch-radon](https://github.com/matteo-ronchetti/torch-radon).

## Disclaimer

This is a personal project and very work-in-progress. It is meant as a learning exercise for me, a pedagogical implementation for others (once I add some comments), and maybe even a tool for implementing proof-of-concept pipelines.

## Features

* Parallel beam
* Fan beam
* Cone Beam
* FBP
* ... all with a flat detector
* FP and BP registered as respective transpose for autodiff with JAX
* End-to-end SIR via autodiff
* `jax.pmap` for multi-GPU speedup

## Todo

* Valid FBP for large fan/cone angles (atm we just do Ramlak filter + BP)
* Other FP methods (Siddon, Footprint, ...)
* Curved detector
* Different voxel basis functions [[1]](https://pubmed.ncbi.nlm.nih.gov/17278818/), [[2]](https://www.researchgate.net/publication/263921475_Projector_and_Backprojector_for_Iterative_CT_Reconstruction_with_Blobs_using_CUDA)
* speedup [bilinear interpolation](https://github.com/clemisch/jaxtomo/blob/d47796a41381975d4e195eee6360bc93561013e3/jaxtomo/util.py#L47) and/or profile FP, as it's rather slow

## Proof of concept

### Fan FBP
![image](https://user-images.githubusercontent.com/5190547/215475790-4551313a-76fc-409c-9b41-6417b61bf69b.png)

### Parallel, fan, and cone projector
![image](https://user-images.githubusercontent.com/5190547/220201015-89feac80-b14e-4899-9af2-2be3c35ce14f.png)

