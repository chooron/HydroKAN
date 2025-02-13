# HydroKAN.jl

Hydrological Acknowledge Based Kolmogorov-Arnold Network

The current work of [KolmogorovArnold.jl](https://github.com/vpuri3/KolmogorovArnold.jl) and others has only implemented the computation of the KAN Layer. However, with the recent update to KAN 2.0, which introduces multiplication operators, additional operations are now required. Moreover, there is a need to implement importance screening functionalities. Therefore, this library primarily focuses on implementing analytical and visualization features for KAN 2.0, largely inspired by the implementation approach of pykan. In my experiments, I have observed that Julia-based KAN computations are faster, which is why I have decided to use the Julia language to build the model. This model will mainly be applied to solving ordinary differential equations, particularly in the context of hydrological model calculations.

The expected functionalities of HydroKAN.jl include the following:

- Support for the construction of MultiKAN, enabling the execution of multiplication operations.

- Support for the kan-compiler functionality as seen in pykan.

- Support for importance evaluation, pruning, and a series of other functionalities as implemented in pykan.

**Note**: I am not a professional software developer, and the development of this library is mainly for my own learning and research purposes in hydrology. I will integrate it into a package after the development is stable.