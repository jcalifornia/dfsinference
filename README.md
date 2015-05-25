dfsinference
============

Inference of bond forces and diffusivities from dynamic force spectoscopy experiment using the method found in the manuscript [arXiv:1502.06415] (http://arxiv.org/abs/1502.06415)

This method used a Bayesian interpretation of Tikhonov regularization in order to provide the estimated
bond force, diffusivity, and error estimates of these functions.

Simulated [data] (https://www.dropbox.com/sh/xjrjcvhkx64xyxg/AACcTJ5jTLsGXkWtobhsq1Oia?dl=0) is available. Check out the included .ipynb file for usage. 

The following are the dependencies of this script:
* numpy
* sympy
* hyperopt for finding the regularization parameters
