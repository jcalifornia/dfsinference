dfsinference
============

Inference of bond forces and diffusivities from dynamic force spectoscopy bond pulling experiments using the method found in the manuscript [arXiv:1502.06415] (http://arxiv.org/abs/1502.06415).

This method used a Bayesian interpretation of Tikhonov regularization in order to provide the estimated
bond force, diffusivity, and error estimates of these functions. The overall approach is empirical Bayes in flavor as we are not integrating over a prior distribution for the regularization parameters. However, it is easy to extend this method into full-Bayes, which we will eventually implement in this script.

We assume that the user is not interested in the repulsive behavior for short bond separations. For this reason, we just assume the Lennard-Jones potential holds at short distances - the scaling for what is meant by short distance must be inputted by the user. At the present moment this involves some slight modification of the script - we hope to implement more user-friendly definitions of length scales and units soon. If you find yourself in need of this feature, please shoot me an email. We also appreciate any contributions to the source code base.

Data is expected in either text format or as a .mat file containing a matrix object named 'A' consisting of time in the first column and bond coordinate position in the second column. Check out the included .ipynb file for usage. 

The following are the dependencies of this script:
* numpy
* sympy
* [hyperopt] (https://github.com/hyperopt/hyperopt) for finding the regularization parameters

Please cite the following paper if you found these ideas useful:

 This method was published in Biophysical Journal in the manuscript

```
@article{chang2015bayesian,
  title={Bayesian Uncertainty Quantification for Bond Energies and Mobilities Using Path Integral Analysis},
  author={Chang, Joshua C and Fok, Pak-Wing and Chou, Tom},
  journal={Biophysical Journal},
  volume={109},
  number={5},
  pages={966--974},
  year={2015},
  publisher={Elsevier}
}

```
