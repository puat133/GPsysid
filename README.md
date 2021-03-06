# Probabilistic approach to catalyst activity estimation for chemical reactors
This repository contains several methods to estimate the catalyst activity of a chemical reactor.
- Linear System Identification approach.
- Gaussian Process bases system identification using Reduced Rank method.
- Partial differential equation parameter estimation with latent force.

To use this package, please install the following prerequisities:

1. `SIPPY` package from [github](https://github.com/CPCLAB-UNIPI/SIPPY). Follow the instruction there.
2. `Numba` version should be later than `0.47`.
3. `Pytorch`
4. `tqdm`
5. 

Explanation of files and folders:

1. `Documentation` contains a pdf file related to the PDE approach.
2. `Python Notebook` is the folder where all python notebook can be found.
3. `chemReactor.py` is the file that runs the finite rank GP system identification.
4. `finiteDimensionalGP` is a library that contains classes for finite rank GP system identification.
5. `GPutils.py` is a numerical library that is used in `chemReactor.py`.
6. `PDE.py` is a numerical library that is used in a Python notebook about PDE approach.
7. `finiteRankGPsysID_example.py` is the test file which runs the finite rank GP system identification for a toy model.

As this repository is a working folder, things might break without notification. Should you face a problem, please contact <muhammad.emzir@aalto.fi>.
