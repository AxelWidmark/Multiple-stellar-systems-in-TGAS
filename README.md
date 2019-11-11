# Multiple-stellar-systems-in-TGAS
Code that infers binary and trinary stellar system populations in the TGAS catalogue. This is based on the following article:
https://arxiv.org/abs/1801.08547


The repository contains two files:

* Model.py

This file contains a class called "model". In the code, there are short comments that explain what the different functions do. Among other things the class has a function called "model.hyperposterior" that takes six hyperparameters and returns a non-normalized posterior value. Throw this into your favorite MCMC sampler and get to work!

* TGASx2MASS_cut_d<200pc.npz

Contains a sample of stellar objects with astrometric and photometric information, from a cross-match between TGAS and 2MASS. This is automatically loaded by the Model.py script.
