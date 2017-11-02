# Multiple-stellar-systems-in-TGAS
Code that infers binary and trinary stellar system populations in the TGAS catalogue

A work-in-progress article that explains what's going on:
https://www.overleaf.com/read/wrvbcsbfzqqz


The repository contains two files at the moment:

* Model.py
This file contains a class called "model". Among other things the class has a function called "model.hyperposterior" that takes six hyperparameters and returns a non-normalized posterior value. Throw this into your favorite MCMC sampler and get to work!

* TGASx2MASS_cut_d<200pc.npz
Contains a sample of stellar objects with astrometric and photometric information, from a cross-match between TGAS and 2MASS
This is automatically loaded by the Model.py script.
