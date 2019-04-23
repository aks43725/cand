# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import pandas as pd
import cand

np.random.seed(11)

data = cand.gendata(n=100, d=20, s=6, sim_case=3, param={'rho': 0.5, 'sig2': 1.0})

if __name__ == '__main__':
  
  # The GA
  models_GA = cand.GA(data['X'], data['y'])
  models_GA.fit()
  
  # The SA, with model sizes appeared in the last GA generation
  count = np.bincount(models_GA.generations['model_size'][-1])
  SA_sizes = np.nonzero(count)[0]
  models_SA = cand.SA(data['X'], data['y'])
  models_SA.fit(SA_sizes, verbose=True)
  
  # The RP:
  models_RP = cand.RP(data['X'], data['y'])
  models_RP.fit()

