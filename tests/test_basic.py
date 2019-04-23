# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import pandas as pd
import cand

ribflv = pd.read_csv('data/riboflavin.csv', index_col=0).T
y = ribflv['q_RIBFLV']
X = ribflv.drop(['q_RIBFLV'], axis=1)
n, d = X.shape

np.random.seed(570)

if __name__ == '__main__':
  
  # The GA
  models_GA = cand.GA(X, y)
  models_GA.fit()
  print('Model sizes:\n{}'.format(models_GA.models.sum(axis=1)))
  print('Fitness values:\n{}'.format(models_GA.fitness))
  
  # The SA, with model sizes appeared in the last GA generation
  count = np.bincount(models_GA.generations['model_size'][-1])
  SA_sizes = np.nonzero(count)[0]
  models_SA = cand.SA(X, y)
  models_SA.fit(SA_sizes, verbose=True)
  print('Model sizes:\n{}'.format(models_SA.models.sum(axis=1)))
  print('GIC values:\n{}'.format(models_SA.ic))
  
  # The RP:
  models_RP = cand.RP(X, y)
  models_RP = models_RP.fit()
  print('Model sizes:\n{}'.format(models_RP.models.sum(axis=1)))
  print('GIC values:\n{}'.format(cand.GIC(X, y, models_RP.models, 'PLIC', n_jobs=-1)))

